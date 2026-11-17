//! One worker connection, attachment authority and bounded event fanout.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use futures_core::Stream;
use mirror_protocol::agent_rpc_client::AgentRpcClient;
use mirror_protocol::daemon::{AgentConfiguration, AttachWorkspaceResponse, Workspace};
use mirror_protocol::mirror::{
    mirror_rpc_client::MirrorRpcClient, ForkSessionRequest, ForkSessionResponse,
    ForkWorkspaceRequest, ForkWorkspaceResponse,
};
use mirror_protocol::{
    chat_record, client_message, session_message, CallToolRequest, CallToolResponse, ClientMessage,
    GetMessagesRequest, GetMessagesResponse, HostToolDefinition, HostToolResult, InputAck,
    InputNack, InputNackReason, SessionMessage, SetApprovalMode,
};
use prost::Message;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::Status;
use uuid::Uuid;

use crate::{AgentProcess, DisplayUpdates, ExecutionLease, SessionProjection, MAX_FRAME_BYTES};

/// A route to one agent; durable history and command deduplication belong to the agent.
#[derive(Clone, Debug)]
pub struct Session {
    commands: mpsc::Sender<Command>,
    client: AgentRpcClient<Channel>,
    lifecycle: MirrorRpcClient<Channel>,
    settings: watch::Receiver<Workspace>,
}

/// Replay and lag handling negotiated by the attachment consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentRecovery {
    Reconnect,
    InBand,
    InBandWithHistory,
}

enum DisplayBatch {
    Updates(VecDeque<Arc<SessionMessage>>),
    Snapshot(Vec<SessionMessage>),
}

#[derive(Debug)]
struct TerminalDisplay {
    updates: VecDeque<Arc<SessionMessage>>,
    failure: Status,
}

struct Attachment {
    replay: Vec<SessionMessage>,
    changes: watch::Receiver<Option<Arc<TerminalDisplay>>>,
    presence: watch::Sender<()>,
    tools: mpsc::Receiver<SessionMessage>,
}

struct Client {
    id: String,
    updates: DisplayUpdates,
    changes: watch::Sender<Option<Arc<TerminalDisplay>>>,
    presence: watch::Receiver<()>,
    pending: Option<watch::Sender<()>>,
    deadline: Instant,
}

impl Client {
    fn is_live(&self) -> bool {
        self.presence.has_changed().is_ok()
            && (self.pending.is_none() || Instant::now() < self.deadline)
    }
}

type CommandReply = oneshot::Sender<Result<Option<SessionMessage>, Status>>;

enum Command {
    Reserve(
        String,
        Option<Vec<HostToolDefinition>>,
        oneshot::Sender<Result<AttachWorkspaceResponse, Status>>,
    ),
    Attach(String, oneshot::Sender<Result<Attachment, Status>>),
    Updates(String, oneshot::Sender<Result<DisplayBatch, Status>>),
    Send(String, Box<ClientMessage>, CommandReply),
    Authorize(
        String,
        bool,
        oneshot::Sender<Result<ExecutionLease, Status>>,
    ),
    Close(oneshot::Sender<()>),
}

struct SessionState {
    id: String,
    projection: SessionProjection,
    requests: mpsc::Sender<ClientMessage>,
    commands: mpsc::Receiver<Command>,
    clients: HashMap<String, Client>,
    settings: watch::Sender<Workspace>,
    host_client: String,
    host_tools: Vec<HostToolDefinition>,
    host_events: Option<mpsc::Sender<SessionMessage>>,
    host_calls: HashSet<String>,
    failed_host_calls: VecDeque<String>,
    startup_messages: Vec<SessionMessage>,
    executions: watch::Sender<()>,
    settings_requests: HashMap<String, (ClientMessage, bool)>,
    settings_bytes: usize,
    pending_responses: Vec<(String, CommandReply)>,
}

impl Session {
    pub async fn start(
        mut agent: Option<AgentProcess>,
        channel: Channel,
        mut workspace: Workspace,
        configuration: AgentConfiguration,
        host_client: String,
    ) -> Result<Self, Status> {
        let lifecycle = MirrorRpcClient::new(channel.clone());
        let mut client = AgentRpcClient::new(channel)
            .max_decoding_message_size(4 * MAX_FRAME_BYTES)
            .max_encoding_message_size(MAX_FRAME_BYTES);
        let inference = configuration
            .inference
            .ok_or_else(|| Status::invalid_argument("inference configuration is required"))?;
        let id = workspace.session_id.clone();
        let opened = async {
            let mut projection = read_projection(&mut client, &id).await?;
            let (requests, incoming) = mpsc::channel(32);
            let configuration_id = format!("daemon:configure:{}", Uuid::new_v4());
            requests
                .send(ClientMessage {
                    session_id: id.clone(),
                    message_id: configuration_id.clone(),
                    payload: Some(client_message::Payload::SetInferenceConfig(
                        inference.clone(),
                    )),
                })
                .await
                .map_err(|_| closed())?;
            // A server may wait for the first message before sending response headers.
            let mut responses = client
                .session(ReceiverStream::new(incoming))
                .await?
                .into_inner();
            let approval_id = format!("{configuration_id}:approval");
            requests
                .send(ClientMessage {
                    session_id: id.clone(),
                    message_id: approval_id.clone(),
                    payload: Some(client_message::Payload::SetApprovalMode(SetApprovalMode {
                        mode: configuration.approval_mode,
                    })),
                })
                .await
                .map_err(|_| closed())?;
            loop {
                let event = responses.message().await?.ok_or_else(closed)?;
                match &event.payload {
                    Some(session_message::Payload::InputAck(ack))
                        if ack.message_id == approval_id =>
                    {
                        break
                    }
                    Some(
                        session_message::Payload::InputNack(_)
                        | session_message::Payload::HostToolCall(_),
                    ) => return Err(Status::failed_precondition("worker rejected startup")),
                    _ => projection.apply(&event)?,
                }
            }
            Ok::<_, Status>((projection, requests, responses))
        };
        let opened = tokio::time::timeout(Duration::from_secs(30), opened)
            .await
            .unwrap_or_else(|_| Err(Status::deadline_exceeded("worker configuration timed out")));
        let (projection, requests, responses) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                if let Some(agent) = agent.as_mut() {
                    agent.close().await;
                }
                return Err(error);
            }
        };
        workspace.inference = Some(inference);
        workspace.approval_mode = Some(configuration.approval_mode);
        let (commands, receiver) = mpsc::channel(32);
        let (settings, observed_settings) = watch::channel(workspace);
        let state = SessionState {
            id,
            projection,
            requests,
            commands: receiver,
            settings,
            host_client,
            host_tools: configuration.host_tools,
            clients: HashMap::new(),
            host_events: None,
            host_calls: HashSet::new(),
            failed_host_calls: VecDeque::new(),
            startup_messages: agent
                .as_mut()
                .map(|agent| std::mem::take(&mut agent.startup_messages))
                .unwrap_or_default(),
            executions: watch::channel(()).0,
            settings_requests: HashMap::new(),
            settings_bytes: 0,
            pending_responses: Vec::new(),
        };
        tokio::spawn(state.run(agent, responses));
        Ok(Self {
            commands,
            client,
            lifecycle,
            settings: observed_settings,
        })
    }

    pub fn settings(&self) -> Workspace {
        self.settings.borrow().clone()
    }
    pub fn is_closed(&self) -> bool {
        self.commands.is_closed()
    }

    /// Reserve authority for a bounded interval until the Session stream owns its lifetime.
    pub async fn reserve(
        &self,
        client_id: String,
        host_tools: Option<Vec<HostToolDefinition>>,
    ) -> Result<AttachWorkspaceResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::Reserve(client_id, host_tools, reply))
            .await
            .map_err(|_| closed())?;
        result.await.map_err(|_| closed())?
    }

    /// Cancel pending attachment when shutdown needs its execution lease released.
    pub async fn attach(
        &self,
        attachment_id: String,
        incoming: impl Stream<Item = Result<ClientMessage, Status>> + Send + 'static,
        mut lease: ExecutionLease,
        recovery: AttachmentRecovery,
    ) -> Result<impl Stream<Item = Result<SessionMessage, Status>> + Send + 'static, Status> {
        let attaching = async {
            let (reply, result) = oneshot::channel();
            self.commands
                .send(Command::Attach(attachment_id.clone(), reply))
                .await
                .map_err(|_| closed())?;
            result.await.map_err(|_| closed())?
        };
        let attachment = tokio::select! {
            biased;
            _ = lease.closed() => return Err(Status::cancelled("attachment closed")),
            result = attaching => result?,
        };
        Ok(attachment.into_stream(self.clone(), attachment_id, incoming, recovery))
    }

    async fn updates(&self, attachment_id: &str) -> Result<DisplayBatch, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::Updates(attachment_id.to_owned(), reply))
            .await
            .map_err(|_| closed())?;
        result.await.map_err(|_| closed())?
    }

    async fn send(
        &self,
        attachment_id: &str,
        request: ClientMessage,
    ) -> Result<Option<SessionMessage>, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .try_send(Command::Send(
                attachment_id.to_owned(),
                Box::new(request),
                reply,
            ))
            .map_err(|_| Status::resource_exhausted("session ingress is full or closed"))?;
        result.await.map_err(|_| closed())?
    }

    /// Host-only operations must not execute another client's extension implementations.
    pub async fn authorize(
        &self,
        attachment_id: String,
        host_only: bool,
    ) -> Result<ExecutionLease, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::Authorize(attachment_id, host_only, reply))
            .await
            .map_err(|_| closed())?;
        result.await.map_err(|_| closed())?
    }

    pub async fn fork(
        &self,
        request: ForkSessionRequest,
        mut lease: ExecutionLease,
    ) -> Result<ForkSessionResponse, Status> {
        let mut lifecycle = self.lifecycle.clone();
        tokio::select! {
            _ = lease.closed() => Err(Status::cancelled("attachment closed")),
            result = lifecycle.fork_session(request) => result.map(tonic::Response::into_inner),
        }
    }

    pub async fn fork_workspace(
        &self,
        request: ForkWorkspaceRequest,
        mut lease: ExecutionLease,
    ) -> Result<ForkWorkspaceResponse, Status> {
        let mut lifecycle = self.lifecycle.clone();
        tokio::select! {
            _ = lease.closed() => Err(Status::cancelled("attachment closed")),
            result = lifecycle.fork_workspace(request) => result.map(tonic::Response::into_inner),
        }
    }

    pub async fn call_tool(
        &self,
        request: CallToolRequest,
        mut lease: ExecutionLease,
    ) -> Result<CallToolResponse, Status> {
        let mut client = self.client.clone();
        tokio::select! {
            _ = lease.closed() => Err(Status::cancelled("attachment closed")),
            result = client.call_tool(request) => result.map(tonic::Response::into_inner),
        }
    }

    pub async fn get_messages(
        &self,
        request: GetMessagesRequest,
    ) -> Result<GetMessagesResponse, Status> {
        tokio::time::timeout(
            Duration::from_secs(10),
            read_history_page(&mut self.client.clone(), request),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("worker history read timed out"))?
    }

    pub async fn close(&self) {
        let (reply, result) = oneshot::channel();
        if self.commands.send(Command::Close(reply)).await.is_ok() {
            let _ = result.await;
        }
    }
}

impl Attachment {
    fn into_stream(
        self,
        session: Session,
        attachment_id: String,
        incoming: impl Stream<Item = Result<ClientMessage, Status>> + Send + 'static,
        recovery: AttachmentRecovery,
    ) -> impl Stream<Item = Result<SessionMessage, Status>> + Send + 'static {
        let Self {
            replay,
            mut changes,
            presence,
            mut tools,
        } = self;
        let display = session.clone();
        let display_id = attachment_id.clone();
        let (local, mut responses) = mpsc::channel(32);
        tokio::spawn(async move {
            let _presence = presence;
            tokio::pin!(incoming);
            loop {
                tokio::select! {
                    biased;
                    _ = local.closed() => break,
                    _ = session.commands.closed() => break,
                    request = incoming.next() => match request {
                        Some(Ok(request)) => tokio::select! {
                            _ = local.closed() => break,
                            sent = submit(&session, &attachment_id, request, &local) => if !sent { break; },
                        },
                        _ => break,
                    }
                }
            }
        });
        async_stream::try_stream! {
            let start = replay_start(&replay);
            let mut pending: VecDeque<_> = replay.into_iter().map(Arc::new).collect();
            if recovery == AttachmentRecovery::InBandWithHistory {
                pending.push_front(Arc::new(start));
                pending.push_back(Arc::new(state_restored()));
            }
            let mut hosting = true;
            let mut replying = true;
            let mut failure: Option<Status> = None;
            let mut dirty = false;
            let mut next_display = Instant::now();
            loop {
                if pending.is_empty() && !replying {
                    if let Some(failure) = failure.take() { Err(failure)?; }
                }
                let message = tokio::select! {
                    biased;
                    changed = changes.changed(), if failure.is_none() => {
                        let terminal = changes.borrow_and_update().clone();
                        match (changed, terminal) {
                            (_, Some(terminal)) => {
                                pending.extend(terminal.updates.iter().cloned());
                                failure = Some(terminal.failure.clone());
                            }
                            (Err(_), None) => failure = Some(closed()),
                            (Ok(()), None) => { dirty = true; continue; }
                        }
                        // Execution authority is already revoked; only accepted output may drain.
                        hosting = false;
                        tools.close();
                        continue;
                    },
                    message = tools.recv(), if hosting => match message {
                        Some(message) => Some(message), None => { hosting = false; continue; }
                    },
                    message = responses.recv(), if replying => match message {
                        Some(message) => Some(message),
                        None if failure.is_none() => break,
                        None => { replying = false; continue; }
                    },
                    _ = std::future::ready(()), if !pending.is_empty() => {
                        pending.pop_front().map(|message| (*message).clone())
                    },
                    _ = tokio::time::sleep_until(next_display), if dirty && failure.is_none() => None,
                };
                if let Some(message) = message {
                    yield message;
                    continue;
                }
                // Display work follows a frame cadence; host calls and replies bypass even replay.
                next_display = Instant::now() + Duration::from_millis(16);
                dirty = false;
                let batch = display.updates(&display_id).await;
                // Shutdown may have handed off the mailbox while this command was waiting.
                if batch.is_err() && changes.borrow().is_some() { continue; }
                match batch? {
                    DisplayBatch::Updates(messages) => pending = messages,
                    DisplayBatch::Snapshot(replay) => {
                        if recovery == AttachmentRecovery::Reconnect {
                            Err(Status::resource_exhausted("attachment lost events; reconnect to catch up"))?;
                        }
                        pending.push_back(Arc::new(replay_start(&replay)));
                        pending.extend(replay.into_iter().map(Arc::new));
                        pending.push_back(Arc::new(state_restored()));
                    }
                }
            }
        }
    }
}

impl SessionState {
    async fn run(
        mut self,
        mut agent: Option<AgentProcess>,
        responses: impl Stream<Item = Result<SessionMessage, Status>>,
    ) {
        tokio::pin!(responses);
        let mut closing = None;
        let mut cleanup = tokio::time::interval(Duration::from_millis(100));
        let startup_deadline = Instant::now() + Duration::from_secs(30);
        let mut attached = false;
        let outgoing = self.requests.clone();
        let failure = loop {
            tokio::select! {
                _ = cleanup.tick() => {
                    if !attached && Instant::now() >= startup_deadline { break closed(); }
                    if attached { self.prune_clients(); }
                },
                permit = outgoing.reserve(), if !self.failed_host_calls.is_empty() => {
                    let Ok(permit) = permit else { break closed(); };
                    if let Some(call_id) = self.failed_host_calls.pop_front() {
                        permit.send(ClientMessage {
                            session_id: self.id.clone(),
                            message_id: format!("daemon:host-unavailable:{call_id}"),
                            payload: Some(client_message::Payload::HostToolResult(HostToolResult {
                                call_id,
                                content: "The execution host is unavailable; the call was not transferred.".into(),
                                success: false,
                                display: None,
                            })),
                        });
                    }
                },
                event = responses.next() => match event {
                    Some(Ok(event)) => if let Err(error) = self.observe(event) { break error; },
                    Some(Err(error)) => break error,
                    None => break closed(),
                },
                command = self.commands.recv() => match command {
                    Some(Command::Reserve(client_id, host_tools, reply)) => {
                        let result = self.reserve(client_id, host_tools);
                        attached |= result.is_ok();
                        let _ = reply.send(result);
                    }
                    Some(Command::Attach(id, reply)) => { let _ = reply.send(self.attach(&id)); }
                    Some(Command::Updates(id, reply)) => { let _ = reply.send(self.updates(&id)); }
                    Some(Command::Send(id, request, reply)) => {
                        // One unacknowledged command per attachment, independent of display lag.
                        // Keep abandoned waiters until their reply, including reused message IDs.
                        if self.pending_responses.len() >= 32 {
                            let _ = reply.send(Err(Status::resource_exhausted("command acknowledgements are pending")));
                            continue;
                        }
                        let message_id = request.message_id.clone();
                        match self.admit(&id, *request) {
                            Ok(None) => self.pending_responses.push((message_id, reply)),
                            result => { let _ = reply.send(result); }
                        }
                    }
                    Some(Command::Authorize(id, host_only, reply)) => { let _ = reply.send(self.authorize(&id, host_only)); }
                    Some(Command::Close(reply)) => { closing = Some(reply); break Status::cancelled("workspace closed"); }
                    None => break closed(),
                }
            }
        };
        self.executions.send_replace(());
        for (_, mut client) in self.clients.drain() {
            // An overflowed mailbox needs a live worker for recovery, not an incomplete tail.
            let updates = client.updates.take().unwrap_or_default();
            client.changes.send_replace(Some(Arc::new(TerminalDisplay {
                updates,
                failure: failure.clone(),
            })));
        }
        drop(self.commands);
        drop(self.pending_responses);
        if let Some(agent) = agent.as_mut() {
            agent.close().await;
        }
        self.executions.closed().await;
        if let Some(reply) = closing {
            let _ = reply.send(());
        }
    }

    fn prune_clients(&mut self) {
        self.clients.retain(|_, client| client.is_live());
        if !self
            .clients
            .values()
            .any(|client| client.id == self.host_client)
        {
            self.host_events = None;
            self.host_client.clear();
            self.fail_host_calls();
        }
    }

    fn reserve(
        &mut self,
        client_id: String,
        host_tools: Option<Vec<HostToolDefinition>>,
    ) -> Result<AttachWorkspaceResponse, Status> {
        self.prune_clients();
        if self.clients.len() >= 16 {
            return Err(Status::resource_exhausted("attachment limit reached"));
        }
        if self.clients.values().any(|client| client.id == client_id) {
            return Err(Status::already_exists("client already attached"));
        }
        let id = Uuid::new_v4().to_string();
        let (pending, presence) = watch::channel(());
        // A new host must supply its live tool definitions; saved metadata grants no authority.
        if self.host_client.is_empty() && host_tools.as_ref() == Some(&self.host_tools) {
            self.host_client = client_id.clone();
        }
        let execution_host = client_id == self.host_client;
        self.clients.insert(
            id.clone(),
            Client {
                id: client_id,
                updates: DisplayUpdates::default(),
                changes: watch::channel(None).0,
                presence,
                pending: Some(pending),
                deadline: Instant::now() + Duration::from_secs(30),
            },
        );
        Ok(AttachWorkspaceResponse {
            attachment_id: id,
            execution_host,
            workspace: Some(self.settings.borrow().clone()),
            startup_messages: self.startup_messages.clone(),
            startup_error: None,
            ..AttachWorkspaceResponse::default()
        })
    }

    fn attach(&mut self, id: &str) -> Result<Attachment, Status> {
        let client = self
            .clients
            .get_mut(id)
            .filter(|client| client.is_live())
            .ok_or_else(|| Status::permission_denied("attachment expired"))?;
        let presence = client
            .pending
            .take()
            .ok_or_else(|| Status::already_exists("attachment already has a Session stream"))?;
        let (tools, incoming_tools) = mpsc::channel(32);
        if client.id == self.host_client {
            self.host_events = Some(tools);
        }
        Ok(Attachment {
            replay: self.projection.replay(&self.id),
            changes: client.changes.subscribe(),
            presence,
            tools: incoming_tools,
        })
    }

    fn updates(&mut self, id: &str) -> Result<DisplayBatch, Status> {
        let client = self
            .clients
            .get_mut(id)
            .filter(|client| client.is_live())
            .ok_or_else(|| Status::permission_denied("attachment expired"))?;
        // The drain and replacement snapshot share one actor turn; later events queue normally.
        Ok(match client.updates.take() {
            Some(messages) => DisplayBatch::Updates(messages),
            None => DisplayBatch::Snapshot(self.projection.replay(&self.id)),
        })
    }

    fn authorize(&self, id: &str, host_only: bool) -> Result<ExecutionLease, Status> {
        let client = self
            .clients
            .get(id)
            .filter(|client| client.is_live())
            .ok_or_else(|| Status::permission_denied("attachment expired"))?;
        if host_only && client.id != self.host_client {
            return Err(Status::permission_denied(
                "operation requires the execution host",
            ));
        }
        if self.executions.receiver_count() >= 32 {
            return Err(Status::resource_exhausted(
                "session execution limit reached",
            ));
        }
        Ok(ExecutionLease {
            attachment: client.presence.clone(),
            worker: self.executions.subscribe(),
        })
    }

    fn admit(
        &mut self,
        attachment_id: &str,
        request: ClientMessage,
    ) -> Result<Option<SessionMessage>, Status> {
        let client = self
            .clients
            .get(attachment_id)
            .filter(|client| client.is_live())
            .ok_or_else(|| Status::permission_denied("attachment expired"))?;
        if request.session_id != self.id {
            return Err(Status::invalid_argument(
                "session does not match attached workspace",
            ));
        }
        if request.message_id.is_empty()
            || request.message_id.starts_with("daemon:")
            || request.encoded_len() > MAX_FRAME_BYTES
        {
            return Err(Status::invalid_argument("invalid command identity or size"));
        }
        match &request.payload {
            None => return Ok(Some(ack(&request.message_id))),
            Some(client_message::Payload::HostToolResult(result))
                if client.id != self.host_client || !self.host_calls.contains(&result.call_id) =>
            {
                return Err(Status::permission_denied(
                    "tool result does not belong to this client",
                ));
            }
            Some(client_message::Payload::DetachTools(_)) if client.id != self.host_client => {
                return Err(Status::permission_denied(
                    "only the execution host may detach tools",
                ));
            }
            Some(
                client_message::Payload::SetInferenceConfig(_)
                | client_message::Payload::SetApprovalMode(_),
            ) => {
                if !self.settings_requests.contains_key(&request.message_id)
                    && (self.settings_requests.len() >= 4096
                        || self.settings_bytes + request.encoded_len() > 4 * MAX_FRAME_BYTES)
                {
                    return Err(Status::resource_exhausted("settings request limit reached"));
                }
                if self
                    .settings_requests
                    .get(&request.message_id)
                    .is_some_and(|(previous, _)| previous != &request)
                {
                    return Err(Status::already_exists(
                        "message ID reused with different settings",
                    ));
                }
            }
            _ => {}
        }
        self.requests
            .try_send(request.clone())
            .map_err(|_| Status::resource_exhausted("worker ingress is full or closed"))?;
        match &request.payload {
            Some(
                client_message::Payload::SetInferenceConfig(_)
                | client_message::Payload::SetApprovalMode(_),
            ) if !self.settings_requests.contains_key(&request.message_id) => {
                self.settings_bytes += request.encoded_len();
                self.settings_requests
                    .insert(request.message_id.clone(), (request, false));
            }
            Some(client_message::Payload::HostToolResult(result)) => {
                self.host_calls.remove(&result.call_id);
            }
            Some(client_message::Payload::DetachTools(_)) => {
                self.host_events = None;
                self.host_client.clear();
                self.fail_host_calls();
            }
            _ => {}
        }
        Ok(None)
    }

    fn fail_host_calls(&mut self) {
        self.failed_host_calls.extend(self.host_calls.drain());
    }

    fn observe(&mut self, event: SessionMessage) -> Result<(), Status> {
        if event.encoded_len() > MAX_FRAME_BYTES {
            return Err(Status::resource_exhausted(
                "worker event exceeded its bound",
            ));
        }
        match &event.payload {
            Some(session_message::Payload::HostToolCall(call)) => {
                if self.host_calls.len() + self.failed_host_calls.len() >= 128
                    || self.failed_host_calls.contains(&call.id)
                    || call.id.is_empty()
                    || !self.host_calls.insert(call.id.clone())
                {
                    return Err(Status::resource_exhausted(
                        "invalid or excessive host tool calls",
                    ));
                }
                if self
                    .host_events
                    .as_ref()
                    .is_none_or(|events| events.try_send(event.clone()).is_err())
                {
                    self.host_calls.remove(&call.id);
                    self.failed_host_calls.push_back(call.id.clone());
                }
                return Ok(());
            }
            Some(session_message::Payload::InputAck(response)) => {
                if let Some((request, acknowledged)) = self
                    .settings_requests
                    .get_mut(&response.message_id)
                    .filter(|(_, acknowledged)| !*acknowledged)
                {
                    *acknowledged = true;
                    self.settings
                        .send_modify(|settings| match &request.payload {
                            Some(client_message::Payload::SetInferenceConfig(config)) => {
                                let mut config = config.clone();
                                if config.model.is_empty() {
                                    config.model = settings
                                        .inference
                                        .as_ref()
                                        .map(|config| config.model.clone())
                                        .unwrap_or_default();
                                }
                                settings.inference = Some(config);
                            }
                            Some(client_message::Payload::SetApprovalMode(mode)) => {
                                settings.approval_mode = Some(mode.mode)
                            }
                            _ => {}
                        });
                }
            }
            _ => {}
        }
        let response_id = match &event.payload {
            Some(session_message::Payload::InputAck(ack)) => Some(&ack.message_id),
            Some(session_message::Payload::InputNack(nack)) => Some(&nack.message_id),
            _ => None,
        };
        if let Some(id) = response_id {
            if let Some(index) = self
                .pending_responses
                .iter()
                .position(|(pending, _)| pending == id)
            {
                let (_, reply) = self.pending_responses.remove(index);
                let _ = reply.send(Ok(Some(event)));
                return Ok(());
            }
        }
        self.projection.apply(&event)?;
        let event = Arc::new(event);
        for client in self
            .clients
            .values_mut()
            .filter(|client| client.pending.is_none())
        {
            if client.updates.push(event.clone()) {
                client.changes.send_replace(None);
            }
        }
        Ok(())
    }
}

async fn submit(
    session: &Session,
    attachment_id: &str,
    request: ClientMessage,
    local: &mpsc::Sender<SessionMessage>,
) -> bool {
    let message_id = request.message_id.clone();
    let response = match session.send(attachment_id, request).await {
        Ok(response) => response,
        // Losing a worker acknowledgement leaves the input unconfirmed, not rejected.
        Err(status) if status.code() == tonic::Code::Unavailable => return false,
        Err(status) => Some(SessionMessage {
            payload: Some(session_message::Payload::InputNack(InputNack {
                message_id,
                reason: if status.code() == tonic::Code::AlreadyExists {
                    InputNackReason::MessageIdConflict
                } else {
                    InputNackReason::InvalidRequest
                } as i32,
                error: status.message().to_owned(),
            })),
            ..SessionMessage::default()
        }),
    };
    match response {
        Some(response) => local.send(response).await.is_ok(),
        None => true,
    }
}

async fn read_history_page(
    client: &mut AgentRpcClient<Channel>,
    mut request: GetMessagesRequest,
) -> Result<GetMessagesResponse, Status> {
    request.limit = if request.limit == 0 {
        100
    } else {
        request.limit.min(128)
    };
    loop {
        match client.get_messages(request.clone()).await {
            Ok(response) => return Ok(response.into_inner()),
            // Tonic reports its encoding size limit as OutOfRange, not ResourceExhausted.
            Err(status) if status.code() == tonic::Code::OutOfRange && request.limit > 1 => {
                request.limit /= 2;
            }
            Err(status) => return Err(status),
        }
    }
}

async fn read_projection(
    client: &mut AgentRpcClient<Channel>,
    id: &str,
) -> Result<SessionProjection, Status> {
    let mut offset = 0;
    let (total, has_user) = loop {
        let page = read_history_page(
            client,
            GetMessagesRequest {
                session_id: id.to_owned(),
                offset,
                limit: 128,
            },
        )
        .await?;
        let has_user = page
            .messages
            .iter()
            .any(|entry| matches!(entry.chat_message(), Some(chat_record::Message::User(_))));
        offset += page.messages.len() as u64;
        if has_user || offset >= page.total {
            break (page.total, has_user);
        }
        if page.messages.is_empty() {
            return Err(Status::data_loss(
                "worker history pagination made no progress",
            ));
        }
    };
    offset = total.saturating_sub(128);
    let mut projection = SessionProjection::new(offset, has_user);
    while offset < total {
        let page = read_history_page(
            client,
            GetMessagesRequest {
                session_id: id.to_owned(),
                offset,
                limit: (total - offset).min(128) as u32,
            },
        )
        .await?;
        if page.messages.is_empty() {
            return Err(Status::data_loss(
                "worker history pagination made no progress",
            ));
        }
        offset += page.messages.len() as u64;
        for entry in page.messages {
            projection.record(entry);
        }
    }
    Ok(projection)
}

fn replay_start(replay: &[SessionMessage]) -> SessionMessage {
    let history_prefix = replay
        .iter()
        .find_map(|message| match &message.payload {
            Some(session_message::Payload::ReplayDone(done)) => Some(done.dropped_entries),
            _ => None,
        })
        .unwrap_or_default();
    SessionMessage {
        payload: Some(session_message::Payload::ReplayStart(
            mirror_protocol::ReplayStart { history_prefix },
        )),
        ..Default::default()
    }
}

fn state_restored() -> SessionMessage {
    SessionMessage {
        payload: Some(session_message::Payload::StateRestored(())),
        ..Default::default()
    }
}

fn closed() -> Status {
    Status::unavailable("session stopped; no automatic restart or replay of commands")
}
fn ack(message_id: &str) -> SessionMessage {
    SessionMessage {
        payload: Some(session_message::Payload::InputAck(InputAck {
            message_id: message_id.to_owned(),
        })),
        ..SessionMessage::default()
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use super::*;
    use assert_matches::assert_matches;
    use mirror_protocol::agent_rpc_server::{AgentRpc, AgentRpcServer};
    use mirror_protocol::{
        runtime_delta, AgentMessage, InferenceConfig, RuntimeDelta, StreamContent,
    };
    use pretty_assertions::assert_eq;
    use tokio::sync::Notify;
    use tonic::transport::Endpoint;
    use tonic::{Request, Response, Streaming};

    fn fixture() -> (
        Session,
        mpsc::Sender<Result<SessionMessage, Status>>,
        mpsc::Receiver<ClientMessage>,
    ) {
        let (commands, receiver) = mpsc::channel(32);
        let (requests, incoming) = mpsc::channel(32);
        let (events, responses) = mpsc::channel(1);
        let (settings, observed) = watch::channel(Workspace::default());
        let state = SessionState {
            id: "session".into(),
            projection: SessionProjection::default(),
            requests,
            commands: receiver,
            clients: HashMap::new(),
            settings,
            host_client: "host".into(),
            host_tools: Vec::new(),
            host_events: None,
            host_calls: HashSet::new(),
            failed_host_calls: VecDeque::new(),
            startup_messages: Vec::new(),
            executions: watch::channel(()).0,
            settings_requests: HashMap::new(),
            settings_bytes: 0,
            pending_responses: Vec::new(),
        };
        tokio::spawn(state.run(None, ReceiverStream::new(responses)));
        let channel = Endpoint::from_static("http://unused.invalid").connect_lazy();
        (
            Session {
                commands,
                client: AgentRpcClient::new(channel.clone()),
                lifecycle: MirrorRpcClient::new(channel),
                settings: observed,
            },
            events,
            incoming,
        )
    }

    async fn attachment(
        session: &Session,
        name: &str,
    ) -> (
        impl Stream<Item = Result<SessionMessage, Status>> + use<>,
        ExecutionLease,
    ) {
        let id = session
            .reserve(name.into(), Some(Vec::new()))
            .await
            .expect("reserve")
            .attachment_id;
        let lease = session.authorize(id.clone(), false).await.expect("lease");
        let stream = session
            .attach(
                id.clone(),
                tokio_stream::pending(),
                session.authorize(id, false).await.expect("stream lease"),
                AttachmentRecovery::InBand,
            )
            .await
            .expect("attach");
        (stream, lease)
    }

    #[tokio::test(start_paused = true)]
    async fn slow_attachment_coalesces_without_replay_or_interrupting_observer() {
        let (session, events, _requests) = fixture();
        let (slow, host) = attachment(&session, "host").await;
        let (healthy, observer) = attachment(&session, "observer").await;
        tokio::pin!(slow, healthy);
        assert_matches!(slow.next().await, Some(Ok(message)) if matches!(message.payload, Some(session_message::Payload::ReplayDone(_))));
        assert_matches!(healthy.next().await, Some(Ok(message)) if matches!(message.payload, Some(session_message::Payload::ReplayDone(_))));
        let chunk = SessionMessage {
            payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(runtime_delta::Payload::StreamContent(StreamContent {
                    content: "x".into(),
                })),
                ..Default::default()
            })),
            ..Default::default()
        };
        for _ in 0..2048 {
            events.send(Ok(chunk.clone())).await.expect("publish");
            assert_eq!(
                healthy.next().await.expect("observer").expect("event"),
                chunk
            );
        }
        let committed = SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: "committed".into(),
                ..Default::default()
            })),
            stream_final: true,
            ..Default::default()
        };
        events.send(Ok(committed.clone())).await.expect("commit");
        assert_eq!(
            healthy.next().await.expect("observer").expect("event"),
            committed
        );
        assert_eq!(
            slow.next().await.expect("slow reader").expect("commit"),
            committed
        );
        events.send(Ok(chunk.clone())).await.expect("live event");
        assert_eq!(slow.next().await.expect("resumed").expect("event"), chunk);
        assert!(!host.is_closed());
        assert!(!observer.is_closed());
        drop(host);
        drop(observer);
        // Closing waits for execution leases, not for display consumers.
        session.close().await;
    }

    #[tokio::test(start_paused = true)]
    async fn history_gap_replays_a_precise_prefix_and_pending_approval_then_live_updates() {
        let (session, events, _requests) = fixture();
        let (slow, host) = attachment(&session, "host").await;
        let (healthy, observer) = attachment(&session, "observer").await;
        tokio::pin!(slow, healthy);
        slow.next()
            .await
            .expect("initial replay")
            .expect("replay done");
        healthy
            .next()
            .await
            .expect("initial replay")
            .expect("replay done");
        let approval = SessionMessage {
            payload: Some(session_message::Payload::HitlRequest(
                mirror_protocol::HitlRequest {
                    call: Some(mirror_protocol::ToolCall {
                        id: "approval".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        events.send(Ok(approval.clone())).await.expect("approval");
        assert_eq!(
            healthy.next().await.expect("observer").expect("approval"),
            approval
        );
        for index in 0..1025 {
            let record = SessionMessage {
                payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                    id: index.to_string(),
                    ..Default::default()
                })),
                ..Default::default()
            };
            events.send(Ok(record.clone())).await.expect("commit");
            assert_eq!(
                healthy.next().await.expect("observer").expect("record"),
                record
            );
        }
        assert_matches!(slow.next().await, Some(Ok(message)) if matches!(&message.payload,
            Some(session_message::Payload::ReplayStart(start)) if start.history_prefix == 897));
        let live = SessionMessage {
            payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(runtime_delta::Payload::TurnStarted(())),
                ..Default::default()
            })),
            ..Default::default()
        };
        events
            .send(Ok(live.clone()))
            .await
            .expect("arrives during replay");
        assert_eq!(healthy.next().await.expect("observer").expect("live"), live);
        for index in 897..1025 {
            assert_matches!(slow.next().await, Some(Ok(message)) if matches!(&message.payload,
                Some(session_message::Payload::AgentMessage(record)) if record.id == index.to_string()));
        }
        assert_matches!(slow.next().await, Some(Ok(message)) if matches!(&message.payload,
            Some(session_message::Payload::ReplayDone(done)) if done.dropped_entries == 897));
        assert_eq!(
            slow.next().await.expect("replay").expect("approval"),
            approval
        );
        assert_matches!(slow.next().await, Some(Ok(message)) if matches!(&message.payload,
            Some(session_message::Payload::StateRestored(_))));
        assert_eq!(slow.next().await.expect("resumed").expect("live"), live);
        assert!(!host.is_closed());
        assert!(!observer.is_closed());
        drop(host);
        drop(observer);
        session.close().await;
    }

    #[tokio::test]
    async fn upstream_failure_is_preserved_and_releases_execution_authority() {
        let (session, events, _requests) = fixture();
        let (stream, mut lease) = attachment(&session, "host").await;
        tokio::pin!(stream);
        stream
            .next()
            .await
            .expect("initial replay")
            .expect("replay done");
        events
            .send(Err(Status::resource_exhausted(
                "agent event stream lagged by 42 events",
            )))
            .await
            .expect("upstream failure");
        assert_matches!(stream.next().await, Some(Err(error))
            if error.code() == tonic::Code::ResourceExhausted
                && error.message() == "agent event stream lagged by 42 events");
        lease.closed().await;
        assert!(session.is_closed());
        drop(lease);
        session.close().await;
    }

    fn committed(id: &str) -> SessionMessage {
        SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: id.into(),
                ..Default::default()
            })),
            stream_final: true,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn upstream_eof_drains_the_mailbox_after_revoking_execution_authority() {
        let (session, events, _requests) = fixture();
        let (stream, mut lease) = attachment(&session, "host").await;
        tokio::pin!(stream);
        stream
            .next()
            .await
            .expect("initial replay")
            .expect("replay done");
        events.send(Ok(committed("last"))).await.expect("commit");
        events
            .send(Ok(SessionMessage {
                payload: Some(session_message::Payload::HostToolCall(
                    mirror_protocol::ToolCall {
                        id: "must not execute after shutdown".into(),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }))
            .await
            .expect("pending host call");
        drop(events);
        lease.closed().await;
        assert!(session.is_closed());
        assert_eq!(
            stream.next().await.expect("final output").expect("commit"),
            committed("last")
        );
        assert_matches!(stream.next().await, Some(Err(error)) if error.code() == tonic::Code::Unavailable);
        drop(lease);
        session.close().await;
    }

    #[tokio::test(start_paused = true)]
    async fn upstream_error_drains_the_inflight_batch_before_the_last_mailbox() {
        let (session, events, _requests) = fixture();
        let (slow, mut host) = attachment(&session, "host").await;
        let (healthy, mut observer) = attachment(&session, "observer").await;
        tokio::pin!(slow, healthy);
        slow.next()
            .await
            .expect("initial replay")
            .expect("replay done");
        healthy
            .next()
            .await
            .expect("initial replay")
            .expect("replay done");
        events
            .send(Ok(committed("first")))
            .await
            .expect("first commit");
        assert_eq!(
            healthy.next().await.expect("observer").expect("commit"),
            committed("first")
        );
        events
            .send(Ok(committed("second")))
            .await
            .expect("second commit");
        assert_eq!(
            healthy.next().await.expect("observer").expect("commit"),
            committed("second")
        );
        assert_eq!(
            slow.next().await.expect("inflight batch").expect("commit"),
            committed("first")
        );
        events
            .send(Ok(committed("third")))
            .await
            .expect("queued after drain");
        events
            .send(Err(Status::internal("worker failed")))
            .await
            .expect("upstream error");
        host.closed().await;
        observer.closed().await;
        assert_eq!(
            slow.next().await.expect("inflight batch").expect("commit"),
            committed("second")
        );
        assert_eq!(
            slow.next().await.expect("final mailbox").expect("commit"),
            committed("third")
        );
        assert_matches!(slow.next().await, Some(Err(error)) if error.message() == "worker failed");
        drop(host);
        drop(observer);
        session.close().await;
    }

    #[tokio::test]
    async fn shutdown_delivers_ack_without_rejecting_an_unconfirmed_command() {
        let (session, events, mut requests) = fixture();
        let id = session
            .reserve("host".into(), Some(Vec::new()))
            .await
            .expect("reserve")
            .attachment_id;
        let mut lease = session.authorize(id.clone(), false).await.expect("lease");
        let (incoming, receiver) = mpsc::channel(1);
        let stream = session
            .attach(
                id.clone(),
                ReceiverStream::new(receiver),
                session.authorize(id, false).await.expect("stream lease"),
                AttachmentRecovery::InBand,
            )
            .await
            .expect("attach");
        tokio::pin!(stream);
        stream
            .next()
            .await
            .expect("initial replay")
            .expect("replay done");
        incoming
            .send(Ok(ClientMessage {
                session_id: "session".into(),
                message_id: "accepted".into(),
                payload: Some(client_message::Payload::Interrupt(Default::default())),
            }))
            .await
            .expect("first command");
        assert_eq!(
            requests.recv().await.expect("worker request").message_id,
            "accepted"
        );
        events
            .send(Ok(ack("accepted")))
            .await
            .expect("worker reply");
        incoming
            .send(Ok(ClientMessage {
                session_id: "session".into(),
                message_id: "unconfirmed".into(),
                payload: Some(client_message::Payload::Interrupt(Default::default())),
            }))
            .await
            .expect("second command");
        assert_eq!(
            requests
                .recv()
                .await
                .expect("prior reply is queued")
                .message_id,
            "unconfirmed"
        );
        drop(events);
        lease.closed().await;
        assert_eq!(
            stream.next().await.expect("queued reply").expect("ack"),
            ack("accepted")
        );
        assert_matches!(stream.next().await, Some(Err(error)) if error.code() == tonic::Code::Unavailable);
        drop(lease);
        session.close().await;
    }

    #[tokio::test]
    async fn replies_and_host_calls_bypass_pending_replay_records() {
        let (session, events, mut requests) = fixture();
        events
            .send(Ok(SessionMessage {
                payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                    id: "history".into(),
                    ..Default::default()
                })),
                ..Default::default()
            }))
            .await
            .expect("history");
        events
            .send(Ok(ack("history observed")))
            .await
            .expect("history barrier");
        let id = session
            .reserve("host".into(), Some(Vec::new()))
            .await
            .expect("reserve")
            .attachment_id;
        let (incoming, receiver) = mpsc::channel(1);
        let stream = session
            .attach(
                id.clone(),
                ReceiverStream::new(receiver),
                session.authorize(id, false).await.expect("lease"),
                AttachmentRecovery::InBandWithHistory,
            )
            .await
            .expect("attach");
        tokio::pin!(stream);
        assert_matches!(stream.next().await, Some(Ok(message)) if matches!(&message.payload,
            Some(session_message::Payload::ReplayStart(_))));
        incoming
            .send(Ok(ClientMessage {
                session_id: "session".into(),
                message_id: "first".into(),
                payload: Some(client_message::Payload::Interrupt(Default::default())),
            }))
            .await
            .expect("command");
        assert_eq!(
            requests.recv().await.expect("worker request").message_id,
            "first"
        );
        events.send(Ok(ack("first"))).await.expect("worker reply");
        incoming
            .send(Ok(ClientMessage {
                session_id: "session".into(),
                message_id: "second".into(),
                payload: Some(client_message::Payload::Interrupt(Default::default())),
            }))
            .await
            .expect("reply barrier");
        assert_eq!(
            requests
                .recv()
                .await
                .expect("prior reply is queued")
                .message_id,
            "second"
        );
        assert_eq!(
            stream.next().await.expect("reply").expect("accepted"),
            ack("first")
        );
        let call = SessionMessage {
            payload: Some(session_message::Payload::HostToolCall(
                mirror_protocol::ToolCall {
                    id: "host call".into(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        events.send(Ok(call.clone())).await.expect("host call");
        events
            .send(Ok(ack("second")))
            .await
            .expect("host delivery barrier");
        assert_eq!(
            stream.next().await.expect("host delivery").expect("call"),
            call
        );
        incoming
            .send(Ok(ClientMessage {
                session_id: "session".into(),
                message_id: "result".into(),
                payload: Some(client_message::Payload::HostToolResult(HostToolResult {
                    call_id: "host call".into(),
                    success: true,
                    ..Default::default()
                })),
            }))
            .await
            .expect("host result");
        assert_matches!(requests.recv().await, Some(request) if matches!(&request.payload,
            Some(client_message::Payload::HostToolResult(result)) if result.success && result.call_id == "host call"));
        session.close().await;
    }

    #[tokio::test]
    async fn unavailable_host_results_wait_for_worker_capacity_without_stopping_it() {
        let (session, events, requests) = fixture();
        session.reserve("host".into(), None).await.expect("reserve");
        for index in 0..64 {
            events
                .send(Ok(SessionMessage {
                    payload: Some(session_message::Payload::HostToolCall(
                        mirror_protocol::ToolCall {
                            id: index.to_string(),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }))
                .await
                .expect("worker remains connected");
        }
        let replies: Vec<_> = ReceiverStream::new(requests).take(64).collect().await;
        assert_eq!(replies.len(), 64);
        assert_matches!(&replies[63].payload, Some(client_message::Payload::HostToolResult(result)) if result.call_id == "63" && !result.success);
        assert!(!session.is_closed());
        session.close().await;
    }

    #[tokio::test]
    async fn command_rejection_is_not_part_of_the_lossy_display_broadcast() {
        let (session, events, mut requests) = fixture();
        let id = session
            .reserve("host".into(), None)
            .await
            .expect("reserve")
            .attachment_id;
        let sender = session.clone();
        let pending = tokio::spawn(async move {
            sender
                .send(
                    &id,
                    ClientMessage {
                        session_id: "session".into(),
                        message_id: "command".into(),
                        payload: Some(client_message::Payload::Interrupt(Default::default())),
                    },
                )
                .await
        });
        assert_eq!(
            requests.recv().await.expect("admitted").message_id,
            "command"
        );
        let rejection = SessionMessage {
            payload: Some(session_message::Payload::InputNack(InputNack {
                message_id: "command".into(),
                error: "declined".into(),
                ..Default::default()
            })),
            ..Default::default()
        };
        events.send(Ok(rejection.clone())).await.expect("reply");
        assert_eq!(
            pending.await.expect("sender").expect("reply"),
            Some(rejection)
        );
        session.close().await;
    }

    struct RestartingServer(Arc<Notify>);

    #[tonic::async_trait]
    impl AgentRpc for RestartingServer {
        type SessionStream = Pin<Box<dyn Stream<Item = Result<SessionMessage, Status>> + Send>>;

        async fn session(
            &self,
            request: Request<Streaming<ClientMessage>>,
        ) -> Result<Response<Self::SessionStream>, Status> {
            let mut incoming = request.into_inner();
            let restart = self.0.clone();
            Ok(Response::new(Box::pin(async_stream::try_stream! {
                loop {
                    let message = tokio::select! {
                        message = incoming.message() => message,
                        () = restart.notified() => Ok(None),
                    }?;
                    let Some(message) = message else { break };
                    yield ack(&message.message_id);
                }
                Err::<(), _>(Status::unavailable("server is restarting; reconnect"))?;
            })))
        }

        async fn get_messages(
            &self,
            _: Request<GetMessagesRequest>,
        ) -> Result<Response<GetMessagesResponse>, Status> {
            Ok(Response::new(GetMessagesResponse::default()))
        }

        async fn call_tool(
            &self,
            _: Request<CallToolRequest>,
        ) -> Result<Response<CallToolResponse>, Status> {
            Err(Status::unimplemented("no tools"))
        }
    }

    #[tokio::test]
    async fn remote_stream_end_reaches_attached_clients_with_its_status() {
        let directory = tempfile::tempdir().expect("socket directory");
        let socket = directory.path().join("server.sock");
        let restart = Arc::new(Notify::new());
        let listener = tokio::net::UnixListener::bind(&socket).expect("listener");
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(RestartingServer(restart.clone())))
                .serve_with_incoming(crate::local_incoming(listener)),
        );
        let channel = crate::local_channel(socket).await.expect("channel");
        let session = Session::start(
            None,
            channel,
            Workspace {
                session_id: "session".into(),
                ..Default::default()
            },
            AgentConfiguration {
                inference: Some(InferenceConfig::default()),
                ..Default::default()
            },
            "client".into(),
        )
        .await
        .expect("session");
        let attachment = session
            .reserve("client".into(), None)
            .await
            .expect("reservation")
            .attachment_id;
        let lease = session
            .authorize(attachment.clone(), false)
            .await
            .expect("lease");
        let events = session
            .attach(
                attachment,
                tokio_stream::pending(),
                lease,
                AttachmentRecovery::Reconnect,
            )
            .await
            .expect("attachment");
        tokio::pin!(events);
        restart.notify_one();
        let ended = loop {
            match events.next().await {
                Some(Ok(_)) => continue,
                Some(Err(status)) => break status,
                None => panic!("stream ended without a status"),
            }
        };

        assert_eq!(ended.code(), tonic::Code::Unavailable);
        assert_eq!(ended.message(), "server is restarting; reconnect");
        server.abort();
    }
}
