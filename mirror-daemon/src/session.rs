//! One worker connection, attachment authority and bounded event fanout.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use futures_core::Stream;
use mirror_protocol::agent_rpc_client::AgentRpcClient;
use mirror_protocol::daemon::{
    AgentConfiguration, AttachWorkspaceResponse, GetConversationMessagesResponse, Workspace,
};
use mirror_protocol::mirror::{
    mirror_rpc_client::MirrorRpcClient, ForkSessionRequest, ForkSessionResponse,
    ForkWorkspaceRequest, ForkWorkspaceResponse,
};
use mirror_protocol::{
    chat_record, client_message, session_message, CallToolRequest, CallToolResponse, ClientMessage,
    GetMessagesRequest, GetMessagesResponse, HostToolDefinition, HostToolResult, InferenceConfig,
    InputAck, InputNack, InputNackReason, SessionMessage, SetApprovalMode,
};
use prost::Message;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::transport::Channel;
use tonic::Status;
use uuid::Uuid;

use crate::{
    history_page, history_page_limit, AgentProcess, AgentSpawner, DisplayUpdates, ExecutionLease,
    SessionProjection, MAX_FRAME_BYTES,
};

/// A route to one agent; durable history and command deduplication belong to the agent.
#[derive(Clone, Debug)]
pub struct Session {
    commands: mpsc::Sender<Command>,
    client: AgentRpcClient<Channel>,
    lifecycle: MirrorRpcClient<Channel>,
    settings: watch::Receiver<Workspace>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExistingSettings {
    Replace,
    Keep,
}

/// Replay and lag handling negotiated by the attachment consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentRecovery {
    Reconnect,
    InBand,
    InBandWithHistory,
    InBandWithSnapshot,
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
    Snapshot(
        u32,
        oneshot::Sender<Result<GetConversationMessagesResponse, Status>>,
    ),
    Reserve(
        String,
        Option<Vec<HostToolDefinition>>,
        Option<String>,
        oneshot::Sender<Result<AttachWorkspaceResponse, Status>>,
    ),
    Attach(String, oneshot::Sender<Result<Attachment, Status>>),
    Updates(String, oneshot::Sender<Result<DisplayBatch, Status>>),
    Send(String, Box<ClientMessage>, CommandReply),
    RefreshMcp(
        mirror_protocol::McpConnections,
        oneshot::Sender<Result<(), Status>>,
    ),
    Authorize(
        String,
        bool,
        oneshot::Sender<Result<ExecutionLease, Status>>,
    ),
    ReleaseReservation(String, oneshot::Sender<()>),
    Close(oneshot::Sender<()>),
}

struct SessionState {
    daemon_configuration: bool,
    id: String,
    spawner: Option<AgentSpawner>,
    daemon_calls: tokio::task::JoinSet<HostToolResult>,
    daemon_results: VecDeque<HostToolResult>,
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
    #[allow(clippy::too_many_arguments)] // Keep transport, configuration and host capabilities explicit at startup.
    pub async fn start(
        mut agent: Option<AgentProcess>,
        channel: Channel,
        mut workspace: Workspace,
        configuration: AgentConfiguration,
        host_client: String,
        daemon_configuration: bool,
        spawner: Option<AgentSpawner>,
        snapshots: Option<crate::SpawnHandle>,
        existing: ExistingSettings,
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
            let keep = existing == ExistingSettings::Keep && !projection.is_empty();
            let personal_context = match &snapshots {
                Some(snapshots) if projection.is_empty() => {
                    Some(snapshots.personal_context(workspace.id.clone()).await?)
                }
                _ => None,
            };
            let configuration_id = format!("daemon:configure:{}", Uuid::new_v4());
            let approval_id = format!("{configuration_id}:approval");
            let inspect = ClientMessage {
                trace_parent: String::new(),
                session_id: id.clone(),
                message_id: configuration_id.clone(),
                personal_context: None,
                payload: Some(client_message::Payload::InspectSettings(())),
            };
            let mut attempt = 0;
            let (mut requests, mut responses) = if keep {
                open_inspection(&mut client, &inspect, &mut attempt).await?
            } else {
                open_replacing(
                    &mut client,
                    ClientMessage {
                        trace_parent: String::new(),
                        session_id: id.clone(),
                        message_id: configuration_id.clone(),
                        personal_context,
                        payload: Some(client_message::Payload::SetInferenceConfig(
                            inference.clone(),
                        )),
                    },
                    &approval_id,
                    configuration.approval_mode,
                )
                .await?
            };
            let mut pushed = !keep;
            let mut awaited = if pushed {
                approval_id.clone()
            } else {
                configuration_id.clone()
            };
            let mut live = None;
            loop {
                let event = match responses.message().await {
                    Err(status)
                        if !pushed && live.is_none() && inspection_retryable(&status, attempt) =>
                    {
                        tokio::time::sleep(inspection_backoff(attempt)).await;
                        attempt += 1;
                        (requests, responses) =
                            open_inspection(&mut client, &inspect, &mut attempt).await?;
                        continue;
                    }
                    event => event?.ok_or_else(closed)?,
                };
                match &event.payload {
                    Some(session_message::Payload::InputAck(ack)) if ack.message_id == awaited => {
                        break
                    }
                    Some(session_message::Payload::SessionSettings(settings)) if !pushed => {
                        live = Some(settings.clone());
                    }
                    Some(session_message::Payload::InputNack(nack))
                        if !pushed
                            && nack.message_id == configuration_id
                            && nack.reason() == InputNackReason::InvalidRequest =>
                    {
                        let inference_id = format!("{configuration_id}:inference");
                        push_settings(
                            &requests,
                            &id,
                            Some((&inference_id, &inference)),
                            &approval_id,
                            configuration.approval_mode,
                        )
                        .await?;
                        pushed = true;
                        awaited = approval_id.clone();
                    }
                    Some(session_message::Payload::InputNack(nack))
                        if !pushed
                            && live.is_none()
                            && nack.message_id == configuration_id
                            && nack.reason() == InputNackReason::TargetNotFound =>
                    {
                        if attempt >= INSPECTION_RETRIES {
                            return Err(Status::unavailable(
                                "worker closed while reporting settings; retry",
                            ));
                        }
                        tokio::time::sleep(inspection_backoff(attempt)).await;
                        attempt += 1;
                        (requests, responses) =
                            open_inspection(&mut client, &inspect, &mut attempt).await?;
                    }
                    Some(
                        session_message::Payload::InputNack(_)
                        | session_message::Payload::HostToolCall(_),
                    ) => return Err(Status::failed_precondition("worker rejected startup")),
                    _ => {
                        projection.apply(&event)?;
                    }
                }
            }
            // Initialization commits before subscription; retain runtime while refreshing history.
            projection.replace_history(read_projection(&mut client, &id).await?);
            let settings = if !pushed {
                let live = live.ok_or_else(|| {
                    Status::failed_precondition("worker acknowledged startup without settings")
                })?;
                (live.inference, Some(live.approval_mode))
            } else {
                (Some(inference), Some(configuration.approval_mode))
            };
            Ok::<_, Status>((projection, requests, responses, settings))
        };
        // Hosted configuration includes repository and sandbox preparation.
        let timeout = Duration::from_secs(if snapshots.is_some() { 15 * 60 } else { 30 });
        let opened = tokio::time::timeout(timeout, opened)
            .await
            .unwrap_or_else(|_| Err(Status::deadline_exceeded("worker configuration timed out")));
        let (projection, requests, responses, (inference, approval_mode)) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                if let Some(agent) = agent.as_mut() {
                    agent.close().await;
                }
                return Err(error);
            }
        };
        workspace.inference = inference;
        workspace.approval_mode = approval_mode;
        let (commands, receiver) = mpsc::channel(32);
        let (settings, observed_settings) = watch::channel(workspace);
        let state = SessionState {
            daemon_configuration,
            id,
            spawner: spawner.clone(),
            daemon_calls: tokio::task::JoinSet::new(),
            daemon_results: VecDeque::new(),
            projection,
            requests,
            commands: receiver,
            settings,
            host_client,
            host_tools: configuration
                .host_tools
                .into_iter()
                .filter(|tool| spawner.is_none() || tool.name != "spawn_agent")
                .collect(),
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

    /// Read display state without reserving an attachment or sending worker commands.
    pub async fn snapshot(&self, limit: u32) -> Result<GetConversationMessagesResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::Snapshot(limit, reply))
            .await
            .map_err(|_| closed())?;
        result.await.map_err(|_| closed())?
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
        replaces_attachment_id: Option<String>,
    ) -> Result<AttachWorkspaceResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::Reserve(
                client_id,
                host_tools,
                replaces_attachment_id,
                reply,
            ))
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

    /// Submit exactly one initial prompt; a missing acknowledgement stays unknown.
    pub async fn submit_prompt(
        &self,
        attachment_id: &str,
        input_id: String,
        content: String,
    ) -> Result<bool, Status> {
        let request = ClientMessage {
            personal_context: None,
            trace_parent: String::new(),
            session_id: self.settings().session_id,
            message_id: input_id.clone(),
            payload: Some(client_message::Payload::Message(Box::new(
                mirror_protocol::AgentMessage {
                    id: input_id,
                    payload: Some(mirror_protocol::agent_message::Payload::Chat(
                        mirror_protocol::ChatRecord {
                            message: Some(chat_record::Message::User(
                                mirror_protocol::UserMessage {
                                    content,
                                    ..Default::default()
                                },
                            )),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
            ))),
        };
        match self
            .send(attachment_id, request)
            .await?
            .and_then(|event| event.payload)
        {
            Some(session_message::Payload::InputAck(_)) => Ok(true),
            Some(session_message::Payload::InputNack(_)) => Ok(false),
            _ => Err(Status::unavailable(
                "initial prompt acknowledgement is unknown",
            )),
        }
    }

    /// Relinquish a daemon's startup reservation without taking a frontend's live stream.
    pub async fn release_reservation(&self, id: String) {
        let (reply, result) = oneshot::channel();
        if self
            .commands
            .send(Command::ReleaseReservation(id, reply))
            .await
            .is_ok()
        {
            let _ = result.await;
        }
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

    /// Read the hosted log immediately before a display cursor without replaying the session.
    pub async fn hosted_history_before(
        &self,
        session_id: &str,
        before: u64,
        limit: u32,
    ) -> Result<GetConversationMessagesResponse, Status> {
        // Hosted sequences start at one; GetMessages offsets count preceding records.
        let mut end = before.saturating_sub(1).min(i64::MAX as u64);
        let mut count = end.min(u64::from(history_page_limit(limit)));
        while count > 0 {
            let offset = end - count;
            let page = self
                .get_messages(GetMessagesRequest {
                    session_id: session_id.to_owned(),
                    offset,
                    limit: count as u32,
                })
                .await?;
            if page.total < end {
                // A caller may supply any exclusive upper bound, including one past the head.
                end = page.total;
                count = count.min(end);
                continue;
            }
            let received = page.messages.len() as u64;
            if received == 0
                || received > count
                || page
                    .messages
                    .iter()
                    .enumerate()
                    .any(|(i, entry)| entry.sequence != Some(offset + i as u64 + 1))
            {
                return Err(Status::data_loss("incomplete hosted history page"));
            }
            if received < count {
                // A size-limited forward read returns the prefix. Re-anchor the suffix at the cursor.
                count = received;
                continue;
            }
            return history_page(page.messages.into_iter().rev(), offset, limit);
        }
        Ok(GetConversationMessagesResponse::default())
    }

    pub async fn refresh_mcp(
        &self,
        connections: mirror_protocol::McpConnections,
    ) -> Result<(), Status> {
        let (reply, result) = oneshot::channel();
        self.commands
            .send(Command::RefreshMcp(connections, reply))
            .await
            .map_err(|_| closed())?;
        result.await.map_err(|_| closed())?
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
            let start = replay_start(&replay, recovery == AttachmentRecovery::InBandWithSnapshot);
            let mut pending: VecDeque<_> = replay.into_iter().map(Arc::new).collect();
            if matches!(recovery, AttachmentRecovery::InBandWithHistory | AttachmentRecovery::InBandWithSnapshot) {
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
                        pending.push_back(Arc::new(replay_start(&replay, false)));
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
                Some(result) = self.daemon_calls.join_next(), if !self.daemon_calls.is_empty() => {
                    match result {
                        Ok(result) => self.daemon_results.push_back(result),
                        Err(_) => break Status::internal("daemon tool task failed"),
                    }
                }
                permit = outgoing.reserve(), if !self.daemon_results.is_empty() => {
                    let Ok(permit) = permit else { break closed(); };
                    if let Some(result) = self.daemon_results.pop_front() {
                        permit.send(ClientMessage {
                            trace_parent: String::new(),
                            personal_context: None,
                            session_id: self.id.clone(),
                            message_id: format!("daemon:tool-result:{}", result.call_id),
                            payload: Some(client_message::Payload::HostToolResult(result)),
                        });
                    }
                }
                _ = cleanup.tick() => {
                    if !attached && Instant::now() >= startup_deadline { break closed(); }
                    if attached { self.prune_clients(); }
                },
                permit = outgoing.reserve(), if !self.failed_host_calls.is_empty() => {
                    let Ok(permit) = permit else { break closed(); };
                    if let Some(call_id) = self.failed_host_calls.pop_front() {
                        permit.send(ClientMessage {
                            trace_parent: String::new(),
                            personal_context: None,
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
                    Some(Command::Snapshot(limit, reply)) => { let _ = reply.send(self.projection.snapshot(limit)); }
                    Some(Command::Reserve(client_id, host_tools, replaces_attachment_id, reply)) => {
                        let result = self.reserve(client_id, host_tools, replaces_attachment_id);
                        attached |= result.is_ok();
                        let _ = reply.send(result);
                    }
                    Some(Command::Attach(id, reply)) => { let _ = reply.send(self.attach(&id)); }
                    Some(Command::Updates(id, reply)) => { let _ = reply.send(self.updates(&id)); }
                    Some(Command::Send(id, mut request, reply)) => {
                        request.personal_context = None;
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
                    Some(Command::RefreshMcp(connections, reply)) => {
                        let result = if self.daemon_configuration && self.settings.borrow().kind != mirror_protocol::daemon::WorkspaceKind::Remote as i32 {
                            self.requests.try_send(ClientMessage {
                                trace_parent: String::new(),
                                personal_context: None,
                                session_id: self.id.clone(),
                                message_id: format!("daemon:mcp:{}", Uuid::new_v4()),
                                payload: Some(client_message::Payload::SetMcpConnections(connections)),
                            }).map_err(|_| Status::resource_exhausted("worker ingress is full or closed"))
                        } else { Ok(()) };
                        let _ = reply.send(result);
                    }
                    Some(Command::Authorize(id, host_only, reply)) => { let _ = reply.send(self.authorize(&id, host_only)); }
                    Some(Command::ReleaseReservation(id, reply)) => {
                        if self.clients.get(&id).is_some_and(|client| client.pending.is_some()) {
                            self.clients.remove(&id);
                            self.prune_clients();
                        }
                        let _ = reply.send(());
                    }
                    Some(Command::Close(reply)) => { closing = Some(reply); break Status::cancelled("workspace closed"); }
                    None => break closed(),
                }
            }
        };
        self.executions.send_replace(());
        self.daemon_calls.shutdown().await;
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
        replaces_attachment_id: Option<String>,
    ) -> Result<AttachWorkspaceResponse, Status> {
        self.prune_clients();
        let replaced = replaces_attachment_id
            .as_deref()
            .filter(|id| self.clients.contains_key(*id));
        let replaces_host = replaced
            .and_then(|id| self.clients.get(id))
            .is_some_and(|client| client.id == self.host_client);
        if replaces_host && host_tools.as_ref() != Some(&self.host_tools) {
            return Err(Status::failed_precondition("execution host tools changed"));
        }
        if self.clients.len() >= 16 && replaced.is_none() {
            return Err(Status::resource_exhausted("attachment limit reached"));
        }
        if self.clients.values().any(|client| client.id == client_id) {
            return Err(Status::already_exists("client already attached"));
        }
        if let Some(id) = replaced {
            self.clients.remove(id);
        }
        if replaces_host {
            self.host_events = None;
            self.host_client.clear();
            // Never replay calls that may already have executed on the replaced host.
            self.fail_host_calls();
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
            Some(client_message::Payload::SetMcpConnections(_))
                if client.id != self.host_client || self.daemon_configuration =>
            {
                return Err(Status::permission_denied(
                    "MCP configuration belongs to the worker's host",
                ));
            }
            Some(client_message::Payload::HostToolResult(result))
                if client.id != self.host_client || !self.host_calls.contains(&result.call_id) =>
            {
                return Err(Status::permission_denied(
                    "tool result does not belong to this client",
                ));
            }
            Some(client_message::Payload::DetachTools(_))
                if self.spawner.is_some() && client.id == self.host_client =>
            {
                // The frontend owns no daemon tool implementations to detach.
                return Ok(Some(ack(&request.message_id)));
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
                    .is_some_and(|(previous, _)| !same_settings(previous, &request))
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

    fn observe(&mut self, mut event: SessionMessage) -> Result<(), Status> {
        if event.encoded_len() > MAX_FRAME_BYTES {
            return Err(Status::resource_exhausted(
                "worker event exceeded its bound",
            ));
        }
        if let Some(session_message::Payload::ToolInventory(inventory)) = &mut event.payload {
            for tool in &mut inventory.tools {
                if tool.source != "host" || tool.parent.is_some() {
                    continue;
                }
                if tool.name == "spawn_agent" && self.spawner.is_some() {
                    tool.source = "builtin".into();
                } else if self.host_events.is_none() {
                    tool.unavailable_reason = Some("The owning desktop is disconnected.".into());
                }
            }
        }
        match &event.payload {
            Some(session_message::Payload::HostToolCall(call)) => {
                if let Some(spawner) = self.spawner.as_ref().filter(|_| {
                    call.function
                        .as_ref()
                        .is_some_and(|function| function.name == "spawn_agent")
                }) {
                    if call.id.is_empty()
                        || self.daemon_calls.len() + self.daemon_results.len() >= 8
                    {
                        return Err(Status::resource_exhausted(
                            "invalid or excessive daemon tool calls",
                        ));
                    }
                    let spawner = spawner.clone();
                    let call = call.clone();
                    let lifetime = self.executions.subscribe();
                    self.daemon_calls
                        .spawn(async move { spawner.call(call, lifetime).await });
                    return Ok(());
                }
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
        if !self.projection.apply(&event)? {
            return Ok(());
        }
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

async fn open_session(
    client: &mut AgentRpcClient<Channel>,
    first: ClientMessage,
) -> Result<
    (
        mpsc::Sender<ClientMessage>,
        tonic::Streaming<SessionMessage>,
    ),
    Status,
> {
    let (requests, incoming) = mpsc::channel(32);
    requests.send(first).await.map_err(|_| closed())?;
    // A server may wait for the first message before sending response headers.
    let responses = client
        .session(ReceiverStream::new(incoming))
        .await?
        .into_inner();
    Ok((requests, responses))
}

const INSPECTION_RETRIES: u32 = 4;

fn inspection_retryable(status: &Status, attempt: u32) -> bool {
    status.code() == tonic::Code::Unavailable && attempt < INSPECTION_RETRIES
}

fn inspection_backoff(attempt: u32) -> Duration {
    Duration::from_millis(250 << attempt)
}

async fn open_inspection(
    client: &mut AgentRpcClient<Channel>,
    inspect: &ClientMessage,
    attempt: &mut u32,
) -> Result<
    (
        mpsc::Sender<ClientMessage>,
        tonic::Streaming<SessionMessage>,
    ),
    Status,
> {
    loop {
        match open_session(client, inspect.clone()).await {
            Err(status) if inspection_retryable(&status, *attempt) => {
                tokio::time::sleep(inspection_backoff(*attempt)).await;
                *attempt += 1;
            }
            opened => return opened,
        }
    }
}

async fn open_replacing(
    client: &mut AgentRpcClient<Channel>,
    first: ClientMessage,
    approval_id: &str,
    approval_mode: i32,
) -> Result<
    (
        mpsc::Sender<ClientMessage>,
        tonic::Streaming<SessionMessage>,
    ),
    Status,
> {
    let session_id = first.session_id.clone();
    let (requests, responses) = open_session(client, first).await?;
    push_settings(&requests, &session_id, None, approval_id, approval_mode).await?;
    Ok((requests, responses))
}

async fn push_settings(
    requests: &mpsc::Sender<ClientMessage>,
    id: &str,
    inference: Option<(&str, &InferenceConfig)>,
    approval_id: &str,
    approval_mode: i32,
) -> Result<(), Status> {
    let inference = inference.map(|(message_id, inference)| {
        (
            message_id,
            client_message::Payload::SetInferenceConfig(inference.clone()),
        )
    });
    let approval = (
        approval_id,
        client_message::Payload::SetApprovalMode(SetApprovalMode {
            mode: approval_mode,
        }),
    );
    for (message_id, payload) in inference.into_iter().chain([approval]) {
        requests
            .send(ClientMessage {
                trace_parent: String::new(),
                session_id: id.to_owned(),
                message_id: message_id.to_owned(),
                personal_context: None,
                payload: Some(payload),
            })
            .await
            .map_err(|_| closed())?;
    }
    Ok(())
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

fn replay_start(replay: &[SessionMessage], tail_only: bool) -> SessionMessage {
    let history_prefix = replay
        .iter()
        .find_map(|message| match &message.payload {
            Some(session_message::Payload::ReplayDone(done)) => Some(done.dropped_entries),
            _ => None,
        })
        .unwrap_or_default();
    let next_before_sequence = replay
        .iter()
        .find_map(|message| match &message.payload {
            Some(session_message::Payload::AgentMessage(entry)) => entry.sequence,
            _ => None,
        })
        .filter(|_| history_prefix > 0);
    let tail_only = tail_only && (history_prefix == 0 || next_before_sequence.is_some());
    SessionMessage {
        payload: Some(session_message::Payload::ReplayStart(
            mirror_protocol::ReplayStart {
                history_prefix: if tail_only { 0 } else { history_prefix },
                includes_runtime: true,
                tail_only,
                next_before_sequence: if tail_only {
                    next_before_sequence
                } else {
                    None
                },
            },
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
fn same_settings(previous: &ClientMessage, request: &ClientMessage) -> bool {
    *previous
        == ClientMessage {
            trace_parent: previous.trace_parent.clone(),
            ..request.clone()
        }
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
        runtime_delta, AgentMessage, ApprovalMode, RuntimeDelta, SessionSettings, StreamContent,
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
        fixture_with_spawner(None, false)
    }

    fn fixture_with_spawner(
        spawner: Option<AgentSpawner>,
        daemon_configuration: bool,
    ) -> (
        Session,
        mpsc::Sender<Result<SessionMessage, Status>>,
        mpsc::Receiver<ClientMessage>,
    ) {
        let (commands, receiver) = mpsc::channel(32);
        let (requests, incoming) = mpsc::channel(32);
        let (events, responses) = mpsc::channel(1);
        let (settings, observed) = watch::channel(Workspace::default());
        let state = SessionState {
            daemon_configuration,
            id: "session".into(),
            spawner,
            daemon_calls: tokio::task::JoinSet::new(),
            daemon_results: VecDeque::new(),
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
            .reserve(name.into(), Some(Vec::new()), None)
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

    #[tokio::test]
    async fn tool_inventory_marks_disconnected_desktop_tools_not_connections_named_host() {
        let (session, events, _requests) = fixture();
        let id = session
            .reserve("observer".into(), None, None)
            .await
            .expect("reserve observer")
            .attachment_id;
        let lease = session.authorize(id.clone(), false).await.expect("lease");
        let observer = session
            .attach(
                id,
                tokio_stream::pending(),
                lease,
                AttachmentRecovery::InBand,
            )
            .await
            .expect("attach observer");
        tokio::pin!(observer);
        observer.next().await.expect("replay").expect("replay done");
        events
            .send(Ok(SessionMessage {
                payload: Some(session_message::Payload::ToolInventory(
                    mirror_protocol::ToolInventory {
                        tools: vec![
                            mirror_protocol::ToolAvailability {
                                name: "desktop_commands".into(),
                                source: "host".into(),
                                enabled: true,
                                ..Default::default()
                            },
                            mirror_protocol::ToolAvailability {
                                name: "mcp__host__read".into(),
                                source: "host".into(),
                                parent: Some("mcp_call".into()),
                                enabled: true,
                                ..Default::default()
                            },
                        ],
                    },
                )),
                ..Default::default()
            }))
            .await
            .expect("inventory event");
        let event = observer.next().await.expect("inventory").expect("event");
        assert_matches!(event.payload, Some(session_message::Payload::ToolInventory(tools)) if tools.tools[0].unavailable_reason.is_some() && tools.tools[1].unavailable_reason.is_none());
        session.close().await;
    }

    #[tokio::test(start_paused = true)]
    async fn snapshot_attachment_brackets_the_initial_tail_without_a_history_read() {
        let (session, events, _requests) = fixture();
        let (original, _lease) = attachment(&session, "host").await;
        tokio::pin!(original);
        original
            .next()
            .await
            .expect("initial replay")
            .expect("replay done");
        let message = SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: "latest".into(),
                sequence: Some(42),
                ..Default::default()
            })),
            ..Default::default()
        };
        events.send(Ok(message.clone())).await.expect("event");
        assert_eq!(
            original
                .next()
                .await
                .expect("observed event")
                .expect("event"),
            message
        );
        let id = session
            .reserve("observer".into(), None, None)
            .await
            .expect("reserve")
            .attachment_id;
        let lease = session.authorize(id.clone(), false).await.expect("lease");
        let stream = session
            .attach(
                id,
                tokio_stream::pending(),
                lease,
                AttachmentRecovery::InBandWithSnapshot,
            )
            .await
            .expect("attach");
        tokio::pin!(stream);
        assert_matches!(stream.next().await.expect("start").expect("start").payload,
            Some(session_message::Payload::ReplayStart(start)) if start.includes_runtime && start.tail_only && start.history_prefix == 0);
        assert_eq!(
            stream.next().await.expect("history").expect("history"),
            message
        );
        assert_matches!(
            stream
                .next()
                .await
                .expect("history boundary")
                .expect("history boundary")
                .payload,
            Some(session_message::Payload::ReplayDone(_))
        );
        assert_matches!(
            stream
                .next()
                .await
                .expect("display boundary")
                .expect("display boundary")
                .payload,
            Some(session_message::Payload::StateRestored(()))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn snapshot_reads_do_not_reserve_attachments_or_send_worker_commands() {
        let (session, _events, mut requests) = fixture();
        let snapshot = session
            .snapshot(32)
            .await
            .expect("snapshot without attaching");
        assert_eq!(snapshot.messages, vec![]);
        assert_eq!(snapshot.runtime.expect("known idle").events, vec![]);
        assert_matches!(requests.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        assert_matches!(session.authorize("not-attached".into(), true).await, Err(_));
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
    async fn daemon_mcp_defaults_cannot_be_replaced_by_an_attached_frontend() {
        let (session, _events, mut requests) = fixture_with_spawner(None, true);
        let attachment = session
            .reserve("host".into(), Some(Vec::new()), None)
            .await
            .expect("host")
            .attachment_id;
        let update = ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".into(),
            message_id: "mcp".into(),
            payload: Some(client_message::Payload::SetMcpConnections(
                Default::default(),
            )),
        };
        assert_eq!(
            session
                .send(&attachment, update)
                .await
                .expect_err("host-owned defaults")
                .code(),
            tonic::Code::PermissionDenied
        );
        session
            .refresh_mcp(Default::default())
            .await
            .expect("host refresh");
        assert_matches!(
            requests.recv().await.expect("worker update").payload,
            Some(client_message::Payload::SetMcpConnections(_))
        );
        session.close().await;
    }

    #[tokio::test]
    async fn settings_retry_with_new_trace_parent_is_a_duplicate() {
        let (session, events, mut requests) = fixture();
        let attachment = session
            .reserve("host".into(), Some(Vec::new()), None)
            .await
            .expect("reserve")
            .attachment_id;
        let settings = |trace_parent: &str, mode: ApprovalMode| ClientMessage {
            trace_parent: trace_parent.into(),
            personal_context: None,
            session_id: "session".into(),
            message_id: "settings".into(),
            payload: Some(client_message::Payload::SetApprovalMode(SetApprovalMode {
                mode: mode.into(),
            })),
        };
        let send = session.send(&attachment, settings("first", ApprovalMode::Bypass));
        tokio::pin!(send);
        let forwarded = tokio::select! {
            result = &mut send => panic!("first send settled before the worker received it: {result:?}"),
            forwarded = requests.recv() => forwarded.expect("forwarded settings"),
        };
        assert_eq!(forwarded.trace_parent, "first");
        events.send(Ok(ack("settings"))).await.expect("ack");
        send.await.expect("first send");
        let send = session.send(&attachment, settings("retry", ApprovalMode::Bypass));
        tokio::pin!(send);
        tokio::select! {
            result = &mut send => result,
            forwarded = requests.recv() => {
                forwarded.expect("forwarded settings");
                events.send(Ok(ack("settings"))).await.expect("ack");
                send.await
            }
        }
        .expect("retry with a new trace parent");
        let conflict = session
            .send(&attachment, settings("retry", ApprovalMode::Hitl))
            .await
            .expect_err("reused ID with different settings");
        assert_eq!(conflict.code(), tonic::Code::AlreadyExists);
        session.close().await;
    }

    #[tokio::test]
    async fn daemon_tools_cannot_be_answered_or_detached_by_a_frontend() {
        let (commands, mut calls) = mpsc::channel(1);
        let (session, events, mut requests) = fixture_with_spawner(
            Some(AgentSpawner {
                registry: crate::SpawnHandle {
                    remote_connections: Default::default(),
                    commands: commands.downgrade(),
                },
                workspace_id: "parent".into(),
                worker_id: "worker".into(),
            }),
            true,
        );
        let attachment = session
            .reserve("host".into(), Some(Vec::new()), None)
            .await
            .expect("frontend")
            .attachment_id;
        let detached = session
            .send(
                &attachment,
                ClientMessage {
                    trace_parent: String::new(),
                    personal_context: None,
                    session_id: "session".into(),
                    message_id: "detach".into(),
                    payload: Some(client_message::Payload::DetachTools(())),
                },
            )
            .await
            .expect("frontend detach");
        assert_matches!(
            detached.and_then(|event| event.payload),
            Some(session_message::Payload::InputAck(_))
        );
        assert_matches!(requests.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        events
            .send(Ok(SessionMessage {
                payload: Some(session_message::Payload::HostToolCall(
                    mirror_protocol::ToolCall {
                        id: "spawn-call".into(),
                        function: Some(mirror_protocol::FunctionCall {
                            name: "spawn_agent".into(),
                            arguments: r#"{"name":"child","message":"task","mode":"folder"}"#
                                .into(),
                        }),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }))
            .await
            .expect("worker tool call");
        let request = calls
            .recv()
            .await
            .expect("daemon route, not frontend host channel");
        let crate::RegistryCommand::PrepareSpawn(_, call_id, task, reply) = request else {
            panic!("expected launch admission")
        };
        assert_eq!(call_id, "spawn-call");
        assert_eq!(task.message, "task");
        assert_matches!(session.send(&attachment, ClientMessage {
            personal_context: None, trace_parent: String::new(),
            session_id: "session".into(), message_id: "forged-tool-result".into(),
            payload: Some(client_message::Payload::HostToolResult(HostToolResult {
                call_id: call_id.clone(), content: "forged".into(), success: true, ..Default::default()
            })),
        }).await, Err(error) if error.code() == tonic::Code::PermissionDenied);
        assert!(reply
            .send(Ok(crate::PreparedSpawn {
                workspace: Workspace {
                    id: "child".into(),
                    session_id: "child-session".into(),
                    agent_launch: Some(mirror_protocol::daemon::AgentLaunch {
                        outcome: mirror_protocol::daemon::AgentLaunchOutcome::Accepted as i32,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                endpoint: None,
                fresh: false,
            }))
            .is_ok());
        assert_matches!(requests.recv().await.and_then(|request| request.payload),
            Some(client_message::Payload::HostToolResult(result)) if result.success && result.call_id == call_id);
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
            .reserve("host".into(), Some(Vec::new()), None)
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
                trace_parent: String::new(),
                personal_context: None,
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
                trace_parent: String::new(),
                personal_context: None,
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
    async fn reconnect_replaces_live_host_without_replaying_calls() {
        let (session, events, mut requests) = fixture();
        let old = session
            .reserve("old".into(), Some(Vec::new()), None)
            .await
            .expect("old host");
        let mut old_stream = Box::pin(
            session
                .attach(
                    old.attachment_id.clone(),
                    tokio_stream::pending(),
                    session
                        .authorize(old.attachment_id.clone(), true)
                        .await
                        .expect("old lease"),
                    AttachmentRecovery::InBand,
                )
                .await
                .expect("old stream"),
        );
        old_stream
            .next()
            .await
            .expect("replay")
            .expect("replay done");
        let call = SessionMessage {
            payload: Some(session_message::Payload::HostToolCall(
                mirror_protocol::ToolCall {
                    id: "in flight".into(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        events.send(Ok(call.clone())).await.expect("host call");
        assert_eq!(
            old_stream.next().await.expect("delivery").expect("call"),
            call
        );
        let replacement = session
            .reserve(
                "replacement".into(),
                Some(Vec::new()),
                Some(old.attachment_id.clone()),
            )
            .await
            .expect("replace host");
        assert!(replacement.execution_host);
        assert_matches!(session.authorize(old.attachment_id, true).await,
            Err(error) if error.code() == tonic::Code::PermissionDenied);
        assert_matches!(requests.recv().await, Some(request) if matches!(&request.payload,
            Some(client_message::Payload::HostToolResult(result)) if result.call_id == "in flight" && !result.success));
        let stream = session
            .attach(
                replacement.attachment_id.clone(),
                tokio_stream::pending(),
                session
                    .authorize(replacement.attachment_id, true)
                    .await
                    .expect("new lease"),
                AttachmentRecovery::InBand,
            )
            .await
            .expect("replacement stream");
        tokio::pin!(stream);
        stream.next().await.expect("replay").expect("replay done");
        drop(old_stream);
        let next = SessionMessage {
            payload: Some(session_message::Payload::HostToolCall(
                mirror_protocol::ToolCall {
                    id: "next".into(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        events.send(Ok(next.clone())).await.expect("next host call");
        assert_eq!(stream.next().await.expect("delivery").expect("call"), next);
        session.close().await;
    }

    #[tokio::test]
    async fn reconnect_observer_cannot_replace_host_authority() {
        let (session, _events, _requests) = fixture();
        let host = session
            .reserve("host".into(), Some(Vec::new()), None)
            .await
            .expect("host");
        let observer = session
            .reserve("observer".into(), None, None)
            .await
            .expect("observer");
        let replacement = session
            .reserve(
                "replacement".into(),
                Some(Vec::new()),
                Some(observer.attachment_id),
            )
            .await
            .expect("replace observer");
        assert!(!replacement.execution_host);
        assert_matches!(session.reserve("mismatch".into(), None, Some(host.attachment_id.clone())).await,
            Err(error) if error.code() == tonic::Code::FailedPrecondition);
        drop(
            session
                .authorize(host.attachment_id, true)
                .await
                .expect("host unchanged"),
        );
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
            .reserve("host".into(), Some(Vec::new()), None)
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
                trace_parent: String::new(),
                personal_context: None,
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
                trace_parent: String::new(),
                personal_context: None,
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
                personal_context: None,
                trace_parent: String::new(),
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
        session
            .reserve("host".into(), None, None)
            .await
            .expect("reserve");
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
            .reserve("host".into(), None, None)
            .await
            .expect("reserve")
            .attachment_id;
        let sender = session.clone();
        let pending = tokio::spawn(async move {
            sender
                .send(
                    &id,
                    ClientMessage {
                        trace_parent: String::new(),
                        personal_context: None,
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

    struct InitializingServer {
        history: watch::Sender<Vec<AgentMessage>>,
        live: Arc<Notify>,
    }

    fn startup_entry(sequence: u64) -> AgentMessage {
        AgentMessage {
            id: format!("entry-{sequence}"),
            sequence: Some(sequence),
            ..Default::default()
        }
    }

    #[tonic::async_trait]
    impl AgentRpc for InitializingServer {
        type SessionStream = Pin<Box<dyn Stream<Item = Result<SessionMessage, Status>> + Send>>;

        async fn session(
            &self,
            request: Request<Streaming<ClientMessage>>,
        ) -> Result<Response<Self::SessionStream>, Status> {
            let mut incoming = request.into_inner();
            let history = self.history.clone();
            let live = self.live.clone();
            Ok(Response::new(Box::pin(async_stream::try_stream! {
                let configuration = incoming.message().await?.ok_or_else(closed)?;
                history.send_replace(vec![startup_entry(1), startup_entry(2)]);
                yield SessionMessage {
                    payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                        payload: Some(runtime_delta::Payload::TurnStarted(())),
                        ..Default::default()
                    })),
                    ..Default::default()
                };
                yield ack(&configuration.message_id);
                let approval = incoming.message().await?.ok_or_else(closed)?;
                yield ack(&approval.message_id);
                live.notified().await;
                for sequence in [1, 2, 3] {
                    yield SessionMessage {
                        payload: Some(session_message::Payload::AgentMessage(startup_entry(sequence))),
                        ..Default::default()
                    };
                }
                let _ = incoming.message().await?;
            })))
        }

        async fn get_messages(
            &self,
            request: Request<GetMessagesRequest>,
        ) -> Result<Response<GetMessagesResponse>, Status> {
            let request = request.into_inner();
            let history = self.history.borrow();
            Ok(Response::new(GetMessagesResponse {
                messages: history
                    .iter()
                    .skip(request.offset as usize)
                    .take(request.limit as usize)
                    .cloned()
                    .collect(),
                total: history.len() as u64,
                ..Default::default()
            }))
        }

        async fn call_tool(
            &self,
            _: Request<CallToolRequest>,
        ) -> Result<Response<CallToolResponse>, Status> {
            Err(Status::unimplemented("no tools"))
        }
    }

    #[tokio::test]
    async fn startup_replays_initialization_and_delivers_live_history_once_to_each_client() {
        let directory = tempfile::tempdir().expect("socket directory");
        let socket = directory.path().join("server.sock");
        let live = Arc::new(Notify::new());
        let listener = tokio::net::UnixListener::bind(&socket).expect("listener");
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(InitializingServer {
                    history: watch::channel(Vec::new()).0,
                    live: live.clone(),
                }))
                .serve_with_incoming(crate::local_incoming(listener)),
        );
        let session = Session::start(
            None,
            crate::local_channel(socket).await.expect("channel"),
            Workspace {
                session_id: "session".into(),
                ..Default::default()
            },
            AgentConfiguration {
                inference: Some(InferenceConfig::default()),
                ..Default::default()
            },
            "host".into(),
            false,
            None,
            None,
            ExistingSettings::Replace,
        )
        .await
        .expect("session");
        let snapshot = session.snapshot(128).await.expect("initialized history");
        assert_eq!(snapshot.messages, vec![startup_entry(1), startup_entry(2)]);
        assert_matches!(&snapshot.runtime.expect("runtime").events[0].payload,
            Some(session_message::Payload::RuntimeDelta(delta)) if matches!(delta.payload, Some(runtime_delta::Payload::TurnStarted(()))));
        let (first, first_lease) = attachment(&session, "host").await;
        let (second, second_lease) = attachment(&session, "observer").await;
        tokio::pin!(first, second);
        let replay = (&mut first)
            .take(4)
            .map(|event| event.expect("replay"))
            .collect::<Vec<_>>()
            .await;
        assert_eq!(
            replay,
            (&mut second)
                .take(4)
                .map(|event| event.expect("replay"))
                .collect::<Vec<_>>()
                .await
        );
        live.notify_one();
        let event = first.next().await.expect("live event").expect("event");
        assert_matches!(&event.payload,
            Some(session_message::Payload::AgentMessage(entry)) if *entry == startup_entry(3));
        assert_eq!(
            second.next().await.expect("observer event").expect("event"),
            event
        );
        assert_eq!(
            session.snapshot(128).await.expect("snapshot").messages,
            vec![startup_entry(1), startup_entry(2), startup_entry(3)]
        );
        drop(first_lease);
        drop(second_lease);
        // Stream leases end when their streams are dropped.
        server.abort();
    }

    struct RestartingServer(Arc<Notify>, Option<mpsc::UnboundedSender<CallToolRequest>>);

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
            request: Request<CallToolRequest>,
        ) -> Result<Response<CallToolResponse>, Status> {
            let Some(calls) = &self.1 else {
                return Err(Status::unimplemented("no tools"));
            };
            let request = request.into_inner();
            calls
                .send(request.clone())
                .expect("observe forwarded request");
            if request.name == "mcp__hosted_mcp__call" {
                return Err(Status::permission_denied(
                    "server denied publication access",
                ));
            }
            Ok(Response::new(CallToolResponse {
                content: "server-owned result".into(),
                success: true,
                display: None,
            }))
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
                .add_service(AgentRpcServer::new(RestartingServer(restart.clone(), None)))
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
            false,
            None,
            None,
            ExistingSettings::Replace,
        )
        .await
        .expect("session");
        let attachment = session
            .reserve("client".into(), None, None)
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

    #[tokio::test]
    async fn publication_tool_requests_and_results_are_forwarded_unchanged() {
        let directory = tempfile::tempdir().expect("socket directory");
        let socket = directory.path().join("server.sock");
        let listener = tokio::net::UnixListener::bind(&socket).expect("listener");
        let (calls, mut received) = mpsc::unbounded_channel();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(RestartingServer(
                    Arc::new(Notify::new()),
                    Some(calls),
                )))
                .serve_with_incoming(crate::local_incoming(listener)),
        );
        let session = Session::start(
            None,
            crate::local_channel(socket).await.expect("channel"),
            Workspace {
                session_id: "session".into(),
                ..Default::default()
            },
            AgentConfiguration {
                inference: Some(InferenceConfig::default()),
                ..Default::default()
            },
            "client".into(),
            false,
            None,
            None,
            ExistingSettings::Replace,
        )
        .await
        .expect("session");
        let attachment = session
            .reserve("client".into(), Some(Vec::new()), None)
            .await
            .expect("reservation")
            .attachment_id;
        let lease = session
            .authorize(attachment.clone(), true)
            .await
            .expect("lease");
        let request = CallToolRequest {
            session_id: "session".into(),
            name: "mcp__hosted_mcp__list".into(),
            arguments: "{}".into(),
            web_provider: None,
        };
        assert_eq!(
            session
                .call_tool(request.clone(), lease)
                .await
                .expect("forwarded result"),
            CallToolResponse {
                content: "server-owned result".into(),
                success: true,
                display: None
            }
        );
        assert_eq!(received.recv().await.expect("forwarded request"), request);
        let request = CallToolRequest {
            name: "mcp__hosted_mcp__call".into(),
            arguments: r#"{"publication_id":"foreign","tool":"echo","arguments":{}}"#.into(),
            ..request
        };
        let lease = session.authorize(attachment, true).await.expect("lease");
        assert_matches!(session.call_tool(request.clone(), lease).await, Err(status) if status.code() == tonic::Code::PermissionDenied && status.message() == "server denied publication access");
        assert_eq!(received.recv().await.expect("forwarded request"), request);
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        session.close().await;
        server.abort();
    }

    #[derive(Clone)]
    enum Inspection {
        Reports(SessionSettings),
        Unknown,
        Refused {
            message: &'static str,
            closed_nack: bool,
            remaining: Arc<std::sync::atomic::AtomicUsize>,
            settings: SessionSettings,
        },
    }

    struct ExistingServer {
        history: Vec<AgentMessage>,
        inspection: Inspection,
        received: mpsc::UnboundedSender<ClientMessage>,
    }

    #[tonic::async_trait]
    impl AgentRpc for ExistingServer {
        type SessionStream = Pin<Box<dyn Stream<Item = Result<SessionMessage, Status>> + Send>>;

        async fn session(
            &self,
            request: Request<Streaming<ClientMessage>>,
        ) -> Result<Response<Self::SessionStream>, Status> {
            let mut incoming = request.into_inner();
            let received = self.received.clone();
            let inspection = self.inspection.clone();
            Ok(Response::new(Box::pin(async_stream::try_stream! {
                while let Some(message) = incoming.message().await? {
                    let message_id = message.message_id.clone();
                    let inspect = matches!(
                        message.payload,
                        Some(client_message::Payload::InspectSettings(()))
                    );
                    let _ = received.send(message);
                    match (inspect, inspection.clone()) {
                        (true, Inspection::Reports(settings)) => {
                            yield SessionMessage {
                                payload: Some(session_message::Payload::SessionSettings(settings)),
                                ..Default::default()
                            };
                            yield ack(&message_id);
                        }
                        (true, Inspection::Refused { message, closed_nack, remaining, settings }) => {
                            if remaining
                                .fetch_update(
                                    std::sync::atomic::Ordering::SeqCst,
                                    std::sync::atomic::Ordering::SeqCst,
                                    |left| left.checked_sub(1),
                                )
                                .is_ok()
                            {
                                if closed_nack {
                                    yield SessionMessage {
                                        payload: Some(session_message::Payload::InputNack(InputNack {
                                            message_id: message_id.clone(),
                                            reason: InputNackReason::TargetNotFound.into(),
                                            error: "agent service closed; reconnect".into(),
                                        })),
                                        ..Default::default()
                                    };
                                }
                                Err(Status::unavailable(message))?;
                            }
                            yield SessionMessage {
                                payload: Some(session_message::Payload::SessionSettings(settings)),
                                ..Default::default()
                            };
                            yield ack(&message_id);
                        }
                        (true, Inspection::Unknown) => {
                            yield SessionMessage {
                                payload: Some(session_message::Payload::InputNack(InputNack {
                                    message_id,
                                    reason: InputNackReason::InvalidRequest.into(),
                                    error: "client message has no payload".into(),
                                })),
                                ..Default::default()
                            };
                        }
                        (false, _) => yield ack(&message_id),
                    }
                }
            })))
        }

        async fn get_messages(
            &self,
            request: Request<GetMessagesRequest>,
        ) -> Result<Response<GetMessagesResponse>, Status> {
            let request = request.into_inner();
            Ok(Response::new(GetMessagesResponse {
                messages: self
                    .history
                    .iter()
                    .skip(request.offset as usize)
                    .take(request.limit as usize)
                    .cloned()
                    .collect(),
                total: self.history.len() as u64,
                ..Default::default()
            }))
        }

        async fn call_tool(
            &self,
            _: Request<CallToolRequest>,
        ) -> Result<Response<CallToolResponse>, Status> {
            Err(Status::unimplemented("no tools"))
        }
    }

    fn model(name: &str) -> InferenceConfig {
        InferenceConfig {
            model: name.into(),
            api: Some("openai-responses".into()),
            ..Default::default()
        }
    }

    async fn start_existing(
        history: Vec<AgentMessage>,
        inspection: Inspection,
        existing: ExistingSettings,
    ) -> (Result<Session, Status>, Vec<ClientMessage>) {
        let (client_io, server_io) = tokio::io::duplex(1 << 20);
        let (received, mut messages) = mpsc::unbounded_channel();
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(ExistingServer {
                    history,
                    inspection,
                    received,
                }))
                .serve_with_incoming(tokio_stream::once(Ok::<_, std::io::Error>(server_io))),
        );
        let mut client_io = Some(client_io);
        let channel = tonic::transport::Endpoint::from_static("http://localhost")
            .connect_with_connector(tower::service_fn(move |_| {
                let io = client_io.take();
                async move {
                    io.map(hyper_util::rt::TokioIo::new).ok_or_else(|| {
                        std::io::Error::other("in-memory transport already connected")
                    })
                }
            }))
            .await
            .expect("channel");
        let session = Session::start(
            None,
            channel,
            Workspace {
                session_id: "session".into(),
                ..Default::default()
            },
            AgentConfiguration {
                inference: Some(model("host-default")),
                approval_mode: ApprovalMode::Hitl.into(),
                ..Default::default()
            },
            "host".into(),
            true,
            None,
            None,
            existing,
        )
        .await;
        let mut startup = Vec::new();
        while let Ok(message) = messages.try_recv() {
            startup.push(message);
        }
        (session, startup)
    }

    fn turn(model_name: &str) -> AgentMessage {
        AgentMessage {
            id: "turn".into(),
            sequence: Some(1),
            inference: Some(model(model_name)),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn start_keep_existing_settings_reads_live_settings_without_changing_them() {
        let live = SessionSettings {
            inference: Some(model("selected-model")),
            approval_mode: ApprovalMode::Bypass.into(),
        };
        let (session, startup) = start_existing(
            vec![turn("previous-turn-model")],
            Inspection::Reports(live),
            ExistingSettings::Keep,
        )
        .await;
        let session = session.expect("session");

        assert_matches!(
            startup
                .iter()
                .map(|message| &message.payload)
                .collect::<Vec<_>>()[..],
            [Some(client_message::Payload::InspectSettings(()))]
        );
        let settings = session.settings();
        assert_eq!(settings.inference, Some(model("selected-model")));
        assert_eq!(settings.approval_mode, Some(ApprovalMode::Bypass.into()));
        session.close().await;
    }

    #[tokio::test]
    async fn start_keep_existing_settings_pushes_settings_to_a_server_without_inspection() {
        let (session, startup) = start_existing(
            vec![turn("previous-turn-model")],
            Inspection::Unknown,
            ExistingSettings::Keep,
        )
        .await;
        let session = session.expect("session");

        assert_eq!(
            startup
                .into_iter()
                .map(|message| message.payload)
                .collect::<Vec<_>>(),
            vec![
                Some(client_message::Payload::InspectSettings(())),
                Some(client_message::Payload::SetInferenceConfig(model(
                    "host-default"
                ))),
                Some(client_message::Payload::SetApprovalMode(SetApprovalMode {
                    mode: ApprovalMode::Hitl.into(),
                })),
            ]
        );
        let settings = session.settings();
        assert_eq!(settings.inference, Some(model("host-default")));
        assert_eq!(settings.approval_mode, Some(ApprovalMode::Hitl.into()));
        session.close().await;
    }

    const LEASED_ELSEWHERE: &str = "session is executing on another node; retry";
    const RESTARTING: &str = "server is restarting; reconnect";

    fn refused(message: &'static str, refusals: usize) -> Inspection {
        Inspection::Refused {
            message,
            closed_nack: false,
            remaining: Arc::new(refusals.into()),
            settings: SessionSettings {
                inference: Some(model("selected-model")),
                approval_mode: ApprovalMode::Bypass.into(),
            },
        }
    }

    async fn assert_keep_retries_inspection_after(message: &'static str) {
        let (session, startup) = start_existing(
            vec![turn("previous-turn-model")],
            refused(message, 1),
            ExistingSettings::Keep,
        )
        .await;
        let session = session.expect("session");

        assert_matches!(
            startup
                .iter()
                .map(|message| &message.payload)
                .collect::<Vec<_>>()[..],
            [
                Some(client_message::Payload::InspectSettings(())),
                Some(client_message::Payload::InspectSettings(())),
            ]
        );
        let settings = session.settings();
        assert_eq!(settings.inference, Some(model("selected-model")));
        assert_eq!(settings.approval_mode, Some(ApprovalMode::Bypass.into()));
        session.close().await;
    }

    async fn assert_keep_fails_without_pushing_after(message: &'static str) {
        let (session, startup) = start_existing(
            vec![turn("previous-turn-model")],
            refused(message, usize::MAX),
            ExistingSettings::Keep,
        )
        .await;

        assert_matches!(session, Err(status) if status.code() == tonic::Code::Unavailable && status.message() == message);
        assert!(startup.iter().all(|message| matches!(
            message.payload,
            Some(client_message::Payload::InspectSettings(()))
        )));
    }

    #[tokio::test(start_paused = true)]
    async fn start_keep_existing_settings_retries_inspection_when_leased_elsewhere() {
        assert_keep_retries_inspection_after(LEASED_ELSEWHERE).await;
    }

    #[tokio::test(start_paused = true)]
    async fn start_keep_existing_settings_retries_inspection_while_the_server_restarts() {
        assert_keep_retries_inspection_after(RESTARTING).await;
    }

    #[tokio::test(start_paused = true)]
    async fn start_keep_existing_settings_fails_without_pushing_when_leased_elsewhere() {
        assert_keep_fails_without_pushing_after(LEASED_ELSEWHERE).await;
    }

    #[tokio::test(start_paused = true)]
    async fn start_keep_existing_settings_fails_without_pushing_while_the_server_restarts() {
        assert_keep_fails_without_pushing_after(RESTARTING).await;
    }

    fn closed_during_inspection(refusals: usize) -> Inspection {
        Inspection::Refused {
            message: RESTARTING,
            closed_nack: true,
            remaining: Arc::new(refusals.into()),
            settings: SessionSettings {
                inference: Some(model("selected-model")),
                approval_mode: ApprovalMode::Bypass.into(),
            },
        }
    }

    #[tokio::test(start_paused = true)]
    async fn start_keep_existing_settings_retries_inspection_after_the_worker_closes() {
        let (session, startup) = start_existing(
            vec![turn("previous-turn-model")],
            closed_during_inspection(1),
            ExistingSettings::Keep,
        )
        .await;
        let session = session.expect("session");

        assert_matches!(
            startup
                .iter()
                .map(|message| &message.payload)
                .collect::<Vec<_>>()[..],
            [
                Some(client_message::Payload::InspectSettings(())),
                Some(client_message::Payload::InspectSettings(())),
            ]
        );
        let settings = session.settings();
        assert_eq!(settings.inference, Some(model("selected-model")));
        assert_eq!(settings.approval_mode, Some(ApprovalMode::Bypass.into()));
        session.close().await;
    }

    #[tokio::test(start_paused = true)]
    async fn start_keep_existing_settings_is_unavailable_when_the_worker_keeps_closing() {
        let (session, startup) = start_existing(
            vec![turn("previous-turn-model")],
            closed_during_inspection(usize::MAX),
            ExistingSettings::Keep,
        )
        .await;

        assert_matches!(session, Err(status) if status.code() == tonic::Code::Unavailable);
        assert!(startup.iter().all(|message| matches!(
            message.payload,
            Some(client_message::Payload::InspectSettings(()))
        )));
    }

    #[tokio::test]
    async fn start_keep_existing_settings_configures_a_new_session() {
        let (session, startup) =
            start_existing(Vec::new(), Inspection::Unknown, ExistingSettings::Keep).await;
        let session = session.expect("session");

        assert_eq!(
            startup
                .into_iter()
                .map(|message| message.payload)
                .collect::<Vec<_>>(),
            vec![
                Some(client_message::Payload::SetInferenceConfig(model(
                    "host-default"
                ))),
                Some(client_message::Payload::SetApprovalMode(SetApprovalMode {
                    mode: ApprovalMode::Hitl.into(),
                })),
            ]
        );
        let settings = session.settings();
        assert_eq!(settings.inference, Some(model("host-default")));
        assert_eq!(settings.approval_mode, Some(ApprovalMode::Hitl.into()));
        session.close().await;
    }
}
