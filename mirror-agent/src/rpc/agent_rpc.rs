use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use crate::tools::ToolOutput;
use futures_util::future::BoxFuture;
use futures_util::Stream;
use futures_util::StreamExt;
use mirror_protocol::agent_message;
use mirror_protocol::agent_rpc_server;
use mirror_protocol::client_message;
use mirror_protocol::runtime_delta;
use mirror_protocol::session_message;
use mirror_protocol::AgentMessage;
use mirror_protocol::ApprovalMode;
use mirror_protocol::ApprovalSource;
use mirror_protocol::CallToolRequest;
use mirror_protocol::CallToolResponse;
use mirror_protocol::ClientMessage;
use mirror_protocol::ContextUsage;
use mirror_protocol::GetMessagesRequest;
use mirror_protocol::GetMessagesResponse;
use mirror_protocol::HitlRequest;
use mirror_protocol::HitlResolved;
use mirror_protocol::HostToolResult;
use mirror_protocol::InferenceConfig;
use mirror_protocol::InputAck;
use mirror_protocol::InputNack;
use mirror_protocol::InputNackReason;
use mirror_protocol::RuntimeDelta;
use mirror_protocol::SessionMessage;
use mirror_protocol::TaskStatus as ProtocolTaskStatus;
use mirror_protocol::TaskUpdate;
use mirror_protocol::ToolCall;
use mirror_protocol::TurnError;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use crate::AgentApprovalMode;
use crate::AgentEvent;
use crate::ReactAgentServiceHandle;
use crate::ReactServiceError;
use crate::ReactServiceEvent;
use crate::RuntimeTaskStatus;
use crate::TaskInfo;

#[derive(Debug, Clone, PartialEq)]
struct SessionRequest {
    session_id: String,
    message_id: String,
    input: SessionInput,
}

#[derive(Debug, Clone, PartialEq)]
enum SessionInput {
    Message(Box<AgentMessage>),
    Compact,
    Interrupt,
    CancelTask {
        task_id: String,
    },
    HitlResponse {
        call_id: String,
        approved: bool,
        source: ApprovalSource,
    },
    RespondToHostTool {
        call_id: String,
        response: ToolOutput,
    },
    DetachTools,
    SetApprovalMode(AgentApprovalMode),
    SetInferenceConfig(InferenceConfig),
}

#[derive(Debug, Clone, PartialEq)]
enum SessionOutput {
    Event(ReactServiceEvent),
    Ack {
        message_id: String,
    },
    Nack {
        message_id: String,
        reason: SessionNackReason,
        error: String,
    },
    EventStreamLagged(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionNackReason {
    InvalidRequest,
    SessionMismatch,
    MessageIdConflict,
    TargetNotFound,
}

#[derive(Clone)]
pub struct AgentRpc {
    service: ReactAgentServiceHandle,
    shutdown: broadcast::Sender<()>,
}

impl AgentRpc {
    pub fn new(service: ReactAgentServiceHandle) -> Self {
        let (shutdown, _) = broadcast::channel(1);
        Self { service, shutdown }
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown.send(());
    }

    pub async fn open_session<Requests>(
        &self,
        requests: Requests,
    ) -> Result<ReceiverStream<Result<SessionMessage, Status>>, Status>
    where
        Requests: Stream<Item = Result<ClientMessage, Status>> + Send + 'static,
    {
        serve_agent_rpc_session(
            self.service.clone(),
            Arc::new(self.service.clone()),
            requests,
            self.shutdown.subscribe(),
        )
        .await
    }
}

#[tonic::async_trait]
impl agent_rpc_server::AgentRpc for AgentRpc {
    type SessionStream = ReceiverStream<Result<SessionMessage, Status>>;

    async fn session(
        &self,
        request: Request<tonic::Streaming<ClientMessage>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        self.open_session(request.into_inner())
            .await
            .map(Response::new)
    }

    async fn call_tool(
        &self,
        request: Request<CallToolRequest>,
    ) -> Result<Response<CallToolResponse>, Status> {
        let request = request.into_inner();
        if request.session_id != self.service.session_id() {
            return Err(Status::not_found("session not found"));
        }
        let mut shutdown = self.shutdown.subscribe();
        let output = tokio::select! {
            result = self.service.call_tool(request.name, request.arguments) => {
                result.map_err(|error| Status::internal(error.to_string()))?
            }
            _ = shutdown.recv() => return Err(Status::unavailable("agent RPC shutting down")),
        };
        Ok(Response::new(CallToolResponse {
            content: output.content,
            success: output.success,
            display: output.display,
        }))
    }

    async fn get_messages(
        &self,
        request: Request<GetMessagesRequest>,
    ) -> Result<Response<GetMessagesResponse>, Status> {
        let request = request.into_inner();
        if request.session_id != self.service.session_id() {
            return Err(Status::not_found(format!(
                "this AgentRPC serves session {:?}, not {:?}",
                self.service.session_id(),
                request.session_id
            )));
        }
        let limit = if request.limit == 0 {
            100
        } else {
            request.limit.min(1000)
        };
        let (messages, total) = self
            .service
            .messages_page(request.offset, limit)
            .await
            .map_err(|error| Status::internal(error.to_string()))?;
        Ok(Response::new(GetMessagesResponse { messages, total }))
    }
}

pub trait SessionSource: Send + Sync {
    fn acquire(
        &self,
        intent: AcquireIntent,
    ) -> BoxFuture<'_, Result<ReactAgentServiceHandle, Status>>;
    fn detached(&self) -> BoxFuture<'_, Result<(), Status>>;
}

impl SessionSource for ReactAgentServiceHandle {
    fn acquire(
        &self,
        _intent: AcquireIntent,
    ) -> BoxFuture<'_, Result<ReactAgentServiceHandle, Status>> {
        Box::pin(async move {
            if self.is_closed() {
                Err(Status::unavailable(SERVICE_CLOSED))
            } else {
                Ok(self.clone())
            }
        })
    }

    fn detached(&self) -> BoxFuture<'_, Result<(), Status>> {
        Box::pin(async { Err(Status::unavailable(SERVICE_CLOSED)) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireIntent {
    Execute,
    Observe,
}

pub fn acquire_intent(message: &ClientMessage) -> AcquireIntent {
    match message.payload {
        Some(
            client_message::Payload::Message(_)
            | client_message::Payload::SetInferenceConfig(_)
            | client_message::Payload::SetApprovalMode(_)
            | client_message::Payload::Compact(_),
        ) => AcquireIntent::Execute,
        _ => AcquireIntent::Observe,
    }
}

const SERVICE_CLOSED: &str = "agent service closed; reconnect";
const CLOSE_GRACE: Duration = Duration::from_secs(5);

pub async fn serve_agent_rpc_session<Requests>(
    service: ReactAgentServiceHandle,
    source: Arc<dyn SessionSource>,
    requests: Requests,
    shutdown: broadcast::Receiver<()>,
) -> Result<ReceiverStream<Result<SessionMessage, Status>>, Status>
where
    Requests: Stream<Item = Result<ClientMessage, Status>> + Send + 'static,
{
    let attached = AttachedService::subscribe(service).await;
    if attached.service.is_closed() {
        return Err(Status::unavailable(SERVICE_CLOSED));
    }
    let (responses, receiver) = mpsc::channel(32);
    tokio::spawn(run_session(
        SessionState::new(attached, source),
        requests,
        shutdown,
        responses,
    ));
    Ok(ReceiverStream::new(receiver))
}

struct SessionState {
    session_id: String,
    attached: Option<AttachedService>,
    source: Arc<dyn SessionSource>,
    previous_inputs: HashMap<String, SessionInput>,
    rejected: Vec<String>,
    approval_mode: Option<AgentApprovalMode>,
    inference: Option<InferenceConfig>,
}

struct AttachedService {
    service: ReactAgentServiceHandle,
    events: broadcast::Receiver<ReactServiceEvent>,
    /// Requests made before this stream subscribed; the broadcast never sends them again.
    unannounced: VecDeque<ToolCall>,
}

impl AttachedService {
    /// Subscribe before reading pending approvals, so a request lands in one or
    /// both; clients ignore a call they are already asking about.
    async fn subscribe(service: ReactAgentServiceHandle) -> Self {
        let events = service.events();
        let unannounced = service.pending_approvals().await.unwrap_or_default().into();
        Self {
            service,
            events,
            unannounced,
        }
    }
}

#[derive(Debug)]
struct ServiceClosed;

enum Step {
    Request(ClientMessage),
    Event(ReactServiceEvent),
    Lagged(u64),
    Detached,
    End(Option<Status>),
}

impl SessionState {
    fn new(attached: AttachedService, source: Arc<dyn SessionSource>) -> Self {
        Self {
            session_id: attached.service.session_id().to_string(),
            attached: Some(attached),
            source,
            previous_inputs: HashMap::new(),
            rejected: Vec::new(),
            approval_mode: None,
            inference: None,
        }
    }

    async fn next_step(&mut self) -> Step {
        let Some(attached) = &mut self.attached else {
            return std::future::pending().await;
        };
        if let Some(call) = attached.unannounced.pop_front() {
            return Step::Event(ReactServiceEvent::HitlRequest(call));
        }
        match attached.events.recv().await {
            Ok(ReactServiceEvent::Closed) | Err(broadcast::error::RecvError::Closed) => {
                Step::Detached
            }
            Ok(event) => Step::Event(event),
            Err(broadcast::error::RecvError::Lagged(count)) => Step::Lagged(count),
        }
    }

    fn remembers(&self, message_id: &str) -> bool {
        self.previous_inputs.contains_key(message_id)
    }

    async fn service(&mut self, intent: AcquireIntent) -> Result<ReactAgentServiceHandle, Status> {
        if let Some(attached) = &self.attached {
            return Ok(attached.service.clone());
        }
        let attached = match self.subscribe(intent).await? {
            Some(attached) => attached,
            None => self
                .subscribe(intent)
                .await?
                .ok_or_else(|| Status::unavailable(SERVICE_CLOSED))?,
        };
        self.restore_settings(&attached.service).await?;
        let service = attached.service.clone();
        self.attached = Some(attached);
        Ok(service)
    }

    async fn subscribe(&self, intent: AcquireIntent) -> Result<Option<AttachedService>, Status> {
        let attached = AttachedService::subscribe(self.source.acquire(intent).await?).await;
        Ok((!attached.service.is_closed()).then_some(attached))
    }

    async fn restore_settings(&self, service: &ReactAgentServiceHandle) -> Result<(), Status> {
        if let Some(mode) = self.approval_mode {
            service
                .set_approval_mode(mode)
                .await
                .map_err(service_status)?;
        }
        if let Some(inference) = &self.inference {
            service
                .set_inference_config(inference.clone())
                .await
                .map_err(service_status)?;
        }
        Ok(())
    }

    fn remember_setting(&mut self, input: &SessionInput) {
        match input {
            SessionInput::SetApprovalMode(mode) => self.approval_mode = Some(*mode),
            SessionInput::SetInferenceConfig(inference) => {
                self.inference = Some(inference.clone());
            }
            _ => {}
        }
    }

    async fn handle(
        &mut self,
        request: ClientMessage,
        responses: &mpsc::Sender<Result<SessionMessage, Status>>,
    ) -> Result<Vec<SessionOutput>, Status> {
        let request = match session_request(request) {
            Ok(request) => request,
            Err(invalid) => return Ok(vec![invalid.into_output()]),
        };
        let fresh = !self.remembers(&request.message_id);
        let message_id = request.message_id.clone();
        let input = request.input.clone();
        let handled = match self.service(input.intent()).await {
            Ok(service) => self.handle_on(&service, request, responses).await,
            Err(status) => Err(status),
        };
        let outputs = match handled {
            Ok(outputs) => outputs,
            Err(status) => {
                let _ = send_output(closed_nack(message_id), responses).await;
                return Err(status);
            }
        };
        if fresh && self.remembers(&message_id) {
            self.remember_setting(&input);
        }
        Ok(outputs)
    }

    async fn handle_on(
        &mut self,
        service: &ReactAgentServiceHandle,
        request: SessionRequest,
        responses: &mpsc::Sender<Result<SessionMessage, Status>>,
    ) -> Result<Vec<SessionOutput>, Status> {
        let intent = request.input.intent();
        match handle_request(service, &mut self.previous_inputs, request.clone()).await {
            Ok(outputs) => Ok(outputs),
            Err(ServiceClosed) => {
                self.drain_closing(responses).await?;
                self.attached = None;
                self.source.detached().await?;
                let service = self.service(self.replay_intent(intent)).await?;
                let mut outputs = self.replay(&service).await;
                match handle_request(&service, &mut self.previous_inputs, request.clone()).await {
                    Ok(handled) => outputs.extend(handled),
                    Err(ServiceClosed) => outputs.push(closed_nack(request.message_id)),
                }
                Ok(outputs)
            }
        }
    }

    async fn drain_closing(
        &mut self,
        responses: &mpsc::Sender<Result<SessionMessage, Status>>,
    ) -> Result<(), Status> {
        let Some(attached) = &mut self.attached else {
            return Ok(());
        };
        let deadline = tokio::time::Instant::now() + CLOSE_GRACE;
        loop {
            match tokio::time::timeout_at(deadline, attached.events.recv()).await {
                Ok(Ok(ReactServiceEvent::Closed))
                | Ok(Err(broadcast::error::RecvError::Closed))
                | Err(_) => return Ok(()),
                Ok(Ok(ReactServiceEvent::InputRejected { message_id }))
                    if self.previous_inputs.contains_key(&message_id) =>
                {
                    self.rejected.push(message_id);
                }
                Ok(Ok(event)) => {
                    if !send_output(SessionOutput::Event(event), responses).await {
                        return Err(Status::cancelled("client stopped reading"));
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(count))) => {
                    return Err(lagged_status(count));
                }
            }
        }
    }

    async fn detached(&mut self) -> Result<Vec<SessionOutput>, Status> {
        self.attached = None;
        self.source.detached().await?;
        if self.rejected.is_empty() {
            return Ok(Vec::new());
        }
        let service = self
            .service(self.replay_intent(AcquireIntent::Observe))
            .await?;
        Ok(self.replay(&service).await)
    }

    fn replay_intent(&self, request: AcquireIntent) -> AcquireIntent {
        let replays_execution = self.rejected.iter().any(|message_id| {
            self.previous_inputs
                .get(message_id)
                .is_some_and(|input| input.intent() == AcquireIntent::Execute)
        });
        if replays_execution {
            AcquireIntent::Execute
        } else {
            request
        }
    }

    async fn replay(&mut self, service: &ReactAgentServiceHandle) -> Vec<SessionOutput> {
        let mut outputs = Vec::new();
        for message_id in std::mem::take(&mut self.rejected) {
            let Some(input) = self.previous_inputs.remove(&message_id) else {
                continue;
            };
            let request = SessionRequest {
                session_id: self.session_id.clone(),
                message_id: message_id.clone(),
                input,
            };
            match handle_request(service, &mut self.previous_inputs, request).await {
                Ok(replayed) => outputs.extend(
                    replayed
                        .into_iter()
                        .filter(|output| !matches!(output, SessionOutput::Ack { .. })),
                ),
                Err(ServiceClosed) => outputs.push(closed_nack(message_id)),
            }
        }
        outputs
    }
}

impl SessionInput {
    fn intent(&self) -> AcquireIntent {
        match self {
            Self::Message(_)
            | Self::SetApprovalMode(_)
            | Self::SetInferenceConfig(_)
            | Self::Compact => AcquireIntent::Execute,
            Self::Interrupt
            | Self::CancelTask { .. }
            | Self::HitlResponse { .. }
            | Self::RespondToHostTool { .. }
            | Self::DetachTools => AcquireIntent::Observe,
        }
    }
}

async fn run_session<Requests>(
    mut state: SessionState,
    requests: Requests,
    mut shutdown: broadcast::Receiver<()>,
    responses: mpsc::Sender<Result<SessionMessage, Status>>,
) where
    Requests: Stream<Item = Result<ClientMessage, Status>> + Send + 'static,
{
    tokio::pin!(requests);
    loop {
        let step = tokio::select! {
            _ = shutdown.recv() => Step::End(None),
            request = requests.next() => match request {
                Some(Ok(request)) => Step::Request(request),
                Some(Err(error)) => Step::End(Some(error)),
                None => Step::End(None),
            },
            step = state.next_step() => step,
        };
        let outputs = match step {
            Step::Request(request) => state.handle(request, &responses).await,
            Step::Event(ReactServiceEvent::InputRejected { message_id })
                if state.remembers(&message_id) =>
            {
                state.rejected.push(message_id);
                Ok(Vec::new())
            }
            Step::Event(event) => Ok(vec![SessionOutput::Event(event)]),
            Step::Lagged(count) => {
                let _ = send_output(SessionOutput::EventStreamLagged(count), &responses).await;
                return;
            }
            Step::Detached => state.detached().await,
            Step::End(status) => {
                if !reject_held(&mut state, &responses).await {
                    return;
                }
                if let Some(status) = status {
                    let _ = responses.send(Err(status)).await;
                }
                return;
            }
        };
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(status) => {
                if reject_held(&mut state, &responses).await {
                    let _ = responses.send(Err(status)).await;
                }
                return;
            }
        };
        for output in outputs {
            if !send_output(output, &responses).await {
                return;
            }
        }
    }
}

async fn handle_request(
    service: &ReactAgentServiceHandle,
    previous_inputs: &mut HashMap<String, SessionInput>,
    request: SessionRequest,
) -> Result<Vec<SessionOutput>, ServiceClosed> {
    let message_id = request.message_id;
    if message_id.is_empty() {
        return Ok(vec![nack_output(
            message_id,
            SessionNackReason::InvalidRequest,
            "message_id must not be empty".to_string(),
        )]);
    }
    if request.session_id != service.session_id() {
        return Ok(vec![nack_output(
            message_id,
            SessionNackReason::SessionMismatch,
            format!(
                "session {:?} does not match {:?}",
                request.session_id,
                service.session_id()
            ),
        )]);
    }
    if let Some(previous) = previous_inputs.get(&message_id) {
        return Ok(if previous == &request.input {
            vec![SessionOutput::Ack { message_id }]
        } else {
            vec![nack_output(
                message_id.clone(),
                SessionNackReason::MessageIdConflict,
                format!("message_id {message_id:?} was reused for different input"),
            )]
        });
    }

    let accepted = match request.input.clone() {
        SessionInput::Message(message) => match service.accept(*message).await {
            Ok(()) => true,
            Err(ReactServiceError::Closed) => return Err(ServiceClosed),
            Err(ReactServiceError::MessageIdConflict(id)) => {
                return Ok(vec![nack_output(
                    message_id,
                    SessionNackReason::MessageIdConflict,
                    format!("message_id {id:?} was reused for different input"),
                )]);
            }
            Err(ReactServiceError::Agent(error)) => {
                return Ok(vec![nack_output(
                    message_id,
                    SessionNackReason::InvalidRequest,
                    error,
                )]);
            }
            Err(error) => {
                return Ok(vec![nack_output(
                    message_id,
                    SessionNackReason::TargetNotFound,
                    error.to_string(),
                )]);
            }
        },
        SessionInput::Compact => match service.compact().await {
            Ok(()) => true,
            Err(ReactServiceError::Closed) => return Err(ServiceClosed),
            Err(error) => {
                return Ok(vec![nack_output(
                    message_id,
                    SessionNackReason::InvalidRequest,
                    error.to_string(),
                )])
            }
        },
        SessionInput::Interrupt => {
            let running = match service.interrupt() {
                Ok(running) => running.await.unwrap_or(false),
                Err(ReactServiceError::Closed) => return Err(ServiceClosed),
                Err(_) => false,
            };
            if !running {
                return Ok(vec![nack_output(
                    message_id,
                    SessionNackReason::TargetNotFound,
                    "no turn is running".to_string(),
                )]);
            }
            true
        }
        SessionInput::CancelTask { task_id } => accepted(service.cancel_task(task_id).await)?,
        SessionInput::HitlResponse {
            call_id,
            approved,
            source,
        } => accepted(service.respond_to_hitl(call_id, approved, source).await)?,
        SessionInput::RespondToHostTool { call_id, response } => {
            accepted(service.respond_to_host_tool(call_id, response).await)?
        }
        SessionInput::DetachTools => accepted(service.detach_tools().await.map(|_| true))?,
        SessionInput::SetApprovalMode(mode) => {
            accepted(service.set_approval_mode(mode).await.map(|()| true))?
        }
        SessionInput::SetInferenceConfig(inference) => {
            if let Some(reasoning) = &inference.reasoning_effort {
                if !crate::providers::is_reasoning_effort(reasoning) {
                    return Ok(vec![nack_output(
                        message_id,
                        SessionNackReason::InvalidRequest,
                        format!(
                            "invalid reasoning effort: {reasoning:?} ({})",
                            crate::providers::reasoning_effort_expectation()
                        ),
                    )]);
                }
            }
            accepted(service.set_inference_config(inference).await.map(|()| true))?
        }
    };
    let mut outputs = Vec::new();
    if accepted {
        previous_inputs.insert(message_id.clone(), request.input);
        outputs.push(SessionOutput::Ack { message_id });
    } else {
        outputs.push(nack_output(
            message_id,
            SessionNackReason::TargetNotFound,
            "request target was not found".to_string(),
        ));
    }
    Ok(outputs)
}

fn accepted(result: Result<bool, ReactServiceError>) -> Result<bool, ServiceClosed> {
    match result {
        Ok(accepted) => Ok(accepted),
        Err(ReactServiceError::Closed) => Err(ServiceClosed),
        Err(_) => Ok(false),
    }
}

async fn reject_held(
    state: &mut SessionState,
    responses: &mpsc::Sender<Result<SessionMessage, Status>>,
) -> bool {
    for message_id in std::mem::take(&mut state.rejected) {
        let rejected = SessionOutput::Event(ReactServiceEvent::InputRejected { message_id });
        if !send_output(rejected, responses).await {
            return false;
        }
    }
    true
}

fn lagged_status(count: u64) -> Status {
    Status::resource_exhausted(format!("agent event stream lagged by {count} events"))
}

fn service_status(error: ReactServiceError) -> Status {
    match error {
        ReactServiceError::Closed => Status::unavailable(SERVICE_CLOSED),
        error => Status::internal(error.to_string()),
    }
}

fn closed_nack(message_id: String) -> SessionOutput {
    nack_output(
        message_id,
        SessionNackReason::TargetNotFound,
        ReactServiceError::Closed.to_string(),
    )
}

fn nack_output(message_id: String, reason: SessionNackReason, error: String) -> SessionOutput {
    SessionOutput::Nack {
        message_id,
        reason,
        error,
    }
}

fn session_request(request: ClientMessage) -> Result<SessionRequest, InvalidClientMessage> {
    let session_id = request.session_id;
    let message_id = request.message_id;
    let input = match request.payload {
        Some(client_message::Payload::Message(message)) => SessionInput::Message(message),
        Some(client_message::Payload::Interrupt(_)) => SessionInput::Interrupt,
        Some(client_message::Payload::CancelTask(cancel)) => SessionInput::CancelTask {
            task_id: cancel.task_id,
        },
        Some(client_message::Payload::HitlResponse(response)) => {
            let source = match ApprovalSource::try_from(response.source) {
                Ok(
                    source @ (ApprovalSource::Unspecified
                    | ApprovalSource::Manual
                    | ApprovalSource::RememberedExact
                    | ApprovalSource::RememberedPrefix),
                ) => source,
                _ => {
                    return Err(invalid_request(
                        message_id,
                        "invalid client approval source".to_string(),
                    ))
                }
            };
            SessionInput::HitlResponse {
                call_id: response.call_id,
                approved: response.approved,
                source,
            }
        }
        Some(client_message::Payload::HostToolResult(result)) => {
            let HostToolResult {
                call_id,
                content,
                display,
                success,
            } = result;
            let response = if success {
                ToolOutput::success(content)
            } else {
                ToolOutput::error(content)
            };
            SessionInput::RespondToHostTool {
                call_id,
                response: match display {
                    Some(display) => response.with_display(display),
                    None => response,
                },
            }
        }
        Some(client_message::Payload::Compact(_)) => SessionInput::Compact,
        Some(client_message::Payload::DetachTools(_)) => SessionInput::DetachTools,
        Some(client_message::Payload::SetApprovalMode(mode)) => {
            match ApprovalMode::try_from(mode.mode) {
                Ok(ApprovalMode::Auto) => SessionInput::SetApprovalMode(AgentApprovalMode::Auto),
                Ok(ApprovalMode::Hitl) => SessionInput::SetApprovalMode(AgentApprovalMode::Hitl),
                _ => {
                    return Err(invalid_request(
                        message_id,
                        format!("invalid approval mode: {}", mode.mode),
                    ));
                }
            }
        }
        Some(client_message::Payload::SetInferenceConfig(inference)) => {
            SessionInput::SetInferenceConfig(inference)
        }
        None => {
            return Err(invalid_request(
                message_id,
                "client message has no payload".to_string(),
            ));
        }
    };
    Ok(SessionRequest {
        session_id,
        message_id,
        input,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct InvalidClientMessage {
    message_id: String,
    error: String,
}

impl InvalidClientMessage {
    fn into_output(self) -> SessionOutput {
        SessionOutput::Nack {
            message_id: self.message_id,
            reason: SessionNackReason::InvalidRequest,
            error: self.error,
        }
    }
}

fn invalid_request(message_id: String, error: String) -> InvalidClientMessage {
    InvalidClientMessage { message_id, error }
}

async fn send_output(
    output: SessionOutput,
    responses: &mpsc::Sender<Result<SessionMessage, Status>>,
) -> bool {
    match encode_session_output(output) {
        EncodedSessionOutput::Messages(messages) => {
            for message in messages {
                if responses.send(Ok(message)).await.is_err() {
                    return false;
                }
            }
            true
        }
        EncodedSessionOutput::Error(error) => {
            _ = responses.send(Err(error)).await;
            false
        }
    }
}

enum EncodedSessionOutput {
    Messages(Vec<SessionMessage>),
    Error(Status),
}

fn encode_session_output(output: SessionOutput) -> EncodedSessionOutput {
    let messages = match output {
        SessionOutput::EventStreamLagged(count) => {
            return EncodedSessionOutput::Error(lagged_status(count));
        }
        SessionOutput::Event(event) => service_responses(event),
        SessionOutput::Ack { message_id } => vec![ack(&message_id)],
        SessionOutput::Nack {
            message_id,
            reason,
            error,
        } => vec![nack(&message_id, nack_reason(reason), error)],
    };
    EncodedSessionOutput::Messages(messages)
}

fn nack_reason(reason: SessionNackReason) -> InputNackReason {
    match reason {
        SessionNackReason::InvalidRequest => InputNackReason::InvalidRequest,
        SessionNackReason::SessionMismatch => InputNackReason::SessionMismatch,
        SessionNackReason::MessageIdConflict => InputNackReason::MessageIdConflict,
        SessionNackReason::TargetNotFound => InputNackReason::TargetNotFound,
    }
}

fn service_responses(event: ReactServiceEvent) -> Vec<SessionMessage> {
    match event {
        ReactServiceEvent::Agent(event) => agent_event_responses(event),
        ReactServiceEvent::HitlRequest(call) => vec![SessionMessage {
            payload: Some(session_message::Payload::HitlRequest(HitlRequest {
                call: Some(call),
            })),
            ..SessionMessage::default()
        }],
        ReactServiceEvent::HitlResolved { call_id, approved } => vec![SessionMessage {
            payload: Some(session_message::Payload::HitlResolved(HitlResolved {
                call_id,
                approved,
            })),
            ..SessionMessage::default()
        }],
        ReactServiceEvent::TaskUpdated(info) => vec![runtime_response(
            runtime_delta::Payload::TaskUpdate(info.into()),
        )],
        ReactServiceEvent::HostToolCall(call) => vec![SessionMessage {
            payload: Some(session_message::Payload::HostToolCall(call)),
            ..SessionMessage::default()
        }],
        ReactServiceEvent::TurnStarted => {
            vec![runtime_response(runtime_delta::Payload::TurnStarted(()))]
        }
        ReactServiceEvent::TurnCompleted => {
            vec![runtime_response(runtime_delta::Payload::TurnCompleted(()))]
        }
        ReactServiceEvent::TurnInterrupted => vec![runtime_response(
            runtime_delta::Payload::TurnInterrupted(()),
        )],
        ReactServiceEvent::TurnError(message) => vec![runtime_response(
            runtime_delta::Payload::TurnError(TurnError { message }),
        )],
        ReactServiceEvent::InputRejected { message_id } => vec![nack(
            &message_id,
            InputNackReason::InvalidRequest,
            "session closed before executing this input; retry with the same message ID"
                .to_string(),
        )],
        ReactServiceEvent::Closed => Vec::new(),
    }
}

impl From<TaskInfo> for TaskUpdate {
    fn from(info: TaskInfo) -> Self {
        let status = match info.status {
            RuntimeTaskStatus::Running => ProtocolTaskStatus::Running,
            RuntimeTaskStatus::Succeeded => ProtocolTaskStatus::Succeeded,
            RuntimeTaskStatus::Failed => ProtocolTaskStatus::Failed,
            RuntimeTaskStatus::Cancelled => ProtocolTaskStatus::Cancelled,
        };
        Self {
            id: info.id,
            tool_name: info.tool_name,
            display: info.display,
            status: status.into(),
            started_at: Some(info.started_at.into()),
            ended_at: info.ended_at.map(Into::into),
            detached: info.detached,
        }
    }
}

/// Forward startup context metadata without copying prompt contents into diagnostics.
pub fn startup_responses(event: ReactServiceEvent) -> Vec<SessionMessage> {
    let ReactServiceEvent::Agent(event) = event else {
        return Vec::new();
    };
    if matches!(event, AgentEvent::ContextUsage { .. }) {
        return agent_event_responses(event);
    }
    let AgentEvent::Entries(mut entries) = event else {
        return Vec::new();
    };
    entries.retain_mut(|entry| {
        let Some(agent_message::Payload::ContextWindow(window)) = &mut entry.payload else {
            return false;
        };
        window.system_prompt.clear();
        window.metadata = None;
        window.compaction = None;
        true
    });
    agent_event_responses(AgentEvent::Entries(entries))
}

fn agent_event_responses(event: AgentEvent) -> Vec<SessionMessage> {
    let payload = match event {
        AgentEvent::Entries(entries) => {
            return entries
                .iter()
                .map(|entry| agent_response(entry, false, None))
                .collect();
        }
        AgentEvent::StreamFinal {
            entry,
            tokens_per_second,
        } => return vec![agent_response(&entry, true, tokens_per_second)],
        AgentEvent::StreamContent(value) => runtime_delta::Payload::StreamContent(value),
        AgentEvent::StreamReasoning(value) => runtime_delta::Payload::StreamReasoning(value),
        AgentEvent::StreamRefusal(value) => runtime_delta::Payload::StreamRefusal(value),
        AgentEvent::StreamToolCall(value) => runtime_delta::Payload::StreamToolCall(value),
        AgentEvent::StreamUsage(value) => runtime_delta::Payload::StreamUsage(value),
        AgentEvent::StreamAborted => runtime_delta::Payload::StreamAborted(()),
        AgentEvent::CompactionStarted => runtime_delta::Payload::CompactionStarted(()),
        AgentEvent::ContextUsage { tokens, window } => {
            runtime_delta::Payload::ContextUsage(ContextUsage {
                tokens: tokens as u64,
                window: window as u64,
            })
        }
    };
    vec![runtime_response(payload)]
}

fn agent_response(
    entry: &AgentMessage,
    stream_final: bool,
    tokens_per_second: Option<f64>,
) -> SessionMessage {
    SessionMessage {
        payload: Some(session_message::Payload::AgentMessage(entry.clone())),
        stream_final,
        tokens_per_second,
        ..SessionMessage::default()
    }
}

fn runtime_response(payload: runtime_delta::Payload) -> SessionMessage {
    SessionMessage {
        payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
            payload: Some(payload),
        })),
        ..SessionMessage::default()
    }
}

fn ack(message_id: &str) -> SessionMessage {
    SessionMessage {
        payload: Some(session_message::Payload::InputAck(InputAck {
            message_id: message_id.to_string(),
        })),
        ..SessionMessage::default()
    }
}

fn nack(message_id: &str, reason: InputNackReason, error: String) -> SessionMessage {
    SessionMessage {
        payload: Some(session_message::Payload::InputNack(InputNack {
            message_id: message_id.to_string(),
            reason: reason.into(),
            error,
        })),
        ..SessionMessage::default()
    }
}

#[cfg(test)]
#[path = "agent_rpc_test.rs"]
mod tests;

#[cfg(test)]
#[path = "agent_rpc_task_test.rs"]
mod task_tests;
