use std::collections::HashMap;
use std::collections::HashSet;
use std::future;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::tools::ToolOutput;
use crate::Store;
use crate::StoreError;
use futures_util::future::join3;
use futures_util::future::BoxFuture;
use futures_util::stream::FuturesUnordered;
use futures_util::FutureExt;
use futures_util::StreamExt;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::ApprovalSource;
use mirror_protocol::ChatRecord;
use mirror_protocol::FunctionCall;
use mirror_protocol::InferenceConfig;
use mirror_protocol::TaskResultRecord;
use mirror_protocol::ToolApprovalOutcome;
use mirror_protocol::ToolCall;
use mirror_protocol::ToolMessage;
use mirror_protocol::UserMessage;
use thiserror::Error;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::time::Instant;
use uuid::Uuid;

use crate::providers::Llm;
use crate::react::ContextHooks;
use crate::react::ReactAgent;
use crate::react::ReactAgentError;
use crate::react::ReactContext;
use crate::react::ReactContextError;
use crate::react::TurnEnd;
use crate::system_prompt::Skill;
use crate::task_manager::TaskEvent;
use crate::task_manager::TaskManager;
use crate::tools::ToolExecutor;
use crate::tools::ToolExecutorError;
use crate::AgentApprovalMode;
use crate::AgentEvent;
use crate::ApprovalDecision;
use crate::ApprovalHook;
use crate::ApprovalRequest;
use crate::ExecutionContext;
use crate::ExecutionContextInput;
use crate::HostToolRequest;
use crate::MentionedSkills;
use crate::TaskInfo;
use crate::TaskManagerHandle;

pub struct ReactAgentService<Model> {
    agent: ReactAgent<Model>,
    tool_executor: Box<dyn ToolExecutor>,
    busy: Arc<AtomicBool>,
    context: ReactContext,
    context_events: mpsc::UnboundedReceiver<AgentEvent>,
    execution_input: ExecutionContextInput,
    approval_mode_store: Option<Arc<dyn ApprovalModeStore>>,
    approval_requests: mpsc::UnboundedReceiver<ApprovalRequest>,
    host_tool_requests: Option<mpsc::UnboundedReceiver<HostToolRequest>>,
    commands: mpsc::UnboundedReceiver<Command>,
    events: broadcast::Sender<ReactServiceEvent>,
    tasks: TaskManagerHandle,
    task_events: mpsc::UnboundedReceiver<TaskEvent>,
    pending_approvals: PendingApprovals,
    approval_resolutions: FuturesUnordered<BoxFuture<'static, ()>>,
    pending_host_tools: HashMap<String, PendingHostTool>,
    inputs: HashMap<String, agent_message::Payload>,
    resume_turn: bool,
}

#[derive(Clone)]
pub struct ReactAgentServiceHandle {
    session_id: String,
    busy: Arc<AtomicBool>,
    commands: mpsc::UnboundedSender<Command>,
    events: broadcast::Sender<ReactServiceEvent>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReactServiceEvent {
    Agent(AgentEvent),
    HitlRequest(ToolCall),
    HitlResolved { call_id: String, approved: bool },
    TaskUpdated(TaskInfo),
    HostToolCall(ToolCall),
    TurnStarted,
    TurnCompleted,
    TurnInterrupted,
    TurnError(String),
    InputRejected { message_id: String },
    Closed,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReactServiceError {
    #[error("agent service closed")]
    Closed,
    #[error("cannot modify the conversation while a turn is running")]
    ConversationBusy,
    #[error("message ID {0:?} was reused for different input")]
    MessageIdConflict(String),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("agent failed: {0}")]
    Agent(String),
}

enum Command {
    CallTool {
        name: String,
        arguments: String,
        response: oneshot::Sender<BoxFuture<'static, Result<ToolOutput, ToolExecutorError>>>,
    },
    GetMessages {
        offset: u64,
        limit: u32,
        response: oneshot::Sender<Result<(Vec<AgentMessage>, u64), ReactServiceError>>,
    },
    Execute {
        message: Box<AgentMessage>,
        accepted: Option<oneshot::Sender<Result<(), ReactServiceError>>>,
    },
    Compact(oneshot::Sender<Result<(), ReactServiceError>>),
    Interrupt(oneshot::Sender<bool>),
    RespondToHitl {
        call_id: String,
        approved: bool,
        source: ApprovalSource,
        result: oneshot::Sender<bool>,
    },
    RespondToHostTool {
        call_id: String,
        response: ToolOutput,
        result: oneshot::Sender<bool>,
    },
    CancelTask {
        task_id: String,
        result: oneshot::Sender<Result<bool, ReactServiceError>>,
    },
    DetachTools(oneshot::Sender<Result<usize, ReactServiceError>>),
    PendingApprovals(oneshot::Sender<Vec<ToolCall>>),
    SetApprovalMode {
        mode: AgentApprovalMode,
        result: oneshot::Sender<Result<(), ReactServiceError>>,
    },
    SetInferenceConfig {
        inference: InferenceConfig,
        result: oneshot::Sender<()>,
    },
    CloseIfIdle(oneshot::Sender<bool>),
    ResumeTurn(oneshot::Sender<bool>),
    Close(oneshot::Sender<()>),
    Abandon(oneshot::Sender<()>),
}

#[derive(Debug)]
enum OperationOutcome {
    Completed,
    Failed,
    Interrupted,
    Closing {
        persist_pending: bool,
        response: oneshot::Sender<()>,
    },
}

/// A decision a client still owes, with the call it was asked about.
struct PendingApproval {
    call: ToolCall,
    response: oneshot::Sender<ApprovalDecision>,
}

type PendingApprovals = HashMap<String, PendingApproval>;

struct PendingHostTool {
    response: oneshot::Sender<ToolOutput>,
    deadline: Instant,
}

/// Durable home for a session's approval mode, written before a change takes effect.
pub trait ApprovalModeStore: Send + Sync {
    fn save(&self, mode: AgentApprovalMode) -> BoxFuture<'_, Result<(), StoreError>>;
}

/// The approval mode a service starts in, and where it saves later changes.
pub struct SessionApprovalMode {
    mode: AgentApprovalMode,
    store: Option<Arc<dyn ApprovalModeStore>>,
}

impl SessionApprovalMode {
    /// Start in `mode` and save every later change to `store` before applying it.
    #[must_use]
    pub fn saved(mode: AgentApprovalMode, store: Arc<dyn ApprovalModeStore>) -> Self {
        Self {
            mode,
            store: Some(store),
        }
    }
}

impl From<AgentApprovalMode> for SessionApprovalMode {
    fn from(mode: AgentApprovalMode) -> Self {
        Self { mode, store: None }
    }
}

impl<Model: Llm + 'static> ReactAgentService<Model> {
    #[allow(clippy::too_many_arguments)]
    pub async fn start(
        mut agent: ReactAgent<Model>,
        tool_executor: Box<dyn ToolExecutor>,
        mut store: Box<dyn Store>,
        session_id: impl Into<String>,
        approval_mode: impl Into<SessionApprovalMode>,
        skills: Vec<Skill>,
        mut hooks: Vec<Box<dyn ContextHooks>>,
        host_tool_requests: Option<mpsc::UnboundedReceiver<HostToolRequest>>,
        events: broadcast::Sender<ReactServiceEvent>,
        approval_parent: Option<&ExecutionContext>,
    ) -> Result<(ReactAgentServiceHandle, impl Future<Output = ()> + Send), ReactServiceError> {
        let session_id = session_id.into();
        let approval_mode = approval_mode.into();
        let entries = store.load(&session_id, false).await?;
        let entries = repair_saved_history(store.as_mut(), &session_id, entries).await?;
        let inputs = entries
            .iter()
            .filter_map(|entry| input_payload(entry).map(|payload| (entry.id.clone(), payload)))
            .collect();
        let inference = entries
            .iter()
            .rev()
            .find_map(|entry| entry.inference.clone());
        let (commands, command_receiver) = mpsc::unbounded_channel();
        let (approval_hook, approval_requests) = ApprovalHook::channel();
        let (task_manager, tasks, task_events, task_runner) = TaskManager::new();
        let execution = ExecutionContext::new(
            approval_mode.mode,
            Some(approval_hook),
            tasks.clone(),
            approval_parent,
            agent.prompt_injection_policy(),
        );
        let execution_input = execution.input();
        execution_input.set_inference_config(inference);
        hooks.insert(0, Box::new(MentionedSkills::new(skills)));
        let (context, mut context_events) =
            ReactContext::new(session_id.clone(), entries, store, execution);
        let mut context = context.with_hooks(hooks);
        let initialized = agent.initialize(&mut context, tool_executor.as_ref()).await;
        while let Ok(event) = context_events.try_recv() {
            publish_context_event(&events, event);
        }
        initialized.map_err(agent_error)?;
        let busy = Arc::new(AtomicBool::new(false));
        let service = Self {
            agent,
            tool_executor,
            busy: busy.clone(),
            context,
            context_events,
            execution_input,
            approval_mode_store: approval_mode.store,
            approval_requests,
            host_tool_requests,
            commands: command_receiver,
            events: events.clone(),
            tasks: tasks.clone(),
            task_events,
            pending_approvals: HashMap::new(),
            approval_resolutions: FuturesUnordered::new(),
            pending_host_tools: HashMap::new(),
            inputs,
            resume_turn: false,
        };
        let runner = async move {
            join3(service.run(), task_manager.run(), task_runner).await;
        };
        Ok((
            ReactAgentServiceHandle {
                session_id,
                busy,
                commands,
                events,
            },
            runner,
        ))
    }
}

impl ReactAgentServiceHandle {
    async fn request<T>(
        &self,
        command: impl FnOnce(oneshot::Sender<T>) -> Command,
    ) -> Result<T, ReactServiceError> {
        let (result, receiver) = oneshot::channel();
        self.commands
            .send(command(result))
            .map_err(|_| ReactServiceError::Closed)?;
        receiver.await.map_err(|_| ReactServiceError::Closed)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.commands.is_closed()
    }

    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.events.receiver_count()
    }

    pub fn events(&self) -> broadcast::Receiver<ReactServiceEvent> {
        self.events.subscribe()
    }

    pub async fn open(&self) -> Result<Vec<AgentMessage>, ReactServiceError> {
        self.messages().await
    }

    pub async fn messages(&self) -> Result<Vec<AgentMessage>, ReactServiceError> {
        self.messages_page(0, u32::MAX)
            .await
            .map(|(entries, _)| entries)
    }

    pub async fn messages_page(
        &self,
        offset: u64,
        limit: u32,
    ) -> Result<(Vec<AgentMessage>, u64), ReactServiceError> {
        self.request(|response| Command::GetMessages {
            offset,
            limit,
            response,
        })
        .await?
    }

    pub async fn call_tool(
        &self,
        name: String,
        arguments: String,
    ) -> Result<ToolOutput, ReactServiceError> {
        let execution = self
            .request(|response| Command::CallTool {
                name,
                arguments,
                response,
            })
            .await?;
        tokio::select! {
            biased;
            () = self.commands.closed() => Err(ReactServiceError::Closed),
            result = execution => result.map_err(|error| ReactServiceError::Agent(error.to_string())),
        }
    }

    pub fn execute(&self, message: AgentMessage) -> Result<(), ReactServiceError> {
        self.queue_execute(message, None)
    }

    /// Accept into the actor's input queue, not necessarily durable history.
    pub async fn accept(&self, message: AgentMessage) -> Result<(), ReactServiceError> {
        let (accepted, receipt) = oneshot::channel();
        self.queue_execute(message, Some(accepted))?;
        receipt.await.map_err(|_| ReactServiceError::Closed)?
    }

    fn queue_execute(
        &self,
        message: AgentMessage,
        accepted: Option<oneshot::Sender<Result<(), ReactServiceError>>>,
    ) -> Result<(), ReactServiceError> {
        match message.payload.as_ref() {
            Some(agent_message::Payload::Chat(chat)) => {
                if chat.rewind_to.is_some() {
                    return Err(ReactServiceError::Agent(
                        "rewind_to is not supported; fork the session with MirrorRPC.ForkSession"
                            .to_string(),
                    ));
                }
                if !matches!(chat.message, Some(chat_record::Message::User(_))) {
                    return Err(ReactServiceError::Agent(
                        "service input is not a user or extension message".to_string(),
                    ));
                }
            }
            Some(agent_message::Payload::Extension(_)) => {}
            _ => {
                return Err(ReactServiceError::Agent(
                    "service input is not a user or extension message".to_string(),
                ));
            }
        }
        self.commands
            .send(Command::Execute {
                message: Box::new(message),
                accepted,
            })
            .map_err(|_| ReactServiceError::Closed)
    }

    pub fn interrupt(&self) -> Result<oneshot::Receiver<bool>, ReactServiceError> {
        let (result, running) = oneshot::channel();
        self.commands
            .send(Command::Interrupt(result))
            .map_err(|_| ReactServiceError::Closed)?;
        Ok(running)
    }

    pub async fn respond_to_hitl(
        &self,
        call_id: String,
        approved: bool,
        source: ApprovalSource,
    ) -> Result<bool, ReactServiceError> {
        self.request(|result| Command::RespondToHitl {
            call_id,
            approved,
            source,
            result,
        })
        .await
    }

    pub async fn respond_to_host_tool(
        &self,
        call_id: String,
        response: ToolOutput,
    ) -> Result<bool, ReactServiceError> {
        self.request(|result| Command::RespondToHostTool {
            call_id,
            response,
            result,
        })
        .await
    }

    pub async fn cancel_task(&self, task_id: String) -> Result<bool, ReactServiceError> {
        self.request(|result| Command::CancelTask { task_id, result })
            .await?
    }

    /// Accept compaction only while idle; completion uses the normal turn events.
    pub async fn compact(&self) -> Result<(), ReactServiceError> {
        self.request(Command::Compact).await?
    }

    pub async fn detach_tools(&self) -> Result<usize, ReactServiceError> {
        self.request(Command::DetachTools).await?
    }

    /// Calls that are still waiting for an approval decision.
    pub async fn pending_approvals(&self) -> Result<Vec<ToolCall>, ReactServiceError> {
        self.request(Command::PendingApprovals).await
    }

    /// Apply `mode` once it is saved; a failed save leaves the previous mode in effect.
    pub async fn set_approval_mode(
        &self,
        mode: AgentApprovalMode,
    ) -> Result<(), ReactServiceError> {
        self.request(|result| Command::SetApprovalMode { mode, result })
            .await?
    }

    pub async fn set_inference_config(
        &self,
        inference: InferenceConfig,
    ) -> Result<(), ReactServiceError> {
        self.request(|result| Command::SetInferenceConfig { inference, result })
            .await
    }

    pub async fn close(&self) -> Result<(), ReactServiceError> {
        match self.request(Command::Close).await {
            Ok(()) | Err(ReactServiceError::Closed) => Ok(()),
            Err(error) => Err(error),
        }
    }

    pub async fn close_if_idle(&self) -> Result<bool, ReactServiceError> {
        self.request(Command::CloseIfIdle).await
    }

    pub async fn resume_turn(&self) -> Result<bool, ReactServiceError> {
        self.request(Command::ResumeTurn).await
    }

    pub async fn abandon(&self) -> Result<(), ReactServiceError> {
        match self.request(Command::Abandon).await {
            Ok(()) | Err(ReactServiceError::Closed) => Ok(()),
            Err(error) => Err(error),
        }
    }
}

impl<Model: Llm> ReactAgentService<Model> {
    fn call_tool(
        &self,
        name: String,
        arguments: String,
    ) -> BoxFuture<'static, Result<ToolOutput, ToolExecutorError>> {
        self.tool_executor
            .call_tool(name, arguments, self.context.execution().clone())
    }

    async fn run(mut self) {
        loop {
            if let Err(error) = self
                .agent
                .refresh_inference(&mut self.context, self.tool_executor.as_ref())
                .await
            {
                let _ = self
                    .events
                    .send(ReactServiceEvent::TurnError(error.to_string()));
            }
            if self.resume_turn {
                self.resume_turn = false;
                if !self.run_turn(Vec::new()).await {
                    return;
                }
                continue;
            }
            let pending = self.context.flush_pending();
            if !pending.is_empty() {
                if !self.run_turn(pending).await {
                    return;
                }
                continue;
            }
            let host_tool_timeout = next_host_tool_timeout(&self.pending_host_tools);
            tokio::select! {
                Some(()) = self.approval_resolutions.next() => {}
                call_id = wait_for_host_tool_timeout(host_tool_timeout) => {
                    if let Some(call_id) = call_id {
                        timeout_host_tool(&mut self.pending_host_tools, &call_id);
                    }
                }
                command = self.commands.recv() => {
                    let Some(command) = command else {
                        self.stop(true).await;
                        return;
                    };
                    if !self.handle_command(command).await {
                        return;
                    }
                }
                Some(event) = self.context_events.recv() => {
                    publish_context_event(&self.events, event);
                }
                Some(request) = self.approval_requests.recv() => {
                    handle_approval(
                        self.execution_input.approval_mode(),
                        &mut self.pending_approvals,
                        &self.events,
                        request,
                    );
                }
                request = receive_host_tool(&mut self.host_tool_requests) => {
                    if let Some(request) = request {
                        forward_host_tool(
                            &mut self.pending_host_tools,
                            &self.events,
                            request,
                        );
                    } else {
                        self.host_tool_requests = None;
                    }
                }
                Some(event) = self.task_events.recv() => {
                    handle_task_event(&self.context.input(), &self.events, event);
                }
            }
        }
    }

    async fn handle_command(&mut self, command: Command) -> bool {
        match command {
            Command::CallTool {
                name,
                arguments,
                response,
            } => {
                let _ = response.send(self.call_tool(name, arguments));
            }
            Command::GetMessages {
                offset,
                limit,
                response,
            } => {
                let _ = response.send(
                    self.context
                        .history(offset, limit)
                        .await
                        .map_err(ReactServiceError::from),
                );
            }
            Command::Execute { message, accepted } => {
                let result = accept_input(&self.context.input(), &mut self.inputs, *message);
                if let Some(accepted) = accepted {
                    let _ = accepted.send(result);
                } else if let Err(error) = result {
                    self.reject_turn(error);
                }
            }
            Command::Compact(result) => {
                let tasks_idle = self.tasks.list().await.is_ok_and(|tasks| tasks.is_empty());
                if !tasks_idle
                    || self.context.has_pending()
                    || !self.pending_approvals.is_empty()
                    || !self.pending_host_tools.is_empty()
                {
                    let _ = result.send(Err(ReactServiceError::ConversationBusy));
                } else {
                    let _ = result.send(Ok(()));
                    return self.run_compaction().await;
                }
            }
            Command::Interrupt(result) => {
                let _ = result.send(false);
            }
            Command::RespondToHitl {
                call_id,
                approved,
                source,
                result,
            } => {
                self.approval_resolutions.push(resolve_hitl(
                    &mut self.pending_approvals,
                    &self.events,
                    call_id,
                    approved,
                    source,
                    Some(result),
                ));
            }
            Command::RespondToHostTool {
                call_id,
                response,
                result,
            } => {
                let _ = result.send(resolve_host_tool(
                    &mut self.pending_host_tools,
                    &call_id,
                    response,
                ));
            }
            Command::CancelTask { task_id, result } => {
                let response = self
                    .tasks
                    .cancel(&task_id)
                    .await
                    .map_err(|error| ReactServiceError::Agent(error.to_string()));
                let _ = result.send(response);
            }
            Command::DetachTools(result) => {
                let response = self
                    .tasks
                    .detach_foreground()
                    .await
                    .map_err(|error| ReactServiceError::Agent(error.to_string()));
                let _ = result.send(response);
            }
            Command::PendingApprovals(result) => {
                let _ = result.send(pending_calls(&self.pending_approvals));
            }
            Command::SetApprovalMode { mode, result } => {
                let saved = save_approval_mode(self.approval_mode_store.as_deref(), mode).await;
                if saved.is_ok() {
                    set_approval_mode(
                        &self.execution_input,
                        &mut self.pending_approvals,
                        &mut self.approval_resolutions,
                        &self.events,
                        mode,
                    );
                }
                let _ = result.send(saved);
            }
            Command::SetInferenceConfig { inference, result } => {
                self.execution_input.set_inference_config(Some(inference));
                let _ = result.send(());
            }
            Command::CloseIfIdle(result) => {
                let tasks_idle = self.tasks.list().await.is_ok_and(|tasks| tasks.is_empty());
                self.publish_task_events();
                if !tasks_idle || self.context.has_pending() {
                    let _ = result.send(false);
                    return true;
                }
                self.stop(true).await;
                let _ = result.send(true);
                return false;
            }
            Command::ResumeTurn(result) => {
                self.resume_turn = awaits_model(self.context.entries());
                let _ = result.send(self.resume_turn);
            }
            Command::Close(result) => {
                self.stop(true).await;
                let _ = result.send(());
                return false;
            }
            Command::Abandon(result) => {
                self.stop(false).await;
                let _ = result.send(());
                return false;
            }
        }
        true
    }

    async fn run_turn(&mut self, pending: Vec<AgentMessage>) -> bool {
        let _busy = BusyFlag::raise(self.busy.clone());
        let should_run = pending.is_empty()
            || pending
                .iter()
                .any(|entry| !entry.to_llm_messages().is_empty());
        if !pending.is_empty() {
            let pending_ids = pending
                .iter()
                .map(|entry| entry.id.clone())
                .collect::<Vec<_>>();
            if let Err(error) = self.context.push(AgentEvent::Entries(pending)).await {
                self.reject_turn(service_error(error));
                for pending_id in pending_ids {
                    if self.inputs.remove(&pending_id).is_some() {
                        let _ = self.events.send(ReactServiceEvent::InputRejected {
                            message_id: pending_id,
                        });
                    }
                }
                self.stop(false).await;
                return false;
            }
        }
        if !should_run {
            return true;
        }
        let execution = self.context.execution().clone();
        let outcome = self
            .drive_operation(|agent, context, tools| agent.run(context, tools).boxed())
            .await;
        let interrupted = matches!(
            outcome,
            OperationOutcome::Interrupted | OperationOutcome::Closing { .. }
        );
        let closing = matches!(outcome, OperationOutcome::Closing { .. });
        if interrupted && !self.context.append_failed() {
            reject_pending_approvals(&mut self.pending_approvals, &self.events);
            cancel_foreground_tasks(&self.tasks).await;
            self.publish_task_events();
            let pending_messages = if closing {
                Vec::new()
            } else {
                self.take_interrupted_messages()
            };
            if let Err(error) = self.finalize_interrupted_tools().await {
                let _ = self
                    .events
                    .send(ReactServiceEvent::TurnError(error.to_string()));
            }
            if !pending_messages.is_empty() {
                let pending_ids = pending_messages
                    .iter()
                    .map(|message| message.id.clone())
                    .collect::<Vec<_>>();
                if let Err(error) = self
                    .context
                    .push(AgentEvent::Entries(pending_messages))
                    .await
                {
                    for pending_id in pending_ids {
                        if self.inputs.remove(&pending_id).is_some() {
                            let _ = self.events.send(ReactServiceEvent::InputRejected {
                                message_id: pending_id,
                            });
                        }
                    }
                    let _ = self
                        .events
                        .send(ReactServiceEvent::TurnError(error.to_string()));
                }
            }
        }
        if matches!(outcome, OperationOutcome::Completed) && !self.context.append_failed() {
            self.context
                .run_turn_end_hooks(TurnEnd {
                    execution: &execution,
                    tools: self.tool_executor.tools(),
                })
                .await;
        }
        self.finish_operation(outcome).await
    }

    async fn run_compaction(&mut self) -> bool {
        let _busy = BusyFlag::raise(self.busy.clone());
        let outcome = self
            .drive_operation(|agent, context, tools| agent.compact_now(context, tools).boxed())
            .await;
        self.finish_operation(outcome).await
    }

    /// Keep session controls responsive while an operation owns the agent and context.
    async fn drive_operation(
        &mut self,
        operation: impl for<'a> FnOnce(
            &'a mut ReactAgent<Model>,
            &'a mut ReactContext,
            &'a dyn ToolExecutor,
        ) -> BoxFuture<'a, Result<(), ReactAgentError>>,
    ) -> OperationOutcome {
        let _ = self.events.send(ReactServiceEvent::TurnStarted);
        let events = self.events.clone();
        let commands = &mut self.commands;
        let context_events = &mut self.context_events;
        let approval_requests = &mut self.approval_requests;
        let host_tool_requests = &mut self.host_tool_requests;
        let task_events = &mut self.task_events;
        let tasks = &self.tasks;
        let pending_approvals = &mut self.pending_approvals;
        let approval_resolutions = &mut self.approval_resolutions;
        let pending_host_tools = &mut self.pending_host_tools;
        let inputs = &mut self.inputs;
        let input = self.context.input();
        let execution_input = &self.execution_input;
        let approval_mode_store = self.approval_mode_store.clone();
        let mut approval_mode = execution_input.approval_mode();
        let execution = self.context.execution().clone();
        let tool_executor = self.tool_executor.as_ref();
        let outcome = {
            let run =
                AssertUnwindSafe(operation(&mut self.agent, &mut self.context, tool_executor))
                    .catch_unwind();
            tokio::pin!(run);
            loop {
                let host_tool_timeout = next_host_tool_timeout(pending_host_tools);
                tokio::select! {
                    Some(()) = approval_resolutions.next() => {}
                    call_id = wait_for_host_tool_timeout(host_tool_timeout) => {
                        if let Some(call_id) = call_id {
                            timeout_host_tool(pending_host_tools, &call_id);
                        }
                    }
                    command = commands.recv() => match command.unwrap_or_else(|| {
                        let (response, _) = oneshot::channel();
                        Command::Close(response)
                    }) {
                        Command::Execute { message, accepted } => {
                            let result = accept_input(&input, inputs, *message);
                            if let Some(accepted) = accepted {
                                let _ = accepted.send(result);
                            } else if let Err(error) = result {
                                let _ = events.send(ReactServiceEvent::TurnError(error.to_string()));
                            }
                        }
                        Command::Compact(result) => {
                            let _ = result.send(Err(ReactServiceError::ConversationBusy));
                        }
                        Command::Interrupt(result) => {
                            let _ = result.send(true);
                            break OperationOutcome::Interrupted;
                        }
                        Command::RespondToHitl { call_id, approved, source, result } => {
                            approval_resolutions.push(resolve_hitl(
                                pending_approvals, &events, call_id, approved, source, Some(result),
                            ));
                        }
                        Command::RespondToHostTool {
                            call_id,
                            response,
                            result,
                        } => {
                            let _ = result.send(resolve_host_tool(
                                pending_host_tools,
                                &call_id,
                                response,
                            ));
                        }
                        Command::CancelTask { task_id, result } => {
                            let response = tasks
                                .cancel(&task_id)
                                .await
                                .map_err(|error| ReactServiceError::Agent(error.to_string()));
                            let _ = result.send(response);
                        }
                        Command::DetachTools(result) => {
                            let response = tasks
                                .detach_foreground()
                                .await
                                .map_err(|error| ReactServiceError::Agent(error.to_string()));
                            let _ = result.send(response);
                        }
                        Command::PendingApprovals(result) => {
                            let _ = result.send(pending_calls(pending_approvals));
                        }
                        Command::SetApprovalMode { mode, result } => {
                            let saved =
                                save_approval_mode(approval_mode_store.as_deref(), mode).await;
                            if saved.is_ok() {
                                approval_mode = mode;
                                set_approval_mode(
                                    execution_input,
                                    pending_approvals,
                                    approval_resolutions,
                                    &events,
                                    mode,
                                );
                            }
                            let _ = result.send(saved);
                        }
                        Command::SetInferenceConfig { inference, result } => {
                            execution_input.set_inference_config(Some(inference));
                            let _ = result.send(());
                        }
                        Command::CallTool { name, arguments, response } => {
                            let _ = response.send(tool_executor.call_tool(name, arguments, execution.clone()));
                        }
                        Command::GetMessages { response, .. } => {
                            let _ = response.send(Err(ReactServiceError::ConversationBusy));
                        }
                        Command::CloseIfIdle(result) | Command::ResumeTurn(result) => {
                            let _ = result.send(false);
                        }
                        Command::Close(result) => {
                            break OperationOutcome::Closing { persist_pending: true, response: result };
                        }
                        Command::Abandon(result) => {
                            break OperationOutcome::Closing { persist_pending: false, response: result };
                        }
                    },
                    result = run.as_mut() => {
                        match result {
                            Ok(Ok(())) => break OperationOutcome::Completed,
                            Ok(Err(error)) => {
                                let _ = events.send(ReactServiceEvent::TurnError(error.to_string()));
                            }
                            Err(payload) => {
                                let _ = events.send(ReactServiceEvent::TurnError(format!(
                                    "agent panicked: {}",
                                    panic_message(payload),
                                )));
                            }
                        }
                        break OperationOutcome::Failed;
                    }
                    Some(event) = context_events.recv() => publish_context_event(&events, event),
                    Some(request) = approval_requests.recv() => {
                        handle_approval(
                            approval_mode,
                            pending_approvals,
                            &events,
                            request,
                        );
                    }
                    request = receive_host_tool(host_tool_requests) => {
                        if let Some(request) = request {
                            forward_host_tool(pending_host_tools, &events, request);
                        } else {
                            *host_tool_requests = None;
                        }
                    }
                    Some(event) = task_events.recv() => handle_task_event(&input, &events, event),
                    else => break OperationOutcome::Failed,
                }
            }
        };
        while let Some(Some(())) = self.approval_resolutions.next().now_or_never() {}
        self.pending_host_tools
            .retain(|_, pending| !pending.response.is_closed());
        outcome
    }

    async fn finish_operation(&mut self, outcome: OperationOutcome) -> bool {
        let interrupted = matches!(
            outcome,
            OperationOutcome::Interrupted | OperationOutcome::Closing { .. }
        );
        let close = match outcome {
            OperationOutcome::Closing {
                persist_pending,
                response,
            } => Some((persist_pending, response)),
            _ => None,
        };
        let closing = close.is_some() || self.context.append_failed();
        if closing {
            let persist_pending = close.as_ref().is_some_and(|(persist, _)| *persist)
                && !self.context.append_failed();
            self.stop(persist_pending).await;
        }
        self.publish_context_events();
        if interrupted {
            let _ = self.events.send(ReactServiceEvent::TurnInterrupted);
        }
        let _ = self.events.send(ReactServiceEvent::TurnCompleted);
        if let Some((_, response)) = close {
            let _ = response.send(());
        }
        !closing
    }

    async fn finalize_interrupted_tools(&mut self) -> Result<(), ReactServiceError> {
        let mut repaired = repair_unanswered_tools(self.context.entries())?;
        let unchanged = unchanged_prefix(self.context.entries(), &repaired);
        if unchanged != self.context.entries().len() {
            return Err(invalid_history(
                "active conversation requires historical tool repair",
            ));
        }
        let repairs = repaired.split_off(unchanged);
        if repairs.is_empty() {
            return Ok(());
        }
        self.context
            .push(AgentEvent::Entries(repairs))
            .await
            .map_err(service_error)
    }

    fn take_interrupted_messages(&mut self) -> Vec<AgentMessage> {
        self.context
            .flush_pending()
            .into_iter()
            .filter_map(|message| {
                if matches!(message.chat_message(), Some(chat_record::Message::User(_))) {
                    self.inputs.remove(&message.id);
                    None
                } else {
                    Some(message)
                }
            })
            .collect()
    }

    fn publish_context_events(&mut self) {
        while let Ok(event) = self.context_events.try_recv() {
            publish_context_event(&self.events, event);
        }
    }

    fn publish_task_events(&mut self) {
        let input = self.context.input();
        while let Ok(event) = self.task_events.try_recv() {
            handle_task_event(&input, &self.events, event);
        }
    }

    async fn stop(&mut self, persist_pending: bool) {
        reject_pending_approvals(&mut self.pending_approvals, &self.events);
        fail_host_tools(&mut self.pending_host_tools);
        let _ = self.tasks.close().await;
        self.publish_task_events();
        let pending = self.context.flush_pending();
        if !persist_pending || self.context.append_failed() {
            for dropped in &pending {
                if self.inputs.remove(&dropped.id).is_some() {
                    let _ = self.events.send(ReactServiceEvent::InputRejected {
                        message_id: dropped.id.clone(),
                    });
                }
            }
        } else if !pending.is_empty() {
            if let Err(error) = self.context.push(AgentEvent::Entries(pending)).await {
                let _ = self
                    .events
                    .send(ReactServiceEvent::TurnError(error.to_string()));
            }
        }
        self.commands.close();
        while let Some(command) = self.commands.recv().await {
            if let Command::Execute { message, .. } = command {
                let _ = self.events.send(ReactServiceEvent::InputRejected {
                    message_id: message.id,
                });
            }
        }
        self.publish_context_events();
        let _ = self.events.send(ReactServiceEvent::Closed);
    }

    fn reject_turn(&self, error: ReactServiceError) {
        let _ = self
            .events
            .send(ReactServiceEvent::TurnError(error.to_string()));
        let _ = self.events.send(ReactServiceEvent::TurnCompleted);
    }
}

async fn cancel_foreground_tasks(tasks: &TaskManagerHandle) {
    let Ok(listed) = tasks.list().await else {
        return;
    };
    for task in listed {
        if !task.detached {
            let _ = tasks.cancel(&task.id).await;
        }
    }
}

struct BusyFlag(Arc<AtomicBool>);

impl BusyFlag {
    fn raise(flag: Arc<AtomicBool>) -> Self {
        flag.store(true, Ordering::Release);
        Self(flag)
    }
}

impl Drop for BusyFlag {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

fn accept_input(
    input: &crate::react::ReactContextInput,
    inputs: &mut HashMap<String, agent_message::Payload>,
    message: AgentMessage,
) -> Result<(), ReactServiceError> {
    let payload = input_payload(&message).ok_or_else(|| {
        ReactServiceError::Agent("service input is not a durable message".to_string())
    })?;
    if let Some(previous) = inputs.get(&message.id) {
        return if previous == &payload {
            Ok(())
        } else {
            Err(ReactServiceError::MessageIdConflict(message.id))
        };
    }
    let message_id = message.id.clone();
    input.push(message).map_err(service_error)?;
    inputs.insert(message_id, payload);
    Ok(())
}

fn input_payload(entry: &AgentMessage) -> Option<agent_message::Payload> {
    match entry.payload.as_ref()? {
        agent_message::Payload::Chat(chat) => {
            let Some(chat_record::Message::User(user)) = &chat.message else {
                return None;
            };
            Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::User(UserMessage {
                    content: user.content.clone(),
                    checkpoint: None,
                })),
                ..chat.clone()
            }))
        }
        agent_message::Payload::Extension(extension) => {
            Some(agent_message::Payload::Extension(extension.clone()))
        }
        _ => None,
    }
}

fn agent_error(error: ReactAgentError) -> ReactServiceError {
    match error {
        ReactAgentError::Context(error) => service_error(error),
        error => ReactServiceError::Agent(error.to_string()),
    }
}

fn service_error(error: ReactContextError) -> ReactServiceError {
    match error {
        ReactContextError::Store(error) => ReactServiceError::Store(error),
        error => ReactServiceError::Agent(error.to_string()),
    }
}

fn publish_context_event(events: &broadcast::Sender<ReactServiceEvent>, event: AgentEvent) {
    let _ = events.send(ReactServiceEvent::Agent(event));
}

async fn receive_host_tool(
    requests: &mut Option<mpsc::UnboundedReceiver<HostToolRequest>>,
) -> Option<HostToolRequest> {
    match requests {
        Some(requests) => requests.recv().await,
        None => future::pending().await,
    }
}

fn forward_host_tool(
    pending: &mut HashMap<String, PendingHostTool>,
    events: &broadcast::Sender<ReactServiceEvent>,
    request: HostToolRequest,
) {
    let call_id = loop {
        let candidate = Uuid::new_v4().to_string();
        if !pending.contains_key(&candidate) {
            break candidate;
        }
    };
    pending.insert(
        call_id.clone(),
        PendingHostTool {
            response: request.response,
            deadline: Instant::now() + request.timeout,
        },
    );
    let event = ReactServiceEvent::HostToolCall(ToolCall {
        id: call_id.clone(),
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: request.name,
            arguments: request.arguments,
        }),
        provider_fields: None,
    });
    if events.send(event).is_err() {
        if let Some(pending) = pending.remove(&call_id) {
            let _ = pending
                .response
                .send(ToolOutput::error("host tool session is unavailable"));
        }
    }
}

fn next_host_tool_timeout(pending: &HashMap<String, PendingHostTool>) -> Option<(String, Instant)> {
    pending
        .iter()
        .min_by_key(|(_, pending)| pending.deadline)
        .map(|(call_id, pending)| (call_id.clone(), pending.deadline))
}

async fn wait_for_host_tool_timeout(timeout: Option<(String, Instant)>) -> Option<String> {
    let Some((call_id, deadline)) = timeout else {
        return future::pending().await;
    };
    tokio::time::sleep_until(deadline).await;
    Some(call_id)
}

fn timeout_host_tool(pending: &mut HashMap<String, PendingHostTool>, call_id: &str) {
    if let Some(pending) = pending.remove(call_id) {
        let _ = pending
            .response
            .send(ToolOutput::error("host tool call timed out"));
    }
}

fn resolve_host_tool(
    pending: &mut HashMap<String, PendingHostTool>,
    call_id: &str,
    response: ToolOutput,
) -> bool {
    let Some(pending) = pending.remove(call_id) else {
        return false;
    };
    pending.response.send(response).is_ok()
}

fn fail_host_tools(pending: &mut HashMap<String, PendingHostTool>) {
    for (_, pending) in pending.drain() {
        let _ = pending
            .response
            .send(ToolOutput::error("host tool session closed"));
    }
}

fn handle_approval(
    mode: AgentApprovalMode,
    pending: &mut PendingApprovals,
    events: &broadcast::Sender<ReactServiceEvent>,
    request: ApprovalRequest,
) {
    if request.response.is_closed() {
        return;
    }
    if mode == AgentApprovalMode::Auto {
        let _ = request
            .response
            .send(ApprovalDecision::new(true, ApprovalSource::Automatic));
    } else {
        let call_id = request.call.id.clone();
        pending.insert(
            call_id,
            PendingApproval {
                call: request.call.clone(),
                response: request.response,
            },
        );
        let _ = events.send(ReactServiceEvent::HitlRequest(request.call));
    }
}

fn resolve_hitl(
    pending: &mut PendingApprovals,
    events: &broadcast::Sender<ReactServiceEvent>,
    call_id: String,
    approved: bool,
    source: ApprovalSource,
    result: Option<oneshot::Sender<bool>>,
) -> BoxFuture<'static, ()> {
    let response = pending.remove(&call_id).map(|pending| pending.response);
    let events = events.clone();
    Box::pin(async move {
        let accepted = if let Some(response) = response {
            let (committed, receipt) = oneshot::channel();
            let _ = response.send(ApprovalDecision {
                approved,
                source,
                committed: Some(committed),
            });
            let accepted = receipt.await.is_ok();
            let _ = events.send(ReactServiceEvent::HitlResolved {
                call_id,
                approved: approved && accepted,
            });
            accepted
        } else {
            false
        };
        if let Some(result) = result {
            let _ = result.send(accepted);
        }
    })
}

fn pending_calls(pending: &PendingApprovals) -> Vec<ToolCall> {
    pending
        .values()
        .filter(|pending| !pending.response.is_closed())
        .map(|pending| pending.call.clone())
        .collect()
}

fn reject_pending_approvals(
    pending: &mut PendingApprovals,
    events: &broadcast::Sender<ReactServiceEvent>,
) {
    for (call_id, pending) in pending.drain() {
        let _ = pending
            .response
            .send(ApprovalDecision::new(false, ApprovalSource::Unspecified));
        let _ = events.send(ReactServiceEvent::HitlResolved {
            call_id,
            approved: false,
        });
    }
}

/// Changes are saved one at a time, in the order this service applies them.
async fn save_approval_mode(
    store: Option<&dyn ApprovalModeStore>,
    mode: AgentApprovalMode,
) -> Result<(), ReactServiceError> {
    match store {
        Some(store) => Ok(store.save(mode).await?),
        None => Ok(()),
    }
}

fn set_approval_mode(
    input: &ExecutionContextInput,
    pending: &mut PendingApprovals,
    resolutions: &mut FuturesUnordered<BoxFuture<'static, ()>>,
    events: &broadcast::Sender<ReactServiceEvent>,
    mode: AgentApprovalMode,
) {
    input.set_approval_mode(mode);
    if mode == AgentApprovalMode::Auto {
        while let Some(call_id) = pending.keys().next().cloned() {
            resolutions.push(resolve_hitl(
                pending,
                events,
                call_id,
                true,
                ApprovalSource::Automatic,
                None,
            ));
        }
    }
}

fn handle_task_event(
    input: &crate::react::ReactContextInput,
    events: &broadcast::Sender<ReactServiceEvent>,
    event: TaskEvent,
) {
    match event {
        TaskEvent::Updated(info) => {
            let _ = events.send(ReactServiceEvent::TaskUpdated(info));
        }
        TaskEvent::Finished(info, result) => {
            let _ = input.push(task_result_message(info, result));
        }
    }
}

async fn repair_saved_history(
    store: &mut dyn Store,
    session_id: &str,
    mut entries: Vec<AgentMessage>,
) -> Result<Vec<AgentMessage>, ReactServiceError> {
    let mut repaired = repair_unanswered_tools(&entries)?;
    let unchanged = unchanged_prefix(&entries, &repaired);
    if unchanged == repaired.len() {
        return Ok(entries);
    }
    if unchanged != entries.len() {
        return Err(invalid_history(
            "saved conversation requires historical tool repair",
        ));
    }
    let mut repairs = repaired.split_off(unchanged);
    store
        .append(session_id, crate::head(&entries), &mut repairs)
        .await?;
    entries.extend(repairs);
    Ok(entries)
}

fn awaits_model(entries: &[AgentMessage]) -> bool {
    entries.iter().rev().find_map(owes_model).unwrap_or(false)
}

fn owes_model(entry: &AgentMessage) -> Option<bool> {
    match entry.payload.as_ref()? {
        agent_message::Payload::Chat(chat) => Some(match chat.message.as_ref()? {
            chat_record::Message::Assistant(assistant) => {
                chat.truncated || !assistant.tool_calls.is_empty()
            }
            chat_record::Message::User(_) | chat_record::Message::Tool(_) => true,
            chat_record::Message::System(_) => false,
        }),
        agent_message::Payload::ContextWindow(window) => {
            let compaction = window.compaction.as_ref()?;
            compaction
                .recent
                .iter()
                .rev()
                .find_map(owes_model)
                .or_else(|| {
                    compaction
                        .summary
                        .as_deref()
                        .is_some_and(|summary| !summary.is_empty())
                        .then_some(true)
                })
        }
        agent_message::Payload::TaskResult(_) => Some(true),
        agent_message::Payload::Extension(extension) => {
            extension.model_message.as_ref().map(|_| true)
        }
        agent_message::Payload::Log(_) | agent_message::Payload::ToolApproval(_) => None,
    }
}

fn unchanged_prefix(saved: &[AgentMessage], repaired: &[AgentMessage]) -> usize {
    saved
        .iter()
        .zip(repaired)
        .take_while(|(saved, repaired)| saved == repaired)
        .count()
}

fn repair_unanswered_tools(
    entries: &[AgentMessage],
) -> Result<Vec<AgentMessage>, ReactServiceError> {
    let mut repaired = Vec::with_capacity(entries.len());
    let mut pending: Vec<(ToolCall, Option<ToolApprovalOutcome>)> = Vec::new();
    let mut call_ids = HashSet::new();
    for entry in entries {
        if matches!(
            entry.payload,
            Some(agent_message::Payload::ContextWindow(_))
        ) {
            append_interrupted_tools(&mut repaired, &mut pending);
            call_ids.clear();
        }
        if let Some(agent_message::Payload::ToolApproval(record)) = &entry.payload {
            let Some((_, outcome)) = pending
                .iter_mut()
                .find(|(call, _)| call.id == record.call_id)
            else {
                return Err(invalid_history("approval has no unanswered call"));
            };
            *outcome = Some(record.outcome());
        }
        for message in entry.to_llm_messages() {
            match message {
                chat_record::Message::Assistant(assistant) => {
                    append_interrupted_tools(&mut repaired, &mut pending);
                    for call in &assistant.tool_calls {
                        if call.id.is_empty() {
                            return Err(invalid_history("tool call ID is empty"));
                        }
                        if !call_ids.insert(call.id.clone()) {
                            return Err(invalid_history(&format!(
                                "tool call ID {:?} is duplicated",
                                call.id
                            )));
                        }
                    }
                    pending = assistant
                        .tool_calls
                        .into_iter()
                        .map(|call| (call, None))
                        .collect();
                }
                chat_record::Message::Tool(tool) => {
                    let Some(index) = pending
                        .iter()
                        .position(|(call, _)| call.id == tool.tool_call_id)
                    else {
                        return Err(invalid_history(&format!(
                            "tool result {:?} has no unanswered call",
                            tool.tool_call_id
                        )));
                    };
                    pending.remove(index);
                }
                _ => append_interrupted_tools(&mut repaired, &mut pending),
            }
        }
        repaired.push(entry.clone());
    }
    append_interrupted_tools(&mut repaired, &mut pending);
    Ok(repaired)
}

fn append_interrupted_tools(
    repaired: &mut Vec<AgentMessage>,
    pending: &mut Vec<(ToolCall, Option<ToolApprovalOutcome>)>,
) {
    for (call, approval) in pending.drain(..) {
        if approval == Some(ToolApprovalOutcome::Requested) {
            repaired.push(AgentMessage::tool_approval(
                &call,
                ToolApprovalOutcome::Canceled,
                ApprovalSource::Unspecified,
            ));
        }
        repaired.push(interrupted_tool(&call));
    }
}

fn invalid_history(detail: &str) -> ReactServiceError {
    ReactServiceError::Agent(format!("saved conversation history is invalid: {detail}"))
}

fn interrupted_tool(call: &ToolCall) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Tool(ToolMessage {
                tool_call_id: call.id.clone(),
                content: "Tool error: interrupted".to_string(),
                name: call.function.as_ref().map(|function| function.name.clone()),
            })),
            success: false,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn task_result_message(info: TaskInfo, result: ToolOutput) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::TaskResult(TaskResultRecord {
            task_id: info.id,
            tool_name: info.tool_name,
            content: result.content,
            display: result.display,
            success: result.success,
        })),
        ..AgentMessage::default()
    }
}

#[cfg(test)]
#[path = "react_agent_service_test.rs"]
pub(crate) mod tests;
