use std::collections::HashSet;
use std::collections::VecDeque;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::providers::{ModelApi, ModelProvider, ModelProviderConfig, DEFAULT_CONTEXT_WINDOW};
use crate::tools::web_test_server::{TestResponse, TestServer};
use crate::tools::ToolDefinition;
use crate::tools::ToolOutput;
use crate::Store;
use crate::StoreError;
use assert_matches::assert_matches;
use async_stream::stream;
use futures_util::future::BoxFuture;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::ApprovalSource;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::ExtensionRecord;
use mirror_protocol::FunctionCall;
use mirror_protocol::LogRecord;
use mirror_protocol::StreamContent;
use mirror_protocol::ToolApprovalOutcome;
use mirror_protocol::ToolCall;
use mirror_protocol::ToolMessage;
use mirror_protocol::Usage;
use mirror_protocol::UserMessage;
use pretty_assertions::{assert_eq, assert_ne};
use serde_json::json;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use uuid::Uuid;

use super::*;
use crate::local::SQLiteStore;
use crate::providers::LlmError;
use crate::providers::StreamEvent;
use crate::react::InMemoryStore;
use crate::react::ReactAgentConfig;
use crate::system_prompt::Skill;
use crate::system_prompt::SystemPromptConfig;
use crate::tools::AnyTool;
use crate::tools::DirectToolExecutor;

pub(crate) async fn test_provider(window: usize) -> ModelProvider {
    let mut model = ScriptedModel::new([]);
    model.settings.context_window = Some(window);
    model.provider().await
}

fn system_prompt() -> SystemPromptConfig {
    SystemPromptConfig {
        custom_prompt: Some("system".to_string()),
        ..SystemPromptConfig::default()
    }
}

struct ScriptedModel {
    settings: ModelProviderConfig,
    responses: mpsc::UnboundedSender<oneshot::Sender<Option<ModelResponse>>>,
    requests: mpsc::UnboundedSender<CompletionRequest>,
    recorded: Option<mpsc::UnboundedReceiver<CompletionRequest>>,
}

enum ModelResponse {
    Completion(Completion),
    Error(LlmError),
    Events(Vec<Result<StreamEvent, LlmError>>),
    Panic(&'static str),
    Delayed {
        started: mpsc::UnboundedSender<()>,
        completion: oneshot::Receiver<Completion>,
    },
}

impl ScriptedModel {
    fn new(responses: impl IntoIterator<Item = ModelResponse>) -> Self {
        let mut script: VecDeque<ModelResponse> = responses.into_iter().collect();
        let (responses, mut next_response) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(reply) = next_response.recv().await {
                let reply: oneshot::Sender<Option<ModelResponse>> = reply;
                let _ = reply.send(script.pop_front());
            }
        });
        let (requests, recorded) = mpsc::unbounded_channel();
        Self {
            settings: ModelProviderConfig {
                api: ModelApi::ChatCompletions,
                model: "scripted".into(),
                context_window: Some(DEFAULT_CONTEXT_WINDOW),
                api_key: Some("synthetic-test-key".into()),
                base_url: Some("http://127.0.0.1:1".into()),
                auth_path: None,
                prompt_cache_key: "session".into(),
                request_timeout: None,
            },
            responses,
            requests,
            recorded: Some(recorded),
        }
    }

    async fn provider(&self) -> ModelProvider {
        ModelProvider::new(self.settings.clone(), None)
            .await
            .expect("test provider")
    }

    fn requests(&mut self) -> RecordedRequests {
        RecordedRequests {
            receiver: self.recorded.take().expect("requests are recorded once"),
            seen: Vec::new(),
        }
    }
}

struct RecordedRequests {
    receiver: mpsc::UnboundedReceiver<CompletionRequest>,
    seen: Vec<CompletionRequest>,
}

impl RecordedRequests {
    fn seen(&mut self) -> &[CompletionRequest] {
        while let Ok(request) = self.receiver.try_recv() {
            self.seen.push(request);
        }
        &self.seen
    }
}

impl Llm for ScriptedModel {
    fn model(&self) -> &str {
        "scripted"
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        let _ = self.requests.send(request);
        let (reply, response) = oneshot::channel();
        let _ = self.responses.send(reply);
        stream! {
            let response = response
                .await
                .ok()
                .flatten()
                .expect("each expected model request has a scripted response");
            match response {
                ModelResponse::Completion(completion) => {
                    yield Ok(StreamEvent::Final(completion));
                }
                ModelResponse::Error(error) => {
                    yield Err(error);
                }
                ModelResponse::Events(events) => {
                    for event in events {
                        yield event;
                    }
                }
                ModelResponse::Panic(message) => panic!("{message}"),
                ModelResponse::Delayed { started, completion } => {
                    let _ = started.send(());
                    match completion.await {
                        Ok(completion) => yield Ok(StreamEvent::Final(completion)),
                        Err(_) => yield Err(LlmError::Request("completion cancelled".to_string())),
                    }
                }
            }
        }
    }
}

struct FailingStore {
    inner: InMemoryStore,
    fail_user_append: bool,
    fail_approval_append: bool,
}

impl FailingStore {
    fn user_append() -> Self {
        Self {
            inner: InMemoryStore::default(),
            fail_user_append: true,
            fail_approval_append: false,
        }
    }
}

impl Store for FailingStore {
    fn load<'a>(
        &'a mut self,
        session_id: &'a str,
        full_history: bool,
    ) -> BoxFuture<'a, Result<Vec<AgentMessage>, StoreError>> {
        self.inner.load(session_id, full_history)
    }

    fn append<'a>(
        &'a mut self,
        session_id: &'a str,
        head: u64,
        entries: &'a mut [AgentMessage],
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        if self.fail_approval_append
            && entries.iter().any(|entry| {
                matches!(
                    &entry.payload, Some(agent_message::Payload::ToolApproval(record))
                        if record.outcome() == ToolApprovalOutcome::Approved
                )
            })
        {
            return Box::pin(async {
                Err(StoreError::Persistence(
                    "injected approval append failure".to_string(),
                ))
            });
        }
        if self.fail_user_append
            && entries
                .iter()
                .any(|entry| matches!(entry.chat_message(), Some(chat_record::Message::User(_))))
        {
            self.fail_user_append = false;
            return Box::pin(async {
                Err(StoreError::Persistence(
                    "injected user append failure".to_string(),
                ))
            });
        }
        self.inner.append(session_id, head, entries)
    }
}

struct ErrorTool;

impl AnyTool for ErrorTool {
    fn name(&self) -> &str {
        "fail"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: "Fail the call.".to_string(),
            parameters: serde_json::Map::new(),
        }
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async { ToolOutput::error("injected tool failure") })
    }
}

struct BlockingTool {
    started: mpsc::UnboundedSender<()>,
}

impl AnyTool for BlockingTool {
    fn name(&self) -> &str {
        "block"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: "Wait forever.".to_string(),
            parameters: serde_json::Map::new(),
        }
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let _ = self.started.send(());
            std::future::pending().await
        })
    }
}

struct HostRequestTool {
    requests: mpsc::UnboundedSender<HostToolRequest>,
}

impl AnyTool for HostRequestTool {
    fn name(&self) -> &str {
        "host"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        tool_definition(self.name())
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let (response, receiver) = oneshot::channel();
            if self
                .requests
                .send(HostToolRequest {
                    name: self.name().to_string(),
                    arguments: arguments.to_string(),
                    timeout: Duration::from_secs(300),
                    response,
                })
                .is_err()
            {
                return ToolOutput::error("host tool session is unavailable");
            }
            receiver
                .await
                .unwrap_or_else(|_| ToolOutput::error("host tool session closed"))
        })
    }
}

struct PanickingTool;

impl AnyTool for PanickingTool {
    fn name(&self) -> &str {
        "panic"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        tool_definition(self.name())
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async { panic!("injected tool panic") })
    }
}

struct PermissionTool;

impl AnyTool for PermissionTool {
    fn name(&self) -> &str {
        "permission"
    }

    fn needs_approval(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        tool_definition(self.name())
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            ToolOutput::success(
                context
                    .tool_permissions()
                    .allow_outside_workspace()
                    .to_string(),
            )
        })
    }
}

struct DetachedTool {
    started: mpsc::UnboundedSender<()>,
}

impl AnyTool for DetachedTool {
    fn name(&self) -> &str {
        "detach"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        tool_definition(self.name())
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let started = self.started.clone();
            match context
                .tasks()
                .spawn(
                    self.name(),
                    None,
                    crate::TaskBehavior::Detach,
                    move |_| async move {
                        let _ = started.send(());
                        std::future::pending().await
                    },
                )
                .await
            {
                Ok(task_id) => ToolOutput::success(format!("Task {task_id} started.")),
                Err(error) => ToolOutput::error(error.to_string()),
            }
        })
    }
}

struct CompletingDetachedTool {
    started: mpsc::UnboundedSender<oneshot::Sender<ToolOutput>>,
}

impl AnyTool for CompletingDetachedTool {
    fn name(&self) -> &str {
        "completing_detach"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        tool_definition(self.name())
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let started = self.started.clone();
            match context
                .tasks()
                .spawn(
                    self.name(),
                    None,
                    crate::TaskBehavior::Detach,
                    move |_| async move {
                        let (complete, completion) = oneshot::channel();
                        let _ = started.send(complete);
                        completion.await.unwrap_or_else(|_| {
                            ToolOutput::error("task completion sender was dropped")
                        })
                    },
                )
                .await
            {
                Ok(task_id) => ToolOutput::success(format!("Task {task_id} started.")),
                Err(error) => ToolOutput::error(error.to_string()),
            }
        })
    }
}

struct ForegroundTaskTool {
    started: mpsc::UnboundedSender<()>,
}

impl AnyTool for ForegroundTaskTool {
    fn name(&self) -> &str {
        "foreground"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        tool_definition(self.name())
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let started = self.started.clone();
            let task_id = match context
                .tasks()
                .spawn(
                    self.name(),
                    None,
                    crate::TaskBehavior::Wait,
                    move |_| async move {
                        let _ = started.send(());
                        std::future::pending().await
                    },
                )
                .await
            {
                Ok(task_id) => task_id,
                Err(error) => return ToolOutput::error(error.to_string()),
            };
            match context.tasks().resolve(&task_id).await {
                Ok(crate::TaskResolution::Finished(output)) => output,
                Ok(crate::TaskResolution::Detached) => ToolOutput::success("detached"),
                Err(error) => ToolOutput::error(error.to_string()),
            }
        })
    }
}

#[tokio::test]
async fn service_routes_client_hosted_tools_with_single_use_call_ids() {
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("model-call", "host")),
        ModelResponse::Completion(completion("done")),
    ]);
    let (requests, host_tool_requests) = mpsc::unbounded_channel();
    let tool_executor = DirectToolExecutor::new(vec![Arc::new(HostRequestTool { requests })]);
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(InMemoryStore::default()),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        Some(host_tool_requests),
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    let runtime = tokio::spawn(runtime);
    let mut events = service.events();

    service
        .execute(user_message(
            "run host tool".to_string(),
            "message-1".to_string(),
            None,
        ))
        .expect("execute turn");
    let call = loop {
        if let ReactServiceEvent::HostToolCall(call) = events.recv().await.expect("service event") {
            break call;
        }
    };

    assert_matches!(Uuid::parse_str(&call.id), Ok(id) if id.get_version_num() == 4);
    assert_matches!(
        call.function.as_ref(),
        Some(FunctionCall { name, arguments })
            if name == "host" && arguments == "{}"
    );
    assert!(service
        .respond_to_host_tool(call.id.clone(), ToolOutput::success("host result"),)
        .await
        .expect("resolve host tool"));
    assert!(!service
        .respond_to_host_tool(call.id, ToolOutput::success("replayed"))
        .await
        .expect("reject replayed result"));
    let remaining_events = wait_for_turn(&mut events).await;
    assert_matches!(
        remaining_events.last(),
        Some(ReactServiceEvent::TurnCompleted)
    );

    close_service(&service, runtime).await;
}

#[tokio::test]
async fn refreshed_prompt_recomputes_retention_or_fails_without_committing_a_window() {
    for added_chars in [12_000, 32_000] {
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("AGENTS.md");
        std::fs::write(&path, "seed").expect("initial instructions");
        let prompt_config = SystemPromptConfig {
            working_directory: Some(directory.path().to_path_buf()),
            global_instructions_dir: Some(directory.path().join("no-global")),
            global_skills_dir: Some(directory.path().join("no-skills")),
            ..SystemPromptConfig::default()
        };
        let threshold = crate::system_prompt::build_system_prompt(&prompt_config)
            .0
            .chars()
            .count()
            / 4
            + 4_000;
        let mut responses = vec![ModelResponse::Completion(completion(
            &"history ".repeat(2_000),
        ))];
        if added_chars == 12_000 {
            responses.extend([
                ModelResponse::Completion(completion("summary")),
                ModelResponse::Completion(completion("done")),
            ]);
        }
        let mut model = ScriptedModel::new(responses);
        model.settings.context_window = Some(threshold * 3);
        let mut requests = model.requests();
        let tool_executor = DirectToolExecutor::new(Vec::new());
        let provider = model.provider().await;
        let agent = ReactAgent::new(
            model,
            provider,
            None,
            &tool_executor,
            prompt_config.clone(),
            ReactAgentConfig {
                compaction_threshold: Some(threshold),
                max_tokens: 8_000,
                ..ReactAgentConfig::default()
            },
        )
        .expect("agent");
        let (service, runtime) = ReactAgentService::start(
            agent,
            Box::new(tool_executor),
            Box::new(InMemoryStore::default()),
            "session",
            AgentApprovalMode::Auto,
            Vec::new(),
            Vec::new(),
            None,
            tokio::sync::broadcast::channel(1024).0,
            None,
        )
        .await
        .expect("service");
        let runtime = tokio::spawn(runtime);
        run_turn(&service, "hello", "first").await;
        std::fs::write(&path, "x".repeat(added_chars)).expect("grow instructions");
        let (refreshed, _) = crate::system_prompt::build_system_prompt(&prompt_config);
        let user = "user input ".repeat(550);
        let mut events = service.events();
        service
            .execute(user_entry(&user, "second"))
            .expect("second turn");
        let events = wait_for_turn(&mut events).await;
        let history = service.messages().await.expect("history");
        let windows = history
            .iter()
            .filter(|entry| {
                matches!(
                    entry.payload,
                    Some(agent_message::Payload::ContextWindow(_))
                )
            })
            .count();
        if added_chars == 12_000 {
            assert!(!events
                .iter()
                .any(|event| matches!(event, ReactServiceEvent::TurnError(_))));
            assert_eq!(windows, 2);
            assert_eq!(requests.seen().len(), 3);
            let request = requests.seen().last().expect("completion request");
            assert_matches!(&request.messages[0], chat_record::Message::System(system) if system.content == refreshed);
            assert!(!request.messages.iter().any(|message| matches!(message, chat_record::Message::User(message) if message.content == user)));
            assert_final_answer(&history, "done");
        } else {
            assert_eq!(windows, 1);
            assert_eq!(requests.seen().len(), 1);
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, ReactServiceEvent::TurnError(_)))
                    .count(),
                1
            );
            assert!(!events.iter().any(|event| matches!(
                event,
                ReactServiceEvent::Agent(AgentEvent::CompactionStarted)
            )));
            assert!(!history.iter().any(|entry| matches!(entry.chat_message(), Some(chat_record::Message::Assistant(message)) if message.content.as_deref().is_some_and(|text| text.contains("maximum of")))));
        }
        close_service(&service, runtime).await;
    }
}

#[tokio::test]
async fn system_prompt_is_frozen_until_compaction_and_persisted() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let path = directory.path().join("AGENTS.md");
    let prompt_config = SystemPromptConfig {
        working_directory: Some(directory.path().to_path_buf()),
        global_instructions_dir: Some(directory.path().join("no-global")),
        global_skills_dir: Some(directory.path().join("no-skills")),
        ..SystemPromptConfig::default()
    };
    std::fs::write(&path, "original preference").expect("write profile");
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(completion("first")),
        ModelResponse::Completion(completion("summary")),
        ModelResponse::Completion(completion("second")),
    ]);
    model.settings.context_window = Some(40_000);
    let mut requests = model.requests();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        prompt_config.clone(),
        ReactAgentConfig {
            compaction_threshold: Some(10_000),
            max_tokens: 1_000,
            ..ReactAgentConfig::default()
        },
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(InMemoryStore::default()),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("start service");
    let runtime = tokio::spawn(runtime);
    std::fs::write(&path, "updated preference").expect("edit profile");
    run_turn(&service, "hello", "user-1").await;
    assert_matches!(&requests.seen()[0].messages[0], chat_record::Message::System(system) if system.content.contains("original preference") && !system.content.contains("updated preference"));
    let events = run_turn(&service, &"history ".repeat(10_000), "user-2").await;
    let batch = events
        .iter()
        .find_map(|event| match event {
            ReactServiceEvent::Agent(AgentEvent::Entries(entries))
                if entries.iter().any(|entry| {
                    matches!(&entry.payload, Some(agent_message::Payload::ContextWindow(window)) if window.compaction.is_some())
                }) =>
            {
                Some(entries)
            }
            _ => None,
        })
        .expect("compaction batch");
    assert_matches!(
        batch.as_slice(),
        [AgentMessage {
            payload: Some(agent_message::Payload::ContextWindow(_)),
            ..
        }]
    );
    assert_matches!(&requests.seen().last().expect("completion").messages[0], chat_record::Message::System(system) if system.content.contains("updated preference") && !system.content.contains("original preference"));
    let entries = service.messages().await.expect("messages");
    assert!(entries.windows(1).any(|window| window == batch.as_slice()));
    assert_eq!(batch[0].inference, Some(default_inference()));
    assert_matches!(&batch[0].payload, Some(agent_message::Payload::ContextWindow(window)) if window.metadata.is_none() && window.compaction.is_some());
    close_service(&service, runtime).await;
    std::fs::remove_file(&path).expect("delete profile");
    let mut store = InMemoryStore::default();
    let checkpoint = entries
        .iter()
        .position(|entry| entry.id == batch[0].id)
        .expect("checkpoint position");
    store
        .append("session", 0, &mut entries[..=checkpoint].to_vec())
        .await
        .expect("restore history at the window boundary");
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("resumed"))]);
    let mut requests = model.requests();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        prompt_config,
        ReactAgentConfig::default(),
    )
    .expect("create resumed agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("resume service");
    let runtime = tokio::spawn(runtime);
    let mut events = service.events();
    service
        .execute(user_message(
            "hello again".to_string(),
            "user-3".to_string(),
            None,
        ))
        .expect("new input");
    wait_for_turn(&mut events).await;
    assert_eq!(requests.seen()[0].inference, Some(default_inference()));
    assert_matches!(&requests.seen()[0].messages[0], chat_record::Message::System(system) if system.content.contains("updated preference"));
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_persists_completed_turn_to_sqlite() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([ModelResponse::Completion(completion("done"))]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;

    let opened = service.open().await.expect("open service");
    assert_eq!(opened.len(), 1);
    run_turn(&service, "hello", "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "done");
    assert_eq!(
        service.messages_page(1, 1).await.expect("service page"),
        (entries[1..2].to_vec(), 3)
    );
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.sequence)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        entries.len()
    );

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn completed_turn_reports_usage_metrics() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([ModelResponse::Completion(Completion {
        usage: Some(Usage {
            completion_tokens: 120,
            ..Usage::default()
        }),
        ..completion("done")
    })]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");

    let events = run_turn(&service, "hello", "user-1").await;

    assert_matches!(
        events.iter().find(|event| matches!(
            event,
            ReactServiceEvent::Agent(AgentEvent::ContextUsage { .. })
        )),
        Some(ReactServiceEvent::Agent(AgentEvent::ContextUsage { tokens, window }))
            if *tokens > 0 && *window == crate::providers::DEFAULT_CONTEXT_WINDOW
    );
    assert_matches!(
        events.iter().find(|event| matches!(
            event,
            ReactServiceEvent::Agent(AgentEvent::StreamFinal { .. })
        )),
        Some(ReactServiceEvent::Agent(AgentEvent::StreamFinal {
            tokens_per_second: Some(value),
            ..
        })) if value.is_finite() && *value > 0.0
    );
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn streaming_usage_carries_provider_counts_and_worker_elapsed_time() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([ModelResponse::Events(vec![
        Ok(StreamEvent::OutputTokens(4)),
        Ok(StreamEvent::OutputTokens(40)),
        Ok(StreamEvent::Final(completion("done"))),
    ])]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let events = run_turn(&service, "hello", "user-1").await;
    let samples = events
        .iter()
        .filter_map(|event| match event {
            ReactServiceEvent::Agent(AgentEvent::StreamUsage(usage)) => Some(usage),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].completion_tokens, 4);
    assert_eq!(samples[1].completion_tokens, 40);
    assert!(samples[0].elapsed_seconds >= 0.0);
    assert!(samples[1].elapsed_seconds >= samples[0].elapsed_seconds);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn model_switch_uses_the_new_endpoints_catalog_for_compaction_and_budgeting() {
    use crate::providers::{ModelApi, ModelProvider, ModelProviderConfig};

    let catalog = json!({"data": [{"id": "large", "context_length": 32768}]});
    let answer = json!({"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]});
    let mut large = TestServer::start(vec![
        TestResponse::json(200, catalog),
        TestResponse::json(200, answer.clone()),
    ])
    .await;
    let mut small = TestServer::start(vec![
        TestResponse::json(
            200,
            json!({"data": [{"id": "small", "context_length": 4096}]}),
        ),
        TestResponse::json(200, answer.clone()),
        TestResponse::json(200, answer),
    ])
    .await;
    let model = ModelProvider::new(
        ModelProviderConfig {
            api: ModelApi::ChatCompletionsNonStreaming,
            model: "large".into(),
            context_window: None,
            api_key: Some("synthetic-large-key".into()),
            base_url: Some(large.base_url.clone()),
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
        },
        None,
    )
    .await
    .expect("provider");
    let tools = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        model.clone(),
        model,
        None,
        &tools,
        system_prompt(),
        ReactAgentConfig {
            max_tokens: 8_000,
            ..Default::default()
        },
    )
    .expect("agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tools),
        Box::new(InMemoryStore::default()),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    let mut events = service.events();
    service
        .execute(user_message(
            "history ".repeat(2_000),
            "user-1".into(),
            None,
        ))
        .expect("first input");
    wait_for_turn(&mut events).await;
    assert_eq!(large.request().await.path, "/models");
    let first = large.request().await;
    assert_eq!(first.path, "/chat/completions");
    assert_eq!(first.body["model"], "large");
    assert_eq!(first.body["max_tokens"], 8_000);
    assert_eq!(
        first.authorization.as_deref(),
        Some("Bearer synthetic-large-key")
    );

    service
        .set_inference_config(mirror_protocol::InferenceConfig {
            model: "small".into(),
            base_url: Some(small.base_url.clone()),
            ..Default::default()
        })
        .await
        .expect("switch model and endpoint");
    service
        .execute(user_message("continue".into(), "user-2".into(), None))
        .expect("second input");
    let events = wait_for_turn(&mut events).await;
    let metadata = small.request().await;
    assert_eq!(metadata.path, "/models");
    assert_ne!(
        metadata.authorization.as_deref(),
        Some("Bearer synthetic-large-key")
    );
    let summary = small.request().await;
    assert_eq!(summary.path, "/chat/completions");
    assert_eq!(summary.body["model"], "small");
    let second = small.request().await;
    assert_eq!(second.path, "/chat/completions");
    assert_eq!(second.body["model"], "small");
    assert_matches!(second.body["max_tokens"].as_u64(), Some(tokens) if tokens > 0 && tokens < 4096);
    assert!(events.iter().any(|event| matches!(
        event,
        ReactServiceEvent::Agent(AgentEvent::CompactionStarted)
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ReactServiceEvent::Agent(AgentEvent::ContextUsage { window: 4096, .. })
    )));
    assert_final_answer(&service.messages().await.expect("messages"), "done");
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn inference_control_during_turn_applies_to_the_next_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (completion_sender, completion_receiver) = oneshot::channel();
    let mut model = ScriptedModel::new([
        ModelResponse::Delayed {
            started,
            completion: completion_receiver,
        },
        ModelResponse::Completion(completion("done")),
    ]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(ErrorTool)],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    let initial = mirror_protocol::InferenceConfig {
        model: "initial-model".to_string(),
        api: Some("openai-responses".to_string()),
        base_url: Some("https://initial.example.test/v1".to_string()),
        reasoning_effort: Some("medium".to_string()),
        ..Default::default()
    };
    let queued = mirror_protocol::InferenceConfig {
        model: "queued-model".to_string(),
        api: Some("openai-chat-completions".to_string()),
        base_url: Some("https://queued.example.test/v1".to_string()),
        reasoning_effort: Some("high".to_string()),
        temperature: Some(0.0),
        top_p: Some(0.9),
        seed: Some(123),
        extra_body: None,
    };

    service
        .set_inference_config(initial.clone())
        .await
        .expect("set initial inference");
    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("first input");
    started_receiver.recv().await.expect("model starts");
    service
        .set_inference_config(queued.clone())
        .await
        .expect("set queued inference");
    completion_sender
        .send(tool_completion("call-1", "fail"))
        .expect("release completion");
    wait_for_turn(&mut events).await;
    assert_eq!(
        requests
            .seen()
            .iter()
            .map(|request| request.inference.clone())
            .collect::<Vec<_>>(),
        vec![Some(initial.clone()), Some(queued.clone()),]
    );
    let completions = service
        .messages()
        .await
        .expect("service messages")
        .into_iter()
        .filter(|entry| {
            matches!(
                entry.chat_message(),
                Some(chat_record::Message::Assistant(_))
            )
        })
        .map(|entry| entry.inference)
        .collect::<Vec<_>>();
    assert_eq!(completions, vec![Some(initial), Some(queued)]);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn explicit_skill_mentions_append_one_persisted_user_context() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let store = SQLiteStore::open(&database, "session", directory.path(), true, false)
        .expect("open SQLite store");
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        ScriptedModel::new([ModelResponse::Completion(completion("done"))]),
        test_provider(DEFAULT_CONTEXT_WINDOW).await,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        "session",
        AgentApprovalMode::Auto,
        vec![Skill {
            name: "review".to_string(),
            description: "Review code.".to_string(),
            body: "Follow the checklist.".to_string(),
            path: std::path::PathBuf::from("skills/review&fix/SKILL.md"),
            disable_model_invocation: false,
        }],
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    let runtime = tokio::spawn(runtime);
    service.open().await.expect("open service");

    run_turn(&service, "$review again $review", "message-1").await;
    let messages = service.messages().await.expect("messages");

    assert_matches!(
        &messages[1].payload,
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage { content, .. })),
            ..
        })) if content == "$review again $review"
    );
    assert_matches!(
        &messages[2].payload,
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage { content, .. })),
            display: Some(display),
            ..
        })) if display == "$review"
            && content == "<skill>\n<name>review</name>\n<path>skills/review&amp;fix/SKILL.md</path>\nFollow the checklist.\n</skill>"
    );
    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, messages);
}

#[tokio::test]
async fn service_recovers_from_failed_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([
        ModelResponse::Error(LlmError::Request("offline".to_string())),
        ModelResponse::Completion(completion("recovered")),
    ]);
    let config = ReactAgentConfig {
        max_exceptions: 2,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;

    service.open().await.expect("open service");
    run_turn(&service, "hello", "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Log(log)) => Some(log),
            _ => None,
        }),
        Some(LogRecord { level, text })
            if level == "error" && text.contains("model request failed: offline")
    );
    assert_final_answer(&entries, "recovered");

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn service_compacts_and_continues_after_truncated_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(truncated_completion("partial")),
        ModelResponse::Completion(completion("continued")),
    ]);
    model.settings.context_window = Some(100);
    let mut requests = model.requests();
    let config = ReactAgentConfig {
        max_tokens: 90,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;

    service.open().await.expect("open service");
    run_turn(&service, &"x".repeat(100), "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Chat(chat)) if chat.truncated => Some(chat),
            _ => None,
        }),
        Some(ChatRecord {
            message: Some(chat_record::Message::Assistant(AssistantMessage {
                content: Some(content),
                ..
            })),
            ..
        }) if content == "partial"
    );
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Log(log))
                if log.text.contains("response was cut short") => Some(log),
            _ => None,
        }),
        Some(LogRecord { level, .. }) if level == "warning"
    );
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ContextWindow(window)) => window.compaction.as_ref(),
            _ => None,
        }),
        Some(_)
    );
    assert_final_answer(&entries, "continued");
    assert_eq!(requests.seen().len(), 2);
    assert_matches!(requests.seen()[0].max_tokens, Some(tokens) if tokens < 80);

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn service_compacts_oversized_history_before_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(completion("summary")),
        ModelResponse::Completion(completion("done")),
    ]);
    model.settings.context_window = Some(4_000);
    let mut requests = model.requests();
    let config = ReactAgentConfig {
        compaction_threshold: Some(1_000),
        max_tokens: 1_000,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;

    service.open().await.expect("open service");
    run_turn(&service, &"history ".repeat(1_000), "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ContextWindow(window)) => window.compaction.as_ref(),
            _ => None,
        }),
        Some(compaction) if compaction.summary.as_deref().unwrap_or_default().contains("summary")
    );
    assert_final_answer(&entries, "done");
    assert_eq!(requests.seen().len(), 2);
    assert_eq!(requests.seen()[0].max_tokens, None);
    assert_matches!(requests.seen()[1].max_tokens, Some(_));

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn service_persists_failed_tool_and_continues() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "fail")),
        ModelResponse::Completion(completion("handled")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(ErrorTool)],
        ReactAgentConfig::default(),
    )
    .await;

    service.open().await.expect("open service");
    run_turn(&service, "use the tool", "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::Tool(tool)),
                success,
                ..
            })) => Some((tool, success)),
            _ => None,
        }),
        Some((ToolMessage { tool_call_id, content, .. }, false))
            if tool_call_id == "call-1" && content == "Tool error: injected tool failure"
    );
    assert_final_answer(&entries, "handled");

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn service_heals_tool_call_interrupted_by_process_exit() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let model = ScriptedModel::new([ModelResponse::Completion(tool_completion(
        "call-1", "block",
    ))]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(BlockingTool { started })],
        ReactAgentConfig::default(),
    )
    .await;

    service.open().await.expect("open service");
    service
        .execute(user_message(
            "block".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("execute turn");
    tokio::time::timeout(Duration::from_secs(5), started_receiver.recv())
        .await
        .expect("tool start timed out")
        .expect("tool start");
    runtime.abort();
    assert!(runtime.await.expect_err("runtime abort").is_cancelled());
    drop(service);

    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("continued"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let opened = service.open().await.expect("heal service");

    assert!(requests.seen().is_empty());
    assert_matches!(
        opened.last().and_then(AgentMessage::chat_message),
        Some(chat_record::Message::Tool(ToolMessage {
            tool_call_id,
            content,
            ..
        })) if tool_call_id == "call-1" && content == "Tool error: interrupted"
    );
    run_turn(&service, "continue", "user-2").await;
    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "continued");
    assert_eq!(requests.seen().len(), 1);

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn resume_turn_answers_a_turn_cut_off_by_process_exit() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let model = ScriptedModel::new([ModelResponse::Completion(tool_completion(
        "call-1", "block",
    ))]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(BlockingTool { started })],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    service
        .execute(user_message(
            "block".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("execute turn");
    tokio::time::timeout(Duration::from_secs(5), started_receiver.recv())
        .await
        .expect("tool start timed out")
        .expect("tool start");
    runtime.abort();
    assert!(runtime.await.expect_err("runtime abort").is_cancelled());
    drop(service);

    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("continued"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let mut events = service.events();
    assert!(service.resume_turn().await.expect("resume the turn"));
    wait_for_turn(&mut events).await;

    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "continued");
    assert_eq!(requests.seen().len(), 1);
    assert_eq!(
        entries
            .iter()
            .filter(|entry| matches!(entry.chat_message(), Some(chat_record::Message::User(_))))
            .count(),
        1,
        "the resumed turn adds no input"
    );
    assert!(!service
        .resume_turn()
        .await
        .expect("ask again after the answer"));
    assert_eq!(requests.seen().len(), 1);

    close_service(&service, runtime).await;
}

#[tokio::test]
async fn resume_turn_continues_after_a_truncated_answer() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut truncated = assistant_tool_entry(Vec::new());
    if let Some(agent_message::Payload::Chat(chat)) = &mut truncated.payload {
        chat.truncated = true;
        if let Some(chat_record::Message::Assistant(assistant)) = &mut chat.message {
            assistant.content = Some("partial".to_string());
        }
    }
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![user_entry("write it", "user-1"), truncated]),
    )
    .await;
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("finished"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let mut events = service.events();

    assert!(service.resume_turn().await.expect("resume the turn"));
    wait_for_turn(&mut events).await;
    assert_final_answer(
        &service.messages().await.expect("service messages"),
        "finished",
    );
    assert_eq!(requests.seen().len(), 1);

    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_surfaces_stale_sqlite_commit_and_replacement_recovers() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (completion_sender, completion_receiver) = oneshot::channel();
    let model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion: completion_receiver,
    }]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;

    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_message(
            "first owner".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("execute turn");
    tokio::time::timeout(Duration::from_secs(5), started_receiver.recv())
        .await
        .expect("completion start timed out")
        .expect("completion start");

    let competing = rusqlite::Connection::open(&database).expect("competing connection");
    let head: i64 = competing
        .query_row(
            "SELECT MAX(sequence) FROM react_entry WHERE session_id = 'session'",
            [],
            |row| row.get(0),
        )
        .expect("load head");
    competing
        .execute(
            "INSERT INTO react_entry(session_id, sequence, entry_id, payload) \
             VALUES ('session', ?, 'competing', ?)",
            rusqlite::params![
                head + 1,
                prost::Message::encode_to_vec(&AgentMessage {
                    id: "competing".to_string(),
                    ..log_entry("competing owner committed")
                }),
            ],
        )
        .expect("advance state");
    completion_sender
        .send(completion("stale result"))
        .expect("release completion");

    let turn_events = wait_for_turn(&mut events).await;
    assert_matches!(
        turn_events.iter().find(|event| matches!(event, ReactServiceEvent::TurnError(_))),
        Some(ReactServiceEvent::TurnError(error))
            if error.contains("conversation state changed")
    );
    runtime.await.expect("stale owner closes");
    assert!(service.is_closed());
    let stale_entries = load_entries(&database, directory.path()).await;
    assert!(!stale_entries.iter().any(|entry| {
        matches!(
            entry.chat_message(),
            Some(chat_record::Message::Assistant(AssistantMessage {
                content: Some(content),
                ..
            })) if content == "stale result"
        )
    }));
    drop(competing);

    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("recovered"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let opened = service.open().await.expect("open replacement");
    assert!(requests.seen().is_empty());
    assert_matches!(
        opened.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Log(log)) => Some(log),
            _ => None,
        }),
        Some(LogRecord { text, .. }) if text == "competing owner committed"
    );

    run_turn(&service, "second owner", "user-2").await;
    let entries = service.messages().await.expect("replacement messages");
    assert_final_answer(&entries, "recovered");
    assert_eq!(requests.seen().len(), 1);

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn open_heals_only_unanswered_calls_after_failed_tool_result() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            assistant_tool_entry(vec![tool_call("call-1", "one"), tool_call("call-2", "two")]),
            tool_entry("call-1", "Tool error: failed", false),
        ]),
    )
    .await;
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("unused"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;

    let opened = service.open().await.expect("heal service");
    let tool_results = opened
        .iter()
        .filter_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::Tool(tool)),
                success,
                ..
            })) => Some((tool, success)),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(requests.seen().is_empty());
    assert_eq!(tool_results.len(), 2);
    assert_matches!(
        tool_results[0],
        (ToolMessage { tool_call_id, content, .. }, false)
            if tool_call_id == "call-1" && content == "Tool error: failed"
    );
    assert_matches!(
        tool_results[1],
        (ToolMessage { tool_call_id, content, .. }, false)
            if tool_call_id == "call-2" && content == "Tool error: interrupted"
    );

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, opened);
}

#[tokio::test]
async fn open_preserves_completed_failed_tool_call() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let seeded = initialized_trace(vec![
        assistant_tool_entry(vec![tool_call("call-1", "fail")]),
        tool_entry("call-1", "Tool error: expected", false),
    ]);
    seed_entries(&database, directory.path(), seeded.clone()).await;
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("unused"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;

    let opened = service.open().await.expect("open service");

    assert!(requests.seen().is_empty());
    assert_eq!(opened.len(), seeded.len());
    assert_eq!(
        opened
            .iter()
            .filter(|entry| matches!(entry.chat_message(), Some(chat_record::Message::Tool(_))))
            .count(),
        1
    );

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, opened);
}

#[tokio::test]
async fn service_creation_rejects_tool_repair_before_a_later_user_message() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            assistant_tool_entry(vec![tool_call("call-1", "write")]),
            user_entry("later", "user-2"),
        ]),
    )
    .await;
    let saved = load_entries(&database, directory.path()).await;

    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error) if error.contains("historical tool repair")
    );
    assert_eq!(load_entries(&database, directory.path()).await, saved);
}

#[tokio::test]
async fn service_creation_rejects_mismatched_tool_result() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let seeded = initialized_trace(vec![
        assistant_tool_entry(vec![tool_call("call-1", "write")]),
        tool_entry("other-call", "wrong result", true),
    ]);
    seed_entries(&database, directory.path(), seeded.clone()).await;
    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error) if error.contains("has no unanswered call")
    );
    assert_eq!(
        load_entries(&database, directory.path()).await.len(),
        seeded.len()
    );
}

#[tokio::test]
async fn open_does_not_continue_persisted_user_turn() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let seeded = initialized_trace(vec![user_entry("unfinished", "user-1")]);
    seed_entries(&database, directory.path(), seeded.clone()).await;
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("new turn"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;

    let opened = service.open().await.expect("open service");
    assert_eq!(opened.len(), seeded.len());
    assert!(requests.seen().is_empty());

    run_turn(&service, "continue", "user-2").await;
    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "new turn");
    assert_eq!(requests.seen().len(), 1);

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn service_deduplicates_inputs_across_concurrent_callers() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (completion_sender, completion_receiver) = oneshot::channel();
    let mut model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion: completion_receiver,
    }]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("first input");
    started_receiver.recv().await.expect("model starts");
    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("idempotent retry");
    service
        .execute(user_message(
            "different".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("queue conflicting input");
    loop {
        if let ReactServiceEvent::TurnError(error) = events.recv().await.expect("conflict event") {
            assert!(error.contains("message ID \"user-1\" was reused"));
            break;
        }
    }
    completion_sender
        .send(completion("done"))
        .expect("release completion");
    wait_for_turn(&mut events).await;

    let entries = service.messages().await.expect("service messages");
    assert_eq!(
        entries.iter().filter(|entry| entry.id == "user-1").count(),
        1
    );
    assert_eq!(requests.seen().len(), 1);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_closes_after_input_persistence_failure() {
    let mut model = ScriptedModel::new(Vec::new());
    let mut requests = model.requests();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(FailingStore::user_append()),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    let runtime = tokio::spawn(runtime);
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(user_message(
            "hello".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("accept first attempt");
    assert_matches!(
        wait_for_turn(&mut events).await.as_slice(),
        [ReactServiceEvent::TurnError(error), ReactServiceEvent::TurnCompleted]
            if error.contains("injected user append failure")
    );
    assert_eq!(
        wait_for_close(&mut events).await,
        [
            ReactServiceEvent::InputRejected {
                message_id: "user-1".to_string(),
            },
            ReactServiceEvent::Closed,
        ]
    );

    runtime.await.expect("failed service closes");
    assert!(service.is_closed());
    assert_matches!(
        service.execute(user_message(
            "hello".to_string(),
            "user-1".to_string(),
            None,
        )),
        Err(ReactServiceError::Closed)
    );
    assert!(requests.seen().is_empty());
}

#[tokio::test]
async fn interrupt_reports_whether_a_turn_was_running() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (_completion_sender, completion_receiver) = oneshot::channel();
    let model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion: completion_receiver,
    }]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");

    assert!(!service
        .interrupt()
        .expect("idle interrupt")
        .await
        .expect("idle reply"));

    let mut events = service.events();
    service
        .execute(user_message("work".to_string(), "user-1".to_string(), None))
        .expect("start turn");
    started_receiver.recv().await.expect("model starts");
    assert!(service
        .interrupt()
        .expect("turn interrupt")
        .await
        .expect("turn reply"));
    let turn = wait_for_turn(&mut events).await;
    assert!(turn.contains(&ReactServiceEvent::TurnStarted));
    assert!(turn.contains(&ReactServiceEvent::TurnInterrupted));
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn input_without_model_content_starts_no_turn() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([ModelResponse::Completion(completion("done"))]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(AgentMessage {
            id: "extension-1".to_string(),
            payload: Some(agent_message::Payload::Extension(ExtensionRecord {
                extension: "test".to_string(),
                kind: "state".to_string(),
                data: None,
                model_message: None,
                survives_compaction: false,
            })),
            ..AgentMessage::default()
        })
        .expect("extension state");
    run_turn(&service, "work", "user-1").await;

    let turn = wait_for_turn(&mut events).await;
    assert_eq!(
        turn.iter()
            .filter(|event| matches!(event, ReactServiceEvent::TurnStarted))
            .count(),
        1
    );
    assert_eq!(
        turn.iter()
            .filter(|event| matches!(event, ReactServiceEvent::TurnCompleted))
            .count(),
        1
    );
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn close_is_idempotent_across_service_clones() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new(Vec::new());
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let clone = service.clone();

    let (first, second) = tokio::join!(service.close(), clone.close());

    assert_matches!(first, Ok(()));
    assert_matches!(second, Ok(()));
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn interrupt_drops_input_accepted_for_the_next_turn() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (_completion_sender, completion_receiver) = oneshot::channel();
    let mut model = ScriptedModel::new([
        ModelResponse::Delayed {
            started,
            completion: completion_receiver,
        },
        ModelResponse::Completion(completion("second completed")),
    ]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("first input");
    started_receiver.recv().await.expect("model starts");
    service
        .execute(user_message(
            "second".to_string(),
            "user-2".to_string(),
            None,
        ))
        .expect("queued input");
    service.interrupt().expect("interrupt");
    let interrupted = wait_for_turn(&mut events).await;
    assert_matches!(
        interrupted
            .iter()
            .find(|event| matches!(event, ReactServiceEvent::TurnInterrupted)),
        Some(_)
    );

    let entries = service.messages().await.expect("service messages");
    assert_eq!(
        entries
            .iter()
            .filter_map(|entry| match entry.chat_message() {
                Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["first"]
    );
    assert_eq!(requests.seen().len(), 1);

    run_turn(&service, "second", "user-2").await;
    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "second completed");
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn interrupt_persists_queued_extension_messages() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (_completion_sender, completion_receiver) = oneshot::channel();
    let mut model = ScriptedModel::new([
        ModelResponse::Delayed {
            started,
            completion: completion_receiver,
        },
        ModelResponse::Completion(completion("next completed")),
    ]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("first input");
    started_receiver.recv().await.expect("model starts");
    service
        .execute(AgentMessage {
            id: "extension-1".to_string(),
            payload: Some(agent_message::Payload::Extension(ExtensionRecord {
                extension: "test".to_string(),
                kind: "message".to_string(),
                data: None,
                model_message: Some(UserMessage {
                    checkpoint: None,
                    content: "extension update".to_string(),
                }),
                survives_compaction: false,
            })),
            ..AgentMessage::default()
        })
        .expect("queued extension message");
    service.interrupt().expect("interrupt");
    wait_for_turn(&mut events).await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find(|entry| entry.id == "extension-1"),
        Some(AgentMessage {
            payload: Some(agent_message::Payload::Extension(_)),
            ..
        })
    );
    assert_eq!(requests.seen().len(), 1);

    run_turn(&service, "next", "user-2").await;
    assert_matches!(
        requests.seen()[1].messages.iter().find(|message| {
            matches!(
                message,
                chat_record::Message::User(UserMessage { content, .. })
                    if content == "extension update"
            )
        }),
        Some(_)
    );
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn interrupt_drops_user_intent_but_persists_extension_and_task_in_order() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (task_started, mut task_started_receiver) = mpsc::unbounded_channel();
    let (model_started, mut model_started_receiver) = mpsc::unbounded_channel();
    let (_model_completion, model_completion_receiver) = oneshot::channel();
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "completing_detach")),
        ModelResponse::Completion(completion("task started")),
        ModelResponse::Delayed {
            started: model_started,
            completion: model_completion_receiver,
        },
    ]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(CompletingDetachedTool {
            started: task_started,
        })],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    run_turn(&service, "start task", "user-1").await;
    let task_completion = task_started_receiver.recv().await.expect("task starts");
    let mut events = service.events();

    service
        .execute(user_message("wait".to_string(), "user-2".to_string(), None))
        .expect("active input");
    model_started_receiver.recv().await.expect("model starts");
    service
        .execute(user_message(
            "drop me".to_string(),
            "user-3".to_string(),
            None,
        ))
        .expect("queued input");
    service
        .execute(AgentMessage {
            id: "extension-1".to_string(),
            payload: Some(agent_message::Payload::Extension(ExtensionRecord {
                extension: "test".to_string(),
                kind: "state".to_string(),
                model_message: Some(UserMessage {
                    checkpoint: None,
                    content: "retain this context".to_string(),
                }),
                ..ExtensionRecord::default()
            })),
            ..AgentMessage::default()
        })
        .expect("queued extension");
    assert_matches!(
        service.messages().await,
        Err(ReactServiceError::ConversationBusy)
    );
    task_completion
        .send(ToolOutput::success("background result"))
        .expect("complete task");
    loop {
        if matches!(
            events.recv().await.expect("service event"),
            ReactServiceEvent::TaskUpdated(crate::TaskInfo {
                status: crate::RuntimeTaskStatus::Succeeded,
                ..
            })
        ) {
            break;
        }
    }
    service.interrupt().expect("interrupt");
    let interrupted = wait_for_turn(&mut events).await;

    let entries = service.messages().await.expect("service messages");
    assert_eq!(
        entries
            .iter()
            .filter_map(|entry| match entry.chat_message() {
                Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["start task", "wait"]
    );
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::TaskResult(result)) => Some(result),
            _ => None,
        }),
        Some(TaskResultRecord { content, success: true, .. })
            if content == "background result"
    );
    let retained = &entries[entries.len() - 2..];
    assert_matches!(
        retained,
        [
            AgentMessage {
                payload: Some(agent_message::Payload::Extension(_)),
                ..
            },
            AgentMessage {
                payload: Some(agent_message::Payload::TaskResult(_)),
                ..
            },
        ]
    );
    let terminal = interrupted
        .iter()
        .filter(|event| {
            matches!(
                event,
                ReactServiceEvent::Agent(AgentEvent::Entries(_))
                    | ReactServiceEvent::TurnInterrupted
                    | ReactServiceEvent::TurnCompleted
            )
        })
        .collect::<Vec<_>>();
    assert_matches!(terminal.as_slice(), [
        ReactServiceEvent::Agent(AgentEvent::Entries(saved)),
        ReactServiceEvent::TurnInterrupted,
        ReactServiceEvent::TurnCompleted,
    ] if saved.as_slice() == retained);
    assert_eq!(requests.seen().len(), 3);

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_lifecycle_holds_across_worker_threads() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (task_started, mut task_started_receiver) = mpsc::unbounded_channel();
    let (model_started, mut model_started_receiver) = mpsc::unbounded_channel();
    let (model_completion, model_completion_receiver) = oneshot::channel();
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "completing_detach")),
        ModelResponse::Completion(completion("task started")),
        ModelResponse::Delayed {
            started: model_started,
            completion: model_completion_receiver,
        },
        ModelResponse::Completion(completion("done")),
        ModelResponse::Completion(completion("done")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(CompletingDetachedTool {
            started: task_started,
        })],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    run_turn(&service, "start task", "user-1").await;
    let task_completion = task_started_receiver.recv().await.expect("task starts");
    let mut events = service.events();

    service
        .execute(user_message("wait".to_string(), "user-2".to_string(), None))
        .expect("second turn");
    model_started_receiver.recv().await.expect("model starts");
    let completer = tokio::spawn(async move {
        task_completion
            .send(ToolOutput::success("background result"))
            .expect("complete task");
    });
    let queued_service = service.clone();
    let queuer = tokio::spawn(async move {
        queued_service
            .execute(user_message(
                "also queued".to_string(),
                "user-3".to_string(),
                None,
            ))
            .expect("queued input");
    });
    loop {
        if matches!(
            events.recv().await.expect("service event"),
            ReactServiceEvent::TaskUpdated(crate::TaskInfo {
                status: crate::RuntimeTaskStatus::Succeeded,
                ..
            })
        ) {
            break;
        }
    }
    completer.await.expect("completer task");
    queuer.await.expect("queuer task");
    model_completion
        .send(completion("waited"))
        .expect("release model");
    let mut turn_events = Vec::new();
    let entries = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            turn_events.extend(wait_for_turn(&mut events).await);
            let Ok(entries) = service.messages().await else {
                continue;
            };
            let has_queued_input = entries.iter().any(|entry| entry.id == "user-3");
            let has_task_result = entries
                .iter()
                .any(|entry| matches!(&entry.payload, Some(agent_message::Payload::TaskResult(_))));
            if has_queued_input && has_task_result {
                return entries;
            }
        }
    })
    .await
    .expect("both concurrent inputs finish");
    while let Ok(event) = events.try_recv() {
        turn_events.push(event);
    }
    assert_matches!(
        turn_events
            .iter()
            .find(|event| matches!(event, ReactServiceEvent::TurnError(_))),
        None
    );
    assert_eq!(
        entries
            .iter()
            .filter_map(|entry| match entry.chat_message() {
                Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["start task", "wait", "also queued"]
    );
    assert_final_answer(&entries, "done");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::TaskResult(result)) => Some(result),
            _ => None,
        }),
        Some(TaskResultRecord { content, success: true, .. })
            if content == "background result"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| matches!(&entry.payload, Some(agent_message::Payload::TaskResult(_))))
            .count(),
        1
    );

    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn close_persists_input_already_accepted_for_the_next_turn() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (_completion_sender, completion_receiver) = oneshot::channel();
    let model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion: completion_receiver,
    }]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");

    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("first input");
    started_receiver.recv().await.expect("model starts");
    service
        .execute(user_message(
            "second".to_string(),
            "user-2".to_string(),
            None,
        ))
        .expect("queued input");
    service.close().await.expect("close service");
    runtime.await.expect("service runtime");

    let entries = load_entries(&database, directory.path()).await;
    assert_eq!(
        entries
            .iter()
            .filter_map(|entry| match entry.chat_message() {
                Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
}

#[tokio::test]
async fn abandon_drops_input_accepted_for_the_next_turn_so_a_retry_executes() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (_completion_sender, completion_receiver) = oneshot::channel();
    let model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion: completion_receiver,
    }]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");

    service
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("first input");
    started_receiver.recv().await.expect("model starts");
    service
        .execute(user_message(
            "second".to_string(),
            "user-2".to_string(),
            None,
        ))
        .expect("queued input");
    let mut events = service.events();
    service.abandon().await.expect("abandon service");
    runtime.await.expect("service runtime");
    let mut rejected = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let ReactServiceEvent::InputRejected { message_id } = event {
            rejected.push(message_id);
        }
    }
    assert_eq!(rejected, vec!["user-2"]);
    assert_eq!(
        user_contents(&load_entries(&database, directory.path()).await),
        vec!["first"]
    );

    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("done"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("reopen service");
    run_turn(&service, "second", "user-2").await;
    let entries = service.messages().await.expect("service messages");
    assert_eq!(user_contents(&entries), vec!["first", "second"]);
    assert_final_answer(&entries, "done");
    assert_eq!(requests.seen().len(), 1);
    close_service(&service, runtime).await;
}

fn user_contents(entries: &[AgentMessage]) -> Vec<&str> {
    entries
        .iter()
        .filter_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn close_if_idle_yields_to_a_queued_input_and_closes_once_idle() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (completion_sender, completion_receiver) = oneshot::channel();
    let model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion: completion_receiver,
    }]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(user_message(
            "hello".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("queue input");
    assert!(!service.close_if_idle().await.expect("conditional close"));
    started_receiver.recv().await.expect("the queued turn runs");
    completion_sender
        .send(completion("done"))
        .expect("complete the turn");
    wait_for_turn(&mut events).await;
    assert_final_answer(&service.messages().await.expect("messages"), "done");

    assert!(service.close_if_idle().await.expect("conditional close"));
    runtime.await.expect("service runtime");
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match events.recv().await {
                Ok(ReactServiceEvent::Closed) => return,
                Ok(_) => {}
                Err(error) => {
                    panic!("event stream failed before the closed event: {error}")
                }
            }
        }
    })
    .await
    .expect("closed event reaches attached streams");
}

#[tokio::test]
async fn close_if_idle_rejects_input_queued_after_the_close_request() {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        ScriptedModel::new([]),
        test_provider(DEFAULT_CONTEXT_WINDOW).await,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(InMemoryStore::default()),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    let mut events = service.events();
    let close = service.close_if_idle();
    tokio::pin!(close);
    assert_matches!(futures_util::poll!(&mut close), std::task::Poll::Pending);
    service
        .execute(user_message(
            "queued".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("input queues behind the close request");

    let runtime = tokio::spawn(runtime);
    assert!(close.await.expect("close idle service"));
    runtime.await.expect("service runtime");
    let mut rejected = Vec::new();
    loop {
        match events.try_recv().expect("closed event is published") {
            ReactServiceEvent::InputRejected { message_id } => rejected.push(message_id),
            ReactServiceEvent::Closed => break,
            _ => {}
        }
    }
    assert_eq!(rejected, vec!["user-1"]);
    assert_matches!(
        service.execute(user_message(
            "too late".to_string(),
            "user-2".to_string(),
            None,
        )),
        Err(ReactServiceError::Closed)
    );
}

#[tokio::test]
async fn close_if_idle_keeps_detached_work_until_its_result_is_processed() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "completing_detach")),
        ModelResponse::Completion(completion("task started")),
        ModelResponse::Completion(completion("task finished")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(CompletingDetachedTool { started })],
        ReactAgentConfig::default(),
    )
    .await;
    run_turn(&service, "start task", "user-1").await;
    let complete = started_receiver.recv().await.expect("detached task starts");
    assert!(!service.is_busy(), "the main turn has finished");
    assert!(!service.close_if_idle().await.expect("check idle"));

    let mut events = service.events();
    complete
        .send(ToolOutput::success("background result"))
        .expect("complete task");
    wait_for_turn(&mut events).await;
    let entries = service.messages().await.expect("messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::TaskResult(result)) => Some(result),
            _ => None,
        }),
        Some(TaskResultRecord { content, success: true, .. }) if content == "background result"
    );
    assert_final_answer(&entries, "task finished");
    assert!(service
        .close_if_idle()
        .await
        .expect("close after task result"));
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn abandon_rejects_only_client_inputs_not_skill_context_or_task_results() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let store = SQLiteStore::open(&database, "session", directory.path(), true, false)
        .expect("open SQLite store");
    let (task_started, mut task_started_receiver) = mpsc::unbounded_channel();
    let (model_started, mut model_started_receiver) = mpsc::unbounded_channel();
    let (_complete, completion_receiver) = oneshot::channel();
    let tool_executor = DirectToolExecutor::new(vec![Arc::new(DetachedTool {
        started: task_started,
    })]);
    let agent = ReactAgent::new(
        ScriptedModel::new([
            ModelResponse::Completion(tool_completion("call-1", "detach")),
            ModelResponse::Completion(completion("task started")),
            ModelResponse::Delayed {
                started: model_started,
                completion: completion_receiver,
            },
        ]),
        test_provider(DEFAULT_CONTEXT_WINDOW).await,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        "session",
        AgentApprovalMode::Auto,
        vec![Skill {
            name: "review".to_string(),
            description: "Review code.".to_string(),
            body: "Follow the checklist.".to_string(),
            path: std::path::PathBuf::from("skills/review/SKILL.md"),
            disable_model_invocation: false,
        }],
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    let runtime = tokio::spawn(runtime);
    run_turn(&service, "start task", "user-1").await;
    task_started_receiver.recv().await.expect("task starts");
    service
        .execute(user_message("wait".to_string(), "user-2".to_string(), None))
        .expect("start turn");
    model_started_receiver.recv().await.expect("turn starts");
    let mut events = service.events();
    service
        .execute(user_message(
            "$review".to_string(),
            "user-3".to_string(),
            None,
        ))
        .expect("queue skill input");
    service.abandon().await.expect("abandon service");
    runtime.await.expect("service runtime");
    let mut rejected = Vec::new();
    let mut task_cancelled = false;
    while let Ok(event) = events.try_recv() {
        match event {
            ReactServiceEvent::InputRejected { message_id } => rejected.push(message_id),
            ReactServiceEvent::TaskUpdated(info) => {
                task_cancelled |= info.status == crate::RuntimeTaskStatus::Cancelled;
            }
            _ => {}
        }
    }
    assert!(task_cancelled, "closing generated a detached task result");
    assert_eq!(rejected, vec!["user-3"]);
}

#[tokio::test]
async fn dropping_service_cancels_detached_tasks_and_stops_runtime() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "detach")),
        ModelResponse::Completion(completion("done")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(DetachedTool { started })],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    run_turn(&service, "detach", "user-1").await;
    started_receiver.recv().await.expect("task starts");
    assert_matches!(
        service.compact().await,
        Err(ReactServiceError::ConversationBusy)
    );

    drop(service);
    tokio::time::timeout(Duration::from_secs(5), runtime)
        .await
        .expect("runtime stop timed out")
        .expect("service runtime");

    let entries = load_entries(&database, directory.path()).await;
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::TaskResult(result)) => Some(result),
            _ => None,
        }),
        Some(result)
            if result.task_id == "1000" && !result.success
                && result.content == "Tool error: task cancelled"
    );
}

#[tokio::test]
async fn failed_approval_commit_is_not_acknowledged_or_executed() {
    let model = ScriptedModel::new([ModelResponse::Completion(tool_completion(
        "call-1",
        "permission",
    ))]);
    let tools = DirectToolExecutor::new(vec![Arc::new(PermissionTool)]);
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tools,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tools),
        Box::new(FailingStore {
            inner: InMemoryStore::default(),
            fail_user_append: false,
            fail_approval_append: true,
        }),
        "session",
        AgentApprovalMode::Hitl,
        Vec::new(),
        Vec::new(),
        None,
        broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    service.open().await.expect("open");
    let mut events = service.events();
    service
        .execute(user_entry("check", "user-1"))
        .expect("execute");
    assert_eq!(next_approval(&mut events).await.id, "call-1");
    assert!(!service
        .respond_to_hitl("call-1".to_string(), true, ApprovalSource::Manual)
        .await
        .expect("rejected acknowledgment"));
    let remaining = wait_for_turn(&mut events).await;
    assert!(remaining.iter().any(|event| matches!(event, ReactServiceEvent::TurnError(error) if error.contains("injected approval append failure"))));
    assert!(!remaining.iter().any(|event| matches!(
        event,
        ReactServiceEvent::HitlResolved { approved: true, .. }
    )));
    assert!(!remaining.iter().any(|event| matches!(event, ReactServiceEvent::Agent(AgentEvent::Entries(entries)) if entries.iter().any(|entry| matches!(entry.chat_message(), Some(chat_record::Message::Tool(_)))))));
    runtime.await.expect("failed service closes");
}

#[tokio::test]
async fn acknowledged_approval_survives_interrupt_and_continue() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut action = tool_completion("call-1", "permission");
    action
        .message
        .tool_calls
        .push(tool_call("call-2", "permission"));
    let model = ScriptedModel::new([ModelResponse::Completion(action)]);
    let (service, runtime) = spawn_service_with_mode(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(PermissionTool)],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_entry("check", "user-1"))
        .expect("execute");
    assert_eq!(next_approval(&mut events).await.id, "call-1");
    assert!(service
        .respond_to_hitl("call-1".to_string(), true, ApprovalSource::RememberedPrefix)
        .await
        .expect("durable approval"));
    assert_eq!(next_approval(&mut events).await.id, "call-2");
    assert!(service
        .interrupt()
        .expect("interrupt")
        .await
        .expect("interrupt response"));
    wait_for_turn(&mut events).await;
    assert!(!service
        .respond_to_hitl("call-2".to_string(), true, ApprovalSource::Manual)
        .await
        .expect("late approval"));
    close_service(&service, runtime).await;
    assert_eq!(
        approval_outcomes(&load_entries(&database, directory.path()).await),
        [
            ("call-1", ToolApprovalOutcome::Requested),
            ("call-1", ToolApprovalOutcome::Approved),
            ("call-2", ToolApprovalOutcome::Requested),
            ("call-2", ToolApprovalOutcome::Canceled),
        ]
    );
    assert!(load_entries(&database, directory.path()).await.iter().any(|entry| matches!(
        &entry.payload, Some(agent_message::Payload::ToolApproval(record)) if record.source() == ApprovalSource::RememberedPrefix
    )));
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("continued"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        vec![Box::new(PermissionTool)],
        ReactAgentConfig::default(),
    )
    .await;
    let mut events = service.events();
    assert!(service.resume_turn().await.expect("continue"));
    wait_for_turn(&mut events).await;
    let request = requests.seen().last().expect("resumed model request");
    let repaired = request
        .messages
        .iter()
        .filter_map(|message| match message {
            chat_record::Message::Tool(tool) => {
                Some((tool.tool_call_id.as_str(), tool.content.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        repaired,
        [
            ("call-1", "Tool error: interrupted"),
            ("call-2", "Tool error: interrupted")
        ]
    );
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn approval_trace_recovers_pending_but_preserves_consumed_permission() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let approved = tool_call("approved-call", "permission");
    let denied = tool_call("denied-call", "permission");
    let pending = tool_call("pending-call", "permission");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            assistant_tool_entry(vec![approved.clone(), denied.clone(), pending.clone()]),
            AgentMessage::tool_approval(
                &approved,
                ToolApprovalOutcome::Requested,
                ApprovalSource::Unspecified,
            ),
            AgentMessage::tool_approval(
                &approved,
                ToolApprovalOutcome::Approved,
                ApprovalSource::Unspecified,
            ),
            AgentMessage::tool_approval(
                &denied,
                ToolApprovalOutcome::Requested,
                ApprovalSource::Unspecified,
            ),
            AgentMessage::tool_approval(
                &denied,
                ToolApprovalOutcome::Denied,
                ApprovalSource::Unspecified,
            ),
            AgentMessage::tool_approval(
                &pending,
                ToolApprovalOutcome::Requested,
                ApprovalSource::Unspecified,
            ),
        ]),
    )
    .await;
    let mut store =
        SQLiteStore::open(&database, "session", directory.path(), false, false).expect("reopen");
    let entries = store.load("session", false).await.expect("load");
    let opened = repair_saved_history(&mut store, "session", entries)
        .await
        .expect("recover");
    assert_eq!(
        approval_outcomes(&opened),
        [
            ("approved-call", ToolApprovalOutcome::Requested),
            ("approved-call", ToolApprovalOutcome::Approved),
            ("denied-call", ToolApprovalOutcome::Requested),
            ("denied-call", ToolApprovalOutcome::Denied),
            ("pending-call", ToolApprovalOutcome::Requested),
            ("pending-call", ToolApprovalOutcome::Canceled),
        ]
    );
    assert_eq!(
        repair_saved_history(&mut store, "session", opened.clone())
            .await
            .expect("recover again"),
        opened
    );
    drop(store);
    assert_eq!(load_entries(&database, directory.path()).await, opened);
}

#[tokio::test]
async fn approval_trace_correlates_approved_and_denied_actions_after_reopen() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut approved = tool_call("approved-call", "permission");
    approved.function.as_mut().expect("function").arguments =
        r#"{"token":"synthetic-secret-approval-trace"}"#.to_string();
    let model = ScriptedModel::new([
        ModelResponse::Completion(Completion {
            message: AssistantMessage {
                tool_calls: vec![approved, tool_call("denied-call", "permission")],
                ..AssistantMessage::default()
            },
            usage: None,
            truncated: false,
        }),
        ModelResponse::Completion(completion("done")),
    ]);
    let (service, runtime) = spawn_service_with_mode(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(PermissionTool)],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_entry("check", "user-1"))
        .expect("execute");
    assert_eq!(next_approval(&mut events).await.id, "approved-call");
    assert!(!service
        .respond_to_hitl("unknown".to_string(), true, ApprovalSource::Manual)
        .await
        .expect("unknown decision"));
    assert!(service
        .respond_to_hitl(
            "approved-call".to_string(),
            true,
            ApprovalSource::RememberedExact
        )
        .await
        .expect("approve"));
    assert_eq!(next_approval(&mut events).await.id, "denied-call");
    assert!(!service
        .respond_to_hitl("approved-call".to_string(), false, ApprovalSource::Manual)
        .await
        .expect("duplicate decision"));
    assert!(service
        .respond_to_hitl("denied-call".to_string(), false, ApprovalSource::Manual)
        .await
        .expect("deny"));
    wait_for_turn(&mut events).await;
    close_service(&service, runtime).await;

    let entries = load_entries(&database, directory.path()).await;
    assert_eq!(
        approval_outcomes(&entries),
        [
            ("approved-call", ToolApprovalOutcome::Requested),
            ("approved-call", ToolApprovalOutcome::Approved),
            ("denied-call", ToolApprovalOutcome::Requested),
            ("denied-call", ToolApprovalOutcome::Denied),
        ]
    );
    let replies = entries
        .iter()
        .filter_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::Tool(tool)) => Some((tool.tool_call_id.as_str(), entry)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0].0, "approved-call");
    assert_eq!(replies[1].0, "denied-call");
    assert_matches!(&replies[0].1.payload, Some(agent_message::Payload::Chat(chat)) if chat.success);
    assert_matches!(&replies[1].1.payload, Some(agent_message::Payload::Chat(chat)) if !chat.success);
    let audit = entries
        .iter()
        .filter(|entry| matches!(entry.payload, Some(agent_message::Payload::ToolApproval(_))))
        .collect::<Vec<_>>();
    assert!(!format!("{audit:?}").contains("synthetic-secret-approval-trace"));
    assert_matches!(&audit[1].payload, Some(agent_message::Payload::ToolApproval(record)) if record.source() == ApprovalSource::RememberedExact);
    assert_matches!(&audit[3].payload, Some(agent_message::Payload::ToolApproval(record)) if record.source() == ApprovalSource::Manual);
    assert!(audit
        .iter()
        .all(|entry| entry.sequence.is_some() && entry.timestamp.is_some()));
    assert!(audit.iter().all(|entry| entry.to_llm_messages().is_empty()));
    assert!(audit.iter().all(|entry| matches!(&entry.payload, Some(agent_message::Payload::ToolApproval(record)) if record.tool_name == "permission")));
}

async fn next_approval(events: &mut broadcast::Receiver<ReactServiceEvent>) -> ToolCall {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let ReactServiceEvent::HitlRequest(call) =
                events.recv().await.expect("service event")
            {
                return call;
            }
        }
    })
    .await
    .expect("approval request timeout")
}

fn approval_outcomes(entries: &[AgentMessage]) -> Vec<(&str, ToolApprovalOutcome)> {
    entries
        .iter()
        .filter_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ToolApproval(record)) => {
                Some((record.call_id.as_str(), record.outcome()))
            }
            _ => None,
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct RecordedModes {
    pub(crate) saved: std::sync::Mutex<Vec<AgentApprovalMode>>,
    failing: std::sync::atomic::AtomicBool,
}

impl ApprovalModeStore for RecordedModes {
    fn save(&self, mode: AgentApprovalMode) -> BoxFuture<'_, Result<(), StoreError>> {
        Box::pin(async move {
            if self.failing.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(StoreError::Persistence(
                    "injected approval mode failure".to_string(),
                ));
            }
            self.saved.lock().expect("saved modes").push(mode);
            Ok(())
        })
    }
}

#[tokio::test]
async fn approval_mode_applies_only_after_it_is_saved() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "permission")),
        ModelResponse::Completion(completion("done")),
    ]);
    let modes = Arc::new(RecordedModes::default());
    let (service, runtime) = spawn_service_with_mode(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(PermissionTool)],
        ReactAgentConfig::default(),
        SessionApprovalMode::saved(AgentApprovalMode::Hitl, modes.clone()),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_entry("check", "user-1"))
        .expect("execute");
    assert_eq!(next_approval(&mut events).await.id, "call-1");

    modes
        .failing
        .store(true, std::sync::atomic::Ordering::SeqCst);
    assert_matches!(
        service.set_approval_mode(AgentApprovalMode::Auto).await,
        Err(ReactServiceError::Store(_))
    );
    assert!(
        service.is_busy(),
        "an unsaved mode must not approve the call"
    );
    modes
        .failing
        .store(false, std::sync::atomic::Ordering::SeqCst);
    service
        .set_approval_mode(AgentApprovalMode::Auto)
        .await
        .expect("saved mode");
    wait_for_turn(&mut events).await;
    close_service(&service, runtime).await;

    assert_eq!(
        *modes.saved.lock().expect("saved modes"),
        [AgentApprovalMode::Auto]
    );
    let entries = load_entries(&database, directory.path()).await;
    assert!(entries.iter().any(|entry| matches!(
        &entry.payload,
        Some(agent_message::Payload::ToolApproval(record))
            if record.outcome() == ToolApprovalOutcome::Approved
                && record.source() == ApprovalSource::Automatic
    )));
}

#[tokio::test]
async fn approval_mode_change_updates_active_tool_permissions() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "permission")),
        ModelResponse::Completion(tool_completion("ungated-call", "fail")),
        ModelResponse::Completion(completion("done")),
    ]);
    let (service, runtime) = spawn_service_with_mode(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(PermissionTool), Box::new(ErrorTool)],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_message(
            "check".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("execute turn");
    loop {
        if matches!(
            events.recv().await.expect("service event"),
            ReactServiceEvent::HitlRequest(_)
        ) {
            break;
        }
    }
    assert_matches!(
        service.compact().await,
        Err(ReactServiceError::ConversationBusy)
    );
    service
        .set_approval_mode(AgentApprovalMode::Auto)
        .await
        .expect("change approval mode");
    let completed = wait_for_turn(&mut events).await;
    assert!(!completed
        .iter()
        .any(|event| matches!(event, ReactServiceEvent::HitlRequest(_))));

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::Tool(tool)) => Some(tool),
            _ => None,
        }),
        Some(tool) if tool.content == "true"
    );
    close_service(&service, runtime).await;
    let entries = load_entries(&database, directory.path()).await;
    assert_eq!(
        approval_outcomes(&entries),
        [
            ("call-1", ToolApprovalOutcome::Requested),
            ("call-1", ToolApprovalOutcome::Approved),
            ("ungated-call", ToolApprovalOutcome::Approved),
        ]
    );
    let sources = entries
        .iter()
        .filter_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ToolApproval(record)) => Some(record.source()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sources,
        [
            ApprovalSource::Unspecified,
            ApprovalSource::Automatic,
            ApprovalSource::NotRequired
        ]
    );
}

#[tokio::test]
async fn direct_calls_execute_registered_host_tools() {
    let (requests, mut host_calls) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![Arc::new(HostRequestTool { requests })]);
    assert_eq!(executor.definitions()[0].name, "host");
    let execution = executor.call_tool(
        "host".to_string(),
        "{}".to_string(),
        ExecutionContext::new(
            AgentApprovalMode::Hitl,
            None,
            TaskManagerHandle::closed(),
            None,
            None,
        ),
    );
    drop(executor);
    let call = tokio::spawn(execution);
    let request = host_calls.recv().await.expect("host callback");
    assert_eq!(request.name, "host");
    assert_eq!(request.arguments, "{}");
    request
        .response
        .send(ToolOutput::success("done"))
        .expect("reply");
    assert_eq!(call.await.expect("caller").expect("result").content, "done");
}

#[tokio::test]
async fn direct_calls_reject_unregistered_tools() {
    let executor = DirectToolExecutor::new(Vec::new());
    let context = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        TaskManagerHandle::closed(),
        None,
        None,
    );
    let output = executor
        .call_tool("missing".to_string(), "{}".to_string(), context)
        .await
        .expect("tool result");
    assert!(!output.success);
    assert_eq!(output.content, "Unknown tool: missing");
}

#[tokio::test]
async fn direct_calls_fail_closed_without_an_approval_handler() {
    let workspace = tempfile::tempdir().expect("workspace");
    let executor = DirectToolExecutor::new(vec![Arc::new(
        crate::tools::WriteTool::new(workspace.path()).expect("write tool"),
    )]);
    let denied = executor
        .call_tool(
            "write".into(),
            r#"{"file_path":"note","content":"must not write"}"#.into(),
            ExecutionContext::new(
                AgentApprovalMode::Hitl,
                None,
                TaskManagerHandle::closed(),
                None,
                None,
            ),
        )
        .await
        .expect("approval unavailable");
    assert!(!denied.success);
    assert_eq!(denied, ToolOutput::error("Tool approval is unavailable."));
    assert!(!workspace.path().join("note").exists());
}

#[tokio::test]
async fn direct_calls_in_one_batch_run_in_order() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let executor = DirectToolExecutor::new(vec![Arc::new(
        crate::tools::BashTool::new(workspace.path()).expect("workspace is valid"),
    )]);
    let context = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        TaskManagerHandle::closed(),
        None,
        None,
    );
    let bash = |id: &str, command: &str| ToolCall {
        id: id.to_string(),
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: "bash".to_string(),
            arguments: json!({ "command": command }).to_string(),
        }),
        ..ToolCall::default()
    };
    let calls = [
        bash("write", "sleep 0.3; echo written > note"),
        bash("read", "cat note"),
    ];

    let outputs = executor
        .execute(Uuid::new_v4(), &calls, &[true, true], &context)
        .await
        .expect("batch runs");

    assert!(outputs[1].content.starts_with("written\n[Completed in "));
}

#[tokio::test]
async fn tool_execution_outlives_its_executor() {
    let execution = {
        let executor = DirectToolExecutor::new(vec![Arc::new(PermissionTool)]);
        let context = ExecutionContext::new(
            AgentApprovalMode::Auto,
            None,
            TaskManagerHandle::closed(),
            None,
            None,
        );
        executor.call_tool("permission".to_string(), "{}".to_string(), context)
    };
    let output = execution.await.expect("execute after dropping executor");
    assert!(output.success);
    assert_eq!(output.content, "true");
}

#[tokio::test]
async fn direct_tool_call_requests_approval_without_starting_a_turn() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (service, runtime) = spawn_service_with_mode(
        &database,
        directory.path(),
        true,
        ScriptedModel::new([]),
        vec![Box::new(PermissionTool)],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    let before = service.messages().await.expect("messages");
    let mut events = service.events();
    let caller = service.clone();
    let call = tokio::spawn(async move {
        caller
            .call_tool("permission".to_string(), "{}".to_string())
            .await
    });
    let approval = next_approval(&mut events).await;
    assert!(!service.is_busy());
    assert!(service
        .respond_to_hitl(approval.id, true, ApprovalSource::Manual)
        .await
        .expect("approve"));
    let output = call
        .await
        .expect("caller")
        .expect("tool output after approval");
    assert!(output.success);
    assert_eq!(
        output.content, "false",
        "workspace permissions are still enforced"
    );
    assert_eq!(service.messages().await.expect("messages"), before);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn interrupting_the_model_does_not_interrupt_a_direct_tool_call() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (started, mut model_started) = mpsc::unbounded_channel();
    let (_complete, completion) = oneshot::channel();
    let (requests, mut tool_requests) = mpsc::unbounded_channel();
    let (service, runtime) = spawn_service_with_mode(
        &directory.path().join("react.db"),
        directory.path(),
        true,
        ScriptedModel::new([ModelResponse::Delayed {
            started,
            completion,
        }]),
        vec![Box::new(HostRequestTool { requests })],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    let mut events = service.events();
    service
        .execute(user_message("work".to_string(), "user-1".to_string(), None))
        .expect("start model turn");
    model_started.recv().await.expect("model started");
    let caller = service.clone();
    let call =
        tokio::spawn(async move { caller.call_tool("host".to_string(), "{}".to_string()).await });
    let request = tool_requests.recv().await.expect("direct call started");
    service.interrupt().expect("interrupt model");
    wait_for_turn(&mut events).await;
    request
        .response
        .send(ToolOutput::success("done"))
        .expect("direct call is still running");
    let output = call.await.expect("caller").expect("tool result");
    assert!(output.success);
    assert_eq!(output.content, "done");
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn direct_tool_call_returns_task_started_and_preserves_task_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (started, mut completions) = mpsc::unbounded_channel();
    let (service, runtime) = spawn_service_with_mode(
        &directory.path().join("react.db"),
        directory.path(),
        true,
        ScriptedModel::new([ModelResponse::Completion(completion(
            "observed task result",
        ))]),
        vec![Box::new(CompletingDetachedTool { started })],
        ReactAgentConfig::default(),
        AgentApprovalMode::Auto,
    )
    .await;
    let before = service.messages().await.expect("messages");
    let mut events = service.events();
    let output = service
        .call_tool("completing_detach".to_string(), "{}".to_string())
        .await
        .expect("call");
    assert!(output.success);
    assert!(output.content.starts_with("Task ") && output.content.ends_with(" started."));
    assert_eq!(service.messages().await.expect("messages"), before);
    completions
        .recv()
        .await
        .expect("started")
        .send(ToolOutput::success("done"))
        .expect("complete");
    wait_for_turn(&mut events).await;
    let entries = service.messages().await.expect("messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::TaskResult(result)) => Some(result),
            _ => None,
        }),
        Some(TaskResultRecord { content, success: true, .. }) if content == "done"
    );
    assert!(!service.is_busy());
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn close_resolves_hitl_before_turn_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([ModelResponse::Completion(tool_completion(
        "call-1",
        "permission",
    ))]);
    let (service, runtime) = spawn_service_with_mode(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(PermissionTool)],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_message(
            "check".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("execute turn");
    loop {
        if matches!(
            events.recv().await.expect("service event"),
            ReactServiceEvent::HitlRequest(_)
        ) {
            break;
        }
    }

    service.close().await.expect("close service");
    let remaining = wait_for_turn(&mut events).await;
    let resolved = remaining
        .iter()
        .position(|event| matches!(event, ReactServiceEvent::HitlResolved { .. }))
        .expect("HITL resolution");
    let completed = remaining
        .iter()
        .position(|event| matches!(event, ReactServiceEvent::TurnCompleted))
        .expect("turn completion");
    assert!(resolved < completed);
    runtime.await.expect("service runtime");
    let entries = load_entries(&database, directory.path()).await;
    assert_eq!(
        approval_outcomes(&entries),
        [
            ("call-1", ToolApprovalOutcome::Requested),
            ("call-1", ToolApprovalOutcome::Canceled),
        ]
    );
}

#[tokio::test]
async fn interrupt_publishes_task_cancellation_before_turn_completion() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let model = ScriptedModel::new([ModelResponse::Completion(tool_completion(
        "call-1",
        "foreground",
    ))]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(ForegroundTaskTool { started })],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();
    service
        .execute(user_message("run".to_string(), "user-1".to_string(), None))
        .expect("execute turn");
    started_receiver.recv().await.expect("task starts");

    service.interrupt().expect("interrupt");
    let remaining = wait_for_turn(&mut events).await;
    let cancelled = remaining
        .iter()
        .position(|event| {
            matches!(
                event,
                ReactServiceEvent::TaskUpdated(crate::TaskInfo {
                    status: crate::RuntimeTaskStatus::Cancelled,
                    ..
                })
            )
        })
        .expect("task cancellation");
    let completed = remaining
        .iter()
        .position(|event| matches!(event, ReactServiceEvent::TurnCompleted))
        .expect("turn completion");
    assert!(cancelled < completed);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_recovers_after_model_panic() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([
        ModelResponse::Panic("injected model panic"),
        ModelResponse::Completion(completion("recovered")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");
    let mut events = service.events();

    service
        .execute(user_message(
            "panic".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("execute turn");
    let failed = wait_for_turn(&mut events).await;
    assert_matches!(
        failed.iter().find(|event| matches!(event, ReactServiceEvent::TurnError(_))),
        Some(ReactServiceEvent::TurnError(error)) if error.contains("injected model panic")
    );
    run_turn(&service, "retry", "user-2").await;
    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "recovered");
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_persists_panicking_tool_as_failure_and_continues() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([
        ModelResponse::Completion(tool_completion("call-1", "panic")),
        ModelResponse::Completion(completion("handled")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        vec![Box::new(PanickingTool)],
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");

    run_turn(&service, "panic", "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::Tool(tool)),
                success,
                ..
            })) => Some((tool, success)),
            _ => None,
        }),
        Some((tool, false))
            if tool.tool_call_id == "call-1"
                && tool.content.contains("injected tool panic")
    );
    assert_final_answer(&entries, "handled");
    close_service(&service, runtime).await;
}

#[tokio::test(start_paused = true)]
async fn service_stops_after_maximum_completion_failures() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Error(LlmError::Request("first".to_string())),
        ModelResponse::Error(LlmError::Request("second".to_string())),
    ]);
    let mut requests = model.requests();
    let config = ReactAgentConfig {
        max_exceptions: 2,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;
    service.open().await.expect("open service");

    run_turn(&service, "fail", "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_final_answer(
        &entries,
        "Reached the maximum of 2 exceptions without completing the task.",
    );
    assert_eq!(requests.seen().len(), 2);
    close_service(&service, runtime).await;
}

#[tokio::test(start_paused = true)]
async fn service_recovers_with_exception_limit_disabled() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Error(LlmError::Request("first".to_string())),
        ModelResponse::Error(LlmError::Request("second".to_string())),
        ModelResponse::Error(LlmError::Request("third".to_string())),
        ModelResponse::Error(LlmError::Request("fourth".to_string())),
        ModelResponse::Error(LlmError::Request("fifth".to_string())),
        ModelResponse::Error(LlmError::Request("sixth".to_string())),
        ModelResponse::Completion(completion("recovered")),
    ]);
    let mut requests = model.requests();
    let config = ReactAgentConfig {
        max_exceptions: 0,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;
    service.open().await.expect("open service");

    let started = tokio::time::Instant::now();
    run_turn(&service, "retry", "user-1").await;

    assert_eq!(started.elapsed(), Duration::from_secs(21));
    let entries = service.messages().await.expect("service messages");
    assert_final_answer(&entries, "recovered");
    assert_eq!(requests.seen().len(), 7);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_aborts_partial_stream_without_final_message() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let model = ScriptedModel::new([ModelResponse::Events(vec![Ok(StreamEvent::Content(
        StreamContent {
            content: "ghost".to_string(),
        },
    ))])]);
    let config = ReactAgentConfig {
        max_exceptions: 1,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;
    service.open().await.expect("open service");

    let events = run_turn(&service, "stream", "user-1").await;

    assert_matches!(
        events
            .iter()
            .find(|event| matches!(event, ReactServiceEvent::Agent(AgentEvent::StreamAborted))),
        Some(_)
    );
    let entries = service.messages().await.expect("service messages");
    assert!(!entries.iter().any(|entry| {
        matches!(
            entry.chat_message(),
            Some(chat_record::Message::Assistant(AssistantMessage {
                content: Some(content),
                ..
            })) if content == "ghost"
        )
    }));
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_retries_truncated_compaction_summary() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(truncated_completion("partial summary")),
        ModelResponse::Completion(completion("complete summary")),
        ModelResponse::Completion(completion("done")),
    ]);
    model.settings.context_window = Some(4_000);
    let mut requests = model.requests();
    let config = compaction_config();
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;
    service.open().await.expect("open service");

    run_turn(&service, &"history ".repeat(1_000), "user-1").await;

    let entries = service.messages().await.expect("service messages");
    let compactions = entries
        .iter()
        .filter_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ContextWindow(window)) => window.compaction.as_ref(),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(compactions.len(), 1);
    assert!(compactions[0]
        .summary
        .as_deref()
        .unwrap_or_default()
        .contains("complete summary"));
    assert!(!compactions[0]
        .summary
        .as_deref()
        .unwrap_or_default()
        .contains("partial summary"));
    assert_final_answer(&entries, "done");
    assert_eq!(requests.seen().len(), 3);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_retries_compaction_stream_output_after_final() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Events(vec![
            Ok(StreamEvent::Final(completion("invalid summary"))),
            Ok(StreamEvent::Content(StreamContent {
                content: "late".to_string(),
            })),
        ]),
        ModelResponse::Completion(completion("valid summary")),
        ModelResponse::Completion(completion("done")),
    ]);
    model.settings.context_window = Some(4_000);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        compaction_config(),
    )
    .await;
    service.open().await.expect("open service");

    run_turn(&service, &"history ".repeat(1_000), "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::ContextWindow(window)) => window.compaction.as_ref(),
            _ => None,
        }),
        Some(compaction)
            if compaction.summary.as_deref().unwrap_or_default().contains("valid summary")
                && !compaction.summary.as_deref().unwrap_or_default().contains("invalid summary")
    );
    assert_final_answer(&entries, "done");
    assert_eq!(requests.seen().len(), 3);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_compacts_when_custom_threshold_exceeds_completion_boundary() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(completion("summary")),
        ModelResponse::Completion(completion("done")),
    ]);
    model.settings.context_window = Some(1_000);
    let mut requests = model.requests();
    let config = ReactAgentConfig {
        compaction_threshold: Some(950),
        max_tokens: 800,
        ..ReactAgentConfig::default()
    };
    let (service, runtime) =
        spawn_service(&database, directory.path(), true, model, Vec::new(), config).await;
    service.open().await.expect("open service");

    run_turn(&service, &"x".repeat(3_700), "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries
            .iter()
            .find(|entry| matches!(&entry.payload, Some(agent_message::Payload::ContextWindow(window)) if window.compaction.is_some())),
        Some(_)
    );
    assert_final_answer(&entries, "done");
    assert_eq!(requests.seen().len(), 2);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_recovers_from_provider_context_overflow() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Error(LlmError::ContextOverflow("too long".to_string())),
        ModelResponse::Completion(completion("done")),
    ]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    service.open().await.expect("open service");

    run_turn(&service, "hello", "user-1").await;

    let entries = service.messages().await.expect("service messages");
    assert_matches!(
        entries.iter().find_map(|entry| match &entry.payload {
            Some(agent_message::Payload::Log(log)) => Some(log),
            _ => None,
        }),
        Some(log) if log.text.contains("too long; compacting")
    );
    assert_matches!(
        entries
            .iter()
            .find(|entry| matches!(&entry.payload, Some(agent_message::Payload::ContextWindow(window)) if window.compaction.is_some())),
        Some(_)
    );
    assert_final_answer(&entries, "done");
    assert_eq!(requests.seen().len(), 2);
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_creation_rejects_orphan_and_duplicate_tool_results() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("orphan.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![tool_entry("call-1", "orphan", true)]),
    )
    .await;
    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error) if error.contains("has no unanswered call")
    );

    let database = directory.path().join("duplicate.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            assistant_tool_entry(vec![tool_call("call-1", "read")]),
            tool_entry("call-1", "first", true),
            tool_entry("call-1", "duplicate", false),
        ]),
    )
    .await;
    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error) if error.contains("has no unanswered call")
    );
}

#[tokio::test]
async fn service_creation_rejects_duplicate_and_empty_tool_call_ids() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("duplicate.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![assistant_tool_entry(vec![
            tool_call("call-1", "one"),
            tool_call("call-1", "two"),
        ])]),
    )
    .await;
    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error) if error.contains("is duplicated")
    );

    let database = directory.path().join("empty.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![assistant_tool_entry(vec![tool_call("", "one")])]),
    )
    .await;
    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error) if error.contains("ID is empty")
    );
}

#[tokio::test]
async fn open_accepts_out_of_order_mixed_tool_batches_and_heals_only_final_batch() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            assistant_tool_entry(vec![tool_call("call-1", "one"), tool_call("call-2", "two")]),
            tool_entry("call-2", "Tool error: failed", false),
            tool_entry("call-1", "success", true),
            assistant_tool_entry(vec![tool_call("call-3", "three")]),
            tool_entry("call-3", "success", true),
            assistant_tool_entry(vec![
                tool_call("call-4", "four"),
                tool_call("call-5", "five"),
            ]),
            tool_entry("call-5", "success", true),
        ]),
    )
    .await;
    let model = ScriptedModel::new(Vec::new());
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;

    let opened = service.open().await.expect("heal final batch");
    let repaired = opened
        .iter()
        .filter_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::Tool(tool)) if tool.content == "Tool error: interrupted" => {
                Some(tool)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(repaired.len(), 1);
    assert_eq!(repaired[0].tool_call_id, "call-4");
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn service_creation_rejects_nonempty_partially_initialized_trace() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    seed_entries(
        &database,
        directory.path(),
        vec![user_entry("orphan", "user-1")],
    )
    .await;

    assert_matches!(
        service_creation_error(&database, directory.path()).await,
        ReactServiceError::Agent(error)
            if error.contains("missing a context window")
    );
}

#[tokio::test]
async fn forked_session_deduplicates_the_prefix_and_accepts_the_abandoned_suffix() {
    let directory = tempfile::tempdir().expect("temporary directory");
    assert!(std::process::Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(directory.path())
        .status()
        .expect("Git init")
        .success());
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([
        ModelResponse::Completion(completion("first answer")),
        ModelResponse::Completion(completion("second answer")),
        ModelResponse::Completion(completion("forked answer")),
    ]);
    let mut requests = model.requests();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let mut store = SQLiteStore::open(&database, "session", directory.path(), true, false)
        .expect("open parent store");
    store.checkpoints_enabled = true;
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("start parent service");
    let runtime = tokio::spawn(runtime);
    service.open().await.expect("open service");
    let extension = AgentMessage {
        id: "extension-1".to_string(),
        payload: Some(agent_message::Payload::Extension(ExtensionRecord {
            extension: "todo".to_string(),
            kind: "state".to_string(),
            ..ExtensionRecord::default()
        })),
        ..AgentMessage::default()
    };
    service
        .execute(extension.clone())
        .expect("append extension");
    run_turn(&service, "first", "user-1").await;
    run_turn(&service, "second", "user-2").await;
    let parent = service.messages().await.expect("parent messages");
    assert_matches!(parent.iter().find(|entry| entry.id == "user-1").and_then(|entry| entry.chat_message()),
        Some(chat_record::Message::User(user)) if user.checkpoint.is_some());
    close_service(&service, runtime).await;
    assert_eq!(requests.seen().len(), 2);

    SQLiteStore::fork(&database, "session", "user-2", "forked")
        .expect("rewind before the second turn");
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("forked answer"))]);
    let mut requests = model.requests();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let store = SQLiteStore::open(&database, "forked", directory.path(), false, false)
        .expect("open forked store");
    let (forked, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        "forked",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("start forked service");
    let runtime = tokio::spawn(runtime);
    let opened = forked.open().await.expect("open forked service");
    assert_eq!(user_contents(&opened), ["first"]);
    assert_matches!(
        forked.execute(AgentMessage {
            id: "rejected-rewind".to_string(),
            payload: Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::User(UserMessage {
                checkpoint: None,
                    content: "replacement".to_string(),
                })),
                rewind_to: Some("user-1".to_string()),
                ..ChatRecord::default()
            })),
            ..AgentMessage::default()
        }),
        Err(ReactServiceError::Agent(error)) if error.contains("MirrorRPC.ForkSession")
    );
    assert!(opened.iter().any(|entry| entry.id == extension.id));
    forked
        .execute(extension.clone())
        .expect("retry inherited extension");

    let mut events = forked.events();
    forked
        .execute(user_message(
            "first".to_string(),
            "user-1".to_string(),
            None,
        ))
        .expect("resubmit the prefix's message");
    forked
        .execute(user_message(
            "second, revised".to_string(),
            "user-2".to_string(),
            None,
        ))
        .expect("reuse the abandoned suffix's message id");
    let turn_events = wait_for_turn(&mut events).await;
    assert_matches!(
        turn_events
            .iter()
            .find(|event| matches!(event, ReactServiceEvent::TurnError(_))),
        None
    );
    let entries = forked.messages().await.expect("forked messages");
    assert_eq!(user_contents(&entries), ["first", "second, revised"]);
    assert_eq!(
        entries
            .iter()
            .filter(|entry| entry.id == extension.id)
            .count(),
        1
    );
    assert_final_answer(&entries, "forked answer");
    assert_eq!(requests.seen().len(), 1);
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.sequence.expect("sequence"))
            .collect::<Vec<_>>(),
        (1..=entries.len() as u64).collect::<Vec<_>>()
    );
    close_service(&forked, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, parent);
}

async fn service_creation_error(database: &Path, workspace: &Path) -> ReactServiceError {
    let store = Box::new(
        SQLiteStore::open(database, "session", workspace, false, false).expect("open SQLite store"),
    );
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        ScriptedModel::new(Vec::new()),
        test_provider(DEFAULT_CONTEXT_WINDOW).await,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    match ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        store,
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    {
        Ok(_) => panic!("service creation unexpectedly succeeded"),
        Err(error) => error,
    }
}

async fn spawn_service(
    database: &Path,
    workspace: &Path,
    create: bool,
    model: ScriptedModel,
    tools: Vec<Box<dyn AnyTool>>,
    config: ReactAgentConfig,
) -> (ReactAgentServiceHandle, tokio::task::JoinHandle<()>) {
    spawn_service_with_mode(
        database,
        workspace,
        create,
        model,
        tools,
        config,
        AgentApprovalMode::Auto,
    )
    .await
}

async fn spawn_service_with_mode(
    database: &Path,
    workspace: &Path,
    create: bool,
    model: ScriptedModel,
    tools: Vec<Box<dyn AnyTool>>,
    config: ReactAgentConfig,
    approval_mode: impl Into<SessionApprovalMode>,
) -> (ReactAgentServiceHandle, tokio::task::JoinHandle<()>) {
    let approval_mode: SessionApprovalMode = approval_mode.into();
    let store = Box::new(
        SQLiteStore::open(database, "session", workspace, create, false)
            .expect("open SQLite store"),
    );
    let tools = tools.into_iter().map(Arc::from).collect();
    let tool_executor = DirectToolExecutor::new(tools);
    let provider = model.provider().await;
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &tool_executor,
        system_prompt(),
        config,
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        store,
        "session",
        approval_mode,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    (service, tokio::spawn(runtime))
}

async fn run_turn(
    service: &ReactAgentServiceHandle,
    instruction: &str,
    message_id: &str,
) -> Vec<ReactServiceEvent> {
    let mut events = service.events();
    service
        .set_inference_config(default_inference())
        .await
        .expect("set inference");
    service
        .execute(user_message(
            instruction.to_string(),
            message_id.to_string(),
            None,
        ))
        .expect("execute turn");
    let events = wait_for_turn(&mut events).await;
    assert_matches!(
        events
            .iter()
            .find(|event| matches!(event, ReactServiceEvent::TurnError(_))),
        None
    );
    events
}

async fn wait_for_turn(
    events: &mut broadcast::Receiver<ReactServiceEvent>,
) -> Vec<ReactServiceEvent> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut received = Vec::new();
        loop {
            let event = events.recv().await.expect("service event");
            let completed = matches!(event, ReactServiceEvent::TurnCompleted);
            received.push(event);
            if completed {
                return received;
            }
        }
    })
    .await
    .expect("turn completion timed out")
}

async fn wait_for_close(
    events: &mut broadcast::Receiver<ReactServiceEvent>,
) -> Vec<ReactServiceEvent> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut received = Vec::new();
        loop {
            let event = events.recv().await.expect("service event");
            let closed = matches!(event, ReactServiceEvent::Closed);
            received.push(event);
            if closed {
                return received;
            }
        }
    })
    .await
    .expect("service close timed out")
}

async fn close_service(service: &ReactAgentServiceHandle, runtime: tokio::task::JoinHandle<()>) {
    service.close().await.expect("close service");
    runtime.await.expect("service runtime");
}

async fn load_entries(database: &Path, workspace: &Path) -> Vec<AgentMessage> {
    SQLiteStore::open(database, "session", workspace, false, false)
        .expect("reopen SQLite store")
        .load("session", true)
        .await
        .expect("load entries")
}

async fn seed_entries(database: &Path, workspace: &Path, mut entries: Vec<AgentMessage>) {
    let mut store = SQLiteStore::open(database, "session", workspace, true, false)
        .expect("create SQLite store");
    assert!(store
        .load("session", true)
        .await
        .expect("load empty trace")
        .is_empty());
    store
        .append("session", 0, &mut entries)
        .await
        .expect("seed trace");
}

fn tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: format!("Run {name}."),
        parameters: serde_json::Map::new(),
    }
}

fn compaction_config() -> ReactAgentConfig {
    ReactAgentConfig {
        compaction_threshold: Some(1_000),
        max_tokens: 1_000,
        ..ReactAgentConfig::default()
    }
}

fn default_inference() -> mirror_protocol::InferenceConfig {
    mirror_protocol::InferenceConfig {
        model: "scripted".to_string(),
        api: None,
        base_url: None,
        reasoning_effort: Some("medium".to_string()),
        ..Default::default()
    }
}

fn user_message(content: String, id: String, display: Option<String>) -> AgentMessage {
    AgentMessage {
        id,
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content,
                checkpoint: None,
            })),
            success: true,
            display,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

/// Reports the history length each time a turn finishes cleanly.
struct TurnRecorder {
    calls: mpsc::UnboundedSender<usize>,
}

impl ContextHooks for TurnRecorder {
    fn on_turn_end<'a>(
        &'a self,
        history: &'a [AgentMessage],
        _turn: TurnEnd<'a>,
    ) -> BoxFuture<'a, ()> {
        let _ = self.calls.send(history.len());
        Box::pin(async {})
    }
}

#[tokio::test]
async fn hooks_hear_on_turn_end_only_after_a_clean_turn() {
    let (calls, mut finished) = mpsc::unbounded_channel();
    let (model_started, mut started) = mpsc::unbounded_channel();
    let (_complete, completion_receiver) = oneshot::channel();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        ScriptedModel::new([
            ModelResponse::Completion(completion("done")),
            ModelResponse::Delayed {
                started: model_started,
                completion: completion_receiver,
            },
        ]),
        test_provider(DEFAULT_CONTEXT_WINDOW).await,
        None,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("create agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(InMemoryStore::default()),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        vec![Box::new(TurnRecorder { calls })],
        None,
        broadcast::channel(1024).0,
        None,
    )
    .await
    .expect("create service");
    let runtime = tokio::spawn(runtime);

    run_turn(&service, "hello", "user-1").await;
    // The context window, the user message and the reply.
    assert_eq!(finished.try_recv().expect("clean turn finished"), 3);

    let mut events = service.events();
    service.compact().await.expect("compact between turns");
    wait_for_turn(&mut events).await;
    assert_matches!(finished.try_recv(), Err(_));

    let mut events = service.events();
    service
        .execute(user_message(
            "again".to_string(),
            "user-2".to_string(),
            None,
        ))
        .expect("start turn");
    started.recv().await.expect("model call started");
    service.interrupt().expect("interrupt");
    wait_for_turn(&mut events).await;
    assert_matches!(finished.try_recv(), Err(_));

    close_service(&service, runtime).await;
}

fn completion(content: &str) -> Completion {
    Completion {
        message: AssistantMessage {
            content: Some(content.to_string()),
            ..AssistantMessage::default()
        },
        usage: None,
        truncated: false,
    }
}

fn truncated_completion(content: &str) -> Completion {
    Completion {
        truncated: true,
        ..completion(content)
    }
}

fn tool_completion(call_id: &str, name: &str) -> Completion {
    Completion {
        message: AssistantMessage {
            tool_calls: vec![tool_call(call_id, name)],
            ..AssistantMessage::default()
        },
        usage: None,
        truncated: false,
    }
}

fn initialized_trace(mut entries: Vec<AgentMessage>) -> Vec<AgentMessage> {
    let mut initialized = vec![AgentMessage {
        payload: Some(agent_message::Payload::ContextWindow(
            mirror_protocol::ContextWindowRecord {
                system_prompt: "system".to_string(),
                ..Default::default()
            },
        )),
        ..AgentMessage::default()
    }];
    initialized.append(&mut entries);
    initialized
}

fn assistant_tool_entry(calls: Vec<ToolCall>) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Assistant(AssistantMessage {
                tool_calls: calls,
                ..AssistantMessage::default()
            })),
            success: true,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn tool_call(id: &str, name: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        }),
        ..ToolCall::default()
    }
}

fn tool_entry(call_id: &str, content: &str, success: bool) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Tool(ToolMessage {
                tool_call_id: call_id.to_string(),
                content: content.to_string(),
                name: None,
            })),
            success,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn user_entry(content: &str, id: &str) -> AgentMessage {
    AgentMessage {
        id: id.to_string(),
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: content.to_string(),
            })),
            success: true,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn log_entry(text: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::Log(LogRecord {
            level: "info".to_string(),
            text: text.to_string(),
        })),
        ..AgentMessage::default()
    }
}

fn assert_final_answer(entries: &[AgentMessage], expected: &str) {
    assert_matches!(
        entries.last().and_then(AgentMessage::chat_message),
        Some(chat_record::Message::Assistant(AssistantMessage {
            content: Some(content),
            tool_calls,
            ..
        })) if content == expected && tool_calls.is_empty()
    );
}

struct ChildApprovalTool;

impl crate::tools::Tool for ChildApprovalTool {
    type Args = serde_json::Value;

    fn name(&self) -> &str {
        "child_approval"
    }

    fn description(&self) -> &str {
        "Request approval from a child context."
    }

    fn parameters(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }

    async fn run(
        &self,
        _args: Self::Args,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, crate::tools::ToolError> {
        let child = ExecutionContext::new(
            AgentApprovalMode::Auto,
            None,
            TaskManagerHandle::closed(),
            Some(context),
            None,
        );
        let approved = child
            .request_approval(&ToolCall {
                id: "child-call".into(),
                ..Default::default()
            })
            .await
            .expect("child approval");
        assert!(approved);
        assert!(child.tool_permissions().allow_outside_workspace());
        Ok(ToolOutput::success("approved"))
    }
}

#[tokio::test]
async fn automatic_mode_releases_child_approval_while_parent_is_idle() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (service, runtime) = spawn_service_with_mode(
        &directory.path().join("react.db"),
        directory.path(),
        true,
        ScriptedModel::new([]),
        vec![Box::new(ChildApprovalTool)],
        ReactAgentConfig::default(),
        AgentApprovalMode::Hitl,
    )
    .await;
    let mut events = service.events();
    let caller = service.clone();
    let call =
        tokio::spawn(async move { caller.call_tool("child_approval".into(), "{}".into()).await });
    let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
        .await
        .expect("approval arrives")
        .expect("service event");
    assert_matches!(event, ReactServiceEvent::HitlRequest(_));
    assert!(!service.is_busy());
    assert_matches!(
        service.compact().await,
        Err(ReactServiceError::ConversationBusy)
    );
    service
        .set_approval_mode(AgentApprovalMode::Auto)
        .await
        .expect("automatic mode");
    let output = tokio::time::timeout(Duration::from_secs(5), call)
        .await
        .expect("approval released")
        .expect("caller")
        .expect("tool result");
    assert!(output.success, "{output:?}");
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn manual_compaction_persists_a_window_without_answering_and_retains_recent_tools() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let history = initialized_trace(vec![
        user_entry(&"old history ".repeat(8_000), "old"),
        user_entry("recent task", "recent"),
        assistant_tool_entry(vec![tool_call("call-1", "permission")]),
        tool_entry("call-1", "tool result", true),
    ]);
    seed_entries(&database, directory.path(), history).await;
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("summary"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let before = service.messages().await.expect("history");
    let mut events = service.events();
    service.compact().await.expect("accept compaction");
    let events = wait_for_turn(&mut events).await;
    assert!(!events
        .iter()
        .any(|event| matches!(event, ReactServiceEvent::TurnError(_))));
    let entries = service.messages().await.expect("compacted history");
    assert_eq!(&entries[..before.len()], before.as_slice());
    let Some(agent_message::Payload::ContextWindow(window)) =
        &entries.last().expect("window").payload
    else {
        panic!("expected context window");
    };
    let compacted = window.compaction.as_ref().expect("compaction");
    assert!(compacted
        .summary
        .as_deref()
        .expect("summary")
        .contains("summary"));
    assert_eq!(compacted.recent.len(), 3);
    assert_eq!(requests.seen().len(), 1);
    assert_eq!(requests.seen()[0].max_tokens, None);
    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, entries);
}

#[tokio::test]
async fn manual_compaction_of_empty_or_short_history_does_not_call_the_model() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let mut model = ScriptedModel::new([ModelResponse::Completion(completion("answer"))]);
    let mut requests = model.requests();
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let mut events = service.events();
    service.compact().await.expect("empty compaction");
    wait_for_turn(&mut events).await;
    assert!(requests.seen().is_empty());
    run_turn(&service, "hello", "user").await;
    let mut events = service.events();
    service.compact().await.expect("short compaction");
    wait_for_turn(&mut events).await;
    assert_eq!(requests.seen().len(), 1);
    let entries = service.messages().await.expect("history");
    let Some(agent_message::Payload::ContextWindow(window)) =
        &entries.last().expect("window").payload
    else {
        panic!("expected context window");
    };
    let compacted = window.compaction.as_ref().expect("compaction");
    assert!(compacted.summary.is_none());
    assert_eq!(compacted.recent.len(), 2);
    assert_matches!(compacted.recent[0].chat_message(),
        Some(chat_record::Message::User(user)) if user.content == "hello");
    assert_matches!(compacted.recent[1].chat_message(),
        Some(chat_record::Message::Assistant(assistant)) if assistant.content.as_deref() == Some("answer"));
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn failed_manual_compaction_preserves_history_and_allows_retry() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            user_entry(&"old history ".repeat(8_000), "old"),
            user_entry("recent", "recent"),
        ]),
    )
    .await;
    let model = ScriptedModel::new([
        ModelResponse::Error(LlmError::ContextOverflow("summary failed".into())),
        ModelResponse::Completion(completion("retry summary")),
    ]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let before = service.messages().await.expect("history");
    let mut events = service.events();
    service.compact().await.expect("accept compaction");
    let failed = wait_for_turn(&mut events).await;
    assert!(failed
        .iter()
        .any(|event| matches!(event, ReactServiceEvent::TurnError(_))));
    assert_eq!(service.messages().await.expect("unchanged history"), before);
    service.compact().await.expect("retry compaction");
    let retried = wait_for_turn(&mut events).await;
    assert!(!retried
        .iter()
        .any(|event| matches!(event, ReactServiceEvent::TurnError(_))));
    close_service(&service, runtime).await;
}

#[tokio::test]
async fn manual_compaction_is_interruptible_and_rejects_concurrent_compaction() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    seed_entries(
        &database,
        directory.path(),
        initialized_trace(vec![
            user_entry(&"old history ".repeat(8_000), "old"),
            user_entry("recent", "recent"),
        ]),
    )
    .await;
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let (_finish, completion) = oneshot::channel();
    let model = ScriptedModel::new([ModelResponse::Delayed {
        started,
        completion,
    }]);
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        false,
        model,
        Vec::new(),
        ReactAgentConfig::default(),
    )
    .await;
    let before = service.messages().await.expect("history");
    let mut events = service.events();
    service.compact().await.expect("accept compaction");
    started_receiver.recv().await.expect("summary started");
    assert_matches!(
        service.compact().await,
        Err(ReactServiceError::ConversationBusy)
    );
    assert!(service
        .interrupt()
        .expect("interrupt")
        .await
        .expect("accepted"));
    let interrupted = wait_for_turn(&mut events).await;
    assert!(interrupted
        .iter()
        .any(|event| matches!(event, ReactServiceEvent::TurnInterrupted)));
    assert_eq!(service.messages().await.expect("unchanged history"), before);
    close_service(&service, runtime).await;
    assert_eq!(load_entries(&database, directory.path()).await, before);
}

#[tokio::test]
async fn manual_compaction_respects_disabled_compaction() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let (service, runtime) = spawn_service(
        &database,
        directory.path(),
        true,
        ScriptedModel::new([]),
        Vec::new(),
        ReactAgentConfig {
            disable_compaction: true,
            ..ReactAgentConfig::default()
        },
    )
    .await;
    let before = service.messages().await.expect("history");
    let mut events = service.events();
    service.compact().await.expect("request accepted");
    let events = wait_for_turn(&mut events).await;
    assert!(events.iter().any(|event| matches!(event,
        ReactServiceEvent::TurnError(error) if error.contains("compaction is disabled"))));
    assert_eq!(service.messages().await.expect("history"), before);
    close_service(&service, runtime).await;
}
