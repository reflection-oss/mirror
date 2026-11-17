use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::compaction::SummaryCompactorFactory;
use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::react_agent_service::tests::test_provider;
use crate::Store;
use assert_matches::assert_matches;
use async_stream::stream;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::ContextWindowRecord;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;

use super::*;
use crate::providers::Llm;
use crate::providers::LlmError;
use crate::providers::StreamEvent;
use crate::react::InMemoryStore;
use crate::system_prompt::SystemPromptConfig;
use crate::task_manager::TaskManager;
use crate::tools::DirectToolExecutor;
use crate::AgentApprovalMode;
use crate::ExecutionContext;

struct ScriptedModel {
    replies: Vec<&'static str>,
    calls: Arc<AtomicUsize>,
    truncated: bool,
    overflow: bool,
}

impl Llm for ScriptedModel {
    fn model(&self) -> &str {
        "scripted"
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        let index = self.calls.fetch_add(1, Ordering::SeqCst);
        let content = self.replies[index].to_string();
        let overflow = self.overflow;
        stream! {
            if overflow {
                yield Err(LlmError::ContextOverflow("provider rejected the context".to_string()));
                return;
            }
            yield Ok(StreamEvent::Final(Completion {
                message: AssistantMessage {
                    content: Some(content),
                    ..AssistantMessage::default()
                },
                usage: None,
                truncated: self.truncated,
            }));
        }
    }
}

fn window_entry() -> AgentMessage {
    window_entry_with("system")
}

fn window_entry_with(system_prompt: &str) -> AgentMessage {
    AgentMessage {
        payload: Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
            system_prompt: system_prompt.to_string(),
            ..ContextWindowRecord::default()
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
                images: Vec::new(),
                content: content.to_string(),
            })),
            success: true,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn execution_context() -> ExecutionContext {
    let (_manager, tasks, _task_events, _task_runner) = TaskManager::new();
    ExecutionContext::new(AgentApprovalMode::Auto, None, tasks, None, None)
}

#[tokio::test]
async fn truncated_answers_remain_bounded_across_compaction() {
    let calls = Arc::new(AtomicUsize::new(0));
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let mut agent = ReactAgent::new(
        ScriptedModel {
            replies: vec!["Partial answer"; 3],
            calls: calls.clone(),
            truncated: true,
            overflow: false,
        },
        test_provider(4_000).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig {
            max_tokens: 3_800,
            ..ReactAgentConfig::default()
        },
    )
    .expect("agent");
    let mut store = InMemoryStore::default();
    let mut entries = vec![window_entry(), user_entry("Finish the task", "user")];
    store
        .append("session", 0, &mut entries)
        .await
        .expect("seed history");
    let (mut context, _events) =
        ReactContext::new("session", entries, Box::new(store), execution_context());
    assert_matches!(
        agent.run(&mut context, &tool_executor).await,
        Err(ReactAgentError::IncompleteCompletion)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let (history, _) = context.history(0, 1000).await.expect("stored history");
    assert_eq!(
        history
            .iter()
            .filter(|entry| matches!(
                entry.payload,
                Some(agent_message::Payload::ContextWindow(_))
            ))
            .count(),
        4
    );
}

#[tokio::test]
async fn empty_truncated_answers_compact_before_retrying() {
    let calls = Arc::new(AtomicUsize::new(0));
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let mut agent = ReactAgent::new(
        ScriptedModel {
            replies: vec![""; 3],
            calls: calls.clone(),
            truncated: true,
            overflow: false,
        },
        test_provider(4_000).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig {
            max_tokens: 3_800,
            ..ReactAgentConfig::default()
        },
    )
    .expect("agent");
    let mut store = InMemoryStore::default();
    let mut entries = vec![window_entry(), user_entry("Finish the task", "user")];
    store
        .append("session", 0, &mut entries)
        .await
        .expect("seed history");
    let (mut context, _events) =
        ReactContext::new("session", entries, Box::new(store), execution_context());
    assert_matches!(
        agent.run(&mut context, &tool_executor).await,
        Err(ReactAgentError::IncompleteCompletion)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    let (history, _) = context.history(0, 1000).await.expect("stored history");
    assert_eq!(
        history
            .iter()
            .filter(|entry| matches!(
                entry.payload,
                Some(agent_message::Payload::ContextWindow(_))
            ))
            .count(),
        4
    );
}

struct PrewarmRecorder {
    prewarmed: std::sync::mpsc::Sender<CompletionRequest>,
}

impl Llm for PrewarmRecorder {
    fn model(&self) -> &str {
        "prewarm-recorder"
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        futures_util::stream::empty()
    }

    fn prewarm(&self, request: CompletionRequest) {
        self.prewarmed
            .send(request)
            .expect("test receiver is alive");
    }
}

#[tokio::test]
async fn initialize_prewarms_the_model_with_the_new_context() {
    let (prewarmed, requests) = std::sync::mpsc::channel();
    let tool_executor = DirectToolExecutor::new(vec![Arc::new(crate::tools::WebSearchTool::new(
        Box::new(|selection| {
            Ok(Box::new(
                crate::tools::web_test_provider::MemoryWebProvider(selection),
            ))
        }),
    ))])
    .with_tool_visibility(|_, context| {
        context.web_provider() == Some(crate::web::WebProviderConfig::Tavily)
    });
    let mut agent = ReactAgent::new(
        PrewarmRecorder { prewarmed },
        test_provider(4_000).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (mut context, _events) = ReactContext::new(
        "session",
        Vec::new(),
        Box::new(InMemoryStore::default()),
        execution_context(),
    );

    agent
        .initialize(&mut context, &tool_executor)
        .await
        .expect("initialize");

    let request = requests.try_recv().expect("initialize prewarms once");
    assert_matches!(
        request.messages.as_slice(),
        [chat_record::Message::System(system)] if system.content.contains("system")
    );
    assert_eq!(request.max_tokens, None);
    assert!(
        request.tools.is_empty(),
        "prewarm must respect tool visibility"
    );
}

#[tokio::test]
async fn compaction_keeps_only_the_active_window_resident() {
    let mut store = InMemoryStore::default();
    let mut history = vec![window_entry()];
    history
        .extend((0..3).map(|index| user_entry(&"history ".repeat(500), &format!("user-{index}"))));
    store
        .append("session", 0, &mut history)
        .await
        .expect("seed history");
    let entries = store.load("session", false).await.expect("load window");
    assert_eq!(entries.len(), 4);
    let calls = Arc::new(AtomicUsize::new(0));
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let mut agent = ReactAgent::new(
        ScriptedModel {
            replies: vec!["summary", "done"],
            calls: calls.clone(),
            truncated: false,
            overflow: false,
        },
        test_provider(4_000).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig {
            compaction_threshold: Some(1_000),
            max_tokens: 1_000,
            ..ReactAgentConfig::default()
        },
    )
    .expect("agent");
    let (mut context, _events) =
        ReactContext::new("session", entries, Box::new(store), execution_context());
    agent
        .initialize(&mut context, &tool_executor)
        .await
        .expect("initialize");

    agent.run(&mut context, &tool_executor).await.expect("run");

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let resident = context.entries().to_vec();
    assert_matches!(
        resident.first().and_then(|entry| entry.payload.as_ref()),
        Some(agent_message::Payload::ContextWindow(window))
            if window.compaction.as_ref().is_some_and(|compaction|
                compaction.model == "scripted" && compaction.inference.is_none()
                    && compaction.summary.as_deref() == Some("summary"))
    );
    assert!(resident
        .iter()
        .all(|entry| { !matches!(entry.chat_message(), Some(chat_record::Message::User(_))) }));
    assert_matches!(
        resident.last().and_then(AgentMessage::chat_message),
        Some(chat_record::Message::Assistant(AssistantMessage { content: Some(content), .. }))
            if content == "done"
    );
    let (stored, _) = context.history(0, 1000).await.expect("stored history");
    assert!(stored.len() > resident.len());
    assert_eq!(stored[stored.len() - resident.len()..], resident[..]);
}

#[tokio::test]
async fn compaction_records_context_tokens_before_and_after() {
    let mut store = InMemoryStore::default();
    let mut history = vec![window_entry()];
    history
        .extend((0..3).map(|index| user_entry(&"history ".repeat(500), &format!("user-{index}"))));
    store
        .append("session", 0, &mut history)
        .await
        .expect("seed history");
    let entries = store.load("session", false).await.expect("load window");
    let before = prompt_size(&entries, 0) as u64;
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let mut agent = ReactAgent::new(
        ScriptedModel {
            replies: vec!["summary", "done"],
            calls: Arc::new(AtomicUsize::new(0)),
            truncated: false,
            overflow: false,
        },
        test_provider(4_000).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig {
            compaction_threshold: Some(1_000),
            max_tokens: 1_000,
            ..ReactAgentConfig::default()
        },
    )
    .expect("agent");
    let (mut context, mut events) =
        ReactContext::new("session", entries, Box::new(store), execution_context());
    agent
        .initialize(&mut context, &tool_executor)
        .await
        .expect("initialize");

    agent.run(&mut context, &tool_executor).await.expect("run");

    let after = std::iter::from_fn(|| events.try_recv().ok())
        .skip_while(|event| !matches!(event, AgentEvent::CompactionStarted))
        .find_map(|event| match event {
            AgentEvent::ContextUsage(usage) => Some(usage.tokens),
            _ => None,
        })
        .expect("context usage after compaction");
    assert!(
        after < before,
        "compaction must shrink {before} tokens, got {after}"
    );
    assert_matches!(
        context.entries().first().and_then(|entry| entry.payload.as_ref()),
        Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
            compaction: Some(compaction),
            ..
        })) if compaction.context_tokens_before == Some(before)
            && compaction.context_tokens_after == Some(after)
    );
}

/// How the scripted model answers a run without compaction.
#[derive(Debug, Default, Clone, Copy)]
struct Replies {
    truncated: bool,
    overflow: bool,
}

async fn agent_without_compaction(
    system_prompt: &str,
    replies: Replies,
    calls: Arc<AtomicUsize>,
    tool_executor: &DirectToolExecutor,
    compactor_factory: impl CompactorFactory + 'static,
) -> ReactAgent<ScriptedModel> {
    ReactAgent::new(
        ScriptedModel {
            replies: vec!["answer"; 3],
            calls,
            truncated: replies.truncated,
            overflow: replies.overflow,
        },
        test_provider(4_000).await,
        None,
        compactor_factory,
        tool_executor,
        SystemPromptConfig {
            custom_prompt: Some(system_prompt.to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig {
            disable_compaction: true,
            max_tokens: 3_800,
            ..ReactAgentConfig::default()
        },
    )
    .expect("agent")
}

async fn run_without_compaction(
    user: &str,
    replies: Replies,
) -> (Result<(), ReactAgentError>, ReactContext, usize) {
    run_without_compaction_from(window_entry(), user, replies).await
}

async fn run_without_compaction_from(
    first: AgentMessage,
    user: &str,
    replies: Replies,
) -> (Result<(), ReactAgentError>, ReactContext, usize) {
    let calls = Arc::new(AtomicUsize::new(0));
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let mut agent = agent_without_compaction(
        "system",
        replies,
        calls.clone(),
        &tool_executor,
        SummaryCompactorFactory,
    )
    .await;
    let mut store = InMemoryStore::default();
    let mut entries = vec![first, user_entry(user, "user")];
    store
        .append("session", 0, &mut entries)
        .await
        .expect("seed history");
    let (mut context, _events) =
        ReactContext::new("session", entries, Box::new(store), execution_context());
    let result = agent.run(&mut context, &tool_executor).await;
    (result, context, calls.load(Ordering::SeqCst))
}

#[tokio::test]
async fn disabled_compaction_starts_a_session_with_a_system_prompt_beyond_the_compaction_line() {
    // About 3,650 tokens: over the 3,600-token compaction line of the 4,000-token
    // window, but under the window itself.
    let system_prompt = "x".repeat(14_600);
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let mut agent = agent_without_compaction(
        &system_prompt,
        Replies::default(),
        Arc::new(AtomicUsize::new(0)),
        &tool_executor,
        SummaryCompactorFactory,
    )
    .await;
    let (mut context, _events) = ReactContext::new(
        "session",
        Vec::new(),
        Box::new(InMemoryStore::default()),
        execution_context(),
    );

    agent
        .initialize(&mut context, &tool_executor)
        .await
        .expect("the system prompt still fits the model window");
    assert_matches!(context.entries(), [entry] if matches!(
        entry.payload.as_ref(),
        Some(agent_message::Payload::ContextWindow(window))
            if window.system_prompt.contains(&system_prompt)
    ));
}

#[tokio::test]
async fn disabled_compaction_rejects_history_without_a_context_window() {
    let (result, _context, calls) =
        run_without_compaction_from(user_entry("earlier", "earlier"), "task", Replies::default())
            .await;

    assert_matches!(
        result,
        Err(ReactAgentError::Context(ReactContextError::InvalidHistory(
            _
        )))
    );
    assert_eq!(calls, 0);
}

#[tokio::test]
async fn disabled_compaction_accepts_a_system_prompt_beyond_the_compaction_line() {
    // About 3,650 tokens: over the 3,600-token compaction line of the 4,000-token
    // window, but under the window itself.
    let system_prompt = "x".repeat(14_600);
    let (result, context, calls) = run_without_compaction_from(
        window_entry_with(&system_prompt),
        "task",
        Replies::default(),
    )
    .await;

    result.expect("the prompt still fits the model window");
    assert_eq!(calls, 1);
    assert_matches!(context.entries()[0].payload.as_ref(),
        Some(agent_message::Payload::ContextWindow(window))
            if window.system_prompt == system_prompt && window.compaction.is_none());
    assert_matches!(context.entries()[1].chat_message(),
        Some(chat_record::Message::User(message)) if message.content == "task");
}

#[tokio::test]
async fn disabled_compaction_uses_context_beyond_the_default_threshold() {
    let user = "x".repeat(15_000);
    let (result, context, calls) = run_without_compaction(&user, Replies::default()).await;

    result.expect("the prompt still fits the model window");
    assert_eq!(calls, 1);
    assert_matches!(context.entries()[0].payload.as_ref(),
        Some(agent_message::Payload::ContextWindow(window)) if window.compaction.is_none());
    assert_matches!(context.entries()[1].chat_message(),
        Some(chat_record::Message::User(message)) if message.content == user);
}

#[tokio::test]
async fn disabled_compaction_retains_history_after_reduced_budget_truncation() {
    let user = "x".repeat(4_000);
    let (result, mut context, calls) = run_without_compaction(
        &user,
        Replies {
            truncated: true,
            ..Replies::default()
        },
    )
    .await;

    assert_matches!(result, Err(ReactAgentError::IncompleteCompletion));
    assert_eq!(calls, 3);
    let (history, _) = context.history(0, 1000).await.expect("stored history");
    assert_eq!(
        history
            .iter()
            .filter(|entry| matches!(
                entry.payload,
                Some(agent_message::Payload::ContextWindow(_))
            ))
            .count(),
        1
    );
    assert_matches!(context.entries()[1].chat_message(),
        Some(chat_record::Message::User(message)) if message.content == user);
}

#[tokio::test]
async fn disabled_compaction_returns_provider_overflow_without_retrying() {
    let (result, context, calls) = run_without_compaction(
        "task",
        Replies {
            overflow: true,
            ..Replies::default()
        },
    )
    .await;

    assert_matches!(
        result,
        Err(ReactAgentError::Model(LlmError::ContextOverflow(_)))
    );
    assert_eq!(calls, 1);
    assert_eq!(context.entries().len(), 2);
}

#[tokio::test]
async fn disabled_compaction_stops_when_the_prompt_exhausts_the_window() {
    let (result, context, calls) =
        run_without_compaction(&"x".repeat(16_000), Replies::default()).await;

    assert_matches!(
        result,
        Err(ReactAgentError::Model(LlmError::ContextOverflow(_)))
    );
    assert_eq!(calls, 0);
    assert_eq!(context.entries().len(), 2);
}

/// A policy that discards older inputs without running a summary model.
#[derive(Debug)]
struct KeepLatestInput {
    threshold: usize,
}

impl CompactorFactory for KeepLatestInput {
    fn configure(
        &self,
        config: CompactorConfig,
        tools: &[ToolDefinition],
        _system_prompt: &str,
    ) -> Result<Box<dyn Compactor>, AgentError> {
        Ok(Box::new(Self {
            threshold: config
                .compaction_threshold
                .unwrap_or(self.threshold)
                .saturating_sub(estimate_tools(tools)),
        }))
    }
}

impl Compactor for KeepLatestInput {
    fn should_compact(&self, entries: &[AgentMessage]) -> bool {
        prompt_size(entries, 0) >= self.threshold
    }

    fn prepare(&self, _entries: &[AgentMessage]) -> (Vec<AgentMessage>, Option<CompletionRequest>) {
        (Vec::new(), None)
    }

    fn finish(
        &self,
        entries: &[AgentMessage],
        completion: Option<Completion>,
    ) -> Result<Compaction, AgentError> {
        assert!(
            completion.is_none(),
            "retention must not run summary inference"
        );
        Ok(Compaction {
            recent: entries
                .iter()
                .rev()
                .find(|entry| matches!(entry.chat_message(), Some(chat_record::Message::User(_))))
                .cloned()
                .into_iter()
                .collect(),
            ..Default::default()
        })
    }
}

#[tokio::test]
async fn late_custom_compaction_retains_selected_context_before_the_completion_boundary() {
    let mut store = InMemoryStore::default();
    let mut history = vec![
        window_entry(),
        user_entry(&"x".repeat(14_400), "old"),
        user_entry("continue", "latest"),
    ];
    store
        .append("session", 0, &mut history)
        .await
        .expect("seed history");
    let entries = store.load("session", false).await.expect("load history");
    assert!(!KeepLatestInput { threshold: 3_700 }.should_compact(&entries));
    assert_matches!(
        completion_budget(prompt_size(&entries, 0), 4_000, 64_000),
        None
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let tools = DirectToolExecutor::new(Vec::new());
    let mut agent = ReactAgent::new(
        ScriptedModel {
            replies: vec!["done"],
            calls: calls.clone(),
            truncated: false,
            overflow: false,
        },
        test_provider(4_000).await,
        None,
        KeepLatestInput { threshold: 3_700 },
        &tools,
        SystemPromptConfig {
            custom_prompt: Some("refreshed".into()),
            ..Default::default()
        },
        ReactAgentConfig {
            compaction_threshold: Some(3_700),
            ..Default::default()
        },
    )
    .expect("agent");
    let (mut context, _events) =
        ReactContext::new("session", entries, Box::new(store), execution_context());
    agent
        .initialize(&mut context, &tools)
        .await
        .expect("initialize");

    agent
        .run(&mut context, &tools)
        .await
        .expect("run with injected compaction");

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "only the action should sample"
    );
    let resident = context.entries().to_vec();
    assert_matches!(resident[0].payload.as_ref(),
        Some(agent_message::Payload::ContextWindow(window))
            if window.system_prompt == "refreshed"
                && window.compaction.as_ref().expect("compaction").summary.is_none()
                && window.compaction.as_ref().expect("compaction").recent.len() == 1
                && window.compaction.as_ref().expect("compaction").model.is_empty()
                && window.compaction.as_ref().expect("compaction").inference.is_none()
                && window.compaction.as_ref().expect("compaction").usage.is_none());
    let (stored, _) = context.history(0, 100).await.expect("durable history");
    assert_eq!(&stored[..history.len()], history.as_slice());
    assert_eq!(stored[history.len()..], resident[..]);
    assert_matches!(&active_messages(&resident)[1], chat_record::Message::User(user) if user.content == "continue");
    assert_eq!(active_messages(&stored), active_messages(&resident));
}

#[derive(Debug)]
struct UnavailableCompaction;

impl CompactorFactory for UnavailableCompaction {
    fn configure(
        &self,
        _config: CompactorConfig,
        _tools: &[ToolDefinition],
        _system_prompt: &str,
    ) -> Result<Box<dyn Compactor>, AgentError> {
        Err(AgentError::InvalidConfiguration(
            "strategy is unavailable".into(),
        ))
    }
}

#[tokio::test]
async fn disabled_compaction_does_not_configure_an_injected_strategy() {
    let calls = Arc::new(AtomicUsize::new(0));
    let tools = DirectToolExecutor::new(Vec::new());
    let mut agent = agent_without_compaction(
        "system",
        Replies::default(),
        calls.clone(),
        &tools,
        UnavailableCompaction,
    )
    .await;
    let (mut context, _events) = ReactContext::new(
        "session",
        Vec::new(),
        Box::new(InMemoryStore::default()),
        execution_context(),
    );

    agent
        .initialize(&mut context, &tools)
        .await
        .expect("initialize without a strategy");
    agent
        .run(&mut context, &tools)
        .await
        .expect("run without a strategy");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_matches!(agent.compact_now(&mut context, &tools).await,
        Err(ReactAgentError::Agent(AgentError::InvalidInput(message))) if message == "compaction is disabled");
}

#[tokio::test(start_paused = true)]
async fn custom_compaction_that_cannot_reclaim_room_stops_at_the_exception_limit() {
    let mut store = InMemoryStore::default();
    let mut history = vec![window_entry(), user_entry(&"x".repeat(14_400), "latest")];
    store
        .append("session", 0, &mut history)
        .await
        .expect("seed history");
    let calls = Arc::new(AtomicUsize::new(0));
    let tools = DirectToolExecutor::new(Vec::new());
    let mut agent = ReactAgent::new(
        ScriptedModel {
            replies: Vec::new(),
            calls: calls.clone(),
            truncated: false,
            overflow: false,
        },
        test_provider(4_000).await,
        None,
        KeepLatestInput { threshold: 3_700 },
        &tools,
        SystemPromptConfig {
            custom_prompt: Some("system".into()),
            ..Default::default()
        },
        ReactAgentConfig {
            max_exceptions: 1,
            ..Default::default()
        },
    )
    .expect("agent");
    let (mut context, _events) =
        ReactContext::new("session", history, Box::new(store), execution_context());

    agent
        .run(&mut context, &tools)
        .await
        .expect("bounded failure");

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (stored, _) = context.history(0, 100).await.expect("history");
    assert_eq!(
        stored
            .iter()
            .filter(|entry| matches!(
                entry.payload,
                Some(agent_message::Payload::ContextWindow(_))
            ))
            .count(),
        2
    );
    assert!(stored.iter().any(|entry| matches!(&entry.payload,
        Some(agent_message::Payload::Log(log)) if log.text.contains("leaves no room under boundary="))));
    assert_matches!(stored.last().and_then(AgentMessage::chat_message),
        Some(chat_record::Message::Assistant(assistant))
            if assistant.content.as_deref() == Some("Reached the maximum of 1 exceptions without completing the task."));
}
