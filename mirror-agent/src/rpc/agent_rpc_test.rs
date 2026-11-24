use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

use crate::compaction::SummaryCompactorFactory;
use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::react_agent_service::tests::test_provider;
use crate::tools::ToolDefinition;
use assert_matches::assert_matches;
use async_stream::stream;
use futures_util::future::BoxFuture;
use mirror_protocol::agent_message;
use mirror_protocol::agent_rpc_server::AgentRpc as _;
use mirror_protocol::chat_record;
use mirror_protocol::client_message;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::ClientMessage;
use mirror_protocol::ExtensionRecord;
use mirror_protocol::FunctionCall;
use mirror_protocol::HitlResponse;
use mirror_protocol::ImageContent;
use mirror_protocol::InferenceConfig;
use mirror_protocol::SetApprovalMode;
use mirror_protocol::ToolCall;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use super::*;
use crate::local::SQLiteStore;
use crate::providers::Llm;
use crate::providers::LlmError;
use crate::providers::StreamEvent;
use crate::react::InMemoryStore;
use crate::react::ReactAgent;
use crate::react::ReactAgentConfig;
use crate::system_prompt::SystemPromptConfig;
use crate::tools::AnyTool;
use crate::tools::DirectToolExecutor;
use crate::ExecutionContext;
use crate::ReactAgentService;
use crate::ReactServiceError;

struct StableModel;

fn system_prompt() -> SystemPromptConfig {
    SystemPromptConfig {
        custom_prompt: Some("system".to_string()),
        ..SystemPromptConfig::default()
    }
}

fn inference() -> InferenceConfig {
    InferenceConfig {
        model: "gpt-test".to_string(),
        api: Some("openai-responses".to_string()),
        base_url: Some("https://example.test/v1".to_string()),
        reasoning_effort: Some("high".to_string()),
        ..Default::default()
    }
}

impl Llm for StableModel {
    fn model(&self) -> &str {
        "stable"
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        stream! {
            yield Ok(StreamEvent::Final(Completion {
                message: AssistantMessage {
                    content: Some("done".to_string()),
                    ..AssistantMessage::default()
                },
                usage: None,
                truncated: false,
            }));
        }
    }
}

struct PendingModel {
    started: mpsc::UnboundedSender<Vec<ToolDefinition>>,
}

impl Llm for PendingModel {
    fn model(&self) -> &str {
        "pending"
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> {
        let _ = self.started.send(request.tools);
        futures_util::stream::pending()
    }
}

#[tokio::test]
async fn web_provider_controls_apply_while_idle_and_running() {
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let tools = DirectToolExecutor::new(vec![Arc::new(crate::tools::WebSearchTool::new(
        Box::new(|selection| {
            Ok(Box::new(
                crate::tools::web_test_provider::MemoryWebProvider(selection),
            ))
        }),
    ))])
    .with_tool_visibility(|_, context| {
        context.web_provider() == Some(crate::web::WebProviderConfig::Tavily)
    });
    let agent = ReactAgent::new(
        PendingModel { started },
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tools,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    assert_matches!(
        &service.messages().await.expect("history")[0].payload,
        Some(agent_message::Payload::ContextWindow(window))
            if window.system_prompt_report.as_ref().expect("report").tools.is_empty()
    );
    let rpc = AgentRpc::new(service.clone());
    let (requests, receiver) = mpsc::channel(4);
    let mut responses = rpc
        .open_session(ReceiverStream::new(receiver))
        .await
        .expect("session");
    requests
        .send(Ok(web_provider_request("first", "tavily")))
        .await
        .expect("send");
    await_ack(&mut responses, "first").await;
    let first = service
        .call_tool("web_search".to_string(), r#"{"query":"test"}"#.to_string())
        .await
        .expect("idle tool");
    assert_eq!(first.display.as_deref(), Some("Tavily: 1 result(s)"));

    service
        .execute(user_input_message("user".to_string(), "wait".to_string()))
        .expect("execute");
    let visible = started_receiver.recv().await.expect("running");
    assert_eq!(visible.len(), 1);
    assert_eq!(visible[0].name, "web_search");
    requests
        .send(Ok(web_provider_request("next", "firecrawl")))
        .await
        .expect("send");
    await_ack(&mut responses, "next").await;
    let request = CallToolRequest {
        session_id: "session".to_string(),
        name: "web_search".to_string(),
        arguments: r#"{"query":"test"}"#.to_string(),
        web_provider: Some("you".to_string()),
    };
    let direct = rpc
        .call_tool(Request::new(request))
        .await
        .expect("direct call")
        .into_inner();
    assert_eq!(direct.display.as_deref(), Some("You.com: 1 result(s)"));
    let active = service
        .call_tool("web_search".to_string(), r#"{"query":"test"}"#.to_string())
        .await
        .expect("active tool");
    assert_eq!(active.display.as_deref(), Some("Firecrawl: 1 result(s)"));
    service
        .interrupt()
        .expect("interrupt")
        .await
        .expect("interrupted");
    service
        .execute(user_input_message("next".to_string(), "wait".to_string()))
        .expect("execute again");
    assert!(started_receiver
        .recv()
        .await
        .expect("next request")
        .is_empty());
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}

fn web_provider_request(message_id: &str, provider: &str) -> ClientMessage {
    ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: message_id.to_string(),
        payload: Some(client_message::Payload::SetWebProvider(
            mirror_protocol::SetWebProvider {
                provider: provider.to_string(),
            },
        )),
    }
}

#[tokio::test]
async fn call_tool_runs_bash_in_the_session_workspace_without_a_turn() {
    let directory = tempfile::tempdir().expect("workspace");
    let tool_executor = DirectToolExecutor::new(vec![Arc::new(
        crate::tools::BashTool::new(directory.path()).expect("bash"),
    )]);
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    let before = service.messages().await.expect("messages");
    let rpc = AgentRpc::new(service.clone());
    let denied = rpc
        .call_tool(Request::new(CallToolRequest {
            session_id: "another-session".to_string(),
            name: "bash".to_string(),
            arguments: "{}".to_string(),
            ..Default::default()
        }))
        .await
        .expect_err("reject wrong session");
    assert_eq!(denied.code(), tonic::Code::NotFound);
    let output = rpc
        .call_tool(Request::new(CallToolRequest {
            session_id: "session".to_string(),
            name: "bash".to_string(),
            arguments: r#"{"command":"pwd"}"#.to_string(),
            ..Default::default()
        }))
        .await
        .expect("tool result")
        .into_inner();
    assert!(output.success);
    assert!(output.content.contains(
        directory
            .path()
            .canonicalize()
            .expect("canonical workspace")
            .to_str()
            .expect("path")
    ));
    assert_eq!(service.messages().await.expect("messages"), before);
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}

#[tokio::test]
async fn session_attaches_mid_turn_and_rejects_queued_input_before_closing() {
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        PendingModel { started },
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    service
        .execute(user_input_message(
            "message-1".to_string(),
            "hello".to_string(),
        ))
        .expect("start turn");
    started_receiver.recv().await.expect("model request starts");
    let mut previous = HashMap::new();
    let compact = session_request(ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".into(),
        message_id: "busy-compact".into(),
        payload: Some(client_message::Payload::Compact(())),
    })
    .expect("compact request");
    assert_matches!(handled(&service, &mut previous, compact).await.as_slice(),
        [SessionOutput::Nack { reason: SessionNackReason::InvalidRequest, error, .. }]
            if error.contains("while a turn is running"));
    assert!(previous.is_empty());

    let rpc = AgentRpc::new(service.clone());
    let (requests, request_receiver) = mpsc::channel::<Result<ClientMessage, Status>>(1);
    let mut responses = rpc
        .open_session(ReceiverStream::new(request_receiver))
        .await
        .expect("attach while the turn runs");
    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".to_string(),
            message_id: "queued-input".to_string(),
            payload: Some(client_message::Payload::Message(Box::new(
                user_input_message("queued-input".to_string(), "next turn".to_string()),
            ))),
        }))
        .await
        .expect("submit queued input");
    loop {
        match responses.next().await {
            Some(Ok(message))
                if matches!(
                    &message.payload,
                    Some(session_message::Payload::InputAck(InputAck { message_id }))
                        if message_id == "queued-input"
                ) =>
            {
                break
            }
            Some(Ok(_)) => {}
            Some(Err(error)) => panic!("AgentRPC stream failed: {error}"),
            None => panic!("AgentRPC stream ended before acknowledging the queued input"),
        }
    }

    service.abandon().await.expect("abandon service");
    let mut rejected = Vec::new();
    let ended = loop {
        match responses.next().await {
            Some(Ok(SessionMessage {
                payload: Some(session_message::Payload::InputNack(nack)),
                ..
            })) => rejected.push(nack.message_id),
            Some(Ok(_)) => {}
            Some(Err(status)) => break status,
            None => panic!("AgentRPC stream ended without a status"),
        }
    };
    assert_eq!(rejected, vec!["queued-input"]);
    assert_eq!(ended.code(), tonic::Code::Unavailable);
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn invalid_message_returns_validation_nack_without_recording_acceptance() {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    let mut inputs = HashMap::new();
    assert_eq!(
        handled(
            &service,
            &mut inputs,
            SessionRequest {
                session_id: "session".to_string(),
                message_id: "missing-payload".to_string(),
                input: SessionInput::Message(Box::default()),
            }
        )
        .await,
        vec![nack_output(
            "missing-payload".to_string(),
            SessionNackReason::InvalidRequest,
            "service input is not a user or extension message".to_string(),
        )]
    );
    assert!(inputs.is_empty());
    service.close().await.expect("close service");
    runtime.await.expect("service runtime");
    assert_matches!(
        handle_request(
            &service,
            &mut inputs,
            SessionRequest {
                session_id: "session".to_string(),
                message_id: "closed-1".to_string(),
                input: SessionInput::Message(Box::new(user_input_message(
                    "closed-1".to_string(),
                    "hello".to_string()
                ))),
            }
        )
        .await,
        Err(ServiceClosed)
    );
}

#[tokio::test]
async fn image_no_model_could_read_is_nacked_and_never_stored() {
    let (service, runtime) = stable_service().await;
    let mut inputs = HashMap::new();
    let oversized = [b"\x89PNG\r\n\x1a\n".as_slice(), &[0; 4 * 1024 * 1024]].concat();

    assert_eq!(
        handled(
            &service,
            &mut inputs,
            image_request("not-an-image", b"<svg/>".to_vec())
        )
        .await,
        vec![nack_output(
            "not-an-image".to_string(),
            SessionNackReason::InvalidRequest,
            "image \"[Image #1]\" is not a PNG, JPEG, GIF, or WebP".to_string(),
        )]
    );
    assert_matches!(
        handled(&service, &mut inputs, image_request("too-large", oversized))
            .await
            .as_slice(),
        [SessionOutput::Nack { reason: SessionNackReason::InvalidRequest, error, .. }]
            if error.ends_with("an image can be at most 3932160 bytes")
    );

    let stored = service.messages().await.expect("messages");
    assert!(!stored
        .iter()
        .any(|message| message.chat_message().is_some()));
    service.close().await.expect("close service");
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn inference_config_accepts_integer_effort_and_rejects_other_strings() {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    tokio::spawn(runtime);
    let request = |message_id: &str, effort: &str| SessionRequest {
        session_id: "session".to_string(),
        message_id: message_id.to_string(),
        input: SessionInput::SetInferenceConfig(InferenceConfig {
            reasoning_effort: Some(effort.to_string()),
            ..inference()
        }),
    };

    let accepted = handled(&service, &mut HashMap::new(), request("budget", "512")).await;
    assert_matches!(
        accepted.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "budget"
    );
    let rejected = handled(&service, &mut HashMap::new(), request("bogus", "lots")).await;
    assert_matches!(
        rejected.as_slice(),
        [SessionOutput::Nack { message_id, reason: SessionNackReason::InvalidRequest, .. }]
            if message_id == "bogus"
    );
    let toggle = |message_id: &str, api: Option<&str>| SessionRequest {
        session_id: "session".into(),
        message_id: message_id.into(),
        input: SessionInput::SetInferenceConfig(InferenceConfig {
            api: api.map(Into::into),
            reasoning_effort: Some("on".into()),
            ..inference()
        }),
    };
    let local = handled(
        &service,
        &mut HashMap::new(),
        toggle("local", Some("llama_cpp")),
    )
    .await;
    assert_matches!(
        local.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "local"
    );
    // A change that names no protocol keeps the worker's Chat Completions provider.
    let hosted = handled(&service, &mut HashMap::new(), toggle("hosted", None)).await;
    assert_matches!(
        hosted.as_slice(),
        [SessionOutput::Nack { message_id, reason: SessionNackReason::InvalidRequest, .. }]
            if message_id == "hosted"
    );
    service.close().await.expect("close service");
}

#[tokio::test]
async fn request_dispatches_directly_to_agent_service() {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    tokio::spawn(runtime);
    let mut events = service.events();
    let config_outputs = handled(
        &service,
        &mut HashMap::new(),
        SessionRequest {
            session_id: "session".to_string(),
            message_id: "config-1".to_string(),
            input: SessionInput::SetInferenceConfig(inference()),
        },
    )
    .await;
    assert_matches!(
        config_outputs.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "config-1"
    );
    let extension = ExtensionRecord {
        extension: "counter".to_string(),
        kind: "state".to_string(),
        data: None,
        model_message: None,
        survives_compaction: false,
    };
    let extension_request = SessionRequest {
        session_id: "session".to_string(),
        message_id: "extension-1".to_string(),
        input: SessionInput::Message(Box::new(extension_input_message(
            "extension-1".to_string(),
            extension.clone(),
        ))),
    };
    let mut extension_inputs = HashMap::new();
    let extension_outputs =
        handled(&service, &mut extension_inputs, extension_request.clone()).await;
    assert_matches!(
        extension_outputs.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "extension-1"
    );
    assert_matches!(
        handled(&service, &mut extension_inputs, extension_request.clone())
        .await
        .as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "extension-1"
    );
    let conflicting_extension = SessionRequest {
        input: SessionInput::Message(Box::new(extension_input_message(
            "extension-1".to_string(),
            ExtensionRecord {
                kind: "different".to_string(),
                ..extension.clone()
            },
        ))),
        ..extension_request
    };
    assert_matches!(
        handled(&service, &mut extension_inputs, conflicting_extension)
            .await
            .as_slice(),
        [SessionOutput::Nack {
            reason: SessionNackReason::MessageIdConflict,
            ..
        }]
    );

    let model_extension = ExtensionRecord {
        extension: "counter".to_string(),
        kind: "message".to_string(),
        data: None,
        model_message: Some(UserMessage {
            parts: Vec::new(),
            documents: Vec::new(),
            checkpoint: None,
            images: Vec::new(),
            content: "The counter changed".to_string(),
        }),
        survives_compaction: false,
    };
    let model_extension_outputs = handled(
        &service,
        &mut HashMap::new(),
        SessionRequest {
            session_id: "session".to_string(),
            message_id: "extension-2".to_string(),
            input: SessionInput::Message(Box::new(extension_input_message(
                "extension-2".to_string(),
                model_extension.clone(),
            ))),
        },
    )
    .await;
    assert_matches!(
        model_extension_outputs.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "extension-2"
    );
    while !matches!(events.recv().await, Ok(ReactServiceEvent::TurnCompleted)) {}

    let outputs = handled(
        &service,
        &mut HashMap::new(),
        SessionRequest {
            session_id: "session".to_string(),
            message_id: "message-1".to_string(),
            input: SessionInput::Message(Box::new(user_input_message(
                "message-1".to_string(),
                "hello".to_string(),
            ))),
        },
    )
    .await;

    assert_eq!(
        outputs,
        vec![SessionOutput::Ack {
            message_id: "message-1".to_string()
        }]
    );
    while !matches!(events.recv().await, Ok(ReactServiceEvent::TurnCompleted)) {}
    let entries = service.messages().await.expect("messages");
    let entry = entries
        .iter()
        .find(|entry| entry.id == "message-1")
        .expect("user entry");
    assert_matches!(
        entry.payload.as_ref(),
        Some(agent_message::Payload::Chat(chat))
            if matches!(
                chat.message.as_ref(),
                Some(chat_record::Message::User(user)) if user.content == "hello"
            )
    );
    assert_eq!(entry.inference, None);
    let extension_entry = entries
        .iter()
        .find(|entry| entry.id == "extension-1")
        .expect("extension entry");
    assert_matches!(
        extension_entry.payload.as_ref(),
        Some(agent_message::Payload::Extension(record)) if record == &extension
    );
    assert_eq!(extension_entry.to_llm_messages(), Vec::new());
    let model_extension_entry = entries
        .iter()
        .find(|entry| entry.id == "extension-2")
        .expect("model extension entry");
    assert_eq!(
        model_extension_entry.to_llm_messages(),
        vec![chat_record::Message::User(UserMessage {
            parts: Vec::new(),
            documents: Vec::new(),
            checkpoint: None,
            images: Vec::new(),
            content: "The counter changed".to_string(),
        })]
    );
    service.close().await.expect("close");
}

#[tokio::test]
async fn direct_request_stream_serves_agent_rpc() {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    tokio::spawn(runtime);
    let rpc = AgentRpc::new(service.clone());
    let (requests, request_receiver) = mpsc::channel(1);
    let mut responses = rpc
        .open_session(ReceiverStream::new(request_receiver))
        .await
        .expect("open RPC session");

    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".to_string(),
            message_id: "message-1".to_string(),
            payload: Some(client_message::Payload::Message(Box::new(
                user_input_message("message-1".to_string(), "hello".to_string()),
            ))),
        }))
        .await
        .expect("send request");

    let acknowledgement = loop {
        match responses.next().await {
            Some(Ok(message))
                if matches!(
                    message.payload.as_ref(),
                    Some(session_message::Payload::InputAck(InputAck {
                        message_id,
                    })) if message_id == "message-1"
                ) =>
            {
                break message
            }
            Some(Ok(_)) => {}
            Some(Err(error)) => panic!("AgentRPC stream failed: {error}"),
            None => panic!("AgentRPC stream ended before acknowledging the request"),
        }
    };
    assert_matches!(
        acknowledgement.payload,
        Some(session_message::Payload::InputAck(InputAck { message_id }))
            if message_id == "message-1"
    );
    service.close().await.expect("close service");
}

#[tokio::test]
async fn message_ids_are_shared_across_rpc_sessions() {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    tokio::spawn(runtime);
    service.open().await.expect("open service");
    let mut first_session = HashMap::new();
    let mut second_session = HashMap::new();
    let request = SessionRequest {
        session_id: "session".to_string(),
        message_id: "message-1".to_string(),
        input: SessionInput::Message(Box::new(user_input_message(
            "message-1".to_string(),
            "hello".to_string(),
        ))),
    };

    assert_eq!(
        handled(&service, &mut first_session, request.clone()).await,
        vec![SessionOutput::Ack {
            message_id: "message-1".to_string(),
        }]
    );
    assert_eq!(
        handled(&service, &mut second_session, request.clone()).await,
        vec![SessionOutput::Ack {
            message_id: "message-1".to_string(),
        }]
    );
    let conflict = SessionRequest {
        input: SessionInput::Message(Box::new(user_input_message(
            "message-1".to_string(),
            "different".to_string(),
        ))),
        ..request
    };
    assert_matches!(
        handled(&service, &mut HashMap::new(), conflict)
            .await
            .as_slice(),
        [SessionOutput::Nack {
            message_id,
            reason: SessionNackReason::MessageIdConflict,
            ..
        }] if message_id == "message-1"
    );

    wait_for_service_idle(&service).await;
    assert_eq!(
        service
            .messages()
            .await
            .expect("messages")
            .iter()
            .filter(|entry| entry.id == "message-1")
            .count(),
        1
    );
    service.close().await.expect("close service");
}

#[tokio::test]
async fn extension_request_identity_survives_service_reconstruction() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let store = SQLiteStore::open(&database, "session", directory.path(), true, false)
        .expect("create SQLite store");
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
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
        Vec::new(),
    )
    .await
    .expect("service");
    tokio::spawn(runtime);
    service.open().await.expect("open service");
    let mut inputs = HashMap::new();
    let first = SessionRequest {
        session_id: "session".to_string(),
        message_id: "message-1".to_string(),
        input: SessionInput::Message(Box::new(user_input_message(
            "message-1".to_string(),
            "first".to_string(),
        ))),
    };
    handled(&service, &mut inputs, first).await;
    wait_for_service_idle(&service).await;
    let extension = SessionRequest {
        session_id: "session".to_string(),
        message_id: "message-2".to_string(),
        input: SessionInput::Message(Box::new(extension_input_message(
            "message-2".to_string(),
            ExtensionRecord {
                extension: "todo".to_string(),
                kind: "state".to_string(),
                model_message: Some(UserMessage {
                    parts: Vec::new(),
                    documents: Vec::new(),
                    checkpoint: None,
                    images: Vec::new(),
                    content: "continue".to_string(),
                }),
                ..ExtensionRecord::default()
            },
        ))),
    };
    let mut events = service.events();
    assert_eq!(
        handled(&service, &mut inputs, extension.clone()).await,
        vec![SessionOutput::Ack {
            message_id: "message-2".to_string(),
        }]
    );
    loop {
        if matches!(
            events.recv().await.expect("turn event"),
            ReactServiceEvent::TurnCompleted
        ) {
            break;
        }
    }
    wait_for_service_idle(&service).await;
    service.close().await.expect("close service");

    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (reopened, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(
            SQLiteStore::open(&database, "session", directory.path(), false, false)
                .expect("reopen SQLite store"),
        ),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
        Vec::new(),
    )
    .await
    .expect("reopened service");
    tokio::spawn(runtime);
    reopened.open().await.expect("open reconstructed service");
    let mut restored = HashMap::new();

    assert_eq!(
        handled(&reopened, &mut restored, extension).await,
        vec![SessionOutput::Ack {
            message_id: "message-2".to_string(),
        }]
    );
    reopened.close().await.expect("close service");
}

async fn wait_for_service_idle(service: &ReactAgentServiceHandle) {
    loop {
        match service.messages().await {
            Ok(_) => return,
            Err(ReactServiceError::ConversationBusy) => tokio::task::yield_now().await,
            Err(error) => panic!("service failed while waiting for idle: {error}"),
        }
    }
}

#[test]
fn approval_source_is_preserved_and_cannot_claim_server_policy() {
    let mut message = ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: "decision".to_string(),
        payload: Some(client_message::Payload::HitlResponse(HitlResponse {
            call_id: "call".to_string(),
            approved: true,
            source: ApprovalSource::RememberedPrefix.into(),
            reason: None,
        })),
    };
    assert_matches!(
        session_request(message.clone())
            .expect("valid decision")
            .input,
        SessionInput::HitlResponse {
            source: ApprovalSource::RememberedPrefix,
            ..
        }
    );
    if let Some(client_message::Payload::HitlResponse(response)) = &mut message.payload {
        response.source = ApprovalSource::Automatic.into();
    }
    assert!(session_request(message).is_err());
}

#[test]
fn request_rejects_missing_payload() {
    let invalid = session_request(ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: "message-1".to_string(),
        payload: None,
    })
    .expect_err("missing payload");

    assert_eq!(invalid.error, "client message has no payload");
}

#[test]
fn request_decodes_host_tool_result() {
    let request = session_request(ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: "host-result".to_string(),
        payload: Some(client_message::Payload::HostToolResult(
            mirror_protocol::HostToolResult {
                call_id: "call-1".to_string(),
                content: "echoed".to_string(),
                display: Some("echo".to_string()),
                success: true,
            },
        )),
    });

    assert_matches!(
        request,
        Ok(SessionRequest {
            input: SessionInput::RespondToHostTool { call_id, response },
            ..
        }) if call_id == "call-1"
            && response == ToolOutput::success("echoed").with_display("echo")
    );
}

#[test]
fn request_decodes_extension_record() {
    let record = ExtensionRecord {
        extension: "counter".to_string(),
        kind: "state".to_string(),
        data: None,
        model_message: None,
        survives_compaction: false,
    };
    let request = session_request(ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: "extension-1".to_string(),
        payload: Some(client_message::Payload::Message(Box::new(
            extension_input_message("extension-1".to_string(), record.clone()),
        ))),
    });

    assert_matches!(
        request,
        Ok(SessionRequest {
            input: SessionInput::Message(decoded),
            ..
        }) if decoded == Box::new(extension_input_message("extension-1".to_string(), record))
    );
}

#[test]
fn request_decodes_user_message() {
    let request = session_request(ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: "message-1".to_string(),
        payload: Some(client_message::Payload::Message(Box::new(
            user_input_message("message-1".to_string(), "hello".to_string()),
        ))),
    })
    .expect("request");

    assert_eq!(request.message_id, "message-1");
    assert_eq!(
        request.input,
        SessionInput::Message(Box::new(user_input_message(
            "message-1".to_string(),
            "hello".to_string(),
        )))
    );
}

#[test]
fn removed_user_input_field_is_not_reinterpreted_as_an_agent_message() {
    let request = <ClientMessage as prost::Message>::decode(&b"\x1a\x05\x0a\x03old"[..])
        .expect("valid protobuf with a reserved field");
    assert!(request.payload.is_none());
    let invalid = session_request(request).expect_err("removed input is rejected");
    assert_eq!(invalid.error, "client message has no payload");
}

#[test]
fn streaming_usage_preserves_worker_measurements() {
    let usage = mirror_protocol::StreamUsage {
        completion_tokens: 120,
        elapsed_seconds: 3.5,
    };
    let responses = agent_event_responses(AgentEvent::StreamUsage(usage));
    assert_eq!(
        responses,
        vec![runtime_response(runtime_delta::Payload::StreamUsage(usage))]
    );
}

#[test]
fn stream_final_encodes_tokens_per_second() {
    let responses = agent_event_responses(AgentEvent::StreamFinal {
        entry: Box::new(AgentMessage::default()),
        tokens_per_second: Some(42.0),
    });

    assert_matches!(
        responses.as_slice(),
        [SessionMessage {
            stream_final: true,
            tokens_per_second: Some(value),
            ..
        }] if *value == 42.0
    );
}

#[test]
fn request_decodes_inference_control() {
    let request = session_request(ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: "message-1".to_string(),
        payload: Some(client_message::Payload::SetInferenceConfig(inference())),
    })
    .expect("inference control");

    assert_eq!(request.input, SessionInput::SetInferenceConfig(inference()));
}

fn user_input_message(id: String, content: String) -> AgentMessage {
    AgentMessage {
        id,
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                parts: Vec::new(),
                documents: Vec::new(),
                content,
                checkpoint: None,
                images: Vec::new(),
            })),
            success: true,
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    }
}

fn image_request(message_id: &str, data: Vec<u8>) -> SessionRequest {
    SessionRequest {
        session_id: "session".to_string(),
        message_id: message_id.to_string(),
        input: SessionInput::Message(Box::new(AgentMessage {
            id: message_id.to_string(),
            payload: Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(chat_record::Message::User(UserMessage {
                    parts: Vec::new(),
                    documents: Vec::new(),
                    content: "[Image #1] look".to_string(),
                    checkpoint: None,
                    images: vec![ImageContent {
                        name: "[Image #1]".to_string(),
                        data,
                    }],
                })),
                success: true,
                ..ChatRecord::default()
            })),
            ..AgentMessage::default()
        })),
    }
}

fn extension_input_message(id: String, record: ExtensionRecord) -> AgentMessage {
    AgentMessage {
        id,
        payload: Some(agent_message::Payload::Extension(record)),
        ..AgentMessage::default()
    }
}

async fn handled(
    service: &ReactAgentServiceHandle,
    previous_inputs: &mut HashMap<String, SessionInput>,
    request: SessionRequest,
) -> Vec<SessionOutput> {
    handle_request(service, previous_inputs, request)
        .await
        .expect("the service is open")
}

struct HandedService {
    service: Mutex<Option<ReactAgentServiceHandle>>,
    acquired: Arc<AtomicUsize>,
}

impl SessionSource for HandedService {
    fn acquire(
        &self,
        _intent: AcquireIntent,
    ) -> BoxFuture<'_, Result<ReactAgentServiceHandle, Status>> {
        Box::pin(async move {
            self.acquired.fetch_add(1, Ordering::SeqCst);
            self.service
                .lock()
                .expect("source lock")
                .take()
                .ok_or_else(|| Status::unavailable("no replacement service"))
        })
    }

    fn detached(&self) -> BoxFuture<'_, Result<(), Status>> {
        Box::pin(async { Ok(()) })
    }
}

async fn stable_service() -> (ReactAgentServiceHandle, JoinHandle<()>) {
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    (service, tokio::spawn(runtime))
}

fn client_input(message_id: &str, content: &str) -> ClientMessage {
    ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: message_id.to_string(),
        payload: Some(client_message::Payload::Message(Box::new(
            user_input_message(message_id.to_string(), content.to_string()),
        ))),
    }
}

async fn next_message(
    responses: &mut ReceiverStream<Result<SessionMessage, Status>>,
) -> SessionMessage {
    match responses.next().await {
        Some(Ok(message)) => message,
        Some(Err(error)) => panic!("AgentRPC stream failed: {error}"),
        None => panic!("AgentRPC stream ended"),
    }
}

async fn await_ack(
    responses: &mut ReceiverStream<Result<SessionMessage, Status>>,
    message_id: &str,
) {
    loop {
        match next_message(responses).await.payload {
            Some(session_message::Payload::InputAck(ack)) if ack.message_id == message_id => return,
            Some(session_message::Payload::InputNack(nack)) if nack.message_id == message_id => {
                panic!("input {message_id} was rejected: {}", nack.error)
            }
            _ => {}
        }
    }
}

async fn await_turn_completed(responses: &mut ReceiverStream<Result<SessionMessage, Status>>) {
    loop {
        if matches!(
            next_message(responses).await.payload,
            Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(runtime_delta::Payload::TurnCompleted(())),
                ..
            }))
        ) {
            return;
        }
    }
}

#[tokio::test]
async fn idle_interrupt_is_nacked_because_no_turn_is_running() {
    let (service, runtime) = stable_service().await;
    let mut previous_inputs = HashMap::new();

    let outputs = handled(
        &service,
        &mut previous_inputs,
        SessionRequest {
            session_id: "session".to_string(),
            message_id: "interrupt-1".to_string(),
            input: SessionInput::Interrupt,
        },
    )
    .await;

    assert_matches!(
        outputs.as_slice(),
        [SessionOutput::Nack { message_id, reason: SessionNackReason::TargetNotFound, error }]
            if message_id == "interrupt-1" && error == "no turn is running"
    );
    assert!(previous_inputs.is_empty());
    service.close().await.expect("close the service");
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn stream_reacquires_a_service_after_its_actor_closes() {
    let (first, first_runtime) = stable_service().await;
    let (second, second_runtime) = stable_service().await;
    let acquired = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(HandedService {
        service: Mutex::new(Some(second.clone())),
        acquired: acquired.clone(),
    });
    let (requests, request_receiver) = mpsc::channel(4);
    let (shutdown, _) = broadcast::channel(1);
    let mut responses = serve_agent_rpc_session(
        first.clone(),
        source,
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
    )
    .await
    .expect("open stream");

    requests
        .send(Ok(client_input("message-1", "hello")))
        .await
        .expect("first input");
    await_ack(&mut responses, "message-1").await;
    await_turn_completed(&mut responses).await;
    first.close().await.expect("close the first service");
    first_runtime.await.expect("first runtime");

    requests
        .send(Ok(client_input("message-2", "again")))
        .await
        .expect("second input");
    await_ack(&mut responses, "message-2").await;
    await_turn_completed(&mut responses).await;

    assert_eq!(acquired.load(Ordering::SeqCst), 1);
    assert!(second
        .messages()
        .await
        .expect("second history")
        .iter()
        .any(|entry| entry.id == "message-2"));
    second.close().await.expect("close the second service");
    second_runtime.await.expect("second runtime");
}

#[tokio::test]
async fn input_racing_a_closed_actor_is_nacked_before_the_stream_ends() {
    let (service, runtime) = stable_service().await;
    let mut state = SessionState::new(
        AttachedService::subscribe(service.clone()).await,
        Arc::new(service.clone()),
    );
    service.close().await.expect("close the service");
    runtime.await.expect("service runtime");
    let (responses, mut received) = mpsc::channel(4);

    let ended = state
        .handle(client_input("late-1", "hello"), &responses)
        .await
        .expect_err("a closed fixed source ends the stream");

    assert_eq!(ended.code(), tonic::Code::Unavailable);
    drop(responses);
    let mut nacked = Vec::new();
    while let Some(message) = received.recv().await {
        if let Some(session_message::Payload::InputNack(nack)) =
            message.expect("session message").payload
        {
            nacked.push(nack.message_id);
        }
    }
    assert_eq!(nacked, ["late-1"]);
}

struct ListedModels(ReactAgentServiceHandle);

impl SessionSource for ListedModels {
    fn acquire(
        &self,
        _intent: AcquireIntent,
    ) -> BoxFuture<'_, Result<ReactAgentServiceHandle, Status>> {
        Box::pin(async move { Ok(self.0.clone()) })
    }

    fn detached(&self) -> BoxFuture<'_, Result<(), Status>> {
        Box::pin(async { Ok(()) })
    }

    fn select_inference(&self, inference: InferenceConfig) -> Result<InferenceConfig, String> {
        if inference.model == "listed" {
            Ok(inference)
        } else {
            Err(format!("model {:?} is not listed", inference.model))
        }
    }
}

#[tokio::test]
async fn source_refuses_an_inference_selection_without_ending_the_stream() {
    let (service, runtime) = stable_service().await;
    let (requests, request_receiver) = mpsc::channel(4);
    let (shutdown, _) = broadcast::channel(1);
    let mut responses = serve_agent_rpc_session(
        service.clone(),
        Arc::new(ListedModels(service.clone())),
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
    )
    .await
    .expect("open stream");
    let selection = |message_id: &str, model: &str| ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: message_id.to_string(),
        payload: Some(client_message::Payload::SetInferenceConfig(
            InferenceConfig {
                model: model.to_string(),
                ..Default::default()
            },
        )),
    };

    requests
        .send(Ok(selection("unlisted", "other")))
        .await
        .expect("unlisted selection");
    requests
        .send(Ok(selection("listed", "listed")))
        .await
        .expect("listed selection");

    let refused = loop {
        if let Some(session_message::Payload::InputNack(nack)) =
            next_message(&mut responses).await.payload
        {
            break nack;
        }
    };
    assert_eq!(refused.message_id, "unlisted");
    assert_eq!(refused.reason(), InputNackReason::InvalidRequest);
    assert_eq!(refused.error, "model \"other\" is not listed");
    await_ack(&mut responses, "listed").await;
    service.close().await.expect("close service");
    runtime.await.expect("service runtime");
}

struct IntentRecordingSource {
    service: Mutex<Option<ReactAgentServiceHandle>>,
    intents: mpsc::UnboundedSender<AcquireIntent>,
}

impl SessionSource for IntentRecordingSource {
    fn acquire(
        &self,
        intent: AcquireIntent,
    ) -> BoxFuture<'_, Result<ReactAgentServiceHandle, Status>> {
        Box::pin(async move {
            let _ = self.intents.send(intent);
            self.service
                .lock()
                .expect("source lock")
                .take()
                .ok_or_else(|| Status::unavailable("no replacement service"))
        })
    }

    fn detached(&self) -> BoxFuture<'_, Result<(), Status>> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn observing_input_reacquires_for_execution_when_a_message_awaits_replay() {
    let (closing, closing_runtime) = stable_service().await;
    let (replacement, replacement_runtime) = stable_service().await;
    let (intents, mut recorded) = mpsc::unbounded_channel();
    let mut state = SessionState::new(
        AttachedService::subscribe(closing.clone()).await,
        Arc::new(IntentRecordingSource {
            service: Mutex::new(Some(replacement.clone())),
            intents,
        }),
    );
    state.previous_inputs.insert(
        "queued-1".to_string(),
        SessionInput::Message(Box::new(user_input_message(
            "queued-1".to_string(),
            "hello".to_string(),
        ))),
    );
    state.rejected.push("queued-1".to_string());
    closing.close().await.expect("close the service");
    closing_runtime.await.expect("service runtime");
    let (responses, _received) = mpsc::channel(16);

    state
        .handle(
            ClientMessage {
                trace_parent: String::new(),
                personal_context: None,
                session_id: "session".to_string(),
                message_id: "interrupt-1".to_string(),
                payload: Some(client_message::Payload::Interrupt(Default::default())),
            },
            &responses,
        )
        .await
        .expect("the interrupt is handled on the replacement");

    assert_eq!(recorded.recv().await, Some(AcquireIntent::Execute));
    replacement.close().await.expect("close the replacement");
    replacement_runtime.await.expect("replacement runtime");
}

#[tokio::test]
async fn half_closed_stream_nacks_rejections_it_was_holding_for_replay() {
    let (service, runtime) = stable_service().await;
    let mut state = SessionState::new(
        AttachedService::subscribe(service.clone()).await,
        Arc::new(service.clone()),
    );
    state.rejected.push("held-1".to_string());
    let (requests, request_receiver) = mpsc::channel::<Result<ClientMessage, Status>>(1);
    drop(requests);
    let (shutdown, _) = broadcast::channel(1);
    let (responses, mut received) = mpsc::channel(4);

    run_session(
        state,
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
        responses,
    )
    .await;

    let mut nacked = Vec::new();
    while let Some(message) = received.recv().await {
        if let Some(session_message::Payload::InputNack(nack)) =
            message.expect("session message").payload
        {
            nacked.push(nack.message_id);
        }
    }
    assert_eq!(nacked, ["held-1"]);
    service.close().await.expect("close the service");
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn stream_replays_an_input_its_closing_actor_rejected() {
    let (started, mut started_receiver) = mpsc::unbounded_channel();
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        PendingModel { started },
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (first, first_runtime) = ReactAgentService::start(
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
        Vec::new(),
    )
    .await
    .expect("service");
    let first_runtime = tokio::spawn(first_runtime);
    let (second, second_runtime) = stable_service().await;
    let acquired = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(HandedService {
        service: Mutex::new(Some(second.clone())),
        acquired: acquired.clone(),
    });
    let (requests, request_receiver) = mpsc::channel(4);
    let (shutdown, _) = broadcast::channel(1);
    let mut responses = serve_agent_rpc_session(
        first.clone(),
        source,
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
    )
    .await
    .expect("open stream");

    requests
        .send(Ok(client_input("message-1", "hello")))
        .await
        .expect("first input");
    started_receiver
        .recv()
        .await
        .expect("the first turn reaches the model");
    requests
        .send(Ok(client_input("message-2", "queued")))
        .await
        .expect("queued input");
    await_ack(&mut responses, "message-2").await;
    // Acceptance must not wait for the blocked model to finish its turn.
    assert_matches!(
        first.messages().await,
        Err(ReactServiceError::ConversationBusy)
    );

    first.abandon().await.expect("abandon the first service");
    first_runtime.await.expect("first runtime");

    let mut replayed = false;
    loop {
        match next_message(&mut responses).await.payload {
            Some(session_message::Payload::InputNack(nack)) => {
                panic!("input {} was rejected: {}", nack.message_id, nack.error)
            }
            Some(session_message::Payload::AgentMessage(entry)) if entry.id == "message-2" => {
                replayed = true;
            }
            Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(runtime_delta::Payload::TurnCompleted(())),
                ..
            })) if replayed => break,
            _ => {}
        }
    }
    assert_eq!(acquired.load(Ordering::SeqCst), 1);
    assert_matches!(
        second
            .messages()
            .await
            .expect("second history")
            .iter()
            .find(|entry| entry.id == "message-2")
            .and_then(AgentMessage::chat_message),
        Some(chat_record::Message::User(UserMessage { content, .. })) if content == "queued"
    );
    second.close().await.expect("close the second service");
    second_runtime.await.expect("second runtime");
}

struct GatedModel {
    calls: AtomicUsize,
}

impl Llm for GatedModel {
    fn model(&self) -> &str {
        "gated"
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        let message = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call-1".to_string(),
                    r#type: "function".to_string(),
                    function: Some(FunctionCall {
                        name: "gated".to_string(),
                        arguments: "{}".to_string(),
                    }),
                    ..ToolCall::default()
                }],
                ..AssistantMessage::default()
            }
        } else {
            AssistantMessage {
                content: Some("done".to_string()),
                ..AssistantMessage::default()
            }
        };
        stream! {
            yield Ok(StreamEvent::Final(Completion {
                message,
                usage: None,
                truncated: false,
            }));
        }
    }
}

struct GatedTool;

impl AnyTool for GatedTool {
    fn name(&self) -> &str {
        "gated"
    }

    fn needs_approval(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "gated".to_string(),
            description: "Run gated.".to_string(),
            parameters: serde_json::Map::new(),
        }
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async { ToolOutput::success("ran") })
    }
}

#[tokio::test]
async fn a_stream_attached_during_an_approval_is_asked_for_it() {
    let tool_executor = DirectToolExecutor::new(vec![Arc::new(GatedTool)]);
    let agent = ReactAgent::new(
        GatedModel {
            calls: AtomicUsize::new(0),
        },
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(InMemoryStore::default()),
        "session",
        AgentApprovalMode::Hitl,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    let mut events = service.events();
    service
        .execute(user_input_message(
            "message-1".to_string(),
            "run it".to_string(),
        ))
        .expect("input");
    while !matches!(
        events.recv().await.expect("service event"),
        ReactServiceEvent::HitlRequest(_)
    ) {}

    let source = Arc::new(HandedService {
        service: Mutex::new(None),
        acquired: Arc::new(AtomicUsize::new(0)),
    });
    let (requests, request_receiver) = mpsc::channel(4);
    let (shutdown, _) = broadcast::channel(1);
    let mut responses = serve_agent_rpc_session(
        service.clone(),
        source,
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
    )
    .await
    .expect("open stream");
    let first = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        next_message(&mut responses),
    )
    .await
    .expect("the attached stream is told about the pending approval");
    let call_id = match first.payload {
        Some(session_message::Payload::HitlRequest(request)) => {
            request.call.expect("gated call").id
        }
        other => panic!("expected the pending approval first, got {other:?}"),
    };
    assert_eq!(call_id, "call-1");
    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".to_string(),
            message_id: "deny-1".to_string(),
            payload: Some(client_message::Payload::HitlResponse(HitlResponse {
                call_id,
                approved: false,
                source: mirror_protocol::ApprovalSource::Manual.into(),
                reason: Some("This action is outside the task.".to_string()),
            })),
        }))
        .await
        .expect("deny");
    await_turn_completed(&mut responses).await;
    let entries = service.messages().await.expect("messages");
    assert_matches!(
        entries.iter().find_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::Tool(tool)) => Some(tool),
            _ => None,
        }),
        Some(tool) if tool.tool_call_id == "call-1"
            && tool.content == "Tool error: This action is outside the task."
    );
    service.close().await.expect("close the service");
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn reacquired_actor_keeps_the_streams_approval_mode() {
    let (first, first_runtime) = stable_service().await;
    let tool_executor = DirectToolExecutor::new(vec![Arc::new(GatedTool)]);
    let agent = ReactAgent::new(
        GatedModel {
            calls: AtomicUsize::new(0),
        },
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (second, second_runtime) = ReactAgentService::start(
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
        Vec::new(),
    )
    .await
    .expect("service");
    let second_runtime = tokio::spawn(second_runtime);
    let source = Arc::new(HandedService {
        service: Mutex::new(Some(second.clone())),
        acquired: Arc::new(AtomicUsize::new(0)),
    });
    let (requests, request_receiver) = mpsc::channel(4);
    let (shutdown, _) = broadcast::channel(1);
    let mut responses = serve_agent_rpc_session(
        first.clone(),
        source,
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
    )
    .await
    .expect("open stream");

    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".to_string(),
            message_id: "approval-1".to_string(),
            payload: Some(client_message::Payload::SetApprovalMode(SetApprovalMode {
                mode: ApprovalMode::Hitl.into(),
            })),
        }))
        .await
        .expect("approval mode");
    await_ack(&mut responses, "approval-1").await;
    first.close().await.expect("close the first service");
    first_runtime.await.expect("first runtime");

    requests
        .send(Ok(client_input("message-1", "run it")))
        .await
        .expect("input");
    let call_id = loop {
        match next_message(&mut responses).await.payload {
            Some(session_message::Payload::HitlRequest(request)) => {
                break request.call.expect("gated call").id;
            }
            Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(runtime_delta::Payload::TurnCompleted(())),
                ..
            })) => panic!("the gated tool ran without approval"),
            _ => {}
        }
    };
    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".to_string(),
            message_id: "deny-1".to_string(),
            payload: Some(client_message::Payload::HitlResponse(HitlResponse {
                call_id,
                approved: false,
                source: mirror_protocol::ApprovalSource::Manual.into(),
                reason: None,
            })),
        }))
        .await
        .expect("deny");
    await_turn_completed(&mut responses).await;
    second.close().await.expect("close the second service");
    second_runtime.await.expect("second runtime");
}

fn approval_mode_request(message_id: &str, mode: ApprovalMode) -> ClientMessage {
    ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".to_string(),
        message_id: message_id.to_string(),
        payload: Some(client_message::Payload::SetApprovalMode(SetApprovalMode {
            mode: mode.into(),
        })),
    }
}

#[tokio::test]
async fn rejected_approval_mode_is_never_saved() {
    let modes = Arc::new(crate::react_agent_service::tests::RecordedModes::default());
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tool_executor,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(InMemoryStore::default()),
        "session",
        crate::SessionApprovalMode::saved(AgentApprovalMode::Hitl, modes.clone()),
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    let source = Arc::new(HandedService {
        service: Mutex::new(None),
        acquired: Arc::new(AtomicUsize::new(0)),
    });
    let (requests, request_receiver) = mpsc::channel(4);
    let (shutdown, _) = broadcast::channel(1);
    let mut responses = serve_agent_rpc_session(
        service.clone(),
        source,
        ReceiverStream::new(request_receiver),
        shutdown.subscribe(),
    )
    .await
    .expect("open stream");

    requests
        .send(Ok(approval_mode_request("", ApprovalMode::Auto)))
        .await
        .expect("unidentified mode");
    assert_matches!(
        next_message(&mut responses).await.payload,
        Some(session_message::Payload::InputNack(_))
    );
    requests
        .send(Ok(approval_mode_request("mode-1", ApprovalMode::Hitl)))
        .await
        .expect("identified mode");
    await_ack(&mut responses, "mode-1").await;
    assert_eq!(
        *modes.saved.lock().expect("saved modes"),
        [AgentApprovalMode::Hitl]
    );
    service.close().await.expect("close the service");
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn inspect_settings_reports_the_live_selection_and_requests_handoff() {
    let (service, runtime) = stable_service().await;
    service
        .set_inference_config(inference())
        .await
        .expect("select a model");
    service
        .set_approval_mode(AgentApprovalMode::Bypass)
        .await
        .expect("bypass approvals");
    let wire = ClientMessage {
        personal_context: None,
        trace_parent: String::new(),
        session_id: "session".into(),
        message_id: "settings-1".into(),
        payload: Some(client_message::Payload::InspectSettings(())),
    };
    assert_eq!(acquire_intent(&wire), AcquireIntent::Execute);
    let request = session_request(wire).expect("valid settings request");

    assert_eq!(
        handled(&service, &mut HashMap::new(), request).await,
        vec![
            SessionOutput::Settings(SessionSettings {
                inference: Some(inference()),
                approval_mode: ApprovalMode::Bypass.into(),
            }),
            SessionOutput::Ack {
                message_id: "settings-1".into(),
            },
        ]
    );
    service.close().await.expect("close the service");
    runtime.await.expect("service runtime");
}

#[tokio::test]
async fn compact_request_uses_session_controls_and_is_idempotent() {
    let (service, runtime) = stable_service().await;
    let mut events = service.events();
    let wire = ClientMessage {
        trace_parent: String::new(),
        personal_context: None,
        session_id: "session".into(),
        message_id: "compact-1".into(),
        payload: Some(client_message::Payload::Compact(())),
    };
    assert_eq!(acquire_intent(&wire), AcquireIntent::Execute);
    let request = session_request(wire).expect("valid compact request");
    assert_eq!(request.input, SessionInput::Compact);
    let mut previous = HashMap::new();
    assert_matches!(handled(&service, &mut previous, request.clone()).await.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "compact-1");
    tokio::time::timeout(Duration::from_secs(5), async {
        while !matches!(
            events.recv().await.expect("event"),
            ReactServiceEvent::TurnCompleted
        ) {}
    })
    .await
    .expect("compaction completed");
    let before = service.messages().await.expect("history");
    assert_matches!(handled(&service, &mut previous, request).await.as_slice(),
        [SessionOutput::Ack { message_id }] if message_id == "compact-1");
    assert_eq!(service.messages().await.expect("history"), before);
    assert!(!before
        .iter()
        .any(|entry| matches!(entry.chat_message(), Some(chat_record::Message::User(_)))));
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}

#[tokio::test]
async fn mcp_updates_reach_a_running_turn_without_entering_history() {
    use crate::mcp::live_mcp_tools;
    use axum::{routing::post, Json, Router};
    use serde_json::{json, Value};

    let port = portpicker::pick_unused_port().expect("port");
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("listener");
    let listed = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let server_listed = listed.clone();
    let server_release = release.clone();
    let endpoint = tokio::spawn(async move {
        axum::serve(listener, Router::new().route("/mcp", post(move |Json(request): Json<Value>| {
            let listed = server_listed.clone();
            let release = server_release.clone();
            async move {
            let result = match request["method"].as_str() {
                Some("initialize") => json!({"protocolVersion":"2025-11-25", "capabilities":{"tools":{}}, "serverInfo":{"name":"fixture", "version":"1"}}),
                Some("tools/list") => {
                    listed.notify_one();
                    release.notified().await;
                    json!({"tools":[{"name":"echo", "description":"Echo text", "inputSchema":{"type":"object"}}]})
                },
                _ => json!({}),
            };
            Json(json!({"jsonrpc":"2.0", "id":request["id"], "result":result}))
        }}))).await.expect("MCP server");
    });
    let source = Arc::new(LocalMcpDiscovery::new(Vec::new(), false).expect("source"));
    let tools = DirectToolExecutor::new(live_mcp_tools(source.clone()));
    let (started, mut running) = mpsc::unbounded_channel();
    let agent = ReactAgent::new(
        PendingModel { started },
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tools,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    service
        .execute(user_input_message("user".into(), "wait".into()))
        .expect("turn");
    assert_eq!(running.recv().await.expect("running").len(), 3);
    let rpc = AgentRpc::new(service.clone()).with_mcp(source);
    let (requests, receiver) = mpsc::channel(4);
    let mut responses = rpc
        .open_session(ReceiverStream::new(receiver))
        .await
        .expect("session");
    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".into(),
            message_id: "connect".into(),
            payload: Some(client_message::Payload::SetMcpConnections(
                mirror_protocol::McpConnections {
                    connections: vec![mirror_protocol::McpConnection {
                        name: "custom".into(),
                        config: Some(mirror_protocol::McpConnectionConfig {
                            url: Some(format!("http://127.0.0.1:{port}/mcp")),
                            ..Default::default()
                        }),
                        secrets: [("TOKEN".into(), "synthetic-private-input".into())].into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )),
        }))
        .await
        .expect("update");
    await_ack(&mut responses, "connect").await;
    let controls = async {
        listed.notified().await;
        assert_eq!(
            service
                .inspect_tools(false)
                .await
                .expect("local inventory")
                .tools
                .len(),
            3
        );
        assert_matches!(
            service
                .set_tool_enabled(mirror_protocol::SetToolEnabled {
                    name: "mcp_search".into(),
                    enabled: false
                })
                .await,
            Err(crate::ReactServiceError::ConversationBusy)
        );
        assert!(service
            .pending_approvals()
            .await
            .expect("approvals")
            .is_empty());
        service
            .interrupt()
            .expect("interrupt")
            .await
            .expect("interrupted");
        release.notify_one();
    };
    let (inventory, ()) = tokio::join!(service.inspect_tools(true), controls);
    assert_matches!(inventory.expect("inventory").tools.last(), Some(tool) if tool.name == "mcp__custom__echo");
    assert!(!format!("{:?}", service.messages().await.expect("history"))
        .contains("synthetic-private-input"));
    assert_matches!(running.try_recv(), Err(mpsc::error::TryRecvError::Empty));
    rpc.shutdown();
    service.close().await.expect("close");
    runtime.await.expect("runtime");
    endpoint.abort();
}

#[tokio::test]
async fn tool_access_controls_persist_and_reject_direct_invocation_without_a_turn() {
    let workspace = tempfile::tempdir().expect("workspace");
    let tools = DirectToolExecutor::new(vec![Arc::new(
        crate::tools::BashTool::new(workspace.path()).expect("bash"),
    )]);
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tools,
        system_prompt(),
        ReactAgentConfig::default(),
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
        Vec::new(),
    )
    .await
    .expect("service");
    let runtime = tokio::spawn(runtime);
    let rpc = AgentRpc::new(service.clone());
    let (requests, receiver) = mpsc::channel(4);
    let mut responses = rpc
        .open_session(ReceiverStream::new(receiver))
        .await
        .expect("session");
    requests
        .send(Ok(ClientMessage {
            trace_parent: String::new(),
            personal_context: None,
            session_id: "session".into(),
            message_id: "disable".into(),
            payload: Some(client_message::Payload::SetToolEnabled(
                mirror_protocol::SetToolEnabled {
                    name: "bash".into(),
                    enabled: false,
                },
            )),
        }))
        .await
        .expect("send");
    await_ack(&mut responses, "disable").await;
    let inventory = service.inspect_tools(false).await.expect("inventory");
    assert!(!inventory.tools[0].enabled);
    let output = rpc
        .call_tool(Request::new(CallToolRequest {
            session_id: "session".into(),
            name: "bash".into(),
            arguments: r#"{"command":"touch forbidden"}"#.into(),
            ..Default::default()
        }))
        .await
        .expect("call")
        .into_inner();
    assert!(!output.success);
    assert!(!workspace.path().join("forbidden").exists());
    let saved = service.messages().await.expect("history");
    assert!(crate::ToolAccess::restore(&saved)
        .expect("restore")
        .disabled
        .contains("bash"));
    assert!(!service.is_busy());
    let tools = DirectToolExecutor::new(vec![Arc::new(
        crate::tools::BashTool::new(workspace.path()).expect("bash"),
    )]);
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        SummaryCompactorFactory,
        &tools,
        system_prompt(),
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (restored, restored_runtime) = ReactAgentService::start(
        agent,
        Box::new(tools),
        Box::new(InMemoryStore::from_entries("session".into(), saved)),
        "session",
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        broadcast::channel(1024).0,
        None,
        Vec::new(),
    )
    .await
    .expect("restored");
    let restored_runtime = tokio::spawn(restored_runtime);
    assert_eq!(
        restored
            .inspect_tools(false)
            .await
            .expect("restored inventory")
            .tools[0]
            .enabled,
        false
    );
    restored.close().await.expect("close restored");
    restored_runtime.await.expect("restored runtime");
    assert_matches!(
        service.accept(crate::ToolAccess::default().record()).await,
        Err(ReactServiceError::Agent(_))
    );
    service
        .set_tool_enabled(mirror_protocol::SetToolEnabled {
            name: "bash".into(),
            enabled: true,
        })
        .await
        .expect("enable");
    assert_eq!(
        service.inspect_tools(false).await.expect("inventory").tools[0].enabled,
        true
    );
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}
