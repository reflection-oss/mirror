use std::path::Path;

use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::react_agent_service::tests::test_provider;
use crate::Store;
use async_stream::stream;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::mirror::mirror_rpc_server::MirrorRpc;
use mirror_protocol::mirror::ForkSessionRequest;
use mirror_protocol::AgentMessage;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::FunctionCall;
use mirror_protocol::ToolCall;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;
use tonic::Request;

use super::*;
use crate::providers::Llm;
use crate::providers::LlmError;
use crate::providers::StreamEvent;
use crate::react::ReactAgent;
use crate::react::ReactAgentConfig;
use crate::system_prompt::SystemPromptConfig;
use crate::tools::DirectToolExecutor;
use crate::AgentApprovalMode;
use crate::ReactAgentService;
use crate::ReactServiceEvent;

struct ToolThenDoneModel;

impl Llm for ToolThenDoneModel {
    fn model(&self) -> &str {
        "tool-then-done"
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        let answered_tool = request
            .messages
            .iter()
            .any(|message| matches!(message, chat_record::Message::Tool(_)));
        stream! {
            let message = if answered_tool {
                AssistantMessage {
                    content: Some("done".to_string()),
                    ..AssistantMessage::default()
                }
            } else {
                AssistantMessage {
                    tool_calls: vec![ToolCall {
                        id: "call-1".to_string(),
                        r#type: "function".to_string(),
                        function: Some(FunctionCall {
                            name: "missing".to_string(),
                            arguments: "{}".to_string(),
                        }),
                        ..ToolCall::default()
                    }],
                    ..AssistantMessage::default()
                }
            };
            yield Ok(StreamEvent::Final(Completion {
                message,
                usage: None,
                truncated: false,
            }));
        }
    }
}

fn user_message(content: &str, id: &str) -> AgentMessage {
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

async fn start_local_rpc(database: &Path, workspace: &Path) -> LocalMirrorRpc {
    let store = SQLiteStore::open(database, "session", workspace, true, false)
        .expect("create SQLite store");
    let tool_executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        ToolThenDoneModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        &tool_executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
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
    )
    .await
    .expect("service");
    tokio::spawn(runtime);
    let mut events = service.events();
    service
        .execute(user_message("hello", "user-1"))
        .expect("execute turn");
    loop {
        if matches!(
            events.recv().await.expect("service event"),
            ReactServiceEvent::TurnCompleted
        ) {
            break;
        }
    }
    LocalMirrorRpc::new(service, database.to_path_buf())
}

fn fork_request(at_entry_id: &str) -> Request<ForkSessionRequest> {
    Request::new(ForkSessionRequest {
        source_session_id: "session".to_string(),
        at_entry_id: at_entry_id.to_string(),
        scope: 0,
    })
}

#[tokio::test]
async fn fork_validates_the_boundary_and_leaves_the_source_running() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let rpc = start_local_rpc(&database, directory.path()).await;
    let entries = rpc.service.messages().await.expect("messages");
    let tool_result = entries
        .iter()
        .find(|entry| matches!(entry.chat_message(), Some(chat_record::Message::Tool(_))))
        .expect("tool result");

    let rejected = rpc
        .fork_session(fork_request(&tool_result.id))
        .await
        .expect_err("a tool result is not a rewind boundary");
    assert_eq!(rejected.code(), tonic::Code::InvalidArgument);
    let missing = rpc
        .fork_session(fork_request("missing"))
        .await
        .expect_err("an unknown entry is not a rewind boundary");
    assert_eq!(missing.code(), tonic::Code::NotFound);
    assert_eq!(rpc.service.messages().await.expect("messages"), entries);
    let forked = rpc
        .fork_session(fork_request("user-1"))
        .await
        .expect("rewind")
        .into_inner()
        .session_id;

    let mut store = SQLiteStore::open(&database, &forked, directory.path(), false, false)
        .expect("open the forked session");
    let view = store
        .load(&forked, true)
        .await
        .expect("load the forked view");
    assert!(view.iter().all(|entry| !matches!(
        entry.payload,
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(_)),
            ..
        }))
    )));
    assert_eq!(
        view.len(),
        rpc.service
            .messages()
            .await
            .expect("source messages")
            .iter()
            .position(|entry| entry.id == "user-1")
            .expect("source keeps the user turn")
    );
    let mut events = rpc.service.events();
    rpc.service
        .execute(user_message("continue", "user-2"))
        .expect("source remains usable");
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !matches!(
            events.recv().await.expect("source event"),
            ReactServiceEvent::TurnCompleted
        ) {}
    })
    .await
    .expect("source completes another turn");
    assert!(store
        .load("session", true)
        .await
        .expect("parent history")
        .iter()
        .any(|entry| entry.id == "user-2"));
    assert_eq!(
        store
            .load(&forked, true)
            .await
            .expect("child prefix stays fixed"),
        view
    );
    rpc.service.close().await.expect("close source");
}

#[tokio::test]
async fn a_contended_fork_keeps_the_async_runtime_responsive() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let rpc = start_local_rpc(&database, directory.path()).await;
    let mut connection = rusqlite::Connection::open(&database).expect("second connection");
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .expect("hold the SQLite write lock");
    let fork = rpc.fork_session(fork_request("user-1"));
    tokio::pin!(fork);
    tokio::select! {
        result = &mut fork => panic!("fork must wait for the transaction: {result:?}"),
        () = tokio::time::sleep(std::time::Duration::from_millis(50)) => (),
    }
    transaction.commit().expect("release the write lock");
    tokio::time::timeout(std::time::Duration::from_secs(5), fork)
        .await
        .expect("fork resumes without blocking timers")
        .expect("fork commits");
    rpc.service.close().await.expect("close source");
}

#[tokio::test]
async fn fork_rejects_public_and_unknown_scope_but_accepts_private() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("react.db");
    let rpc = start_local_rpc(&database, directory.path()).await;

    let mut public = fork_request("user-1");
    public.get_mut().scope = SessionScope::Public.into();
    let rejected = rpc
        .fork_session(public)
        .await
        .expect_err("local sessions cannot be public");
    assert_eq!(rejected.code(), tonic::Code::Unimplemented);

    let mut unknown = fork_request("user-1");
    unknown.get_mut().scope = i32::MAX;
    let invalid = rpc
        .fork_session(unknown)
        .await
        .expect_err("reject unknown scope");
    assert_eq!(invalid.code(), tonic::Code::InvalidArgument);

    let mut private = fork_request("user-1");
    private.get_mut().scope = SessionScope::Private.into();
    let forked = rpc
        .fork_session(private)
        .await
        .expect("fork a private local session")
        .into_inner();
    let mut store = SQLiteStore::open(
        &database,
        &forked.session_id,
        directory.path(),
        false,
        false,
    )
    .expect("open private fork");
    store
        .load(&forked.session_id, true)
        .await
        .expect("private fork was persisted");
    rpc.service.close().await.expect("close source");
}
