use std::time::Duration;

use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::react_agent_service::tests::test_provider;
use assert_matches::assert_matches;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::client_message;
use mirror_protocol::runtime_delta;
use mirror_protocol::session_message;
use mirror_protocol::AgentMessage;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::InputNackReason;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;

use super::*;
use crate::providers::{Llm, LlmError, StreamEvent};
use crate::react::{InMemoryStore, ReactAgent, ReactAgentConfig};
use crate::system_prompt::SystemPromptConfig;
use crate::tools::DirectToolExecutor;
use crate::{AgentApprovalMode, ReactAgentService};

struct StableModel;

impl Llm for StableModel {
    fn model(&self) -> &str {
        "stable"
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        futures_util::stream::iter([Ok(StreamEvent::Final(Completion {
            message: AssistantMessage {
                content: Some("done".to_string()),
                ..AssistantMessage::default()
            },
            usage: None,
            truncated: false,
        }))])
    }
}

async fn service() -> ReactAgentServiceHandle {
    let executor = DirectToolExecutor::new(Vec::new());
    let agent = ReactAgent::new(
        StableModel,
        test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW).await,
        None,
        &executor,
        SystemPromptConfig {
            custom_prompt: Some("system".to_string()),
            ..SystemPromptConfig::default()
        },
        ReactAgentConfig::default(),
    )
    .expect("agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(executor),
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
    tokio::spawn(runtime);
    service
}

#[tokio::test]
async fn half_closed_input_drains_the_completed_turn() {
    let service = service().await;
    let rpc = AgentInProcessRpc::new(service.clone());
    let mut session = rpc.connect().await.expect("connect");
    session
        .send(ClientMessage {
            session_id: "session".to_string(),
            message_id: "message-1".to_string(),
            payload: Some(client_message::Payload::Message(Box::new(AgentMessage {
                id: "message-1".to_string(),
                payload: Some(agent_message::Payload::Chat(ChatRecord {
                    message: Some(chat_record::Message::User(UserMessage {
                        checkpoint: None,
                        content: "hello".to_string(),
                    })),
                    ..ChatRecord::default()
                })),
                ..AgentMessage::default()
            }))),
        })
        .await
        .expect("send");
    session.close_input();
    let answer = tokio::time::timeout(Duration::from_secs(5), async {
        let mut answer = None;
        while let Some(message) = session.recv().await.expect("receive") {
            match message.payload {
                Some(session_message::Payload::AgentMessage(AgentMessage {
                    payload:
                        Some(agent_message::Payload::Chat(ChatRecord {
                            message: Some(chat_record::Message::Assistant(assistant)),
                            ..
                        })),
                    ..
                })) => answer = assistant.content,
                Some(session_message::Payload::RuntimeDelta(delta))
                    if matches!(
                        delta.payload,
                        Some(runtime_delta::Payload::TurnCompleted(()))
                    ) =>
                {
                    return answer;
                }
                _ => {}
            }
        }
        panic!("session ended before completing the turn");
    })
    .await
    .expect("turn completes");
    assert_eq!(answer.as_deref(), Some("done"));
    assert_matches!(session.send(ClientMessage::default()).await, Err(error)
        if error.code() == tonic::Code::FailedPrecondition);
    service.close().await.expect("close");
}

#[tokio::test]
async fn malformed_input_is_rejected_in_band() {
    let service = service().await;
    let mut session = AgentInProcessRpc::new(service.clone())
        .connect()
        .await
        .expect("connect");
    session
        .send(ClientMessage {
            session_id: "session".to_string(),
            message_id: "missing-payload".to_string(),
            payload: None,
        })
        .await
        .expect("send malformed input");
    let response = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let response = session
                .recv()
                .await
                .expect("receive")
                .expect("open session");
            // Session state events can precede the input response.
            match &response.payload {
                Some(
                    session_message::Payload::AgentMessage(_)
                    | session_message::Payload::RuntimeDelta(_),
                ) => {}
                _ => return response,
            }
        }
    })
    .await
    .expect("input response arrives");
    assert_matches!(response.payload, Some(session_message::Payload::InputNack(nack))
        if nack.message_id == "missing-payload"
            && nack.reason == InputNackReason::InvalidRequest as i32);
    service.close().await.expect("close");
}
