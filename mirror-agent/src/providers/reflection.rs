//! Reflection policy composed with the standard Chat Completions implementation.

use std::path::PathBuf;

use futures_core::Stream;
use mirror_protocol::chat_record;
use reqwest::Client;

use super::{
    ChatCompletions, ChatCompletionsConfig, CompletionRequest, Llm, LlmError, ModelProviderConfig,
    ModelRequestGuard, StreamEvent,
};

/// The first-party endpoint; Reflection credentials must never follow custom URLs.
pub const REFLECTION_BASE_URL: &str = "https://api.reflection.ai/openai/v1";

/// Reflection authentication and endpoint policy over Chat Completions.
pub struct Reflection {
    chat_completions: ChatCompletions,
}

impl Reflection {
    pub fn new(model: String, api_key: String, client: Client) -> Self {
        let chat_config = ChatCompletionsConfig {
            model,
            api_key,
            base_url: REFLECTION_BASE_URL.into(),
        };
        Self {
            chat_completions: ChatCompletions::new(chat_config, client),
        }
    }
}

impl Reflection {
    pub(super) fn stream_guarded<'a>(
        &'a self,
        mut request: CompletionRequest,
        guard: Option<&'a dyn ModelRequestGuard>,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        // Reflection rejects other providers' structured reasoning in replayed history.
        for message in &mut request.messages {
            if let chat_record::Message::Assistant(message) = message {
                message.reasoning_details.clear();
            }
        }
        self.chat_completions.stream_guarded(request, guard, true)
    }
}

impl Llm for Reflection {
    fn model(&self) -> &str {
        self.chat_completions.model()
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + Send {
        self.stream_guarded(request, None)
    }
}

pub(super) async fn reflection_key(config: &ModelProviderConfig) -> Result<String, LlmError> {
    if config
        .base_url
        .as_deref()
        .is_some_and(|url| url.trim_end_matches('/') != REFLECTION_BASE_URL)
    {
        return Err(LlmError::Request("reflection does not accept a custom base URL; use openai-chat-completions for other endpoints".into()));
    }
    let mut key = config
        .api_key
        .clone()
        .or_else(|| std::env::var("REFLECTION_API_KEY").ok())
        .filter(|key| !key.trim().is_empty());
    if key.is_none() {
        if let Some(home) = std::env::var_os("HOME") {
            match tokio::fs::read_to_string(PathBuf::from(home).join(".mirror/reflection_api_key"))
                .await
            {
                Ok(saved) => key = Some(saved.trim().to_string()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => {
                    return Err(LlmError::Request(
                        "could not read saved Reflection API key".into(),
                    ))
                }
            }
        }
    }
    key.filter(|key| !key.is_empty()).ok_or_else(|| LlmError::Request("run mirror --api reflection or set REFLECTION_API_KEY; create a key at https://platform.reflection.ai/api-keys".into()))
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use axum::{routing::post, Json, Router};
    use futures_util::TryStreamExt;
    use mirror_protocol::{
        AssistantMessage, FunctionCall, SystemMessage, ToolCall, ToolMessage, UserMessage,
    };
    use pretty_assertions::assert_eq;
    use serde_json::{json, Value};
    use tokio::{net::TcpListener, sync::oneshot};

    use super::*;
    use crate::providers::responses::json_to_struct;

    #[tokio::test]
    async fn stream_replays_cross_provider_history_without_reasoning_details() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener");
        let address = listener.local_addr().expect("listener address");
        let router = Router::new().route("/chat/completions", post(|Json(body): Json<Value>| async move {
            assert_eq!(body["messages"], json!([
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant", "content": "Checking the file",
                    "reasoning_content": "I should inspect it",
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"README.md\"}"}}]
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "File contents"},
                {"role": "user", "content": "hi"}
            ]));
            ([ ("content-type", "text/event-stream") ], concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            ))
        }));
        let (shutdown, stopped) = oneshot::channel();
        let server = axum::serve(listener, router).with_graceful_shutdown(async {
            let _ = stopped.await;
        });
        let model = Reflection {
            chat_completions: ChatCompletions::new(
                ChatCompletionsConfig {
                    model: "test-model".into(),
                    api_key: "synthetic-test-key".into(),
                    base_url: format!("http://{address}"),
                },
                crate::providers::http_client(None).expect("test HTTP client"),
            ),
        };
        let request = CompletionRequest {
            messages: vec![
                chat_record::Message::System(SystemMessage {
                    content: "Be helpful".into(),
                }),
                chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    content: "Read the file".into(),
                }),
                chat_record::Message::Assistant(AssistantMessage {
                    content: Some("Checking the file".into()),
                    reasoning_content: Some("I should inspect it".into()),
                    reasoning_details: vec![json_to_struct(
                        json!({"type": "reasoning.encrypted", "data": "synthetic-provider-state"}),
                    )
                    .expect("reasoning detail")],
                    tool_calls: vec![ToolCall {
                        id: "call-1".into(),
                        r#type: "function".into(),
                        function: Some(FunctionCall {
                            name: "read".into(),
                            arguments: r#"{"path":"README.md"}"#.into(),
                        }),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                chat_record::Message::Tool(ToolMessage {
                    tool_call_id: "call-1".into(),
                    content: "File contents".into(),
                    name: Some("read".into()),
                }),
                chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    content: "hi".into(),
                }),
            ],
            tools: vec![],
            max_tokens: None,
            inference: None,
        };
        let client = async {
            let events = model.stream(request).try_collect::<Vec<_>>().await;
            let _ = shutdown.send(());
            events.expect("Reflection accepts replayed history")
        };
        let (served, events) = tokio::join!(async { server.await }, client);
        served.expect("test server completed");
        assert_matches!(events.as_slice(), [StreamEvent::Content(_), StreamEvent::Final(completion)] if completion.message.content.as_deref() == Some("Hello"));
    }
}
