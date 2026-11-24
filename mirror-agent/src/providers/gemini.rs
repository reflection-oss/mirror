use futures_core::Stream;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ToolCall;
use reqwest::Client;
use serde_json::json;
use serde_json::Map;
use serde_json::Value;

use super::chat_completions::ChatWireExtension;
use super::responses::struct_to_json;
use super::ChatCompletions;
use super::ChatCompletionsConfig;
use super::CompletionRequest;
use super::Llm;
use super::LlmError;
use super::ModelApi;
use super::ModelRequestGuard;
use super::StreamEvent;

const THOUGHT_SIGNATURE_FIELD: &str = "google_thought_signature";

pub struct Gemini {
    chat_completions: ChatCompletions,
}

impl Gemini {
    pub fn new(config: ChatCompletionsConfig, client: Client) -> Self {
        Self {
            chat_completions: ChatCompletions::with_extension(config, client, &GeminiWire),
        }
    }

    pub(super) fn stream_guarded<'a>(
        &'a self,
        request: CompletionRequest,
        guard: Option<&'a dyn ModelRequestGuard>,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        self.chat_completions.stream_guarded(request, guard, true)
    }
}

impl Llm for Gemini {
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

#[derive(Debug)]
struct GeminiWire;

impl ChatWireExtension for GeminiWire {
    fn replay_assistant(&self, message: &AssistantMessage, wire: &mut Map<String, Value>) {
        wire.remove("reasoning_content");
        wire.remove("reasoning_details");
        replay_thought_signature(wire, message.provider_fields.as_ref());
    }

    fn replay_tool_call(&self, call: &ToolCall, wire: &mut Map<String, Value>) {
        if let Some(arguments) = wire
            .get_mut("function")
            .and_then(|function| function.get_mut("arguments"))
            .filter(|arguments| {
                arguments
                    .as_str()
                    .is_some_and(|text| text.trim().is_empty())
            })
        {
            *arguments = json!("{}");
        }
        replay_thought_signature(wire, call.provider_fields.as_ref());
    }

    fn replay_tool_result(&self, wire: &mut Map<String, Value>) {
        let Some(content) = wire.get_mut("content") else {
            return;
        };
        let references_a_part = content
            .as_str()
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .is_some_and(|parsed| has_reference_key(&parsed));
        if references_a_part {
            *content = json!(json!({"output": content.take()}).to_string());
        }
    }

    fn merge_extra_content(
        &self,
        provider_fields: &mut Map<String, Value>,
        extra_content: Value,
    ) -> Result<(), LlmError> {
        let Some(signature) = extra_content
            .get("google")
            .and_then(|google| google.get("thought_signature"))
            .and_then(Value::as_str)
        else {
            return Ok(());
        };
        if provider_fields
            .get(THOUGHT_SIGNATURE_FIELD)
            .is_some_and(|previous| previous.as_str() != Some(signature))
        {
            // Signatures are opaque complete values, not text deltas.
            return Err(LlmError::Request(
                "conflicting streamed thought signatures".into(),
            ));
        }
        provider_fields.insert(THOUGHT_SIGNATURE_FIELD.into(), json!(signature));
        Ok(())
    }
}

fn has_reference_key(value: &Value) -> bool {
    match value {
        Value::Object(fields) => {
            fields.contains_key("$ref") || fields.values().any(has_reference_key)
        }
        Value::Array(items) => items.iter().any(has_reference_key),
        _ => false,
    }
}

fn replay_thought_signature(wire: &mut Map<String, Value>, fields: Option<&prost_types::Struct>) {
    if let Some(signature) = fields
        .map(struct_to_json)
        .and_then(|mut fields| fields.as_object_mut()?.remove(THOUGHT_SIGNATURE_FIELD))
    {
        wire.insert(
            "extra_content".into(),
            json!({"google": {"thought_signature": signature}}),
        );
    }
}

pub(super) fn validate_developer_api_request(request: &CompletionRequest) -> Result<(), LlmError> {
    if request
        .max_tokens
        .is_some_and(|tokens| tokens == 0 || tokens > 65_536)
    {
        return Err(LlmError::Request(
            "Gemini 4 Argon supports 1..65536 output tokens per stream; long decode is not supported by the OpenAI-compatible API".into(),
        ));
    }
    if let Some(inference) = &request.inference {
        ModelApi::Gemini.validate_sampling(inference)?;
        if inference
            .reasoning_effort
            .as_deref()
            .is_some_and(|effort| !matches!(effort, "none" | "low" | "medium" | "high"))
        {
            return Err(LlmError::Request(
                "Gemini 4 Argon reasoning effort must be none (provider default), low, medium, or high".into(),
            ));
        }
        if inference
            .extra_body
            .as_deref()
            .is_some_and(|extra| !extra.trim().is_empty())
        {
            return Err(LlmError::Request(
                "gemini does not support extra_body overrides; use openai-chat-completions for custom requests".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use axum::routing::post;
    use axum::Json;
    use axum::Router;
    use futures_util::TryStreamExt;
    use mirror_protocol::chat_record;
    use mirror_protocol::FunctionCall;
    use mirror_protocol::InferenceConfig;
    use mirror_protocol::ToolMessage;
    use mirror_protocol::UserMessage;
    use pretty_assertions::assert_eq;
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio::sync::oneshot;

    use super::*;
    use crate::providers::responses::json_to_struct;

    async fn exchange(
        request: CompletionRequest,
        response: &'static str,
    ) -> (Value, Result<Vec<StreamEvent>, LlmError>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener");
        let address = listener.local_addr().expect("listener address");
        let (sent, mut received) = mpsc::channel(1);
        let router = Router::new().route(
            "/chat/completions",
            post(move |Json(body): Json<Value>| {
                let sent = sent.clone();
                async move {
                    sent.send(body).await.expect("capture request");
                    ([("content-type", "text/event-stream")], response)
                }
            }),
        );
        let (shutdown, stopped) = oneshot::channel();
        let server = axum::serve(listener, router).with_graceful_shutdown(async {
            let _ = stopped.await;
        });
        let model = Gemini::new(
            ChatCompletionsConfig {
                model: "google/gemini-4-neon".into(),
                api_key: "synthetic-test-key".into(),
                base_url: format!("http://{address}"),
            },
            crate::providers::http_client(None).expect("test HTTP client"),
        );
        let client = async {
            let events = model.stream(request).try_collect::<Vec<_>>().await;
            let body = received.recv().await.expect("captured request");
            shutdown.send(()).expect("stop test server");
            (body, events)
        };
        let (served, exchanged) = tokio::join!(async { server.await }, client);
        served.expect("test server exits");
        exchanged
    }

    fn request(messages: Vec<chat_record::Message>) -> CompletionRequest {
        CompletionRequest {
            messages,
            tools: vec![],
            max_tokens: None,
            inference: None,
        }
    }

    fn user(content: &str) -> chat_record::Message {
        chat_record::Message::User(UserMessage {
            content: content.into(),
            ..Default::default()
        })
    }

    const DONE: &str = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );

    #[tokio::test]
    async fn stream_keeps_thought_signatures_from_the_response() {
        let (_, events) = exchange(
            request(vec![user("list files")]),
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Listing\",",
                "\"extra_content\":{\"google\":{\"thought_signature\":\"message-signature\"}},",
                "\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{}\"},",
                "\"extra_content\":{\"google\":{\"thought_signature\":\"tool-signature\"}}}]},",
                "\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n",
            ),
        )
        .await;

        let events = events.expect("stream completes");
        let completion =
            assert_matches!(events.last(), Some(StreamEvent::Final(completion)) => completion);
        assert_eq!(
            completion
                .message
                .provider_fields
                .as_ref()
                .map(struct_to_json),
            Some(json!({"google_thought_signature": "message-signature"}))
        );
        assert_eq!(
            completion.message.tool_calls[0]
                .provider_fields
                .as_ref()
                .map(struct_to_json),
            Some(json!({"google_thought_signature": "tool-signature"}))
        );
    }

    #[tokio::test]
    async fn stream_rejects_conflicting_thought_signatures() {
        let (_, events) = exchange(
            request(vec![user("hi")]),
            concat!(
                "data: {\"choices\":[{\"delta\":{\"extra_content\":{\"google\":{\"thought_signature\":\"first\"}}}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"extra_content\":{\"google\":{\"thought_signature\":\"second\"}}},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            ),
        )
        .await;

        assert_matches!(events, Err(LlmError::Request(error)) if error.contains("conflicting streamed thought signatures"));
    }

    #[tokio::test]
    async fn stream_replays_history_in_gemini_shape() {
        let signature = |value: &str| {
            Some(
                json_to_struct(json!({"google_thought_signature": value}))
                    .expect("provider fields"),
            )
        };
        let assistant = chat_record::Message::Assistant(AssistantMessage {
            content: Some("Checking".into()),
            reasoning_content: Some("plan".into()),
            provider_fields: signature("message-signature"),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                r#type: "function".into(),
                function: Some(FunctionCall {
                    name: "session_info".into(),
                    arguments: String::new(),
                }),
                provider_fields: signature("tool-signature"),
            }],
            ..Default::default()
        });

        let (body, events) = exchange(request(vec![user("hi"), assistant]), DONE).await;

        events.expect("stream completes");
        assert_eq!(
            body["messages"][1],
            json!({
                "role": "assistant",
                "content": "Checking",
                "extra_content": {"google": {"thought_signature": "message-signature"}},
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "session_info", "arguments": "{}"},
                    "extra_content": {"google": {"thought_signature": "tool-signature"}},
                }],
            })
        );
    }

    #[tokio::test]
    async fn stream_wraps_tool_results_that_hold_reference_keys() {
        let tool = |id: &str, content: &str| {
            chat_record::Message::Tool(ToolMessage {
                tool_call_id: id.into(),
                content: content.into(),
                ..Default::default()
            })
        };
        let schema = r##"{"tools":[{"parameters":{"$ref":"#/$defs/Condition"}}]}"##;

        let (body, events) = exchange(
            request(vec![
                user("hi"),
                tool("call_1", schema),
                tool("call_2", r#"{"titles":["a"]}"#),
                tool("call_3", "costs $ref dollars"),
            ]),
            DONE,
        )
        .await;

        events.expect("stream completes");
        assert_eq!(
            body["messages"][1],
            json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json!({"output": schema}).to_string(),
            })
        );
        assert_eq!(body["messages"][2]["content"], json!(r#"{"titles":["a"]}"#));
        assert_eq!(body["messages"][3]["content"], json!("costs $ref dollars"));
    }

    #[test]
    fn developer_api_rejects_unsupported_controls_before_inference() {
        let mut request = CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: Some(65_536),
            inference: Some(InferenceConfig {
                reasoning_effort: Some("medium".into()),
                ..Default::default()
            }),
        };
        assert_matches!(validate_developer_api_request(&request), Ok(()));
        request.max_tokens = Some(65_537);
        assert_matches!(validate_developer_api_request(&request), Err(LlmError::Request(error)) if error.contains("long decode"));
        request.max_tokens = None;
        request.inference = Some(InferenceConfig {
            reasoning_effort: Some("xhigh".into()),
            ..Default::default()
        });
        assert_matches!(validate_developer_api_request(&request), Err(LlmError::Request(error)) if error.contains("reasoning effort"));
        request.inference = Some(InferenceConfig {
            temperature: Some(0.5),
            ..Default::default()
        });
        assert_matches!(
            validate_developer_api_request(&request),
            Err(LlmError::Request(_))
        );
        request.inference = Some(InferenceConfig {
            extra_body: Some("{}".into()),
            ..Default::default()
        });
        assert_matches!(validate_developer_api_request(&request), Err(LlmError::Request(error)) if error.contains("extra_body"));
    }
}
