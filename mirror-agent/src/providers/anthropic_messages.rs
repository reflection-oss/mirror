//! Streaming Anthropic Messages model provider.

use std::collections::BTreeMap;

use crate::providers::Completion;
use crate::providers::CompletionRequest;
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures_core::Stream;
use futures_util::StreamExt;
use mirror_protocol::chat_record;
use mirror_protocol::AssistantMessage;
use mirror_protocol::FunctionCall;
use mirror_protocol::StreamContent;
use mirror_protocol::StreamReasoning;
use mirror_protocol::StreamToolCall;
use mirror_protocol::ToolCall;
use mirror_protocol::Usage;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use serde_json::Map;
use serde_json::Value;

use super::apply_extra_body;
use super::responses::json_to_struct;
use super::responses::struct_to_json;
use super::Llm;
use super::LlmError;
use super::ModelRequestGuard;
use super::StreamEvent;

/// The Messages API version pinned in the `anthropic-version` header.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// A streaming model backed by the Anthropic Messages endpoint.
#[derive(Debug)]
pub struct AnthropicMessages {
    config: AnthropicMessagesConfig,
    client: Client,
}

/// Configuration for an Anthropic Messages model.
#[derive(Debug, Clone)]
pub struct AnthropicMessagesConfig {
    pub model: String,
    pub api_key: String,
    pub base_url: String,
}

impl AnthropicMessages {
    /// Create a provider with a reusable HTTP client.
    pub fn new(config: AnthropicMessagesConfig, client: Client) -> Self {
        Self { config, client }
    }

    fn request_body(&self, request: CompletionRequest) -> Result<Value, LlmError> {
        let effort = request
            .inference
            .as_ref()
            .and_then(|inference| inference.reasoning_effort.clone())
            .filter(|effort| !effort.is_empty());
        let system = request
            .messages
            .iter()
            .filter_map(|message| match message {
                chat_record::Message::System(message) => Some(message.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let messages = request
            .messages
            .iter()
            .filter(|message| !matches!(message, chat_record::Message::System(_)))
            .map(serialize_message)
            .collect::<Result<Vec<_>, _>>()?;
        // Anthropic requires an explicit completion ceiling on every request.
        let max_tokens = request.max_tokens.unwrap_or(4096);
        let mut body = Map::from_iter([
            ("model".to_string(), json!(self.config.model)),
            ("max_tokens".to_string(), json!(max_tokens)),
            ("stream".to_string(), Value::Bool(true)),
            ("messages".to_string(), Value::Array(messages)),
        ]);
        // A breakpoint on the last stable block (tools render before system, so
        // the system block ends the shared prefix) caches tools plus system.
        let mut tools = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                })
            })
            .collect::<Vec<_>>();
        if !system.is_empty() {
            body.insert(
                "system".to_string(),
                json!([{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]),
            );
        } else if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
        if !tools.is_empty() {
            body.insert("tools".to_string(), Value::Array(tools));
        }
        if let Some(effort) = effort.filter(|effort| effort != "none") {
            // `display: summarized` restores readable reasoning, which the API
            // otherwise omits by default on the current Opus and Fable models.
            body.insert(
                "thinking".to_string(),
                json!({"type": "adaptive", "display": "summarized"}),
            );
            body.insert(
                "output_config".to_string(),
                json!({"effort": map_effort(&effort)}),
            );
        }
        apply_extra_body(&mut body, request.inference.as_ref())?;
        Ok(Value::Object(body))
    }
}

impl Llm for AnthropicMessages {
    fn model(&self) -> &str {
        &self.config.model
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> {
        self.stream_guarded(request, None)
    }
}

impl AnthropicMessages {
    pub(super) fn stream_guarded<'a>(
        &'a self,
        request: CompletionRequest,
        guard: Option<&'a dyn ModelRequestGuard>,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        let client = self.client.clone();
        let endpoint = format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'));
        let model = self.config.model.clone();
        let api_key = self.config.api_key.clone();
        let body = self.request_body(request);
        try_stream! {
            let body = body?;
            if let Some(guard) = guard {
                guard.verify().await?;
            }
            let response = client
                .post(&endpoint)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header("x-api-key", &api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body)
                .send()
                .await
                .map_err(|error| request_error(&model, error))?;
            let response = if response.status().is_success() {
                response
            } else {
                let status = response.status();
                let detail = response.text().await.unwrap_or_default();
                Err(response_error(&model, format!("{status}: {detail}")))?
            };

            let mut state = CompletionState::default();
            let mut completed = false;
            let mut events = response.bytes_stream().eventsource();
            while let Some(event) = events.next().await {
                let payload = event
                    .map_err(|error| request_error(&model, error))?
                    .data;
                let event: ApiEvent = serde_json::from_str(&payload)
                    .map_err(|error| request_error(&model, error))?;
                match event {
                    ApiEvent::Error { error } => {
                        Err(response_error(&model, format_api_error(&error)))?;
                    }
                    ApiEvent::MessageStop => {
                        completed = true;
                        break;
                    }
                    event => {
                        for event in state.apply(event) {
                            yield event;
                        }
                    }
                }
            }
            if !completed {
                Err(request_error(
                    &model,
                    "model stream ended without a terminal event",
                ))?;
            }
            yield StreamEvent::Final(state.finish()?);
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiEvent {
    MessageStart {
        message: ApiMessageStart,
    },
    ContentBlockStart {
        index: u32,
        content_block: ApiContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: ApiDelta,
    },
    ContentBlockStop {
        #[allow(dead_code)]
        index: u32,
    },
    MessageDelta {
        delta: ApiMessageDelta,
        usage: Option<ApiUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: ApiError,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct ApiMessageStart {
    #[serde(default)]
    usage: ApiUsage,
}

#[derive(Debug, Default, Deserialize)]
struct ApiUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        #[serde(default)]
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiDelta {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct ApiMessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
}

#[derive(Debug)]
enum BlockState {
    Text(String),
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        arguments: String,
    },
}

#[derive(Debug, Default)]
struct CompletionState {
    blocks: BTreeMap<u32, BlockState>,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    cache_write_tokens: u64,
    stop_reason: Option<String>,
}

impl CompletionState {
    fn apply(&mut self, event: ApiEvent) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        match event {
            ApiEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;
                self.cached_tokens = message.usage.cache_read_input_tokens;
                self.cache_write_tokens = message.usage.cache_creation_input_tokens;
                self.output_tokens = message.usage.output_tokens;
                if self.output_tokens > 0 {
                    events.push(StreamEvent::OutputTokens(self.output_tokens));
                }
            }
            ApiEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ApiContentBlock::Text { text } => {
                    if !text.is_empty() {
                        events.push(StreamEvent::Content(StreamContent {
                            content: text.clone(),
                        }));
                    }
                    self.blocks.insert(index, BlockState::Text(text));
                }
                ApiContentBlock::Thinking {
                    thinking,
                    signature,
                } => {
                    if !thinking.is_empty() {
                        events.push(StreamEvent::Reasoning(StreamReasoning {
                            reasoning: thinking.clone(),
                        }));
                    }
                    self.blocks.insert(
                        index,
                        BlockState::Thinking {
                            thinking,
                            signature,
                        },
                    );
                }
                ApiContentBlock::RedactedThinking { data } => {
                    self.blocks
                        .insert(index, BlockState::RedactedThinking { data });
                }
                ApiContentBlock::ToolUse { id, name } => {
                    events.push(StreamEvent::ToolCall(StreamToolCall {
                        index,
                        name: name.clone(),
                        arguments: String::new(),
                    }));
                    self.blocks.insert(
                        index,
                        BlockState::ToolUse {
                            id,
                            name,
                            arguments: String::new(),
                        },
                    );
                }
                ApiContentBlock::Unknown => {}
            },
            ApiEvent::ContentBlockDelta { index, delta } => {
                match (self.blocks.get_mut(&index), delta) {
                    (Some(BlockState::Text(content)), ApiDelta::TextDelta { text }) => {
                        content.push_str(&text);
                        events.push(StreamEvent::Content(StreamContent { content: text }));
                    }
                    (
                        Some(BlockState::Thinking { thinking, .. }),
                        ApiDelta::ThinkingDelta { thinking: delta },
                    ) => {
                        thinking.push_str(&delta);
                        events.push(StreamEvent::Reasoning(StreamReasoning { reasoning: delta }));
                    }
                    (
                        Some(BlockState::Thinking { signature, .. }),
                        ApiDelta::SignatureDelta { signature: delta },
                    ) => {
                        signature.push_str(&delta);
                    }
                    (
                        Some(BlockState::ToolUse { arguments, .. }),
                        ApiDelta::InputJsonDelta { partial_json },
                    ) => {
                        arguments.push_str(&partial_json);
                        events.push(StreamEvent::ToolCall(StreamToolCall {
                            index,
                            name: String::new(),
                            arguments: partial_json,
                        }));
                    }
                    _ => {}
                }
            }
            ApiEvent::MessageDelta { delta, usage } => {
                if let Some(stop_reason) = delta.stop_reason {
                    self.stop_reason = Some(stop_reason);
                }
                if let Some(usage) = usage {
                    self.output_tokens = usage.output_tokens;
                    events.push(StreamEvent::OutputTokens(self.output_tokens));
                }
            }
            ApiEvent::ContentBlockStop { .. }
            | ApiEvent::Ping
            | ApiEvent::MessageStop
            | ApiEvent::Error { .. }
            | ApiEvent::Unknown => {}
        }
        events
    }

    fn finish(mut self) -> Result<Completion, LlmError> {
        let truncated = match self.stop_reason.as_deref() {
            Some("end_turn" | "tool_use" | "stop_sequence") => false,
            Some("max_tokens") => {
                self.drop_incomplete_tool_call();
                true
            }
            Some("refusal") => {
                return Err(LlmError::Request(
                    "model completion was declined by a safety classifier".to_string(),
                ));
            }
            Some(reason) => {
                return Err(LlmError::Request(format!(
                    "unsupported model stop reason: {reason}"
                )));
            }
            None => {
                return Err(LlmError::Request(
                    "model completion ended without a stop reason".to_string(),
                ));
            }
        };
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut reasoning_details = Vec::new();
        let mut tool_calls = Vec::new();
        for block in self.blocks.into_values() {
            match block {
                BlockState::Text(text) => content.push_str(&text),
                BlockState::Thinking {
                    thinking,
                    signature,
                } => {
                    reasoning.push_str(&thinking);
                    reasoning_details.push(json_to_struct(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": signature,
                    }))?);
                }
                BlockState::RedactedThinking { data } => {
                    reasoning_details.push(json_to_struct(json!({
                        "type": "redacted_thinking",
                        "data": data,
                    }))?);
                }
                BlockState::ToolUse {
                    id,
                    name,
                    arguments,
                } => tool_calls.push(ToolCall {
                    id,
                    r#type: "function".to_string(),
                    function: Some(FunctionCall { name, arguments }),
                    provider_fields: None,
                }),
            }
        }
        if tool_calls.iter().any(|call| {
            call.id.is_empty()
                || call
                    .function
                    .as_ref()
                    .is_none_or(|function| function.name.is_empty())
        }) {
            return Err(LlmError::Request(
                "model completion produced an incomplete tool call".to_string(),
            ));
        }
        Ok(Completion {
            message: AssistantMessage {
                content: (!content.is_empty()).then_some(content),
                tool_calls,
                reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                reasoning_details,
                refusal: None,
                provider_fields: None,
            },
            usage: Some(Usage {
                prompt_tokens: self.input_tokens,
                completion_tokens: self.output_tokens,
                cached_tokens: self.cached_tokens,
                cache_write_tokens: self.cache_write_tokens,
            }),
            truncated,
        })
    }

    fn drop_incomplete_tool_call(&mut self) {
        let Some((&index, BlockState::ToolUse { id, arguments, .. })) =
            self.blocks.last_key_value()
        else {
            return;
        };
        let complete = !id.is_empty() && serde_json::from_str::<Value>(arguments).is_ok();
        if !complete {
            self.blocks.remove(&index);
        }
    }
}

fn serialize_message(message: &chat_record::Message) -> Result<Value, LlmError> {
    match message {
        // System messages are hoisted into the top-level `system` field.
        chat_record::Message::System(_) => Err(LlmError::Request(
            "system messages must be hoisted before serialization".to_string(),
        )),
        chat_record::Message::User(message) => {
            Ok(json!({"role": "user", "content": message.content}))
        }
        chat_record::Message::Tool(message) => Ok(json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }],
        })),
        chat_record::Message::Assistant(message) => serialize_assistant(message),
    }
}

fn serialize_assistant(message: &AssistantMessage) -> Result<Value, LlmError> {
    let mut content = Vec::new();
    // Thinking blocks must precede text and tool_use, and replay verbatim so
    // their signatures still validate on the same model.
    for detail in &message.reasoning_details {
        let value = struct_to_json(detail);
        if matches!(
            value.get("type").and_then(Value::as_str),
            Some("thinking" | "redacted_thinking")
        ) {
            content.push(value);
        }
    }
    if let Some(text) = &message.content {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    for call in &message.tool_calls {
        let function = call
            .function
            .as_ref()
            .ok_or_else(|| LlmError::Request(format!("tool call {:?} has no function", call.id)))?;
        let input = if function.arguments.is_empty() {
            json!({})
        } else {
            serde_json::from_str(&function.arguments).map_err(|error| {
                LlmError::Request(format!("invalid tool call arguments: {error}"))
            })?
        };
        content.push(json!({
            "type": "tool_use",
            "id": call.id,
            "name": function.name,
            "input": input,
        }));
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    Ok(json!({"role": "assistant", "content": content}))
}

fn map_effort(effort: &str) -> &str {
    // The CLI's `minimal` tier has no Messages API equivalent; `low` is closest.
    match effort {
        "minimal" => "low",
        effort => effort,
    }
}

fn format_api_error(error: &ApiError) -> String {
    let message = error
        .message
        .clone()
        .unwrap_or_else(|| "model request failed".to_string());
    match &error.kind {
        Some(kind) => format!("{message} ({kind})"),
        None => message,
    }
}

fn request_error(model: &str, error: impl std::fmt::Display) -> LlmError {
    LlmError::Request(format!("{error} [model={model}]"))
}

fn response_error(model: &str, detail: String) -> LlmError {
    if detail.contains("prompt is too long") || detail.contains("exceeds the maximum") {
        LlmError::ContextOverflow(detail)
    } else {
        request_error(model, detail)
    }
}

#[cfg(test)]
mod tests {
    use crate::tools::ToolDefinition;
    use assert_matches::assert_matches;
    use mirror_protocol::InferenceConfig;
    use mirror_protocol::SystemMessage;
    use mirror_protocol::ToolMessage;
    use mirror_protocol::UserMessage;
    use pretty_assertions::assert_eq;

    use super::*;

    fn provider() -> AnthropicMessages {
        AnthropicMessages::new(
            AnthropicMessagesConfig {
                model: "claude-opus-4-8".to_string(),
                api_key: "test-key".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
            },
            super::super::http_client(None).expect("test HTTP client configuration is valid"),
        )
    }

    #[test]
    fn request_body_uses_messages_shapes() {
        let request = CompletionRequest {
            messages: vec![
                chat_record::Message::System(SystemMessage {
                    content: "be helpful".to_string(),
                }),
                chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    content: "hello".to_string(),
                }),
            ],
            tools: vec![ToolDefinition {
                name: "bash".to_string(),
                description: "Run a command.".to_string(),
                parameters: Map::from_iter([("type".to_string(), json!("object"))]),
            }],
            max_tokens: Some(1024),
            inference: Some(InferenceConfig {
                model: "claude-opus-4-8".to_string(),
                api: None,
                base_url: None,
                reasoning_effort: Some("high".to_string()),
                ..Default::default()
            }),
        };

        let body = provider()
            .request_body(request)
            .expect("valid messages serialize");

        assert_eq!(
            body,
            json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "stream": true,
                "messages": [{"role": "user", "content": "hello"}],
                "system": [{
                    "type": "text",
                    "text": "be helpful",
                    "cache_control": {"type": "ephemeral"},
                }],
                "tools": [{
                    "name": "bash",
                    "description": "Run a command.",
                    "input_schema": {"type": "object"},
                }],
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "high"},
            })
        );
    }

    #[test]
    fn request_body_omits_thinking_when_effort_is_none() {
        let request = CompletionRequest {
            messages: vec![chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "hi".to_string(),
            })],
            tools: Vec::new(),
            max_tokens: None,
            inference: Some(InferenceConfig {
                model: "claude-opus-4-8".to_string(),
                api: None,
                base_url: None,
                reasoning_effort: Some("none".to_string()),
                ..Default::default()
            }),
        };

        let body = provider()
            .request_body(request)
            .expect("valid messages serialize");

        assert_eq!(body.get("thinking"), None);
        assert_eq!(body["max_tokens"], json!(4096));
    }

    #[test]
    fn tool_results_serialize_as_user_tool_result_blocks() {
        let message = chat_record::Message::Tool(ToolMessage {
            tool_call_id: "toolu_1".to_string(),
            content: "ok".to_string(),
            name: Some("bash".to_string()),
        });

        let value = serialize_message(&message).expect("tool message serializes");

        assert_eq!(
            value,
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "ok",
                }],
            })
        );
    }

    #[test]
    fn stream_state_assembles_thinking_text_and_tool_calls() {
        let events = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash"}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"cmd\":\"ls\"}"}}),
            json!({"type": "content_block_stop", "index": 2}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
        ];
        let mut state = CompletionState::default();

        for event in events {
            state.apply(serde_json::from_value(event).expect("valid event"));
        }
        let completion = state.finish().expect("stream completes");

        assert_eq!(completion.message.content.as_deref(), Some("Hello"));
        assert_eq!(
            completion.message.reasoning_content.as_deref(),
            Some("plan")
        );
        assert_eq!(completion.message.tool_calls[0].id, "toolu_1");
        assert_eq!(
            completion.message.tool_calls[0]
                .function
                .as_ref()
                .expect("tool call has a function")
                .arguments,
            "{\"cmd\":\"ls\"}"
        );
        assert_eq!(
            completion
                .message
                .reasoning_details
                .iter()
                .map(struct_to_json)
                .collect::<Vec<_>>(),
            vec![json!({"type": "thinking", "thinking": "plan", "signature": "sig"})]
        );
        let usage = completion.usage.expect("usage is present");
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 7);
        assert!(!completion.truncated);
    }

    #[test]
    fn assistant_replays_thinking_before_tool_use() {
        let message = AssistantMessage {
            content: Some("done".to_string()),
            tool_calls: vec![ToolCall {
                id: "toolu_1".to_string(),
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: "bash".to_string(),
                    arguments: "{\"cmd\":\"ls\"}".to_string(),
                }),
                provider_fields: None,
            }],
            reasoning_content: Some("plan".to_string()),
            reasoning_details: vec![json_to_struct(json!({
                "type": "thinking",
                "thinking": "plan",
                "signature": "sig",
            }))
            .expect("thinking detail")],
            refusal: None,
            provider_fields: None,
        };

        let value = serialize_assistant(&message).expect("assistant serializes");

        assert_eq!(
            value,
            json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": "sig"},
                    {"type": "text", "text": "done"},
                    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}},
                ],
            })
        );
    }

    #[test]
    fn context_overflow_is_classified() {
        let error = response_error(
            "claude-opus-4-8",
            "prompt is too long: 1200000 tokens > 1000000 maximum".to_string(),
        );

        assert_matches!(error, LlmError::ContextOverflow(_));
    }
}
