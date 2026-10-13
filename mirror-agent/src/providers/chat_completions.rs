//! OpenAI Chat Completions model provider with SSE and JSON transports.

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
use mirror_protocol::StreamRefusal;
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
use super::ModelApi;
use super::ModelRequestGuard;
use super::StreamEvent;

/// A streaming model backed by an OpenAI-compatible Chat Completions endpoint.
#[derive(Debug)]
pub struct ChatCompletions {
    config: ChatCompletionsConfig,
    client: Client,
}

/// Configuration for an OpenAI-compatible Chat Completions model.
#[derive(Clone)]
pub struct ChatCompletionsConfig {
    pub model: String,
    pub api_key: String,
    pub base_url: String,
}

impl std::fmt::Debug for ChatCompletionsConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatCompletionsConfig")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl ChatCompletions {
    /// Create a provider with a reusable HTTP client.
    pub fn new(config: ChatCompletionsConfig, client: Client) -> Self {
        Self { config, client }
    }

    fn request_body(&self, request: CompletionRequest) -> Result<Value, LlmError> {
        let messages = request
            .messages
            .iter()
            .map(serialize_message)
            .collect::<Result<Vec<_>, _>>()?;
        let mut body = Map::from_iter([
            ("model".to_string(), json!(self.config.model)),
            ("messages".to_string(), Value::Array(messages)),
            ("stream".to_string(), Value::Bool(true)),
            ("stream_options".to_string(), json!({"include_usage": true})),
        ]);
        if let Some(max_tokens) = request.max_tokens {
            body.insert("max_tokens".to_string(), json!(max_tokens));
        }
        if let Some(inference) = &request.inference {
            ModelApi::ChatCompletions.validate_sampling(inference)?;
            if let Some(value) = inference.temperature {
                body.insert("temperature".to_string(), json!(value));
            }
            if let Some(value) = inference.top_p {
                body.insert("top_p".to_string(), json!(value));
            }
            if let Some(value) = inference.seed {
                body.insert("seed".to_string(), json!(value));
            }
            if let Some(effort) = inference
                .reasoning_effort
                .as_deref()
                .filter(|effort| *effort != "none")
            {
                let (key, value) = self.reasoning_effort_field(effort);
                body.insert(key.to_string(), value);
            }
        }
        if !request.tools.is_empty() {
            body.insert(
                "tools".to_string(),
                Value::Array(
                    request
                        .tools
                        .into_iter()
                        .map(|tool| {
                            json!({
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.parameters,
                                },
                            })
                        })
                        .collect(),
                ),
            );
        }
        if self.is_openrouter() {
            body.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            body.insert("usage".to_string(), json!({"include": true}));
        }
        apply_extra_body(&mut body, request.inference.as_ref())?;
        Ok(Value::Object(body))
    }

    fn is_openrouter(&self) -> bool {
        self.config.base_url.contains("openrouter.ai")
    }

    /// The wire field carrying a reasoning effort on this endpoint's dialect.
    ///
    /// OpenRouter takes a `reasoning` object: a level name as `effort`, a
    /// token count as `max_tokens`. Every other Chat Completions endpoint
    /// takes `reasoning_effort`: a level name, or an integer budget on servers
    /// that accept one.
    fn reasoning_effort_field(&self, effort: &str) -> (&'static str, Value) {
        match (self.is_openrouter(), effort.parse::<u64>()) {
            (true, Ok(tokens)) => ("reasoning", json!({"max_tokens": tokens})),
            (true, Err(_)) => ("reasoning", json!({"effort": effort})),
            (false, Ok(tokens)) => ("reasoning_effort", json!(tokens)),
            (false, Err(_)) => ("reasoning_effort", json!(effort)),
        }
    }
}

impl Llm for ChatCompletions {
    fn model(&self) -> &str {
        &self.config.model
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> {
        self.stream_guarded(request, None, true)
    }
}

impl ChatCompletions {
    pub(super) fn stream_guarded<'a>(
        &'a self,
        request: CompletionRequest,
        guard: Option<&'a dyn ModelRequestGuard>,
        streaming: bool,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        let client = self.client.clone();
        let endpoint = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let model = self.config.model.clone();
        let api_key = self.config.api_key.clone();
        let body = self.request_body(request);
        try_stream! {
            let mut body = body?;
            if !streaming {
                body["stream"] = json!(false);
                body.as_object_mut()
                    .expect("request_body constructs a JSON object")
                    .remove("stream_options");
            }
            let response = loop {
                if let Some(guard) = guard {
                    guard.verify().await?;
                }
                let response = client
                    .post(&endpoint)
                    .header(reqwest::header::ACCEPT, if streaming { "text/event-stream" } else { "application/json" })
                    .bearer_auth(&api_key)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|error| request_error(&model, error))?;
                if response.status().is_success() {
                    break response;
                }
                let status = response.status();
                let detail = response.text().await.unwrap_or_default();
                // Some endpoints reject this optional extension without naming it.
                if matches!(status.as_u16(), 400 | 422) {
                    if let Some(options) = body.get_mut("stream_options").and_then(Value::as_object_mut) {
                        if options.remove("continuous_usage_stats").is_some() {
                            continue;
                        }
                    }
                }
                Err(response_error(&model, format!("{status}: {detail}")))?;
            };

            if !streaming {
                let completion: ApiCompletion = response.json().await
                    .map_err(|error| request_error(&model, error))?;
                yield StreamEvent::Final(completion.finish(&model)?);
                return;
            }

            let mut state = CompletionState::default();
            let mut completed = false;
            let mut events = response.bytes_stream().eventsource();
            while let Some(event) = events.next().await {
                let payload = event
                    .map_err(|error| request_error(&model, error))?
                    .data;
                if payload == "[DONE]" {
                    completed = true;
                    break;
                }
                let chunk: ApiChunk = serde_json::from_str(&payload)
                    .map_err(|error| request_error(&model, error))?;
                if let Some(error) = &chunk.error {
                    Err(response_error(&model, format_api_error(error)))?;
                }
                for event in state.apply(chunk)? {
                    yield event;
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

#[derive(Debug, Default, Deserialize)]
struct ApiChunk {
    #[serde(default)]
    choices: Vec<ApiChoice>,
    usage: Option<ApiUsage>,
    error: Option<ApiError>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiChoice {
    #[serde(default)]
    delta: ApiDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiDelta<Call = ApiToolCallFragment> {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    refusal: Option<String>,
    reasoning_details: Option<Vec<Value>>,
    tool_calls: Option<Vec<Call>>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiToolCallFragment {
    index: u32,
    id: Option<String>,
    function: Option<ApiFunctionFragment>,
}

#[derive(Debug, Deserialize)]
struct ApiFunctionFragment {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiCompletion {
    #[serde(default)]
    choices: Vec<ApiCompletedChoice>,
    usage: Option<ApiUsage>,
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct ApiCompletedChoice {
    message: ApiDelta<ApiToolCall>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiToolCall {
    id: String,
    function: ApiFunctionFragment,
}

impl ApiCompletion {
    fn finish(self, model: &str) -> Result<Completion, LlmError> {
        if let Some(error) = self.error {
            return Err(response_error(model, format_api_error(&error)));
        }
        let choice = self
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| request_error(model, "model completion has no choices"))?;
        let message = choice.message;
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(index, call)| {
                let index = u32::try_from(index).map_err(|error| request_error(model, error))?;
                Ok((
                    index,
                    ToolCallState {
                        id: call.id,
                        name: call.function.name.unwrap_or_default(),
                        arguments: call.function.arguments.unwrap_or_default(),
                    },
                ))
            })
            .collect::<Result<_, LlmError>>()?;
        CompletionState {
            content: message.content.unwrap_or_default(),
            reasoning: message
                .reasoning_content
                .or(message.reasoning)
                .unwrap_or_default(),
            refusal: message.refusal.unwrap_or_default(),
            reasoning_details: message.reasoning_details.unwrap_or_default(),
            tool_calls,
            usage: self.usage,
            finish_reason: choice.finish_reason,
        }
        .finish()
    }
}

#[derive(Debug, Default, Deserialize)]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    prompt_tokens_details: Option<ApiUsageDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiUsageDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: Option<String>,
    code: Option<Value>,
}

#[derive(Debug, Default)]
struct CompletionState {
    content: String,
    reasoning: String,
    refusal: String,
    reasoning_details: Vec<Value>,
    tool_calls: BTreeMap<u32, ToolCallState>,
    usage: Option<ApiUsage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
}

impl CompletionState {
    fn apply(&mut self, chunk: ApiChunk) -> Result<Vec<StreamEvent>, LlmError> {
        let mut events = Vec::new();
        if let Some(usage) = chunk.usage {
            events.push(StreamEvent::OutputTokens(usage.completion_tokens));
            self.usage = Some(usage);
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
            return Ok(events);
        };
        self.finish_reason = choice.finish_reason.or(self.finish_reason.take());
        let delta = choice.delta;
        let reasoning = delta.reasoning_content.or(delta.reasoning);
        if let Some(reasoning) = reasoning.filter(|value| !value.is_empty()) {
            self.reasoning.push_str(&reasoning);
            events.push(StreamEvent::Reasoning(StreamReasoning { reasoning }));
        }
        if let Some(content) = delta.content.filter(|value| !value.is_empty()) {
            self.content.push_str(&content);
            events.push(StreamEvent::Content(StreamContent { content }));
        }
        if let Some(refusal) = delta.refusal.filter(|value| !value.is_empty()) {
            self.refusal.push_str(&refusal);
            events.push(StreamEvent::Refusal(StreamRefusal { refusal }));
        }
        for detail in delta.reasoning_details.unwrap_or_default() {
            let position = detail
                .get("index")
                .filter(|index| !index.is_null())
                .and_then(|index| {
                    let kind = detail.get("type");
                    self.reasoning_details.iter().position(|prior| {
                        prior.get("type") == kind && prior.get("index") == Some(index)
                    })
                });
            if let Some(position) = position {
                merge_reasoning_detail(&mut self.reasoning_details[position], detail);
            } else {
                self.reasoning_details.push(detail);
            }
        }
        for fragment in delta.tool_calls.unwrap_or_default() {
            let state = self.tool_calls.entry(fragment.index).or_default();
            if let Some(id) = fragment.id {
                state.id.push_str(&id);
            }
            let mut name = String::new();
            let mut arguments = String::new();
            if let Some(function) = fragment.function {
                name = function.name.unwrap_or_default();
                arguments = function.arguments.unwrap_or_default();
                state.name.push_str(&name);
                state.arguments.push_str(&arguments);
            }
            events.push(StreamEvent::ToolCall(StreamToolCall {
                index: fragment.index,
                name,
                arguments,
            }));
        }
        Ok(events)
    }

    fn finish(mut self) -> Result<Completion, LlmError> {
        let truncated = match self.finish_reason.as_deref() {
            Some("stop" | "tool_calls") => false,
            Some("length") => {
                self.drop_incomplete_tool_call();
                true
            }
            Some("content_filter") if !self.refusal.is_empty() => false,
            Some("content_filter") => {
                return Err(LlmError::Request(
                    "model completion was blocked by a content filter".to_string(),
                ));
            }
            Some(reason) => {
                return Err(LlmError::Request(format!(
                    "unsupported model finish reason: {reason}"
                )));
            }
            None => {
                return Err(LlmError::Request(
                    "model completion ended without a finish reason".to_string(),
                ));
            }
        };
        let tool_calls = self
            .tool_calls
            .into_values()
            .map(|call| ToolCall {
                id: call.id,
                r#type: "function".to_string(),
                function: Some(FunctionCall {
                    name: call.name,
                    arguments: call.arguments,
                }),
                provider_fields: None,
            })
            .collect::<Vec<_>>();
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
        let reasoning_details = self
            .reasoning_details
            .into_iter()
            .map(json_to_struct)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Completion {
            message: AssistantMessage {
                content: (!self.content.is_empty()).then_some(self.content),
                tool_calls,
                reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
                reasoning_details,
                refusal: (!self.refusal.is_empty()).then_some(self.refusal),
                provider_fields: None,
            },
            usage: self.usage.map(api_usage),
            truncated,
        })
    }

    fn drop_incomplete_tool_call(&mut self) {
        let Some((index, call)) = self.tool_calls.last_key_value() else {
            return;
        };
        let complete = !call.id.is_empty()
            && !call.name.is_empty()
            && serde_json::from_str::<Value>(&call.arguments).is_ok();
        if !complete {
            let index = *index;
            self.tool_calls.remove(&index);
        }
    }
}

fn merge_reasoning_detail(prior: &mut Value, detail: Value) {
    let (Some(prior), Value::Object(detail)) = (prior.as_object_mut(), detail) else {
        return;
    };
    for field in ["text", "summary", "data"] {
        let Some(incoming) = detail.get(field) else {
            continue;
        };
        let previous = prior.get(field).and_then(Value::as_str).unwrap_or_default();
        let incoming = incoming.as_str().unwrap_or_default();
        let merged = if incoming.starts_with(previous) {
            incoming.to_string()
        } else if previous.ends_with(incoming) {
            previous.to_string()
        } else {
            format!("{previous}{incoming}")
        };
        prior.insert(field.to_string(), Value::String(merged));
    }
    for (name, value) in detail {
        if !matches!(name.as_str(), "text" | "summary" | "data") && !value.is_null() {
            prior.insert(name, value);
        }
    }
}

fn serialize_message(message: &chat_record::Message) -> Result<Value, LlmError> {
    match message {
        chat_record::Message::System(message) => {
            Ok(json!({"role": "system", "content": message.content}))
        }
        chat_record::Message::User(message) => {
            Ok(json!({"role": "user", "content": message.content}))
        }
        chat_record::Message::Tool(message) => Ok(json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        })),
        chat_record::Message::Assistant(message) => serialize_assistant(message),
    }
}

fn serialize_assistant(message: &AssistantMessage) -> Result<Value, LlmError> {
    let mut value = Map::from_iter([("role".to_string(), json!("assistant"))]);
    if let Some(content) = &message.content {
        value.insert("content".to_string(), json!(content));
    }
    // The Reflection API requires nonempty reasoning even when the model emitted none.
    let reasoning_content = message
        .reasoning_content
        .as_deref()
        .filter(|reasoning| !reasoning.is_empty())
        .unwrap_or(" ");
    value.insert("reasoning_content".to_string(), json!(reasoning_content));
    if !message.reasoning_details.is_empty() {
        value.insert(
            "reasoning_details".to_string(),
            Value::Array(
                message
                    .reasoning_details
                    .iter()
                    .map(struct_to_json)
                    .collect(),
            ),
        );
    }
    if let Some(refusal) = &message.refusal {
        value.insert("refusal".to_string(), json!(refusal));
    }
    if !message.tool_calls.is_empty() {
        let calls = message
            .tool_calls
            .iter()
            .map(|call| {
                let function = call.function.as_ref().ok_or_else(|| {
                    LlmError::Request(format!("tool call {:?} has no function", call.id))
                })?;
                Ok(json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "arguments": function.arguments,
                    },
                }))
            })
            .collect::<Result<Vec<_>, _>>()?;
        value.insert("tool_calls".to_string(), Value::Array(calls));
    }
    Ok(Value::Object(value))
}

fn api_usage(usage: ApiUsage) -> Usage {
    let details = usage.prompt_tokens_details.unwrap_or_default();
    Usage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_tokens: details.cached_tokens,
        cache_write_tokens: details.cache_write_tokens,
    }
}

fn format_api_error(error: &ApiError) -> String {
    let message = error
        .message
        .clone()
        .unwrap_or_else(|| "model request failed".to_string());
    match &error.code {
        Some(code) => format!("{message} (code={code})"),
        None => message,
    }
}

fn request_error(model: &str, error: impl std::fmt::Display) -> LlmError {
    LlmError::Request(format!("{error} [model={model}]"))
}

fn response_error(model: &str, detail: String) -> LlmError {
    if detail.contains("context_length_exceeded")
        || detail.contains("context_window_exceeded")
        || detail.contains("maximum context length")
    {
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

    fn provider() -> ChatCompletions {
        provider_with_base_url("http://test/v1")
    }

    fn provider_with_base_url(base_url: &str) -> ChatCompletions {
        ChatCompletions::new(
            ChatCompletionsConfig {
                model: "test-model".to_string(),
                api_key: "test-key".to_string(),
                base_url: base_url.to_string(),
            },
            super::super::http_client(None).expect("test HTTP client configuration is valid"),
        )
    }

    fn effort_request(effort: &str, extra_body: Option<&str>) -> CompletionRequest {
        CompletionRequest {
            messages: vec![chat_record::Message::User(UserMessage {
                content: "hello".to_string(),
                checkpoint: None,
            })],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                model: "test-model".to_string(),
                reasoning_effort: Some(effort.to_string()),
                extra_body: extra_body.map(str::to_string),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn request_body_spells_effort_for_each_dialect() {
        let openai = provider();
        let openrouter = provider_with_base_url("https://openrouter.ai/api/v1");

        for effort in ["medium", "max"] {
            let body = openai
                .request_body(effort_request(effort, None))
                .expect("serializes");
            assert_eq!(body["reasoning_effort"], json!(effort));
            assert_eq!(body.get("reasoning"), None);
        }

        let body = openai
            .request_body(effort_request("64", None))
            .expect("serializes");
        assert_eq!(body["reasoning_effort"], json!(64));

        let body = openrouter
            .request_body(effort_request("medium", None))
            .expect("serializes");
        assert_eq!(body["reasoning"], json!({"effort": "medium"}));
        assert_eq!(body.get("reasoning_effort"), None);

        let body = openrouter
            .request_body(effort_request("64", None))
            .expect("serializes");
        assert_eq!(body["reasoning"], json!({"max_tokens": 64}));

        let body = openai
            .request_body(effort_request("none", None))
            .expect("serializes");
        assert_eq!(body.get("reasoning_effort"), None);
    }

    #[test]
    fn request_body_merges_extra_body_last() {
        let body = provider()
            .request_body(effort_request(
                "medium",
                Some(r#"{"reasoning_effort": 256, "chat_template_kwargs": {"enable_thinking": false}}"#),
            ))
            .expect("serializes");
        assert_eq!(body["reasoning_effort"], json!(256));
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"enable_thinking": false})
        );
        assert_eq!(body["model"], json!("test-model"));
    }

    #[test]
    fn continuous_usage_is_opt_in_through_extra_body() {
        let provider = provider();
        let standard = provider
            .request_body(effort_request("medium", None))
            .expect("request");
        assert_eq!(standard["stream_options"], json!({"include_usage": true}));
        let custom = provider
            .request_body(effort_request(
                "medium",
                Some(r#"{"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#),
            ))
            .expect("request");
        assert_eq!(
            custom["stream_options"],
            json!({"include_usage": true, "continuous_usage_stats": true})
        );
    }

    #[test]
    fn request_body_rejects_extra_body_that_is_not_an_object() {
        assert_matches!(
            provider().request_body(effort_request("medium", Some("[1]"))),
            Err(LlmError::Request(message)) if message.contains("JSON object")
        );
    }

    #[test]
    fn request_body_sends_tool_results_without_tool_name() {
        let request = CompletionRequest {
            messages: vec![
                chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    content: "list files".to_string(),
                }),
                chat_record::Message::Tool(ToolMessage {
                    tool_call_id: "call_1".to_string(),
                    content: "ok".to_string(),
                    name: Some("bash".to_string()),
                }),
            ],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                model: "test-model".to_string(),
                ..Default::default()
            }),
        };

        let body = provider()
            .request_body(request)
            .expect("valid messages serialize");

        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": "list files"},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ])
        );
    }

    #[test]
    fn request_body_uses_chat_completions_shapes() {
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
                model: "test-model".to_string(),
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
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "hello"},
                ],
                "stream": true,
                "stream_options": {"include_usage": true},
                "max_tokens": 1024,
                "reasoning_effort": "high",
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command.",
                        "parameters": {"type": "object"},
                    },
                }],
            })
        );
    }

    #[test]
    fn request_body_uses_per_request_sampling_and_does_not_retain_it() {
        let provider = provider();
        let request = CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                temperature: Some(0.0),
                top_p: Some(1.0),
                seed: Some(i64::MAX),
                ..Default::default()
            }),
        };
        let body = provider.request_body(request).expect("sampling request");
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["top_p"], 1.0);
        assert_eq!(body["seed"], i64::MAX);

        let body = provider
            .request_body(CompletionRequest {
                messages: vec![],
                tools: vec![],
                max_tokens: None,
                inference: None,
            })
            .expect("endpoint defaults");
        assert_eq!(body.get("temperature"), None);
        assert_eq!(body.get("top_p"), None);
        assert_eq!(body.get("seed"), None);
    }

    #[tokio::test]
    async fn stream_rejects_invalid_sampling_before_sending_a_request() {
        let provider = provider();
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                temperature: Some(f64::NAN),
                ..Default::default()
            }),
        });
        futures_util::pin_mut!(stream);
        assert_matches!(stream.next().await, Some(Err(LlmError::Request(message)))
            if message == "temperature must be between 0 and 2");
    }

    #[test]
    fn openrouter_request_enables_prompt_caching_and_usage() {
        let request = CompletionRequest {
            messages: vec![chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "hello".to_string(),
            })],
            tools: Vec::new(),
            max_tokens: None,
            inference: None,
        };

        let body = provider_with_base_url("https://openrouter.ai/api/v1")
            .request_body(request)
            .expect("valid messages serialize");

        assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(body["usage"], json!({"include": true}));
    }

    #[tokio::test]
    async fn unsupported_continuous_usage_keeps_final_usage_available() {
        assert_continuous_usage_fallback(
            "400 Bad Request",
            r#"{"error":{"message":"Unknown parameter: stream_options.continuous_usage_stats"}}"#,
        )
        .await;
    }

    #[tokio::test]
    async fn generic_bad_request_retries_without_continuous_usage() {
        assert_continuous_usage_fallback("400 Bad Request", "Invalid request").await;
    }

    #[tokio::test]
    async fn generic_unprocessable_entity_retries_without_continuous_usage() {
        assert_continuous_usage_fallback("422 Unprocessable Entity", "Invalid request").await;
    }

    async fn assert_continuous_usage_fallback(status: &'static str, rejected: &'static str) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("test server");
        let address = listener.local_addr().expect("server address");
        let server = tokio::spawn(async move {
            let stream = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\n",
                "data: [DONE]\n\n",
            );
            for (status, body) in [(status, rejected), ("200 OK", stream)] {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let mut request = [0; 16384];
                let received = socket.read(&mut request).await.expect("read request");
                assert!(received > 0);
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("response");
            }
        });
        let provider = provider_with_base_url(&format!("http://{address}"));
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                extra_body: Some(
                    r#"{"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#
                        .into(),
                ),
                ..Default::default()
            }),
        });
        futures_util::pin_mut!(stream);
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::Content(_))));
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::OutputTokens(2))));
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::Final(completion))) => {
            assert_eq!(completion.usage.expect("usage").completion_tokens, 2);
        });
        assert!(stream.next().await.is_none());
        server.await.expect("server completed");
    }

    #[tokio::test]
    async fn continuous_usage_fallback_does_not_retry_a_second_rejection() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("test server");
        let address = listener.local_addr().expect("server address");
        let server = tokio::spawn(async move {
            for detail in ["Invalid request", "Still invalid"] {
                let (mut socket, _) = listener.accept().await.expect("accept request");
                let mut request = [0; 16384];
                assert!(socket.read(&mut request).await.expect("read request") > 0);
                let response = format!(
                    "HTTP/1.1 422 Unprocessable Entity\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{detail}",
                    detail.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("response");
            }
        });
        let provider = provider_with_base_url(&format!("http://{address}"));
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                extra_body: Some(
                    r#"{"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#
                        .into(),
                ),
                ..Default::default()
            }),
        });
        futures_util::pin_mut!(stream);
        let result = stream.next().await.expect("request result");
        assert_matches!(result, Err(error) => {
            assert!(error.to_string().contains("Still invalid"));
        });
        assert!(stream.next().await.is_none());
        server.await.expect("server completed");
    }

    #[test]
    fn usage_only_chunks_are_forwarded_without_fabricating_text() {
        let mut state = CompletionState::default();
        let chunk: ApiChunk = serde_json::from_value(json!({
            "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 42}
        }))
        .expect("usage chunk");
        assert_eq!(
            state.apply(chunk).expect("usage update"),
            vec![StreamEvent::OutputTokens(42)]
        );
    }

    #[test]
    fn stream_state_assembles_text_tools_and_usage() {
        let first: ApiChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {
                    "reasoning": "plan",
                    "content": "Hel",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_",
                        "function": {"name": "ba", "arguments": "{\"cmd\":"},
                    }],
                },
            }],
        }))
        .expect("first chunk is valid");
        let second: ApiChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {
                    "content": "lo",
                    "tool_calls": [{
                        "index": 0,
                        "id": "1",
                        "function": {"name": "sh", "arguments": "\"ls\"}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }))
        .expect("second chunk is valid");
        let mut state = CompletionState::default();

        let first_events = state.apply(first).expect("first chunk applies");
        let second_events = state.apply(second).expect("second chunk applies");
        let completion = state.finish().expect("stream completes");

        assert_eq!(
            first_events,
            vec![
                StreamEvent::Reasoning(StreamReasoning {
                    reasoning: "plan".to_string(),
                }),
                StreamEvent::Content(StreamContent {
                    content: "Hel".to_string(),
                }),
                StreamEvent::ToolCall(StreamToolCall {
                    index: 0,
                    name: "ba".to_string(),
                    arguments: "{\"cmd\":".to_string(),
                }),
            ]
        );
        assert_eq!(
            second_events,
            vec![
                StreamEvent::OutputTokens(5),
                StreamEvent::Content(StreamContent {
                    content: "lo".to_string(),
                }),
                StreamEvent::ToolCall(StreamToolCall {
                    index: 0,
                    name: "sh".to_string(),
                    arguments: "\"ls\"}".to_string(),
                }),
            ]
        );
        assert_eq!(completion.message.content.as_deref(), Some("Hello"));
        assert_eq!(
            completion.message.reasoning_content.as_deref(),
            Some("plan")
        );
        assert_eq!(completion.message.tool_calls[0].id, "call_1");
        assert_eq!(
            completion.message.tool_calls[0]
                .function
                .as_ref()
                .expect("tool call has a function")
                .name,
            "bash"
        );
        assert_eq!(
            completion.usage.expect("usage is present").prompt_tokens,
            10
        );
    }

    #[test]
    fn stream_state_merges_reasoning_details_by_type_and_index() {
        let first: ApiChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {
                    "reasoning_details": [
                        {"type": "reasoning.text", "index": 0, "text": "plan"},
                    ],
                },
            }],
        }))
        .expect("first chunk is valid");
        let second: ApiChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {
                    "reasoning_details": [
                        {"type": "reasoning.summary", "index": 1, "summary": "short"},
                    ],
                },
            }],
        }))
        .expect("second chunk is valid");
        let third: ApiChunk = serde_json::from_value(json!({
            "choices": [{
                "delta": {
                    "reasoning_details": [{
                        "type": "reasoning.text",
                        "index": 0,
                        "text": " done",
                        "signature": "sig",
                    }],
                },
                "finish_reason": "stop",
            }],
        }))
        .expect("third chunk is valid");
        let mut state = CompletionState::default();

        assert_eq!(state.apply(first).expect("first chunk applies"), Vec::new());
        assert_eq!(
            state.apply(second).expect("second chunk applies"),
            Vec::new()
        );
        assert_eq!(state.apply(third).expect("third chunk applies"), Vec::new());
        let completion = state.finish().expect("stream completes");

        assert_eq!(
            completion
                .message
                .reasoning_details
                .iter()
                .map(struct_to_json)
                .collect::<Vec<_>>(),
            vec![
                json!({
                    "type": "reasoning.text",
                    "index": 0.0,
                    "text": "plan done",
                    "signature": "sig",
                }),
                json!({
                    "type": "reasoning.summary",
                    "index": 1.0,
                    "summary": "short",
                }),
            ]
        );
    }

    #[test]
    fn non_streaming_completion_preserves_refusal_reasoning_and_usage() {
        let response: ApiCompletion = serde_json::from_value(json!({
            "choices": [{
                "message": {"content": null, "reasoning": "plan", "refusal": "Cannot help"},
                "finish_reason": "content_filter",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2}},
        }))
        .expect("complete response");
        let completion = response.finish("test-model").expect("refusal completion");
        assert_eq!(completion.message.refusal.as_deref(), Some("Cannot help"));
        assert_eq!(
            completion.message.reasoning_content.as_deref(),
            Some("plan")
        );
        assert_eq!(completion.message.content, None);
        assert_eq!(
            completion.usage,
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cached_tokens: 3,
                cache_write_tokens: 2,
            })
        );
        assert!(!completion.truncated);
    }

    #[test]
    fn non_streaming_length_limit_drops_incomplete_last_tool_call() {
        let response: ApiCompletion = serde_json::from_value(json!({
            "choices": [{
                "message": {"tool_calls": [
                    {"id": "first", "function": {"name": "ls", "arguments": "{}"}},
                    {"id": "second", "function": {"name": "bash", "arguments": "{\"command\":"}},
                ]},
                "finish_reason": "length",
            }],
        }))
        .expect("truncated response");
        let completion = response.finish("test-model").expect("truncated completion");
        assert!(completion.truncated);
        assert_eq!(completion.message.tool_calls.len(), 1);
        assert_eq!(completion.message.tool_calls[0].id, "first");
        assert_eq!(completion.usage, None);
    }

    #[test]
    fn non_streaming_error_preserves_context_overflow_classification() {
        let response: ApiCompletion = serde_json::from_value(json!({
            "error": {"message": "too many tokens", "code": "context_length_exceeded"},
        }))
        .expect("error response");
        assert_matches!(
            response.finish("test-model"),
            Err(LlmError::ContextOverflow(_))
        );
    }

    #[test]
    fn non_streaming_completion_rejects_empty_choices() {
        let response: ApiCompletion =
            serde_json::from_value(json!({"choices": []})).expect("empty response");
        assert_matches!(response.finish("test-model"), Err(LlmError::Request(message))
            if message.contains("no choices"));
    }

    #[test]
    fn non_streaming_completion_requires_a_finish_reason() {
        let response: ApiCompletion = serde_json::from_value(json!({
            "choices": [{"message": {"content": "unfinished"}}],
        }))
        .expect("unfinished response");
        assert_matches!(response.finish("test-model"), Err(LlmError::Request(message))
            if message.contains("without a finish reason"));
    }

    #[test]
    fn non_streaming_completion_rejects_a_missing_message() {
        assert_matches!(
            serde_json::from_value::<ApiCompletion>(json!({
                "choices": [{"finish_reason": "stop"}],
            })),
            Err(_)
        );
    }

    #[test]
    fn context_overflow_is_classified() {
        let error = response_error(
            "test-model",
            "This model's maximum context length was exceeded".to_string(),
        );

        assert_matches!(error, LlmError::ContextOverflow(_));
    }
}
