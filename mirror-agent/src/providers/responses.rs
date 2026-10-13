//! Streaming OpenAI Responses API model provider.

use std::collections::BTreeMap;

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
use prost_types::value::Kind;
use prost_types::Struct;
use reqwest::header::HeaderMap;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use serde_json::Map;
use serde_json::Value;

use super::apply_extra_body;
use super::Llm;
use super::LlmError;
use super::ModelApi;
use super::ModelRequestGuard;
use super::StreamEvent;
use crate::providers::Completion;
use crate::providers::CompletionRequest;

#[cfg(test)]
#[path = "responses_test.rs"]
mod tests;

/// A streaming model backed by an OpenAI Responses API endpoint.
#[derive(Debug)]
pub struct ResponsesClient {
    config: ResponsesConfig,
    client: Client,
}

/// Configuration for an OpenAI Responses API model.
#[derive(Debug, Clone)]
pub struct ResponsesConfig {
    pub model: String,
    pub api_key: String,
    pub base_url: String,
    pub prompt_cache_key: Option<String>,
}

impl ResponsesClient {
    /// Create a provider with a reusable HTTP client.
    pub fn new(config: ResponsesConfig, client: Client) -> Self {
        Self { config, client }
    }

    pub(super) fn request_body(&self, request: CompletionRequest) -> Result<Value, LlmError> {
        if let Some(inference) = &request.inference {
            ModelApi::Responses.validate_sampling(inference)?;
        }
        let mut input = Vec::new();
        for (index, message) in request.messages.iter().enumerate() {
            input.extend(serialize_message(message, index)?);
        }
        let mut body = Map::from_iter([
            ("model".to_string(), json!(self.config.model)),
            ("input".to_string(), Value::Array(input)),
            ("store".to_string(), Value::Bool(false)),
            ("stream".to_string(), Value::Bool(true)),
            (
                "include".to_string(),
                json!(["reasoning.encrypted_content"]),
            ),
        ]);
        if let Some(max_tokens) = request.max_tokens {
            body.insert("max_output_tokens".to_string(), json!(max_tokens));
        }
        if let Some(prompt_cache_key) = &self.config.prompt_cache_key {
            body.insert("prompt_cache_key".to_string(), json!(prompt_cache_key));
        }
        if let Some(reasoning_effort) = request
            .inference
            .as_ref()
            .and_then(|inference| inference.reasoning_effort.as_deref())
            .filter(|effort| *effort != "none")
        {
            body.insert("reasoning".to_string(), json!({"effort": reasoning_effort}));
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
                                "name": tool.name,
                                "description": tool.description,
                                "strict": false,
                                "parameters": tool.parameters,
                            })
                        })
                        .collect(),
                ),
            );
        }
        apply_extra_body(&mut body, request.inference.as_ref())?;
        Ok(Value::Object(body))
    }
}

impl Llm for ResponsesClient {
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

impl ResponsesClient {
    pub(super) fn stream_guarded<'a>(
        &'a self,
        request: CompletionRequest,
        guard: Option<&'a dyn ModelRequestGuard>,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        self.stream_body(
            self.request_body(request),
            HeaderMap::new(),
            self.config.api_key.clone(),
            guard,
        )
    }

    pub(super) fn stream_body<'a>(
        &'a self,
        body: Result<Value, LlmError>,
        headers: HeaderMap,
        bearer_token: String,
        guard: Option<&'a dyn ModelRequestGuard>,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        let client = self.client.clone();
        let endpoint = format!("{}/responses", self.config.base_url.trim_end_matches('/'));
        let model = self.config.model.clone();
        try_stream! {
            let body = body?;
            if let Some(guard) = guard {
                guard.verify().await?;
            }
            let response = client
                .post(endpoint)
                .headers(headers)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .bearer_auth(bearer_token)
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

            let mut completed_items = Vec::new();
            let mut events = response.bytes_stream().eventsource();
            while let Some(event) = events.next().await {
                let payload = event
                    .map_err(|error| request_error(&model, error))?
                    .data;
                if payload == "[DONE]" {
                    continue;
                }
                let event: ApiEvent = serde_json::from_str(&payload)
                    .map_err(|error| request_error(&model, error))?;
                if event.kind == "error" || event.kind == "response.failed" {
                    Err(response_error(&model, stream_error(&event)))?;
                }
                if let Some(usage) = event.response.as_ref().and_then(|response| response.usage.as_ref()) {
                    yield StreamEvent::OutputTokens(usage.output_tokens);
                }
                if event.kind == "response.output_item.done" {
                    if let Some(item) = event.item.clone() {
                        completed_items.push(item);
                    }
                }
                if matches!(
                    event.kind.as_str(),
                    "response.completed" | "response.incomplete" | "response.done"
                ) {
                    let mut final_response = event.response.unwrap_or_default();
                    if final_response.output.is_empty() {
                        final_response.output = completed_items;
                    }
                    yield StreamEvent::Final(build_completion(final_response)?);
                    return;
                }
                if let Some(partial) = stream_partial(event)? {
                    yield partial;
                }
            }
            Err(request_error(
                &model,
                "model stream ended without a terminal response",
            ))?;
        }
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

#[derive(Debug, Default, Deserialize)]
struct ApiResponse {
    status: Option<String>,
    #[serde(default)]
    output: Vec<Value>,
    usage: Option<ApiUsage>,
    incomplete_details: Option<Map<String, Value>>,
    error: Option<ApiError>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    input_tokens_details: Option<ApiUsageDetails>,
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
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
    code: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ApiEvent {
    #[serde(rename = "type")]
    kind: String,
    delta: Option<String>,
    #[serde(default)]
    output_index: u32,
    item: Option<Value>,
    response: Option<ApiResponse>,
    message: Option<String>,
    code: Option<Value>,
    error: Option<ApiError>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiItem {
    #[serde(rename = "type", default)]
    kind: String,
    id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
    content: Option<Vec<ApiContentPart>>,
}

#[derive(Debug, Default, Deserialize)]
struct ApiContentPart {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    refusal: String,
}

fn stream_partial(event: ApiEvent) -> Result<Option<StreamEvent>, LlmError> {
    let partial = match event.kind.as_str() {
        "response.output_text.delta" => event
            .delta
            .filter(|delta| !delta.is_empty())
            .map(|content| StreamEvent::Content(StreamContent { content })),
        "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => event
            .delta
            .filter(|delta| !delta.is_empty())
            .map(|reasoning| StreamEvent::Reasoning(StreamReasoning { reasoning })),
        "response.refusal.delta" => event
            .delta
            .filter(|delta| !delta.is_empty())
            .map(|refusal| StreamEvent::Refusal(StreamRefusal { refusal })),
        "response.output_item.added" => {
            let Some(item) = event.item else {
                return Ok(None);
            };
            let item: ApiItem = serde_json::from_value(item)
                .map_err(|error| LlmError::Request(error.to_string()))?;
            (item.kind == "function_call").then(|| {
                StreamEvent::ToolCall(StreamToolCall {
                    index: event.output_index,
                    name: item.name.unwrap_or_default(),
                    arguments: String::new(),
                })
            })
        }
        "response.function_call_arguments.delta" => event
            .delta
            .filter(|delta| !delta.is_empty())
            .map(|arguments| {
                StreamEvent::ToolCall(StreamToolCall {
                    index: event.output_index,
                    name: String::new(),
                    arguments,
                })
            }),
        _ => None,
    };
    Ok(partial)
}

fn stream_error(event: &ApiEvent) -> String {
    if let Some(error) = &event.error {
        return format_api_error(error);
    }
    if let Some(error) = event
        .response
        .as_ref()
        .and_then(|response| response.error.as_ref())
    {
        return format_api_error(error);
    }
    if let Some(message) = &event.message {
        return match &event.code {
            Some(code) => format!("{message} (code={code})"),
            None => message.clone(),
        };
    }
    "model stream failed".to_string()
}

fn format_api_error(error: &ApiError) -> String {
    let message = error
        .message
        .clone()
        .or_else(|| error.kind.clone())
        .unwrap_or_else(|| "model request failed".to_string());
    match &error.code {
        Some(code) => format!("{message} (code={code})"),
        None => message,
    }
}

fn build_completion(response: ApiResponse) -> Result<Completion, LlmError> {
    let mut message = build_assistant_message(response.output)?;
    let truncated = match response.status.as_deref() {
        Some("completed") => false,
        Some("incomplete") => match response
            .incomplete_details
            .as_ref()
            .and_then(|details| details.get("reason"))
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => {
                drop_incomplete_tool_call(&mut message);
                true
            }
            Some("content_filter") if message.refusal.is_some() => false,
            Some("content_filter") => {
                return Err(LlmError::Request(
                    "model completion was blocked by a content filter".to_string(),
                ));
            }
            Some(reason) => {
                return Err(LlmError::Request(format!(
                    "model completion incomplete: {reason}"
                )));
            }
            None => return Err(LlmError::Request("model completion incomplete".to_string())),
        },
        status => {
            return Err(LlmError::Request(format!(
                "unsupported model response status: {status:?}"
            )));
        }
    };
    if message.tool_calls.iter().any(|call| {
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
        message,
        usage: response.usage.map(api_usage),
        truncated,
    })
}

fn build_assistant_message(output: Vec<Value>) -> Result<AssistantMessage, LlmError> {
    let mut content = String::new();
    let mut refusal = String::new();
    let mut reasoning_details = Vec::new();
    let mut tool_calls = Vec::new();
    let mut message_item_id = None;
    for raw in output {
        let item: ApiItem = serde_json::from_value(raw.clone())
            .map_err(|error| LlmError::Request(error.to_string()))?;
        match item.kind.as_str() {
            "reasoning" => reasoning_details.push(json_to_struct(raw)?),
            "message" => {
                message_item_id = message_item_id.or(item.id);
                for part in item.content.unwrap_or_default() {
                    match part.kind.as_str() {
                        "output_text" => content.push_str(&part.text),
                        "refusal" => refusal.push_str(&part.refusal),
                        _ => {}
                    }
                }
            }
            "function_call" => {
                let mut provider_fields = BTreeMap::new();
                if let Some(id) = item.id {
                    provider_fields.insert("item_id".to_string(), string_value(id));
                }
                tool_calls.push(ToolCall {
                    id: item.call_id.unwrap_or_default(),
                    r#type: "function".to_string(),
                    function: Some(FunctionCall {
                        name: item.name.unwrap_or_default(),
                        arguments: item.arguments.unwrap_or_default(),
                    }),
                    provider_fields: (!provider_fields.is_empty()).then_some(Struct {
                        fields: provider_fields,
                    }),
                });
            }
            _ => {}
        }
    }
    let provider_fields = message_item_id.map(|id| Struct {
        fields: BTreeMap::from([("message_item_id".to_string(), string_value(id))]),
    });
    Ok(AssistantMessage {
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        reasoning_content: None,
        reasoning_details,
        refusal: (!refusal.is_empty()).then_some(refusal),
        provider_fields,
    })
}

fn api_usage(usage: ApiUsage) -> Usage {
    let details = usage.input_tokens_details.unwrap_or_default();
    Usage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        cached_tokens: details.cached_tokens,
        cache_write_tokens: details.cache_write_tokens,
    }
}

fn drop_incomplete_tool_call(message: &mut AssistantMessage) {
    let Some(last) = message.tool_calls.last() else {
        return;
    };
    let complete = !last.id.is_empty()
        && last.function.as_ref().is_some_and(|function| {
            !function.name.is_empty() && serde_json::from_str::<Value>(&function.arguments).is_ok()
        });
    if !complete {
        message.tool_calls.pop();
    }
}

fn serialize_message(message: &chat_record::Message, index: usize) -> Result<Vec<Value>, LlmError> {
    match message {
        chat_record::Message::System(message) => Ok(vec![text_input("system", &message.content)]),
        chat_record::Message::User(message) => Ok(vec![text_input("user", &message.content)]),
        chat_record::Message::Tool(message) => Ok(vec![json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content,
        })]),
        chat_record::Message::Assistant(message) => serialize_assistant(message, index),
    }
}

fn text_input(role: &str, content: &str) -> Value {
    json!({
        "role": role,
        "content": [{"type": "input_text", "text": content}],
    })
}

fn serialize_assistant(message: &AssistantMessage, index: usize) -> Result<Vec<Value>, LlmError> {
    let mut items = Vec::new();
    for detail in &message.reasoning_details {
        let value = struct_to_json(detail);
        if value.get("type").and_then(Value::as_str) == Some("reasoning") {
            items.push(value);
        }
    }
    if message.content.is_some() || message.refusal.is_some() {
        let mut content = Vec::new();
        if let Some(text) = &message.content {
            content.push(json!({"type": "output_text", "text": text, "annotations": []}));
        }
        if let Some(refusal) = &message.refusal {
            content.push(json!({"type": "refusal", "refusal": refusal}));
        }
        let id = message
            .provider_fields
            .as_ref()
            .and_then(|fields| fields.fields.get("message_item_id"))
            .and_then(prost_string)
            .map(str::to_string)
            .unwrap_or_else(|| format!("msg_mirror_{index}"));
        items.push(json!({
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "id": id,
            "content": content,
        }));
    }
    for call in &message.tool_calls {
        let Some(function) = &call.function else {
            return Err(LlmError::Request(format!(
                "tool call {:?} has no function",
                call.id
            )));
        };
        let mut item = json!({
            "type": "function_call",
            "call_id": call.id,
            "name": function.name,
            "arguments": function.arguments,
        });
        if let Some(id) = call
            .provider_fields
            .as_ref()
            .and_then(|fields| fields.fields.get("item_id"))
            .and_then(prost_string)
        {
            item["id"] = Value::String(id.to_string());
        }
        items.push(item);
    }
    Ok(items)
}

pub(super) fn json_to_struct(value: Value) -> Result<Struct, LlmError> {
    let Value::Object(fields) = value else {
        return Err(LlmError::Request("expected a JSON object".to_string()));
    };
    Ok(Struct {
        fields: fields
            .into_iter()
            .map(|(key, value)| (key, json_to_prost(value)))
            .collect(),
    })
}

fn json_to_prost(value: Value) -> prost_types::Value {
    match value {
        Value::Null => Kind::NullValue(0).into(),
        Value::Bool(value) => value.into(),
        Value::Number(value) => value.as_f64().unwrap_or_default().into(),
        Value::String(value) => value.into(),
        Value::Array(values) => values
            .into_iter()
            .map(json_to_prost)
            .collect::<Vec<_>>()
            .into(),
        Value::Object(fields) => fields
            .into_iter()
            .map(|(key, value)| (key, json_to_prost(value)))
            .collect::<BTreeMap<_, _>>()
            .into(),
    }
}

pub(super) fn struct_to_json(value: &Struct) -> Value {
    Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_to_json(value)))
            .collect(),
    )
}

fn prost_to_json(value: &prost_types::Value) -> Value {
    match value.kind.as_ref() {
        Some(Kind::NullValue(_)) | None => Value::Null,
        Some(Kind::BoolValue(value)) => Value::Bool(*value),
        Some(Kind::NumberValue(value)) => json!(value),
        Some(Kind::StringValue(value)) => Value::String(value.clone()),
        Some(Kind::ListValue(value)) => {
            Value::Array(value.values.iter().map(prost_to_json).collect())
        }
        Some(Kind::StructValue(value)) => struct_to_json(value),
    }
}

fn string_value(value: String) -> prost_types::Value {
    prost_types::Value {
        kind: Some(Kind::StringValue(value)),
    }
}

fn prost_string(value: &prost_types::Value) -> Option<&str> {
    match value.kind.as_ref() {
        Some(Kind::StringValue(value)) => Some(value),
        _ => None,
    }
}
