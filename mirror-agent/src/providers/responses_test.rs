use crate::tools::ToolDefinition;
use assert_matches::assert_matches;
use mirror_protocol::InferenceConfig;
use mirror_protocol::SystemMessage;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;

use super::*;

fn provider() -> ResponsesClient {
    ResponsesClient::new(
        ResponsesConfig {
            model: "gpt-test".to_string(),
            api_key: "test-key".to_string(),
            base_url: "http://test/v1".to_string(),
            prompt_cache_key: Some("session-1".to_string()),
        },
        super::super::http_client(None).expect("test HTTP client configuration is valid"),
    )
}

#[test]
fn request_body_contains_messages_tools_and_session_settings() {
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
            model: "gpt-test".to_string(),
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
            "model": "gpt-test",
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "be helpful"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 1024,
            "prompt_cache_key": "session-1",
            "reasoning": {"effort": "high"},
            "tools": [{
                "type": "function",
                "name": "bash",
                "description": "Run a command.",
                "strict": false,
                "parameters": {"type": "object"},
            }],
        })
    );
}

#[test]
fn request_body_merges_extra_body_last() {
    let body = provider()
        .request_body(CompletionRequest {
            messages: vec![chat_record::Message::User(UserMessage {
                content: "hello".to_string(),
                checkpoint: None,
            })],
            tools: vec![],
            max_tokens: None,
            inference: Some(InferenceConfig {
                model: "gpt-test".to_string(),
                extra_body: Some(r#"{"store": true, "metadata": {"run": "r1"}}"#.to_string()),
                ..Default::default()
            }),
        })
        .expect("valid messages serialize");
    assert_eq!(body["store"], json!(true));
    assert_eq!(body["metadata"], json!({"run": "r1"}));
}

#[test]
fn request_body_omits_disabled_reasoning() {
    let request = CompletionRequest {
        messages: vec![],
        tools: vec![],
        max_tokens: None,
        inference: Some(InferenceConfig {
            model: "gpt-test".to_string(),
            api: None,
            base_url: None,
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        }),
    };

    let body = provider()
        .request_body(request)
        .expect("empty input serializes");

    assert_eq!(body.get("reasoning"), None);
}

#[tokio::test]
async fn stream_rejects_unsupported_sampling_before_sending_a_request() {
    let provider = provider();
    let stream = provider.stream(CompletionRequest {
        messages: vec![],
        tools: vec![],
        max_tokens: None,
        inference: Some(InferenceConfig {
            seed: Some(0),
            ..Default::default()
        }),
    });
    futures_util::pin_mut!(stream);
    assert_matches!(stream.next().await, Some(Err(LlmError::Request(message)))
        if message == "sampling controls require openai-chat-completions");
}

#[test]
fn completion_assembles_message_reasoning_tool_call_and_usage() {
    let reasoning = json!({
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "encrypted",
    });
    let response = ApiResponse {
        status: Some("completed".to_string()),
        output: vec![
            reasoning.clone(),
            json!({
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "hello"}],
            }),
            json!({
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "{\"cmd\":\"ls\"}",
            }),
        ],
        usage: Some(ApiUsage {
            input_tokens: 100,
            output_tokens: 5,
            input_tokens_details: Some(ApiUsageDetails {
                cached_tokens: 80,
                cache_write_tokens: 0,
            }),
        }),
        ..ApiResponse::default()
    };

    let completion = build_completion(response).expect("valid response completes");

    assert_eq!(completion.message.content.as_deref(), Some("hello"));
    assert_eq!(
        completion.message.reasoning_details,
        vec![json_to_struct(reasoning).expect("reasoning is an object")]
    );
    assert_eq!(completion.message.tool_calls.len(), 1);
    assert_eq!(completion.message.tool_calls[0].id, "call_1");
    assert_eq!(
        completion.message.tool_calls[0]
            .function
            .as_ref()
            .expect("function call has a function")
            .name,
        "bash"
    );
    assert_eq!(
        completion.usage,
        Some(Usage {
            prompt_tokens: 100,
            completion_tokens: 5,
            cached_tokens: 80,
            cache_write_tokens: 0,
        })
    );
    assert!(!completion.truncated);
}

#[test]
fn assistant_replay_preserves_provider_items() {
    let message = build_assistant_message(vec![
        json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "encrypted",
        }),
        json!({
            "type": "message",
            "id": "msg_1",
            "content": [{"type": "output_text", "text": "hello"}],
        }),
        json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "bash",
            "arguments": "{\"cmd\":\"ls\"}",
        }),
    ])
    .expect("valid output builds a message");

    let items = serialize_assistant(&message, 1).expect("assistant message serializes");

    assert_eq!(
        items,
        vec![
            json!({
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "encrypted",
            }),
            json!({
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "msg_1",
                "content": [{
                    "type": "output_text",
                    "text": "hello",
                    "annotations": [],
                }],
            }),
            json!({
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "{\"cmd\":\"ls\"}",
            }),
        ]
    );
}

#[test]
fn incomplete_completion_keeps_content_and_drops_partial_tool_call() {
    let response = ApiResponse {
        status: Some("incomplete".to_string()),
        output: vec![
            json!({
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "hello"}],
            }),
            json!({
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "{\"cmd\":\"ls\"}",
            }),
            json!({
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "bash",
                "arguments": "{\"cmd\":",
            }),
        ],
        incomplete_details: Some(Map::from_iter([(
            "reason".to_string(),
            json!("max_output_tokens"),
        )])),
        ..ApiResponse::default()
    };

    let completion = build_completion(response).expect("length truncation is usable");

    assert!(completion.truncated);
    assert_eq!(completion.message.content.as_deref(), Some("hello"));
    assert_eq!(completion.message.tool_calls.len(), 1);
    assert_eq!(completion.message.tool_calls[0].id, "call_1");
}

#[test]
fn completed_response_rejects_incomplete_tool_call() {
    let response = ApiResponse {
        status: Some("completed".to_string()),
        output: vec![json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": "",
            "arguments": "{}",
        })],
        ..ApiResponse::default()
    };

    let error = build_completion(response).expect_err("tool name is required");

    assert_eq!(
        error,
        LlmError::Request("model completion produced an incomplete tool call".to_string())
    );
}

#[test]
fn content_filter_refusal_is_a_valid_completion() {
    let response = ApiResponse {
        status: Some("incomplete".to_string()),
        output: vec![json!({
            "type": "message",
            "content": [{"type": "refusal", "refusal": "I cannot help."}],
        })],
        incomplete_details: Some(Map::from_iter([(
            "reason".to_string(),
            json!("content_filter"),
        )])),
        ..ApiResponse::default()
    };

    let completion = build_completion(response).expect("refusal explains the filtered output");

    assert_eq!(
        completion.message.refusal.as_deref(),
        Some("I cannot help.")
    );
    assert!(!completion.truncated);
}

#[test]
fn stream_deltas_map_to_protocol_events() {
    let tool_start = stream_partial(ApiEvent {
        kind: "response.output_item.added".to_string(),
        delta: None,
        output_index: 2,
        item: Some(json!({"type": "function_call", "name": "bash"})),
        response: None,
        message: None,
        code: None,
        error: None,
    })
    .expect("valid tool event maps");
    let arguments = stream_partial(ApiEvent {
        kind: "response.function_call_arguments.delta".to_string(),
        delta: Some("{\"cmd\":".to_string()),
        output_index: 2,
        item: None,
        response: None,
        message: None,
        code: None,
        error: None,
    })
    .expect("valid argument event maps");
    let content = stream_partial(ApiEvent {
        kind: "response.output_text.delta".to_string(),
        delta: Some("hello".to_string()),
        output_index: 3,
        item: None,
        response: None,
        message: None,
        code: None,
        error: None,
    })
    .expect("valid content event maps");

    assert_eq!(
        tool_start,
        Some(StreamEvent::ToolCall(StreamToolCall {
            index: 2,
            name: "bash".to_string(),
            arguments: String::new(),
        }))
    );
    assert_eq!(
        arguments,
        Some(StreamEvent::ToolCall(StreamToolCall {
            index: 2,
            name: String::new(),
            arguments: "{\"cmd\":".to_string(),
        }))
    );
    assert_eq!(
        content,
        Some(StreamEvent::Content(StreamContent {
            content: "hello".to_string(),
        }))
    );
}

#[test]
fn context_overflow_is_classified() {
    let error = response_error(
        "gpt-test",
        "This model's maximum context length was exceeded (code=context_length_exceeded)"
            .to_string(),
    );

    assert_matches!(error, LlmError::ContextOverflow(_));
}
