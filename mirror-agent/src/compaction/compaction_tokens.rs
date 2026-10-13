//! Prompt-size estimates and completion budgets.

use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;

use super::active_window::active_window;
use crate::tools::ToolDefinition;

const CHARS_PER_TOKEN: usize = 4;
const BAND_PERCENT: usize = 5;

/// Prompt plus completion boundary below the full model window.
pub fn boundary(window: usize) -> usize {
    window * (100 - BAND_PERCENT) / 100
}

/// Minimum useful completion before compaction is required.
pub fn min_completion(window: usize, max_tokens: usize) -> usize {
    max_tokens.min(window * BAND_PERCENT / 100)
}

/// Prompt size that triggers compaction.
pub fn compaction_line(window: usize, max_tokens: usize) -> usize {
    boundary(window).saturating_sub(min_completion(window, max_tokens))
}

/// Convert a character count to the shared approximate token count.
pub fn estimate_chars(chars: usize) -> usize {
    chars / CHARS_PER_TOKEN
}

/// Convert an approximate token count to characters.
pub fn chars_for_tokens(tokens: usize) -> usize {
    tokens.saturating_mul(CHARS_PER_TOKEN)
}

/// Estimate serialized tool-schema size.
pub fn estimate_tools(tools: &[ToolDefinition]) -> usize {
    if tools.is_empty() {
        return 0;
    }
    let schemas = tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": &tool.name,
                    "description": &tool.description,
                    "parameters": &tool.parameters,
                },
            })
        })
        .collect::<Vec<_>>();
    estimate_chars(
        serde_json::Value::Array(schemas)
            .to_string()
            .chars()
            .count(),
    )
}

/// Estimate message size using the Python harness's four-characters rule.
pub fn estimate_messages(messages: &[chat_record::Message]) -> usize {
    let chars = messages.iter().map(message_chars).sum();
    estimate_chars(chars)
}

/// Estimate the replayed prompt, anchored on the latest reported usage.
pub fn prompt_size(entries: &[AgentMessage], tool_tokens: usize) -> usize {
    let mut messages = Vec::new();
    let mut measured = 0;
    let mut tail_start = 0;
    for entry in active_window(entries) {
        match &entry.payload {
            Some(agent_message::Payload::Chat(chat)) => {
                messages.extend(entry.to_llm_messages());
                if let Some(usage) = &chat.usage {
                    measured = usage
                        .prompt_tokens
                        .saturating_add(usage.completion_tokens)
                        .try_into()
                        .unwrap_or(usize::MAX);
                    tail_start = messages.len();
                }
            }
            _ => messages.extend(entry.to_llm_messages()),
        }
    }
    if measured == 0 {
        estimate_messages(&messages).saturating_add(tool_tokens)
    } else {
        measured.saturating_add(estimate_messages(&messages[tail_start..]))
    }
}

/// Choose a completion budget or require compaction.
pub fn completion_budget(prompt: usize, window: usize, max_tokens: usize) -> Option<usize> {
    let room = boundary(window).saturating_sub(prompt);
    (room >= min_completion(window, max_tokens)).then(|| max_tokens.min(room))
}

fn message_chars(message: &chat_record::Message) -> usize {
    match message {
        chat_record::Message::System(message) => message.content.len(),
        chat_record::Message::User(message) => message.content.len(),
        chat_record::Message::Tool(message) => message.content.len(),
        chat_record::Message::Assistant(message) => {
            let reasoning_details = if message.reasoning_details.is_empty() {
                0
            } else {
                super::compactor::reasoning_details_json(&message.reasoning_details)
                    .to_string()
                    .chars()
                    .count()
            };
            message.content.as_ref().map_or(0, String::len)
                + message.reasoning_content.as_ref().map_or(0, String::len)
                + reasoning_details
                + message.refusal.as_ref().map_or(0, String::len)
                + message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        call.function
                            .as_ref()
                            .map_or(0, |function| function.name.len() + function.arguments.len())
                    })
                    .sum::<usize>()
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use mirror_protocol::AssistantMessage;
    use mirror_protocol::TaskResultRecord;
    use pretty_assertions::{assert_eq, assert_ne};
    use prost_types::Struct;

    use super::*;

    #[test]
    fn replays_task_results_as_user_context() {
        let task = AgentMessage {
            payload: Some(agent_message::Payload::TaskResult(TaskResultRecord {
                task_id: "task-1".to_string(),
                tool_name: "bash".to_string(),
                content: "done".to_string(),
                display: None,
                success: true,
            })),
            ..AgentMessage::default()
        };

        assert_eq!(
            task.to_llm_messages(),
            vec![chat_record::Message::User(mirror_protocol::UserMessage {
                checkpoint: None,
                content: "<task_result id=\"task-1\" tool=\"bash\" status=\"succeeded\">\ndone\n</task_result>".to_string(),
            })]
        );
    }

    #[test]
    fn estimates_reasoning_details_and_complete_tool_schemas() {
        let message = chat_record::Message::Assistant(AssistantMessage {
            reasoning_details: vec![Struct {
                fields: [(
                    "text".to_string(),
                    "Responses reasoning details".to_string().into(),
                )]
                .into_iter()
                .collect(),
            }],
            ..AssistantMessage::default()
        });
        assert_ne!(estimate_messages(&[message]), 0);

        let parameters = serde_json::Map::from_iter([(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        )]);
        let payload_chars = "read".len()
            + "Read files".len()
            + serde_json::Value::Object(parameters.clone())
                .to_string()
                .chars()
                .count();
        let tool = ToolDefinition {
            name: "read".to_string(),
            description: "Read files".to_string(),
            parameters,
        };
        assert!(estimate_tools(&[tool]) > estimate_chars(payload_chars));
    }

    #[test]
    fn budgets_reserve_five_percent_each_for_safety_and_completion() {
        assert_eq!(boundary(100_000), 95_000);
        assert_eq!(min_completion(100_000, 64_000), 5_000);
        assert_eq!(compaction_line(100_000, 64_000), 90_000);
        assert_eq!(completion_budget(90_000, 100_000, 64_000), Some(5_000));
        assert_matches!(completion_budget(90_001, 100_000, 64_000), None);
        assert_eq!(compaction_line(100_000, 2_000), 93_000);
    }
}
