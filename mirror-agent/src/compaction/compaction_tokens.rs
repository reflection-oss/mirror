//! Prompt-size estimates and completion budgets.

use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::ContextUsage;
use mirror_protocol::ContextWindowRecord;

use super::active_window::active_messages;
use super::active_window::active_window;
use crate::tools::ToolDefinition;

const CHARS_PER_TOKEN: usize = 4;
const BAND_PERCENT: usize = 5;
/// What a provider charges for a large image; the measured usage of the next
/// model call replaces this estimate.
const IMAGE_TOKENS: usize = 1_800;

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

pub fn context_usage(entries: &[AgentMessage], tool_tokens: usize, window: usize) -> ContextUsage {
    let tokens = prompt_size(entries, tool_tokens);
    let mut instruction_chars = 0;
    let mut conversation_chars = 0;
    let mut tool_result_chars = 0;
    for message in active_messages(entries) {
        let chars = message_chars(&message);
        match message {
            chat_record::Message::System(_) => instruction_chars += chars,
            chat_record::Message::Tool(_) => tool_result_chars += chars,
            chat_record::Message::User(_) | chat_record::Message::Assistant(_) => {
                conversation_chars += chars
            }
        }
    }
    let task_result_chars = task_result_chars(active_window(entries));
    let instructions = estimate_chars(instruction_chars).saturating_add(tool_tokens);
    let tool_results = estimate_chars(tool_result_chars + task_result_chars);
    let estimated = instructions
        + tool_results
        + estimate_chars(conversation_chars.saturating_sub(task_result_chars));
    let share = |part: usize| {
        part.saturating_mul(tokens)
            .checked_div(estimated)
            .unwrap_or(0)
    };
    let instruction_tokens = share(instructions);
    let tool_result_tokens = share(tool_results);
    ContextUsage {
        tokens: tokens as u64,
        window: window as u64,
        instruction_tokens: Some(instruction_tokens as u64),
        message_tokens: Some(tokens.saturating_sub(instruction_tokens + tool_result_tokens) as u64),
        tool_result_tokens: Some(tool_result_tokens as u64),
    }
}

fn task_result_chars<'a>(entries: impl IntoIterator<Item = &'a AgentMessage>) -> usize {
    entries
        .into_iter()
        .map(|entry| match &entry.payload {
            Some(agent_message::Payload::TaskResult(_)) => {
                entry.to_llm_messages().iter().map(message_chars).sum()
            }
            Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
                compaction: Some(compaction),
                ..
            })) => task_result_chars(&compaction.recent),
            _ => 0,
        })
        .sum()
}

pub fn measure_compaction(entries: &[AgentMessage], window: &mut AgentMessage, tool_tokens: usize) {
    let before = prompt_size(entries, tool_tokens);
    let after = prompt_size(std::slice::from_ref(window), tool_tokens);
    if let Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
        compaction: Some(compaction),
        ..
    })) = &mut window.payload
    {
        compaction.context_tokens_before = Some(before as u64);
        compaction.context_tokens_after = Some(after as u64);
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
        chat_record::Message::User(message) => {
            message.text_content().len() + message.images.len() * chars_for_tokens(IMAGE_TOKENS)
        }
        chat_record::Message::Tool(message) => message.content.len(),
        chat_record::Message::Assistant(message) => {
            let reasoning_details = if message.reasoning_details.is_empty() {
                0
            } else {
                super::summary_compactor::reasoning_details_json(&message.reasoning_details)
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
    use mirror_protocol::ChatRecord;
    use mirror_protocol::Compaction;
    use mirror_protocol::FunctionCall;
    use mirror_protocol::ImageContent;
    use mirror_protocol::TaskResultRecord;
    use mirror_protocol::ToolCall;
    use mirror_protocol::ToolMessage;
    use mirror_protocol::Usage;
    use mirror_protocol::UserMessage;
    use pretty_assertions::{assert_eq, assert_ne};
    use prost_types::Struct;

    use super::*;

    #[test]
    fn estimate_counts_each_image_beside_the_text() {
        let text_only = UserMessage {
            parts: Vec::new(),
            documents: Vec::new(),
            content: "u".repeat(400),
            checkpoint: None,
            images: Vec::new(),
        };
        let with_images = UserMessage {
            images: vec![ImageContent::default(), ImageContent::default()],
            ..text_only.clone()
        };

        assert_eq!(
            estimate_messages(&[chat_record::Message::User(text_only)]),
            100
        );
        assert_eq!(
            estimate_messages(&[chat_record::Message::User(with_images)]),
            100 + 2 * IMAGE_TOKENS
        );
    }

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
                parts: Vec::new(),
                documents: Vec::new(),
                checkpoint: None,
                images: Vec::new(),
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
    fn context_usage_splits_the_prompt_into_instructions_messages_and_tool_results() {
        let mut entries = vec![
            AgentMessage {
                payload: Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
                    system_prompt: "s".repeat(400),
                    ..ContextWindowRecord::default()
                })),
                ..AgentMessage::default()
            },
            chat(chat_record::Message::User(UserMessage {
                parts: Vec::new(),
                documents: Vec::new(),
                checkpoint: None,
                images: Vec::new(),
                content: "u".repeat(200),
            })),
            chat(chat_record::Message::Assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    function: Some(FunctionCall {
                        name: "read".to_string(),
                        arguments: "a".repeat(196),
                    }),
                    ..ToolCall::default()
                }],
                ..AssistantMessage::default()
            })),
            chat(chat_record::Message::Tool(ToolMessage {
                content: "t".repeat(800),
                ..ToolMessage::default()
            })),
        ];

        assert_eq!(
            context_usage(&entries, 50, 1_000),
            ContextUsage {
                tokens: 450,
                window: 1_000,
                instruction_tokens: Some(150),
                message_tokens: Some(100),
                tool_result_tokens: Some(200),
            }
        );

        if let Some(agent_message::Payload::Chat(record)) = &mut entries[2].payload {
            record.usage = Some(Usage {
                prompt_tokens: 800,
                completion_tokens: 100,
                ..Usage::default()
            });
        }
        assert_eq!(
            context_usage(&entries, 50, 1_000),
            ContextUsage {
                tokens: 1_100,
                window: 1_000,
                instruction_tokens: Some(366),
                message_tokens: Some(246),
                tool_result_tokens: Some(488),
            }
        );
    }

    #[test]
    fn context_usage_counts_task_results_as_tool_results() {
        let entries = vec![
            AgentMessage {
                payload: Some(agent_message::Payload::ContextWindow(ContextWindowRecord {
                    system_prompt: "s".repeat(400),
                    compaction: Some(Compaction {
                        recent: vec![task_result()],
                        ..Compaction::default()
                    }),
                    ..ContextWindowRecord::default()
                })),
                ..AgentMessage::default()
            },
            task_result(),
        ];

        assert_eq!(
            context_usage(&entries, 0, 1_000),
            ContextUsage {
                tokens: 300,
                window: 1_000,
                instruction_tokens: Some(100),
                message_tokens: Some(0),
                tool_result_tokens: Some(200),
            }
        );
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

    fn chat(message: chat_record::Message) -> AgentMessage {
        AgentMessage {
            payload: Some(agent_message::Payload::Chat(ChatRecord {
                message: Some(message),
                success: true,
                ..ChatRecord::default()
            })),
            ..AgentMessage::default()
        }
    }

    fn task_result() -> AgentMessage {
        AgentMessage {
            payload: Some(agent_message::Payload::TaskResult(TaskResultRecord {
                task_id: "t".to_string(),
                tool_name: "bash".to_string(),
                content: "r".repeat(333),
                display: None,
                success: true,
            })),
            ..AgentMessage::default()
        }
    }
}
