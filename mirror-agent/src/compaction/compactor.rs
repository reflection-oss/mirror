//! Conversation summarization and token-budget policy.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::Compaction;
use mirror_protocol::LogRecord;
use mirror_protocol::SystemMessage;
use mirror_protocol::UserMessage;
use prost_types::value::Kind;
use prost_types::Struct;

use super::active_window::active_messages;
use super::active_window::active_window;
use super::compaction_tokens::chars_for_tokens;
use super::compaction_tokens::compaction_line;
use super::compaction_tokens::estimate_chars;
use super::compaction_tokens::estimate_messages;
use super::compaction_tokens::estimate_tools;
use super::compaction_tokens::prompt_size;
use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::tools::ToolDefinition;
use crate::AgentError;

/// Summarize conversation history under a configured token budget.
pub struct Compactor {
    context_window: usize,
    compaction_threshold: usize,
    tool_tokens: usize,
    summary_input_max_chars: usize,
    summary_system_prompt: String,
    max_user_tokens: usize,
    recent_max_chars: usize,
    tool_result_max_chars: usize,
}

impl Compactor {
    pub fn new(
        config: CompactorConfig,
        tools: &[ToolDefinition],
        system_prompt: &str,
    ) -> Result<Self, AgentError> {
        let requested_threshold = config
            .compaction_threshold
            .unwrap_or_else(|| compaction_line(config.context_window, config.max_tokens));
        let compaction_threshold =
            requested_threshold.min(compaction_line(config.context_window, config.max_tokens));
        let tool_tokens = estimate_tools(tools);
        let system_tokens = if system_prompt.trim().is_empty() {
            0
        } else {
            estimate_messages(&[chat_record::Message::System(SystemMessage {
                content: system_prompt.to_string(),
            })])
        };
        let (summary_system_prompt, _) =
            crate::system_prompt::build_system_prompt(&crate::system_prompt::SystemPromptConfig {
                custom_prompt: Some(COMPACTION_SYSTEM_PROMPT.into()),
                prompt_injection_policy: config.prompt_injection_policy,
                include_project_instructions: false,
                include_skills: false,
                ..Default::default()
            });
        let summary_system_tokens =
            estimate_messages(&[chat_record::Message::System(SystemMessage {
                content: summary_system_prompt.clone(),
            })]);
        let summary_wrapper_tokens = estimate_chars(
            "<conversation>\n\n</conversation>\n\n".len() + COMPACTION_USER_PROMPT.len(),
        );
        let summary_input_max_chars = chars_for_tokens(
            compaction_threshold
                .saturating_sub(summary_system_tokens)
                .saturating_sub(summary_wrapper_tokens),
        );
        let max_user_tokens = compaction_threshold
            .saturating_sub(tool_tokens)
            .saturating_sub(system_tokens);
        if requested_threshold > config.context_window || max_user_tokens == 0 {
            return Err(AgentError::InvalidConfiguration(format!(
                "Invalid prompt budget: tools={tool_tokens}, system={system_tokens}, \
                 budget={compaction_threshold}, requested={requested_threshold}, \
                 context_window={}. Reduce the tool catalog or system prompt, or increase the context window.",
                config.context_window
            )));
        }
        let summary_reserve = SUMMARY_RESERVE_TOKENS.min(max_user_tokens / 2);
        let recent_max_chars = RECENT_MAX_CHARS.min(chars_for_tokens(
            max_user_tokens.saturating_sub(summary_reserve),
        ));
        Ok(Self {
            context_window: config.context_window,
            compaction_threshold,
            tool_tokens,
            summary_input_max_chars,
            summary_system_prompt,
            max_user_tokens,
            recent_max_chars,
            tool_result_max_chars: config.tool_result_max_chars,
        })
    }

    pub fn messages_with(
        &self,
        entries: &[AgentMessage],
        additional: &[AgentMessage],
    ) -> Vec<chat_record::Message> {
        active_messages(entries.iter().chain(additional))
    }

    pub fn should_compact(&self, entries: &[AgentMessage], additional: &[AgentMessage]) -> bool {
        self.prompt_tokens_with(entries, additional) >= self.compaction_threshold
    }

    pub fn prepare_with(
        &self,
        entries: &[AgentMessage],
        additional: &[AgentMessage],
    ) -> (Vec<AgentMessage>, Option<CompletionRequest>) {
        let material = self.material_with(entries, additional);
        let mut records = Vec::new();
        if material.oversized_user {
            records.push(AgentMessage {
                payload: Some(agent_message::Payload::Log(LogRecord {
                    level: "warning".to_string(),
                    text: "last user message exceeded budget so it was summarised anyway"
                        .to_string(),
                })),
                ..AgentMessage::default()
            });
        }
        let request = (!material.to_summarize.is_empty()).then(|| {
            let prompt = format!(
                "<conversation>\n{}\n</conversation>\n\n{COMPACTION_USER_PROMPT}",
                material.to_summarize
            );
            CompletionRequest {
                messages: vec![
                    chat_record::Message::System(SystemMessage {
                        content: self.summary_system_prompt.clone(),
                    }),
                    chat_record::Message::User(UserMessage {
                        content: prompt,
                        checkpoint: None,
                    }),
                ],
                tools: Vec::new(),
                max_tokens: None,
                inference: inference_config(entries, additional),
            }
        });
        (records, request)
    }

    pub fn finish(
        &self,
        entries: &[AgentMessage],
        completion: Option<Completion>,
    ) -> Result<Compaction, AgentError> {
        let material = self.material_with(entries, &[]);
        let (summary, usage) = if material.to_summarize.is_empty() {
            if completion.is_some() {
                return Err(AgentError::InvalidInput(
                    "summary completion exists for an empty summary request".to_string(),
                ));
            }
            (material.previous_summary.clone(), None)
        } else {
            let completion = completion.ok_or_else(|| {
                AgentError::InvalidInput(
                    "summary completion is missing for non-empty history".to_string(),
                )
            })?;
            (
                completion.message.content.unwrap_or_default(),
                completion.usage,
            )
        };
        Ok(Compaction {
            summary: (!summary.is_empty()).then_some(summary),
            recent: material.recent,
            usage,
            ..Default::default()
        })
    }

    fn material_with(
        &self,
        entries: &[AgentMessage],
        additional: &[AgentMessage],
    ) -> CompactionMaterial {
        let active = active_window(entries.iter().chain(additional));
        let oversized_user = active
            .iter()
            .rev()
            .filter_map(|entry| entry.chat_message())
            .find(|message| matches!(message, chat_record::Message::User(_)))
            .is_some_and(|message| {
                estimate_messages(std::slice::from_ref(message)) > self.max_user_tokens
            });
        let mut previous_summary = String::new();
        let mut conversation = Vec::new();
        for entry in active {
            if let Some(agent_message::Payload::ContextWindow(window)) = &entry.payload {
                if let Some(compaction) = &window.compaction {
                    previous_summary = compaction.summary.clone().unwrap_or_default();
                    conversation.extend(compaction.recent.iter().cloned());
                }
            } else {
                conversation.push(entry.clone());
            }
        }
        let snapshots: BTreeMap<_, _> = conversation
            .iter()
            .filter_map(|entry| match &entry.payload {
                Some(agent_message::Payload::Extension(record)) if record.survives_compaction => {
                    Some((record.extension.clone(), entry.clone()))
                }
                _ => None,
            })
            .collect();
        conversation.retain(|entry| {
            matches!(
                entry.payload,
                Some(agent_message::Payload::Chat(_) | agent_message::Payload::TaskResult(_))
            )
        });
        let start = recent_start(&conversation, self.recent_max_chars);
        let mut recent: Vec<_> = snapshots.into_values().collect();
        recent.extend_from_slice(&conversation[start..]);
        let transcript = conversation[..start]
            .iter()
            .flat_map(AgentMessage::to_llm_messages)
            .filter_map(|message| serialize_message(&message, self.tool_result_max_chars))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut to_summarize = if transcript.is_empty() {
            String::new()
        } else {
            [previous_summary.as_str(), transcript.as_str()]
                .into_iter()
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        if to_summarize.chars().count() > self.summary_input_max_chars {
            (_, to_summarize) = split_recent(&to_summarize, self.summary_input_max_chars);
        }
        CompactionMaterial {
            previous_summary,
            to_summarize,
            recent,
            oversized_user,
        }
    }

    pub fn context_window(&self) -> usize {
        self.context_window
    }

    pub fn compaction_threshold(&self) -> usize {
        self.compaction_threshold
    }

    pub fn prompt_tokens(&self, entries: &[AgentMessage]) -> usize {
        prompt_size(entries, self.tool_tokens)
    }

    pub fn prompt_tokens_with(
        &self,
        entries: &[AgentMessage],
        additional: &[AgentMessage],
    ) -> usize {
        let mut combined = entries.to_vec();
        combined.extend_from_slice(additional);
        prompt_size(&combined, self.tool_tokens)
    }
}

fn inference_config(
    entries: &[AgentMessage],
    additional: &[AgentMessage],
) -> Option<mirror_protocol::InferenceConfig> {
    additional
        .iter()
        .rev()
        .chain(entries.iter().rev())
        .find_map(|entry| entry.inference.clone())
}

struct CompactionMaterial {
    previous_summary: String,
    to_summarize: String,
    recent: Vec<AgentMessage>,
    oversized_user: bool,
}

fn recent_start(entries: &[AgentMessage], max_chars: usize) -> usize {
    let mut pending = BTreeSet::new();
    let mut boundaries = vec![(0, 0)];
    let mut chars = 0;
    for (index, entry) in entries.iter().enumerate() {
        if pending.is_empty() && matches!(entry.chat_message(), Some(chat_record::Message::User(_)))
        {
            boundaries.push((index, chars));
        }
        for message in entry.to_llm_messages() {
            match &message {
                chat_record::Message::Assistant(assistant) => {
                    pending.extend(assistant.tool_calls.iter().map(|call| call.id.clone()));
                }
                chat_record::Message::Tool(tool) => {
                    pending.remove(&tool.tool_call_id);
                }
                _ => {}
            }
            chars += chars_for_tokens(estimate_messages(std::slice::from_ref(&message)));
        }
    }
    boundaries
        .into_iter()
        .find(|(_, prefix)| chars - prefix <= max_chars)
        .map_or(entries.len(), |(index, _)| index)
}

/// Token limits used to construct a compactor.
#[derive(Debug, Clone)]
pub struct CompactorConfig {
    pub context_window: usize,
    pub compaction_threshold: Option<usize>,
    pub max_tokens: usize,
    pub tool_result_max_chars: usize,
    pub prompt_injection_policy: Option<String>,
}

const COMPACTION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Read the quoted conversation and produce a structured summary in the requested format. Do not continue the conversation or answer questions from it. Output only the summary.";
const COMPACTION_USER_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this exact format:\n\n## Goal\n[What is the user trying to accomplish?]\n\n## Constraints & Preferences\n- [Important constraints, preferences, or requirements]\n\n## Progress\n### Done\n- [x] [Completed work]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Current blockers, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered next steps]\n\n## Critical Context\n- [Data, examples, paths, function names, and errors needed to continue]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";
const RECENT_MAX_CHARS: usize = 80_000;
const SUMMARY_RESERVE_TOKENS: usize = 16_384;

fn serialize_message(
    message: &chat_record::Message,
    tool_result_max_chars: usize,
) -> Option<String> {
    match message {
        chat_record::Message::System(_) => None,
        chat_record::Message::User(message) => Some(format!("[User]: {}", message.content)),
        chat_record::Message::Tool(message) => Some(format!(
            "[Tool result]: {}",
            truncate_for_summary(&message.content, tool_result_max_chars)
        )),
        chat_record::Message::Assistant(message) => {
            let mut parts = Vec::new();
            if let Some(reasoning) = plaintext_reasoning(message) {
                parts.push(format!("[Assistant thinking]: {reasoning}"));
            }
            if let Some(content) = &message.content {
                if !content.is_empty() {
                    parts.push(format!("[Assistant]: {content}"));
                }
            }
            if let Some(refusal) = &message.refusal {
                if !refusal.is_empty() {
                    parts.push(format!("[Assistant]: {refusal}"));
                }
            }
            let calls = message
                .tool_calls
                .iter()
                .filter_map(|call| call.function.as_ref())
                .map(|function| {
                    let arguments = render_arguments(&function.arguments);
                    format!("{}({arguments})", function.name)
                })
                .collect::<Vec<_>>();
            if !calls.is_empty() {
                parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
            }
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        }
    }
}

fn plaintext_reasoning(message: &mirror_protocol::AssistantMessage) -> Option<String> {
    if let Some(reasoning) = message
        .reasoning_content
        .as_ref()
        .filter(|reasoning| !reasoning.trim().is_empty())
    {
        return Some(reasoning.clone());
    }
    let chunks = message
        .reasoning_details
        .iter()
        .flat_map(|detail| detail_texts(&struct_to_json(detail)))
        .collect::<Vec<_>>();
    (!chunks.is_empty()).then(|| chunks.join("\n"))
}

fn detail_texts(detail: &serde_json::Value) -> Vec<String> {
    let Some(fields) = detail.as_object() else {
        return Vec::new();
    };
    for key in ["text", "content", "summary"] {
        let Some(value) = fields.get(key) else {
            continue;
        };
        if let Some(text) = value
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            return vec![text.to_string()];
        }
        let texts = value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|part| part.as_object()?.get("text")?.as_str())
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !texts.is_empty() {
            return texts;
        }
    }
    Vec::new()
}

pub(super) fn reasoning_details_json(details: &[Struct]) -> serde_json::Value {
    serde_json::Value::Array(details.iter().map(struct_to_json).collect())
}

fn struct_to_json(value: &Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_to_json(value)))
            .collect(),
    )
}

fn prost_to_json(value: &prost_types::Value) -> serde_json::Value {
    match value.kind.as_ref() {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(*value),
        Some(Kind::NumberValue(value)) => serde_json::json!(value),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value.clone()),
        Some(Kind::ListValue(value)) => {
            serde_json::Value::Array(value.values.iter().map(prost_to_json).collect())
        }
        Some(Kind::StructValue(value)) => struct_to_json(value),
    }
}

fn render_arguments(arguments: &str) -> String {
    let Ok(serde_json::Value::Object(values)) = serde_json::from_str(arguments) else {
        return arguments.to_string();
    };
    values
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let end = text
        .char_indices()
        .nth(max_chars)
        .map_or(text.len(), |(index, _)| index);
    let omitted = text.chars().count().saturating_sub(max_chars);
    format!(
        "{}\n\n[... {omitted} more characters truncated]",
        &text[..end]
    )
}

fn split_recent(text: &str, max_chars: usize) -> (String, String) {
    if max_chars == 0 {
        return (text.to_string(), String::new());
    }
    let count = text.chars().count();
    if count <= max_chars {
        return (String::new(), text.to_string());
    }
    let split = text
        .char_indices()
        .nth(count - max_chars)
        .map_or(0, |(index, _)| index);
    (text[..split].to_string(), text[split..].to_string())
}

#[cfg(test)]
mod tests {
    //! Compaction replay, budgeting, and truncation behavior.

    use assert_matches::assert_matches;
    use mirror_protocol::agent_message;
    use mirror_protocol::chat_record;
    use mirror_protocol::AgentMessage;
    use mirror_protocol::AssistantMessage;
    use mirror_protocol::ChatRecord;
    use mirror_protocol::Compaction;
    use mirror_protocol::ExtensionRecord;
    use mirror_protocol::FunctionCall;
    use mirror_protocol::SystemMessage;
    use mirror_protocol::ToolCall;
    use mirror_protocol::ToolMessage;
    use mirror_protocol::UserMessage;
    use pretty_assertions::assert_eq;
    use pretty_assertions::assert_ne;

    use super::*;

    #[test]
    fn summary_inherits_policy_without_agent_identity() {
        let compactor = Compactor::new(
            CompactorConfig {
                context_window: 1_000,
                compaction_threshold: Some(500),
                max_tokens: 100,
                tool_result_max_chars: 2_000,
                prompt_injection_policy: Some("Shared safety policy.\n\n".into()),
            },
            &[],
            "Agent tone and identity.",
        )
        .expect("compactor");
        let entries = [chat(chat_record::Message::User(UserMessage {
            checkpoint: None,
            content: "history ".repeat(1_000),
        }))];
        let (_, request) = compactor.prepare_with(&entries, &[]);
        let request = request.expect("summary request");
        assert_matches!(&request.messages[0], chat_record::Message::System(message)
            if message.content.starts_with("Shared safety policy.\n\nYou are a context summarization assistant.")
                && !message.content.contains("Agent tone and identity."));
    }

    #[test]
    fn latest_compaction_window_is_self_contained() {
        let system = |text: &str| {
            chat(chat_record::Message::System(SystemMessage {
                content: text.to_string(),
            }))
        };
        let snapshot = window(
            "updated",
            Compaction {
                summary: Some("summary".to_string()),
                ..Compaction::default()
            },
        );
        let entries = vec![
            system("original"),
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "old history".to_string(),
            })),
            snapshot,
        ];
        let compactor = compactor();
        let messages = compactor.messages_with(&entries, &[]);
        assert_matches!(&messages[0], chat_record::Message::System(system) if system.content == "updated");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages, compactor.messages_with(&entries[2..], &[]));
        assert_eq!(
            compactor.prompt_tokens(&entries),
            compactor.prompt_tokens(&entries[2..])
        );
        assert_eq!(
            compactor.prompt_tokens(&entries),
            estimate_messages(&messages) + compactor.tool_tokens
        );
    }

    #[test]
    fn empty_system_prompt_preserves_compaction_without_a_system_message() {
        let entries = [window(
            "",
            Compaction {
                summary: Some("summary".to_string()),
                recent: vec![chat(chat_record::Message::User(UserMessage {
                    checkpoint: None,
                    content: "continue".to_string(),
                }))],
                ..Compaction::default()
            },
        )];
        let compactor = compactor();
        let messages = compactor.messages_with(&entries, &[]);

        assert_eq!(messages.len(), 2);
        assert_matches!(&messages[0], chat_record::Message::User(message) if message.content.contains("<summary>\nsummary\n</summary>"));
        assert_matches!(&messages[1], chat_record::Message::User(message) if message.content == "continue");
        assert_eq!(
            compactor.prompt_tokens(&entries),
            estimate_messages(&messages) + compactor.tool_tokens
        );
    }

    #[test]
    fn serializes_messages_as_inert_summary_text() {
        let assistant = chat_record::Message::Assistant(AssistantMessage {
            content: Some("Done.".to_string()),
            reasoning_content: Some("Check.".to_string()),
            tool_calls: vec![ToolCall {
                function: Some(FunctionCall {
                    name: "read".to_string(),
                    arguments: r#"{"file_path":"a.py","offset":1}"#.to_string(),
                }),
                ..ToolCall::default()
            }],
            ..AssistantMessage::default()
        });

        assert_eq!(
            serialize_message(&assistant, 2000),
            Some(
                "[Assistant thinking]: Check.\n\n[Assistant]: Done.\n\n[Assistant tool calls]: read(file_path=\"a.py\", offset=1)"
                    .to_string()
            )
        );
        assert_ne!(
            serialize_message(&assistant, 2000),
            Some("Done.".to_string())
        );
        assert_matches!(
            serialize_message(
                &chat_record::Message::System(SystemMessage::default()),
                2000,
            ),
            None
        );
        assert_eq!(
            serialize_message(
                &chat_record::Message::Tool(ToolMessage {
                    content: "abcdef".to_string(),
                    ..ToolMessage::default()
                }),
                3,
            ),
            Some("[Tool result]: abc\n\n[... 3 more characters truncated]".to_string())
        );
    }

    #[test]
    fn preparation_exposes_summary_as_an_external_model_command() {
        let entries = vec![
            chat(chat_record::Message::System(SystemMessage {
                content: "system".to_string(),
            })),
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "old history ".repeat(1000),
            })),
        ];
        let compactor = compactor();

        let (records, request) = compactor.prepare_with(&entries, &[]);

        assert!(!records.is_empty());
        let request = request.expect("non-empty history needs summarization");
        assert!(request.tools.is_empty());
        assert_matches!(
            request.messages.as_slice(),
            [
                chat_record::Message::System(_),
                chat_record::Message::User(_)
            ]
        );
    }

    #[test]
    fn recorded_summary_becomes_one_durable_compaction_record() {
        let entries = vec![
            chat(chat_record::Message::System(SystemMessage {
                content: "system".to_string(),
            })),
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "old history ".repeat(1000),
            })),
        ];

        let result = compactor()
            .finish(
                &entries,
                Some(crate::providers::Completion {
                    message: AssistantMessage {
                        content: Some("SUMMARY".to_string()),
                        ..AssistantMessage::default()
                    },
                    usage: None,
                    truncated: false,
                }),
            )
            .expect("recorded summary");

        assert!(result.summary.as_deref() == Some("SUMMARY"));
    }

    #[test]
    fn compaction_carries_latest_extension_snapshots() {
        let entries = vec![
            window(
                "system",
                Compaction {
                    summary: Some("prior summary".to_string()),
                    recent: vec![snapshot("notes", "state"), snapshot("todo", "old")],
                    ..Compaction::default()
                },
            ),
            extension("scratch", "state"),
            snapshot("todo", "latest"),
        ];

        let result = compactor()
            .finish(&entries, None)
            .expect("recorded summary");

        assert_eq!(
            result.recent,
            vec![snapshot("notes", "state"), snapshot("todo", "latest")]
        );
        assert_matches!(result.summary.as_deref(), Some("prior summary"));
    }

    #[test]
    fn recent_context_is_structured_and_survives_repeated_compaction() {
        let recent_user = chat(chat_record::Message::User(UserMessage {
            checkpoint: None,
            content: "recent question".to_string(),
        }));
        let mut recent_answer = chat(chat_record::Message::Assistant(AssistantMessage {
            content: Some("recent answer".to_string()),
            ..AssistantMessage::default()
        }));
        if let Some(agent_message::Payload::Chat(record)) = &mut recent_answer.payload {
            record.usage = Some(mirror_protocol::Usage {
                prompt_tokens: 1_000_000,
                ..Default::default()
            });
        }
        let entries = vec![
            window("system", Compaction::default()),
            snapshot("todo", "old"),
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "old material ".repeat(1000),
            })),
            recent_user.clone(),
            recent_answer.clone(),
            snapshot("todo", "latest"),
        ];
        let compactor = compactor();
        let (_, request) = compactor.prepare_with(&entries, &[]);
        let request = request.expect("older context needs summarization");
        assert_matches!(&request.messages[1], chat_record::Message::User(user) if !user.content.contains("recent question"));
        let result = compactor
            .finish(
                &entries,
                Some(Completion {
                    message: AssistantMessage {
                        content: Some("summary".to_string()),
                        ..Default::default()
                    },
                    usage: None,
                    truncated: false,
                }),
            )
            .expect("compact");
        assert_eq!(result.summary.as_deref(), Some("summary"));
        assert_eq!(
            result.recent,
            vec![snapshot("todo", "latest"), recent_user, recent_answer]
        );
        let checkpoint = window("refreshed", result);
        let replay = compactor.messages_with(std::slice::from_ref(&checkpoint), &[]);
        assert_eq!(
            compactor.prompt_tokens(std::slice::from_ref(&checkpoint)),
            estimate_messages(&replay)
        );
        assert_matches!(&replay[1], chat_record::Message::User(user) if user.content.contains("<summary>"));
        assert_matches!(&replay[2], chat_record::Message::User(user) if user.content == "recent question");
        let next = vec![checkpoint, snapshot("todo", "newest")];
        let (_, request) = compactor.prepare_with(&next, &[]);
        assert!(request.is_none());
        let result = compactor
            .finish(&next, None)
            .expect("retain existing summary");
        assert_eq!(result.summary.as_deref(), Some("summary"));
        assert_eq!(result.recent.len(), 3);
        assert_eq!(result.recent[0], snapshot("todo", "newest"));
    }

    #[test]
    fn recent_cutoff_never_splits_a_tool_exchange() {
        let entries = vec![
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "old ".repeat(100),
            })),
            chat(chat_record::Message::Assistant(AssistantMessage {
                tool_calls: vec![ToolCall {
                    id: "call".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            })),
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "injected".to_string(),
            })),
            chat(chat_record::Message::Tool(ToolMessage {
                tool_call_id: "call".to_string(),
                content: "result".to_string(),
                ..Default::default()
            })),
            chat(chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "new turn".to_string(),
            })),
        ];
        assert_eq!(recent_start(&entries, 40), 4);
        assert_eq!(recent_start(&entries[..4], 40), 4);
        assert_eq!(recent_start(&entries, 1000), 0);
    }

    #[test]
    fn custom_threshold_is_capped_to_completion_line() {
        let compactor = Compactor::new(
            CompactorConfig {
                context_window: 1_000,
                compaction_threshold: Some(950),
                max_tokens: 800,
                tool_result_max_chars: 2_000,
                prompt_injection_policy: None,
            },
            &[],
            "system",
        )
        .expect("valid compactor");
        let entries = [chat(chat_record::Message::User(UserMessage {
            checkpoint: None,
            content: "x".repeat(3_600),
        }))];

        assert_eq!(compactor.compaction_threshold(), 900);
        assert!(compactor.should_compact(&entries, &[]));
    }

    fn compactor() -> Compactor {
        Compactor::new(
            CompactorConfig {
                context_window: 500,
                compaction_threshold: Some(400),
                max_tokens: 100,
                tool_result_max_chars: 2000,
                prompt_injection_policy: None,
            },
            &[],
            "system",
        )
        .expect("valid test compactor")
    }

    fn window(system: &str, compaction: Compaction) -> AgentMessage {
        AgentMessage {
            payload: Some(agent_message::Payload::ContextWindow(
                mirror_protocol::ContextWindowRecord {
                    system_prompt: system.to_string(),
                    compaction: Some(compaction),
                    ..Default::default()
                },
            )),
            ..AgentMessage::default()
        }
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

    fn extension(extension: &str, kind: &str) -> AgentMessage {
        AgentMessage {
            payload: Some(agent_message::Payload::Extension(ExtensionRecord {
                extension: extension.to_string(),
                kind: kind.to_string(),
                ..ExtensionRecord::default()
            })),
            ..AgentMessage::default()
        }
    }

    fn snapshot(name: &str, kind: &str) -> AgentMessage {
        let mut message = extension(name, kind);
        if let Some(agent_message::Payload::Extension(record)) = &mut message.payload {
            record.survives_compaction = true;
        }
        message
    }
}
