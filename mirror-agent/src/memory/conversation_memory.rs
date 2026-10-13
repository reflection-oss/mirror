//! One conversation's memory: what to recall after the user speaks and what
//! to attach when a tool touches a file. What the model has already been shown
//! is read from the conversation's active window, never tracked here.

use std::collections::HashSet;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::ExtensionRecord;
use mirror_protocol::UserMessage;

use super::now;
use super::render_recall;
use super::search_memory_tool::search_result_ids;
use super::select_from;
use super::MemoryQuery;
use super::MemoryResult;
use super::MemoryStore;
use super::SearchMemoryTool;
use super::Selected;
use super::SelectionRequest;
use crate::compaction::active_window;
use crate::react::ContextHooks;
use crate::Workspace;

/// One conversation's part in one owner's memory: what it recalls after a
/// message, attaches after a tool batch, and starts in the background when a
/// turn ends.
pub struct ConversationMemory {
    store: Arc<dyn MemoryStore>,
    workspace: Arc<dyn Workspace>,
    conversation_id: String,
}

pub const MEMORY_EXTENSION: &str = "memory";
pub const RECALL_KIND: &str = "recall";

/// What the active context window already shows the model.
#[derive(Debug, Default)]
pub struct Surfaced {
    pub ids: HashSet<String>,
    pub chars: usize,
}

impl ConversationMemory {
    pub fn new(
        store: Arc<dyn MemoryStore>,
        workspace: Arc<dyn Workspace>,
        conversation_id: impl Into<String>,
    ) -> Self {
        Self {
            store,
            workspace,
            conversation_id: conversation_id.into(),
        }
    }

    pub fn store(&self) -> &dyn MemoryStore {
        self.store.as_ref()
    }

    pub fn workspace(&self) -> &dyn Workspace {
        self.workspace.as_ref()
    }

    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    /// The scope every read for this conversation applies: this repository's
    /// records plus repository-agnostic ones.
    pub fn scope(&self) -> MemoryQuery {
        MemoryQuery {
            repository: self.workspace.repository().map(str::to_string),
            ..MemoryQuery::default()
        }
    }

    /// Run the channels `message` and `new_paths` enable against what
    /// `history` already shows, and return the recall to append, if any.
    pub async fn recall(
        &self,
        history: &[AgentMessage],
        message: Option<&str>,
        new_paths: &[String],
    ) -> MemoryResult<Option<AgentMessage>> {
        let surfaced = surfaced(active_window(history));
        let request = SelectionRequest {
            message,
            new_paths,
            surfaced: &surfaced.ids,
            surfaced_chars: surfaced.chars,
            now: now(),
        };
        let selected = select_from(self.store.as_ref(), &self.scope(), &request).await?;
        Ok(recall_record(&selected))
    }

    async fn recall_or_log(
        &self,
        history: &[AgentMessage],
        message: Option<&str>,
        new_paths: &[String],
    ) -> Vec<AgentMessage> {
        match self.recall(history, message, new_paths).await {
            Ok(recall) => recall.into_iter().collect(),
            Err(error) => {
                // ConversationMemory is never a reason to fail a turn.
                eprintln!("memory recall failed: {error}");
                Vec::new()
            }
        }
    }
}

impl ContextHooks for ConversationMemory {
    /// The user's new messages run the lexical channel on their text.
    fn on_input<'a>(
        &'a self,
        history: &'a [AgentMessage],
        input: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        Box::pin(async move {
            let Some(message) = user_text(input) else {
                return Vec::new();
            };
            self.recall_or_log(history, Some(&message), &[]).await
        })
    }

    /// A tool batch attaches memories for the files it touched.
    fn on_tool_replies<'a>(
        &'a self,
        history: &'a [AgentMessage],
        replies: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        Box::pin(async move {
            let paths = touched_paths(self.workspace.as_ref(), replies);
            if paths.is_empty() {
                return Vec::new();
            }
            self.recall_or_log(history, None, &paths).await
        })
    }
}

/// The text of the user's own messages among `input`, joined. Slash commands
/// are not searched.
fn user_text(input: &[AgentMessage]) -> Option<String> {
    let texts: Vec<&str> = input
        .iter()
        .filter_map(|entry| match entry.chat_message() {
            Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
            _ => None,
        })
        .filter(|text| !text.trim_start().starts_with('/'))
        .collect();
    (!texts.is_empty()).then(|| texts.join("\n\n"))
}

/// The checkout-relative paths the tools in `replies` reported touching, read
/// from the tool results' metadata.
fn touched_paths(workspace: &dyn Workspace, replies: &[AgentMessage]) -> Vec<String> {
    let touched = replies
        .iter()
        .filter(|entry| matches!(entry.chat_message(), Some(chat_record::Message::Tool(_))))
        .filter_map(|entry| entry.metadata.as_ref())
        .flat_map(|metadata| metadata.touched.iter());
    let mut paths = Vec::new();
    for touched in touched {
        if let Some(relative) = workspace.relative(&touched.path) {
            if !paths.contains(&relative) {
                paths.push(relative);
            }
        }
    }
    paths
}

/// What the active context window already shows the model: the records in
/// its recalls and in successful `search_memory` results. They are not
/// shown again, and the recalls' size counts against the window's retrieval
/// budget. Compaction summarizes them away, which frees both.
pub fn surfaced<'a>(window: impl IntoIterator<Item = &'a AgentMessage>) -> Surfaced {
    let mut surfaced = Surfaced::default();
    let mut search_calls = HashSet::new();
    for entry in window {
        match &entry.payload {
            Some(agent_message::Payload::Extension(extension))
                if extension.extension == MEMORY_EXTENSION && extension.kind == RECALL_KIND =>
            {
                surfaced.ids.extend(recall_ids(extension));
                surfaced.chars += extension
                    .model_message
                    .as_ref()
                    .map_or(0, |message| message.content.len());
            }
            Some(agent_message::Payload::Chat(chat)) => match &chat.message {
                Some(chat_record::Message::Assistant(assistant)) => {
                    search_calls.extend(
                        assistant
                            .tool_calls
                            .iter()
                            .filter(|call| {
                                call.function
                                    .as_ref()
                                    .is_some_and(|function| function.name == SearchMemoryTool::NAME)
                            })
                            .map(|call| call.id.clone()),
                    );
                }
                Some(chat_record::Message::Tool(tool))
                    if chat.success && search_calls.contains(&tool.tool_call_id) =>
                {
                    surfaced.ids.extend(search_result_ids(&tool.content));
                }
                _ => {}
            },
            _ => {}
        }
    }
    surfaced
}

/// The record ids one recall carries.
fn recall_ids(extension: &ExtensionRecord) -> impl Iterator<Item = String> + '_ {
    extension
        .data
        .as_ref()
        .and_then(|data| data.fields.get("ids"))
        .and_then(|value| match &value.kind {
            Some(prost_types::value::Kind::StringValue(ids)) => Some(ids.as_str()),
            _ => None,
        })
        .unwrap_or_default()
        .split(',')
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// The recall record for `selected`, or `None` when nothing was admitted.
fn recall_record(selected: &[Selected]) -> Option<AgentMessage> {
    if selected.is_empty() {
        return None;
    }
    let rendered: Vec<String> = selected
        .iter()
        .map(|selected| selected.rendered.clone())
        .collect();
    let ids = selected
        .iter()
        .map(|selected| selected.record.id.as_str())
        .collect::<Vec<_>>()
        .join(",");
    Some(memory_record(
        RECALL_KIND,
        &[("ids", ids)],
        Some(render_recall(&rendered)),
    ))
}

/// An extension record this memory writes into the conversation.
pub(super) fn memory_record(
    kind: &str,
    fields: &[(&str, String)],
    model_message: Option<String>,
) -> AgentMessage {
    let data = prost_types::Struct {
        fields: fields
            .iter()
            .map(|(key, value)| {
                (
                    (*key).to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue(value.clone())),
                    },
                )
            })
            .collect(),
    };
    AgentMessage {
        payload: Some(agent_message::Payload::Extension(ExtensionRecord {
            extension: MEMORY_EXTENSION.to_string(),
            kind: kind.to_string(),
            data: Some(data),
            model_message: model_message.map(|content| UserMessage {
                content,
                checkpoint: None,
            }),
            // Recall is for the turn it precedes; compaction drops it with that turn.
            survives_compaction: false,
        })),
        ..AgentMessage::default()
    }
}

#[cfg(test)]
#[path = "conversation_memory_test.rs"]
mod tests;
