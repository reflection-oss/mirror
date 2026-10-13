use crate::Store;
use crate::StoreError;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use thiserror::Error;
use tokio::sync::mpsc;

use super::ContextHooks;
use super::TurnEnd;
use crate::AgentEvent;
use crate::ExecutionContext;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReactContextError {
    #[error("context closed")]
    Closed,
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("saved conversation history is invalid: {0}")]
    InvalidHistory(String),
}

pub struct ReactContext {
    session_id: String,
    entries: Vec<AgentMessage>,
    store: Box<dyn Store>,
    append_failed: bool,
    events: mpsc::UnboundedSender<AgentEvent>,
    pending: mpsc::UnboundedReceiver<AgentMessage>,
    pending_sender: ReactContextInput,
    execution: ExecutionContext,
    hooks: Vec<Box<dyn ContextHooks>>,
}

#[derive(Clone)]
pub struct ReactContextInput {
    sender: mpsc::UnboundedSender<AgentMessage>,
}

impl ReactContextInput {
    pub fn push(&self, message: AgentMessage) -> Result<(), ReactContextError> {
        self.sender
            .send(message)
            .map_err(|_| ReactContextError::Closed)
    }
}

impl ReactContext {
    pub fn new(
        session_id: impl Into<String>,
        entries: Vec<AgentMessage>,
        store: Box<dyn Store>,
        execution: ExecutionContext,
    ) -> (Self, mpsc::UnboundedReceiver<AgentEvent>) {
        let (events, event_receiver) = mpsc::unbounded_channel();
        let (pending_sender, pending) = mpsc::unbounded_channel();
        (
            Self {
                session_id: session_id.into(),
                entries,
                store,
                append_failed: false,
                events,
                pending,
                pending_sender: ReactContextInput {
                    sender: pending_sender,
                },
                execution,
                hooks: Vec::new(),
            },
            event_receiver,
        )
    }

    /// Give the context the hooks that follow what it commits.
    #[must_use]
    pub fn with_hooks(mut self, hooks: Vec<Box<dyn ContextHooks>>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Fork the active context without persisting the child or publishing its events.
    pub async fn in_memory_fork(
        &mut self,
        execution: ExecutionContext,
    ) -> Result<Self, ReactContextError> {
        let session_id = uuid::Uuid::new_v4().simple().to_string();
        let entries = crate::compaction::active_window(self.entries.iter())
            .into_iter()
            .cloned()
            .collect();
        let mut store = super::InMemoryStore::from_entries(self.session_id.clone(), entries);
        store.fork(&self.session_id, None, &session_id)?;
        let entries = store.load(&session_id, false).await?;
        execution
            .input()
            .set_inference_config(self.execution.inference_config());
        let (context, _) = Self::new(session_id, entries, Box::new(store), execution);
        Ok(context)
    }

    pub async fn history(
        &mut self,
        offset: u64,
        limit: u32,
    ) -> Result<(Vec<AgentMessage>, u64), StoreError> {
        self.store
            .messages_page(&self.session_id, offset, limit)
            .await
    }

    pub fn entries(&self) -> &[AgentMessage] {
        &self.entries
    }

    pub fn strip_before_active_window(&mut self) -> Result<(), ReactContextError> {
        let start = self
            .entries
            .iter()
            .rposition(|entry| {
                matches!(
                    entry.payload,
                    Some(mirror_protocol::agent_message::Payload::ContextWindow(_))
                )
            })
            .ok_or_else(|| {
                ReactContextError::InvalidHistory(
                    "nonempty conversation is missing a context window".to_string(),
                )
            })?;
        self.entries.drain(..start);
        Ok(())
    }

    pub fn system_prompt(&self) -> Result<&str, ReactContextError> {
        self.entries
            .iter()
            .rev()
            .find_map(|entry| match &entry.payload {
                Some(mirror_protocol::agent_message::Payload::ContextWindow(window)) => {
                    Some(window.system_prompt.as_str())
                }
                _ => None,
            })
            .ok_or_else(|| {
                ReactContextError::InvalidHistory(
                    "nonempty conversation is missing a context window".to_string(),
                )
            })
    }

    pub(crate) fn execution(&self) -> &ExecutionContext {
        &self.execution
    }

    pub fn input(&self) -> ReactContextInput {
        self.pending_sender.clone()
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn flush_pending(&mut self) -> Vec<AgentMessage> {
        let mut pending = Vec::new();
        while let Ok(message) = self.pending.try_recv() {
            pending.push(message);
        }
        pending
    }

    /// Commit `event`'s entries, then whatever the hooks add after them,
    /// publishing both in order.
    pub async fn push(&mut self, mut event: AgentEvent) -> Result<(), ReactContextError> {
        let mut followed = Vec::new();
        match &mut event {
            AgentEvent::Entries(entries) => {
                self.append(entries).await?;
                followed = follow(&self.hooks, &self.entries, entries).await;
                if !followed.is_empty() {
                    self.append(&mut followed).await?;
                }
            }
            AgentEvent::StreamFinal { entry, .. } => {
                self.append(std::slice::from_mut(entry.as_mut())).await?;
            }
            AgentEvent::StreamContent(_)
            | AgentEvent::StreamReasoning(_)
            | AgentEvent::StreamRefusal(_)
            | AgentEvent::StreamToolCall(_)
            | AgentEvent::StreamUsage(_)
            | AgentEvent::StreamAborted
            | AgentEvent::CompactionStarted
            | AgentEvent::ContextUsage { .. } => {}
        }
        self.publish(event);
        if !followed.is_empty() {
            self.publish(AgentEvent::Entries(followed));
        }
        Ok(())
    }

    /// Run every hook's `on_turn_end`: the turn finished cleanly.
    ///
    /// Takes `&mut self` so the future is `Send`: the store is `Send` but not
    /// `Sync`, so a shared borrow of the context could not cross threads.
    pub async fn run_turn_end_hooks(&mut self, turn: TurnEnd<'_>) {
        for hook in &self.hooks {
            hook.on_turn_end(&self.entries, turn).await;
        }
    }

    /// Publish a live view without appending or rewriting durable history.
    pub fn publish(&self, event: AgentEvent) {
        let _ = self.events.send(event);
    }

    pub(crate) fn append_failed(&self) -> bool {
        self.append_failed
    }

    async fn append(&mut self, entries: &mut [AgentMessage]) -> Result<(), ReactContextError> {
        if self.append_failed {
            return Err(StoreError::Persistence(
                "conversation append failed; reopen the session before writing".to_string(),
            )
            .into());
        }
        if let Err(error) = self
            .store
            .append(&self.session_id, crate::head(&self.entries), entries)
            .await
        {
            self.append_failed = true;
            return Err(error.into());
        }
        self.entries.extend_from_slice(entries);
        Ok(())
    }
}

/// Records the hooks add after `committed`: the user's own messages drive
/// `on_input`, tool replies drive `on_tool_replies`.
async fn follow(
    hooks: &[Box<dyn ContextHooks>],
    history: &[AgentMessage],
    committed: &[AgentMessage],
) -> Vec<AgentMessage> {
    let input = committed
        .iter()
        .filter(|entry| is_user_input(entry))
        .cloned()
        .collect::<Vec<_>>();
    let replies = committed
        .iter()
        .filter(|entry| is_tool_reply(entry))
        .cloned()
        .collect::<Vec<_>>();
    let mut followed = Vec::new();
    for hook in hooks {
        if !input.is_empty() {
            followed.extend(hook.on_input(history, &input).await);
        }
        if !replies.is_empty() {
            followed.extend(hook.on_tool_replies(history, &replies).await);
        }
    }
    followed
}

/// A message the user wrote, as opposed to context injected beside it,
/// which carries a `display`.
fn is_user_input(entry: &AgentMessage) -> bool {
    matches!(
        &entry.payload,
        Some(agent_message::Payload::Chat(chat))
            if chat.display.is_none()
                && matches!(chat.message, Some(chat_record::Message::User(_)))
    )
}

fn is_tool_reply(entry: &AgentMessage) -> bool {
    matches!(entry.chat_message(), Some(chat_record::Message::Tool(_)))
}

#[cfg(test)]
#[path = "context_test.rs"]
mod tests;
