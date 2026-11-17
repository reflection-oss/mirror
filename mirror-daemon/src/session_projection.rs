//! A bounded, disposable view built from the worker's single event stream.

use std::collections::{BTreeMap, VecDeque};

use mirror_protocol::{
    agent_message, chat_record, runtime_delta, session_message, AgentMessage, ReplayDone,
    RuntimeDelta, SessionMessage, TaskStatus,
};
use prost::Message;
use tonic::Status;

use crate::merge_display_update;

/// Replay recent committed records and the current live state, never a second store.
#[derive(Debug, Default)]
pub struct SessionProjection {
    history: VecDeque<AgentMessage>,
    history_bytes: usize,
    dropped_entries: u64,
    has_user_message: bool,
    preview: VecDeque<SessionMessage>,
    preview_bytes: usize,
    turn: Option<RuntimeDelta>,
    turn_error: Option<RuntimeDelta>,
    context_usage: Option<RuntimeDelta>,
    stream_usage: Option<RuntimeDelta>,
    compaction: Option<RuntimeDelta>,
    tasks: BTreeMap<String, RuntimeDelta>,
    task_bytes: usize,
    approvals: BTreeMap<String, SessionMessage>,
    approval_bytes: usize,
}

/// Maximum command or live event size before transport and projection admission.
pub const MAX_FRAME_BYTES: usize = 10 * 1024 * 1024;
const HISTORY_BYTES: usize = 4 * MAX_FRAME_BYTES;
const HISTORY_ENTRIES: usize = 128;

impl SessionProjection {
    /// Start a replay at the history offset preceding the records to load.
    pub fn new(dropped_entries: u64, has_user_message: bool) -> Self {
        Self {
            dropped_entries,
            has_user_message,
            ..Self::default()
        }
    }

    /// Apply one worker event; display caches never terminate the worker.
    pub fn apply(&mut self, message: &SessionMessage) -> Result<(), Status> {
        match &message.payload {
            Some(session_message::Payload::AgentMessage(entry)) => {
                if message.stream_final
                    || matches!(
                        entry.chat_message(),
                        Some(chat_record::Message::Assistant(_))
                    )
                {
                    self.clear_preview();
                }
                if matches!(&entry.payload, Some(agent_message::Payload::ContextWindow(window)) if window.compaction.is_some())
                {
                    self.compaction = None;
                }
                self.record(entry.clone());
            }
            Some(session_message::Payload::RuntimeDelta(delta)) => self.runtime(message, delta),
            Some(session_message::Payload::HitlRequest(request)) => {
                let call = request
                    .call
                    .as_ref()
                    .filter(|call| !call.id.is_empty())
                    .ok_or_else(|| Status::data_loss("approval request has no call identity"))?;
                if let Some(previous) = self.approvals.get(&call.id) {
                    if previous != message {
                        return Err(Status::data_loss("worker reused an approval call identity"));
                    }
                    return Ok(());
                }
                if self.approvals.len() >= 128
                    || self.approval_bytes + message.encoded_len() > MAX_FRAME_BYTES
                {
                    return Err(Status::resource_exhausted(
                        "pending approvals exceeded their bound",
                    ));
                }
                self.approval_bytes += message.encoded_len();
                self.approvals.insert(call.id.clone(), message.clone());
            }
            Some(session_message::Payload::HitlResolved(resolved)) => {
                if let Some(previous) = self.approvals.remove(&resolved.call_id) {
                    self.approval_bytes -= previous.encoded_len();
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Capture a replay while the owning actor reserves the live subscription.
    pub fn replay(&self, session_id: &str) -> Vec<SessionMessage> {
        let mut messages = self
            .history
            .iter()
            .cloned()
            .map(|entry| SessionMessage {
                payload: Some(session_message::Payload::AgentMessage(entry)),
                ..SessionMessage::default()
            })
            .collect::<Vec<_>>();
        messages.push(SessionMessage {
            payload: Some(session_message::Payload::ReplayDone(ReplayDone {
                conversation_id: session_id.to_owned(),
                dropped_entries: self.dropped_entries,
                has_user_message: self.has_user_message,
            })),
            ..SessionMessage::default()
        });
        messages.extend(
            self.turn
                .iter()
                .chain(self.turn_error.iter())
                .chain(self.context_usage.iter())
                .chain(self.compaction.iter())
                .chain(self.tasks.values())
                .map(|delta| SessionMessage {
                    payload: Some(session_message::Payload::RuntimeDelta(delta.clone())),
                    ..SessionMessage::default()
                }),
        );
        messages.extend(self.preview.iter().cloned());
        if let Some(usage) = &self.stream_usage {
            messages.push(SessionMessage {
                payload: Some(session_message::Payload::RuntimeDelta(usage.clone())),
                ..SessionMessage::default()
            });
        }
        messages.extend(self.approvals.values().cloned());
        messages
    }

    /// Retain the newest committed records within the replay budget.
    pub fn record(&mut self, entry: AgentMessage) {
        if self.history.iter().any(|previous| previous.id == entry.id) {
            return;
        }
        self.has_user_message |=
            matches!(entry.chat_message(), Some(chat_record::Message::User(_)));
        self.history_bytes += entry.encoded_len();
        self.history.push_back(entry);
        while self.history_bytes > HISTORY_BYTES || self.history.len() > HISTORY_ENTRIES {
            if let Some(removed) = self.history.pop_front() {
                self.history_bytes -= removed.encoded_len();
                self.dropped_entries += 1;
            }
        }
    }

    fn runtime(&mut self, message: &SessionMessage, delta: &RuntimeDelta) {
        match &delta.payload {
            Some(runtime_delta::Payload::TurnStarted(_)) => {
                self.turn = Some(delta.clone());
                self.turn_error = None;
            }
            Some(
                runtime_delta::Payload::TurnCompleted(_)
                | runtime_delta::Payload::TurnInterrupted(_),
            ) => {
                self.turn = None;
                self.compaction = None;
                self.clear_preview();
            }
            Some(runtime_delta::Payload::StreamAborted(_)) => self.clear_preview(),
            Some(runtime_delta::Payload::ContextUsage(_)) => {
                self.context_usage = Some(delta.clone())
            }
            Some(runtime_delta::Payload::StreamUsage(_)) => self.stream_usage = Some(delta.clone()),
            Some(runtime_delta::Payload::CompactionStarted(_)) => {
                self.compaction = Some(delta.clone())
            }
            Some(runtime_delta::Payload::TurnError(_)) => {
                self.compaction = None;
                self.turn_error = Some(delta.clone());
            }
            Some(runtime_delta::Payload::Rewound(rewound)) => {
                let mut remaining = rewound.removed_entries;
                while remaining > 0 {
                    let Some(entry) = self.history.pop_back() else {
                        break;
                    };
                    self.history_bytes -= entry.encoded_len();
                    remaining -= 1;
                }
                self.dropped_entries = self.dropped_entries.saturating_sub(remaining);
                self.clear_preview();
            }
            Some(runtime_delta::Payload::TaskUpdate(update)) => {
                if let Some(previous) = self.tasks.remove(&update.id) {
                    self.task_bytes -= previous.encoded_len();
                }
                if update.status == TaskStatus::Running as i32 {
                    let mut update = update.clone();
                    // Child messages are deltas, not a snapshot of the task's current status.
                    update.messages.clear();
                    let id = update.id.clone();
                    let latest = RuntimeDelta {
                        payload: Some(runtime_delta::Payload::TaskUpdate(update)),
                        ..Default::default()
                    };
                    self.task_bytes += latest.encoded_len();
                    self.tasks.insert(id, latest);
                }
                while self.tasks.len() > 128 || self.task_bytes > MAX_FRAME_BYTES {
                    if let Some((_, removed)) = self.tasks.pop_first() {
                        self.task_bytes -= removed.encoded_len();
                    }
                }
            }
            Some(
                runtime_delta::Payload::StreamContent(_)
                | runtime_delta::Payload::StreamReasoning(_)
                | runtime_delta::Payload::StreamRefusal(_)
                | runtime_delta::Payload::StreamToolCall(_),
            ) => {
                self.preview(message);
            }
            _ => {}
        }
    }

    fn preview(&mut self, message: &SessionMessage) {
        let merged = self.preview.back_mut().is_some_and(|previous| {
            let size = previous.encoded_len();
            if size + message.encoded_len() > 64 * 1024 || !merge_display_update(previous, message)
            {
                return false;
            }
            self.preview_bytes += previous.encoded_len() - size;
            true
        });
        if !merged {
            self.preview_bytes += message.encoded_len();
            self.preview.push_back(message.clone());
        }
        // Late clients get a bounded tail; live clients and committed history remain complete.
        while self.preview_bytes > MAX_FRAME_BYTES || self.preview.len() > 4096 {
            if let Some(removed) = self.preview.pop_front() {
                self.preview_bytes -= removed.encoded_len();
            }
        }
    }

    fn clear_preview(&mut self) {
        self.preview.clear();
        self.preview_bytes = 0;
        self.stream_usage = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use mirror_protocol::{ContextUsage, StreamContent, StreamUsage, TaskUpdate};
    use pretty_assertions::assert_eq;

    fn event(payload: runtime_delta::Payload) -> SessionMessage {
        SessionMessage {
            payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(payload),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[test]
    fn long_stream_replays_content_without_terminating_the_worker() {
        let mut projection = SessionProjection::default();
        let chunk = event(runtime_delta::Payload::StreamContent(StreamContent {
            content: "x".into(),
        }));
        for _ in 0..5000 {
            projection.apply(&chunk).expect("stream continues");
        }
        assert_eq!(
            projection.replay("session")[1],
            event(runtime_delta::Payload::StreamContent(StreamContent {
                content: "x".repeat(5000)
            }))
        );
    }

    #[test]
    fn oversized_preview_retains_a_bounded_tail_and_accepts_completion() {
        let mut projection = SessionProjection::default();
        let chunk = event(runtime_delta::Payload::StreamContent(StreamContent {
            content: "x".repeat(MAX_FRAME_BYTES / 2),
        }));
        projection.apply(&chunk).expect("first chunk");
        projection.apply(&chunk).expect("second chunk");
        let tail = event(runtime_delta::Payload::StreamContent(StreamContent {
            content: "tail".into(),
        }));
        projection
            .apply(&tail)
            .expect("stream continues after eviction");
        let replay = projection.replay("session");
        assert_matches!(replay.last(), Some(last) if last == &tail);
        assert!(replay.iter().map(Message::encoded_len).sum::<usize>() <= MAX_FRAME_BYTES);
        projection
            .apply(&event(runtime_delta::Payload::TurnCompleted(())))
            .expect("completion");
        assert_eq!(projection.replay("session").len(), 1);
    }

    #[test]
    fn replay_does_not_restore_resolved_approvals() {
        let mut projection = SessionProjection::default();
        let request = SessionMessage {
            payload: Some(session_message::Payload::HitlRequest(
                mirror_protocol::HitlRequest {
                    call: Some(mirror_protocol::ToolCall {
                        id: "call".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        projection.apply(&request).expect("pending approval");
        assert_eq!(projection.replay("session").last(), Some(&request));
        projection
            .apply(&SessionMessage {
                payload: Some(session_message::Payload::HitlResolved(
                    mirror_protocol::HitlResolved {
                        call_id: "call".into(),
                        approved: true,
                    },
                )),
                ..Default::default()
            })
            .expect("resolution");
        assert_eq!(projection.replay("session").len(), 1);
    }

    #[test]
    fn rewind_updates_the_authoritative_history_prefix() {
        let mut projection = SessionProjection::default();
        for index in 0..140 {
            projection.record(AgentMessage {
                id: index.to_string(),
                ..Default::default()
            });
        }
        projection
            .apply(&event(runtime_delta::Payload::Rewound(
                mirror_protocol::Rewound {
                    entry_id: "1".into(),
                    removed_entries: 139,
                },
            )))
            .expect("rewind");
        assert_matches!(&projection.replay("session")[0].payload,
            Some(session_message::Payload::ReplayDone(done)) if done.dropped_entries == 1);
    }

    #[test]
    fn attach_replays_current_runtime_state_and_forgets_completed_tasks() {
        let mut projection = SessionProjection::default();
        let turn = event(runtime_delta::Payload::TurnStarted(()));
        let context = event(runtime_delta::Payload::ContextUsage(ContextUsage {
            tokens: 42,
            window: 100,
            ..ContextUsage::default()
        }));
        let compaction = event(runtime_delta::Payload::CompactionStarted(()));
        let task = event(runtime_delta::Payload::TaskUpdate(TaskUpdate {
            id: "tool".into(),
            tool_name: "bash".into(),
            status: TaskStatus::Running as i32,
            ..Default::default()
        }));
        let usage = event(runtime_delta::Payload::StreamUsage(StreamUsage {
            completion_tokens: 5,
            elapsed_seconds: 1.0,
        }));
        projection.apply(&turn).expect("turn");
        projection.apply(&context).expect("context");
        projection.apply(&compaction).expect("compaction");
        projection.apply(&task).expect("task");
        projection.apply(&usage).expect("usage");
        assert_eq!(
            &projection.replay("session")[1..],
            &[turn, context.clone(), compaction, task, usage]
        );
        projection
            .apply(&event(runtime_delta::Payload::TaskUpdate(TaskUpdate {
                id: "tool".into(),
                status: TaskStatus::Succeeded as i32,
                ..Default::default()
            })))
            .expect("task finished");
        projection
            .apply(&event(runtime_delta::Payload::TurnCompleted(())))
            .expect("turn finished");
        assert_eq!(&projection.replay("session")[1..], &[context]);
    }
}
