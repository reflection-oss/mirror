//! Consumer-paced display work, not a queue entry for every provider fragment.

use std::collections::VecDeque;
use std::sync::Arc;

use mirror_protocol::{chat_record, runtime_delta, session_message, SessionMessage};
use prost::Message;

use crate::MAX_FRAME_BYTES;

/// One attachment's bounded pending display; execution never travels through it.
#[derive(Debug, Default)]
pub struct DisplayUpdates {
    messages: VecDeque<Arc<SessionMessage>>,
    bytes: usize,
    snapshot_needed: bool,
}

const MAX_UPDATES: usize = 1024;
const MERGED_BYTES: usize = 64 * 1024;

impl DisplayUpdates {
    /// Return whether an empty mailbox became ready, coalescing obsolete display work.
    pub fn push(&mut self, message: Arc<SessionMessage>) -> bool {
        let wake = self.messages.is_empty() && !self.snapshot_needed;
        if self.snapshot_needed {
            return false;
        }
        if supersedes_preview(&message) {
            let start = preview_start(self.messages.iter().map(AsRef::as_ref));
            let mut preview = self.messages.split_off(start);
            preview.retain(|previous| !is_preview(previous));
            self.messages.extend(preview);
            self.bytes = self.messages.iter().map(|entry| entry.encoded_len()).sum();
        }
        self.append(message);
        if self.bytes > MAX_FRAME_BYTES || self.messages.len() > MAX_UPDATES {
            self.messages.clear();
            self.bytes = 0;
            self.snapshot_needed = true;
        }
        wake
    }

    /// Drain once; `None` requests an authoritative snapshot instead of partial history.
    pub fn take(&mut self) -> Option<VecDeque<Arc<SessionMessage>>> {
        self.bytes = 0;
        if std::mem::take(&mut self.snapshot_needed) {
            None
        } else {
            Some(std::mem::take(&mut self.messages))
        }
    }

    fn append(&mut self, message: Arc<SessionMessage>) {
        let size = message.encoded_len();
        if let Some(index) = merge_index(self.messages.iter().map(AsRef::as_ref), &message) {
            let previous = &mut self.messages[index];
            let previous_size = previous.encoded_len();
            if previous_size + size <= MERGED_BYTES
                && merge_display_update(Arc::make_mut(previous), &message)
            {
                self.bytes = self.bytes - previous_size + previous.encoded_len();
                return;
            }
        }
        self.bytes += size;
        self.messages.push_back(message);
    }
}

fn is_preview(message: &SessionMessage) -> bool {
    matches!(&message.payload,
        Some(session_message::Payload::RuntimeDelta(delta)) if matches!(&delta.payload,
            Some(runtime_delta::Payload::StreamContent(_)
                | runtime_delta::Payload::StreamReasoning(_)
                | runtime_delta::Payload::StreamRefusal(_)
                | runtime_delta::Payload::StreamToolCall(_)
                | runtime_delta::Payload::StreamUsage(_))))
}

fn is_usage(message: &SessionMessage) -> bool {
    matches!(&message.payload,
        Some(session_message::Payload::RuntimeDelta(delta)) if matches!(&delta.payload,
            Some(runtime_delta::Payload::StreamUsage(_))))
}

// Only independent child streams and cumulative usage can move past each other.
fn merge_index<'a>(
    mut messages: impl DoubleEndedIterator<Item = &'a SessionMessage> + ExactSizeIterator,
    next: &SessionMessage,
) -> Option<usize> {
    messages.rposition(|previous| !commutes(previous, next))
}

fn commutes(previous: &SessionMessage, next: &SessionMessage) -> bool {
    if previous.stream_final || next.stream_final {
        return false;
    }
    let (
        Some(session_message::Payload::RuntimeDelta(before)),
        Some(session_message::Payload::RuntimeDelta(after)),
    ) = (&previous.payload, &next.payload)
    else {
        return false;
    };
    if before.turn_id != after.turn_id {
        return false;
    }
    if is_preview(previous) && is_preview(next) && is_usage(previous) != is_usage(next) {
        return true;
    }
    matches!((&before.payload, &after.payload),
        (Some(runtime_delta::Payload::TaskUpdate(before)), Some(runtime_delta::Payload::TaskUpdate(after)))
            if before.id != after.id && !before.messages.is_empty() && !after.messages.is_empty())
}

// Closed streams may retain partial output as decorations, even after later commits.
fn preview_start<'a>(
    mut messages: impl DoubleEndedIterator<Item = &'a SessionMessage> + ExactSizeIterator,
) -> usize {
    messages
        .rposition(|message| {
            supersedes_preview(message)
                || matches!(&message.payload,
                Some(session_message::Payload::RuntimeDelta(delta)) if matches!(&delta.payload,
                    Some(runtime_delta::Payload::StreamAborted(_)
                        | runtime_delta::Payload::TurnStarted(_)
                        | runtime_delta::Payload::TurnCompleted(_)
                        | runtime_delta::Payload::TurnInterrupted(_))))
        })
        .map_or(0, |index| index + 1)
}

fn supersedes_preview(message: &SessionMessage) -> bool {
    message.stream_final
        || matches!(&message.payload,
            Some(session_message::Payload::AgentMessage(entry))
                if matches!(entry.chat_message(), Some(chat_record::Message::Assistant(_))))
        || matches!(&message.payload,
            Some(session_message::Payload::RuntimeDelta(delta)) if matches!(&delta.payload,
                Some(runtime_delta::Payload::Rewound(_))))
}

/// Merge compatible display fragments without crossing lifecycle or record boundaries.
pub fn merge_display_update(previous: &mut SessionMessage, next: &SessionMessage) -> bool {
    if previous.stream_final
        || next.stream_final
        || previous.tokens_per_second != next.tokens_per_second
    {
        return false;
    }
    let (
        Some(session_message::Payload::RuntimeDelta(previous)),
        Some(session_message::Payload::RuntimeDelta(next)),
    ) = (&mut previous.payload, &next.payload)
    else {
        return false;
    };
    if previous.turn_id != next.turn_id {
        return false;
    }
    match (&mut previous.payload, &next.payload) {
        (
            Some(runtime_delta::Payload::StreamContent(previous)),
            Some(runtime_delta::Payload::StreamContent(next)),
        ) => previous.content.push_str(&next.content),
        (
            Some(runtime_delta::Payload::StreamReasoning(previous)),
            Some(runtime_delta::Payload::StreamReasoning(next)),
        ) => previous.reasoning.push_str(&next.reasoning),
        (
            Some(runtime_delta::Payload::StreamRefusal(previous)),
            Some(runtime_delta::Payload::StreamRefusal(next)),
        ) => previous.refusal.push_str(&next.refusal),
        (
            Some(runtime_delta::Payload::StreamToolCall(previous)),
            Some(runtime_delta::Payload::StreamToolCall(next)),
        ) if previous.index == next.index => {
            previous.name.push_str(&next.name);
            previous.arguments.push_str(&next.arguments);
        }
        (
            Some(runtime_delta::Payload::StreamUsage(previous)),
            Some(runtime_delta::Payload::StreamUsage(next)),
        ) => *previous = *next,
        (
            Some(runtime_delta::Payload::ContextUsage(previous)),
            Some(runtime_delta::Payload::ContextUsage(next)),
        ) => *previous = *next,
        (
            Some(runtime_delta::Payload::TaskUpdate(previous)),
            Some(runtime_delta::Payload::TaskUpdate(next)),
        ) if previous.id == next.id
            && previous.status == next.status
            && previous.tool_name == next.tool_name
            && previous.detached == next.detached
            // Metadata-only updates drive parent task UI; child-bearing updates do not.
            && previous.messages.is_empty() == next.messages.is_empty() =>
        {
            let mut messages = std::mem::take(&mut previous.messages);
            for message in &next.messages {
                if supersedes_preview(message) {
                    let start = preview_start(messages.iter());
                    let mut preview = messages.split_off(start);
                    preview.retain(|previous| !is_preview(previous));
                    messages.extend(preview);
                }
                let merged = merge_index(messages.iter(), message)
                    .is_some_and(|index| merge_display_update(&mut messages[index], message));
                if !merged {
                    messages.push(message.clone());
                }
            }
            *previous = next.clone();
            previous.messages = messages;
        }
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use mirror_protocol::{
        AgentMessage, HitlRequest, RuntimeDelta, StreamContent, StreamUsage, TaskUpdate,
    };
    use pretty_assertions::assert_eq;

    fn event(payload: runtime_delta::Payload) -> Arc<SessionMessage> {
        Arc::new(SessionMessage {
            payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(payload),
                ..Default::default()
            })),
            ..Default::default()
        })
    }

    fn text(content: &str) -> Arc<SessionMessage> {
        event(runtime_delta::Payload::StreamContent(StreamContent {
            content: content.into(),
        }))
    }

    #[test]
    fn token_and_usage_bursts_are_one_pending_text_and_latest_usage() {
        let mut updates = DisplayUpdates::default();
        assert!(updates.push(text("start")));
        for index in 0..4096 {
            updates.push(text("x"));
            updates.push(event(runtime_delta::Payload::StreamUsage(StreamUsage {
                completion_tokens: index + 1,
                elapsed_seconds: 1.0,
            })));
        }
        assert!(!updates.push(text("end")));
        let messages = updates
            .take()
            .expect("text never consumes a slot per token");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0], text(&format!("start{}end", "x".repeat(4096))));
        assert_matches!(&messages[1].payload,
            Some(session_message::Payload::RuntimeDelta(delta)) if matches!(&delta.payload,
                Some(runtime_delta::Payload::StreamUsage(usage)) if usage.completion_tokens == 4096));
        assert!(updates.push(text("next batch")));
    }

    #[test]
    fn committed_output_replaces_unsent_previews_without_erasing_approvals() {
        let mut updates = DisplayUpdates::default();
        let approval = Arc::new(SessionMessage {
            payload: Some(session_message::Payload::HitlRequest(HitlRequest::default())),
            ..Default::default()
        });
        let committed = Arc::new(SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: "committed".into(),
                ..Default::default()
            })),
            stream_final: true,
            ..Default::default()
        });
        updates.push(text("obsolete prefix"));
        updates.push(approval.clone());
        updates.push(text("obsolete suffix"));
        updates.push(committed.clone());
        assert_eq!(updates.take(), Some(VecDeque::from([approval, committed])));
    }

    #[test]
    fn child_text_is_coalesced_without_losing_child_messages() {
        let mut updates = DisplayUpdates::default();
        updates.push(event(runtime_delta::Payload::TaskUpdate(TaskUpdate {
            id: "child".into(),
            messages: vec![(*text("hello")).clone()],
            ..Default::default()
        })));
        updates.push(event(runtime_delta::Payload::TaskUpdate(TaskUpdate {
            id: "child".into(),
            display: Some("current".into()),
            messages: vec![(*text(" world")).clone()],
            ..Default::default()
        })));
        let messages = updates.take().expect("coalesced child");
        assert_eq!(messages.len(), 1);
        assert_matches!(&messages[0].payload,
            Some(session_message::Payload::RuntimeDelta(delta)) if matches!(&delta.payload,
                Some(runtime_delta::Payload::TaskUpdate(task))
                    if task.display.as_deref() == Some("current") && task.messages == vec![(*text("hello world")).clone()]));
    }

    fn child(id: &str, messages: Vec<SessionMessage>) -> Arc<SessionMessage> {
        event(runtime_delta::Payload::TaskUpdate(TaskUpdate {
            id: id.into(),
            tool_name: "subagent".into(),
            status: mirror_protocol::TaskStatus::Running as i32,
            messages,
            ..Default::default()
        }))
    }

    #[test]
    fn interleaved_children_coalesce_their_own_text_and_usage() {
        let mut updates = DisplayUpdates::default();
        let usage = event(runtime_delta::Payload::StreamUsage(StreamUsage {
            completion_tokens: 42,
            elapsed_seconds: 1.0,
        }));
        for _ in 0..2048 {
            updates.push(child("first", vec![(*text("a")).clone(), (*usage).clone()]));
            updates.push(child(
                "second",
                vec![(*text("b")).clone(), (*usage).clone()],
            ));
        }
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([
                child(
                    "first",
                    vec![(*text(&"a".repeat(2048))).clone(), (*usage).clone()]
                ),
                child(
                    "second",
                    vec![(*text(&"b".repeat(2048))).clone(), (*usage).clone()]
                ),
            ]))
        );
    }

    #[test]
    fn child_commit_replaces_only_its_own_pending_preview() {
        let mut updates = DisplayUpdates::default();
        let committed = SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: "child record".into(),
                ..Default::default()
            })),
            stream_final: true,
            ..Default::default()
        };
        updates.push(child("first", vec![(*text("obsolete")).clone()]));
        updates.push(child("second", vec![(*text("retained")).clone()]));
        updates.push(child("first", vec![committed.clone()]));
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([
                child("first", vec![committed]),
                child("second", vec![(*text("retained")).clone()]),
            ]))
        );
    }

    #[test]
    fn child_messages_do_not_swallow_parent_task_metadata() {
        let mut updates = DisplayUpdates::default();
        let metadata = child("child", vec![]);
        let content = child("child", vec![(*text("content")).clone()]);
        updates.push(metadata.clone());
        updates.push(content.clone());
        updates.push(metadata.clone());
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([metadata.clone(), content, metadata]))
        );
    }

    fn committed(id: &str) -> Arc<SessionMessage> {
        Arc::new(SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: id.into(),
                ..Default::default()
            })),
            stream_final: true,
            ..Default::default()
        })
    }

    #[test]
    fn interrupted_and_aborted_output_survives_completion_and_later_commits() {
        let mut updates = DisplayUpdates::default();
        let interrupted = event(runtime_delta::Payload::TurnInterrupted(()));
        let aborted = event(runtime_delta::Payload::StreamAborted(()));
        let completed = event(runtime_delta::Payload::TurnCompleted(()));
        let error = event(runtime_delta::Payload::TurnError(
            mirror_protocol::TurnError {
                message: "provider failed".into(),
            },
        ));
        updates.push(text("interrupted partial"));
        updates.push(interrupted.clone());
        updates.push(completed.clone());
        updates.push(text("aborted partial"));
        updates.push(aborted.clone());
        updates.push(error.clone());
        updates.push(completed.clone());
        updates.push(text("superseded preview"));
        updates.push(committed("next turn"));
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([
                text("interrupted partial"),
                interrupted,
                completed.clone(),
                text("aborted partial"),
                aborted,
                error,
                completed,
                committed("next turn"),
            ]))
        );
    }

    #[test]
    fn child_partial_output_survives_closed_streams_and_a_later_commit() {
        let mut updates = DisplayUpdates::default();
        let interrupted = (*event(runtime_delta::Payload::TurnInterrupted(()))).clone();
        let aborted = (*event(runtime_delta::Payload::StreamAborted(()))).clone();
        let completed = (*event(runtime_delta::Payload::TurnCompleted(()))).clone();
        updates.push(child("child", vec![(*text("interrupted partial")).clone()]));
        updates.push(child("child", vec![interrupted.clone(), completed.clone()]));
        updates.push(child("child", vec![(*text("aborted partial")).clone()]));
        updates.push(child("child", vec![aborted.clone(), completed.clone()]));
        updates.push(child("child", vec![(*text("superseded preview")).clone()]));
        updates.push(child("child", vec![(*committed("next turn")).clone()]));
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([child(
                "child",
                vec![
                    (*text("interrupted partial")).clone(),
                    interrupted,
                    completed.clone(),
                    (*text("aborted partial")).clone(),
                    aborted,
                    completed,
                    (*committed("next turn")).clone(),
                ]
            )]))
        );
    }

    #[test]
    fn different_turn_identities_do_not_merge() {
        let mut updates = DisplayUpdates::default();
        let next = Arc::new(SessionMessage {
            payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                turn_id: Some("next".into()),
                payload: Some(runtime_delta::Payload::StreamContent(StreamContent {
                    content: "next".into(),
                })),
            })),
            ..Default::default()
        });
        updates.push(text("previous"));
        updates.push(next.clone());
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([text("previous"), next]))
        );
    }

    #[test]
    fn lifecycle_events_remain_between_their_stream_fragments() {
        let mut updates = DisplayUpdates::default();
        let started = event(runtime_delta::Payload::TurnStarted(()));
        updates.push(text("before"));
        updates.push(started.clone());
        updates.push(text("after"));
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([text("before"), started, text("after")]))
        );
    }

    #[test]
    fn retained_history_gap_requires_a_snapshot_and_then_accepts_live_updates() {
        let mut updates = DisplayUpdates::default();
        for index in 0..=MAX_UPDATES {
            updates.push(Arc::new(SessionMessage {
                payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                    id: index.to_string(),
                    ..Default::default()
                })),
                ..Default::default()
            }));
        }
        assert!(!updates.push(text("snapshot covers this")));
        assert_eq!(updates.take(), None);
        assert!(updates.push(text("after snapshot")));
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([text("after snapshot")]))
        );
    }

    #[test]
    fn tool_fragments_merge_without_crossing_call_indices() {
        let mut updates = DisplayUpdates::default();
        let first = event(runtime_delta::Payload::StreamToolCall(
            mirror_protocol::StreamToolCall {
                index: 0,
                name: "ba".into(),
                arguments: "{".into(),
            },
        ));
        let second = event(runtime_delta::Payload::StreamToolCall(
            mirror_protocol::StreamToolCall {
                index: 0,
                name: "sh".into(),
                arguments: "}".into(),
            },
        ));
        let other = event(runtime_delta::Payload::StreamToolCall(
            mirror_protocol::StreamToolCall {
                index: 1,
                name: "read".into(),
                arguments: "{}".into(),
            },
        ));
        updates.push(first);
        updates.push(second);
        updates.push(other.clone());
        assert_eq!(
            updates.take(),
            Some(VecDeque::from([
                event(runtime_delta::Payload::StreamToolCall(
                    mirror_protocol::StreamToolCall {
                        index: 0,
                        name: "bash".into(),
                        arguments: "{}".into(),
                    }
                )),
                other,
            ]))
        );
    }

    #[test]
    fn pending_bytes_are_bounded_even_when_few_events_are_large() {
        let mut updates = DisplayUpdates::default();
        updates.push(text(&"x".repeat(MAX_FRAME_BYTES / 2)));
        updates.push(text(&"x".repeat(MAX_FRAME_BYTES / 2)));
        assert_eq!(updates.take(), None);
    }
}
