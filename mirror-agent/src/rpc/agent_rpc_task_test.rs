use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;

#[test]
fn translates_task_updates_to_runtime_deltas() {
    let started_at = std::time::UNIX_EPOCH + std::time::Duration::from_secs(12);
    let ended_at = started_at + std::time::Duration::from_secs(3);

    let responses = service_responses(ReactServiceEvent::TaskUpdated(TaskInfo {
        id: "1234".to_string(),
        tool_name: "bash".to_string(),
        display: Some("working".to_string()),
        status: RuntimeTaskStatus::Succeeded,
        started_at,
        ended_at: Some(ended_at),
        detached: true,
    }));

    assert_matches!(
        responses.as_slice(),
        [SessionMessage {
            payload: Some(session_message::Payload::RuntimeDelta(RuntimeDelta {
                payload: Some(runtime_delta::Payload::TaskUpdate(TaskUpdate {
                    id,
                    tool_name,
                    status,
                    detached: true,
                    ..
                })),
            })),
            ..
        }] if id == "1234"
            && tool_name == "bash"
            && *status == ProtocolTaskStatus::Succeeded as i32
    );
}

#[test]
fn preserves_task_timestamps_before_the_unix_epoch() {
    let update = TaskUpdate::from(TaskInfo {
        id: "1234".to_string(),
        tool_name: "bash".to_string(),
        display: None,
        status: RuntimeTaskStatus::Running,
        started_at: std::time::UNIX_EPOCH - std::time::Duration::from_secs(1),
        ended_at: None,
        detached: false,
    });

    assert_eq!(
        update.started_at,
        Some(prost_types::Timestamp {
            seconds: -1,
            nanos: 0,
        })
    );
}
