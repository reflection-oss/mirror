use assert_matches::assert_matches;
use mirror_protocol::ToolMessage;
use pretty_assertions::assert_eq;

use super::*;

const TIMEOUT: Duration = Duration::from_secs(30);

#[test]
fn from_reply_returns_the_tool_message_of_a_completed_call() {
    let record = ChatRecord {
        message: Some(chat_record::Message::Tool(ToolMessage {
            content: "listed".to_string(),
            ..ToolMessage::default()
        })),
        success: true,
        ..ChatRecord::default()
    };

    let outcome = SandboxCallOutcome::from_reply(
        Ok(CallReply::Outcome(CallOutcome::Completed(Box::new(record)))),
        TIMEOUT,
    );

    assert_matches!(outcome, SandboxCallOutcome::Completed(output) => {
        assert_eq!(output.content, "listed");
        assert!(output.success);
    });
}

#[test]
fn from_reply_fails_a_completed_call_without_a_tool_message() {
    let outcome = SandboxCallOutcome::from_reply(
        Ok(CallReply::Outcome(CallOutcome::Completed(Box::default()))),
        TIMEOUT,
    );

    assert_matches!(
        outcome,
        SandboxCallOutcome::Failed(detail) if detail.contains("non-tool message")
    );
}

#[test]
fn from_reply_is_unknown_when_a_cancelled_runner_was_not_seen_to_stop() {
    let outcome = SandboxCallOutcome::from_reply(
        Ok(CallReply::Outcome(CallOutcome::Cancelled(false))),
        TIMEOUT,
    );

    assert_matches!(outcome, SandboxCallOutcome::Unknown(_));
}

#[test]
fn from_reply_is_unknown_when_the_stream_ended_first() {
    let outcome =
        SandboxCallOutcome::from_reply(Ok(CallReply::StreamEnded("closed".to_string())), TIMEOUT);

    assert_matches!(outcome, SandboxCallOutcome::Unknown(detail) if detail.ends_with("closed"));
}
