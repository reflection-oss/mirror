use std::time::Duration;

use mirror_protocol::chat_record;
use mirror_protocol::ChatRecord;
use mirror_sandbox::CallOutcome;
use mirror_sandbox::CallReply;
use tokio::sync::oneshot::error::RecvError;

use super::ToolOutput;

/// What became of one tool call sent to a sandbox.
#[derive(Debug)]
pub enum SandboxCallOutcome<T = ToolOutput> {
    Completed(T),
    /// Nothing was dispatched; repeating the call is safe.
    Rejected(String),
    Failed(String),
    Unknown(String),
}

impl<T> SandboxCallOutcome<T> {
    /// A bounded name for the outcome, for span and log fields.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Completed(_) => "ok",
            Self::Rejected(_) => "rejected",
            Self::Failed(_) => "failed",
            Self::Unknown(_) => "unknown",
        }
    }
}

impl SandboxCallOutcome {
    /// What `sandboxd`'s reply to a call that ran under `timeout` means for the
    /// tool. A reply that never arrived leaves the outcome unknown.
    pub fn from_reply(reply: Result<CallReply, RecvError>, timeout: Duration) -> Self {
        let outcome = match reply {
            Ok(CallReply::Outcome(outcome)) => outcome,
            Ok(CallReply::StreamEnded(reason)) => {
                return Self::Unknown(format!(
                    "the sandboxd stream ended before the call reported: {reason}"
                ))
            }
            Err(_) => return Self::Unknown("the sandboxd session dropped the call".to_string()),
        };
        match outcome {
            CallOutcome::Completed(record) => match tool_output(*record) {
                Ok(output) => Self::Completed(output),
                Err(detail) => Self::Failed(detail),
            },
            CallOutcome::Rejected(message) => {
                Self::Rejected(format!("sandboxd rejected the call: {message}"))
            }
            CallOutcome::Failed(message) => Self::Failed(message),
            CallOutcome::Cancelled(true) => Self::Failed(
                "the call was cancelled and its runner process group stopped".to_string(),
            ),
            CallOutcome::Cancelled(false) => Self::Unknown(
                "the call was cancelled but its runner process group was not observed stopped"
                    .to_string(),
            ),
            CallOutcome::DeadlineExceeded(true) => Self::Failed(format!(
                "the call exceeded its {timeout:?} deadline and its runner process group stopped"
            )),
            CallOutcome::DeadlineExceeded(false) => Self::Unknown(format!(
                "the call exceeded its {timeout:?} deadline but its runner process group was not \
                 observed stopped"
            )),
        }
    }
}

/// `sandboxd` reports a tool's result as the chat record the tool produced.
fn tool_output(record: ChatRecord) -> Result<ToolOutput, String> {
    let Some(chat_record::Message::Tool(message)) = record.message else {
        return Err("sandboxd returned a non-tool message".to_string());
    };
    Ok(ToolOutput {
        content: message.content,
        display: record.display,
        success: record.success,
        touched: Vec::new(),
        exit_code: None,
    })
}

#[cfg(test)]
#[path = "sandbox_call_outcome_test.rs"]
mod tests;
