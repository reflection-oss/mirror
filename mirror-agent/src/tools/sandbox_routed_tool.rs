//! A workspace tool whose calls run in a sandbox instead of on this machine.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use mirror_protocol::FunctionCall;
use mirror_protocol::ToolCall;
use mirror_sandbox::MAX_MESSAGE_BYTES;
use prost::Message;

use super::bash_tool::DEFAULT_BASH_TIMEOUT;
use super::{AnyTool, SandboxCallOutcome, SandboxToolExecutor, ToolDefinition, ToolOutput};
use crate::ExecutionContext;

/// Presents a native tool's definition and runs its calls through a sandbox.
pub struct SandboxRoutedTool {
    tool: Arc<dyn AnyTool>,
    executor: Arc<dyn SandboxToolExecutor>,
}

impl SandboxRoutedTool {
    pub fn new(tool: Arc<dyn AnyTool>, executor: Arc<dyn SandboxToolExecutor>) -> Self {
        Self { tool, executor }
    }
}

impl AnyTool for SandboxRoutedTool {
    fn name(&self) -> &str {
        self.tool.name()
    }

    fn needs_approval(&self) -> bool {
        self.tool.needs_approval()
    }

    fn definition(&self) -> ToolDefinition {
        self.tool.definition()
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let call = function_call(self.name(), arguments);
            if call.encoded_len() > MAX_MESSAGE_BYTES {
                return ToolOutput::error(format!(
                    "tool request exceeds {MAX_MESSAGE_BYTES} bytes"
                ));
            }
            match self.executor.execute(&call, tool_timeout(&call)).await {
                SandboxCallOutcome::Completed(output) => output,
                SandboxCallOutcome::Rejected(detail) => {
                    ToolOutput::error(format!("sandbox tool was not executed: {detail}"))
                }
                SandboxCallOutcome::Failed(detail) => ToolOutput::error(detail),
                SandboxCallOutcome::Unknown(detail) => ToolOutput::error(format!(
                    "sandbox tool outcome is unknown; the call may still be running: {detail}"
                )),
            }
        })
    }
}

fn function_call(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        }),
        ..ToolCall::default()
    }
}

const DEFAULT_TOOL_TIMEOUT: Duration = Duration::from_secs(30);
const BASH_TIMEOUT_MARGIN: Duration = Duration::from_secs(5);
const MAX_TOOL_TIMEOUT: Duration = DEFAULT_BASH_TIMEOUT.saturating_add(BASH_TIMEOUT_MARGIN);

/// Bash gets the native tool's own deadline, which model arguments cannot
/// change, plus time for the runner to report it.
fn tool_timeout(call: &ToolCall) -> Duration {
    let Some(function) = call
        .function
        .as_ref()
        .filter(|function| function.name == "bash")
    else {
        return DEFAULT_TOOL_TIMEOUT;
    };
    serde_json::from_str::<serde_json::Value>(&function.arguments)
        .ok()
        .and_then(|arguments| arguments.get("timeout_seconds")?.as_f64())
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
        .and_then(|timeout| timeout.checked_add(BASH_TIMEOUT_MARGIN))
        .unwrap_or(DEFAULT_BASH_TIMEOUT + BASH_TIMEOUT_MARGIN)
        .min(MAX_TOOL_TIMEOUT)
}

#[cfg(test)]
#[path = "sandbox_routed_tool_test.rs"]
mod tests;
