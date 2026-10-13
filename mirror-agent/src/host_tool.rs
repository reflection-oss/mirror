//! Generic tools executed by the client process through the agent session.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use std::time::TryFromFloatSecsError;

use serde::Deserialize;
use serde_json::Map;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::tools::ToolDefinition;
use crate::tools::ToolOutput;

use crate::tools::AnyTool;
use crate::ExecutionContext;

/// A model-facing tool whose implementation lives in the client process.
pub struct HostTool {
    name: String,
    description: String,
    parameters: Map<String, Value>,
    needs_approval: bool,
    timeout: Duration,
    requests: mpsc::UnboundedSender<HostToolRequest>,
}

/// One immutable host tool definition supplied at worker startup.
#[derive(Debug, Deserialize)]
pub struct HostToolConfig {
    name: String,
    description: String,
    parameters: Map<String, Value>,
    needs_approval: bool,
    #[serde(default = "default_host_tool_timeout_seconds")]
    timeout_seconds: f64,
}

/// One client-hosted tool invocation awaiting a result.
pub struct HostToolRequest {
    pub name: String,
    pub arguments: String,
    pub timeout: Duration,
    pub response: oneshot::Sender<ToolOutput>,
}

const DEFAULT_HOST_TOOL_TIMEOUT_SECONDS: f64 = 300.0;

impl HostTool {
    /// Create a client-hosted tool using the service request channel.
    pub fn new(
        config: HostToolConfig,
        requests: mpsc::UnboundedSender<HostToolRequest>,
    ) -> Result<Self, TryFromFloatSecsError> {
        let timeout = Duration::try_from_secs_f64(config.timeout_seconds)?;
        Ok(Self {
            name: config.name,
            description: config.description,
            parameters: config.parameters,
            needs_approval: config.needs_approval,
            timeout,
            requests,
        })
    }
}

fn default_host_tool_timeout_seconds() -> f64 {
    DEFAULT_HOST_TOOL_TIMEOUT_SECONDS
}

impl AnyTool for HostTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn needs_approval(&self) -> bool {
        self.needs_approval
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let (response, receiver) = oneshot::channel();
            if self
                .requests
                .send(HostToolRequest {
                    name: self.name.clone(),
                    arguments: arguments.to_string(),
                    timeout: self.timeout,
                    response,
                })
                .is_err()
            {
                return ToolOutput::error("host tool session is unavailable");
            }
            receiver
                .await
                .unwrap_or_else(|_| ToolOutput::error("host tool session closed"))
        })
    }
}

#[cfg(test)]
#[path = "host_tool_test.rs"]
mod tests;
