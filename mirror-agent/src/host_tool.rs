//! Generic tools executed by the client process through the agent session.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use mirror_protocol::HostToolDefinition;
use mirror_protocol::ToolAvailability;
use serde_json::Map;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::tools::ToolOutput;
use crate::tools::{ToolDefinition, ToolError};

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

/// One client-hosted tool invocation awaiting a result.
pub struct HostToolRequest {
    pub name: String,
    pub arguments: String,
    pub timeout: Duration,
    pub response: oneshot::Sender<ToolOutput>,
}

impl HostTool {
    /// Create a client-hosted tool using the service request channel.
    pub fn new(
        config: HostToolDefinition,
        requests: mpsc::UnboundedSender<HostToolRequest>,
    ) -> Result<Self, ToolError> {
        let timeout = Duration::try_from_secs_f64(config.timeout_seconds)
            .map_err(|_| ToolError::InvalidArguments("invalid host tool timeout".into()))?;
        let parameters = serde_json::from_str(&config.parameters_json)
            .map_err(|_| ToolError::InvalidArguments("invalid host tool parameters".into()))?;
        Ok(Self {
            name: config.name,
            description: config.description,
            parameters,
            needs_approval: config.needs_approval,
            timeout,
            requests,
        })
    }
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

    fn inventory<'a>(
        &'a self,
        context: &'a ExecutionContext,
        _include_nested: bool,
    ) -> Pin<Box<dyn Future<Output = Vec<ToolAvailability>> + Send + 'a>> {
        Box::pin(async move {
            vec![ToolAvailability {
                name: self.name.clone(),
                description: self.description.clone(),
                source: "host".into(),
                enabled: context.tool_enabled(&self.name),
                ..Default::default()
            }]
        })
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
