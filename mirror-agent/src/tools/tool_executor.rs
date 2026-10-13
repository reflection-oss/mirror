//! Ordered execution boundary for one committed tool batch.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::tools::ToolDefinition;
use crate::tools::ToolOutput;
use mirror_protocol::FunctionCall;
use mirror_protocol::ToolCall;
use thiserror::Error;
use uuid::Uuid;

use crate::ExecutionContext;

use super::AnyTool;

/// Executes one immutable model tool batch.
pub trait ToolExecutor: Send + Sync {
    /// Tools whose contracts are exposed by this executor.
    fn tools(&self) -> &[Arc<dyn AnyTool>];

    /// Model-facing definitions derived from the executor's tools.
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools().iter().map(|tool| tool.definition()).collect()
    }

    /// Whether one named tool requires approval before execution.
    fn needs_approval(&self, name: &str) -> bool {
        self.tools()
            .iter()
            .find(|tool| tool.name() == name)
            .is_some_and(|tool| tool.needs_approval())
    }

    /// Return an owned future yielding one output per call in model-call order.
    fn execute(
        &self,
        batch_id: Uuid,
        calls: &[ToolCall],
        allowed: &[bool],
        context: &ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolOutput>, ToolExecutorError>> + Send>>;

    /// Execute one registered tool with normal approval checks.
    fn call_tool(
        &self,
        name: String,
        arguments: String,
        context: ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolExecutorError>> + Send>> {
        let tool = self
            .tools()
            .iter()
            .find(|tool| tool.name() == name)
            .cloned();
        let call = ToolCall {
            id: Uuid::new_v4().to_string(),
            r#type: "function".to_string(),
            function: Some(FunctionCall {
                name: name.clone(),
                arguments,
            }),
            ..ToolCall::default()
        };
        let execution = self.execute(
            Uuid::new_v4(),
            std::slice::from_ref(&call),
            &[true],
            &context,
        );
        Box::pin(async move {
            let Some(tool) = tool else {
                return Ok(ToolOutput::failure(format!("Unknown tool: {name}")));
            };
            if tool.needs_approval() {
                match context.request_approval(&call).await {
                    Ok(true) => {}
                    Ok(false) => return Ok(ToolOutput::error("Tool call rejected by the user.")),
                    Err(_) => return Ok(ToolOutput::error("Tool approval is unavailable.")),
                }
            }
            let outputs = execution.await?;
            let actual = outputs.len();
            let [output]: [ToolOutput; 1] =
                outputs
                    .try_into()
                    .map_err(|_| ToolExecutorError::ResultCount {
                        expected: 1,
                        actual,
                    })?;
            Ok(output)
        })
    }
}

/// A failure at the tool-batch execution boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolExecutorError {
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error("tool executor returned {actual} results for {expected} calls")]
    ResultCount { expected: usize, actual: usize },
}
