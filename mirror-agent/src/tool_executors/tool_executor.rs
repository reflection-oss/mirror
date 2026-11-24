//! Ordered execution boundary for one committed tool batch.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::tools::{AnyTool, ToolDefinition, ToolOutput};
use mirror_protocol::FunctionCall;
use mirror_protocol::ToolCall;
use mirror_protocol::ToolInventory;
use thiserror::Error;
use uuid::Uuid;

use crate::ApprovalDecision;
use crate::ApprovalKind;
use crate::ExecutionContext;

/// Executes one immutable model tool batch.
pub trait ToolExecutor: Send + Sync {
    /// Tools whose contracts are exposed by this executor.
    fn tools(&self) -> &[Arc<dyn AnyTool>];

    /// Model-facing definitions derived from the executor's tools.
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools().iter().map(|tool| tool.definition()).collect()
    }

    /// Definitions visible to the model under the current execution settings.
    fn definitions_for(&self, context: &ExecutionContext) -> Vec<ToolDefinition> {
        self.definitions()
            .into_iter()
            .filter(|tool| context.tool_enabled(&tool.name))
            .collect()
    }

    /// Authoritative registered tools, with opt-in discovery of nested tools.
    fn inventory(
        &self,
        context: &ExecutionContext,
        include_nested: bool,
    ) -> Pin<Box<dyn Future<Output = ToolInventory> + Send>> {
        let registered = self.tools().to_vec();
        let context = context.clone();
        Box::pin(async move {
            let mut tools = Vec::new();
            for tool in registered {
                tools.extend(tool.inventory(&context, include_nested).await);
            }
            ToolInventory { tools }
        })
    }

    /// Whether one named tool requires approval before execution.
    fn needs_approval(&self, name: &str) -> bool {
        self.tools()
            .iter()
            .find(|tool| tool.name() == name)
            .is_some_and(|tool| tool.needs_approval())
    }

    fn always_asks(&self, name: &str) -> bool {
        self.tools()
            .iter()
            .find(|tool| tool.name() == name)
            .is_some_and(|tool| tool.always_asks())
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
        let context = context.for_tool_execution();
        let tool = self
            .tools()
            .iter()
            .find(|tool| tool.name() == name)
            .cloned();
        let call = ToolCall {
            id: context
                .tool_call_id()
                .map_or_else(|| Uuid::new_v4().to_string(), str::to_owned),
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
            if !context.tool_enabled(&name) {
                return Ok(ToolOutput::error(format!(
                    "Tool disabled for this chat: {name}"
                )));
            }
            let Some(tool) = tool else {
                return Ok(ToolOutput::failure(format!("Unknown tool: {name}")));
            };
            if let Some(reason) = context.review_tool(&call).await? {
                return Ok(ToolOutput::error(reason));
            }
            let kind = if tool.always_asks() {
                Some(ApprovalKind::AlwaysAsk)
            } else if tool.needs_approval() {
                Some(ApprovalKind::Ordinary)
            } else {
                None
            };
            if let Some(kind) = kind {
                match context
                    .approval_decision(&call, kind)
                    .await
                    .map(ApprovalDecision::acknowledge)
                {
                    Ok(decision) if !decision.approved => {
                        return Ok(ToolOutput::error(
                            decision.denial_message().unwrap_or("Tool call rejected."),
                        ));
                    }
                    Ok(_) => {}
                    Err(_) => return Ok(ToolOutput::error("Tool approval is unavailable.")),
                }
            }
            let outputs = execution.await?;
            context.check_pre_tool_hooks()?;
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
    #[error(transparent)]
    PreTool(#[from] crate::hooks::PreToolError),
    #[error("tool execution failed: {0}")]
    Execution(String),
    #[error("tool executor returned {actual} results for {expected} calls")]
    ResultCount { expected: usize, actual: usize },
}
