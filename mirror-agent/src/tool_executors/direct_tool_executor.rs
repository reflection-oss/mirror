//! In-process execution of typed Mirror tools.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures_util::FutureExt;
use mirror_protocol::ToolCall;
use mirror_protocol::ToolInventory;
use uuid::Uuid;

use crate::ExecutionContext;

use super::tool_batch::{run_batch, BatchCall};
use crate::tool_executors::{ToolExecutor, ToolExecutorError};
use crate::tools::{AnyTool, ToolDefinition, ToolOutput};

/// Executes tool batches directly in the agent process.
pub struct DirectToolExecutor {
    tools: Vec<Arc<dyn AnyTool>>,
    indices: HashMap<String, usize>,
    tool_visibility: Option<Arc<ToolVisibility>>,
}

type ToolVisibility = dyn Fn(&str, &ExecutionContext) -> bool + Send + Sync;

impl DirectToolExecutor {
    /// Own the tools used for direct execution.
    #[must_use]
    pub fn new(tools: Vec<Arc<dyn AnyTool>>) -> Self {
        let indices = tools
            .iter()
            .enumerate()
            .map(|(index, tool)| (tool.name().to_string(), index))
            .collect();
        Self {
            tools,
            indices,
            tool_visibility: None,
        }
    }

    /// Filter model-visible contracts without removing tools needed by in-flight calls.
    #[must_use]
    pub fn with_tool_visibility(
        mut self,
        visible: impl Fn(&str, &ExecutionContext) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.tool_visibility = Some(Arc::new(visible));
        self
    }

    fn execute_call(
        &self,
        call: &ToolCall,
        allowed: bool,
        context: &ExecutionContext,
    ) -> BatchCall {
        let tool = call
            .function
            .as_ref()
            .and_then(|function| self.indices.get(&function.name))
            .and_then(|index| self.tools.get(*index))
            .cloned();
        let concurrency_safe = tool.as_ref().is_some_and(|tool| tool.is_concurrency_safe());
        let call = call.clone();
        let context = context
            .clone()
            .with_web_provider(None)
            .with_tool_call_id(call.id.clone());
        let execution = async move {
            if call.r#type != "function" {
                return ToolOutput::error(format!("unsupported tool call type {:?}", call.r#type));
            }
            let Some(function) = &call.function else {
                return ToolOutput::error("missing tool function");
            };
            let Some(tool) = tool else {
                return ToolOutput::failure(format!("Unknown tool: {}", function.name));
            };
            if !allowed {
                return ToolOutput::error("Tool call rejected.");
            }
            if !context.tool_enabled(&function.name) {
                return ToolOutput::error(format!(
                    "Tool disabled for this chat: {}",
                    function.name
                ));
            }
            if let Err(error) = context.check_pre_tool_hooks() {
                return ToolOutput::error(error.to_string());
            }
            let started = Instant::now();
            let mut output = tool.execute(&function.arguments, &context).await;
            output.duration = Some(started.elapsed());
            output
        }
        .boxed();
        BatchCall {
            concurrency_safe,
            execution,
        }
    }
}

impl ToolExecutor for DirectToolExecutor {
    fn tools(&self) -> &[Arc<dyn AnyTool>] {
        &self.tools
    }

    fn definitions_for(&self, context: &ExecutionContext) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|tool| context.tool_enabled(tool.name()))
            .filter(|tool| {
                self.tool_visibility
                    .as_ref()
                    .is_none_or(|visible| visible(tool.name(), context))
            })
            .map(|tool| tool.definition())
            .collect()
    }

    fn inventory(
        &self,
        context: &ExecutionContext,
        include_nested: bool,
    ) -> Pin<Box<dyn Future<Output = ToolInventory> + Send>> {
        let registered = self.tools.clone();
        let visibility = self.tool_visibility.clone();
        let context = context.clone();
        Box::pin(async move {
            let mut tools = Vec::new();
            for tool in registered {
                tools.extend(tool.inventory(&context, include_nested).await);
            }
            for tool in &mut tools {
                if visibility
                    .as_ref()
                    .is_some_and(|visible| !visible(&tool.name, &context))
                {
                    tool.unavailable_reason = Some("Not configured for this chat.".into());
                }
            }
            ToolInventory { tools }
        })
    }

    fn execute(
        &self,
        _batch_id: Uuid,
        calls: &[ToolCall],
        allowed: &[bool],
        context: &ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolOutput>, ToolExecutorError>> + Send>> {
        let calls = calls
            .iter()
            .zip(allowed)
            .map(|(call, allowed)| self.execute_call(call, *allowed, context))
            .collect();
        Box::pin(run_batch(calls).map(Ok))
    }
}

#[cfg(test)]
#[path = "direct_tool_executor_test.rs"]
mod tests;
