//! In-process execution of typed Mirror tools.

use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::FutureExt;
use mirror_protocol::ToolCall;
use uuid::Uuid;

use crate::ExecutionContext;

use super::{AnyTool, ToolExecutor, ToolExecutorError, ToolOutput};

/// Executes tool batches directly in the agent process.
pub struct DirectToolExecutor {
    tools: Vec<Arc<dyn AnyTool>>,
    indices: HashMap<String, usize>,
}

impl DirectToolExecutor {
    /// Own the tools used for direct execution.
    #[must_use]
    pub fn new(tools: Vec<Arc<dyn AnyTool>>) -> Self {
        let indices = tools
            .iter()
            .enumerate()
            .map(|(index, tool)| (tool.name().to_string(), index))
            .collect();
        Self { tools, indices }
    }

    fn execute_call(
        &self,
        call: &ToolCall,
        allowed: bool,
        context: &ExecutionContext,
    ) -> impl Future<Output = ToolOutput> + Send + 'static {
        let tool = call
            .function
            .as_ref()
            .and_then(|function| self.indices.get(&function.name))
            .and_then(|index| self.tools.get(*index))
            .cloned();
        let call = call.clone();
        let context = context.clone();
        async move {
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
                return ToolOutput::error("Tool call rejected by the user.");
            }
            tool.execute(&function.arguments, &context).await
        }
    }
}

impl ToolExecutor for DirectToolExecutor {
    fn tools(&self) -> &[Arc<dyn AnyTool>] {
        &self.tools
    }

    fn execute(
        &self,
        _batch_id: Uuid,
        calls: &[ToolCall],
        allowed: &[bool],
        context: &ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolOutput>, ToolExecutorError>> + Send>> {
        // Calls run one after another, in order, so a call sees the effects of
        // the calls before it.
        let executions: Vec<_> = calls
            .iter()
            .zip(allowed)
            .map(|(call, allowed)| self.execute_call(call, *allowed, context))
            .collect();
        Box::pin(async move {
            let mut outputs = Vec::with_capacity(executions.len());
            for execution in executions {
                outputs.push(
                    AssertUnwindSafe(execution)
                        .catch_unwind()
                        .await
                        .unwrap_or_else(|payload| ToolOutput::error(panic_message(payload))),
                );
            }
            Ok(outputs)
        })
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "tool panicked".to_string())
}
