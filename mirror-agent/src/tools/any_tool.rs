//! Argument-type-erased tool execution.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

use crate::tools::ToolDefinition;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

use super::Tool;
use super::ToolError;

/// The object-safe interface shared by every typed tool.
pub trait AnyTool: Send + Sync {
    /// The model-facing tool name.
    fn name(&self) -> &str;

    /// Whether calls require human approval.
    fn needs_approval(&self) -> bool;

    /// The model-facing function definition.
    fn definition(&self) -> ToolDefinition;

    /// Whether this tool and its descendants support remote execution.
    fn supports_remote(&self) -> bool {
        false
    }

    /// Parse and execute remotely without a local fallback.
    fn execute_remote<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
        _remote: &'a super::RemoteToolExecutor,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async {
            ToolOutput::error(format!(
                "tool {:?} has no remote implementation",
                self.name()
            ))
        })
    }

    /// Parse and execute one model-generated JSON argument object.
    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>>;
}

/// Reject ambiguous tool names before constructing an executor.
pub fn validate_tools<'a>(
    tools: impl IntoIterator<Item = &'a dyn AnyTool>,
) -> Result<(), ToolError> {
    let mut names = HashSet::new();
    for tool in tools {
        if !names.insert(tool.name()) {
            return Err(ToolError::InvalidArguments(format!(
                "duplicate tool name {:?}",
                tool.name()
            )));
        }
    }
    Ok(())
}

impl<T: Tool> AnyTool for T {
    fn name(&self) -> &str {
        Tool::name(self)
    }

    fn needs_approval(&self) -> bool {
        Tool::needs_approval(self)
    }

    fn supports_remote(&self) -> bool {
        Tool::supports_remote(self)
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: Tool::name(self).to_string(),
            description: Tool::description(self).to_string(),
            parameters: Tool::parameters(self),
        }
    }

    fn execute<'a>(
        &'a self,
        arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let args = match parse_arguments(arguments, Tool::name(self)) {
                Ok(args) => args,
                Err(output) => return output,
            };
            self.run(args, context).await.unwrap_or_else(tool_error)
        })
    }

    fn execute_remote<'a>(
        &'a self,
        arguments: &'a str,
        context: &'a ExecutionContext,
        remote: &'a super::RemoteToolExecutor,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let args = match parse_arguments(arguments, Tool::name(self)) {
                Ok(args) => args,
                Err(output) => return output,
            };
            self.run_remote(args, context, remote)
                .await
                .unwrap_or_else(tool_error)
        })
    }
}

fn parse_arguments<T: serde::de::DeserializeOwned>(
    arguments: &str,
    name: &str,
) -> Result<T, ToolOutput> {
    let arguments = if arguments.trim().is_empty() {
        "{}"
    } else {
        arguments
    };
    serde_json::from_str(arguments)
        .map_err(|error| ToolOutput::failure(format!("Invalid arguments for tool {name}: {error}")))
}

fn tool_error(error: ToolError) -> ToolOutput {
    match error {
        ToolError::Execution(message) => ToolOutput::error(message),
        error => ToolOutput::error(error.to_string()),
    }
}
