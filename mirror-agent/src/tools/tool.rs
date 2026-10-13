//! Typed tool execution primitives.

use std::future::Future;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use thiserror::Error;

use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// A tool with typed JSON arguments and asynchronous execution.
pub trait Tool: Send + Sync {
    type Args: DeserializeOwned + Serialize + Send;

    /// The model-facing tool name.
    fn name(&self) -> &str;

    /// A natural-language description of what the tool does.
    fn description(&self) -> &str;

    /// The JSON Schema used to validate model-generated arguments.
    fn parameters(&self) -> Map<String, Value>;

    /// Whether calls require human approval.
    fn needs_approval(&self) -> bool {
        false
    }

    /// Whether this tool and its descendants support remote execution.
    fn supports_remote(&self) -> bool {
        false
    }

    /// Execute remotely; unsupported tools fail without touching the host workspace.
    fn run_remote(
        &self,
        _args: Self::Args,
        _context: &ExecutionContext,
        _remote: &super::RemoteToolExecutor,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        async {
            Err(ToolError::Execution(format!(
                "tool {:?} has no remote implementation",
                self.name()
            )))
        }
    }

    /// Execute one validated call.
    fn run(
        &self,
        args: Self::Args,
        context: &ExecutionContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send;
}

/// A typed failure produced while validating or running a tool.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolError {
    #[error("no connection for {0}")]
    NotConnected(String),
    #[error("invalid tool arguments: {0}")]
    InvalidArguments(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct EchoArgs {
        text: String,
    }

    struct EchoTool;

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct EmptyArgs {}

    struct EmptyTool;

    impl Tool for EchoTool {
        type Args = EchoArgs;

        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo text."
        }

        fn parameters(&self) -> Map<String, Value> {
            Map::from_iter([
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    serde_json::json!({"text": {"type": "string"}}),
                ),
                ("required".to_string(), serde_json::json!(["text"])),
                ("additionalProperties".to_string(), Value::Bool(false)),
            ])
        }

        async fn run(
            &self,
            args: Self::Args,
            context: &ExecutionContext,
        ) -> Result<ToolOutput, ToolError> {
            assert!(!context.tool_permissions().allow_outside_workspace());
            Ok(ToolOutput::success(args.text))
        }
    }

    impl Tool for EmptyTool {
        type Args = EmptyArgs;

        fn name(&self) -> &str {
            "empty"
        }

        fn description(&self) -> &str {
            "Accept no arguments."
        }

        fn parameters(&self) -> Map<String, Value> {
            Map::from_iter([
                ("type".to_string(), Value::String("object".to_string())),
                ("properties".to_string(), Value::Object(Map::new())),
                ("additionalProperties".to_string(), Value::Bool(false)),
            ])
        }

        async fn run(
            &self,
            _args: Self::Args,
            _context: &ExecutionContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("ran"))
        }
    }

    #[test]
    fn definition_exposes_function_schema() {
        let definition = crate::tools::AnyTool::definition(&EchoTool);

        assert_eq!(definition.name, "echo");
        assert_eq!(definition.description, "Echo text.");
        assert_eq!(definition.parameters["additionalProperties"], false);
    }

    #[test]
    fn executor_preserves_tool_definitions() {
        use crate::tools::{AnyTool, DirectToolExecutor, ReadTool, ToolExecutor};

        let workspace = tempfile::tempdir().expect("workspace");
        let read = ReadTool::new(workspace.path()).expect("read tool");
        let raw = AnyTool::definition(&read);
        let executor = DirectToolExecutor::new(vec![
            std::sync::Arc::new(read),
            std::sync::Arc::new(EchoTool),
        ]);

        let definitions = executor.definitions();
        assert_eq!(definitions[0], raw);
        assert_eq!(
            definitions[0].description,
            "Read a UTF-8 text file from the workspace."
        );
        assert_eq!(definitions[1], AnyTool::definition(&EchoTool));
        assert_eq!(executor.definitions(), definitions);
    }

    #[tokio::test]
    async fn execute_validates_and_runs_typed_arguments() {
        let result = crate::tools::AnyTool::execute(
            &EchoTool,
            r#"{"text":"hello"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_ne!(result, ToolOutput::error("hello"));
        assert_eq!(result, ToolOutput::success("hello"));
    }

    #[tokio::test]
    async fn execute_treats_empty_arguments_as_an_empty_object() {
        let result =
            crate::tools::AnyTool::execute(&EmptyTool, "  ", &ExecutionContext::default()).await;

        assert_ne!(result, ToolOutput::error("ran"));
        assert_eq!(result, ToolOutput::success("ran"));
        assert_matches!(result, ToolOutput { success: true, .. });
    }

    #[tokio::test]
    async fn execute_returns_invalid_json_as_a_tool_error() {
        let result = crate::tools::AnyTool::execute(
            &EchoTool,
            r#"{"unknown":true}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(result, ToolOutput { success: false, .. });
        assert!(result
            .content
            .starts_with("Invalid arguments for tool echo: unknown field `unknown`"));
    }
}
