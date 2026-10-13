use std::future::Future;
use std::pin::Pin;

use rmcp::model::{CallToolResult, JsonObject, Tool};

use crate::tools::ToolError;

/// Routes MCP discovery and calls; implementations own authentication and connection lifecycles.
pub trait McpRouter: Send + Sync {
    fn list_tools<'a>(
        &'a self,
        server: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Tool>, ToolError>> + Send + 'a>>;

    fn call_tool<'a>(
        &'a self,
        server: &'a str,
        tool: &'a str,
        arguments: JsonObject,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>>;
}
