//! Construction of workspace tools shared by native and Python frontends.

use std::path::Path;
use std::sync::Arc;

use super::{
    AnyTool, BashTool, EditTool, FindTool, GrepTool, LsTool, ReadTool, ToolError, WriteTool,
};

/// The tools that act on a workspace's files and processes, which a sandbox
/// runs in place of this machine.
pub const WORKSPACE_TOOL_NAMES: [&str; 7] = ["read", "bash", "write", "edit", "grep", "glob", "ls"];

/// Construct a named native tool without changing its model-facing contract.
pub fn native_tool(name: &str, workspace: &Path) -> Result<Arc<dyn AnyTool>, ToolError> {
    Ok(match name {
        "bash" => Arc::new(BashTool::new(workspace)?),
        "read" => Arc::new(ReadTool::new(workspace)?),
        "write" => Arc::new(WriteTool::new(workspace)?),
        "edit" => Arc::new(EditTool::new(workspace)?),
        "grep" => Arc::new(GrepTool::new(workspace)?),
        "glob" => Arc::new(FindTool::new(workspace)?),
        "ls" => Arc::new(LsTool::new(workspace)?),
        _ => {
            return Err(ToolError::InvalidArguments(format!(
                "unknown native tool {name:?}"
            )))
        }
    })
}

#[cfg(test)]
#[path = "native_tool_test.rs"]
mod tests;
