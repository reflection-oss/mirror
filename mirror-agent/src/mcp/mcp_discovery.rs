use std::sync::Arc;

use super::McpTool;
use crate::tools::ToolError;

/// Discovers the owning runtime's current tools without changing execution authority.
#[async_trait::async_trait]
pub trait McpDiscovery: Send + Sync {
    async fn discover(&self) -> Result<Vec<Arc<McpTool>>, ToolError>;
}
