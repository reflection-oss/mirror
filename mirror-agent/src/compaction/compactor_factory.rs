//! Compactor configuration for the current model and prompt.

use crate::tools::ToolDefinition;
use crate::AgentError;

use super::{Compactor, CompactorConfig};

/// Construct a compactor for the current model, prompt, and tool catalog.
///
/// Configuration is refreshed after model changes and before each compaction.
/// Factories never execute inference or mutate conversation history.
pub trait CompactorFactory: Send + Sync {
    /// Validate and resolve budgets for the current request configuration.
    fn configure(
        &self,
        config: CompactorConfig,
        tools: &[ToolDefinition],
        system_prompt: &str,
    ) -> Result<Box<dyn Compactor>, AgentError>;
}
