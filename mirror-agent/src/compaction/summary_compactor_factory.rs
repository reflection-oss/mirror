//! Default summarization factory for Mirror context windows.

use crate::tools::ToolDefinition;
use crate::AgentError;

use super::{Compactor, CompactorConfig, CompactorFactory, SummaryCompactor};

/// Construct the default summary compactor for the current request configuration.
#[derive(Debug, Default)]
pub struct SummaryCompactorFactory;

impl CompactorFactory for SummaryCompactorFactory {
    fn configure(
        &self,
        config: CompactorConfig,
        tools: &[ToolDefinition],
        system_prompt: &str,
    ) -> Result<Box<dyn Compactor>, AgentError> {
        Ok(Box::new(SummaryCompactor::new(
            config,
            tools,
            system_prompt,
        )?))
    }
}
