//! Configured compaction decisions without inference or history mutation.

use mirror_protocol::{AgentMessage, Compaction};

use crate::providers::{Completion, CompletionRequest};
use crate::AgentError;

/// A compactor with budgets resolved for one prompt configuration.
///
/// The agent executes the optional summary request and passes its completion
/// back to `finish`. The resulting record must be self-contained: protocol
/// projection and resumed sessions only see its summary and retained records.
/// Preserve complete tool exchanges and extension snapshots needed on resume.
pub trait Compactor: Send + Sync {
    /// Whether the active history needs compaction before the next action.
    fn should_compact(&self, entries: &[AgentMessage]) -> bool;

    /// Produce diagnostic records and an optional summary model request.
    fn prepare(&self, entries: &[AgentMessage]) -> (Vec<AgentMessage>, Option<CompletionRequest>);

    /// Build a self-contained replacement context from the recorded completion.
    fn finish(
        &self,
        entries: &[AgentMessage],
        completion: Option<Completion>,
    ) -> Result<Compaction, AgentError>;
}
