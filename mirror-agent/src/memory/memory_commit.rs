//! An atomic change set; each change is checked against its own record's version.

use super::MemoryActor;
use super::MemoryChange;

#[derive(Debug, Clone)]
pub struct MemoryCommit {
    pub actor: MemoryActor,
    /// The conversation this change belongs to, recorded on every event.
    pub session_id: Option<String>,
    pub changes: Vec<MemoryChange>,
}
