use super::MemoryEventOp;
use super::MemoryRecord;
use super::MemoryStatus;

/// One change in a commit. An update carries the record as it should read
/// next, at exactly one version past the stored one.
#[derive(Debug, Clone)]
pub enum MemoryChange {
    Insert(MemoryRecord),
    Update {
        record: MemoryRecord,
        operation: MemoryEventOp,
    },
}

impl MemoryChange {
    /// The update that takes `record` out of selection under `operation`,
    /// which is `Retire` when something replaces it and `Forget` when the
    /// user withdraws it.
    pub fn retire(mut record: MemoryRecord, operation: MemoryEventOp) -> Self {
        record.version += 1;
        record.status = MemoryStatus::Retired;
        Self::Update { record, operation }
    }
}
