use super::MemoryStatus;

/// The change a writer is recording; the head is always the fold of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryEventOp {
    Accept,
    Correct,
    Retire,
    Forget,
}

impl MemoryEventOp {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accept => "accept",
            Self::Correct => "correct",
            Self::Retire => "retire",
            Self::Forget => "forget",
        }
    }

    /// Whether this is the change that takes a stored record from `before` to
    /// `after`. Active records are corrected in place, and anything live is
    /// retired (or forgotten by the user); a retired record does not come back.
    /// `Accept` only ever starts a record.
    pub fn describes(self, before: MemoryStatus, after: MemoryStatus) -> bool {
        match (before, after) {
            (MemoryStatus::Active, MemoryStatus::Active) => self == Self::Correct,
            (MemoryStatus::Retired, MemoryStatus::Active) => false,
            (MemoryStatus::Retired, MemoryStatus::Retired) => false,
            (MemoryStatus::Active, MemoryStatus::Retired) => {
                matches!(self, Self::Retire | Self::Forget)
            }
        }
    }
}
