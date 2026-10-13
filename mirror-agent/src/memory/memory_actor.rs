/// Who authored a change. The runtime chooses this; model output never does.
///
/// Against the hosted runtime's principals: `User` and `Import` are the person
/// who owns the store (their words, typed or imported from another agent's
/// notes); `Extractor` and `Consolidator` are the workload acting on their
/// behalf. Background writers may only change what the workload wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryActor {
    User,
    Extractor,
    Consolidator,
    Import,
}

impl MemoryActor {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Extractor => "extractor",
            Self::Consolidator => "consolidator",
            Self::Import => "import",
        }
    }

    /// The actor stored as `name` in a record or event row.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "user" => Self::User,
            "extractor" => Self::Extractor,
            "consolidator" => Self::Consolidator,
            "import" => Self::Import,
            _ => return None,
        })
    }

    /// Whether the change came from the person rather than a background pass.
    pub fn is_person(self) -> bool {
        matches!(self, Self::User | Self::Import)
    }
}
