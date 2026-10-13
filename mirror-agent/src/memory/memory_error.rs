//! Invalid input or unavailable memory storage; never `anyhow`.

/// A memory validation or storage failure with a user-facing message.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MemoryError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
    #[error("memory record changed; regenerate the change against the current version")]
    Stale,
    #[error("memory: {0}")]
    Invalid(String),
    #[error("memory cannot store credentials; use a password manager or secret store")]
    Credential,
}

pub(crate) type MemoryResult<T> = std::result::Result<T, MemoryError>;

pub(crate) fn invalid(message: impl Into<String>) -> MemoryError {
    MemoryError::Invalid(message.into())
}
