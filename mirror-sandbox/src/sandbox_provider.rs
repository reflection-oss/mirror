//! Provider-neutral sandbox lifecycle and its outcomes.

use async_trait::async_trait;

#[async_trait]
/// Provider-neutral lifecycle operations on stable sandbox identities.
pub trait SandboxProvider: Send + Sync {
    /// Repeated calls with the same request ID must resolve to the same sandbox.
    async fn create(
        &self,
        request_id: uuid::Uuid,
        session_id: uuid::Uuid,
    ) -> Result<String, SandboxProviderError>;

    async fn wait_until_ready(&self, provider_sandbox_id: &str)
        -> Result<(), SandboxProviderError>;

    /// A sandbox that is neither ready nor terminally gone, such as one the
    /// provider is still starting, is a `Retryable` error.
    async fn liveness(
        &self,
        provider_sandbox_id: &str,
    ) -> Result<SandboxLiveness, SandboxProviderError>;

    /// Request suspension without replacing the sandbox identity or workspace.
    /// Repeating the request for an already suspended sandbox must be safe.
    async fn suspend(&self, provider_sandbox_id: &str) -> Result<(), SandboxProviderError>;

    /// Return only when the sandbox is suspended and no longer executes work.
    async fn wait_until_suspended(
        &self,
        provider_sandbox_id: &str,
    ) -> Result<(), SandboxProviderError>;

    /// Request execution under the same identity with the preserved workspace.
    /// Repeating the request for an already running sandbox must be safe.
    async fn resume(&self, provider_sandbox_id: &str) -> Result<(), SandboxProviderError>;

    /// Repeating deletion must be safe.
    async fn delete(&self, provider_sandbox_id: &str) -> Result<(), SandboxProviderError>;
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum SandboxProviderError {
    /// Repeating the request with the same identity is safe.
    #[error("sandbox provider is temporarily unavailable: {0}")]
    Retryable(String),
    /// The request is meaningful but cannot succeed without changing it.
    #[error("sandbox provider rejected the request: {0}")]
    Rejected(String),
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxLiveness {
    Ready,
    Suspended,
    Gone,
}

impl SandboxProviderError {
    /// A bounded name for the failure, for span and log fields.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Retryable(_) => "provider_retryable",
            Self::Rejected(_) => "provider_rejected",
        }
    }
}
