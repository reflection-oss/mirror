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

    async fn wait_until_ready(
        &self,
        provider_sandbox_id: &str,
    ) -> Result<SandboxPlacement, SandboxProviderError>;

    /// A sandbox that is neither ready nor terminally gone, such as one the
    /// provider is still starting, is a `Retryable` error.
    async fn liveness(
        &self,
        provider_sandbox_id: &str,
    ) -> Result<SandboxLiveness, SandboxProviderError>;

    /// Bring a failed guest back up on its existing workspace before returning it to use.
    /// Report the ready replacement's placement. Recovery must preserve the
    /// workspace even when cancelled or unsuccessful.
    async fn recover_workspace(
        &self,
        _name: &str,
    ) -> Result<SandboxPlacement, SandboxProviderError> {
        Err(SandboxProviderError::Rejected(
            "workspace recovery is unsupported".into(),
        ))
    }

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

    /// Stop uncertain execution; providers may persist a ban on future use.
    async fn quarantine(&self, name: &str) -> Result<(), SandboxProviderError> {
        self.suspend(name).await?;
        self.wait_until_suspended(name).await
    }

    /// Repeating deletion must be safe.
    async fn delete(&self, provider_sandbox_id: &str) -> Result<(), SandboxProviderError>;

    async fn list(&self) -> Result<Vec<ProvidedSandbox>, SandboxProviderError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvidedSandbox {
    pub provider_sandbox_id: String,
    pub session_id: uuid::Uuid,
    pub created_at: std::time::SystemTime,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SandboxPlacement {
    pub node: Option<String>,
}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxLiveness {
    Ready(SandboxPlacement),
    Suspended,
    /// Execution has terminated, but the sandbox still owns its workspace.
    Failed,
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
