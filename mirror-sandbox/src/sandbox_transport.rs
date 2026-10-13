use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;

use super::SandboxdStream;

#[async_trait]
pub trait SandboxTransport: Send + Sync {
    /// Starts `sandboxd serve` in the sandbox with `workspace` as its tool
    /// workspace and opens the stream to it. Nothing runs in the sandbox until
    /// a call is sent, so a failure here is always pre-dispatch.
    async fn connect(
        &self,
        provider_sandbox_id: &str,
        workspace: &str,
    ) -> Result<SandboxdStream, SandboxTransportError>;

    async fn stream(
        &self,
        sandbox_id: &str,
        request: SandboxStream,
    ) -> Result<SandboxStreamOutcome, SandboxTransportError>;
}

pub struct SandboxStream {
    pub command: String,
    pub workdir: Option<String>,
    pub timeout: Duration,
    pub stdin: Option<Pin<Box<dyn AsyncRead + Send>>>,
    pub stdout: Pin<Box<dyn AsyncWrite + Send>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SandboxStreamOutcome {
    pub exit_code: i32,
    pub stderr: Vec<u8>,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum SandboxTransportError {
    /// The sandbox exists but is not accepting connections yet; retry shortly.
    #[error("sandbox is not ready: {0}")]
    NotReady(String),
    /// Resuming the sandbox under the same identity brings its workspace back.
    #[error("sandbox is suspended: {0}")]
    Suspended(String),
    /// Delete the sandbox's identity before replacing it.
    #[error("sandbox is gone: {0}")]
    Gone(String),
    /// A streamed command may still be running; do not reuse its sandbox.
    #[error("sandbox operation outcome is unknown; the command may still be running: {0}")]
    OutcomeUnknown(String),
    /// Repeating the operation will not help without a change.
    #[error("sandbox transport failed: {0}")]
    Failed(String),
}

/// Where the sandbox image installs `sandboxd`.
pub const SANDBOXD_PATH: &str = "/usr/local/bin/mirror-sandboxd";
pub const SANDBOXD_WORKSPACE_VARIABLE: &str = "MIRROR_WORKSPACE";
/// Set to `1` when the workspace is shared from a host whose changes the
/// guest kernel may cache stale, such as a virtiofs mount. `sandboxd` then
/// drops the guest's dentry and inode caches before each call.
pub const SANDBOXD_REFRESH_FILE_CACHE_VARIABLE: &str = "MIRROR_REFRESH_FILE_CACHE";

impl SandboxTransportError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NotReady(_) => "transport_not_ready",
            Self::Suspended(_) => "transport_suspended",
            Self::Gone(_) => "transport_gone",
            Self::OutcomeUnknown(_) => "transport_outcome_unknown",
            Self::Failed(_) => "transport_failed",
        }
    }
}
