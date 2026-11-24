//! Why a local model operation failed, as the service reports it to callers.

use std::path::PathBuf;

use tokio::task::JoinError;

use super::engine::EngineError;
use super::settings::SettingsError;
use super::{CatalogError, DownloadError};
use crate::providers::LlmError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LocalModelError {
    #[error("Local model root and worker paths must be absolute")]
    RelativePaths,
    #[error("Cannot bind local model control port {port}: {source}")]
    Bind { port: u16, source: std::io::Error },
    #[error("Port {0} belongs to another service or runtime directory")]
    ForeignPort(u16),
    #[error("Local runtime is unavailable")]
    Unavailable,
    #[error("Expected a local model operation")]
    MissingOperation,
    #[error("This server belongs to another Mirror daemon; ask its owner to do this")]
    OwnerOnly,
    #[error("Local model runtime stopped")]
    Stopped,
    /// An error the service reported in its reply.
    #[error("{0}")]
    Rejected(String),
    #[error(transparent)]
    Rpc(#[from] tonic::Status),
    #[error(
        "llama-server (or llama) was not found; install llama.cpp, or set MIRROR_LLAMA_CPP_EXECUTABLE, \
         and restart the Mirror daemon"
    )]
    NotInstalled,
    #[error("llama.cpp is not running")]
    NotRunning,
    #[error("This reference came from an earlier Mirror daemon")]
    StaleReference,
    #[error("Restoring acts on the managed server only")]
    RestoreElsewhere,
    #[error("llama.cpp did not become ready in time")]
    StartTimedOut,
    #[error("The local model operation timed out")]
    TimedOut,
    #[error("Context window {0} is not supported")]
    UnsupportedContext(u64),
    #[error("Idle delay {0}s is not supported")]
    UnsupportedSleepIdle(u64),
    #[error("Cannot prepare {}: {source}", .path.display())]
    Prepare {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("A local model file operation did not finish: {0}")]
    Interrupted(#[from] JoinError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error(transparent)]
    Server(#[from] LlmError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error(transparent)]
    Download(#[from] DownloadError),
}
