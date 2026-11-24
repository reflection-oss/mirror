use super::{execution_runtime, sandboxd_service::SandboxdService};
use crate::guest::GuestState;
use crate::tool_runner::RunnerCommand;
use mirror_sandbox::serve_connection;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, watch};

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("could not make sandboxd non-dumpable: {0}")]
    Dumpable(nix::errno::Errno),
    #[error("could not enable child subreaping: {0}")]
    Subreaper(#[from] nix::errno::Errno),
    #[error("sandboxd gRPC transport failed: {0}")]
    Transport(#[from] std::io::Error),
    #[error("sandboxd execution runtime failed: {0}")]
    Runtime(#[from] tokio::task::JoinError),
}
pub async fn serve<R, W>(input: R, output: W, runner: RunnerCommand) -> Result<(), ServeError>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    #[cfg(target_os = "linux")]
    {
        nix::sys::prctl::set_dumpable(false).map_err(ServeError::Dumpable)?;
        nix::sys::prctl::set_child_subreaper(true)?;
    }
    let (closing, closed) = watch::channel(false);
    let (commands, receiver) = mpsc::channel(64);
    let (guest, readiness) = watch::channel(GuestState::Unconfigured);
    let runtime = tokio::spawn(execution_runtime::run(receiver, runner, closed, readiness));
    let result = serve_connection(
        input,
        output,
        SandboxdService {
            commands,
            guest: Arc::new(guest),
            closing: closing.subscribe(),
        },
        closing.clone(),
    )
    .await;
    closing.send_replace(true);
    runtime.await?;
    result?;
    Ok(())
}
#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
