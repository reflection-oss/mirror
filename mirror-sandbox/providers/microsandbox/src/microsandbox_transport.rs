//! Commands in a running microVM over `msb exec --stream`.

use std::io;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use mirror_sandbox::SandboxStream;
use mirror_sandbox::SandboxStreamOutcome;
use mirror_sandbox::SandboxTransport;
use mirror_sandbox::SandboxTransportError;
use mirror_sandbox::SandboxdStream;
use mirror_sandbox::SANDBOXD_PATH;
use mirror_sandbox::SANDBOXD_REFRESH_FILE_CACHE_VARIABLE;
use mirror_sandbox::SANDBOXD_WORKSPACE_VARIABLE;
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::time::Instant;

use crate::MicrosandboxProvider;
use crate::MsbExecStream;
use crate::MsbFailure;
use crate::SandboxStatus;

/// Stderr kept from a streamed command, enough to explain a failure.
const STREAM_STDERR_BYTES: u64 = 64 * 1024;

impl MicrosandboxProvider {
    /// `msb exec` on a stopped sandbox boots a throwaway VM rather than
    /// failing, so every command first confirms the sandbox is running.
    async fn require_running(&self, name: &str) -> Result<(), SandboxTransportError> {
        let inspection = self
            .inspect(name)
            .await
            .map_err(|failure| match failure {
                MsbFailure::SandboxMissing => gone(name),
                MsbFailure::TimedOut(detail) => SandboxTransportError::NotReady(detail),
                failure @ (MsbFailure::SandboxExists | MsbFailure::Failed(_)) => {
                    SandboxTransportError::Failed(failure.to_string())
                }
            })?
            .ok_or_else(|| gone(name))?;
        match inspection.status {
            SandboxStatus::Running => self
                .confirm_isolation(name, &inspection)
                .map_err(SandboxTransportError::Failed),
            SandboxStatus::Stopped | SandboxStatus::Crashed | SandboxStatus::Paused => {
                Err(SandboxTransportError::Suspended(format!(
                    "sandbox {name} is {:?}",
                    inspection.status
                )))
            }
            status => Err(SandboxTransportError::NotReady(format!(
                "sandbox {name} is {status:?}"
            ))),
        }
    }
}

#[async_trait]
impl SandboxTransport for MicrosandboxProvider {
    async fn connect(
        &self,
        name: &str,
        workspace: &str,
    ) -> Result<SandboxdStream, SandboxTransportError> {
        // `sandboxd` runs as root in the guest, so its tools are confined only
        // by the workspace it is given.
        if Path::new(workspace) != self.config.workspace {
            return Err(SandboxTransportError::Failed(format!(
                "sandbox {name} mounts {:?}, not the workspace {workspace:?}",
                self.config.workspace
            )));
        }
        self.require_running(name).await?;
        let process = self
            .msb
            .command()
            .args(["exec", "--quiet", "--stream", "--env"])
            .arg(format!("{SANDBOXD_WORKSPACE_VARIABLE}={workspace}"))
            .arg("--env")
            .arg(format!("{SANDBOXD_REFRESH_FILE_CACHE_VARIABLE}=1"))
            .args([name, "--", SANDBOXD_PATH, "serve"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                SandboxTransportError::Failed(format!(
                    "msb exec into {name} could not start: {error}"
                ))
            })?;
        Ok(SandboxdStream::split(MsbExecStream::new(process)?))
    }

    async fn stream(
        &self,
        name: &str,
        request: SandboxStream,
    ) -> Result<SandboxStreamOutcome, SandboxTransportError> {
        let timeout = request.timeout;
        let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
            SandboxTransportError::Failed("stream timeout is too large".to_string())
        })?;
        self.require_running(name).await?;
        let mut command = self.msb.command();
        // The guest kills the command at its timeout even if this host stops
        // waiting first.
        command
            .args(["exec", "--quiet", "--stream", "--timeout"])
            .arg(format!("{}s", guest_seconds(timeout)));
        if let Some(workdir) = &request.workdir {
            command.args(["--workdir", workdir]);
        }
        let mut process = command
            .args([name, "--", "/bin/sh", "-c", &request.command])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                SandboxTransportError::Failed(format!(
                    "msb exec into {name} could not start: {error}"
                ))
            })?;
        let (Some(mut input), Some(mut output), Some(errors)) = (
            process.stdin.take(),
            process.stdout.take(),
            process.stderr.take(),
        ) else {
            return Err(SandboxTransportError::Failed(
                "msb exec was started without piped standard streams".to_string(),
            ));
        };
        let SandboxStream {
            stdin, mut stdout, ..
        } = request;
        let completion = async {
            let feed = async move {
                if let Some(mut stdin) = stdin {
                    tokio::io::copy(&mut stdin, &mut input).await?;
                }
                Ok::<_, io::Error>(())
            };
            let finish = async {
                let (copied, stderr) = tokio::join!(
                    tokio::io::copy(&mut output, &mut stdout),
                    read_bounded(errors, STREAM_STDERR_BYTES),
                );
                (copied, stderr, process.wait().await)
            };
            let mut feed = std::pin::pin!(feed);
            let mut finish = std::pin::pin!(finish);
            // A command may exit without reading all of its input, and the
            // input may never end, so the feed is dropped once it has.
            let (fed, (copied, stderr, status)) = tokio::select! {
                fed = &mut feed => (Some(fed), finish.await),
                finished = &mut finish => (None, finished),
            };
            if let Some(Err(error)) = fed {
                if error.kind() != io::ErrorKind::BrokenPipe {
                    return Err(outcome_unknown(format!("streaming stdin failed: {error}")));
                }
            }
            copied.map_err(|error| outcome_unknown(format!("streaming stdout failed: {error}")))?;
            let stderr = stderr
                .map_err(|error| outcome_unknown(format!("reading stderr failed: {error}")))?;
            let status = status.map_err(|error| {
                outcome_unknown(format!("msb exec could not be awaited: {error}"))
            })?;
            let exit_code = status.code().ok_or_else(|| {
                outcome_unknown(format!("msb exec ended without an exit code: {status}"))
            })?;
            Ok(SandboxStreamOutcome { exit_code, stderr })
        };
        tokio::time::timeout_at(deadline, completion)
            .await
            .unwrap_or_else(|_| {
                Err(outcome_unknown(format!(
                    "the command in {name} did not finish within {timeout:?}"
                )))
            })
    }
}

/// `msb --timeout` takes whole seconds; rounding up keeps the guest from
/// stopping a command before the host's deadline.
fn guest_seconds(timeout: Duration) -> u64 {
    let seconds = timeout.as_secs() + u64::from(timeout.subsec_nanos() > 0);
    seconds.max(1)
}

fn gone(name: &str) -> SandboxTransportError {
    SandboxTransportError::Gone(format!("sandbox {name} does not exist"))
}

fn outcome_unknown(detail: String) -> SandboxTransportError {
    SandboxTransportError::OutcomeUnknown(detail)
}

/// Keeps the first `limit` bytes and discards the rest, so the writer never
/// blocks on a full pipe.
async fn read_bounded(mut reader: impl AsyncRead + Unpin, limit: u64) -> io::Result<Vec<u8>> {
    let mut kept = Vec::new();
    (&mut reader).take(limit).read_to_end(&mut kept).await?;
    tokio::io::copy(&mut reader, &mut tokio::io::sink()).await?;
    Ok(kept)
}

#[cfg(test)]
#[path = "microsandbox_transport_test.rs"]
mod tests;
