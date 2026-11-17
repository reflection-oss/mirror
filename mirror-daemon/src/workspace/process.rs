//! A running workspace command with bounded pipes and owned, awaited cleanup.

use std::future::Future;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use mirror_protocol::WorkerSandbox;
use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;
use tempfile::TempDir;
use tokio::io::{self, DuplexStream};
use tokio::process::{Child, Command};
use tokio::sync::oneshot;
use tonic::Status;

use crate::AgentOptions;

use super::{run_workspace_command, sandbox_command, sandbox_configuration_json};

/// Drain output concurrently with `wait`; `cancel` or dropping the handle stops execution.
#[derive(Debug)]
pub struct Process {
    pub stdin: Option<DuplexStream>,
    pub stdout: Option<DuplexStream>,
    pub stderr: Option<DuplexStream>,
    cancel: Option<oneshot::Sender<()>>,
    completion: Completion,
}

/// Bounded byte streams and cancellation owned by a workspace's process supervisor.
#[derive(Debug)]
pub struct ProcessIo {
    pub stdin: DuplexStream,
    pub stdout: DuplexStream,
    pub stderr: DuplexStream,
    pub cancellation: oneshot::Receiver<()>,
}

#[derive(Debug)]
enum Completion {
    Pending(oneshot::Receiver<Result<i32, Status>>),
    Ready(Result<i32, Status>),
}

#[derive(Debug)]
struct SandboxCompletion {
    directory: TempDir,
    quarantine: Command,
}

#[derive(Debug, serde::Deserialize)]
enum SandboxOutcome {
    Ok(i32),
    Err(String),
    Unknown(String),
}

impl Process {
    /// Spawn with pipes and a private process group, preserving the supplied environment.
    pub fn spawn(command: Command) -> Result<Self, Status> {
        Self::start(command, None)
    }

    /// Start a guest command with privately owned completion evidence and cleanup.
    pub fn spawn_sandbox(
        options: &AgentOptions,
        workspace: &Path,
        sandbox: &WorkerSandbox,
        argv: &[String],
    ) -> Result<Self, Status> {
        if argv.is_empty() {
            return Err(Status::invalid_argument("process requires an executable"));
        }
        let script = shlex::try_join(argv.iter().map(String::as_str))
            .map_err(|_| Status::invalid_argument("process arguments contain NUL"))?;
        // Completion evidence is private host state, never mounted into the guest.
        let directory = tempfile::Builder::new()
            .permissions(std::fs::Permissions::from_mode(0o700))
            .tempdir()
            .map_err(|_| Status::internal("process directory unavailable"))?;
        let mut command = sandbox_command(options);
        command
            .args(["sandbox", "exec"])
            .arg(sandbox_configuration_json(sandbox)?)
            .arg(workspace)
            .arg(directory.path().join("outcome"))
            .arg(script);
        let mut quarantine = sandbox_command(options);
        quarantine.args(["sandbox", "quarantine", &sandbox.id]);
        Self::start(
            command,
            Some(SandboxCompletion {
                directory,
                quarantine,
            }),
        )
    }

    fn start(mut command: Command, sandbox: Option<SandboxCompletion>) -> Result<Self, Status> {
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| Status::unavailable("could not start workspace process"))?;
        Ok(Self::run(move |pipes| supervise(child, pipes, sandbox)))
    }

    /// Own a supervisor until completion; cancellation must settle execution before returning.
    pub fn run<F: Future<Output = Result<i32, Status>> + Send + 'static>(
        supervise: impl FnOnce(ProcessIo) -> F + Send + 'static,
    ) -> Self {
        let (stdin, input) = io::duplex(65536);
        let (stdout, output) = io::duplex(65536);
        let (stderr, errors) = io::duplex(65536);
        let (cancel, cancelled) = oneshot::channel();
        let (complete, completion) = oneshot::channel();
        tokio::spawn(async move {
            let result = supervise(ProcessIo {
                stdin: input,
                stdout: output,
                stderr: errors,
                cancellation: cancelled,
            })
            .await;
            let _ = complete.send(result);
        });
        Self {
            stdin: Some(stdin),
            stdout: Some(stdout),
            stderr: Some(stderr),
            cancel: Some(cancel),
            completion: Completion::Pending(completion),
        }
    }

    /// Wait for output forwarding, reaping and cleanup; nonzero exits are results.
    pub async fn wait(&mut self) -> Result<i32, Status> {
        match &mut self.completion {
            Completion::Ready(result) => result.clone(),
            Completion::Pending(receiver) => {
                let result = receiver.await.unwrap_or_else(|_| {
                    Err(Status::aborted(
                        "process supervisor stopped; completion is unknown",
                    ))
                });
                self.completion = Completion::Ready(result.clone());
                result
            }
        }
    }

    /// Cancel execution and await bounded cleanup, even if a prior wait was interrupted.
    pub async fn cancel(&mut self) -> Result<(), Status> {
        drop(self.stdin.take());
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        self.wait().await.map(|_| ())
    }
}

impl SandboxCompletion {
    async fn finish(self) -> Result<i32, Status> {
        let result = match tokio::fs::read(self.directory.path().join("outcome")).await {
            Ok(bytes) => serde_json::from_slice::<SandboxOutcome>(&bytes).ok(),
            Err(_) => None,
        };
        match result {
            Some(SandboxOutcome::Ok(code)) => return Ok(code),
            Some(SandboxOutcome::Err(detail)) => {
                // Native diagnostics are deliberately not forwarded to clients.
                drop(detail);
                return Err(Status::failed_precondition("sandbox process failed"));
            }
            Some(SandboxOutcome::Unknown(detail)) => drop(detail),
            None => {}
        }
        run_workspace_command(self.quarantine, &[], Duration::from_secs(90)).await?;
        Err(Status::aborted(
            "sandbox process completion is unknown; quarantined, never replayed",
        ))
    }
}

async fn supervise(
    mut child: Child,
    pipes: ProcessIo,
    sandbox: Option<SandboxCompletion>,
) -> Result<i32, Status> {
    let ProcessIo {
        mut stdin,
        stdout: output,
        stderr: errors,
        cancellation: cancelled,
    } = pipes;
    let mut input = child
        .stdin
        .take()
        .ok_or_else(|| Status::internal("process stdin unavailable"))?;
    let feeding = async move {
        let result = io::copy(&mut stdin, &mut input).await;
        drop(input);
        match result {
            Ok(_) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
            Err(_) => Err(Status::internal("process stdin failed")),
        }
    };
    let group = child.id().map(|id| Pid::from_raw(id as i32));
    let result = {
        let running = async {
            let mut stdout = child
                .stdout
                .take()
                .ok_or_else(|| Status::internal("process stdout unavailable"))?;
            let mut stderr = child
                .stderr
                .take()
                .ok_or_else(|| Status::internal("process stderr unavailable"))?;
            let (_, _, status) = tokio::try_join!(
                async {
                    let mut output = output;
                    io::copy(&mut stdout, &mut output).await
                },
                async {
                    let mut errors = errors;
                    io::copy(&mut stderr, &mut errors).await
                },
                async {
                    let status = child.wait().await?;
                    // Descendants must release inherited pipes before output can finish draining.
                    if let Some(group) = group {
                        let _ = killpg(group, Signal::SIGKILL);
                    }
                    Ok::<_, std::io::Error>(status)
                },
            )
            .map_err(|_| Status::internal("process I/O or wait failed"))?;
            Ok::<_, Status>(status)
        };
        tokio::pin!(running);
        let running = async {
            tokio::select! {
                result = &mut running => result,
                result = feeding => { result?; running.await }
            }
        };
        tokio::select! {
            biased;
            _ = cancelled => None,
            result = running => Some(result),
        }
    };
    let (status, error) = match result {
        Some(Ok(status)) => (Ok(status), None),
        failure => (
            terminate(&mut child, group).await,
            failure.and_then(Result::err),
        ),
    };
    let code = match sandbox {
        Some(sandbox) => sandbox.finish().await?,
        None => {
            let status = status
                .as_ref()
                .map_err(|_| Status::aborted("process completion is unknown; never replayed"))?;
            status.code().unwrap_or(128 + status.signal().unwrap_or(0))
        }
    };
    status?;
    error.map_or(Ok(code), Err)
}

async fn terminate(child: &mut Child, group: Option<Pid>) -> Result<ExitStatus, Status> {
    if let Some(group) = group {
        let _ = killpg(group, Signal::SIGTERM);
    }
    let result = tokio::time::timeout(Duration::from_secs(35), child.wait()).await;
    // A reaped launcher can still have descendants holding pipes or touching the workspace.
    if let Some(group) = group {
        let _ = killpg(group, Signal::SIGKILL);
    }
    match result {
        Ok(result) => result.map_err(|_| Status::internal("process cleanup failed")),
        Err(_) => tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .map_err(|_| Status::deadline_exceeded("process cleanup timed out"))?
            .map_err(|_| Status::internal("process cleanup failed")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn exited_child_drains_output_without_waiting_for_background_descendants() {
        let mut command = Command::new("/bin/sh");
        command.env_clear().args([
            "-c",
            "/bin/sleep 60 & printf done; printf warning >&2; exit 7",
        ]);
        let mut process = Process::spawn(command).expect("fixture shell starts");
        drop(process.stdin.take());
        let mut stdout = process.stdout.take().expect("stdout is piped");
        let mut stderr = process.stderr.take().expect("stderr is piped");
        let mut output = String::new();
        let mut errors = String::new();
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(
                stdout.read_to_string(&mut output),
                stderr.read_to_string(&mut errors),
                process.wait(),
            )
        })
        .await;
        process
            .cancel()
            .await
            .expect("fixture processes are reaped");
        let (stdout, stderr, status) = result.expect("direct child exit releases inherited pipes");
        assert_matches!(stdout, Ok(4));
        assert_matches!(stderr, Ok(7));
        assert_matches!(status, Ok(7));
        assert_eq!(output, "done");
        assert_eq!(errors, "warning");
    }
}
