//! Bounded native workspace subprocesses with no inherited authentication environment.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tonic::Status;

use crate::AgentOptions;

/// Construct a host operation with only the daemon's explicit environment.
pub fn workspace_command(
    options: &AgentOptions,
    program: impl AsRef<OsStr>,
    cwd: &Path,
) -> Command {
    let mut command = Command::new(program);
    command
        .env_clear()
        .env("HOME", &options.home)
        .env("PATH", &options.path)
        .env("GIT_TERMINAL_PROMPT", "0")
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .kill_on_drop(true);
    command
}

/// Finish or reap the whole operation; raw subprocess diagnostics never reach RPC clients.
pub async fn run_workspace_command(
    mut command: Command,
    input: &[u8],
    timeout: Duration,
) -> Result<String, Status> {
    let mut child = command
        .spawn()
        .map_err(|_| Status::unavailable("could not start workspace command"))?;
    let operation = async {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("workspace stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("workspace stdout unavailable"))?;
        let mut bytes = Vec::new();
        let write = async {
            stdin.write_all(input).await?;
            drop(stdin);
            Ok::<_, std::io::Error>(())
        };
        let mut stdout = stdout.take(1024 * 1024 + 1);
        tokio::try_join!(write, stdout.read_to_end(&mut bytes))?;
        if bytes.len() > 1024 * 1024 {
            return Err(std::io::Error::other("workspace output exceeded its bound"));
        }
        if !child.wait().await?.success() {
            return Err(std::io::Error::other("workspace command failed"));
        }
        String::from_utf8(bytes).map_err(std::io::Error::other)
    };
    match tokio::time::timeout(timeout, operation).await {
        Ok(Ok(output)) => Ok(output),
        result => {
            if let Some(id) = child.id() {
                let _ = killpg(Pid::from_raw(id as i32), Signal::SIGKILL);
            }
            let _ = child.kill().await;
            Err(if result.is_err() {
                Status::deadline_exceeded("workspace command timed out")
            } else {
                Status::failed_precondition(
                    "workspace command failed; check the checkout and installed native tools",
                )
            })
        }
    }
}
