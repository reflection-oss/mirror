//! Subprocess ownership and the existing agent startup protocol.

use std::io;
use std::os::unix::fs::PermissionsExt;
use std::time::Duration;

use base64::Engine;
use mirror_protocol::daemon::AgentConfiguration;
use mirror_protocol::{SessionMessage, WorkerSandbox, WorkerStartup};
use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;
use prost::Message;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tonic::transport::Channel;

use crate::{local_channel, AgentOptions};

/// One child and its parent lifeline; no agent is adopted from a PID file.
#[derive(Debug)]
pub struct AgentProcess {
    pub startup_messages: Vec<SessionMessage>,
    child: Child,
    output: JoinHandle<()>,
    _directory: TempDir,
}

/// Startup reports survive failure; raw subprocess diagnostics never leave the daemon.
#[derive(Debug)]
pub struct AgentStartError {
    pub reason: &'static str,
    pub messages: Vec<SessionMessage>,
}

impl From<io::Error> for AgentStartError {
    fn from(_: io::Error) -> Self {
        Self {
            reason: "agent startup failed",
            messages: Vec::new(),
        }
    }
}

const START_TIMEOUT: Duration = Duration::from_secs(30);
const STOP_TIMEOUT: Duration = Duration::from_secs(3);

impl AgentProcess {
    /// Complete host-agent startup with a private socket and framed configuration.
    pub async fn start(
        options: &AgentOptions,
        session_id: &str,
        mut command: Command,
        configuration: &AgentConfiguration,
        sandbox: Option<WorkerSandbox>,
        managed_reference: &str,
    ) -> Result<(Self, Channel), AgentStartError> {
        let directory = tempfile::Builder::new()
            .prefix("a")
            .permissions(std::fs::Permissions::from_mode(0o700))
            .tempdir_in(&options.runtime_directory)?;
        let socket = directory.path().join("rpc");
        let config = WorkerStartup {
            configuration: Some(configuration.clone()),
            rpc_address: format!("unix:{}", socket.display()),
            session_id: session_id.into(),
            database: crate::workspace::text(&options.database)
                .map_err(io::Error::other)?
                .into(),
            create_session: true,
            allow_workspace_mismatch: false,
            exit_on_stdin_close: true,
            sandbox,
            managed_reference: managed_reference.into(),
        }
        .encode_to_vec();
        command.arg("--config-stdin");
        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::other("agent stdin unavailable"))?;
        let startup = async {
            let size = u32::try_from(config.len()).map_err(io::Error::other)?;
            stdin.write_all(&size.to_be_bytes()).await?;
            stdin.write_all(&config).await
        };
        tokio::time::timeout(START_TIMEOUT, startup)
            .await
            .map_err(|_| {
                io::Error::new(io::ErrorKind::TimedOut, "agent configuration timed out")
            })??;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("agent stdout unavailable"))?;
        let mut lines = BufReader::new(stdout.take((4 * crate::MAX_FRAME_BYTES) as u64)).lines();
        let mut startup_messages = Vec::new();
        let mut startup_bytes = 0;
        let mut reason = "agent startup failed";
        let ready = async {
            while let Some(line) = lines.next_line().await? {
                if let Some(encoded) = line.strip_prefix("MIRROR_SESSION_MESSAGE=") {
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(io::Error::other)?;
                    startup_bytes += bytes.len();
                    if startup_bytes > crate::MAX_FRAME_BYTES {
                        return Err(io::Error::other(
                            "agent startup metadata exceeded its bound",
                        ));
                    }
                    startup_messages
                        .push(SessionMessage::decode(bytes.as_slice()).map_err(io::Error::other)?);
                }
                if let Some(address) = line.strip_prefix("MIRROR_RPC_ADDRESS=") {
                    if address != format!("unix:{}", socket.display()) {
                        return Err(io::Error::other(
                            "agent did not bind the requested private socket",
                        ));
                    }
                    return local_channel(socket.clone())
                        .await
                        .map_err(io::Error::other);
                }
                if line.starts_with("MIRROR_RPC_ERROR=") {
                    if line.starts_with("MIRROR_RPC_ERROR=agent failed: invalid agent configuration: Invalid prompt budget:") {
                        reason = "Invalid prompt budget";
                    }
                    return Err(io::Error::other("agent startup failed"));
                }
            }
            Err(io::Error::other("agent exited before reporting readiness"))
        };
        let channel = match tokio::time::timeout(START_TIMEOUT, ready).await {
            Ok(Ok(channel)) => channel,
            _ => {
                let _ = child.kill().await;
                return Err(AgentStartError {
                    reason,
                    messages: startup_messages,
                });
            }
        };
        let mut stdout = lines.into_inner().into_inner().into_inner();
        let output = tokio::spawn(async move {
            // Agent stdout is a control channel, not a diagnostic payload sink.
            let _ = tokio::io::copy(&mut stdout, &mut tokio::io::sink()).await;
        });
        Ok((
            Self {
                startup_messages,
                child,
                output,
                _directory: directory,
            },
            channel,
        ))
    }

    /// Close the lifeline, then bound cleanup of this still-owned child.
    pub async fn close(&mut self) {
        drop(self.child.stdin.take());
        if tokio::time::timeout(STOP_TIMEOUT, self.child.wait())
            .await
            .is_err()
        {
            if let Some(id) = self.child.id() {
                let _ = killpg(Pid::from_raw(id as i32), Signal::SIGKILL);
            }
            let _ = self.child.kill().await;
        }
        self.output.abort();
    }
}

impl Drop for AgentProcess {
    fn drop(&mut self) {
        self.output.abort();
    }
}
