//! The sandbox a local worker runs its workspace tools in.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mirror_protocol::ToolCall;
use mirror_sandbox::SandboxLiveness;
use mirror_sandbox::SandboxProvider;
use mirror_sandbox::SandboxStream;
use mirror_sandbox::SandboxTransport;
use mirror_sandbox::SandboxdSession;
use mirror_sandbox::MAX_OUTPUT_BYTES;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;
use uuid::Uuid;

use crate::system_prompt::EnvironmentInfo;
use crate::system_prompt::ENVIRONMENT_PROBE;
use crate::tools::SandboxCallOutcome;
use crate::tools::SandboxToolExecutor;

/// Runs tool calls in one sandbox through `sandboxd`. The sandbox is created,
/// resumed or reconnected when a call needs it, and outlives this worker.
pub struct LocalSandbox<P> {
    provider: Arc<P>,
    connections: mpsc::Sender<ConnectionRequest>,
}

/// Which sandbox to use. The same request ID always finds the same sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalSandboxIdentity {
    pub request_id: Uuid,
    pub session_id: Uuid,
}

/// An open session to the named sandbox's `sandboxd`.
#[derive(Clone)]
struct Connection {
    name: String,
    session: Arc<SandboxdSession>,
}

type ConnectionRequest = oneshot::Sender<Result<Connection, String>>;

/// Cancels a dispatched call if its caller stops waiting for the result.
struct CancelOnDrop {
    session: Option<Arc<SandboxdSession>>,
    call_id: String,
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

impl<P> LocalSandbox<P>
where
    P: SandboxProvider + SandboxTransport + 'static,
{
    /// `workspace` is the directory, as the guest sees it, that tools run in.
    pub fn start(provider: Arc<P>, identity: LocalSandboxIdentity, workspace: String) -> Self {
        let (connections, requests) = mpsc::channel(16);
        drop(tokio::spawn(maintain_connection(
            provider.clone(),
            identity,
            workspace,
            requests,
        )));
        Self {
            provider,
            connections,
        }
    }

    /// Describes the sandbox, as seen from inside it, for the system prompt.
    pub async fn environment(&self, working_directory: &Path) -> Result<EnvironmentInfo, String> {
        let connection = self.connection().await?;
        let (stdout, mut probe) = tokio::io::duplex(4096);
        let outcome = self
            .provider
            .stream(
                &connection.name,
                SandboxStream {
                    command: "exec bash -s".to_string(),
                    workdir: None,
                    timeout: PROBE_TIMEOUT,
                    stdin: Some(Box::pin(ENVIRONMENT_PROBE.as_bytes())),
                    stdout: Box::pin(stdout),
                },
            )
            .await
            .map_err(|error| format!("probing sandbox {} failed: {error}", connection.name))?;
        let mut output = String::new();
        probe.read_to_string(&mut output).await.map_err(|error| {
            format!(
                "reading sandbox {}'s probe failed: {error}",
                connection.name
            )
        })?;
        if outcome.exit_code != 0 {
            output.clear();
        }
        Ok(EnvironmentInfo::from_probe(
            working_directory,
            "Linux",
            &output,
        ))
    }

    async fn connection(&self) -> Result<Connection, String> {
        let (reply, connection) = oneshot::channel();
        self.connections
            .send(reply)
            .await
            .map_err(|_| "the sandbox connection has stopped".to_string())?;
        connection
            .await
            .map_err(|_| "the sandbox connection has stopped".to_string())?
    }
}

#[async_trait]
impl<P> SandboxToolExecutor for LocalSandbox<P>
where
    P: SandboxProvider + SandboxTransport + 'static,
{
    async fn execute(&self, call: &ToolCall, timeout: Duration) -> SandboxCallOutcome {
        let connection = match self.connection().await {
            Ok(connection) => connection,
            Err(detail) => return SandboxCallOutcome::Rejected(detail),
        };
        let call_id = Uuid::new_v4().to_string();
        let replied = match connection
            .session
            .execute(call_id.clone(), call.clone(), timeout, MAX_OUTPUT_BYTES)
            .await
        {
            Ok(replied) => replied,
            Err(error) => return SandboxCallOutcome::Rejected(error.to_string()),
        };
        let cancel = CancelOnDrop {
            session: Some(connection.session),
            call_id,
        };
        let reply = replied.await;
        cancel.disarm();
        SandboxCallOutcome::from_reply(reply, timeout)
    }
}

impl CancelOnDrop {
    fn disarm(mut self) {
        self.session = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let call_id = std::mem::take(&mut self.call_id);
        let tasks = session.tasks().clone();
        drop(tasks.spawn(async move { session.cancel(call_id).await }));
    }
}

/// Owns the sandbox's session, so concurrent calls share one connection and
/// only one of them reopens it after the sandbox stops or the stream ends.
async fn maintain_connection<P>(
    provider: Arc<P>,
    identity: LocalSandboxIdentity,
    workspace: String,
    mut requests: mpsc::Receiver<ConnectionRequest>,
) where
    P: SandboxProvider + SandboxTransport,
{
    let mut current: Option<Connection> = None;
    while let Some(reply) = requests.recv().await {
        let open = current
            .as_ref()
            .filter(|connection| !connection.session.is_closed())
            .cloned();
        let connection = match open {
            Some(connection) => Ok(connection),
            None => connect(provider.as_ref(), identity, &workspace).await,
        };
        current = connection.as_ref().ok().cloned();
        let _ = reply.send(connection);
    }
    if let Some(connection) = current {
        connection.session.close().await;
    }
}

async fn connect<P>(
    provider: &P,
    identity: LocalSandboxIdentity,
    workspace: &str,
) -> Result<Connection, String>
where
    P: SandboxProvider + SandboxTransport,
{
    let name = provider
        .create(identity.request_id, identity.session_id)
        .await
        .map_err(|error| format!("creating the sandbox failed: {error}"))?;
    ensure_running(provider, &name).await?;
    let stream = provider
        .connect(&name, workspace)
        .await
        .map_err(|error| format!("connecting to sandbox {name} failed: {error}"))?;
    let session = SandboxdSession::open(stream, watch::channel(false).0)
        .await
        .map_err(|error| format!("opening sandboxd in sandbox {name} failed: {error}"))?;
    Ok(Connection {
        name,
        session: Arc::new(session),
    })
}

async fn ensure_running<P: SandboxProvider>(provider: &P, name: &str) -> Result<(), String> {
    match provider.liveness(name).await {
        Ok(SandboxLiveness::Ready) => return Ok(()),
        Ok(SandboxLiveness::Suspended) => provider
            .resume(name)
            .await
            .map_err(|error| format!("resuming sandbox {name} failed: {error}"))?,
        Ok(SandboxLiveness::Gone) => return Err(format!("sandbox {name} was deleted")),
        // Anything else, including a sandbox that is still starting, settles
        // while waiting for it to be ready.
        Ok(_) | Err(_) => {}
    }
    provider
        .wait_until_ready(name)
        .await
        .map_err(|error| format!("sandbox {name} did not become ready: {error}"))
}

#[cfg(test)]
#[path = "local_sandbox_test.rs"]
mod tests;
