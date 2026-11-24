//! The sandbox a local worker runs its workspace tools in.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mirror_protocol::ToolCall;
use mirror_sandbox::{
    SandboxLiveness, SandboxProvider, SandboxStream, SandboxTransport, SandboxdSession,
    MAX_OUTPUT_BYTES, SANDBOXD_PATH,
};
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, oneshot, watch};
use uuid::Uuid;

use crate::system_prompt::{EnvironmentInfo, ResolvedInstructions};
use crate::tools::{SandboxCallOutcome, SandboxToolExecutor};

/// Owns one sandbox connection; uncertain execution quarantines it without replay.
pub struct LocalSandbox {
    connections: mpsc::Sender<ConnectionRequest>,
}

/// Which sandbox to use. The same request ID always finds the same sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalSandboxIdentity {
    pub request_id: Uuid,
    pub session_id: Uuid,
}

struct Connection {
    name: String,
    session: Arc<SandboxdSession>,
    environment: EnvironmentInfo,
    instructions: ResolvedInstructions,
}

type ConnectionRequest = oneshot::Sender<Result<Arc<Connection>, String>>;

/// An abandoned local call stops the VM rather than guessing whether it completed.
struct CloseOnDrop {
    session: Option<Arc<SandboxdSession>>,
    connections: mpsc::Sender<ConnectionRequest>,
}

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let connections = self.connections.clone();
            drop(runtime.spawn(async move {
                session.close().await;
                drop(connections);
            }));
        }
    }
}
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

impl LocalSandbox {
    /// `workspace` is the directory, as the guest sees it, that tools run in.
    pub fn start<P>(provider: Arc<P>, identity: LocalSandboxIdentity, workspace: String) -> Self
    where
        P: SandboxProvider + SandboxTransport + 'static,
    {
        let (connections, requests) = mpsc::channel(16);
        drop(tokio::spawn(maintain_connection(
            provider, identity, workspace, requests,
        )));
        Self { connections }
    }

    /// Prepare the VM before extensions or model tools can submit work.
    pub async fn prepare(&self) -> Result<(), String> {
        self.connection().await.map(drop)
    }

    /// Environment facts collected in the guest during preparation.
    pub async fn environment(&self) -> Result<EnvironmentInfo, String> {
        Ok(self.connection().await?.environment.clone())
    }

    /// Instructions collected in the guest, never by following host-side links.
    pub async fn instructions(&self) -> Result<ResolvedInstructions, String> {
        Ok(self.connection().await?.instructions.clone())
    }

    async fn connection(&self) -> Result<Arc<Connection>, String> {
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
impl SandboxToolExecutor for LocalSandbox {
    async fn execute(&self, call: &ToolCall, timeout: Duration) -> SandboxCallOutcome {
        let connection = match self.connection().await {
            Ok(connection) => connection,
            Err(detail) => return SandboxCallOutcome::Rejected(detail),
        };
        let mut guard = CloseOnDrop {
            session: Some(connection.session.clone()),
            connections: self.connections.clone(),
        };
        let replied = match connection
            .session
            .execute(
                Uuid::new_v4().to_string(),
                call.clone(),
                timeout,
                MAX_OUTPUT_BYTES,
            )
            .await
        {
            Ok(replied) => replied,
            Err(error) => return SandboxCallOutcome::Rejected(error.to_string()),
        };
        let outcome = SandboxCallOutcome::from_reply(replied.await, timeout);
        if matches!(outcome, SandboxCallOutcome::Unknown(_)) {
            connection.session.close().await;
            let _ = self.connection().await;
        }
        guard.session = None;
        outcome
    }
}

/// Owns preparation and the connection independently of any waiting caller.
async fn maintain_connection<P>(
    provider: Arc<P>,
    identity: LocalSandboxIdentity,
    workspace: String,
    mut requests: mpsc::Receiver<ConnectionRequest>,
) where
    P: SandboxProvider + SandboxTransport,
{
    let Some(reply) = requests.recv().await else {
        return;
    };
    let connection = connect(provider.as_ref(), identity, &workspace).await;
    let _ = reply.send(connection.clone());
    if let Ok(connection) = &connection {
        let mut closing = connection.session.closing();
        loop {
            tokio::select! {
                biased;
                () = async { let _ = closing.wait_for(|closed| *closed).await; } => break,
                reply = requests.recv() => match reply {
                    Some(reply) => { let _ = reply.send(Ok(connection.clone())); }
                    None => {
                        connection.session.close().await;
                        return;
                    }
                },
            }
        }
    }
    let reason = match connection {
        Ok(connection) => {
            let detail = quarantine(provider.as_ref(), &connection.name).await;
            connection.session.close().await;
            format!("sandbox {} is quarantined after connection loss; {detail}; commands will not be replayed", connection.name)
        }
        Err(error) => error,
    };
    while let Some(reply) = requests.recv().await {
        let _ = reply.send(Err(reason.clone()));
    }
}

async fn quarantine<P: SandboxProvider>(provider: &P, name: &str) -> String {
    match tokio::time::timeout(PROBE_TIMEOUT, provider.quarantine(name)).await {
        Ok(Ok(())) => "execution stopped".to_string(),
        Ok(Err(error)) => format!("stop could not be confirmed: {error}"),
        Err(_) => "stop could not be confirmed before its deadline".to_string(),
    }
}

async fn connect<P>(
    provider: &P,
    identity: LocalSandboxIdentity,
    workspace: &str,
) -> Result<Arc<Connection>, String>
where
    P: SandboxProvider + SandboxTransport,
{
    let name = provider
        .create(identity.request_id, identity.session_id)
        .await
        .map_err(|error| format!("creating the sandbox failed: {error}"))?;
    ensure_running(provider, &name).await?;
    let (environment, instructions) = match probe(provider, &name, workspace).await {
        Ok(context) => context,
        Err(error) => {
            let detail = quarantine(provider, &name).await;
            return Err(format!("sandbox {name} is quarantined: {error}; {detail}"));
        }
    };
    let stream = provider
        .connect(&name, workspace)
        .await
        .map_err(|error| format!("connecting to sandbox {name} failed: {error}"))?;
    let session = SandboxdSession::open(stream, watch::channel(false).0)
        .await
        .map_err(|error| format!("opening sandboxd in sandbox {name} failed: {error}"))?;
    Ok(Arc::new(Connection {
        name,
        session: Arc::new(session),
        environment,
        instructions,
    }))
}

/// One bounded read-only probe, owned by preparation rather than a streaming API.
async fn probe<P: SandboxTransport>(
    provider: &P,
    name: &str,
    workspace: &str,
) -> Result<(EnvironmentInfo, ResolvedInstructions), String> {
    let (stdout, reader) = tokio::io::duplex(4096);
    let mut reader = reader.take(MAX_OUTPUT_BYTES + 1);
    let mut output = Vec::new();
    let capture = async {
        let (outcome, read) = tokio::join!(
            provider.stream(
                name,
                SandboxStream {
                    command: format!("{SANDBOXD_PATH} context"),
                    workdir: Some(workspace.to_string()),
                    timeout: PROBE_TIMEOUT,
                    cancellation: Box::pin(std::future::pending()),
                    stdin: None,
                    stdout: Box::pin(stdout),
                    stderr: None,
                }
            ),
            reader.read_to_end(&mut output),
        );
        read.map_err(|error| error.to_string())?;
        let outcome = outcome.map_err(|error| error.to_string())?;
        if outcome.exit_code != 0 || output.len() as u64 > MAX_OUTPUT_BYTES {
            return Err("sandbox context probe failed or exceeded its byte limit".into());
        }
        serde_json::from_slice(&output).map_err(|error| error.to_string())
    };
    // The stream owns its deadline and needs time to confirm process-group cleanup.
    tokio::time::timeout(
        PROBE_TIMEOUT + mirror_sandbox::CANCEL_ANSWER_TIMEOUT + Duration::from_secs(5),
        capture,
    )
    .await
    .map_err(|_| "sandbox context probe exceeded its deadline".to_string())?
}

async fn ensure_running<P: SandboxProvider>(provider: &P, name: &str) -> Result<(), String> {
    match provider.liveness(name).await {
        Ok(SandboxLiveness::Ready(_)) => return Ok(()),
        Ok(SandboxLiveness::Suspended) => provider
            .resume(name)
            .await
            .map_err(|error| format!("resuming sandbox {name} failed: {error}"))?,
        Ok(SandboxLiveness::Gone) => return Err(format!("sandbox {name} was deleted")),
        // Starting sandboxes settle while waiting for readiness.
        Ok(_) | Err(_) => {}
    }
    provider
        .wait_until_ready(name)
        .await
        .map(drop)
        .map_err(|error| format!("sandbox {name} did not become ready: {error}"))
}

#[cfg(test)]
#[path = "local_sandbox_test.rs"]
mod tests;
