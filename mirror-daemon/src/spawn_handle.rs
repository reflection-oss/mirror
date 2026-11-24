//! Private worker capability for bounded launch admission and receipts.

use mirror_protocol::daemon::{AgentLaunchOutcome, Workspace};
use tokio::sync::{mpsc, oneshot};
use tonic::Status;

use crate::{AgentSpawner, PreparedSpawn, RegistryCommand, Session, SpawnTask};

/// Private worker-to-registry capability; frontends cannot construct one over RPC.
#[derive(Clone)]
pub struct SpawnHandle {
    pub remote_connections: crate::workspace::RemoteConnections,
    pub commands: mpsc::WeakSender<RegistryCommand>,
}

impl SpawnHandle {
    /// The registry persists the first snapshot before any remote initialization can use it.
    pub async fn personal_context(
        &self,
        id: String,
    ) -> Result<mirror_protocol::PersonalContext, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(RegistryCommand::PersonalContext(id, reply))?;
        result
            .await
            .map_err(|_| Status::unavailable("daemon stopped"))?
    }

    pub async fn prepare(
        &self,
        caller: AgentSpawner,
        call_id: String,
        task: SpawnTask,
    ) -> Result<PreparedSpawn, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(RegistryCommand::PrepareSpawn(caller, call_id, task, reply))?;
        result.await.map_err(|_| {
            Status::aborted(
                "launch admission response was lost; inspect the catalog before retrying",
            )
        })?
    }

    pub async fn start(
        &self,
        caller: AgentSpawner,
        id: String,
        endpoint: Option<String>,
    ) -> Result<(Session, String), Status> {
        let (reply, result) = oneshot::channel();
        self.submit(RegistryCommand::StartSpawn(caller, id, endpoint, reply))?;
        let response = result
            .await
            .map_err(|_| Status::unavailable("daemon stopped"))??;
        if response.startup_error.is_some() {
            return Err(Status::failed_precondition("child worker startup failed"));
        }
        let (reply, result) = oneshot::channel();
        self.submit(RegistryCommand::Authorize(
            response.attachment_id.clone(),
            None,
            true,
            reply,
        ))?;
        let (session, _lease) = result
            .await
            .map_err(|_| Status::unavailable("daemon stopped"))??;
        Ok((session, response.attachment_id))
    }

    pub async fn finish(
        &self,
        id: String,
        outcome: AgentLaunchOutcome,
    ) -> Result<Workspace, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(RegistryCommand::FinishSpawn(id, outcome, reply))?;
        result
            .await
            .map_err(|_| Status::unavailable("daemon stopped"))?
    }

    fn submit(&self, command: RegistryCommand) -> Result<(), Status> {
        self.commands
            .upgrade()
            .ok_or_else(|| Status::unavailable("daemon stopped"))?
            .try_send(command)
            .map_err(|_| Status::resource_exhausted("daemon ingress is full or closed"))
    }
}
