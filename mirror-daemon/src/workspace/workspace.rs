//! Capabilities of an explicitly authorized execution workspace.

use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;

use mirror_protocol::daemon::AgentConfiguration;
use mirror_protocol::WorkerSandbox;
use tonic::transport::Channel;
use tonic::Status;

use crate::{AgentOptions, AgentProcess, AgentStartError};

use super::{Local, Process, Sandbox};

/// Workspace execution differs by location; session identity and admission do not.
#[tonic::async_trait]
pub trait Workspace: Debug + Send + Sync {
    /// Open the agent connection; only local workspaces own an agent subprocess.
    async fn start_agent(
        &self,
        session_id: &str,
        configuration: &AgentConfiguration,
    ) -> Result<(Option<AgentProcess>, Channel), AgentStartError>;

    /// Start a piped process in the same execution environment as agent tools.
    async fn exec(&self, argv: &[String]) -> Result<Process, Status>;

    /// Stop execution without deleting the workspace or its history.
    async fn stop(&self) -> Result<(), Status>;
}

/// Runtime configuration comes from a live client, never from saved metadata.
pub fn from_configuration(
    options: &AgentOptions,
    directory: PathBuf,
    sandbox: Option<WorkerSandbox>,
) -> Arc<dyn Workspace> {
    let local = Local::new(options.clone(), directory);
    match sandbox {
        Some(configuration) => Arc::new(Sandbox::new(local, configuration)),
        None => Arc::new(local),
    }
}
