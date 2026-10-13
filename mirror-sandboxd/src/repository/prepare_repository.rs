use std::io::Write;
use std::path::Path;

use prost::Message;

use crate::run_once::read_request;
use crate::{prepare_checkout, RepositoryDefinition, RunOnceError};
use mirror_agent::system_prompt::{
    gather_environment_info, load_workspace_context, load_workspace_skills, EnvironmentInfo,
    ResolvedInstructions,
};

#[derive(Debug, thiserror::Error)]
pub enum PrepareRepositoryError {
    #[error(transparent)]
    Request(#[from] RunOnceError),
    #[error("repository preparation failed: {0}")]
    Prepare(#[from] std::io::Error),
}

/// What `prepare-repository` writes: the checkout's prompt context and facts
/// about the sandbox that holds it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PreparedContext {
    #[serde(flatten)]
    pub context: ResolvedInstructions,
    /// Absent from guests that predate environment facts.
    #[serde(default)]
    pub environment: Option<EnvironmentInfo>,
}

pub async fn prepare_repository_stdio(
    workspace: &Path,
    skills_only: bool,
) -> Result<(), PrepareRepositoryError> {
    let encoded = read_request().await?;
    let requested =
        RepositoryDefinition::decode(encoded.as_slice()).map_err(RunOnceError::Decode)?;
    prepare_checkout(workspace, &requested).await?;
    let root = workspace.join("repository");
    let cwd = root.join(&requested.directory);
    let context = if skills_only {
        load_workspace_skills(&cwd, &root)?
    } else {
        load_workspace_context(&cwd, &root)?
    };
    let prepared = PreparedContext {
        context,
        environment: Some(gather_environment_info(&cwd)),
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &prepared).map_err(std::io::Error::other)?;
    stdout.flush()?;
    Ok(())
}
