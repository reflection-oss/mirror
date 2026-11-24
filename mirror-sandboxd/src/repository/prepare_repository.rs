use std::io::Write;
use std::path::Path;
use std::time::Instant;

use prost::Message;

use super::repository_checkout::elapsed_ms;
use super::workspace_script::report_failures;
use crate::tool_runner::{read_request, RunOnceError};
use crate::{prepare_checkout, CheckoutStages, RepositoryDefinition};
use mirror_agent::system_prompt::{
    gather_clean_checkout_environment_info, gather_environment_facts, gather_environment_info,
    load_workspace_context, load_workspace_skills, EnvironmentInfo, ResolvedInstructions,
};

#[derive(Debug, thiserror::Error)]
pub enum PrepareRepositoryError {
    #[error(transparent)]
    Request(#[from] RunOnceError),
    #[error("repository preparation failed: {0}")]
    Prepare(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PreparedContext {
    #[serde(flatten)]
    pub context: ResolvedInstructions,
    #[serde(default)]
    pub environment: Option<EnvironmentInfo>,
    #[serde(default)]
    pub stages: Option<PreparationStages>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PreparationStages {
    pub checkout: CheckoutStages,
    pub context_ms: u64,
    pub environment_ms: u64,
}

/// Names the workspace-relative paths, one per line, that the caller declares the sandbox
/// image recreates on startup and that hold no work of the user's.
pub const DISPOSABLE_PATHS_VARIABLE: &str = "MIRROR_DISPOSABLE_PATHS";

pub async fn prepare_repository_stdio(
    workspace: &Path,
    skills_only: bool,
) -> Result<(), PrepareRepositoryError> {
    #[cfg(target_os = "linux")]
    nix::sys::prctl::set_child_subreaper(true).map_err(std::io::Error::from)?;
    let encoded = read_request().await?;
    let requested =
        RepositoryDefinition::decode(encoded.as_slice()).map_err(RunOnceError::Decode)?;
    let checkout = prepare_checkout(workspace, &requested, &disposable_paths()).await?;
    let root = workspace.join("repository");
    let cwd = root.join(&requested.directory);
    let loading = Instant::now();
    let context = if skills_only {
        load_workspace_skills(&cwd, &root)?
    } else {
        load_workspace_context(&cwd, &root)?
    };
    let context_ms = elapsed_ms(loading);
    let gathering = Instant::now();
    let environment = if skills_only {
        gather_environment_facts(&cwd)
    } else {
        let ran_scripts = checkout.setup.is_some() || checkout.resume.is_some();
        let mut environment = if checkout.reused || ran_scripts {
            gather_environment_info(&cwd)
        } else {
            gather_clean_checkout_environment_info(&cwd)
        };
        report_failures(&mut environment, &checkout);
        environment
    };
    let prepared = PreparedContext {
        context,
        environment: Some(environment),
        stages: Some(PreparationStages {
            checkout,
            context_ms,
            environment_ms: elapsed_ms(gathering),
        }),
    };
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &prepared).map_err(std::io::Error::other)?;
    stdout.flush()?;
    Ok(())
}

fn disposable_paths() -> Vec<String> {
    let declared = std::env::var(DISPOSABLE_PATHS_VARIABLE).unwrap_or_default();
    declared
        .lines()
        .map(|path| path.strip_prefix("./").unwrap_or(path))
        .map(|path| path.strip_suffix('/').unwrap_or(path))
        .filter(|path| !path.is_empty())
        .map(str::to_string)
        .collect()
}
