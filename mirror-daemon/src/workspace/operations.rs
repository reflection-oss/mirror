//! Provision explicit workspace intent without merging preferences or starting workers.

use std::path::Path;

use mirror_protocol::daemon::{
    create_workspace_request::Source, AgentLaunch, CreateWorkspaceRequest, SandboxSource,
    Workspace, WorkspaceKind, WorkspaceState,
};
use mirror_protocol::WorkerSandbox;
use tonic::Status;
use uuid::Uuid;

use crate::AgentOptions;

use super::{
    create_sandbox_workspace, create_worktree, delete_sandbox_workspace, describe_repository,
    drop_worktree, host_directory, is_worktree_directory, text, WorkspaceRecord, WorkspaceStore,
};

/// Reopen membership without preparing the directory or starting execution.
pub async fn open_workspace(store: &mut WorkspaceStore, id: &str) -> Result<Workspace, Status> {
    let mut record = store
        .get(id)
        .ok_or_else(|| Status::not_found("workspace not found"))?
        .clone();
    match WorkspaceState::try_from(record.workspace.state) {
        Ok(WorkspaceState::Available) => return Ok(record.workspace),
        Ok(WorkspaceState::Closed) => {}
        _ => {
            return Err(Status::failed_precondition(
                "workspace is not available to open",
            ));
        }
    }
    record.workspace.state = WorkspaceState::Available as i32;
    let workspace = record.workspace.clone();
    store.save([record]).await?;
    Ok(workspace)
}

/// Reserve identity before side effects so retries never reprovision interrupted work.
pub async fn create_workspace(
    options: &AgentOptions,
    store: &mut WorkspaceStore,
    request: CreateWorkspaceRequest,
    launch: Option<AgentLaunch>,
) -> Result<Workspace, Status> {
    if Uuid::parse_str(&request.workspace_id).is_err() {
        return Err(Status::invalid_argument("workspace_id must be a UUID"));
    }
    if let Some(record) = store.get(&request.workspace_id) {
        // Retries compare the original intent, not subsequently edited metadata.
        return if record.creation == request {
            Ok(record.workspace.clone())
        } else {
            Err(Status::already_exists(
                "workspace ID has different creation parameters",
            ))
        };
    }
    let kind = match request.source.as_ref() {
        Some(Source::Directory(_) | Source::WorktreeFrom(_)) => WorkspaceKind::Directory,
        Some(Source::Sandbox(source)) => {
            if source.image.is_empty()
                || !(1..=255).contains(&source.cpus)
                || source.memory_mib < 256
            {
                return Err(Status::invalid_argument("invalid sandbox configuration"));
            }
            WorkspaceKind::Sandbox
        }
        Some(Source::Remote(source)) => {
            if !source.endpoint.is_empty() || source.workspace.is_some() {
                return Err(Status::invalid_argument(
                    "remote endpoints must be supplied when attaching, not stored",
                ));
            }
            if source.session_id.as_deref() != Some(request.workspace_id.as_str()) {
                return Err(Status::invalid_argument(
                    "remote session_id must equal workspace_id",
                ));
            }
            WorkspaceKind::Remote
        }
        None => return Err(Status::invalid_argument("workspace source is required")),
    };
    let mut record = WorkspaceRecord {
        workspace: Workspace {
            id: request.workspace_id.clone(),
            session_id: if kind == WorkspaceKind::Remote {
                request.workspace_id.clone()
            } else {
                Uuid::new_v4().to_string()
            },
            name: request.name.clone(),
            kind: kind as i32,
            is_worktree: matches!(&request.source, Some(Source::WorktreeFrom(_))),
            created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            preferences: request.preferences.clone(),
            agent_launch: launch,
            state: WorkspaceState::Provisioning as i32,
            ..Workspace::default()
        },
        creation: request,
        checkout: None,
        sandbox_id: None,
        discovered: false,
        personal_context: None,
    };
    if let Some(launch) = &mut record.workspace.agent_launch {
        launch.session_id = record.workspace.session_id.clone();
    }
    store.save([record.clone()]).await?;
    match provision(options, store, &mut record).await {
        Ok(()) => record.workspace.state = WorkspaceState::Available as i32,
        Err(error) => {
            record.workspace.state = WorkspaceState::Failed as i32;
            record.workspace.error = Some(error.message().to_owned());
            store.save([record]).await?;
            return Err(error);
        }
    }
    let workspace = record.workspace.clone();
    store.save([record]).await?;
    Ok(workspace)
}

async fn provision(
    options: &AgentOptions,
    store: &WorkspaceStore,
    record: &mut WorkspaceRecord,
) -> Result<(), Status> {
    let directory = match record.creation.source.as_ref() {
        Some(Source::Directory(directory)) => {
            if record.creation.create_directory {
                let resolved =
                    super::resolve_directory(options, directory, super::text(&options.home)?)
                        .await?;
                if resolved.directory != *directory {
                    return Err(Status::failed_precondition(
                        "workspace directory identity changed",
                    ));
                }
                if !resolved.exists {
                    tokio::fs::create_dir(directory).await.map_err(|_| {
                        Status::failed_precondition("could not create workspace directory")
                    })?;
                }
            }
            let directory = host_directory(options, directory).await?;
            record.workspace.is_worktree = is_worktree_directory(&directory).await?;
            directory
        }
        Some(Source::WorktreeFrom(directory)) => {
            let checkout = create_worktree(options, directory, Some(&record.workspace.id)).await?;
            let directory = checkout.directory.clone().into();
            record.checkout = Some(checkout);
            directory
        }
        Some(Source::Sandbox(source)) => {
            let source_directory = source
                .source_workspace_id
                .as_deref()
                .map(|id| {
                    store
                        .get(id)
                        .map(|record| record.workspace.directory.as_str())
                        .ok_or_else(|| Status::not_found("workspace not found"))
                })
                .transpose()?;
            let directory =
                create_sandbox_workspace(options, &record.workspace.id, source_directory).await?;
            record.sandbox_id = Some(record.workspace.id.clone());
            directory
        }
        Some(Source::Remote(_)) => return Ok(()),
        None => return Err(Status::invalid_argument("workspace source is required")),
    };
    record.workspace.directory = text(&directory)?.to_owned();
    let database = options.database.clone();
    let session_id = record.workspace.session_id.clone();
    let created_at = record
        .workspace
        .created_at
        .ok_or_else(|| Status::internal("workspace creation timestamp is missing"))?;
    tokio::task::spawn_blocking(move || {
        mirror_agent::local::create_conversation_metadata(
            &database,
            &session_id,
            &directory,
            created_at,
        )
    })
    .await
    .map_err(|_| Status::internal("conversation reservation task failed"))?
    .map_err(|error| Status::failed_precondition(error.to_string()))?;
    Ok(())
}

/// Execution authority comes from AttachWorkspace, not these persisted provisioning facts.
pub fn sandbox_configuration(
    options: &AgentOptions,
    record: &WorkspaceRecord,
) -> Option<WorkerSandbox> {
    match (&record.creation.source, &record.sandbox_id) {
        (Some(Source::Sandbox(source)), Some(id)) => Some(WorkerSandbox {
            id: id.clone(),
            sandboxd: options
                .bundle
                .join("linux/mirror-sandboxd")
                .to_string_lossy()
                .into_owned(),
            source: Some(SandboxSource {
                source_workspace_id: None,
                ..source.clone()
            }),
        }),
        _ => None,
    }
}

/// Call only after live workers/processes and overlapping store directories are excluded.
pub async fn delete_files(options: &AgentOptions, record: &WorkspaceRecord) -> Result<(), Status> {
    if let Some(sandbox) = sandbox_configuration(options, record) {
        return delete_sandbox_workspace(options, &record.workspace.directory, &sandbox).await;
    }
    if record.workspace.kind != WorkspaceKind::Directory as i32 || !record.workspace.is_worktree {
        return Err(Status::failed_precondition(
            "only managed worktrees and sandboxes can delete files",
        ));
    }
    let directory = host_directory(options, &record.workspace.directory).await?;
    if directory != Path::new(&record.workspace.directory) {
        return Err(Status::failed_precondition(
            "workspace directory identity changed",
        ));
    }
    if !directory.starts_with(options.home.join(".mirror/worktrees"))
        || !is_worktree_directory(&directory).await?
    {
        return Err(Status::failed_precondition(
            "file deletion requires a managed linked checkout root",
        ));
    }
    let checkout = match &record.checkout {
        Some(checkout) if checkout.directory == record.workspace.directory => checkout.clone(),
        _ => describe_repository(options, text(&directory)?)
            .await?
            .current
            .ok_or_else(|| Status::failed_precondition("checkout is unavailable"))?,
    };
    drop_worktree(options, &checkout).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    fn options(home: &Path) -> AgentOptions {
        AgentOptions {
            executable: home.join("no-worker"),
            bundle: home.to_owned(),
            database: home.join("state.db"),
            home: home.to_owned(),
            runtime_directory: home.to_owned(),
            state_directory: home.to_owned(),
            path: String::new(),
        }
    }

    #[tokio::test]
    async fn folder_creation_needs_no_vcs_process_and_timestamps_before_agent_start() {
        let home = tempfile::tempdir().expect("home");
        let directory = home.path().canonicalize().expect("canonical home");
        std::fs::create_dir(directory.join(".git")).expect("repository marker");
        let options = options(&directory);
        let mut store = WorkspaceStore::open(&directory).await.expect("catalog");
        let request = CreateWorkspaceRequest {
            workspace_id: Uuid::new_v4().to_string(),
            source: Some(Source::Directory(directory.to_string_lossy().into_owned())),
            ..Default::default()
        };
        let workspace = create_workspace(&options, &mut store, request.clone(), None)
            .await
            .expect("folder without Git executable");
        assert_eq!(workspace.state, WorkspaceState::Available as i32);
        assert!(!workspace.is_worktree);
        assert_matches!(workspace.created_at, Some(timestamp) if timestamp.seconds > 0);
        assert_eq!(
            create_workspace(&options, &mut store, request, None)
                .await
                .expect("retry"),
            workspace
        );
        let summaries = mirror_agent::local::list_conversations(
            &options.database,
            &mirror_protocol::daemon::ListConversationsRequest {
                limit: 10,
                recent_first: true,
                ..Default::default()
            },
        )
        .expect("immediate Recent metadata");
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].id, workspace.session_id);
        assert_eq!(summaries[0].updated_at, workspace.created_at);
        assert!(!directory.join("state.db.locks").exists());
    }

    #[tokio::test]
    async fn delete_files_rejects_a_folder_even_with_a_forged_worktree_flag() {
        let home = tempfile::tempdir().expect("home");
        let home = home.path().canonicalize().expect("canonical home");
        let directory = home.join(".mirror/worktrees/project/nested");
        std::fs::create_dir_all(&directory).expect("folder");
        let file = directory.join("uncommitted.txt");
        std::fs::write(&file, "keep me").expect("uncommitted file");
        let mut record = WorkspaceRecord {
            workspace: Workspace {
                directory: directory.to_string_lossy().into_owned(),
                kind: WorkspaceKind::Directory as i32,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_matches!(delete_files(&options(&home), &record).await, Err(error) if error.code() == tonic::Code::FailedPrecondition);
        record.workspace.is_worktree = true;
        assert_matches!(delete_files(&options(&home), &record).await, Err(error) if error.code() == tonic::Code::FailedPrecondition);
        assert_eq!(
            std::fs::read_to_string(file).expect("preserved file"),
            "keep me"
        );
    }

    #[tokio::test]
    async fn snapshot_describes_worktree_intent_without_inspecting_checkout() {
        let home = tempfile::tempdir().expect("isolated home");
        let directory = home.path().canonicalize().expect("canonical home");
        let options = options(&directory);
        let mut store = WorkspaceStore::open(&directory).await.expect("catalog");
        let id = Uuid::new_v4().to_string();
        store
            .save([WorkspaceRecord {
                workspace: Workspace {
                    id: id.clone(),
                    session_id: id.clone(),
                    directory: "/missing/checkout".into(),
                    kind: WorkspaceKind::Directory as i32,
                    state: WorkspaceState::Available as i32,
                    ..Default::default()
                },
                creation: CreateWorkspaceRequest {
                    workspace_id: id,
                    source: Some(Source::WorktreeFrom("/missing/source".into())),
                    ..Default::default()
                },
                ..Default::default()
            }])
            .await
            .expect("save intent");
        let models = mirror_agent::local_models::Host::bind(
            directory.join("models"),
            options.executable.clone(),
            None,
            0,
        )
        .await
        .expect("isolated model service");
        let (registry, task) =
            crate::WorkspaceRegistry::spawn(options.clone(), models.service.clone())
                .await
                .expect("registry");
        let snapshot = registry
            .snapshot(Default::default())
            .await
            .expect("snapshot");
        registry.shutdown();
        task.await.expect("stop registry");
        models.close().await;
        assert_eq!(snapshot.workspaces.len(), 1);
        assert!(snapshot.workspaces[0].is_worktree);
        assert!(!options.database.exists());
    }
}
