//! Reserve a workspace identity around the worker's snapshot-backed history fork.

use std::path::Path;

use mirror_protocol::daemon::{
    create_workspace_request::Source, CreateWorkspaceRequest, Workspace, WorkspaceKind,
    WorkspaceState,
};
use mirror_protocol::mirror::{ForkWorkspaceRequest, ForkWorkspaceResponse};
use tonic::Status;
use uuid::Uuid;

use crate::{AgentOptions, ExecutionLease, Session};

use super::{describe_repository, fork_destination, WorkspaceRecord, WorkspaceStore};

/// Keep interrupted fork identities visible without retrying or deleting partial checkouts.
pub async fn fork_workspace(
    options: &AgentOptions,
    store: &mut WorkspaceStore,
    source: &WorkspaceRecord,
    session: Session,
    lease: ExecutionLease,
    mut request: ForkWorkspaceRequest,
) -> Result<ForkWorkspaceResponse, Status> {
    Uuid::parse_str(&request.session_id)
        .map_err(|_| Status::invalid_argument("child session ID must be a UUID"))?;
    if source.sandbox_id.is_some() || source.workspace.kind != WorkspaceKind::Directory as i32 {
        return Err(Status::failed_precondition(
            "forking requires a host workspace",
        ));
    }
    if store.get(&request.session_id).is_some() {
        return Err(Status::already_exists(
            "fork identity already exists; inspect its outcome",
        ));
    }
    if request.directory.is_empty() {
        request.directory =
            fork_destination(options, &source.workspace.directory, &request.session_id).await?;
    }
    if !Path::new(&request.directory).is_absolute() {
        return Err(Status::invalid_argument("fork directory must be absolute"));
    }
    let mut record = WorkspaceRecord {
        workspace: Workspace {
            id: request.session_id.clone(),
            session_id: request.session_id.clone(),
            name: request.name.clone(),
            directory: request.directory.clone(),
            kind: WorkspaceKind::Directory as i32,
            is_worktree: true,
            created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            preferences: source.workspace.preferences.clone(),
            state: WorkspaceState::Provisioning as i32,
            ..Default::default()
        },
        creation: CreateWorkspaceRequest {
            workspace_id: request.session_id.clone(),
            name: request.name.clone(),
            preferences: source.workspace.preferences.clone(),
            source: Some(Source::WorktreeFrom(source.workspace.directory.clone())),
            create_directory: false,
        },
        checkout: None,
        sandbox_id: None,
        discovered: false,
    };
    store.save([record.clone()]).await?;
    let result = finish_fork(options, session, lease, request, &mut record).await;
    match &result {
        Ok(_) => record.workspace.state = WorkspaceState::Available as i32,
        Err(error) => {
            record.workspace.state = WorkspaceState::Failed as i32;
            record.workspace.error = Some(error.message().to_owned());
        }
    }
    store.save([record]).await?;
    result
}

async fn finish_fork(
    options: &AgentOptions,
    session: Session,
    lease: ExecutionLease,
    request: ForkWorkspaceRequest,
    record: &mut WorkspaceRecord,
) -> Result<ForkWorkspaceResponse, Status> {
    let directory = request.directory.clone();
    let response = session.fork_workspace(request, lease).await?;
    record.checkout = describe_repository(options, &directory).await?.current;
    record.workspace.directory.clone_from(&response.workspace);
    Ok(response)
}
