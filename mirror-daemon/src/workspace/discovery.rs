//! Repository discovery adds metadata without granting execution or filesystem ownership.

use std::path::Path;

use mirror_protocol::daemon::{
    create_workspace_request::Source, CreateWorkspaceRequest, WatchWorkspacesRequest,
    WatchWorkspacesResponse, Workspace, WorkspaceKind, WorkspaceState,
};
use tonic::Status;
use uuid::Uuid;

use super::{Repository, WorkspaceRecord, WorkspaceStore};

pub async fn discover_workspaces(
    store: &mut WorkspaceStore,
    include_unmanaged: bool,
    repository: &Repository,
) -> Result<(), Status> {
    let mut records: Vec<WorkspaceRecord> = Vec::new();
    for checkout in &repository.checkouts {
        let current = repository
            .current
            .as_ref()
            .is_some_and(|current| current.id == checkout.id);
        if (!include_unmanaged && !current && !checkout.managed)
            || store.ignores(&checkout.directory)
            || store
                .records()
                .chain(records.iter())
                .any(|record| record.workspace.directory == checkout.directory)
        {
            continue;
        }
        let id = Uuid::new_v4().to_string();
        let name = checkout.name.clone().unwrap_or_else(|| {
            Path::new(&checkout.directory)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
        records.push(WorkspaceRecord {
            workspace: Workspace {
                id: id.clone(),
                name: name.clone(),
                directory: checkout.directory.clone(),
                session_id: Uuid::new_v4().to_string(),
                kind: WorkspaceKind::Directory.into(),
                is_worktree: repository
                    .checkouts
                    .first()
                    .is_some_and(|primary| primary.id != checkout.id),
                created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                state: WorkspaceState::Available.into(),
                ..Default::default()
            },
            creation: CreateWorkspaceRequest {
                workspace_id: id,
                name,
                source: Some(Source::Directory(checkout.directory.clone())),
                ..Default::default()
            },
            checkout: Some(checkout.clone()),
            sandbox_id: None,
            discovered: !current,
        });
    }
    if !records.is_empty() {
        store.save(records).await?;
    }
    Ok(())
}

pub fn workspace_snapshot(
    store: &WorkspaceStore,
    request: &WatchWorkspacesRequest,
    repository: Option<&Repository>,
) -> WatchWorkspacesResponse {
    let mut workspaces: Vec<Workspace> = store
        .records()
        .filter(|record| record.workspace.id > request.after_id)
        .filter(|record| {
            let workspace = &record.workspace;
            let checkout_visible = !record.discovered
                || request.include_unmanaged
                || record
                    .checkout
                    .as_ref()
                    .is_some_and(|checkout| checkout.managed)
                || repository
                    .and_then(|repository| repository.current.as_ref())
                    .is_some_and(|current| current.directory == workspace.directory);
            let in_repository = repository.is_none_or(|repository| {
                repository.checkouts.iter().any(|checkout| {
                    Path::new(&workspace.directory).starts_with(&checkout.directory)
                })
            });
            checkout_visible
                && in_repository
                && (request.include_closed || workspace.state != WorkspaceState::Closed as i32)
        })
        .take(if request.limit == 0 {
            usize::MAX
        } else {
            request.limit as usize + 1
        })
        .map(|record| record.workspace.clone())
        .collect();
    let more = request.limit != 0 && workspaces.len() > request.limit as usize;
    if more {
        workspaces.truncate(request.limit as usize);
    }
    let next_after_id = workspaces
        .last()
        .filter(|_| more)
        .map(|workspace| workspace.id.clone())
        .unwrap_or_default();
    WatchWorkspacesResponse {
        workspaces,
        next_after_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Checkout;
    use pretty_assertions::{assert_eq, assert_ne};

    #[tokio::test]
    async fn discovery_distinguishes_primary_folder_from_unmanaged_worktree() {
        let directory = tempfile::tempdir().expect("catalog directory");
        let mut store = WorkspaceStore::open(directory.path())
            .await
            .expect("catalog");
        let primary = Checkout {
            id: "primary".into(),
            directory: "/fixture/project".into(),
            provider: "git".into(),
            ..Default::default()
        };
        let secondary = Checkout {
            id: "secondary".into(),
            directory: "/fixture/checkout".into(),
            provider: "git".into(),
            ..Default::default()
        };
        let repository = Repository {
            key: "fixture".into(),
            current: Some(primary.clone()),
            checkouts: vec![primary, secondary],
        };
        discover_workspaces(&mut store, true, &repository)
            .await
            .expect("discover");
        let snapshot = workspace_snapshot(
            &store,
            &WatchWorkspacesRequest {
                include_unmanaged: true,
                ..Default::default()
            },
            Some(&repository),
        );
        let primary = snapshot
            .workspaces
            .iter()
            .find(|record| record.directory == "/fixture/project")
            .expect("primary");
        let secondary = snapshot
            .workspaces
            .iter()
            .find(|record| record.directory == "/fixture/checkout")
            .expect("secondary");
        assert_eq!(primary.kind, WorkspaceKind::Directory as i32);
        assert!(!primary.is_worktree);
        assert_eq!(secondary.kind, WorkspaceKind::Directory as i32);
        assert!(secondary.is_worktree);
        assert_eq!(secondary.state, WorkspaceState::Available as i32);
        let first = workspace_snapshot(
            &store,
            &WatchWorkspacesRequest {
                include_unmanaged: true,
                limit: 1,
                ..Default::default()
            },
            None,
        );
        assert_eq!(first.workspaces.len(), 1);
        assert_eq!(first.next_after_id, first.workspaces[0].id);
        let next = workspace_snapshot(
            &store,
            &WatchWorkspacesRequest {
                include_unmanaged: true,
                limit: 1,
                after_id: first.next_after_id,
                ..Default::default()
            },
            None,
        );
        assert_eq!(next.workspaces.len(), 1);
        assert_ne!(first.workspaces[0].id, next.workspaces[0].id);
        assert!(next.next_after_id.is_empty());
    }
}
