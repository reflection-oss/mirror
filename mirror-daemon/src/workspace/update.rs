//! Apply a complete metadata field mask before committing any changes.

use mirror_protocol::daemon::{UpdateWorkspaceRequest, Workspace};
use tonic::Status;

use super::WorkspaceStore;

pub async fn update_workspace(
    store: &mut WorkspaceStore,
    request: UpdateWorkspaceRequest,
) -> Result<Workspace, Status> {
    let fields = request
        .update_mask
        .as_ref()
        .filter(|mask| !mask.paths.is_empty())
        .ok_or_else(|| Status::invalid_argument("nonempty update_mask is required"))?;
    let mut record = store
        .get(&request.workspace_id)
        .ok_or_else(|| Status::not_found("workspace not found"))?
        .clone();
    for field in &fields.paths {
        patch(&mut record.workspace, &request, field)?;
    }
    let workspace = record.workspace.clone();
    store.save([record]).await?;
    Ok(workspace)
}

fn patch(
    workspace: &mut Workspace,
    request: &UpdateWorkspaceRequest,
    field: &str,
) -> Result<(), Status> {
    match field {
        "name" => workspace.name.clone_from(&request.name),
        "preferences" => workspace.preferences.clone_from(&request.preferences),
        field if field.starts_with("preferences.") => {
            let preferences = workspace.preferences.get_or_insert_default();
            let source = request.preferences.clone().unwrap_or_default();
            match field {
                "preferences.approval_mode" => preferences.approval_mode = source.approval_mode,
                "preferences.extensions" => preferences.extensions = source.extensions,
                "preferences.tag_id" => preferences.tag_id = source.tag_id,
                "preferences.definition" => preferences.definition = source.definition,
                field => {
                    let definition = preferences.definition.get_or_insert_default();
                    let source = source.definition.unwrap_or_default();
                    match field {
                        "preferences.definition.model" => definition.model = source.model,
                        "preferences.definition.model_api" => {
                            definition.model_api = source.model_api
                        }
                        "preferences.definition.model_base_url" => {
                            definition.model_base_url = source.model_base_url
                        }
                        "preferences.definition.reasoning_effort" => {
                            definition.reasoning_effort = source.reasoning_effort
                        }
                        "preferences.definition.speed" => definition.speed = source.speed,
                        "preferences.definition.system_prompt" => {
                            definition.system_prompt = source.system_prompt
                        }
                        "preferences.definition.tools_allowlist" => {
                            definition.tools_allowlist = source.tools_allowlist
                        }
                        "preferences.definition.tools_allowlist.names" => {
                            definition.tools_allowlist.get_or_insert_default().names =
                                source.tools_allowlist.unwrap_or_default().names
                        }
                        _ => {
                            return Err(Status::invalid_argument(format!(
                                "unsupported workspace field: {field}"
                            )))
                        }
                    }
                }
            }
        }
        _ => {
            return Err(Status::invalid_argument(format!(
                "unsupported workspace field: {field}"
            )))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::WorkspaceRecord;
    use assert_matches::assert_matches;
    use mirror_protocol::daemon::{CreateWorkspaceRequest, WorkspacePreferences};
    use mirror_protocol::mirror::AgentDefinition;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn speed_update_survives_restart_and_can_be_cleared_without_changing_model() {
        let directory = tempfile::tempdir().expect("workspace directory");
        let id = uuid::Uuid::new_v4().to_string();
        let mut store = WorkspaceStore::open(directory.path())
            .await
            .expect("open store");
        store
            .save([WorkspaceRecord {
                workspace: Workspace {
                    id: id.clone(),
                    session_id: "session".into(),
                    preferences: Some(WorkspacePreferences {
                        definition: Some(AgentDefinition {
                            model: Some("retained-model".into()),
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                creation: CreateWorkspaceRequest {
                    workspace_id: id.clone(),
                    ..Default::default()
                },
                ..Default::default()
            }])
            .await
            .expect("save workspace");
        let request = UpdateWorkspaceRequest {
            workspace_id: id.clone(),
            preferences: Some(WorkspacePreferences {
                definition: Some(AgentDefinition {
                    speed: Some("fast".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            update_mask: Some(prost_types::FieldMask {
                paths: vec!["preferences.definition.speed".into()],
            }),
            ..Default::default()
        };
        update_workspace(&mut store, request.clone())
            .await
            .expect("set speed");
        let mut reopened = WorkspaceStore::open(directory.path())
            .await
            .expect("reopen store");
        let definition = reopened
            .get(&id)
            .expect("workspace")
            .workspace
            .preferences
            .as_ref()
            .expect("preferences")
            .definition
            .as_ref()
            .expect("definition");
        assert_eq!(definition.speed.as_deref(), Some("fast"));
        assert_eq!(definition.model.as_deref(), Some("retained-model"));
        let cleared = update_workspace(
            &mut reopened,
            UpdateWorkspaceRequest {
                preferences: None,
                ..request
            },
        )
        .await
        .expect("clear speed");
        let definition = cleared
            .preferences
            .expect("preferences")
            .definition
            .expect("definition");
        assert_matches!(definition.speed, None);
        assert_eq!(definition.model.as_deref(), Some("retained-model"));
    }
}
