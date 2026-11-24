//! Host-relative independent launches; interrupted requests are receipts, never retry queues.

use mirror_protocol::daemon::{AgentLaunchOutcome, Workspace};
use mirror_protocol::mirror::{ListWorkspacesRequest, SessionScope, StartSessionRequest};
use mirror_protocol::{HostToolDefinition, HostToolResult, ToolCall};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::watch;
use tonic::Status;

use crate::SpawnHandle;

/// A private route bound to one daemon-owned worker, not a frontend or saved catalog entry.
#[derive(Clone)]
pub struct AgentSpawner {
    pub registry: SpawnHandle,
    pub workspace_id: String,
    pub worker_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpawnTask {
    pub name: String,
    pub message: String,
    #[serde(default)]
    pub mode: SpawnMode,
    pub directory: Option<String>,
    pub remote_workspace: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpawnMode {
    Folder,
    #[default]
    Worktree,
    Remote,
}

pub struct PreparedSpawn {
    pub workspace: Workspace,
    pub endpoint: Option<String>,
    pub fresh: bool,
}

impl SpawnTask {
    pub fn parse(arguments: &str) -> Result<Self, Status> {
        let task: Self = serde_json::from_str(arguments)
            .map_err(|_| Status::invalid_argument("invalid spawn_agent arguments"))?;
        if task.name.trim().is_empty()
            || task.name.len() > 120
            || task.message.trim().is_empty()
            || task.message.len() > 65536
            || (task.mode == SpawnMode::Remote && task.directory.is_some())
            || (task.mode != SpawnMode::Remote && task.remote_workspace.is_some())
        {
            return Err(Status::invalid_argument("invalid spawn_agent task"));
        }
        Ok(task)
    }
}

impl AgentSpawner {
    pub fn definition() -> HostToolDefinition {
        HostToolDefinition {
            name: "spawn_agent".into(),
            description: "Launch one independent agent using this host's configuration. Folder/worktree stay on this execution host; remote uses this host's configured Mirror connection. Never hops to the laptop or another SSH host. Returns acceptance, not completion. Never automatically retry an unknown outcome, poll, or send follow-ups. Children inherit the calling agent's current approval mode.".into(),
            parameters_json: json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "message": {"type": "string"},
                    "mode": {"type": "string", "enum": ["folder", "worktree", "remote"]},
                    "directory": {"type": "string", "description": "Host-local path, relative to this agent's directory. Defaults to this agent's directory."},
                    "remote_workspace": {"type": "string", "description": "Configured hosted workspace name; omitted creates a hosted scratch session."}
                },
                "required": ["name", "message"],
                "additionalProperties": false
            }).to_string(),
            needs_approval: true,
            timeout_seconds: 300.0,
        }
    }

    pub async fn call(&self, call: ToolCall, mut lifetime: watch::Receiver<()>) -> HostToolResult {
        let observed = lifetime.clone();
        let result = self.spawn(&call, &observed);
        let content = tokio::select! {
            result = result => result.unwrap_or_else(|error| json!({
                "outcome": if error.code() == tonic::Code::Aborted { "unknown" } else { "not_started" }, "error": error.message()
            })),
            _ = tokio::time::sleep(std::time::Duration::from_secs(300)) => json!({
                "outcome": "unknown", "error": "Launch timed out; inspect its receipt. Do not repeat the task."
            }),
            _ = lifetime.changed() => json!({
                "outcome": "unknown", "error": "Calling worker stopped; inspect its launch receipts. Do not repeat the task."
            }),
        };
        HostToolResult {
            call_id: call.id,
            success: matches!(
                content["outcome"].as_str(),
                Some("accepted" | "accepted_but_record_incomplete")
            ),
            content: content.to_string(),
            display: None,
        }
    }

    async fn spawn(
        &self,
        call: &ToolCall,
        lifetime: &watch::Receiver<()>,
    ) -> Result<serde_json::Value, Status> {
        ensure_live(lifetime)?;
        let function = call
            .function
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing tool function"))?;
        let task = SpawnTask::parse(&function.arguments)?;
        let prepared = self
            .registry
            .prepare(self.clone(), call.id.clone(), task.clone())
            .await?;
        let mut workspace = prepared.workspace;
        if !prepared.fresh {
            return Ok(receipt(&workspace));
        }
        // Reservation precedes all dispatch. Every later error is conservatively unknown.
        let dispatch = async {
            ensure_live(lifetime)?;
            if let Some(endpoint) = &prepared.endpoint {
                let remote = self
                    .registry
                    .remote_connections
                    .connect(endpoint, &workspace.session_id)
                    .await?;
                let preset = match task.remote_workspace {
                    Some(name) => Some(
                        remote
                            .runtime()
                            .list_workspaces(ListWorkspacesRequest {})
                            .await?
                            .into_inner()
                            .workspaces
                            .into_iter()
                            .find(|workspace| workspace.name == name)
                            .ok_or_else(|| Status::not_found("unknown hosted workspace"))?,
                    ),
                    None => None,
                };
                ensure_live(lifetime)?;
                remote
                    .lifecycle()
                    .start_session(StartSessionRequest {
                        session_id: workspace.session_id.clone(),
                        scope: SessionScope::Private as i32,
                        workspace: preset,
                        // An SSH parent is not a hosted tree identity.
                        parent_session_id: String::new(),
                        agent_id: String::new(),
                    })
                    .await?;
            }
            ensure_live(lifetime)?;
            let (session, attachment) = self
                .registry
                .start(self.clone(), workspace.id.clone(), prepared.endpoint)
                .await?;
            let result = async {
                ensure_live(lifetime)?;
                let input_id = workspace
                    .agent_launch
                    .as_ref()
                    .ok_or_else(|| Status::internal("launch receipt missing"))?
                    .input_id
                    .clone();
                session
                    .submit_prompt(&attachment, input_id, task.message)
                    .await
            }
            .await;
            session.release_reservation(attachment).await;
            result
        }
        .await;
        let outcome = match dispatch {
            Ok(true) => AgentLaunchOutcome::Accepted,
            Ok(false) => AgentLaunchOutcome::NotStarted,
            Err(_) => AgentLaunchOutcome::Unknown,
        };
        if let Ok(saved) = self.registry.finish(workspace.id.clone(), outcome).await {
            workspace = saved;
        } else if outcome == AgentLaunchOutcome::Accepted {
            let mut result = receipt(&workspace);
            result["outcome"] = "accepted_but_record_incomplete".into();
            return Ok(result);
        }
        Ok(receipt(&workspace))
    }
}

fn ensure_live(lifetime: &watch::Receiver<()>) -> Result<(), Status> {
    if lifetime.has_changed().unwrap_or(true) {
        return Err(Status::cancelled("calling worker stopped"));
    }
    Ok(())
}

fn receipt(workspace: &Workspace) -> serde_json::Value {
    let outcome = workspace
        .agent_launch
        .as_ref()
        .map(|launch| launch.outcome());
    json!({
        "workspace_id": workspace.id,
        "conversation_id": workspace.session_id,
        "name": workspace.name,
        "outcome": match outcome {
            Some(AgentLaunchOutcome::Accepted) => "accepted",
            Some(AgentLaunchOutcome::NotStarted) => "not_started",
            _ => "unknown",
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn destinations_cannot_supply_another_host_or_connection() {
        assert_matches!(
            SpawnTask::parse(r#"{"name":"child","message":"task","sshHost":"laptop"}"#),
            Err(_)
        );
        assert_matches!(
            SpawnTask::parse(
                r#"{"name":"child","message":"task","endpoint":"https://other.invalid","mode":"remote"}"#
            ),
            Err(_)
        );
        assert_matches!(
            SpawnTask::parse(
                r#"{"name":"child","message":"task","mode":"remote","directory":"/host"}"#
            ),
            Err(_)
        );
        assert_matches!(
            SpawnTask::parse(r#"{"name":"child","message":"task","remote_workspace":"hosted"}"#),
            Err(_)
        );
        assert_eq!(
            SpawnTask::parse(r#"{"name":"child","message":"task"}"#)
                .expect("default")
                .mode,
            SpawnMode::Worktree
        );
    }
}
