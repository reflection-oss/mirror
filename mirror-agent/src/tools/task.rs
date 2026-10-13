//! Model-facing task lifecycle operations.

use std::time::SystemTime;

use crate::tools::ToolOutput;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::Tool;
use super::ToolError;
use crate::ExecutionContext;
use crate::RuntimeTaskStatus;
use crate::TaskInfo;

/// Manage non-blocking tasks owned by the active conversation.
#[derive(Debug, Clone, Copy)]
pub struct Task;

impl Tool for Task {
    type Args = TaskArgs;

    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Manage non-blocking tasks started by tools. List running tasks, inspect a task's paginated log, wait for completion, or cancel a running task."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "action".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "enum": ["list", "inspect", "wait", "cancel"],
                        "description": "Operation to perform on non-blocking tasks."
                    }),
                ),
                (
                    "task_id".to_string(),
                    serde_json::json!({
                        "type": ["string", "null"],
                        "default": null,
                        "description": "Task id. Required for inspect, wait, and cancel."
                    }),
                ),
                (
                    "offset".to_string(),
                    serde_json::json!({
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Zero-based task-log offset for inspect."
                    }),
                ),
                (
                    "limit".to_string(),
                    serde_json::json!({
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Maximum task-log entries returned by inspect."
                    }),
                ),
            ]),
            &["action"],
        )
    }

    fn supports_remote(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: Self::Args,
        context: &ExecutionContext,
        _remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        // This tool uses a narrow host service, not the workspace or remote transport.
        self.run(args, context).await
    }

    async fn run(
        &self,
        args: TaskArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        if !(1..=20).contains(&args.limit) {
            return Err(ToolError::InvalidArguments(
                "limit must be between 1 and 20".to_string(),
            ));
        }
        let tasks = context.tasks();
        match args.action {
            TaskAction::List => {
                let tasks = tasks
                    .list()
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?;
                let tasks = tasks.iter().map(task_payload).collect::<Vec<_>>();
                Ok(ToolOutput::success(
                    serde_json::json!({"tasks": tasks}).to_string(),
                ))
            }
            TaskAction::Inspect => {
                let task_id = args.task_id.as_deref().ok_or_else(|| {
                    ToolError::InvalidArguments("task_id is required for inspect".to_string())
                })?;
                let Some(info) = tasks
                    .get(task_id)
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?
                else {
                    return Ok(ToolOutput::error(format!("task '{task_id}' not found")));
                };
                let (entries, total) = tasks
                    .logs(task_id, args.offset, args.limit)
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?;
                let next_offset = args.offset.saturating_add(entries.len());
                let entries = entries
                    .iter()
                    .map(|entry| {
                        serde_json::json!({
                            "kind": entry.kind,
                            "content": entry.content,
                            "elapsed_seconds": elapsed(entry.timestamp, info.started_at),
                        })
                    })
                    .collect::<Vec<_>>();
                let mut payload = task_payload(&info);
                payload["log"] = serde_json::json!({
                    "offset": args.offset,
                    "limit": args.limit,
                    "total": total,
                    "next_offset": (next_offset < total).then_some(next_offset),
                    "entries": entries,
                });
                Ok(ToolOutput::success(payload.to_string()))
            }
            TaskAction::Wait => {
                let task_id = args.task_id.as_deref().ok_or_else(|| {
                    ToolError::InvalidArguments("task_id is required for wait".to_string())
                })?;
                if tasks
                    .get(task_id)
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?
                    .is_none()
                {
                    return Ok(ToolOutput::error(format!("task '{task_id}' not found")));
                }
                match tasks.wait(task_id).await {
                    Ok(_) => Ok(ToolOutput::success(format!(
                        "Task {task_id} completed; its result was delivered separately."
                    ))),
                    Err(error) => Ok(ToolOutput::error(error.to_string())),
                }
            }
            TaskAction::Cancel => {
                let task_id = args.task_id.as_deref().ok_or_else(|| {
                    ToolError::InvalidArguments("task_id is required for cancel".to_string())
                })?;
                let Some(info) = tasks
                    .get(task_id)
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?
                else {
                    return Ok(ToolOutput::error(format!("task '{task_id}' not found")));
                };
                if info.status != RuntimeTaskStatus::Running {
                    return Ok(ToolOutput::error(format!(
                        "task '{task_id}' is already {}",
                        status_name(info.status)
                    )));
                }
                if tasks
                    .cancel(task_id)
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?
                {
                    return Ok(ToolOutput::success(format!("Task {task_id} cancelled.")));
                }
                let Some(info) = tasks
                    .get(task_id)
                    .await
                    .map_err(|error| ToolError::Execution(error.to_string()))?
                else {
                    return Ok(ToolOutput::error(format!("task '{task_id}' not found")));
                };
                Ok(ToolOutput::error(format!(
                    "task '{task_id}' is already {}",
                    status_name(info.status)
                )))
            }
        }
    }
}

/// Arguments accepted by the task tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskArgs {
    pub action: TaskAction,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

/// Operation requested from the task tool.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum TaskAction {
    /// List running tasks.
    List,
    /// Read one page of a task's log.
    Inspect,
    /// Wait until a task reaches a terminal state.
    Wait,
    /// Cancel running task work.
    Cancel,
}

fn default_limit() -> usize {
    5
}

fn task_payload(info: &TaskInfo) -> Value {
    serde_json::json!({
        "id": info.id,
        "tool_name": info.tool_name,
        "status": status_name(info.status),
        "display": info.display,
        "elapsed_seconds": elapsed(info.ended_at.unwrap_or_else(SystemTime::now), info.started_at),
    })
}

fn status_name(status: RuntimeTaskStatus) -> &'static str {
    match status {
        RuntimeTaskStatus::Running => "running",
        RuntimeTaskStatus::Succeeded => "succeeded",
        RuntimeTaskStatus::Failed => "failed",
        RuntimeTaskStatus::Cancelled => "cancelled",
    }
}

fn elapsed(end: SystemTime, start: SystemTime) -> f64 {
    let seconds = end.duration_since(start).unwrap_or_default().as_secs_f64();
    (seconds * 10.0).round() / 10.0
}

#[cfg(test)]
#[path = "task_test.rs"]
mod tests;
