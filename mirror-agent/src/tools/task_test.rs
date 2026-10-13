use std::future;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use super::*;
use crate::tools::AnyTool;
use crate::AgentApprovalMode;

fn start_tasks() -> (
    crate::TaskManagerHandle,
    mpsc::UnboundedReceiver<crate::task_manager::TaskEvent>,
) {
    let (manager, tasks, events, task_runner) = crate::task_manager::TaskManager::new();
    tokio::spawn(manager.run());
    tokio::spawn(task_runner);
    (tasks, events)
}

#[test]
fn schema_describes_task_actions_and_pagination() {
    let definition = AnyTool::definition(&Task);

    assert_eq!(definition.name, "task");
    assert_eq!(
        definition.parameters["required"],
        serde_json::json!(["action"])
    );
    assert_eq!(definition.parameters["properties"]["limit"]["maximum"], 20);
    assert_eq!(definition.parameters["additionalProperties"], false);
}

#[tokio::test]
async fn lists_and_inspects_paginated_logs() {
    let (tasks, _events) = start_tasks();
    let context = ExecutionContext::new(AgentApprovalMode::Hitl, None, tasks.clone(), None, None);
    let (release, released) = oneshot::channel();
    let task_id = tasks
        .spawn(
            "bash",
            Some("working".to_string()),
            crate::TaskBehavior::Detach,
            |task| async move {
                task.log("progress", "first");
                task.log("progress", "second");
                released.await.expect("released");
                ToolOutput::success("done")
            },
        )
        .await
        .expect("spawn");
    tokio::task::yield_now().await;

    let listed = Task
        .run(
            TaskArgs {
                action: TaskAction::List,
                task_id: None,
                offset: 0,
                limit: 5,
            },
            &context,
        )
        .await
        .expect("list");
    let listed: Value = serde_json::from_str(&listed.content).expect("list json");
    assert_eq!(listed["tasks"][0]["id"], task_id);
    assert_eq!(listed["tasks"][0]["status"], "running");

    let inspected = Task
        .run(
            TaskArgs {
                action: TaskAction::Inspect,
                task_id: Some(task_id.clone()),
                offset: 1,
                limit: 1,
            },
            &context,
        )
        .await
        .expect("inspect");
    let inspected: Value = serde_json::from_str(&inspected.content).expect("inspect json");
    assert_eq!(inspected["log"]["total"], 2);
    assert_eq!(inspected["log"]["next_offset"], Value::Null);
    assert_eq!(inspected["log"]["entries"][0]["content"], "second");

    release.send(()).expect("release");
    let _ = tasks.wait(&task_id).await.expect("wait");
}

#[tokio::test]
async fn waits_cancels_and_reports_unknown_tasks() {
    let (tasks, _events) = start_tasks();
    let context = ExecutionContext::new(AgentApprovalMode::Hitl, None, tasks.clone(), None, None);
    let task_id = tasks
        .spawn("bash", None, crate::TaskBehavior::Detach, |_| {
            future::pending()
        })
        .await
        .expect("spawn");

    let cancelled = Task
        .run(
            TaskArgs {
                action: TaskAction::Cancel,
                task_id: Some(task_id.clone()),
                offset: 0,
                limit: 5,
            },
            &context,
        )
        .await
        .expect("cancel");
    assert_eq!(
        cancelled,
        ToolOutput::success(format!("Task {task_id} cancelled."))
    );

    let waited = Task
        .run(
            TaskArgs {
                action: TaskAction::Wait,
                task_id: Some(task_id),
                offset: 0,
                limit: 5,
            },
            &context,
        )
        .await
        .expect("wait");
    assert_matches!(waited, ToolOutput { success: false, .. });
    assert!(waited.content.contains("was cancelled"));

    let unknown = AnyTool::execute(
        &Task,
        r#"{"action":"inspect","task_id":"missing"}"#,
        &context,
    )
    .await;
    assert_eq!(unknown, ToolOutput::error("task 'missing' not found"));
}

#[tokio::test]
async fn validates_action_specific_arguments() {
    let missing = Task
        .run(
            TaskArgs {
                action: TaskAction::Inspect,
                task_id: None,
                offset: 0,
                limit: 5,
            },
            &ExecutionContext::default(),
        )
        .await;
    assert_matches!(missing, Err(ToolError::InvalidArguments(_)));

    let (tasks, _events) = start_tasks();
    let context = ExecutionContext::new(AgentApprovalMode::Hitl, None, tasks, None, None);
    let invalid = Task
        .run(
            TaskArgs {
                action: TaskAction::List,
                task_id: None,
                offset: 0,
                limit: 21,
            },
            &context,
        )
        .await;
    assert_matches!(invalid, Err(ToolError::InvalidArguments(_)));
}
