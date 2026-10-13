use std::future;
use std::task::Poll;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use tokio::sync::oneshot;

use super::*;

fn start_handle() -> (TaskManagerHandle, mpsc::UnboundedReceiver<TaskEvent>) {
    let (manager, handle, events, task_runner) = TaskManager::new();
    tokio::spawn(manager.run());
    tokio::spawn(task_runner);
    (handle, events)
}

#[tokio::test]
async fn reports_lifecycle_updates_and_detached_completion() {
    let (handle, mut events) = start_handle();
    let (release, released) = oneshot::channel();
    let task_id = handle
        .spawn(
            "test",
            Some("starting".to_string()),
            TaskBehavior::Detach,
            |context| async move {
                context.update(Some("working".to_string()));
                released.await.expect("released");
                ToolOutput::success("done")
            },
        )
        .await
        .expect("spawn");

    assert_eq!(task_id, "1000");
    assert_matches!(
        events.recv().await,
        Some(TaskEvent::Updated(TaskInfo {
            status: TaskStatus::Running,
            display: Some(display),
            ..
        })) if display == "starting"
    );
    assert_matches!(
        events.recv().await,
        Some(TaskEvent::Updated(TaskInfo {
            status: TaskStatus::Running,
            display: Some(display),
            ..
        })) if display == "working"
    );

    release.send(()).expect("release");
    assert_eq!(
        handle.wait(&task_id).await.expect("wait"),
        ToolOutput::success("done")
    );
    assert_matches!(
        events.recv().await,
        Some(TaskEvent::Updated(TaskInfo {
            status: TaskStatus::Succeeded,
            ..
        }))
    );
    assert_matches!(
        events.recv().await,
        Some(TaskEvent::Finished(_, ToolOutput { success: true, .. }))
    );
    assert_eq!(handle.list().await.expect("list"), Vec::new());
}

#[tokio::test]
async fn detaches_foreground_work_without_cancelling_it() {
    let (handle, _events) = start_handle();
    let (release, released) = oneshot::channel();
    let task_id = handle
        .spawn("test", None, TaskBehavior::Wait, |_| async move {
            released.await.expect("released");
            ToolOutput::success("done")
        })
        .await
        .expect("spawn");
    let resolver_handle = handle.clone();
    let resolver_id = task_id.clone();
    let resolver = tokio::spawn(async move { resolver_handle.resolve(&resolver_id).await });
    tokio::task::yield_now().await;

    assert_eq!(handle.detach_foreground().await.expect("detach"), 1);
    assert_eq!(
        resolver.await.expect("resolver").expect("resolve"),
        TaskResolution::Detached
    );
    assert_eq!(handle.detach_foreground().await.expect("detach"), 0);

    release.send(()).expect("release");
    assert_eq!(
        handle.wait(&task_id).await.expect("wait"),
        ToolOutput::success("done")
    );
}

#[tokio::test]
async fn dropping_foreground_resolution_cancels_the_task() {
    let (handle, _events) = start_handle();
    let task_id = handle
        .spawn("test", None, TaskBehavior::Wait, |_| future::pending())
        .await
        .expect("spawn");
    let resolver_handle = handle.clone();
    let resolver_id = task_id.clone();
    let resolver = tokio::spawn(async move { resolver_handle.resolve(&resolver_id).await });
    tokio::task::yield_now().await;

    resolver.abort();
    assert_matches!(
        handle.wait(&task_id).await,
        Err(TaskError::Cancelled(id)) if id == task_id
    );
    assert_eq!(
        handle
            .get(&task_id)
            .await
            .expect("get")
            .expect("task")
            .status,
        TaskStatus::Cancelled
    );
}

#[tokio::test]
async fn dropping_detached_resolution_does_not_cancel_the_task() {
    let (handle, _events) = start_handle();
    let task_id = handle
        .spawn("test", None, TaskBehavior::Detach, |_| future::pending())
        .await
        .expect("spawn");
    let mut resolution = Box::pin(handle.resolve(&task_id));

    assert_matches!(futures_util::poll!(resolution.as_mut()), Poll::Pending);
    drop(resolution);
    tokio::task::yield_now().await;

    assert_eq!(handle.list().await.expect("list")[0].id, task_id);
    assert!(handle.cancel(&task_id).await.expect("cancel"));
}

#[tokio::test]
async fn paginates_logs_without_cancelling_a_dropped_waiter() {
    let (handle, _events) = start_handle();
    let (release, released) = oneshot::channel();
    let task_id = handle
        .spawn("test", None, TaskBehavior::Detach, |context| async move {
            context.log("progress", "first");
            context.log("progress", "second");
            context.log("result", "third");
            released.await.expect("released");
            ToolOutput::success("done")
        })
        .await
        .expect("spawn");
    tokio::task::yield_now().await;

    let (entries, total) = handle.logs(&task_id, 1, 1).await.expect("logs");
    assert_eq!(total, 3);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].content, "second");

    let waiter_handle = handle.clone();
    let waiter_id = task_id.clone();
    let waiter = tokio::spawn(async move { waiter_handle.wait(&waiter_id).await });
    tokio::task::yield_now().await;
    waiter.abort();
    assert_eq!(handle.list().await.expect("list").len(), 1);

    release.send(()).expect("release");
    assert_eq!(
        handle.wait(&task_id).await.expect("wait"),
        ToolOutput::success("done")
    );
}

#[tokio::test]
async fn cancelling_spawn_does_not_leave_a_running_task() {
    let (handle, mut events) = start_handle();
    let mut spawn =
        Box::pin(handle.spawn("test", None, TaskBehavior::Detach, |_| future::pending()));

    assert_matches!(futures_util::poll!(spawn.as_mut()), Poll::Pending);
    drop(spawn);
    tokio::task::yield_now().await;

    assert_eq!(handle.list().await.expect("list"), Vec::new());
    assert_matches!(events.try_recv(), Err(mpsc::error::TryRecvError::Empty));
}

#[tokio::test]
async fn panicking_task_finishes_as_a_failed_result() {
    let (handle, _events) = start_handle();
    let task_id = handle
        .spawn("test", None, TaskBehavior::Detach, |_| async {
            panic!("broken");
        })
        .await
        .expect("spawn");

    let result = handle.wait(&task_id).await.expect("wait");

    assert_eq!(result, ToolOutput::error("broken"));
    assert_eq!(
        handle
            .get(&task_id)
            .await
            .expect("get")
            .expect("task")
            .status,
        TaskStatus::Failed
    );
}

#[tokio::test]
async fn cancel_and_close_wait_for_task_cleanup() {
    let (handle, _events) = start_handle();
    let first = handle
        .spawn("first", None, TaskBehavior::Detach, |_| future::pending())
        .await
        .expect("spawn first");
    let second = handle
        .spawn("second", None, TaskBehavior::Detach, |_| future::pending())
        .await
        .expect("spawn second");

    assert!(handle.cancel(&first).await.expect("cancel"));
    assert!(!handle.cancel(&first).await.expect("cancel again"));
    assert_eq!(handle.list().await.expect("list")[0].id, second);

    handle.close().await.expect("close");
    assert_matches!(handle.list().await, Err(TaskError::Closed));
}
