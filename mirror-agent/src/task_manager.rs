//! Sequential owner of one conversation's task lifecycle.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::time::SystemTime;

use futures_util::stream::FuturesUnordered;
use futures_util::FutureExt;
use futures_util::StreamExt;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::tools::ToolOutput;

/// Own task state for one conversation.
pub(crate) struct TaskManager {
    commands: mpsc::UnboundedReceiver<Command>,
    events: mpsc::UnboundedSender<TaskEvent>,
    tasks: Vec<Task>,
    next_id: u64,
    closing: bool,
    close_waiters: Vec<oneshot::Sender<()>>,
}

/// Cloneable local handle for one conversation's task manager.
#[derive(Debug, Clone)]
pub struct TaskManagerHandle {
    commands: mpsc::UnboundedSender<Command>,
    task_sender: mpsc::UnboundedSender<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>,
}

/// Current state of one task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskInfo {
    /// Conversation-local task identifier.
    pub id: String,
    /// Tool that started the task.
    pub tool_name: String,
    /// Latest terminal summary.
    pub display: Option<String>,
    /// Current lifecycle state.
    pub status: TaskStatus,
    /// Time the task was accepted.
    pub started_at: SystemTime,
    /// Time the task reached a terminal state.
    pub ended_at: Option<SystemTime>,
    /// Whether the active agent turn is no longer waiting for this task.
    pub detached: bool,
}

/// One append-only task log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskLogEntry {
    /// Producer-defined log category.
    pub kind: String,
    /// Full inspectable log content.
    pub content: String,
    /// Time this entry was appended.
    pub timestamp: SystemTime,
}

/// Task lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskStatus {
    /// Work is still running.
    Running,
    /// Work completed successfully.
    Succeeded,
    /// Work completed with a tool error.
    Failed,
    /// Work was cancelled before completion.
    Cancelled,
}

/// Whether a caller initially waits for task completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskBehavior {
    /// Keep the task attached to the active tool call.
    Wait,
    /// Return immediately and let the task continue independently.
    Detach,
}

/// Result of observing foreground work.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskResolution {
    /// Work finished while still attached.
    Finished(ToolOutput),
    /// Work continues independently from the active tool call.
    Detached,
}

/// Operations available to running task work.
#[derive(Debug, Clone)]
pub struct TaskContext {
    task_id: String,
    commands: mpsc::UnboundedSender<Command>,
    task_sender: mpsc::UnboundedSender<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>,
}

struct Task {
    id: String,
    tool_name: String,
    display: Option<String>,
    started_at: SystemTime,
    behavior: TaskBehavior,
    log: Vec<TaskLogEntry>,
    state: TaskState,
}

enum TaskState {
    Running {
        cancellation: Cancellation,
        waiters: Vec<oneshot::Sender<Result<ToolOutput, TaskError>>>,
        resolvers: Vec<oneshot::Sender<Result<TaskResolution, TaskError>>>,
        cancel_waiters: Vec<oneshot::Sender<bool>>,
    },
    Finished {
        result: ToolOutput,
        ended_at: SystemTime,
    },
    Cancelled {
        ended_at: SystemTime,
    },
}

enum Cancellation {
    Available(oneshot::Sender<()>),
    Requested,
}

/// Internal lifecycle events consumed exclusively by `ReactAgentService`.
#[derive(Debug)]
pub(crate) enum TaskEvent {
    Updated(TaskInfo),
    Finished(TaskInfo, ToolOutput),
}

impl Default for TaskManagerHandle {
    fn default() -> Self {
        Self::closed()
    }
}

impl TaskManagerHandle {
    pub(crate) fn closed() -> Self {
        let (commands, receiver) = mpsc::unbounded_channel();
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        drop(receiver);
        drop(task_receiver);
        Self {
            commands,
            task_sender,
        }
    }

    /// Start task work and return its conversation-local id.
    pub async fn spawn<F, Fut>(
        &self,
        tool_name: impl Into<String>,
        display: Option<String>,
        behavior: TaskBehavior,
        work: F,
    ) -> Result<String, TaskError>
    where
        F: FnOnce(TaskContext) -> Fut,
        Fut: Future<Output = ToolOutput> + Send + 'static,
    {
        let (cancel, cancelled) = oneshot::channel();
        let task_id = self
            .request(|response| Command::Spawn {
                tool_name: tool_name.into(),
                display,
                behavior,
                cancel,
                response,
            })
            .await??;
        let context = TaskContext {
            task_id: task_id.clone(),
            commands: self.commands.clone(),
            task_sender: self.task_sender.clone(),
        };
        if self
            .task_sender
            .send(Box::pin(run_task(
                task_id.clone(),
                work(context),
                cancelled,
                self.commands.clone(),
            )))
            .is_err()
        {
            let _ = self.commands.send(Command::Cancel {
                task_id: task_id.clone(),
            });
            return Err(TaskError::Closed);
        }
        Ok(task_id)
    }

    /// Resolve foreground work, cancelling it if this observer is dropped.
    pub async fn resolve(&self, task_id: &str) -> Result<TaskResolution, TaskError> {
        let mut cancel = CancelOnDrop {
            task_id: Some(task_id.to_string()),
            commands: self.commands.clone(),
        };
        let resolution = self
            .request(|response| Command::Resolve {
                task_id: task_id.to_string(),
                response,
            })
            .await?;
        cancel.task_id = None;
        resolution
    }

    /// Return the latest snapshot for one task.
    pub async fn get(&self, task_id: &str) -> Result<Option<TaskInfo>, TaskError> {
        self.request(|response| Command::Get {
            task_id: task_id.to_string(),
            response,
        })
        .await
    }

    /// Return running tasks in creation order.
    pub async fn list(&self) -> Result<Vec<TaskInfo>, TaskError> {
        self.request(|response| Command::List { response }).await
    }

    /// Return a stable page from one task's append-only log.
    pub async fn logs(
        &self,
        task_id: &str,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<TaskLogEntry>, usize), TaskError> {
        self.request(|response| Command::Logs {
            task_id: task_id.to_string(),
            offset,
            limit,
            response,
        })
        .await?
    }

    /// Wait for completion without making caller cancellation cancel the task.
    pub async fn wait(&self, task_id: &str) -> Result<ToolOutput, TaskError> {
        self.request(|response| Command::Wait {
            task_id: task_id.to_string(),
            response,
        })
        .await?
    }

    /// Cancel one running task and wait for its cleanup.
    pub async fn cancel(&self, task_id: &str) -> Result<bool, TaskError> {
        self.request(|response| Command::RequestCancel {
            task_id: task_id.to_string(),
            response,
        })
        .await
    }

    /// Detach every running foreground task.
    pub async fn detach_foreground(&self) -> Result<usize, TaskError> {
        self.request(|response| Command::DetachForeground { response })
            .await
    }

    /// Cancel every running task and close the lifecycle actor.
    pub async fn close(&self) -> Result<(), TaskError> {
        self.request(|response| Command::Close { response }).await
    }

    async fn request<T>(
        &self,
        command: impl FnOnce(oneshot::Sender<T>) -> Command,
    ) -> Result<T, TaskError> {
        let (response, receiver) = oneshot::channel();
        self.commands
            .send(command(response))
            .map_err(|_| TaskError::Closed)?;
        receiver.await.map_err(|_| TaskError::Closed)
    }
}

impl TaskContext {
    /// The conversation-local task id.
    pub(crate) fn task_id(&self) -> &str {
        &self.task_id
    }

    pub(crate) fn spawn(
        &self,
        task: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), TaskError> {
        self.task_sender
            .send(Box::pin(task))
            .map_err(|_| TaskError::Closed)
    }

    /// Replace the task's current terminal summary.
    pub fn update(&self, display: Option<String>) {
        let _ = self.commands.send(Command::Display {
            task_id: self.task_id.clone(),
            display,
        });
    }

    /// Append an inspectable task log entry.
    pub fn log(&self, kind: impl Into<String>, content: impl Into<String>) {
        let _ = self.commands.send(Command::Log {
            task_id: self.task_id.clone(),
            kind: kind.into(),
            content: content.into(),
        });
    }
}

/// Task lifecycle command failure.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskError {
    /// The lifecycle actor is no longer accepting commands.
    #[error("task runtime closed")]
    Closed,
    /// No more conversation-local task identifiers are available.
    #[error("task identifier space exhausted")]
    IdExhausted,
    /// No task has the requested identifier.
    #[error("task '{0}' not found")]
    NotFound(String),
    /// The task was cancelled before producing a result.
    #[error("task '{0}' was cancelled")]
    Cancelled(String),
}

enum Command {
    Spawn {
        tool_name: String,
        display: Option<String>,
        behavior: TaskBehavior,
        cancel: oneshot::Sender<()>,
        response: oneshot::Sender<Result<String, TaskError>>,
    },
    Resolve {
        task_id: String,
        response: oneshot::Sender<Result<TaskResolution, TaskError>>,
    },
    Get {
        task_id: String,
        response: oneshot::Sender<Option<TaskInfo>>,
    },
    List {
        response: oneshot::Sender<Vec<TaskInfo>>,
    },
    Logs {
        task_id: String,
        offset: usize,
        limit: usize,
        response: oneshot::Sender<Result<(Vec<TaskLogEntry>, usize), TaskError>>,
    },
    Wait {
        task_id: String,
        response: oneshot::Sender<Result<ToolOutput, TaskError>>,
    },
    RequestCancel {
        task_id: String,
        response: oneshot::Sender<bool>,
    },
    CancelForeground {
        task_id: String,
    },
    DetachForeground {
        response: oneshot::Sender<usize>,
    },
    Display {
        task_id: String,
        display: Option<String>,
    },
    Log {
        task_id: String,
        kind: String,
        content: String,
    },
    Finish {
        task_id: String,
        result: ToolOutput,
    },
    Cancel {
        task_id: String,
    },
    Close {
        response: oneshot::Sender<()>,
    },
}

impl TaskManager {
    /// Create the manager, its handle, and its local task runner.
    pub(crate) fn new() -> (
        Self,
        TaskManagerHandle,
        mpsc::UnboundedReceiver<TaskEvent>,
        impl Future<Output = ()>,
    ) {
        let (commands, command_receiver) = mpsc::unbounded_channel();
        let (events, event_receiver) = mpsc::unbounded_channel();
        let (task_sender, mut task_receiver) = mpsc::unbounded_channel();
        let task_runner = async move {
            let mut running = FuturesUnordered::new();
            loop {
                if running.is_empty() {
                    let Some(task) = task_receiver.recv().await else {
                        return;
                    };
                    running.push(task);
                    continue;
                }
                tokio::select! {
                    task = task_receiver.recv() => match task {
                        Some(task) => running.push(task),
                        None => {
                            while running.next().await.is_some() {}
                            return;
                        }
                    },
                    _ = running.next() => {}
                }
            }
        };
        (
            Self {
                commands: command_receiver,
                events,
                tasks: Vec::new(),
                next_id: 1_000,
                closing: false,
                close_waiters: Vec::new(),
            },
            TaskManagerHandle {
                commands,
                task_sender,
            },
            event_receiver,
            task_runner,
        )
    }

    /// Process commands until closure completes.
    pub(crate) async fn run(mut self) {
        while let Some(command) = self.commands.recv().await {
            for task in &mut self.tasks {
                task.prune_closed_waiters();
            }
            self.handle(command);
            if self.closing && !self.tasks.iter().any(Task::is_running) {
                for response in self.close_waiters.drain(..) {
                    let _ = response.send(());
                }
                return;
            }
        }
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Spawn {
                tool_name,
                display,
                behavior,
                cancel,
                response,
            } => {
                if self.closing {
                    let _ = response.send(Err(TaskError::Closed));
                    return;
                }
                let id = match self.new_id() {
                    Ok(id) => id,
                    Err(error) => {
                        let _ = response.send(Err(error));
                        return;
                    }
                };
                self.tasks
                    .push(Task::new(id.clone(), tool_name, display, behavior, cancel));
                if response.send(Ok(id.clone())).is_err() {
                    self.tasks.retain(|task| task.id != id);
                } else if let Some(task) = Self::task(&self.tasks, &id) {
                    let _ = self.events.send(TaskEvent::Updated(task.info()));
                }
            }
            Command::Resolve { task_id, response } => self.resolve(&task_id, response),
            Command::Get { task_id, response } => {
                let _ = response.send(Self::task(&self.tasks, &task_id).map(Task::info));
            }
            Command::List { response } => {
                let tasks = self
                    .tasks
                    .iter()
                    .filter(|task| task.is_running())
                    .map(Task::info)
                    .collect();
                let _ = response.send(tasks);
            }
            Command::Logs {
                task_id,
                offset,
                limit,
                response,
            } => {
                let result = Self::task(&self.tasks, &task_id).map_or_else(
                    || Err(TaskError::NotFound(task_id)),
                    |task| {
                        Ok((
                            task.log.iter().skip(offset).take(limit).cloned().collect(),
                            task.log.len(),
                        ))
                    },
                );
                let _ = response.send(result);
            }
            Command::Wait { task_id, response } => self.wait(&task_id, response),
            Command::RequestCancel { task_id, response } => {
                self.request_cancel(&task_id, response);
            }
            Command::CancelForeground { task_id } => self.cancel_foreground(&task_id),
            Command::DetachForeground { response } => {
                let detached = self.detach_foreground();
                let _ = response.send(detached);
            }
            Command::Display { task_id, display } => self.display(&task_id, display),
            Command::Log {
                task_id,
                kind,
                content,
            } => self.log(&task_id, kind, content),
            Command::Finish { task_id, result } => self.finish(&task_id, result),
            Command::Cancel { task_id } => self.cancel(&task_id),
            Command::Close { response } => {
                self.closing = true;
                self.close_waiters.push(response);
                for task in &mut self.tasks {
                    task.request_cancel(None);
                }
            }
        }
    }

    fn new_id(&mut self) -> Result<String, TaskError> {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or(TaskError::IdExhausted)?;
        Ok(id.to_string())
    }

    fn task<'a>(tasks: &'a [Task], task_id: &str) -> Option<&'a Task> {
        tasks.iter().find(|task| task.id == task_id)
    }

    fn task_mut<'a>(tasks: &'a mut [Task], task_id: &str) -> Option<&'a mut Task> {
        tasks.iter_mut().find(|task| task.id == task_id)
    }

    fn resolve(
        &mut self,
        task_id: &str,
        response: oneshot::Sender<Result<TaskResolution, TaskError>>,
    ) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            let _ = response.send(Err(TaskError::NotFound(task_id.to_string())));
            return;
        };
        if task.behavior == TaskBehavior::Detach {
            let _ = response.send(Ok(TaskResolution::Detached));
            return;
        }
        match &mut task.state {
            TaskState::Running { resolvers, .. } => {
                resolvers.push(response);
            }
            TaskState::Finished { result, .. } => {
                let _ = response.send(Ok(TaskResolution::Finished(result.clone())));
            }
            TaskState::Cancelled { .. } => {
                let _ = response.send(Err(TaskError::Cancelled(task_id.to_string())));
            }
        }
    }

    fn wait(&mut self, task_id: &str, response: oneshot::Sender<Result<ToolOutput, TaskError>>) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            let _ = response.send(Err(TaskError::NotFound(task_id.to_string())));
            return;
        };
        match &mut task.state {
            TaskState::Running { waiters, .. } => {
                waiters.push(response);
            }
            TaskState::Finished { result, .. } => {
                let _ = response.send(Ok(result.clone()));
            }
            TaskState::Cancelled { .. } => {
                let _ = response.send(Err(TaskError::Cancelled(task_id.to_string())));
            }
        }
    }

    fn request_cancel(&mut self, task_id: &str, response: oneshot::Sender<bool>) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            let _ = response.send(false);
            return;
        };
        task.request_cancel(Some(response));
    }

    fn cancel_foreground(&mut self, task_id: &str) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            return;
        };
        if task.behavior == TaskBehavior::Wait {
            task.request_cancel(None);
        }
    }

    fn detach_foreground(&mut self) -> usize {
        let mut count = 0;
        for task in &mut self.tasks {
            if task.detach() {
                let _ = self.events.send(TaskEvent::Updated(task.info()));
                count += 1;
            }
        }
        count
    }

    fn display(&mut self, task_id: &str, display: Option<String>) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            return;
        };
        if !task.is_running() {
            return;
        }
        task.display = display;
        let _ = self.events.send(TaskEvent::Updated(task.info()));
    }

    fn log(&mut self, task_id: &str, kind: String, content: String) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            return;
        };
        task.log.push(TaskLogEntry {
            kind,
            content,
            timestamp: SystemTime::now(),
        });
    }

    fn finish(&mut self, task_id: &str, result: ToolOutput) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            return;
        };
        if !task.finish(result.clone()) {
            return;
        }
        let info = task.info();
        let _ = self.events.send(TaskEvent::Updated(info.clone()));
        if info.detached {
            let _ = self.events.send(TaskEvent::Finished(info, result.clone()));
        }
    }

    fn cancel(&mut self, task_id: &str) {
        let Some(task) = Self::task_mut(&mut self.tasks, task_id) else {
            return;
        };
        if task.cancel() {
            let info = task.info();
            let _ = self.events.send(TaskEvent::Updated(info.clone()));
            if info.detached {
                let _ = self.events.send(TaskEvent::Finished(
                    info,
                    ToolOutput::error("task cancelled"),
                ));
            }
        }
    }
}

impl Task {
    fn new(
        id: String,
        tool_name: String,
        display: Option<String>,
        behavior: TaskBehavior,
        cancel: oneshot::Sender<()>,
    ) -> Self {
        Self {
            id,
            tool_name,
            display,
            started_at: SystemTime::now(),
            behavior,
            log: Vec::new(),
            state: TaskState::Running {
                cancellation: Cancellation::Available(cancel),
                waiters: Vec::new(),
                resolvers: Vec::new(),
                cancel_waiters: Vec::new(),
            },
        }
    }

    fn info(&self) -> TaskInfo {
        let (status, ended_at) = match &self.state {
            TaskState::Running { .. } => (TaskStatus::Running, None),
            TaskState::Finished { result, ended_at } => (
                if result.success {
                    TaskStatus::Succeeded
                } else {
                    TaskStatus::Failed
                },
                Some(*ended_at),
            ),
            TaskState::Cancelled { ended_at } => (TaskStatus::Cancelled, Some(*ended_at)),
        };
        TaskInfo {
            id: self.id.clone(),
            tool_name: self.tool_name.clone(),
            display: self.display.clone(),
            status,
            started_at: self.started_at,
            ended_at,
            detached: self.behavior == TaskBehavior::Detach,
        }
    }

    fn is_running(&self) -> bool {
        matches!(self.state, TaskState::Running { .. })
    }

    fn prune_closed_waiters(&mut self) {
        if let TaskState::Running {
            waiters,
            resolvers,
            cancel_waiters,
            ..
        } = &mut self.state
        {
            waiters.retain(|response| !response.is_closed());
            resolvers.retain(|response| !response.is_closed());
            cancel_waiters.retain(|response| !response.is_closed());
        }
    }

    fn request_cancel(&mut self, response: Option<oneshot::Sender<bool>>) {
        let TaskState::Running {
            cancellation,
            cancel_waiters,
            ..
        } = &mut self.state
        else {
            if let Some(response) = response {
                let _ = response.send(false);
            }
            return;
        };
        if let Some(response) = response {
            cancel_waiters.push(response);
        }
        if let Cancellation::Available(cancel) =
            std::mem::replace(cancellation, Cancellation::Requested)
        {
            let _ = cancel.send(());
        }
    }

    fn detach(&mut self) -> bool {
        let TaskState::Running { resolvers, .. } = &mut self.state else {
            return false;
        };
        if self.behavior == TaskBehavior::Detach {
            return false;
        }
        self.behavior = TaskBehavior::Detach;
        for response in resolvers.drain(..) {
            let _ = response.send(Ok(TaskResolution::Detached));
        }
        true
    }

    fn finish(&mut self, result: ToolOutput) -> bool {
        let previous = std::mem::replace(
            &mut self.state,
            TaskState::Finished {
                result: result.clone(),
                ended_at: SystemTime::now(),
            },
        );
        let TaskState::Running {
            waiters,
            resolvers,
            cancel_waiters,
            ..
        } = previous
        else {
            self.state = previous;
            return false;
        };
        for response in waiters {
            let _ = response.send(Ok(result.clone()));
        }
        for response in resolvers {
            let _ = response.send(Ok(TaskResolution::Finished(result.clone())));
        }
        for response in cancel_waiters {
            let _ = response.send(false);
        }
        true
    }

    fn cancel(&mut self) -> bool {
        let previous = std::mem::replace(
            &mut self.state,
            TaskState::Cancelled {
                ended_at: SystemTime::now(),
            },
        );
        let TaskState::Running {
            waiters,
            resolvers,
            cancel_waiters,
            ..
        } = previous
        else {
            self.state = previous;
            return false;
        };
        for response in waiters {
            let _ = response.send(Err(TaskError::Cancelled(self.id.clone())));
        }
        for response in resolvers {
            let _ = response.send(Err(TaskError::Cancelled(self.id.clone())));
        }
        for response in cancel_waiters {
            let _ = response.send(true);
        }
        true
    }
}

struct CancelOnDrop {
    task_id: Option<String>,
    commands: mpsc::UnboundedSender<Command>,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(task_id) = self.task_id.take() {
            let _ = self.commands.send(Command::CancelForeground { task_id });
        }
    }
}

async fn run_task<F>(
    task_id: String,
    future: F,
    cancelled: oneshot::Receiver<()>,
    commands: mpsc::UnboundedSender<Command>,
) where
    F: Future<Output = ToolOutput>,
{
    let outcome = AssertUnwindSafe(async move {
        tokio::pin!(future);
        tokio::select! {
            biased;
            _ = cancelled => None,
            result = &mut future => Some(result),
        }
    })
    .catch_unwind()
    .await;
    let command = match outcome {
        Ok(Some(result)) => Command::Finish { task_id, result },
        Ok(None) => Command::Cancel { task_id },
        Err(payload) => Command::Finish {
            task_id,
            result: ToolOutput::error(
                payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "task panicked".to_string()),
            ),
        },
    };
    let _ = commands.send(command);
}

#[cfg(test)]
#[path = "task_manager_test.rs"]
mod tests;
