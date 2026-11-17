use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::os::unix::process::CommandExt;
use std::process::ExitStatus;
use std::process::Stdio;
use std::time::Duration;

use mirror_protocol::ChatRecord;
use mirror_sandbox::{
    execute_event, run_command_response, run_command_result, CallOutcome, ExecuteEvent,
    ExecuteRequest, ExecuteResponse, RunCommandRequest, RunCommandResponse, RunCommandResult,
    RunCommandStart, CANCEL_ANSWER_TIMEOUT, MAX_DISPATCH_CALL_ID_BYTES, MAX_MESSAGE_BYTES,
};
use nix::unistd::Pid;
use prost::Message;
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;

use super::process_supervision::{supervise_process, ProcessInterrupt, ProcessSupervision};
use super::run_command::run_command;
use crate::guest::{Guest, GuestState};
use crate::tool_runner::{RunnerCommand, REQUEST_BYTES_VARIABLE};

pub(crate) enum ExecutionCommand {
    Execute(ExecutionRequest),
    Cancel(String),
}

pub(crate) enum ExecutionRequest {
    Tool {
        request: ExecuteRequest,
        events: mpsc::Sender<ExecuteEvent>,
    },
    Command {
        request: RunCommandStart,
        input: Box<tonic::Streaming<RunCommandRequest>>,
        events: mpsc::Sender<RunCommandResponse>,
    },
}

const MAX_ACTIVE_CALLS: usize = 64;
const CALL_STOP_GRACE: Duration = Duration::from_secs(16);
const DIAGNOSTIC_BYTES: u64 = 8 * 1024;

struct Execution {
    stdout: io::Result<(Vec<u8>, bool)>,
    stderr: io::Result<(Vec<u8>, bool)>,
    status: io::Result<ExitStatus>,
}

pub(crate) async fn run(
    mut commands: mpsc::Receiver<ExecutionCommand>,
    runner: RunnerCommand,
    mut closing: watch::Receiver<bool>,
    readiness: watch::Receiver<GuestState>,
) {
    let mut calls = HashMap::<String, Option<oneshot::Sender<()>>>::new();
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            biased;
            _ = closing.wait_for(|closed| *closed) => break,
            completed = tasks.join_next(), if !tasks.is_empty() => {
                let Some(Ok(id)) = completed else { break; };
                calls.remove(&id);
            }
            command = commands.recv() => match command {
                Some(ExecutionCommand::Cancel(id)) => {
                    if let Some(interrupt) = calls.get_mut(&id).and_then(Option::take) {
                        let _ = interrupt.send(());
                    }
                }
                Some(ExecutionCommand::Execute(request)) => {
                    let (id, valid) = match &request {
                        ExecutionRequest::Tool { request, .. } => (&request.call_id, request.call.is_some()),
                        ExecutionRequest::Command { request, .. } => (&request.call_id, true),
                    };
                    let admitted = readiness.borrow().admit();
                    let rejection = if id.len() > MAX_DISPATCH_CALL_ID_BYTES {
                        Some("sandboxd call ID exceeds the dispatch limit".to_string())
                    } else if calls.len() >= MAX_ACTIVE_CALLS {
                        Some("sandboxd concurrent call limit reached".to_string())
                    } else if calls.contains_key(id) {
                        Some("sandboxd call is already running".to_string())
                    } else if !valid {
                        Some("execute has no tool call".to_string())
                    } else { admitted.as_ref().err().cloned() };
                    let delivered = match &request {
                        ExecutionRequest::Tool { events, .. } => events.try_send(ExecuteEvent {
                            event: Some(match &rejection {
                                Some(reason) => execute_event::Event::Completed(ExecuteResponse {
                                    outcome: Some(CallOutcome::Rejected(reason.clone())),
                                }),
                                None => execute_event::Event::Accepted(true),
                            }),
                        }).is_ok(),
                        ExecutionRequest::Command { events, .. } => events.try_send(RunCommandResponse {
                            event: Some(match &rejection {
                                Some(reason) => run_command_response::Event::Completed(RunCommandResult {
                                    outcome: Some(run_command_result::Outcome::Rejected(reason.clone())),
                                }),
                                None => run_command_response::Event::Accepted(true),
                            }),
                        }).is_ok(),
                    };
                    if rejection.is_some() || !delivered { continue; }
                    let id = id.clone();
                    let (interrupt, interrupted) = oneshot::channel();
                    calls.insert(id.clone(), Some(interrupt));
                    let runner = runner.clone();
                    let guest = admitted.ok().flatten();
                    tasks.spawn(async move {
                        match request {
                            ExecutionRequest::Tool { request, events } => {
                                let cancelled = async {
                                    tokio::select! {
                                        _ = interrupted => {},
                                        () = events.closed() => {},
                                    }
                                };
                                let outcome = run_call(
                                    &runner,
                                    guest.as_deref(),
                                    request.call.map(|call| call.encode_to_vec()).unwrap_or_default(),
                                    Duration::from_millis(request.timeout_millis),
                                    request.max_output_bytes,
                                    cancelled,
                                ).await;
                                let mut event = ExecuteEvent {
                                    event: Some(execute_event::Event::Completed(ExecuteResponse {
                                        outcome: Some(outcome),
                                    })),
                                };
                                if event.encoded_len() > MAX_MESSAGE_BYTES {
                                    event.event = Some(execute_event::Event::Completed(ExecuteResponse {
                                        outcome: Some(CallOutcome::Failed(
                                            "the tool result exceeded the RPC message limit and was discarded".into(),
                                        )),
                                    }));
                                }
                                let _ = events.try_send(event);
                            }
                            ExecutionRequest::Command { request, mut input, events } => {
                                let cancelled = async {
                                    tokio::select! {
                                        _ = interrupted => {},
                                        () = events.closed() => {},
                                    }
                                };
                                let outcome = run_command(&request, guest.as_deref(), &mut input, &events, cancelled).await;
                                let _ = tokio::time::timeout(CANCEL_ANSWER_TIMEOUT, events.send(RunCommandResponse {
                                    event: Some(run_command_response::Event::Completed(RunCommandResult {
                                        outcome: Some(outcome),
                                    })),
                                })).await;
                            }
                        }
                        id
                    });
                }
                None => break,
            }
        }
    }
    for interrupt in calls.into_values().flatten() {
        let _ = interrupt.send(());
    }
    if tokio::time::timeout(CALL_STOP_GRACE, async {
        while tasks.join_next().await.is_some() {}
    })
    .await
    .is_err()
    {
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
    }
}

async fn run_call(
    runner: &RunnerCommand,
    guest: Option<&Guest>,
    request: Vec<u8>,
    timeout: Duration,
    max_output_bytes: u64,
    cancelled: impl Future<Output = ()>,
) -> CallOutcome {
    let mut command = std::process::Command::new(&runner.program);
    command
        .args(&runner.args)
        .env(REQUEST_BYTES_VARIABLE, request.len().to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.process_group(0);
    if let Some(guest) = guest {
        guest.configure_command(&mut command);
    }
    let mut command = Command::from(command);
    command.kill_on_drop(true);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("could not start the tool runner: {error}");
            return CallOutcome::Rejected(message);
        }
    };
    let Some(group) = child
        .id()
        .and_then(|pid| i32::try_from(pid).ok())
        .map(Pid::from_raw)
    else {
        let cleanup = child.kill().await;
        let message = match cleanup {
            Ok(()) => "the spawned tool runner did not have a valid process id".to_string(),
            Err(error) => format!(
                "the spawned tool runner did not have a valid process id and could not be cleaned up: {error}"
            ),
        };
        return CallOutcome::Failed(message);
    };
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let output_limit = max_output_bytes.min(MAX_MESSAGE_BYTES as u64);
    let execution = async move {
        let feed = async {
            if let Some(mut stdin) = stdin {
                let _ = stdin.write_all(&request).await;
                let _ = stdin.shutdown().await;
            }
        };
        let (stdout, stderr, ()) = tokio::join!(
            read_bounded(stdout, output_limit),
            read_bounded(stderr, DIAGNOSTIC_BYTES),
            feed
        );
        let status = child.wait().await;
        Execution {
            stdout,
            stderr,
            status,
        }
    };
    match supervise_process(group, timeout, cancelled, |_| execution).await {
        ProcessSupervision::Completed(result) => finish(result),
        ProcessSupervision::Interrupted {
            reason,
            finished,
            confirmed,
        } => {
            if let Some(finished) = finished {
                let outcome = finish(finished);
                if matches!(outcome, CallOutcome::Completed(_)) {
                    return outcome;
                }
            }
            match reason {
                ProcessInterrupt::Cancel => CallOutcome::Cancelled(confirmed),
                ProcessInterrupt::Deadline => CallOutcome::DeadlineExceeded(confirmed),
            }
        }
    }
}

fn finish(execution: Execution) -> CallOutcome {
    let failed = CallOutcome::Failed;
    let status = match execution.status {
        Ok(status) => status,
        Err(error) => return failed(format!("the tool runner could not be awaited: {error}")),
    };
    let (stderr, _) = execution.stderr.unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr).trim().to_string();
    if !status.success() {
        return failed(format!("the tool runner exited with {status}: {stderr}"));
    }
    let (record, truncated) = match execution.stdout {
        Ok(stdout) => stdout,
        Err(error) => {
            return failed(format!(
                "the tool runner's output could not be read: {error}"
            ))
        }
    };
    if truncated {
        return failed(format!(
            "the tool result exceeded the {} byte limit and was discarded",
            record.len()
        ));
    }
    match ChatRecord::decode(record.as_slice()) {
        Ok(record) => CallOutcome::Completed(Box::new(record)),
        Err(error) => failed(format!(
            "the tool runner returned an invalid record: {error}"
        )),
    }
}

async fn read_bounded<R: AsyncRead + Unpin>(
    reader: Option<R>,
    limit: u64,
) -> io::Result<(Vec<u8>, bool)> {
    let Some(mut reader) = reader else {
        return Ok((Vec::new(), false));
    };
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut kept = Vec::new();
    let mut truncated = false;
    let mut buffer = vec![0u8; 16 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok((kept, truncated));
        }
        let room = limit.saturating_sub(kept.len());
        if read > room {
            truncated = true;
        }
        kept.extend_from_slice(&buffer[..read.min(room)]);
    }
}
