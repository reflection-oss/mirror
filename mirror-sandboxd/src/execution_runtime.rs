use std::collections::HashMap;
use std::future::Future;
use std::io;
use std::os::unix::process::CommandExt;
use std::pin::Pin;
use std::process::ExitStatus;
use std::process::Stdio;
use std::time::Duration;

use mirror_protocol::ChatRecord;
use nix::errno::Errno;
use nix::sys::signal::killpg;
use nix::sys::signal::Signal;
use nix::sys::wait::waitpid;
use nix::sys::wait::WaitPidFlag;
use nix::sys::wait::WaitStatus;
use nix::unistd::Pid;
use prost::Message;
use tokio::io::AsyncRead;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tokio::time::Instant;

use crate::{RunnerCommand, REQUEST_BYTES_VARIABLE};
use mirror_sandbox::{
    execute_event, CallOutcome, ExecuteEvent, ExecuteRequest, ExecuteResponse,
    MAX_DISPATCH_CALL_ID_BYTES, MAX_MESSAGE_BYTES, STOP_GRACE,
};
use tokio::sync::watch;

const MAX_ACTIVE_CALLS: usize = 64;
const CALL_STOP_GRACE: Duration = Duration::from_secs(16);
const DIAGNOSTIC_BYTES: u64 = 8 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(20);

type Events = mpsc::Sender<ExecuteEvent>;
pub(crate) enum ExecutionCommand {
    Execute {
        request: ExecuteRequest,
        events: Events,
    },
    Cancel(String),
}
#[derive(Clone, Copy)]
enum Interrupt {
    Cancel,
    Deadline,
}
struct Execution {
    stdout: io::Result<(Vec<u8>, bool)>,
    stderr: io::Result<(Vec<u8>, bool)>,
    status: io::Result<ExitStatus>,
}
struct Stopped {
    finished: Option<Execution>,
    confirmed: bool,
}

pub(crate) async fn run(
    mut commands: mpsc::Receiver<ExecutionCommand>,
    runner: RunnerCommand,
    mut closing: watch::Receiver<bool>,
    egress: crate::egress_runtime::GuestEgress,
) {
    let mut calls = HashMap::<String, Option<oneshot::Sender<Interrupt>>>::new();
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            biased;
            _ = closing.wait_for(|closed| *closed) => break,
            completed = tasks.join_next(), if !tasks.is_empty() => {
                let Some(Ok((id, events, outcome))) = completed else { break; };
                calls.remove(&id);
                complete(&events, outcome);
            }
            command = commands.recv() => match command {
                Some(ExecutionCommand::Cancel(id)) => {
                    if let Some(interrupt) = calls.get_mut(&id).and_then(Option::take) {
                        let _ = interrupt.send(Interrupt::Cancel);
                    }
                }
                Some(ExecutionCommand::Execute { request, events }) => {
                    let rejection = if request.call_id.len() > MAX_DISPATCH_CALL_ID_BYTES {
                        Some("sandboxd call ID exceeds the dispatch limit")
                    } else if calls.len() >= MAX_ACTIVE_CALLS {
                        Some("sandboxd concurrent call limit reached")
                    } else if calls.contains_key(&request.call_id) {
                        Some("sandboxd call is already running")
                    } else if request.call.is_none() {
                        Some("execute has no tool call")
                    } else { None };
                    if let Some(reason) = rejection {
                        complete(&events, CallOutcome::Rejected(reason.into()));
                        continue;
                    }
                    if events.try_send(ExecuteEvent { event: Some(execute_event::Event::Accepted(true)) }).is_err() { continue; }
                    let (interrupt, interrupted) = oneshot::channel();
                    calls.insert(request.call_id.clone(), Some(interrupt));
                    let runner = runner.clone();
                    let egress = egress.get().cloned();
                    tasks.spawn(async move {
                        let (abandoned, abandoned_rx) = oneshot::channel();
                        let relay = async {
                            let reason = tokio::select! {
                                reason = interrupted => reason.unwrap_or(Interrupt::Cancel),
                                () = events.closed() => Interrupt::Cancel,
                            };
                            let _ = abandoned.send(reason);
                        };
                        let execution = run_call(&runner, egress.as_deref(),
                            request.call.map(|call| call.encode_to_vec()).unwrap_or_default(),
                            Duration::from_millis(request.timeout_millis), request.max_output_bytes, abandoned_rx);
                        tokio::pin!(execution);
                        let outcome = tokio::select! {
                            outcome = &mut execution => outcome,
                            () = relay => execution.await,
                        };
                        (request.call_id, events, outcome)
                    });
                }
                None => break,
            }
        }
    }
    for interrupt in calls.into_values().flatten() {
        let _ = interrupt.send(Interrupt::Cancel);
    }
    if tokio::time::timeout(CALL_STOP_GRACE, async {
        while let Some(Ok((_, events, outcome))) = tasks.join_next().await {
            complete(&events, outcome);
        }
    })
    .await
    .is_err()
    {
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
    }
}

fn complete(events: &Events, outcome: CallOutcome) {
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

async fn run_call(
    runner: &RunnerCommand,
    egress: Option<&crate::egress_connector::EgressConnector>,
    request: Vec<u8>,
    timeout: Duration,
    max_output_bytes: u64,
    interrupted: oneshot::Receiver<Interrupt>,
) -> CallOutcome {
    let mut command = std::process::Command::new(&runner.program);
    command
        .args(&runner.args)
        .env(REQUEST_BYTES_VARIABLE, request.len().to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command.process_group(0);
    if let Some(egress) = egress {
        egress.configure_command(&mut command);
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
    tokio::pin!(execution);
    let interrupt = tokio::select! {
        result = &mut execution => return finish(result),
        interrupt = interrupted => match interrupt {
            Ok(interrupt) => interrupt,
            Err(_) => return finish(execution.await),
        },
        () = tokio::time::sleep(timeout) => Interrupt::Deadline,
    };
    let stopped = stop_group(group, execution).await;
    if let Some(finished) = stopped.finished {
        let outcome = finish(finished);
        if matches!(outcome, CallOutcome::Completed(_)) {
            return outcome;
        }
    }
    let confirmed = stopped.confirmed;
    match interrupt {
        Interrupt::Cancel => CallOutcome::Cancelled(confirmed),
        Interrupt::Deadline => CallOutcome::DeadlineExceeded(confirmed),
    }
}

async fn stop_group(
    group: Pid,
    mut execution: Pin<&mut impl Future<Output = Execution>>,
) -> Stopped {
    let _ = killpg(group, Signal::SIGTERM);
    let mut finished = tokio::time::timeout(STOP_GRACE, &mut execution).await.ok();
    if finished.is_none() {
        let _ = killpg(group, Signal::SIGKILL);
        finished = tokio::time::timeout(STOP_GRACE, &mut execution).await.ok();
    }
    let Some(finished) = finished else {
        return Stopped {
            finished: None,
            confirmed: false,
        };
    };
    let deadline = Instant::now() + STOP_GRACE;
    loop {
        reap_group(group);
        match killpg(group, None::<Signal>) {
            Err(Errno::ESRCH) => {
                return Stopped {
                    finished: Some(finished),
                    confirmed: true,
                };
            }
            _ if Instant::now() >= deadline => {
                return Stopped {
                    finished: Some(finished),
                    confirmed: false,
                };
            }
            _ => {
                let _ = killpg(group, Signal::SIGKILL);
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

fn reap_group(group: Pid) {
    while matches!(
        waitpid(Pid::from_raw(-group.as_raw()), Some(WaitPidFlag::WNOHANG)),
        Ok(WaitStatus::Exited(..) | WaitStatus::Signaled(..))
    ) {}
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
