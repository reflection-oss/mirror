use std::future::Future;
use std::io;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::Stdio;
use std::time::Duration;

use mirror_sandbox::{
    run_command_request, run_command_response, run_command_result::Outcome, RunCommandRequest,
    RunCommandResponse, RunCommandStart, COMMAND_CHUNK_BYTES,
};
use nix::unistd::Pid;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, watch};

use super::process_supervision::{supervise_process, ProcessInterrupt, ProcessSupervision};
use crate::guest::Guest;

pub async fn run_command(
    request: &RunCommandStart,
    guest: Option<&Guest>,
    input: &mut tonic::Streaming<RunCommandRequest>,
    events: &mpsc::Sender<RunCommandResponse>,
    cancelled: impl Future<Output = ()>,
) -> Outcome {
    let mut command = std::process::Command::new("/bin/sh");
    command
        .args(["-c", &request.command])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if let Some(guest) = guest {
        guest.configure_command(&mut command);
    }
    if let Some(workdir) = &request.workdir {
        command.current_dir(workdir);
    }
    let mut child = match tokio::process::Command::from(command)
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return Outcome::Rejected(format!("could not start command: {error}")),
    };
    let Some(group) = child
        .id()
        .and_then(|pid| i32::try_from(pid).ok())
        .map(Pid::from_raw)
    else {
        let _ = child.kill().await;
        return Outcome::Cancelled(false);
    };
    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let execution = supervise_process(
        group,
        Duration::from_millis(request.timeout_millis),
        cancelled,
        |stopping| async move {
            let feed = feed_input(input, stdin, stopping.clone());
            let finish = async {
                let (stdout, stderr, status) = tokio::join!(
                    pump_output(
                        stdout,
                        events,
                        run_command_response::Event::Stdout,
                        stopping.clone()
                    ),
                    pump_output(
                        stderr,
                        events,
                        run_command_response::Event::Stderr,
                        stopping
                    ),
                    child.wait(),
                );
                (status, stdout.and(stderr))
            };
            tokio::pin!(finish);
            tokio::select! {
                biased;
                result = &mut finish => result,
                fed = feed => {
                    let (status, output) = finish.await;
                    (status, fed.and(output))
                }
            }
        },
    )
    .await;
    match execution {
        ProcessSupervision::Completed((Ok(status), Ok(()))) => {
            match status
                .code()
                .or_else(|| status.signal().map(|signal| -signal))
            {
                Some(code) => Outcome::ExitCode(code),
                None => Outcome::Failed("command exited without a status".into()),
            }
        }
        ProcessSupervision::Completed((Ok(_), Err(error))) => {
            Outcome::Failed(format!("command I/O failed: {error}"))
        }
        ProcessSupervision::Completed((Err(_), _)) => Outcome::Cancelled(false),
        ProcessSupervision::Interrupted {
            reason, confirmed, ..
        } => match reason {
            ProcessInterrupt::Cancel => Outcome::Cancelled(confirmed),
            ProcessInterrupt::Deadline => Outcome::DeadlineExceeded(confirmed),
        },
    }
}

async fn feed_input(
    input: &mut tonic::Streaming<RunCommandRequest>,
    stdin: Option<tokio::process::ChildStdin>,
    mut stopping: watch::Receiver<bool>,
) -> io::Result<()> {
    let Some(mut stdin) = stdin else {
        return Err(io::Error::other("command has no stdin"));
    };
    let feed = async {
        while let Some(request) = input.message().await.map_err(io::Error::other)? {
            let Some(run_command_request::Input::Stdin(bytes)) = request.input else {
                return Err(io::Error::other("expected command stdin"));
            };
            if bytes.len() > COMMAND_CHUNK_BYTES {
                return Err(io::Error::other("command stdin chunk exceeds its limit"));
            }
            match stdin.write_all(&bytes).await {
                Err(error) if error.kind() == io::ErrorKind::BrokenPipe => return Ok(()),
                result => result?,
            }
        }
        stdin.shutdown().await
    };
    tokio::select! {
        result = feed => result,
        _ = stopping.wait_for(|stopping| *stopping) => Ok(()),
    }
}

async fn pump_output(
    reader: Option<impl AsyncRead + Unpin>,
    events: &mpsc::Sender<RunCommandResponse>,
    event: fn(Vec<u8>) -> run_command_response::Event,
    mut stopping: watch::Receiver<bool>,
) -> io::Result<()> {
    let Some(mut reader) = reader else {
        return Err(io::Error::other("command has no output pipe"));
    };
    let mut buffer = vec![0; COMMAND_CHUNK_BYTES];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            return Ok(());
        }
        if *stopping.borrow() {
            continue;
        }
        let event = event(buffer[..count].to_vec());
        tokio::select! {
            result = events.send(RunCommandResponse { event: Some(event) }) => {
                result.map_err(|_| io::Error::other("command output closed"))?;
            }
            _ = stopping.wait_for(|stopping| *stopping) => {}
        }
    }
}

#[cfg(test)]
#[path = "run_command_test.rs"]
mod tests;
