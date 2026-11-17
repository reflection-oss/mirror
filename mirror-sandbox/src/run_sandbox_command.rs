//! Byte-oriented process I/O over the shared sandboxd command lifecycle.

use std::future::Future;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::watch;
use tonic::metadata::MetadataMap;
use tonic::transport::Channel;
use uuid::Uuid;

use crate::{
    run_command_response::Event, run_command_result::Outcome, sandboxd_client::SandboxdClient,
    CommandRpc, RunCommandStart, SandboxStream, SandboxStreamOutcome, SandboxTransportError,
    SandboxdStream, COMMAND_CHUNK_BYTES,
};

const ADMISSION_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const STDERR_LIMIT: usize = 64 * 1024;

/// Cancellation must be awaited: dropping this future cannot confirm guest exit.
pub async fn run_sandbox_command(
    connection: impl Future<Output = Result<SandboxdStream, SandboxTransportError>> + Send,
    request: SandboxStream<'_>,
) -> Result<SandboxStreamOutcome, SandboxTransportError> {
    let (closing, _) = watch::channel(false);
    let connection_closing = closing.clone();
    let connect = async move {
        let stream = connection.await?;
        crate::connect(stream.reader, stream.writer, connection_closing)
            .await
            .map_err(|error| SandboxTransportError::Failed(error.to_string()))
    };
    let result = run_command(connect, request, MetadataMap::new(), ADMISSION_TIMEOUT).await;
    closing.send_replace(true);
    result
}

/// Run one command over a local or hosted channel without replaying dispatch.
pub async fn run_command(
    connection: impl Future<Output = Result<SandboxdClient<Channel>, SandboxTransportError>> + Send,
    mut request: SandboxStream<'_>,
    metadata: MetadataMap,
    admission_timeout: Duration,
) -> Result<SandboxStreamOutcome, SandboxTransportError> {
    let deadline = tokio::time::Instant::now()
        .checked_add(request.timeout)
        .ok_or_else(|| SandboxTransportError::Failed("command timeout is too large".into()))?;
    let client = tokio::select! {
        result = tokio::time::timeout(request.timeout.min(CONNECT_TIMEOUT), connection) => {
            result.map_err(|_| SandboxTransportError::NotReady("sandboxd connection timed out".into()))??
        }
        () = &mut request.cancellation => return Err(SandboxTransportError::Failed("command cancelled before dispatch".into())),
    };
    let input = futures_util::stream::try_unfold(request.stdin.take(), |stdin| async move {
        let Some(mut stdin) = stdin else {
            return Ok(None);
        };
        let mut buffer = vec![0; COMMAND_CHUNK_BYTES];
        let count = stdin.read(&mut buffer).await.map_err(io_error)?;
        buffer.truncate(count);
        Ok((count != 0).then_some((buffer, Some(stdin))))
    });
    let mut captured = Vec::new();
    let output = futures_util::sink::unfold(
        (&mut request.stdout, &mut request.stderr, &mut captured),
        |(stdout, stderr, captured), event| async move {
            match event {
                Event::Stdout(bytes) => stdout.write_all(&bytes).await.map_err(io_error)?,
                Event::Stderr(bytes) => {
                    if let Some(stderr) = stderr.as_mut() {
                        stderr.write_all(&bytes).await.map_err(io_error)?;
                    } else {
                        captured.extend_from_slice(
                            &bytes[..bytes.len().min(STDERR_LIMIT - captured.len())],
                        );
                    }
                }
                _ => {}
            }
            Ok((stdout, stderr, captured))
        },
    );
    let command = CommandRpc {
        start: RunCommandStart {
            call_id: Uuid::new_v4().to_string(),
            command: request.command,
            workdir: request.workdir,
            timeout_millis: u64::try_from(
                deadline
                    .saturating_duration_since(tokio::time::Instant::now())
                    .as_millis(),
            )
            .unwrap_or(u64::MAX),
        },
        input: Box::pin(input),
        output: Box::pin(output),
        cancellation: Box::pin(&mut request.cancellation),
    };
    let completion = command.run(client, metadata, admission_timeout).await?;
    let result = outcome(completion.result.outcome, captured)?;
    if let Some(error) = completion.interruption {
        return Err(SandboxTransportError::Failed(error.to_string()));
    }
    // A failed output sink cannot make an observed exit unknown.
    let delivery = async {
        request.stdout.shutdown().await?;
        if let Some(stderr) = request.stderr.as_mut() {
            stderr.shutdown().await?;
        }
        Ok::<_, std::io::Error>(())
    };
    tokio::select! {
        result = tokio::time::timeout_at(deadline, delivery) => {
            result.map_err(|_| SandboxTransportError::Failed("command exited but output delivery timed out".into()))?
                .map_err(|error| SandboxTransportError::Failed(format!("command exited but output delivery failed: {error}")))?;
        }
        () = &mut request.cancellation => return Err(SandboxTransportError::Failed("command exited but output delivery was cancelled".into())),
    }
    Ok(result)
}

fn outcome(
    terminal: Option<Outcome>,
    stderr: Vec<u8>,
) -> Result<SandboxStreamOutcome, SandboxTransportError> {
    match terminal {
        Some(Outcome::ExitCode(exit_code)) => Ok(SandboxStreamOutcome { exit_code, stderr }),
        Some(Outcome::Rejected(detail) | Outcome::Failed(detail)) => {
            Err(SandboxTransportError::Failed(detail))
        }
        Some(Outcome::Cancelled(true)) => Err(SandboxTransportError::Failed(
            "command cancelled; its process group stopped".into(),
        )),
        Some(Outcome::DeadlineExceeded(true)) => Err(SandboxTransportError::Failed(
            "command deadline exceeded; its process group stopped".into(),
        )),
        _ => Err(SandboxTransportError::OutcomeUnknown(
            "sandboxd did not confirm command termination".into(),
        )),
    }
}

fn io_error(error: std::io::Error) -> SandboxTransportError {
    SandboxTransportError::Failed(error.to_string())
}
