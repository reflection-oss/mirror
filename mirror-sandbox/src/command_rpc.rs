//! Shared sandboxd admission, streaming, and confirmed termination without replay.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use futures_util::{Sink, SinkExt, Stream, StreamExt};
use prost::Message;
use tokio::sync::mpsc;
use tonic::metadata::MetadataMap;
use tonic::transport::Channel;
use tonic::Request;

use crate::{
    run_command_request::Input, run_command_response::Event, run_command_result::Outcome,
    sandboxd_client::SandboxdClient, CancelRequest, RunCommandRequest, RunCommandResponse,
    RunCommandResult, RunCommandStart, SandboxTransportError, CANCEL_ANSWER_TIMEOUT,
    COMMAND_CHUNK_BYTES, MAX_MESSAGE_BYTES,
};

/// Protocol-level I/O; adapters own byte sinks, RPC forwarding, and containment.
pub struct CommandRpc<'a> {
    pub start: RunCommandStart,
    pub input: Pin<Box<dyn Stream<Item = Result<Vec<u8>, SandboxTransportError>> + Send + 'a>>,
    pub output: Pin<Box<dyn Sink<Event, Error = SandboxTransportError> + Send + 'a>>,
    pub cancellation: Pin<Box<dyn Future<Output = ()> + Send + 'a>>,
}

/// Terminal evidence is retained even when local I/O or cancellation interrupted delivery.
#[derive(Debug)]
pub struct CommandCompletion {
    pub result: RunCommandResult,
    pub interruption: Option<SandboxTransportError>,
}

const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);

impl CommandRpc<'_> {
    /// Await cancellation to completion; dropping this future cannot confirm guest exit.
    pub async fn run(
        mut self,
        mut client: SandboxdClient<Channel>,
        metadata: MetadataMap,
        admission_timeout: Duration,
    ) -> Result<CommandCompletion, SandboxTransportError> {
        let deadline = tokio::time::Instant::now()
            .checked_add(Duration::from_millis(self.start.timeout_millis))
            .ok_or_else(|| SandboxTransportError::Failed("command timeout is too large".into()))?;
        let call_id = self.start.call_id.clone();
        let start = RunCommandRequest {
            input: Some(Input::Start(self.start)),
        };
        if start.encoded_len() > MAX_MESSAGE_BYTES {
            return Err(SandboxTransportError::Failed(
                "command exceeds the RPC message limit".into(),
            ));
        }
        let (input, receiver) = mpsc::channel(4);
        input.send(start).await.map_err(unknown)?;
        let admission = admit(&mut client, receiver, metadata.clone());
        // Cancellation cannot overtake admission and leave an unregistered call ID.
        let (mut events, first) = tokio::time::timeout_at(
            tokio::time::Instant::now()
                .checked_add(admission_timeout)
                .unwrap_or(deadline)
                .min(deadline),
            admission,
        )
        .await
        .map_err(unknown)??;
        match first {
            Some(Event::Accepted(true)) => {}
            Some(Event::Completed(result)) => return terminal(result, None),
            _ => return Err(unknown("sandboxd did not acknowledge command admission")),
        }
        let running = async {
            self.output.send(Event::Accepted(true)).await?;
            let feed = feed_input(input, self.input);
            let output = receive_output(&mut events, &mut self.output);
            tokio::pin!(output);
            tokio::select! {
                biased;
                result = &mut output => result,
                result = feed => { result?; output.await }
            }
        };
        let completion = tokio::select! {
            biased;
            result = running => result,
            () = &mut self.cancellation => Err(SandboxTransportError::Failed("command cancelled".into())),
            () = tokio::time::sleep_until(deadline) => Err(SandboxTransportError::Failed("command deadline exceeded".into())),
        };
        match completion {
            Ok(result) => terminal(result, None),
            Err(error) => {
                let result = cancel_and_wait(&mut client, &mut events, call_id, metadata).await?;
                terminal(result, Some(error))
            }
        }
    }
}

async fn admit(
    client: &mut SandboxdClient<Channel>,
    receiver: mpsc::Receiver<RunCommandRequest>,
    metadata: MetadataMap,
) -> Result<(tonic::Streaming<RunCommandResponse>, Option<Event>), SandboxTransportError> {
    let requests = futures_util::stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|request| (request, receiver))
    });
    let mut requests = Request::new(requests);
    *requests.metadata_mut() = metadata;
    let mut events = client
        .run_command(requests)
        .await
        .map_err(admission_error)?
        .into_inner();
    let first = events.message().await.map_err(unknown)?;
    Ok((events, first.and_then(|event| event.event)))
}

async fn feed_input(
    input: mpsc::Sender<RunCommandRequest>,
    mut chunks: impl Stream<Item = Result<Vec<u8>, SandboxTransportError>> + Unpin,
) -> Result<(), SandboxTransportError> {
    while let Some(bytes) = chunks.next().await {
        let bytes = bytes?;
        if bytes.len() > COMMAND_CHUNK_BYTES {
            return Err(SandboxTransportError::Failed(
                "stdin chunk exceeds the limit".into(),
            ));
        }
        if input
            .send(RunCommandRequest {
                input: Some(Input::Stdin(bytes)),
            })
            .await
            .is_err()
        {
            break;
        }
    }
    Ok(())
}

async fn receive_output(
    events: &mut tonic::Streaming<RunCommandResponse>,
    output: &mut (impl Sink<Event, Error = SandboxTransportError> + Unpin),
) -> Result<RunCommandResult, SandboxTransportError> {
    while let Some(event) = events.message().await.map_err(unknown)? {
        match event.event {
            Some(Event::Completed(result)) => return Ok(result),
            Some(event) => match &event {
                Event::Stdout(bytes) | Event::Stderr(bytes)
                    if bytes.len() <= COMMAND_CHUNK_BYTES =>
                {
                    output.send(event).await?;
                }
                _ => return Err(unknown("unexpected or oversized sandboxd command event")),
            },
            None => return Err(unknown("empty sandboxd command event")),
        }
    }
    Err(unknown("sandboxd command ended without a terminal result"))
}

async fn cancel_and_wait(
    client: &mut SandboxdClient<Channel>,
    events: &mut tonic::Streaming<RunCommandResponse>,
    call_id: String,
    metadata: MetadataMap,
) -> Result<RunCommandResult, SandboxTransportError> {
    let mut request = Request::new(CancelRequest { call_id });
    *request.metadata_mut() = metadata;
    tokio::time::timeout(CANCEL_ANSWER_TIMEOUT, async {
        // A terminal response can race even a failed Cancel RPC.
        let _ = tokio::time::timeout(CONTROL_TIMEOUT, client.cancel(request)).await;
        while let Some(event) = events.message().await.map_err(unknown)? {
            if let Some(Event::Completed(result)) = event.event {
                return Ok(result);
            }
        }
        Err(unknown(
            "sandboxd ended before command termination was confirmed",
        ))
    })
    .await
    .unwrap_or_else(|_| Err(unknown("sandboxd did not confirm command termination")))
}

fn terminal(
    result: RunCommandResult,
    interruption: Option<SandboxTransportError>,
) -> Result<CommandCompletion, SandboxTransportError> {
    match &result.outcome {
        Some(
            Outcome::ExitCode(_)
            | Outcome::Rejected(_)
            | Outcome::Failed(_)
            | Outcome::Cancelled(true)
            | Outcome::DeadlineExceeded(true),
        ) => Ok(CommandCompletion {
            result,
            interruption,
        }),
        _ => Err(unknown("sandboxd did not confirm command termination")),
    }
}

fn admission_error(error: tonic::Status) -> SandboxTransportError {
    match error.code() {
        tonic::Code::Unimplemented
        | tonic::Code::InvalidArgument
        | tonic::Code::NotFound
        | tonic::Code::PermissionDenied
        | tonic::Code::Unauthenticated
        | tonic::Code::FailedPrecondition
        | tonic::Code::ResourceExhausted => SandboxTransportError::Failed(error.to_string()),
        _ => unknown(error),
    }
}

fn unknown(error: impl std::fmt::Display) -> SandboxTransportError {
    SandboxTransportError::OutcomeUnknown(error.to_string())
}
