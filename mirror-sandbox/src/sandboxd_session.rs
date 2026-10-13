//! One open connection to a sandbox's `sandboxd` and the calls in flight on it.

use crate::{
    execute_event, sandboxd_client::SandboxdClient, CallOutcome, CancelRequest, ExecuteRequest,
    SandboxdStream, MAX_MESSAGE_BYTES, STOP_GRACE,
};
use mirror_protocol::ToolCall;
use prost::Message;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;
use tonic::transport::Channel;

pub struct SandboxdSession {
    commands: mpsc::Sender<SessionCommand>,
    closing: watch::Sender<bool>,
    tasks: tokio_util::task::TaskTracker,
}
impl std::fmt::Debug for SandboxdSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxdSession")
            .field("closed", &self.is_closed())
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
pub enum CallReply {
    Outcome(CallOutcome),
    StreamEnded(String),
}
#[derive(Debug, thiserror::Error)]
pub enum SandboxdSessionError {
    #[error("sandboxd gRPC connection failed: {0}")]
    Connect(String),
    #[error("sandboxd connection is closed")]
    Closed,
    #[error("sandboxd request exceeds {MAX_MESSAGE_BYTES} bytes")]
    RequestTooLarge,
}
const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_ADMISSION_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a host waits for `sandboxd` to answer a cancel with the call's fate.
pub const CANCEL_ANSWER_TIMEOUT: Duration = STOP_GRACE.saturating_mul(4);
/// The most output one call may return, whatever the caller asks for.
pub const MAX_OUTPUT_BYTES: u64 = 8_000_000;
const MAX_PENDING_CALLS: usize = 64;

enum SessionCommand {
    Execute {
        request: ExecuteRequest,
        reply: oneshot::Sender<CallReply>,
    },
    Cancel(String),
    Client(oneshot::Sender<SandboxdClient<Channel>>),
}
struct PendingCall {
    reply: oneshot::Sender<CallReply>,
    cancelled: watch::Sender<bool>,
}
impl SandboxdSession {
    pub async fn open(
        stream: SandboxdStream,
        closing: watch::Sender<bool>,
    ) -> Result<Self, SandboxdSessionError> {
        let client = tokio::time::timeout(
            CONNECT_TIMEOUT,
            crate::connect(stream.reader, stream.writer, closing.clone()),
        )
        .await
        .map_err(|_| SandboxdSessionError::Connect("connection timed out".into()))?
        .map_err(|error| SandboxdSessionError::Connect(error.to_string()))?;
        let (commands, receiver) = mpsc::channel(MAX_PENDING_CALLS);
        let tasks = tokio_util::task::TaskTracker::new();
        drop(tasks.spawn(drive(client, receiver, closing.clone())));
        Ok(Self {
            commands,
            closing,
            tasks,
        })
    }
    pub async fn close(&self) {
        self.closing.send_replace(true);
        self.tasks.close();
        self.tasks.wait().await;
    }
    pub fn is_closed(&self) -> bool {
        *self.closing.borrow() || self.commands.is_closed()
    }

    /// A client on this session's connection, for RPCs other than tool calls.
    ///
    /// The connection stays open while any client is held, so holders must
    /// release theirs once the session starts closing.
    pub async fn client(&self) -> Result<SandboxdClient<Channel>, SandboxdSessionError> {
        let (reply, client) = oneshot::channel();
        self.commands
            .send(SessionCommand::Client(reply))
            .await
            .map_err(|_| SandboxdSessionError::Closed)?;
        client.await.map_err(|_| SandboxdSessionError::Closed)
    }

    /// Watches whether the session has started closing.
    pub fn closing(&self) -> watch::Receiver<bool> {
        self.closing.subscribe()
    }

    /// Tasks spawned here are awaited when the session closes.
    pub fn tasks(&self) -> &tokio_util::task::TaskTracker {
        &self.tasks
    }

    /// How many more commands the session can queue before senders wait.
    pub fn command_capacity(&self) -> usize {
        self.commands.capacity()
    }

    pub async fn execute(
        &self,
        call_id: String,
        call: ToolCall,
        timeout: Duration,
        max_output_bytes: u64,
    ) -> Result<oneshot::Receiver<CallReply>, SandboxdSessionError> {
        if self.is_closed() {
            return Err(SandboxdSessionError::Closed);
        }
        let request = ExecuteRequest {
            call_id,
            call: Some(call),
            timeout_millis: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            max_output_bytes: max_output_bytes.min(MAX_OUTPUT_BYTES),
        };
        if request.encoded_len() > MAX_MESSAGE_BYTES {
            return Err(SandboxdSessionError::RequestTooLarge);
        }
        let (reply, receiver) = oneshot::channel();
        self.commands
            .send(SessionCommand::Execute { request, reply })
            .await
            .map_err(|_| SandboxdSessionError::Closed)?;
        Ok(receiver)
    }
    pub async fn cancel(&self, call_id: String) {
        let _ = self.commands.send(SessionCommand::Cancel(call_id)).await;
    }
}

async fn drive(
    client: SandboxdClient<Channel>,
    mut commands: mpsc::Receiver<SessionCommand>,
    closing: watch::Sender<bool>,
) {
    let mut closed = closing.subscribe();
    let mut calls = HashMap::<String, PendingCall>::new();
    let mut tasks = JoinSet::new();
    let reason = loop {
        tokio::select! {
            biased;
            () = async { let _ = closed.wait_for(|closed| *closed).await; } => break "sandbox lifecycle closed".to_string(),
            completed = tasks.join_next(), if !tasks.is_empty() => {
                let Some(Ok((id, result))) = completed else { break "sandboxd RPC task stopped".into(); };
                if let CallReply::StreamEnded(reason) = &result { break reason.clone(); }
                if let Some(call) = calls.remove(&id) { let _ = call.reply.send(result); }
            }
            command = commands.recv() => match command {
                Some(SessionCommand::Execute { request, reply }) => {
                    if calls.contains_key(&request.call_id) || calls.len() >= MAX_PENDING_CALLS {
                        let _ = reply.send(CallReply::Outcome(CallOutcome::Rejected("sandboxd call is already in flight or the execute queue is full".into())));
                        continue;
                    }
                    let (cancelled, cancellation) = watch::channel(false);
                    calls.insert(request.call_id.clone(), PendingCall { reply, cancelled });
                    let client = client.clone();
                    tasks.spawn(async move {
                        let id = request.call_id.clone();
                        let result = execute_rpc(client, request, cancellation).await;
                        (id, result.unwrap_or_else(CallReply::StreamEnded))
                    });
                }
                Some(SessionCommand::Cancel(id)) => {
                    if let Some(call) = calls.get(&id) { call.cancelled.send_replace(true); }
                }
                Some(SessionCommand::Client(reply)) => {
                    let _ = reply.send(client.clone());
                }
                None => break "the session was dropped".into(),
            }
        }
    };
    closing.send_replace(true);
    tasks.abort_all();
    while tasks.join_next().await.is_some() {}
    drop(client);
    commands.close();
    while let Some(command) = commands.recv().await {
        if let SessionCommand::Execute { reply, .. } = command {
            let _ = reply.send(CallReply::StreamEnded(reason.clone()));
        }
    }
    for call in calls.into_values() {
        let _ = call.reply.send(CallReply::StreamEnded(reason.clone()));
    }
}

async fn execute_rpc(
    mut client: SandboxdClient<Channel>,
    request: ExecuteRequest,
    mut cancellation: watch::Receiver<bool>,
) -> Result<CallReply, String> {
    let call_id = request.call_id.clone();
    let output_limit = usize::try_from(request.max_output_bytes).unwrap_or(MAX_MESSAGE_BYTES);
    let response_limit = output_limit + 1 + prost::length_delimiter_len(output_limit);
    let event_limit = response_limit + 1 + prost::length_delimiter_len(response_limit);
    client = client.max_decoding_message_size(event_limit);
    let timeout = Duration::from_millis(request.timeout_millis);
    let deadline = tokio::time::Instant::now() + timeout;
    let admission = async {
        let mut responses = client.execute(request).await?.into_inner();
        let first = responses.message().await?;
        Ok::<_, tonic::Status>((responses, first))
    };
    let (mut responses, first) =
        tokio::time::timeout(timeout.min(MAX_ADMISSION_TIMEOUT), admission)
            .await
            .map_err(|_| "sandboxd call admission timed out".to_string())?
            .map_err(|error| format!("sandboxd execute failed: {error}"))?;
    match first.and_then(|event| event.event) {
        Some(execute_event::Event::Completed(response)) => {
            return response
                .outcome
                .map(CallReply::Outcome)
                .ok_or_else(|| "sandboxd result has no outcome".into())
        }
        Some(execute_event::Event::Accepted(true)) => {}
        _ => return Err("sandboxd did not acknowledge admission".into()),
    }
    // A cancellation cannot overtake a backpressured Execute request: admission
    // confirms that the guest has registered this call ID before Cancel is sent.
    let interrupted = async {
        tokio::select! {
            () = tokio::time::sleep_until(deadline) => {}
            _ = cancellation.wait_for(|cancelled| *cancelled) => {}
        }
    };
    let result = tokio::select! {
        result = responses.message() => result,
        () = interrupted => {
            tokio::time::timeout(CONTROL_TIMEOUT, client.cancel(CancelRequest { call_id })).await
                .map_err(|_| "sandboxd cancel timed out".to_string())?
                .map_err(|error| format!("sandboxd cancel failed: {error}"))?;
            tokio::time::timeout(CANCEL_ANSWER_TIMEOUT, responses.message()).await
                .map_err(|_| "sandboxd did not answer the cancel in time".to_string())?
        }
    }
    .map_err(|error| format!("sandboxd execution stream failed: {error}"))?;
    match result.and_then(|event| event.event) {
        Some(execute_event::Event::Completed(response)) => response
            .outcome
            .map(CallReply::Outcome)
            .ok_or_else(|| "sandboxd result has no outcome".into()),
        _ => Err("sandboxd execution ended without a result".into()),
    }
}
#[cfg(test)]
#[path = "sandboxd_session_tests.rs"]
mod tests;
