use super::*;
use crate::{sandboxd_server::Sandboxd, CancelResponse, ExecuteEvent, ExecuteResponse};
use assert_matches::assert_matches;
use mirror_protocol::{chat_record, ChatRecord, ToolMessage};
use pretty_assertions::assert_eq;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

type Events = mpsc::Sender<Result<ExecuteEvent, Status>>;
#[derive(Clone)]
struct Probe {
    requests: mpsc::Sender<(ExecuteRequest, Events)>,
    cancellations: mpsc::Sender<String>,
}
#[tonic::async_trait]
impl Sandboxd for Probe {
    type EgressStream =
        tokio_stream::wrappers::ReceiverStream<Result<crate::EgressEvent, tonic::Status>>;
    async fn egress(
        &self,
        _: tonic::Request<tonic::Streaming<crate::EgressUpdate>>,
    ) -> Result<tonic::Response<Self::EgressStream>, tonic::Status> {
        Err(tonic::Status::unimplemented("execution-only fixture"))
    }

    type ExecuteStream = ReceiverStream<Result<ExecuteEvent, Status>>;
    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        let (events, receiver) = mpsc::channel(2);
        self.requests
            .send((request.into_inner(), events))
            .await
            .map_err(|_| Status::unavailable("probe closed"))?;
        Ok(Response::new(ReceiverStream::new(receiver)))
    }
    async fn cancel(
        &self,
        request: Request<CancelRequest>,
    ) -> Result<Response<CancelResponse>, Status> {
        self.cancellations
            .send(request.into_inner().call_id)
            .await
            .map_err(|_| Status::unavailable("probe closed"))?;
        Ok(Response::new(CancelResponse {}))
    }
}
struct Fixture {
    session: SandboxdSession,
    requests: mpsc::Receiver<(ExecuteRequest, Events)>,
    cancellations: mpsc::Receiver<String>,
    disconnect: watch::Sender<bool>,
}
async fn open() -> Fixture {
    let (host, guest) = tokio::io::duplex(128);
    let (input, output) = tokio::io::split(guest);
    let (requests, received) = mpsc::channel(128);
    let (cancellations, cancelled) = mpsc::channel(128);
    let (closing, _) = watch::channel(false);
    drop(tokio::spawn(crate::serve_connection(
        input,
        output,
        Probe {
            requests,
            cancellations,
        },
        closing.clone(),
    )));
    let session = SandboxdSession::open(SandboxdStream::split(host), watch::channel(false).0)
        .await
        .expect("gRPC session");
    Fixture {
        session,
        requests: received,
        cancellations: cancelled,
        disconnect: closing,
    }
}
async fn call(session: &SandboxdSession, id: &str) -> oneshot::Receiver<CallReply> {
    session
        .execute(id.into(), ToolCall::default(), Duration::from_secs(60), 100)
        .await
        .expect("queue execute")
}
async fn accepted(events: &Events) {
    events
        .send(Ok(ExecuteEvent {
            event: Some(execute_event::Event::Accepted(true)),
        }))
        .await
        .expect("admission");
}
async fn complete(events: &Events, outcome: CallOutcome) {
    events
        .send(Ok(ExecuteEvent {
            event: Some(execute_event::Event::Completed(ExecuteResponse {
                outcome: Some(outcome),
            })),
        }))
        .await
        .expect("completion");
}
#[tokio::test]
async fn result_at_output_limit_is_accepted_with_its_rpc_envelopes() {
    let mut fixture = open().await;
    let record = ChatRecord {
        message: Some(chat_record::Message::Tool(ToolMessage {
            content: "x".repeat(128),
            ..ToolMessage::default()
        })),
        display: Some("display".into()),
        success: true,
        ..ChatRecord::default()
    };
    let pending = fixture
        .session
        .execute(
            "boundary".into(),
            ToolCall::default(),
            Duration::from_secs(60),
            record.encoded_len() as u64,
        )
        .await
        .expect("execute");
    let (request, events) = fixture.requests.recv().await.expect("request");
    assert_eq!(request.max_output_bytes, record.encoded_len() as u64);
    accepted(&events).await;
    complete(&events, CallOutcome::Completed(Box::new(record.clone()))).await;
    assert_matches!(pending.await, Ok(CallReply::Outcome(CallOutcome::Completed(actual))) if *actual == record);
    assert!(!fixture.session.is_closed());
    fixture.session.close().await;
}

#[tokio::test]
async fn oversized_immediate_result_closes_the_stream() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "content").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    complete(
        &events,
        CallOutcome::Completed(Box::new(ChatRecord {
            message: Some(chat_record::Message::Tool(ToolMessage {
                content: "é".repeat(60),
                ..ToolMessage::default()
            })),
            ..ChatRecord::default()
        })),
    )
    .await;
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(reason)) if reason.contains("message length too large"));
    assert!(fixture.session.is_closed());
}

#[tokio::test]
async fn output_limit_counts_display_as_well_as_content() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "display").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    accepted(&events).await;
    complete(
        &events,
        CallOutcome::Completed(Box::new(ChatRecord {
            message: Some(chat_record::Message::Tool(ToolMessage {
                content: "ok".into(),
                ..ToolMessage::default()
            })),
            display: Some("x".repeat(100)),
            ..ChatRecord::default()
        })),
    )
    .await;
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(reason)) if reason.contains("message length too large"));
    assert!(fixture.session.is_closed());
}

#[tokio::test]
async fn oversized_guest_failure_cannot_bypass_the_output_limit() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "failed").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    accepted(&events).await;
    complete(&events, CallOutcome::Failed("x".repeat(101))).await;
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(reason)) if reason.contains("message length too large"));
    assert!(fixture.session.is_closed());
}

#[tokio::test]
async fn oversized_guest_rejection_cannot_bypass_the_output_limit() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "rejected").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    complete(&events, CallOutcome::Rejected("x".repeat(101))).await;
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(reason)) if reason.contains("message length too large"));
    assert!(fixture.session.is_closed());
}

#[tokio::test]
async fn caller_cannot_raise_the_host_output_limit() {
    let mut fixture = open().await;
    let pending = fixture
        .session
        .execute(
            "limit".into(),
            ToolCall::default(),
            Duration::from_secs(60),
            u64::MAX,
        )
        .await
        .expect("execute");
    let (request, events) = fixture.requests.recv().await.expect("request");
    assert_eq!(request.max_output_bytes, 8_000_000);
    complete(&events, CallOutcome::Completed(Box::default())).await;
    assert_matches!(
        pending.await,
        Ok(CallReply::Outcome(CallOutcome::Completed(_)))
    );
    fixture.session.close().await;
}

#[tokio::test]
async fn calls_complete_independently_in_any_order() {
    let mut fixture = open().await;
    let first = call(&fixture.session, "a").await;
    let (request, first_events) = fixture.requests.recv().await.expect("first request");
    assert_eq!(request.call_id, "a");
    accepted(&first_events).await;
    let second = call(&fixture.session, "b").await;
    let (_, second_events) = fixture.requests.recv().await.expect("second request");
    accepted(&second_events).await;
    complete(&second_events, CallOutcome::Completed(Box::default())).await;
    assert_matches!(
        second.await,
        Ok(CallReply::Outcome(CallOutcome::Completed(_)))
    );
    complete(&first_events, CallOutcome::Completed(Box::default())).await;
    assert_matches!(
        first.await,
        Ok(CallReply::Outcome(CallOutcome::Completed(_)))
    );
    fixture.session.close().await;
}
#[tokio::test]
async fn cancel_waits_for_admission_and_reports_the_guests_confirmation() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "cancel").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    fixture.session.cancel("cancel".into()).await;
    tokio::task::yield_now().await;
    assert!(
        fixture.cancellations.try_recv().is_err(),
        "Cancel cannot precede admission"
    );
    accepted(&events).await;
    assert_eq!(
        fixture.cancellations.recv().await.as_deref(),
        Some("cancel")
    );
    complete(&events, CallOutcome::Cancelled(true)).await;
    assert_matches!(
        pending.await,
        Ok(CallReply::Outcome(CallOutcome::Cancelled(true)))
    );
    fixture.session.close().await;
}
#[tokio::test(start_paused = true)]
async fn admission_backpressure_outlives_the_control_timeout() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "slow-admission").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    tokio::time::sleep(CONTROL_TIMEOUT + Duration::from_secs(1)).await;
    assert!(!fixture.session.is_closed());
    accepted(&events).await;
    complete(&events, CallOutcome::Completed(Box::default())).await;
    assert_matches!(
        pending.await,
        Ok(CallReply::Outcome(CallOutcome::Completed(_)))
    );
    fixture.session.close().await;
}
#[tokio::test(start_paused = true)]
async fn admission_timeout_closes_uncertain_execution() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "stalled").await;
    let (_, _events) = fixture.requests.recv().await.expect("request");
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(reason)) if reason.contains("admission timed out"));
    assert!(fixture.session.is_closed());
}
#[tokio::test(start_paused = true)]
async fn admission_respects_a_shorter_call_deadline() {
    let mut fixture = open().await;
    let started = tokio::time::Instant::now();
    let pending = fixture
        .session
        .execute(
            "short".into(),
            ToolCall::default(),
            Duration::from_secs(5),
            100,
        )
        .await
        .expect("execute");
    let (_, _events) = fixture.requests.recv().await.expect("request");
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(_)));
    assert!(started.elapsed() < MAX_ADMISSION_TIMEOUT);
}
#[tokio::test(start_paused = true)]
async fn admitted_call_deadline_cancels_and_closes_all_uncertain_streams() {
    let mut fixture = open().await;
    let started = tokio::time::Instant::now();
    let first = call(&fixture.session, "hung").await;
    let (_, first_events) = fixture.requests.recv().await.expect("first request");
    tokio::time::sleep(Duration::from_secs(10)).await;
    accepted(&first_events).await;
    let second = fixture
        .session
        .execute(
            "other".into(),
            ToolCall::default(),
            Duration::from_secs(120),
            100,
        )
        .await
        .expect("second call");
    let (_, second_events) = fixture.requests.recv().await.expect("second request");
    accepted(&second_events).await;

    let cancelled = tokio::time::timeout(Duration::from_secs(51), fixture.cancellations.recv())
        .await
        .expect("the host enforces the deadline including admission time");
    assert_eq!(cancelled.as_deref(), Some("hung"));
    assert!(started.elapsed() >= Duration::from_secs(60));
    assert_matches!(
        tokio::time::timeout(Duration::from_secs(30), first).await,
        Ok(Ok(CallReply::StreamEnded(reason))) if reason.contains("cancel")
    );
    assert_matches!(second.await, Ok(CallReply::StreamEnded(_)));
    assert!(fixture.session.is_closed());
    tokio::time::timeout(Duration::from_secs(1), first_events.closed())
        .await
        .expect("timed out RPC is closed");
    tokio::time::timeout(Duration::from_secs(1), second_events.closed())
        .await
        .expect("other uncertain RPC is closed");
    assert!(
        fixture.requests.try_recv().is_err(),
        "calls are not replayed"
    );
}
#[tokio::test(start_paused = true)]
async fn admitted_call_deadline_allows_the_guest_to_confirm_cleanup() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "deadline").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    accepted(&events).await;
    let cancelled = tokio::time::timeout(Duration::from_secs(61), fixture.cancellations.recv())
        .await
        .expect("host cancels at the deadline");
    assert_eq!(cancelled.as_deref(), Some("deadline"));
    tokio::time::sleep(crate::STOP_GRACE).await;
    complete(&events, CallOutcome::DeadlineExceeded(true)).await;
    assert_matches!(
        pending.await,
        Ok(CallReply::Outcome(CallOutcome::DeadlineExceeded(true)))
    );
    assert!(!fixture.session.is_closed());
    fixture.session.close().await;
}
#[tokio::test(start_paused = true)]
async fn acknowledged_cancel_without_a_result_closes_the_stream() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "cancel").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    accepted(&events).await;
    fixture.session.cancel("cancel".into()).await;
    assert_eq!(
        fixture.cancellations.recv().await.as_deref(),
        Some("cancel")
    );
    assert_matches!(
        tokio::time::timeout(Duration::from_secs(30), pending).await,
        Ok(Ok(CallReply::StreamEnded(reason))) if reason.contains("cancel")
    );
    assert!(fixture.session.is_closed());
    tokio::time::timeout(Duration::from_secs(1), events.closed())
        .await
        .expect("unconfirmed cancellation closes the RPC");
}
#[tokio::test]
async fn lifecycle_close_settles_and_cancels_pending_rpcs() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "pending").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    tokio::time::timeout(Duration::from_secs(1), fixture.session.close())
        .await
        .expect("bounded shutdown");
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(_)));
    tokio::time::timeout(Duration::from_secs(1), events.closed())
        .await
        .expect("gRPC response is cancelled");
}
#[tokio::test]
async fn close_ends_the_connection_once_borrowed_clients_are_released() {
    let fixture = open().await;
    drop(fixture.session.client().await.expect("client"));
    let mut guest_disconnected = fixture.disconnect.subscribe();
    fixture.session.close().await;
    tokio::time::timeout(
        Duration::from_secs(1),
        guest_disconnected.wait_for(|disconnected| *disconnected),
    )
    .await
    .expect("guest sees the connection end while the session is still held")
    .expect("guest connection state");
    assert_matches!(
        fixture.session.client().await,
        Err(SandboxdSessionError::Closed)
    );
}
#[tokio::test]
async fn broken_rpc_settles_all_calls_without_replay() {
    let mut fixture = open().await;
    let first = call(&fixture.session, "first").await;
    let (_, first_events) = fixture.requests.recv().await.expect("first");
    accepted(&first_events).await;
    let second = call(&fixture.session, "second").await;
    let (_, _second_events) = fixture.requests.recv().await.expect("second");
    first_events
        .send(Err(Status::unavailable("connection lost")))
        .await
        .expect("fail response");
    assert_matches!(first.await, Ok(CallReply::StreamEnded(_)));
    assert_matches!(second.await, Ok(CallReply::StreamEnded(_)));
    assert!(fixture.session.is_closed());
    assert!(fixture.requests.try_recv().is_err());
    fixture.disconnect.send_replace(true);
}
#[tokio::test]
async fn dropping_session_closes_pending_rpcs() {
    let mut fixture = open().await;
    let pending = call(&fixture.session, "dropped").await;
    let (_, events) = fixture.requests.recv().await.expect("request");
    drop(fixture.session);
    assert_matches!(pending.await, Ok(CallReply::StreamEnded(_)));
    tokio::time::timeout(Duration::from_secs(1), events.closed())
        .await
        .expect("RPC cancelled");
}
#[tokio::test]
async fn full_execute_queue_rejects_only_excess_calls() {
    let mut fixture = open().await;
    let mut pending = Vec::new();
    let mut streams = Vec::new();
    for id in 0..MAX_PENDING_CALLS {
        pending.push(call(&fixture.session, &format!("call-{id}")).await);
        let (_, events) = fixture.requests.recv().await.expect("request");
        accepted(&events).await;
        streams.push(events);
    }
    let overflow = call(&fixture.session, "overflow").await;
    assert_matches!(overflow.await, Ok(CallReply::Outcome(CallOutcome::Rejected(message))) if message.contains("queue is full"));
    complete(&streams[0], CallOutcome::Completed(Box::default())).await;
    assert_matches!(
        pending.remove(0).await,
        Ok(CallReply::Outcome(CallOutcome::Completed(_)))
    );
    assert!(!fixture.session.is_closed());
    fixture.session.close().await;
}

#[tokio::test]
async fn idle_connection_loss_closes_the_session_before_dispatch() {
    let fixture = open().await;
    fixture.disconnect.send_replace(true);
    tokio::time::timeout(Duration::from_secs(1), fixture.session.commands.closed())
        .await
        .expect("idle connection loss is observed");
    assert!(fixture.session.is_closed());
    assert_matches!(
        fixture
            .session
            .execute(
                "late".into(),
                ToolCall::default(),
                Duration::from_secs(5),
                100
            )
            .await,
        Err(SandboxdSessionError::Closed)
    );
}
