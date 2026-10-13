use super::*;
use assert_matches::assert_matches;
use mirror_protocol::{chat_record, ChatRecord, FunctionCall, ToolCall, ToolMessage};
use mirror_sandbox::{
    connect, execute_event, sandboxd_client::SandboxdClient, CallOutcome, CancelRequest,
    ExecuteEvent, ExecuteRequest, MAX_DISPATCH_CALL_ID_BYTES, MAX_MESSAGE_BYTES,
};
use pretty_assertions::assert_eq;
use prost::Message;
use std::io::Write;
use std::time::Duration;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

type Client = SandboxdClient<Channel>;
async fn start(script: &str) -> (Client, JoinHandle<Result<(), ServeError>>) {
    let (host, guest) = tokio::io::duplex(64 * 1024);
    let (input, output) = tokio::io::split(guest);
    let served = tokio::spawn(serve(
        input,
        output,
        RunnerCommand {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
        },
    ));
    let (input, output) = tokio::io::split(host);
    (
        connect(input, output, watch::channel(false).0)
            .await
            .expect("gRPC connection"),
        served,
    )
}
fn execute(id: &str, timeout: Duration) -> ExecuteRequest {
    ExecuteRequest {
        call_id: id.into(),
        timeout_millis: timeout.as_millis() as u64,
        max_output_bytes: MAX_MESSAGE_BYTES as u64,
        call: Some(ToolCall {
            r#type: "function".into(),
            function: Some(FunctionCall {
                name: id.into(),
                arguments: "{}".into(),
            }),
            ..ToolCall::default()
        }),
    }
}
fn record(content: &str) -> ChatRecord {
    ChatRecord {
        message: Some(chat_record::Message::Tool(ToolMessage {
            content: content.into(),
            ..ToolMessage::default()
        })),
        success: true,
        ..ChatRecord::default()
    }
}
fn record_file(content: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().expect("record file");
    file.write_all(&record(content).encode_to_vec())
        .expect("write record");
    file
}
async fn admitted(client: &mut Client, request: ExecuteRequest) -> tonic::Streaming<ExecuteEvent> {
    let mut response = client
        .execute(request)
        .await
        .expect("execute RPC")
        .into_inner();
    assert_matches!(
        response.message().await.expect("admission"),
        Some(ExecuteEvent {
            event: Some(execute_event::Event::Accepted(true))
        })
    );
    response
}
async fn outcome(response: &mut tonic::Streaming<ExecuteEvent>) -> CallOutcome {
    let event = tokio::time::timeout(Duration::from_secs(20), response.message())
        .await
        .expect("bounded completion")
        .expect("RPC result")
        .expect("completion");
    match event.event {
        Some(execute_event::Event::Completed(response)) => response.outcome.expect("outcome"),
        other => panic!("expected completion, got {other:?}"),
    }
}
async fn stopped(task: JoinHandle<Result<(), ServeError>>) {
    assert_matches!(
        tokio::time::timeout(Duration::from_secs(20), task).await,
        Ok(Ok(Ok(())))
    );
}

#[tokio::test]
async fn execute_returns_shared_record_and_completed_id_can_be_reused() {
    let file = record_file("sandboxd ok");
    let (mut client, served) =
        start(&format!("cat >/dev/null; cat {}", file.path().display())).await;
    let mut first = admitted(&mut client, execute("same", Duration::from_secs(10))).await;
    assert_eq!(
        outcome(&mut first).await,
        CallOutcome::Completed(Box::new(record("sandboxd ok")))
    );
    let mut second = admitted(&mut client, execute("same", Duration::from_secs(10))).await;
    assert_eq!(
        outcome(&mut second).await,
        CallOutcome::Completed(Box::new(record("sandboxd ok")))
    );
    drop((first, second, client));
    stopped(served).await;
}

#[tokio::test]
async fn egress_refresh_timeout_fails_closed_and_rejects_a_late_identity() {
    use mirror_sandbox::{egress_update, EgressIdentity, EgressProxyConfig, EgressUpdate};
    use tokio_stream::wrappers::ReceiverStream;
    let certificate = rcgen::generate_simple_self_signed(vec!["localhost".into()]).expect("CA");
    let (mut client, served) = start("cat >/dev/null").await;
    let (updates, receiver) = mpsc::channel(1);
    updates
        .send(EgressUpdate {
            update: Some(egress_update::Update::Configure(EgressProxyConfig {
                host: "localhost".into(),
                port: 443,
                ca_pem: certificate.cert.pem(),
            })),
        })
        .await
        .expect("configure");
    let mut events = client
        .egress(ReceiverStream::new(receiver))
        .await
        .expect("renewal RPC")
        .into_inner();
    let initial = events
        .message()
        .await
        .expect("initial request")
        .expect("event");
    let expiry = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
        + 3;
    updates
        .send(EgressUpdate {
            update: Some(egress_update::Update::Identity(EgressIdentity {
                request_id: initial.request_id,
                expires_at: expiry,
                token: b"mse4.initial.signature".to_vec(),
            })),
        })
        .await
        .expect("identity");
    assert!(
        events
            .message()
            .await
            .expect("installed")
            .expect("ack")
            .installed
    );
    let request = tokio::time::timeout(Duration::from_secs(3), events.message())
        .await
        .expect("automatic refresh")
        .expect("refresh")
        .expect("request");
    assert_eq!(request.request_id, 2);
    assert!(!request.installed);
    let error = tokio::time::timeout(Duration::from_secs(3), events.message())
        .await
        .expect("bounded expiry")
        .expect_err("no replacement");
    assert_eq!(error.code(), tonic::Code::PermissionDenied);
    assert!(!error.to_string().contains("mse4."));
    tokio::time::timeout(Duration::from_secs(2), updates.closed())
        .await
        .expect("closed renewal input");
    drop((updates, events, client));
    stopped(served).await;
}

#[tokio::test]
async fn cancel_reaps_the_process_group_before_confirmation() {
    let (mut client, served) =
        start("(trap '' TERM; exec sleep 30) >/dev/null 2>&1 & cat >/dev/null; sleep 30").await;
    let mut response = admitted(&mut client, execute("cancel", Duration::from_secs(60))).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    client
        .cancel(CancelRequest {
            call_id: "cancel".into(),
        })
        .await
        .expect("cancel");
    assert_eq!(outcome(&mut response).await, CallOutcome::Cancelled(true));
    drop((response, client));
    stopped(served).await;
}

#[tokio::test]
async fn deadline_stops_the_run_and_reports_confirmation() {
    let (mut client, served) = start("cat >/dev/null; sleep 30").await;
    let mut response = admitted(&mut client, execute("deadline", Duration::from_millis(200))).await;
    assert_eq!(
        outcome(&mut response).await,
        CallOutcome::DeadlineExceeded(true)
    );
    drop((response, client));
    stopped(served).await;
}

#[tokio::test]
async fn dropping_an_rpc_stops_its_process_and_keeps_other_calls_usable() {
    let directory = tempfile::tempdir().expect("directory");
    let pid_file = directory.path().join("pid");
    let (mut client, served) = start(&format!(
        "echo $$ > {}; cat >/dev/null; exec sleep 30",
        pid_file.display()
    ))
    .await;
    let response = admitted(&mut client, execute("abandoned", Duration::from_secs(60))).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while !pid_file.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("runner starts");
    let pid: i32 = std::fs::read_to_string(&pid_file)
        .expect("pid")
        .trim()
        .parse()
        .expect("pid number");
    drop(response);
    tokio::time::timeout(Duration::from_secs(15), async {
        while nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("abandoned process is reaped");
    let mut next = admitted(&mut client, execute("next", Duration::from_millis(100))).await;
    assert_eq!(
        outcome(&mut next).await,
        CallOutcome::DeadlineExceeded(true)
    );
    drop((next, client));
    stopped(served).await;
}

#[tokio::test]
async fn runner_failure_keeps_bounded_diagnostics() {
    let (mut client, served) =
        start("cat >/dev/null; echo boom >&2; head -c 300000 /dev/zero | tr '\\0' x >&2; exit 3")
            .await;
    let mut response = admitted(&mut client, execute("failure", Duration::from_secs(10))).await;
    assert_matches!(outcome(&mut response).await, CallOutcome::Failed(message) if message.contains("exit status: 3") && message.contains("boom"));
    drop((response, client));
    stopped(served).await;
}

#[tokio::test]
async fn oversized_id_is_rejected_without_running_the_tool() {
    let dir = tempfile::tempdir().expect("directory");
    let marker = dir.path().join("started");
    let (mut client, served) = start(&format!("touch {}", marker.display())).await;
    let mut response = client
        .execute(execute(
            &"x".repeat(MAX_DISPATCH_CALL_ID_BYTES + 1),
            Duration::from_secs(1),
        ))
        .await
        .expect("execute")
        .into_inner();
    assert_matches!(outcome(&mut response).await, CallOutcome::Rejected(message) if message.len() < 128);
    assert!(!marker.exists());
    drop((response, client));
    stopped(served).await;
}

#[tokio::test]
async fn concurrent_rpcs_complete_independently() {
    let file = record_file("done");
    let (mut client, served) = start(&format!(
        "case \"$(cat)\" in *slow*) sleep 1;; esac; cat {}",
        file.path().display()
    ))
    .await;
    let mut slow = admitted(&mut client, execute("slow", Duration::from_secs(10))).await;
    let mut fast = admitted(&mut client, execute("fast", Duration::from_secs(10))).await;
    assert_eq!(
        outcome(&mut fast).await,
        CallOutcome::Completed(Box::new(record("done")))
    );
    assert_eq!(
        outcome(&mut slow).await,
        CallOutcome::Completed(Box::new(record("done")))
    );
    drop((slow, fast, client));
    stopped(served).await;
}

#[tokio::test]
async fn duplicate_call_id_stays_reserved_during_cancellation() {
    let (mut client, served) = start("trap '' TERM; cat >/dev/null; sleep 30").await;
    let mut running = admitted(&mut client, execute("duplicate", Duration::from_secs(60))).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    client
        .cancel(CancelRequest {
            call_id: "duplicate".into(),
        })
        .await
        .expect("cancel");
    let mut duplicate = client
        .execute(execute("duplicate", Duration::from_secs(60)))
        .await
        .expect("duplicate")
        .into_inner();
    assert_matches!(outcome(&mut duplicate).await, CallOutcome::Rejected(message) if message.contains("already running"));
    assert_eq!(outcome(&mut running).await, CallOutcome::Cancelled(true));
    drop((running, duplicate, client));
    stopped(served).await;
}

#[tokio::test]
async fn completed_record_wins_a_race_with_cancellation() {
    let file = record_file("done after all");
    let (mut client, served) = start(&format!(
        "trap 'cat {}; exit 0' TERM; cat >/dev/null; sleep 30",
        file.path().display()
    ))
    .await;
    let mut response = admitted(&mut client, execute("race", Duration::from_secs(60))).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    client
        .cancel(CancelRequest {
            call_id: "race".into(),
        })
        .await
        .expect("cancel");
    assert_eq!(
        outcome(&mut response).await,
        CallOutcome::Completed(Box::new(record("done after all")))
    );
    drop((response, client));
    stopped(served).await;
}

#[tokio::test]
async fn concurrent_limit_rejects_excess_without_interrupting_admitted_calls() {
    let (mut client, served) = start("cat >/dev/null; exec sleep 60").await;
    let mut responses = Vec::new();
    for id in 0..64 {
        responses.push(
            admitted(
                &mut client,
                execute(&format!("call-{id}"), Duration::from_secs(120)),
            )
            .await,
        );
    }
    let mut overflow = client
        .execute(execute("overflow", Duration::from_secs(120)))
        .await
        .expect("overflow")
        .into_inner();
    assert_matches!(outcome(&mut overflow).await, CallOutcome::Rejected(message) if message.contains("concurrent call limit"));
    drop((responses, overflow, client));
    stopped(served).await;
}

#[tokio::test]
async fn maximum_length_call_id_is_dispatched() {
    let file = record_file("accepted");
    let (mut client, served) =
        start(&format!("cat >/dev/null; cat {}", file.path().display())).await;
    let mut response = admitted(
        &mut client,
        execute(
            &"x".repeat(MAX_DISPATCH_CALL_ID_BYTES),
            Duration::from_secs(10),
        ),
    )
    .await;
    assert_eq!(
        outcome(&mut response).await,
        CallOutcome::Completed(Box::new(record("accepted")))
    );
    drop((response, client));
    stopped(served).await;
}

#[tokio::test]
async fn oversized_response_reports_failure_and_keeps_connection_usable() {
    let file = record_file(&"x".repeat(MAX_MESSAGE_BYTES - 16));
    let encoded = std::fs::read(file.path()).expect("record");
    assert!(encoded.len() < MAX_MESSAGE_BYTES);
    let event = ExecuteEvent {
        event: Some(execute_event::Event::Completed(
            mirror_sandbox::ExecuteResponse {
                outcome: Some(CallOutcome::Completed(Box::new(
                    ChatRecord::decode(encoded.as_slice()).expect("record"),
                ))),
            },
        )),
    };
    assert!(event.encoded_len() > MAX_MESSAGE_BYTES);
    drop((encoded, event));
    let small = record_file("small");
    let (mut client, served) = start(&format!(
        "if grep -aq large; then cat {}; else cat {}; fi",
        file.path().display(),
        small.path().display()
    ))
    .await;
    let mut large = admitted(&mut client, execute("large", Duration::from_secs(20))).await;
    assert_matches!(outcome(&mut large).await, CallOutcome::Failed(message) if message.contains("message limit"));
    let mut next = admitted(&mut client, execute("small", Duration::from_secs(10))).await;
    assert_eq!(
        outcome(&mut next).await,
        CallOutcome::Completed(Box::new(record("small")))
    );
    drop((large, next, client));
    stopped(served).await;
}

#[tokio::test]
async fn connection_loss_stops_and_reaps_running_processes() {
    let directory = tempfile::tempdir().expect("directory");
    let pid_file = directory.path().join("pid");
    let (host, mut to_host) = tokio::io::duplex(4096);
    let (mut to_guest, guest) = tokio::io::duplex(4096);
    let pump = tokio::spawn(async move {
        let _ = tokio::io::copy_bidirectional(&mut to_host, &mut to_guest).await;
    });
    let (input, output) = tokio::io::split(guest);
    let served = tokio::spawn(serve(
        input,
        output,
        RunnerCommand {
            program: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                format!(
                    "echo $$ > {}; cat >/dev/null; exec sleep 30",
                    pid_file.display()
                ),
            ],
        },
    ));
    let (input, output) = tokio::io::split(host);
    let mut client = connect(input, output, watch::channel(false).0)
        .await
        .expect("connect");
    let mut response = admitted(&mut client, execute("running", Duration::from_secs(60))).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while !pid_file.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("runner starts");
    let pid: i32 = std::fs::read_to_string(&pid_file)
        .expect("pid")
        .trim()
        .parse()
        .expect("pid number");
    pump.abort();
    let _ = pump.await;
    stopped(served).await;
    assert_eq!(
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
        Err(nix::errno::Errno::ESRCH)
    );
    assert_matches!(response.message().await, Err(_));
}
