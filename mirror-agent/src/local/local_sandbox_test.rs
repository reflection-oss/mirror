use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use assert_matches::assert_matches;
use mirror_protocol::chat_record;
use mirror_protocol::ChatRecord;
use mirror_protocol::FunctionCall;
use mirror_protocol::ToolMessage;
use mirror_sandbox::execute_event;
use mirror_sandbox::sandboxd_server::Sandboxd;
use mirror_sandbox::CallOutcome;
use mirror_sandbox::CancelRequest;
use mirror_sandbox::CancelResponse;
use mirror_sandbox::EgressEvent;
use mirror_sandbox::EgressUpdate;
use mirror_sandbox::ExecuteEvent;
use mirror_sandbox::ExecuteRequest;
use mirror_sandbox::ExecuteResponse;
use mirror_sandbox::ProvidedSandbox;
use mirror_sandbox::SandboxPlacement;
use mirror_sandbox::SandboxProviderError;
use mirror_sandbox::SandboxStreamOutcome;
use mirror_sandbox::SandboxTransportError;
use mirror_sandbox::SandboxdStream;
use pretty_assertions::assert_eq;
use tokio::io::AsyncWriteExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use super::*;

const IDENTITY: LocalSandboxIdentity = LocalSandboxIdentity {
    request_id: Uuid::from_u128(1),
    session_id: Uuid::from_u128(2),
};
const TIMEOUT: Duration = Duration::from_secs(30);

/// A sandbox whose `sandboxd` answers every call with the tool's name.
#[derive(Default)]
struct FakeSandbox {
    fail_create: bool,
    fail_suspend: bool,
    fail_probe: bool,
    probe_padding: usize,
    stopped: tokio::sync::Notify,
    started: Arc<tokio::sync::Notify>,
    suspended: AtomicBool,
    connects: AtomicUsize,
    resumes: AtomicUsize,
    /// Ends each `sandboxd` connection served so far.
    servers: Mutex<Vec<watch::Sender<bool>>>,
}

impl FakeSandbox {
    fn drop_connections(&self) {
        for server in self.servers.lock().expect("servers").drain(..) {
            server.send_replace(true);
        }
    }
}

#[async_trait]
impl SandboxProvider for FakeSandbox {
    async fn create(&self, _: Uuid, _: Uuid) -> Result<String, SandboxProviderError> {
        if self.fail_create {
            return Err(SandboxProviderError::Rejected("no such image".to_string()));
        }
        Ok("fake-sandbox".to_string())
    }

    async fn wait_until_ready(&self, _: &str) -> Result<SandboxPlacement, SandboxProviderError> {
        Ok(SandboxPlacement::default())
    }

    async fn liveness(&self, _: &str) -> Result<SandboxLiveness, SandboxProviderError> {
        if self.suspended.load(Ordering::SeqCst) {
            return Ok(SandboxLiveness::Suspended);
        }
        Ok(SandboxLiveness::Ready(SandboxPlacement::default()))
    }

    async fn suspend(&self, _: &str) -> Result<(), SandboxProviderError> {
        self.stopped.notify_one();
        if self.fail_suspend {
            return Err(SandboxProviderError::Retryable("stop unavailable".into()));
        }
        self.suspended.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn wait_until_suspended(&self, _: &str) -> Result<(), SandboxProviderError> {
        Ok(())
    }

    async fn resume(&self, _: &str) -> Result<(), SandboxProviderError> {
        self.resumes.fetch_add(1, Ordering::SeqCst);
        self.suspended.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn delete(&self, _: &str) -> Result<(), SandboxProviderError> {
        Ok(())
    }

    async fn list(&self) -> Result<Vec<ProvidedSandbox>, SandboxProviderError> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl SandboxTransport for FakeSandbox {
    async fn connect(&self, _: &str, _: &str) -> Result<SandboxdStream, SandboxTransportError> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        if self.suspended.load(Ordering::SeqCst) {
            return Err(SandboxTransportError::Suspended("stopped".to_string()));
        }
        let (host, guest) = tokio::io::duplex(64 * 1024);
        let (input, output) = tokio::io::split(guest);
        let (server, _) = watch::channel(false);
        self.servers.lock().expect("servers").push(server.clone());
        drop(tokio::spawn(mirror_sandbox::serve_connection(
            input,
            output,
            NameEcho(self.started.clone()),
            server,
        )));
        Ok(SandboxdStream::split(host))
    }

    /// Answers the trusted sandboxd context probe.
    async fn stream(
        &self,
        _: &str,
        mut request: SandboxStream<'_>,
    ) -> Result<SandboxStreamOutcome, SandboxTransportError> {
        if self.fail_probe {
            return Err(SandboxTransportError::OutcomeUnknown(
                "probe connection lost".into(),
            ));
        }
        assert_eq!(request.command, format!("{SANDBOXD_PATH} context"));
        let mut context = vec![b' '; self.probe_padding];
        context.extend(
            serde_json::to_vec(&serde_json::json!([
                {"working_directory": "/workspace", "is_git_repo": false,
                 "platform": "Linux", "os_version": "Linux 6.12.34", "shell": "bash 5.2.15"},
                {"instructions": [], "skills": []},
            ]))
            .expect("context"),
        );
        request
            .stdout
            .write_all(&context)
            .await
            .expect("write probe output");
        Ok(SandboxStreamOutcome::default())
    }
}

#[derive(Clone)]
struct NameEcho(Arc<tokio::sync::Notify>);

#[tonic::async_trait]
impl Sandboxd for NameEcho {
    type RunCommandStream = ReceiverStream<Result<mirror_sandbox::RunCommandResponse, Status>>;
    async fn run_command(
        &self,
        _: Request<tonic::Streaming<mirror_sandbox::RunCommandRequest>>,
    ) -> Result<Response<Self::RunCommandStream>, Status> {
        Err(Status::unimplemented("tool-only fixture"))
    }

    type EgressStream = ReceiverStream<Result<EgressEvent, Status>>;

    async fn egress(
        &self,
        _: Request<tonic::Streaming<EgressUpdate>>,
    ) -> Result<Response<Self::EgressStream>, Status> {
        Err(Status::unimplemented("execution-only fixture"))
    }

    type ExecuteStream = ReceiverStream<Result<ExecuteEvent, Status>>;

    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        let name = request
            .into_inner()
            .call
            .and_then(|call| call.function)
            .map(|function| function.name)
            .unwrap_or_default();
        let record = ChatRecord {
            message: Some(chat_record::Message::Tool(ToolMessage {
                content: format!("ran {name}"),
                ..ToolMessage::default()
            })),
            success: true,
            ..ChatRecord::default()
        };
        let (events, receiver) = mpsc::channel(2);
        let accepted = ExecuteEvent {
            event: Some(execute_event::Event::Accepted(true)),
        };
        let completed = ExecuteEvent {
            event: Some(execute_event::Event::Completed(ExecuteResponse {
                outcome: Some(if name == "uncertain" {
                    CallOutcome::DeadlineExceeded(false)
                } else {
                    CallOutcome::Completed(Box::new(record))
                }),
                metadata: None,
            })),
        };
        if name == "wait" {
            self.0.notify_one();
            std::future::pending::<()>().await;
        }
        events.send(Ok(accepted)).await.expect("accept");
        events.send(Ok(completed)).await.expect("complete");
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    async fn cancel(&self, _: Request<CancelRequest>) -> Result<Response<CancelResponse>, Status> {
        Ok(Response::new(CancelResponse {}))
    }
}

fn call(name: &str) -> ToolCall {
    ToolCall {
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        }),
        ..ToolCall::default()
    }
}

#[tokio::test]
async fn calls_share_one_session_to_the_sandbox() {
    let sandbox = Arc::new(FakeSandbox::default());
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".to_string());

    let read = local.execute(&call("read"), TIMEOUT).await;
    let listed = local.execute(&call("ls"), TIMEOUT).await;

    assert_matches!(read, SandboxCallOutcome::Completed(output) if output.content == "ran read");
    assert_matches!(listed, SandboxCallOutcome::Completed(output) if output.content == "ran ls");
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn unconfirmed_completion_quarantines_before_returning() {
    let sandbox = Arc::new(FakeSandbox::default());
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".into());

    assert_matches!(
        local.execute(&call("uncertain"), TIMEOUT).await,
        SandboxCallOutcome::Unknown(_)
    );
    assert!(sandbox.suspended.load(Ordering::SeqCst));
    assert_matches!(
        local.execute(&call("write"), TIMEOUT).await,
        SandboxCallOutcome::Rejected(_)
    );
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn abandoned_call_quarantines_without_a_waiting_caller() {
    let sandbox = Arc::new(FakeSandbox::default());
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".into());
    let execution = tokio::spawn(async move { local.execute(&call("wait"), TIMEOUT).await });
    sandbox.started.notified().await;

    execution.abort();
    assert!(execution
        .await
        .expect_err("caller cancelled")
        .is_cancelled());
    tokio::time::timeout(TIMEOUT, sandbox.stopped.notified())
        .await
        .expect("VM stopped");
    assert!(sandbox.suspended.load(Ordering::SeqCst));
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn connection_loss_stops_the_sandbox_and_rejects_concurrent_calls() {
    let sandbox = Arc::new(FakeSandbox::default());
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".to_string());
    assert_matches!(
        local.execute(&call("read"), TIMEOUT).await,
        SandboxCallOutcome::Completed(_)
    );

    sandbox.drop_connections();
    tokio::time::timeout(TIMEOUT, sandbox.stopped.notified())
        .await
        .expect("connection loss stops even an idle sandbox");
    let (read, list) = (call("read"), call("ls"));
    let (read, listed) =
        tokio::join!(local.execute(&read, TIMEOUT), local.execute(&list, TIMEOUT),);

    assert_matches!(read, SandboxCallOutcome::Rejected(detail) if detail.contains("quarantined"));
    assert_matches!(listed, SandboxCallOutcome::Rejected(detail) if detail.contains("quarantined"));
    assert!(sandbox.suspended.load(Ordering::SeqCst));
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 1);
    assert_eq!(sandbox.resumes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_stop_does_not_allow_reconnection() {
    let sandbox = Arc::new(FakeSandbox {
        fail_suspend: true,
        ..FakeSandbox::default()
    });
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".into());
    assert_matches!(
        local.execute(&call("read"), TIMEOUT).await,
        SandboxCallOutcome::Completed(_)
    );

    sandbox.drop_connections();
    tokio::time::timeout(TIMEOUT, sandbox.stopped.notified())
        .await
        .expect("attempt to stop the sandbox");
    let result = local.execute(&call("write"), TIMEOUT).await;

    assert_matches!(result, SandboxCallOutcome::Rejected(detail) if detail.contains("stop could not be confirmed"));
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 1);
    assert_eq!(sandbox.resumes.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn a_suspended_sandbox_is_resumed_before_the_call() {
    let sandbox = Arc::new(FakeSandbox {
        suspended: AtomicBool::new(true),
        ..FakeSandbox::default()
    });
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".to_string());

    let read = local.execute(&call("read"), TIMEOUT).await;

    assert_matches!(read, SandboxCallOutcome::Completed(_));
    assert_eq!(sandbox.resumes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_sandbox_that_cannot_be_created_rejects_the_call() {
    let sandbox = Arc::new(FakeSandbox {
        fail_create: true,
        ..FakeSandbox::default()
    });
    let local = LocalSandbox::start(sandbox, IDENTITY, "/workspace".to_string());

    let read = local.execute(&call("read"), TIMEOUT).await;

    assert_matches!(
        read,
        SandboxCallOutcome::Rejected(detail) if detail.contains("no such image")
    );
}

#[tokio::test]
async fn uncertain_environment_probe_quarantines_tool_execution() {
    let sandbox = Arc::new(FakeSandbox {
        fail_probe: true,
        ..FakeSandbox::default()
    });
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".into());

    let environment = local.environment().await;
    let result = local.execute(&call("write"), TIMEOUT).await;

    assert_matches!(environment, Err(detail) if detail.contains("probe connection lost"));
    assert_matches!(result, SandboxCallOutcome::Rejected(detail) if detail.contains("quarantined"));
    assert!(sandbox.suspended.load(Ordering::SeqCst));
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn environment_drains_more_than_the_pipe_capacity() {
    let local = LocalSandbox::start(
        Arc::new(FakeSandbox {
            probe_padding: 6000,
            ..FakeSandbox::default()
        }),
        IDENTITY,
        "/workspace".into(),
    );

    assert_matches!(local.environment().await, Ok(_));
}

#[tokio::test]
async fn environment_describes_the_sandbox_from_inside_it() {
    let local = LocalSandbox::start(
        Arc::new(FakeSandbox::default()),
        IDENTITY,
        "/workspace".to_string(),
    );

    let environment = local.environment().await.expect("environment");

    assert_eq!(environment.platform, "Linux");
    assert_eq!(environment.os_version, "Linux 6.12.34");
    assert_eq!(environment.shell, "bash 5.2.15");
}
