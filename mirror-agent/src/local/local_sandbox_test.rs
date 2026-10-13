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

    async fn wait_until_ready(&self, _: &str) -> Result<(), SandboxProviderError> {
        Ok(())
    }

    async fn liveness(&self, _: &str) -> Result<SandboxLiveness, SandboxProviderError> {
        if self.suspended.load(Ordering::SeqCst) {
            return Ok(SandboxLiveness::Suspended);
        }
        Ok(SandboxLiveness::Ready)
    }

    async fn suspend(&self, _: &str) -> Result<(), SandboxProviderError> {
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
            input, output, NameEcho, server,
        )));
        Ok(SandboxdStream::split(host))
    }

    /// Answers the environment probe as a Linux guest with bash would.
    async fn stream(
        &self,
        _: &str,
        mut request: SandboxStream,
    ) -> Result<SandboxStreamOutcome, SandboxTransportError> {
        let mut script = String::new();
        if let Some(stdin) = request.stdin.as_mut() {
            stdin.read_to_string(&mut script).await.expect("read probe");
        }
        assert_eq!(script, ENVIRONMENT_PROBE);
        request
            .stdout
            .write_all(b"Linux 6.12.34\n5.2.15\n")
            .await
            .expect("write probe output");
        Ok(SandboxStreamOutcome::default())
    }
}

#[derive(Clone)]
struct NameEcho;

#[tonic::async_trait]
impl Sandboxd for NameEcho {
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
                outcome: Some(CallOutcome::Completed(Box::new(record))),
            })),
        };
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
async fn concurrent_calls_after_a_lost_connection_reopen_it_once() {
    let sandbox = Arc::new(FakeSandbox::default());
    let local = LocalSandbox::start(sandbox.clone(), IDENTITY, "/workspace".to_string());
    let first = local.connection().await.expect("connection");
    let mut closing = first.session.closing();

    sandbox.drop_connections();
    closing
        .wait_for(|closing| *closing)
        .await
        .expect("the session closes");
    let (read, list) = (call("read"), call("ls"));
    let (read, listed) =
        tokio::join!(local.execute(&read, TIMEOUT), local.execute(&list, TIMEOUT),);

    assert_matches!(read, SandboxCallOutcome::Completed(output) if output.content == "ran read");
    assert_matches!(listed, SandboxCallOutcome::Completed(output) if output.content == "ran ls");
    assert_eq!(sandbox.connects.load(Ordering::SeqCst), 2);
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
async fn environment_describes_the_sandbox_from_inside_it() {
    let local = LocalSandbox::start(
        Arc::new(FakeSandbox::default()),
        IDENTITY,
        "/workspace".to_string(),
    );

    let environment = local
        .environment(Path::new("/workspace"))
        .await
        .expect("environment");

    assert_eq!(environment.platform, "Linux");
    assert_eq!(environment.os_version, "Linux 6.12.34");
    assert_eq!(environment.shell, "bash 5.2.15");
}
