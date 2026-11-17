use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use hyper_util::rt::TokioIo;
use mirror_agent::agent_rpc_client::AgentRpcClient;
use mirror_agent::daemon::AgentConfiguration;
use mirror_agent::{ApprovalMode, ClientMessage, InferenceConfig, WorkerStartup};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use prost::Message;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::io::Lines;
use tokio::net::UnixStream;
use tokio::process::Child;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Endpoint;
use tower::service_fn;

#[tokio::test]
async fn sigterm_closes_worker_with_open_rpc_session() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut child, mut lines) = start_worker(directory.path(), false).await;
    let address = worker_address(&mut lines).await;
    let socket = address
        .strip_prefix("unix:")
        .expect("Unix worker address")
        .to_owned();
    assert_eq!(
        std::fs::metadata(&socket)
            .expect("socket metadata")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    let channel = Endpoint::from_static("http://localhost")
        .connect_with_connector(service_fn(move |_| {
            let socket = socket.clone();
            async move { UnixStream::connect(socket).await.map(TokioIo::new) }
        }))
        .await
        .expect("connect to worker");
    let mut client = AgentRpcClient::new(channel);
    let (_requests, request_stream) = mpsc::channel::<ClientMessage>(1);
    let _responses = client
        .session(ReceiverStream::new(request_stream))
        .await
        .expect("open RPC session");

    let process_id = child.id().expect("worker process ID");
    kill(Pid::from_raw(process_id as i32), Signal::SIGTERM).expect("signal worker");
    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("worker did not stop after SIGTERM")
        .expect("wait for worker");
    assert!(status.success(), "worker exited with {status}");
}

#[tokio::test]
async fn second_worker_for_an_open_conversation_names_the_holder() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut first, mut first_lines) = start_worker(directory.path(), false).await;
    worker_address(&mut first_lines).await;

    let (mut second, mut second_lines) = start_worker(directory.path(), false).await;
    let error = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let line = second_lines
                .next_line()
                .await
                .expect("read second worker")
                .expect("second worker exited without an error");
            if let Some(error) = line.strip_prefix("MIRROR_RPC_ERROR=") {
                break error.to_string();
            }
        }
    })
    .await
    .expect("second worker error deadline");

    assert_eq!(
        error,
        format!(
            "conversation \"session\" is open in another Mirror process (pid {})",
            first.id().expect("first worker process ID")
        )
    );
    let status = tokio::time::timeout(Duration::from_secs(5), second.wait())
        .await
        .expect("second worker did not exit")
        .expect("wait for second worker");
    assert!(!status.success());
    let process_id = first.id().expect("first worker process ID");
    kill(Pid::from_raw(process_id as i32), Signal::SIGTERM).expect("signal worker");
    first.wait().await.expect("wait for first worker");
}

#[tokio::test]
async fn worker_exits_on_parent_eof_or_idle_drain() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut child, mut lines) = start_worker(directory.path(), true).await;
    worker_address(&mut lines).await;

    let drain_directory = tempfile::tempdir().expect("temporary drain directory");
    let address = format!(
        "unix:{}",
        drain_directory.path().join("agent.sock").display()
    );
    let (mut draining, mut drain_lines) = start_worker(drain_directory.path(), true).await;
    assert_eq!(worker_address(&mut drain_lines).await, address);

    drop(child.stdin.take());
    draining
        .stdin
        .as_mut()
        .expect("parent lifeline")
        .write_all(b"D")
        .await
        .expect("request idle drain");
    let drained = tokio::time::timeout(Duration::from_secs(5), draining.wait())
        .await
        .expect("worker did not drain with stdin still open")
        .expect("wait for drained worker");
    assert!(drained.success(), "drained worker exited with {drained}");

    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("worker did not stop after stdin closed")
        .expect("wait for worker");
    assert!(status.success(), "worker exited with {status}");
}

#[tokio::test]
async fn worker_rejects_tcp_before_opening_history() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let database = directory.path().join("state.db");
    let config = WorkerStartup {
        rpc_address: "127.0.0.1:0".into(),
        session_id: "session".into(),
        database: database.to_str().expect("database path").into(),
        ..Default::default()
    }
    .encode_to_vec();
    let mut child = Command::new(worker_binary().canonicalize().expect("worker binary"))
        .arg("--config-stdin")
        .env_clear()
        .current_dir(directory.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("start worker");
    let mut stdin = child.stdin.take().expect("worker stdin");
    stdin
        .write_all(&(config.len() as u32).to_be_bytes())
        .await
        .expect("startup size");
    stdin.write_all(&config).await.expect("startup frame");
    drop(stdin);
    let output = tokio::time::timeout(Duration::from_secs(5), child.wait_with_output())
        .await
        .expect("worker rejection deadline")
        .expect("wait for worker");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("worker RPC address must be a unix: socket path"));
    assert!(!database.exists());
}

async fn start_worker(
    directory: &Path,
    exit_on_stdin_close: bool,
) -> (Child, Lines<BufReader<ChildStdout>>) {
    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))
        .expect("private socket directory");
    let auth = directory.join("auth.json");
    std::fs::write(
        &auth,
        r#"{"tokens":{"access_token":"test","account_id":"test"}}"#,
    )
    .expect("write credentials");
    let config = WorkerStartup {
        configuration: Some(AgentConfiguration {
            inference: Some(InferenceConfig {
                model: "test".into(),
                api: Some("openai-codex-responses".into()),
                ..Default::default()
            }),
            auth_path: Some(auth.to_str().expect("auth path").into()),
            approval_mode: ApprovalMode::Auto.into(),
            ..Default::default()
        }),
        rpc_address: format!("unix:{}", directory.join("agent.sock").display()),
        session_id: "session".into(),
        database: directory
            .join("state.db")
            .to_str()
            .expect("database path")
            .into(),
        create_session: true,
        exit_on_stdin_close,
        ..Default::default()
    }
    .encode_to_vec();
    let mut child = Command::new(worker_binary())
        .arg("--config-stdin")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", directory)
        .env("MIRROR_WORKSPACE", directory)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("start worker");
    let stdin = child.stdin.as_mut().expect("worker stdin");
    stdin
        .write_all(&(config.len() as u32).to_be_bytes())
        .await
        .expect("startup size");
    stdin.write_all(&config).await.expect("startup frame");
    let stdout = child.stdout.take().expect("worker stdout");
    (child, BufReader::new(stdout).lines())
}

async fn worker_address(lines: &mut Lines<BufReader<ChildStdout>>) -> String {
    let report = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .expect("report deadline")
        .expect("read report")
        .expect("prompt report");
    assert!(report.starts_with("MIRROR_SESSION_MESSAGE="));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let line = lines
                .next_line()
                .await
                .expect("read worker address")
                .expect("worker exited before publishing its address");
            if let Some(address) = line.strip_prefix("MIRROR_RPC_ADDRESS=") {
                break address.to_string();
            }
            assert!(line.starts_with("MIRROR_SESSION_MESSAGE="));
        }
    })
    .await
    .expect("worker startup timed out")
}

fn worker_binary() -> PathBuf {
    std::env::var_os("MIRROR_AGENT_BIN")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_mirror-agent").map(PathBuf::from))
        .expect("worker binary path")
}
