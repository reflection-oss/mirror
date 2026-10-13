use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use mirror_agent::agent_rpc_client::AgentRpcClient;
use mirror_agent::ClientMessage;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::io::Lines;
use tokio::process::Child;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[tokio::test]
async fn sigterm_closes_worker_with_open_rpc_session() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut child, mut lines) = start_worker(directory.path(), false, Stdio::inherit());
    let address = worker_address(&mut lines).await;
    let mut client = AgentRpcClient::connect(format!("http://{address}"))
        .await
        .expect("connect to worker");
    let (_requests, request_stream) = mpsc::channel::<ClientMessage>(1);
    let _responses = client
        .session(ReceiverStream::new(request_stream))
        .await
        .expect("open RPC session");

    let process_id = child.id().expect("worker process ID");
    assert_eq!(unsafe { libc::kill(process_id as i32, libc::SIGTERM) }, 0);
    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("worker did not stop after SIGTERM")
        .expect("wait for worker");
    assert!(status.success(), "worker exited with {status}");
}

#[tokio::test]
async fn second_worker_for_an_open_conversation_names_the_holder() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut first, mut first_lines) = start_worker(directory.path(), false, Stdio::null());
    worker_address(&mut first_lines).await;

    let (mut second, mut second_lines) = start_worker(directory.path(), false, Stdio::null());
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
    assert_eq!(unsafe { libc::kill(process_id as i32, libc::SIGTERM) }, 0);
    first.wait().await.expect("wait for first worker");
}

#[tokio::test]
async fn worker_exits_when_its_parent_closes_stdin() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let (mut child, mut lines) = start_worker(directory.path(), true, Stdio::piped());
    worker_address(&mut lines).await;

    drop(child.stdin.take());

    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("worker did not stop after stdin closed")
        .expect("wait for worker");
    assert!(status.success(), "worker exited with {status}");
}

fn start_worker(
    directory: &Path,
    exit_on_stdin_close: bool,
    stdin: Stdio,
) -> (Child, Lines<BufReader<ChildStdout>>) {
    let auth = directory.join("auth.json");
    std::fs::write(
        &auth,
        r#"{"tokens":{"access_token":"test","account_id":"test"}}"#,
    )
    .expect("write credentials");
    let config = serde_json::json!({
        "rpc_address": "127.0.0.1:0",
        "conversation_id": "session",
        "model": "test",
        "database": directory.join("state.db"),
        "create_session": true,
        "exit_on_stdin_close": exit_on_stdin_close,
    });
    let mut child = Command::new(worker_binary())
        .arg(config.to_string())
        .env("MIRROR_WORKSPACE", directory)
        .env("MIRROR_CODEX_AUTH_PATH", auth)
        .stdin(stdin)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("start worker");
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
