use std::path::PathBuf;
use std::time::Duration;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use serde_json::json;
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;

use super::*;
use crate::sandboxd_file;
use crate::EgressHost;
use crate::FakeMsb;
use crate::MicrosandboxConfig;

const NAME: &str = "mirror-ws-test";

/// An `msb inspect` document whose status the fake fills in.
fn document(default_egress: &str) -> Value {
    json!({
        "status": "__STATUS__",
        "config": {
            "labels": {},
            "network": {
                "policy": {
                    "default_egress": default_egress,
                    "default_ingress": "deny",
                    "rules": [],
                },
                "ports": [],
            },
            "mounts": [
                {"host": "/workspace", "guest": "/workspace", "options": {"readonly": false}},
                {
                    "host": sandboxd_file(),
                    "guest": "/usr/local/bin/mirror-sandboxd",
                    "options": {"readonly": true},
                },
            ],
        },
        "active_config": null,
    })
}

async fn provider(fake: &FakeMsb) -> MicrosandboxProvider {
    let config = MicrosandboxConfig {
        image: "debian:bookworm-slim".to_string(),
        workspace: PathBuf::from("/workspace"),
        sandboxd: PathBuf::from("/opt/mirror/mirror-sandboxd"),
        egress: Vec::<EgressHost>::new(),
        cpus: None,
        memory_mib: None,
    };
    MicrosandboxProvider::new(fake.msb().await, config).expect("valid config")
}

fn command(command: &str, timeout: Duration) -> (SandboxStream, tokio::io::DuplexStream) {
    let (stdout, captured) = tokio::io::duplex(64 * 1024);
    let request = SandboxStream {
        command: command.to_string(),
        workdir: None,
        timeout,
        stdin: Some(Box::pin(&b"from-stdin"[..])),
        stdout: Box::pin(stdout),
    };
    (request, captured)
}

#[tokio::test]
async fn connect_starts_sandboxd_on_the_workspace_with_the_file_cache_refresh() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("deny"));
    let provider = provider(&fake).await;

    let mut stream = provider.connect(NAME, "/workspace").await.expect("connect");
    stream.writer.write_all(b"ping\n").await.expect("write");
    let mut echoed = [0; 5];
    stream.reader.read_exact(&mut echoed).await.expect("read");

    assert_eq!(&echoed, b"ping\n");
    assert_eq!(
        fake.calls().last().map(String::as_str),
        Some(
            "exec --quiet --stream --env MIRROR_WORKSPACE=/workspace \
             --env MIRROR_REFRESH_FILE_CACHE=1 mirror-ws-test -- \
             /usr/local/bin/mirror-sandboxd serve"
        )
    );
}

#[tokio::test]
async fn connect_refuses_a_stopped_vm_rather_than_booting_a_throwaway_one() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Stopped", &document("deny"));
    let provider = provider(&fake).await;

    assert_matches!(
        provider.connect(NAME, "/workspace").await,
        Err(SandboxTransportError::Suspended(_))
    );
    assert!(!fake.calls().iter().any(|call| call.starts_with("exec ")));
}

#[tokio::test]
async fn connect_refuses_a_vm_whose_network_policy_changed() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("allow"));
    let provider = provider(&fake).await;

    assert_matches!(
        provider.connect(NAME, "/workspace").await,
        Err(SandboxTransportError::Failed(detail)) if detail.contains("does not deny network traffic")
    );
}

#[tokio::test]
async fn connect_refuses_a_workspace_the_vm_does_not_mount() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("deny"));
    let provider = provider(&fake).await;

    assert_matches!(
        provider.connect(NAME, "/").await,
        Err(SandboxTransportError::Failed(detail)) if detail.contains("not the workspace \"/\"")
    );
    assert!(!fake.calls().iter().any(|call| call.starts_with("exec ")));
}

#[tokio::test]
async fn connect_to_a_missing_vm_reports_it_gone() {
    let fake = FakeMsb::new();
    let provider = provider(&fake).await;

    assert_matches!(
        provider.connect(NAME, "/workspace").await,
        Err(SandboxTransportError::Gone(_))
    );
}

#[tokio::test]
async fn stream_returns_the_exit_code_output_and_stderr() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("deny"));
    let provider = provider(&fake).await;
    let (request, mut captured) = command(
        "printf 'out:'; cat; printf problem >&2; exit 3",
        Duration::from_secs(30),
    );

    let outcome = provider.stream(NAME, request).await.expect("stream");
    let mut stdout = String::new();
    captured.read_to_string(&mut stdout).await.expect("stdout");

    assert_eq!(
        outcome,
        SandboxStreamOutcome {
            exit_code: 3,
            stderr: b"problem".to_vec(),
        }
    );
    assert_eq!(stdout, "out:from-stdin");
}

#[tokio::test]
async fn stream_returns_once_the_command_exits_even_if_its_input_never_ends() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("deny"));
    let provider = provider(&fake).await;
    let (mut request, _captured) = command("exit 4", Duration::from_secs(30));
    let (_never_written, endless) = tokio::io::duplex(64);
    request.stdin = Some(Box::pin(endless));

    let outcome = tokio::time::timeout(Duration::from_secs(10), provider.stream(NAME, request))
        .await
        .expect("stream returns before the timeout");

    assert_matches!(outcome, Ok(SandboxStreamOutcome { exit_code: 4, .. }));
}

#[tokio::test]
async fn stream_rounds_a_fractional_timeout_up_for_the_guest() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("deny"));
    let provider = provider(&fake).await;
    let (request, _captured) = command("exit 0", Duration::from_millis(1500));

    provider.stream(NAME, request).await.expect("stream");

    assert_matches!(
        fake.calls().last(),
        Some(call) if call.starts_with("exec --quiet --stream --timeout 2s ")
    );
}

#[tokio::test]
async fn stream_past_its_timeout_has_an_unknown_outcome() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document("deny"));
    let provider = provider(&fake).await;
    let (request, _captured) = command("exec sleep 30", Duration::from_millis(200));

    assert_matches!(
        provider.stream(NAME, request).await,
        Err(SandboxTransportError::OutcomeUnknown(_))
    );
}
