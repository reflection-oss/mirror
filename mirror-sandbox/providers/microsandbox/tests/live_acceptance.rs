//! Runs workspace tools in a real microVM through `sandboxd`.
//!
//! Needs microsandbox's `msb` (on `PATH` or at `$MIRROR_MSB`) and a host that
//! can run microVMs. Set `MIRROR_LIVE_SANDBOXD` to a Linux `mirror-sandboxd`
//! for the host's architecture; `MIRROR_LIVE_IMAGE` overrides the image.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use mirror_protocol::chat_record;
use mirror_protocol::FunctionCall;
use mirror_protocol::ToolCall;
use mirror_sandbox::CallOutcome;
use mirror_sandbox::CallReply;
use mirror_sandbox::SandboxLiveness;
use mirror_sandbox::SandboxProvider;
use mirror_sandbox::SandboxTransport;
use mirror_sandbox::SandboxTransportError;
use mirror_sandbox::SandboxdSession;
use mirror_sandbox_microsandbox::MicrosandboxConfig;
use mirror_sandbox_microsandbox::MicrosandboxProvider;
use mirror_sandbox_microsandbox::Msb;
use tokio::sync::watch;
use uuid::Uuid;

const CALL_TIMEOUT: Duration = Duration::from_secs(60);

#[tokio::test]
#[ignore = "boots microVMs; needs msb and MIRROR_LIVE_SANDBOXD"]
async fn workspace_tools_run_in_the_vm_and_see_host_edits_at_once() {
    let sandboxd = PathBuf::from(
        std::env::var_os("MIRROR_LIVE_SANDBOXD")
            .expect("MIRROR_LIVE_SANDBOXD names a Linux mirror-sandboxd for this host"),
    );
    let image =
        std::env::var("MIRROR_LIVE_IMAGE").unwrap_or_else(|_| "debian:bookworm-slim".to_string());
    let workspace = tempfile::tempdir().expect("workspace");
    let root = std::fs::canonicalize(workspace.path()).expect("canonical workspace");
    let msb = Msb::locate().await.expect("msb is installed and usable");
    let provider = MicrosandboxProvider::new(
        msb,
        MicrosandboxConfig {
            image,
            workspace: root.clone(),
            sandboxd,
            egress: Vec::new(),
            cpus: Some(1),
            memory_mib: Some(512),
        },
    )
    .expect("valid config");
    let name = provider
        .create(Uuid::new_v4(), Uuid::new_v4())
        .await
        .expect("create");

    let exercised = exercise(&provider, &name, &root).await;
    provider.delete(&name).await.expect("delete");
    provider
        .delete(&name)
        .await
        .expect("deleting again succeeds");

    exercised.expect("the sandbox behaves");
    assert_eq!(
        provider.liveness(&name).await.expect("liveness"),
        SandboxLiveness::Gone
    );
}

async fn exercise(provider: &MicrosandboxProvider, name: &str, root: &Path) -> Result<(), String> {
    provider
        .wait_until_ready(name)
        .await
        .map_err(|error| error.to_string())?;
    let workspace = root.to_str().ok_or("workspace path is not UTF-8")?;
    let session = open_session(provider, name, workspace).await?;

    bash(&session, "printf from-vm > made-in-vm.txt").await?;
    let made = std::fs::read_to_string(root.join("made-in-vm.txt")).map_err(|e| e.to_string())?;
    expect_eq("file the guest wrote", &made, "from-vm")?;

    std::fs::write(root.join("shared.txt"), "host-one").map_err(|e| e.to_string())?;
    expect_contains(
        "first host edit",
        &bash(&session, "cat shared.txt").await?,
        "host-one",
    )?;
    std::fs::write(root.join("replacement.txt"), "host-two").map_err(|e| e.to_string())?;
    std::fs::rename(root.join("replacement.txt"), root.join("shared.txt"))
        .map_err(|e| e.to_string())?;
    expect_contains(
        "renamed host edit",
        &bash(&session, "cat shared.txt").await?,
        "host-two",
    )?;
    session.close().await;

    provider.suspend(name).await.map_err(|e| e.to_string())?;
    provider
        .wait_until_suspended(name)
        .await
        .map_err(|e| e.to_string())?;
    match provider.connect(name, workspace).await {
        Err(SandboxTransportError::Suspended(_)) => {}
        Err(other) => return Err(format!("connecting to a stopped VM failed with {other}")),
        Ok(_) => return Err("connecting to a stopped VM succeeded".to_string()),
    }
    provider.resume(name).await.map_err(|e| e.to_string())?;
    provider
        .wait_until_ready(name)
        .await
        .map_err(|e| e.to_string())?;
    let session = open_session(provider, name, workspace).await?;
    expect_contains(
        "workspace after resume",
        &bash(&session, "cat made-in-vm.txt").await?,
        "from-vm",
    )?;
    session.close().await;
    Ok(())
}

async fn open_session(
    provider: &MicrosandboxProvider,
    name: &str,
    workspace: &str,
) -> Result<SandboxdSession, String> {
    let stream = provider
        .connect(name, workspace)
        .await
        .map_err(|e| e.to_string())?;
    SandboxdSession::open(stream, watch::channel(false).0)
        .await
        .map_err(|e| e.to_string())
}

/// Runs `command` with the `bash` tool and returns the tool's output.
async fn bash(session: &SandboxdSession, command: &str) -> Result<String, String> {
    let call = ToolCall {
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: "bash".to_string(),
            arguments: serde_json::json!({ "command": command }).to_string(),
        }),
        ..ToolCall::default()
    };
    let reply = session
        .execute(Uuid::new_v4().to_string(), call, CALL_TIMEOUT, 64 * 1024)
        .await
        .map_err(|e| e.to_string())?
        .await
        .map_err(|e| e.to_string())?;
    let record = match reply {
        CallReply::Outcome(CallOutcome::Completed(record)) => *record,
        other => return Err(format!("{command:?} did not complete: {other:?}")),
    };
    match record.message {
        Some(chat_record::Message::Tool(message)) if record.success => Ok(message.content),
        other => Err(format!("{command:?} failed: {other:?}")),
    }
}

fn expect_eq(what: &str, actual: &str, expected: &str) -> Result<(), String> {
    if actual == expected {
        return Ok(());
    }
    Err(format!("{what} was {actual:?}, not {expected:?}"))
}

fn expect_contains(what: &str, actual: &str, expected: &str) -> Result<(), String> {
    if actual.contains(expected) {
        return Ok(());
    }
    Err(format!("{what} was {actual:?}, without {expected:?}"))
}
