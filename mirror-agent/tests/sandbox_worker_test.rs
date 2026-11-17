//! Runs a worker whose workspace tools run in a microsandbox VM.
//!
//! Needs microsandbox's `msb` (on `PATH` or at `$MIRROR_MSB`) and a host that
//! can run microVMs. Set `MIRROR_LIVE_SANDBOXD` to a Linux `mirror-sandboxd`
//! for the host's architecture; `MIRROR_LIVE_IMAGE` overrides the image.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use hyper_util::rt::TokioIo;
use mirror_agent::agent_message;
use mirror_agent::agent_rpc_client::AgentRpcClient;
use mirror_agent::daemon::{AgentConfiguration, SandboxSource};
use mirror_agent::CallToolRequest;
use mirror_agent::{
    ApprovalMode, GetMessagesRequest, InferenceConfig, WorkerSandbox, WorkerStartup,
};
use mirror_sandbox::SandboxProvider;
use mirror_sandbox_microsandbox::sandbox_name;
use mirror_sandbox_microsandbox::MicrosandboxConfig;
use mirror_sandbox_microsandbox::MicrosandboxProvider;
use mirror_sandbox_microsandbox::Msb;
use prost::Message;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::net::UnixStream;
use tokio::process::Command;
use tonic::transport::Endpoint;
use tower::service_fn;
use uuid::Uuid;

#[tokio::test]
#[ignore = "boots a microVM; needs msb and MIRROR_LIVE_SANDBOXD"]
async fn sandboxed_worker_runs_bash_in_the_vm_and_describes_it() {
    let sandboxd = PathBuf::from(
        std::env::var_os("MIRROR_LIVE_SANDBOXD")
            .expect("MIRROR_LIVE_SANDBOXD names a Linux mirror-sandboxd for this host"),
    );
    let image =
        std::env::var("MIRROR_LIVE_IMAGE").unwrap_or_else(|_| "debian:bookworm-slim".to_string());
    let workspace = tempfile::tempdir().expect("workspace");
    let root = std::fs::canonicalize(workspace.path()).expect("canonical workspace");
    let auth = root.join("auth.json");
    std::fs::write(
        &auth,
        r#"{"tokens":{"access_token":"test","account_id":"test"}}"#,
    )
    .expect("write credentials");
    let sandbox_id = Uuid::new_v4();
    let conversation_id = Uuid::new_v4().to_string();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700))
        .expect("private socket directory");
    let config = WorkerStartup {
        configuration: Some(AgentConfiguration {
            inference: Some(InferenceConfig {
                model: "test".into(),
                api: Some("openai-codex-responses".into()),
                ..Default::default()
            }),
            auth_path: Some(auth.to_str().expect("auth path").into()),
            approval_mode: ApprovalMode::Auto.into(),
            include_environment: Some(true),
            ..Default::default()
        }),
        rpc_address: format!("unix:{}", root.join("agent.sock").display()),
        session_id: conversation_id.clone(),
        database: root
            .join("state.db")
            .to_str()
            .expect("database path")
            .into(),
        create_session: true,
        sandbox: Some(WorkerSandbox {
            id: sandbox_id.to_string(),
            sandboxd: sandboxd.to_str().expect("sandboxd path").into(),
            source: Some(SandboxSource {
                image,
                cpus: 1,
                memory_mib: 512,
                ..Default::default()
            }),
        }),
        ..Default::default()
    }
    .encode_to_vec();
    let mut worker = Command::new(worker_binary())
        .arg("--config-stdin")
        .env("MIRROR_WORKSPACE", &root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .expect("start worker");
    let mut stdin = worker.stdin.take().expect("worker stdin");
    stdin
        .write_all(&(config.len() as u32).to_be_bytes())
        .await
        .expect("startup size");
    stdin.write_all(&config).await.expect("startup frame");
    drop(stdin);
    let mut lines = BufReader::new(worker.stdout.take().expect("worker stdout")).lines();
    let address = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            let line = lines
                .next_line()
                .await
                .expect("read worker output")
                .expect("worker exited before publishing its address");
            if let Some(address) = line.strip_prefix("MIRROR_RPC_ADDRESS=") {
                break address.to_string();
            }
        }
    })
    .await
    .expect("worker startup deadline");
    let socket = address
        .strip_prefix("unix:")
        .expect("Unix worker address")
        .to_owned();
    let channel = Endpoint::from_static("http://localhost")
        .connect_with_connector(service_fn(move |_| {
            let socket = socket.clone();
            async move { UnixStream::connect(socket).await.map(TokioIo::new) }
        }))
        .await
        .expect("connect to worker");
    let mut client = AgentRpcClient::new(channel);

    let context = client
        .get_messages(GetMessagesRequest {
            session_id: conversation_id.clone(),
            offset: 0,
            limit: 1,
        })
        .await
        .expect("read history")
        .into_inner()
        .messages
        .into_iter()
        .next()
        .and_then(|message| message.payload);
    let ran = client
        .call_tool(CallToolRequest {
            session_id: conversation_id,
            name: "bash".to_string(),
            arguments: r#"{"command":"uname -s > made-in-vm.txt"}"#.to_string(),
            ..Default::default()
        })
        .await
        .expect("call bash")
        .into_inner();
    worker.kill().await.expect("stop worker");
    remove_sandbox(sandbox_id, root.clone(), sandboxd).await;

    let Some(agent_message::Payload::ContextWindow(window)) = context else {
        panic!("first record is not a context window: {context:?}");
    };
    assert!(
        window.system_prompt.contains("\n- Platform: Linux\n"),
        "{}",
        window.system_prompt
    );
    assert!(ran.success, "{}", ran.content);
    assert_eq!(
        std::fs::read_to_string(root.join("made-in-vm.txt")).expect("file the VM wrote"),
        "Linux\n"
    );
}

/// The VM outlives the worker by design, so the test removes it.
async fn remove_sandbox(sandbox_id: Uuid, workspace: PathBuf, sandboxd: PathBuf) {
    let provider = MicrosandboxProvider::new(
        Msb::locate().await.expect("msb"),
        MicrosandboxConfig {
            image: "unused".to_string(),
            workspace,
            sandboxd,
            egress: Vec::new(),
            cpus: None,
            memory_mib: None,
        },
    )
    .expect("provider");
    provider
        .delete(&sandbox_name(sandbox_id))
        .await
        .expect("delete the VM");
}

fn worker_binary() -> PathBuf {
    std::env::var_os("MIRROR_AGENT_BIN")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_mirror-agent").map(PathBuf::from))
        .expect("worker binary path")
}
