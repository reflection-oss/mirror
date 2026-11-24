use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use hyper_util::rt::TokioIo;
use mirror_agent::agent_message;
use mirror_agent::agent_rpc_client::AgentRpcClient;
use mirror_agent::daemon::{AgentConfiguration, WebCredentials};
use mirror_agent::mirror::agent_definition::ToolAllowlist;
use mirror_agent::{ApprovalMode, GetMessagesRequest, InferenceConfig, WorkerStartup};
use prost::Message;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::net::UnixStream;
use tokio::process::Command;
use tonic::transport::Endpoint;
use tower::service_fn;

#[tokio::test]
async fn worker_describes_its_workspace_when_the_client_asks() {
    let workspace = tempfile::tempdir().expect("workspace");

    let initialized = std::process::Command::new("git")
        .current_dir(workspace.path())
        .args(["init", "-b", "feature"])
        .output()
        .expect("initialize repository");
    assert!(initialized.status.success());

    let prompt = initial_context(
        workspace.path(),
        AgentConfiguration {
            include_environment: Some(true),
            ..Default::default()
        },
        WebCredentials::default(),
    )
    .await
    .system_prompt;

    let header = format!(
        "# Environment\n- Working directory: {}\n- Git repository: yes\n",
        workspace.path().display()
    );
    assert!(prompt.starts_with(&header), "{prompt}");
    assert!(prompt.contains("\n- Shell: bash "), "{prompt}");
    assert!(prompt.contains("Git worktree: no"), "{prompt}");
    assert!(prompt.contains("Current branch: feature"), "{prompt}");
    assert!(prompt.contains("-1000"), "{prompt}");
    assert!(prompt.contains("Status:\n??"), "{prompt}");
}

#[tokio::test]
async fn worker_loads_context_without_an_environment_section_unless_asked() {
    let workspace = tempfile::tempdir().expect("workspace");
    std::fs::write(workspace.path().join("AGENTS.md"), "Workspace instruction.")
        .expect("instructions");
    let skill = workspace.path().join(".agents/skills/example");
    std::fs::create_dir_all(&skill).expect("skill directory");
    std::fs::write(
        skill.join("SKILL.md"),
        "---\nname: example\ndescription: Workspace skill.\n---\nRead files.",
    )
    .expect("skill");

    let prompt = initial_context(
        workspace.path(),
        AgentConfiguration::default(),
        WebCredentials::default(),
    )
    .await
    .system_prompt;

    assert!(!prompt.contains("# Environment"), "{prompt}");
    assert!(prompt.contains("Workspace instruction."), "{prompt}");
    assert!(prompt.contains("Workspace skill."), "{prompt}");
}

#[tokio::test]
async fn worker_omits_unconfigured_web_tools_by_default() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        AgentConfiguration::default(),
        WebCredentials::default(),
    )
    .await;
    let tools = context.system_prompt_report.expect("prompt report").tools;
    assert!(tools.iter().any(|tool| tool.source == "bash"));
    assert!(!tools
        .iter()
        .any(|tool| tool.source == "web_search" || tool.source == "web_browse"));
}

#[tokio::test]
async fn worker_skips_unconfigured_web_tools_in_an_allowlist() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        AgentConfiguration {
            tools: Some(ToolAllowlist {
                names: vec!["bash".into(), "web_search".into(), "web_browse".into()],
            }),
            web_provider: Some("tavily".into()),
            ..Default::default()
        },
        WebCredentials {
            you_api_key: Some("synthetic-you-key".into()),
            ..Default::default()
        },
    )
    .await;
    let tools = context.system_prompt_report.expect("prompt report").tools;
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.source.as_str())
            .collect::<Vec<_>>(),
        ["bash"]
    );
}

#[tokio::test]
async fn worker_omits_web_tools_with_a_blank_key() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        AgentConfiguration {
            tools: Some(ToolAllowlist {
                names: vec!["bash".into(), "web_search".into(), "web_browse".into()],
            }),
            web_provider: Some("you".into()),
            ..Default::default()
        },
        WebCredentials {
            you_api_key: Some(" \t ".into()),
            ..Default::default()
        },
    )
    .await;
    let tools = context.system_prompt_report.expect("prompt report").tools;
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.source.as_str())
            .collect::<Vec<_>>(),
        ["bash"]
    );
}

#[tokio::test]
async fn worker_offers_web_tools_with_a_you_key() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        AgentConfiguration {
            tools: Some(ToolAllowlist {
                names: vec!["bash".into(), "web_search".into(), "web_browse".into()],
            }),
            web_provider: Some("you".into()),
            ..Default::default()
        },
        WebCredentials {
            you_api_key: Some("synthetic-you-key".into()),
            ..Default::default()
        },
    )
    .await;
    let tools = context.system_prompt_report.expect("prompt report").tools;
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.source.as_str())
            .collect::<Vec<_>>(),
        ["bash", "web_search", "web_browse"]
    );
}

#[tokio::test]
async fn worker_allowlist_selects_search_without_browse() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        AgentConfiguration {
            tools: Some(ToolAllowlist {
                names: vec!["bash".into(), "web_search".into()],
            }),
            ..Default::default()
        },
        WebCredentials {
            you_api_key: Some("synthetic-you-key".into()),
            ..Default::default()
        },
    )
    .await;
    let tools = context.system_prompt_report.expect("prompt report").tools;
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.source.as_str())
            .collect::<Vec<_>>(),
        ["bash", "web_search"]
    );
}

#[tokio::test]
async fn worker_omits_web_tools_when_only_another_provider_has_a_key() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        AgentConfiguration {
            tools: Some(ToolAllowlist {
                names: vec!["bash".into(), "web_search".into(), "web_browse".into()],
            }),
            ..Default::default()
        },
        WebCredentials {
            firecrawl_api_key: Some("synthetic-firecrawl-key".into()),
            ..Default::default()
        },
    )
    .await;
    let tools = context.system_prompt_report.expect("prompt report").tools;
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.source.as_str())
            .collect::<Vec<_>>(),
        ["bash"]
    );
}

#[tokio::test]
async fn pre_tool_hooks_run_without_a_cli_and_do_not_inherit_worker_secrets() {
    let workspace = tempfile::tempdir().expect("workspace");
    std::fs::create_dir(workspace.path().join(".mirror")).expect("config directory");
    std::fs::write(workspace.path().join(".mirror/hooks.json"), r#"{
        "hooks": {"PreToolUse": [{"matcher":"Write", "hooks": [{"type":"command", "command":
            "test -z \"${MIRROR_TEST_SECRET-}${HOME-}\" || exit 1; printf 'Runtime denial.' >&2; exit 2"
        }]}]}
    }"#).expect("user hook");
    let (mut child, mut client) = start_worker(
        workspace.path(),
        AgentConfiguration::default(),
        WebCredentials::default(),
    )
    .await;
    let result = client
        .call_tool(mirror_agent::CallToolRequest {
            session_id: "session".into(),
            name: "write".into(),
            arguments: r#"{"file_path":"blocked","content":"should not be written"}"#.into(),
            ..Default::default()
        })
        .await
        .expect("reviewed call")
        .into_inner();
    assert_eq!(result.content, "Tool error: Runtime denial.");
    assert!(!workspace.path().join("blocked").exists());
    child.kill().await.expect("stop worker");
}

async fn initial_context(
    workspace: &Path,
    configuration: AgentConfiguration,
    web_credentials: WebCredentials,
) -> mirror_agent::ContextWindowRecord {
    let (mut child, mut client) = start_worker(workspace, configuration, web_credentials).await;
    let messages = client
        .get_messages(GetMessagesRequest {
            session_id: "session".into(),
            offset: 0,
            limit: 1,
        })
        .await
        .expect("read history")
        .into_inner()
        .messages;
    child.kill().await.expect("stop worker");
    match messages
        .into_iter()
        .next()
        .and_then(|message| message.payload)
    {
        Some(agent_message::Payload::ContextWindow(window)) => window,
        payload => panic!("first record is not a context window: {payload:?}"),
    }
}

async fn start_worker(
    workspace: &Path,
    configuration: AgentConfiguration,
    web_credentials: WebCredentials,
) -> (
    tokio::process::Child,
    AgentRpcClient<tonic::transport::Channel>,
) {
    let auth = workspace.join("auth.json");
    std::fs::write(
        &auth,
        r#"{"tokens":{"access_token":"test","account_id":"test"}}"#,
    )
    .expect("write credentials");
    std::fs::set_permissions(workspace, std::fs::Permissions::from_mode(0o700))
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
            web_credentials: Some(web_credentials),
            ..configuration
        }),
        rpc_address: format!("unix:{}", workspace.join("agent.sock").display()),
        session_id: "session".into(),
        database: workspace
            .join("state.db")
            .to_str()
            .expect("database path")
            .into(),
        create_session: true,
        ..Default::default()
    }
    .encode_to_vec();
    let mut command = Command::new(worker_binary());
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("MIRROR_TEST_SECRET", "synthetic-worker-secret")
        .env("TZ", "Pacific/Honolulu");
    let mut child = command
        .arg("--config-stdin")
        .env("MIRROR_WORKSPACE", workspace)
        .env("HOME", workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("start worker");
    let mut stdin = child.stdin.take().expect("worker stdin");
    stdin
        .write_all(&(config.len() as u32).to_be_bytes())
        .await
        .expect("frame length");
    stdin.write_all(&config).await.expect("startup frame");
    let mut lines = BufReader::new(child.stdout.take().expect("worker stdout")).lines();
    let address = tokio::time::timeout(Duration::from_secs(10), async {
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
    (child, AgentRpcClient::new(channel))
}

fn worker_binary() -> PathBuf {
    std::env::var_os("MIRROR_AGENT_BIN")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_mirror-agent").map(PathBuf::from))
        .expect("worker binary path")
}
