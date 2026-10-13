use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use mirror_agent::agent_message;
use mirror_agent::agent_rpc_client::AgentRpcClient;
use mirror_agent::GetMessagesRequest;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Command;

#[tokio::test]
async fn worker_describes_its_workspace_when_the_client_asks() {
    let workspace = tempfile::tempdir().expect("workspace");

    let prompt = initial_context(
        workspace.path(),
        serde_json::json!({"include_environment": true}),
        None,
    )
    .await
    .system_prompt;

    let header = format!(
        "# Environment\n- Working directory: {}\n- Git repository: no\n",
        workspace.path().display()
    );
    assert!(prompt.starts_with(&header), "{prompt}");
    assert!(prompt.contains("\n- Shell: bash "), "{prompt}");
}

#[tokio::test]
async fn worker_adds_no_environment_section_unless_asked() {
    let workspace = tempfile::tempdir().expect("workspace");

    let prompt = initial_context(workspace.path(), serde_json::json!({}), None)
        .await
        .system_prompt;

    assert!(!prompt.contains("# Environment"), "{prompt}");
}

#[tokio::test]
async fn worker_omits_unconfigured_web_tools_by_default() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(workspace.path(), serde_json::json!({}), None).await;
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
        serde_json::json!({"tools": ["bash", "web_search", "web_browse"]}),
        None,
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
        serde_json::json!({"tools": ["bash", "web_search", "web_browse"]}),
        Some(" \t "),
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
async fn worker_offers_web_tools_with_a_configured_key() {
    let workspace = tempfile::tempdir().expect("workspace");
    let context = initial_context(
        workspace.path(),
        serde_json::json!({"tools": ["bash", "web_search", "web_browse"]}),
        Some("synthetic-firecrawl-key"),
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

async fn initial_context(
    workspace: &Path,
    overrides: serde_json::Value,
    firecrawl_key: Option<&str>,
) -> mirror_agent::ContextWindowRecord {
    let auth = workspace.join("auth.json");
    std::fs::write(
        &auth,
        r#"{"tokens":{"access_token":"test","account_id":"test"}}"#,
    )
    .expect("write credentials");
    let mut config = serde_json::json!({
        "rpc_address": "127.0.0.1:0",
        "conversation_id": "session",
        "model": "test",
        "database": workspace.join("state.db"),
        "create_session": true,
    });
    config
        .as_object_mut()
        .expect("config object")
        .extend(overrides.as_object().expect("override object").clone());
    let mut command = Command::new(worker_binary());
    command.env_remove("FIRECRAWL_API_KEY");
    if let Some(key) = firecrawl_key {
        command.env("FIRECRAWL_API_KEY", key);
    }
    let mut child = command
        .arg(config.to_string())
        .env("MIRROR_WORKSPACE", workspace)
        .env("MIRROR_CODEX_AUTH_PATH", auth)
        .env("HOME", workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("start worker");
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
    let mut client = AgentRpcClient::connect(format!("http://{address}"))
        .await
        .expect("connect to worker");
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

fn worker_binary() -> PathBuf {
    std::env::var_os("MIRROR_AGENT_BIN")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_mirror-agent").map(PathBuf::from))
        .expect("worker binary path")
}
