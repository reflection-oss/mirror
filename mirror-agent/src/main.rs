mod http;
mod memory_command;

use std::error::Error;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use mirror_agent::rpc::startup_responses;
use prost::Message;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use mirror_agent::agent_rpc_server::AgentRpcServer;
use mirror_agent::local::LocalMirrorRpc;
use mirror_agent::local::LocalSandbox;
use mirror_agent::local::LocalSandboxIdentity;
use mirror_agent::local::LocalWorkspace;
use mirror_agent::local::SQLiteStore;
use mirror_agent::mcp::{discover_mcp_connections, mcp_tools, LocalMcpConnection};
use mirror_agent::memory::open_memory;
use mirror_agent::memory::MemoryEnvironment;
use mirror_agent::memory::SaveMemoryTool;
use mirror_agent::memory::SearchMemoryTool;
use mirror_agent::memory::SqliteMemoryDatabase;
use mirror_agent::memory::SqliteMemoryStore;
use mirror_agent::mirror::mirror_rpc_server::MirrorRpcServer;
use mirror_agent::providers::ModelApi;
use mirror_agent::providers::ModelProvider;
use mirror_agent::providers::ModelProviderConfig;
use mirror_agent::react::ContextHooks;
use mirror_agent::react::ReactAgent;
use mirror_agent::react::ReactAgentConfig;
use mirror_agent::system_prompt::gather_environment_info;
use mirror_agent::system_prompt::pinned_memories_path;
use mirror_agent::system_prompt::resolve_prompt_instructions;
use mirror_agent::system_prompt::NestedInstructions;
use mirror_agent::system_prompt::Skill;
use mirror_agent::system_prompt::SystemPromptConfig;
use mirror_agent::tools::validate_tools;
use mirror_agent::tools::AnyTool;
use mirror_agent::tools::DirectToolExecutor;
use mirror_agent::tools::FirecrawlClient;
use mirror_agent::tools::SandboxRoutedTool;
use mirror_agent::tools::SubagentTool;
use mirror_agent::tools::Task;
use mirror_agent::tools::WebBrowseTool;
use mirror_agent::tools::WebSearchTool;
use mirror_agent::tools::WORKSPACE_TOOL_NAMES;
use mirror_agent::tools::{native_tool, ToolError};
use mirror_agent::AgentApprovalMode;
use mirror_agent::AgentRpc;
use mirror_agent::HostTool;
use mirror_agent::HostToolConfig;
use mirror_agent::ReactAgentService;
use mirror_sandbox_microsandbox::EgressHost;
use mirror_sandbox_microsandbox::MicrosandboxConfig;
use mirror_sandbox_microsandbox::MicrosandboxProvider;
use mirror_sandbox_microsandbox::Msb;
use serde::Deserialize;
use std::io::Read;
use std::path::Path;
use tokio::io::AsyncWriteExt;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use uuid::Uuid;

#[derive(Deserialize)]
struct WorkerConfig {
    rpc_address: String,
    conversation_id: String,
    model: String,
    system_prompt: Option<String>,
    prompt_injection_policy: Option<String>,
    tools: Option<Vec<String>>,
    compaction_model: Option<String>,
    compaction_model_api: Option<String>,
    compaction_model_base_url: Option<String>,
    api: Option<String>,
    model_api: Option<String>,
    approval_mode: Option<String>,
    max_iterations: Option<usize>,
    compaction_threshold: Option<usize>,
    context_window: Option<usize>,
    max_tokens: Option<usize>,
    model_base_url: Option<String>,
    base_url: Option<String>,
    database: Option<PathBuf>,
    create_session: Option<bool>,
    allow_workspace_mismatch: Option<bool>,
    memory_dir: Option<PathBuf>,
    #[serde(default)]
    include_environment: bool,
    #[serde(default)]
    exit_on_stdin_close: bool,
    checkpoints_enabled: Option<bool>,
    #[serde(default)]
    host_tools: Vec<HostToolConfig>,
    #[serde(default)]
    mcp_connections: Vec<LocalMcpConnection>,
    #[serde(default)]
    additional_skills: Vec<Skill>,
    sandbox: Option<WorkerSandboxConfig>,
}

/// Runs the workspace tools in a microsandbox VM instead of on this machine.
#[derive(Deserialize)]
struct WorkerSandboxConfig {
    /// The same ID finds the same VM across worker restarts.
    id: Uuid,
    /// The OCI image the VM boots.
    image: String,
    /// A Linux `mirror-sandboxd` for this machine's architecture.
    sandboxd: PathBuf,
    /// Hosts the VM may reach over HTTPS; all other traffic is denied.
    #[serde(default)]
    egress: Vec<String>,
    cpus: Option<u8>,
    memory_mib: Option<u32>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    if matches!(
        std::env::args().nth(1).as_deref(),
        Some("serve" | "--help" | "-h" | "--version" | "-V")
    ) {
        return match http::run().await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        };
    }
    if std::env::args().nth(1).as_deref() == Some("memory") {
        return match memory_command::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        };
    }
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let message = format!(
                "MIRROR_RPC_ERROR={error}
"
            );
            let mut stdout = tokio::io::stdout();
            let _ = stdout.write_all(message.as_bytes()).await;
            let _ = stdout.flush().await;
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let config = worker_config()?;
    let workspace = std::env::var_os("MIRROR_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    let sandbox = match &config.sandbox {
        Some(sandbox) => Some(start_sandbox(sandbox, &config.conversation_id, &workspace).await?),
        None => None,
    };
    let environment = match (&sandbox, config.include_environment) {
        (_, false) => None,
        (Some(sandbox), true) => Some(sandbox.environment(&workspace).await?),
        (None, true) => Some(gather_environment_info(&workspace)),
    };
    let firecrawl = FirecrawlClient::from_env()?;
    let mut api = config
        .model_api
        .as_deref()
        .or(config.api.as_deref())
        .unwrap_or("openai-codex-responses")
        .parse()?;
    if config.model_base_url.is_some() && api == ModelApi::CodexResponses {
        api = ModelApi::ChatCompletions;
    }
    let provider_config = ModelProviderConfig {
        context_window: config.context_window,
        api,
        model: config.model.clone(),
        api_key: None,
        base_url: config.model_base_url.clone().or(config.base_url.clone()),
        auth_path: std::env::var_os("MIRROR_CODEX_AUTH_PATH").map(Into::into),
        prompt_cache_key: config.conversation_id.clone(),
        request_timeout: Some(mirror_agent::providers::DEFAULT_REQUEST_TIMEOUT),
    };
    let model = ModelProvider::new(provider_config.clone(), None).await?;
    let agent_config = ReactAgentConfig {
        max_iterations: config.max_iterations.unwrap_or(usize::MAX),
        compaction_threshold: config.compaction_threshold,
        max_tokens: config.max_tokens.unwrap_or(64_000),
        ..ReactAgentConfig::default()
    };
    let mut tools = WORKSPACE_TOOL_NAMES
        .into_iter()
        .map(|name| {
            let tool = native_tool(name, &workspace)?;
            Ok(match &sandbox {
                Some(sandbox) => {
                    Arc::new(SandboxRoutedTool::new(tool, sandbox.clone())) as Arc<dyn AnyTool>
                }
                None => tool,
            })
        })
        .collect::<Result<Vec<_>, ToolError>>()?;
    let subagent_tools = tools
        .iter()
        .filter(|tool| {
            config
                .tools
                .as_ref()
                .is_none_or(|names| names.iter().any(|name| name == tool.name()))
        })
        .cloned()
        .collect();
    let subagent = SubagentTool::new(
        &workspace,
        &config.conversation_id,
        provider_config.clone(),
        agent_config,
        environment.clone(),
        subagent_tools,
    )?;
    tools.push(Arc::new(Task));
    let has_web_tools = firecrawl.is_configured();
    if has_web_tools {
        tools.push(Arc::new(WebSearchTool::new(firecrawl.clone())));
        tools.push(Arc::new(WebBrowseTool::new(firecrawl)));
    }
    tools.push(Arc::new(subagent));
    let memory = match &config.memory_dir {
        Some(directory) => open_memory(MemoryEnvironment {
            store: Box::new(SqliteMemoryStore::new(SqliteMemoryDatabase::open(
                directory,
            )?)?),
            workspace: Box::new(LocalWorkspace::new(&workspace)?),
            conversation_id: config.conversation_id.clone(),
            private_session: true,
            pinned_memories: pinned_memories_path()?,
        }),
        None => None,
    };
    let pinned_memories = memory.as_ref().map(|memory| memory.pinned_memories.clone());
    let has_memory = memory.is_some();
    let mut hooks: Vec<Box<dyn ContextHooks>> = vec![Box::new(NestedInstructions::new(&workspace))];
    if let Some(memory) = memory {
        tools.extend(memory.tools);
        hooks.push(memory.hooks);
    }
    let remote_tools = discover_mcp_connections(&config.mcp_connections).await;
    let mcp = mcp_tools(remote_tools)?;
    let has_mcp = !mcp.is_empty();
    tools.extend(mcp);
    if let Some(names) = &config.tools {
        let memory_tools = [SearchMemoryTool::NAME, SaveMemoryTool::NAME];
        tools = names
            .iter()
            .filter(|name| has_memory || !memory_tools.contains(&name.as_str()))
            .filter(|name| {
                has_mcp || !matches!(name.as_str(), "mcp_search" | "mcp_describe" | "mcp_call")
            })
            .filter(|name| has_web_tools || !matches!(name.as_str(), "web_search" | "web_browse"))
            .map(|name| {
                tools
                    .iter()
                    .find(|tool| tool.name() == name)
                    .cloned()
                    .ok_or_else(|| ToolError::InvalidArguments(format!("unknown tool {name:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    let host_tool_requests = if config.host_tools.is_empty() {
        None
    } else {
        let (requests, receiver) = tokio::sync::mpsc::unbounded_channel();
        for config in config.host_tools {
            tools.push(Arc::new(HostTool::new(config, requests.clone())?));
        }
        Some(receiver)
    };
    let compaction_model = if let Some(model) = config.compaction_model {
        let base_url = config
            .compaction_model_base_url
            .clone()
            .or_else(|| provider_config.base_url.clone());
        let mut api = config
            .compaction_model_api
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or(provider_config.api);
        if base_url.is_some() && api == ModelApi::CodexResponses {
            api = ModelApi::ChatCompletions;
        }
        Some(
            ModelProvider::new(
                ModelProviderConfig {
                    api,
                    model,
                    base_url,
                    ..provider_config.clone()
                },
                None,
            )
            .await?,
        )
    } else {
        None
    };
    validate_tools(tools.iter().map(|tool| tool.as_ref()))?;
    let prompt_config = SystemPromptConfig {
        custom_prompt: config.system_prompt,
        prompt_injection_policy: config.prompt_injection_policy,
        working_directory: Some(workspace.clone()),
        environment,
        pinned_memories,
        additional_skills: config.additional_skills,
        ..SystemPromptConfig::default()
    };
    let skills = resolve_prompt_instructions(&prompt_config).skills;
    let tool_executor = DirectToolExecutor::new(tools);
    let agent = ReactAgent::new(
        model.clone(),
        model,
        compaction_model,
        &tool_executor,
        prompt_config,
        agent_config,
    )?;
    let approval_mode = match config.approval_mode.as_deref() {
        Some("approve") => AgentApprovalMode::Hitl,
        _ => AgentApprovalMode::Auto,
    };
    let database = config.database.unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace.clone())
            .join(".mirror/state.db")
    });
    let mut store = SQLiteStore::open(
        &database,
        &config.conversation_id,
        &workspace,
        config.create_session.unwrap_or(false),
        config.allow_workspace_mismatch.unwrap_or(false),
    )?;
    store.checkpoints_enabled = config.checkpoints_enabled.unwrap_or(false);
    let (events, mut startup_events) = tokio::sync::broadcast::channel(1024);
    let startup = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        config.conversation_id,
        approval_mode,
        skills,
        hooks,
        host_tool_requests,
        events,
        None,
    )
    .await;
    let mut stdout = tokio::io::stdout();
    while let Ok(event) = startup_events.try_recv() {
        for response in startup_responses(event) {
            let encoded = STANDARD.encode(response.encode_to_vec());
            stdout
                .write_all(format!("MIRROR_SESSION_MESSAGE={encoded}\n").as_bytes())
                .await?;
        }
    }
    stdout.flush().await?;
    let (service, runtime) = startup?;
    tokio::spawn(runtime);
    let service_control = service.clone();
    let mirror_rpc = LocalMirrorRpc::new(service.clone(), database);
    let rpc = AgentRpc::new(service);
    let rpc_control = rpc.clone();
    let listener = tokio::net::TcpListener::bind(&config.rpc_address).await?;
    let address = listener.local_addr()?;
    let mut stdout = tokio::io::stdout();
    stdout
        .write_all(format!("MIRROR_RPC_ADDRESS={address}\n").as_bytes())
        .await?;
    stdout.flush().await?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let parent_closed = config.exit_on_stdin_close.then(stdin_closed);
    let shutdown = async move {
        tokio::select! {
            _ = terminate.recv() => {}
            _ = interrupt.recv() => {}
            _ = async {
                match parent_closed {
                    Some(closed) => {
                        let _ = closed.await;
                    }
                    None => std::future::pending().await,
                }
            } => {}
        }
        rpc_control.shutdown();
    };
    let serve_result = Server::builder()
        .add_service(
            AgentRpcServer::new(rpc)
                .max_decoding_message_size(10 * 1024 * 1024)
                .max_encoding_message_size(10 * 1024 * 1024),
        )
        .add_service(MirrorRpcServer::new(mirror_rpc))
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
        .await;
    let close_result = service_control.close().await;
    serve_result?;
    close_result?;
    Ok(())
}

fn stdin_closed() -> tokio::sync::oneshot::Receiver<()> {
    let (closed, receiver) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 256];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) => break,
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let _ = closed.send(());
    });
    receiver
}

/// Starts the sandbox the workspace tools run in. The VM mounts `workspace` at
/// the same path, so a tool path names the same file on both sides.
async fn start_sandbox(
    config: &WorkerSandboxConfig,
    conversation_id: &str,
    workspace: &Path,
) -> Result<Arc<LocalSandbox<MicrosandboxProvider>>, Box<dyn Error>> {
    let session_id = Uuid::parse_str(conversation_id).map_err(|_| {
        format!("a sandboxed conversation needs a UUID conversation ID, not {conversation_id:?}")
    })?;
    let workspace = std::fs::canonicalize(workspace)?;
    let guest_workspace = workspace
        .to_str()
        .ok_or_else(|| format!("workspace {workspace:?} is not UTF-8"))?
        .to_string();
    let egress = config
        .egress
        .iter()
        .map(|host| EgressHost::parse(host))
        .collect::<Result<Vec<_>, _>>()?;
    let provider = MicrosandboxProvider::new(
        Msb::locate().await?,
        MicrosandboxConfig {
            image: config.image.clone(),
            workspace,
            sandboxd: config.sandboxd.clone(),
            egress,
            cpus: config.cpus,
            memory_mib: config.memory_mib,
        },
    )?;
    let identity = LocalSandboxIdentity {
        request_id: config.id,
        session_id,
    };
    Ok(Arc::new(LocalSandbox::start(
        Arc::new(provider),
        identity,
        guest_workspace,
    )))
}

fn worker_config() -> Result<WorkerConfig, Box<dyn Error>> {
    if let Some(raw) = std::env::args().nth(1) {
        return Ok(serde_json::from_str(&raw)?);
    }
    Ok(WorkerConfig {
        rpc_address: std::env::var("MIRROR_RPC_ADDRESS")
            .unwrap_or_else(|_| "127.0.0.1:0".to_string()),
        conversation_id: std::env::var("MIRROR_CONVERSATION_ID")
            .unwrap_or_else(|_| "default".to_string()),
        model: std::env::var("MIRROR_MODEL").unwrap_or_else(|_| "gpt-6-astra".to_string()),
        system_prompt: None,
        prompt_injection_policy: None,
        tools: None,
        compaction_model: None,
        compaction_model_api: None,
        compaction_model_base_url: None,
        api: Some("openai-codex-responses".to_string()),
        model_api: None,
        approval_mode: None,
        max_iterations: None,
        compaction_threshold: None,
        context_window: None,
        max_tokens: None,
        model_base_url: None,
        base_url: None,
        database: None,
        create_session: Some(true),
        allow_workspace_mismatch: Some(false),
        memory_dir: None,
        include_environment: false,
        exit_on_stdin_close: false,
        checkpoints_enabled: None,
        host_tools: Vec::new(),
        mcp_connections: Vec::new(),
        additional_skills: Vec::new(),
        sandbox: None,
    })
}
