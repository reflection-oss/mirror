mod http;
mod memory_command;
mod sandbox_command;
mod sandbox_config;

use std::error::Error;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures_core::Stream;
use mirror_agent::compaction::SummaryCompactorFactory;
use mirror_agent::rpc::startup_responses;
use prost::Message;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use mirror_agent::agent_rpc_server::AgentRpcServer;
use mirror_agent::local::LocalMirrorRpc;
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
use mirror_agent::tools::RootSearchGuard;
use mirror_agent::tools::SandboxRoutedTool;
use mirror_agent::tools::SubagentTool;
use mirror_agent::tools::Task;
use mirror_agent::tools::WebBrowseTool;
use mirror_agent::tools::WebSearchTool;
use mirror_agent::tools::WORKSPACE_TOOL_NAMES;
use mirror_agent::tools::{native_tool, ToolError};
use mirror_agent::web::{web_provider_from_config, WebProviderConfig};
use mirror_agent::AgentApprovalMode;
use mirror_agent::AgentRpc;
use mirror_agent::HostTool;
use mirror_agent::{
    agent_message, session_message, AgentMessage, ApprovalMode, LogRecord, ReactAgentService,
    SessionMessage, WorkerStartup,
};
use nix::unistd::geteuid;
use std::io::{self, Read};
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tonic::transport::Server;
use uuid::Uuid;

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
    if std::env::args().nth(1).as_deref() == Some("sandbox") {
        return match sandbox_command::run().await {
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
    let startup = worker_startup()?;
    let socket = startup
        .rpc_address
        .strip_prefix("unix:")
        .ok_or("worker RPC address must be a unix: socket path")?;
    let config = startup
        .configuration
        .ok_or("worker configuration is required")?;
    let inference = config
        .inference
        .as_ref()
        .ok_or("inference configuration is required")?;
    let approval_mode = match ApprovalMode::try_from(config.approval_mode)? {
        ApprovalMode::Auto => AgentApprovalMode::Auto,
        ApprovalMode::Bypass => AgentApprovalMode::Bypass,
        ApprovalMode::Hitl => AgentApprovalMode::Hitl,
        ApprovalMode::Unspecified => return Err("approval mode is required".into()),
    };
    let mcp_connections: Vec<LocalMcpConnection> = if config.mcp_connections_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&config.mcp_connections_json)?
    };
    if startup.sandbox.is_some() && config.memory_dir.is_some() {
        return Err("sandbox workers cannot load host memory".into());
    }
    if startup.sandbox.is_some()
        && mcp_connections
            .iter()
            .any(|connection| !connection.is_http())
    {
        return Err("sandbox workers cannot start host stdio MCP servers".into());
    }
    let workspace = std::env::var_os("MIRROR_WORKSPACE")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    let sandbox = match startup.sandbox.as_ref() {
        Some(sandbox) => Some(
            sandbox_config::SandboxConfig::try_from(sandbox)?
                .start(Uuid::parse_str(&startup.session_id)?, &workspace)
                .await?,
        ),
        None => None,
    };
    let resolved = match &sandbox {
        Some(sandbox) => Some(sandbox.instructions().await?),
        None => None,
    };
    let environment = match (&sandbox, config.include_environment.unwrap_or(false)) {
        (_, false) => None,
        (Some(sandbox), true) => Some(sandbox.environment().await?),
        (None, true) => Some(gather_environment_info(&workspace)),
    };
    let provider_config = ModelProviderConfig {
        context_window: config.context_window.map(usize::try_from).transpose()?,
        api: inference
            .api
            .as_deref()
            .ok_or("inference API is required")?
            .parse()?,
        model: inference.model.clone(),
        api_key: config.api_key.clone(),
        base_url: inference.base_url.clone().filter(|url| !url.is_empty()),
        auth_path: config.auth_path.as_ref().map(PathBuf::from),
        prompt_cache_key: startup.session_id.clone(),
        request_timeout: Some(mirror_agent::providers::DEFAULT_REQUEST_TIMEOUT),
    };
    let model = ModelProvider::new(provider_config.clone(), None).await?;
    let agent_config = ReactAgentConfig {
        max_iterations: config
            .max_iterations
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(usize::MAX),
        compaction_threshold: config
            .compaction_threshold
            .map(usize::try_from)
            .transpose()?,
        max_tokens: config
            .max_tokens
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(64_000),
        ..ReactAgentConfig::default()
    };
    let canonical_workspace = std::fs::canonicalize(&workspace).ok();
    let mut tools = WORKSPACE_TOOL_NAMES
        .into_iter()
        .map(|name| {
            let tool = native_tool(name, &workspace)?;
            let tool = match &sandbox {
                Some(sandbox) => {
                    Arc::new(SandboxRoutedTool::new(tool, sandbox.clone())) as Arc<dyn AnyTool>
                }
                None => tool,
            };
            Ok(if name == "bash" && config.root_search_guard {
                Arc::new(RootSearchGuard::new(tool, canonical_workspace.clone()))
                    as Arc<dyn AnyTool>
            } else {
                tool
            })
        })
        .collect::<Result<Vec<_>, ToolError>>()?;
    let subagent_tools = tools
        .iter()
        .filter(|tool| {
            config
                .tools
                .as_ref()
                .is_none_or(|tools| tools.names.iter().any(|name| name == tool.name()))
        })
        .cloned()
        .collect();
    let mut subagent = SubagentTool::new(
        &workspace,
        &startup.session_id,
        provider_config.clone(),
        agent_config,
        environment.clone(),
        subagent_tools,
    )?;
    if let Some(resolved) = &resolved {
        subagent = subagent.with_resolved_context(resolved.clone());
    }
    tools.push(Arc::new(Task));
    let default_web_provider = config.web_provider.as_deref().map(str::parse).transpose()?;
    let web_keys = config.web_credentials.unwrap_or_default();
    let web_key = move |selection: Option<WebProviderConfig>| match selection
        .or(default_web_provider)
        .unwrap_or_default()
    {
        WebProviderConfig::You => web_keys.you_api_key.clone(),
        WebProviderConfig::Tavily => web_keys.tavily_api_key.clone(),
        WebProviderConfig::Firecrawl => web_keys.firecrawl_api_key.clone(),
        _ => None,
    };
    let visibility_key = web_key.clone();
    let make_provider = move |selection: Option<WebProviderConfig>| {
        web_provider_from_config(selection.or(default_web_provider), web_key(selection))
    };
    tools.push(Arc::new(WebSearchTool::new(Box::new(
        make_provider.clone(),
    ))));
    tools.push(Arc::new(WebBrowseTool::new(Box::new(make_provider))));
    tools.push(Arc::new(subagent));
    let memory = match &config.memory_dir {
        Some(directory) => open_memory(MemoryEnvironment {
            store: Box::new(SqliteMemoryStore::new(SqliteMemoryDatabase::open(
                Path::new(directory),
            )?)?),
            workspace: Box::new(LocalWorkspace::new(&workspace)?),
            conversation_id: startup.session_id.clone(),
            private_session: true,
            pinned_memories: pinned_memories_path()?,
        }),
        None => None,
    };
    let pinned_memories = memory.as_ref().map(|memory| memory.pinned_memories.clone());
    let has_memory = memory.is_some();
    let mut hooks: Vec<Box<dyn ContextHooks>> = Vec::new();
    if sandbox.is_none() {
        hooks.push(Box::new(NestedInstructions::new(&workspace)));
    }
    if let Some(memory) = memory {
        tools.extend(memory.tools);
        hooks.push(memory.hooks);
    }
    let mut mcp_errors = Vec::new();
    let remote_tools =
        discover_mcp_connections(&mcp_connections, |error| mcp_errors.push(error)).await;
    let mut stdout = tokio::io::stdout();
    for error in mcp_errors {
        // The daemon discards stderr; only payload-free failures belong on the client stream.
        let warning = SessionMessage {
            payload: Some(session_message::Payload::AgentMessage(AgentMessage {
                id: Uuid::new_v4().to_string(),
                payload: Some(agent_message::Payload::Log(LogRecord {
                    level: "warning".into(),
                    text: format!(
                        "{error}. This connection's tools were not loaded. \
                                 Check Connections or /connections and network access, \
                                 then start a new conversation to retry."
                    ),
                })),
                ..Default::default()
            })),
            ..Default::default()
        };
        let encoded = STANDARD.encode(warning.encode_to_vec());
        stdout
            .write_all(format!("MIRROR_SESSION_MESSAGE={encoded}\n").as_bytes())
            .await?;
    }
    stdout.flush().await?;
    let mcp = mcp_tools(remote_tools)?;
    let has_mcp = !mcp.is_empty();
    tools.extend(mcp);
    if let Some(names) = &config.tools {
        let memory_tools = [SearchMemoryTool::NAME, SaveMemoryTool::NAME];
        tools = names
            .names
            .iter()
            .filter(|name| has_memory || !memory_tools.contains(&name.as_str()))
            .filter(|name| {
                has_mcp || !matches!(name.as_str(), "mcp_search" | "mcp_describe" | "mcp_call")
            })
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
        let mut api = config
            .compaction_model_api
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or(provider_config.api);
        let base_url = config.compaction_model_base_url.clone().or_else(|| {
            (api == provider_config.api)
                .then(|| provider_config.base_url.clone())
                .flatten()
        });
        if base_url.is_some() && api == ModelApi::CodexResponses {
            api = ModelApi::ChatCompletions;
        }
        Some(
            ModelProvider::new(
                ModelProviderConfig {
                    api,
                    model,
                    api_key: config.compaction_api_key.or_else(|| {
                        (api == provider_config.api && base_url == provider_config.base_url)
                            .then(|| provider_config.api_key.clone())
                            .flatten()
                    }),
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
        include_project_instructions: true,
        include_skills: true,
        environment,
        pinned_memories,
        additional_skills: config
            .additional_skills
            .into_iter()
            .map(|skill| Skill {
                name: skill.name,
                description: skill.description,
                body: skill.body,
                path: skill.path.into(),
                disable_model_invocation: skill.disable_model_invocation,
            })
            .collect(),
        resolved,
        ..SystemPromptConfig::default()
    };
    let skills = resolve_prompt_instructions(&prompt_config).skills;
    let tool_executor =
        DirectToolExecutor::new(tools).with_tool_visibility(move |name, context| {
            !matches!(name, "web_search" | "web_browse")
                || visibility_key(context.web_provider()).is_some_and(|key| !key.trim().is_empty())
        });
    let agent = ReactAgent::new(
        model.clone(),
        model,
        compaction_model,
        SummaryCompactorFactory,
        &tool_executor,
        prompt_config,
        agent_config,
    )?;
    if startup.database.is_empty() {
        return Err("worker database is required".into());
    }
    let database = PathBuf::from(startup.database);
    let mut store = SQLiteStore::open(
        &database,
        &startup.session_id,
        &workspace,
        startup.create_session,
        startup.allow_workspace_mismatch,
    )?;
    store.checkpoints_enabled =
        config.checkpoints_enabled.unwrap_or(false) && startup.sandbox.is_none();
    let (events, mut startup_events) = tokio::sync::broadcast::channel(1024);
    let service_start = ReactAgentService::start(
        agent,
        Box::new(tool_executor),
        Box::new(store),
        startup.session_id,
        approval_mode,
        skills,
        hooks,
        host_tool_requests,
        events,
        None,
    )
    .await;
    while let Ok(event) = startup_events.try_recv() {
        for response in startup_responses(event) {
            let encoded = STANDARD.encode(response.encode_to_vec());
            stdout
                .write_all(format!("MIRROR_SESSION_MESSAGE={encoded}\n").as_bytes())
                .await?;
        }
    }
    stdout.flush().await?;
    let (service, runtime) = service_start?;
    tokio::spawn(runtime);
    let service_control = service.clone();
    let mirror_rpc = LocalMirrorRpc::new(
        service.clone(),
        database,
        startup.sandbox.is_none().then_some(workspace),
    );
    let rpc = AgentRpc::new(service);
    let rpc_control = rpc.clone();
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let parent_control = startup.exit_on_stdin_close.then(stdin_control);
    let idle_service = service_control.clone();
    let shutdown = async move {
        tokio::select! {
            _ = terminate.recv() => {}
            _ = interrupt.recv() => {}
            _ = async {
                match parent_control {
                    Some((drain, closed)) => tokio::select! {
                        _ = closed => {},
                        _ = async {
                            let _ = drain.await;
                            // Only the service knows about queued inputs, approvals and background tools.
                            while !idle_service.is_closed() && !idle_service.close_if_idle().await.unwrap_or(false) {
                                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            }
                        } => {},
                    },
                    None => std::future::pending().await,
                }
            } => {}
        }
        rpc_control.shutdown();
    };
    let server = Server::builder()
        .add_service(
            AgentRpcServer::new(rpc)
                .max_decoding_message_size(10 * 1024 * 1024)
                .max_encoding_message_size(10 * 1024 * 1024),
        )
        .add_service(MirrorRpcServer::new(mirror_rpc));
    let serve_result: Result<(), Box<dyn Error>> = async {
        let listener = bind_private_socket(Path::new(socket))?;
        announce_rpc_address(&startup.rpc_address).await?;
        server
            .serve_with_incoming_shutdown(unix_incoming(listener), shutdown)
            .await?;
        Ok(())
    }
    .await;
    let close_result = service_control.close().await;
    serve_result?;
    close_result?;
    Ok(())
}

fn bind_private_socket(socket: &Path) -> io::Result<UnixListener> {
    let parent = socket
        .parent()
        .ok_or_else(|| io::Error::other("socket needs a parent"))?;
    let metadata = std::fs::symlink_metadata(parent)?;
    if !metadata.is_dir() || metadata.uid() != geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "agent socket directory must be user-owned and private",
        ));
    }
    // The caller owns the socket directory and its cleanup.
    let listener = UnixListener::bind(socket)?;
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

fn unix_incoming(listener: UnixListener) -> impl Stream<Item = io::Result<UnixStream>> {
    async_stream::try_stream! {
        loop {
            let (stream, _) = listener.accept().await?;
            if stream.peer_cred().is_ok_and(|peer| peer.uid() == geteuid().as_raw()) {
                yield stream;
            }
        }
    }
}

async fn announce_rpc_address(address: &str) -> io::Result<()> {
    let mut stdout = tokio::io::stdout();
    stdout
        .write_all(format!("MIRROR_RPC_ADDRESS={address}\n").as_bytes())
        .await?;
    stdout.flush().await
}

fn stdin_control() -> (
    tokio::sync::oneshot::Receiver<()>,
    tokio::sync::oneshot::Receiver<()>,
) {
    let (drain, draining) = tokio::sync::oneshot::channel();
    let (closed, receiver) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 256];
        let mut drain = Some(drain);
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    if buffer[..count].contains(&b'D') {
                        if let Some(drain) = drain.take() {
                            let _ = drain.send(());
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let _ = closed.send(());
    });
    (draining, receiver)
}

fn worker_startup() -> Result<WorkerStartup, Box<dyn Error>> {
    if std::env::args().nth(1).as_deref() != Some("--config-stdin") {
        return Err("worker requires --config-stdin and a private protobuf startup frame".into());
    }
    let mut stdin = std::io::stdin().lock();
    let mut size = [0; 4];
    stdin.read_exact(&mut size)?;
    let size = u32::from_be_bytes(size) as usize;
    if size > 64 * 1024 * 1024 {
        return Err("worker configuration exceeds 64 MiB".into());
    }
    let mut bytes = vec![0; size];
    stdin.read_exact(&mut bytes)?;
    Ok(WorkerStartup::decode(bytes.as_slice())?)
}
