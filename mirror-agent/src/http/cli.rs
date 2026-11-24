use std::error::Error;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use mirror_agent::compaction::SummaryCompactorFactory;
use mirror_agent::local::LocalWorkspace;
use mirror_agent::mcp::{discover_mcp_connections, mcp_tools, LocalMcpConnection};
use mirror_agent::memory::{
    open_memory, MemoryEnvironment, SqliteMemoryDatabase, SqliteMemoryStore,
};
use mirror_agent::providers::{ModelApi, ModelProvider, ModelProviderConfig};
use mirror_agent::react::{ContextHooks, InMemoryStore, ReactAgent, ReactAgentConfig};
use mirror_agent::system_prompt::{pinned_memories_path, NestedInstructions, SystemPromptConfig};
use mirror_agent::tools::{
    validate_tools, AnyTool, BashTool, DirectToolExecutor, EditTool, FindTool, GrepTool, LsTool,
    ReadTool, RemoteToolExecutor, RootSearchGuard, ToolError, ToolExecutor, WebBrowseTool,
    WebSearchTool, WriteTool,
};
use mirror_agent::web::{web_provider_from_config, WebProviderConfig};
use mirror_agent::{AgentApprovalMode, InferenceConfig, ReactAgentService, Store};
use uuid::Uuid;

use super::{router, JsonlStore, MemoryTrace, PromptRunner, SessionTrace, TracedModel};

#[derive(Parser)]
#[command(
    name = "mirror-agent",
    version,
    about = "Serve a sandbox-local Mirror agent over HTTP"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Expose one configured agent over sandbox-local HTTP.
    Serve {
        #[command(flatten)]
        agent: AgentArgs,
        #[arg(long, default_value = "127.0.0.1:4097")]
        bind: SocketAddr,
        /// Write committed conversation records to this JSONL file; defaults to mirror.jsonl
        /// unless --in-memory-output is set. The file and its .model-calls.jsonl sibling are
        /// truncated at startup.
        #[arg(
            long,
            default_value = "mirror.jsonl",
            default_value_if("in_memory_output", "true", None)
        )]
        output: Option<PathBuf>,
        /// Expose unauthenticated trace snapshots; combine with --output to also write files.
        #[arg(long)]
        in_memory_output: bool,
    },
}

#[derive(Args)]
struct AgentArgs {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "openai-chat-completions")]
    api: ModelApi,
    #[arg(long)]
    base_url: Option<reqwest::Url>,
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    /// Execute supported tools through an Irregular-compatible send-command URL.
    #[arg(long)]
    remote: Option<reqwest::Url>,
    /// Add custom instructions before any opted-in project and skill context.
    #[arg(long)]
    system_prompt: Option<String>,
    /// Include AGENTS.md / CLAUDE.md instructions in the system prompt.
    #[arg(long)]
    include_project_instructions: bool,
    /// Include the available skills catalog in the system prompt.
    #[arg(long)]
    include_skills: bool,
    /// Give the agent the personal memory stored under this directory. Needs a
    /// local workspace; the store's contents reach the model endpoint.
    #[arg(long)]
    memory_dir: Option<PathBuf>,
    /// MCP connection declarations as a JSON array, supplied by the trusted launcher.
    #[arg(long, value_name = "JSON")]
    mcp_connections: Option<String>,
    /// Web search and browse backend (you, tavily, or firecrawl); defaults to You.com.
    #[arg(long)]
    web_provider: Option<WebProviderConfig>,
    /// A level name (`none`, `low`, `medium`, `high`, `xhigh`, `max`) or a
    /// non-negative integer budget for endpoints that take one; llama.cpp takes
    /// `on`, `off` or a level its model's template supports.
    #[arg(long)]
    reasoning_effort: Option<String>,
    /// A JSON object merged into every model request after the fields above,
    /// spelled as the endpoint expects (for example
    /// `{"chat_template_kwargs": {"enable_thinking": false}}`).
    #[arg(long, value_parser = json_object)]
    extra_body: Option<String>,
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long)]
    seed: Option<i64>,
    /// Override model context discovery through GET /models.
    #[arg(long)]
    context_window: Option<usize>,
    #[arg(long, default_value_t = 64_000)]
    max_tokens: usize,
    #[arg(long)]
    compaction_threshold: Option<usize>,
    /// Preserve history and stop on context exhaustion instead of compacting.
    #[arg(long, conflicts_with = "compaction_threshold")]
    disable_compaction: bool,
    #[arg(long, default_value_t = 1000)]
    max_iterations: usize,
    /// Consecutive retryable failures, including all-invalid JSON tool turns; zero disables the limit.
    #[arg(long, default_value_t = 0)]
    max_exceptions: usize,
    #[arg(long, default_value_t = 2000)]
    tool_result_max_chars: usize,
    #[arg(long, default_value = "1800", value_parser = duration)]
    turn_timeout_seconds: Duration,
    /// Limit on one model request, including generation. Unset by default:
    /// the turn timeout bounds a turn, and a long generation is never
    /// abandoned and resampled.
    #[arg(long, value_parser = duration)]
    request_timeout_seconds: Option<Duration>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "bash")]
    tools: Vec<ToolName>,
    /// Advertise bash as a system builtin through the Reflection eval proxy.
    #[arg(long)]
    use_native_bash: bool,
    /// Advertise selected web tools as the system builtins search and browse.
    #[arg(long)]
    use_native_web: bool,
    #[arg(
        long,
        help = "Refuse recursive grep/rg over / that does not exclude /proc before running it."
    )]
    root_search_guard: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ToolName {
    Bash,
    Read,
    Write,
    Edit,
    Grep,
    Find,
    Ls,
    #[value(name = "web_search", alias = "web-search")]
    WebSearch,
    #[value(name = "web_browse", alias = "web-browse")]
    WebBrowse,
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let Command::Serve {
        agent,
        bind,
        output,
        in_memory_output,
    } = Cli::parse().command;
    let output = output.map(std::path::absolute).transpose()?;
    if let Some(output) = &output {
        let parent = output.parent().ok_or("output must name a trace file")?;
        tokio::fs::create_dir_all(parent).await?;
    }
    let (runner, traces) = agent.start(output, in_memory_output).await?;
    let result = serve(&runner, bind, traces).await;
    let closed = runner.close().await;
    result?;
    closed?;
    Ok(())
}

impl AgentArgs {
    async fn start(
        &self,
        output: Option<PathBuf>,
        in_memory_output: bool,
    ) -> Result<(PromptRunner, Option<SessionTrace>), Box<dyn Error>> {
        let limits = self.limits()?;
        if let Some(effort) = self
            .reasoning_effort
            .as_deref()
            .filter(|effort| !self.api.accepts_reasoning_effort(effort))
        {
            return Err(format!(
                "invalid reasoning effort {effort:?}: {}",
                self.api.reasoning_effort_expectation()
            )
            .into());
        }
        let inference = InferenceConfig {
            model: self.model.clone(),
            api: Some(self.api.as_str().into()),
            base_url: self.base_url.as_ref().map(ToString::to_string),
            reasoning_effort: self.reasoning_effort.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            extra_body: self.native_extra_body()?,
            context_window: None,
        };
        self.api.validate_sampling(&inference)?;
        let remote = self.remote.is_some();
        if remote && self.memory_dir.is_some() {
            return Err("--memory-dir needs the workspace on this machine; with --remote the checkout is remote and memory has no way to read it yet".into());
        }
        let workspace = if remote {
            self.workspace.clone()
        } else {
            self.workspace.canonicalize()?
        };
        let default_web_provider = self.web_provider;
        let make_provider = move |selection: Option<WebProviderConfig>| {
            let selection = selection.or(default_web_provider);
            let api_key = std::env::var(selection.unwrap_or_default().key_env()).ok();
            web_provider_from_config(selection, api_key)
        };
        let web_search = Arc::new(WebSearchTool::new(Box::new(make_provider)).with_name(
            if self.use_native_web {
                "search"
            } else {
                "web_search"
            },
        ));
        let web_browse = Arc::new(WebBrowseTool::new(Box::new(make_provider)).with_name(
            if self.use_native_web {
                "browse"
            } else {
                "web_browse"
            },
        ));
        let mut tools = self
            .tools
            .iter()
            .map(|name| -> Result<Arc<dyn AnyTool>, ToolError> {
                Ok(match name {
                    ToolName::Bash => self.bash_tool(remote, &workspace)?,
                    ToolName::Read if remote => Arc::new(ReadTool::remote()),
                    ToolName::Read => Arc::new(ReadTool::new(&workspace)?),
                    ToolName::Write if remote => Arc::new(WriteTool::remote()),
                    ToolName::Write => Arc::new(WriteTool::new(&workspace)?),
                    ToolName::Edit if remote => Arc::new(EditTool::remote()),
                    ToolName::Edit => Arc::new(EditTool::new(&workspace)?),
                    ToolName::Grep if remote => Arc::new(GrepTool::remote()),
                    ToolName::Grep => Arc::new(GrepTool::new(&workspace)?),
                    ToolName::Find if remote => Arc::new(FindTool::remote()),
                    ToolName::Find => Arc::new(FindTool::new(&workspace)?),
                    ToolName::Ls if remote => Arc::new(LsTool::remote()),
                    ToolName::Ls => Arc::new(LsTool::new(&workspace)?),
                    ToolName::WebSearch => web_search.clone(),
                    ToolName::WebBrowse => web_browse.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let id = Uuid::new_v4().to_string();
        let memory = match &self.memory_dir {
            Some(directory) => open_memory(MemoryEnvironment {
                store: Box::new(SqliteMemoryStore::new(SqliteMemoryDatabase::open(
                    directory,
                )?)?),
                workspace: Box::new(LocalWorkspace::new(&workspace)?),
                conversation_id: id.clone(),
                private_session: true,
                pinned_memories: pinned_memories_path()?,
            }),
            None => None,
        };
        let pinned_memories = memory.as_ref().map(|memory| memory.pinned_memories.clone());
        let mut hooks: Vec<Box<dyn ContextHooks>> = Vec::new();
        if !remote && self.include_project_instructions {
            hooks.push(Box::new(NestedInstructions::new(&workspace)));
        }
        if let Some(memory) = memory {
            tools.extend(memory.tools);
            hooks.push(memory.hooks);
        }
        if let Some(config) = &self.mcp_connections {
            let connections: Vec<LocalMcpConnection> =
                serde_json::from_str(config).map_err(|_| {
                    "invalid --mcp-connections: expected a JSON array of MCP connections"
                })?;
            if remote && connections.iter().any(|connection| !connection.is_http()) {
                return Err(
                    "--remote supports HTTP MCP connections, not host stdio servers".into(),
                );
            }
            tools.extend(mcp_tools(
                discover_mcp_connections(&connections, |error| eprintln!("{error}")).await,
            )?);
        }
        let prompt = SystemPromptConfig {
            custom_prompt: self.system_prompt.clone(),
            include_project_instructions: !remote && self.include_project_instructions,
            include_skills: !remote && self.include_skills,
            working_directory: (!remote).then(|| workspace.clone()),
            environment: None,
            pinned_memories,
            ..SystemPromptConfig::default()
        };
        validate_tools(tools.iter().map(|tool| tool.as_ref()))?;
        let executor: Box<dyn ToolExecutor> = match &self.remote {
            Some(url) => Box::new(RemoteToolExecutor::new(tools, url.clone())?),
            None => Box::new(DirectToolExecutor::new(tools)),
        };
        let model = self.provider(&id).await?;
        let inference_provider = model.clone();
        let mut store: Box<dyn Store> = match &output {
            Some(output) => Box::new(JsonlStore::create(output)?),
            None => Box::new(InMemoryStore::default()),
        };
        let (calls, traces) = if in_memory_output {
            let (traced, native) = MemoryTrace::new(store);
            store = Box::new(traced);
            let (calls, model_calls) = tokio::sync::watch::channel(Vec::new());
            (
                Some(calls),
                Some(SessionTrace {
                    native,
                    model_calls,
                }),
            )
        } else {
            (None, None)
        };
        let model = TracedModel::create(
            model,
            output
                .map(|path| path.with_extension("model-calls.jsonl"))
                .as_deref(),
            calls,
        )
        .await?;
        let agent = ReactAgent::new(
            model,
            inference_provider,
            None,
            SummaryCompactorFactory,
            executor.as_ref(),
            prompt,
            limits,
        )?;
        let (service, runtime) = ReactAgentService::start(
            agent,
            executor,
            store,
            id,
            AgentApprovalMode::Bypass,
            Vec::new(),
            hooks,
            None,
            tokio::sync::broadcast::channel(1024).0,
            None,
            mirror_agent::hooks::PreToolHook::load_user(&if remote {
                std::env::current_dir()?
            } else {
                workspace.clone()
            })
            .await?,
        )
        .await?;
        tokio::spawn(runtime);
        service.set_inference_config(inference).await?;
        Ok((
            PromptRunner::new(service, self.turn_timeout_seconds),
            traces,
        ))
    }

    fn bash_tool(&self, remote: bool, workspace: &Path) -> Result<Arc<dyn AnyTool>, ToolError> {
        let tool: Arc<dyn AnyTool> = if remote {
            Arc::new(BashTool::remote())
        } else {
            Arc::new(BashTool::new(workspace)?)
        };
        Ok(if self.root_search_guard {
            Arc::new(RootSearchGuard::new(
                tool,
                (!remote).then(|| workspace.to_path_buf()),
            ))
        } else {
            tool
        })
    }

    fn native_extra_body(&self) -> Result<Option<String>, Box<dyn Error>> {
        if !self.use_native_bash && !self.use_native_web {
            return Ok(self.extra_body.clone());
        }
        let flag = if self.use_native_bash {
            "--use-native-bash"
        } else {
            "--use-native-web"
        };
        if !matches!(
            self.api,
            ModelApi::ChatCompletions | ModelApi::ChatCompletionsNonStreaming
        ) {
            return Err(format!("{flag} requires a Chat Completions API").into());
        }
        if self.use_native_bash && !self.tools.contains(&ToolName::Bash) {
            return Err("--use-native-bash requires --tools bash".into());
        }
        if self.use_native_web
            && !self.tools.contains(&ToolName::WebSearch)
            && !self.tools.contains(&ToolName::WebBrowse)
        {
            return Err("--use-native-web requires --tools web_search or web_browse".into());
        }
        let builtins: Vec<_> = self
            .tools
            .iter()
            .filter_map(|tool| match tool {
                ToolName::Bash if self.use_native_bash => Some("bash"),
                ToolName::WebSearch if self.use_native_web => Some("search"),
                ToolName::WebBrowse if self.use_native_web => Some("browse"),
                _ => None,
            })
            .collect();
        let mut extra: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(self.extra_body.as_deref().unwrap_or("{}"))?;
        if extra.contains_key("tools") {
            return Err(format!("{flag} conflicts with extra-body tools").into());
        }
        let kwargs = extra
            .entry("chat_template_kwargs")
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
            .ok_or("chat_template_kwargs must be an object")?;
        if kwargs.contains_key("builtin_tools") {
            return Err(format!("{flag} conflicts with extra-body builtin_tools").into());
        }
        kwargs.insert("builtin_tools".into(), serde_json::json!(builtins));
        Ok(Some(serde_json::to_string(&extra)?))
    }

    async fn provider(&self, id: &str) -> Result<ModelProvider, Box<dyn Error>> {
        if self.model.trim().is_empty() {
            return Err("model must not be empty".into());
        }
        if self.base_url.as_ref().is_some_and(|url| {
            !matches!(url.scheme(), "http" | "https")
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
        }) {
            return Err("base-url must be an HTTP(S) URL without credentials".into());
        }
        Ok(ModelProvider::new(
            ModelProviderConfig {
                context_window: self.context_window,
                api: self.api,
                model: self.model.clone(),
                api_key: if matches!(
                    self.api,
                    ModelApi::Reflection
                        | ModelApi::Gemini
                        | ModelApi::Vercel
                        | ModelApi::Vertex
                        | ModelApi::LlamaCpp
                ) {
                    None
                } else {
                    std::env::var("OPENAI_API_KEY").ok()
                },
                base_url: self.base_url.as_ref().map(ToString::to_string),
                auth_path: std::env::var_os("MIRROR_CODEX_AUTH_PATH").map(Into::into),
                prompt_cache_key: id.into(),
                request_timeout: self.request_timeout_seconds,
                managed_reference: None,
            },
            None,
        )
        .await?)
    }

    fn limits(&self) -> Result<ReactAgentConfig, Box<dyn Error>> {
        if self.max_tokens == 0
            || self
                .context_window
                .is_some_and(|window| window == 0 || window > usize::MAX / 100)
            || self.max_iterations == 0
            || self.compaction_threshold == Some(0)
        {
            return Err(
                "limits must be positive and context-window must fit token-budget arithmetic"
                    .into(),
            );
        }
        Ok(ReactAgentConfig {
            max_tokens: self.max_tokens,
            max_iterations: self.max_iterations,
            max_exceptions: self.max_exceptions,
            compaction_threshold: self.compaction_threshold,
            disable_compaction: self.disable_compaction,
            tool_result_max_chars: self.tool_result_max_chars,
        })
    }
}

async fn serve(
    runner: &PromptRunner,
    bind: SocketAddr,
    traces: Option<SessionTrace>,
) -> Result<(), Box<dyn Error>> {
    let router = router(runner.clone(), traces);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let signal = shutdown_signal()?;
    println!("MIRROR_HTTP_ADDRESS=http://{}", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown({
            let runner = runner.clone();
            async move {
                signal.await;
                let _ = runner.close().await;
            }
        })
        .await?;
    Ok(())
}

fn json_object(value: &str) -> Result<String, String> {
    match serde_json::from_str::<serde_json::Value>(value) {
        Ok(serde_json::Value::Object(_)) => Ok(value.to_string()),
        Ok(_) => Err("expected a JSON object".to_string()),
        Err(error) => Err(format!("expected a JSON object: {error}")),
    }
}

fn duration(value: &str) -> Result<Duration, String> {
    let seconds = value.parse::<f64>().map_err(|error| error.to_string())?;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 604_800.0 {
        return Err("timeouts must be greater than zero and at most 604800 seconds".into());
    }
    Duration::try_from_secs_f64(seconds).map_err(|error| error.to_string())
}

fn shutdown_signal() -> std::io::Result<impl std::future::Future<Output = ()>> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    Ok(async move {
        tokio::select! { _ = terminate.recv() => {}, _ = interrupt.recv() => {} }
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn parse_agent(args: &[&str]) -> AgentArgs {
        let Command::Serve { agent, .. } = Cli::try_parse_from(
            ["mirror-agent", "serve", "--model", "test-model"]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .expect("parse server arguments")
        .command;
        agent
    }

    #[test]
    fn native_web_marks_only_selected_web_tools_for_non_streaming() {
        let agent = parse_agent(&[
            "--api",
            "openai-chat-completions-non-streaming",
            "--tools",
            "bash,web_browse",
            "--use-native-web",
        ]);
        let extra: serde_json::Value = serde_json::from_str(
            &agent
                .native_extra_body()
                .expect("valid native options")
                .expect("extra body"),
        )
        .expect("JSON");
        assert_eq!(
            extra,
            serde_json::json!({"chat_template_kwargs":{"builtin_tools":["browse"]}})
        );
    }

    #[test]
    fn native_web_requires_selected_web_tools() {
        let agent = parse_agent(&["--use-native-web"]);
        assert_eq!(
            agent
                .native_extra_body()
                .expect_err("missing web tools")
                .to_string(),
            "--use-native-web requires --tools web_search or web_browse"
        );
    }

    #[test]
    fn native_bash_still_requires_bash() {
        let agent = parse_agent(&["--tools", "web_search", "--use-native-bash"]);
        assert_eq!(
            agent
                .native_extra_body()
                .expect_err("missing bash")
                .to_string(),
            "--use-native-bash requires --tools bash"
        );
    }

    #[test]
    fn native_web_rejects_other_apis() {
        let agent = parse_agent(&[
            "--api",
            "openai-responses",
            "--tools",
            "web_search",
            "--use-native-web",
        ]);
        assert_eq!(
            agent
                .native_extra_body()
                .expect_err("unsupported API")
                .to_string(),
            "--use-native-web requires a Chat Completions API"
        );
    }

    #[test]
    fn native_web_rejects_tool_overrides() {
        let agent = parse_agent(&[
            "--tools",
            "web_search",
            "--use-native-web",
            "--extra-body",
            r#"{"tools":[]}"#,
        ]);
        assert_eq!(
            agent
                .native_extra_body()
                .expect_err("tool override")
                .to_string(),
            "--use-native-web conflicts with extra-body tools"
        );
    }

    #[test]
    fn native_web_rejects_builtin_overrides() {
        let agent = parse_agent(&[
            "--tools",
            "web_search",
            "--use-native-web",
            "--extra-body",
            r#"{"chat_template_kwargs":{"builtin_tools":[]}}"#,
        ]);
        assert_eq!(
            agent
                .native_extra_body()
                .expect_err("builtin override")
                .to_string(),
            "--use-native-web conflicts with extra-body builtin_tools"
        );
    }

    #[test]
    fn native_web_rejects_non_object_template_kwargs() {
        let agent = parse_agent(&[
            "--tools",
            "web_search",
            "--use-native-web",
            "--extra-body",
            r#"{"chat_template_kwargs":false}"#,
        ]);
        assert_eq!(
            agent
                .native_extra_body()
                .expect_err("invalid template kwargs")
                .to_string(),
            "chat_template_kwargs must be an object"
        );
    }

    #[test]
    fn root_search_guard_is_off_unless_asked_for() {
        assert!(!parse_agent(&[]).root_search_guard);
        assert!(parse_agent(&["--root-search-guard"]).root_search_guard);
    }

    #[test]
    fn file_output_remains_the_default() {
        let Command::Serve {
            output,
            in_memory_output,
            ..
        } = Cli::try_parse_from(["mirror-agent", "serve", "--model", "test"])
            .expect("default output")
            .command;
        assert_eq!(output, Some(PathBuf::from("mirror.jsonl")));
        assert!(!in_memory_output);
    }

    #[test]
    fn memory_output_does_not_default_to_a_file() {
        let Command::Serve {
            output,
            in_memory_output,
            ..
        } = Cli::try_parse_from([
            "mirror-agent",
            "serve",
            "--model",
            "test",
            "--in-memory-output",
        ])
        .expect("memory output")
        .command;
        assert_eq!(output, None);
        assert!(in_memory_output);
    }

    #[test]
    fn memory_output_composes_with_an_explicit_file() {
        let Command::Serve {
            output,
            in_memory_output,
            ..
        } = Cli::try_parse_from([
            "mirror-agent",
            "serve",
            "--model",
            "test",
            "--in-memory-output",
            "--output",
            "trace.jsonl",
        ])
        .expect("both outputs")
        .command;
        assert_eq!(output, Some(PathBuf::from("trace.jsonl")));
        assert!(in_memory_output);
    }

    #[test]
    fn exception_limit_is_disabled_by_default() {
        let Command::Serve { agent, .. } =
            Cli::try_parse_from(["mirror-agent", "serve", "--model", "test-model"])
                .expect("parse server arguments")
                .command;

        assert_eq!(agent.limits().expect("valid limits").max_exceptions, 0);
        assert!(!agent.limits().expect("valid limits").disable_compaction);
    }

    #[test]
    fn exception_limit_accepts_an_explicit_value() {
        let Command::Serve { agent, .. } = Cli::try_parse_from([
            "mirror-agent",
            "serve",
            "--model",
            "test-model",
            "--max-exceptions",
            "2",
        ])
        .expect("parse server arguments")
        .command;

        assert_eq!(agent.limits().expect("valid limits").max_exceptions, 2);
    }

    #[test]
    fn server_accepts_disabling_compaction() {
        let Command::Serve { agent, .. } = Cli::try_parse_from([
            "mirror-agent",
            "serve",
            "--model",
            "test-model",
            "--disable-compaction",
        ])
        .expect("parse server arguments")
        .command;

        assert!(agent.limits().expect("valid limits").disable_compaction);
        assert!(Cli::try_parse_from([
            "mirror-agent",
            "serve",
            "--model",
            "test-model",
            "--disable-compaction",
            "--compaction-threshold",
            "1000",
        ])
        .is_err());
    }
}
