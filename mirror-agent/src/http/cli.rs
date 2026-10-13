use std::error::Error;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};
use mirror_agent::local::LocalWorkspace;
use mirror_agent::memory::{
    open_memory, MemoryEnvironment, SqliteMemoryDatabase, SqliteMemoryStore,
};
use mirror_agent::providers::{
    is_reasoning_effort, reasoning_effort_expectation, ModelApi, ModelProvider, ModelProviderConfig,
};
use mirror_agent::react::{ContextHooks, ReactAgent, ReactAgentConfig};
use mirror_agent::system_prompt::{pinned_memories_path, NestedInstructions, SystemPromptConfig};
use mirror_agent::tools::{
    validate_tools, AnyTool, BashTool, DirectToolExecutor, EditTool, FindTool, GrepTool, LsTool,
    ReadTool, RemoteToolExecutor, ToolError, ToolExecutor, WriteTool,
};
use mirror_agent::{AgentApprovalMode, InferenceConfig, ReactAgentService};
use uuid::Uuid;

use super::{router, JsonlStore, PromptRunner, TracedModel};

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
        /// Append each committed conversation record to this JSONL file.
        #[arg(long, default_value = "mirror.jsonl")]
        output: PathBuf,
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
    /// A level name (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) or a
    /// non-negative integer budget for endpoints that take one.
    #[arg(long, value_parser = reasoning_effort)]
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
    /// Maximum retryable exceptions per turn; zero disables the limit.
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
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let Command::Serve {
        agent,
        bind,
        output,
    } = Cli::parse().command;
    let output = std::path::absolute(output)?;
    let parent = output.parent().ok_or("output must name a trace file")?;
    tokio::fs::create_dir_all(parent).await?;
    let runner = agent.start(output).await?;
    let result = serve(&runner, bind).await;
    let closed = runner.close().await;
    result?;
    closed?;
    Ok(())
}

impl AgentArgs {
    async fn start(&self, output: PathBuf) -> Result<PromptRunner, Box<dyn Error>> {
        let limits = self.limits()?;
        let mut inference = InferenceConfig {
            model: self.model.clone(),
            api: Some(
                match self.api {
                    ModelApi::ChatCompletions => "openai-chat-completions",
                    ModelApi::ChatCompletionsNonStreaming => {
                        "openai-chat-completions-non-streaming"
                    }
                    ModelApi::Reflection => "reflection",
                    ModelApi::Responses => "openai-responses",
                    ModelApi::CodexResponses => "openai-codex-responses",
                    ModelApi::AnthropicMessages => "anthropic-messages",
                }
                .into(),
            ),
            base_url: self.base_url.as_ref().map(ToString::to_string),
            reasoning_effort: self.reasoning_effort.clone(),
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            extra_body: self.extra_body.clone(),
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
        let mut tools = self
            .tools
            .iter()
            .map(|name| -> Result<Arc<dyn AnyTool>, ToolError> {
                Ok(match name {
                    ToolName::Bash if remote => Arc::new(BashTool::remote()),
                    ToolName::Bash => Arc::new(BashTool::new(&workspace)?),
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
        if self.use_native_bash {
            if !matches!(
                self.api,
                ModelApi::ChatCompletions | ModelApi::ChatCompletionsNonStreaming
            ) {
                return Err("--use-native-bash requires a Chat Completions API".into());
            }
            let bash_name = tools
                .iter()
                .find(|tool| tool.name() == "bash")
                .ok_or("--use-native-bash requires --tools bash")?
                .name();
            let mut extra: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(self.extra_body.as_deref().unwrap_or("{}"))?;
            if extra.contains_key("tools") {
                return Err("--use-native-bash conflicts with extra-body tools".into());
            }
            let kwargs = extra
                .entry("chat_template_kwargs")
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .ok_or("chat_template_kwargs must be an object")?;
            if kwargs.contains_key("builtin_tools") {
                return Err("--use-native-bash conflicts with extra-body builtin_tools".into());
            }
            kwargs.insert("builtin_tools".into(), serde_json::json!([bash_name]));
            inference.extra_body = Some(serde_json::to_string(&extra)?);
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
        let store = JsonlStore::create(&output)?;
        let inference_provider = model.clone();
        let model = TracedModel::create(model, &output.with_extension("model-calls.jsonl")).await?;
        let agent = ReactAgent::new(
            model,
            inference_provider,
            None,
            executor.as_ref(),
            prompt,
            limits,
        )?;
        let (service, runtime) = ReactAgentService::start(
            agent,
            executor,
            Box::new(store),
            id,
            AgentApprovalMode::Auto,
            Vec::new(),
            hooks,
            None,
            tokio::sync::broadcast::channel(1024).0,
            None,
        )
        .await?;
        tokio::spawn(runtime);
        service.set_inference_config(inference).await?;
        Ok(PromptRunner::new(service, self.turn_timeout_seconds))
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
                api_key: if self.api == ModelApi::Reflection {
                    None
                } else {
                    std::env::var("OPENAI_API_KEY").ok()
                },
                base_url: self.base_url.as_ref().map(ToString::to_string),
                auth_path: std::env::var_os("MIRROR_CODEX_AUTH_PATH").map(Into::into),
                prompt_cache_key: id.into(),
                request_timeout: self.request_timeout_seconds,
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

async fn serve(runner: &PromptRunner, bind: SocketAddr) -> Result<(), Box<dyn Error>> {
    let router = router(runner.clone());
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

fn reasoning_effort(value: &str) -> Result<String, String> {
    if is_reasoning_effort(value) {
        return Ok(value.to_string());
    }
    Err(reasoning_effort_expectation())
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
