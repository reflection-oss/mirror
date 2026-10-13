//! Python construction and lifecycle for the in-process Rust agent service.

use std::sync::Arc;

use futures_util::future::Either;
use mirror_agent_core::providers::CompletionRequest;
use mirror_agent_core::providers::{Llm, LlmError, ModelProvider, StreamEvent};
use mirror_agent_core::react::{InMemoryStore, ReactAgent};
use mirror_agent_core::rpc::AgentInProcessRpc;
use mirror_agent_core::system_prompt::SystemPromptConfig;
use mirror_agent_core::tools::validate_tools;
use mirror_agent_core::tools::AnyTool;
use mirror_agent_core::tools::DirectToolExecutor;
use mirror_agent_core::AgentApprovalMode;
use mirror_agent_core::{ReactAgentService, ReactAgentServiceHandle};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use pyo3_async_runtimes::TaskLocals;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::model::{PyScriptedModel, ScriptedModel};
use crate::session::start_session;
use crate::session::PySession;
use crate::session::SessionSender;
use crate::tool::PythonTool;
use crate::PyAgentConfig;
use crate::PyTool;

/// In-process Rust agent service configured by Python.
#[pyclass(name = "AgentService")]
pub struct PyAgentService {
    commands: mpsc::UnboundedSender<ServiceCommand>,
}

type ServiceResult<T> = Result<T, String>;

struct ServiceInputs {
    config: PyAgentConfig,
    tools: Vec<PythonTool>,
    model: Option<ScriptedModel>,
    include_repository_context: bool,
}

enum ServiceCommand {
    Connect(oneshot::Sender<ServiceResult<SessionSender>>),
    Close(Option<oneshot::Sender<ServiceResult<()>>>),
}

struct ServiceActor {
    service: ReactAgentServiceHandle,
    rpc: AgentInProcessRpc,
    commands: mpsc::UnboundedReceiver<ServiceCommand>,
}

enum ServiceModel {
    Provider(Box<ModelProvider>),
    Scripted(ScriptedModel),
}

impl Llm for ServiceModel {
    fn model(&self) -> &str {
        match self {
            Self::Provider(model) => model.model(),
            Self::Scripted(model) => model.model(),
        }
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        match self {
            Self::Provider(model) => Either::Left(model.stream(request)),
            Self::Scripted(model) => Either::Right(model.stream(request)),
        }
    }
}

#[pymethods]
impl PyAgentService {
    /// Start an in-process service with an explicit model endpoint and tools.
    #[new]
    #[pyo3(signature = (
        tools: "collections.abc.Sequence[Tool]",
        config,
        *,
        model=None,
        include_repository_context=true,
    ) -> "AgentService")]
    fn new(
        py: Python<'_>,
        tools: Vec<PyRef<'_, PyTool>>,
        config: PyRef<'_, PyAgentConfig>,
        model: Option<PyRef<'_, PyScriptedModel>>,
        include_repository_context: bool,
    ) -> PyResult<Self> {
        let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let runtime_tools = tools
            .iter()
            .map(|tool| tool.runtime_tool(py, locals.clone_ref(py)))
            .collect();
        let config = config.clone();
        let (commands, command_receiver) = mpsc::unbounded_channel();
        start_service_thread(
            ServiceInputs {
                config,
                tools: runtime_tools,
                model: model.map(|model| model.runtime_model()),
                include_repository_context,
            },
            command_receiver,
        )?;
        Ok(Self { commands })
    }

    /// Attach a bidirectional protobuf session.
    #[pyo3(signature = () -> "collections.abc.Awaitable[Session]")]
    fn connect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let commands = self.commands.clone();
        future_into_py(py, async move {
            let (result, receiver) = oneshot::channel();
            send(&commands, ServiceCommand::Connect(result))?;
            Ok(PySession::new(receive(receiver).await?))
        })
    }

    /// Close the service and wait for active work to terminate.
    #[pyo3(signature = () -> "collections.abc.Awaitable[None]")]
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let commands = self.commands.clone();
        future_into_py(py, async move {
            let (result, receiver) = oneshot::channel();
            if commands.send(ServiceCommand::Close(Some(result))).is_err() {
                return Ok(());
            }
            receive(receiver).await
        })
    }
}

impl Drop for PyAgentService {
    fn drop(&mut self) {
        let _ = self.commands.send(ServiceCommand::Close(None));
    }
}

impl ServiceActor {
    async fn run(mut self) {
        while let Some(command) = self.commands.recv().await {
            match command {
                ServiceCommand::Connect(result) => {
                    let response = self
                        .rpc
                        .connect()
                        .await
                        .map(start_session)
                        .map_err(|error| error.to_string());
                    let _ = result.send(response);
                }
                ServiceCommand::Close(result) => {
                    let response = self
                        .service
                        .close()
                        .await
                        .map_err(|error| error.to_string());
                    if let Some(result) = result {
                        let _ = result.send(response);
                    }
                    return;
                }
            }
        }
        let _ = self.service.close().await;
    }
}

fn start_service_thread(
    inputs: ServiceInputs,
    commands: mpsc::UnboundedReceiver<ServiceCommand>,
) -> PyResult<()> {
    let (initialized, initialization) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("mirror-agent-service".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = initialized.send(Err(error.to_string()));
                    return;
                }
            };
            tokio::task::LocalSet::new().block_on(&runtime, async move {
                let service = match build_service(inputs).await {
                    Ok(service) => service,
                    Err(error) => {
                        let _ = initialized.send(Err(error));
                        return;
                    }
                };
                let _ = initialized.send(Ok(()));
                ServiceActor {
                    rpc: AgentInProcessRpc::new(service.clone()),
                    service,
                    commands,
                }
                .run()
                .await;
            });
        })
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    initialization
        .recv()
        .map_err(|_| PyRuntimeError::new_err("agent service failed during startup"))?
        .map_err(PyRuntimeError::new_err)
}

async fn build_service(inputs: ServiceInputs) -> ServiceResult<ReactAgentServiceHandle> {
    let config = inputs.config;
    let mut provider_config = config.provider;
    if inputs.model.is_some() {
        provider_config
            .context_window
            .get_or_insert(mirror_agent_core::providers::DEFAULT_CONTEXT_WINDOW);
    }
    let provider = ModelProvider::new(provider_config, None)
        .await
        .map_err(|error| error.to_string())?;
    let model = match inputs.model {
        Some(model) => ServiceModel::Scripted(model),
        None => ServiceModel::Provider(Box::new(provider.clone())),
    };
    validate_tools(inputs.tools.iter().map(|tool| tool as &dyn AnyTool))
        .map_err(|error| error.to_string())?;
    let prompt_config = SystemPromptConfig {
        custom_prompt: config.custom_system_prompt,
        include_project_instructions: inputs.include_repository_context,
        include_skills: inputs.include_repository_context,
        ..SystemPromptConfig::default()
    };
    let tools = inputs
        .tools
        .into_iter()
        .map(|tool| Arc::new(tool) as Arc<dyn AnyTool>)
        .collect();
    let executor = DirectToolExecutor::new(tools);
    let agent = ReactAgent::new(
        model,
        provider,
        None,
        &executor,
        prompt_config,
        config.agent,
    )
    .map_err(|error| error.to_string())?;
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(executor),
        Box::new(InMemoryStore::default()),
        config.conversation_id,
        AgentApprovalMode::Auto,
        Vec::new(),
        Vec::new(),
        None,
        tokio::sync::broadcast::channel(1024).0,
        None,
    )
    .await
    .map_err(|error| error.to_string())?;
    tokio::spawn(runtime);
    Ok(service)
}

fn send(commands: &mpsc::UnboundedSender<ServiceCommand>, command: ServiceCommand) -> PyResult<()> {
    commands
        .send(command)
        .map_err(|_| PyRuntimeError::new_err("agent service closed"))
}

async fn receive<T>(receiver: oneshot::Receiver<ServiceResult<T>>) -> PyResult<T> {
    receiver
        .await
        .map_err(|_| PyRuntimeError::new_err("agent service closed"))?
        .map_err(PyRuntimeError::new_err)
}
