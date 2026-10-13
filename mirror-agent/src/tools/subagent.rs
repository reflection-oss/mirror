//! Isolated child-agent execution through the task runtime.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use crate::tools::ToolOutput;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use mirror_protocol::AssistantMessage;
use mirror_protocol::ChatRecord;
use mirror_protocol::UserMessage;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::validate_tools;
use super::AnyTool;
use super::Tool;
use super::ToolError;
use super::{DirectToolExecutor, RemoteToolExecutor, ToolExecutor};
use crate::providers::ModelProvider;
use crate::providers::ModelProviderConfig;
use crate::providers::ModelRequestGuard;
use crate::react::InMemoryStore;
use crate::react::ReactAgent;
use crate::react::ReactAgentConfig;
use crate::system_prompt::load_skills;
use crate::system_prompt::EnvironmentInfo;
use crate::system_prompt::ResolvedInstructions;
use crate::system_prompt::SystemPromptConfig;
use crate::AgentEvent;
use crate::ExecutionContext;
use crate::ReactAgentService;
use crate::ReactAgentServiceHandle;
use crate::ReactServiceEvent;
use crate::TaskContext;
use crate::TaskResolution;

/// Run isolated agent tasks with non-recursive tools.
#[derive(Clone)]
pub struct SubagentTool {
    workspace_root: PathBuf,
    working_directory: PathBuf,
    parent_conversation_id: String,
    provider: ModelProviderConfig,
    request_guard: Option<Arc<dyn ModelRequestGuard>>,
    agent: ReactAgentConfig,
    environment: Option<EnvironmentInfo>,
    tools: Vec<Arc<dyn AnyTool>>,
    resolved: Option<ResolvedInstructions>,
}

/// Arguments accepted by the subagent tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SubagentArgs {
    pub prompt: String,
}

impl SubagentTool {
    /// Create a task-backed subagent sharing the supplied tools.
    pub fn new(
        workspace_root: impl AsRef<Path>,
        parent_conversation_id: impl Into<String>,
        provider: ModelProviderConfig,
        agent: ReactAgentConfig,
        environment: Option<EnvironmentInfo>,
        tools: Vec<Arc<dyn AnyTool>>,
    ) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: super::canonical_workspace(&workspace_root)?,
            working_directory: workspace_root.as_ref().to_path_buf(),
            parent_conversation_id: parent_conversation_id.into(),
            provider,
            request_guard: None,
            agent,
            environment,
            tools,
            resolved: None,
        })
    }

    /// Supply hosted prompt context without discovering files on the server.
    #[must_use]
    pub fn with_resolved_context(mut self, resolved: ResolvedInstructions) -> Self {
        self.resolved = Some(resolved);
        self
    }

    /// Apply the parent's authorization to every subagent model request.
    #[must_use]
    pub fn with_request_guard(mut self, guard: Arc<dyn ModelRequestGuard>) -> Self {
        self.request_guard = Some(guard);
        self
    }
}

impl Tool for SubagentTool {
    type Args = SubagentArgs;

    fn name(&self) -> &str {
        "subagent"
    }

    fn description(&self) -> &str {
        "Run an isolated agent task and return its final answer."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([(
                "prompt".to_string(),
                serde_json::json!({"type": "string", "description": "Task for the subagent."}),
            )]),
            &["prompt"],
        )
    }

    fn supports_remote(&self) -> bool {
        self.tools.iter().all(|tool| tool.supports_remote())
    }

    async fn run_remote(
        &self,
        args: SubagentArgs,
        context: &ExecutionContext,
        remote: &RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        let executor = remote.with_tools(self.tools.clone())?;
        let prompt = SystemPromptConfig {
            prompt_injection_policy: context.prompt_injection_policy(),
            include_project_instructions: false,
            include_skills: false,
            ..SystemPromptConfig::default()
        };
        let task_id = self
            .spawn_executor(args.prompt, Box::new(executor), prompt, context)
            .await?;
        resolve_task(&task_id, context).await
    }

    async fn run(
        &self,
        args: SubagentArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let task_id = self.spawn(args.prompt, self.tools.clone(), context).await?;
        resolve_task(&task_id, context).await
    }
}

impl SubagentTool {
    /// Start a detached child agent over `tools` with `prompt` as its task and
    /// return the task id; its final answer arrives as that task's result.
    ///
    /// The model-facing tool uses this with its own tools. Harness features
    /// that need a bounded child conversation pass their own tool list.
    pub async fn spawn(
        &self,
        prompt: String,
        tools: Vec<Arc<dyn AnyTool>>,
        context: &ExecutionContext,
    ) -> Result<String, ToolError> {
        validate_tools(tools.iter().map(|tool| tool.as_ref()))?;
        let prompt_config = SystemPromptConfig {
            prompt_injection_policy: context.prompt_injection_policy(),
            working_directory: Some(self.working_directory.clone()),
            environment: self.environment.clone(),
            resolved: self.resolved.clone(),
            ..SystemPromptConfig::default()
        };
        self.spawn_executor(
            prompt,
            Box::new(DirectToolExecutor::new(tools)),
            prompt_config,
            context,
        )
        .await
    }

    async fn spawn_executor(
        &self,
        prompt: String,
        tool_executor: Box<dyn ToolExecutor>,
        system_prompt: SystemPromptConfig,
        context: &ExecutionContext,
    ) -> Result<String, ToolError> {
        let tasks = context.tasks();
        let approval_parent = context.clone();
        let workspace_root = self.workspace_root.clone();
        let parent_conversation_id = self.parent_conversation_id.clone();
        let provider = self.provider.clone();
        let request_guard = self.request_guard.clone();
        let inference = context.inference_config().ok_or_else(|| {
            ToolError::Execution("active turn has no inference config".to_string())
        })?;
        let agent_config = self.agent;
        tasks
            .spawn(
                "subagent",
                Some("Starting…".to_string()),
                crate::TaskBehavior::Detach,
                move |task_context| async move {
                    let result: Result<ToolOutput, ToolError> = async {
                        let conversation_id = format!(
                            "{parent_conversation_id}-subagent-{}",
                            task_context.task_id()
                        );
                        let provider = provider
                            .for_inference(&inference)
                            .map_err(|error| ToolError::Execution(error.to_string()))?;
                        let model = ModelProvider::new(
                            ModelProviderConfig {
                                prompt_cache_key: conversation_id.clone(),
                                ..provider
                            },
                            request_guard,
                        )
                        .await
                        .map_err(|error| ToolError::Execution(error.to_string()))?;
                        let skills = if system_prompt.include_skills {
                            system_prompt.resolved.as_ref().map_or_else(
                                || load_skills(&workspace_root, None),
                                |resolved| resolved.skills.clone(),
                            )
                        } else {
                            Vec::new()
                        };
                        let agent = ReactAgent::new(
                            model.clone(),
                            model,
                            None,
                            tool_executor.as_ref(),
                            system_prompt,
                            agent_config,
                        )
                        .map_err(|error| ToolError::Execution(error.to_string()))?;
                        let (service, runtime) = ReactAgentService::start(
                            agent,
                            tool_executor,
                            Box::new(InMemoryStore::default()),
                            conversation_id,
                            approval_parent.approval_mode(),
                            skills,
                            Vec::new(),
                            None,
                            tokio::sync::broadcast::channel(1024).0,
                            Some(&approval_parent),
                        )
                        .await
                        .map_err(|error| ToolError::Execution(error.to_string()))?;
                        task_context
                            .spawn(runtime)
                            .map_err(|error| ToolError::Execution(error.to_string()))?;
                        let mut events = service.events();
                        service
                            .set_inference_config(inference)
                            .await
                            .map_err(|error| ToolError::Execution(error.to_string()))?;
                        let result = match service.execute(AgentMessage {
                            id: "subagent-user".to_string(),
                            payload: Some(agent_message::Payload::Chat(ChatRecord {
                                message: Some(chat_record::Message::User(UserMessage {
                                    checkpoint: None,
                                    content: prompt,
                                })),
                                success: true,
                                ..ChatRecord::default()
                            })),
                            ..AgentMessage::default()
                        }) {
                            Ok(_) => {
                                collect_subagent_result(&service, &mut events, task_context).await
                            }
                            Err(error) => ToolOutput::error(error.to_string()),
                        };
                        let _ = service.close().await;
                        Ok(result)
                    }
                    .await;
                    result.unwrap_or_else(|error| ToolOutput::error(error.to_string()))
                },
            )
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))
    }
}

async fn resolve_task(task_id: &str, context: &ExecutionContext) -> Result<ToolOutput, ToolError> {
    match context
        .tasks()
        .resolve(task_id)
        .await
        .map_err(|error| ToolError::Execution(error.to_string()))?
    {
        TaskResolution::Detached => Ok(ToolOutput::success(format!(
            "Subagent task {task_id} started. Its final answer will be delivered automatically."
        ))),
        TaskResolution::Finished(result) => Ok(result),
    }
}

async fn collect_subagent_result(
    service: &ReactAgentServiceHandle,
    events: &mut tokio::sync::broadcast::Receiver<ReactServiceEvent>,
    task: TaskContext,
) -> ToolOutput {
    let mut display = None;
    let mut has_trace = false;
    let mut turn_error = None;
    loop {
        match events.recv().await {
            Ok(ReactServiceEvent::TurnCompleted) => break,
            Ok(ReactServiceEvent::TurnError(error)) => turn_error = Some(error),
            Ok(ReactServiceEvent::Agent(AgentEvent::ContextUsage { .. })) if !has_trace => {
                update_display(&task, &mut display, "Thinking…".to_string());
            }
            Ok(ReactServiceEvent::Agent(AgentEvent::CompactionStarted)) if !has_trace => {
                update_display(&task, &mut display, "Compacting context…".to_string());
            }
            Ok(ReactServiceEvent::Agent(AgentEvent::StreamFinal { entry, .. })) => {
                if let Some(next) = record_entry(&task, &entry) {
                    has_trace = true;
                    update_display(&task, &mut display, next);
                }
            }
            Ok(ReactServiceEvent::Agent(AgentEvent::Entries(entries))) => {
                for entry in &entries {
                    if let Some(next) = record_entry(&task, entry) {
                        has_trace = true;
                        update_display(&task, &mut display, next);
                    }
                }
            }
            Ok(_) => {}
            Err(error) => {
                return ToolOutput::error(format!("subagent event stream failed: {error}"))
            }
        }
    }
    if let Some(error) = turn_error {
        return ToolOutput::error(error);
    }
    let entries = match service.messages().await {
        Ok(entries) => entries,
        Err(error) => return ToolOutput::error(error.to_string()),
    };
    final_answer(&entries)
        .map(ToolOutput::success)
        .unwrap_or_else(|| ToolOutput::error("subagent did not produce a final answer"))
}

fn final_answer(entries: &[AgentMessage]) -> Option<String> {
    let Some(AgentMessage {
        payload:
            Some(agent_message::Payload::Chat(ChatRecord {
                message:
                    Some(chat_record::Message::Assistant(AssistantMessage {
                        content, refusal, ..
                    })),
                ..
            })),
        ..
    }) = entries.last()
    else {
        return None;
    };
    content
        .clone()
        .filter(|content| !content.is_empty())
        .or_else(|| refusal.clone().filter(|refusal| !refusal.is_empty()))
}

fn update_display(task: &TaskContext, current: &mut Option<String>, next: String) {
    if current.as_deref() == Some(next.as_str()) {
        return;
    }
    *current = Some(next.clone());
    task.update(Some(next));
}

fn record_entry(task: &TaskContext, entry: &AgentMessage) -> Option<String> {
    match &entry.payload {
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::System(_)),
            ..
        }))
        | Some(agent_message::Payload::TaskResult(_))
        | Some(agent_message::Payload::Extension(_)) => None,
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(user)),
            ..
        })) => {
            task.log("user", &user.content);
            None
        }
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Tool(tool)),
            ..
        })) => {
            let name = tool.name.as_deref().unwrap_or("tool");
            task.log(format!("tool:{name}"), &tool.content);
            Some(format!("{name}: {}", summary(&tool.content)))
        }
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::Assistant(assistant)),
            ..
        })) => {
            let mut log = [
                assistant.reasoning_content.as_deref(),
                assistant.content.as_deref(),
                assistant.refusal.as_deref(),
            ]
            .into_iter()
            .flatten()
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
            log.extend(assistant.tool_calls.iter().filter_map(|call| {
                let function = call.function.as_ref()?;
                Some(format!("{}({})", function.name, function.arguments))
            }));
            if !log.is_empty() {
                task.log("assistant", log.join("\n\n"));
            }
            let calls = assistant
                .tool_calls
                .iter()
                .filter_map(|call| call.function.as_ref())
                .map(|function| {
                    let arguments = one_line(&function.arguments);
                    if arguments.is_empty() {
                        function.name.clone()
                    } else {
                        format!("{}({})", function.name, summary(&arguments))
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            if !calls.is_empty() {
                return Some(format!("Running {calls}"));
            }
            assistant
                .content
                .as_deref()
                .filter(|answer| !answer.is_empty())
                .or_else(|| {
                    assistant
                        .refusal
                        .as_deref()
                        .filter(|answer| !answer.is_empty())
                })
                .map(|answer| format!("Answer: {}", summary(answer)))
        }
        Some(agent_message::Payload::Log(log)) => {
            task.log(format!("log:{}", log.level), &log.text);
            None
        }
        Some(agent_message::Payload::ContextWindow(window)) => {
            if let Some(summary) = window
                .compaction
                .as_ref()
                .and_then(|compaction| compaction.summary.as_deref())
            {
                task.log("compaction", summary);
            }
            None
        }
        Some(agent_message::Payload::Chat(ChatRecord { message: None, .. }))
        | Some(agent_message::Payload::ToolApproval(_))
        | None => None,
    }
}

fn one_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn summary(value: &str) -> String {
    let value = one_line(value);
    if value.chars().count() <= 160 {
        return value;
    }
    let mut result = value.chars().take(159).collect::<String>();
    result.push('…');
    result
}

#[cfg(test)]
#[path = "subagent_test.rs"]
mod tests;
