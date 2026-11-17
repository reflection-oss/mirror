//! Process-local approvals and tool execution context.

use mirror_protocol::AgentMessage;
use mirror_protocol::ApprovalSource;
use mirror_protocol::InferenceConfig;
use mirror_protocol::StreamContent;
use mirror_protocol::StreamReasoning;
use mirror_protocol::StreamRefusal;
use mirror_protocol::StreamToolCall;
use mirror_protocol::ToolCall;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;

use crate::web::WebProviderConfig;
use crate::TaskManagerHandle;

/// Live facilities available while an imperative agent executes tools.
#[derive(Clone)]
pub struct ExecutionContext {
    approval_mode: watch::Receiver<AgentApprovalMode>,
    inference_config: watch::Receiver<Option<InferenceConfig>>,
    web_provider: watch::Receiver<Option<WebProviderConfig>>,
    input: ExecutionContextInput,
    approval_hook: Option<ApprovalHook>,
    prompt_injection_policy: Option<String>,
    tasks: TaskManagerHandle,
}

/// Producer for live settings consumed by an active execution context.
#[derive(Clone)]
pub(crate) struct ExecutionContextInput {
    approval_mode: watch::Sender<AgentApprovalMode>,
    inference_config: watch::Sender<Option<InferenceConfig>>,
    web_provider: watch::Sender<Option<WebProviderConfig>>,
}

impl ExecutionContextInput {
    pub(crate) fn approval_mode(&self) -> AgentApprovalMode {
        *self.approval_mode.borrow()
    }

    pub(crate) fn set_approval_mode(&self, mode: AgentApprovalMode) {
        self.approval_mode.send_replace(mode);
    }

    pub(crate) fn set_web_provider(&self, provider: WebProviderConfig) {
        self.web_provider.send_replace(Some(provider));
    }

    pub(crate) fn set_inference_config(&self, inference: Option<InferenceConfig>) {
        self.inference_config.send_replace(inference);
    }
}

impl ExecutionContext {
    pub fn new(
        approval_mode: AgentApprovalMode,
        approval_hook: Option<ApprovalHook>,
        tasks: TaskManagerHandle,
        approval_parent: Option<&Self>,
        prompt_injection_policy: Option<String>,
    ) -> Self {
        let (approval_mode, approval_mode_receiver) = watch::channel(approval_mode);
        let (inference_config, inference_config_receiver) = watch::channel(None);
        let (web_provider, web_provider_receiver) = watch::channel(None);
        Self {
            approval_mode: approval_parent.map_or(approval_mode_receiver, |parent| {
                parent.approval_mode.clone()
            }),
            inference_config: inference_config_receiver,
            web_provider: approval_parent
                .map_or(web_provider_receiver, |parent| parent.web_provider.clone()),
            input: ExecutionContextInput {
                approval_mode,
                inference_config,
                web_provider,
            },
            approval_hook: approval_parent
                .map_or(approval_hook, |parent| parent.approval_hook.clone()),
            prompt_injection_policy: prompt_injection_policy
                .or_else(|| approval_parent.and_then(Self::prompt_injection_policy)),
            tasks,
        }
    }

    pub(crate) fn prompt_injection_policy(&self) -> Option<String> {
        self.prompt_injection_policy.clone()
    }

    pub(crate) fn inference_config(&self) -> Option<InferenceConfig> {
        self.inference_config.borrow().clone()
    }

    /// The live web-provider override, or `None` for the assembly's default.
    pub fn web_provider(&self) -> Option<WebProviderConfig> {
        *self.web_provider.borrow()
    }

    pub(crate) fn with_web_provider(mut self, provider: Option<WebProviderConfig>) -> Self {
        let (_, receiver) = watch::channel(provider.or(self.web_provider()));
        self.web_provider = receiver;
        self
    }

    pub fn approval_mode(&self) -> AgentApprovalMode {
        *self.approval_mode.borrow()
    }

    pub(crate) fn input(&self) -> ExecutionContextInput {
        self.input.clone()
    }

    pub fn tool_permissions(&self) -> ToolPermissions {
        ToolPermissions::new(self.approval_mode().skips_approval())
    }

    pub fn tasks(&self) -> &TaskManagerHandle {
        &self.tasks
    }

    pub async fn request_approval(&self, call: &ToolCall) -> Result<bool, ExecutionContextError> {
        Ok(self
            .approval_decision(call, ApprovalKind::Ordinary)
            .await?
            .acknowledge()
            .approved)
    }

    pub(crate) async fn approval_decision(
        &self,
        call: &ToolCall,
        kind: ApprovalKind,
    ) -> Result<ApprovalDecision, ExecutionContextError> {
        match &self.approval_hook {
            Some(hook) => hook.decision(call, kind).await,
            None if kind == ApprovalKind::Ordinary && self.approval_mode().skips_approval() => {
                Ok(ApprovalDecision::new(true, ApprovalSource::Automatic))
            }
            None => Err(ExecutionContextError::Approval(
                "approval channel unavailable".to_owned(),
            )),
        }
    }
}

/// A live agent update published while a session changes.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AgentEvent {
    Entries(Vec<AgentMessage>),
    StreamContent(StreamContent),
    StreamReasoning(StreamReasoning),
    StreamRefusal(StreamRefusal),
    StreamToolCall(StreamToolCall),
    StreamFinal {
        entry: Box<AgentMessage>,
        tokens_per_second: Option<f64>,
    },
    StreamUsage(mirror_protocol::StreamUsage),
    StreamAborted,
    CompactionStarted,
    ContextUsage(mirror_protocol::ContextUsage),
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExecutionContextError {
    #[error("approval failed: {0}")]
    Approval(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ApprovalKind {
    Ordinary,
    AlwaysAsk,
}

pub(crate) struct ApprovalRequest {
    pub call: ToolCall,
    pub kind: ApprovalKind,
    pub response: oneshot::Sender<ApprovalDecision>,
}

/// Permission whose acknowledgment is released only after the caller commits it.
#[derive(Debug)]
pub(crate) struct ApprovalDecision {
    pub approved: bool,
    pub source: ApprovalSource,
    pub reason: Option<String>,
    pub committed: Option<oneshot::Sender<()>>,
}

impl ApprovalDecision {
    pub fn new(approved: bool, source: ApprovalSource) -> Self {
        Self {
            approved,
            source,
            reason: None,
            committed: None,
        }
    }

    pub fn denial_message(&self) -> Option<&str> {
        self.reason
            .as_deref()
            .filter(|reason| !self.approved && !reason.trim().is_empty())
    }

    pub fn acknowledge(mut self) -> Self {
        if let Some(committed) = self.committed.take() {
            let _ = committed.send(());
        }
        self
    }
}

#[derive(Clone)]
pub struct ApprovalHook {
    requests: mpsc::UnboundedSender<ApprovalRequest>,
}

impl ApprovalHook {
    pub(crate) fn channel() -> (Self, mpsc::UnboundedReceiver<ApprovalRequest>) {
        let (requests, receiver) = mpsc::unbounded_channel();
        (Self { requests }, receiver)
    }

    pub async fn approve(&self, call: &ToolCall) -> Result<bool, ExecutionContextError> {
        Ok(self
            .decision(call, ApprovalKind::Ordinary)
            .await?
            .acknowledge()
            .approved)
    }

    async fn decision(
        &self,
        call: &ToolCall,
        kind: ApprovalKind,
    ) -> Result<ApprovalDecision, ExecutionContextError> {
        let (response, receiver) = oneshot::channel();
        self.requests
            .send(ApprovalRequest {
                call: call.clone(),
                kind,
                response,
            })
            .map_err(|_| ExecutionContextError::Approval("approval channel closed".to_string()))?;
        receiver
            .await
            .map_err(|_| ExecutionContextError::Approval("approval channel closed".to_string()))
    }
}

/// Tool approval policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AgentApprovalMode {
    Auto,
    Hitl,
    Bypass,
}

impl AgentApprovalMode {
    pub fn skips_approval(self) -> bool {
        matches!(self, Self::Auto | Self::Bypass)
    }
}

/// Immutable permissions inherited by a tool batch.
#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct ToolPermissions {
    allow_outside_workspace: bool,
}

impl ToolPermissions {
    pub fn new(allow_outside_workspace: bool) -> Self {
        Self {
            allow_outside_workspace,
        }
    }

    pub fn allow_outside_workspace(&self) -> bool {
        self.allow_outside_workspace
    }
}

#[cfg(test)]
#[path = "execution_context_test.rs"]
mod tests;
