//! Python-owned agent construction settings.

use mirror_agent_core::providers::ModelApi;
use mirror_agent_core::providers::ModelProviderConfig;
use mirror_agent_core::react::ReactAgentConfig;
use pyo3::prelude::*;

/// Configuration for one in-process Rust agent service.
#[pyclass(name = "AgentConfig", frozen)]
#[derive(Debug, Clone)]
pub struct PyAgentConfig {
    pub(crate) provider: ModelProviderConfig,
    pub(crate) agent: ReactAgentConfig,
    pub(crate) conversation_id: String,
    pub(crate) custom_system_prompt: Option<String>,
}

#[pymethods]
impl PyAgentConfig {
    /// Create the settings required by an eval-hosted agent.
    #[new]
    #[pyo3(signature = (
        model_name,
        conversation_id,
        *,
        base_url,
        custom_system_prompt=None,
        tool_result_max_chars=None,
        max_iterations=None,
        context_window=None,
    ) -> "AgentConfig")]
    fn new(
        model_name: String,
        conversation_id: String,
        base_url: String,
        custom_system_prompt: Option<String>,
        tool_result_max_chars: Option<usize>,
        max_iterations: Option<usize>,
        context_window: Option<usize>,
    ) -> Self {
        Self {
            provider: ModelProviderConfig {
                context_window,
                api: ModelApi::ChatCompletions,
                model: model_name,
                api_key: None,
                base_url: Some(base_url),
                auth_path: None,
                prompt_cache_key: conversation_id.clone(),
                request_timeout: Some(mirror_agent_core::providers::DEFAULT_REQUEST_TIMEOUT),
            },
            agent: ReactAgentConfig {
                max_iterations: max_iterations.unwrap_or(usize::MAX),
                max_exceptions: 3,
                compaction_threshold: None,
                disable_compaction: false,
                max_tokens: 64_000,
                tool_result_max_chars: tool_result_max_chars.unwrap_or(2_000),
            },
            conversation_id,
            custom_system_prompt,
        }
    }
}
