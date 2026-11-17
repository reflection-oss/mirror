//! Shared inputs for configuring a compactor.

/// Token limits used to construct a compactor.
#[derive(Debug, Clone)]
pub struct CompactorConfig {
    pub context_window: usize,
    pub compaction_threshold: Option<usize>,
    pub max_tokens: usize,
    pub tool_result_max_chars: usize,
    pub prompt_injection_policy: Option<String>,
}
