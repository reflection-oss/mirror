//! The values a reasoning effort setting may take.

use super::ModelApi;

/// Named reasoning effort levels accepted by the harness; provider support varies.
pub const REASONING_EFFORT_LEVELS: [&str; 6] = ["none", "low", "medium", "high", "xhigh", "max"];

/// Whether `value` is a level name or a non-negative integer budget.
///
/// A budget is forwarded as an integer to endpoints that accept one; each
/// provider decides how to spell either form on its wire.
pub fn is_reasoning_effort(value: &str) -> bool {
    REASONING_EFFORT_LEVELS.contains(&value) || value.parse::<u64>().is_ok()
}

/// The error text for a value `is_reasoning_effort` refused.
pub fn reasoning_effort_expectation() -> String {
    format!(
        "expected one of {} or a non-negative integer",
        REASONING_EFFORT_LEVELS.join(", ")
    )
}

/// The thinking toggles a llama.cpp chat template can expose besides the named levels.
const LLAMA_CPP_THINKING_TOGGLES: [&str; 2] = ["on", "off"];

impl ModelApi {
    /// Whether `value` is a reasoning effort this protocol can carry.
    pub fn accepts_reasoning_effort(self, value: &str) -> bool {
        match self {
            Self::LlamaCpp => {
                LLAMA_CPP_THINKING_TOGGLES.contains(&value)
                    || REASONING_EFFORT_LEVELS.contains(&value)
            }
            _ => is_reasoning_effort(value),
        }
    }

    /// The error text for a value `accepts_reasoning_effort` refused.
    pub fn reasoning_effort_expectation(self) -> String {
        match self {
            Self::LlamaCpp => format!(
                "expected one of {}, {}",
                REASONING_EFFORT_LEVELS.join(", "),
                LLAMA_CPP_THINKING_TOGGLES.join(", ")
            ),
            _ => reasoning_effort_expectation(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_levels_and_budgets_only() {
        for level in REASONING_EFFORT_LEVELS {
            assert!(is_reasoning_effort(level), "{level}");
        }
        assert!(is_reasoning_effort("0"));
        assert!(is_reasoning_effort("512"));
        assert!(!is_reasoning_effort("-1"));
        assert!(!is_reasoning_effort("lots"));
        assert!(!is_reasoning_effort("64 tokens"));
        assert!(!is_reasoning_effort(""));
    }

    #[test]
    fn llama_cpp_takes_thinking_toggles_and_no_budgets() {
        assert!(ModelApi::LlamaCpp.accepts_reasoning_effort("on"));
        assert!(!ModelApi::LlamaCpp.accepts_reasoning_effort("512"));
        assert!(!ModelApi::ChatCompletions.accepts_reasoning_effort("on"));
    }
}
