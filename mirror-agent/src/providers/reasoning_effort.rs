//! The values a reasoning effort setting may take.

/// Named reasoning effort levels accepted by the harness; provider support varies.
pub const REASONING_EFFORT_LEVELS: [&str; 7] =
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

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
}
