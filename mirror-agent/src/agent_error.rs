//! Agent configuration and input validation failures.

/// A failure while configuring the agent or validating its input.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AgentError {
    #[error("invalid agent configuration: {0}")]
    InvalidConfiguration(String),
    #[error("agent input is inconsistent with its committed command: {0}")]
    InvalidInput(String),
}
