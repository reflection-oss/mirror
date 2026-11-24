//! Runtime-owned reviewers and failure propagation through nested execution.

use std::sync::Arc;

use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use mirror_protocol::ToolCall;
use tokio::sync::watch;

use super::{PreToolError, PreToolHook};

/// Internal execution state; callers configure a list of `PreToolHook` instead.
#[derive(Debug, Clone, Default)]
pub(crate) struct PreToolHooks {
    commands: Arc<[PreToolHook]>,
    failure: Option<watch::Sender<Option<PreToolError>>>,
}

impl PreToolHooks {
    pub fn new(commands: Vec<PreToolHook>, parent: Option<&Self>) -> Self {
        let mut inherited = parent.map_or_else(Vec::new, |parent| parent.commands.to_vec());
        inherited.extend(commands);
        Self {
            commands: inherited.into(),
            failure: None,
        }
    }

    /// Share failures within a batch and its nested tools, not with later turns.
    pub fn scoped(&self) -> Self {
        let mut hooks = self.clone();
        hooks.failure.get_or_insert_with(|| watch::channel(None).0);
        hooks
    }

    pub fn check(&self) -> Result<(), PreToolError> {
        match self
            .failure
            .as_ref()
            .and_then(|failure| failure.borrow().clone())
        {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub async fn review(&self, call: &ToolCall) -> Result<Option<String>, PreToolError> {
        self.check()?;
        let result = self.run(call).await;
        if let (Err(error), Some(failure)) = (&result, &self.failure) {
            // Nested tools may turn errors into output; the enclosing batch must still stop.
            failure.send_replace(Some(error.clone()));
        }
        result
    }

    async fn run(&self, call: &ToolCall) -> Result<Option<String>, PreToolError> {
        let mut pending = self
            .commands
            .iter()
            .map(|hook| hook.review(call))
            .collect::<FuturesUnordered<_>>();
        let mut denial = None;
        while let Some(result) = pending.next().await {
            if let Some(reason) = result? {
                denial.get_or_insert(reason);
            }
        }
        Ok(denial)
    }
}

#[cfg(test)]
#[path = "pre_tool_hooks_test.rs"]
mod tests;
