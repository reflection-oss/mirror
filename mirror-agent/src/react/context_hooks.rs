//! Hooks that follow what the conversation commits.

use std::sync::Arc;

use futures_util::future::BoxFuture;
use mirror_protocol::AgentMessage;

use crate::tools::AnyTool;
use crate::ExecutionContext;

/// The hooks a context runs at fixed points of the conversation: after the
/// user's input is committed, after tool replies are committed, and when a
/// turn ends.
///
/// `history` is the active context window including the entries just
/// committed. `on_input` and `on_tool_replies` return records the context
/// commits right after what triggered them, so a hook that must not repeat
/// itself reads what it already added from `history` instead of keeping
/// state of its own. A fork's context has no hooks.
pub trait ContextHooks: Send + Sync {
    /// Records that follow `input`, the user's new messages, once committed.
    fn on_input<'a>(
        &'a self,
        history: &'a [AgentMessage],
        input: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        let _ = (history, input);
        Box::pin(async { Vec::new() })
    }

    /// Records that follow `replies`, one tool batch's results, once committed.
    fn on_tool_replies<'a>(
        &'a self,
        history: &'a [AgentMessage],
        replies: &'a [AgentMessage],
    ) -> BoxFuture<'a, Vec<AgentMessage>> {
        let _ = (history, replies);
        Box::pin(async { Vec::new() })
    }

    /// A turn finished without an error or an interruption.
    fn on_turn_end<'a>(
        &'a self,
        history: &'a [AgentMessage],
        turn: TurnEnd<'a>,
    ) -> BoxFuture<'a, ()> {
        let _ = (history, turn);
        Box::pin(async {})
    }
}

/// What a hook may use when a turn ends: the session's execution
/// facilities (tasks, approval mode) and its tools.
#[derive(Clone, Copy)]
pub struct TurnEnd<'a> {
    pub execution: &'a ExecutionContext,
    pub tools: &'a [Arc<dyn AnyTool>],
}
