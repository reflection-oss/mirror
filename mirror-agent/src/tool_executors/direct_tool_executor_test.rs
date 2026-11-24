use std::sync::Arc;
use std::time::Duration;

use mirror_protocol::FunctionCall;
use pretty_assertions::assert_eq;
use serde_json::Map;
use tokio::sync::mpsc;

use super::*;
use crate::tools::ToolDefinition;

/// Sleeps for `delay`, then reports its name on `finished`.
struct LoggedTool {
    name: &'static str,
    concurrency_safe: bool,
    delay: Duration,
    finished: mpsc::UnboundedSender<&'static str>,
}

impl AnyTool for LoggedTool {
    fn name(&self) -> &str {
        self.name
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.to_string(),
            description: String::new(),
            parameters: Map::new(),
        }
    }

    fn is_concurrency_safe(&self) -> bool {
        self.concurrency_safe
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            tokio::time::sleep(self.delay).await;
            self.finished
                .send(self.name)
                .expect("finish order receiver");
            ToolOutput::success(self.name)
        })
    }
}

struct PanickingTool;

impl AnyTool for PanickingTool {
    fn name(&self) -> &str {
        "explode"
    }

    fn needs_approval(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "explode".to_string(),
            description: String::new(),
            parameters: Map::new(),
        }
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }

    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        _context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async { panic!("tool exploded") })
    }
}

struct SwallowingTool {
    inner: DirectToolExecutor,
}

impl AnyTool for SwallowingTool {
    fn name(&self) -> &str {
        "wrapper"
    }
    fn needs_approval(&self) -> bool {
        false
    }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().into(),
            description: String::new(),
            parameters: Map::new(),
        }
    }
    fn execute<'a>(
        &'a self,
        _arguments: &'a str,
        context: &'a ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = ToolOutput> + Send + 'a>> {
        Box::pin(async move {
            let _ = self
                .inner
                .call_tool("nested".into(), "{}".into(), context.clone())
                .await;
            ToolOutput::success("ignored nested failure")
        })
    }
}

#[tokio::test]
async fn pre_tool_failure_in_a_nested_call_prevents_queued_side_effects() {
    let workspace = tempfile::tempdir().expect("workspace");
    let path = workspace.path().join("hooks.json");
    std::fs::write(&path, r#"{"hooks":{"PreToolUse":[{"matcher":"^nested$","hooks":[{"type":"command","command":"exit 1"}]}]}}"#).expect("hook config");
    let hooks = crate::hooks::PreToolHook::load(&path, workspace.path())
        .await
        .expect("hooks");
    let context = ExecutionContext::new(
        crate::AgentApprovalMode::Auto,
        None,
        crate::TaskManagerHandle::closed(),
        None,
        None,
        hooks,
    )
    .for_tool_execution();
    let (finished, mut effects) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        Arc::new(SwallowingTool {
            inner: DirectToolExecutor::new(vec![logged_tool(
                &finished,
                "nested",
                false,
                Duration::ZERO,
            )]),
        }),
        logged_tool(&finished, "later", false, Duration::ZERO),
    ]);
    let outputs = executor
        .execute(
            Uuid::new_v4(),
            &[tool_call("wrapper"), tool_call("later")],
            &[true, true],
            &context,
        )
        .await
        .expect("batch");
    assert_eq!(outputs[1].content, "Tool error: pre-tool hook failed");
    assert_matches::assert_matches!(
        context.check_pre_tool_hooks(),
        Err(crate::hooks::PreToolError::Failed)
    );
    assert!(
        effects.try_recv().is_err(),
        "no tool side effects after reviewer failure"
    );
}

fn logged_tool(
    finished: &mpsc::UnboundedSender<&'static str>,
    name: &'static str,
    concurrency_safe: bool,
    delay: Duration,
) -> Arc<dyn AnyTool> {
    Arc::new(LoggedTool {
        name,
        concurrency_safe,
        delay,
        finished: finished.clone(),
    })
}

fn tool_call(name: &str) -> ToolCall {
    ToolCall {
        id: format!("call-{name}"),
        r#type: "function".to_string(),
        function: Some(FunctionCall {
            name: name.to_string(),
            arguments: "{}".to_string(),
        }),
        ..ToolCall::default()
    }
}

/// The content and success of an output, without its measured duration.
fn outcome(output: &ToolOutput) -> (&str, bool) {
    (&output.content, output.success)
}

#[tokio::test(start_paused = true)]
async fn execute_overlaps_safe_calls_and_runs_unsafe_calls_alone() {
    let (finished, mut finish_order) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        logged_tool(&finished, "slow_read", true, Duration::from_secs(5)),
        logged_tool(&finished, "fast_read", true, Duration::ZERO),
        logged_tool(&finished, "write", false, Duration::from_secs(1)),
        logged_tool(&finished, "later_read", true, Duration::ZERO),
    ]);
    let calls = ["slow_read", "fast_read", "write", "later_read"].map(tool_call);

    let outputs = executor
        .execute(
            Uuid::new_v4(),
            &calls,
            &[true; 4],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");

    drop(executor);
    drop(finished);
    let mut order = Vec::new();
    finish_order.recv_many(&mut order, calls.len()).await;
    assert_eq!(order, ["fast_read", "slow_read", "write", "later_read"]);
    assert_eq!(
        outputs.iter().map(outcome).collect::<Vec<_>>(),
        [
            ("slow_read", true),
            ("fast_read", true),
            ("write", true),
            ("later_read", true),
        ]
    );
}

#[tokio::test(start_paused = true)]
async fn execute_starts_a_queued_call_when_any_running_call_finishes() {
    let (finished, _finish_order) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        logged_tool(&finished, "slow_read", true, Duration::from_secs(30)),
        logged_tool(&finished, "fast_read", true, Duration::ZERO),
    ]);
    let calls = [
        "slow_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "fast_read",
        "slow_read",
    ]
    .map(tool_call);
    let started = tokio::time::Instant::now();

    executor
        .execute(
            Uuid::new_v4(),
            &calls,
            &[true; 11],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");

    assert_eq!(started.elapsed(), Duration::from_secs(30));
}

#[tokio::test]
async fn execute_turns_a_panic_into_that_call_s_error() {
    let (finished, _finish_order) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(vec![
        Arc::new(PanickingTool),
        logged_tool(&finished, "read", true, Duration::ZERO),
    ]);
    let calls = ["explode", "read"].map(tool_call);

    let outputs = executor
        .execute(
            Uuid::new_v4(),
            &calls,
            &[true; 2],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");

    assert_eq!(
        outputs.iter().map(outcome).collect::<Vec<_>>(),
        [("Tool error: tool exploded", false), ("read", true)]
    );
}

#[tokio::test]
async fn tool_access_is_rechecked_when_an_already_queued_call_starts() {
    let executor = DirectToolExecutor::new(vec![Arc::new(PanickingTool)]);
    let context = ExecutionContext::default();
    let execution = executor.execute(Uuid::new_v4(), &[tool_call("explode")], &[true], &context);
    context
        .input()
        .set_disabled_tools(["explode".to_string()].into());

    let outputs = execution.await.expect("disabled batch");
    assert!(!outputs[0].success);
    assert_eq!(
        outputs[0].content,
        "Tool error: Tool disabled for this chat: explode"
    );
    assert!(executor.definitions_for(&context).is_empty());
    assert!(!executor.inventory(&context, false).await.tools[0].enabled);
    let output = executor
        .call_tool("explode".into(), "{}".into(), context)
        .await
        .expect("direct call");
    assert_eq!(
        output.content,
        "Tool error: Tool disabled for this chat: explode"
    );
}
