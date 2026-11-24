use std::path::Path;
use std::sync::Arc;

use assert_matches::assert_matches;
use mirror_protocol::{FunctionCall, ToolCall};
use pretty_assertions::assert_eq;
use serde_json::json;
use tokio::sync::{broadcast, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use super::*;
use crate::compaction::SummaryCompactorFactory;
use crate::providers::{Completion, CompletionRequest, Llm, LlmError, StreamEvent};
use crate::react::{InMemoryStore, ReactAgent, ReactAgentConfig};
use crate::system_prompt::SystemPromptConfig;
use crate::tools::{BashTool, DirectToolExecutor, ToolExecutor, ToolOutput};
use crate::{AgentApprovalMode, AgentRpc, ExecutionContext, ReactAgentService, TaskManagerHandle};

async fn load_hooks(workspace: &Path, groups: serde_json::Value) -> PreToolHooks {
    let path = workspace.join("hooks.json");
    std::fs::write(&path, json!({"hooks": {"PreToolUse": groups}}).to_string())
        .expect("write trusted hooks");
    PreToolHooks::new(
        PreToolHook::load(&path, workspace)
            .await
            .expect("load hooks"),
        None,
    )
}

fn bash_call() -> ToolCall {
    ToolCall {
        id: "call-1".into(),
        function: Some(FunctionCall {
            name: "bash".into(),
            arguments: r#"{"command":"touch should-not-exist"}"#.into(),
        }),
        ..ToolCall::default()
    }
}

#[tokio::test]
async fn pre_tool_hooks_receive_input_and_keep_the_loaded_configuration() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(workspace.path(), json!([
        {"matcher": "^read$", "hooks": [{"type": "command", "command": "exit 1"}]},
        {"matcher": "^Bash$", "hooks": [{"type": "command", "command": "cat > input.json; printf 'Outside scope.' >&2; exit 2"}]}
    ])).await;
    std::fs::remove_file(workspace.path().join("hooks.json")).expect("remove configuration");

    assert_eq!(
        hooks.review(&bash_call()).await,
        Ok(Some("Outside scope.".into()))
    );
    let input: serde_json::Value = serde_json::from_slice(
        &std::fs::read(workspace.path().join("input.json")).expect("hook input"),
    )
    .expect("JSON input");
    assert_eq!(
        input,
        json!({
            "hook_event_name": "PreToolUse",
            "cwd": workspace.path(),
            "tool_name": "Bash",
            "tool_use_id": "call-1",
            "tool_input": {"command": "touch should-not-exist"}
        })
    );
}

#[tokio::test]
async fn pre_tool_hooks_accept_standard_no_decision_output() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(workspace.path(), json!([{"hooks": [
        {"type": "command", "command": "printf '%s' '{\"continue\":true,\"suppressOutput\":true}'"},
        {"type": "command", "command": "printf '%s' '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\"}}'"},
        {"type": "command", "command": "printf 'Reviewed successfully.'"}
    ]}])).await;

    assert_eq!(hooks.review(&bash_call()).await, Ok(None));
}

#[tokio::test]
async fn pre_tool_hooks_any_matching_command_can_deny() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(workspace.path(), json!([{"hooks": [
        {"type": "command", "command": "exit 0"},
        {"type": "command", "command": "printf '%s' '{\"continue\":true,\"suppressOutput\":true,\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Not allowed.\"}}'"}
    ]}])).await;
    assert_eq!(
        hooks.review(&bash_call()).await,
        Ok(Some("Not allowed.".into()))
    );
}

#[tokio::test]
async fn pre_tool_hooks_empty_denial_is_generic() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "exit 2"}]}
        ]),
    )
    .await;
    assert_eq!(
        hooks.review(&bash_call()).await,
        Ok(Some("Tool call rejected.".into()))
    );
}

#[tokio::test]
async fn pre_tool_hooks_failure_does_not_expose_diagnostics() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "printf 'synthetic-secret' >&2; exit 1"}]}
        ]),
    )
    .await;
    let error = hooks
        .review(&bash_call())
        .await
        .expect_err("failure stops execution");
    assert_eq!(error.to_string(), "pre-tool hook failed");
}

#[tokio::test]
async fn pre_tool_hooks_invalid_output_fails_closed() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "printf '{\"continue\":\"synthetic-secret'"}]}
        ]),
    )
    .await;
    let error = hooks.review(&bash_call()).await.expect_err("invalid JSON");
    assert_eq!(
        error.to_string(),
        "invalid pre-tool hook output: stdout starts with '{' but is not valid JSON"
    );
}

#[tokio::test]
async fn pre_tool_hooks_reject_unsupported_allow_decisions() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(workspace.path(), json!([{"hooks": [{
        "type": "command",
        "command": "printf '%s' '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"allow\"}}'"
    }]}])).await;

    let error = hooks
        .review(&bash_call())
        .await
        .expect_err("unsupported decision");
    assert_eq!(
        error.to_string(),
        "invalid pre-tool hook output: expected only continue:true, suppressOutput and hookSpecificOutput (PreToolUse deny); other fields or decisions are unsupported"
    );
}

#[tokio::test]
async fn pre_tool_hooks_reject_unsupported_fields_without_exposing_them() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(workspace.path(), json!([{"hooks": [{
        "type": "command", "command": "printf '%s' '{\"continue\":true,\"synthetic-secret\":true}'"
    }]}])).await;

    let error = hooks
        .review(&bash_call())
        .await
        .expect_err("unsupported field");
    assert_eq!(
        error.to_string(),
        "invalid pre-tool hook output: expected only continue:true, suppressOutput and hookSpecificOutput (PreToolUse deny); other fields or decisions are unsupported"
    );
}

#[tokio::test]
async fn pre_tool_hooks_reject_unsupported_stop_requests() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([{"hooks": [{
            "type": "command", "command": "printf '%s' '{\"continue\":false}'"
        }]}]),
    )
    .await;

    let error = hooks
        .review(&bash_call())
        .await
        .expect_err("unsupported stop request");
    assert_eq!(
        error.to_string(),
        "invalid pre-tool hook output: continue:false is unsupported; use exit 2 or permissionDecision:deny to reject a tool call"
    );
}

#[tokio::test]
async fn pre_tool_hooks_bound_output() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "head -c 65537 /dev/zero"}]}
        ]),
    )
    .await;
    assert_matches!(
        hooks.review(&bash_call()).await,
        Err(PreToolError::OutputTooLarge)
    );
}

#[tokio::test(start_paused = true)]
async fn pre_tool_hooks_timeout_fails_closed() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "exec /bin/sleep 60", "timeout": 1}]}
        ]),
    )
    .await;
    assert_matches!(
        hooks.review(&bash_call()).await,
        Err(PreToolError::TimedOut)
    );
}

#[tokio::test]
async fn pre_tool_hooks_invalid_configuration_is_not_silently_disabled() {
    let workspace = tempfile::tempdir().expect("workspace");
    let path = workspace.path().join("hooks.json");
    std::fs::write(
        &path,
        r#"{"hooks":{"PreToolUse":[{"matcher":"[","hooks":[]}]}}"#,
    )
    .expect("write invalid matcher");
    assert_matches!(
        PreToolHook::load(&path, workspace.path()).await,
        Err(PreToolError::Configuration("invalid matcher"))
    );
}

#[tokio::test]
async fn pre_tool_hooks_reject_unsupported_configuration_without_exposing_it() {
    let workspace = tempfile::tempdir().expect("workspace");
    let path = workspace.path().join("hooks.json");
    std::fs::write(
        &path,
        r#"{"hooks":{"PreToolUse":[{"hooks":[{"type":"command","command":"exit 0","synthetic-secret":true}]}]}}"#,
    )
    .expect("write unsupported configuration");

    let error = PreToolHook::load(&path, workspace.path())
        .await
        .expect_err("unsupported field");
    assert_eq!(
        error.to_string(),
        "invalid pre-tool hook configuration: expected only hooks.PreToolUse groups with matcher and command hooks (type, command, timeout); other events, fields and hook types are unsupported"
    );
}

#[tokio::test]
async fn pre_tool_hooks_child_execution_retains_parent_hooks() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "printf 'Blocked child.' >&2; exit 2"}]}
        ]),
    )
    .await;
    let parent = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        TaskManagerHandle::closed(),
        None,
        None,
        hooks.commands.to_vec(),
    );
    let child = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        TaskManagerHandle::closed(),
        Some(&parent),
        None,
        Vec::new(),
    );
    drop(parent);
    let executor = DirectToolExecutor::new(vec![Arc::new(
        BashTool::new(workspace.path()).expect("bash"),
    )]);
    let result = executor
        .call_tool(
            "bash".into(),
            r#"{"command":"touch should-not-exist"}"#.into(),
            child,
        )
        .await;
    assert_eq!(
        result.expect("reviewed call"),
        ToolOutput::error("Blocked child.")
    );
    assert!(!workspace.path().join("should-not-exist").exists());
}

struct ToolModel;

impl Llm for ToolModel {
    fn model(&self) -> &str {
        "synthetic"
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl futures_core::Stream<Item = Result<StreamEvent, LlmError>> + Send {
        futures_util::stream::once(async {
            Ok(StreamEvent::Final(Completion {
                message: mirror_protocol::AssistantMessage {
                    tool_calls: vec![bash_call()],
                    ..Default::default()
                },
                usage: None,
                truncated: false,
            }))
        })
    }
}

async fn service(
    workspace: &Path,
    hooks: PreToolHooks,
    approval: AgentApprovalMode,
) -> (crate::ReactAgentServiceHandle, tokio::task::JoinHandle<()>) {
    let tools = DirectToolExecutor::new(vec![Arc::new(BashTool::new(workspace).expect("bash"))]);
    let agent = ReactAgent::new(
        ToolModel,
        crate::react_agent_service::tests::test_provider(crate::providers::DEFAULT_CONTEXT_WINDOW)
            .await,
        None,
        SummaryCompactorFactory,
        &tools,
        SystemPromptConfig {
            custom_prompt: Some("synthetic".into()),
            include_project_instructions: false,
            include_skills: false,
            ..Default::default()
        },
        ReactAgentConfig {
            max_iterations: 1,
            ..Default::default()
        },
    )
    .expect("agent");
    let (service, runtime) = ReactAgentService::start(
        agent,
        Box::new(tools),
        Box::new(InMemoryStore::default()),
        "session",
        approval,
        Vec::new(),
        Vec::new(),
        None,
        broadcast::channel(1024).0,
        None,
        hooks.commands.to_vec(),
    )
    .await
    .expect("service");
    (service, tokio::spawn(runtime))
}

#[tokio::test]
async fn pre_tool_hooks_are_not_duplicated_or_removed_by_rpc_attachments() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(workspace.path(), json!([
        {"hooks": [{"type": "command", "command": "printf x >> reviews; printf 'Blocked.' >&2; exit 2"}]}
    ])).await;
    let (service, runtime) = service(workspace.path(), hooks, AgentApprovalMode::Auto).await;
    let rpc = AgentRpc::new(service.clone());
    let (first, incoming) = mpsc::channel(1);
    let first_stream = rpc
        .open_session(ReceiverStream::new(incoming))
        .await
        .expect("first client");
    let (second, incoming) = mpsc::channel(1);
    let second_stream = rpc
        .open_session(ReceiverStream::new(incoming))
        .await
        .expect("second client");

    assert_eq!(
        service
            .call_tool("bash".into(), "{}".into())
            .await
            .expect("review"),
        ToolOutput::error("Blocked.")
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("reviews")).expect("review count"),
        "x"
    );
    drop((first, first_stream, second, second_stream));
    assert_eq!(
        service
            .call_tool("bash".into(), "{}".into())
            .await
            .expect("headless review"),
        ToolOutput::error("Blocked.")
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("reviews")).expect("review count"),
        "xx"
    );
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}

#[tokio::test]
async fn pre_tool_hooks_continue_does_not_bypass_manual_approval() {
    let workspace = tempfile::tempdir().expect("workspace");
    let hooks = load_hooks(
        workspace.path(),
        json!([
            {"hooks": [{"type": "command", "command": "printf '%s' '{\"continue\":true,\"suppressOutput\":false}'"}]}
        ]),
    )
    .await;
    let (service, runtime) = service(workspace.path(), hooks, AgentApprovalMode::Hitl).await;
    let mut events = service.events();
    let caller = service.clone();
    let call = tokio::spawn(async move {
        caller
            .call_tool(
                "bash".into(),
                r#"{"command":"touch should-not-exist"}"#.into(),
            )
            .await
    });
    let approval = loop {
        if let crate::ReactServiceEvent::HitlRequest(request) = events.recv().await.expect("event")
        {
            break request.call.expect("call");
        }
    };
    service
        .respond_to_hitl(approval.id, false, mirror_protocol::ApprovalSource::Manual)
        .await
        .expect("deny");
    assert_eq!(
        call.await.expect("caller").expect("result"),
        ToolOutput::error("Tool call rejected.")
    );
    assert!(!workspace.path().join("should-not-exist").exists());
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}

#[tokio::test]
async fn pre_tool_hooks_review_model_calls_without_a_cli() {
    let workspace = tempfile::tempdir().expect("workspace");
    check_model_hook(
        workspace.path(),
        "printf 'Blocked model.' >&2; exit 2",
        "Blocked model.",
    )
    .await;
}

#[tokio::test]
async fn hook_failure_records_a_result_without_executing_the_call() {
    let workspace = tempfile::tempdir().expect("workspace");
    check_model_hook(workspace.path(), "exit 1", "pre-tool hook failed").await;
}

async fn check_model_hook(workspace: &Path, command: &str, expected: &str) {
    let hooks = load_hooks(
        workspace,
        json!([{"hooks":[{"type":"command","command":command}]}]),
    )
    .await;
    let (service, runtime) = service(workspace, hooks, AgentApprovalMode::Auto).await;
    let mut events = service.events();
    service
        .execute(mirror_protocol::AgentMessage {
            id: "user".into(),
            payload: Some(mirror_protocol::agent_message::Payload::Chat(
                mirror_protocol::ChatRecord {
                    message: Some(mirror_protocol::chat_record::Message::User(
                        mirror_protocol::UserMessage {
                            content: "work".into(),
                            ..Default::default()
                        },
                    )),
                    success: true,
                    ..Default::default()
                },
            )),
            ..Default::default()
        })
        .expect("execute");
    loop {
        if matches!(
            events.recv().await.expect("event"),
            crate::ReactServiceEvent::TurnCompleted | crate::ReactServiceEvent::TurnError(_)
        ) {
            break;
        }
    }
    let entries = service.messages().await.expect("history");
    assert!(entries.iter().any(|entry| matches!(entry.chat_message(), Some(mirror_protocol::chat_record::Message::Tool(tool)) if tool.tool_call_id == "call-1" && tool.content == format!("Tool error: {expected}"))));
    assert!(!workspace.join("should-not-exist").exists());
    service.close().await.expect("close");
    runtime.await.expect("runtime");
}

#[tokio::test]
async fn registered_plugin_hooks_require_trust_and_use_the_plugin_root() {
    let config = tempfile::tempdir().expect("config");
    let workspace = tempfile::tempdir().expect("workspace");
    let plugin = config.path().join("reviewer's plugin");
    std::fs::create_dir_all(plugin.join(".claude-plugin")).expect("manifest directory");
    std::fs::create_dir(plugin.join("hooks")).expect("hook directory");
    std::fs::write(
        plugin.join(".claude-plugin/plugin.json"),
        r#"{"name":"review"}"#,
    )
    .expect("manifest");
    std::fs::write(plugin.join("hooks/hooks.json"), json!({"description":"Review shell calls", "hooks":{"PreToolUse":[{"matcher":"Bash", "hooks":[{"type":"command", "statusMessage":"Reviewing", "command":"sh \"${CLAUDE_PLUGIN_ROOT}/review.sh\""}]}]}}).to_string()).expect("plugin hooks");
    std::fs::write(plugin.join("review.sh"), "test \"$PLUGIN_ROOT\" = \"$CLAUDE_PLUGIN_ROOT\" || exit 1; cat > input.json; echo 'Reviewed.' >&2; exit 2").expect("script");
    crate::plugins::add_plugin(&config.path().join("plugins"), &plugin).expect("register plugin");
    let path = config.path().join("hooks.json");
    assert!(PreToolHook::load(&path, workspace.path())
        .await
        .expect("no trust")
        .is_empty());
    std::fs::write(&path, r#"{"plugins":["review"]}"#).expect("trust hooks");
    let hooks = PreToolHooks::new(
        PreToolHook::load(&path, workspace.path())
            .await
            .expect("trusted hooks"),
        None,
    );
    let mut call = bash_call();
    call.function.as_mut().expect("function").arguments.clear();
    assert_eq!(hooks.review(&call).await, Ok(Some("Reviewed.".into())));
    let input: serde_json::Value =
        serde_json::from_slice(&std::fs::read(workspace.path().join("input.json")).expect("stdin"))
            .expect("JSON");
    assert_eq!(input["tool_input"], json!({}));
    assert_eq!(input["tool_name"], "Bash");
}

#[tokio::test]
async fn codex_manifest_hook_replaces_default_and_cannot_escape_plugin() {
    let config = tempfile::tempdir().expect("config");
    let root = config.path().join("plugins/check");
    std::fs::create_dir_all(root.join(".codex-plugin")).expect("plugin");
    std::fs::create_dir(root.join("hooks")).expect("hook directory");
    std::fs::write(root.join("hooks/hooks.json"), "not json").expect("unused default");
    std::fs::write(
        root.join("selected.json"),
        r#"{"hooks":{"PreToolUse":[{"hooks":[{"type":"command","command":"exit 2"}]}]}}"#,
    )
    .expect("selected hook");
    let manifest = root.join(".codex-plugin/plugin.json");
    std::fs::write(&manifest, r#"{"name":"check","hooks":"./selected.json"}"#).expect("manifest");
    let path = config.path().join("hooks.json");
    std::fs::write(&path, r#"{"plugins":["check"]}"#).expect("trust hooks");
    let hooks = PreToolHooks::new(
        PreToolHook::load(&path, config.path())
            .await
            .expect("trusted hooks"),
        None,
    );
    assert_eq!(
        hooks.review(&bash_call()).await,
        Ok(Some("Tool call rejected.".into()))
    );
    std::fs::rename(
        root.join("selected.json"),
        config.path().join("outside.json"),
    )
    .expect("move file");
    std::os::unix::fs::symlink(
        config.path().join("outside.json"),
        root.join("selected.json"),
    )
    .expect("link outside");
    assert_matches!(
        PreToolHook::load(&path, config.path()).await,
        Err(PreToolError::Configuration(_))
    );
}
