use pretty_assertions::assert_eq;

use super::*;

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new(
            AgentApprovalMode::Hitl,
            None,
            crate::TaskManagerHandle::default(),
            None,
            None,
        )
    }
}

#[test]
fn execution_context_carries_inference_config() {
    let context = ExecutionContext::default();
    let input = context.input();
    let inference = mirror_protocol::InferenceConfig {
        model: "child-model".to_string(),
        api: Some("openai-responses".to_string()),
        base_url: Some("https://example.test/v1".to_string()),
        reasoning_effort: Some("high".to_string()),
        ..Default::default()
    };

    input.set_inference_config(Some(inference.clone()));
    assert_eq!(context.inference_config(), Some(inference));

    input.set_inference_config(None);
    assert_eq!(context.inference_config(), None);
}

#[test]
fn automatic_approval_allows_workspace_writes() {
    let context = ExecutionContext::new(
        AgentApprovalMode::Auto,
        None,
        crate::TaskManagerHandle::default(),
        None,
        None,
    );

    assert!(context.tool_permissions().allow_outside_workspace());
}

#[test]
fn input_updates_live_approval_mode() {
    let context = ExecutionContext::default();
    let input = context.input();

    input.set_approval_mode(AgentApprovalMode::Auto);

    assert_eq!(context.approval_mode(), AgentApprovalMode::Auto);
    assert!(context.tool_permissions().allow_outside_workspace());
}

#[test]
fn inherited_approvals_follow_live_parent_permissions() {
    let parent = ExecutionContext::default();
    let child = ExecutionContext::new(
        AgentApprovalMode::Hitl,
        None,
        Default::default(),
        Some(&parent),
        None,
    );

    assert_eq!(child.approval_mode(), AgentApprovalMode::Hitl);
    assert!(!child.tool_permissions().allow_outside_workspace());
    parent.input().set_approval_mode(AgentApprovalMode::Auto);
    assert_eq!(child.approval_mode(), AgentApprovalMode::Auto);
    assert!(child.tool_permissions().allow_outside_workspace());
    parent.input().set_approval_mode(AgentApprovalMode::Hitl);
    assert!(!child.tool_permissions().allow_outside_workspace());
}
