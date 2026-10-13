use assert_matches::assert_matches;
use pretty_assertions::{assert_eq, assert_ne};

use super::*;
use crate::tools::validate_tools;
use crate::tools::AnyTool;
use crate::tools::BashTool;
use crate::tools::ReadTool;
use crate::tools::ToolError;

#[tokio::test]
async fn executes_client_hosted_tool_through_request_channel() {
    let (requests, mut receiver) = mpsc::unbounded_channel();
    let tool = HostTool::new(
        HostToolConfig {
            name: "echo".to_string(),
            description: "Echo text.".to_string(),
            parameters: [("type".to_string(), Value::String("object".to_string()))]
                .into_iter()
                .collect(),
            needs_approval: true,
            timeout_seconds: 42.5,
        },
        requests,
    )
    .expect("host tool");
    let definition = tool.definition();
    let context = ExecutionContext::default();
    let execute = tool.execute(r#"{"text":"hello"}"#, &context);
    let respond = async {
        let request = receiver.recv().await.expect("host tool request");
        assert_eq!(request.name, "echo");
        assert_eq!(request.arguments, r#"{"text":"hello"}"#);
        assert_eq!(request.timeout, Duration::from_secs_f64(42.5));
        _ = request.response.send(ToolOutput::success("echoed"));
    };

    let (result, ()) = tokio::join!(execute, respond);

    assert_eq!(definition.name, "echo");
    assert_eq!(definition.parameters["type"], "object");
    assert!(tool.needs_approval());
    assert_ne!(result, ToolOutput::error("echo"));
    assert_matches!(result, ToolOutput { success: true, .. });
    assert_eq!(result.content, "echoed");
}

#[test]
fn rejects_duplicate_host_tool_names() {
    let (requests, _receiver) = mpsc::unbounded_channel();
    let first = host_tool("echo", requests.clone());
    let second = host_tool("echo", requests);
    let tools: [&dyn AnyTool; 2] = [&first, &second];

    assert_matches!(
        validate_tools(tools),
        Err(ToolError::InvalidArguments(message)) if message == "duplicate tool name \"echo\""
    );
}

#[test]
fn rejects_host_tool_name_that_collides_with_read_only_builtin() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let read = ReadTool::new(workspace.path()).expect("read tool");
    let (requests, _receiver) = mpsc::unbounded_channel();
    let host_tool = host_tool("read", requests);
    let tools: [&dyn AnyTool; 2] = [&read, &host_tool];

    assert!(!AnyTool::needs_approval(&read));
    assert_matches!(
        validate_tools(tools),
        Err(ToolError::InvalidArguments(message)) if message == "duplicate tool name \"read\""
    );
}

#[test]
fn rejects_host_tool_name_that_collides_with_approval_gated_builtin() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let bash = BashTool::new(workspace.path()).expect("bash tool");
    let (requests, _receiver) = mpsc::unbounded_channel();
    let host_tool = host_tool("bash", requests);
    let tools: [&dyn AnyTool; 2] = [&bash, &host_tool];

    assert!(AnyTool::needs_approval(&bash));
    assert_matches!(
        validate_tools(tools),
        Err(ToolError::InvalidArguments(message)) if message == "duplicate tool name \"bash\""
    );
}

fn host_tool(name: &str, requests: mpsc::UnboundedSender<HostToolRequest>) -> HostTool {
    HostTool::new(
        HostToolConfig {
            name: name.to_string(),
            description: "Host tool.".to_string(),
            parameters: Map::new(),
            needs_approval: false,
            timeout_seconds: DEFAULT_HOST_TOOL_TIMEOUT_SECONDS,
        },
        requests,
    )
    .expect("host tool")
}
