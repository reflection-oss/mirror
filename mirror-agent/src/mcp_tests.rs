use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::Json;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use rmcp::model::{CallToolResult, ContentBlock, Tool};
use serde_json::{json, Map, Value};

use super::mcp_tool::McpToolDefinition;
use super::{
    discover_mcp_connections, mcp_tools, valid_mcp_name, valid_mcp_server_name, valid_mcp_url,
    LocalMcpConnection, Mcp, McpAuthorization, McpClient, McpClientError, McpRouter, McpTool,
};
use crate::tools::{DirectToolExecutor, ToolDefinition, ToolError, ToolExecutor};
use crate::{
    AgentApprovalMode, ApprovalDecision, ApprovalHook, ApprovalSource, ExecutionContext,
    TaskManagerHandle,
};

#[derive(Default)]
struct Calls(AtomicUsize);

impl McpRouter for Calls {
    fn list_tools<'a>(
        &'a self,
        _server: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Tool>, ToolError>> + Send + 'a>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn call_tool<'a>(
        &'a self,
        server: &'a str,
        tool: &'a str,
        arguments: Map<String, Value>,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            assert_eq!(server, "issues");
            assert_eq!(tool, "createIssue");
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(CallToolResult::success(vec![ContentBlock::text(
                arguments["title"].as_str().expect("validated title"),
            )]))
        })
    }
}

fn tool(calls: Arc<Calls>) -> Arc<McpTool> {
    Arc::new(McpTool::new(McpToolDefinition {
        definition: ToolDefinition {
            name: "mcp__issues__createIssue".into(),
            description: "Create an issue with a title".into(),
            parameters: serde_json::from_value(json!({"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"], "additionalProperties": false})).expect("schema"),
        },
        connection: "issues".into(), tool: "createIssue".into(), needs_approval: true,
    }, calls))
}

fn context(mode: AgentApprovalMode, hook: Option<ApprovalHook>) -> ExecutionContext {
    ExecutionContext::new(mode, hook, TaskManagerHandle::default(), None, None)
}

#[tokio::test]
async fn original_names_cannot_bypass_the_router() {
    let calls = Arc::new(Calls::default());
    let executor = DirectToolExecutor::new(mcp_tools(vec![tool(calls.clone())]).expect("tools"));
    assert_eq!(
        executor
            .definitions()
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["mcp_search", "mcp_describe", "mcp_call"]
    );
    let output = executor
        .call_tool(
            "mcp__issues__createIssue".into(),
            r#"{"title":"must not run"}"#.into(),
            context(AgentApprovalMode::Auto, None),
        )
        .await
        .expect("direct call");
    assert!(!output.success);
    assert_eq!(calls.0.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn batch_validates_before_approval_and_uses_the_underlying_identity() {
    let calls = Arc::new(Calls::default());
    let executor = DirectToolExecutor::new(mcp_tools(vec![tool(calls.clone())]).expect("tools"));
    let (hook, mut approvals) = ApprovalHook::channel();
    let context = context(AgentApprovalMode::Hitl, Some(hook));
    let execute = executor.call_tool("mcp_call".into(), r#"{"calls":[{"name":"mcp__issues__createIssue","arguments":{"title":42}},{"name":"mcp__issues__createIssue","arguments":{"title":"denied"}},{"name":"mcp__issues__createIssue","arguments":{"title":"allowed"}}]}"#.into(), context);
    let approve = async {
        let denied = approvals.recv().await.expect("denied request");
        assert_eq!(
            denied.call.function.as_ref().expect("function").name,
            "mcp__issues__createIssue"
        );
        assert_eq!(
            denied.call.function.as_ref().expect("function").arguments,
            r#"{"title":"denied"}"#
        );
        denied
            .response
            .send(ApprovalDecision::new(false, ApprovalSource::Manual))
            .expect("reject");
        let allowed = approvals.recv().await.expect("allowed request");
        allowed
            .response
            .send(ApprovalDecision::new(true, ApprovalSource::Manual))
            .expect("approve");
    };
    let (output, ()) = tokio::join!(execute, approve);
    let output = output.expect("call");
    let output: Value = serde_json::from_str(&output.content).expect("results");
    assert_eq!(output["results"][0]["success"], false);
    assert_eq!(output["results"][1]["success"], false);
    assert_eq!(output["results"][2]["content"], "allowed");
    assert_eq!(calls.0.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn invalid_envelope_executes_nothing() {
    let calls = Arc::new(Calls::default());
    let executor = DirectToolExecutor::new(mcp_tools(vec![tool(calls.clone())]).expect("tools"));
    let output = executor.call_tool("mcp_call".into(), r#"{"calls":[{"name":"mcp__issues__createIssue","arguments":{"title":"must not run"}},{}]}"#.into(), context(AgentApprovalMode::Auto, None)).await.expect("call");
    assert!(!output.success);
    assert_eq!(calls.0.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn large_call_results_preserve_status_and_report_truncation() {
    let calls = Arc::new(Calls::default());
    let executor = DirectToolExecutor::new(mcp_tools(vec![tool(calls.clone())]).expect("tools"));
    let context = context(AgentApprovalMode::Auto, None);
    let request = json!({"calls":[
        {"name":"mcp__issues__createIssue", "arguments":{"title":"\"🥑".repeat(40_000)}},
        {"name":"mcp__issues__createIssue", "arguments":{"title":"later call"}}
    ]});
    let output = executor
        .call_tool("mcp_call".into(), request.to_string(), context)
        .await
        .expect("call");
    assert!(output.success);
    assert!(output.content.len() <= 32_768);
    let value: Value = serde_json::from_str(&output.content).expect("valid JSON");
    assert_eq!(value["results"][0]["success"], true);
    assert_eq!(value["results"][0]["truncated"], true);
    assert_eq!(value["results"][1]["content"], "later call");
    assert_eq!(calls.0.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn oversized_schema_fails_explicitly_without_truncating_json() {
    let calls = Arc::new(Calls::default());
    let mut binding = tool(calls.clone()).binding().clone();
    binding
        .definition
        .parameters
        .insert("description".into(), json!("x".repeat(40_000)));
    let remote = Arc::new(McpTool::new(binding, calls));
    let executor = DirectToolExecutor::new(mcp_tools(vec![remote]).expect("tools"));
    let context = context(AgentApprovalMode::Auto, None);
    let output = executor
        .call_tool(
            "mcp_describe".into(),
            r#"{"names":["mcp__issues__createIssue"]}"#.into(),
            context,
        )
        .await
        .expect("describe");
    assert!(!output.success);
    assert!(output.content.contains("schemas are never truncated"));
    assert!(output.content.len() < 32_768);
}

#[test]
fn search_matches_schema_keys_and_exact_names_but_requires_the_rarest_term() {
    let mcp = Mcp::new(vec![tool(Arc::new(Calls::default()))]).expect("catalog");
    assert!(mcp.search("issues nonexistent", 5).is_empty());
    assert_eq!(mcp.search("TITLE", 5)[0].name, "mcp__issues__createIssue");
    assert_eq!(
        mcp.search(" MCP__ISSUES__CREATEISSUE ", 5)[0].name,
        "mcp__issues__createIssue"
    );
}

async fn mcp(headers: HeaderMap, Json(request): Json<Value>) -> Response {
    assert!(!headers.contains_key("authorization"));
    let result = match request["method"].as_str() {
        Some("initialize") => {
            json!({"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name":"test", "version":"1"}})
        }
        Some("notifications/initialized") => return StatusCode::ACCEPTED.into_response(),
        Some("tools/list") => {
            json!({"tools":[{"name":"echo", "description":"Echo text", "inputSchema":{"type":"object", "properties":{"text":{"type":"string"}}, "required":["text"]}}]})
        }
        Some("tools/call") => {
            return Json(json!({
                "jsonrpc": "2.0", "id": request["id"],
                "error": {"code": -32603, "message": "synthetic-fresh"}
            }))
            .into_response()
        }
        _ => return StatusCode::BAD_REQUEST.into_response(),
    };
    Json(json!({"jsonrpc":"2.0", "id":request["id"], "result":result})).into_response()
}

async fn oauth_metadata(headers: HeaderMap) -> Json<Value> {
    let host = headers["host"].to_str().expect("host");
    Json(json!({
        "issuer": format!("http://{host}"),
        "authorization_endpoint": format!("http://{host}/authorize"),
        "token_endpoint": format!("http://{host}/token"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post"]
    }))
}

async fn protected_resource(headers: HeaderMap) -> Json<Value> {
    let host = headers["host"].to_str().expect("host");
    Json(
        json!({"resource": format!("http://{host}/mcp"), "authorization_servers": [format!("http://{host}")]}),
    )
}

async fn refresh_token(body: String) -> Json<Value> {
    assert!(body.contains("grant_type=refresh_token"));
    assert!(body.contains("client_secret=synthetic-client-secret"));
    assert!(body.contains("refresh_token=synthetic-refresh"));
    Json(
        json!({"access_token":"synthetic-fresh", "token_type":"Bearer", "expires_in":3600, "refresh_token":"synthetic-rotated"}),
    )
}

async fn authenticated_mcp(mut headers: HeaderMap, request: Json<Value>) -> Response {
    assert_eq!(
        headers.remove("authorization").expect("auth"),
        "Bearer synthetic-fresh"
    );
    mcp(headers, request).await
}

#[tokio::test]
async fn shared_client_uses_bearer_auth_and_keeps_upstream_errors_private() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().route("/mcp", post(authenticated_mcp)),
        )
        .await
        .expect("server");
    });
    let client = McpClient::http(
        reqwest::Client::new(),
        &format!("http://{address}/mcp"),
        McpAuthorization::Bearer("synthetic-fresh".into()),
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("connected client");
    assert_eq!(client.list_tools().await.expect("catalog").len(), 1);
    let error = client
        .call_tool("echo", None)
        .await
        .expect_err("upstream error");
    assert_matches::assert_matches!(error, McpClientError::Request);
    assert!(!error.to_string().contains("synthetic-fresh"));
    server.abort();
}

#[tokio::test]
async fn runtime_refreshes_cli_credentials_without_python_and_saves_them_privately() {
    use axum::routing::get;
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("credentials directory");
    let path = directory.path().join("fixture.json");
    std::fs::write(&path, json!({"client_info":{"client_id":"fixture-client","client_secret":"synthetic-client-secret","redirect_uris":["http://localhost/callback"]}, "tokens":{"access_token":"synthetic-expired","token_type":"Bearer","refresh_token":"synthetic-refresh","expires_in":3600}, "expires_at":1}).to_string()).expect("synthetic credentials");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let address = listener.local_addr().expect("address");
    let app = axum::Router::new()
        .route("/mcp", post(authenticated_mcp))
        .route("/token", post(refresh_token))
        .route(
            "/.well-known/oauth-protected-resource/mcp",
            get(protected_resource),
        )
        .route(
            "/.well-known/oauth-authorization-server",
            get(oauth_metadata),
        );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server");
    });
    let connection: LocalMcpConnection = serde_json::from_value(json!({"name":"fixture", "credential_directory":directory.path(), "config":{"url":format!("http://{address}/mcp"),"auth":{"type":"oauth"}}})).expect("connection");
    let tools = discover_mcp_connections(&[connection]).await;
    assert_eq!(tools.len(), 1);
    let saved: Value = serde_json::from_slice(&std::fs::read(&path).expect("saved credentials"))
        .expect("credential JSON");
    assert_eq!(saved["tokens"]["refresh_token"], "synthetic-rotated");
    assert_eq!(
        std::fs::metadata(&path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert!(!serde_json::to_string(tools[0].binding())
        .expect("binding")
        .contains("synthetic-"));
    server.abort();
}

#[test]
fn server_names_are_canonical_but_upstream_names_and_credential_stems_keep_case() {
    assert!(valid_mcp_server_name("test-server_2"));
    assert!(!valid_mcp_server_name(""));
    assert!(!valid_mcp_server_name("Test"));
    assert!(!valid_mcp_server_name("2test"));
    assert!(!valid_mcp_server_name("a__b"));
    assert!(valid_mcp_name("CredentialAlias"));
    assert!(!valid_mcp_name("../credentials"));
}

#[test]
fn endpoints_require_tls_or_loopback_and_exclude_embedded_authentication() {
    assert!(valid_mcp_url("https://example.test/mcp"));
    assert!(valid_mcp_url("http://localhost:8080/mcp"));
    assert!(valid_mcp_url("http://127.0.0.1/mcp"));
    assert!(valid_mcp_url("http://[::1]/mcp"));
    assert!(!valid_mcp_url("http://example.test/mcp"));
    assert!(!valid_mcp_url("https://user:synthetic@example.test/mcp"));
    assert!(!valid_mcp_url("https://example.test/mcp?token=synthetic"));
    assert!(!valid_mcp_url("https://example.test/mcp#synthetic"));
    assert!(!valid_mcp_url("file:///tmp/mcp"));
}
