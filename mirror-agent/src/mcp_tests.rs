use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use assert_matches::assert_matches;
use axum::extract::Json;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use mirror_protocol::{FunctionCall, ToolCall};
use portpicker::pick_unused_port;
use pretty_assertions::{assert_eq, assert_ne};
use rmcp::model::{
    CallToolResult, ContentBlock, InitializeRequestParams, InitializeResult, ListToolsResult,
    PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::never::NeverSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler};
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use super::mcp_tool::McpToolDefinition;
use super::{
    discover_mcp_connections, live_mcp_tools, mcp_tools, valid_mcp_name, valid_mcp_server_name,
    valid_mcp_url, LocalMcpConnection, LocalMcpDiscovery, Mcp, McpAuthorization, McpClient,
    McpClientError, McpRouter, McpTool,
};
use crate::tools::{
    AnyTool, DirectToolExecutor, RemoteToolExecutor, ToolDefinition, ToolError, ToolExecutor,
};
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
        _tool_call_id: Option<&'a str>,
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

struct ToolCallIds(mpsc::UnboundedSender<Option<String>>);

impl McpRouter for ToolCallIds {
    fn list_tools<'a>(
        &'a self,
        _server: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Tool>, ToolError>> + Send + 'a>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn call_tool<'a>(
        &'a self,
        _server: &'a str,
        _tool: &'a str,
        _arguments: Map<String, Value>,
        tool_call_id: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>> {
        let _ = self.0.send(tool_call_id.map(str::to_owned));
        Box::pin(async { Ok(CallToolResult::success(Vec::new())) })
    }
}

fn tool(calls: Arc<dyn McpRouter>) -> Arc<McpTool> {
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
    ExecutionContext::new(
        mode,
        hook,
        TaskManagerHandle::default(),
        None,
        None,
        Vec::new(),
    )
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
async fn remote_workspace_calls_use_the_mcp_router_not_the_shell() {
    let calls = Arc::new(Calls::default());
    let executor = RemoteToolExecutor::new(
        mcp_tools(vec![tool(calls.clone())]).expect("tools"),
        reqwest::Url::parse("http://127.0.0.1:1/exec").expect("URL"),
    )
    .expect("MCP does not need a reachable remote shell");
    let output = executor.call_tool(
        "mcp_call".into(),
        r#"{"calls":[{"name":"mcp__issues__createIssue","arguments":{"title":"created through MCP"}}]}"#.into(),
        context(AgentApprovalMode::Auto, None),
    ).await.expect("MCP call");
    let result: Value = serde_json::from_str(&output.content).expect("result");
    assert_eq!(result["results"][0]["success"], true);
    assert_eq!(result["results"][0]["content"], "created through MCP");
    assert_eq!(calls.0.load(Ordering::SeqCst), 1);
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
async fn batched_calls_carry_the_model_call_id_and_their_position_across_retries() {
    let (sender, mut ids) = mpsc::unbounded_channel();
    let executor = DirectToolExecutor::new(
        mcp_tools(vec![tool(Arc::new(ToolCallIds(sender)))]).expect("tools"),
    );
    let call = ToolCall {
        id: "call_1".into(),
        r#type: "function".into(),
        function: Some(FunctionCall {
            name: "mcp_call".into(),
            arguments: r#"{"calls":[{"name":"mcp__issues__createIssue","arguments":{"title":"first"}},{"name":"mcp__issues__createIssue","arguments":{"title":"second"}}]}"#.into(),
        }),
        ..ToolCall::default()
    };
    let context = context(AgentApprovalMode::Auto, None);

    executor
        .execute(
            Uuid::new_v4(),
            std::slice::from_ref(&call),
            &[true],
            &context,
        )
        .await
        .expect("first attempt");
    executor
        .execute(
            Uuid::new_v4(),
            std::slice::from_ref(&call),
            &[true],
            &context,
        )
        .await
        .expect("retry");

    let mut received = Vec::new();
    ids.recv_many(&mut received, 10).await;
    assert_eq!(
        received,
        [
            Some("call_1/0".to_owned()),
            Some("call_1/1".to_owned()),
            Some("call_1/0".to_owned()),
            Some("call_1/1".to_owned()),
        ]
    );
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
            assert_eq!(request["params"]["protocolVersion"], "2025-11-25");
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
        .call_tool("echo", None, None)
        .await
        .expect_err("upstream error");
    assert_matches!(error, McpClientError::JsonRpc { code: -32603 });
    assert!(!error.to_string().contains("synthetic-fresh"));
    assert!(error.to_string().contains("do not blindly retry writes"));
    server.abort();
}

async fn error_server(
    method: &'static str,
    response: impl Fn(Value) -> Response + Clone + Send + Sync + 'static,
) -> (String, tokio::task::JoinHandle<()>) {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let port = pick_unused_port().expect("unused port");
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("listener");
    let url = format!("http://{}/mcp", listener.local_addr().expect("address"));
    let router = axum::Router::new().route(
        "/mcp",
        post(move |headers: HeaderMap, request: Json<Value>| {
            let response = response.clone();
            async move {
                if request.0["method"] == method {
                    response(request.0["id"].clone())
                } else {
                    mcp(headers, request).await
                }
            }
        }),
    );
    let task = tokio::spawn(async move { axum::serve(listener, router).await.expect("server") });
    (url, task)
}

#[tokio::test]
async fn discovery_reports_connection_refusal() {
    let port = pick_unused_port().expect("unused port");
    let connection = serde_json::from_value(json!({
        "name": "unreachable", "config": {"url": format!("http://127.0.0.1:{port}/mcp")}
    }))
    .expect("connection");
    let mut errors = Vec::new();
    let tools =
        discover_mcp_connections(&[connection], |error| errors.push(error.to_string())).await;

    assert!(tools.is_empty());
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("MCP discovery for unreachable: cannot connect to MCP server"));
    assert!(errors[0].contains("check network access and server configuration"));
    assert!(!errors[0].contains("may have executed"));
}

#[tokio::test(start_paused = true)]
async fn shared_client_reports_http_timeout_before_startup_deadline() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let port = pick_unused_port().expect("unused port");
    let _listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("listener");
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .expect("client");
    let error = McpClient::http(
        http,
        &format!("http://127.0.0.1:{port}/mcp"),
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .err()
    .expect("request timeout");

    assert_matches!(error, McpClientError::Timeout);
}

#[derive(Clone)]
struct StatelessServer;

impl ServerHandler for StatelessServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
    }

    fn supported_protocol_versions(&self) -> std::borrow::Cow<'static, [ProtocolVersion]> {
        std::borrow::Cow::Owned(vec![ProtocolVersion::V_2026_07_28])
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, ErrorData> {
        Err(ErrorData::invalid_request(
            "initialize is not supported",
            None,
        ))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let schema = json!({"type": "object"})
            .as_object()
            .cloned()
            .expect("schema");
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "echo",
            "Echo text",
            Arc::new(schema),
        )]))
    }
}

#[tokio::test]
async fn shared_client_uses_stateless_discovery_when_the_server_supports_it() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let service = StreamableHttpService::new(
        || Ok(StatelessServer),
        Arc::new(NeverSessionManager::default()),
        StreamableHttpServerConfig::default()
            .with_legacy_session_mode(false)
            .with_stateless_protocol_metadata_required(true),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let url = format!("http://{}/mcp", listener.local_addr().expect("address"));
    let server = tokio::spawn(async move {
        axum::serve(listener, axum::Router::new().nest_service("/mcp", service))
            .await
            .expect("server");
    });

    let client = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("stateless client");

    assert_eq!(client.list_tools().await.expect("catalog").len(), 1);
    server.abort();
}

async fn legacy_server(
    discover: impl Fn(Value) -> Response + Clone + Send + Sync + 'static,
) -> (
    String,
    mpsc::UnboundedReceiver<String>,
    tokio::task::JoinHandle<()>,
) {
    let (methods, receiver) = mpsc::unbounded_channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let url = format!("http://{}/mcp", listener.local_addr().expect("address"));
    let router = axum::Router::new().route(
        "/mcp",
        post(move |headers: HeaderMap, request: Json<Value>| {
            let discover = discover.clone();
            let methods = methods.clone();
            async move {
                let method = request.0["method"].as_str().unwrap_or_default().to_owned();
                let _ = methods.send(method.clone());
                if method == "server/discover" {
                    return discover(request.0["id"].clone());
                }
                if headers
                    .get("mcp-protocol-version")
                    .is_some_and(|version| version == "2026-07-28")
                {
                    return StatusCode::BAD_REQUEST.into_response();
                }
                mcp(headers, request).await
            }
        }),
    );
    let task = tokio::spawn(async move { axum::serve(listener, router).await.expect("server") });
    (url, receiver, task)
}

fn received(methods: &mut mpsc::UnboundedReceiver<String>) -> Vec<String> {
    std::iter::from_fn(|| methods.try_recv().ok()).collect()
}

#[tokio::test]
async fn shared_client_falls_back_to_initialize_when_discovery_is_unsupported() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let (url, mut methods, server) = legacy_server(|id| {
        Json(json!({
            "jsonrpc": "2.0", "id": id,
            "error": {"code": -32601, "message": "Method not supported"}
        }))
        .into_response()
    })
    .await;

    let client = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("legacy client");

    assert_eq!(client.list_tools().await.expect("catalog").len(), 1);
    assert_eq!(
        received(&mut methods),
        [
            "server/discover",
            "initialize",
            "notifications/initialized",
            "tools/list"
        ]
    );
    server.abort();
}

#[tokio::test]
async fn shared_client_retries_initialize_after_a_bare_discovery_failure() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let (url, mut methods, server) =
        legacy_server(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()).await;

    let client = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("retried client");

    assert_eq!(client.list_tools().await.expect("catalog").len(), 1);
    assert_eq!(
        received(&mut methods),
        [
            "server/discover",
            "initialize",
            "notifications/initialized",
            "tools/list"
        ]
    );
    server.abort();
}

#[tokio::test]
async fn shared_client_reports_authentication_without_leaking_challenges() {
    let (url, server) = error_server("server/discover", |_| {
        (
            StatusCode::UNAUTHORIZED,
            [(
                "www-authenticate",
                "Bearer realm=\"synthetic-private-upstream-error\"",
            )],
            "synthetic-private-upstream-error",
        )
            .into_response()
    })
    .await;
    let error = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .err()
    .expect("authentication failure");

    assert_matches!(error, McpClientError::Authentication);
    assert!(!format!("{error:?} {error}").contains("synthetic-private"));
    server.abort();
}

#[tokio::test]
async fn discovery_reports_forbidden_initialization_without_guessing_its_cause() {
    let (url, server) = error_server("initialize", |_| {
        (
            StatusCode::FORBIDDEN,
            [(
                "www-authenticate",
                "Bearer scope=\"synthetic-private-upstream-error\"",
            )],
            "synthetic-private-upstream-error",
        )
            .into_response()
    })
    .await;
    let connection = serde_json::from_value(json!({"name":"fixture", "config":{"url":url}}))
        .expect("connection");
    let mut errors = Vec::new();
    let tools =
        discover_mcp_connections(&[connection], |error| errors.push(error.to_string())).await;

    assert!(tools.is_empty());
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("MCP discovery for fixture: MCP access denied"));
    assert!(errors[0].contains("account permissions and server API configuration"));
    assert!(!errors[0].contains("SERVICE_DISABLED"));
    assert!(!errors[0].contains("synthetic-private"));
    server.abort();
}

#[tokio::test]
async fn shared_client_preserves_safe_gateway_diagnostics() {
    let (url, server) = error_server("tools/list", |id| {
        Json(json!({
        "jsonrpc":"2.0", "id":id,
        "error":{"code":-32603, "message":"synthetic-private-upstream-error", "data":{
            "mcp_error":{"kind":"http", "status":429}, "private":"synthetic-private-upstream-error"
        }}
    })).into_response()
    })
    .await;
    let client = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("client");
    let error = client.list_tools().await.expect_err("upstream failure");

    assert_matches!(error, McpClientError::Http { status: 429 });
    assert!(!format!("{error:?} {error}").contains("synthetic-private"));
    server.abort();
}

#[tokio::test]
async fn shared_client_rejects_unrecognized_gateway_diagnostics() {
    let (url, server) = error_server("tools/list", |id| {
        Json(json!({
        "jsonrpc":"2.0", "id":id,
        "error":{"code":-32002, "message":"synthetic-private-upstream-error", "data":{
            "mcp_error":{"kind":"http", "status":403, "message":"synthetic-private-upstream-error"}
        }}
    })).into_response()
    })
    .await;
    let client = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("client");
    let error = client.list_tools().await.expect_err("upstream failure");

    assert_matches!(error, McpClientError::JsonRpc { code: -32002 });
    assert!(!format!("{error:?} {error}").contains("synthetic-private"));
    server.abort();
}

#[tokio::test]
async fn shared_client_keeps_untyped_http_bodies_private() {
    let (url, server) = error_server("tools/list", |_| {
        (
            StatusCode::FORBIDDEN,
            "synthetic-private-upstream-error SERVICE_DISABLED",
        )
            .into_response()
    })
    .await;
    let client = McpClient::http(
        reqwest::Client::new(),
        &url,
        McpAuthorization::None,
        Duration::from_secs(10),
        Duration::from_secs(10),
    )
    .await
    .expect("client");
    let error = client.list_tools().await.expect_err("upstream failure");

    assert_matches!(error, McpClientError::Request);
    assert!(!format!("{error:?} {error}").contains("synthetic-private"));
    assert!(!error.to_string().contains("SERVICE_DISABLED"));
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
    let tools = discover_mcp_connections(&[connection], |error| panic!("{error}")).await;
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

#[tokio::test]
async fn live_catalog_adds_and_removes_a_custom_server_without_replacing_the_executor() {
    let (url, server) = error_server("tools/call", |id| {
        Json(json!({"jsonrpc":"2.0", "id":id, "result":{"content":[{"type":"text", "text":"called once"}]}})).into_response()
    }).await;
    let source = Arc::new(LocalMcpDiscovery::new(Vec::new(), false).expect("source"));
    let executor = DirectToolExecutor::new(live_mcp_tools(source.clone()));
    let context = context(AgentApprovalMode::Auto, None);
    let empty = executor
        .call_tool(
            "mcp_search".into(),
            r#"{"queries":["echo"]}"#.into(),
            context.clone(),
        )
        .await
        .expect("search");
    assert_eq!(
        serde_json::from_str::<Value>(&empty.content).expect("json")["tools"],
        json!({})
    );

    assert_eq!(executor.inventory(&context, true).await.tools.len(), 3);
    source
        .update(
            vec![
                serde_json::from_value(json!({"name":"custom", "config":{"url":url}}))
                    .expect("connection"),
            ],
            false,
        )
        .expect("add");
    assert_matches!(executor.inventory(&context, true).await.tools.last(), Some(tool) if tool.name == "mcp__custom__echo" && tool.enabled);
    context
        .input()
        .set_disabled_tools(["mcp__custom__echo".into()].into());
    assert_matches!(executor.inventory(&context, true).await.tools.last(), Some(tool) if !tool.enabled);
    let blocked = executor
        .call_tool(
            "mcp_call".into(),
            r#"{"calls":[{"name":"mcp__custom__echo","arguments":{"text":"must not run"}}]}"#
                .into(),
            context.clone(),
        )
        .await
        .expect("blocked call");
    assert_eq!(
        serde_json::from_str::<Value>(&blocked.content).expect("json")["results"][0]["success"],
        false
    );
    context.input().set_disabled_tools(Default::default());
    let found = executor
        .call_tool(
            "mcp_search".into(),
            r#"{"queries":["echo"]}"#.into(),
            context.clone(),
        )
        .await
        .expect("search");
    assert_eq!(
        serde_json::from_str::<Value>(&found.content).expect("json")["results"][0]["matches"],
        json!(["mcp__custom__echo"])
    );
    let described = executor
        .call_tool(
            "mcp_describe".into(),
            r#"{"names":["mcp__custom__echo"]}"#.into(),
            context.clone(),
        )
        .await
        .expect("describe");
    assert_eq!(
        serde_json::from_str::<Value>(&described.content).expect("json")["tools"]
            ["mcp__custom__echo"]["parameters"]["required"],
        json!(["text"])
    );
    let called = executor
        .call_tool(
            "mcp_call".into(),
            r#"{"calls":[{"name":"mcp__custom__echo","arguments":{"text":"hello"}}]}"#.into(),
            context.clone(),
        )
        .await
        .expect("call");
    assert_eq!(
        serde_json::from_str::<Value>(&called.content).expect("json")["results"][0]["content"],
        "called once"
    );

    source.update(Vec::new(), false).expect("remove");
    assert_eq!(executor.inventory(&context, true).await.tools.len(), 3);
    let removed = executor
        .call_tool(
            "mcp_call".into(),
            r#"{"calls":[{"name":"mcp__custom__echo","arguments":{"text":"must not run"}}]}"#
                .into(),
            context,
        )
        .await
        .expect("call");
    assert_eq!(
        serde_json::from_str::<Value>(&removed.content).expect("json")["results"][0]["success"],
        false
    );
    server.abort();
}

#[tokio::test]
async fn replacing_connections_during_approval_does_not_dispatch_the_old_route() {
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = calls.clone();
    let (url, server) = error_server("tools/call", move |id| {
        recorded.fetch_add(1, Ordering::SeqCst);
        Json(json!({"jsonrpc":"2.0", "id":id, "result":{"content":[]}})).into_response()
    })
    .await;
    let source = Arc::new(
        LocalMcpDiscovery::new(
            vec![
                serde_json::from_value(json!({"name":"custom", "config":{"url":url}}))
                    .expect("connection"),
            ],
            false,
        )
        .expect("source"),
    );
    let executor = DirectToolExecutor::new(live_mcp_tools(source.clone()));
    let (hook, mut approvals) = ApprovalHook::channel();
    let execute = executor.call_tool(
        "mcp_call".into(),
        r#"{"calls":[{"name":"mcp__custom__echo","arguments":{"text":"must not run"}}]}"#.into(),
        context(AgentApprovalMode::Hitl, Some(hook)),
    );
    let approve = async {
        let approval = approvals.recv().await.expect("approval");
        source
            .update(Vec::new(), false)
            .expect("remove while awaiting approval");
        approval
            .response
            .send(ApprovalDecision::new(true, ApprovalSource::Manual))
            .expect("approve");
    };
    let (output, ()) = tokio::join!(execute, approve);
    let output: Value = serde_json::from_str(&output.expect("call").content).expect("json");
    assert_eq!(output["results"][0]["success"], false);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    server.abort();
}

#[tokio::test]
async fn tool_access_blocks_nested_mcp_calls_locally_and_remotely() {
    let calls = Arc::new(Calls::default());
    let local = DirectToolExecutor::new(mcp_tools(vec![tool(calls.clone())]).expect("tools"));
    let remote = RemoteToolExecutor::new(
        mcp_tools(vec![tool(calls.clone())]).expect("tools"),
        reqwest::Url::parse("http://127.0.0.1:1/exec").expect("URL"),
    )
    .expect("remote");
    let context = context(AgentApprovalMode::Auto, None);
    context
        .input()
        .set_disabled_tools(["mcp__issues__createIssue".into()].into());
    let arguments =
        r#"{"calls":[{"name":"mcp__issues__createIssue","arguments":{"title":"must not run"}}]}"#;
    let local_result = local
        .call_tool("mcp_call".into(), arguments.into(), context.clone())
        .await
        .expect("local");
    let remote_result = remote
        .call_tool("mcp_call".into(), arguments.into(), context.clone())
        .await
        .expect("remote");
    assert_eq!(
        serde_json::from_str::<Value>(&local_result.content).expect("result")["results"][0]
            ["success"],
        false
    );
    assert_eq!(remote_result.content, local_result.content);
    assert_eq!(calls.0.load(Ordering::SeqCst), 0);
    let inventory = local.inventory(&context, true).await;
    assert_matches!(inventory.tools.iter().find(|tool| tool.name == "mcp__issues__createIssue"), Some(tool) if !tool.enabled && tool.source == "issues" && tool.parent.as_deref() == Some("mcp_call"));
    context.input().set_disabled_tools(Default::default());
    let allowed = local
        .call_tool("mcp_call".into(), arguments.into(), context)
        .await
        .expect("enabled");
    assert_eq!(
        serde_json::from_str::<Value>(&allowed.content).expect("result")["results"][0]["success"],
        true
    );
    assert_eq!(calls.0.load(Ordering::SeqCst), 1);
}

struct EchoNames(Vec<&'static str>);

impl McpRouter for EchoNames {
    fn list_tools<'a>(
        &'a self,
        _server: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Tool>, ToolError>> + Send + 'a>> {
        Box::pin(async {
            Ok(self
                .0
                .iter()
                .map(|name| {
                    Tool::new(
                        *name,
                        "Echo routing names",
                        Arc::new(serde_json::from_value(json!({"type": "object"})).unwrap()),
                    )
                })
                .collect())
        })
    }

    fn call_tool<'a>(
        &'a self,
        server: &'a str,
        tool: &'a str,
        _arguments: Map<String, Value>,
        _tool_call_id: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "{server}/{tool}"
            ))]))
        })
    }
}

#[tokio::test]
async fn long_names_remain_distinct_selectable_and_route_original_names() {
    let server = "hosted_finance_operations_and_invoice_reporting_for_the_entire_team";
    let router = Arc::new(EchoNames(vec![
        "download_invoice_for_current_billing_period",
        "download_invoice_for_previous_billing_period",
    ]));
    let tools = McpTool::discover(router.clone(), server, false)
        .await
        .unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].name().len(), 64);
    assert_ne!(tools[0].name(), tools[1].name());
    assert!(McpTool::matches_server(tools[0].name(), server));
    assert!(!McpTool::matches_server(tools[0].name(), "hosted_mcp"));
    let again = McpTool::discover(router, server, false).await.unwrap();
    assert_eq!(tools[0].name(), again[0].name());
    assert_eq!(
        tools[0]
            .execute("{}", &context(AgentApprovalMode::Auto, None))
            .await
            .content,
        format!("{server}/download_invoice_for_current_billing_period")
    );
}

#[tokio::test]
async fn short_server_with_long_tool_keeps_the_original_binding() {
    let router = Arc::new(EchoNames(vec![
        "download_invoice_and_all_attachments_for_current_billing_period",
    ]));
    let tools = McpTool::discover(router, "hosted_invoices", false)
        .await
        .unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name().len(), 64);
    assert!(McpTool::matches_server(tools[0].name(), "hosted_invoices"));
    assert_eq!(
        tools[0]
            .execute("{}", &context(AgentApprovalMode::Auto, None))
            .await
            .content,
        "hosted_invoices/download_invoice_and_all_attachments_for_current_billing_period"
    );
}
