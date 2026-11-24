use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientConfig, JsonObject, ProtocolVersion,
    RequestMetaObject, Tool,
};
use rmcp::service::{
    ClientInitializeError, ClientLifecycleMode, ClientServiceExt, RunningService, ServiceError,
};
use rmcp::transport::auth::{AuthClient, AuthError};
use rmcp::transport::streamable_http_client::{
    AuthRequiredError, InsufficientScopeError, StreamableHttpClient,
    StreamableHttpClientTransportConfig, StreamableHttpError,
};
use rmcp::transport::{
    DynamicTransportError, IntoTransport, StreamableHttpClientTransport, TokioChildProcess,
};
use rmcp::{ErrorData, RoleClient};
use serde::{Deserialize, Serialize};

use super::{valid_mcp_url, McpAuthorization, McpRouter};
use crate::tools::ToolError;

/// An MCP session with bounded discovery and calls, independent of credential storage.
pub struct McpClient {
    service: RunningService<RoleClient, ClientConfig>,
    startup_timeout: Duration,
    tool_timeout: Duration,
}

/// Payload-free failures suitable for diagnostics even on authenticated transports.
#[derive(Debug, Deserialize, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum McpClientError {
    #[error("cannot connect to MCP server; check network access and server configuration")]
    Connection,
    #[error("MCP request timed out; its outcome may be unknown, so do not blindly retry writes")]
    Timeout,
    #[error("MCP request failed; it may have executed, so do not blindly retry writes")]
    Request,
    #[error("MCP authentication failed; check the connected account")]
    Authentication,
    #[error("MCP access denied; check account permissions and server API configuration")]
    PermissionDenied,
    #[error("MCP account is missing or ambiguous; connect or select an account")]
    ConnectionRequired,
    #[error(
        "MCP credentials are unavailable; check the account or contact the server administrator"
    )]
    CredentialUnavailable,
    #[error("MCP server returned HTTP {status}; do not blindly retry writes")]
    Http { status: u16 },
    #[error("MCP server returned JSON-RPC error {code}; do not blindly retry writes")]
    JsonRpc { code: i32 },
}

impl McpClient {
    /// Connect over HTTP using authentication already resolved by the runtime.
    pub async fn http(
        http: reqwest::Client,
        url: &str,
        authorization: McpAuthorization,
        startup_timeout: Duration,
        tool_timeout: Duration,
    ) -> Result<Self, McpClientError> {
        if !valid_mcp_url(url) {
            return Err(McpClientError::Connection);
        }
        let _ = rustls::crypto::ring::default_provider().install_default();
        let config = StreamableHttpClientTransportConfig::with_uri(url.to_owned());
        let service = match authorization {
            McpAuthorization::None => connect_http(http, config, startup_timeout).await,
            McpAuthorization::Bearer(token) => {
                connect_http(http, config.auth_header(token), startup_timeout).await
            }
            McpAuthorization::OAuth(manager) => {
                connect_http(AuthClient::new(http, *manager), config, startup_timeout).await
            }
        }?;
        Ok(Self {
            service,
            startup_timeout,
            tool_timeout,
        })
    }

    /// Launch a stdio server; the caller supplies its explicitly scoped environment.
    pub async fn stdio(
        process: impl Fn() -> tokio::process::Command,
        startup_timeout: Duration,
        tool_timeout: Duration,
    ) -> Result<Self, McpClientError> {
        let service = connect(startup_timeout, || {
            let mut process = process();
            process.stderr(Stdio::null());
            TokioChildProcess::new(process).map_err(|_| McpClientError::Connection)
        })
        .await?;
        Ok(Self {
            service,
            startup_timeout,
            tool_timeout,
        })
    }

    /// Connect to a caller-owned byte stream, such as stdio in an isolated worker.
    pub async fn stream(
        reader: impl tokio::io::AsyncRead + Unpin + Send + 'static,
        writer: impl tokio::io::AsyncWrite + Unpin + Send + 'static,
        startup_timeout: Duration,
        tool_timeout: Duration,
    ) -> Result<Self, McpClientError> {
        let service = tokio::time::timeout(
            startup_timeout,
            client_config().serve_with_lifecycle((reader, writer), ClientLifecycleMode::Initialize),
        )
        .await
        .map_err(|_| McpClientError::Timeout)?
        .map_err(initialize_error)?;
        Ok(Self {
            service,
            startup_timeout,
            tool_timeout,
        })
    }

    /// Whether the transport can no longer serve requests.
    pub fn is_closed(&self) -> bool {
        self.service.is_closed()
    }

    /// Discover the complete upstream catalog within its discovery deadline.
    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpClientError> {
        request(self.startup_timeout, self.service.list_all_tools()).await
    }

    /// Invoke one tool without retrying ambiguous failures.
    pub async fn call_tool(
        &self,
        tool: &str,
        arguments: Option<JsonObject>,
        meta: Option<RequestMetaObject>,
    ) -> Result<CallToolResult, McpClientError> {
        let mut params = CallToolRequestParams::new(tool.to_owned());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
        params.meta = meta;
        request(self.tool_timeout, self.service.call_tool(params)).await
    }
}

impl McpRouter for McpClient {
    fn list_tools<'a>(
        &'a self,
        server: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Tool>, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            self.list_tools().await.map_err(|error| {
                ToolError::Execution(format!("MCP discovery for {server}: {error}"))
            })
        })
    }

    fn call_tool<'a>(
        &'a self,
        server: &'a str,
        tool: &'a str,
        arguments: JsonObject,
        _tool_call_id: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            self.call_tool(tool, Some(arguments), None)
                .await
                .map_err(|error| {
                    ToolError::Execution(format!("MCP call for {server}: {error}; check /connect"))
                })
        })
    }
}

async fn connect_http<C: StreamableHttpClient>(
    client: C,
    config: StreamableHttpClientTransportConfig,
    startup_timeout: Duration,
) -> Result<RunningService<RoleClient, ClientConfig>, McpClientError> {
    connect(startup_timeout, || {
        Ok(StreamableHttpClientTransport::with_client(
            client.clone(),
            config.clone(),
        ))
    })
    .await
}

async fn connect<T, E, A>(
    startup_timeout: Duration,
    transport: impl Fn() -> Result<T, McpClientError>,
) -> Result<RunningService<RoleClient, ClientConfig>, McpClientError>
where
    T: IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    tokio::time::timeout(startup_timeout, async {
        let auto = ClientLifecycleMode::Auto {
            preferred_versions: vec![ProtocolVersion::V_2026_07_28],
            legacy_version: Some(ProtocolVersion::LATEST_WITH_INITIALIZE),
        };
        let error = match client_config()
            .serve_with_lifecycle(transport()?, auto)
            .await
        {
            Ok(service) => return Ok(service),
            Err(error) => error,
        };
        let initialized = matches!(error, ClientInitializeError::LegacyFallbackFailed { .. });
        let error = initialize_error(error);
        if initialized
            || matches!(
                error,
                McpClientError::Authentication | McpClientError::PermissionDenied
            )
        {
            return Err(error);
        }
        client_config()
            .serve_with_lifecycle(transport()?, ClientLifecycleMode::Initialize)
            .await
            .map_err(initialize_error)
    })
    .await
    .map_err(|_| McpClientError::Timeout)?
}

fn client_config() -> ClientConfig {
    ClientConfig::default().with_protocol_version(ProtocolVersion::LATEST_WITH_INITIALIZE)
}

async fn request<T>(
    timeout: Duration,
    request: impl Future<Output = Result<T, ServiceError>>,
) -> Result<T, McpClientError> {
    tokio::time::timeout(timeout, request)
        .await
        .map_err(|_| McpClientError::Timeout)?
        .map_err(|error| match error {
            ServiceError::McpError(error) => rpc_error(error),
            ServiceError::TransportSend(error) => transport_error(error),
            ServiceError::Timeout { .. } => McpClientError::Timeout,
            _ => McpClientError::Request,
        })
}

fn initialize_error(error: ClientInitializeError) -> McpClientError {
    match error {
        ClientInitializeError::TransportError { error, .. } => transport_error(error),
        ClientInitializeError::JsonRpcError(error) => rpc_error(error),
        ClientInitializeError::LegacyFallbackFailed { fallback, .. } => initialize_error(*fallback),
        _ => McpClientError::Connection,
    }
}

fn rpc_error(error: ErrorData) -> McpClientError {
    // Gateways may forward only this closed diagnostic shape, never upstream text or data.
    error
        .data
        .and_then(|data| data.get("mcp_error").cloned())
        .and_then(|data| serde_json::from_value(data).ok())
        .unwrap_or(McpClientError::JsonRpc { code: error.code.0 })
}

fn transport_error(error: DynamicTransportError) -> McpClientError {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error.error.as_ref());
    while let Some(error) = source {
        if error.is::<AuthRequiredError>() {
            return McpClientError::Authentication;
        }
        if error.is::<InsufficientScopeError>() {
            return McpClientError::PermissionDenied;
        }
        match error.downcast_ref::<AuthError>() {
            Some(
                AuthError::AuthorizationRequired
                | AuthError::TokenRefreshRejected(_)
                | AuthError::TokenExpired,
            ) => return McpClientError::Authentication,
            Some(AuthError::InsufficientScope { .. }) => return McpClientError::PermissionDenied,
            _ => {}
        }
        // rmcp's Client variant does not expose its payload through Error::source().
        let http_error = match error.downcast_ref::<StreamableHttpError<reqwest::Error>>() {
            Some(StreamableHttpError::Client(error)) => Some(error),
            _ => error.downcast_ref::<reqwest::Error>(),
        };
        if let Some(error) = http_error {
            if error.is_timeout() {
                return McpClientError::Timeout;
            }
            if let Some(status) = error.status() {
                return McpClientError::Http {
                    status: status.as_u16(),
                };
            }
            if error.is_connect() {
                return McpClientError::Connection;
            }
        }
        source = error.source();
    }
    McpClientError::Request
}
