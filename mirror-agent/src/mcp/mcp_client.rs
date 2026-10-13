use std::future::Future;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;

use rmcp::model::{CallToolRequestParams, CallToolResult, ClientConfig, JsonObject, Tool};
use rmcp::service::{RunningService, ServiceError};
use rmcp::transport::auth::AuthClient;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};

use super::{valid_mcp_url, McpAuthorization, McpRouter};
use crate::tools::ToolError;

/// An MCP session with bounded discovery and calls, independent of credential storage.
pub struct McpClient {
    service: RunningService<RoleClient, ClientConfig>,
    startup_timeout: Duration,
    tool_timeout: Duration,
}

/// Payload-free failures suitable for diagnostics even on authenticated transports.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum McpClientError {
    #[error("cannot connect to MCP server")]
    Connection,
    #[error("MCP request timed out; its outcome may be unknown, so do not blindly retry writes")]
    Timeout,
    #[error("MCP request failed; it may have executed, so do not blindly retry writes")]
    Request,
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
        let service = tokio::time::timeout(startup_timeout, async {
            let mut config = StreamableHttpClientTransportConfig::with_uri(url.to_owned());
            match authorization {
                McpAuthorization::None => {}
                McpAuthorization::Bearer(token) => config = config.auth_header(token),
                McpAuthorization::OAuth(manager) => {
                    return ClientConfig::default()
                        .serve(StreamableHttpClientTransport::with_client(
                            AuthClient::new(http, *manager),
                            config,
                        ))
                        .await
                        .map_err(|_| McpClientError::Connection);
                }
            }
            ClientConfig::default()
                .serve(StreamableHttpClientTransport::with_client(http, config))
                .await
                .map_err(|_| McpClientError::Connection)
        })
        .await
        .map_err(|_| McpClientError::Timeout)??;
        Ok(Self {
            service,
            startup_timeout,
            tool_timeout,
        })
    }

    /// Launch a stdio server; the caller supplies its explicitly scoped environment.
    pub async fn stdio(
        mut process: tokio::process::Command,
        startup_timeout: Duration,
        tool_timeout: Duration,
    ) -> Result<Self, McpClientError> {
        process.stderr(Stdio::null());
        let transport = TokioChildProcess::new(process).map_err(|_| McpClientError::Connection)?;
        let service =
            tokio::time::timeout(startup_timeout, ClientConfig::default().serve(transport))
                .await
                .map_err(|_| McpClientError::Timeout)?
                .map_err(|_| McpClientError::Connection)?;
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
    ) -> Result<CallToolResult, McpClientError> {
        let mut params = CallToolRequestParams::new(tool.to_owned());
        if let Some(arguments) = arguments {
            params = params.with_arguments(arguments);
        }
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
                ToolError::Execution(format!(
                    "MCP discovery for {server}: {error}; check /connect"
                ))
            })
        })
    }

    fn call_tool<'a>(
        &'a self,
        server: &'a str,
        tool: &'a str,
        arguments: JsonObject,
    ) -> Pin<Box<dyn Future<Output = Result<CallToolResult, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            self.call_tool(tool, Some(arguments))
                .await
                .map_err(|error| {
                    ToolError::Execution(format!("MCP call for {server}: {error}; check /connect"))
                })
        })
    }
}

async fn request<T>(
    timeout: Duration,
    request: impl Future<Output = Result<T, ServiceError>>,
) -> Result<T, McpClientError> {
    tokio::time::timeout(timeout, request)
        .await
        .map_err(|_| McpClientError::Timeout)?
        .map_err(|_| McpClientError::Request)
}
