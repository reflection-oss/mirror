use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rmcp::transport::auth::{
    AuthError, CredentialRefreshGuard, CredentialStore, OAuthClientConfig, StoredCredentials,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{
    discover_tools, valid_mcp_name, valid_mcp_server_name, valid_mcp_url, McpAuthorization,
    McpClient, McpCredentials, McpTool,
};
use crate::tools::ToolError;

/// Resolves the CLI's environment references and file-backed OAuth credentials.
struct LocalMcpCredentials<'a>(&'a LocalMcpConnection);

/// Configuration for one MCP server used by the local CLI worker.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LocalMcpConnection {
    pub name: String,
    config: ConnectionConfig,
    /// CLI OAuth storage on the local host.
    #[serde(default)]
    credential_directory: PathBuf,
}

/// A secret-free, runtime-resolved MCP server declaration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConnectionConfig {
    url: Option<String>,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    auth: Option<McpAuth>,
    enabled_tools: Option<Vec<String>>,
    #[serde(default = "startup_timeout")]
    startup_timeout_seconds: f64,
    #[serde(default = "tool_timeout")]
    tool_timeout_seconds: f64,
}

/// Credentials are looked up by the runtime, never serialized into a tool definition.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum McpAuth {
    Bearer {
        #[serde(rename = "tokenEnvVar")]
        token_env_var: String,
    },
    Oauth {
        credential: Option<String>,
    },
}

/// OAuth storage compatible with the CLI's interactive sign-in files.
struct OAuthFile(PathBuf);

#[async_trait::async_trait]
impl McpCredentials for LocalMcpCredentials<'_> {
    type Error = AuthError;

    async fn authorize(&self, url: &str) -> Result<McpAuthorization, AuthError> {
        match &self.0.config.auth {
            None => Ok(McpAuthorization::None),
            Some(McpAuth::Bearer { token_env_var }) => std::env::var(token_env_var)
                .map(McpAuthorization::Bearer)
                .map_err(|_| AuthError::AuthorizationRequired),
            Some(McpAuth::Oauth { credential }) => {
                let credential = credential.as_deref().unwrap_or(&self.0.name);
                let directory = &self.0.credential_directory;
                if !valid_mcp_name(credential) || directory.as_os_str().is_empty() {
                    return Err(AuthError::AuthorizationRequired);
                }
                let store = OAuthFile(directory.join(format!("{credential}.json")));
                let config = store.client_config().await?;
                McpAuthorization::oauth(url, store, config).await
            }
        }
    }
}

impl LocalMcpConnection {
    async fn discover(&self) -> Result<Vec<Arc<McpTool>>, ToolError> {
        let startup_timeout = Duration::try_from_secs_f64(self.config.startup_timeout_seconds)
            .map_err(|_| ToolError::InvalidArguments("Invalid MCP startup timeout".into()))?;
        let tool_timeout = Duration::try_from_secs_f64(self.config.tool_timeout_seconds)
            .map_err(|_| ToolError::InvalidArguments("Invalid MCP tool timeout".into()))?;
        if !valid_mcp_server_name(&self.name) || startup_timeout.is_zero() || tool_timeout.is_zero()
        {
            return Err(ToolError::InvalidArguments(
                "Invalid MCP name or timeout".into(),
            ));
        }
        tokio::time::timeout(startup_timeout, async {
            let client = Arc::new(self.connect(startup_timeout, tool_timeout).await?);
            let mut tools = McpTool::discover(client, &self.name, false).await?;
            tools.retain(|tool| {
                self.config
                    .enabled_tools
                    .as_ref()
                    .is_none_or(|names| names.contains(&tool.binding().tool))
            });
            Ok(tools)
        })
        .await
        .unwrap_or_else(|_| {
            Err(ToolError::Execution(format!(
                "MCP startup timed out for {}",
                self.name
            )))
        })
    }

    async fn connect(
        &self,
        startup_timeout: Duration,
        tool_timeout: Duration,
    ) -> Result<McpClient, ToolError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let failure = || {
            ToolError::Execution(format!(
                "Cannot connect MCP server {}; check configuration or /connect {}",
                self.name, self.name
            ))
        };
        match (&self.config.url, &self.config.command) {
            (Some(url), None) => {
                if !valid_mcp_url(url) {
                    return Err(failure());
                }
                let authorization = LocalMcpCredentials(self)
                    .authorize(url)
                    .await
                    .map_err(|_| failure())?;
                let http = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|_| failure())?;
                McpClient::http(http, url, authorization, startup_timeout, tool_timeout)
                    .await
                    .map_err(|_| failure())
            }
            (None, Some(command)) => {
                if self.config.auth.is_some() {
                    return Err(failure());
                }
                let mut process = tokio::process::Command::new(command);
                process.args(&self.config.args).env_clear();
                // A configured child does not inherit the agent's provider authentication.
                for key in ["PATH", "HOME", "TMPDIR", "LANG"] {
                    if let Some(value) = std::env::var_os(key) {
                        process.env(key, value);
                    }
                }
                for (key, source) in &self.config.env {
                    process.env(key, std::env::var(source).map_err(|_| failure())?);
                }
                McpClient::stdio(process, startup_timeout, tool_timeout)
                    .await
                    .map_err(|_| failure())
            }
            _ => Err(ToolError::InvalidArguments(
                "MCP connection must have exactly one of url or command".into(),
            )),
        }
    }
}

/// Discover local connections concurrently; one unavailable server does not disable others.
pub async fn discover_mcp_connections(connections: &[LocalMcpConnection]) -> Vec<Arc<McpTool>> {
    discover_tools(
        connections.iter().map(LocalMcpConnection::discover),
        |error| eprintln!("{error}"),
    )
    .await
}

fn startup_timeout() -> f64 {
    15.0
}
fn tool_timeout() -> f64 {
    300.0
}

impl OAuthFile {
    async fn client_config(&self) -> Result<OAuthClientConfig, AuthError> {
        let path = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let value = read(&path)?;
            let info = &value["client_info"];
            let client_id = info["client_id"]
                .as_str()
                .ok_or(AuthError::AuthorizationRequired)?;
            let redirect = info["redirect_uris"][0]
                .as_str()
                .ok_or(AuthError::AuthorizationRequired)?;
            let mut config = OAuthClientConfig::new(client_id, redirect);
            config.client_secret = info["client_secret"].as_str().map(str::to_owned);
            Ok(config)
        })
        .await
        .map_err(|_| failure("Credential reader stopped"))?
    }
}

fn read(path: &Path) -> Result<Value, AuthError> {
    let file = File::open(path).map_err(|_| AuthError::AuthorizationRequired)?;
    serde_json::from_reader(file)
        .map_err(|_| AuthError::CredentialStoreError("Invalid MCP credential file".into()))
}

fn write(path: &Path, value: &Value) -> Result<(), AuthError> {
    let parent = path
        .parent()
        .ok_or_else(|| failure("Missing credential directory"))?;
    let mut file = tempfile::NamedTempFile::new_in(parent)
        .map_err(|_| failure("Cannot create credential file"))?;
    serde_json::to_writer(&mut file, value).map_err(|_| failure("Cannot write credential file"))?;
    file.flush()
        .map_err(|_| failure("Cannot flush credential file"))?;
    file.persist(path)
        .map_err(|_| failure("Cannot save credential file"))?;
    Ok(())
}

fn failure(message: &str) -> AuthError {
    AuthError::CredentialStoreError(message.to_owned())
}

#[async_trait::async_trait]
impl CredentialStore for OAuthFile {
    async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
        let path = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let value = read(&path)?;
            let Some(tokens) = value.get("tokens").filter(|value| !value.is_null()) else {
                return Ok(None);
            };
            let client_id = value["client_info"]["client_id"]
                .as_str()
                .ok_or(AuthError::AuthorizationRequired)?
                .to_owned();
            let token_response = serde_json::from_value(tokens.clone())
                .map_err(|_| failure("Invalid OAuth token file"))?;
            let received = value["expires_at"]
                .as_f64()
                .zip(tokens["expires_in"].as_f64())
                .map(|(expiry, lifetime)| (expiry - lifetime).max(0.0) as u64);
            Ok(Some(
                StoredCredentials::new(
                    client_id,
                    Some(token_response),
                    tokens["scope"]
                        .as_str()
                        .unwrap_or_default()
                        .split_whitespace()
                        .map(str::to_owned)
                        .collect(),
                    received,
                )
                .with_issuer(value["issuer"].as_str().map(str::to_owned)),
            ))
        })
        .await
        .map_err(|_| failure("Credential reader stopped"))?
    }

    async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
        let path = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let mut value = read(&path)?;
            let tokens = serde_json::to_value(credentials.token_response)
                .map_err(|_| failure("Cannot encode OAuth token"))?;
            value["expires_at"] = credentials
                .token_received_at
                .zip(tokens["expires_in"].as_u64())
                .map(|(start, lifetime)| json!(start.saturating_add(lifetime)))
                .unwrap_or(Value::Null);
            value["tokens"] = tokens;
            value["issuer"] = json!(credentials.issuer);
            write(&path, &value)
        })
        .await
        .map_err(|_| failure("Credential writer stopped"))?
    }

    async fn clear(&self) -> Result<(), AuthError> {
        let path = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let mut value = read(&path)?;
            if let Some(object) = value.as_object_mut() {
                object.remove("tokens");
                object.remove("expires_at");
            }
            write(&path, &value)
        })
        .await
        .map_err(|_| failure("Credential writer stopped"))?
    }

    async fn acquire_refresh_guard(&self) -> Result<Option<CredentialRefreshGuard>, AuthError> {
        let path = self.0.with_extension("lock");
        tokio::task::spawn_blocking(move || {
            let file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .mode(0o600)
                .open(path)
                .map_err(|_| failure("Cannot open credential lock"))?;
            file.lock()
                .map_err(|_| failure("Cannot lock credentials"))?;
            Ok(Some(CredentialRefreshGuard::new(file)))
        })
        .await
        .map_err(|_| failure("Credential lock task stopped"))?
    }
}
