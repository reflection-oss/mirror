//! Streaming Responses provider billed through a Codex ChatGPT login.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use async_stream::try_stream;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use futures_core::Stream;
use futures_util::pin_mut;
use futures_util::StreamExt;
use mirror_protocol::chat_record;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use serde_json::Value;

use super::apply_extra_body;
use super::Llm;
use super::LlmError;
use super::ModelRequestGuard;
use super::ResponsesClient;
use super::ResponsesConfig;
use super::StreamEvent;
use crate::providers::CompletionRequest;

#[cfg(test)]
#[path = "codex_responses_test.rs"]
mod tests;

pub(super) const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const LOGIN_HINT: &str = "log in with the Codex CLI first: `codex login --device-auth`";
const TOKEN_REFRESH_WINDOW: Duration = Duration::from_secs(300);

/// Configuration for the ChatGPT plan-billed Codex backend.
#[derive(Debug, Clone)]
pub struct CodexResponsesConfig {
    pub model: String,
    pub base_url: Option<String>,
    pub auth_path: Option<PathBuf>,
    pub prompt_cache_key: String,
}

/// A Responses provider authenticated by the Codex CLI login.
pub struct CodexResponses {
    responses: ResponsesClient,
    headers: HeaderMap,
    auth_path: PathBuf,
    auth_client: Client,
}

impl CodexResponses {
    /// Validate that a plan-billed Codex login is available.
    pub fn validate_login(auth_path: Option<&Path>) -> Result<(), LlmError> {
        match auth_path {
            Some(path) => load_auth(path)?,
            None => load_auth(&default_auth_path()?)?,
        };
        Ok(())
    }

    /// Create a provider from the OAuth credentials written by `codex login`.
    pub fn new(config: CodexResponsesConfig, client: Client) -> Result<Self, LlmError> {
        let auth_path = match config.auth_path {
            Some(path) => path,
            None => default_auth_path()?,
        };
        Self::validate_login(Some(&auth_path))?;
        let mut headers = HeaderMap::new();
        headers.insert("originator", HeaderValue::from_static("mirror"));
        headers.insert(
            reqwest::header::USER_AGENT,
            HeaderValue::from_static(super::USER_AGENT),
        );
        insert_header(&mut headers, "session-id", &config.prompt_cache_key)?;
        insert_header(
            &mut headers,
            "x-client-request-id",
            &config.prompt_cache_key,
        )?;
        let responses = ResponsesClient::new(
            ResponsesConfig {
                model: config.model,
                api_key: String::new(),
                base_url: config
                    .base_url
                    .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
                prompt_cache_key: Some(config.prompt_cache_key),
            },
            client.clone(),
        );
        Ok(Self {
            responses,
            headers,
            auth_path,
            auth_client: client,
        })
    }

    fn request_body(&self, mut request: CompletionRequest) -> Result<Value, LlmError> {
        let instructions = request
            .messages
            .iter()
            .filter_map(|message| match message {
                chat_record::Message::System(message) => Some(message.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        request
            .messages
            .retain(|message| !matches!(message, chat_record::Message::System(_)));
        request.max_tokens = None;
        let inference = request.inference.clone();
        let mut body = self.responses.request_body(request)?;
        let Value::Object(fields) = &mut body else {
            return Err(LlmError::Request("expected a JSON object".to_string()));
        };
        fields.insert("instructions".to_string(), Value::String(instructions));
        fields.insert("stream".to_string(), Value::Bool(true));
        fields.insert("text".to_string(), json!({"verbosity": "low"}));
        fields.insert("tool_choice".to_string(), json!("auto"));
        fields.insert("parallel_tool_calls".to_string(), Value::Bool(true));
        fields.remove("max_output_tokens");
        if let Some(reasoning) = fields.get_mut("reasoning").and_then(Value::as_object_mut) {
            reasoning.insert("summary".to_string(), json!("auto"));
        }
        // The Responses builder already merged the caller's extra body, but
        // the Codex fields above replace some of those keys; merging again
        // keeps the caller's spelling authoritative.
        apply_extra_body(fields, inference.as_ref())?;
        Ok(body)
    }

    async fn fresh_auth(&self, rejected_token: Option<&str>) -> Result<AuthTokens, LlmError> {
        let auth_path = self.auth_path.clone();
        let mut auth = tokio::task::spawn_blocking(move || load_auth(&auth_path))
            .await
            .map_err(|error| {
                LlmError::Request(format!("Codex credential task failed: {error}"))
            })??;
        let should_refresh = match rejected_token {
            Some(token) => auth.access_token == token,
            None => token_needs_refresh(&auth.access_token),
        };
        if !should_refresh {
            return Ok(auth);
        }
        if auth.refresh_token.is_empty() {
            return Err(LlmError::Request(format!(
                "Codex login expired and has no refresh token; {LOGIN_HINT}"
            )));
        }
        let response = self
            .auth_client
            .post(TOKEN_URL)
            .json(&json!({
                "client_id": CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": auth.refresh_token,
            }))
            .send()
            .await
            .map_err(|error| LlmError::Request(format!("Codex token refresh failed: {error}")))?;
        if !response.status().is_success() {
            return Err(LlmError::Request(format!(
                "Codex token refresh failed ({}); {LOGIN_HINT}",
                response.status()
            )));
        }
        let refreshed: RefreshResponse = response
            .json()
            .await
            .map_err(|error| LlmError::Request(format!("invalid Codex token response: {error}")))?;
        auth.access_token = refreshed.access_token;
        if let Some(refresh_token) = refreshed.refresh_token {
            auth.refresh_token = refresh_token;
        }
        if let Some(id_token) = refreshed.id_token {
            auth.id_token = id_token;
        }
        let auth_path = self.auth_path.clone();
        let persisted_auth = auth.clone();
        tokio::task::spawn_blocking(move || persist_auth(&auth_path, &persisted_auth))
            .await
            .map_err(|error| {
                LlmError::Request(format!("Codex credential task failed: {error}"))
            })??;
        Ok(auth)
    }
}

impl Llm for CodexResponses {
    fn model(&self) -> &str {
        self.responses.model()
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> {
        self.stream_guarded(request, None)
    }
}

impl CodexResponses {
    pub(super) fn stream_guarded<'a>(
        &'a self,
        request: CompletionRequest,
        guard: Option<&'a dyn ModelRequestGuard>,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + 'a {
        try_stream! {
            let body = self.request_body(request)?;
            let mut rejected_token = None;
            for attempt in 0..2 {
                let auth = self.fresh_auth(rejected_token.as_deref()).await?;
                let mut headers = self.headers.clone();
                insert_header(&mut headers, "chatgpt-account-id", &auth.account_id)?;
                let attempted_token = auth.access_token.clone();
                let events = self.responses.stream_body(
                    Ok(body.clone()),
                    headers,
                    auth.access_token,
                    guard,
                );
                pin_mut!(events);
                let mut retry = false;
                while let Some(event) = events.next().await {
                    match event {
                        Err(error) if attempt == 0 && is_unauthorized(&error) => {
                            rejected_token = Some(attempted_token);
                            retry = true;
                            break;
                        }
                        event => yield event?,
                    }
                }
                if !retry {
                    return;
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct AuthFile {
    tokens: Option<AuthTokens>,
}

#[derive(Clone, Deserialize)]
struct AuthTokens {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    refresh_token: String,
    #[serde(default)]
    account_id: String,
    #[serde(default)]
    id_token: String,
}

#[derive(Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

#[derive(Deserialize)]
struct JwtClaims {
    exp: Option<u64>,
}

fn default_auth_path() -> Result<PathBuf, LlmError> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| LlmError::Request(format!("HOME is not set; {LOGIN_HINT}")))?;
    Ok(PathBuf::from(home).join(".codex/auth.json"))
}

fn load_auth(path: &Path) -> Result<AuthTokens, LlmError> {
    let contents = std::fs::read_to_string(path).map_err(|error| {
        LlmError::Request(format!(
            "unreadable Codex credentials at {}: {error}; {LOGIN_HINT}",
            path.display()
        ))
    })?;
    let auth: AuthFile = serde_json::from_str(&contents).map_err(|error| {
        LlmError::Request(format!(
            "unreadable Codex credentials at {}: {error}",
            path.display()
        ))
    })?;
    let tokens = auth.tokens.ok_or_else(|| missing_login(path))?;
    if tokens.access_token.is_empty() || tokens.account_id.is_empty() {
        return Err(missing_login(path));
    }
    Ok(tokens)
}

fn missing_login(path: &Path) -> LlmError {
    LlmError::Request(format!(
        "no ChatGPT login in {} (API-key mode is not plan-billed); {LOGIN_HINT}",
        path.display()
    ))
}

fn is_unauthorized(error: &LlmError) -> bool {
    matches!(error, LlmError::Request(detail) if detail.starts_with("401 Unauthorized:"))
}

fn token_needs_refresh(access_token: &str) -> bool {
    let Some(payload) = access_token.split('.').nth(1) else {
        return false;
    };
    let Ok(decoded) = URL_SAFE_NO_PAD.decode(payload) else {
        return false;
    };
    let Ok(claims) = serde_json::from_slice::<JwtClaims>(&decoded) else {
        return false;
    };
    let Some(expiry) = claims.exp else {
        return false;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    expiry.saturating_sub(now) < TOKEN_REFRESH_WINDOW.as_secs()
}

fn persist_auth(path: &Path, auth: &AuthTokens) -> Result<(), LlmError> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| LlmError::Request(format!("could not read Codex login: {error}")))?;
    let mut data: Value = serde_json::from_str(&contents)
        .map_err(|error| LlmError::Request(format!("could not update Codex login: {error}")))?;
    let tokens = data
        .get_mut("tokens")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| missing_login(path))?;
    tokens.insert("access_token".to_string(), json!(auth.access_token));
    tokens.insert("refresh_token".to_string(), json!(auth.refresh_token));
    if !auth.id_token.is_empty() {
        tokens.insert("id_token".to_string(), json!(auth.id_token));
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("auth"),
        nonce
    ));
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, &data)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        std::fs::rename(&temporary, path)
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(LlmError::Request(format!(
            "could not persist refreshed Codex login: {error}"
        )));
    }
    Ok(())
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) -> Result<(), LlmError> {
    let value = HeaderValue::from_str(value)
        .map_err(|error| LlmError::Request(format!("invalid {name} header: {error}")))?;
    headers.insert(name, value);
    Ok(())
}
