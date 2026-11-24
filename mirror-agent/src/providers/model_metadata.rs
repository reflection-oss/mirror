//! Bounded model-metadata requests and OpenAI-compatible context-limit parsing.

use std::time::Duration;

use reqwest::{Client, RequestBuilder, Response};
use serde_json::Value;

use super::{LlmError, ModelApi, ModelProvider, ModelProviderConfig};
use crate::with_fallback_roots;

pub const DEFAULT_CONTEXT_WINDOW: usize = 256_000;

/// Below this, Mirror's own prompt and tools leave a session little room before compaction.
pub const MINIMUM_AGENT_CONTEXT_WINDOW: usize = 32_768;

/// Warn when a discovered window is too small to work comfortably in.
fn small_context_warning(window: usize) -> Option<String> {
    (window < MINIMUM_AGENT_CONTEXT_WINDOW).then(|| {
        format!(
            "the context window is {window} tokens, and agent sessions work best with at least \
             {MINIMUM_AGENT_CONTEXT_WINDOW}; history will compact early."
        )
    })
}

/// A provider default used because the model's context limit was not discovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextWindowFallback {
    pub reason: &'static str,
    pub tokens: usize,
}

impl ContextWindowFallback {
    /// Describe the fallback without URLs, credentials, or response bodies.
    pub fn warning(&self, model: &str) -> String {
        format!(
            "could not discover the context window for {model} ({}); using a {}-token fallback. \
             Pass --context-window to set it.",
            self.reason, self.tokens
        )
    }
}

/// Discover the model's context limit, using a provider fallback on failure.
pub async fn model_context_window(provider: &ModelProvider, config: &ModelProviderConfig) -> usize {
    resolve_model_context_window(provider, config)
        .await
        .unwrap_or_else(|fallback| fallback.tokens)
}

/// The context window a session can use, and what its user should be told about it.
#[derive(Debug, PartialEq, Eq)]
pub struct SessionContextWindow {
    pub tokens: usize,
    pub warning: Option<String>,
}

/// Discover the window of the selected model.
///
/// A local model fails when it has no allocated window instead of falling
/// back to a default.
pub async fn session_context_window(
    provider: &ModelProvider,
    config: &ModelProviderConfig,
) -> Result<SessionContextWindow, LlmError> {
    let (tokens, warning) = if config.api == ModelApi::LlamaCpp {
        provider.verify_request().await?;
        let (base_url, _) = config.connection().await?;
        let allocated = provider
            .llama_cpp(&base_url)?
            .check_model(&config.model)
            .await?;
        // An explicit window is the user's decision, not a discovery to second-guess.
        match config.context_window {
            Some(limit) => (limit.min(allocated), None),
            None => (allocated, small_context_warning(allocated)),
        }
    } else {
        match resolve_model_context_window(provider, config).await {
            Ok(tokens) => (tokens, None),
            Err(fallback) => (fallback.tokens, Some(fallback.warning(&config.model))),
        }
    };
    Ok(SessionContextWindow { tokens, warning })
}

/// Discover the model's context limit, or report the fallback used instead.
pub async fn resolve_model_context_window(
    provider: &ModelProvider,
    config: &ModelProviderConfig,
) -> Result<usize, ContextWindowFallback> {
    let fallback = match config.api {
        ModelApi::CodexResponses => 272_000,
        ModelApi::AnthropicMessages => 200_000,
        _ => DEFAULT_CONTEXT_WINDOW,
    };
    // Bound connection resolution and authorization, not only the HTTP request.
    tokio::time::timeout(
        Duration::from_secs(10),
        discover_context_window(provider, config),
    )
    .await
    .unwrap_or(Err("model metadata discovery timed out"))
    .map_err(|reason| {
        eprintln!("{reason}; using a {fallback}-token fallback");
        ContextWindowFallback {
            reason,
            tokens: fallback,
        }
    })
}

async fn discover_context_window(
    provider: &ModelProvider,
    config: &ModelProviderConfig,
) -> Result<usize, &'static str> {
    if let Some(window) = config.context_window {
        return Ok(window);
    }
    // Google's window is shared by input, thinking, and output; no catalog probe is needed.
    if config.api == ModelApi::Gemini && config.model == "gemini-4-argon" {
        return Ok(1_048_576);
    }
    // The plan-billed Codex backend has no model catalog.
    if config.api == ModelApi::CodexResponses {
        return Ok(272_000);
    }
    let (base_url, api_key) = provider
        .connection(config)
        .await
        .map_err(|_| "model metadata connection unavailable")?;
    provider
        .verify_request()
        .await
        .map_err(|_| "model metadata request not authorized")?;
    let api_key = (!api_key.is_empty()).then_some(api_key.as_str());
    if config.api != ModelApi::AnthropicMessages {
        return openai_context_window(&base_url, api_key, &config.model).await;
    }
    let mut url = reqwest::Url::parse(&format!("{}/v1/models", base_url.trim_end_matches('/')))
        .map_err(|_| "invalid Anthropic model endpoint")?;
    url.path_segments_mut()
        .map_err(|_| "invalid Anthropic model endpoint")?
        .push(&config.model);
    let mut request = metadata_client()?
        .get(url)
        .header("anthropic-version", "2023-06-01");
    if let Some(key) = api_key {
        request = request.header("x-api-key", key);
    }
    let payload = model_metadata(request).await?;
    // A per-model lookup may resolve an alias to a canonical ID.
    let model = payload["id"]
        .as_str()
        .ok_or("model metadata did not contain an ID")?;
    context_window_from_models(&payload, model)
}

/// Metadata requests must not forward credentials through redirects.
pub fn metadata_client() -> Result<Client, &'static str> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    with_fallback_roots(Client::builder())
        .user_agent(super::USER_AGENT)
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|_| "could not create model metadata client")
}

/// Fetch metadata without including URLs, credentials, or response bodies in diagnostics.
pub async fn model_metadata(request: RequestBuilder) -> Result<Value, &'static str> {
    let response = request
        .send()
        .await
        .map_err(|_| "model metadata request failed")?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(METADATA_NOT_FOUND);
    }
    if !response.status().is_success() {
        return Err("model metadata endpoint returned an error");
    }
    read_limited_json(response).await
}

/// The most model metadata Mirror reads from one response.
pub(crate) const METADATA_LIMIT: usize = 4 * 1024 * 1024;

/// Read a JSON body of at most `METADATA_LIMIT`, without including it in diagnostics.
pub(super) async fn read_limited_json(response: Response) -> Result<Value, &'static str> {
    let body = read_limited(response, METADATA_LIMIT).await?;
    serde_json::from_slice(&body).map_err(|_| "model metadata was not valid JSON")
}

/// Read a body of at most `limit` bytes, failing as soon as it grows past that.
pub(crate) async fn read_limited(
    mut response: Response,
    limit: usize,
) -> Result<Vec<u8>, &'static str> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "could not read model metadata")?
    {
        if body.len() + chunk.len() > limit {
            return Err("model metadata exceeded the size limit");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Query a model catalog shared by OpenAI-compatible providers.
pub async fn openai_context_window(
    base_url: &str,
    api_key: Option<&str>,
    model: &str,
) -> Result<usize, &'static str> {
    let base_url = base_url.trim_end_matches('/');
    let mut request = metadata_client()?.get(format!("{base_url}/models"));
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let listed = context_window_from_models(&model_metadata(request).await?, model);
    if listed != Err(MISSING_MODEL) {
        return listed;
    }
    let mut url = reqwest::Url::parse(&format!("{base_url}/models"))
        .map_err(|_| "invalid model metadata endpoint")?;
    url.path_segments_mut()
        .map_err(|_| "invalid model metadata endpoint")?
        .push(model);
    let mut request = metadata_client()?.get(url);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let payload = model_metadata(request).await.map_err(|error| {
        if error == METADATA_NOT_FOUND {
            MISSING_MODEL
        } else {
            error
        }
    })?;
    let id = payload["id"].as_str().ok_or(MISSING_MODEL)?;
    context_window_from_models(&payload, id)
}

const MISSING_MODEL: &str = "model metadata did not contain the selected model";
const METADATA_NOT_FOUND: &str = "model metadata endpoint was not found";

/// Read only the requested model; never borrow another deployment's limit.
pub fn context_window_from_models(payload: &Value, model: &str) -> Result<usize, &'static str> {
    let entry = payload
        .get("data")
        .and_then(Value::as_array)
        .and_then(|entries| entries.iter().find(|entry| entry["id"] == model))
        .or_else(|| (payload["id"] == model).then_some(payload))
        .ok_or(MISSING_MODEL)?;
    [
        "max_model_len",
        "context_length",
        "max_input_tokens",
        "context_window",
    ]
    .into_iter()
    .find_map(|key| positive_limit(&entry[key]))
    .or_else(|| {
        ["n_ctx", "n_ctx_train"]
            .into_iter()
            .find_map(|key| positive_limit(&entry["meta"][key]))
    })
    .ok_or("model metadata did not report a valid context length")
}

pub(super) fn positive_limit(value: &Value) -> Option<usize> {
    usable_window(value.as_u64()?)
}

/// A token count a context window may be, which budget arithmetic can scale safely.
pub(super) fn usable_window(tokens: u64) -> Option<usize> {
    let tokens = usize::try_from(tokens).ok()?;
    // Budget arithmetic multiplies a window by percentages before dividing.
    (tokens > 0 && tokens <= usize::MAX / 100).then_some(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_reflection_models_contract_without_confusing_output_and_context_limits() {
        let model = json!({
            "id": "reflection-501B-A23B",
            "object": "model",
            "created": 1791158400,
            "owned_by": "reflection",
            "shutdown_date": null,
            "context_length": 262144,
            "max_output_tokens": 65536
        });
        assert_eq!(
            context_window_from_models(&model, "reflection-501B-A23B"),
            Ok(262144)
        );
        assert_eq!(
            context_window_from_models(
                &json!({"object": "list", "data": [model]}),
                "reflection-501B-A23B"
            ),
            Ok(262144)
        );
        assert!(context_window_from_models(
            &json!({"id": "reflection-501B-A23B", "max_output_tokens": 65536}),
            "reflection-501B-A23B",
        )
        .is_err());
    }

    #[tokio::test]
    async fn discovers_reflection_models_contract_over_http() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let base_url = format!(
            "http://{}/openai/v1",
            listener.local_addr().expect("address")
        );
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("connection");
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                socket.read_exact(&mut byte).await.expect("request header");
                request.push(byte[0]);
            }
            let request = String::from_utf8(request).expect("HTTP headers");
            assert!(request.starts_with("GET /openai/v1/models HTTP/1.1\r\n"));
            assert!(request.contains("Bearer synthetic-reflection-key"));
            assert!(request
                .to_ascii_lowercase()
                .contains("\r\nuser-agent: mirror-cli\r\n"));
            let body = r#"{"object":"list","data":[{"id":"reflection-501B-A23B","object":"model","created":1791158400,"owned_by":"reflection","shutdown_date":null,"context_length":262144}]}"#;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .expect("response");
        });
        // Reflection uses the same catalog lookup, but forbids overriding its endpoint.
        let config = ModelProviderConfig {
            context_window: None,
            api: ModelApi::ChatCompletions,
            model: "reflection-501B-A23B".into(),
            api_key: Some("synthetic-reflection-key".into()),
            base_url: Some(base_url),
            auth_path: None,
            prompt_cache_key: "session".into(),
            request_timeout: None,
            managed_reference: None,
        };
        let provider = ModelProvider::new(config.clone(), None)
            .await
            .expect("provider");
        assert_eq!(model_context_window(&provider, &config).await, 262144);
        server.await.expect("server");
    }

    #[tokio::test]
    async fn unlisted_model_resolves_through_the_per_model_endpoint() {
        use crate::tools::web_test_server::{TestResponse, TestServer};
        let mut server = TestServer::start(vec![
            TestResponse::json(
                200,
                json!({"data": [{"id": "beam-alias", "context_length": 1_048_576}]}),
            ),
            TestResponse::json(200, json!({"id": "beam-full", "context_length": 1_048_576})),
        ])
        .await;

        assert_eq!(
            openai_context_window(&server.base_url, Some("synthetic-key"), "beam-full").await,
            Ok(1_048_576)
        );
        assert_eq!(server.request().await.path, "/models");
        let lookup = server.request().await;
        assert_eq!(lookup.path, "/models/beam-full");
        assert_eq!(
            lookup.authorization.as_deref(),
            Some("Bearer synthetic-key")
        );
    }

    #[tokio::test]
    async fn unknown_model_is_missing_after_the_per_model_lookup() {
        use crate::tools::web_test_server::{TestResponse, TestServer};
        let server = TestServer::start(vec![
            TestResponse::json(200, json!({"data": []})),
            TestResponse::json(404, json!({"error": {"code": "model_not_found"}})),
        ])
        .await;

        assert_eq!(
            openai_context_window(&server.base_url, None, "missing").await,
            Err(MISSING_MODEL)
        );
    }

    #[tokio::test]
    async fn per_model_lookup_failure_reports_the_endpoint_error() {
        use crate::tools::web_test_server::{TestResponse, TestServer};
        let server = TestServer::start(vec![
            TestResponse::json(200, json!({"data": []})),
            TestResponse::json(503, json!({"error": "unavailable"})),
        ])
        .await;

        assert_eq!(
            openai_context_window(&server.base_url, None, "beam-full").await,
            Err("model metadata endpoint returned an error")
        );
    }

    #[test]
    fn only_windows_below_the_agent_minimum_warn() {
        assert!(small_context_warning(8192)
            .expect("warning")
            .contains("8192 tokens"));
        assert_eq!(small_context_warning(32_768), None);
    }

    #[test]
    fn fallback_warning_names_the_model_reason_and_override() {
        let fallback = ContextWindowFallback {
            reason: MISSING_MODEL,
            tokens: DEFAULT_CONTEXT_WINDOW,
        };

        assert_eq!(
            fallback.warning("beam-full"),
            "could not discover the context window for beam-full (model metadata did not contain the selected model); using a 256000-token fallback. Pass --context-window to set it."
        );
    }

    #[test]
    fn selects_the_matching_model_and_valid_positive_limit() {
        let payload = json!({"data": [
            {"id": "other", "max_model_len": 999999},
            {"id": "served", "max_model_len": false, "context_length": 131072}
        ]});
        assert_eq!(
            context_window_from_models(&payload, "served").expect("limit"),
            131072
        );
        assert!(context_window_from_models(&payload, "missing").is_err());
        assert_eq!(
            context_window_from_models(
                &json!({"id": "served", "meta": {"n_ctx": 32768}}),
                "served"
            )
            .expect("limit"),
            32768
        );
        assert!(context_window_from_models(
            &json!({"id": "served", "max_model_len": u64::MAX}),
            "served"
        )
        .is_err());
        assert!(context_window_from_models(
            &json!({"id": "served", "context_length": 0}),
            "served"
        )
        .is_err());
    }
}
