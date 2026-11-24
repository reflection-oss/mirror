use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use reqwest::Client;
use reqwest::Url;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use super::LlmError;
use super::ModelProviderConfig;

const METADATA_HOST: &str = "metadata.google.internal";
const TOKEN_PATH: &str = "computeMetadata/v1/instance/service-accounts/default/token";
const TOKEN_TIMEOUT: Duration = Duration::from_secs(5);
const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(120);
const ENDPOINT_SHAPE: &str =
    "https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/endpoints/openapi";

type TokenReply = oneshot::Sender<Result<String, LlmError>>;

#[derive(Clone, Default)]
pub(super) struct VertexTokens {
    requests: Arc<OnceLock<mpsc::Sender<TokenReply>>>,
}

struct WorkloadToken {
    value: String,
    fresh_until: Instant,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

pub fn is_vertex_endpoint(base_url: &str) -> bool {
    let Ok(url) = Url::parse(base_url.trim_end_matches('/')) else {
        return false;
    };
    let host = url.host_str().unwrap_or_default();
    let regional = host
        .strip_suffix("-aiplatform.googleapis.com")
        .is_some_and(is_resource_name);
    let segments: Vec<&str> = url
        .path_segments()
        .map(Iterator::collect)
        .unwrap_or_default();
    url.scheme() == "https"
        && (host == "aiplatform.googleapis.com" || regional)
        && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && matches!(
            segments.as_slice(),
            ["v1", "projects", project, "locations", location, "endpoints", "openapi"]
                if is_resource_name(project) && is_resource_name(location)
        )
}

fn is_resource_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-')
}

pub(super) fn vertex_model(model: &str) -> Result<String, LlmError> {
    let name = model.strip_prefix("google/").unwrap_or(model);
    if !name.starts_with("gemini-") || name.contains('/') {
        return Err(LlmError::Request(format!(
            "vertex serves Gemini models only, such as gemini-4-neon [model={model}]"
        )));
    }
    Ok(format!("google/{name}"))
}

impl VertexTokens {
    pub(super) async fn connection(
        &self,
        config: &ModelProviderConfig,
    ) -> Result<(String, String), LlmError> {
        vertex_model(&config.model)?;
        let base_url = match &config.base_url {
            Some(base_url) => base_url.trim_end_matches('/').to_string(),
            None => {
                let project = nonempty_env("GOOGLE_CLOUD_PROJECT").ok_or_else(|| {
                    LlmError::Request(format!(
                        "vertex needs a base URL shaped like {ENDPOINT_SHAPE}, or GOOGLE_CLOUD_PROJECT"
                    ))
                })?;
                format!(
                    "https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"
                )
            }
        };
        if !is_vertex_endpoint(&base_url) {
            return Err(LlmError::Request(format!(
                "vertex sends Google credentials only to {ENDPOINT_SHAPE}; use openai-chat-completions for a custom endpoint"
            )));
        }
        let supplied = config
            .api_key
            .clone()
            .filter(|token| !token.trim().is_empty())
            .or_else(|| nonempty_env("GOOGLE_OAUTH_ACCESS_TOKEN"));
        let token = match supplied {
            Some(token) => token,
            None => {
                let host =
                    nonempty_env("GCE_METADATA_HOST").unwrap_or_else(|| METADATA_HOST.into());
                self.workload_token(host).await?
            }
        };
        Ok((base_url, token))
    }

    async fn workload_token(&self, metadata_host: String) -> Result<String, LlmError> {
        let requests = self.requests.get_or_init(|| {
            let (requests, received) = mpsc::channel(8);
            tokio::spawn(serve_workload_tokens(received, metadata_host));
            requests
        });
        let (reply, token) = oneshot::channel();
        requests.send(reply).await.map_err(|_| unavailable())?;
        token.await.map_err(|_| unavailable())?
    }
}

async fn serve_workload_tokens(mut requests: mpsc::Receiver<TokenReply>, metadata_host: String) {
    let client = super::llm::build_http_client(
        super::llm::http_client_builder(Some(TOKEN_TIMEOUT)).no_proxy(),
    );
    let mut cached: Option<WorkloadToken> = None;
    while let Some(reply) = requests.recv().await {
        if cached
            .as_ref()
            .is_none_or(|token| Instant::now() >= token.fresh_until)
        {
            cached = match &client {
                Ok(client) => fetch_workload_token(client, &metadata_host).await.ok(),
                Err(_) => None,
            };
        }
        let token = cached
            .as_ref()
            .map(|token| token.value.clone())
            .ok_or_else(unavailable);
        let _ = reply.send(token);
    }
}

fn unavailable() -> LlmError {
    LlmError::Request(
        "vertex found no Google workload credentials; run on Google Cloud with a service account that may call Vertex AI, or set GOOGLE_OAUTH_ACCESS_TOKEN".into(),
    )
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

async fn fetch_workload_token(
    client: &Client,
    metadata_host: &str,
) -> Result<WorkloadToken, LlmError> {
    let response = client
        .get(format!("http://{metadata_host}/{TOKEN_PATH}"))
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|_| unavailable())?;
    if !response.status().is_success() {
        return Err(unavailable());
    }
    let token: TokenResponse = response.json().await.map_err(|_| unavailable())?;
    if token.access_token.is_empty() {
        return Err(unavailable());
    }
    let lifetime = Duration::from_secs(token.expires_in).saturating_sub(TOKEN_REFRESH_MARGIN);
    Ok(WorkloadToken {
        value: token.access_token,
        fresh_until: Instant::now() + lifetime,
    })
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;
    use crate::providers::ModelApi;
    use crate::tools::web_test_server::TestResponse;
    use crate::tools::web_test_server::TestServer;

    const ENDPOINT: &str =
        "https://aiplatform.googleapis.com/v1/projects/example-project/locations/global/endpoints/openapi";

    fn config(base_url: Option<&str>, token: Option<&str>) -> ModelProviderConfig {
        ModelProviderConfig {
            api: ModelApi::Vertex,
            model: "gemini-4-neon".into(),
            context_window: None,
            api_key: token.map(Into::into),
            base_url: base_url.map(Into::into),
            auth_path: None,
            prompt_cache_key: "test".into(),
            request_timeout: None,
            managed_reference: None,
        }
    }

    #[test]
    fn vertex_endpoint_accepts_only_google_openai_compatible_endpoints() {
        assert!(is_vertex_endpoint(ENDPOINT));
        assert!(is_vertex_endpoint(&format!("{ENDPOINT}/")));
        assert!(is_vertex_endpoint(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/example-project/locations/us-central1/endpoints/openapi"
        ));
        assert!(!is_vertex_endpoint(
            &ENDPOINT.replace("https://", "http://")
        ));
        assert!(!is_vertex_endpoint(&ENDPOINT.replace(
            "aiplatform.googleapis.com",
            "aiplatform.googleapis.com.example"
        )));
        assert!(!is_vertex_endpoint(&ENDPOINT.replace(
            "aiplatform.googleapis.com",
            "evil-aiplatform.googleapis.com.example"
        )));
        assert!(!is_vertex_endpoint(
            &ENDPOINT.replace("https://", "https://user@")
        ));
        assert!(!is_vertex_endpoint(&format!("{ENDPOINT}/chat/completions")));
        assert!(!is_vertex_endpoint(&format!("{ENDPOINT}?alt=sse")));
    }

    #[test]
    fn vertex_model_names_the_google_publisher() {
        assert_eq!(
            vertex_model("gemini-4-neon").expect("model"),
            "google/gemini-4-neon"
        );
        assert_eq!(
            vertex_model("google/gemini-4-neon").expect("model"),
            "google/gemini-4-neon"
        );
        assert_matches!(vertex_model("meta/llama-4"), Err(LlmError::Request(error)) if error.contains("Gemini models only"));
    }

    #[tokio::test]
    async fn vertex_connection_uses_a_supplied_token_at_the_listed_endpoint() {
        assert_eq!(
            VertexTokens::default()
                .connection(&config(Some(ENDPOINT), Some("synthetic-token")))
                .await
                .expect("connection"),
            (ENDPOINT.to_string(), "synthetic-token".to_string())
        );
    }

    #[tokio::test]
    async fn vertex_connection_refuses_to_send_credentials_off_google() {
        assert_matches!(
            VertexTokens::default()
                .connection(&config(Some("https://proxy.example/v1"), Some("synthetic-token")))
                .await,
            Err(LlmError::Request(error)) if error.contains("only to https://aiplatform.googleapis.com")
        );
    }

    fn token_response(token: &str, expires_in: u64) -> TestResponse {
        TestResponse::json(
            200,
            json!({"access_token": token, "expires_in": expires_in, "token_type": "Bearer"}),
        )
    }

    fn metadata_host(server: &TestServer) -> String {
        server.base_url.trim_start_matches("http://").to_string()
    }

    #[tokio::test]
    async fn workload_token_is_fetched_once_while_fresh() {
        let mut server = TestServer::start(vec![token_response("synthetic-first", 3599)]).await;
        let tokens = VertexTokens::default();
        let shared = tokens.clone();

        let (first, second) = tokio::join!(
            tokens.workload_token(metadata_host(&server)),
            shared.workload_token(metadata_host(&server)),
        );

        assert_eq!(first.expect("token"), "synthetic-first");
        assert_eq!(second.expect("token"), "synthetic-first");
        assert_eq!(server.request().await.path, format!("/{TOKEN_PATH}"));
    }

    #[tokio::test]
    async fn workload_token_is_fetched_again_inside_the_refresh_margin() {
        let server = TestServer::start(vec![
            token_response("synthetic-expiring", 60),
            token_response("synthetic-renewed", 3599),
        ])
        .await;
        let tokens = VertexTokens::default();

        let first = tokens.workload_token(metadata_host(&server)).await;
        let second = tokens.workload_token(metadata_host(&server)).await;

        assert_eq!(first.expect("token"), "synthetic-expiring");
        assert_eq!(second.expect("token"), "synthetic-renewed");
    }

    #[tokio::test]
    async fn workload_token_recovers_after_a_failed_fetch() {
        let server = TestServer::start(vec![
            TestResponse::json(404, json!({})),
            token_response("synthetic-recovered", 3599),
        ])
        .await;
        let tokens = VertexTokens::default();

        assert_matches!(
            tokens.workload_token(metadata_host(&server)).await,
            Err(LlmError::Request(error)) if error.contains("GOOGLE_OAUTH_ACCESS_TOKEN")
        );
        assert_eq!(
            tokens
                .workload_token(metadata_host(&server))
                .await
                .expect("token"),
            "synthetic-recovered"
        );
    }
}
