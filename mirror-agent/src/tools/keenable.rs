//! Typed Keenable search and fetch client, usable without an API key.

use std::time::Duration;

use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

use super::FirecrawlClient;
use super::Page;
use super::SearchResult;

/// A small client shared by Keenable-backed tools.
#[derive(Clone)]
pub struct KeenableClient {
    client: Client,
    api_key: Option<String>,
    base_url: String,
    retry_delays: Vec<Duration>,
}

impl std::fmt::Debug for KeenableClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KeenableClient")
            .field("base_url", &self.base_url)
            .field("retry_delays", &self.retry_delays)
            .finish_non_exhaustive()
    }
}

/// A Keenable request failure safe to return to the model.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum KeenableError {
    #[error("Could not reach Keenable.")]
    Unreachable,
    #[error("{0}")]
    Response(String),
    #[error("Keenable client initialization failed: {0}")]
    Request(String),
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    max_results: usize,
}

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<SearchItem>,
}

#[derive(Deserialize)]
struct SearchItem {
    url: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    snippet: Option<String>,
}

#[derive(Deserialize)]
struct FetchResponse {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: Option<String>,
}

const DEFAULT_BASE_URL: &str = "https://api.keenable.ai/v1";
const APP_TITLE: &str = "Mirror";
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const FETCH_TIMEOUT: Duration = Duration::from_secs(90);
const RETRYABLE_STATUSES: &[u16] = &[408, 429, 500, 502, 503, 504];

impl KeenableClient {
    /// Create a client using the optional `KEENABLE_API_KEY` and production endpoints.
    pub fn from_env() -> Result<Self, KeenableError> {
        Self::new(
            std::env::var("KEENABLE_API_KEY").ok(),
            DEFAULT_BASE_URL,
            vec![Duration::from_secs(1), Duration::from_secs(2)],
        )
    }

    fn new(
        api_key: Option<String>,
        base_url: impl Into<String>,
        retry_delays: Vec<Duration>,
    ) -> Result<Self, KeenableError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| KeenableError::Request(error.to_string()))?;
        Ok(Self {
            client,
            api_key: api_key
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty()),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            retry_delays,
        })
    }

    /// Create an internal client with deterministic test configuration.
    #[cfg(test)]
    pub(crate) fn for_test(
        api_key: Option<&str>,
        base_url: impl Into<String>,
        retry_delays: Vec<Duration>,
    ) -> Self {
        Self::new(api_key.map(str::to_string), base_url, retry_delays).expect("test HTTP client")
    }

    /// Search for normalized results consumed by `WebSearchTool`.
    pub(crate) async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, KeenableError> {
        let request = self
            .client
            .post(self.endpoint("search"))
            .json(&SearchRequest {
                query,
                max_results: limit,
            })
            .timeout(SEARCH_TIMEOUT);
        let body = parse::<SearchResponse>(self.send(request).await?, "search").await?;
        Ok(body
            .results
            .into_iter()
            .take(limit)
            .map(|result| SearchResult {
                title: result.title,
                link: result.url,
                snippet: result.snippet.or(result.description),
            })
            .collect())
    }

    /// Fetch one normalized page consumed by `WebBrowseTool`.
    pub(crate) async fn scrape(&self, url: &str) -> Result<Page, KeenableError> {
        let endpoint = reqwest::Url::parse_with_params(
            &self.endpoint("fetch"),
            [("url", url), ("live", "true")],
        )
        .map_err(|error| KeenableError::Request(error.to_string()))?;
        let request = self.client.get(endpoint).timeout(FETCH_TIMEOUT);
        let body = parse::<FetchResponse>(self.send(request).await?, "fetch").await?;
        Ok(Page {
            url: body.url.unwrap_or_else(|| url.to_string()),
            title: body.title,
            text: body.content.unwrap_or_default(),
        })
    }

    /// Keyless requests use the public endpoints, which require an app title.
    fn endpoint(&self, operation: &str) -> String {
        match self.api_key {
            Some(_) => format!("{}/{operation}", self.base_url),
            None => format!("{}/{operation}/public", self.base_url),
        }
    }

    async fn send(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, KeenableError> {
        let request = match &self.api_key {
            Some(api_key) => request.bearer_auth(api_key),
            None => request.header("X-Keenable-Title", APP_TITLE),
        };
        let mut retry_delays = self.retry_delays.iter();
        loop {
            let attempt = request
                .try_clone()
                .ok_or_else(|| KeenableError::Request("request is not retryable".to_string()))?;
            let error = match attempt.send().await {
                Ok(response) if !RETRYABLE_STATUSES.contains(&response.status().as_u16()) => {
                    return Ok(response);
                }
                Ok(response) => KeenableError::Response(format!(
                    "Keenable returned HTTP {}.",
                    response.status().as_u16()
                )),
                Err(_) => KeenableError::Unreachable,
            };
            let Some(delay) = retry_delays.next() else {
                return Err(error);
            };
            tokio::time::sleep(*delay).await;
        }
    }
}

async fn parse<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
    operation: &str,
) -> Result<T, KeenableError> {
    let status = response.status();
    if matches!(status.as_u16(), 401..=403) {
        return Err(KeenableError::Response(
            "Keenable rejected KEENABLE_API_KEY.".to_string(),
        ));
    }
    if !status.is_success() {
        return Err(KeenableError::Response(format!(
            "Keenable {operation} returned HTTP {}.",
            status.as_u16()
        )));
    }
    response.json().await.map_err(|_| {
        KeenableError::Response(format!(
            "Keenable {operation} returned an invalid response."
        ))
    })
}

/// The provider behind the web tools.
#[derive(Debug, Clone)]
pub enum WebClient {
    Firecrawl(FirecrawlClient),
    Keenable(KeenableClient),
}

impl WebClient {
    /// Use Firecrawl when `FIRECRAWL_API_KEY` is set, otherwise keyless Keenable.
    pub fn from_env() -> Result<Self, String> {
        let firecrawl = FirecrawlClient::from_env().map_err(|error| error.to_string())?;
        if firecrawl.is_configured() {
            return Ok(Self::Firecrawl(firecrawl));
        }
        KeenableClient::from_env()
            .map(Self::Keenable)
            .map_err(|error| error.to_string())
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Firecrawl(_) => "firecrawl",
            Self::Keenable(_) => "keenable",
        }
    }

    pub(crate) async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, String> {
        match self {
            Self::Firecrawl(client) => client.search(query, limit).await.map_err(|e| e.to_string()),
            Self::Keenable(client) => client.search(query, limit).await.map_err(|e| e.to_string()),
        }
    }

    pub(crate) async fn scrape(&self, url: &str) -> Result<Page, String> {
        match self {
            Self::Firecrawl(client) => client.scrape(url).await.map_err(|e| e.to_string()),
            Self::Keenable(client) => client.scrape(url).await.map_err(|e| e.to_string()),
        }
    }
}

impl From<FirecrawlClient> for WebClient {
    fn from(client: FirecrawlClient) -> Self {
        Self::Firecrawl(client)
    }
}

impl From<KeenableClient> for WebClient {
    fn from(client: KeenableClient) -> Self {
        Self::Keenable(client)
    }
}

#[cfg(test)]
#[path = "keenable_test.rs"]
mod tests;
