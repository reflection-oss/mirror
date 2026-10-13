//! Typed Firecrawl search and scrape client.

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;
use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

/// Authorization checked before each Firecrawl request, including retries.
pub trait FirecrawlRequestGuard: Send + Sync {
    /// Reject dispatch when execution is no longer authorized.
    fn verify(&self) -> BoxFuture<'_, Result<(), FirecrawlError>>;
}

/// A small client shared by Firecrawl-backed tools.
#[derive(Clone)]
pub struct FirecrawlClient {
    client: Client,
    api_key: Option<String>,
    base_url: String,
    retry_delays: Vec<Duration>,
    request_guard: Option<Arc<dyn FirecrawlRequestGuard>>,
}

impl std::fmt::Debug for FirecrawlClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FirecrawlClient")
            .field("base_url", &self.base_url)
            .field("retry_delays", &self.retry_delays)
            .finish_non_exhaustive()
    }
}

/// One internal search result shared by the Firecrawl-backed tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SearchResult {
    pub title: Option<String>,
    pub link: String,
    pub snippet: Option<String>,
}

/// One internal scraped page shared by the Firecrawl-backed tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Page {
    pub url: String,
    pub title: Option<String>,
    pub text: String,
}

/// A Firecrawl request failure safe to return to the model.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum FirecrawlError {
    #[error("FIRECRAWL_API_KEY is not set. Create a Firecrawl API key at https://firecrawl.dev/ and export it.")]
    MissingApiKey,
    #[error("Could not reach Firecrawl.")]
    Unreachable,
    #[error("{0}")]
    Response(String),
    #[error("Firecrawl client initialization failed: {0}")]
    Request(String),
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    limit: usize,
    sources: [&'static str; 1],
}

#[derive(Deserialize)]
struct SearchResponse {
    success: bool,
    data: Option<SearchData>,
}

#[derive(Deserialize)]
struct SearchData {
    #[serde(default)]
    web: Vec<SearchItem>,
}

#[derive(Deserialize)]
struct SearchItem {
    url: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScrapeRequest<'a> {
    url: &'a str,
    formats: [&'static str; 1],
    only_main_content: bool,
}

#[derive(Deserialize)]
struct ScrapeResponse {
    success: bool,
    data: Option<ScrapeData>,
}

#[derive(Deserialize)]
struct ScrapeData {
    #[serde(default)]
    markdown: Option<String>,
    #[serde(default)]
    metadata: ScrapeMetadata,
}

#[derive(Default, Deserialize)]
struct ScrapeMetadata {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default, rename = "sourceURL")]
    source_url: Option<String>,
}

const DEFAULT_BASE_URL: &str = "https://api.firecrawl.dev/v2";
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const SCRAPE_TIMEOUT: Duration = Duration::from_secs(90);
const RETRYABLE_STATUSES: &[u16] = &[408, 429, 500, 502, 503, 504];

impl FirecrawlClient {
    /// Create a client using `FIRECRAWL_API_KEY` and production endpoints.
    pub fn from_env() -> Result<Self, FirecrawlError> {
        Self::new(
            std::env::var("FIRECRAWL_API_KEY").ok(),
            DEFAULT_BASE_URL,
            vec![Duration::from_secs(1), Duration::from_secs(2)],
        )
    }

    /// Whether requests have a non-blank API key configured.
    pub fn is_configured(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
    }

    fn new(
        api_key: Option<String>,
        base_url: impl Into<String>,
        retry_delays: Vec<Duration>,
    ) -> Result<Self, FirecrawlError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| FirecrawlError::Request(error.to_string()))?;
        Ok(Self {
            client,
            api_key,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            retry_delays,
            request_guard: None,
        })
    }

    /// Require authorization before every search or scrape HTTP request.
    #[must_use]
    pub fn with_request_guard(mut self, guard: Arc<dyn FirecrawlRequestGuard>) -> Self {
        self.request_guard = Some(guard);
        self
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
    ) -> Result<Vec<SearchResult>, FirecrawlError> {
        let response = self
            .post(
                "search",
                &SearchRequest {
                    query,
                    limit,
                    sources: ["web"],
                },
                SEARCH_TIMEOUT,
            )
            .await?;
        let body = parse::<SearchResponse>(response, "search").await?;
        if !body.success {
            return Err(FirecrawlError::Response(
                "Firecrawl search returned an unsuccessful response.".to_string(),
            ));
        }
        Ok(body
            .data
            .map(|data| data.web)
            .unwrap_or_default()
            .into_iter()
            .take(limit)
            .map(|result| SearchResult {
                title: result.title,
                link: result.url,
                snippet: result.description,
            })
            .collect())
    }

    /// Scrape one normalized page consumed by `WebBrowseTool`.
    pub(crate) async fn scrape(&self, url: &str) -> Result<Page, FirecrawlError> {
        let response = self
            .post(
                "scrape",
                &ScrapeRequest {
                    url,
                    formats: ["markdown"],
                    only_main_content: true,
                },
                SCRAPE_TIMEOUT,
            )
            .await?;
        let body = parse::<ScrapeResponse>(response, "scrape").await?;
        let data = body.data.filter(|_| body.success).ok_or_else(|| {
            FirecrawlError::Response("Firecrawl could not fetch the webpage.".to_string())
        })?;
        Ok(Page {
            url: data
                .metadata
                .url
                .or(data.metadata.source_url)
                .unwrap_or_else(|| url.to_string()),
            title: data.metadata.title,
            text: data.markdown.unwrap_or_default(),
        })
    }

    async fn post(
        &self,
        operation: &str,
        payload: &impl Serialize,
        timeout: Duration,
    ) -> Result<reqwest::Response, FirecrawlError> {
        let api_key = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or(FirecrawlError::MissingApiKey)?;
        let endpoint = format!("{}/{operation}", self.base_url);
        let mut retry_delays = self.retry_delays.iter();
        loop {
            if let Some(guard) = &self.request_guard {
                guard.verify().await?;
            }
            let result = self
                .client
                .post(&endpoint)
                .bearer_auth(api_key)
                .json(payload)
                .timeout(timeout)
                .send()
                .await;
            let error = match result {
                Ok(response) if !RETRYABLE_STATUSES.contains(&response.status().as_u16()) => {
                    return Ok(response);
                }
                Ok(response) => FirecrawlError::Response(format!(
                    "Firecrawl returned HTTP {}.",
                    response.status().as_u16()
                )),
                Err(_) => FirecrawlError::Unreachable,
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
) -> Result<T, FirecrawlError> {
    let status = response.status();
    if matches!(status.as_u16(), 401 | 402) {
        return Err(FirecrawlError::Response(
            "Firecrawl rejected FIRECRAWL_API_KEY.".to_string(),
        ));
    }
    if !status.is_success() {
        return Err(FirecrawlError::Response(format!(
            "Firecrawl {operation} returned HTTP {}.",
            status.as_u16()
        )));
    }
    response.json().await.map_err(|_| {
        FirecrawlError::Response(format!(
            "Firecrawl {operation} returned an invalid response."
        ))
    })
}

#[cfg(test)]
#[path = "firecrawl_test.rs"]
mod tests;
