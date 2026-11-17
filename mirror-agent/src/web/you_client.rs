//! You.com search and full-page Markdown transport.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{Page, SearchResult, WebError, WebProvider};
use crate::with_fallback_roots;

/// You.com client shared by search and browsing tools.
#[derive(Clone)]
pub struct YouClient {
    client: Client,
    api_key: Option<String>,
    base_url: String,
    request_guard: Option<Arc<dyn YouRequestGuard>>,
}

/// Authorization checked before each You.com request, including retries.
pub trait YouRequestGuard: Send + Sync {
    /// Reject dispatch when execution is no longer authorized.
    fn verify(&self) -> BoxFuture<'_, Result<(), YouError>>;
}

/// A provider failure safe to return to the model, without response bodies.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum YouError {
    #[error("YOU_API_KEY is not set. Create a You.com API key at https://you.com/ and export it.")]
    MissingApiKey,
    #[error("Could not reach You.com.")]
    Unreachable,
    #[error("{0}")]
    Response(String),
    #[error("You.com client initialization failed: {0}")]
    Request(String),
}

#[derive(Serialize)]
struct ContentsRequest<'a> {
    urls: [&'a str; 1],
    formats: [&'static str; 1],
    crawl_timeout: u64,
}

#[derive(Deserialize)]
struct ContentsPage {
    url: Option<String>,
    title: Option<String>,
    markdown: Option<String>,
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    count: usize,
    offset: usize,
    snippet_engine: &'static str,
    snippet_engine_target_chars: usize,
}

#[derive(Deserialize)]
struct SearchResponse {
    results: Option<SearchResults>,
}

#[derive(Deserialize)]
struct SearchResults {
    #[serde(default)]
    web: Vec<SearchItem>,
}

#[derive(Deserialize)]
struct SearchItem {
    url: String,
    title: String,
    description: Option<String>,
    snippets: Option<Vec<String>>,
}

const BASE_URL: &str = "https://ydc-index.io/v1";
// Leave time for the API envelope beyond its 60-second crawl window.
const CONTENTS_TIMEOUT: Duration = Duration::from_secs(75);
const RETRYABLE_STATUSES: &[u16] = &[408, 429, 500, 502, 503, 504];

impl std::fmt::Debug for YouClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("YouClient").finish_non_exhaustive()
    }
}

impl YouClient {
    /// Create a client with an explicit credential and the production endpoint.
    pub fn from_api_key(api_key: Option<String>) -> Result<Self, YouError> {
        Self::new(api_key, BASE_URL)
    }

    /// Whether requests have a non-blank API key configured.
    pub fn is_configured(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
    }

    fn new(api_key: Option<String>, base_url: &str) -> Result<Self, YouError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = with_fallback_roots(Client::builder())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            // Custom authentication headers must never follow provider redirects.
            .redirect(reqwest::redirect::Policy::none())
            // Retry loops must recheck authorization before every dispatch.
            .retry(reqwest::retry::never())
            .build()
            .map_err(|error| YouError::Request(error.to_string()))?;
        Ok(Self {
            client,
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            request_guard: None,
        })
    }

    /// Require authorization before every HTTP request, including retries.
    #[must_use]
    pub fn with_request_guard(mut self, guard: Arc<dyn YouRequestGuard>) -> Self {
        self.request_guard = Some(guard);
        self
    }

    /// Construct a loopback client without environment configuration.
    #[cfg(test)]
    pub(crate) fn for_test(api_key: Option<&str>, base_url: &str) -> Self {
        Self::new(api_key.map(str::to_string), &format!("{base_url}/v1")).expect("test HTTP client")
    }

    /// Fetch normalized results for the model-facing search tool.
    pub(crate) async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, YouError> {
        let response = self.fetch(query, limit).await?;
        let body = response.json::<SearchResponse>().await.map_err(|_| {
            YouError::Response("You.com search returned an invalid response.".to_string())
        })?;
        Ok(body
            .results
            .map(|results| results.web)
            .unwrap_or_default()
            .into_iter()
            .take(limit)
            .map(|result| SearchResult {
                title: Some(result.title),
                link: result.url,
                snippet: result
                    .snippets
                    .unwrap_or_default()
                    .into_iter()
                    .find(|snippet| {
                        // Python's strip also treats U+001C..U+001F as whitespace.
                        let cleaned = snippet.trim_matches(|character: char| {
                            character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                        });
                        !cleaned.is_empty() && !cleaned.eq_ignore_ascii_case("null")
                    })
                    .or_else(|| {
                        result
                            .description
                            .filter(|description| !description.is_empty())
                    }),
            })
            .collect())
    }

    /// Fetch a complete Markdown page; unavailable pages must not be cached.
    pub(crate) async fn contents(&self, url: &str) -> Result<Page, YouError> {
        let mut pages = self.fetch_contents(url).await?;
        if pages.len() > 1 {
            return Err(YouError::Response(
                "Malformed You.com Contents response".to_string(),
            ));
        }
        let page = pages.pop();
        let unavailable = || {
            YouError::Response(
                "This website could not be accessed. Try a different source.".to_string(),
            )
        };
        let page = page.ok_or_else(unavailable)?;
        let text = page
            .markdown
            .filter(|text| {
                !text
                    .trim_matches(|character: char| {
                        character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                    })
                    .is_empty()
            })
            .ok_or_else(unavailable)?;
        Ok(Page {
            url: page
                .url
                .filter(|url| !url.is_empty())
                .unwrap_or_else(|| url.to_string()),
            title: page.title,
            text,
        })
    }

    async fn fetch_contents(&self, url: &str) -> Result<Vec<ContentsPage>, YouError> {
        let api_key = self.api_key()?;
        let mut attempt = 0;
        loop {
            if let Some(guard) = &self.request_guard {
                guard.verify().await?;
            }
            let response = self
                .client
                .post(format!("{}/contents", self.base_url))
                .header("X-API-Key", api_key)
                .json(&ContentsRequest {
                    urls: [url],
                    formats: ["markdown"],
                    crawl_timeout: 60,
                })
                .timeout(CONTENTS_TIMEOUT)
                .send()
                .await;
            let error = match response {
                Ok(response) if response.status().is_success() => match response.bytes().await {
                    Ok(body) => {
                        return serde_json::from_slice(&body).map_err(|_| {
                            YouError::Response("Malformed You.com Contents response".to_string())
                        })
                    }
                    Err(_) => YouError::Unreachable,
                },
                Ok(response) => {
                    let status = response.status().as_u16();
                    let error = contents_status_error(status);
                    if !RETRYABLE_STATUSES.contains(&status) {
                        return Err(error);
                    }
                    error
                }
                Err(_) => YouError::Unreachable,
            };
            if attempt == 2 {
                return Err(error);
            }
            // Contents shares one three-attempt budget across HTTP and transport failures.
            tokio::time::sleep(Duration::from_secs(2)).await;
            attempt += 1;
        }
    }

    fn api_key(&self) -> Result<&str, YouError> {
        self.api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or(YouError::MissingApiKey)
    }

    async fn fetch(&self, query: &str, limit: usize) -> Result<reqwest::Response, YouError> {
        let api_key = self.api_key()?;
        let arguments = SearchRequest {
            query,
            count: limit,
            offset: 0,
            snippet_engine: "rerank_v5",
            snippet_engine_target_chars: 200,
        };
        let mut attempt = 0;
        loop {
            if let Some(guard) = &self.request_guard {
                guard.verify().await?;
            }
            let response = self
                .client
                .get(format!("{}/search", self.base_url))
                .header("X-API-Key", api_key)
                .query(&arguments)
                .send()
                .await;
            let delay = match response {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status().as_u16();
                    let error = status_error(status);
                    if !RETRYABLE_STATUSES.contains(&status) || attempt == 5 {
                        return Err(error);
                    }
                    // Anti-SERP blocks ban the account briefly; short retries cannot help.
                    if matches!(status, 503 | 504) {
                        (15 * 2_u64.pow(attempt)).min(30)
                    } else {
                        (2 * 2_u64.pow(attempt)).clamp(4, 32)
                    }
                }
                Err(_) if attempt < 2 => 2_u64.pow(attempt),
                Err(_) => return Err(YouError::Unreachable),
            };
            tokio::time::sleep(Duration::from_secs(delay)).await;
            attempt += 1;
        }
    }
}

#[async_trait]
impl WebProvider for YouClient {
    fn name(&self) -> &'static str {
        "you"
    }

    fn is_configured(&self) -> bool {
        self.is_configured()
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, WebError> {
        Ok(self.search(query, limit).await?)
    }

    async fn contents(&self, url: &str) -> Result<Page, WebError> {
        Ok(self.contents(url).await?)
    }
}

fn contents_status_error(status: u16) -> YouError {
    YouError::Response(match status {
        400 | 422 => "Invalid URL for You.com Contents. Try a different URL.".to_string(),
        401..=403 => "You.com rejected YOU_API_KEY.".to_string(),
        _ => format!("You.com Contents API returned HTTP {status}"),
    })
}

fn status_error(status: u16) -> YouError {
    YouError::Response(match status {
        401..=403 => "You.com rejected YOU_API_KEY.".to_string(),
        503 => "You.com search blocked by upstream anti-SERP measures (HTTP 503).".to_string(),
        504 => "You.com search upstream timeout during anti-SERP blocking (HTTP 504).".to_string(),
        _ => format!("You.com search returned HTTP {status}."),
    })
}

#[cfg(test)]
#[path = "you_client_test.rs"]
mod tests;
