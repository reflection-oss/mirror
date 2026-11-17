//! Tavily Search and Extract transport; no SDK, synthesis, or provider fallback.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

use super::{Page, SearchResult, WebError, WebProvider};
use crate::with_fallback_roots;

/// Tavily client shared by search and browsing tools.
#[derive(Clone)]
pub struct TavilyClient {
    client: Client,
    api_key: Option<String>,
    base_url: String,
}

/// A Tavily failure safe to return to the model, without response bodies.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TavilyError {
    #[error("TAVILY_API_KEY is not set. Create a Tavily API key at https://app.tavily.com/ and export it.")]
    MissingApiKey,
    #[error("Could not reach Tavily.")]
    Unreachable,
    #[error("Tavily client initialization failed.")]
    Initialization,
    #[error("{0}")]
    Response(String),
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    max_results: usize,
    search_depth: &'static str,
    topic: &'static str,
    auto_parameters: bool,
    include_answer: bool,
    include_raw_content: bool,
    include_images: bool,
}

#[derive(Deserialize)]
struct SearchResponse {
    results: Vec<SearchItem>,
}

#[derive(Deserialize)]
struct SearchItem {
    title: Option<String>,
    url: String,
    content: Option<String>,
}

#[derive(Serialize)]
struct ExtractRequest<'a> {
    urls: [&'a str; 1],
    format: &'static str,
    extract_depth: &'static str,
    timeout: u64,
}

#[derive(Deserialize)]
struct ExtractResponse {
    results: Vec<ExtractPage>,
    // Provider error bodies are not retained or returned to the model.
    failed_results: Vec<serde::de::IgnoredAny>,
}

#[derive(Deserialize)]
struct ExtractPage {
    url: String,
    raw_content: String,
}

impl std::fmt::Debug for TavilyClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TavilyClient")
            .finish_non_exhaustive()
    }
}

impl TavilyClient {
    /// Create a client with an explicit credential and the production endpoint.
    pub fn from_api_key(api_key: Option<String>) -> Result<Self, TavilyError> {
        Self::new(api_key, "https://api.tavily.com")
    }

    fn new(api_key: Option<String>, base_url: &str) -> Result<Self, TavilyError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = with_fallback_roots(Client::builder())
            .connect_timeout(Duration::from_secs(10))
            // Never forward credentials to redirects or repeat a billable request implicitly.
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| TavilyError::Initialization)?;
        Ok(Self {
            client,
            api_key,
            base_url: base_url.to_string(),
        })
    }

    async fn post<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &impl Serialize,
        timeout: Duration,
    ) -> Result<T, TavilyError> {
        let key = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or(TavilyError::MissingApiKey)?;
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .bearer_auth(key)
            .json(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|_| TavilyError::Unreachable)?;
        let status = response.status();
        if !status.is_success() {
            return Err(TavilyError::Response(match status.as_u16() {
                401 | 403 => "Tavily rejected TAVILY_API_KEY.".into(),
                429 => "Tavily rate limit exceeded. Try again later.".into(),
                432 | 433 => "Tavily credit limit exceeded. Check your Tavily plan.".into(),
                code => format!("Tavily {path} returned HTTP {code}."),
            }));
        }
        response.json().await.map_err(|_| {
            TavilyError::Response(format!("Tavily {path} returned an invalid response."))
        })
    }
}

#[async_trait]
impl WebProvider for TavilyClient {
    fn name(&self) -> &'static str {
        "tavily"
    }

    fn is_configured(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, WebError> {
        let response: SearchResponse = self
            .post(
                "/search",
                &SearchRequest {
                    query,
                    max_results: limit,
                    search_depth: "basic",
                    topic: "general",
                    auto_parameters: false,
                    include_answer: false,
                    include_raw_content: false,
                    include_images: false,
                },
                Duration::from_secs(30),
            )
            .await?;
        Ok(response
            .results
            .into_iter()
            .take(limit)
            .map(|item| SearchResult {
                title: item.title,
                link: item.url,
                snippet: item.content,
            })
            .collect())
    }

    async fn contents(&self, url: &str) -> Result<Page, WebError> {
        // No query/chunk limit: grep must see all extracted text, not ranked excerpts.
        let mut response: ExtractResponse = self
            .post(
                "/extract",
                &ExtractRequest {
                    urls: [url],
                    format: "markdown",
                    extract_depth: "basic",
                    timeout: 60,
                },
                Duration::from_secs(75),
            )
            .await?;
        if response.results.len() > 1 {
            return Err(TavilyError::Response("Malformed Tavily Extract response.".into()).into());
        }
        let page = response
            .results
            .pop()
            .filter(|page| {
                response.failed_results.is_empty()
                    && !page
                        .raw_content
                        .trim_matches(|character: char| {
                            character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                        })
                        .is_empty()
            })
            .ok_or_else(|| {
                TavilyError::Response(
                    "This website could not be accessed. Try a different source.".into(),
                )
            })?;
        Ok(Page {
            url: page.url,
            // Extract does not promise a title or a redirect destination.
            title: None,
            text: page.raw_content,
        })
    }
}
