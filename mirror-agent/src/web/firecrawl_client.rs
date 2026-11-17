//! Firecrawl v2 Search and Scrape transport without synthesis or fallback.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

use super::{Page, SearchResult, WebError, WebProvider};
use crate::with_fallback_roots;

/// Firecrawl client shared by search and browsing tools.
#[derive(Clone)]
pub struct FirecrawlClient {
    client: Client,
    api_key: Option<String>,
    base_url: String,
}

/// A Firecrawl failure safe to return without response bodies or credentials.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FirecrawlError {
    #[error("FIRECRAWL_API_KEY is not set. Create a Firecrawl API key at https://www.firecrawl.dev/ and export it.")]
    MissingApiKey,
    #[error("Could not reach Firecrawl.")]
    Unreachable,
    #[error("Firecrawl client initialization failed.")]
    Initialization,
    #[error("{0}")]
    Response(String),
}

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    limit: usize,
    sources: [&'static str; 1],
    highlights: bool,
    timeout: u64,
}

#[derive(Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
}

#[derive(Deserialize)]
struct SearchResults {
    web: Vec<SearchItem>,
}

#[derive(Deserialize)]
struct SearchItem {
    title: Option<String>,
    url: String,
    description: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScrapeRequest<'a> {
    url: &'a str,
    formats: [&'static str; 1],
    only_main_content: bool,
    only_clean_content: bool,
    skip_tls_verification: bool,
    store_in_cache: bool,
    timeout: u64,
}

#[derive(Deserialize)]
struct ScrapePage {
    markdown: String,
    metadata: Option<ScrapeMetadata>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScrapeMetadata {
    title: Option<String>,
    url: Option<String>,
    status_code: Option<u16>,
    error: Option<serde::de::IgnoredAny>,
}

impl std::fmt::Debug for FirecrawlClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FirecrawlClient")
            .finish_non_exhaustive()
    }
}

impl FirecrawlClient {
    /// Create a client with an explicit credential and the production endpoint.
    pub fn from_api_key(api_key: Option<String>) -> Result<Self, FirecrawlError> {
        Self::new(api_key, "https://api.firecrawl.dev")
    }

    fn new(api_key: Option<String>, base_url: &str) -> Result<Self, FirecrawlError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = with_fallback_roots(Client::builder())
            .connect_timeout(Duration::from_secs(10))
            // Never forward credentials to redirects or implicitly repeat billable requests.
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| FirecrawlError::Initialization)?;
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
    ) -> Result<T, FirecrawlError> {
        let key = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .ok_or(FirecrawlError::MissingApiKey)?;
        let response = self
            .client
            .post(format!("{}{path}", self.base_url))
            .bearer_auth(key)
            .json(body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|_| FirecrawlError::Unreachable)?;
        let status = response.status();
        if !status.is_success() {
            return Err(FirecrawlError::Response(match status.as_u16() {
                401 => "Firecrawl rejected FIRECRAWL_API_KEY.".into(),
                402 => "Firecrawl credit limit exceeded. Check your Firecrawl plan.".into(),
                403 => "Firecrawl denied access to this request.".into(),
                429 => "Firecrawl rate limit exceeded. Try again later.".into(),
                code => format!("Firecrawl {path} returned HTTP {code}."),
            }));
        }
        let response: ApiResponse<T> = response.json().await.map_err(|_| {
            FirecrawlError::Response(format!("Firecrawl {path} returned an invalid response."))
        })?;
        if !response.success {
            return Err(FirecrawlError::Response(format!(
                "Firecrawl {path} failed."
            )));
        }
        response.data.ok_or_else(|| {
            FirecrawlError::Response(format!("Firecrawl {path} returned an invalid response."))
        })
    }
}

#[async_trait]
impl WebProvider for FirecrawlClient {
    fn name(&self) -> &'static str {
        "firecrawl"
    }

    fn is_configured(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, WebError> {
        let response: SearchResults = self
            .post(
                "/v2/search",
                &SearchRequest {
                    query,
                    limit,
                    sources: ["web"],
                    highlights: false,
                    timeout: 30_000,
                },
                Duration::from_secs(45),
            )
            .await?;
        Ok(response
            .web
            .into_iter()
            .take(limit)
            .map(|item| SearchResult {
                title: item.title,
                link: item.url,
                snippet: item.description,
            })
            .collect())
    }

    async fn contents(&self, url: &str) -> Result<Page, WebError> {
        let page: ScrapePage = self
            .post(
                "/v2/scrape",
                &ScrapeRequest {
                    url,
                    formats: ["markdown"],
                    // Grep needs the full extracted text, not a cleaned or summarized subset.
                    only_main_content: false,
                    only_clean_content: false,
                    skip_tls_verification: false,
                    store_in_cache: false,
                    timeout: 60_000,
                },
                Duration::from_secs(75),
            )
            .await?;
        let metadata = page.metadata.unwrap_or_default();
        if metadata.status_code.is_some_and(|status| status >= 400)
            || metadata.error.is_some()
            || page
                .markdown
                .trim_matches(|character: char| {
                    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
                })
                .is_empty()
        {
            return Err(FirecrawlError::Response(
                "This website could not be accessed. Try a different source.".into(),
            )
            .into());
        }
        Ok(Page {
            url: metadata
                .url
                .filter(|url| !url.is_empty())
                .unwrap_or_else(|| url.to_string()),
            title: metadata.title,
            text: page.markdown,
        })
    }
}
