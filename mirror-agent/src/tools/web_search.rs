//! Public web search through Firecrawl or Keenable.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::SearchResult;
use super::Tool;
use super::ToolError;
use super::WebClient;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// Search public webpages through the configured web provider.
#[derive(Debug, Clone)]
pub struct WebSearchTool {
    client: WebClient,
}

/// Arguments accepted by web search.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchArgs {
    pub query: String,
}

const NUM_RESULTS: usize = 10;
const MAX_OUTPUT_CHARS: usize = 20_000;
const MAX_TITLE_CHARS: usize = 500;
const MAX_SNIPPET_CHARS: usize = 1_500;
const MAX_URL_CHARS: usize = 2_048;

impl WebSearchTool {
    /// Create a web search tool backed by a shared web client.
    pub fn new(client: impl Into<WebClient>) -> Self {
        Self {
            client: client.into(),
        }
    }
}

impl Tool for WebSearchTool {
    type Args = WebSearchArgs;

    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search for content on the internet. Results include the website title, address (URL), and description."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([(
                "query".to_string(),
                serde_json::json!({"type": "string", "description": "The search query."}),
            )]),
            &["query"],
        )
    }

    fn supports_remote(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: Self::Args,
        context: &ExecutionContext,
        _remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        // This tool uses a narrow host service, not the workspace or remote transport.
        self.run(args, context).await
    }

    async fn run(
        &self,
        args: WebSearchArgs,
        _context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let results = match self.client.search(&args.query, NUM_RESULTS).await {
            Ok(results) => results,
            Err(error) => return Ok(ToolOutput::error(error)),
        };
        let (content, count) = serialize_results(results)?;
        Ok(ToolOutput::success(content)
            .with_display(format!("{}: {count} result(s)", self.client.name())))
    }
}

fn serialize_results(results: Vec<SearchResult>) -> Result<(String, usize), ToolError> {
    let mut normalized = results
        .into_iter()
        .filter_map(|result| {
            let link = result_link(&result.link)?;
            Some(serde_json::json!({
                "title": bounded_text(result.title.as_deref(), MAX_TITLE_CHARS),
                "link": link,
                "snippet": bounded_text(result.snippet.as_deref(), MAX_SNIPPET_CHARS),
            }))
        })
        .collect::<Vec<_>>();
    loop {
        let content = serde_json::to_string_pretty(&serde_json::json!({
            "results": normalized,
        }))
        .map_err(|error| ToolError::Execution(error.to_string()))?;
        if content.chars().count() <= MAX_OUTPUT_CHARS || normalized.is_empty() {
            let count = normalized.len();
            return Ok((content, count));
        }
        normalized.pop();
    }
}

fn result_link(link: &str) -> Option<String> {
    let link = link.trim();
    if link.is_empty() || link.chars().count() > MAX_URL_CHARS {
        return None;
    }
    let parsed = reqwest::Url::parse(link).ok()?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return None;
    }
    Some(link.to_string())
}

fn bounded_text(text: Option<&str>, limit: usize) -> Option<String> {
    let text = text?;
    if text.is_empty() {
        return None;
    }
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return None;
    }
    if normalized.chars().count() <= limit {
        return Some(normalized);
    }
    let mut bounded = normalized.chars().take(limit - 1).collect::<String>();
    bounded = bounded.trim_end().to_string();
    bounded.push('…');
    Some(bounded)
}

#[cfg(test)]
#[path = "web_search_test.rs"]
mod tests;
