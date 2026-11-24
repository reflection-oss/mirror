//! Fetch, cache, preview, and search webpages with provider-isolated caches.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use super::object_schema;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::web::{WebError, WebProviderConfig, WebProviderFactory};
use crate::ExecutionContext;

/// Fetch and inspect public webpages through a conversation-local cache.
#[derive(Debug, Clone)]
pub struct WebBrowseTool {
    requests: mpsc::UnboundedSender<BrowseRequest>,
    name: &'static str,
}

/// Arguments accepted by webpage browsing and search.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WebBrowseArgs {
    pub url: String,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub output_mode: BrowseOutputMode,
    #[serde(default)]
    pub ignore_case: bool,
    #[serde(default)]
    pub context_lines: usize,
    #[serde(default)]
    pub multiline: bool,
    #[serde(default = "default_max_lines")]
    pub max_lines: usize,
    #[serde(default)]
    pub offset: usize,
}

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BrowseOutputMode {
    #[default]
    Content,
    Count,
}

#[derive(Debug, Clone)]
struct Page {
    final_url: String,
    title: Option<String>,
    text: String,
    estimated_tokens: usize,
    line_count: usize,
}

// Struct serialization preserves field order and omits absent optional fields.
#[derive(Serialize)]
struct BrowseResponse<'a> {
    url: &'a str,
    final_url: &'a str,
    cached: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_preview: Option<&'a str>,
    estimated_text_tokens: usize,
    line_count: usize,
    truncated: bool,
    hint: &'a str,
}

#[derive(Serialize)]
struct GrepResponse<'a> {
    url: &'a str,
    final_url: &'a str,
    cached: bool,
    pattern: &'a Option<String>,
    output_mode: BrowseOutputMode,
    output: &'a str,
    truncated: bool,
    timed_out: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
}

struct BrowseRequest {
    url: String,
    provider: Option<WebProviderConfig>,
    response: oneshot::Sender<Result<(Page, bool), WebError>>,
}

struct BrowseCache {
    make_provider: Box<WebProviderFactory>,
    requests: mpsc::UnboundedReceiver<BrowseRequest>,
    pages: HashMap<(&'static str, String), Page>,
}

const MAX_TOKENS_PER_PAGE: usize = 8_000;
const MAX_GREP_LINES: usize = 1_000;
const MAX_TITLE_CHARS: usize = 500;
const MAX_URL_CHARS: usize = 2_048;
// Python's Unicode \w excludes combining marks and connector punctuation except _.
static TOKEN_CHUNKS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s\x1c-\x1f]")
        .expect("the token estimate pattern is a valid constant")
});
static LINE_BREAKS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\r\n|[\n\r\x0b\x0c\x1c-\x1e\x{85}\x{2028}\x{2029}]")
        .expect("the splitlines pattern is a valid constant")
});

impl WebBrowseTool {
    /// Resolve adapters per request; clones share the provider-isolated cache.
    pub fn new(make_provider: Box<WebProviderFactory>) -> Self {
        let (requests, receiver) = mpsc::unbounded_channel();
        tokio::spawn(
            BrowseCache {
                make_provider,
                requests: receiver,
                pages: HashMap::new(),
            }
            .run(),
        );
        Self {
            requests,
            name: "web_browse",
        }
    }

    /// Override the model-facing tool name.
    #[must_use]
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    async fn get_page(
        &self,
        url: String,
        provider: Option<WebProviderConfig>,
    ) -> Result<(Page, bool), ToolError> {
        let (response, receiver) = oneshot::channel();
        self.requests
            .send(BrowseRequest {
                url,
                provider,
                response,
            })
            .map_err(|_| ToolError::Execution("webpage cache closed".to_string()))?;
        receiver
            .await
            .map_err(|_| ToolError::Execution("webpage cache closed".to_string()))?
            .map_err(|error| ToolError::Execution(error.to_string()))
    }
}

impl Tool for WebBrowseTool {
    type Args = WebBrowseArgs;

    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "Visit a webpage and inspect its content. Pass only a URL to fetch and cache an approximately 8000-token preview; include a regex pattern to search the full cached page content."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "url".to_string(),
                    serde_json::json!({
                        "type": "string",
                        "description": "The HTTP(S) webpage URL, usually obtained from a web search. The page is cached by URL for later searches on this tool instance."
                    }),
                ),
                (
                    "pattern".to_string(),
                    serde_json::json!({
                        "type": ["string", "null"],
                        "minLength": 1,
                        "default": null,
                        "description": "Optional regular expression. If omitted, fetch and preview the page; if provided, search the full cached page text."
                    }),
                ),
                (
                    "output_mode".to_string(),
                    serde_json::json!({
                        "enum": ["content", "count"],
                        "default": "content",
                        "description": "Search output mode: content returns matching lines; count returns only the number of matching lines."
                    }),
                ),
                (
                    "ignore_case".to_string(),
                    serde_json::json!({
                        "type": "boolean",
                        "default": false,
                        "description": "Search without matching letter case."
                    }),
                ),
                (
                    "context_lines".to_string(),
                    serde_json::json!({
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 20,
                        "default": 0,
                        "description": "Number of lines before and after each matching line."
                    }),
                ),
                (
                    "multiline".to_string(),
                    serde_json::json!({
                        "type": "boolean",
                        "default": false,
                        "description": "Allow the regular expression to match across line breaks."
                    }),
                ),
                (
                    "max_lines".to_string(),
                    serde_json::json!({
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_GREP_LINES,
                        "default": 250,
                        "description": "Maximum search output lines to return after offset. Pass 0 for unlimited lines up to output caps."
                    }),
                ),
                (
                    "offset".to_string(),
                    serde_json::json!({
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Skip first N search output lines before applying max_lines."
                    }),
                ),
            ]),
            &["url"],
        )
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }

    async fn run(
        &self,
        args: WebBrowseArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        validate_args(&args)?;
        let Some(url) = validated_url(&args.url) else {
            return Ok(ToolOutput::error("web_browse requires an HTTP(S) URL."));
        };
        let (page, cached) = match self.get_page(url, context.web_provider()).await {
            Ok(page) => page,
            Err(ToolError::Execution(message)) => {
                return Ok(ToolOutput::error(message));
            }
            Err(error) => return Err(error),
        };
        let display = page.final_url.clone();
        if args.pattern.is_none() {
            return Ok(
                ToolOutput::success(preview_response(&args.url, &page, cached)?)
                    .with_display(display),
            );
        }
        let (content, error) = grep_response(&args, &page, cached).await?;
        Ok(ToolOutput {
            content,
            display: Some(display),
            success: !error,
            touched: Vec::new(),
            exit_code: None,
            timed_out: false,
            duration: None,
        })
    }
}

fn default_max_lines() -> usize {
    250
}

fn validate_args(args: &WebBrowseArgs) -> Result<(), ToolError> {
    if args.pattern.as_ref().is_some_and(String::is_empty) {
        return Err(ToolError::InvalidArguments(
            "pattern must not be empty".to_string(),
        ));
    }
    if args.context_lines > 20 || args.max_lines > MAX_GREP_LINES {
        return Err(ToolError::InvalidArguments(
            "context_lines cannot exceed 20 and max_lines cannot exceed 1000".to_string(),
        ));
    }
    Ok(())
}

impl BrowseCache {
    async fn run(mut self) {
        while let Some(request) = self.requests.recv().await {
            let BrowseRequest {
                url,
                provider,
                mut response,
            } = request;
            let result = tokio::select! {
                biased;
                _ = response.closed() => continue,
                result = self.fetch(&url, provider) => result,
            };
            let _ = response.send(result);
        }
    }

    async fn fetch(
        &mut self,
        requested_url: &str,
        provider: Option<WebProviderConfig>,
    ) -> Result<(Page, bool), WebError> {
        let client = (self.make_provider)(provider)?;
        let key = (client.name(), requested_url.to_string());
        if let Some(page) = self.pages.get(&key) {
            return Ok((page.clone(), true));
        }
        let result = client.contents(requested_url).await?;
        let text = normalize_text(&result.text);
        let final_url = validated_url(&result.url).unwrap_or_else(|| requested_url.to_string());
        let page = Page {
            final_url: final_url.clone(),
            title: result
                .title
                .map(|title| title.chars().take(MAX_TITLE_CHARS).collect()),
            estimated_tokens: estimate_tokens(&text),
            line_count: text.lines().count(),
            text,
        };
        self.pages.insert(key, page.clone());
        self.pages.insert((client.name(), final_url), page.clone());
        Ok((page, false))
    }
}

fn preview_response(requested_url: &str, page: &Page, cached: bool) -> Result<String, ToolError> {
    let (preview, truncated) = truncate(&page.text, MAX_TOKENS_PER_PAGE);
    model_json(&BrowseResponse {
        url: requested_url,
        final_url: &page.final_url,
        cached,
        title: page.title.as_deref(),
        text_preview: (!preview.is_empty()).then_some(preview.as_str()),
        estimated_text_tokens: page.estimated_tokens,
        line_count: page.line_count,
        truncated,
        hint: "Full page text is cached. Browse previews and search results are capped around 8000 tokens. Add a regex pattern and optional context_lines to inspect relevant parts of the page.",
    })
}

async fn grep_response(
    args: &WebBrowseArgs,
    page: &Page,
    cached: bool,
) -> Result<(String, bool), ToolError> {
    let result = match super::ripgrep::run_with_input(&grep_arguments(args), &page.text).await {
        Ok(result) => result,
        Err(error) => super::ripgrep::RgResult {
            return_code: 126,
            stdout: String::new(),
            stderr: error.to_string(),
            timed_out: false,
        },
    };
    let mut output = result.stdout;
    if result.timed_out && !output.ends_with('\n') {
        output = output
            .rsplit_once('\n')
            .map(|(complete, _)| complete.to_string())
            .unwrap_or_default();
    }
    let error = if !matches!(result.return_code, 0 | 1) && !result.timed_out {
        let detail = if result.stderr.is_empty() {
            output.as_str()
        } else {
            result.stderr.as_str()
        };
        let (detail, _) = truncate(detail, MAX_TOKENS_PER_PAGE);
        Some(format!(
            "Ripgrep error (exit {}): {detail}",
            result.return_code
        ))
    } else {
        None
    };
    let failed = error.is_some();
    let (formatted, truncated) = if failed {
        (String::new(), false)
    } else {
        format_grep_result(args, &output, result.timed_out)
    };
    Ok((
        model_json(&GrepResponse {
            url: &args.url,
            final_url: &page.final_url,
            cached,
            pattern: &args.pattern,
            output_mode: args.output_mode,
            output: &formatted,
            truncated,
            timed_out: result.timed_out,
            error: error.as_deref(),
        })?,
        failed,
    ))
}

fn grep_arguments(args: &WebBrowseArgs) -> Vec<String> {
    let mut arguments = vec![
        "--color".to_string(),
        "never".to_string(),
        "--no-heading".to_string(),
        "-j".to_string(),
        "1".to_string(),
    ];
    match args.output_mode {
        BrowseOutputMode::Count => arguments.push("--count".to_string()),
        BrowseOutputMode::Content => {
            arguments.push("--line-number".to_string());
            if args.context_lines > 0 {
                arguments.extend(["--context".to_string(), args.context_lines.to_string()]);
            }
        }
    }
    if args.ignore_case {
        arguments.push("--ignore-case".to_string());
    }
    if args.multiline {
        arguments.extend(["-U".to_string(), "--multiline-dotall".to_string()]);
    }
    arguments.extend([
        "-e".to_string(),
        args.pattern.clone().expect("validated pattern"),
        "--".to_string(),
        "-".to_string(),
    ]);
    arguments
}

fn format_grep_result(args: &WebBrowseArgs, output: &str, timed_out: bool) -> (String, bool) {
    let total_lines = output.lines().count();
    let (selected, line_truncated) = match args.output_mode {
        BrowseOutputMode::Count => (output.trim_end_matches('\n').to_string(), false),
        BrowseOutputMode::Content => {
            let mut lines = output
                .lines()
                .skip(args.offset)
                .map(str::to_string)
                .collect::<Vec<_>>();
            let truncated = args.max_lines > 0 && lines.len() > args.max_lines;
            if truncated {
                lines.truncate(args.max_lines);
            }
            (lines.join("\n"), truncated)
        }
    };
    let (selected, token_truncated) = truncate(&selected, MAX_TOKENS_PER_PAGE);
    let truncated = line_truncated || token_truncated;
    (
        format_grep_output(
            &selected,
            args.output_mode,
            timed_out,
            truncated,
            total_lines,
            args.offset,
        ),
        truncated,
    )
}

fn format_grep_output(
    output: &str,
    output_mode: BrowseOutputMode,
    timed_out: bool,
    truncated: bool,
    total_lines: usize,
    offset: usize,
) -> String {
    if output.trim().is_empty() {
        if offset > 0 && total_lines > 0 {
            let unit = if total_lines == 1 { "line" } else { "lines" };
            return format!("No output: offset {offset} skips all {total_lines} result {unit}.");
        }
        return if timed_out {
            "Search timed out with no results.".to_string()
        } else {
            "No matches found.".to_string()
        };
    }
    let (count, unit) = match output_mode {
        BrowseOutputMode::Count if output.trim().parse::<usize>().is_ok() => {
            let count = output.trim().parse::<usize>().expect("checked count");
            (count, if count == 1 { "match" } else { "matches" })
        }
        BrowseOutputMode::Content => {
            let count = output.lines().count();
            (count, if count == 1 { "line" } else { "lines" })
        }
        BrowseOutputMode::Count => (output.lines().count(), "lines"),
    };
    let mut header = format!("Found {count} {unit}");
    if timed_out {
        header.push_str(" (search timed out, results may be incomplete)");
    }
    if truncated {
        header.push_str(" (results truncated)");
    }
    format!("{header}\n{output}")
}

fn model_json(payload: &impl Serialize) -> Result<String, ToolError> {
    serde_json::to_string_pretty(payload).map_err(|error| ToolError::Execution(error.to_string()))
}

fn validated_url(url: &str) -> Option<String> {
    let url = url.trim();
    if url.is_empty() || url.chars().count() > MAX_URL_CHARS {
        return None;
    }
    let parsed = reqwest::Url::parse(url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return None;
    }
    Some(url.to_string())
}

fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

fn split_lines(text: &str) -> impl Iterator<Item = &str> {
    // Unlike str::lines, Python splitlines includes Unicode separators and bare CR.
    let mut lines = LINE_BREAKS.split(text).peekable();
    std::iter::from_fn(move || {
        let line = lines.next()?;
        (!line.is_empty() || lines.peek().is_some()).then_some(line)
    })
}

fn normalize_text(text: &str) -> String {
    split_lines(text)
        .map(|line| line.trim_matches(python_whitespace))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    TOKEN_CHUNKS
        .find_iter(text)
        .count()
        .max(text.len().div_ceil(4))
}

fn truncate(text: &str, max_tokens: usize) -> (String, bool) {
    if max_tokens == 0 {
        return (String::new(), !text.is_empty());
    }
    let mut selected = Vec::new();
    let mut used = 0;
    for line in split_lines(text) {
        let separator = usize::from(!selected.is_empty());
        let cost = estimate_tokens(line);
        if used + separator + cost <= max_tokens {
            selected.push(line.to_string());
            used += separator + cost;
            continue;
        }
        if let Some(remaining) = max_tokens.checked_sub(used + separator).filter(|n| *n > 0) {
            let partial = truncate_line(line, remaining);
            if !partial.is_empty() {
                selected.push(partial);
            }
        }
        return (selected.join("\n"), true);
    }
    (text.to_string(), false)
}

fn truncate_line(text: &str, max_tokens: usize) -> String {
    if estimate_tokens(text) <= max_tokens {
        return text.to_string();
    }
    let mut boundaries = text
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    boundaries.push(text.len());
    let mut lower = 0;
    let mut upper = boundaries.len() - 1;
    while lower < upper {
        let midpoint = (lower + upper).div_ceil(2);
        if estimate_tokens(&text[..boundaries[midpoint]]) <= max_tokens {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    text[..boundaries[lower]]
        .trim_end_matches(python_whitespace)
        .to_string()
}

#[cfg(test)]
#[path = "web_browse_test.rs"]
mod tests;
