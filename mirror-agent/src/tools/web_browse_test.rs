use pretty_assertions::assert_eq;

use super::*;
use crate::tools::web_test_server::TestResponse;
use crate::tools::web_test_server::TestServer;
use crate::tools::FirecrawlClient;

fn args(url: &str) -> WebBrowseArgs {
    WebBrowseArgs {
        url: url.to_string(),
        pattern: None,
        output_mode: BrowseOutputMode::Content,
        ignore_case: false,
        context_lines: 0,
        multiline: false,
        max_lines: 250,
        offset: 0,
    }
}

#[tokio::test]
async fn previews_then_searches_the_cached_page_and_final_url() {
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        serde_json::json!({
            "success": true,
            "data": {
                "markdown": " first line \n\nneedle line\nlast line ",
                "metadata": {
                    "title": "Example",
                    "url": "https://example.com/final"
                }
            }
        }),
    )])
    .await;
    let client = FirecrawlClient::for_test(Some("secret"), &server.base_url, Vec::new());
    let tool = WebBrowseTool::new(client);

    let preview = tool
        .run(
            args("https://example.com/page"),
            &ExecutionContext::default(),
        )
        .await
        .expect("preview");
    let preview_payload: Value = serde_json::from_str(&preview.content).expect("preview JSON");
    assert_eq!(
        preview_payload,
        serde_json::json!({
            "url": "https://example.com/page",
            "final_url": "https://example.com/final",
            "cached": false,
            "title": "Example",
            "text_preview": "first line\nneedle line\nlast line",
            "estimated_text_tokens": 8,
            "line_count": 3,
            "truncated": false,
            "hint": "Full page text is cached. Browse previews and search results are capped around 8000 tokens. Add a regex pattern and optional context_lines to inspect relevant parts of the page."
        })
    );
    assert_eq!(
        preview.display.as_deref(),
        Some("https://example.com/final")
    );

    let mut search_args = args("https://example.com/page");
    search_args.pattern = Some("needle".to_string());
    let searched = tool
        .run(search_args, &ExecutionContext::default())
        .await
        .expect("search");
    let search_payload: Value = serde_json::from_str(&searched.content).expect("search JSON");
    assert_eq!(search_payload["cached"], true);
    assert_eq!(search_payload["output"], "Found 1 line\n2:needle line");
    assert_eq!(search_payload["truncated"], false);
    assert_eq!(search_payload["timed_out"], false);

    let final_preview = tool
        .run(
            args("https://example.com/final"),
            &ExecutionContext::default(),
        )
        .await
        .expect("final URL preview");
    assert_eq!(
        serde_json::from_str::<Value>(&final_preview.content).expect("final JSON")["cached"],
        true
    );
    let request = server.request().await;
    assert_eq!(request.path, "/scrape");
    assert_eq!(
        request.body,
        serde_json::json!({
            "url": "https://example.com/page",
            "formats": ["markdown"],
            "onlyMainContent": true
        })
    );
}

#[tokio::test]
async fn supports_count_context_case_multiline_offset_and_limits() {
    let server = TestServer::start(vec![TestResponse::json(
        200,
        serde_json::json!({
            "success": true,
            "data": {
                "markdown": "before\nNeedle one\nmiddle\nneedle two\nafter",
                "metadata": {}
            }
        }),
    )])
    .await;
    let client = FirecrawlClient::for_test(Some("secret"), &server.base_url, Vec::new());
    let tool = WebBrowseTool::new(client);

    let mut count_args = args("https://example.com");
    count_args.pattern = Some("needle".to_string());
    count_args.output_mode = BrowseOutputMode::Count;
    count_args.ignore_case = true;
    let counted = tool
        .run(count_args, &ExecutionContext::default())
        .await
        .expect("count");
    assert_eq!(
        serde_json::from_str::<Value>(&counted.content).expect("count JSON")["output"],
        "Found 2 matches\n2"
    );

    let mut content_args = args("https://example.com");
    content_args.pattern = Some("Needle one.*needle two".to_string());
    content_args.ignore_case = true;
    content_args.multiline = true;
    content_args.context_lines = 1;
    content_args.max_lines = 1;
    let content = tool
        .run(content_args, &ExecutionContext::default())
        .await
        .expect("content");
    let content_payload: Value = serde_json::from_str(&content.content).expect("content JSON");
    assert!(content_payload["output"]
        .as_str()
        .expect("output")
        .starts_with("Found 1 line (results truncated)"));
    assert_eq!(content_payload["truncated"], true);

    let mut offset_args = args("https://example.com");
    offset_args.pattern = Some("needle".to_string());
    offset_args.ignore_case = true;
    offset_args.offset = 10;
    let offset = tool
        .run(offset_args, &ExecutionContext::default())
        .await
        .expect("offset");
    assert_eq!(
        serde_json::from_str::<Value>(&offset.content).expect("offset JSON")["output"],
        "No output: offset 10 skips all 2 result lines."
    );
}

#[tokio::test]
async fn reports_invalid_urls_patterns_and_regex_errors() {
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        serde_json::json!({
            "success": true,
            "data": {"markdown": "page", "metadata": {}}
        }),
    )])
    .await;
    let client = FirecrawlClient::for_test(Some("secret"), &server.base_url, Vec::new());
    let tool = WebBrowseTool::new(client);

    let invalid_url = tool
        .run(args("javascript:alert(1)"), &ExecutionContext::default())
        .await
        .expect("invalid URL result");
    assert_eq!(
        invalid_url,
        ToolOutput::error("web_browse requires an HTTP(S) URL.")
    );

    let mut empty_pattern = args("https://example.com");
    empty_pattern.pattern = Some(String::new());
    assert!(matches!(
        tool.run(empty_pattern, &ExecutionContext::default()).await,
        Err(ToolError::InvalidArguments(_))
    ));

    let mut invalid_regex = args("https://example.com");
    invalid_regex.pattern = Some("[".to_string());
    let regex_result = tool
        .run(invalid_regex, &ExecutionContext::default())
        .await
        .expect("regex result");
    assert!(!regex_result.success);
    assert!(regex_result.content.contains("Ripgrep error (exit 2)"));
    let _ = server.request().await;
}

#[test]
fn bounds_preview_tokens() {
    let text = "word ".repeat(20_000);
    let (preview, truncated) = truncate(&text, MAX_TOKENS_PER_PAGE);

    assert!(truncated);
    assert!(estimate_tokens(&preview) <= MAX_TOKENS_PER_PAGE);
    assert!(estimate_tokens(&text) > MAX_TOKENS_PER_PAGE);
}
