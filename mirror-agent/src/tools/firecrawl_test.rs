use std::time::Duration;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::tools::web_test_server::TestResponse;
use crate::tools::web_test_server::TestServer;

#[tokio::test]
async fn searches_and_scrapes_typed_responses() {
    let mut server = TestServer::start(vec![
        TestResponse::json(
            200,
            serde_json::json!({
                "success": true,
                "data": {"web": [{
                    "url": "https://example.com/result",
                    "title": "Result",
                    "description": "Description",
                    "ignored": true
                }]}
            }),
        ),
        TestResponse::json(
            200,
            serde_json::json!({
                "success": true,
                "data": {
                    "markdown": "Page text",
                    "metadata": {
                        "title": "Page",
                        "sourceURL": "https://example.com/final"
                    }
                }
            }),
        ),
    ])
    .await;
    let client = FirecrawlClient::for_test(Some("secret"), &server.base_url, Vec::new());

    assert_eq!(
        client.search("query", 10).await.expect("search"),
        vec![SearchResult {
            title: Some("Result".to_string()),
            link: "https://example.com/result".to_string(),
            snippet: Some("Description".to_string()),
        }]
    );
    let search_request = server.request().await;
    assert_eq!(search_request.path, "/search");
    assert_eq!(
        search_request.authorization.as_deref(),
        Some("Bearer secret")
    );
    assert_eq!(
        search_request.body,
        serde_json::json!({"query": "query", "limit": 10, "sources": ["web"]})
    );

    assert_eq!(
        client
            .scrape("https://example.com/requested")
            .await
            .expect("scrape"),
        Page {
            url: "https://example.com/final".to_string(),
            title: Some("Page".to_string()),
            text: "Page text".to_string(),
        }
    );
    let scrape_request = server.request().await;
    assert_eq!(scrape_request.path, "/scrape");
    assert_eq!(
        scrape_request.body,
        serde_json::json!({
            "url": "https://example.com/requested",
            "formats": ["markdown"],
            "onlyMainContent": true
        })
    );
}

#[tokio::test]
async fn retries_transient_statuses_and_reports_the_final_status() {
    let mut server = TestServer::start(vec![
        TestResponse::text(503, "unavailable"),
        TestResponse::text(429, "limited"),
    ])
    .await;
    let client = FirecrawlClient::for_test(Some("secret"), &server.base_url, vec![Duration::ZERO]);

    assert_eq!(
        client.search("query", 10).await,
        Err(FirecrawlError::Response(
            "Firecrawl returned HTTP 429.".to_string()
        ))
    );
    assert_eq!(server.request().await.path, "/search");
    assert_eq!(server.request().await.path, "/search");
}

#[tokio::test]
async fn reports_credentials_auth_and_malformed_responses() {
    let missing = FirecrawlClient::for_test(None, "http://127.0.0.1:1", Vec::new());
    assert_eq!(
        missing.search("query", 10).await,
        Err(FirecrawlError::MissingApiKey)
    );

    let mut auth_server = TestServer::start(vec![TestResponse::text(401, "denied")]).await;
    let auth = FirecrawlClient::for_test(Some("bad"), &auth_server.base_url, Vec::new());
    assert_eq!(
        auth.search("query", 10).await,
        Err(FirecrawlError::Response(
            "Firecrawl rejected FIRECRAWL_API_KEY.".to_string()
        ))
    );
    let _ = auth_server.request().await;

    let mut malformed_server = TestServer::start(vec![TestResponse::text(200, "not JSON")]).await;
    let malformed =
        FirecrawlClient::for_test(Some("secret"), &malformed_server.base_url, Vec::new());
    assert_matches!(
        malformed.scrape("https://example.com").await,
        Err(FirecrawlError::Response(message))
            if message == "Firecrawl scrape returned an invalid response."
    );
    let _ = malformed_server.request().await;
}

#[tokio::test]
async fn revoked_authority_stops_firecrawl_before_retry_dispatch() {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    struct FirstRequestOnly(AtomicBool);

    impl FirecrawlRequestGuard for FirstRequestOnly {
        fn verify(&self) -> BoxFuture<'_, Result<(), FirecrawlError>> {
            Box::pin(async move {
                if self.0.swap(false, Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(FirecrawlError::Response("lease revoked".to_string()))
                }
            })
        }
    }

    let mut server = TestServer::start(vec![
        TestResponse::text(503, "unavailable"),
        TestResponse::json(
            200,
            serde_json::json!({"success": true, "data": {"web": []}}),
        ),
    ])
    .await;
    let client = FirecrawlClient::for_test(Some("secret"), &server.base_url, vec![Duration::ZERO])
        .with_request_guard(Arc::new(FirstRequestOnly(AtomicBool::new(true))));
    assert_eq!(
        client.search("query", 10).await,
        Err(FirecrawlError::Response("lease revoked".to_string()))
    );
    assert_eq!(server.request().await.path, "/search");
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.request())
            .await
            .is_err()
    );
}
