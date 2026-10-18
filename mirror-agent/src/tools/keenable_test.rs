use std::time::Duration;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::tools::web_test_server::TestResponse;
use crate::tools::web_test_server::TestServer;

#[tokio::test]
async fn searches_and_fetches_public_endpoints_without_a_key() {
    let mut server = TestServer::start(vec![
        TestResponse::json(
            200,
            serde_json::json!({
                "query": "query",
                "results": [{
                    "url": "https://example.com/result",
                    "title": "Result",
                    "description": "Description",
                    "snippet": "Snippet",
                    "published_at": "2026-01-15T10:30:00Z"
                }]
            }),
        ),
        TestResponse::json(
            200,
            serde_json::json!({
                "url": "https://example.com/final",
                "title": "Page",
                "content": "Page text"
            }),
        ),
    ])
    .await;
    let client = KeenableClient::for_test(None, &server.base_url, Vec::new());

    assert_eq!(
        client.search("query", 10).await.expect("search"),
        vec![SearchResult {
            title: Some("Result".to_string()),
            link: "https://example.com/result".to_string(),
            snippet: Some("Snippet".to_string()),
        }]
    );
    let search_request = server.request().await;
    assert_eq!(search_request.path, "/search/public");
    assert_eq!(search_request.authorization, None);
    assert_eq!(
        search_request.body,
        serde_json::json!({"query": "query", "max_results": 10})
    );

    assert_eq!(
        client
            .scrape("https://example.com/requested")
            .await
            .expect("fetch"),
        Page {
            url: "https://example.com/final".to_string(),
            title: Some("Page".to_string()),
            text: "Page text".to_string(),
        }
    );
    assert_eq!(
        server.request().await.path,
        "/fetch/public?url=https%3A%2F%2Fexample.com%2Frequested&live=true"
    );
}

#[tokio::test]
async fn uses_keyed_endpoints_with_an_api_key() {
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        serde_json::json!({"results": [{"url": "https://example.com", "description": "Description"}]}),
    )])
    .await;
    let client = KeenableClient::for_test(Some(" secret "), &server.base_url, Vec::new());

    assert_eq!(
        client.search("query", 10).await.expect("search")[0].snippet,
        Some("Description".to_string())
    );
    let request = server.request().await;
    assert_eq!(request.path, "/search");
    assert_eq!(request.authorization.as_deref(), Some("Bearer secret"));
}

#[tokio::test]
async fn retries_transient_statuses_and_reports_failures() {
    let mut server = TestServer::start(vec![
        TestResponse::text(503, "unavailable"),
        TestResponse::text(429, "limited"),
    ])
    .await;
    let client = KeenableClient::for_test(None, &server.base_url, vec![Duration::ZERO]);
    assert_eq!(
        client.search("query", 10).await,
        Err(KeenableError::Response(
            "Keenable returned HTTP 429.".to_string()
        ))
    );
    assert_eq!(server.request().await.path, "/search/public");
    assert_eq!(server.request().await.path, "/search/public");

    let mut auth_server = TestServer::start(vec![TestResponse::text(401, "denied")]).await;
    let auth = KeenableClient::for_test(Some("bad"), &auth_server.base_url, Vec::new());
    assert_eq!(
        auth.search("query", 10).await,
        Err(KeenableError::Response(
            "Keenable rejected KEENABLE_API_KEY.".to_string()
        ))
    );
    let _ = auth_server.request().await;

    let mut malformed_server = TestServer::start(vec![TestResponse::text(200, "not JSON")]).await;
    let malformed = KeenableClient::for_test(None, &malformed_server.base_url, Vec::new());
    assert_matches!(
        malformed.scrape("https://example.com").await,
        Err(KeenableError::Response(message))
            if message == "Keenable fetch returned an invalid response."
    );
    let _ = malformed_server.request().await;
}
