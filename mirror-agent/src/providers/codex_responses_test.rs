use assert_matches::assert_matches;
use mirror_protocol::InferenceConfig;
use mirror_protocol::SystemMessage;
use mirror_protocol::UserMessage;
use pretty_assertions::assert_eq;

use super::*;

struct TestAuthFile(PathBuf);

impl TestAuthFile {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mirror-codex-auth-{nonce}.json"));
        std::fs::write(
            &path,
            json!({
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": "sk-must-not-be-used",
                "tokens": {
                    "access_token": "oauth-token",
                    "refresh_token": "refresh-token",
                    "account_id": "account-1",
                },
            })
            .to_string(),
        )
        .expect("test auth file is writable");
        Self(path)
    }
}

impl Drop for TestAuthFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn provider(auth_path: &Path) -> CodexResponses {
    CodexResponses::new(
        CodexResponsesConfig {
            model: "gpt-test".to_string(),
            base_url: Some("http://test/backend-api/codex".to_string()),
            auth_path: Some(auth_path.to_path_buf()),
            prompt_cache_key: "session-1".to_string(),
        },
        super::super::http_client(None).expect("test HTTP client configuration is valid"),
    )
    .expect("test auth is valid")
}

#[tokio::test]
async fn responses_and_codex_forward_cumulative_usage() {
    use super::super::{ModelApi, ModelProvider, ModelProviderConfig};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let auth = TestAuthFile::new();
    for api in [ModelApi::Responses, ModelApi::CodexResponses] {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("server address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = [0; 16384];
            let received = socket.read(&mut request).await.expect("read request");
            assert!(received > 0, "provider sent a request");
            assert!(String::from_utf8_lossy(&request[..received])
                .to_ascii_lowercase()
                .contains("\r\nuser-agent: mirror-cli\r\n"));
            let body = concat!(
                "data: {\"type\":\"response.in_progress\",\"response\":{\"usage\":{\"output_tokens\":4}}}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":10,\"output_tokens\":12},\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\"}]}]}}\n\n",
            );
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write stream");
        });
        let provider = ModelProvider::new(
            ModelProviderConfig {
                context_window: Some(crate::providers::DEFAULT_CONTEXT_WINDOW),
                api,
                model: "test".to_string(),
                api_key: Some("test-key".to_string()),
                base_url: Some(format!("http://{address}")),
                auth_path: Some(auth.0.clone()),
                prompt_cache_key: "test-session".to_string(),
                request_timeout: None,
            },
            None,
        )
        .await
        .expect("provider");
        let stream = provider.stream(CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            inference: None,
        });
        pin_mut!(stream);
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::OutputTokens(4))));
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::Content(_))));
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::OutputTokens(12))));
        assert_matches!(stream.next().await, Some(Ok(StreamEvent::Final(completion))) => {
            assert_eq!(completion.usage.expect("final usage").completion_tokens, 12);
        });
        assert!(stream.next().await.is_none());
        server.await.expect("server completed");
    }
}

#[test]
fn request_moves_system_prompt_to_codex_fields() {
    let auth = TestAuthFile::new();
    let request = CompletionRequest {
        messages: vec![
            chat_record::Message::System(SystemMessage {
                content: "be helpful".to_string(),
            }),
            chat_record::Message::User(UserMessage {
                checkpoint: None,
                content: "hello".to_string(),
            }),
        ],
        tools: vec![],
        max_tokens: Some(1024),
        inference: Some(InferenceConfig {
            model: "gpt-test".to_string(),
            api: None,
            base_url: None,
            reasoning_effort: Some("medium".to_string()),
            ..Default::default()
        }),
    };

    let body = provider(&auth.0)
        .request_body(request)
        .expect("valid request serializes");

    assert_eq!(body["instructions"], "be helpful");
    assert_eq!(
        body["input"],
        json!([{
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }])
    );
    assert_eq!(body["text"], json!({"verbosity": "low"}));
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(
        body["reasoning"],
        json!({"effort": "medium", "summary": "auto"})
    );
    assert!(body.get("max_output_tokens").is_none());
}

#[test]
fn request_keeps_extra_body_over_codex_defaults() {
    let auth = TestAuthFile::new();
    let request = CompletionRequest {
        messages: vec![chat_record::Message::User(UserMessage {
            content: "hello".to_string(),
            checkpoint: None,
        })],
        tools: vec![],
        max_tokens: None,
        inference: Some(InferenceConfig {
            model: "gpt-test".to_string(),
            extra_body: Some(
                r#"{"text": {"verbosity": "high"}, "metadata": {"run": "r1"}}"#.to_string(),
            ),
            ..Default::default()
        }),
    };

    let body = provider(&auth.0)
        .request_body(request)
        .expect("valid request serializes");

    assert_eq!(body["text"], json!({"verbosity": "high"}));
    assert_eq!(body["metadata"], json!({"run": "r1"}));
    assert_eq!(body["tool_choice"], "auto");
}

#[test]
fn request_preserves_an_empty_system_prompt() {
    let auth = TestAuthFile::new();
    let body = provider(&auth.0)
        .request_body(CompletionRequest {
            messages: vec![chat_record::Message::System(SystemMessage {
                content: String::new(),
            })],
            tools: vec![],
            max_tokens: None,
            inference: None,
        })
        .expect("valid request serializes");

    assert_eq!(body["instructions"], "");
}

#[test]
fn auth_uses_chatgpt_tokens_and_ignores_api_key_mode() {
    let auth = TestAuthFile::new();

    let loaded = load_auth(&auth.0).expect("ChatGPT login loads");

    assert_eq!(loaded.access_token, "oauth-token");
    assert_eq!(loaded.account_id, "account-1");
}

#[test]
fn missing_login_has_codex_login_hint() {
    let path = std::env::temp_dir().join("mirror-codex-auth-does-not-exist.json");

    let error = load_auth(&path).err().expect("missing login fails");

    assert!(error.to_string().contains("codex login --device-auth"));
}

#[test]
fn unauthorized_response_requests_one_token_refresh() {
    let error = LlmError::Request("401 Unauthorized: expired [model=gpt-test]".to_string());

    assert!(is_unauthorized(&error));
}

#[test]
fn token_refresh_window_uses_jwt_expiry() {
    let expiry = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock is after the Unix epoch")
        .as_secs()
        + 60;
    let payload = URL_SAFE_NO_PAD.encode(json!({"exp": expiry}).to_string());
    let token = format!("header.{payload}.signature");

    assert!(token_needs_refresh(&token));
    assert!(!token_needs_refresh("opaque-token"));
}

#[tokio::test]
async fn revoked_authority_prevents_the_codex_retry_from_reaching_the_server() {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use futures_util::future::BoxFuture;
    use futures_util::TryStreamExt;

    use crate::tools::web_test_server::TestResponse;
    use crate::tools::web_test_server::TestServer;

    struct FirstRequestOnly {
        active: AtomicBool,
        auth_path: PathBuf,
    }

    impl ModelRequestGuard for FirstRequestOnly {
        fn verify(&self) -> BoxFuture<'_, Result<(), LlmError>> {
            Box::pin(async move {
                if !self.active.swap(false, Ordering::SeqCst) {
                    return Err(LlmError::Request("lease revoked".to_string()));
                }
                // Simulate another process refreshing the rejected credential.
                std::fs::write(
                    &self.auth_path,
                    json!({"tokens": {
                        "access_token": "updated-token",
                        "refresh_token": "refresh-token",
                        "account_id": "account-1"
                    }})
                    .to_string(),
                )
                .expect("replace credentials");
                Ok(())
            })
        }
    }

    let auth = TestAuthFile::new();
    let mut server = TestServer::start(vec![
        TestResponse::text(401, "expired"),
        TestResponse::text(200, "data: [DONE]\n\n"),
    ])
    .await;
    let guard = FirstRequestOnly {
        active: AtomicBool::new(true),
        auth_path: auth.0.clone(),
    };
    let provider = CodexResponses::new(
        CodexResponsesConfig {
            model: "gpt-test".to_string(),
            base_url: Some(server.base_url.clone()),
            auth_path: Some(auth.0.clone()),
            prompt_cache_key: "session-1".to_string(),
        },
        super::super::http_client(None).expect("HTTP client"),
    )
    .expect("provider");
    let request = CompletionRequest {
        messages: vec![],
        tools: vec![],
        max_tokens: None,
        inference: None,
    };
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        provider
            .stream_guarded(request, Some(&guard))
            .try_collect::<Vec<_>>(),
    )
    .await
    .expect("request completes");
    assert_eq!(result, Err(LlmError::Request("lease revoked".to_string())));
    assert_eq!(
        server.request().await.authorization.as_deref(),
        Some("Bearer oauth-token")
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.request())
            .await
            .is_err()
    );
}
