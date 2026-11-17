//! Model provider interface and streaming events.

use std::time::Duration;

use crate::providers::Completion;
use crate::providers::CompletionRequest;
use crate::with_fallback_roots;
use futures_core::Stream;
use futures_util::future::BoxFuture;
use mirror_protocol::StreamContent;
use mirror_protocol::StreamReasoning;
use mirror_protocol::StreamRefusal;
use mirror_protocol::StreamToolCall;
use reqwest::Client;
use reqwest::ClientBuilder;
use thiserror::Error;

pub(super) const USER_AGENT: &str = "mirror-cli";

/// A model request limit for callers with no turn deadline of their own.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// A model HTTP client; `request_timeout` bounds each whole request,
/// including the time the model spends generating.
pub(crate) fn http_client(request_timeout: Option<Duration>) -> Result<Client, LlmError> {
    build_http_client(http_client_builder(request_timeout))
}

pub(super) fn build_http_client(builder: ClientBuilder) -> Result<Client, LlmError> {
    builder
        .build()
        .map_err(|error| LlmError::Request(error_chain(&error)))
}

fn error_chain(error: &reqwest::Error) -> String {
    std::iter::successors(Some(error as &dyn std::error::Error), |cause| {
        cause.source()
    })
    .map(ToString::to_string)
    .collect::<Vec<_>>()
    .join(": ")
}

/// Shared TLS, proxy and timeout configuration for HTTP and WebSocket clients.
pub(super) fn http_client_builder(request_timeout: Option<Duration>) -> ClientBuilder {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
    let mut builder = with_fallback_roots(Client::builder())
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(10));
    if let Some(timeout) = request_timeout {
        builder = builder.timeout(timeout);
    }
    builder
}

/// Authorization checked immediately before each model HTTP request.
pub trait ModelRequestGuard: Send + Sync {
    /// Reject dispatch when execution is no longer authorized.
    fn verify(&self) -> BoxFuture<'_, Result<(), LlmError>>;
}

/// A streaming model provider.
pub trait Llm: Send + Sync {
    /// The model identifier recorded in traces.
    fn model(&self) -> &str;

    /// Run one model request and stream partial events followed by a final event.
    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + Send;

    /// Start preparing for a request like `request`; a no-op by default.
    fn prewarm(&self, _request: CompletionRequest) {}
}

/// An incremental or final event from a streaming model request.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StreamEvent {
    Content(StreamContent),
    Reasoning(StreamReasoning),
    Refusal(StreamRefusal),
    ToolCall(StreamToolCall),
    /// Cumulative generated tokens reported by the inference provider.
    OutputTokens(u64),
    Final(Completion),
}

/// A model provider failure.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum LlmError {
    #[error("model request failed: {0}")]
    Request(String),
    #[error("model context overflow: {0}")]
    ContextOverflow(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;

    #[test]
    fn client_build_errors_report_their_cause() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let builder = Client::builder()
            .tls_version_min(reqwest::tls::Version::TLS_1_3)
            .tls_version_max(reqwest::tls::Version::TLS_1_2);
        assert_matches!(
            build_http_client(builder),
            Err(LlmError::Request(message))
                if message == "builder error: empty supported tls versions"
        );
    }
}
