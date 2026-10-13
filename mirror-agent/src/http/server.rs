//! Sandbox-local HTTP access to one already-configured agent.

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;

use super::{PromptError, PromptRunner};

/// A protocol or HTTP operation failure.
#[derive(Debug, thiserror::Error)]
#[error("{1}")]
pub struct HttpError(StatusCode, String);

/// Expose one configured agent; the caller owns listener and shutdown lifetimes.
pub fn router(runner: PromptRunner) -> Router {
    Router::new()
        .route("/message", post(prompt))
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .layer(middleware::from_fn(guard_request))
        .route("/health", get(|| async { StatusCode::NO_CONTENT }))
        .with_state(runner)
}

async fn guard_request(request: Request, next: Next) -> Result<Response, HttpError> {
    if request.headers().contains_key(header::ORIGIN) {
        return Err(HttpError(
            StatusCode::FORBIDDEN,
            "browser origins are not supported".into(),
        ));
    }
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok(response)
}

async fn prompt(
    State(runner): State<PromptRunner>,
    headers: HeaderMap,
    content: String,
) -> Result<StatusCode, HttpError> {
    let media_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next());
    if !media_type.is_some_and(|value| value.trim().eq_ignore_ascii_case("text/plain")) {
        return Err(HttpError(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "expected Content-Type: text/plain".into(),
        ));
    }
    runner.prompt(content).await.map_err(|error| {
        let status = match &error {
            PromptError::Busy | PromptError::Interrupted => StatusCode::CONFLICT,
            PromptError::Empty => StatusCode::BAD_REQUEST,
            PromptError::TimedOut => StatusCode::GATEWAY_TIMEOUT,
            PromptError::Failed(_) => StatusCode::BAD_GATEWAY,
            PromptError::Service(_) => StatusCode::SERVICE_UNAVAILABLE,
        };
        HttpError(status, error.to_string())
    })?;
    Ok(StatusCode::NO_CONTENT)
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}
