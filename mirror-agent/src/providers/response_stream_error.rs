//! Payload-free diagnostics for failures while reading a model response stream.

use std::error::Error;
use std::io;

use eventsource_stream::EventStreamError;
use reqwest::StatusCode;

use super::LlmError;

/// Retain the response status and safe error categories, never wire data or URLs.
pub fn response_stream_error(
    model: &str,
    status: StatusCode,
    error: EventStreamError<reqwest::Error>,
) -> LlmError {
    let detail = match error {
        EventStreamError::Transport(error) => transport_detail(&error),
        EventStreamError::Utf8(_) => "invalid UTF-8 in response stream".to_string(),
        EventStreamError::Parser(_) => "invalid SSE framing in response stream".to_string(),
    };
    LlmError::Request(format!(
        "{detail} [HTTP {}, model={model}]",
        status.as_u16()
    ))
}

fn transport_detail(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        return "response body timed out".to_string();
    }
    // Reqwest's Decode also wraps HTTP body read failures, not just decompression.
    let mut detail = "response body read/decode failed".to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(error) = cause.downcast_ref::<io::Error>() {
            // Display/Debug of the error itself may contain URLs or response bytes.
            detail.push_str(&format!(" (I/O {:?})", error.kind()));
            break;
        }
        source = cause.source();
    }
    detail
}
