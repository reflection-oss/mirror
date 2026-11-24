//! The llama.cpp router's `/models/sse` stream, read as events.

use std::future::ready;

use eventsource_stream::Eventsource;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::Value;

/// Any event from the router's `/models/sse` stream, such as a model's state change or a
/// download's progress.
#[derive(Debug, Deserialize)]
pub struct RouterEvent {
    pub model: String,
    pub event: String,
    #[serde(default)]
    pub data: Value,
}

/// The router's events, until its stream ends or breaks; events that do not parse are skipped.
pub fn router_events(response: reqwest::Response) -> impl Stream<Item = RouterEvent> + Send {
    response
        .bytes_stream()
        .eventsource()
        .take_while(|event| ready(event.is_ok()))
        .filter_map(|event| {
            ready(
                event
                    .ok()
                    .and_then(|event| serde_json::from_str(&event.data).ok()),
            )
        })
}
