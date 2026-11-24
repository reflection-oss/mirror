//! The llama.cpp router's `/models/sse` stream, read as model state changes.

use std::future::ready;
use std::pin::Pin;

use futures_core::Stream;
use futures_util::StreamExt;

use super::{router_events, RouterEvent};

/// The router's state changes, until its stream ends or breaks.
pub struct ModelEvents {
    events: Pin<Box<dyn Stream<Item = ModelEvent> + Send>>,
}

/// A model's state change from the router's `/models/sse` stream.
#[derive(Debug, PartialEq)]
pub struct ModelEvent {
    pub model: String,
    /// `loading`, `loaded`, `sleeping`, `unloaded`, and so on.
    pub status: String,
}

impl ModelEvents {
    pub(super) fn new(response: reqwest::Response) -> Self {
        let events = router_events(response).filter_map(|event| ready(model_event(event)));
        Self {
            events: Box::pin(events),
        }
    }

    /// The next state change; `None` once the stream ends or breaks.
    pub async fn next(&mut self) -> Option<ModelEvent> {
        self.events.next().await
    }
}

/// The state change one event reports, ignoring other events.
fn model_event(event: RouterEvent) -> Option<ModelEvent> {
    if !matches!(event.event.as_str(), "model_status" | "status_change") {
        return None;
    }
    Some(ModelEvent {
        status: event.data["status"].as_str()?.to_owned(),
        model: event.model,
    })
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    fn event(data: &str) -> RouterEvent {
        serde_json::from_str(data).expect("router event")
    }

    #[test]
    fn state_changes_are_reported_and_other_events_are_ignored() {
        let loading = r#"{"model":"m","event":"status_change","data":{"status":"loading"}}"#;
        let download = r#"{"model":"m","event":"download_progress","data":{"done":1}}"#;

        assert_matches!(
            model_event(event(loading)),
            Some(ModelEvent { model, status }) if model == "m" && status == "loading"
        );
        assert_matches!(model_event(event(download)), None);
    }
}
