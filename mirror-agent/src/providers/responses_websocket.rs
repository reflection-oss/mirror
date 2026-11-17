//! Persistent WebSocket transport for the OpenAI Responses API.
//!
//! A request that extends the previous one sends only its new input with
//! `previous_response_id`, which works with `store: false`.

use std::collections::BTreeMap;
use std::time::Duration;

use futures_core::Stream;
use futures_util::SinkExt;
use futures_util::StreamExt;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderValue;
use reqwest::Client;
use reqwest::StatusCode;
use reqwest::Upgraded;
use serde_json::json;
use serde_json::Map;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_tungstenite::tungstenite::handshake::client::generate_key;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use super::responses::replayed_output;
use super::LlmError;

#[cfg(test)]
#[path = "responses_websocket_test.rs"]
mod tests;

/// A handle to one session's Responses WebSocket connection; clones share it.
#[derive(Clone, Debug)]
pub struct ResponsesWebSocket {
    requests: mpsc::UnboundedSender<WebSocketRequest>,
    request_timeout: Option<Duration>,
}

/// The `https` Responses endpoint to upgrade and the headers to send with it.
#[derive(Clone, Debug, PartialEq)]
pub struct WebSocketTarget {
    pub url: String,
    pub headers: HeaderMap,
}

/// An event from a request sent over the WebSocket.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum WebSocketEvent {
    /// The request was sent.
    Sent(WebSocketDelivery),
    /// One Responses streaming event, as JSON text.
    Payload(String),
    /// Send the request over HTTP instead, for the stated reason.
    Unavailable(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WebSocketDelivery {
    pub incremental: bool,
    pub reason: &'static str,
}

#[derive(Debug)]
struct WebSocketRequest {
    target: WebSocketTarget,
    body: Map<String, Value>,
    events: Option<EventSender>,
    deadline: Option<Deadline>,
}

#[derive(Clone, Copy, Debug)]
struct Deadline {
    at: Instant,
    limit: Duration,
}

type EventSender = mpsc::UnboundedSender<Result<WebSocketEvent, LlmError>>;

struct WebSocketSession {
    client: Client,
    connection: Option<Connection>,
    upgrade_failures: BTreeMap<String, u32>,
}

struct Connection {
    target: WebSocketTarget,
    socket: WebSocketStream<Upgraded>,
    opened_at: Instant,
    requests_sent: usize,
    previous: Option<PreviousResponse>,
}

#[derive(Debug, PartialEq)]
struct PreviousResponse {
    response_id: String,
    settings: Map<String, Value>,
    input: Vec<Value>,
}

enum Exchange {
    Completed {
        response_id: Option<String>,
        output: Vec<Value>,
    },
    Failed,
    NotStarted(String),
    Abandoned {
        kept: bool,
    },
}

enum UpgradeError {
    Unauthorized(LlmError),
    Unsupported(String),
    Transient(String),
}

const RESPONSES_WEBSOCKETS_BETA: &str = "responses_websockets=2026-02-06";
/// The server closes connections after 60 minutes.
const MAX_CONNECTION_AGE: Duration = Duration::from_secs(55 * 60);
/// Reasoning can run for minutes without events.
const IDLE_TIMEOUT: Duration = super::DEFAULT_REQUEST_TIMEOUT;
const INTERRUPT_DEADLINE: Duration = Duration::from_secs(10);
const PREWARM_DEADLINE: Duration = Duration::from_secs(30);
const MAX_UPGRADE_FAILURES: u32 = 5;
const PREVIOUS_RESPONSE_NOT_FOUND: &str = "previous_response_not_found";
const CONNECTION_LIMIT_REACHED: &str = "websocket_connection_limit_reached";
/// Fields that shape one response, not the conversation it continues.
const PER_RESPONSE_FIELDS: [&str; 4] = [
    "input",
    "previous_response_id",
    "generate",
    "max_output_tokens",
];
const WEBSOCKET_HOSTS: [&str; 2] = ["api.openai.com", "chatgpt.com"];

impl ResponsesWebSocket {
    /// Start an owned HTTP/1.1 connection task. Requires a Tokio runtime.
    pub fn spawn(request_timeout: Option<Duration>) -> Result<Self, LlmError> {
        // HTTP/2 strips the connection headers required by the WebSocket upgrade.
        let client =
            super::build_http_client(super::http_client_builder(request_timeout).http1_only())?;
        let (requests, mut receiver) = mpsc::unbounded_channel::<WebSocketRequest>();
        tokio::spawn(async move {
            let mut session = WebSocketSession {
                client,
                connection: None,
                upgrade_failures: BTreeMap::new(),
            };
            while let Some(request) = receiver.recv().await {
                session.handle(request).await;
            }
        });
        Ok(Self {
            requests,
            request_timeout,
        })
    }

    /// Whether the endpoint at `base_url` serves WebSocket mode over TLS.
    pub fn serves(base_url: &str) -> bool {
        reqwest::Url::parse(base_url).is_ok_and(|url| {
            url.scheme() == "https"
                && url
                    .host_str()
                    .is_some_and(|host| WEBSOCKET_HOSTS.contains(&host))
        })
    }

    /// Send one Responses request body and stream its events.
    pub fn stream(
        &self,
        target: WebSocketTarget,
        body: Map<String, Value>,
    ) -> impl Stream<Item = Result<WebSocketEvent, LlmError>> + Send + 'static {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let deadline = self.request_timeout.map(Deadline::after);
        let sent = self.requests.send(WebSocketRequest {
            target,
            body,
            events: Some(events),
            deadline,
        });
        async_stream::stream! {
            if sent.is_err() {
                yield Ok(WebSocketEvent::Unavailable("connection task stopped".to_string()));
                return;
            }
            loop {
                let event = tokio::select! {
                    biased;
                    limit = expired(deadline) => Err(limit),
                    event = receiver.recv() => Ok(event),
                };
                match event {
                    Ok(Some(event)) => yield event,
                    Ok(None) => return,
                    Err(limit) => {
                        yield Err(LlmError::Request(format!(
                            "Responses WebSocket request timed out after {limit:?}"
                        )));
                        return;
                    }
                }
            }
        }
    }

    /// Load `body` into server state without generating.
    pub fn prewarm(&self, target: WebSocketTarget, mut body: Map<String, Value>) {
        body.insert("generate".to_string(), Value::Bool(false));
        let _ = self.requests.send(WebSocketRequest {
            target,
            body,
            events: None,
            deadline: Some(Deadline::after(PREWARM_DEADLINE)),
        });
    }
}

impl Deadline {
    fn after(limit: Duration) -> Self {
        Self {
            at: Instant::now() + limit,
            limit,
        }
    }
}

impl WebSocketSession {
    async fn handle(&mut self, request: WebSocketRequest) {
        let WebSocketRequest {
            target,
            body,
            events,
            deadline,
        } = request;
        let Some(deadline) = deadline else {
            self.deliver(target, body, events, None).await;
            return;
        };
        if deadline.at <= Instant::now() {
            return;
        }
        let delivery = self.deliver(target, body, events, Some(deadline));
        if tokio::time::timeout_at(deadline.at + INTERRUPT_DEADLINE, delivery)
            .await
            .is_err()
        {
            self.connection = None;
        }
    }

    async fn deliver(
        &mut self,
        target: WebSocketTarget,
        mut body: Map<String, Value>,
        events: Option<EventSender>,
        deadline: Option<Deadline>,
    ) {
        if self.upgrade_failures.get(&target.url) >= Some(&MAX_UPGRADE_FAILURES) {
            unavailable(events.as_ref(), "WebSocket mode is off for this endpoint");
            return;
        }
        body.remove("stream");
        if body.remove("background") == Some(Value::Bool(true)) {
            unavailable(events.as_ref(), "background responses use HTTP");
            return;
        }
        loop {
            let reused = self.reusable(&target);
            if !reused {
                if let Err(error) = self.connect(&target).await {
                    self.report_upgrade_error(&target, events.as_ref(), error);
                    return;
                }
            }
            // Connecting can outlast the deadline; nobody waits on what would be sent.
            if deadline.is_some_and(|deadline| deadline.at <= Instant::now()) {
                return;
            }
            let Some(connection) = self.connection.as_mut() else {
                return;
            };
            match connection.send(&body, events.as_ref(), deadline).await {
                Exchange::Completed { .. } | Exchange::Abandoned { kept: true } => return,
                Exchange::Failed | Exchange::Abandoned { kept: false } => {
                    self.connection = None;
                    return;
                }
                Exchange::NotStarted(_) if reused => self.connection = None,
                Exchange::NotStarted(reason) => {
                    self.connection = None;
                    self.report_upgrade_error(
                        &target,
                        events.as_ref(),
                        UpgradeError::Transient(reason),
                    );
                    return;
                }
            }
        }
    }

    fn reusable(&mut self, target: &WebSocketTarget) -> bool {
        let reusable = self.connection.as_ref().is_some_and(|connection| {
            &connection.target == target && connection.opened_at.elapsed() < MAX_CONNECTION_AGE
        });
        if !reusable {
            self.connection = None;
        }
        reusable
    }

    async fn connect(&mut self, target: &WebSocketTarget) -> Result<(), UpgradeError> {
        let socket = upgrade(&self.client, target).await?;
        self.upgrade_failures.remove(&target.url);
        self.connection = Some(Connection {
            target: target.clone(),
            socket,
            opened_at: Instant::now(),
            requests_sent: 0,
            previous: None,
        });
        Ok(())
    }

    fn report_upgrade_error(
        &mut self,
        target: &WebSocketTarget,
        events: Option<&EventSender>,
        error: UpgradeError,
    ) {
        let reason = match error {
            UpgradeError::Unauthorized(error) => {
                send(events, Err(error));
                return;
            }
            UpgradeError::Unsupported(reason) => {
                self.upgrade_failures
                    .insert(target.url.clone(), MAX_UPGRADE_FAILURES);
                reason
            }
            UpgradeError::Transient(reason) => {
                *self.upgrade_failures.entry(target.url.clone()).or_default() += 1;
                reason
            }
        };
        unavailable(events, &reason);
    }
}

impl Connection {
    async fn send(
        &mut self,
        body: &Map<String, Value>,
        events: Option<&EventSender>,
        deadline: Option<Deadline>,
    ) -> Exchange {
        let (outgoing, delivery) = self.prepare(body);
        self.requests_sent += 1;
        send(events, Ok(WebSocketEvent::Sent(delivery)));
        let exchange = self.exchange(outgoing, events, deadline).await;
        self.previous = match &exchange {
            Exchange::Completed {
                response_id: Some(response_id),
                output,
            } => continuation(response_id.clone(), body, output.clone()),
            _ => None,
        };
        exchange
    }

    fn prepare(&self, body: &Map<String, Value>) -> (Map<String, Value>, WebSocketDelivery) {
        let full = |reason| {
            (
                body.clone(),
                WebSocketDelivery {
                    incremental: false,
                    reason,
                },
            )
        };
        let Some(previous) = &self.previous else {
            return full(if self.requests_sent == 0 {
                "new_connection"
            } else {
                "no_previous_response"
            });
        };
        match incremental_body(previous, body) {
            Ok(outgoing) => (
                outgoing,
                WebSocketDelivery {
                    incremental: true,
                    reason: "continued",
                },
            ),
            Err(reason) => full(reason),
        }
    }

    async fn exchange(
        &mut self,
        mut body: Map<String, Value>,
        events: Option<&EventSender>,
        deadline: Option<Deadline>,
    ) -> Exchange {
        body.insert("type".to_string(), json!("response.create"));
        if self.send_frame(Value::Object(body)).await.is_err() {
            return Exchange::NotStarted("could not send the request".to_string());
        }
        let mut started = false;
        let mut response_id = None;
        let mut output = Vec::new();
        loop {
            let message = tokio::select! {
                message = tokio::time::timeout(IDLE_TIMEOUT, self.socket.next()) => Some(message),
                () = consumer_gone(events) => None,
                _ = expired(deadline) => None,
            };
            let Some(message) = message else {
                return self.interrupt(response_id).await;
            };
            let text = match message {
                Ok(Some(Ok(Message::Text(text)))) => text.as_str().to_string(),
                Ok(Some(Ok(Message::Ping(payload)))) => {
                    let _ = self.socket.send(Message::Pong(payload)).await;
                    continue;
                }
                Ok(Some(Ok(Message::Binary(_) | Message::Pong(_) | Message::Frame(_)))) => continue,
                Ok(Some(Ok(Message::Close(frame)))) => {
                    let reason = frame.map_or_else(
                        || "connection closed".to_string(),
                        |frame| frame.reason.to_string(),
                    );
                    return interrupted(started, reason, events);
                }
                Ok(Some(Err(error))) => return interrupted(started, error.to_string(), events),
                Ok(None) => return interrupted(started, "connection closed".to_string(), events),
                Err(_) => return interrupted(started, "idle timeout".to_string(), events),
            };
            let event: Value = serde_json::from_str(&text).unwrap_or_default();
            if !started && is_restartable_error(&event) {
                return Exchange::NotStarted(text);
            }
            started = true;
            response_id = response_id.or_else(|| event_response_id(&event));
            let kind = event
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if kind == "response.output_item.done" {
                output.extend(event.get("item").cloned());
            }
            if !send(events, Ok(WebSocketEvent::Payload(text))) {
                return self.interrupt(response_id).await;
            }
            match kind.as_str() {
                "response.completed" | "response.incomplete" | "response.done" => {
                    let reported = event
                        .pointer("/response/output")
                        .and_then(Value::as_array)
                        .filter(|reported| !reported.is_empty());
                    return Exchange::Completed {
                        response_id,
                        output: reported.cloned().unwrap_or(output),
                    };
                }
                "response.failed" | "error" => return Exchange::Failed,
                _ => {}
            }
        }
    }

    async fn interrupt(&mut self, response_id: Option<String>) -> Exchange {
        let Some(response_id) = response_id else {
            return Exchange::Abandoned { kept: false };
        };
        let frame = json!({
            "type": "response.interrupt",
            "response_id": response_id,
            "mode": "discard_partial_items",
        });
        if self.send_frame(frame).await.is_err() {
            return Exchange::Abandoned { kept: false };
        }
        let kept = tokio::time::timeout(INTERRUPT_DEADLINE, self.drain_interrupted(&response_id))
            .await
            .unwrap_or(false);
        Exchange::Abandoned { kept }
    }

    async fn drain_interrupted(&mut self, response_id: &str) -> bool {
        let mut accepted = false;
        let mut ended = false;
        while !(accepted && ended) {
            let text = match self.socket.next().await {
                Some(Ok(Message::Text(text))) => text.as_str().to_string(),
                Some(Ok(Message::Ping(payload))) => {
                    let _ = self.socket.send(Message::Pong(payload)).await;
                    continue;
                }
                Some(Ok(Message::Binary(_) | Message::Pong(_) | Message::Frame(_))) => continue,
                Some(Ok(Message::Close(_)) | Err(_)) | None => return false,
            };
            let event: Value = serde_json::from_str(&text).unwrap_or_default();
            match event.get("type").and_then(Value::as_str) {
                Some("response.interrupt.accepted") => {
                    accepted =
                        event.get("response_id").and_then(Value::as_str) == Some(response_id);
                }
                Some("response.interrupt.failed" | "error") => return false,
                Some(
                    "response.completed"
                    | "response.incomplete"
                    | "response.failed"
                    | "response.cancelled"
                    | "response.done",
                ) => ended = true,
                _ => {}
            }
        }
        true
    }

    async fn send_frame(&mut self, frame: Value) -> Result<(), ()> {
        self.socket
            .send(Message::Text(frame.to_string().into()))
            .await
            .map_err(|_| ())
    }
}

fn interrupted(started: bool, reason: String, events: Option<&EventSender>) -> Exchange {
    if !started {
        return Exchange::NotStarted(reason);
    }
    send(
        events,
        Err(LlmError::Request(format!(
            "Responses WebSocket closed before the response completed: {reason}"
        ))),
    );
    Exchange::Failed
}

fn send(events: Option<&EventSender>, event: Result<WebSocketEvent, LlmError>) -> bool {
    events.is_none_or(|events| events.send(event).is_ok())
}

fn unavailable(events: Option<&EventSender>, reason: &str) {
    send(events, Ok(WebSocketEvent::Unavailable(reason.to_string())));
}

async fn consumer_gone(events: Option<&EventSender>) {
    match events {
        Some(events) => events.closed().await,
        None => std::future::pending().await,
    }
}

async fn expired(deadline: Option<Deadline>) -> Duration {
    let Some(deadline) = deadline else {
        return std::future::pending().await;
    };
    tokio::time::sleep_until(deadline.at).await;
    deadline.limit
}

fn incremental_body(
    previous: &PreviousResponse,
    body: &Map<String, Value>,
) -> Result<Map<String, Value>, &'static str> {
    if previous.settings != settings(body) {
        return Err("settings_changed");
    }
    let input = input(body);
    if !input.starts_with(&previous.input) {
        return Err("input_diverged");
    }
    let mut outgoing = body.clone();
    outgoing.insert(
        "previous_response_id".to_string(),
        Value::String(previous.response_id.clone()),
    );
    outgoing.insert(
        "input".to_string(),
        Value::Array(input[previous.input.len()..].to_vec()),
    );
    Ok(outgoing)
}

fn continuation(
    response_id: String,
    body: &Map<String, Value>,
    output: Vec<Value>,
) -> Option<PreviousResponse> {
    let mut input = input(body);
    // Compare against the replayed form, which the next request sends.
    input.extend(replayed_output(output).ok()?);
    Some(PreviousResponse {
        response_id,
        settings: settings(body),
        input,
    })
}

fn settings(body: &Map<String, Value>) -> Map<String, Value> {
    let mut settings = body.clone();
    for field in PER_RESPONSE_FIELDS {
        settings.remove(field);
    }
    settings
}

fn input(body: &Map<String, Value>) -> Vec<Value> {
    body.get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn event_response_id(event: &Value) -> Option<String> {
    event
        .pointer("/response/id")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn is_restartable_error(event: &Value) -> bool {
    event.get("type").and_then(Value::as_str) == Some("error")
        && event
            .pointer("/error/code")
            .or_else(|| event.get("code"))
            .and_then(Value::as_str)
            .is_some_and(|code| {
                [PREVIOUS_RESPONSE_NOT_FOUND, CONNECTION_LIMIT_REACHED].contains(&code)
            })
}

/// Upgrade through `client`, so TLS and proxy settings match the HTTP transport.
async fn upgrade(
    client: &Client,
    target: &WebSocketTarget,
) -> Result<WebSocketStream<Upgraded>, UpgradeError> {
    let key = generate_key();
    let response = client
        .get(&target.url)
        .headers(target.headers.clone())
        .header("OpenAI-Beta", RESPONSES_WEBSOCKETS_BETA)
        .header(reqwest::header::CONNECTION, "Upgrade")
        .header(reqwest::header::UPGRADE, "websocket")
        .header(reqwest::header::SEC_WEBSOCKET_VERSION, "13")
        .header(reqwest::header::SEC_WEBSOCKET_KEY, &key)
        .send()
        .await
        .map_err(|error| UpgradeError::Transient(format!("WebSocket upgrade failed: {error}")))?;
    let status = response.status();
    if status != StatusCode::SWITCHING_PROTOCOLS {
        // Fallback reasons go into traces, so they carry only the status.
        let reason = format!("WebSocket upgrade returned {status}");
        return Err(match status {
            StatusCode::UNAUTHORIZED => {
                let detail = response.text().await.unwrap_or_default();
                UpgradeError::Unauthorized(LlmError::Request(format!("{status}: {detail}")))
            }
            StatusCode::NOT_FOUND | StatusCode::UPGRADE_REQUIRED => {
                UpgradeError::Unsupported(reason)
            }
            _ => UpgradeError::Transient(reason),
        });
    }
    let expected = derive_accept_key(key.as_bytes());
    if response
        .headers()
        .get(reqwest::header::SEC_WEBSOCKET_ACCEPT)
        != HeaderValue::from_str(&expected).ok().as_ref()
    {
        return Err(UpgradeError::Unsupported(
            "WebSocket upgrade returned an invalid accept key".to_string(),
        ));
    }
    let upgraded = response
        .upgrade()
        .await
        .map_err(|error| UpgradeError::Transient(format!("WebSocket upgrade failed: {error}")))?;
    Ok(WebSocketStream::from_raw_socket(upgraded, Role::Client, None).await)
}
