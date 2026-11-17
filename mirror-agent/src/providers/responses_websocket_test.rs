use assert_matches::assert_matches;
use futures_util::SinkExt;
use futures_util::StreamExt;
use pretty_assertions::assert_eq;
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio_tungstenite::accept_async;

use super::*;

fn object(value: Value) -> Map<String, Value> {
    let Value::Object(fields) = value else {
        panic!("test value is a JSON object");
    };
    fields
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": [{"type": "input_text", "text": text}]})
}

fn assistant_output(text: &str) -> Value {
    json!({
        "type": "message",
        "id": format!("msg_{text}"),
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

fn completed(response_id: &str, output: &Value) -> String {
    json!({
        "type": "response.completed",
        "response": {"id": response_id, "status": "completed", "output": [output]},
    })
    .to_string()
}

fn sent(incremental: bool, reason: &'static str) -> WebSocketEvent {
    WebSocketEvent::Sent(WebSocketDelivery {
        incremental,
        reason,
    })
}

fn target(address: std::net::SocketAddr) -> WebSocketTarget {
    WebSocketTarget {
        url: format!("http://{address}/v1/responses"),
        headers: HeaderMap::new(),
    }
}

async fn accept(listener: &TcpListener) -> WebSocketStream<TcpStream> {
    let (stream, _) = listener.accept().await.expect("connection");
    accept_async(stream).await.expect("upgrade")
}

async fn reject(listener: &TcpListener, status_line: &str) {
    let (mut stream, _) = listener.accept().await.expect("connection");
    let mut request = vec![0; 4096];
    let _ = stream.read(&mut request).await.expect("request");
    let response =
        format!("HTTP/1.1 {status_line}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    stream
        .write_all(response.as_bytes())
        .await
        .expect("response");
}

async fn read_request(socket: &mut WebSocketStream<TcpStream>) -> Value {
    loop {
        match socket
            .next()
            .await
            .expect("client frame")
            .expect("valid frame")
        {
            Message::Text(text) => return serde_json::from_str(text.as_str()).expect("JSON frame"),
            Message::Ping(_) | Message::Pong(_) => {}
            other => panic!("unexpected frame {other:?}"),
        }
    }
}

async fn reply(socket: &mut WebSocketStream<TcpStream>, event: String) {
    socket
        .send(Message::Text(event.into()))
        .await
        .expect("send");
}

async fn events(
    websocket: &ResponsesWebSocket,
    target: WebSocketTarget,
    body: Value,
) -> Vec<WebSocketEvent> {
    websocket
        .stream(target, object(body))
        .map(|event| event.expect("request succeeds"))
        .collect()
        .await
}

fn websocket() -> ResponsesWebSocket {
    ResponsesWebSocket::spawn(None).expect("test WebSocket client")
}

#[test]
fn incremental_body_sends_only_new_input() {
    let previous = PreviousResponse {
        response_id: "resp_1".to_string(),
        settings: object(json!({"model": "gpt-test"})),
        input: vec![user("one"), assistant_output("two")],
    };
    let body = object(json!({
        "model": "gpt-test",
        "max_output_tokens": 100,
        "input": [user("one"), assistant_output("two"), user("three")],
    }));

    assert_eq!(
        incremental_body(&previous, &body).map(Value::Object),
        Ok(json!({
            "model": "gpt-test",
            "max_output_tokens": 100,
            "previous_response_id": "resp_1",
            "input": [user("three")],
        }))
    );
}

#[test]
fn incremental_body_names_why_a_request_goes_in_full() {
    let previous = PreviousResponse {
        response_id: "resp_1".to_string(),
        settings: object(json!({"model": "gpt-test"})),
        input: vec![user("one")],
    };

    assert_eq!(
        incremental_body(
            &previous,
            &object(json!({"model": "other", "input": [user("one")]}))
        ),
        Err("settings_changed")
    );
    assert_eq!(
        incremental_body(
            &previous,
            &object(json!({"model": "gpt-test", "input": [user("edited")]}))
        ),
        Err("input_diverged")
    );
}

#[test]
fn restartable_errors_are_recognized_by_code() {
    assert!(is_restartable_error(&json!({
        "type": "error",
        "status": 400,
        "error": {"code": "previous_response_not_found", "message": "gone"},
    })));
    assert!(is_restartable_error(&json!({
        "type": "error",
        "error": {"code": "websocket_connection_limit_reached"},
    })));
    assert!(!is_restartable_error(&json!({
        "type": "error",
        "error": {"code": "invalid_request_error"},
    })));
}

#[test]
fn only_known_hosts_serve_websocket_mode() {
    assert!(ResponsesWebSocket::serves("https://api.openai.com/v1"));
    assert!(!ResponsesWebSocket::serves("http://api.openai.com/v1"));
    assert!(ResponsesWebSocket::serves(
        "https://chatgpt.com/backend-api/codex"
    ));
    assert!(!ResponsesWebSocket::serves("http://127.0.0.1:8000/v1"));
    assert!(!ResponsesWebSocket::serves("not a url"));
}

#[tokio::test]
async fn reused_connection_continues_with_previous_response_id() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let first_output = assistant_output("hello");
    let server_output = first_output.clone();
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        let first = read_request(&mut socket).await;
        reply(&mut socket, completed("resp_1", &server_output)).await;
        let second = read_request(&mut socket).await;
        reply(&mut socket, completed("resp_2", &assistant_output("bye"))).await;
        (first, second)
    });
    let websocket = websocket();
    let mut second_input = vec![user("hi")];
    second_input.extend(replayed_output(vec![first_output.clone()]).expect("replayable output"));
    second_input.push(user("again"));

    let first_events = events(
        &websocket,
        target(address),
        json!({"model": "gpt-test", "stream": true, "input": [user("hi")]}),
    )
    .await;
    let second_events = events(
        &websocket,
        target(address),
        json!({"model": "gpt-test", "stream": true, "input": second_input}),
    )
    .await;
    let (first, second) = server.await.expect("server");

    assert_eq!(
        first_events,
        vec![
            sent(false, "new_connection"),
            WebSocketEvent::Payload(completed("resp_1", &first_output)),
        ]
    );
    assert_eq!(second_events[0], sent(true, "continued"));
    assert_eq!(
        first,
        json!({"type": "response.create", "model": "gpt-test", "input": [user("hi")]})
    );
    assert_eq!(
        second,
        json!({
            "type": "response.create",
            "model": "gpt-test",
            "previous_response_id": "resp_1",
            "input": [user("again")],
        })
    );
}

#[tokio::test]
async fn missing_previous_response_retries_in_full_on_a_new_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let output = assistant_output("hello");
    let server_output = output.clone();
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        read_request(&mut socket).await;
        reply(&mut socket, completed("resp_1", &server_output)).await;
        read_request(&mut socket).await;
        let missing = json!({
            "type": "error",
            "status": 400,
            "error": {"code": "previous_response_not_found", "message": "gone"},
        });
        reply(&mut socket, missing.to_string()).await;
        let mut socket = accept(&listener).await;
        let retried = read_request(&mut socket).await;
        reply(&mut socket, completed("resp_2", &assistant_output("bye"))).await;
        retried
    });
    let websocket = websocket();
    events(
        &websocket,
        target(address),
        json!({"model": "gpt-test", "input": [user("hi")]}),
    )
    .await;
    let mut second_input = vec![user("hi")];
    second_input.extend(replayed_output(vec![output]).expect("replayable output"));
    second_input.push(user("again"));

    let second_events = events(
        &websocket,
        target(address),
        json!({"model": "gpt-test", "input": second_input.clone()}),
    )
    .await;
    let retried = server.await.expect("server");

    assert_eq!(
        retried,
        json!({"type": "response.create", "model": "gpt-test", "input": second_input})
    );
    assert_eq!(
        second_events,
        vec![
            sent(true, "continued"),
            sent(false, "new_connection"),
            WebSocketEvent::Payload(completed("resp_2", &assistant_output("bye"))),
        ]
    );
}

#[tokio::test]
async fn endpoint_without_websocket_mode_stays_on_http() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        reject(&listener, "426 Upgrade Required").await;
    });
    let websocket = websocket();

    let first = events(&websocket, target(address), json!({"input": []})).await;
    server.await.expect("server");
    let second = events(&websocket, target(address), json!({"input": []})).await;

    assert_matches!(first.as_slice(), [WebSocketEvent::Unavailable(reason)] if reason == "WebSocket upgrade returned 426 Upgrade Required");
    assert_eq!(
        second,
        vec![WebSocketEvent::Unavailable(
            "WebSocket mode is off for this endpoint".to_string()
        )]
    );
}

#[tokio::test]
async fn transient_upgrade_failure_uses_http_once_then_retries_the_websocket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        reject(&listener, "503 Service Unavailable").await;
        let mut socket = accept(&listener).await;
        read_request(&mut socket).await;
        reply(&mut socket, completed("resp_1", &assistant_output("hi"))).await;
    });
    let websocket = websocket();

    let first = events(&websocket, target(address), json!({"input": []})).await;
    let second = events(&websocket, target(address), json!({"input": []})).await;
    server.await.expect("server");

    assert_matches!(first.as_slice(), [WebSocketEvent::Unavailable(reason)] if reason == "WebSocket upgrade returned 503 Service Unavailable");
    assert_eq!(
        second,
        vec![
            sent(false, "new_connection"),
            WebSocketEvent::Payload(completed("resp_1", &assistant_output("hi"))),
        ]
    );
}

#[tokio::test]
async fn unauthorized_upgrade_is_reported_to_the_caller() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        reject(&listener, "401 Unauthorized").await;
    });
    let websocket = websocket();

    let events = websocket
        .stream(target(address), object(json!({"input": []})))
        .collect::<Vec<_>>()
        .await;

    assert_matches!(
        events.as_slice(),
        [Err(LlmError::Request(detail))] if detail.starts_with("401 Unauthorized:")
    );
}

#[tokio::test]
async fn abandoned_response_is_interrupted_and_the_connection_kept() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        read_request(&mut socket).await;
        let created = json!({"type": "response.created", "response": {"id": "resp_1"}});
        reply(&mut socket, created.to_string()).await;
        let interrupt = read_request(&mut socket).await;
        let accepted = json!({"type": "response.interrupt.accepted", "response_id": "resp_1"});
        reply(&mut socket, accepted.to_string()).await;
        let cancelled = json!({"type": "response.incomplete", "response": {"id": "resp_1"}});
        reply(&mut socket, cancelled.to_string()).await;
        let next = read_request(&mut socket).await;
        reply(&mut socket, completed("resp_2", &assistant_output("hi"))).await;
        (interrupt, next)
    });
    let websocket = websocket();
    let mut stream = websocket
        .stream(
            target(address),
            object(json!({"model": "gpt-test", "input": [user("hi")]})),
        )
        .boxed();
    stream
        .next()
        .await
        .expect("sent")
        .expect("request succeeds");
    stream
        .next()
        .await
        .expect("created")
        .expect("request succeeds");
    drop(stream);

    let next_events = events(
        &websocket,
        target(address),
        json!({"model": "gpt-test", "input": [user("hi")]}),
    )
    .await;
    let (interrupt, next) = server.await.expect("server");

    assert_eq!(
        interrupt,
        json!({"type": "response.interrupt", "response_id": "resp_1", "mode": "discard_partial_items"})
    );
    assert_eq!(next["input"], json!([user("hi")]));
    assert_eq!(next_events[0], sent(false, "no_previous_response"));
}

#[tokio::test]
async fn prewarm_loads_the_conversation_for_the_next_request() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        let warmup = read_request(&mut socket).await;
        let warmed =
            json!({"type": "response.completed", "response": {"id": "resp_warm", "output": []}});
        reply(&mut socket, warmed.to_string()).await;
        let request = read_request(&mut socket).await;
        reply(&mut socket, completed("resp_1", &assistant_output("hi"))).await;
        (warmup, request)
    });
    let websocket = websocket();

    websocket.prewarm(
        target(address),
        object(json!({"model": "gpt-test", "input": [user("system")]})),
    );
    let request_events = events(
        &websocket,
        target(address),
        json!({"model": "gpt-test", "max_output_tokens": 100, "input": [user("system"), user("hi")]}),
    )
    .await;
    let (warmup, request) = server.await.expect("server");

    assert_eq!(
        warmup,
        json!({"type": "response.create", "model": "gpt-test", "generate": false, "input": [user("system")]})
    );
    assert_eq!(
        request,
        json!({
            "type": "response.create",
            "model": "gpt-test",
            "max_output_tokens": 100,
            "previous_response_id": "resp_warm",
            "input": [user("hi")],
        })
    );
    assert_eq!(request_events[0], sent(true, "continued"));
}

#[tokio::test]
async fn background_responses_go_over_http() {
    let events = events(
        &websocket(),
        target("127.0.0.1:9".parse().expect("address")),
        json!({"background": true, "input": []}),
    )
    .await;

    assert_eq!(
        events,
        vec![WebSocketEvent::Unavailable(
            "background responses use HTTP".to_string()
        )]
    );
}

#[tokio::test]
async fn request_timeout_bounds_a_silent_response() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        read_request(&mut socket).await;
        let _ = socket.next().await;
    });
    let websocket =
        ResponsesWebSocket::spawn(Some(Duration::from_secs(60))).expect("test WebSocket client");
    let mut stream = websocket
        .stream(target(address), object(json!({"input": []})))
        .boxed();
    let sent = stream.next().await;
    tokio::time::pause();

    let rest = stream.collect::<Vec<_>>().await;
    server.abort();

    assert_matches!(sent, Some(Ok(WebSocketEvent::Sent(_))));
    assert_matches!(
        rest.as_slice(),
        [Err(LlmError::Request(detail))]
            if detail == "Responses WebSocket request timed out after 60s"
    );
}

#[tokio::test]
async fn request_past_its_deadline_once_connected_is_not_sent() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        let first = read_request(&mut socket).await;
        reply(&mut socket, completed("resp_1", &assistant_output("hi"))).await;
        first
    });
    let mut session = WebSocketSession {
        client: super::super::http_client(None).expect("test HTTP client"),
        connection: None,
        upgrade_failures: BTreeMap::new(),
    };

    session
        .deliver(
            target(address),
            object(json!({"model": "gpt-test", "generate": false, "input": [user("stale")]})),
            None,
            Some(Deadline::after(Duration::ZERO)),
        )
        .await;
    session
        .deliver(
            target(address),
            object(json!({"model": "gpt-test", "input": [user("hi")]})),
            None,
            None,
        )
        .await;
    let first = server.await.expect("server");

    assert_eq!(
        first,
        json!({"type": "response.create", "model": "gpt-test", "input": [user("hi")]})
    );
}

#[tokio::test]
async fn request_queued_behind_a_stalled_prewarm_times_out_on_its_own_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let (prewarm_read, prewarm_received) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        read_request(&mut socket).await;
        let _ = prewarm_read.send(());
        let _ = socket.next().await;
    });
    let websocket =
        ResponsesWebSocket::spawn(Some(Duration::from_secs(5))).expect("test WebSocket client");
    websocket.prewarm(
        target(address),
        object(json!({"model": "gpt-test", "input": [user("system")]})),
    );
    prewarm_received
        .await
        .expect("the prewarm reaches the server");
    tokio::time::pause();
    let started = Instant::now();

    let events = websocket
        .stream(
            target(address),
            object(json!({"model": "gpt-test", "input": [user("system"), user("hi")]})),
        )
        .collect::<Vec<_>>()
        .await;
    let waited = started.elapsed();
    server.abort();

    assert_matches!(
        events.as_slice(),
        [Err(LlmError::Request(detail))]
            if detail == "Responses WebSocket request timed out after 5s"
    );
    assert!(
        waited < PREWARM_DEADLINE,
        "waited {waited:?} behind the prewarm"
    );
}

#[tokio::test]
async fn timed_out_response_is_interrupted() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("listener");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let mut socket = accept(&listener).await;
        read_request(&mut socket).await;
        let created = json!({"type": "response.created", "response": {"id": "resp_1"}});
        reply(&mut socket, created.to_string()).await;
        read_request(&mut socket).await
    });
    let websocket =
        ResponsesWebSocket::spawn(Some(Duration::from_secs(60))).expect("test WebSocket client");
    let mut stream = websocket
        .stream(
            target(address),
            object(json!({"model": "gpt-test", "input": [user("hi")]})),
        )
        .boxed();
    stream
        .next()
        .await
        .expect("sent")
        .expect("request succeeds");
    stream
        .next()
        .await
        .expect("created")
        .expect("request succeeds");
    tokio::time::pause();

    let rest = stream.collect::<Vec<_>>().await;
    let interrupt = server.await.expect("server");

    assert_matches!(
        rest.as_slice(),
        [Err(LlmError::Request(detail))]
            if detail == "Responses WebSocket request timed out after 60s"
    );
    assert_eq!(
        interrupt,
        json!({"type": "response.interrupt", "response_id": "resp_1", "mode": "discard_partial_items"})
    );
}
