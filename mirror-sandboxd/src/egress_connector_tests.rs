use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use rustls::pki_types::PrivatePkcs8KeyDer;
use rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

use super::*;

fn proxy() -> (EgressProxyConfig, TlsAcceptor) {
    let key =
        rcgen::generate_simple_self_signed(vec!["localhost".into()]).expect("fixture certificate");
    let server = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(
            vec![key.cert.der().clone()],
            PrivatePkcs8KeyDer::from(key.key_pair.serialize_der()).into(),
        )
        .expect("TLS server");
    (
        EgressProxyConfig {
            host: "localhost".into(),
            port: 443,
            ca_pem: key.cert.pem(),
        },
        TlsAcceptor::from(Arc::new(server)),
    )
}

fn expiry(seconds: u64) -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs()
        + seconds
}

#[test]
fn only_bounded_https_connect_requests_are_accepted() {
    assert_eq!(
        connect_target(b"CONNECT api.github.com:443 HTTP/1.1\r\nHost: api.github.com:443\r\n\r\n"),
        Some("api.github.com:443")
    );
    assert_eq!(
        connect_target(
            b"CONNECT bcr.bazel.build:443 HTTP/1.1\r\nUser-Agent: Java/21\r\nHost: bcr.bazel.build\r\n\r\n"
        ),
        Some("bcr.bazel.build:443")
    );
    for request in [
        "GET http://example.com HTTP/1.1\r\n\r\n",
        "CONNECT example.com:80 HTTP/1.1\r\n\r\n",
        "CONNECT user@example.com:443 HTTP/1.1\r\n\r\n",
        "CONNECT example.com:443 HTTP/1.1\r\nHost: other.com:443\r\n\r\n",
        "CONNECT example.com:443 HTTP/1.1\r\nContent-Length: 9\r\n\r\n",
        "CONNECT example.com:443 HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n",
    ] {
        assert!(connect_target(request.as_bytes()).is_none(), "{request}");
    }
}

#[tokio::test]
async fn current_token_is_sent_only_inside_tls_and_never_in_child_environment() {
    let (mut config, acceptor) = proxy();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("proxy listener");
    config.port = u32::from(listener.local_addr().expect("proxy port").port());
    let server = tokio::spawn(async move {
        let mut seen = Vec::new();
        for _ in 0..2 {
            let (socket, _) = listener.accept().await.expect("connection");
            let mut tls = acceptor
                .accept(socket)
                .await
                .expect("protected outer transport");
            seen.push(
                String::from_utf8(read_header(&mut tls).await.expect("CONNECT")).expect("UTF-8"),
            );
            tls.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await
                .expect("connected");
            let mut bytes = [0; 4];
            tls.read_exact(&mut bytes)
                .await
                .expect("forwarded application bytes");
            assert_eq!(&bytes, b"ping");
            tls.write_all(b"pong").await.expect("reply");
        }
        seen
    });
    let truststore = tempfile::tempdir().expect("truststore directory");
    let connector = EgressConnector::start(config, &truststore.path().join("truststore.jks"))
        .await
        .expect("connector");
    assert!(truststore.path().join("truststore.jks").is_file());
    let address = connector.address;
    for token in ["mse4.first.signature", "mse4.second.signature"] {
        assert!(connector.install(expiry(30), token.as_bytes()));
        let mut command = Command::new("sh");
        connector.configure_command(&mut command);
        let env: Vec<_> = command
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert!(env
            .iter()
            .all(|(_, value)| !value.as_deref().unwrap_or_default().contains(token)));
        assert!(env.iter().any(|(key, value)| key == "HTTPS_PROXY"
            && value.as_deref() == Some(&format!("http://{address}"))));
        assert!(env.iter().any(|(key, value)| key == "GRPC_PROXY_EXP"
            && value.as_deref() == Some(&address.to_string())));
        let mut client = TcpStream::connect(address).await.expect("local proxy");
        client
            .write_all(
                b"CONNECT api.github.com:443 HTTP/1.1\r\nProxy-Authorization: attacker\r\n\r\n",
            )
            .await
            .expect("client CONNECT");
        assert!(read_header(&mut client)
            .await
            .expect("response")
            .starts_with(b"HTTP/1.1 200"));
        client.write_all(b"ping").await.expect("request bytes");
        let mut response = [0; 4];
        client
            .read_exact(&mut response)
            .await
            .expect("upstream bytes");
        assert_eq!(&response, b"pong");
    }
    let seen = server.await.expect("proxy server");
    assert!(seen[0].contains("Bearer mse4.first.signature"));
    assert!(seen[1].contains("Bearer mse4.second.signature"));
    assert!(seen.iter().all(|request| !request.contains("attacker")));
    drop(connector);
    tokio::time::timeout(Duration::from_secs(1), async {
        while TcpStream::connect(address).await.is_ok() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("aborted listener closes");
}

#[tokio::test]
async fn expired_unbounded_or_malformed_identity_cannot_be_installed() {
    let (config, _) = proxy();
    let truststore = tempfile::tempdir().expect("truststore directory");
    let connector = EgressConnector::start(config, &truststore.path().join("truststore.jks"))
        .await
        .expect("connector");
    assert!(!connector.install(1, b"mse4.token.signature"));
    assert!(!connector.install(expiry(120), b"mse4.token.signature"));
    assert!(!connector.install(expiry(30), b"token\r\ninjected"));
    assert!(connector.install(expiry(30), b"mse4.token.signature"));
}

#[tokio::test]
async fn untrusted_outer_tls_never_receives_a_bearer() {
    let (mut config, _) = proxy();
    let (_, wrong_acceptor) = proxy();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("listener");
    config.port = u32::from(listener.local_addr().expect("port").port());
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("connection");
        assert!(wrong_acceptor.accept(socket).await.is_err());
    });
    let truststore = tempfile::tempdir().expect("truststore directory");
    let connector = EgressConnector::start(config, &truststore.path().join("truststore.jks"))
        .await
        .expect("connector");
    assert!(connector.install(expiry(30), b"mse4.protected.signature"));
    let mut client = TcpStream::connect(connector.address).await.expect("client");
    client
        .write_all(b"CONNECT api.github.com:443 HTTP/1.1\r\n\r\n")
        .await
        .expect("CONNECT");
    assert!(read_header(&mut client)
        .await
        .expect("denial")
        .starts_with(b"HTTP/1.1 502"));
    server.await.expect("TLS fixture");
}

#[tokio::test]
async fn an_admitted_transfer_can_finish_after_its_original_identity_expires() {
    let (mut config, acceptor) = proxy();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("listener");
    config.port = u32::from(listener.local_addr().expect("port").port());
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("connection");
        let mut tls = acceptor.accept(socket).await.expect("TLS");
        read_header(&mut tls).await.expect("CONNECT");
        tls.write_all(b"HTTP/1.1 200 Connected\r\n\r\n")
            .await
            .expect("response");
        let mut request = [0; 4];
        tls.read_exact(&mut request)
            .await
            .expect("admitted request");
        assert_eq!(&request, b"ping");
        tokio::time::sleep(Duration::from_secs(3)).await;
        tls.write_all(b"pong")
            .await
            .expect("complete admitted transfer");
    });
    let truststore = tempfile::tempdir().expect("truststore directory");
    let connector = EgressConnector::start(config, &truststore.path().join("truststore.jks"))
        .await
        .expect("connector");
    let original_expiry = expiry(2);
    assert!(connector.install(original_expiry, b"mse4.original.signature"));
    let mut client = TcpStream::connect(connector.address).await.expect("client");
    client
        .write_all(b"CONNECT api.github.com:443 HTTP/1.1\r\n\r\n")
        .await
        .expect("CONNECT");
    assert!(read_header(&mut client)
        .await
        .expect("response")
        .starts_with(b"HTTP/1.1 200"));
    client
        .write_all(b"ping")
        .await
        .expect("request admitted before expiry");
    assert!(connector.install(expiry(60), b"mse4.replacement.signature"));
    let mut response = [0; 4];
    tokio::time::timeout(Duration::from_secs(5), client.read_exact(&mut response))
        .await
        .expect("transfer completes")
        .expect("response");
    assert_eq!(&response, b"pong");
    assert!(remaining(original_expiry).is_none());
    server.await.expect("upstream completed");
}

#[tokio::test]
async fn dropping_control_closes_an_established_tunnel() {
    let (mut config, acceptor) = proxy();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("listener");
    config.port = u32::from(listener.local_addr().expect("port").port());
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("connection");
        let mut tls = acceptor.accept(socket).await.expect("TLS");
        read_header(&mut tls).await.expect("CONNECT");
        tls.write_all(b"HTTP/1.1 200 Connected\r\n\r\n")
            .await
            .expect("response");
        let mut bytes = [0; 1];
        assert!(!matches!(tls.read(&mut bytes).await, Ok(1)));
    });
    let truststore = tempfile::tempdir().expect("truststore directory");
    let connector = EgressConnector::start(config, &truststore.path().join("truststore.jks"))
        .await
        .expect("connector");
    let ca = connector.ca.path().to_owned();
    assert!(connector.install(expiry(30), b"mse4.protected.signature"));
    let mut client = TcpStream::connect(connector.address).await.expect("client");
    client
        .write_all(b"CONNECT api.github.com:443 HTTP/1.1\r\n\r\n")
        .await
        .expect("CONNECT");
    assert!(read_header(&mut client)
        .await
        .expect("response")
        .starts_with(b"HTTP/1.1 200"));
    drop(connector);
    let mut byte = [0; 1];
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), client.read(&mut byte))
            .await
            .expect("closed")
            .expect("read"),
        0
    );
    assert!(!ca.exists());
    server.await.expect("upstream closed");
}

#[tokio::test]
async fn running_bash_tool_uses_automatic_refresh_without_environment_changes() {
    use crate::RunnerCommand;
    use mirror_sandbox::{
        egress_update, EgressEvent, EgressIdentity, EgressUpdate, ExecuteRequest,
    };
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    let (mut config, acceptor) = proxy();
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("listener");
    config.port = u32::from(listener.local_addr().expect("address").port());
    let (seen, mut requests) = mpsc::channel(2);
    let upstream = tokio::spawn(async move {
        for _ in 0..2 {
            let (socket, _) = listener.accept().await.expect("connection");
            let mut tls = acceptor.accept(socket).await.expect("outer TLS");
            let header = read_header(&mut tls).await.expect("CONNECT");
            seen.send(String::from_utf8(header).expect("header"))
                .await
                .expect("evidence");
            tls.write_all(b"HTTP/1.1 200 Connected\r\n\r\n")
                .await
                .expect("response");
            let mut bytes = [0; 4];
            tls.read_exact(&mut bytes)
                .await
                .expect("application request");
            assert_eq!(&bytes, b"ping");
            tls.write_all(b"pong").await.expect("application response");
        }
    });
    let directory = tempfile::tempdir().expect("evidence");
    let before = directory.path().join("before");
    let after = directory.path().join("after");
    let gate = directory.path().join("continue");
    let script = format!(
        r#"
set -eu
address=${{HTTPS_PROXY#http://}}
host=${{address%:*}}
port=${{address##*:}}
request() {{
  exec 3<>/dev/tcp/$host/$port
  printf 'CONNECT api.github.com:443 HTTP/1.1\r\n\r\n' >&3
  IFS= read -r status <&3
  [[ "$status" == $'HTTP/1.1 200 Connection Established\r' ]]
  IFS= read -r empty <&3
  printf ping >&3
  IFS= read -r -N4 reply <&3
  [[ "$reply" == pong ]]
  exec 3>&-
}}
printf '%s %s %s\n' "$$" "$HTTPS_PROXY" "$GH_TOKEN" > '{before}'
request
while [[ ! -e '{gate}' ]]; do sleep 0.01; done
request
printf '%s %s %s\n' "$$" "$HTTPS_PROXY" "$GH_TOKEN" > '{after}'
"#,
        before = before.display(),
        gate = gate.display(),
        after = after.display()
    );
    let (host, guest) = tokio::io::duplex(64 * 1024);
    let (input, output) = tokio::io::split(guest);
    let served = tokio::spawn(crate::serve(
        input,
        output,
        RunnerCommand {
            program: std::env::var_os("SANDBOXD_BIN")
                .expect("Bazel supplies the production worker")
                .into(),
            args: vec!["run-once".into()],
        },
    ));
    let (input, output) = tokio::io::split(host);
    let closing = watch::channel(false).0;
    let mut client = mirror_sandbox::connect(input, output, closing.clone())
        .await
        .expect("control stream");
    let (updates, receiver) = mpsc::channel(1);
    updates
        .send(EgressUpdate {
            update: Some(egress_update::Update::Configure(config)),
        })
        .await
        .expect("configure");
    let mut events = client
        .egress(ReceiverStream::new(receiver))
        .await
        .expect("egress RPC")
        .into_inner();
    assert_eq!(
        events.message().await.expect("request"),
        Some(EgressEvent {
            request_id: 1,
            installed: false
        })
    );
    let original_expiry = expiry(6);
    updates
        .send(EgressUpdate {
            update: Some(egress_update::Update::Identity(EgressIdentity {
                request_id: 1,
                expires_at: original_expiry,
                token: b"mse4.first.signature".to_vec(),
            })),
        })
        .await
        .expect("initial identity");
    assert_eq!(
        events.message().await.expect("ack"),
        Some(EgressEvent {
            request_id: 1,
            installed: true
        })
    );
    let mut execution = client
        .execute(ExecuteRequest {
            call_id: "running".into(),
            call: Some(mirror_protocol::ToolCall {
                r#type: "function".into(),
                function: Some(mirror_protocol::FunctionCall {
                    name: "bash".into(),
                    arguments: serde_json::json!({"command": script}).to_string(),
                }),
                ..Default::default()
            }),
            timeout_millis: 20_000,
            max_output_bytes: 1024,
        })
        .await
        .expect("execute")
        .into_inner();
    assert!(execution.message().await.expect("admission").is_some());
    assert!(
        tokio::time::timeout(Duration::from_secs(3), requests.recv())
            .await
            .expect("child request")
            .expect("first CONNECT")
            .contains("Bearer mse4.first.signature")
    );
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), events.message())
            .await
            .expect("automatic renewal")
            .expect("request"),
        Some(EgressEvent {
            request_id: 2,
            installed: false
        })
    );
    updates
        .send(EgressUpdate {
            update: Some(egress_update::Update::Identity(EgressIdentity {
                request_id: 2,
                expires_at: expiry(60),
                token: b"mse4.second.signature".to_vec(),
            })),
        })
        .await
        .expect("refreshed identity");
    assert_eq!(
        events.message().await.expect("ack"),
        Some(EgressEvent {
            request_id: 2,
            installed: true
        })
    );
    tokio::time::sleep(remaining(original_expiry).unwrap_or_default() + Duration::from_millis(20))
        .await;
    std::fs::write(&gate, "continue").expect("release same child after old identity expires");
    assert!(
        tokio::time::timeout(Duration::from_secs(3), requests.recv())
            .await
            .expect("child request")
            .expect("second CONNECT")
            .contains("Bearer mse4.second.signature")
    );
    let completed = tokio::time::timeout(Duration::from_secs(3), execution.message())
        .await
        .expect("completion")
        .expect("response")
        .expect("event");
    assert_matches!(
        completed.event,
        Some(mirror_sandbox::execute_event::Event::Completed(
            mirror_sandbox::ExecuteResponse {
                outcome: Some(mirror_sandbox::CallOutcome::Completed(_))
            }
        ))
    );
    let original = std::fs::read_to_string(before).expect("original environment");
    assert_eq!(
        std::fs::read_to_string(after).expect("same child"),
        original
    );
    assert!(original.contains("http://127.0.0.1:") && !original.contains("mse4."));
    drop((execution, events, updates, client));
    tokio::time::timeout(Duration::from_secs(10), served)
        .await
        .expect("bounded shutdown")
        .expect("guest task")
        .expect("shutdown");
    upstream.await.expect("TLS fixture");
}
