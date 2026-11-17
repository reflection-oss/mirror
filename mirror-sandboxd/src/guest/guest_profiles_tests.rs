use std::io::Write;
use std::time::Duration;

use assert_matches::assert_matches;
use mirror_protocol::{chat_record, ChatRecord, FunctionCall, ToolCall, ToolMessage};
use pretty_assertions::assert_eq;
use prost::Message;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;

use super::*;
use crate::{serve, RunnerCommand, ServeError};
use mirror_sandbox::sandboxd_client::SandboxdClient;
use mirror_sandbox::{
    connect, egress_update, execute_event, guest_ready, CallOutcome, EgressEvent,
    EgressProxyConfig, EgressUpdate, ExecuteEvent, ExecuteRequest, SandboxStream,
    SandboxStreamOutcome, SandboxTransportError, StaticEndpoint, StaticRoute,
};

type Client = SandboxdClient<Channel>;

async fn start(script: &str) -> (Client, JoinHandle<Result<(), ServeError>>) {
    let (host, guest) = tokio::io::duplex(64 * 1024);
    let (input, output) = tokio::io::split(guest);
    let served = tokio::spawn(serve(
        input,
        output,
        RunnerCommand {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
        },
    ));
    let (input, output) = tokio::io::split(host);
    let client = connect(input, output, watch::channel(false).0)
        .await
        .expect("gRPC connection");
    (client, served)
}

async fn configure(
    client: &mut Client,
    local_proxy_port: u16,
    profiles: Vec<GuestProfile>,
) -> (
    mpsc::Sender<EgressUpdate>,
    tonic::Streaming<EgressEvent>,
    guest_ready::Outcome,
) {
    let certificate = rcgen::generate_simple_self_signed(vec!["localhost".into()]).expect("CA");
    let (updates, receiver) = mpsc::channel(1);
    updates
        .send(EgressUpdate {
            update: Some(egress_update::Update::Configure(EgressProxyConfig {
                host: "localhost".into(),
                port: 443,
                ca_pem: certificate.cert.pem(),
                profiles,
                local_proxy_port: u32::from(local_proxy_port),
            })),
        })
        .await
        .expect("configure");
    let mut events = client
        .egress(ReceiverStream::new(receiver))
        .await
        .expect("egress RPC")
        .into_inner();
    let report = tokio::time::timeout(Duration::from_secs(20), events.message())
        .await
        .expect("bounded ready phase")
        .expect("ready report")
        .expect("event");
    let outcome = report
        .ready
        .and_then(|ready| ready.outcome)
        .expect("ready outcome");
    (updates, events, outcome)
}

async fn execute(client: &mut Client) -> CallOutcome {
    let mut response = client
        .execute(ExecuteRequest {
            call_id: "call".into(),
            timeout_millis: 10_000,
            max_output_bytes: 1024 * 1024,
            call: Some(ToolCall {
                r#type: "function".into(),
                function: Some(FunctionCall {
                    name: "call".into(),
                    arguments: "{}".into(),
                }),
                ..ToolCall::default()
            }),
        })
        .await
        .expect("execute RPC")
        .into_inner();
    let first = next_event(&mut response).await;
    let completed = match first {
        Some(execute_event::Event::Accepted(true)) => next_event(&mut response).await,
        other => other,
    };
    match completed {
        Some(execute_event::Event::Completed(completed)) => completed.outcome.expect("outcome"),
        other => panic!("expected a completed call, got {other:?}"),
    }
}

async fn command(
    client: &Client,
    script: &str,
) -> Result<SandboxStreamOutcome, SandboxTransportError> {
    mirror_sandbox::run_command(
        async { Ok(client.clone()) },
        SandboxStream {
            command: script.into(),
            workdir: None,
            timeout: Duration::from_secs(10),
            cancellation: Box::pin(std::future::pending()),
            stdin: None,
            stdout: Box::pin(tokio::io::sink()),
            stderr: None,
        },
        tonic::metadata::MetadataMap::new(),
        Duration::from_secs(5),
    )
    .await
}

async fn next_event(response: &mut tonic::Streaming<ExecuteEvent>) -> Option<execute_event::Event> {
    tokio::time::timeout(Duration::from_secs(20), response.message())
        .await
        .expect("bounded execution")
        .expect("execute stream")
        .expect("event")
        .event
}

async fn stopped(served: JoinHandle<Result<(), ServeError>>) {
    assert_matches!(
        tokio::time::timeout(Duration::from_secs(20), served).await,
        Ok(Ok(Ok(())))
    );
}

fn record(content: &str) -> ChatRecord {
    ChatRecord {
        message: Some(chat_record::Message::Tool(ToolMessage {
            content: content.into(),
            ..ToolMessage::default()
        })),
        success: true,
        ..ChatRecord::default()
    }
}

fn record_file(content: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().expect("record file");
    file.write_all(&record(content).encode_to_vec())
        .expect("write record");
    file
}

fn variables(name: &str, value: &str) -> GuestProfile {
    GuestProfile {
        name: "guest".into(),
        environment: HashMap::from([(name.into(), value.into())]),
        ..GuestProfile::default()
    }
}

fn file(path: &str, contents: &str) -> GuestFile {
    GuestFile {
        path: path.into(),
        mode: 0o644,
        contents: contents.as_bytes().to_vec(),
    }
}

fn files(files: Vec<GuestFile>) -> GuestProfile {
    GuestProfile {
        name: "guest".into(),
        files,
        ..GuestProfile::default()
    }
}

fn endpoint(port: u16) -> StaticEndpoint {
    StaticEndpoint {
        listen: format!("127.0.0.1:{port}"),
        routes: vec![StaticRoute {
            path: "/".into(),
            required_headers: HashMap::new(),
            status: 200,
            response_headers: HashMap::new(),
            body: b"ok".to_vec(),
        }],
        response_headers: HashMap::new(),
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("probe port")
        .local_addr()
        .expect("probe address")
        .port()
}

async fn answers(port: u16) -> bool {
    let _ = rustls::crypto::ring::default_provider().install_default();
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("HTTP client")
        .get(format!("http://127.0.0.1:{port}/"))
        .send()
        .await
        .is_ok_and(|response| response.status() == reqwest::StatusCode::OK)
}

#[tokio::test]
async fn ready_profiles_reach_the_child_environment() {
    let directory = tempfile::tempdir().expect("guest files");
    let written = directory.path().join("home/.config/guest.txt");
    let result = record_file("profiles applied");
    let (mut client, served) = start(&format!(
        "set -e; cat >/dev/null; test \"$GUEST_SETTING\" = from-profile; \
         test \"$(cat {written})\" = written; cat {result}",
        written = written.display(),
        result = result.path().display()
    ))
    .await;
    let (updates, events, outcome) = configure(
        &mut client,
        free_port(),
        vec![GuestProfile {
            name: "guest".into(),
            environment: HashMap::from([("GUEST_SETTING".into(), "from-profile".into())]),
            files: vec![file(&written.display().to_string(), "written")],
            endpoints: Vec::new(),
        }],
    )
    .await;
    assert_eq!(outcome, guest_ready::Outcome::Ready(true));
    assert_eq!(
        execute(&mut client).await,
        CallOutcome::Completed(Box::new(record("profiles applied")))
    );
    assert_matches!(
        command(&client, &format!(
            "set -eu; test \"$GUEST_SETTING\" = from-profile; \
             test \"$(cat {})\" = written; test \"$NO_PROXY\" = 127.0.0.1,localhost; \
             case \"$HTTPS_PROXY\" in http://127.0.0.1:*) ;; *) exit 1 ;; esac",
            written.display(),
        )).await,
        Ok(result) if result.exit_code == 0
    );
    drop((updates, events, client));
    stopped(served).await;
}

#[tokio::test]
async fn a_port_in_use_fails_ready_and_rejects_execute_with_the_reason() {
    let occupied = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("occupied port");
    let port = occupied.local_addr().expect("address").port();
    let (mut client, served) = start("cat >/dev/null").await;
    let (updates, events, outcome) = configure(
        &mut client,
        free_port(),
        vec![GuestProfile {
            name: "guest".into(),
            endpoints: vec![endpoint(port)],
            ..GuestProfile::default()
        }],
    )
    .await;
    let reason = format!("endpoint 127.0.0.1:{port} could not listen");
    assert_matches!(
        outcome,
        guest_ready::Outcome::Failed(failed) if failed.starts_with(&reason)
    );
    assert_matches!(
        execute(&mut client).await,
        CallOutcome::Rejected(rejected)
            if rejected.starts_with(&format!("sandbox guest is not ready: {reason}"))
    );
    assert_matches!(
        command(&client, "exit 0").await,
        Err(SandboxTransportError::Failed(rejected))
            if rejected.starts_with(&format!("sandbox guest is not ready: {reason}"))
    );
    drop((updates, events, client, occupied));
    stopped(served).await;
}

#[tokio::test]
async fn a_reconnect_reapplies_the_profiles_on_the_same_ports() {
    let directory = tempfile::tempdir().expect("guest files");
    let written = directory.path().join("guest.txt");
    let proxy_port = free_port();
    let endpoint_port = free_port();
    let result = record_file("same proxy");
    let script = format!(
        "set -e; cat >/dev/null; test \"$HTTPS_PROXY\" = http://127.0.0.1:{proxy_port}; \
         test \"$GRPC_PROXY_EXP\" = 127.0.0.1:{proxy_port}; \
         case \"$JAVA_TOOL_OPTIONS\" in *-Dhttps.proxyPort={proxy_port}\\ *) ;; *) exit 1 ;; esac; \
         cat {result}",
        result = result.path().display()
    );
    let profiles = vec![GuestProfile {
        name: "guest".into(),
        files: vec![file(&written.display().to_string(), "written")],
        endpoints: vec![endpoint(endpoint_port)],
        ..GuestProfile::default()
    }];
    let (mut first, served) = start(&script).await;
    let (updates, events, outcome) = configure(&mut first, proxy_port, profiles.clone()).await;
    assert_eq!(outcome, guest_ready::Outcome::Ready(true));
    assert_eq!(
        execute(&mut first).await,
        CallOutcome::Completed(Box::new(record("same proxy")))
    );
    assert!(answers(endpoint_port).await);
    std::fs::remove_file(&written).expect("remove written file");
    drop((updates, events, first));
    stopped(served).await;
    let (mut second, served) = start(&script).await;
    let (updates, events, outcome) = configure(&mut second, proxy_port, profiles).await;
    assert_eq!(outcome, guest_ready::Outcome::Ready(true));
    assert_eq!(
        std::fs::read_to_string(&written).expect("rewritten file"),
        "written"
    );
    assert!(answers(endpoint_port).await);
    assert_eq!(
        execute(&mut second).await,
        CallOutcome::Completed(Box::new(record("same proxy")))
    );
    assert!(tokio::net::TcpStream::connect(("127.0.0.1", proxy_port))
        .await
        .is_ok());
    drop((updates, events, second));
    stopped(served).await;
}

#[tokio::test]
async fn a_symlinked_directory_fails_ready_without_writing_through_it() {
    let directory = tempfile::tempdir().expect("guest files");
    let elsewhere = directory.path().join("elsewhere");
    std::fs::create_dir(&elsewhere).expect("link target");
    let link = directory.path().join("config");
    std::os::unix::fs::symlink(&elsewhere, &link).expect("planted link");
    let path = link.join("guest.txt").display().to_string();
    let (mut client, served) = start("cat >/dev/null").await;
    let (updates, events, outcome) = configure(
        &mut client,
        free_port(),
        vec![files(vec![file(&path, "written")])],
    )
    .await;
    assert_eq!(
        outcome,
        guest_ready::Outcome::Failed(format!("could not write {path}: config is a symbolic link"))
    );
    assert!(!elsewhere.join("guest.txt").exists());
    drop((updates, events, client));
    stopped(served).await;
}

#[tokio::test]
async fn a_symlinked_file_fails_ready_without_writing_through_it() {
    let directory = tempfile::tempdir().expect("guest files");
    let target = directory.path().join("target.txt");
    std::fs::write(&target, "original").expect("link target");
    let link = directory.path().join("guest.txt");
    std::os::unix::fs::symlink(&target, &link).expect("planted link");
    let path = link.display().to_string();
    let (mut client, served) = start("cat >/dev/null").await;
    let (updates, events, outcome) = configure(
        &mut client,
        free_port(),
        vec![files(vec![file(&path, "written")])],
    )
    .await;
    assert_eq!(
        outcome,
        guest_ready::Outcome::Failed(format!(
            "could not write {path}: guest.txt is a symbolic link"
        ))
    );
    assert_eq!(
        std::fs::read_to_string(&target).expect("link target"),
        "original"
    );
    drop((updates, events, client));
    stopped(served).await;
}

#[test]
fn paths_that_name_one_file_are_refused_as_duplicates() {
    let refused = GuestProfiles::validate(vec![files(vec![
        file("/tmp/guest/settings", "first"),
        file("/tmp//guest/./settings", "second"),
    ])]);
    assert_eq!(
        refused.err(),
        Some("guest profile \"guest\": file \"/tmp//guest/./settings\" is written twice".into())
    );
}

#[test]
fn relative_file_paths_are_refused() {
    let refused = GuestProfiles::validate(vec![files(vec![file("home/.toolrc", "settings")])]);
    assert_eq!(
        refused.err(),
        Some(
            "guest profile \"guest\": file \"home/.toolrc\" needs an absolute path and a mode"
                .into()
        )
    );
}

#[test]
fn duplicate_variables_across_profiles_are_refused() {
    let refused = GuestProfiles::validate(vec![
        GuestProfile {
            name: "first".into(),
            ..variables("SERVICE_TOKEN", "placeholder")
        },
        GuestProfile {
            name: "second".into(),
            ..variables("SERVICE_TOKEN", "placeholder")
        },
    ]);
    assert_eq!(
        refused.err(),
        Some(
            "guest profile \"second\": variable \"SERVICE_TOKEN\" is set by more than one profile"
                .into()
        )
    );
}

#[test]
fn variables_owned_by_sandboxd_are_refused() {
    assert_eq!(
        GuestProfiles::validate(vec![variables("NO_PROXY", "")]).err(),
        Some("guest profile \"guest\": variable \"NO_PROXY\" is owned by sandboxd".into())
    );
    assert_eq!(
        GuestProfiles::validate(vec![variables("MIRROR_WORKSPACE", "/elsewhere")]).err(),
        Some("guest profile \"guest\": variable \"MIRROR_WORKSPACE\" is owned by sandboxd".into())
    );
}
