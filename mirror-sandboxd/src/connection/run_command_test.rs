use std::future::pending;
use std::io::Cursor;
use std::time::Duration;

use assert_matches::assert_matches;
use mirror_sandbox::{
    connect, run_command_request, run_command_response::Event, run_command_result::Outcome,
    run_sandbox_command, RunCommandRequest, RunCommandResponse, RunCommandResult, RunCommandStart,
    SandboxStream, SandboxTransportError, SandboxdStream,
};
use pretty_assertions::assert_eq;
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use crate::{serve, RunnerCommand, ServeError};

fn start() -> (DuplexStream, JoinHandle<Result<(), ServeError>>) {
    let (host, guest) = tokio::io::duplex(64 * 1024);
    let (input, output) = tokio::io::split(guest);
    let served = tokio::spawn(serve(
        input,
        output,
        RunnerCommand {
            program: "/bin/false".into(),
            args: Vec::new(),
        },
    ));
    (host, served)
}

fn command(script: &str) -> SandboxStream<'static> {
    SandboxStream {
        command: script.into(),
        workdir: None,
        timeout: Duration::from_secs(30),
        cancellation: Box::pin(pending()),
        stdin: None,
        stdout: Box::pin(tokio::io::sink()),
        stderr: None,
    }
}

fn start_request(id: &str, script: &str, timeout_millis: u64) -> RunCommandRequest {
    RunCommandRequest {
        input: Some(run_command_request::Input::Start(RunCommandStart {
            call_id: id.into(),
            command: script.into(),
            workdir: None,
            timeout_millis,
        })),
    }
}

async fn stopped(served: JoinHandle<Result<(), ServeError>>) {
    assert_matches!(
        tokio::time::timeout(Duration::from_secs(20), served).await,
        Ok(Ok(Ok(())))
    );
}

#[tokio::test]
async fn run_sandbox_command_preserves_binary_pipes_and_exit_status() {
    let (host, served) = start();
    let (output, mut stdout) = tokio::io::duplex(1024);
    let (errors, mut stderr) = tokio::io::duplex(1024);
    let bytes = b"\0\xffraw\r\n".repeat(20_000);
    let mut request = command("cat; printf '\\000\\377error\\r\\n' >&2; exit 17");
    request.stdin = Some(Box::pin(Cursor::new(bytes.clone())));
    request.stdout = Box::pin(output);
    request.stderr = Some(Box::pin(errors));
    let (mut captured, mut diagnostics) = (Vec::new(), Vec::new());

    let (result, read, read_errors) = tokio::join!(
        run_sandbox_command(async { Ok(SandboxdStream::split(host)) }, request),
        stdout.read_to_end(&mut captured),
        stderr.read_to_end(&mut diagnostics),
    );

    read.expect("stdout");
    read_errors.expect("stderr");
    assert_matches!(result, Ok(result) if result.exit_code == 17 && result.stderr.is_empty());
    assert_eq!(captured, bytes);
    assert_eq!(diagnostics, b"\0\xfferror\r\n");
    stopped(served).await;
}

#[tokio::test]
async fn run_sandbox_command_exits_without_waiting_for_stdin_eof() {
    let (host, served) = start();
    let (_writer, stdin) = tokio::io::duplex(64);
    let mut request = command("exit 4");
    request.stdin = Some(Box::pin(stdin));

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        run_sandbox_command(async { Ok(SandboxdStream::split(host)) }, request),
    )
    .await
    .expect("the guest exit must not wait for stdin");

    assert_matches!(result, Ok(result) if result.exit_code == 4);
    stopped(served).await;
}

#[tokio::test]
async fn run_sandbox_command_confirms_cancellation_with_unread_stdin() {
    let (host, served) = start();
    let (output, mut stdout) = tokio::io::duplex(64);
    let (cancel, cancelled) = oneshot::channel();
    let mut request = command("printf ready; exec sleep 30");
    request.stdout = Box::pin(output);
    let (mut writer, stdin) = tokio::io::duplex(64 * 1024);
    request.stdin = Some(Box::pin(stdin));
    request.cancellation = Box::pin(async { cancelled.await.expect("cancel request") });
    let running = tokio::spawn(run_sandbox_command(
        async { Ok(SandboxdStream::split(host)) },
        request,
    ));
    stdout.read_exact(&mut [0; 5]).await.expect("guest started");

    assert!(tokio::time::timeout(
        Duration::from_millis(200),
        writer.write_all(&vec![0; 8 * 1024 * 1024]),
    )
    .await
    .is_err());
    cancel.send(()).expect("cancel");
    let result = tokio::time::timeout(Duration::from_secs(20), running)
        .await
        .expect("bounded cancellation")
        .expect("command task");

    assert_matches!(result, Err(SandboxTransportError::Failed(detail)) if detail == "command cancelled; its process group stopped");
    stopped(served).await;
}

#[tokio::test]
async fn run_sandbox_command_reports_unknown_when_confirmation_is_lost() {
    let (mut upstream, served) = start();
    let (host, mut bridge) = tokio::io::duplex(64 * 1024);
    let relay =
        tokio::spawn(
            async move { tokio::io::copy_bidirectional(&mut upstream, &mut bridge).await },
        );
    let (output, mut stdout) = tokio::io::duplex(64);
    let mut request = command("printf ready; exec sleep 30");
    request.stdout = Box::pin(output);
    let running = tokio::spawn(run_sandbox_command(
        async { Ok(SandboxdStream::split(host)) },
        request,
    ));
    stdout.read_exact(&mut [0; 5]).await.expect("guest started");

    relay.abort();
    let _ = relay.await;
    let result = tokio::time::timeout(Duration::from_secs(20), running)
        .await
        .expect("bounded transport failure")
        .expect("command task");

    assert_matches!(result, Err(SandboxTransportError::OutcomeUnknown(_)));
    stopped(served).await;
}

#[tokio::test]
async fn run_command_timeout_keeps_another_admitted_command_running() {
    let (host, served) = start();
    let (reader, writer) = tokio::io::split(host);
    let mut client = connect(reader, writer, watch::channel(false).0)
        .await
        .expect("connect");
    let (input, receiver) = mpsc::channel(2);
    input
        .send(start_request("survivor", "printf ready; cat", 30_000))
        .await
        .expect("start survivor");
    let mut survivor = client
        .run_command(ReceiverStream::new(receiver))
        .await
        .expect("survivor RPC")
        .into_inner();
    assert_matches!(
        survivor.message().await,
        Ok(Some(RunCommandResponse {
            event: Some(Event::Accepted(true))
        }))
    );
    assert_matches!(survivor.message().await, Ok(Some(RunCommandResponse { event: Some(Event::Stdout(bytes)) })) if bytes == b"ready");

    let mut timed_out = client
        .run_command(tokio_stream::iter([start_request(
            "timeout",
            "exec sleep 30",
            100,
        )]))
        .await
        .expect("timeout RPC")
        .into_inner();
    assert_matches!(
        timed_out.message().await,
        Ok(Some(RunCommandResponse {
            event: Some(Event::Accepted(true))
        }))
    );
    assert_matches!(
        timed_out.message().await,
        Ok(Some(RunCommandResponse {
            event: Some(Event::Completed(RunCommandResult {
                outcome: Some(Outcome::DeadlineExceeded(true))
            }))
        }))
    );

    input
        .send(RunCommandRequest {
            input: Some(run_command_request::Input::Stdin(b"still running".to_vec())),
        })
        .await
        .expect("survivor stdin");
    drop(input);
    assert_matches!(survivor.message().await, Ok(Some(RunCommandResponse { event: Some(Event::Stdout(bytes)) })) if bytes == b"still running");
    assert_matches!(
        survivor.message().await,
        Ok(Some(RunCommandResponse {
            event: Some(Event::Completed(RunCommandResult {
                outcome: Some(Outcome::ExitCode(0))
            }))
        }))
    );
    drop((timed_out, survivor, client));
    stopped(served).await;
}

#[tokio::test]
async fn serve_gives_commands_the_workspace_from_its_argument() {
    let binary = std::fs::canonicalize(std::env::var("SANDBOXD_BIN").expect("guest binary"))
        .expect("resolve guest binary");
    let workspace = tempfile::tempdir().expect("workspace");
    let mut sandboxd = tokio::process::Command::new(binary)
        .arg("serve")
        .arg(workspace.path())
        .env_remove("MIRROR_WORKSPACE")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("start sandboxd");
    let stream = SandboxdStream {
        reader: Box::new(sandboxd.stdout.take().expect("sandboxd stdout")),
        writer: Box::new(sandboxd.stdin.take().expect("sandboxd stdin")),
    };
    let (output, mut stdout) = tokio::io::duplex(4096);
    let mut request = command(r#"printf %s "$MIRROR_WORKSPACE""#);
    request.stdout = Box::pin(output);
    let mut captured = Vec::new();

    let (result, read) = tokio::join!(
        run_sandbox_command(async { Ok(stream) }, request),
        stdout.read_to_end(&mut captured),
    );

    read.expect("stdout");
    assert_matches!(result, Ok(result) if result.exit_code == 0);
    assert_eq!(captured, workspace.path().as_os_str().as_encoded_bytes());
}
