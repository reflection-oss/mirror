use std::sync::Arc;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;
use crate::tools::web_test_server::{TestResponse, TestServer};
use crate::tools::{AnyTool, BashTool, DirectToolExecutor, RemoteToolExecutor, ToolExecutor};
use crate::ExecutionContext;

#[tokio::test]
async fn forwards_raw_commands_without_credentials_or_output_changes() {
    let output = format!("BEGIN{}END", "界".repeat(40_000));
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        json!({
            "output": output, "exit_code": 7, "alerts": ["challenge_is_down"],
        }),
    )])
    .await;
    let url = Url::parse(&format!("{}/exec?instance=one&timeout=1", server.base_url)).expect("URL");
    let remote = RemoteToolExecutor::new(Vec::new(), url.clone()).expect("executor");
    let result = remote
        .execute_command("true".into(), Duration::from_secs(630))
        .await
        .expect("result");
    assert_eq!(result.output, output);
    assert_eq!(result.exit_code, 7);
    assert_eq!(result.alerts, Some(vec![json!("challenge_is_down")]));
    let request = server.request().await;
    assert_eq!(request.path, "/exec?instance=one&timeout=630");
    assert_eq!(request.body, json!(true));
    assert_eq!(request.authorization, None);
}

#[tokio::test]
async fn transport_errors_never_retry_or_echo_response_bodies() {
    for status in [302, 429, 500] {
        let mut server =
            TestServer::start(vec![TestResponse::text(status, "synthetic-secret")]).await;
        let url = Url::parse(&server.base_url).expect("URL");
        let remote = RemoteToolExecutor::new(Vec::new(), url.clone()).expect("executor");
        let error = remote
            .execute_command("true".into(), Duration::from_secs(30))
            .await
            .expect_err("failure")
            .to_string();
        assert!(error.contains(&format!("HTTP {status}")));
        assert!(error.contains("not retried"));
        assert!(!error.contains("synthetic-secret"));
        let _ = server.request().await;
    }
}

#[tokio::test]
async fn rejects_missing_exit_code() {
    let mut server =
        TestServer::start(vec![TestResponse::json(200, json!({"output": "done"}))]).await;
    let url = Url::parse(&server.base_url).expect("URL");
    let remote = RemoteToolExecutor::new(Vec::new(), url.clone()).expect("executor");
    let error = remote
        .execute_command("true".into(), Duration::from_secs(30))
        .await
        .expect_err("invalid response")
        .to_string();
    assert!(error.contains("invalid command-result JSON"));
    let _ = server.request().await;
}

#[test]
fn rejects_credentials_and_non_http_urls() {
    for url in [
        "file:///exec",
        "https://user:synthetic@example.test/exec",
        "http://localhost/exec#fragment",
    ] {
        assert_matches!(
            RemoteToolExecutor::new(Vec::new(), Url::parse(url).expect("URL")),
            Err(ToolError::InvalidArguments(_))
        );
    }
}

#[test]
fn executor_rejects_unsupported_tools_without_contacting_endpoint() {
    let workspace = tempfile::tempdir().expect("workspace");
    let write: Arc<dyn AnyTool> = Arc::new(crate::tools::web_test_server::LocalOnlyTool(
        workspace.path().join("file"),
    ));
    assert!(!write.supports_remote());
    let url = Url::parse("http://localhost:1/exec").expect("URL");
    let error =
        RemoteToolExecutor::new(vec![write.clone()], url.clone()).expect_err("unsupported tool");
    assert_matches!(error, ToolError::InvalidArguments(_));
    assert!(error
        .to_string()
        .contains("do not support remote execution"));
    let remote = RemoteToolExecutor::new(Vec::new(), url).expect("executor");
    assert_matches!(
        remote.with_tools(vec![write]),
        Err(ToolError::InvalidArguments(_))
    );
}

#[tokio::test]
async fn unsupported_remote_calls_never_fall_back_to_host_writes() {
    let workspace = tempfile::tempdir().expect("workspace");
    std::fs::write(workspace.path().join("file"), "host-only contents").expect("fixture");
    let write = crate::tools::web_test_server::LocalOnlyTool(workspace.path().join("file"));
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        Url::parse("http://localhost:1/exec").expect("URL"),
    )
    .expect("remote");
    let result = write
        .execute_remote(
            r#"{"file_path":"file","old_string":"host-only contents","new_string":"replacement"}"#,
            &ExecutionContext::default(),
            &remote,
        )
        .await;
    assert!(!result.success);
    assert!(result.content.contains("no remote implementation"));
    assert!(!result.content.contains("host-only contents"));
    assert_eq!(
        std::fs::read_to_string(workspace.path().join("file")).expect("host file"),
        "host-only contents"
    );
}

#[tokio::test]
async fn local_and_remote_dispatch_have_identical_argument_and_unknown_tool_errors() {
    let workspace = tempfile::tempdir().expect("workspace");
    let bash = Arc::new(BashTool::new(workspace.path()).expect("bash"));
    let local = DirectToolExecutor::new(vec![bash.clone()]);
    let remote = RemoteToolExecutor::new(
        vec![bash],
        Url::parse("http://localhost:1/exec").expect("URL"),
    )
    .expect("remote executor");
    for (name, arguments) in [
        ("bash", r#"{"command":"true","detached":true}"#),
        ("bash", ""),
        ("unknown", "{}"),
    ] {
        let local_result = local
            .call_tool(name.into(), arguments.into(), ExecutionContext::default())
            .await
            .expect("local result");
        let remote_result = remote
            .call_tool(name.into(), arguments.into(), ExecutionContext::default())
            .await
            .expect("remote result");
        assert!(!remote_result.success);
        assert_eq!(remote_result, local_result);
    }
}

#[tokio::test]
async fn transport_errors_do_not_expose_url_query_credentials() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let url = Url::parse(&format!(
        "http://{}/exec?token=synthetic-endpoint-secret",
        listener.local_addr().expect("address")
    ))
    .expect("URL");
    drop(listener);
    let remote = RemoteToolExecutor::new(Vec::new(), url.clone()).expect("executor");
    let error = remote
        .execute_command("true".into(), Duration::from_secs(30))
        .await
        .expect_err("closed endpoint")
        .to_string();
    assert!(error.contains("not retried"));
    assert!(!error.contains("synthetic-endpoint-secret"));
}

#[tokio::test]
async fn rejected_remote_call_never_contacts_the_endpoint() {
    let executor = RemoteToolExecutor::new(
        vec![Arc::new(BashTool::new(".").expect("bash"))],
        Url::parse("http://localhost:1/exec").expect("URL"),
    )
    .expect("remote executor");
    let call = mirror_protocol::ToolCall {
        r#type: "function".into(),
        function: Some(mirror_protocol::FunctionCall {
            name: "bash".into(),
            arguments: r#"{"command":"true"}"#.into(),
        }),
        ..Default::default()
    };
    let outputs = executor
        .execute(
            uuid::Uuid::new_v4(),
            &[call],
            &[false],
            &ExecutionContext::default(),
        )
        .await
        .expect("batch");
    assert_eq!(
        outputs,
        vec![crate::tools::ToolOutput::error(
            "Tool call rejected by the user."
        )]
    );
}

#[tokio::test]
async fn oversized_response_fails_explicitly_without_replaying_the_command() {
    let mut server = TestServer::start(vec![TestResponse::text(
        200,
        "x".repeat(2 * 1024 * 1024 + 1),
    )])
    .await;
    let url = Url::parse(&server.base_url).expect("URL");
    let error = RemoteToolExecutor::new(Vec::new(), url.clone())
        .expect("executor")
        .execute_command("true".into(), Duration::from_secs(30))
        .await
        .expect_err("bounded response");
    assert!(error.to_string().contains("exceeds 2 MiB"));
    assert!(error.to_string().contains("not retried"));
    assert_eq!(server.request().await.authorization, None);
}

#[test]
fn transport_debug_does_not_expose_endpoint_query_secrets() {
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        Url::parse("https://example.test/exec?token=synthetic-secret").expect("URL"),
    )
    .expect("remote");
    assert!(!format!("{remote:?}").contains("synthetic-secret"));
}

#[tokio::test]
async fn transport_preserves_literal_process_arguments_and_exclusive_file_creation() {
    let workspace = tempfile::tempdir().expect("remote workspace");
    let server = TestServer::shell(workspace.path()).await;
    let remote = RemoteToolExecutor::new(Vec::new(), Url::parse(&server.base_url).expect("URL"))
        .expect("remote");
    remote
        .create_file(Path::new("nested/input"), b"raw\xff bytes\n")
        .await
        .expect("file");
    assert_matches!(
        remote
            .create_file(Path::new("nested/input"), b"overwrite")
            .await,
        Err(ToolError::Execution(_))
    );
    let mut output = remote
        .capture("cat nested/input", Duration::from_secs(5))
        .await
        .expect("capture");
    assert_eq!(
        output.stdout.stripped().expect("text").kept,
        "raw\u{fffd} bytes"
    );
    assert!(output.stderr.is_empty);
    assert_eq!(output.return_code, 0);
    assert_eq!(output.timed_out_after, None);
    let mut literal = remote
        .capture(
            "printf '%s\n' \"\\$(exit 7); 'literal' | &\"",
            Duration::from_secs(5),
        )
        .await
        .expect("literal arguments");
    assert_eq!(
        literal.stdout.stripped().expect("text").kept,
        "$(exit 7); 'literal' | &"
    );
    assert_eq!(literal.return_code, 0);
    remote
        .create_file(Path::new("empty"), b"")
        .await
        .expect("empty file");
    assert!(std::fs::read(workspace.path().join("empty"))
        .expect("empty")
        .is_empty());
}

#[tokio::test]
async fn large_capture_is_disk_backed_and_deleted_after_reading() {
    use std::os::unix::fs::PermissionsExt;

    let workspace = tempfile::tempdir().expect("workspace");
    let server = TestServer::shell(workspace.path()).await;
    let remote = RemoteToolExecutor::new(Vec::new(), Url::parse(&server.base_url).expect("URL"))
        .expect("executor");
    let (result, _): (CaptureResult, _) = remote
        .send_request(
            &Request::Capture {
                command: "printf '%01048593d' 0; printf err >&2",
                timeout_seconds: 5.0,
                pipe_drain_seconds: 1.0,
                environment: BTreeMap::new(),
                remove_environment: &[],
            },
            Duration::from_secs(10),
        )
        .await
        .expect("capture");
    let CaptureStream::File { path } = &result.stdout else {
        panic!("large stream must be disk-backed");
    };
    let path = path.clone();
    assert_eq!(
        std::fs::metadata(&path)
            .expect("capture file")
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    let captured = remote
        .read_capture(result.stdout, 100, Vec::new())
        .await
        .expect("read chunks");
    assert_eq!(captured.raw_chars, 1024 * 1024 + 17);
    assert!(!Path::new(&path).exists());
    assert_matches!(result.stderr, CaptureStream::Inline(encoded) if encoded == "ZXJy");
}

#[tokio::test]
async fn remote_edit_rejects_a_changed_snapshot_without_overwriting() {
    let workspace = tempfile::tempdir().expect("workspace");
    let path = workspace.path().join("file");
    std::fs::write(&path, "original").expect("file");
    let server = TestServer::shell(workspace.path()).await;
    let remote = RemoteToolExecutor::new(Vec::new(), Url::parse(&server.base_url).expect("URL"))
        .expect("executor");
    let (snapshot, _): (EditSnapshot, _) = remote
        .send_request(
            &Request::EditSnapshot {
                path: "file",
                allow_outside_workspace: false,
            },
            Duration::from_secs(30),
        )
        .await
        .expect("snapshot");
    std::fs::write(&path, "concurrent change").expect("concurrent writer");
    let staged = workspace.path().join("staged");
    remote
        .create_file(&staged, b"replacement")
        .await
        .expect("stage");
    let error = remote
        .send_request::<RemotePath>(
            &Request::CommitEdit {
                path: "file",
                staged: &staged,
                digest: &snapshot.digest,
                allow_outside_workspace: false,
            },
            Duration::from_secs(30),
        )
        .await
        .err()
        .expect("conflict");
    assert!(error
        .to_string()
        .contains("file changed during remote edit"));
    assert_eq!(
        std::fs::read_to_string(&path).expect("read"),
        "concurrent change"
    );
    assert!(!staged.exists());
}
