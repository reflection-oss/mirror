use std::io::Write;
use std::path::Path;

use mirror_agent::tools::AnyTool;
use mirror_agent::tools::BashTool;
use mirror_agent::tools::EditTool;
use mirror_agent::tools::FindTool;
use mirror_agent::tools::GrepTool;
use mirror_agent::tools::LsTool;
use mirror_agent::tools::ReadTool;
use mirror_agent::tools::ToolError;
use mirror_agent::tools::ToolOutput;
use mirror_agent::tools::WriteTool;
use mirror_agent::AgentApprovalMode;
use mirror_agent::ExecutionContext;
use mirror_agent::TaskManagerHandle;
use mirror_protocol::chat_record;
use mirror_protocol::ChatRecord;
use mirror_protocol::ToolCall;
use mirror_protocol::ToolMessage;
use nix::errno::Errno;
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;
use prost::Message;
use tokio::io::AsyncReadExt;
use tokio::signal::unix::signal;
use tokio::signal::unix::SignalKind;

use mirror_sandbox::MAX_MESSAGE_BYTES;
use mirror_sandbox::SANDBOXD_REFRESH_FILE_CACHE_VARIABLE;

#[derive(Debug, thiserror::Error)]
pub enum RunOnceError {
    #[error("{REQUEST_BYTES_VARIABLE} must name the request length: {0}")]
    RequestLength(String),
    #[error("tool request of {0} bytes exceeds the {MAX_MESSAGE_BYTES} byte limit")]
    RequestTooLarge(usize),
    #[error("tool request could not be read: {0}")]
    Io(#[from] std::io::Error),
    #[error("tool request is not a valid tool call: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error("the tool run was stopped by SIGTERM")]
    Terminated,
    #[error("the shared workspace's file cache could not be refreshed: {0}")]
    RefreshFileCache(std::io::Error),
}

pub const REQUEST_BYTES_VARIABLE: &str = "MIRROR_TOOL_REQUEST_BYTES";

async fn run_tool_call(workspace: &Path, request: ToolCall) -> Result<ChatRecord, ToolError> {
    let tools: Vec<Box<dyn AnyTool>> = vec![
        Box::new(ReadTool::new(workspace)?),
        Box::new(BashTool::new(workspace)?),
        Box::new(WriteTool::new(workspace)?),
        Box::new(EditTool::new(workspace)?),
        Box::new(GrepTool::new(workspace)?),
        Box::new(FindTool::new(workspace)?),
        Box::new(LsTool::new(workspace)?),
    ];
    let output = if request.r#type != "function" {
        ToolOutput::error(format!("unsupported tool call type {:?}", request.r#type))
    } else if let Some(function) = request.function {
        let detached = function.name == "bash"
            && serde_json::from_str::<serde_json::Value>(&function.arguments)
                .ok()
                .and_then(|arguments| arguments.get("detached")?.as_bool())
                .unwrap_or(false);
        if detached {
            ToolOutput::error("detached sandbox commands are not supported")
        } else if let Some(tool) = tools.into_iter().find(|tool| tool.name() == function.name) {
            let context = ExecutionContext::new(
                AgentApprovalMode::Auto,
                None,
                TaskManagerHandle::default(),
                None,
                None,
            );
            tool.execute(&function.arguments, &context).await
        } else {
            ToolOutput::error(format!("unknown sandbox tool {:?}", function.name))
        }
    } else {
        ToolOutput::error("missing tool function")
    };
    Ok(ChatRecord {
        message: Some(chat_record::Message::Tool(ToolMessage {
            content: output.content,
            ..ToolMessage::default()
        })),
        display: output.display,
        success: output.success,
        ..ChatRecord::default()
    })
}

async fn run_tool_call_until_terminated(
    workspace: &Path,
    request: ToolCall,
) -> Result<ChatRecord, RunOnceError> {
    let mut terminate = signal(SignalKind::terminate())?;
    tokio::select! {
        record = run_tool_call(workspace, request) => Ok(record?),
        _ = terminate.recv() => Err(RunOnceError::Terminated),
    }
}

pub(crate) async fn read_request() -> Result<Vec<u8>, RunOnceError> {
    let request_bytes: usize = std::env::var(REQUEST_BYTES_VARIABLE)
        .map_err(|error| RunOnceError::RequestLength(error.to_string()))?
        .parse()
        .map_err(|error: std::num::ParseIntError| RunOnceError::RequestLength(error.to_string()))?;
    if request_bytes > MAX_MESSAGE_BYTES {
        return Err(RunOnceError::RequestTooLarge(request_bytes));
    }
    let mut encoded = vec![0; request_bytes];
    tokio::io::stdin().read_exact(&mut encoded).await?;
    Ok(encoded)
}

pub async fn run_once_stdio(workspace: &Path) -> Result<(), RunOnceError> {
    // Tool implementations may create their own process groups. Keep their
    // orphaned children owned by this one-call worker during cancellation.
    #[cfg(target_os = "linux")]
    nix::sys::prctl::set_child_subreaper(true).map_err(std::io::Error::from)?;
    let encoded = read_request().await?;
    let request = ToolCall::decode(encoded.as_slice())?;
    if std::env::var_os(SANDBOXD_REFRESH_FILE_CACHE_VARIABLE).is_some_and(|value| value == "1") {
        refresh_file_cache().await?;
    }
    let record = run_tool_call_until_terminated(workspace, request).await;
    if matches!(record, Err(RunOnceError::Terminated)) {
        reap_cancelled_children().await?;
    }
    let record = record?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&record.encode_to_vec())?;
    stdout.flush()?;
    Ok(())
}

/// Drops the guest's dentry and inode caches so the call sees the host's latest
/// view of a shared workspace rather than metadata cached before a host edit.
async fn refresh_file_cache() -> Result<(), RunOnceError> {
    tokio::fs::write("/proc/sys/vm/drop_caches", "2")
        .await
        .map_err(RunOnceError::RefreshFileCache)
}

async fn reap_cancelled_children() -> Result<(), std::io::Error> {
    // The tool future has been dropped, so its process guards have killed its
    // children and nobody needs their exit statuses. Waiting here cannot race
    // another call: each worker executes exactly one. The daemon still bounds
    // this cleanup with its SIGTERM grace period before killing the worker.
    loop {
        match waitpid(Pid::from_raw(-1), Some(WaitPidFlag::WNOHANG)) {
            Err(Errno::ECHILD) => return Ok(()),
            Err(Errno::EINTR) => {}
            Err(error) => return Err(error.into()),
            Ok(WaitStatus::StillAlive) => {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            Ok(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::Stdio;
    use std::time::Duration;

    use assert_matches::assert_matches;
    use mirror_protocol::FunctionCall;
    use nix::sys::signal::{kill, Signal};
    use tokio::io::AsyncWriteExt;

    use super::*;

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            r#type: "function".to_string(),
            function: Some(FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            }),
            ..ToolCall::default()
        }
    }

    fn tool_message(record: &ChatRecord) -> &ToolMessage {
        match &record.message {
            Some(chat_record::Message::Tool(message)) => message,
            other => panic!("expected a tool message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn runs_a_workspace_tool_and_returns_its_record() {
        let workspace = tempfile::tempdir().expect("workspace");

        let record = run_tool_call(
            workspace.path(),
            call("bash", r#"{"command":"printf sandboxd-ran"}"#),
        )
        .await
        .expect("bash runs");

        assert!(record.success);
        assert!(tool_message(&record).content.contains("sandboxd-ran"));
    }

    #[tokio::test]
    async fn sigterm_stops_a_running_call() {
        let workspace = tempfile::tempdir().expect("workspace");
        let path = workspace.path().to_path_buf();
        let run = tokio::spawn(async move {
            run_tool_call_until_terminated(&path, call("bash", r#"{"command":"sleep 30"}"#)).await
        });
        tokio::task::yield_now().await;

        nix::sys::signal::kill(nix::unistd::Pid::this(), nix::sys::signal::Signal::SIGTERM)
            .expect("signal this process");

        let outcome = tokio::time::timeout(Duration::from_secs(5), run)
            .await
            .expect("the run stops promptly")
            .expect("run task");
        assert_matches!(outcome, Err(RunOnceError::Terminated));
    }

    #[tokio::test]
    async fn cancelled_worker_reaps_the_shell_and_its_child() {
        let workspace = tempfile::tempdir().expect("workspace");
        let binary = std::fs::canonicalize(std::env::var("SANDBOXD_BIN").expect("guest binary"))
            .expect("resolve guest binary");
        let request = call(
            "bash",
            r#"{"command":"sleep 30 & child=$!; echo $$ $child > pids; wait"}"#,
        )
        .encode_to_vec();
        let mut worker = tokio::process::Command::new(binary)
            .arg("run-once")
            .current_dir(workspace.path())
            .env(REQUEST_BYTES_VARIABLE, request.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("start worker");
        worker
            .stdin
            .take()
            .expect("worker stdin")
            .write_all(&request)
            .await
            .expect("send call");
        let pids = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(contents) = tokio::fs::read_to_string(workspace.path().join("pids")).await
                {
                    if contents.ends_with('\n') {
                        return contents
                            .split_whitespace()
                            .map(|pid| Pid::from_raw(pid.parse().expect("process id")))
                            .collect::<Vec<_>>();
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("shell and child started");
        assert_eq!(pids.len(), 2);
        let worker_pid =
            Pid::from_raw(i32::try_from(worker.id().expect("worker id")).expect("pid fits"));
        kill(worker_pid, Signal::SIGTERM).expect("cancel worker");
        let status = tokio::time::timeout(Duration::from_secs(5), worker.wait())
            .await
            .expect("worker exits within its grace period")
            .expect("wait worker");
        assert!(!status.success(), "cancelled worker reported success");
        assert_matches!(kill(pids[0], None), Err(Errno::ESRCH));
        assert_matches!(kill(pids[1], None), Err(Errno::ESRCH));
    }

    #[tokio::test]
    async fn unknown_and_detached_calls_are_failed_tool_results() {
        let workspace = tempfile::tempdir().expect("workspace");

        let unknown = run_tool_call(workspace.path(), call("teleport", "{}"))
            .await
            .expect("record");
        let detached = run_tool_call(
            workspace.path(),
            call("bash", r#"{"command":"sleep 60","detached":true}"#),
        )
        .await
        .expect("record");

        assert!(!unknown.success);
        assert!(tool_message(&unknown)
            .content
            .contains("unknown sandbox tool"));
        assert!(!detached.success);
        assert!(tool_message(&detached)
            .content
            .contains("detached sandbox commands are not supported"));
    }
}
