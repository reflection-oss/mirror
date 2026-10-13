//! Remote execution of typed Mirror tools.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use base64::prelude::{Engine, BASE64_STANDARD};
use futures_util::FutureExt;
use mirror_protocol::ToolCall;
use reqwest::{Client, Url};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::bash_tool::{
    append_duration, format_spill, format_unspilled, needs_spill, sandbox_error_needles,
    spill_error, spill_path, validate_timeout, BashArgs, BashOutput, COMMAND_ENVIRONMENT,
    MAX_OUTPUT_CHARS, PIPE_DRAIN_GRACE, REMOVED_ENVIRONMENT,
};
use super::captured_stream::CapturedStream;
use super::read_tool::{format_content, ReadArgs};
use super::stream_capture::StreamCapture;
use super::{AnyTool, ToolError, ToolExecutor, ToolExecutorError, ToolOutput};
use crate::ExecutionContext;

/// Executes tool batches against one mandatory remote shell.
#[derive(Clone)]
pub struct RemoteToolExecutor {
    tools: Vec<Arc<dyn AnyTool>>,
    indices: HashMap<String, usize>,
    client: Client,
    url: Url,
}

#[derive(Debug, Deserialize)]
struct CommandResult {
    output: String,
    exit_code: i32,
    #[serde(default)]
    alerts: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
struct CaptureResult {
    stdout: CaptureStream,
    stderr: CaptureStream,
    return_code: i32,
    timed_out: bool,
    elapsed_seconds: f64,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CaptureStream {
    Inline(String),
    File { path: String },
}

#[derive(Deserialize)]
struct ReadResult {
    content: CaptureStream,
    exceeded_byte_cap: bool,
    root: String,
    path: String,
}

#[derive(Deserialize)]
struct DirectoryResult {
    root: String,
    path: String,
    content: Option<CaptureStream>,
}

#[derive(Deserialize)]
pub(super) struct RemotePath {
    pub root: PathBuf,
    pub path: PathBuf,
    pub is_dir: bool,
}

#[derive(Deserialize)]
struct WriteResult {
    root: PathBuf,
    path: PathBuf,
    before: CaptureStream,
}

#[derive(Deserialize)]
struct EditSnapshot {
    content: CaptureStream,
    digest: String,
}

#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Request<'a> {
    EditSnapshot {
        path: &'a str,
        allow_outside_workspace: bool,
    },
    CommitEdit {
        path: &'a str,
        staged: &'a Path,
        digest: &'a str,
        allow_outside_workspace: bool,
    },
    WriteFile {
        path: &'a str,
        staged: &'a Path,
        allow_outside_workspace: bool,
    },
    ResolvePath {
        path: &'a str,
        allow_outside_workspace: bool,
    },
    Ripgrep {
        arguments: &'a [String],
        cwd: &'a Path,
    },
    ListDirectory {
        path: &'a str,
        allow_outside_workspace: bool,
    },
    ReadFile {
        args: &'a ReadArgs,
        allow_outside_workspace: bool,
    },
    Capture {
        command: &'a str,
        timeout_seconds: f64,
        pipe_drain_seconds: f64,
        environment: BTreeMap<&'a str, &'a str>,
        remove_environment: &'a [&'a str],
    },
    ReadCapture {
        path: &'a str,
        offset: usize,
    },
    CreateFile {
        path: &'a std::path::Path,
        content: &'a str,
        append: bool,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Response<T> {
    Failure { error: i32 },
    Rejected { message: String },
    Success(T),
}

#[derive(Deserialize)]
struct FileResult {}

impl std::fmt::Debug for RemoteToolExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RemoteToolExecutor")
            .finish_non_exhaustive()
    }
}

fn validate_remote_tools(tools: &[Arc<dyn AnyTool>]) -> Result<(), ToolError> {
    super::validate_tools(tools.iter().map(|tool| tool.as_ref()))?;
    for tool in tools {
        if !tool.supports_remote() {
            return Err(ToolError::InvalidArguments(format!(
                "tool {:?} or its descendants do not support remote execution",
                tool.name()
            )));
        }
    }
    Ok(())
}

impl RemoteToolExecutor {
    /// Bind tools to a remote shell without contacting the endpoint.
    pub fn new(tools: Vec<Arc<dyn AnyTool>>, url: Url) -> Result<Self, ToolError> {
        validate_remote_tools(&tools)?;
        let indices = tools
            .iter()
            .enumerate()
            .map(|(index, tool)| (tool.name().to_string(), index))
            .collect();
        Ok(Self {
            tools,
            indices,
            client: client(&url)?,
            url,
        })
    }

    /// Keep the same remote target for a child agent's tool set.
    pub(super) fn with_tools(&self, tools: Vec<Arc<dyn AnyTool>>) -> Result<Self, ToolError> {
        validate_remote_tools(&tools)?;
        let indices = tools
            .iter()
            .enumerate()
            .map(|(index, tool)| (tool.name().to_string(), index))
            .collect();
        Ok(Self {
            tools,
            indices,
            client: self.client.clone(),
            url: self.url.clone(),
        })
    }

    fn execute_call(
        &self,
        call: &ToolCall,
        allowed: bool,
        context: &ExecutionContext,
    ) -> impl Future<Output = ToolOutput> + Send + 'static {
        let tool = call
            .function
            .as_ref()
            .and_then(|function| self.indices.get(&function.name))
            .and_then(|index| self.tools.get(*index))
            .cloned();
        let call = call.clone();
        let context = context.clone();
        let remote = self.clone();
        async move {
            if call.r#type != "function" {
                return ToolOutput::error(format!("unsupported tool call type {:?}", call.r#type));
            }
            let Some(function) = &call.function else {
                return ToolOutput::error("missing tool function");
            };
            let Some(tool) = tool else {
                return ToolOutput::failure(format!("Unknown tool: {}", function.name));
            };
            if !allowed {
                return ToolOutput::error("Tool call rejected by the user.");
            }
            tool.execute_remote(&function.arguments, &context, &remote)
                .await
        }
    }
}

impl ToolExecutor for RemoteToolExecutor {
    fn tools(&self) -> &[Arc<dyn AnyTool>] {
        &self.tools
    }

    fn execute(
        &self,
        _batch_id: Uuid,
        calls: &[ToolCall],
        allowed: &[bool],
        context: &ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ToolOutput>, ToolExecutorError>> + Send>> {
        // Calls run one after another, in order, so a call sees the effects of
        // the calls before it.
        let executions: Vec<_> = calls
            .iter()
            .zip(allowed)
            .map(|(call, allowed)| self.execute_call(call, *allowed, context))
            .collect();
        Box::pin(async move {
            let mut outputs = Vec::with_capacity(executions.len());
            for execution in executions {
                outputs.push(
                    AssertUnwindSafe(execution)
                        .catch_unwind()
                        .await
                        .unwrap_or_else(|payload| ToolOutput::error(panic_message(payload))),
                );
            }
            Ok(outputs)
        })
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "tool panicked".to_string())
}

/// Bind to an Irregular-compatible send-command URL.
fn client(url: &Url) -> Result<Client, ToolError> {
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(ToolError::InvalidArguments(
            "remote must be an HTTP(S) URL without credentials or a fragment".into(),
        ));
    }
    let _ = rustls::crypto::ring::default_provider().install_default();
    Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()
        .map_err(|_| ToolError::Execution("could not create remote shell HTTP client".into()))
}

impl RemoteToolExecutor {
    pub(super) async fn run_edit(
        &self,
        args: &super::EditArgs,
        allow_outside_workspace: bool,
    ) -> Result<ToolOutput, ToolError> {
        let (snapshot, _): (EditSnapshot, _) = self
            .send_request(
                &Request::EditSnapshot {
                    path: &args.file_path,
                    allow_outside_workspace,
                },
                Duration::from_secs(30),
            )
            .await?;
        let bytes = self.read_bytes(snapshot.content, 64 * 1024 * 1024).await?;
        let text = String::from_utf8(bytes)
            .map_err(|_| ToolError::Execution("stream did not contain valid UTF-8".into()))?;
        let (updated, count) = super::edit::replace_text(&text, args)?;
        let staged = spill_path();
        self.create_file(&staged, updated.as_bytes()).await?;
        let (resolved, _): (RemotePath, _) = self
            .send_request(
                &Request::CommitEdit {
                    path: &args.file_path,
                    staged: &staged,
                    digest: &snapshot.digest,
                    allow_outside_workspace,
                },
                Duration::from_secs(30),
            )
            .await?;
        Ok(
            super::edit::format_result(&text, &updated, count, args).touching(super::touched(
                &resolved.root,
                &resolved.path,
                mirror_protocol::TouchedKind::Write,
            )),
        )
    }

    pub(super) async fn run_write(
        &self,
        args: &super::WriteArgs,
        allow_outside_workspace: bool,
    ) -> Result<ToolOutput, ToolError> {
        let staged = spill_path();
        self.create_file(&staged, args.content.as_bytes()).await?;
        let (result, _): (WriteResult, _) = self
            .send_request(
                &Request::WriteFile {
                    path: &args.file_path,
                    staged: &staged,
                    allow_outside_workspace,
                },
                Duration::from_secs(30),
            )
            .await?;
        let before = self.read_bytes(result.before, 64 * 1024 * 1024).await?;
        Ok(
            super::write::format_result(&String::from_utf8_lossy(&before), args).touching(
                super::touched(
                    &result.root,
                    &result.path,
                    mirror_protocol::TouchedKind::Write,
                ),
            ),
        )
    }

    pub(super) async fn resolve_path(
        &self,
        path: &str,
        allow_outside_workspace: bool,
    ) -> Result<RemotePath, ToolError> {
        let (path, _) = self
            .send_request(
                &Request::ResolvePath {
                    path,
                    allow_outside_workspace,
                },
                Duration::from_secs(30),
            )
            .await?;
        Ok(path)
    }

    pub(super) async fn run_rg(
        &self,
        arguments: &[String],
        cwd: &Path,
    ) -> Result<super::ripgrep::RgResult, ToolError> {
        let result = self.run_rg_once(arguments, cwd).await?;
        let output = [result.stdout.as_str(), result.stderr.as_str()].concat();
        if !matches!(result.return_code, 0 | 1)
            && !result.timed_out
            && ["EAGAIN", "Resource temporarily unavailable"]
                .iter()
                .any(|pattern| output.contains(pattern))
        {
            let retry = [vec!["-j".to_string(), "1".to_string()], arguments.to_vec()].concat();
            return self.run_rg_once(&retry, cwd).await;
        }
        Ok(result)
    }

    async fn run_rg_once(
        &self,
        arguments: &[String],
        cwd: &Path,
    ) -> Result<super::ripgrep::RgResult, ToolError> {
        if arguments.iter().any(|argument| argument.contains('\0')) {
            return Err(ToolError::Execution(
                "nul byte found in provided data".into(),
            ));
        }
        let (result, _): (CaptureResult, _) = self
            .send_request(
                &Request::Ripgrep { arguments, cwd },
                Duration::from_secs(60),
            )
            .await?;
        let stdout = self.read_bytes(result.stdout, 64 * 1024 * 1024).await?;
        let stderr = self.read_bytes(result.stderr, 64 * 1024 * 1024).await?;
        Ok(super::ripgrep::RgResult {
            return_code: if result.timed_out {
                124
            } else {
                result.return_code
            },
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            timed_out: result.timed_out,
        })
    }

    pub(super) async fn run_read(
        &self,
        args: &ReadArgs,
        allow_outside_workspace: bool,
    ) -> Result<ToolOutput, ToolError> {
        if args.file_path.contains('\0') {
            return Err(ToolError::Execution(
                "file name contained an unexpected NUL byte".into(),
            ));
        }
        let (result, _): (ReadResult, _) = self
            .send_request(
                &Request::ReadFile {
                    args,
                    allow_outside_workspace,
                },
                Duration::from_secs(30),
            )
            .await?;
        let content = self
            .read_bytes(result.content, args.max_chars * 4 + 4)
            .await?;
        Ok(
            format_content(content, result.exceeded_byte_cap, args).touching(super::touched(
                Path::new(&result.root),
                Path::new(&result.path),
                mirror_protocol::TouchedKind::Read,
            )),
        )
    }

    async fn read_bytes(
        &self,
        stream: CaptureStream,
        byte_cap: usize,
    ) -> Result<Vec<u8>, ToolError> {
        let mut content = Vec::new();
        match stream {
            CaptureStream::Inline(encoded) => {
                content = BASE64_STANDARD
                    .decode(encoded)
                    .map_err(|_| invalid_result())?;
                if content.len() > byte_cap {
                    return Err(invalid_result());
                }
            }
            CaptureStream::File { path } => loop {
                let (encoded, _): (String, _) = self
                    .send_request(
                        &Request::ReadCapture {
                            path: &path,
                            offset: content.len(),
                        },
                        Duration::from_secs(30),
                    )
                    .await?;
                let bytes = BASE64_STANDARD
                    .decode(encoded)
                    .map_err(|_| invalid_result())?;
                if bytes.len() > 1024 * 1024 || bytes.len() > byte_cap - content.len() {
                    return Err(invalid_result());
                }
                let finished = bytes.len() < 1024 * 1024;
                content.extend_from_slice(&bytes);
                if finished {
                    break;
                }
            },
        }
        Ok(content)
    }

    pub(super) async fn run_ls(
        &self,
        path: &str,
        allow_outside_workspace: bool,
    ) -> Result<ToolOutput, ToolError> {
        let (result, _): (DirectoryResult, _) = self
            .send_request(
                &Request::ListDirectory {
                    path,
                    allow_outside_workspace,
                },
                Duration::from_secs(30),
            )
            .await?;
        let Some(content) = result.content else {
            return Ok(ToolOutput::error(format!("{path} is not a directory.")));
        };
        let bytes = self.read_bytes(content, 64 * 1024 * 1024).await?;
        let entries = serde_json::from_slice(&bytes).map_err(|_| invalid_result())?;
        Ok(
            super::ls_tool::format_entries(entries).touching(super::touched(
                Path::new(&result.root),
                Path::new(&result.path),
                mirror_protocol::TouchedKind::List,
            )),
        )
    }

    pub(super) async fn run_bash(&self, args: BashArgs) -> Result<ToolOutput, ToolError> {
        if args.detached {
            return Err(ToolError::InvalidArguments(
                "remote execution does not support detached commands".into(),
            ));
        }
        let timeout = validate_timeout(args.timeout_seconds)?;
        let output = self.capture(&args.command, timeout).await?;
        let (mut result, elapsed) = format_unspilled(output, args.command).await?;
        if result.success && needs_spill(&result.content) {
            let path = spill_path();
            let saved = self.create_file(&path, result.content.as_bytes()).await;
            if let Ok(alerts) = &saved {
                if !alerts.is_empty() {
                    result.display = Some(format!(
                        "{}\nAlerts: {}",
                        result.display.unwrap_or_default(),
                        Value::Array(alerts.clone())
                    ));
                }
            }
            result.content =
                format_spill(&result.content, saved.map(|_| path).map_err(spill_error));
        }
        result.content = append_duration(result.content, elapsed);
        Ok(result)
    }
    /// Capture remote output with the caller's retention policy.
    async fn capture(&self, command: &str, timeout: Duration) -> Result<BashOutput, ToolError> {
        if command.contains('\0') {
            return Err(ToolError::Execution(
                "nul byte found in provided data".into(),
            ));
        }
        let (result, alerts): (CaptureResult, _) = self
            .send_request(
                &Request::Capture {
                    command,
                    timeout_seconds: timeout.as_secs_f64(),
                    pipe_drain_seconds: PIPE_DRAIN_GRACE.as_secs_f64(),
                    environment: COMMAND_ENVIRONMENT.iter().copied().collect(),
                    remove_environment: &REMOVED_ENVIRONMENT,
                },
                timeout.saturating_add(Duration::from_secs(30)),
            )
            .await?;
        let stdout = self
            .read_capture(result.stdout, MAX_OUTPUT_CHARS, Vec::new())
            .await?;
        let stderr = self
            .read_capture(result.stderr, MAX_OUTPUT_CHARS, sandbox_error_needles())
            .await?;
        Ok(BashOutput {
            stdout,
            stderr,
            return_code: result.return_code,
            timed_out_after: result.timed_out.then_some(timeout),
            elapsed: Duration::try_from_secs_f64(result.elapsed_seconds)
                .map_err(|_| invalid_result())?,
            alerts,
        })
    }

    async fn read_capture(
        &self,
        stream: CaptureStream,
        max_chars: usize,
        needles: Vec<&'static str>,
    ) -> Result<CapturedStream, ToolError> {
        let (sink, capture) = StreamCapture::spawn(max_chars, needles);
        match stream {
            CaptureStream::Inline(encoded) => {
                let bytes = BASE64_STANDARD
                    .decode(encoded)
                    .map_err(|_| invalid_result())?;
                sink.send(bytes).await.map_err(|_| invalid_result())?;
            }
            CaptureStream::File { path } => {
                let mut offset = 0;
                loop {
                    let (encoded, _): (String, _) = self
                        .send_request(
                            &Request::ReadCapture {
                                path: &path,
                                offset,
                            },
                            Duration::from_secs(30),
                        )
                        .await?;
                    let bytes = BASE64_STANDARD
                        .decode(encoded)
                        .map_err(|_| invalid_result())?;
                    let length = bytes.len();
                    if length > 1024 * 1024 {
                        return Err(invalid_result());
                    }
                    sink.send(bytes).await.map_err(|_| invalid_result())?;
                    if length < 1024 * 1024 {
                        break;
                    }
                    offset += length;
                }
            }
        }
        drop(sink);
        capture.await.map_err(|_| invalid_result())
    }

    /// Create a remote file without exceeding shell endpoint argument limits.
    async fn create_file(&self, path: &Path, content: &[u8]) -> Result<Vec<Value>, ToolError> {
        let mut alerts = Vec::new();
        let mut remaining = content;
        let mut append = false;
        loop {
            let (chunk, rest) = remaining.split_at(remaining.len().min(32 * 1024));
            let (_, file_alerts) = self
                .send_request::<FileResult>(
                    &Request::CreateFile {
                        path,
                        content: &BASE64_STANDARD.encode(chunk),
                        append,
                    },
                    Duration::from_secs(30),
                )
                .await?;
            alerts.extend(file_alerts);
            if rest.is_empty() {
                return Ok(alerts);
            }
            remaining = rest;
            append = true;
        }
    }

    async fn send_request<T: DeserializeOwned>(
        &self,
        request: &Request<'_>,
        timeout: Duration,
    ) -> Result<(T, Vec<Value>), ToolError> {
        let input = serde_json::to_vec(request)
            .map_err(|error| ToolError::InvalidArguments(error.to_string()))?;
        let input = BASE64_STANDARD.encode(input);
        let script = REMOTE_SCRIPT.replace('\'', "'\"'\"'");
        // A quoted here-document prevents shell expansion and keeps the request out of Python argv.
        let command = format!(
        "python3 -I -c '{script}' <<'MIRROR_ENVIRONMENT_INPUT'\n{input}\nMIRROR_ENVIRONMENT_INPUT"
    );
        let result = self.execute_command(command, timeout).await?;
        if result.exit_code != 0 {
            return Err(invalid_result());
        }
        let response: Response<T> =
            serde_json::from_str(&result.output).map_err(|_| invalid_result())?;
        match response {
            Response::Rejected { message } => Err(ToolError::Execution(message)),
            Response::Success(output) => Ok((output, result.alerts.unwrap_or_default())),
            Response::Failure { error } => Err(ToolError::Execution(
                std::io::Error::from_raw_os_error(error).to_string(),
            )),
        }
    }

    /// Execute one raw command; response data is never evaluated on the agent host.
    async fn execute_command(
        &self,
        command: String,
        timeout: Duration,
    ) -> Result<CommandResult, ToolError> {
        let mut url = self.url.clone();
        let query = url
            .query_pairs()
            .filter(|(key, _)| key != "timeout")
            .map(|(key, value)| (key.into_owned(), value.into_owned()))
            .collect::<Vec<_>>();
        url.set_query(None);
        url.query_pairs_mut()
            .extend_pairs(query)
            .append_pair("timeout", &timeout.as_secs_f64().to_string());
        let mut response = self
            .client
            .post(url)
            .timeout(timeout.saturating_add(Duration::from_secs(30)))
            .body(command)
            .send()
            .await
            .map_err(|_| {
                ToolError::Execution(
                "remote request failed; command may still be running or have executed; not retried"
                    .into(),
            )
            })?;
        if !response.status().is_success() {
            return Err(ToolError::Execution(format!(
                "remote shell returned HTTP {}; command outcome is unknown; not retried",
                response.status().as_u16(),
            )));
        }
        // One base64 capture chunk plus JSON framing; reject before allocating an unbounded body.
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| invalid_result())? {
            if chunk.len() > 2 * 1024 * 1024 - body.len() {
                return Err(ToolError::Execution(
                    "remote response exceeds 2 MiB; command outcome is unknown; not retried".into(),
                ));
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body).map_err(|_| ToolError::Execution(
        "remote shell returned invalid command-result JSON; command may have executed; not retried".into(),
    ))
    }
}
fn invalid_result() -> ToolError {
    ToolError::Execution("remote execution wrapper failed or returned invalid output; Python 3 is required; command outcome is unknown; not retried".into())
}

// Runs only at the remote endpoint; stdout/stderr are spooled rather than buffered.
const REMOTE_SCRIPT: &str = r#"
import base64
import contextlib
import hashlib
import json
import os
import pathlib
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def _kill_group(process):
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def _capture(process, request, stdout, stderr):
    started = time.monotonic()
    deadline = started + request["timeout_seconds"]
    drain_deadline = float("inf")
    timed_out = False
    with selectors.DefaultSelector() as selector:
        assert process.stdout is not None and process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if now >= min(deadline, drain_deadline):
                timed_out = deadline <= drain_deadline
                if timed_out:
                    _kill_group(process)
                break
            for key, _ in selector.select(
                min(0.01, deadline - now, drain_deadline - now)
            ):
                data = os.read(key.fd, 65536)
                if data:
                    key.data.write(data)
                else:
                    selector.unregister(key.fileobj)
            if process.poll() is not None and drain_deadline == float("inf"):
                drain_deadline = time.monotonic() + request["pipe_drain_seconds"]
    return_code = process.wait()
    return {
        "stdout": _finish_stream(stdout),
        "stderr": _finish_stream(stderr),
        "return_code": -1 if return_code < 0 or timed_out else return_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
    }


def capture(request):
    environment = os.environ.copy()
    for name in request["remove_environment"]:
        environment.pop(name, None)
    environment.update(request["environment"])
    with (
        tempfile.NamedTemporaryFile(delete=False) as stdout,
        tempfile.NamedTemporaryFile(delete=False) as stderr,
        subprocess.Popen(
            ["/bin/bash", "-c", request["command"]],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
        ) as process,
    ):
        try:
            return _capture(process, request, stdout.file, stderr.file)
        except BaseException:
            _kill_group(process)
            pathlib.Path(stdout.name).unlink(missing_ok=True)
            pathlib.Path(stderr.name).unlink(missing_ok=True)
            raise


def _finish_stream(stream):
    stream.flush()
    if stream.tell() > 65536:
        return {"path": str(stream.name)}
    stream.seek(0)
    content = base64.b64encode(stream.read()).decode("ascii")
    pathlib.Path(stream.name).unlink()
    return content


def read_capture(path, offset):
    with open(path, "rb") as stream:
        stream.seek(offset)
        content = stream.read(1024 * 1024)
    if len(content) < 1024 * 1024:
        pathlib.Path(path).unlink()
    return base64.b64encode(content).decode("ascii")


def create_file(path, content, append):
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND if append else os.O_CREAT | os.O_EXCL
    fd = os.open(destination, os.O_WRONLY | flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(base64.b64decode(content))
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def digest_file(stream):
    digest = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def edit_snapshot(request):
    _, path = resolve_path(request["path"], request["allow_outside_workspace"])
    with tempfile.NamedTemporaryFile(delete=False) as content:
        try:
            snapshot(path, content)
            digest = digest_file(content)
            content.seek(0, os.SEEK_END)
            return {"content": _finish_stream(content.file), "digest": digest}
        except BaseException:
            pathlib.Path(content.name).unlink(missing_ok=True)
            raise


def commit_edit(request):
    staged = pathlib.Path(request["staged"])
    try:
        root, path = resolve_path(request["path"], request["allow_outside_workspace"])
        with staged.open("rb") as source, path.open("r+b") as output:
            if digest_file(output) != request["digest"]:
                raise ValueError("file changed during remote edit; read it again before retrying")
            output.seek(0)
            output.truncate()
            shutil.copyfileobj(source, output)
        return {"root": str(root), "path": str(path), "is_dir": False}
    finally:
        staged.unlink(missing_ok=True)


def snapshot(path, output, missing_ok=False):
    try:
        source = open(path, "rb")
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    with source:
        while chunk := source.read(1024 * 1024):
            if output.tell() + len(chunk) > 64 * 1024 * 1024:
                raise ValueError("remote file exceeds 64 MiB")
            output.write(chunk)
    output.flush()


def write_file(request):
    staged = pathlib.Path(request["staged"])
    try:
        root, path = resolve_path(request["path"], request["allow_outside_workspace"], for_write=True)
        with tempfile.NamedTemporaryFile(delete=False) as before:
            try:
                snapshot(path, before, missing_ok=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                with staged.open("rb") as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output)
                content = _finish_stream(before.file)
            except BaseException:
                pathlib.Path(before.name).unlink(missing_ok=True)
                raise
        return {"root": str(root), "path": str(path), "before": content}
    finally:
        staged.unlink(missing_ok=True)


def ripgrep(request):
    environment = os.environ.copy()
    environment.pop("REFLECTION_API_KEY", None)
    with (
        tempfile.NamedTemporaryFile(delete=False) as stdout,
        tempfile.NamedTemporaryFile(delete=False) as stderr,
        subprocess.Popen(
            ["rg", "--no-config", *request["arguments"]], cwd=request["cwd"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, start_new_session=True,
        ) as process,
    ):
        try:
            return _capture(process, {"timeout_seconds": 30, "pipe_drain_seconds": 30}, stdout.file, stderr.file)
        except BaseException:
            _kill_group(process)
            pathlib.Path(stdout.name).unlink(missing_ok=True)
            pathlib.Path(stderr.name).unlink(missing_ok=True)
            raise


def resolve_path(path, allow_outside, for_write=False):
    if "\0" in path:
        raise ValueError("file name contained an unexpected NUL byte")
    root = pathlib.Path.cwd().resolve(strict=True)
    candidate = os.path.join(str(root), path)
    if for_write:
        existing = candidate
        while not os.path.lexists(existing):
            existing = os.path.dirname(existing.rstrip("/"))
        resolved = pathlib.Path(os.path.normpath(pathlib.Path(os.path.realpath(existing, strict=True)) / os.path.relpath(candidate, existing)))
    else:
        resolved = pathlib.Path(os.path.realpath(candidate, strict=True))
    if not allow_outside and not resolved.is_relative_to(root):
        raise ValueError(f"path {resolved} is outside workspace {root}")
    return root, resolved


def list_directory(request):
    root, path = resolve_path(request["path"], request["allow_outside_workspace"])
    content = None
    if path.is_dir():
        with tempfile.NamedTemporaryFile(delete=False) as output:
            try:
                output.write(b"[")
                separator = b""
                with os.scandir(path) as directory:
                    for entry in directory:
                        name = os.fsencode(entry.name).decode("utf-8", "replace")
                        name += "/" if entry.is_dir(follow_symlinks=False) else ""
                        output.write(separator + json.dumps(name).encode())
                        separator = b"," 
                        if output.tell() > 64 * 1024 * 1024 - 1:
                            raise ValueError("remote directory listing exceeds 64 MiB")
                output.write(b"]")
                content = _finish_stream(output.file)
            except BaseException:
                pathlib.Path(output.name).unlink(missing_ok=True)
                raise
    return {"root": str(root), "path": str(path), "content": content}


def read_file(request):
    args = request["args"]
    root, path = resolve_path(args["file_path"], request["allow_outside_workspace"])
    first_line = args["offset"] or 1
    end_line = first_line + args["limit"] if args["limit"] is not None else None
    byte_cap = args["max_chars"] * 4 + 4
    selected = bytearray()
    line = 1
    exceeded = False
    with open(path, "rb") as source:
        done = False
        while not done:
            chunk = source.read(8192)
            if not chunk:
                break
            for byte in chunk:
                if line >= first_line and (end_line is None or line < end_line):
                    if len(selected) == byte_cap:
                        exceeded = True
                        done = True
                        break
                    selected.append(byte)
                if byte == 10:
                    line += 1
                    if end_line is not None and line >= end_line:
                        done = True
                        break
    with tempfile.NamedTemporaryFile(delete=False) as output:
        try:
            output.write(selected)
            content = _finish_stream(output.file)
        except BaseException:
            pathlib.Path(output.name).unlink(missing_ok=True)
            raise
    return {"content": content, "exceeded_byte_cap": exceeded, "root": str(root), "path": str(path)}


def main():
    request = json.loads(base64.b64decode(sys.stdin.buffer.read()))
    try:
        if request["operation"] == "edit_snapshot":
            print(json.dumps(edit_snapshot(request)))
        elif request["operation"] == "commit_edit":
            print(json.dumps(commit_edit(request)))
        elif request["operation"] == "write_file":
            print(json.dumps(write_file(request)))
        elif request["operation"] == "resolve_path":
            root, path = resolve_path(request["path"], request["allow_outside_workspace"])
            print(json.dumps({"root": str(root), "path": str(path), "is_dir": path.is_dir()}))
        elif request["operation"] == "ripgrep":
            print(json.dumps(ripgrep(request)))
        elif request["operation"] == "list_directory":
            print(json.dumps(list_directory(request)))
        elif request["operation"] == "read_file":
            print(json.dumps(read_file(request)))
        elif request["operation"] == "capture":
            result = capture(request)
            print(json.dumps(result))
        elif request["operation"] == "read_capture":
            print(json.dumps(read_capture(request["path"], request["offset"])))
        elif request["operation"] == "create_file":
            create_file(request["path"], request["content"], request["append"])
            print("{}")
        else:
            raise ValueError("unknown remote operation")
    except ValueError as error:
        print(json.dumps({"message": str(error)}))
    except OSError as error:
        print(json.dumps({"error": error.errno}))


if __name__ == "__main__":
    main()
"#;

#[cfg(test)]
#[path = "remote_tool_executor_test.rs"]
mod tests;
