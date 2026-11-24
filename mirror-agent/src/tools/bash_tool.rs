use super::process_group::ProcessGroup;
use super::stream_capture::StreamCapture;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::Duration;
use std::time::Instant;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::mpsc;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::captured_stream::{CapturedStream, RAW_END_CHARS};
use super::partial_text::PartialText;
use super::Tool;
use super::ToolError;
use super::{object_schema, RemoteToolExecutor};
use crate::tools::ToolOutput;
use crate::ExecutionContext;
use crate::TaskBehavior;
use crate::TaskContext;

#[derive(Debug, Clone)]
pub struct BashTool {
    workspace_root: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub(crate) struct BashOutput {
    pub return_code: i32,
    pub stdout: CapturedStream,
    pub stderr: CapturedStream,
    pub timed_out_after: Option<Duration>,
    pub elapsed: Duration,
    pub alerts: Vec<Value>,
}

const PREVIEW_CHARS: usize = 2_000;
pub(crate) const MAX_OUTPUT_CHARS: usize = 64 * 1024 * 1024;
const MAX_PARTIAL_OUTPUT_CHARS: usize = 2 * RAW_END_CHARS;
pub(crate) const PIPE_DRAIN_GRACE: Duration = Duration::from_secs(3);
const SPILL_THRESHOLD_CHARS: usize = 30_000;
const SPILL_DIR: &str = "/tmp/tool-results";

const NON_ERROR_EXIT_1: &[(&str, &str)] = &[
    ("grep", "No matches found"),
    ("rg", "No matches found"),
    ("find", "Some directories were inaccessible"),
    ("diff", "Files differ"),
    ("test", "Condition is false"),
    ("[", "Condition is false"),
];

const SANDBOX_ERROR_PATTERNS: &[(&str, &str)] = &[
    (
        "Operation not permitted",
        "This operation was blocked by the sandbox security policy.",
    ),
    (
        "Permission denied",
        "The sandbox denied access to this resource.",
    ),
    (
        "Read-only file system",
        "The sandbox file system is read-only at this location.",
    ),
];

pub(crate) const BASH_DESCRIPTION: &str = "Execute a bash command. The output contains the command's stdout and stderr.\nCommands time out after 10 minutes. You can set a shorter timeout within the bash command.";

impl BashTool {
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

impl Tool for BashTool {
    type Args = BashArgs;

    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        BASH_DESCRIPTION
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([(
                "command".to_string(),
                serde_json::json!({"type": "string", "description": "The command to execute"}),
            )]),
            &["command"],
        )
    }

    fn needs_approval(&self) -> bool {
        true
    }

    fn supports_remote(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: BashArgs,
        _context: &ExecutionContext,
        remote: &RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        remote.run_bash(args).await
    }

    async fn run(
        &self,
        args: BashArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let workspace = self.workspace_root.clone().ok_or_else(|| {
            ToolError::Execution("remote-only Bash tool cannot execute locally".into())
        })?;
        let timeout = if args.detached {
            None
        } else {
            Some(validate_timeout(args.timeout_seconds)?)
        };
        if !args.detached && context.tasks().is_closed() {
            return run_command(args.command, workspace, timeout, None).await;
        }
        let behavior = if args.detached {
            TaskBehavior::Detach
        } else {
            TaskBehavior::Wait
        };
        let command = args.command;
        let task_id = context
            .tasks()
            .spawn(
                "bash",
                Some(command.clone()),
                behavior,
                move |task_context| async move {
                    match run_command(command, workspace, timeout, Some(task_context)).await {
                        Ok(mut output) => {
                            if args.detached {
                                output.display = None;
                            }
                            output
                        }
                        Err(error) => ToolOutput::error(error.to_string()),
                    }
                },
            )
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        match context.tasks().resolve(&task_id).await {
            Ok(crate::TaskResolution::Finished(output)) => Ok(output),
            Ok(crate::TaskResolution::Detached) => {
                Ok(ToolOutput::success(format!("Task {task_id} started.")))
            }
            Err(error) => Err(ToolError::Execution(error.to_string())),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BashArgs {
    pub command: String,
    #[serde(skip, default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
    #[serde(skip)]
    pub detached: bool,
}

pub(crate) const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_mins(10);

fn default_timeout_seconds() -> f64 {
    DEFAULT_BASH_TIMEOUT.as_secs_f64()
}

pub(crate) fn validate_timeout(seconds: f64) -> Result<Duration, ToolError> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(ToolError::InvalidArguments(
            "timeout_seconds must be finite and greater than zero".to_string(),
        ));
    }
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        ToolError::InvalidArguments(
            "timeout_seconds must be representable as a duration".to_string(),
        )
    })
}

pub(crate) fn sandbox_error_needles() -> Vec<&'static str> {
    SANDBOX_ERROR_PATTERNS
        .iter()
        .map(|(pattern, _)| *pattern)
        .collect()
}

pub(crate) const COMMAND_ENVIRONMENT: [(&str, &str); 6] = [
    ("TERM", "dumb"),
    ("NO_COLOR", "1"),
    ("COLORTERM", ""),
    ("PAGER", "cat"),
    ("GIT_PAGER", "cat"),
    ("GH_PAGER", "cat"),
];

pub(crate) const REMOVED_ENVIRONMENT: [&str; 9] = [
    "REFLECTION_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "AI_GATEWAY_API_KEY",
    "FIRECRAWL_API_KEY",
    "YOU_API_KEY",
    "TAVILY_API_KEY",
];

async fn run_command(
    command: String,
    workspace: PathBuf,
    timeout: Option<Duration>,
    task_context: Option<TaskContext>,
) -> Result<ToolOutput, ToolError> {
    let started = Instant::now();
    let mut output = run_process(&command, &workspace, timeout, task_context).await?;
    output.elapsed = started.elapsed();
    format_result(output, command).await
}

async fn format_result(mut output: BashOutput, command: String) -> Result<ToolOutput, ToolError> {
    let timed_out = output.timed_out_after.is_some();
    let success = !timed_out;
    let exit_code = success.then_some(output.return_code);
    let display = command.clone();
    let content = tokio::task::spawn_blocking(move || {
        let body = format_body(&mut output, &command)?;
        let body = if success { maybe_spill(body) } else { body };
        Ok::<_, std::io::Error>(append_duration(body, output.elapsed))
    })
    .await
    .map_err(|error| ToolError::Execution(error.to_string()))?
    .map_err(|error| ToolError::Execution(error.to_string()))?;
    Ok(ToolOutput {
        content,
        display: Some(display),
        success,
        touched: Vec::new(),
        exit_code,
        timed_out,
        duration: None,
    })
}

pub(crate) async fn format_unspilled(
    mut output: BashOutput,
    command: String,
) -> Result<(ToolOutput, Duration), ToolError> {
    let timed_out = output.timed_out_after.is_some();
    let success = !timed_out;
    let exit_code = success.then_some(output.return_code);
    let elapsed = output.elapsed;
    let alerts = std::mem::take(&mut output.alerts);
    let render_command = command.clone();
    let body = tokio::task::spawn_blocking(move || format_body(&mut output, &render_command))
        .await
        .map_err(|error| ToolError::Execution(error.to_string()))?
        .map_err(|error| ToolError::Execution(error.to_string()))?;
    let display = if alerts.is_empty() {
        command
    } else {
        format!("{command}\nAlerts: {}", Value::Array(alerts))
    };
    Ok((
        ToolOutput {
            content: body,
            display: Some(display),
            success,
            touched: Vec::new(),
            exit_code,
            timed_out,
            duration: None,
        },
        elapsed,
    ))
}

fn format_body(output: &mut BashOutput, command: &str) -> std::io::Result<String> {
    Ok(match output.timed_out_after {
        Some(timeout) => format_timeout(output, timeout),
        None => cap_output(format_completed(output, command)?),
    })
}

pub(crate) fn append_duration(output: String, elapsed: Duration) -> String {
    append_line(
        output,
        &format!("[Completed in {:.1}s]", elapsed.as_secs_f64()),
    )
}

fn cap_output(output: PartialText) -> String {
    if output.chars <= MAX_OUTPUT_CHARS {
        return output.kept;
    }
    let mut kept = output.kept;
    let cap = kept
        .char_indices()
        .nth(MAX_OUTPUT_CHARS)
        .map_or(kept.len(), |(index, _)| index);
    let remaining = output.newlines - kept[..cap].matches('\n').count() + 1;
    let warning = format!("\n[{remaining} lines truncated, exceeded 64MB cap]");
    let (end, _) = kept
        .char_indices()
        .nth(MAX_OUTPUT_CHARS - warning.len())
        .expect("output exceeds the cap, which exceeds the ASCII warning length");
    kept.truncate(end);
    kept.push_str(&warning);
    kept
}

fn append_line(text: String, line: &str) -> String {
    if text.is_empty() {
        line.to_string()
    } else {
        format!("{text}\n{line}")
    }
}

fn format_timeout(output: &BashOutput, timeout: Duration) -> String {
    let mut parts = vec![format!("Command timed out after {}ms", timeout.as_millis())];
    if !output.stdout.is_empty {
        parts.push(format!(
            "Partial stdout:\n{}",
            elide_partial_stream(&output.stdout)
        ));
    }
    if !output.stderr.is_empty {
        parts.push(format!(
            "Partial stderr:\n{}",
            elide_partial_stream(&output.stderr)
        ));
    }
    parts.join("\n")
}

fn elide_partial_stream(stream: &CapturedStream) -> String {
    let length = stream.raw_chars;
    if length <= MAX_PARTIAL_OUTPUT_CHARS {
        return stream.raw_head().to_string();
    }
    format!(
        "{}\n... [{} chars omitted] ...\n{}",
        stream.raw_head(),
        length - MAX_PARTIAL_OUTPUT_CHARS,
        stream.raw_tail
    )
}

fn format_completed(output: &mut BashOutput, command: &str) -> std::io::Result<PartialText> {
    let stdout = output.stdout.stripped()?;
    let stderr = match output.stderr.first_needle() {
        Some(index) => output.stderr.stripped_with_suffix(&format!(
            "\n[Sandbox note: {}]",
            SANDBOX_ERROR_PATTERNS[index].1
        ))?,
        None => output.stderr.stripped()?,
    };
    Ok(match non_error_exit_note(command, output.return_code) {
        Some(note) => append_partial_line(
            format_streams(stdout, stderr, 0),
            &format!("{note} (exit code {})", output.return_code),
        ),
        None => format_streams(stdout, stderr, output.return_code),
    })
}

fn format_streams(stdout: PartialText, stderr: PartialText, return_code: i32) -> PartialText {
    let mut parts = Vec::new();
    if return_code != 0 {
        parts.push(PartialText::whole(format!("Exit code: {return_code}")));
    }
    if stdout.chars > 0 {
        parts.push(stdout);
    }
    if stderr.chars > 0 {
        parts.push(PartialText::whole("stderr:\n".to_string()).concat(stderr));
    }
    let mut parts = parts.into_iter();
    let first = parts
        .next()
        .unwrap_or_else(|| PartialText::whole(String::new()));
    parts.fold(first, |joined, part| {
        joined
            .concat(PartialText::whole("\n".to_string()))
            .concat(part)
    })
}

fn append_partial_line(text: PartialText, line: &str) -> PartialText {
    let line = PartialText::whole(line.to_string());
    if text.chars == 0 {
        line
    } else {
        text.concat(PartialText::whole("\n".to_string()))
            .concat(line)
    }
}

fn non_error_exit_note(command: &str, return_code: i32) -> Option<&'static str> {
    if return_code != 1 {
        return None;
    }
    let last = split_commands(command).pop()?;
    let program = last.split_whitespace().next()?;
    NON_ERROR_EXIT_1
        .iter()
        .find(|(name, _)| *name == program)
        .map(|(_, note)| *note)
}

fn split_commands(command: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let (mut in_single, mut in_double, mut escaped) = (false, false, false);
    let mut characters = command.chars().peekable();
    while let Some(character) = characters.next() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        match character {
            '\\' if !in_single => {
                current.push(character);
                escaped = true;
            }
            '\'' if !in_double => {
                current.push(character);
                in_single = !in_single;
            }
            '"' if !in_single => {
                current.push(character);
                in_double = !in_double;
            }
            _ if in_single || in_double => current.push(character),
            ';' => parts.push(std::mem::take(&mut current)),
            '&' | '|' if characters.peek() == Some(&character) => {
                characters.next();
                parts.push(std::mem::take(&mut current));
            }
            '|' => parts.push(std::mem::take(&mut current)),
            _ => current.push(character),
        }
    }
    parts.push(current);
    parts
        .into_iter()
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn maybe_spill(output: String) -> String {
    if !needs_spill(&output) {
        return output;
    }
    let path = spill_path();
    let result = write_file(&path, output.as_bytes())
        .map(|()| path)
        .map_err(|error| error.to_string());
    format_spill(&output, result)
}

pub(crate) fn spill_path() -> PathBuf {
    Path::new(SPILL_DIR).join(format!("{}.txt", uuid::Uuid::new_v4().simple()))
}

pub(crate) fn spill_error(error: ToolError) -> String {
    match error {
        ToolError::Execution(message) => message,
        error => error.to_string(),
    }
}

pub(crate) fn needs_spill(output: &str) -> bool {
    output.chars().count() > SPILL_THRESHOLD_CHARS
}

pub(crate) fn format_spill(output: &str, result: Result<PathBuf, String>) -> String {
    let output_chars = output.chars().count();
    match result {
        Ok(path) => format!(
            "<persisted-output>\nOutput too large ({}). Full output saved to: {}\nPreview ({}):\n{}\n</persisted-output>",
            human_size(output_chars),
            path.display(),
            human_size(PREVIEW_CHARS),
            preview(output, PREVIEW_CHARS)
        ),
        Err(error) => format!(
            "Output too large ({}), and saving it failed: {error}. Only a preview is shown; re-run a narrower command to see the rest.\n{}",
            human_size(output_chars),
            preview(output, PREVIEW_CHARS)
        ),
    }
}

fn human_size(chars: usize) -> String {
    if chars < 1024 {
        format!("{chars} bytes")
    } else if chars < 1024 * 1024 {
        format!("{:.1}KB", chars as f64 / 1024.0)
    } else {
        format!("{:.1}MB", chars as f64 / (1024.0 * 1024.0))
    }
}

fn preview(content: &str, max_chars: usize) -> String {
    let content_chars = content.chars().count();
    if content_chars <= max_chars {
        return content.to_string();
    }
    let head_budget = max_chars / 2;
    let tail_budget = max_chars - head_budget;
    let mut head = content.chars().take(head_budget).collect::<String>();
    if let Some(last_newline) = head.rfind('\n') {
        if head[..last_newline].chars().count() > head_budget / 2 {
            head.truncate(last_newline);
        }
    }
    let mut tail_chars = content.chars().rev().take(tail_budget).collect::<Vec<_>>();
    tail_chars.reverse();
    let mut tail = tail_chars.into_iter().collect::<String>();
    if let Some(first_newline) = tail.find('\n') {
        if tail[..first_newline].chars().count() < tail_budget / 2 {
            tail = tail[first_newline + 1..].to_string();
        }
    }
    let head_chars = head.chars().count();
    let tail_chars = tail.chars().count();
    let omitted = content
        .chars()
        .skip(head_chars)
        .take(content_chars - head_chars - tail_chars)
        .filter(|character| *character == '\n')
        .count()
        + 1;
    format!("{head}\n\n... [{omitted} lines omitted] ...\n\n{tail}")
}

#[cfg(test)]
#[path = "bash_tool_test.rs"]
mod tests;

const TASK_DISPLAY_INTERVAL: Duration = Duration::from_millis(100);

fn write_file(path: &Path, content: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    if let Err(error) = file.write_all(content) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

struct TaskDisplay {
    context: TaskContext,
    pending: Option<String>,
    updated_at: Option<Instant>,
}

impl TaskDisplay {
    fn new(context: TaskContext) -> Self {
        Self {
            context,
            pending: None,
            updated_at: None,
        }
    }

    fn add(&mut self, chunk: &[u8]) {
        let Some(display) = output_display(chunk) else {
            return;
        };
        self.pending = Some(display);
        let now = Instant::now();
        if self
            .updated_at
            .is_none_or(|updated_at| now.duration_since(updated_at) >= TASK_DISPLAY_INTERVAL)
        {
            self.flush(now);
        }
    }

    fn flush(&mut self, now: Instant) {
        if let Some(display) = self.pending.take() {
            self.context.update(Some(display));
            self.updated_at = Some(now);
        }
    }
}

impl Drop for TaskDisplay {
    fn drop(&mut self) {
        self.flush(Instant::now());
    }
}

async fn run_process(
    command_text: &str,
    workspace_root: &Path,
    timeout: Option<Duration>,
    task_context: Option<TaskContext>,
) -> Result<BashOutput, ToolError> {
    let mut std_command = std::process::Command::new("/bin/bash");
    for name in REMOVED_ENVIRONMENT {
        std_command.env_remove(name);
    }
    std_command
        .envs(COMMAND_ENVIRONMENT)
        .args(["-c", command_text])
        .current_dir(workspace_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    std_command.process_group(0);
    let mut command = Command::from(std_command);
    command.kill_on_drop(true);
    let child = command
        .spawn()
        .map_err(|error| ToolError::Execution(error.to_string()))?;
    let mut process_group = ProcessGroup::new(child);
    let child = process_group.child();
    let stdout = child.stdout.take().expect("piped stdout is present");
    let stderr = child.stderr.take().expect("piped stderr is present");
    let (stdout_sink, stdout_capture) = StreamCapture::spawn(MAX_OUTPUT_CHARS, Vec::new());
    let (stderr_sink, stderr_capture) =
        StreamCapture::spawn(MAX_OUTPUT_CHARS, sandbox_error_needles());
    let execution = async {
        let mut readers = std::pin::pin!(futures_util::future::try_join(
            read_output(stdout, stdout_sink, task_context.clone()),
            read_output(stderr, stderr_sink, task_context),
        ));
        let exited = tokio::select! {
            status = child.wait() => Some(status?),
            read = &mut readers => {
                read?;
                None
            }
        };
        let status = match exited {
            Some(status) => {
                if let Ok(read) = tokio::time::timeout(PIPE_DRAIN_GRACE, readers).await {
                    read?;
                }
                status
            }
            None => child.wait().await?,
        };
        Ok::<_, std::io::Error>(process_return_code(status))
    };
    let result = if let Some(timeout) = timeout {
        tokio::time::timeout(timeout, execution).await.ok()
    } else {
        Some(execution.await)
    };
    let (return_code, timed_out_after) = match result {
        Some(result) => {
            let return_code = result.map_err(|error| ToolError::Execution(error.to_string()))?;
            process_group.disarm();
            (return_code, None)
        }
        None => {
            process_group
                .terminate()
                .await
                .map_err(|error| ToolError::Execution(error.to_string()))?;
            (-1, timeout)
        }
    };
    Ok(BashOutput {
        return_code,
        stdout: finished(stdout_capture).await?,
        stderr: finished(stderr_capture).await?,
        timed_out_after,
        elapsed: Duration::ZERO,
        alerts: Vec::new(),
    })
}

async fn finished(
    capture: tokio::task::JoinHandle<CapturedStream>,
) -> Result<CapturedStream, ToolError> {
    capture
        .await
        .map_err(|error| ToolError::Execution(format!("capturing output failed: {error}")))
}

fn process_return_code(status: ExitStatus) -> i32 {
    status
        .code()
        .or_else(|| status.signal().map(|_| -1))
        .unwrap_or(2)
}

async fn read_output(
    mut reader: impl AsyncRead + Unpin,
    sink: mpsc::Sender<Vec<u8>>,
    task_context: Option<TaskContext>,
) -> std::io::Result<()> {
    let mut display = task_context.map(TaskDisplay::new);
    let mut buffer = vec![0; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(());
        }
        if let Some(display) = &mut display {
            display.add(&buffer[..read]);
        }
        sink.send(buffer[..read].to_vec())
            .await
            .map_err(|_| std::io::Error::other("the output capture stopped"))?;
    }
}

fn output_display(chunk: &[u8]) -> Option<String> {
    let decoded = String::from_utf8_lossy(chunk);
    decoded
        .split(['\n', '\r'])
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
}
