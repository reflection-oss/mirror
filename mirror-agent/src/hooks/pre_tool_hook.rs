//! Command hooks loaded only from trusted operator configuration.

use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use mirror_protocol::ToolCall;
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

use super::hook_config;
use crate::tools::ProcessGroup;

/// A command reviewer executed by the agent runtime, independently of clients.
#[derive(Debug, Clone)]
pub struct PreToolHook {
    matcher: Option<Regex>,
    command: String,
    timeout: Duration,
    cwd: PathBuf,
    plugin_root: Option<PathBuf>,
}

/// Hook failures stop execution without exposing subprocess diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PreToolError {
    #[error("invalid pre-tool hook configuration: {0}")]
    Configuration(&'static str),
    #[error("pre-tool hook timed out")]
    TimedOut,
    #[error("pre-tool hook failed")]
    Failed,
    #[error("invalid pre-tool hook output: {0}")]
    Output(&'static str),
    #[error("pre-tool hook output exceeded its limit")]
    OutputTooLarge,
}

#[derive(Serialize)]
struct HookInput<'a> {
    hook_event_name: &'static str,
    cwd: &'a Path,
    tool_name: &'a str,
    tool_use_id: &'a str,
    tool_input: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct HookOutput {
    #[serde(rename = "continue")]
    continue_processing: Option<bool>,
    #[serde(rename = "suppressOutput")]
    _suppress_output: Option<bool>,
    hook_specific_output: Option<HookDecision>,
}

#[derive(Deserialize)]
#[serde(tag = "hookEventName", deny_unknown_fields)]
enum HookDecision {
    PreToolUse {
        #[serde(rename = "permissionDecision")]
        decision: Option<Deny>,
        #[serde(rename = "permissionDecisionReason")]
        reason: Option<String>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Deny {
    Deny,
}

pub(super) const MAX_HOOK_BYTES: u64 = 64 * 1024;

impl PreToolHook {
    /// Load user hooks, never discovering executable configuration in a checkout.
    pub async fn load_user(cwd: &Path) -> Result<Vec<Self>, PreToolError> {
        let Some(home) = std::env::var_os("HOME") else {
            return Ok(Vec::new());
        };
        Self::load(&PathBuf::from(home).join(".mirror/hooks.json"), cwd).await
    }

    /// Load a trusted hook file once; absence means no hooks, invalid data is fatal.
    pub async fn load(path: &Path, cwd: &Path) -> Result<Vec<Self>, PreToolError> {
        let file = match tokio::fs::File::open(path).await {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(_) => return Err(PreToolError::Configuration("cannot read file")),
        };
        let bytes = read_bounded(file).await?;
        hook_config::load(&bytes, path, cwd)
    }

    pub(super) fn new(
        matcher: Option<Regex>,
        command: String,
        timeout: Duration,
        cwd: &Path,
        plugin_root: Option<&Path>,
    ) -> Self {
        Self {
            matcher,
            command,
            timeout,
            cwd: cwd.into(),
            plugin_root: plugin_root.map(PathBuf::from),
        }
    }

    pub(crate) async fn review(&self, call: &ToolCall) -> Result<Option<String>, PreToolError> {
        let function = call.function.as_ref().ok_or(PreToolError::Failed)?;
        let native_name = function.name.as_str();
        let mut tool_name = match native_name {
            "bash" => "Bash",
            "read" => "Read",
            "write" => "Write",
            "edit" => "Edit",
            "grep" => "Grep",
            name => name,
        };
        if let Some(matcher) = &self.matcher {
            if !matcher.is_match(tool_name) {
                if !matcher.is_match(native_name) {
                    return Ok(None);
                }
                tool_name = native_name;
            }
        }
        let input = serde_json::to_vec(&HookInput {
            hook_event_name: "PreToolUse",
            cwd: &self.cwd,
            tool_name,
            tool_use_id: &call.id,
            tool_input: if function.arguments.trim().is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&function.arguments).map_err(|_| PreToolError::Failed)?
            },
        })
        .map_err(|_| PreToolError::Failed)?;
        self.run(&input).await
    }

    async fn run(&self, input: &[u8]) -> Result<Option<String>, PreToolError> {
        let mut command = std::process::Command::new("/bin/sh");
        command
            .args(["-c", &self.command])
            .current_dir(&self.cwd)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        // Hooks receive tool input, not the worker's provider credentials or shell startup environment.
        if let Some(path) = std::env::var_os("PATH") {
            command.env("PATH", path);
        }
        if let Some(root) = &self.plugin_root {
            command
                .env("CLAUDE_PLUGIN_ROOT", root)
                .env("PLUGIN_ROOT", root);
        }
        let child = tokio::process::Command::from(command)
            .kill_on_drop(true)
            .spawn()
            .map_err(|_| PreToolError::Failed)?;
        let mut group = ProcessGroup::new(child);
        let child = group.child();
        let mut stdin = child.stdin.take().ok_or(PreToolError::Failed)?;
        let stdout = child.stdout.take().ok_or(PreToolError::Failed)?;
        let stderr = child.stderr.take().ok_or(PreToolError::Failed)?;
        let execution = async {
            let (_, stdout, stderr, status) = tokio::try_join!(
                async {
                    match stdin.write_all(input).await {
                        Ok(()) => {}
                        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {}
                        Err(_) => return Err(PreToolError::Failed),
                    }
                    drop(stdin);
                    Ok(())
                },
                read_bounded(stdout),
                read_bounded(stderr),
                async { child.wait().await.map_err(|_| PreToolError::Failed) },
            )?;
            parse_output(status.code(), &stdout, &stderr)
        };
        // Dropping the group also kills descendants on cancellation, timeout or malformed output.
        let result = tokio::time::timeout(self.timeout, execution)
            .await
            .map_err(|_| PreToolError::TimedOut)?;
        if result.is_ok() {
            group.disarm();
        }
        result
    }
}

fn parse_output(
    code: Option<i32>,
    stdout: &[u8],
    stderr: &[u8],
) -> Result<Option<String>, PreToolError> {
    match code {
        Some(0) => {
            let stdout = std::str::from_utf8(stdout)
                .map_err(|_| PreToolError::Output("stdout must be UTF-8"))?;
            // Claude-style plain-text success output carries no decision and is discarded.
            if !stdout.trim_start().starts_with('{') {
                return Ok(None);
            }
            let output: HookOutput = serde_json::from_str(stdout).map_err(|error| {
                PreToolError::Output(if error.is_data() {
                    "expected only continue:true, suppressOutput and hookSpecificOutput (PreToolUse deny); other fields or decisions are unsupported"
                } else {
                    "stdout starts with '{' but is not valid JSON"
                })
            })?;
            if output.continue_processing == Some(false) {
                return Err(PreToolError::Output(
                    "continue:false is unsupported; use exit 2 or permissionDecision:deny to reject a tool call",
                ));
            }
            match output.hook_specific_output {
                None => Ok(None),
                Some(HookDecision::PreToolUse { decision, reason }) => match (decision, reason) {
                    (Some(Deny::Deny), reason) => {
                        Ok(Some(denial(reason.as_deref().unwrap_or_default())))
                    }
                    (None, None) => Ok(None),
                    (None, Some(_)) => Err(PreToolError::Output(
                        "permissionDecisionReason requires permissionDecision:deny",
                    )),
                },
            }
        }
        Some(2) => Ok(Some(denial(&String::from_utf8_lossy(stderr)))),
        _ => Err(PreToolError::Failed),
    }
}

fn denial(reason: &str) -> String {
    if reason.trim().is_empty() {
        "Tool call rejected.".into()
    } else {
        reason.trim().into()
    }
}

async fn read_bounded(reader: impl AsyncRead + Unpin) -> Result<Vec<u8>, PreToolError> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_HOOK_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|_| PreToolError::Failed)?;
    if bytes.len() as u64 > MAX_HOOK_BYTES {
        return Err(PreToolError::OutputTooLarge);
    }
    Ok(bytes)
}
