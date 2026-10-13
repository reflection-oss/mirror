//! Workspace text-file reading.

use std::path::Path;
use std::path::PathBuf;

use mirror_protocol::TouchedKind;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use tokio::io::AsyncReadExt;

use super::object_schema;
use super::resolve_existing;
use super::touched;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// Read UTF-8 text files from a workspace.
#[derive(Debug, Clone)]
pub struct ReadTool {
    workspace_root: Option<PathBuf>,
}

impl ReadTool {
    /// Create a reader rooted at a canonical workspace directory.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    /// Create a remote-only reader without accessing the host workspace.
    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

impl Tool for ReadTool {
    type Args = ReadArgs;

    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file from the workspace."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "file_path".to_string(),
                    serde_json::json!({"type": "string", "description": "Path to the file to read, relative to the workspace root, or absolute"}),
                ),
                (
                    "offset".to_string(),
                    serde_json::json!({"type": ["integer", "null"], "minimum": 1, "description": "Line number to start reading from (1-indexed)"}),
                ),
                (
                    "limit".to_string(),
                    serde_json::json!({"type": ["integer", "null"], "minimum": 1, "description": "Maximum number of lines to read"}),
                ),
                (
                    "max_chars".to_string(),
                    serde_json::json!({"type": "integer", "minimum": 1, "maximum": 1_000_000, "default": 20_000, "description": "Maximum number of characters to return; longer output is truncated"}),
                ),
            ]),
            &["file_path"],
        )
    }

    fn supports_remote(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: ReadArgs,
        context: &ExecutionContext,
        remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        validate_arguments(&args)?;
        remote
            .run_read(&args, context.tool_permissions().allow_outside_workspace())
            .await
    }

    async fn run(
        &self,
        args: ReadArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        validate_arguments(&args)?;
        let workspace_root = self.workspace_root.as_ref().ok_or_else(|| {
            ToolError::Execution("remote-only read tool cannot execute locally".into())
        })?;
        let path = resolve_existing(
            workspace_root,
            &args.file_path,
            context.tool_permissions().allow_outside_workspace(),
        )
        .await?;
        let (content, exceeded_byte_cap) = read_selected_bytes(&path, &args).await?;
        let touched = touched(workspace_root, &path, TouchedKind::Read);
        Ok(format_content(content, exceeded_byte_cap, &args).touching(touched))
    }
}

pub(super) fn format_content(
    content: Vec<u8>,
    exceeded_byte_cap: bool,
    args: &ReadArgs,
) -> ToolOutput {
    let content = String::from_utf8_lossy(&content).into_owned();
    let truncated = exceeded_byte_cap || content.chars().count() > args.max_chars;
    if !truncated {
        return ToolOutput::success(content);
    }
    let content = content.chars().take(args.max_chars).collect::<String>();
    ToolOutput::success(format!(
        "{content}\n...[truncated at {} chars — use offset/limit to read more]",
        args.max_chars
    ))
}

/// Arguments accepted by the read tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadArgs {
    pub file_path: String,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
}

fn default_max_chars() -> usize {
    20_000
}

fn validate_arguments(args: &ReadArgs) -> Result<(), ToolError> {
    if args.offset == Some(0) || args.limit == Some(0) || !(1..=1_000_000).contains(&args.max_chars)
    {
        return Err(ToolError::InvalidArguments(
            "offset, limit, and max_chars must be positive; max_chars cannot exceed 1000000"
                .to_string(),
        ));
    }
    Ok(())
}

async fn read_selected_bytes(path: &Path, args: &ReadArgs) -> Result<(Vec<u8>, bool), ToolError> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(super::execution_error)?;
    let first_line = args.offset.unwrap_or(1);
    let end_line = args.limit.map(|limit| first_line.saturating_add(limit));
    let byte_cap = args.max_chars.saturating_mul(4).saturating_add(4);
    let mut selected = Vec::with_capacity(byte_cap.min(64 * 1024));
    let mut buffer = [0_u8; 8192];
    let mut line = 1_usize;
    let mut exceeded_byte_cap = false;
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(super::execution_error)?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            if line >= first_line && end_line.is_none_or(|end| line < end) {
                if selected.len() < byte_cap {
                    selected.push(*byte);
                } else {
                    exceeded_byte_cap = true;
                    return Ok((selected, exceeded_byte_cap));
                }
            }
            if *byte == b'\n' {
                line += 1;
                if end_line.is_some_and(|end| line >= end) {
                    return Ok((selected, exceeded_byte_cap));
                }
            }
        }
    }
    Ok((selected, exceeded_byte_cap))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use mirror_protocol::TouchedPath;

    use super::*;

    #[tokio::test]
    async fn reads_selected_lines_and_truncates_characters() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("notes.txt"), "first\nsecond\nthird\n")
            .expect("write fixture");
        let tool = ReadTool::new(workspace.path()).expect("read tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"notes.txt","offset":2,"limit":2,"max_chars":8}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(&result, ToolOutput { success: true, .. });
        assert_eq!(
            result.touched,
            vec![TouchedPath {
                path: "notes.txt".to_string(),
                kind: TouchedKind::Read as i32,
            }]
        );
        assert_ne!(result.content, "second\nthird\n");
        assert_eq!(
            result.content,
            "second\nt\n...[truncated at 8 chars — use offset/limit to read more]"
        );
    }

    #[tokio::test]
    async fn rejects_workspace_escape() {
        let parent = tempfile::tempdir().expect("temporary parent");
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).expect("create workspace");
        fs::write(parent.path().join("outside"), "private").expect("write outside fixture");
        let tool = ReadTool::new(&workspace).expect("read tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"../outside"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("outside workspace"));
    }

    #[tokio::test]
    async fn remote_read_matches_local_content_and_metadata() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};

        let workspace = tempfile::tempdir().expect("workspace");
        fs::write(
            workspace.path().join("text"),
            b"\nfirst\r\n\xe7\x95\x8c\xff\nlast",
        )
        .expect("fixture");
        fs::write(workspace.path().join("empty"), "").expect("empty");
        std::os::unix::fs::symlink("text", workspace.path().join("link")).expect("symlink");
        std::os::unix::fs::symlink("loop", workspace.path().join("loop")).expect("symlink loop");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = ReadTool::new(workspace.path()).expect("reader");
        let context = ExecutionContext::default();
        for arguments in [
            serde_json::json!({"file_path":"text"}),
            serde_json::json!({"file_path":"text", "offset":2, "limit":2}),
            serde_json::json!({"file_path":"text", "offset":3, "max_chars":1}),
            serde_json::json!({"file_path":"text", "offset":99}),
            serde_json::json!({"file_path":"text", "limit":1}),
            serde_json::json!({"file_path":"empty"}),
            serde_json::json!({"file_path":"link"}),
            serde_json::json!({"file_path":workspace.path().join("text")}),
            serde_json::json!({"file_path":"missing"}),
            serde_json::json!({"file_path":"loop"}),
            serde_json::json!({"file_path":"."}),
            serde_json::json!({"file_path":"bad\0path"}),
        ] {
            let arguments = arguments.to_string();
            assert_eq!(
                ReadTool::remote()
                    .execute_remote(&arguments, &context, &remote)
                    .await,
                local.execute(&arguments, &context).await,
                "{arguments}",
            );
        }
    }

    #[tokio::test]
    async fn remote_read_matches_local_large_output_and_byte_cap() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};

        let workspace = tempfile::tempdir().expect("workspace");
        fs::write(workspace.path().join("large"), "界".repeat(1_400_000)).expect("fixture");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = ReadTool::new(workspace.path()).expect("reader");
        let context = ExecutionContext::default();
        let arguments = r#"{"file_path":"large","max_chars":1000000}"#;
        assert_eq!(
            ReadTool::remote()
                .execute_remote(arguments, &context, &remote)
                .await,
            local.execute(arguments, &context).await,
        );
    }

    #[tokio::test]
    async fn remote_read_matches_local_escape_permissions() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};

        let parent = tempfile::tempdir().expect("parent");
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).expect("workspace");
        fs::write(parent.path().join("outside"), "outside contents").expect("fixture");
        std::os::unix::fs::symlink("../outside", workspace.join("escape")).expect("symlink");
        let server = TestServer::shell(&workspace).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = ReadTool::new(&workspace).expect("reader");
        let context = ExecutionContext::default();
        let arguments = r#"{"file_path":"escape"}"#;
        let denied = ReadTool::remote()
            .execute_remote(arguments, &context, &remote)
            .await;
        assert_eq!(denied, local.execute(arguments, &context).await);
        assert!(!denied.success);
        context
            .input()
            .set_approval_mode(crate::AgentApprovalMode::Auto);
        let allowed = ReadTool::remote()
            .execute_remote(arguments, &context, &remote)
            .await;
        assert_eq!(allowed, local.execute(arguments, &context).await);
        assert_eq!(allowed.content, "outside contents");
    }

    #[tokio::test]
    async fn remote_read_uses_only_remote_workspace() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor, ToolExecutor};

        let host = tempfile::tempdir().expect("host");
        let workspace = tempfile::tempdir().expect("remote");
        fs::write(host.path().join("data"), "HOST_SECRET_CANARY").expect("host fixture");
        fs::write(workspace.path().join("data"), "remote contents").expect("remote fixture");
        let mut server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            vec![std::sync::Arc::new(
                ReadTool::new(host.path()).expect("reader"),
            )],
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let result = remote
            .call_tool(
                "read".into(),
                r#"{"file_path":"data"}"#.into(),
                ExecutionContext::default(),
            )
            .await
            .expect("result");
        assert_eq!(result.content, "remote contents");
        let request = server.request().await;
        assert_eq!(request.authorization, None);
        assert!(!request
            .body
            .to_string()
            .contains(host.path().to_str().expect("path")));
        assert!(
            !ReadTool::remote()
                .execute(r#"{"file_path":"data"}"#, &ExecutionContext::default())
                .await
                .success
        );
    }

    #[tokio::test]
    async fn remote_read_validates_before_contacting_endpoint() {
        use crate::tools::{AnyTool, RemoteToolExecutor};

        let workspace = tempfile::tempdir().expect("workspace");
        let local = ReadTool::new(workspace.path()).expect("reader");
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse("http://localhost:1").expect("URL"),
        )
        .expect("executor");
        let context = ExecutionContext::default();
        for arguments in [
            r#"{"file_path":"missing","offset":0}"#,
            r#"{"file_path":"missing","limit":0}"#,
            r#"{"file_path":"missing","max_chars":1000001}"#,
            r#"{"file_path":"missing","max_chars":0}"#,
            r#"{"file_path":"missing","unknown":true}"#,
        ] {
            assert_eq!(
                ReadTool::remote()
                    .execute_remote(arguments, &context, &remote)
                    .await,
                local.execute(arguments, &context).await,
            );
        }
    }
}
