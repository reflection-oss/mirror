//! Workspace text-file creation and overwrite.

use std::io;
use std::path::Path;
use std::path::PathBuf;

use mirror_protocol::TouchedKind;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::resolve_for_write;
use super::touched;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// Create or overwrite UTF-8 workspace files.
#[derive(Debug, Clone)]
pub struct WriteTool {
    workspace_root: Option<PathBuf>,
}

impl WriteTool {
    /// Create a writer rooted at a canonical workspace directory.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    /// Create a remote-only file writer.
    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

impl Tool for WriteTool {
    type Args = WriteArgs;

    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write a UTF-8 text file in the workspace, overwriting it if it exists and creating parent directories as needed."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "file_path".to_string(),
                    serde_json::json!({"type": "string", "description": "Path to the file to write, relative to the workspace root, or absolute"}),
                ),
                (
                    "content".to_string(),
                    serde_json::json!({"type": "string", "description": "Complete content to write to the file"}),
                ),
            ]),
            &["file_path", "content"],
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
        args: WriteArgs,
        context: &ExecutionContext,
        remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        remote
            .run_write(&args, context.tool_permissions().allow_outside_workspace())
            .await
    }

    async fn run(
        &self,
        args: WriteArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let workspace_root = self.workspace_root.as_ref().ok_or_else(|| {
            ToolError::Execution("remote-only write tool cannot execute locally".into())
        })?;
        let path = resolve_for_write(
            workspace_root,
            &args.file_path,
            context.tool_permissions().allow_outside_workspace(),
        )
        .await?;
        let before = match tokio::fs::read(&path).await {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(super::execution_error(error)),
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(super::execution_error)?;
        }
        tokio::fs::write(&path, &args.content)
            .await
            .map_err(super::execution_error)?;
        Ok(format_result(&before, &args).touching(touched(
            workspace_root,
            &path,
            TouchedKind::Write,
        )))
    }
}

pub(super) fn format_result(before: &str, args: &WriteArgs) -> ToolOutput {
    let diff = super::diff::unified_diff(before, &args.content, &args.file_path);
    let written = args.content.lines().count();
    let mut result = ToolOutput::success(format!("Wrote {written} line(s) to {}.", args.file_path));
    if let Some(display) = super::diff::diff_display(&diff) {
        result = result.with_display(display);
    }
    result
}

/// Arguments accepted by the file-writing tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WriteArgs {
    pub file_path: String,
    pub content: String,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use mirror_protocol::TouchedPath;

    use super::*;

    #[tokio::test]
    async fn creates_parent_directories_and_returns_diff_display() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let tool = WriteTool::new(workspace.path()).expect("write tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"nested/file.txt","content":"hello\n"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(result.content, "Wrote 1 line(s) to nested/file.txt.");
        assert_eq!(
            result.touched,
            vec![TouchedPath {
                path: "nested/file.txt".to_string(),
                kind: TouchedKind::Write as i32,
            }]
        );
        assert!(result
            .display
            .as_deref()
            .is_some_and(|diff| diff.contains("+hello")));
        assert_eq!(
            fs::read_to_string(workspace.path().join("nested/file.txt")).unwrap(),
            "hello\n"
        );
        assert!(Tool::needs_approval(&tool));
    }

    #[tokio::test]
    async fn rejects_creation_through_symlink_outside_workspace() {
        let parent = tempfile::tempdir().expect("temporary parent");
        let workspace = parent.path().join("workspace");
        let outside = parent.path().join("outside");
        fs::create_dir(&workspace).expect("create workspace");
        fs::create_dir(&outside).expect("create outside directory");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, workspace.join("escape")).expect("create symlink");
        let tool = WriteTool::new(&workspace).expect("write tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"escape/file.txt","content":"private"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(&result, ToolOutput { success: false, .. });
        assert_ne!(result.content, "Wrote 1 line(s) to escape/file.txt.");
        assert!(!outside.join("file.txt").exists());
    }

    #[tokio::test]
    async fn rejects_parent_escape_through_missing_directory() {
        let parent = tempfile::tempdir().expect("temporary parent");
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).expect("create workspace");
        let tool = WriteTool::new(&workspace).expect("write tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"missing/../../outside.txt","content":"private"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(&result, ToolOutput { success: false, .. });
        assert_ne!(
            result.content,
            "Wrote 1 line(s) to missing/../../outside.txt."
        );
        assert!(!parent.path().join("outside.txt").exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_dangling_symlink_outside_workspace() {
        let parent = tempfile::tempdir().expect("temporary parent");
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).expect("create workspace");
        let outside = parent.path().join("outside.txt");
        std::os::unix::fs::symlink(&outside, workspace.join("escape.txt")).expect("create symlink");
        let tool = WriteTool::new(&workspace).expect("write tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"escape.txt","content":"private"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(&result, ToolOutput { success: false, .. });
        assert_ne!(result.content, "Wrote 1 line(s) to escape.txt.");
        assert!(!outside.exists());
    }
    #[tokio::test]
    async fn remote_write_matches_local_content_diff_and_metadata() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let workspace = tempfile::tempdir().expect("workspace");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = WriteTool::new(workspace.path()).expect("local");
        let context = ExecutionContext::default();
        for content in [
            "",
            "hello\n",
            "界\r\nsecond",
            "same",
            "x".repeat(100_000).as_str(),
        ] {
            let path = workspace.path().join("data");
            fs::write(&path, b"before\xff\n").expect("before");
            let args = serde_json::json!({"file_path":"data", "content":content}).to_string();
            let expected = local.execute(&args, &context).await;
            let expected_bytes = fs::read(&path).expect("local bytes");
            fs::write(&path, b"before\xff\n").expect("reset");
            assert_eq!(
                WriteTool::remote()
                    .execute_remote(&args, &context, &remote)
                    .await,
                expected
            );
            assert_eq!(fs::read(&path).expect("remote bytes"), expected_bytes);
        }
        let args = r#"{"file_path":"nested/new","content":"new\n"}"#;
        let expected = local.execute(args, &context).await;
        fs::remove_dir_all(workspace.path().join("nested")).expect("reset");
        assert_eq!(
            WriteTool::remote()
                .execute_remote(args, &context, &remote)
                .await,
            expected
        );
        assert_eq!(
            fs::read_to_string(workspace.path().join("nested/new")).expect("new"),
            "new\n"
        );
        assert!(!WriteTool::remote().execute(args, &context).await.success);
    }

    #[tokio::test]
    async fn remote_write_matches_local_path_rejections_without_mutation() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let parent = tempfile::tempdir().expect("parent");
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).expect("workspace");
        fs::write(parent.path().join("outside"), "canary").expect("outside");
        std::os::unix::fs::symlink("../outside", workspace.join("escape")).expect("symlink");
        let server = TestServer::shell(&workspace).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = WriteTool::new(&workspace).expect("local");
        let context = ExecutionContext::default();
        for path in [".", "escape", "../outside", "bad\0path"] {
            let args = serde_json::json!({"file_path":path,"content":"replacement"}).to_string();
            assert_eq!(
                WriteTool::remote()
                    .execute_remote(&args, &context, &remote)
                    .await,
                local.execute(&args, &context).await,
                "{path}"
            );
        }
        assert_eq!(
            fs::read_to_string(parent.path().join("outside")).expect("unchanged"),
            "canary"
        );
    }
    #[tokio::test]
    async fn remote_write_preserves_symlinks_modes_and_host_files() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        use std::os::unix::fs::PermissionsExt;
        let host = tempfile::tempdir().expect("host");
        let workspace = tempfile::tempdir().expect("remote");
        fs::write(host.path().join("link"), "host canary").expect("host file");
        fs::write(workspace.path().join("data"), "old").expect("remote file");
        fs::set_permissions(
            workspace.path().join("data"),
            fs::Permissions::from_mode(0o640),
        )
        .expect("mode");
        std::os::unix::fs::symlink("data", workspace.path().join("link")).expect("link");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let tool = WriteTool::new(host.path()).expect("host-bound tool");
        let output = tool
            .execute_remote(
                r#"{"file_path":"link","content":"new"}"#,
                &ExecutionContext::default(),
                &remote,
            )
            .await;
        assert!(output.success, "{output:?}");
        assert_eq!(
            fs::read_to_string(host.path().join("link")).expect("host unchanged"),
            "host canary"
        );
        assert_eq!(
            fs::read_to_string(workspace.path().join("data")).expect("remote write"),
            "new"
        );
        assert!(fs::symlink_metadata(workspace.path().join("link"))
            .expect("link")
            .is_symlink());
        assert_eq!(
            fs::metadata(workspace.path().join("data"))
                .expect("mode")
                .permissions()
                .mode()
                & 0o777,
            0o640
        );
        assert!(AnyTool::needs_approval(&WriteTool::remote()));
    }
}
