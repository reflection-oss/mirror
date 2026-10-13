//! Workspace directory listing.

use std::path::Path;
use std::path::PathBuf;

use mirror_protocol::TouchedKind;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::resolve_existing;
use super::touched;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// List entries in a workspace directory.
#[derive(Debug, Clone)]
pub struct LsTool {
    workspace_root: Option<PathBuf>,
}

impl LsTool {
    /// Create a directory lister rooted at a canonical workspace directory.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    /// Create a remote-only directory lister.
    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

impl Tool for LsTool {
    type Args = LsArgs;

    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List a directory; entries ending in / are subdirectories."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([(
                "path".to_string(),
                serde_json::json!({"type": "string", "default": ".", "description": "Directory to list, relative to the workspace root, or absolute"}),
            )]),
            &[],
        )
    }

    fn supports_remote(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: LsArgs,
        context: &ExecutionContext,
        remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        remote
            .run_ls(
                &args.path,
                context.tool_permissions().allow_outside_workspace(),
            )
            .await
    }

    async fn run(&self, args: LsArgs, context: &ExecutionContext) -> Result<ToolOutput, ToolError> {
        let workspace_root = self.workspace_root.as_ref().ok_or_else(|| {
            ToolError::Execution("remote-only ls tool cannot execute locally".into())
        })?;
        let directory = resolve_existing(
            workspace_root,
            &args.path,
            context.tool_permissions().allow_outside_workspace(),
        )
        .await?;
        if !tokio::fs::metadata(&directory)
            .await
            .map_err(super::execution_error)?
            .is_dir()
        {
            return Ok(ToolOutput::error(format!(
                "{} is not a directory.",
                args.path
            )));
        }
        let touched = touched(workspace_root, &directory, TouchedKind::List);
        let mut directory = tokio::fs::read_dir(directory)
            .await
            .map_err(super::execution_error)?;
        let mut entries = Vec::new();
        while let Some(entry) = directory
            .next_entry()
            .await
            .map_err(super::execution_error)?
        {
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry
                .file_type()
                .await
                .map_err(super::execution_error)?
                .is_dir()
            {
                name.push('/');
            }
            entries.push(name);
        }
        Ok(format_entries(entries).touching(touched))
    }
}

pub(super) fn format_entries(mut entries: Vec<String>) -> ToolOutput {
    entries.sort();
    ToolOutput::success(if entries.is_empty() {
        "(empty directory)".to_string()
    } else {
        entries.join("\n")
    })
}

/// Arguments accepted by the directory-listing tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LsArgs {
    #[serde(default = "default_path")]
    pub path: String,
}

fn default_path() -> String {
    ".".to_string()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use mirror_protocol::TouchedPath;

    use super::*;

    #[tokio::test]
    async fn lists_sorted_files_and_marks_directories() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("z.txt"), "z").expect("write fixture");
        fs::create_dir(workspace.path().join("a")).expect("create fixture directory");
        let tool = LsTool::new(workspace.path()).expect("ls tool");

        let result =
            crate::tools::AnyTool::execute(&tool, "{}", &ExecutionContext::default()).await;

        assert_matches!(&result, ToolOutput { success: true, .. });
        assert_eq!(
            result.touched,
            vec![TouchedPath {
                path: ".".to_string(),
                kind: TouchedKind::List as i32,
            }]
        );
        assert_ne!(result.content, "z.txt\na/");
        assert_eq!(result.content, "a/\nz.txt");
    }

    #[tokio::test]
    async fn reports_file_as_not_a_directory() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("file.txt"), "text").expect("write fixture");
        let tool = LsTool::new(workspace.path()).expect("ls tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"path":"file.txt"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(result, ToolOutput::error("file.txt is not a directory."));
    }
    #[tokio::test]
    async fn remote_ls_matches_local_entries_errors_and_permissions() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let parent = tempfile::tempdir().expect("parent");
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).expect("workspace");
        fs::create_dir(workspace.join("empty")).expect("empty");
        fs::write(workspace.join("z.txt"), "text").expect("file");
        fs::write(workspace.join(".hidden界"), "text").expect("hidden");
        std::os::unix::fs::symlink("empty", workspace.join("directory-link")).expect("symlink");
        std::os::unix::fs::symlink("missing", workspace.join("broken")).expect("broken symlink");
        let server = TestServer::shell(&workspace).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = LsTool::new(&workspace).expect("local");
        let context = ExecutionContext::default();
        for path in [
            ".",
            "empty",
            "directory-link",
            "z.txt",
            "missing",
            "broken",
            "..",
            "bad\0path",
        ] {
            let args = serde_json::json!({"path":path}).to_string();
            assert_eq!(
                LsTool::remote()
                    .execute_remote(&args, &context, &remote)
                    .await,
                local.execute(&args, &context).await,
                "{path}"
            );
        }
        context
            .input()
            .set_approval_mode(crate::AgentApprovalMode::Auto);
        assert_eq!(
            LsTool::remote()
                .execute_remote(r#"{"path":".."}"#, &context, &remote)
                .await,
            local.execute(r#"{"path":".."}"#, &context).await
        );
        assert!(!LsTool::remote().execute("{}", &context).await.success);
    }
}
