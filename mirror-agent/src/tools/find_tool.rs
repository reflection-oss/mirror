//! Workspace file-pattern discovery.

use std::path::Path;
use std::path::PathBuf;

use mirror_protocol::TouchedKind;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::relative_display;
use super::resolve_existing;
use super::touched;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// Find files under a workspace by glob pattern.
#[derive(Debug, Clone)]
pub struct FindTool {
    workspace_root: Option<PathBuf>,
}

impl FindTool {
    /// Create a finder rooted at a canonical workspace directory.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    /// Create a remote-only file finder.
    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

impl Tool for FindTool {
    type Args = FindArgs;

    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Fast file pattern matching using ripgrep, including hidden and gitignored files while excluding VCS and cache directories."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "pattern".to_string(),
                    serde_json::json!({"type": "string", "description": "Glob matched against file paths under path, such as \"*.rs\" or \"src/**/*.ts\". Returns up to 100 files, newest first."}),
                ),
                (
                    "path".to_string(),
                    serde_json::json!({"type": ["string", "null"], "description": "Directory to search, relative to the workspace root, or absolute. Defaults to the workspace root."}),
                ),
            ]),
            &["pattern"],
        )
    }

    fn supports_remote(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: FindArgs,
        context: &ExecutionContext,
        remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        let path = normalize_path(args.path.as_deref())?.unwrap_or(".");
        let resolved = remote
            .resolve_path(path, context.tool_permissions().allow_outside_workspace())
            .await?;
        if !resolved.is_dir {
            return Ok(ToolOutput::error(format!("Path is not a directory: {path}. Glob searches directories only; pass the parent directory and use pattern to filter files.")));
        }
        let result = remote
            .run_rg(&arguments(&args.pattern, &resolved.path), &resolved.root)
            .await?;
        let output = render_result(result, &resolved.root)?;
        if !output.success {
            return Ok(output);
        }
        Ok(output.touching(touched(&resolved.root, &resolved.path, TouchedKind::List)))
    }

    async fn run(
        &self,
        args: FindArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let workspace_root = self.workspace_root.as_ref().ok_or_else(|| {
            ToolError::Execution("remote-only glob tool cannot execute locally".into())
        })?;
        let path = normalize_path(args.path.as_deref())?;
        let search_root = resolve_existing(
            workspace_root,
            path.unwrap_or("."),
            context.tool_permissions().allow_outside_workspace(),
        )
        .await?;
        if !tokio::fs::metadata(&search_root)
            .await
            .map_err(super::execution_error)?
            .is_dir()
        {
            return Ok(ToolOutput::error(format!(
                "Path is not a directory: {}. Glob searches directories only; pass the parent directory and use pattern to filter files.",
                path.unwrap_or(".")
            )));
        }
        let arguments = arguments(&args.pattern, &search_root);
        let result = super::ripgrep::run(&arguments, workspace_root)
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        let output = render_result(result, workspace_root)?;
        if !output.success {
            return Ok(output);
        }
        Ok(output.touching(touched(workspace_root, &search_root, TouchedKind::List)))
    }
}

fn arguments(pattern: &str, search_root: &Path) -> Vec<String> {
    let mut arguments = vec![
        "--files".to_string(),
        "--hidden".to_string(),
        "--no-ignore".to_string(),
        "--sortr=modified".to_string(),
    ];
    arguments.extend(super::ripgrep::exclusion_arguments());
    arguments.extend([
        "--glob".to_string(),
        pattern.to_string(),
        "--".to_string(),
        search_root.display().to_string(),
    ]);
    arguments
}

/// Arguments accepted by the file-pattern tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FindArgs {
    pub pattern: String,
    pub path: Option<String>,
}

const MAX_RESULTS: usize = 100;
const MAX_RAW_OUTPUT_CHARS: usize = 1_000_000;

fn render_result(
    result: super::ripgrep::RgResult,
    workspace_root: &Path,
) -> Result<ToolOutput, ToolError> {
    let raw_output_truncated = result.stdout.chars().count() > MAX_RAW_OUTPUT_CHARS;
    let mut output = if raw_output_truncated {
        result.stdout.chars().take(MAX_RAW_OUTPUT_CHARS).collect()
    } else {
        result.stdout
    };
    if (raw_output_truncated || result.timed_out) && !output.ends_with('\n') {
        output = output
            .rsplit_once('\n')
            .map_or(String::new(), |(head, _)| head.to_string());
    }
    if !matches!(result.return_code, 0 | 1) && !result.timed_out {
        let detail = if result.stderr.trim().is_empty() {
            output.trim()
        } else {
            result.stderr.trim()
        };
        return Ok(ToolOutput::error(format!(
            "Glob error (exit {}): {}",
            result.return_code,
            detail.chars().take(500).collect::<String>()
        )));
    }
    let mut files = output
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| relative_display(Path::new(line), workspace_root))
        .collect::<Vec<_>>();
    if result.timed_out && files.is_empty() {
        return Ok(ToolOutput::success(
            "Glob search timed out with no results. Try a narrower pattern or smaller path.",
        ));
    }
    let omitted = files.len().saturating_sub(MAX_RESULTS);
    files.truncate(MAX_RESULTS);
    if files.is_empty() {
        return Ok(ToolOutput::success("No files matched."));
    }
    if omitted > 0 {
        files.push(format!("... and {omitted} more files omitted"));
    }
    if raw_output_truncated {
        files.push(
            "... and additional files may be omitted because raw output was truncated".to_string(),
        );
    }
    if result.timed_out {
        files.push(
            "... and additional files may be omitted because the search timed out".to_string(),
        );
    }
    Ok(ToolOutput::success(files.join("\n")))
}

fn normalize_path(path: Option<&str>) -> Result<Option<&str>, ToolError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if path.contains('\0') {
        return Err(ToolError::InvalidArguments(
            "Invalid path: contains null bytes.".to_string(),
        ));
    }
    let path = path.trim();
    if path.is_empty() || matches!(path.to_ascii_lowercase().as_str(), "null" | "undefined") {
        Ok(None)
    } else {
        Ok(Some(path))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use mirror_protocol::TouchedPath;

    use super::*;

    #[tokio::test]
    async fn finds_hidden_and_ignored_files_but_excludes_vcs_data() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir_all(workspace.path().join(".hidden")).expect("create hidden directory");
        fs::create_dir_all(workspace.path().join(".git")).expect("create VCS directory");
        fs::write(workspace.path().join(".hidden/match.rs"), "match").expect("write fixture");
        fs::write(workspace.path().join(".git/ignored.rs"), "ignored").expect("write fixture");
        let tool = FindTool::new(workspace.path()).expect("glob tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"pattern":"**/*.rs"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(&result, ToolOutput { success: true, .. });
        assert_eq!(
            result.touched,
            vec![TouchedPath {
                path: ".".to_string(),
                kind: TouchedKind::List as i32,
            }]
        );
        assert_ne!(result.content, ".git/ignored.rs");
        assert_eq!(result.content, ".hidden/match.rs");
    }

    #[tokio::test]
    async fn treats_model_path_sentinels_as_omitted() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("match.txt"), "match").expect("write fixture");
        let tool = FindTool::new(workspace.path()).expect("glob tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"pattern":"*.txt","path":"undefined"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(result.content, "match.txt");
    }
    #[tokio::test]
    async fn remote_glob_matches_local_patterns_exclusions_and_errors() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let workspace = tempfile::tempdir().expect("workspace");
        fs::create_dir_all(workspace.path().join(".git")).expect("git");
        fs::create_dir_all(workspace.path().join(".hidden")).expect("hidden");
        fs::write(workspace.path().join(".git/ignored.rs"), "ignored").expect("file");
        fs::write(workspace.path().join(".hidden/file.rs"), "hidden").expect("file");
        fs::write(workspace.path().join("space '界.rs"), "text").expect("file");
        fs::write(workspace.path().join(".gitignore"), "*.rs").expect("ignore");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = FindTool::new(workspace.path()).expect("local");
        let context = ExecutionContext::default();
        for args in [
            serde_json::json!({"pattern":"**/*.rs"}),
            serde_json::json!({"pattern":"*", "path":".hidden"}),
            serde_json::json!({"pattern":"*.missing"}),
            serde_json::json!({"pattern":"["}),
            serde_json::json!({"pattern":"*", "path":" null "}),
            serde_json::json!({"pattern":"*", "path":".gitignore"}),
            serde_json::json!({"pattern":"*", "path":"missing"}),
            serde_json::json!({"pattern":"*", "path":".."}),
            serde_json::json!({"pattern":"*", "path":"bad\0path"}),
        ] {
            let args = args.to_string();
            assert_eq!(
                FindTool::remote()
                    .execute_remote(&args, &context, &remote)
                    .await,
                local.execute(&args, &context).await,
                "{args}"
            );
        }
        assert!(
            !FindTool::remote()
                .execute(r#"{"pattern":"*"}"#, &context)
                .await
                .success
        );
    }
}
