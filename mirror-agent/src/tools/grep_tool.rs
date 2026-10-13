//! Workspace content search.

use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;

use mirror_protocol::TouchedKind;
use regex::Regex;
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

/// Search file contents using ripgrep.
#[derive(Debug, Clone)]
pub struct GrepTool {
    workspace_root: Option<PathBuf>,
}

impl GrepTool {
    /// Create a searcher rooted at a canonical workspace directory.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    /// Create a remote-only content searcher.
    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

fn arguments(args: &GrepArgs, search_root: &Path) -> Vec<String> {
    let mut arguments = vec![
        "--max-columns".to_string(),
        "500".to_string(),
        "--max-columns-preview".to_string(),
        "--hidden".to_string(),
    ];
    match args.output_mode {
        GrepOutputMode::FilesWithMatches => {
            arguments.extend([
                "--files-with-matches".to_string(),
                "--sortr=modified".to_string(),
            ]);
        }
        GrepOutputMode::Count => {
            arguments.extend(["--count".to_string(), "--sortr=modified".to_string()]);
        }
        GrepOutputMode::Content => {
            arguments.push("--line-number".to_string());
            if let Some(context) = args.context {
                arguments.extend(["--context".to_string(), context.to_string()]);
            } else {
                if let Some(before) = args.before_context {
                    arguments.extend(["--before-context".to_string(), before.to_string()]);
                }
                if let Some(after) = args.after_context {
                    arguments.extend(["--after-context".to_string(), after.to_string()]);
                }
            }
        }
    }
    if args.case_insensitive {
        arguments.push("--ignore-case".to_string());
    }
    if args.multiline {
        arguments.extend(["-U".to_string(), "--multiline-dotall".to_string()]);
    }
    arguments.extend(super::ripgrep::exclusion_arguments());
    if let Some(pattern) = &args.glob {
        arguments.extend(["--glob".to_string(), pattern.clone()]);
    }
    if let Some(file_type) = &args.file_type {
        arguments.extend(["--type".to_string(), file_type.clone()]);
    }
    arguments.extend([
        "-e".to_string(),
        args.pattern.clone(),
        "--".to_string(),
        search_root.display().to_string(),
    ]);
    arguments
}

impl Tool for GrepTool {
    type Args = GrepArgs;

    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "A powerful search tool built on ripgrep with regex, glob, file-type, context, and output-mode support."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "pattern".to_string(),
                    serde_json::json!({"type": "string", "description": "Regular expression to search for, in ripgrep syntax"}),
                ),
                (
                    "path".to_string(),
                    serde_json::json!({"type": ["string", "null"], "description": "File or directory to search, relative to the workspace root, or absolute. Defaults to the workspace root."}),
                ),
                (
                    "glob".to_string(),
                    serde_json::json!({"type": ["string", "null"], "description": "Only search files matching this glob, such as \"*.rs\" or \"*.{ts,tsx}\""}),
                ),
                (
                    "type".to_string(),
                    serde_json::json!({"type": ["string", "null"], "description": "Only search files of this ripgrep file type, such as \"py\" or \"rust\""}),
                ),
                (
                    "output_mode".to_string(),
                    serde_json::json!({"enum": ["content", "files_with_matches", "count"], "default": "files_with_matches", "description": "content returns matching lines; files_with_matches returns file paths, newest first; count returns the number of matches in each file"}),
                ),
                (
                    "-i".to_string(),
                    serde_json::json!({"type": "boolean", "default": false, "description": "Ignore letter case"}),
                ),
                (
                    "-n".to_string(),
                    serde_json::json!({"type": "boolean", "default": true, "description": "Show line numbers in content output"}),
                ),
                (
                    "-A".to_string(),
                    serde_json::json!({"type": ["integer", "null"], "minimum": 0, "description": "Lines to show after each match, in content output"}),
                ),
                (
                    "-B".to_string(),
                    serde_json::json!({"type": ["integer", "null"], "minimum": 0, "description": "Lines to show before each match, in content output"}),
                ),
                (
                    "-C".to_string(),
                    serde_json::json!({"type": ["integer", "null"], "minimum": 0, "description": "Lines to show before and after each match, in content output; overrides -A and -B"}),
                ),
                (
                    "multiline".to_string(),
                    serde_json::json!({"type": "boolean", "default": false, "description": "Let the pattern span lines, with . matching newlines"}),
                ),
                (
                    "head_limit".to_string(),
                    serde_json::json!({"type": "integer", "minimum": 0, "default": 250, "description": "Maximum output lines or files to return after offset; 0 for no limit"}),
                ),
                (
                    "offset".to_string(),
                    serde_json::json!({"type": "integer", "minimum": 0, "default": 0, "description": "Output lines or files to skip before applying head_limit"}),
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
        args: GrepArgs,
        context: &ExecutionContext,
        remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        let searched = args.path.as_deref().unwrap_or(".");
        let resolved = match remote
            .resolve_path(
                searched,
                context.tool_permissions().allow_outside_workspace(),
            )
            .await
        {
            Ok(path) => path,
            Err(_) => {
                return Ok(ToolOutput::error(format!(
                    "Path not found: {searched}. Check the path and try again."
                )))
            }
        };
        let result = remote
            .run_rg(&arguments(&args, &resolved.path), &resolved.root)
            .await?;
        render_result(
            result,
            &args,
            &resolved.root,
            touched(&resolved.root, &resolved.path, TouchedKind::List),
        )
    }

    async fn run(
        &self,
        args: GrepArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let workspace_root = self.workspace_root.as_ref().ok_or_else(|| {
            ToolError::Execution("remote-only grep tool cannot execute locally".into())
        })?;
        let searched = args.path.as_deref().unwrap_or(".");
        let search_root = match resolve_existing(
            workspace_root,
            searched,
            context.tool_permissions().allow_outside_workspace(),
        )
        .await
        {
            Ok(path) => path,
            Err(_) => {
                return Ok(ToolOutput::error(format!(
                    "Path not found: {searched}. Check the path and try again."
                )));
            }
        };
        let touched = touched(workspace_root, &search_root, TouchedKind::List);
        let result = super::ripgrep::run(&arguments(&args, &search_root), workspace_root)
            .await
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        render_result(result, &args, workspace_root, touched)
    }
}

fn render_result(
    result: super::ripgrep::RgResult,
    args: &GrepArgs,
    workspace_root: &Path,
    touched: mirror_protocol::TouchedPath,
) -> Result<ToolOutput, ToolError> {
    let searched = args.path.as_deref().unwrap_or(".");
    let mut output = result.stdout;
    if result.timed_out {
        output = drop_last_line(output);
    }
    let (output, output_truncated) = truncate_output(output);
    if !matches!(result.return_code, 0 | 1) && !result.timed_out {
        let detail = if result.stderr.trim().is_empty() {
            output.trim()
        } else {
            result.stderr.trim()
        };
        let lowered = detail.to_ascii_lowercase();
        if lowered.contains("no such file or directory") || lowered.contains("enoent") {
            return Ok(ToolOutput::error(format!(
                "Path not found: {searched}. Check the path and try again."
            )));
        }
        if [
            "permission denied",
            "operation not permitted",
            "eacces",
            "eperm",
        ]
        .iter()
        .any(|pattern| lowered.contains(pattern))
        {
            return Ok(ToolOutput::error(format!(
                "Permission denied while searching {searched}. Check permissions and try again."
            )));
        }
        return Ok(ToolOutput::error(format!(
            "Grep error (exit {}): {}",
            result.return_code,
            detail.chars().take(500).collect::<String>()
        )));
    }
    let mut lines = output.lines().map(str::to_string).collect::<Vec<_>>();
    if result.timed_out && lines.is_empty() {
        return Ok(ToolOutput::success(
                "Search timed out with no results. Pattern may be too broad or cause catastrophic backtracking. Try a simpler pattern or narrower path.",
            )
            .touching(touched));
    }
    lines = lines
        .into_iter()
        .map(|line| relativize_line(line, workspace_root, args.output_mode, args.line_numbers))
        .collect();
    let total = lines.len();
    let total_matches = matches!(args.output_mode, GrepOutputMode::Count)
        .then(|| {
            lines
                .iter()
                .map(|line| parse_count(line))
                .sum::<Option<usize>>()
        })
        .flatten();
    lines = lines.into_iter().skip(args.offset).collect();
    let truncated = args.head_limit > 0 && lines.len() > args.head_limit;
    if truncated {
        lines.truncate(args.head_limit);
    }
    let mut filtered = lines.join("\n");
    if output_truncated && !filtered.is_empty() {
        filtered.push_str(
            "\n\n(Output truncated at 20000 chars. Narrow your search with glob/path filters.)",
        );
    }
    if filtered.trim().is_empty() {
        if args.offset > 0 && total > 0 {
            let unit = if matches!(args.output_mode, GrepOutputMode::Content) {
                if total == 1 {
                    "line"
                } else {
                    "lines"
                }
            } else if total == 1 {
                "file"
            } else {
                "files"
            };
            return Ok(ToolOutput::success(format!(
                "No results at offset {} (only {total} {unit} available).",
                args.offset
            ))
            .touching(touched));
        }
        return Ok(ToolOutput::success("No matches found.").touching(touched));
    }
    let mut header = if let Some(matches) = total_matches {
        format!(
            "Found {matches} matches across {total} {}",
            if total == 1 { "file" } else { "files" }
        )
    } else {
        format!(
            "Found {total} {}",
            if matches!(args.output_mode, GrepOutputMode::Content) {
                "lines"
            } else {
                "files"
            }
        )
    };
    if result.timed_out {
        header.push_str(" (search timed out, results may be incomplete)");
    }
    if truncated {
        header.push_str(&format!(" limit: {}", args.head_limit));
    }
    Ok(ToolOutput::success(format!("{header}\n{filtered}")).touching(touched))
}

/// Grep output representation.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum GrepOutputMode {
    Content,
    #[default]
    FilesWithMatches,
    Count,
}

/// Arguments accepted by the content-search tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GrepArgs {
    pub pattern: String,
    pub path: Option<String>,
    pub glob: Option<String>,
    #[serde(rename = "type")]
    pub file_type: Option<String>,
    #[serde(default)]
    pub output_mode: GrepOutputMode,
    #[serde(default, rename = "-i")]
    pub case_insensitive: bool,
    #[serde(default = "default_true", rename = "-n")]
    pub line_numbers: bool,
    #[serde(default, rename = "-A")]
    pub after_context: Option<usize>,
    #[serde(default, rename = "-B")]
    pub before_context: Option<usize>,
    #[serde(default, rename = "-C")]
    pub context: Option<usize>,
    #[serde(default)]
    pub multiline: bool,
    #[serde(default = "default_head_limit")]
    pub head_limit: usize,
    #[serde(default)]
    pub offset: usize,
}

const MAX_RESULT_CHARS: usize = 20_000;

fn default_true() -> bool {
    true
}

fn default_head_limit() -> usize {
    250
}

fn drop_last_line(output: String) -> String {
    if !output.is_empty() && !output.ends_with('\n') {
        output
            .rsplit_once('\n')
            .map_or(String::new(), |(head, _)| head.to_string())
    } else {
        output
    }
}

fn truncate_output(output: String) -> (String, bool) {
    if output.chars().count() <= MAX_RESULT_CHARS {
        return (output, false);
    }
    let truncated = output.chars().take(MAX_RESULT_CHARS).collect::<String>();
    (
        truncated
            .rsplit_once('\n')
            .map_or(String::new(), |(head, _)| head.to_string()),
        true,
    )
}

fn relativize_line(line: String, root: &Path, mode: GrepOutputMode, line_numbers: bool) -> String {
    match mode {
        GrepOutputMode::FilesWithMatches => relative_display(Path::new(&line), root),
        GrepOutputMode::Count => relativize_count_line(&line, root),
        GrepOutputMode::Content => relativize_content_line(&line, root, line_numbers),
    }
}

fn relativize_content_line(line: &str, root: &Path, line_numbers: bool) -> String {
    static PATTERN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"^(?P<path>/.*?)(?P<path_sep>[:-])(?P<number>\d+)(?P<line_sep>[:-])(?P<content>.*)$",
        )
        .expect("static regex is valid")
    });
    let Some(parts) = PATTERN.captures(line) else {
        return line.to_string();
    };
    let path = relative_display(Path::new(&parts["path"]), root);
    if line_numbers {
        format!(
            "{path}{}{}{}{}",
            &parts["path_sep"], &parts["number"], &parts["line_sep"], &parts["content"]
        )
    } else {
        format!("{path}{}{}", &parts["path_sep"], &parts["content"])
    }
}

fn relativize_count_line(line: &str, root: &Path) -> String {
    let Some((path, count)) = line.rsplit_once(':') else {
        return line.to_string();
    };
    if !count.chars().all(|character| character.is_ascii_digit()) {
        return line.to_string();
    }
    format!("{}:{count}", relative_display(Path::new(path), root))
}

fn parse_count(line: &str) -> Option<usize> {
    line.rsplit_once(':')
        .map_or(line, |(_, count)| count)
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use mirror_protocol::TouchedPath;

    use super::*;

    #[tokio::test]
    async fn content_mode_supports_context_and_relative_paths() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(
            workspace.path().join("sample.py"),
            "before\nneedle\nafter\n",
        )
        .expect("write fixture");
        let tool = GrepTool::new(workspace.path()).expect("grep tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"pattern":"needle","output_mode":"content","-C":1}"#,
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
        assert_ne!(result.content, "No matches found.");
        assert_eq!(
            result.content,
            "Found 3 lines\nsample.py-1-before\nsample.py:2:needle\nsample.py-3-after"
        );
    }

    #[tokio::test]
    async fn count_mode_aggregates_matches() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("one.txt"), "hit\nhit\n").expect("write fixture");
        fs::write(workspace.path().join("two.txt"), "hit\n").expect("write fixture");
        let tool = GrepTool::new(workspace.path()).expect("grep tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"pattern":"hit","output_mode":"count"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert!(result
            .content
            .starts_with("Found 3 matches across 2 files\n"));
    }

    #[tokio::test]
    async fn offset_does_not_report_a_limit_when_all_remaining_results_fit() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("sample.txt"), "hit\nhit\nhit\n").expect("write fixture");
        let tool = GrepTool::new(workspace.path()).expect("grep tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"pattern":"hit","output_mode":"content","offset":2,"head_limit":2}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(result.content, "Found 3 lines\nsample.txt:3:hit");
    }

    #[tokio::test]
    async fn multiline_search_uses_ripgrep_semantics() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("sample.txt"), "start\nend\n").expect("write fixture");
        let tool = GrepTool::new(workspace.path()).expect("grep tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"pattern":"start.*end","output_mode":"files_with_matches","multiline":true}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(result.content, "Found 1 files\nsample.txt");
    }
    #[tokio::test]
    async fn remote_grep_matches_local_modes_context_limits_and_errors() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::create_dir(workspace.path().join(".git")).expect("git");
        std::fs::write(workspace.path().join(".git/hidden.txt"), "match").expect("excluded");
        std::fs::write(
            workspace.path().join("space '界.txt"),
            "before\nmatch one\nMATCH two\nafter\n",
        )
        .expect("file");
        std::fs::write(workspace.path().join(".hidden.txt"), "match\n").expect("hidden");
        std::fs::write(
            workspace.path().join("large.txt"),
            "match line\n".repeat(3000),
        )
        .expect("large");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = GrepTool::new(workspace.path()).expect("local");
        let context = ExecutionContext::default();
        for args in [
            serde_json::json!({"pattern":"match"}),
            serde_json::json!({"pattern":"match", "output_mode":"count", "-i":true}),
            serde_json::json!({"pattern":"match", "output_mode":"content", "path":"space '界.txt", "-C":1}),
            serde_json::json!({"pattern":"match", "output_mode":"content", "path":"space '界.txt", "-n":false, "-A":1, "-B":1}),
            serde_json::json!({"pattern":"match", "head_limit":1, "offset":1}),
            serde_json::json!({"pattern":"match", "offset":99}),
            serde_json::json!({"pattern":"match", "glob":"*.txt", "type":"txt"}),
            serde_json::json!({"pattern":"match one\nMATCH", "multiline":true, "output_mode":"content", "path":"space '界.txt"}),
            serde_json::json!({"pattern":"match", "path":"large.txt", "output_mode":"content", "head_limit":0}),
            serde_json::json!({"pattern":"absent"}),
            serde_json::json!({"pattern":"["}),
            serde_json::json!({"pattern":"match", "path":"missing"}),
            serde_json::json!({"pattern":"match", "path":".."}),
        ] {
            let args = args.to_string();
            assert_eq!(
                GrepTool::remote()
                    .execute_remote(&args, &context, &remote)
                    .await,
                local.execute(&args, &context).await,
                "{args}"
            );
        }
        assert!(
            !GrepTool::remote()
                .execute(r#"{"pattern":"match"}"#, &context)
                .await
                .success
        );
    }
}
