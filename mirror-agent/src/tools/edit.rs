//! Workspace text replacement.

use std::path::Path;
use std::path::PathBuf;

use mirror_protocol::TouchedKind;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use super::object_schema;
use super::resolve_existing;
use super::touched;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// Replace text in existing workspace files.
#[derive(Debug, Clone)]
pub struct EditTool {
    workspace_root: Option<PathBuf>,
}

impl EditTool {
    /// Create an editor rooted at a canonical workspace directory.
    pub fn new(workspace_root: impl AsRef<Path>) -> Result<Self, ToolError> {
        Ok(Self {
            workspace_root: Some(super::canonical_workspace(workspace_root)?),
        })
    }

    /// Create a remote-only text editor.
    pub fn remote() -> Self {
        Self {
            workspace_root: None,
        }
    }
}

impl Tool for EditTool {
    type Args = EditArgs;

    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Replace old_string with new_string in a workspace file, requiring a unique match by default and retrying with whitespace and Unicode normalization."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([
                (
                    "file_path".to_string(),
                    serde_json::json!({"type": "string", "description": "Path to the file to edit, relative to the workspace root, or absolute"}),
                ),
                (
                    "old_string".to_string(),
                    serde_json::json!({"type": "string", "description": "Exact text to replace, including whitespace and newlines. It must occur once in the file unless replace_all is true."}),
                ),
                (
                    "new_string".to_string(),
                    serde_json::json!({"type": "string", "description": "Text to replace old_string with"}),
                ),
                (
                    "replace_all".to_string(),
                    serde_json::json!({"type": "boolean", "default": false, "description": "Replace every occurrence of old_string instead of requiring exactly one"}),
                ),
            ]),
            &["file_path", "old_string", "new_string"],
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
        args: EditArgs,
        context: &ExecutionContext,
        remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        if args.old_string.is_empty() {
            return Err(ToolError::InvalidArguments(
                "old_string must not be empty.".to_string(),
            ));
        }
        remote
            .run_edit(&args, context.tool_permissions().allow_outside_workspace())
            .await
    }

    async fn run(
        &self,
        args: EditArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        if args.old_string.is_empty() {
            return Err(ToolError::InvalidArguments(
                "old_string must not be empty.".to_string(),
            ));
        }
        let workspace_root = self.workspace_root.as_ref().ok_or_else(|| {
            ToolError::Execution("remote-only edit tool cannot execute locally".into())
        })?;
        let path = resolve_existing(
            workspace_root,
            &args.file_path,
            context.tool_permissions().allow_outside_workspace(),
        )
        .await?;
        let text = tokio::fs::read_to_string(&path)
            .await
            .map_err(super::execution_error)?;
        let (updated, count) = replace_text(&text, &args)?;
        tokio::fs::write(&path, &updated)
            .await
            .map_err(super::execution_error)?;
        Ok(
            format_result(&text, &updated, count, &args).touching(touched(
                workspace_root,
                &path,
                TouchedKind::Write,
            )),
        )
    }
}

pub(super) fn replace_text(text: &str, args: &EditArgs) -> Result<(String, usize), ToolError> {
    let exact_count = text.matches(&args.old_string).count();
    let (updated, count) = if exact_count > 0 {
        validate_count(exact_count, args)?;
        let replacements = if args.replace_all { exact_count } else { 1 };
        (
            text.replacen(&args.old_string, &args.new_string, replacements),
            replacements,
        )
    } else {
        let normalized = normalize_for_fuzzy_match(text);
        let old = normalize_for_fuzzy_match(&args.old_string);
        let count = if old.is_empty() {
            0
        } else {
            normalized.matches(&old).count()
        };
        validate_count(count, args)?;
        let replacements = if args.replace_all { count } else { 1 };
        (
            fuzzy_replace(text, &normalized, &old, &args.new_string, replacements)?,
            replacements,
        )
    };
    Ok((updated, count))
}

pub(super) fn format_result(
    text: &str,
    updated: &str,
    count: usize,
    args: &EditArgs,
) -> ToolOutput {
    let diff = super::diff::unified_diff(text, updated, &args.file_path);
    if diff.is_empty() {
        return ToolOutput::success(format!("No changes to {}.", args.file_path));
    }
    let mut result = ToolOutput::success(format!(
        "Made {count} replacement(s) in {}.",
        args.file_path
    ));
    if let Some(display) = super::diff::diff_display(&diff) {
        result = result.with_display(display);
    }
    result
}

/// Arguments accepted by the text-replacement tool.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EditArgs {
    pub file_path: String,
    pub old_string: String,
    pub new_string: String,
    #[serde(default)]
    pub replace_all: bool,
}

fn validate_count(count: usize, args: &EditArgs) -> Result<(), ToolError> {
    if count == 0 {
        return Err(ToolError::InvalidArguments(format!(
            "old_string not found in {}. It must match the file exactly, including whitespace and newlines; copy the text from a fresh read of the file.",
            args.file_path
        )));
    }
    if count > 1 && !args.replace_all {
        return Err(ToolError::InvalidArguments(format!(
            "old_string occurs {count} times in {}; the match must be unique. Add surrounding lines to make it unique, or pass replace_all=true to replace every occurrence.",
            args.file_path
        )));
    }
    Ok(())
}

fn normalize_for_fuzzy_match(text: &str) -> String {
    text.nfkc()
        .collect::<String>()
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .chars()
        .map(|character| match character {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => '\'',
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{201f}' => '"',
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{00a0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
            | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200a}' | '\u{202f}' | '\u{205f}'
            | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

fn fuzzy_replace(
    original: &str,
    normalized: &str,
    old: &str,
    new: &str,
    replacements: usize,
) -> Result<String, ToolError> {
    let original_lines = original.split_inclusive('\n').collect::<Vec<_>>();
    let normalized_lines = normalized.split_inclusive('\n').collect::<Vec<_>>();
    if original_lines.len() != normalized_lines.len()
        || original_lines
            .iter()
            .zip(&normalized_lines)
            .any(|(original, normalized)| line_ending(original) != line_ending(normalized))
    {
        return Err(ToolError::InvalidArguments(
            "old_string could not be matched without rewriting the file line endings; match it exactly, including whitespace and newlines."
                .to_string(),
        ));
    }
    let mut starts = Vec::with_capacity(normalized_lines.len());
    let mut offset = 0;
    for line in &normalized_lines {
        starts.push(offset);
        offset += line.len();
    }
    let mut regions = Vec::<(usize, usize, usize)>::new();
    for (position, _) in normalized.match_indices(old).take(replacements) {
        let start = line_index(&starts, position);
        let end = line_index(&starts, position + old.len() - 1) + 1;
        if let Some(last) = regions.last_mut().filter(|last| start < last.1) {
            last.1 = last.1.max(end);
            last.2 += 1;
        } else {
            regions.push((start, end, 1));
        }
    }
    let mut result = String::new();
    let mut copied = 0;
    for (start, end, count) in regions {
        result.push_str(&original_lines[copied..start].concat());
        let region_start = starts[start];
        let region_end = starts.get(end).copied().unwrap_or(offset);
        result.push_str(&normalized[region_start..region_end].replacen(old, new, count));
        copied = end;
    }
    result.push_str(&original_lines[copied..].concat());
    Ok(result)
}

fn line_ending(line: &str) -> &str {
    if line.ends_with("\r\n") {
        "\r\n"
    } else if line.ends_with('\n') {
        "\n"
    } else {
        ""
    }
}

fn line_index(starts: &[usize], position: usize) -> usize {
    starts
        .partition_point(|start| *start <= position)
        .saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    use mirror_protocol::TouchedPath;

    use super::*;

    #[tokio::test]
    async fn replaces_a_unique_exact_match() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let path = workspace.path().join("file.txt");
        fs::write(&path, "before\nold\nafter\n").expect("write fixture");
        let tool = EditTool::new(workspace.path()).expect("edit tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"file.txt","old_string":"old","new_string":"new"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(result.content, "Made 1 replacement(s) in file.txt.");
        assert_eq!(
            result.touched,
            vec![TouchedPath {
                path: "file.txt".to_string(),
                kind: TouchedKind::Write as i32,
            }]
        );
        assert_eq!(fs::read_to_string(path).unwrap(), "before\nnew\nafter\n");
        assert!(Tool::needs_approval(&tool));
    }

    #[tokio::test]
    async fn fuzzy_match_preserves_untouched_lines() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let path = workspace.path().join("file.txt");
        fs::write(&path, "keep  \nsmart “quote”  \nlast  \n").expect("write fixture");
        let tool = EditTool::new(workspace.path()).expect("edit tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"file.txt","old_string":"smart \"quote\"","new_string":"changed"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert!(result.success);
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "keep  \nchanged\nlast  \n"
        );
    }

    #[tokio::test]
    async fn fuzzy_match_rejects_line_ending_changes() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let path = workspace.path().join("file.txt");
        let original = "before\r\nsmart “quote”\r\nafter\r\n";
        fs::write(&path, original).expect("write fixture");
        let tool = EditTool::new(workspace.path()).expect("edit tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"file.txt","old_string":"smart \"quote\"","new_string":"changed"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(&result, ToolOutput { success: false, .. });
        assert_ne!(result.content, "Made 1 replacement(s) in file.txt.");
        assert_eq!(fs::read_to_string(path).unwrap(), original);
    }

    #[tokio::test]
    async fn duplicate_match_requires_replace_all() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::write(workspace.path().join("file.txt"), "same same").expect("write fixture");
        let tool = EditTool::new(workspace.path()).expect("edit tool");

        let result = crate::tools::AnyTool::execute(
            &tool,
            r#"{"file_path":"file.txt","old_string":"same","new_string":"new"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert!(!result.success);
        assert!(result.content.contains("occurs 2 times"));
    }
    #[tokio::test]
    async fn remote_edit_matches_local_replacements_errors_and_diffs() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let workspace = tempfile::tempdir().expect("workspace");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let local = EditTool::new(workspace.path()).expect("local");
        let context = ExecutionContext::default();
        let path = workspace.path().join("data");
        for (before, old, new, all) in [
            ("hello\n", "hello", "world", false),
            ("a a\n", "a", "b", false),
            ("a a\n", "a", "b", true),
            ("hello\n", "missing", "world", false),
            ("hello\n", "", "world", false),
            ("hello\n", "hello", "hello", false),
            ("hello\n", "hello", "", false),
            ("“quoted”\n", "\"quoted\"", "new", false),
            ("ＡＢＣ   \n", "ABC\n", "new\n", false),
            ("first\r\nsecond\r\n", "second", "new", false),
            ("first\r\nsecond\r\n", "first\nsecond", "new", false),
        ] {
            std::fs::write(&path, before).expect("before");
            let args = serde_json::json!({"file_path":"data", "old_string":old, "new_string":new, "replace_all":all}).to_string();
            let expected = local.execute(&args, &context).await;
            let expected_bytes = std::fs::read(&path).expect("local bytes");
            std::fs::write(&path, before).expect("reset");
            assert_eq!(
                EditTool::remote()
                    .execute_remote(&args, &context, &remote)
                    .await,
                expected,
                "{args}"
            );
            assert_eq!(std::fs::read(&path).expect("remote bytes"), expected_bytes);
        }
        std::fs::write(&path, b"bad\xff").expect("invalid UTF-8");
        let args = r#"{"file_path":"data","old_string":"bad","new_string":"good"}"#;
        assert_eq!(
            EditTool::remote()
                .execute_remote(args, &context, &remote)
                .await,
            local.execute(args, &context).await
        );
        assert_eq!(std::fs::read(&path).expect("unchanged"), b"bad\xff");
        assert!(!EditTool::remote().execute(args, &context).await.success);
    }
    #[tokio::test]
    async fn remote_edit_never_touches_host_files_and_preserves_symlinks() {
        use crate::tools::web_test_server::TestServer;
        use crate::tools::{AnyTool, RemoteToolExecutor};
        let host = tempfile::tempdir().expect("host");
        let workspace = tempfile::tempdir().expect("remote");
        std::fs::write(host.path().join("link"), "host canary").expect("host file");
        std::fs::write(workspace.path().join("data"), "old").expect("remote file");
        std::os::unix::fs::symlink("data", workspace.path().join("link")).expect("link");
        let server = TestServer::shell(workspace.path()).await;
        let remote = RemoteToolExecutor::new(
            Vec::new(),
            reqwest::Url::parse(&server.base_url).expect("URL"),
        )
        .expect("executor");
        let tool = EditTool::new(host.path()).expect("host-bound tool");
        let output = tool
            .execute_remote(
                r#"{"file_path":"link","old_string":"old","new_string":"new"}"#,
                &ExecutionContext::default(),
                &remote,
            )
            .await;
        assert!(output.success, "{output:?}");
        assert_eq!(
            std::fs::read_to_string(host.path().join("link")).expect("host unchanged"),
            "host canary"
        );
        assert_eq!(
            std::fs::read_to_string(workspace.path().join("data")).expect("remote edit"),
            "new"
        );
        assert!(std::fs::symlink_metadata(workspace.path().join("link"))
            .expect("link")
            .is_symlink());
        assert!(AnyTool::needs_approval(&EditTool::remote()));
    }
}
