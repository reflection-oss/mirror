use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use assert_matches::assert_matches;
use base64::prelude::{Engine, BASE64_STANDARD};
use pretty_assertions::assert_eq;
use serde_json::json;
use tokio::sync::mpsc;

use super::*;
use crate::tools::web_test_server::{TestResponse, TestServer};
use crate::tools::{AnyTool, DirectToolExecutor, RemoteToolExecutor, ToolExecutor};
use crate::AgentApprovalMode;

/// A stream that wrote `bytes`, captured as `run_command` captures stderr.
fn stream(bytes: &[u8]) -> CapturedStream {
    CapturedStream::from_bytes(bytes, MAX_OUTPUT_CHARS, sandbox_error_needles())
}

fn tool(workspace: &tempfile::TempDir) -> BashTool {
    BashTool::new(workspace.path()).expect("workspace is valid")
}

/// The result without its trailing `[Completed in …]` line.
fn body(content: &str) -> &str {
    let start = content
        .rfind("[Completed in ")
        .unwrap_or_else(|| panic!("no completion line in {content:?}"));
    assert!(
        content.ends_with("s]"),
        "completion line ends the result: {content:?}"
    );
    content[..start]
        .strip_suffix('\n')
        .unwrap_or(&content[..start])
}

async fn run(workspace: &tempfile::TempDir, command: &str) -> ToolOutput {
    tool(workspace)
        .run(
            BashArgs {
                command: command.to_string(),
                timeout_seconds: 30.0,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("command runs")
}

fn start_tasks() -> (
    crate::TaskManagerHandle,
    mpsc::UnboundedReceiver<crate::task_manager::TaskEvent>,
) {
    let (manager, tasks, events, task_runner) = crate::task_manager::TaskManager::new();
    tokio::spawn(manager.run());
    tokio::spawn(task_runner);
    (tasks, events)
}

#[test]
fn schema_requires_a_command_and_approval() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let tool = tool(&workspace);
    let definition = AnyTool::definition(&tool);

    assert_eq!(definition.name, "bash");
    assert_eq!(definition.description, "Execute a bash command. The output contains the command's stdout and stderr.\nCommands time out after 10 minutes. You can set a shorter timeout within the bash command.");
    assert_eq!(
        definition.parameters["required"],
        serde_json::json!(["command"])
    );
    assert_eq!(definition.parameters["additionalProperties"], false);
    assert!(AnyTool::needs_approval(&tool));
}

#[test]
fn model_arguments_are_command_only_with_a_ten_minute_budget() {
    let args: BashArgs = serde_json::from_str(r#"{"command":"true"}"#).expect("command arguments");
    assert_eq!(args.timeout_seconds, 600.0);
    assert!(!args.detached);
    assert_eq!(
        serde_json::to_value(&args).expect("serialize"),
        serde_json::json!({"command": "true"})
    );
    assert_matches!(
        serde_json::from_str::<BashArgs>(r#"{"command":"true","timeout_seconds":1}"#),
        Err(_)
    );
    assert_matches!(
        serde_json::from_str::<BashArgs>(r#"{"command":"true","detached":true}"#),
        Err(_)
    );
    assert_matches!(
        serde_json::from_str::<BashArgs>(r#"{"command":"true","timeout":1000}"#),
        Err(_)
    );
}

#[tokio::test]
async fn runs_in_the_workspace_and_captures_stdout() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let tool = tool(&workspace);

    let result = tool
        .run(
            BashArgs {
                command: "printf '%s' \"$PWD\"".to_string(),
                timeout_seconds: 30.0,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("command runs");

    assert_matches!(result, ToolOutput { success: true, .. });
    assert_eq!(result.display.as_deref(), Some("printf '%s' \"$PWD\""));
    assert_eq!(
        body(&result.content),
        workspace
            .path()
            .canonicalize()
            .expect("workspace canonicalizes")
            .display()
            .to_string()
    );
}

#[tokio::test]
async fn executes_bash_syntax_from_model_arguments() {
    let workspace = tempfile::tempdir().expect("workspace");
    let output = AnyTool::execute(
        &tool(&workspace),
        r#"{"command":"items=(hello world); printf '%s' \"${items[1]}\""}"#,
        &ExecutionContext::default(),
    )
    .await;
    assert_matches!(&output, ToolOutput { success: true, .. });
    assert_eq!(body(&output.content), "world");
}

#[tokio::test]
async fn preserves_stderr_and_nonzero_exit_status() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let tool = tool(&workspace);

    let result = tool
        .run(
            BashArgs {
                command: "printf out; printf err >&2; exit 7".to_string(),
                timeout_seconds: 30.0,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("command runs");

    assert!(result.success);
    assert_eq!(result.exit_code, Some(7));
    assert_eq!(body(&result.content), "Exit code: 7\nout\nstderr:\nerr");
}

#[tokio::test]
async fn a_shell_killed_by_a_signal_reports_exit_code_minus_one() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let tool = tool(&workspace);

    let result = tool
        .run(
            BashArgs {
                command: "kill -KILL $$".to_string(),
                timeout_seconds: 30.0,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("command runs");

    assert!(result.success);
    assert_eq!(body(&result.content), "Exit code: -1");
}

#[tokio::test]
async fn timeout_kills_the_command_process_group() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let marker = workspace.path().join("leaked");
    let tool = tool(&workspace);
    let command = delayed_marker_command(&marker);

    let result = tool
        .run(
            BashArgs {
                command,
                timeout_seconds: 0.05,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("timeout is a tool result");
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(!result.success);
    assert_eq!(result.exit_code, None);
    assert!(result.content.starts_with("Command timed out after 50ms"));
    body(&result.content);
    assert!(!marker.exists());
}

#[tokio::test]
async fn timeout_covers_inherited_output_pipes() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let marker = workspace.path().join("leaked");
    let tool = tool(&workspace);
    let command = format!("(sleep 0.2; printf leaked > '{}') &", marker.display());

    let result = tokio::time::timeout(
        Duration::from_secs(1),
        tool.run(
            BashArgs {
                command,
                timeout_seconds: 0.05,
                detached: false,
            },
            &ExecutionContext::default(),
        ),
    )
    .await
    .expect("tool timeout bounds pipe reads")
    .expect("timeout is a tool result");
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(!result.success);
    assert_eq!(result.exit_code, None);
    assert!(result.content.starts_with("Command timed out after 50ms"));
    body(&result.content);
    assert!(!marker.exists());
}

#[tokio::test]
async fn dropping_the_call_kills_the_command_process_group() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let marker = workspace.path().join("leaked");
    let tool = tool(&workspace);
    let args = BashArgs {
        command: delayed_marker_command(&marker),
        timeout_seconds: 30.0,
        detached: false,
    };

    {
        let context = ExecutionContext::default();
        let call = tool.run(args, &context);
        tokio::pin!(call);
        tokio::select! {
            result = &mut call => panic!("command unexpectedly completed: {result:?}"),
            () = tokio::time::sleep(Duration::from_millis(50)) => {}
        }
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(!marker.exists());
}

#[tokio::test]
async fn detached_command_ignores_foreground_timeout_and_can_be_waited() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let (tasks, _events) = start_tasks();
    let context = ExecutionContext::new(AgentApprovalMode::Hitl, None, tasks.clone(), None, None);
    let result = tool(&workspace)
        .run(
            BashArgs {
                command: "sleep 0.05; printf done".to_string(),
                timeout_seconds: 1e308,
                detached: true,
            },
            &context,
        )
        .await
        .expect("detach");
    let task_id = result
        .content
        .strip_prefix("Task ")
        .and_then(|value| value.strip_suffix(" started."))
        .expect("task id");

    let output = tasks.wait(task_id).await.expect("wait");
    assert_eq!(body(&output.content), "done");
    assert!(output.success);
}

#[tokio::test]
async fn cancelling_detached_command_kills_its_process_group() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let marker = workspace.path().join("leaked");
    let (tasks, _events) = start_tasks();
    let context = ExecutionContext::new(AgentApprovalMode::Hitl, None, tasks.clone(), None, None);
    let result = tool(&workspace)
        .run(
            BashArgs {
                command: delayed_marker_command(&marker),
                timeout_seconds: 30.0,
                detached: true,
            },
            &context,
        )
        .await
        .expect("detach");
    let task_id = result
        .content
        .strip_prefix("Task ")
        .and_then(|value| value.strip_suffix(" started."))
        .expect("task id");

    assert!(tasks.cancel(task_id).await.expect("cancel"));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!marker.exists());
}

#[tokio::test]
async fn rejects_nonpositive_timeout() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let result = tool(&workspace)
        .run(
            BashArgs {
                command: "true".to_string(),
                timeout_seconds: 0.0,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await;

    assert_matches!(result, Err(ToolError::InvalidArguments(message)) if message.contains("timeout_seconds must be finite"));
}

#[tokio::test]
async fn rejects_unrepresentable_timeout() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let result = tool(&workspace)
        .run(
            BashArgs {
                command: "true".to_string(),
                timeout_seconds: 1e308,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await;

    assert_matches!(result, Err(ToolError::InvalidArguments(message)) if message.contains("representable as a duration"));
}

#[test]
fn spills_large_output_with_a_bounded_preview() {
    let output = (0..4_000)
        .map(|index| format!("line {index}\n"))
        .collect::<String>();

    let result = maybe_spill(output);
    let path = result
        .split_once("Full output saved to: ")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .map(|(path, _)| PathBuf::from(path))
        .expect("result names spill path");

    assert!(result.starts_with(
        "<persisted-output>\nOutput too large (38.0KB). Full output saved to: /tmp/tool-results/"
    ));
    assert!(result.contains("\nPreview (2.0KB):\nline 0\n"));
    assert!(result.contains("lines omitted"));
    assert!(result.ends_with("line 3999\n\n</persisted-output>"));
    assert!(path.exists());
    std::fs::remove_file(path).expect("remove spill file");
}

#[test]
fn spill_threshold_counts_characters() {
    let output = "😀".repeat(10_000);

    let result = maybe_spill(output.clone());

    assert_eq!(result, output);
}

#[tokio::test]
async fn large_timed_out_output_returns_partial_output_without_spilling() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = tool(&workspace)
        .run(
            BashArgs {
                command: "head -c 40000 /dev/zero | tr '\\0' a; sleep 5".to_string(),
                timeout_seconds: 1.0,
                detached: false,
            },
            &ExecutionContext::default(),
        )
        .await
        .expect("timeout is a tool result");

    assert!(!result.success);
    assert_eq!(
        body(&result.content),
        format!(
            "Command timed out after 1000ms\nPartial stdout:\n{}\n... [35904 chars omitted] ...\n{}",
            "a".repeat(2_048),
            "a".repeat(2_048)
        )
    );
}

#[test]
fn timed_out_streams_keep_their_head_and_tail_by_character() {
    let stdout = format!("{}{}", "é".repeat(3_000), "z".repeat(2_000));

    let elided = elide_partial_stream(&stream(stdout.as_bytes()));

    assert_eq!(
        elided,
        format!(
            "{}\n... [904 chars omitted] ...\n{}{}",
            "é".repeat(2_048),
            "é".repeat(48),
            "z".repeat(2_000)
        )
    );
}

#[tokio::test]
async fn a_background_job_holding_the_pipes_returns_after_three_seconds_and_keeps_running() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let marker = workspace.path().join("finished");
    let command = format!(
        "(sleep 4; printf done > '{}') & echo started; exit 3",
        marker.display()
    );
    let started = std::time::Instant::now();

    let result = run(&workspace, &command).await;
    let elapsed = started.elapsed();
    tokio::time::sleep(Duration::from_millis(1_500)).await;

    assert_eq!(body(&result.content), "Exit code: 3\nstarted");
    assert!(
        (Duration::from_secs(3)..Duration::from_millis(3_900)).contains(&elapsed),
        "returned after {elapsed:?}"
    );
    assert!(marker.exists(), "the background job kept running");
}

#[tokio::test]
async fn empty_output_is_the_completion_line_alone() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = run(&workspace, "true").await;

    assert!(result.content.starts_with("[Completed in "));
    assert_eq!(body(&result.content), "");
}

#[tokio::test]
async fn grep_without_matches_is_a_result_not_a_failure() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = run(&workspace, "printf 'a\\n' | grep zzz").await;

    assert_eq!(body(&result.content), "No matches found (exit code 1)");
}

#[tokio::test]
async fn unmapped_exit_code_one_is_a_failure() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = run(&workspace, "grep -q zzz /dev/null; false").await;

    assert_eq!(body(&result.content), "Exit code: 1");
}

#[tokio::test]
async fn sandbox_refusals_carry_a_note() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = run(
        &workspace,
        "printf 'cat: x: Permission denied\\n' >&2; exit 2",
    )
    .await;

    assert_eq!(
        body(&result.content),
        "Exit code: 2\nstderr:\ncat: x: Permission denied\n\n[Sandbox note: The sandbox denied access to this resource.]"
    );
}

#[tokio::test]
async fn leading_and_trailing_blank_lines_are_dropped() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = run(&workspace, "printf '\\n\\nhello\\n\\n  world\\n\\n'").await;

    assert_eq!(body(&result.content), "hello\n\n  world");
}

#[tokio::test]
async fn exit_notes_follow_the_last_command_outside_quotes() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = run(&workspace, "echo \"a; b && c\" \\; | grep 'x | y'").await;

    assert_eq!(body(&result.content), "No matches found (exit code 1)");
}

#[tokio::test]
async fn completed_output_caps_combined_streams_before_spilling() {
    let output = BashOutput {
        return_code: 0,
        stdout: stream(&vec![b'a'; MAX_OUTPUT_CHARS - "\nstderr:\n".len()]),
        stderr: stream(b"first\nsecond\nthird"),
        elapsed: Duration::from_secs(1),
        ..BashOutput::default()
    };

    let result = format_result(output, "noisy".into())
        .await
        .expect("format result")
        .content;
    let path = result
        .split_once("Full output saved to: ")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .map(|(path, _)| PathBuf::from(path))
        .expect("result names spill path");
    let persisted = std::fs::read_to_string(&path).expect("persisted output");
    std::fs::remove_file(path).expect("remove spill file");

    assert_eq!(persisted.chars().count(), MAX_OUTPUT_CHARS);
    assert!(persisted.ends_with("\n[3 lines truncated, exceeded 64MB cap]"));
    assert!(result.contains("Output too large (64.0MB)."));
    assert!(result.ends_with("[Completed in 1.0s]"));
}

#[tokio::test]
async fn a_stream_past_the_cap_reads_as_it_would_held_whole() {
    let output = BashOutput {
        return_code: 0,
        stdout: stream(&vec![b'a'; MAX_OUTPUT_CHARS + 5]),
        stderr: stream(b"x\ny"),
        elapsed: Duration::from_secs(1),
        ..BashOutput::default()
    };

    let result = format_result(output, "noisy".into())
        .await
        .expect("format result")
        .content;
    let path = result
        .split_once("Full output saved to: ")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .map(|(path, _)| PathBuf::from(path))
        .expect("result names spill path");
    let persisted = std::fs::read_to_string(&path).expect("persisted output");
    std::fs::remove_file(path).expect("remove spill file");

    // Past the cap are five more `a`s, then `\nstderr:\nx\ny`: three line
    // breaks, so four lines.
    let warning = "\n[4 lines truncated, exceeded 64MB cap]";
    assert_eq!(
        persisted,
        format!("{}{warning}", "a".repeat(MAX_OUTPUT_CHARS - warning.len()))
    );
}

#[test]
fn output_cap_preserves_unicode_at_the_boundary() {
    let warning = "\n[1 lines truncated, exceeded 64MB cap]";
    let prefix = "a".repeat(MAX_OUTPUT_CHARS - warning.len() - 1);
    let output = format!("{prefix}{}", "😀".repeat(warning.len() + 2));

    let result = cap_output(PartialText::whole(output));

    assert_eq!(result.chars().count(), MAX_OUTPUT_CHARS);
    assert!(result.starts_with(&prefix));
    assert!(result.ends_with(&format!("😀{warning}")));
}

#[test]
fn output_at_the_character_cap_is_unchanged() {
    let output = format!("{}😀", "a".repeat(MAX_OUTPUT_CHARS - 1));

    let result = cap_output(PartialText::whole(output));

    assert_eq!(result.chars().count(), MAX_OUTPUT_CHARS);
    assert!(result.ends_with('😀'));
    assert!(!result.contains("truncated"));
}

#[tokio::test]
async fn characters_split_across_pipe_reads_read_as_written() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    // One ASCII byte first, so two-byte characters straddle read boundaries.
    let result = run(&workspace, "printf x; yes é | head -n 50000 | tr -d '\\n'").await;
    let path = result
        .content
        .split_once("Full output saved to: ")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .map(|(path, _)| PathBuf::from(path))
        .expect("result names spill path");
    let persisted = std::fs::read_to_string(&path).expect("persisted output");
    std::fs::remove_file(path).expect("remove spill file");

    assert_eq!(persisted, format!("x{}", "é".repeat(50_000)));
}

#[tokio::test]
async fn command_capture_keeps_more_than_32mb_in_one_stream() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let expected = "a".repeat(33 * 1024 * 1024);
    std::fs::write(workspace.path().join("output"), &expected).expect("write command output");

    let result = run(&workspace, "cat output").await;
    let path = result
        .content
        .split_once("Full output saved to: ")
        .and_then(|(_, rest)| rest.split_once('\n'))
        .map(|(path, _)| PathBuf::from(path))
        .expect("result names spill path");
    let persisted = std::fs::read_to_string(&path).expect("persisted output");
    std::fs::remove_file(path).expect("remove spill file");

    assert_eq!(persisted.len(), expected.len());
    assert_eq!(persisted, expected);
}

#[tokio::test]
async fn output_precedes_the_exit_note() {
    let output = BashOutput {
        return_code: 1,
        stdout: stream(b"< a\n---\n> b\n"),
        stderr: stream(b""),
        elapsed: Duration::from_millis(300),
        ..BashOutput::default()
    };

    let result = format_result(output, "diff a b".into())
        .await
        .expect("format result")
        .content;

    assert_eq!(
        result,
        "< a\n---\n> b\nFiles differ (exit code 1)\n[Completed in 0.3s]"
    );
}

#[test]
fn preview_snaps_to_line_boundaries() {
    let content = "first line\nsecond line\nthird line\nfourth line";

    let result = preview(content, 30);

    assert_eq!(
        result,
        "first line\n\n... [4 lines omitted] ...\n\nfourth line"
    );
}

fn delayed_marker_command(marker: &Path) -> String {
    format!("(sleep 0.2; printf leaked > '{}') & wait", marker.display())
}

#[tokio::test]
async fn commands_run_without_colour_or_pagers() {
    let workspace = tempfile::tempdir().expect("workspace");
    let output = run(
        &workspace,
        r#"printf '%s|%s|%s|%s|%s|%s' "$TERM" "$NO_COLOR" "${COLORTERM-unset}" "$PAGER" "$GIT_PAGER" "$GH_PAGER""#,
    )
    .await;
    assert_eq!(body(&output.content), "dumb|1||cat|cat|cat");
}

/// One recorded bash result, split back into the command's raw output.
#[derive(Debug, serde::Deserialize)]
struct RecordedBashResult {
    kind: String,
    sequence_id: String,
    command: String,
    stdout: String,
    stderr: String,
    return_code: i32,
    timeout_ms: Option<u64>,
    expected: String,
}

/// The result with its spill file's random name replaced, so a spilled result
/// compares by everything else.
fn without_spill_path(result: &str) -> String {
    const LABEL: &str = "Full output saved to: ";
    let Some(start) = result.find(LABEL).map(|index| index + LABEL.len()) else {
        return result.to_string();
    };
    let end = result[start..]
        .find('\n')
        .map_or(result.len(), |index| start + index);
    format!("{}<spill file>{}", &result[..start], &result[end..])
}

/// Formats each recorded result's raw output and compares it with the result
/// recorded for it. Bazel passes the fixture's path in `RECORDED_BASH_RESULTS`;
/// a build without the fixture has nothing to compare against.
#[tokio::test]
async fn reproduces_recorded_bash_results() {
    let Some(path) = std::env::var_os("RECORDED_BASH_RESULTS") else {
        assert!(
            std::env::var_os("TEST_SRCDIR").is_none(),
            "Bazel runs this test with RECORDED_BASH_RESULTS set"
        );
        return;
    };
    let fixture = std::fs::read_to_string(&path).expect("fixture is readable");
    let cases: Vec<RecordedBashResult> = fixture
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is a recorded result"))
        .collect();
    assert!(!cases.is_empty());
    let mut mismatches = Vec::new();
    for case in &cases {
        let output = BashOutput {
            return_code: case.return_code,
            stdout: stream(case.stdout.as_bytes()),
            stderr: stream(case.stderr.as_bytes()),
            timed_out_after: case.timeout_ms.map(Duration::from_millis),
            ..BashOutput::default()
        };
        let result = format_result(output, case.command.clone())
            .await
            .expect("format result")
            .content;
        let actual = without_spill_path(body(&result));
        if actual != without_spill_path(&case.expected) {
            mismatches.push(format!(
                "{} ({}):\n  expected {:?}\n  actual   {:?}",
                case.kind, case.sequence_id, case.expected, actual
            ));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// One recorded command, with a runnable stand-in for it that exercises the
/// same execution behaviour, and the result and duration it must produce.
#[derive(Debug, serde::Deserialize)]
struct RecordedBashExecution {
    kind: String,
    sequence_id: String,
    command: String,
    expected: String,
    min_seconds: f64,
    max_seconds: f64,
}

/// Runs each recorded execution's command through the tool and compares its
/// result and wall time with the recording. Bazel passes the fixture's path in
/// `RECORDED_BASH_EXECUTIONS`; a build without the fixture has nothing to
/// compare against.
#[tokio::test]
async fn reproduces_recorded_bash_executions() {
    let Some(path) = std::env::var_os("RECORDED_BASH_EXECUTIONS") else {
        assert!(
            std::env::var_os("TEST_SRCDIR").is_none(),
            "Bazel runs this test with RECORDED_BASH_EXECUTIONS set"
        );
        return;
    };
    let fixture = std::fs::read_to_string(&path).expect("fixture is readable");
    let cases: Vec<RecordedBashExecution> = fixture
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line is a recorded execution"))
        .collect();
    assert!(!cases.is_empty());
    let mut mismatches = Vec::new();
    for case in &cases {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let started = std::time::Instant::now();
        let result = run(&workspace, &case.command).await;
        let elapsed = started.elapsed().as_secs_f64();
        let actual = body(&result.content);
        if actual != case.expected || !(case.min_seconds..case.max_seconds).contains(&elapsed) {
            mismatches.push(format!(
                "{} ({}): expected {:?} in {}..{}s, got {:?} in {elapsed:.1}s",
                case.kind,
                case.sequence_id,
                case.expected,
                case.min_seconds,
                case.max_seconds,
                actual
            ));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn normalized(content: &str) -> String {
    let (body, duration) = content
        .rsplit_once("[Completed in ")
        .expect("duration suffix");
    let seconds = duration.strip_suffix("s]").expect("seconds suffix");
    assert_eq!(seconds.split('.').nth(1).expect("decimal part").len(), 1);
    assert!(seconds.parse::<f64>().expect("elapsed time") >= 0.0);
    let path = regex::Regex::new(r"Full output saved to: /tmp/tool-results/[0-9a-f]{32}\.txt")
        .expect("spill path pattern");
    path.replace(body, "Full output saved to: <spill file>")
        .into_owned()
}

fn spill_path(content: &str) -> &str {
    content
        .split_once("Full output saved to: ")
        .expect("spill path")
        .1
        .split_once('\n')
        .expect("path end")
        .0
}

#[tokio::test]
async fn real_remote_commands_match_local_model_text() {
    let local_workspace = tempfile::tempdir().expect("local workspace");
    let remote_workspace = tempfile::tempdir().expect("remote workspace");
    let mut server = TestServer::shell(remote_workspace.path()).await;
    let local = BashTool::new(local_workspace.path()).expect("local tool");
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    for command in [
        "true", "false", "exit 7", "exit 124", "exit 137", "kill -KILL $$",
        "printf '\\n  \\n'", "printf '\\n  hello 界  \\n\\n'",
        "printf 'first\\n\\nlast\\n'", "printf 'out\\n'; printf 'err\\n' >&2; exit 7",
        "printf '\\n  err  \\n\\n' >&2", "printf 'Permission denied\\n' >&2; exit 2",
        "printf 'Operation not permitted\\n' >&2", "printf 'Read-only file system\\n' >&2",
        "printf '\\000\\377\\342\\202'; printf '\\355\\240\\200' >&2",
        "grep missing <<< present", "printf 'a|b\\n' | grep missing",
        "diff <(printf 'a\\n') <(printf 'b\\n')", "test 1 = 2", "[ 1 = 2 ]",
        "printf '%s' \"$(printf '\"quoted\"')\"",
        "printf '%s|%s|%s|%s|%s|%s' \"$TERM\" \"$NO_COLOR\" \"${COLORTERM-unset}\" \"$PAGER\" \"$GIT_PAGER\" \"$GH_PAGER\"",
        "printf '%s' \"${OPENAI_API_KEY-unset}\"",
    ] {
        let arguments = json!({"command": command}).to_string();
        let context = ExecutionContext::default();
        let local_result = local.execute(&arguments, &context).await;
        let remote_result = local.execute_remote(&arguments, &context, &remote).await;
        assert_eq!(remote_result.success, local_result.success, "{command}");
        assert_eq!(remote_result.display, local_result.display, "{command}");
        assert_eq!(remote_result.metadata(), local_result.metadata(), "{command}");
        assert_eq!(normalized(&remote_result.content), normalized(&local_result.content), "{command}");
        assert_eq!(server.request().await.authorization, None);
    }
}

#[tokio::test]
async fn shared_dispatch_keeps_execution_in_the_remote_workspace() {
    let local_workspace = tempfile::tempdir().expect("local workspace");
    let remote_workspace = tempfile::tempdir().expect("remote workspace");
    let server = TestServer::shell(remote_workspace.path()).await;
    let bash = Arc::new(BashTool::new(local_workspace.path()).expect("tool"));
    let local = DirectToolExecutor::new(vec![bash.clone()]);
    let remote = RemoteToolExecutor::new(
        vec![bash],
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("executor");
    assert_eq!(local.definitions(), remote.definitions());
    assert!(remote.needs_approval("bash"));
    let output = remote
        .call_tool(
            "bash".into(),
            r#"{"command":"printf done > result.txt; printf err >&2; exit 7"}"#.into(),
            ExecutionContext::new(
                AgentApprovalMode::Auto,
                None,
                crate::TaskManagerHandle::default(),
                None,
                None,
            ),
        )
        .await
        .expect("output");
    assert_eq!(normalized(&output.content), "Exit code: 7\nstderr:\nerr\n");
    assert_eq!(output.exit_code, Some(7));
    assert!(!local_workspace.path().join("result.txt").exists());
    assert_eq!(
        std::fs::read_to_string(remote_workspace.path().join("result.txt")).expect("remote file"),
        "done"
    );
}

#[tokio::test]
async fn large_unicode_output_uses_identical_previews_and_readable_remote_spills() {
    let workspace = tempfile::tempdir().expect("workspace");
    let server = TestServer::shell(workspace.path()).await;
    let tool = BashTool::new(workspace.path()).expect("tool");
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    for command in [
        "printf '%030000d' 0",
        "printf '%030001d' 0",
        "printf x; yes é | head -n 50000 | tr -d '\\n'",
        "printf 'αβγ\\n%.0s' {1..10000}; printf '末尾\\n' >&2; exit 7",
        "printf '%067108900d' 0",
    ] {
        let arguments = json!({"command": command}).to_string();
        let context = ExecutionContext::default();
        let local_result = tool.execute(&arguments, &context).await;
        let remote_result = tool.execute_remote(&arguments, &context, &remote).await;
        assert_eq!(
            normalized(&remote_result.content),
            normalized(&local_result.content),
            "{command}"
        );
        if remote_result.content.starts_with("<persisted-output>") {
            let expected = std::fs::read(spill_path(&local_result.content)).expect("local spill");
            let actual = std::fs::read(spill_path(&remote_result.content)).expect("remote spill");
            assert_eq!(actual.len(), expected.len(), "complete spill length");
            assert!(
                actual == expected,
                "complete spill bytes differ for {command}"
            );
            let permissions = std::fs::metadata(spill_path(&remote_result.content))
                .expect("remote metadata")
                .permissions();
            assert_eq!(permissions.mode() & 0o777, 0o600);
            let read = json!({"command": format!("head -c 100 {}; rm -- {}", spill_path(&remote_result.content), spill_path(&remote_result.content))}).to_string();
            let read = tool.execute_remote(&read, &context, &remote).await;
            assert_eq!(
                normalized(&read.content),
                format!("{}\n", String::from_utf8_lossy(&expected[..100]))
            );
            std::fs::remove_file(spill_path(&local_result.content)).expect("remove local spill");
        }
    }
}

#[tokio::test]
async fn remote_timeouts_match_local_partial_output_and_character_elision() {
    let workspace = tempfile::tempdir().expect("workspace");
    let server = TestServer::shell(workspace.path()).await;
    let tool = BashTool::new(workspace.path()).expect("tool");
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    let command = "printf '界%.0s' {1..5000}; printf '%06000d' 0 >&2; sleep 2";
    let arguments = || BashArgs {
        command: command.into(),
        timeout_seconds: 0.25,
        detached: false,
    };
    let context = ExecutionContext::default();
    let local_result = tool
        .run(arguments(), &context)
        .await
        .expect("local timeout");
    let remote_result = tool
        .run_remote(arguments(), &context, &remote)
        .await
        .expect("remote timeout");
    assert!(!remote_result.success);
    assert_eq!(remote_result.metadata(), local_result.metadata());
    assert_eq!(
        normalized(&remote_result.content),
        normalized(&local_result.content)
    );
    assert!(remote_result
        .content
        .starts_with("Command timed out after 250ms\nPartial stdout:\n"));
}

#[tokio::test]
async fn endpoint_alerts_stay_out_of_model_text() {
    let response = json!({"stdout": "b3V0", "stderr": "ZXJy", "return_code": 2, "timed_out": false, "elapsed_seconds": 0.125});
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        json!({
            "output": response.to_string(), "exit_code": 0, "alerts": ["challenge_is_down"],
        }),
    )])
    .await;
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    let result = BashTool::new(".")
        .expect("tool")
        .execute_remote(
            r#"{"command":"false"}"#,
            &ExecutionContext::default(),
            &remote,
        )
        .await;
    assert_eq!(
        result.content,
        "Exit code: 2\nout\nstderr:\nerr\n[Completed in 0.1s]"
    );
    assert!(result
        .display
        .expect("display")
        .contains("challenge_is_down"));
    assert_eq!(server.request().await.authorization, None);
}

#[tokio::test]
async fn malformed_capture_never_falls_back_to_host_execution_or_echoes_payloads() {
    let workspace = tempfile::tempdir().expect("workspace");
    let mut server = TestServer::start(vec![TestResponse::json(
        200,
        json!({
            "output": "synthetic-secret", "exit_code": 0,
        }),
    )])
    .await;
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    let tool = BashTool::new(workspace.path()).expect("tool");
    let output = tool
        .execute_remote(
            r#"{"command":"touch must-not-exist"}"#,
            &ExecutionContext::default(),
            &remote,
        )
        .await;
    assert_matches!(output, ToolOutput { success: false, .. });
    assert!(!output.content.contains("synthetic-secret"));
    assert!(output.content.contains("not retried"));
    assert!(!workspace.path().join("must-not-exist").exists());
    assert_eq!(server.request().await.authorization, None);
}

#[tokio::test]
async fn inherited_pipes_obey_the_timeout_and_kill_remote_children() {
    let workspace = tempfile::tempdir().expect("workspace");
    let server = TestServer::shell(workspace.path()).await;
    let tool = BashTool::new(workspace.path()).expect("tool");
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    let output = tool
        .run_remote(
            BashArgs {
                command: "(sleep 0.5; printf leaked > leaked) &".into(),
                timeout_seconds: 0.1,
                detached: false,
            },
            &ExecutionContext::default(),
            &remote,
        )
        .await
        .expect("timeout result");
    assert!(!output.success);
    assert_eq!(output.exit_code, None);
    assert_eq!(
        normalized(&output.content),
        "Command timed out after 100ms\n"
    );
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(!workspace.path().join("leaked").exists());
}

#[tokio::test]
async fn spill_failure_uses_the_local_preview_and_does_not_echo_endpoint_data() {
    let output = "界".repeat(30_001);
    let response = json!({"stdout": BASE64_STANDARD.encode(&output), "stderr": "", "return_code": 0, "timed_out": false, "elapsed_seconds": 0.125});
    let server = TestServer::start(vec![
        TestResponse::json(200, json!({"output": response.to_string(), "exit_code": 0})),
        TestResponse::text(500, "synthetic-secret"),
    ])
    .await;
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse(&server.base_url).expect("URL"),
    )
    .expect("remote");
    let result = BashTool::new(".")
        .expect("tool")
        .execute_remote(
            r#"{"command":"true"}"#,
            &ExecutionContext::default(),
            &remote,
        )
        .await;
    assert_eq!(result.exit_code, Some(0));
    assert_eq!(
        result.content,
        append_duration(
            format_spill(
                &output,
                Err(
                    "remote shell returned HTTP 500; command outcome is unknown; not retried"
                        .into()
                )
            ),
            Duration::from_secs_f64(0.125)
        )
    );
    assert!(!result.content.contains("synthetic-secret"));
}

#[tokio::test]
async fn shared_validation_rejects_invalid_timeouts_before_environment_io() {
    let workspace = tempfile::tempdir().expect("workspace");
    let tool = BashTool::new(workspace.path()).expect("tool");
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse("http://localhost:1/exec").expect("URL"),
    )
    .expect("remote");
    let args = || BashArgs {
        command: "touch must-not-exist".into(),
        timeout_seconds: 0.0,
        detached: false,
    };
    let local = tool
        .run(args(), &ExecutionContext::default())
        .await
        .expect_err("invalid timeout");
    let remote = tool
        .run_remote(args(), &ExecutionContext::default(), &remote)
        .await
        .expect_err("invalid timeout");
    assert_eq!(local, remote);
    assert!(!workspace.path().join("must-not-exist").exists());
}

#[tokio::test]
async fn detached_execution_requires_environment_support() {
    let remote = RemoteToolExecutor::new(
        Vec::new(),
        reqwest::Url::parse("http://localhost:1/exec").expect("URL"),
    )
    .expect("remote");
    let result = BashTool::new(".")
        .expect("tool")
        .run_remote(
            BashArgs {
                command: "true".into(),
                timeout_seconds: 600.0,
                detached: true,
            },
            &ExecutionContext::default(),
            &remote,
        )
        .await;
    assert_matches!(result, Err(ToolError::InvalidArguments(message)) if message.contains("does not support detached"));
}

#[test]
fn task_display_uses_latest_carriage_return_frame() {
    assert_eq!(
        output_display(b"Downloading 10%\rDownloading 90%\r"),
        Some("Downloading 90%".to_string())
    );
}

#[tokio::test]
async fn task_display_coalesces_bursty_updates() {
    let (tasks, mut events) = start_tasks();
    let task_id = tasks
        .spawn("bash", None, TaskBehavior::Detach, |context| async move {
            let mut display = TaskDisplay::new(context);
            for index in 0..100 {
                display.add(format!("line {index}\n").as_bytes());
            }
            ToolOutput::success("done")
        })
        .await
        .expect("spawn");
    let _ = tasks.wait(&task_id).await.expect("wait");
    let mut displays = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let crate::task_manager::TaskEvent::Updated(crate::TaskInfo {
            status: crate::RuntimeTaskStatus::Running,
            display: Some(display),
            ..
        }) = event
        {
            displays.push(display);
        }
    }

    assert_eq!(displays, vec!["line 0", "line 99"]);
}

#[tokio::test]
async fn local_process_preserves_file_bytes_and_exclusive_creation() {
    let workspace = tempfile::tempdir().expect("workspace");
    let path = workspace.path().join("nested/input");
    write_file(&path, b"raw\xff bytes\n").expect("file");
    assert_eq!(
        write_file(&path, b"overwrite")
            .expect_err("exclusive file")
            .kind(),
        std::io::ErrorKind::AlreadyExists
    );
    let mut output = run_process(
        "cat nested/input",
        workspace.path(),
        Some(Duration::from_secs(5)),
        None,
    )
    .await
    .expect("capture");
    assert_eq!(
        output.stdout.stripped().expect("text").kept,
        "raw\u{fffd} bytes"
    );
    assert!(output.stderr.is_empty);
    assert_eq!(output.return_code, 0);
    assert_eq!(output.timed_out_after, None);
}

#[test]
fn local_constructor_rejects_missing_workspace() {
    let workspace = tempfile::tempdir().expect("workspace");
    assert_matches!(
        BashTool::new(workspace.path().join("missing")),
        Err(ToolError::Execution(_))
    );
}

#[tokio::test]
async fn local_workspace_is_fixed_when_constructed() {
    let workspace = tempfile::tempdir().expect("workspace");
    let first = workspace.path().join("first");
    let second = workspace.path().join("second");
    let link = workspace.path().join("current");
    std::fs::create_dir(&first).expect("first");
    std::fs::create_dir(&second).expect("second");
    std::fs::write(first.join("marker"), "original").expect("marker");
    std::fs::write(second.join("marker"), "replacement").expect("marker");
    std::os::unix::fs::symlink(&first, &link).expect("link");
    let tool = BashTool::new(&link).expect("tool");
    std::fs::remove_file(&link).expect("unlink");
    std::os::unix::fs::symlink(&second, &link).expect("retarget");
    let result = tool
        .execute(r#"{"command":"cat marker"}"#, &ExecutionContext::default())
        .await;
    assert!(result.success);
    assert_eq!(body(&result.content), "original");
}

#[tokio::test]
async fn remote_only_bash_cannot_execute_locally() {
    let result = BashTool::remote()
        .execute(
            r#"{"command":"printf should-not-run"}"#,
            &ExecutionContext::default(),
        )
        .await;
    assert!(!result.success);
    assert!(result.content.contains("cannot execute locally"));
}
