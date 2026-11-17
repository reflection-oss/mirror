//! Runtime environment facts for the system prompt.

use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use crate::is_repository_root;

/// Runtime environment facts surfaced to the agent.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentInfo {
    pub working_directory: PathBuf,
    pub is_git_repo: bool,
    pub platform: String,
    pub os_version: String,
    pub shell: String,
    /// Repository prompt section captured in the workspace at session start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_status: Option<String>,
}

/// A bash script that prints the facts [`EnvironmentInfo::from_probe`] reads:
/// exactly two lines, each empty when its fact is unavailable.
pub const ENVIRONMENT_PROBE: &str =
    r#"printf '%s\n' "$(uname -sr 2>/dev/null)" "${BASH_VERSION%%(*}""#;

impl EnvironmentInfo {
    /// Environment facts from [`ENVIRONMENT_PROBE`]'s output on a `platform`
    /// machine whose `working_directory` is also readable here.
    pub fn from_probe(working_directory: &Path, platform: &str, probe_output: &str) -> Self {
        let mut lines = probe_output.lines().map(str::trim);
        let mut next = || lines.next().filter(|line| !line.is_empty());
        let os_version = next().map_or_else(|| platform.to_string(), str::to_string);
        let shell = next().map_or_else(|| "bash".to_string(), |version| format!("bash {version}"));
        let root = working_directory
            .ancestors()
            .find(|ancestor| is_repository_root(ancestor));
        Self {
            working_directory: working_directory.components().collect(),
            is_git_repo: root.is_some_and(|root| root.join(".git").exists()),
            platform: platform.to_string(),
            os_version,
            shell,
            repository_status: root.and_then(|root| {
                if root.join(".jj").is_dir() {
                    Some(gather_jj_info(working_directory))
                } else {
                    gather_git_info(working_directory)
                }
            }),
        }
    }
}

/// Gather environment facts for a selected working directory on this machine.
pub fn gather_environment_info(working_directory: &Path) -> EnvironmentInfo {
    let platform = match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "Darwin",
        "windows" => "Windows",
        other => other,
    };
    EnvironmentInfo::from_probe(working_directory, platform, &run_probe(ENVIRONMENT_PROBE))
}

/// This machine's local date, timezone name and UTC offset.
pub(super) fn local_date() -> Option<String> {
    Some(run_probe("date '+%Y-%m-%d %Z %z'").trim().to_string()).filter(|date| !date.is_empty())
}

fn gather_git_info(directory: &Path) -> Option<String> {
    let directories = git(directory, &["rev-parse", "--git-dir", "--git-common-dir"])?;
    let mut directories = directories.lines();
    let git_dir = std::fs::canonicalize(directory.join(directories.next()?)).ok()?;
    let common_dir = std::fs::canonicalize(directory.join(directories.next()?)).ok()?;
    let default_branch = git(
        directory,
        &[
            "symbolic-ref",
            "--quiet",
            "--short",
            "refs/remotes/origin/HEAD",
        ],
    )
    .and_then(|reference| reference.strip_prefix("origin/").map(str::to_string))
    .or_else(|| {
        [
            "refs/remotes/origin/main",
            "refs/remotes/origin/master",
            "refs/heads/main",
            "refs/heads/master",
        ]
        .into_iter()
        .find(|reference| git(directory, &["show-ref", "--verify", "--quiet", reference]).is_some())
        .and_then(|reference| reference.rsplit('/').next().map(str::to_string))
    });
    let branch = git(directory, &["symbolic-ref", "--quiet", "--short", "HEAD"]).or_else(|| {
        git(directory, &["rev-parse", "--verify", "HEAD"]).map(|_| "HEAD (detached)".into())
    });
    Some(format!(
        "- Git worktree: {}\n\n# Git status\nThis is a snapshot at session start; re-check with git before acting on it.\n\nCurrent branch: {}\nMain branch (for PRs): {}\nGit user: {}\n\nStatus:\n{}\n\nRecent commits:\n{}",
        if git_dir != common_dir { "yes" } else { "no" },
        branch.as_deref().unwrap_or("(unknown)"),
        default_branch.as_deref().unwrap_or("(unknown)"),
        git(directory, &["config", "user.name"]).as_deref().unwrap_or("(unknown)"),
        short_status(git(directory, &["status", "--short"]), "git status"),
        git(directory, &["log", "--oneline", "-n", "3"]).as_deref().unwrap_or("(unavailable)"),
    ))
}

fn gather_jj_info(directory: &Path) -> String {
    let commits = jj(
        directory,
        &[
            "log",
            "--no-graph",
            "--limit",
            "3",
            "-r",
            "ancestors(@) ~ @ ~ root()",
            "-T",
            r#"commit_id.short() ++ " " ++ description.first_line() ++ "\n""#,
        ],
    )
    .unwrap_or_else(|| "(unavailable)".into());
    format!(
        "- Jujutsu repository: yes\n\n# Jujutsu status\nThis is a snapshot at session start; re-check with jj before acting on it.\n\nJujutsu user: {}\n\nStatus:\n{}\n\nRecent commits:\n{}",
        jj(directory, &["config", "get", "user.name"]).as_deref().unwrap_or("(unknown)"),
        short_status(jj(directory, &["status"]), "jj status"),
        if commits.is_empty() { "(none)" } else { &commits },
    )
}

fn short_status(status: Option<String>, command: &str) -> String {
    let mut status = status.unwrap_or_else(|| "(unavailable)".into());
    if status.chars().count() > 2_000 {
        status = status.chars().take(2_000).collect();
        status.push_str(&format!(
            "\n... (truncated; run {command} for the full listing)"
        ));
    }
    if status.is_empty() {
        "(clean)".into()
    } else {
        status
    }
}

fn jj(directory: &Path, args: &[&str]) -> Option<String> {
    let mut command = Command::new("jj");
    #[cfg(test)]
    if let Some(binary) = std::env::var_os("JJ_TEST_BIN") {
        command = Command::new(std::fs::canonicalize(binary).ok()?);
    }
    // Observe fresh file changes without updating the workspace or operation head.
    command
        .current_dir(directory)
        .args(["--no-integrate-operation", "--color=never", "--no-pager"])
        .env("LC_ALL", "C");
    command_output(command, args)
}

fn git(directory: &Path, args: &[&str]) -> Option<String> {
    let mut command = Command::new("git");
    command
        .current_dir(directory)
        .args(["--no-optional-locks", "-c", "core.fsmonitor=false"])
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_NAMESPACE");
    command_output(command, args)
}

fn command_output(mut command: Command, args: &[&str]) -> Option<String> {
    let mut child = command
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let mut output = Vec::new();
    // Retain enough status bytes for 2,001 UTF-8 characters, then drain the rest.
    let limit = match args {
        ["status", ..] => 8_004,
        _ => u64::MAX,
    };
    let read = stdout
        .by_ref()
        .take(limit)
        .read_to_end(&mut output)
        .and_then(|_| std::io::copy(&mut stdout, &mut std::io::sink()));
    drop(stdout);
    let status = child.wait().ok()?;
    (read.is_ok() && status.success())
        .then(|| String::from_utf8_lossy(&output).trim_end().to_string())
}

fn run_probe(script: &str) -> String {
    let inherited = ["PATH", "TZ"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name, value)));
    Command::new("/bin/bash")
        .args(["-c", script])
        .env_clear()
        .envs(inherited)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn gather_describes_the_selected_directory() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("repository marker");

        let environment = gather_environment_info(workspace.path());

        assert_eq!(environment.working_directory, workspace.path());
        assert!(environment.is_git_repo);
        assert!(environment.os_version.starts_with(&environment.platform));
        assert!(environment.shell.starts_with("bash "));
    }

    #[test]
    fn gather_reports_a_directory_outside_any_repository() {
        let workspace = tempfile::tempdir().expect("temporary workspace");

        let environment = gather_environment_info(workspace.path());
        assert!(!environment.is_git_repo);
        assert_matches!(environment.repository_status, None);
    }

    #[test]
    fn jujutsu_workspace_inside_a_git_checkout_is_not_a_git_repository() {
        let checkout = tempfile::tempdir().expect("temporary checkout");
        fs::create_dir(checkout.path().join(".git")).expect("git marker");
        let workspace = checkout.path().join("workspaces/feature");
        fs::create_dir_all(workspace.join(".jj")).expect("jujutsu marker");

        let environment = gather_environment_info(&workspace);
        assert!(!environment.is_git_repo);
        assert_matches!(environment.repository_status, Some(section) if section.contains("Jujutsu repository: yes") && section.contains("Status:\n(unavailable)"));
    }

    #[test]
    fn gather_captures_branches_user_status_and_three_commits() {
        let workspace = repository(4);
        run_git(workspace.path(), &["switch", "-c", "feature"]);
        fs::write(workspace.path().join("tracked.txt"), "staged").expect("staged change");
        run_git(workspace.path(), &["add", "tracked.txt"]);
        fs::write(workspace.path().join("tracked.txt"), "changed").expect("unstaged change");
        fs::write(workspace.path().join("new.txt"), "new").expect("untracked file");
        let nested = workspace.path().join("nested");
        fs::create_dir(&nested).expect("nested working directory");

        let section = snapshot(&nested);

        assert!(section.contains("Git worktree: no"), "{section}");
        assert!(section.contains("Current branch: feature"), "{section}");
        assert!(section.contains("Main branch (for PRs): main"), "{section}");
        assert!(section.contains("Git user: Prompt Test"), "{section}");
        assert!(section.contains("MM ../tracked.txt"), "{section}");
        assert!(section.contains("?? ../new.txt"), "{section}");
        let commits = section
            .split_once("Recent commits:\n")
            .expect("recent commits")
            .1;
        let hash = git(workspace.path(), &["rev-parse", "--short", "HEAD"]).expect("short hash");
        assert!(
            commits.starts_with(&format!("{hash} commit 3\n")),
            "{commits}"
        );
        assert_eq!(commits.lines().count(), 3);
        assert_matches!(commits.lines().nth(1), Some(line) if line.ends_with(" commit 2"));
        assert_matches!(commits.lines().nth(2), Some(line) if line.ends_with(" commit 1"));
    }

    #[test]
    fn gather_recognizes_linked_and_detached_worktrees_without_the_primary_path() {
        let workspace = repository(1);
        let parent = tempfile::tempdir().expect("worktree parent");
        let worktree = parent.path().join("linked");
        run_git(
            workspace.path(),
            &[
                "worktree",
                "add",
                "--detach",
                worktree.to_str().expect("path"),
            ],
        );

        let section = snapshot(&worktree);

        assert!(section.contains("Git worktree: yes"));
        assert!(section.contains("Current branch: HEAD (detached)"));
        assert!(section.contains("Status:\n(clean)"));
        assert!(!section.contains(workspace.path().to_str().expect("primary path")));
    }

    #[test]
    fn gather_prefers_remote_branches_over_local_fallbacks() {
        let workspace = repository(1);
        run_git(
            workspace.path(),
            &["update-ref", "refs/remotes/origin/master", "HEAD"],
        );
        let section = snapshot(workspace.path());
        assert!(section.contains("Main branch (for PRs): master"));

        run_git(
            workspace.path(),
            &["update-ref", "refs/remotes/origin/develop", "HEAD"],
        );
        run_git(
            workspace.path(),
            &[
                "symbolic-ref",
                "refs/remotes/origin/HEAD",
                "refs/remotes/origin/develop",
            ],
        );
        let section = snapshot(workspace.path());
        assert!(section.contains("Main branch (for PRs): develop"));
    }

    #[test]
    fn empty_repository_has_a_branch_but_no_commits_and_failed_status_is_not_clean() {
        let workspace = repository(0);
        let section = snapshot(workspace.path());
        assert!(section.contains("Current branch: main"));
        assert!(section.contains("Recent commits:\n(unavailable)"));
        assert!(section.contains("Status:\n(clean)"));

        fs::write(workspace.path().join(".git/index"), "invalid index").expect("corrupt index");
        let section = snapshot(workspace.path());
        assert!(section.contains("Status:\n(unavailable)"));
        assert!(!section.contains("(clean)"));
    }

    #[test]
    fn gather_caps_large_status_without_splitting_unicode() {
        let workspace = repository(0);
        for index in 0..5_000 {
            fs::write(
                workspace.path().join(format!("untracked-{index:03}-é.txt")),
                "",
            )
            .expect("untracked file");
        }
        run_git(workspace.path(), &["config", "core.quotePath", "false"]);

        let section = snapshot(workspace.path());

        let status = section.split_once("Status:\n").expect("status").1;
        let (listing, notice) = status.split_once("\n... ").expect("truncation notice");
        assert_eq!(listing.chars().count(), 2_000);
        assert!(listing.contains('é'));
        assert!(!listing.contains('�'));
        assert!(notice.contains("run git status"));
    }

    #[test]
    fn jj_snapshot_reports_fresh_changes_and_commits_without_advancing_the_workspace() {
        let workspace = tempfile::tempdir().expect("JJ workspace");
        let root = workspace.path();
        run_jj(root, &["git", "init", "--no-colocate"]);
        run_jj(
            root,
            &["config", "set", "--repo", "user.name", "Prompt Test"],
        );
        run_jj(root, &["commit", "-m", "commit 0"]);
        run_jj(root, &["commit", "-m", "commit 1"]);
        run_jj(root, &["commit", "-m", "commit 2"]);
        run_jj(root, &["commit", "-m", "commit 3"]);
        fs::write(root.join("new.txt"), "fresh change").expect("new file");
        let checkout = fs::read(root.join(".jj/working_copy/checkout")).expect("checkout state");
        let tree_state = fs::read(root.join(".jj/working_copy/tree_state")).expect("tree state");
        let operation = run_jj(
            root,
            &[
                "--ignore-working-copy",
                "op",
                "log",
                "--no-graph",
                "--limit",
                "1",
                "-T",
                "id",
            ],
        );

        let environment = gather_environment_info(root);
        let section = environment.repository_status.expect("JJ snapshot");

        assert!(!environment.is_git_repo);
        assert!(section.contains("Jujutsu repository: yes"));
        assert!(section.contains("Jujutsu user: Prompt Test"));
        assert!(section.contains("A new.txt"));
        let commits = section
            .split_once("Recent commits:\n")
            .expect("recent commits")
            .1;
        assert_eq!(commits.lines().count(), 3);
        assert_matches!(commits.lines().next(), Some(line) if line.ends_with(" commit 3"));
        assert_matches!(commits.lines().nth(1), Some(line) if line.ends_with(" commit 2"));
        assert_matches!(commits.lines().nth(2), Some(line) if line.ends_with(" commit 1"));
        assert_eq!(
            fs::read(root.join(".jj/working_copy/checkout")).expect("checkout state"),
            checkout
        );
        assert_eq!(
            fs::read(root.join(".jj/working_copy/tree_state")).expect("tree state"),
            tree_state
        );
        assert_eq!(
            run_jj(
                root,
                &[
                    "--ignore-working-copy",
                    "op",
                    "log",
                    "--no-graph",
                    "--limit",
                    "1",
                    "-T",
                    "id"
                ]
            ),
            operation
        );
    }

    #[test]
    fn jj_takes_precedence_in_colocated_and_secondary_workspaces() {
        let workspace = tempfile::tempdir().expect("colocated workspace");
        run_jj(workspace.path(), &["git", "init", "--colocate"]);
        let primary = gather_environment_info(workspace.path());
        assert!(primary.is_git_repo);
        assert_matches!(primary.repository_status, Some(section) if section.contains("# Jujutsu status") && !section.contains("# Git status"));

        let parent = tempfile::tempdir().expect("secondary parent");
        let secondary = parent.path().join("secondary");
        run_jj(
            workspace.path(),
            &[
                "workspace",
                "add",
                "--name",
                "secondary",
                secondary.to_str().expect("path"),
            ],
        );
        let nested = secondary.join("nested");
        fs::create_dir(&nested).expect("nested directory");
        fs::write(nested.join("new.txt"), "secondary change").expect("new file");

        let environment = gather_environment_info(&nested);
        let section = environment.repository_status.expect("secondary snapshot");

        assert_eq!(environment.working_directory, nested);
        assert!(!environment.is_git_repo);
        assert!(section.contains("# Jujutsu status"));
        assert!(section.contains("A new.txt"));
        assert!(!section.contains(workspace.path().to_str().expect("primary path")));
    }

    #[test]
    fn older_environment_payloads_need_no_repository_snapshot() {
        let facts: EnvironmentInfo = serde_json::from_value(serde_json::json!({
            "working_directory": "/workspace", "is_git_repo": true,
            "platform": "Linux", "os_version": "Linux 6.1", "shell": "bash 5.2"
        }))
        .expect("older guest payload");

        assert_matches!(facts.repository_status, None);
    }

    fn snapshot(directory: &Path) -> String {
        gather_environment_info(directory)
            .repository_status
            .expect("repository snapshot")
    }

    fn repository(commits: usize) -> tempfile::TempDir {
        let workspace = tempfile::tempdir().expect("repository");
        run_git(workspace.path(), &["init", "-b", "main"]);
        run_git(workspace.path(), &["config", "user.name", "Prompt Test"]);
        run_git(
            workspace.path(),
            &["config", "user.email", "prompt-test@example.invalid"],
        );
        for revision in 0..commits {
            fs::write(workspace.path().join("tracked.txt"), revision.to_string())
                .expect("tracked file");
            run_git(workspace.path(), &["add", "tracked.txt"]);
            run_git(
                workspace.path(),
                &["commit", "-m", &format!("commit {revision}")],
            );
        }
        workspace
    }

    fn run_jj(directory: &Path, args: &[&str]) -> String {
        let binary = std::env::var_os("JJ_TEST_BIN")
            .map(|path| {
                fs::canonicalize(path)
                    .expect("JJ test binary")
                    .into_os_string()
            })
            .unwrap_or_else(|| "jj".into());
        let output = Command::new(binary)
            .current_dir(directory)
            .args([
                "--color=never",
                "--no-pager",
                "--config",
                "user.name='Test'",
                "--config",
                "user.email='test@example.invalid'",
            ])
            .args(args)
            .env("JJ_CONFIG", "/dev/null")
            .output()
            .expect("run JJ");
        assert!(
            output.status.success(),
            "jj {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("UTF-8 output")
            .trim_end()
            .to_string()
    }

    fn run_git(directory: &Path, args: &[&str]) {
        let output = Command::new("git")
            .current_dir(directory)
            .args(args)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .expect("run Git");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
