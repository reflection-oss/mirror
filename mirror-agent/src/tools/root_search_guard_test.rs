use pretty_assertions::assert_eq;

use super::*;
use crate::tools::BashTool;

fn guarded_bash(workspace: &tempfile::TempDir) -> RootSearchGuard {
    RootSearchGuard::new(
        Arc::new(BashTool::new(workspace.path()).expect("workspace is valid")),
        Some(workspace.path().to_path_buf()),
    )
}

fn refused(command: &str) -> bool {
    root_search_refusal(command, None).is_some()
}

#[test]
fn refuses_recursive_grep_over_root_behind_redirects_and_pipes() {
    assert!(refused(
        r#"grep -rl "AwilixNotInitializedError\|totalDuration" / 2>/dev/null | grep -v '/app/src' | head"#
    ));
}

#[test]
fn refuses_root_search_after_cd_timeout_and_sudo() {
    assert!(refused(
        "cd /app && timeout 280 sudo grep -RIl needle / | head"
    ));
}

#[test]
fn refuses_root_search_in_command_substitution() {
    assert!(refused("echo $(grep -R --include=*.py needle /*)"));
}

#[test]
fn refuses_root_search_whose_other_operands_are_substitutions() {
    assert!(refused(
        r#"cd /testbed && grep -rn "def assignment?" $(ruby -rrubocop -e 'puts RuboCop::ConfigLoader.default_configuration.path' 2>/dev/null) / 2>/dev/null | head"#
    ));
    assert!(refused("grep -r needle `gem env gemdir` /"));
    assert!(refused("(cd /tmp && grep -r needle /) | head"));
}

#[test]
fn allows_searches_of_substituted_directories() {
    assert!(!refused("grep -rn needle $(gem env gemdir) /testbed"));
    assert!(!refused("grep -rn needle \"$(dirname \"$(which ruby)\")\""));
}

#[test]
fn refuses_searches_of_the_current_directory_after_cd_to_root() {
    assert!(refused(
        r#"cd / && grep -rl "DevLogLevels" . 2>/dev/null | grep -v proc | head"#
    ));
    assert!(refused("cd / && rg needle"));
    assert!(!refused("cd / && cd /testbed && grep -rn needle ."));
    assert!(!refused("cd /testbed && grep -rn needle ."));
}

#[test]
fn refuses_grep_over_every_file_find_lists_under_root() {
    assert!(refused(
        r#"find / -type f 2>/dev/null | xargs grep -l "unauthorized" 2>/dev/null | head"#
    ));
    assert!(refused(
        "find / -maxdepth 4 -type f |& xargs -0 rg -l needle"
    ));
    assert!(refused("find / -type f -exec grep -l needle {} +"));
    assert!(refused("cd / && find . -type f | xargs grep -l needle"));
    assert!(refused("find / -type f | (xargs grep -l needle)"));
    assert!(refused(
        "find / -type f | { xargs -n 1 -I{} grep -l needle {}; }"
    ));
    assert!(refused(
        "find / -type f -print0 | xargs -0 -P4 timeout 60 grep -l needle"
    ));
}

#[test]
fn allows_find_feeds_that_filter_names_or_skip_proc() {
    assert!(!refused(
        "find / -name '*.py' 2>/dev/null | xargs grep -l needle"
    ));
    assert!(!refused(
        "find / -path /proc -prune -o -type f -print | xargs grep -l needle"
    ));
    assert!(!refused("find / -xdev -type f | xargs grep -l needle"));
    assert!(!refused(
        "find / -type f -executable 2>/dev/null | xargs grep -l needle | head"
    ));
    assert!(!refused("find /testbed -type f | xargs grep -l needle"));
    assert!(!refused(
        "find / -type f | wc -l; ls | xargs grep -l needle"
    ));
    assert!(!refused("find / -type f || xargs grep -l needle"));
    assert!(!refused("find / -type f | xargs echo grep"));
    assert!(!refused("find / -type f | xargs -I grep echo grep"));
}

#[test]
fn names_the_find_filter_that_fixes_the_search() {
    assert_eq!(
        root_search_refusal(
            "find / -type f | xargs grep -l needle",
            Some(Path::new("/testbed"))
        ),
        Some(
            "Command blocked: Blocked: grep over the files find / lists reads everything under \
             /proc, /sys and /dev, and does not finish within the 10-minute command limit. The \
             current working directory is /testbed. Give find a -name filter, or search a \
             narrower path."
                .to_string()
        )
    );
}

#[test]
fn tracks_cd_within_the_subshell_it_runs_in() {
    assert!(refused(
        "cd / && echo $(cd /tmp && pwd) && grep -rl needle ."
    ));
    assert!(refused("(cd / && grep -rl needle .)"));
    assert!(!refused("(cd / && true) && grep -rl needle ."));
}

#[test]
fn refuses_root_searches_substituted_into_unquoted_heredocs() {
    assert!(refused("cat <<EOF\nfound: $(rg -l needle /)\nEOF"));
    assert!(!refused("cat <<'EOF'\nfound: $(rg -l needle /)\nEOF"));
}

#[test]
fn refuses_ripgrep_over_root() {
    assert!(refused("rg -l needle /"));
}

#[test]
fn allows_searches_that_exclude_proc() {
    assert!(!refused("grep -rl needle / --exclude-dir=proc"));
    assert!(!refused("rg -g '!proc' -l needle /"));
}

#[test]
fn allows_searches_of_specific_directories() {
    assert!(!refused("grep -rl needle /app/node_modules /root /opt"));
}

#[test]
fn allows_non_recursive_and_non_search_commands() {
    assert!(!refused("grep needle /etc/hosts"));
    assert!(!refused("find / -name '*.pth'"));
    assert!(!refused("echo 'grep -r needle /'"));
}

#[test]
fn treats_slash_as_the_pattern_when_it_comes_first() {
    assert!(!refused("grep -r / src"));
    assert!(refused("grep -r -e needle /"));
}

#[test]
fn names_the_working_directory_and_the_flags_that_fix_the_search() {
    assert_eq!(
        root_search_refusal("grep -r needle /", Some(Path::new("/testbed"))),
        Some(
            "Command blocked: Blocked: recursive grep over / walks the whole filesystem, \
             including /proc, /sys and /dev, and does not finish within the 10-minute command \
             limit. The current working directory is /testbed. Search a narrower path, or add \
             --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev."
                .to_string()
        )
    );
}

#[test]
fn leaves_out_the_working_directory_when_it_is_unknown() {
    assert_eq!(
        root_search_refusal("rg -l needle /", None),
        Some(
            "Command blocked: Blocked: recursive rg over / walks the whole filesystem, \
             including /proc, /sys and /dev, and does not finish within the 10-minute command \
             limit. Search a narrower path, or add -g '!proc' -g '!sys' -g '!dev'."
                .to_string()
        )
    );
}

#[tokio::test]
async fn guard_refuses_a_root_search_without_running_it() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = guarded_bash(&workspace)
        .execute(
            r#"{"command": "touch ran; grep -rl needle / 2>/dev/null | head"}"#,
            &ExecutionContext::default(),
        )
        .await;

    assert!(!result.success);
    assert!(result
        .content
        .starts_with("Command blocked: Blocked: recursive grep over /"));
    assert!(!workspace.path().join("ran").exists());
}

#[tokio::test]
async fn guard_runs_other_commands() {
    let workspace = tempfile::tempdir().expect("temporary workspace");

    let result = guarded_bash(&workspace)
        .execute(r#"{"command": "touch ran"}"#, &ExecutionContext::default())
        .await;

    assert!(result.success, "{}", result.content);
    assert!(workspace.path().join("ran").exists());
}
