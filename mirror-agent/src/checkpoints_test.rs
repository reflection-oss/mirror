use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use assert_matches::assert_matches;
use mirror_protocol::Checkpoint;
use pretty_assertions::assert_eq;

use super::{CheckpointError, Checkpoints};

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_AUTHOR_NAME", "Test")
        .env("GIT_AUTHOR_EMAIL", "test@localhost")
        .env("GIT_COMMITTER_NAME", "Test")
        .env("GIT_COMMITTER_EMAIL", "test@localhost")
        .output()
        .expect("run Git");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("UTF-8 output")
        .trim_end_matches('\n')
        .to_string()
}

async fn capture(root: &Path) -> Checkpoint {
    Checkpoints::new(root.to_path_buf())
        .capture()
        .await
        .expect("capture")
        .expect("Git repo")
}

#[tokio::test]
async fn captures_worktree_without_changing_head_index_or_files() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    git(root, &["init", "--quiet"]);
    fs::write(root.join("tracked"), "base").expect("tracked file");
    fs::write(root.join("deleted"), "base").expect("deleted file");
    fs::write(root.join(".gitignore"), "ignored\n").expect("ignore rules");
    git(root, &["add", "."]);
    git(root, &["commit", "--quiet", "-m", "base"]);
    let head = git(root, &["rev-parse", "HEAD"]);
    fs::write(root.join("tracked"), "staged").expect("staged contents");
    git(root, &["add", "tracked"]);
    fs::write(root.join("tracked"), "working").expect("unstaged contents");
    fs::remove_file(root.join("deleted")).expect("remove tracked file");
    fs::write(root.join("untracked"), [0, 255, 42]).expect("untracked binary");
    fs::write(root.join("ignored"), "private").expect("ignored file");
    let index = fs::read(root.join(".git/index")).expect("index before capture");
    let status = git(root, &["status", "--porcelain"]);
    let hook = root.join(".git/hooks/post-index-change");
    fs::write(&hook, "#!/bin/sh\nprintf invoked > .git/hook-ran\n").expect("hook");
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).expect("executable hook");
    let snapshot = capture(root).await;
    let commit = snapshot.snapshot_commit;

    assert!(
        !root.join(".git/hook-ran").exists(),
        "capture must not invoke repository hooks"
    );
    assert_eq!(snapshot.head_commit, Some(head.clone()));
    assert_eq!(git(root, &["rev-parse", "HEAD"]), head);
    assert_eq!(
        fs::read(root.join(".git/index")).expect("index after capture"),
        index
    );
    assert_eq!(git(root, &["status", "--porcelain"]), status);
    assert_eq!(
        fs::read_to_string(root.join("tracked")).expect("worktree"),
        "working"
    );
    assert_eq!(
        git(root, &["show", &format!("{commit}:tracked")]),
        "working"
    );
    assert_eq!(
        git(root, &["ls-tree", "--name-only", &commit]),
        ".gitignore\ntracked\nuntracked"
    );
    assert_eq!(git(root, &["for-each-ref", "refs/mirror/checkpoints/"]), "");
    assert_eq!(
        git(root, &["rev-parse", &format!("{commit}:untracked")]),
        git(root, &["hash-object", "untracked"])
    );

    git(root, &["commit", "--quiet", "-m", "new base"]);
    let new_head = git(root, &["rev-parse", "HEAD"]);
    let next = capture(root).await;
    let next_commit = next.snapshot_commit;
    assert_eq!(next.head_commit, Some(new_head.clone()));
    assert_eq!(
        git(root, &["rev-parse", &format!("{next_commit}^")]),
        new_head
    );
    assert_eq!(
        git(root, &["rev-parse", &format!("{next_commit}^{{tree}}")]),
        git(root, &["rev-parse", &format!("{commit}^{{tree}}")])
    );
    git(root, &["prune", "--expire=now"]);
    assert!(!Command::new("git")
        .current_dir(root)
        .args(["cat-file", "-e", &next_commit])
        .output()
        .expect("check pruned snapshot")
        .status
        .success());
}

#[tokio::test]
async fn captures_unborn_repository_and_staged_ignored_files() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    git(root, &["init", "--quiet"]);
    fs::write(root.join(".gitignore"), "tracked\n").expect("ignore rules");
    fs::write(root.join("tracked"), "staged").expect("tracked file");
    git(root, &["add", "--force", "tracked"]);
    fs::write(root.join("tracked"), "working").expect("working contents");
    let snapshot = capture(root).await;
    let commit = snapshot.snapshot_commit;
    assert_matches!(snapshot.head_commit, None);
    assert_eq!(
        git(root, &["show", &format!("{commit}:tracked")]),
        "working"
    );
    assert_eq!(git(root, &["show", ":tracked"]), "staged");
    assert_eq!(git(root, &["rev-list", "--count", &commit]), "1");
}

#[tokio::test]
async fn captures_from_linked_worktree_subdirectory_with_split_index() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    let linked = root.join("linked");
    git(root, &["init", "--quiet"]);
    git(root, &["commit", "--allow-empty", "--quiet", "-m", "base"]);
    git(
        root,
        &[
            "worktree",
            "add",
            "--detach",
            linked.to_str().expect("path"),
        ],
    );
    fs::create_dir(linked.join("subdirectory")).expect("subdirectory");
    fs::write(linked.join("tracked"), "base").expect("file");
    git(&linked, &["add", "."]);
    git(&linked, &["update-index", "--split-index"]);
    fs::write(linked.join("tracked"), "working").expect("file");
    let snapshot = capture(&linked.join("subdirectory")).await;
    let commit = snapshot.snapshot_commit;
    assert_eq!(
        Path::new(&snapshot.repository_root),
        fs::canonicalize(&linked).expect("canonical root")
    );
    assert_eq!(
        git(&linked, &["show", &format!("{commit}:tracked")]),
        "working"
    );
    assert_eq!(git(&linked, &["show", ":tracked"]), "base");
}

#[tokio::test]
async fn captures_modified_files_even_when_index_flags_hide_changes() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    git(root, &["init", "--quiet"]);
    fs::write(root.join("assumed"), "base").expect("file");
    fs::write(root.join("skipped"), "base").expect("file");
    fs::create_dir(root.join("excluded")).expect("excluded directory");
    fs::write(root.join("excluded/file"), "sparse contents").expect("excluded file");
    git(root, &["add", "."]);
    git(root, &["commit", "--quiet", "-m", "base"]);
    git(root, &["sparse-checkout", "set", "--cone", "included"]);
    assert!(!root.join("excluded/file").exists());
    git(root, &["update-index", "--assume-unchanged", "assumed"]);
    git(root, &["update-index", "--skip-worktree", "skipped"]);
    fs::write(root.join("assumed"), "changed").expect("file");
    fs::write(root.join("skipped"), "changed").expect("file");
    let index = fs::read(root.join(".git/index")).expect("index");
    let commit = capture(root).await.snapshot_commit;
    assert_eq!(
        git(root, &["show", &format!("{commit}:assumed")]),
        "changed"
    );
    assert_eq!(
        git(root, &["show", &format!("{commit}:skipped")]),
        "changed"
    );
    assert_eq!(
        git(root, &["show", &format!("{commit}:excluded/file")]),
        "sparse contents"
    );
    assert_eq!(fs::read(root.join(".git/index")).expect("index"), index);
}

#[tokio::test]
async fn captures_same_size_edits_with_unchanged_mtime() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    git(root, &["init", "--quiet"]);
    git(root, &["config", "core.trustctime", "false"]);
    let path = root.join("file");
    fs::write(&path, "before").expect("file");
    let modified = fs::metadata(&path)
        .expect("metadata")
        .modified()
        .expect("mtime");
    git(root, &["add", "."]);
    fs::File::options()
        .write(true)
        .open(root.join(".git/index"))
        .expect("index")
        .set_modified(modified)
        .expect("racy index timestamp");
    fs::write(&path, "after!").expect("edit");
    fs::File::options()
        .write(true)
        .open(&path)
        .expect("file")
        .set_modified(modified)
        .expect("restore mtime");
    let commit = capture(root).await.snapshot_commit;
    assert_eq!(git(root, &["show", &format!("{commit}:file")]), "after!");
}

#[tokio::test]
async fn failed_staging_leaves_no_objects_and_successful_packs_are_readable() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    git(root, &["init", "--quiet"]);
    git(root, &["config", "core.bigFileThreshold", "1k"]);
    fs::write(root.join("aaa.bin"), vec![42; 2048]).expect("large file");
    fs::create_dir(root.join("zzz")).expect("nested repository");
    git(&root.join("zzz"), &["init", "--quiet"]);
    let before = git(root, &["count-objects", "-v"]);
    assert_matches!(
        Checkpoints::new(root.to_path_buf()).capture().await,
        Err(CheckpointError::Git { .. })
    );
    assert_eq!(git(root, &["count-objects", "-v"]), before);
    assert!(!root.join(".git/index").exists());
    fs::remove_dir_all(root.join("zzz")).expect("remove unborn nested repository");
    let commit = capture(root).await.snapshot_commit;
    assert_eq!(
        git(root, &["rev-parse", &format!("{commit}:aaa.bin")]),
        git(root, &["hash-object", "aaa.bin"])
    );
    git(root, &["fsck", "--no-reflogs"]);
}

#[tokio::test]
async fn captures_edits_through_a_new_external_hardlink() {
    let directory = tempfile::tempdir().expect("workspace");
    let external = tempfile::tempdir().expect("external files");
    let root = directory.path();
    git(root, &["init", "--quiet"]);
    fs::write(root.join("file"), "before").expect("file");
    let checkpoints = Checkpoints::new(root.to_path_buf());
    checkpoints.capture().await.expect("initial capture");

    let alias = external.path().join("alias");
    fs::hard_link(root.join("file"), &alias).expect("create external alias");
    fs::write(&alias, "after!").expect("edit through external alias");
    let snapshot = checkpoints
        .capture()
        .await
        .expect("capture")
        .expect("Git repo");

    assert_eq!(
        git(
            root,
            &["show", &format!("{}:file", snapshot.snapshot_commit)]
        ),
        "after!"
    );
}

#[tokio::test]
async fn captures_replacement_repository_after_ancestor_rename() {
    let directory = tempfile::tempdir().expect("workspace parent");
    let parent = directory.path().join("parent");
    let root = parent.join("repo");
    fs::create_dir_all(&root).expect("repository directory");
    git(&root, &["init", "--quiet"]);
    fs::write(root.join("file"), "before").expect("file");
    let checkpoints = Checkpoints::new(root.clone());
    checkpoints.capture().await.expect("initial capture");

    fs::rename(&parent, directory.path().join("old-parent")).expect("move ancestor");
    fs::create_dir_all(&root).expect("replacement repository directory");
    git(&root, &["init", "--quiet"]);
    fs::write(root.join("file"), "after!").expect("replacement file");
    let snapshot = checkpoints
        .capture()
        .await
        .expect("capture")
        .expect("Git repo");

    assert_eq!(
        git(
            &root,
            &["show", &format!("{}:file", snapshot.snapshot_commit)]
        ),
        "after!"
    );
}

#[tokio::test]
async fn omits_checkpoint_outside_git() {
    let directory = tempfile::tempdir().expect("workspace");
    let checkpoints = Checkpoints::new(directory.path().to_path_buf());
    assert_matches!(checkpoints.capture().await, Ok(None));
    git(directory.path(), &["init", "--bare", "--quiet"]);
    assert_matches!(checkpoints.capture().await, Ok(None));
}

fn jj(root: &Path, args: &[&str]) -> String {
    let binary = std::env::var_os("JJ_TEST_BIN")
        .map(|path| {
            fs::canonicalize(path)
                .expect("jj test binary")
                .into_os_string()
        })
        .unwrap_or_else(|| "jj".into());
    let output = Command::new(binary)
        .current_dir(root)
        .args([
            "--config",
            "user.name='Test'",
            "--config",
            "user.email='test@localhost'",
        ])
        .args(args)
        .env("JJ_CONFIG", "/dev/null")
        .output()
        .expect("run jj");
    assert!(
        output.status.success(),
        "jj {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("UTF-8 output")
        .trim_end_matches('\n')
        .to_string()
}

fn jj_head(root: &Path) -> String {
    jj(
        root,
        &[
            "--ignore-working-copy",
            "log",
            "--no-graph",
            "-r",
            "@",
            "-T",
            "commit_id",
        ],
    )
}

fn jj_operation(root: &Path) -> String {
    jj(
        root,
        &[
            "--ignore-working-copy",
            "op",
            "log",
            "--limit",
            "1",
            "--no-graph",
            "-T",
            "id",
        ],
    )
}

fn jj_snapshot_file(root: &Path, commit: &str, path: &str) -> String {
    let git_dir = jj(root, &["--ignore-working-copy", "git", "root"]);
    git(
        root,
        &["--git-dir", &git_dir, "show", &format!("{commit}:{path}")],
    )
}

#[tokio::test]
async fn captures_jj_without_colocation_or_updating_the_working_copy() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    jj(root, &["git", "init", "--no-colocate"]);
    fs::write(root.join("tracked"), "before").expect("tracked file");
    fs::write(root.join("deleted"), "before").expect("deleted file");
    fs::write(root.join(".gitignore"), "ignored\n").expect("ignore rules");
    jj(root, &["bookmark", "create", "base"]);
    let head = jj_head(root);
    let operation = jj_operation(root);
    let checkout = fs::read(root.join(".jj/working_copy/checkout")).expect("checkout state");
    let tree_state = fs::read(root.join(".jj/working_copy/tree_state")).expect("tree state");
    fs::write(root.join("tracked"), "after").expect("edit");
    fs::remove_file(root.join("deleted")).expect("delete");
    fs::write(root.join("untracked"), "new").expect("untracked file");
    fs::write(root.join("ignored"), "private").expect("ignored file");
    fs::create_dir(root.join("nested")).expect("subdirectory");

    let checkpoint = capture(&root.join("nested")).await;

    assert_eq!(checkpoint.head_commit, Some(head.clone()));
    assert_eq!(
        Path::new(&checkpoint.repository_root),
        fs::canonicalize(root).expect("root")
    );
    assert_eq!(
        jj_snapshot_file(root, &checkpoint.snapshot_commit, "tracked"),
        "after"
    );
    assert_eq!(
        jj_snapshot_file(root, &checkpoint.snapshot_commit, "untracked"),
        "new"
    );
    let git_dir = jj(root, &["--ignore-working-copy", "git", "root"]);
    assert_eq!(
        git(
            root,
            &[
                "--git-dir",
                &git_dir,
                "ls-tree",
                "--name-only",
                &checkpoint.snapshot_commit
            ]
        ),
        ".gitignore\ntracked\nuntracked"
    );
    assert_eq!(jj_head(root), head);
    assert_eq!(jj_operation(root), operation);
    assert_eq!(
        fs::read(root.join(".jj/working_copy/checkout")).expect("checkout state"),
        checkout
    );
    assert_eq!(
        fs::read(root.join(".jj/working_copy/tree_state")).expect("tree state"),
        tree_state
    );
    assert_eq!(
        fs::read_to_string(root.join("tracked")).expect("worktree"),
        "after"
    );
    assert_eq!(
        jj(
            root,
            &[
                "--ignore-working-copy",
                "log",
                "--no-graph",
                "-r",
                "base",
                "-T",
                "commit_id"
            ]
        ),
        head
    );
}

#[tokio::test]
async fn captures_the_selected_jj_workspace_not_the_shared_git_checkout() {
    let directory = tempfile::tempdir().expect("workspace parent");
    let root = directory.path();
    jj(root, &["git", "init", "--colocate"]);
    fs::write(root.join("file"), "original").expect("file");
    jj(root, &["new"]);
    jj(root, &["workspace", "add", "--name", "second", "second"]);
    let second = root.join("second");
    fs::write(second.join("file"), "second workspace").expect("edit");
    let operation = jj_operation(root);
    let original_head = jj_head(root);
    let second_head = jj_head(&second);

    let checkpoint = capture(&second).await;

    assert_eq!(checkpoint.head_commit, Some(second_head.clone()));
    assert_eq!(
        jj_snapshot_file(&second, &checkpoint.snapshot_commit, "file"),
        "second workspace"
    );
    assert_eq!(jj_head(root), original_head);
    assert_eq!(jj_head(&second), second_head);
    assert_eq!(jj_operation(root), operation);
    assert_eq!(
        fs::read_to_string(root.join("file")).expect("original file"),
        "original"
    );
}

#[tokio::test]
async fn colocated_jj_capture_preserves_git_head_and_index() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    jj(root, &["git", "init", "--colocate"]);
    fs::write(root.join("file"), "base").expect("file");
    jj(root, &["new"]);
    let head = git(root, &["rev-parse", "HEAD"]);
    let index = fs::read(root.join(".git/index")).expect("index");
    let operation = jj_operation(root);
    fs::write(root.join("file"), "working").expect("edit");

    let checkpoint = capture(root).await;

    assert_eq!(
        jj_snapshot_file(root, &checkpoint.snapshot_commit, "file"),
        "working"
    );
    assert_eq!(git(root, &["rev-parse", "HEAD"]), head);
    assert_eq!(fs::read(root.join(".git/index")).expect("index"), index);
    assert_eq!(jj_operation(root), operation);
}

#[tokio::test]
async fn jj_capture_preserves_sparse_files() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    jj(root, &["git", "init", "--no-colocate"]);
    fs::write(root.join("included"), "base").expect("file");
    fs::write(root.join("excluded"), "sparse contents").expect("file");
    jj(root, &["sparse", "set", "--clear", "--add", "included"]);
    assert!(!root.join("excluded").exists());
    fs::write(root.join("included"), "edited").expect("edit");

    let checkpoint = capture(root).await;

    assert_eq!(
        jj_snapshot_file(root, &checkpoint.snapshot_commit, "included"),
        "edited"
    );
    assert_eq!(
        jj_snapshot_file(root, &checkpoint.snapshot_commit, "excluded"),
        "sparse contents"
    );
    assert!(!root.join("excluded").exists());
}

#[tokio::test]
async fn jj_capture_includes_large_new_files_when_auto_tracking_is_disabled() {
    let directory = tempfile::tempdir().expect("workspace");
    let root = directory.path();
    jj(root, &["git", "init", "--no-colocate"]);
    jj(
        root,
        &["config", "set", "--repo", "snapshot.auto-track", "none()"],
    );
    fs::write(root.join("large"), vec![b'x'; 2 * 1024 * 1024]).expect("large new file");
    let operation = jj_operation(root);

    let checkpoint = capture(root).await;

    assert_eq!(
        jj_snapshot_file(root, &checkpoint.snapshot_commit, "large").len(),
        2 * 1024 * 1024
    );
    assert_eq!(jj_operation(root), operation);
    assert_eq!(
        jj(
            root,
            &[
                "--ignore-working-copy",
                "config",
                "get",
                "snapshot.auto-track"
            ]
        ),
        "none()"
    );
}

#[tokio::test]
async fn broken_jj_workspace_reports_failure_instead_of_skipping_capture() {
    let directory = tempfile::tempdir().expect("workspace");
    fs::create_dir(directory.path().join(".jj")).expect("invalid jj marker");

    assert_matches!(
        Checkpoints::new(directory.path().to_path_buf())
            .capture()
            .await,
        Err(CheckpointError::Jujutsu(_))
    );
}
