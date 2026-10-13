use std::fs;
use std::path::Path;
use std::process::Command;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;

fn git(path: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(args)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A checkout whose tools' root is a subdirectory of the repository toplevel.
fn monorepo() -> (tempfile::TempDir, LocalWorkspace) {
    let directory = tempfile::tempdir().expect("directory");
    let toplevel = directory.path().join("project");
    fs::create_dir_all(toplevel.join("crates/agent/src")).expect("create tree");
    fs::write(toplevel.join("crates/agent/src/lib.rs"), "pub fn f() {}\n").expect("write");
    git(&toplevel, &["init", "-q"]);
    git(
        &toplevel,
        &[
            "remote",
            "add",
            "origin",
            "git@github.com:example/project.git",
        ],
    );
    let workspace = LocalWorkspace::new(&toplevel.join("crates/agent")).expect("workspace");
    (directory, workspace)
}

#[tokio::test]
async fn names_the_repository_and_relativizes_to_its_toplevel() {
    let (_directory, workspace) = monorepo();

    assert_eq!(workspace.repository(), Some("github.com/example/project"));
    assert_eq!(
        workspace.relative("src/lib.rs").as_deref(),
        Some("crates/agent/src/lib.rs"),
        "relative input resolves against the tools' root"
    );
    assert_eq!(
        workspace.relative("src/new_file.rs").as_deref(),
        Some("crates/agent/src/new_file.rs"),
        "a file that does not exist yet is named lexically"
    );
    assert_matches!(workspace.relative("../../../outside.rs"), None);
}

#[tokio::test]
async fn inspect_reports_what_is_at_each_checkout_relative_path() {
    let (_directory, workspace) = monorepo();
    let paths = [
        "crates/agent/src/lib.rs".to_string(),
        "crates/agent/src".to_string(),
        "crates/agent/Cargo.lock".to_string(),
    ];

    let states = workspace.inspect(&paths).await.expect("inspect");
    assert_eq!(
        states,
        vec![PathState::File, PathState::Directory, PathState::Missing]
    );
}

#[tokio::test]
async fn outside_a_repository_there_is_no_name_and_paths_resolve_against_the_root() {
    let directory = tempfile::tempdir().expect("directory");
    fs::write(directory.path().join("notes.txt"), "notes").expect("write");
    let workspace = LocalWorkspace::new(directory.path()).expect("workspace");

    assert_matches!(workspace.repository(), None);
    assert_matches!(workspace.relative("notes.txt"), None);
    assert_eq!(
        workspace
            .inspect(&["notes.txt".to_string()])
            .await
            .expect("inspect"),
        vec![PathState::File]
    );
}
