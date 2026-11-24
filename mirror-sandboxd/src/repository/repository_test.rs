use std::ffi::OsString;
use std::fs;
use std::net::{Ipv4Addr, TcpListener};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;

use pretty_assertions::assert_eq;
use prost::Message;
use tokio::io::AsyncWriteExt;

use super::*;
use crate::tool_runner::REQUEST_BYTES_VARIABLE;

const PROJECT_URL: &str = "https://github.com/acme/project.git";

fn write(root: &Path, path: &str, content: &str) {
    let path = root.join(path);
    fs::create_dir_all(path.parent().expect("parent")).expect("create directories");
    fs::write(path, content).expect("write fixture");
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("workspace");
    fs::create_dir(root.path().join(".git")).expect("git root");
    write(root.path(), "AGENTS.md", "Root guidance.");
    write(root.path(), "CLAUDE.md", "Shadowed root guidance.");
    write(root.path(), "src/CLAUDE.md", "Nested guidance.");
    write(
        root.path(),
        ".agents/skills/review/SKILL.md",
        "---\nname: review\ndescription: Review code.\n---\nPinned skill body.",
    );
    root
}

fn context(root: &Path) -> std::io::Result<mirror_agent::system_prompt::ResolvedInstructions> {
    mirror_agent::system_prompt::load_workspace_context(&root.join("src"), root)
}

fn seed_git(directory: &Path, args: &[&str]) -> String {
    let output = std::process::Command::new("git")
        .current_dir(directory)
        .args([
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
        ])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .expect("Git is required by the local checkout tests");
    assert!(output.status.success(), "seed Git command failed");
    String::from_utf8(output.stdout)
        .expect("Git output")
        .trim()
        .to_string()
}

struct LocalGitHub {
    repositories: tempfile::TempDir,
    tools: tempfile::TempDir,
}

impl LocalGitHub {
    fn start() -> Self {
        let repositories = tempfile::tempdir().expect("repositories");
        let tools = tempfile::tempdir().expect("tools");
        let git = host_git();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("daemon listener");
        let port = listener.local_addr().expect("daemon address").port();
        let base = format!("--base-path={}", repositories.path().display());
        let daemon = git.clone();
        std::thread::spawn(move || serve_git_daemon(&listener, &daemon, &base));
        let wrapper = tools.path().join("git");
        fs::write(
            &wrapper,
            format!(
                r#"#!/bin/sh
[ "$GIT_SSL_CAINFO" = '{ca}' ] || exit 1
[ -z "${{GIT_DIR+x}}" ] || exit 1
exec '{git}' -c url.git://127.0.0.1:{port}/.insteadOf=https://github.com/ "$@"
"#,
                ca = mirror_sandbox::EGRESS_CA_PATH,
                git = git.display()
            ),
        )
        .expect("git wrapper");
        fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).expect("wrapper mode");
        Self {
            repositories,
            tools,
        }
    }

    fn search_path(&self) -> OsString {
        let inherited = std::env::var_os("PATH").unwrap_or_default();
        std::env::join_paths(
            std::iter::once(self.tools.path().to_path_buf())
                .chain(std::env::split_paths(&inherited)),
        )
        .expect("guest PATH")
    }
}

fn serve_git_daemon(listener: &TcpListener, git: &Path, base: &str) {
    for connection in listener.incoming().flatten() {
        let Ok(output) = connection.try_clone() else {
            return;
        };
        let Ok(mut daemon) = std::process::Command::new(git)
            .args([
                "daemon",
                "--inetd",
                "--export-all",
                "--log-destination=none",
                base,
            ])
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .stdin(OwnedFd::from(connection))
            .stdout(OwnedFd::from(output))
            .stderr(Stdio::null())
            .spawn()
        else {
            return;
        };
        std::thread::spawn(move || daemon.wait());
    }
}

fn host_git() -> PathBuf {
    std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|directory| directory.join("git"))
        .find(|candidate| candidate.is_file())
        .expect("Git is required by the checkout tests")
}

fn project() -> RepositoryDefinition {
    RepositoryDefinition {
        url: PROJECT_URL.into(),
        revision: "main".into(),
        ..RepositoryDefinition::default()
    }
}

fn project_origin(github: &LocalGitHub) -> PathBuf {
    let origin = github.repositories.path().join("acme/project.git");
    fs::create_dir_all(&origin).expect("origin");
    seed_git(&origin, &["init", "--quiet", "--initial-branch=main"]);
    write(&origin, "AGENTS.md", "Earlier guidance.");
    seed_git(&origin, &["add", "."]);
    seed_git(&origin, &["commit", "--quiet", "-m", "initial"]);
    for index in 0..100 {
        write(
            &origin,
            &format!("files/{index}.txt"),
            &format!("{index}\n"),
        );
    }
    write(&origin, "AGENTS.md", "Seed guidance.");
    write(&origin, "value.txt", "original\n");
    seed_git(&origin, &["add", "."]);
    seed_git(&origin, &["commit", "--quiet", "-m", "fixture"]);
    origin
}

fn update_origin(origin: &Path) -> String {
    write(origin, "value.txt", "updated\n");
    seed_git(origin, &["commit", "--quiet", "--all", "-m", "update"]);
    seed_git(origin, &["rev-parse", "HEAD"])
}

fn project_seed(root: &Path, origin: &Path) -> PathBuf {
    let seed = root.join("project");
    fs::create_dir(&seed).expect("seed");
    seed_git(&seed, &["init", "--quiet"]);
    seed_git(&seed, &["remote", "add", "origin", PROJECT_URL]);
    let source = format!("file://{}", origin.display());
    seed_git(
        &seed,
        &[
            "-c",
            "protocol.file.allow=always",
            "fetch",
            "--quiet",
            "--depth=1",
            &source,
            "main",
        ],
    );
    seed_git(&seed, &["checkout", "--quiet", "--detach", "FETCH_HEAD"]);
    seed
}

fn object_counts(repository: &Path) -> (u64, u64) {
    let counts = seed_git(repository, &["count-objects", "-v"]);
    let count = |key: &str| {
        counts
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .and_then(|value| value.trim().parse().ok())
            .expect("object count")
    };
    (count("count:"), count("in-pack:"))
}

async fn prepare_in_guest(
    sandbox: &Path,
    github: &LocalGitHub,
    request: &RepositoryDefinition,
    arguments: &[&str],
) -> Output {
    prepare_in_guest_declaring(sandbox, github, request, arguments, "").await
}

async fn prepare_in_guest_declaring(
    sandbox: &Path,
    github: &LocalGitHub,
    request: &RepositoryDefinition,
    arguments: &[&str],
    disposable_paths: &str,
) -> Output {
    let disposable_paths = [(DISPOSABLE_PATHS_VARIABLE, PathBuf::from(disposable_paths))];
    prepare_in_guest_with(sandbox, github, request, arguments, &disposable_paths).await
}

async fn prepare_in_pod(
    sandbox: &Path,
    pod: &Path,
    github: &LocalGitHub,
    request: &RepositoryDefinition,
    arguments: &[&str],
) -> Output {
    fs::create_dir_all(pod.join("tmp")).expect("pod temporary directory");
    let pod = [("HOME", pod.join("home")), ("TMPDIR", pod.join("tmp"))];
    prepare_in_guest_with(sandbox, github, request, arguments, &pod).await
}

async fn prepare_in_guest_with(
    sandbox: &Path,
    github: &LocalGitHub,
    request: &RepositoryDefinition,
    arguments: &[&str],
    environment: &[(&str, PathBuf)],
) -> Output {
    let binary = fs::canonicalize(std::env::var("SANDBOXD_BIN").expect("guest binary"))
        .expect("resolve guest binary");
    let request = request.encode_to_vec();
    let mut worker = tokio::process::Command::new(&binary)
        .arg("prepare-repository")
        .args(arguments)
        .current_dir(sandbox)
        .env("PATH", github.search_path())
        .env("GIT_SSL_CAINFO", mirror_sandbox::EGRESS_CA_PATH)
        .env("GIT_DIR", sandbox.join("unexpected-git-dir"))
        .env(REQUEST_BYTES_VARIABLE, request.len().to_string())
        .env(DISPOSABLE_PATHS_VARIABLE, "")
        .envs(environment.iter().cloned())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("start guest entrypoint");
    worker
        .stdin
        .take()
        .expect("worker stdin")
        .write_all(&request)
        .await
        .expect("send request");
    worker.wait_with_output().await.expect("guest exits")
}

#[test]
fn linked_skill_roots_are_deduplicated() {
    let root = fixture();
    fs::create_dir(root.path().join(".claude")).expect("claude");
    symlink("../.agents/skills", root.path().join(".claude/skills")).expect("alias");
    let resolved = context(root.path()).expect("context");
    assert_eq!(resolved.skills.len(), 1);
    assert_eq!(resolved.skills[0].name, "review");
}

#[test]
fn working_directories_require_unambiguous_instruction_ancestry() {
    let root = fixture();
    symlink("src", root.path().join("alias")).expect("working directory alias");
    assert!(repository_checkout::validate_directory(root.path(), "src").is_ok());
    assert!(repository_checkout::validate_directory(root.path(), "alias").is_err());
}

#[test]
fn unavailable_instruction_targets_are_errors_not_absent_guidance() {
    let root = fixture();
    fs::remove_file(root.path().join("AGENTS.md")).expect("remove target");
    symlink("missing.md", root.path().join("AGENTS.md")).expect("unavailable instruction target");
    let result = context(root.path());
    assert!(result.is_err());
}

#[test]
fn context_rejects_oversized_files() {
    let root = fixture();
    fs::write(root.path().join("AGENTS.md"), vec![b'x'; 256 * 1024 + 1]).expect("large");
    assert!(context(root.path()).is_err());
}

#[test]
fn context_discovers_nested_collections_and_matches_local_loading() {
    let root = fixture();
    write(
        root.path(),
        ".agents/skills/collection/nested/SKILL.md",
        "---\nname: nested\ndescription: Nested skill.\n---\nNested body.",
    );
    let guest = context(root.path()).expect("guest context");
    let empty = root.path().join("no-global");
    let local = mirror_agent::system_prompt::ResolvedInstructions {
        instructions: mirror_agent::system_prompt::load_instructions(
            &root.path().join("src"),
            Some(&empty),
        ),
        skills: mirror_agent::system_prompt::load_skills(&root.path().join("src"), Some(&empty)),
    };
    assert_eq!(guest, local);
    assert!(guest.skills.iter().any(|skill| skill.name == "nested"));
    assert!(!guest
        .instructions
        .iter()
        .any(|(_, text)| text.contains("Shadowed root guidance")));
}

#[tokio::test]
async fn credential_urls_and_unowned_checkouts_fail_without_exposing_credentials() {
    let sandbox = tempfile::tempdir().expect("sandbox");
    let request = RepositoryDefinition {
        url: "https://user:TOP_SECRET@example.com/repo.git".to_string(),
        revision: "main".to_string(),
        ..RepositoryDefinition::default()
    };
    let error = prepare_checkout(sandbox.path(), &request, &[])
        .await
        .expect_err("credential URL")
        .to_string();
    assert!(!error.contains("TOP_SECRET"));
    write(sandbox.path(), "repository/user-work", "untouched");
    let request = RepositoryDefinition {
        url: "https://example.com/repo.git".to_string(),
        ..request
    };
    assert!(prepare_checkout(sandbox.path(), &request, &[])
        .await
        .is_err());
    assert_eq!(
        fs::read_to_string(sandbox.path().join("repository/user-work")).expect("work"),
        "untouched"
    );
}

#[tokio::test]
async fn seeded_checkout_fetches_only_objects_the_seed_lacks_and_owns_them() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    let shared = seed_git(&seed, &["rev-parse", "HEAD"]);
    seed_git(
        &seed,
        &[
            "config",
            "http.extraHeader",
            "Authorization: SECRET_SEED_HEADER",
        ],
    );
    seed_git(
        &seed,
        &["config", "credential.helper", "!echo SECRET_SEED_HELPER"],
    );
    write(&seed, "untracked-private-file", "SECRET_SEED_FILE");
    write(&seed, "value.txt", "dirty seed work\n");
    assert_eq!(object_counts(&seed), (0, 105));
    let head = update_origin(&origin);

    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(output.status.success());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    let checkout = prepared.stages.expect("stage timings").checkout;
    assert!(checkout.seeded && !checkout.reused);
    assert!(checkout.seed_lag_seconds.is_some_and(|lag| lag >= 0));
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(object_counts(&work), (3, 105));
    assert_eq!(
        fs::read_to_string(work.join("value.txt")).expect("checkout"),
        "updated\n"
    );
    assert!(!work.join("untracked-private-file").exists());
    assert!(!fs::read_to_string(work.join(".git/config"))
        .expect("config")
        .contains("SECRET"));
    assert!(!work.join(".git/objects/info/alternates").exists());
    assert!(!checkout.tree_adopted);
    assert_eq!(
        fs::read_to_string(seed.join("value.txt")).expect("seed work retained"),
        "dirty seed work\n"
    );
    assert!(seed.join("untracked-private-file").exists());

    fs::remove_dir_all(&seed).expect("sandbox restored without the seed");
    seed_git(&work, &["fsck", "--full", "--no-dangling"]);
    assert_eq!(seed_git(&work, &["log", "--format=%H", &shared]), shared);
    write(&work, "value.txt", "edited\n");
    let retry = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(retry.status.success());
    assert_eq!(
        fs::read_to_string(work.join("value.txt")).expect("user edit retained"),
        "edited\n"
    );
}

#[tokio::test]
async fn disposable_seed_tree_becomes_the_checkout_and_the_seed_is_removed() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    let unchanged_file = fs::metadata(seed.join("files/1.txt"))
        .expect("seed file")
        .ino();
    write(&origin, "added.txt", "added\n");
    seed_git(&origin, &["add", "added.txt"]);
    seed_git(&origin, &["rm", "--quiet", "files/0.txt"]);
    let head = update_origin(&origin);

    let output =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "./project/\n").await;

    assert!(output.status.success());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    let checkout = prepared.stages.expect("stage timings").checkout;
    assert!(checkout.seeded && checkout.tree_adopted);
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(
        fs::metadata(work.join("files/1.txt"))
            .expect("checkout file")
            .ino(),
        unchanged_file
    );
    assert_eq!(
        fs::read_to_string(work.join("value.txt")).expect("changed file"),
        "updated\n"
    );
    assert_eq!(
        fs::read_to_string(work.join("added.txt")).expect("added file"),
        "added\n"
    );
    assert!(!work.join("files/0.txt").exists());
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
    assert!(!seed.exists());
    seed_git(&work, &["fsck", "--full", "--no-dangling"]);
}

#[tokio::test]
async fn disposable_seed_tree_is_merged_without_a_forced_checkout() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    project_seed(sandbox.path(), &origin);
    let head = update_origin(&origin);
    let wrapper = github.tools.path().join("git");
    let refusing_git = fs::read_to_string(&wrapper).expect("git wrapper").replacen(
        "#!/bin/sh\n",
        "#!/bin/sh\ncase \" $* \" in *\" --force \"*|*\" clean \"*) exit 1 ;; esac\n",
        1,
    );
    fs::write(&wrapper, refusing_git).expect("refuse forced checkouts and cleans");

    let output =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "project").await;

    assert!(output.status.success());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    assert!(
        prepared
            .stages
            .expect("stage timings")
            .checkout
            .tree_adopted
    );
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
}

#[tokio::test]
async fn disposable_seed_leaves_entries_its_commit_does_not_track_behind() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    write(&seed, "image-settings", "made by the image");
    let head = update_origin(&origin);

    let output =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "project").await;

    assert!(output.status.success());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    assert!(
        prepared
            .stages
            .expect("stage timings")
            .checkout
            .tree_adopted
    );
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert!(!work.join("image-settings").exists());
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
    assert!(!seed.exists());
}

#[tokio::test]
async fn disposable_seed_brings_what_sits_below_its_tracked_directories() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    write(&seed, "files/image-settings", "made by the image");
    let head = update_origin(&origin);

    let output =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "project").await;

    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(
        seed_git(&work, &["status", "--porcelain"]),
        "?? files/image-settings"
    );
}

#[tokio::test]
async fn disposable_seed_edited_on_a_path_the_revision_changes_is_checked_out_by_force() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    write(&seed, "value.txt", "edited\n");
    // The same length as the tracked content, so only the file's time or content tells.
    write(&seed, "AGENTS.md", "Edit guidance.");
    fs::remove_file(seed.join("files/2.txt")).expect("delete a tracked file");
    write(&seed, "untracked-private-file", "SECRET_SEED_FILE");
    let head = update_origin(&origin);

    let output =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "project").await;

    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(
        fs::read_to_string(work.join("AGENTS.md")).expect("tracked file"),
        "Seed guidance."
    );
    assert_eq!(
        fs::read_to_string(work.join("files/2.txt")).expect("restored file"),
        "2\n"
    );
    assert_eq!(
        fs::read_to_string(work.join("value.txt")).expect("changed file"),
        "updated\n"
    );
    assert!(!work.join("untracked-private-file").exists());
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
}

#[tokio::test]
async fn disposable_seed_with_an_unreadable_index_is_checked_out_in_full() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    fs::write(seed.join(".git/index"), "not an index").expect("damage the index");
    let head = update_origin(&origin);

    let output =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "project").await;

    assert!(output.status.success());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    assert!(
        !prepared
            .stages
            .expect("stage timings")
            .checkout
            .tree_adopted
    );
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
    assert!(!seed.exists());
}

#[tokio::test]
async fn failed_preparation_removes_the_disposable_seed_it_took_the_tree_from() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    let head = update_origin(&origin);
    let missing_directory = RepositoryDefinition {
        directory: "missing".into(),
        ..project()
    };

    let failed =
        prepare_in_guest_declaring(sandbox.path(), &github, &missing_directory, &[], "project")
            .await;

    assert!(!failed.status.success());
    assert!(!seed.exists());
    assert!(!sandbox.path().join("repository").exists());
    let retry = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(retry.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
}

#[tokio::test]
async fn preparation_killed_while_using_a_disposable_seed_leaves_no_partial_seed() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    let head = update_origin(&origin);
    let wrapper = github.tools.path().join("git");
    let working_git = fs::read_to_string(&wrapper).expect("git wrapper");
    let killing_git = working_git.replacen(
        "#!/bin/sh\n",
        "#!/bin/sh\ncase \" $* \" in *\" read-tree \"*) kill -KILL $PPID; exit 1 ;; esac\n",
        1,
    );
    fs::write(&wrapper, killing_git).expect("kill the guest as it merges the fetched revision");

    let killed =
        prepare_in_guest_declaring(sandbox.path(), &github, &project(), &[], "project").await;

    assert!(!killed.status.success());
    assert!(!seed.exists());
    fs::write(&wrapper, working_git).expect("restore Git");
    let retry = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(retry.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), head);
    assert_eq!(seed_git(&work, &["status", "--porcelain", "--ignored"]), "");
}

#[tokio::test]
async fn seed_cloned_from_another_repository_is_not_used() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let seed = project_seed(sandbox.path(), &origin);
    seed_git(
        &seed,
        &[
            "remote",
            "set-url",
            "origin",
            "https://user:SECRET@example.com/project.git",
        ],
    );
    update_origin(&origin);

    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(object_counts(&work), (0, 105));
    assert!(!fs::read_to_string(work.join(".git/config"))
        .expect("config")
        .contains("SECRET"));
}

#[tokio::test]
async fn linked_seed_is_not_used() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    let elsewhere = tempfile::tempdir().expect("outside the sandbox");
    let seed = project_seed(elsewhere.path(), &origin);
    let sandbox = tempfile::tempdir().expect("sandbox");
    symlink(&seed, sandbox.path().join("project")).expect("seed alias");
    update_origin(&origin);

    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(output.status.success());
    assert_eq!(object_counts(&sandbox.path().join("repository")), (0, 105));
}

#[tokio::test]
async fn partial_clone_seed_is_not_used() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    seed_git(&origin, &["config", "uploadpack.allowFilter", "true"]);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let source = format!("file://{}", origin.display());
    seed_git(
        sandbox.path(),
        &[
            "-c",
            "protocol.file.allow=always",
            "clone",
            "--quiet",
            "--no-checkout",
            "--depth=1",
            "--filter=blob:none",
            &source,
            "project",
        ],
    );
    let seed = sandbox.path().join("project");
    seed_git(&seed, &["remote", "set-url", "origin", PROJECT_URL]);
    update_origin(&origin);

    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(output.status.success());
    assert_eq!(object_counts(&sandbox.path().join("repository")), (0, 105));
}

fn workspace_origin(github: &LocalGitHub, setup: &str) -> PathBuf {
    let origin = project_origin(github);
    write(&origin, ".mirror/setup", setup);
    write(
        &origin,
        ".mirror/resume",
        "#!/bin/sh\necho \"$MIRROR_SANDBOX\" >> resumes\n",
    );
    seed_git(&origin, &["add", "."]);
    seed_git(&origin, &["commit", "--quiet", "-m", "workspace scripts"]);
    seed_git(&origin, &["checkout", "--quiet", "-b", "feature"]);
    write(&origin, "value.txt", "feature\n");
    write(
        &origin,
        ".mirror/setup",
        "#!/bin/sh\ntouch branch-setup-ran\n",
    );
    write(
        &origin,
        ".mirror/resume",
        "#!/bin/sh\ntouch branch-resume-ran\n",
    );
    seed_git(&origin, &["commit", "--quiet", "--all", "-m", "feature"]);
    seed_git(&origin, &["checkout", "--quiet", "main"]);
    origin
}

fn feature() -> RepositoryDefinition {
    RepositoryDefinition {
        revision: "feature".into(),
        ..project()
    }
}

#[tokio::test]
async fn setup_runs_on_the_default_branch_before_the_requested_revision_and_resume_after_it() {
    let github = LocalGitHub::start();
    let origin = workspace_origin(
        &github,
        "#!/bin/sh\nprintf '%s %s' \"$MIRROR_SANDBOX\" \"$(cat value.txt)\" > setup-saw\n",
    );
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");

    let output = prepare_in_pod(sandbox.path(), pod.path(), &github, &feature(), &[]).await;

    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(
        seed_git(&work, &["rev-parse", "HEAD"]),
        seed_git(&origin, &["rev-parse", "feature"])
    );
    assert_eq!(
        fs::read_to_string(work.join("setup-saw")).expect("setup output"),
        "1 original"
    );
    assert_eq!(
        fs::read_to_string(work.join("value.txt")).expect("checkout"),
        "feature\n"
    );
    assert_eq!(
        fs::read_to_string(work.join("resumes")).expect("resume output"),
        "1\n"
    );
    assert!(!work.join("branch-setup-ran").exists());
    assert!(!work.join("branch-resume-ran").exists());
    assert!(!work.join(".git/mirror-setup-pending").exists());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    let status = prepared
        .environment
        .and_then(|environment| environment.repository_status)
        .expect("repository status");
    assert!(status.contains("?? setup-saw"), "{status}");
    let checkout = prepared.stages.expect("stage timings").checkout;
    assert_eq!(
        checkout.default_commit,
        Some(seed_git(&origin, &["rev-parse", "main"]))
    );
    assert_eq!(checkout.setup.as_ref().map(ScriptRun::outcome), Some("ok"));
    assert_eq!(checkout.resume.as_ref().map(ScriptRun::outcome), Some("ok"));
}

#[tokio::test]
async fn resume_runs_again_only_after_a_wake() {
    let github = LocalGitHub::start();
    workspace_origin(&github, "#!/bin/sh\n");
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");
    let resumes = sandbox.path().join("repository/resumes");
    let first = prepare_in_pod(sandbox.path(), pod.path(), &github, &project(), &[]).await;
    assert!(first.status.success());

    let reused = prepare_in_pod(
        sandbox.path(),
        pod.path(),
        &github,
        &project(),
        &["--skills-only"],
    )
    .await;
    assert!(reused.status.success());
    assert_eq!(fs::read_to_string(&resumes).expect("resumed once"), "1\n");
    fs::remove_dir_all(pod.path().join("tmp")).expect("restart the pod");
    let woken = prepare_in_pod(
        sandbox.path(),
        pod.path(),
        &github,
        &project(),
        &["--skills-only"],
    )
    .await;

    assert!(woken.status.success());
    assert_eq!(
        fs::read_to_string(&resumes).expect("resumed on wake"),
        "1\n1\n"
    );
}

#[tokio::test]
async fn failed_setup_still_prepares_the_checkout_and_names_its_log() {
    let github = LocalGitHub::start();
    let origin = workspace_origin(&github, "#!/bin/sh\necho broken\nexit 3\n");
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");

    let output = prepare_in_pod(sandbox.path(), pod.path(), &github, &project(), &[]).await;

    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    let main = seed_git(&origin, &["rev-parse", "main"]);
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), main);
    let log = pod.path().join("home/.cache/mirror/setup.log");
    assert_eq!(fs::read_to_string(&log).expect("setup log"), "broken\n");
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    let status = prepared
        .environment
        .and_then(|environment| environment.repository_status)
        .expect("repository status");
    let notice = format!(
        "# Workspace scripts\n- `.mirror/setup` from the default branch at {} exited with \
         status 3 after",
        &main[..12]
    );
    assert!(status.contains(&notice), "{status}");
    assert!(
        status.ends_with(&format!("Its output is in {}.", log.display())),
        "{status}"
    );
}

#[tokio::test]
async fn preparation_stopped_before_the_requested_revision_sets_up_again() {
    let github = LocalGitHub::start();
    workspace_origin(&github, "#!/bin/sh\necho ran >> \"$HOME/setups\"\n");
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");
    let wrapper = github.tools.path().join("git");
    let working_git = fs::read_to_string(&wrapper).expect("git wrapper");
    let killing_git = working_git.replacen(
        "#!/bin/sh\n",
        "#!/bin/sh\ncase \" $* \" in *\" --force \"*) kill -KILL $PPID; exit 1 ;; esac\n",
        1,
    );
    fs::write(&wrapper, killing_git).expect("kill the guest at its forced checkout");
    let killed = prepare_in_pod(sandbox.path(), pod.path(), &github, &feature(), &[]).await;
    assert!(!killed.status.success());
    fs::write(&wrapper, working_git).expect("restore Git");

    let retry = prepare_in_pod(sandbox.path(), pod.path(), &github, &feature(), &[]).await;

    assert!(retry.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(
        fs::read_to_string(work.join("value.txt")).expect("checkout"),
        "feature\n"
    );
    assert_eq!(
        fs::read_to_string(pod.path().join("home/setups")).expect("setup runs"),
        "ran\nran\n"
    );
    assert!(!work.join(".git/mirror-setup-pending").exists());
}

#[tokio::test]
async fn remote_without_a_default_branch_is_checked_out_without_workspace_scripts() {
    let github = LocalGitHub::start();
    let origin = workspace_origin(&github, "#!/bin/sh\ntouch setup-ran\n");
    let feature_commit = seed_git(&origin, &["rev-parse", "feature"]);
    seed_git(&origin, &["symbolic-ref", "HEAD", "refs/heads/absent"]);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");

    let output = prepare_in_pod(sandbox.path(), pod.path(), &github, &feature(), &[]).await;

    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    assert_eq!(seed_git(&work, &["rev-parse", "HEAD"]), feature_commit);
    assert!(!work.join("setup-ran").exists());
    assert!(!work.join("resumes").exists());
    let prepared: PreparedContext = serde_json::from_slice(&output.stdout).expect("context");
    let checkout = prepared.stages.expect("stage timings").checkout;
    assert_eq!(checkout.default_commit, None);
    assert_eq!(checkout.setup, None);
    assert_eq!(checkout.resume, None);
}

#[tokio::test]
async fn failed_fetch_from_a_remote_with_a_default_branch_fails_the_preparation() {
    let github = LocalGitHub::start();
    workspace_origin(&github, "#!/bin/sh\ntouch setup-ran\n");
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");
    let wrapper = github.tools.path().join("git");
    let working_git = fs::read_to_string(&wrapper).expect("git wrapper");
    let failing_git = working_git.replacen(
        "#!/bin/sh\n",
        "#!/bin/sh\ncase \" $* \" in *\" fetch \"*\" HEAD \") exit 1 ;; esac\n",
        1,
    );
    fs::write(&wrapper, failing_git).expect("fail the fetch that names the default branch");

    let output = prepare_in_pod(sandbox.path(), pod.path(), &github, &feature(), &[]).await;

    assert!(!output.status.success());
    assert!(!sandbox.path().join("repository").exists());
}

#[tokio::test]
async fn cancelled_preparation_stops_a_setup_script_that_ignores_sigterm() {
    let github = LocalGitHub::start();
    workspace_origin(
        &github,
        "#!/bin/sh\ntrap '' TERM\nsleep 600 &\necho $! > \"$HOME/child\"\nwhile :; do echo beat >> \"$HOME/beats\"; done\n",
    );
    let sandbox = tempfile::tempdir().expect("sandbox");
    let pod = tempfile::tempdir().expect("pod");
    fs::create_dir_all(pod.path().join("tmp")).expect("pod temporary directory");
    fs::create_dir_all(pod.path().join("home")).expect("pod home");
    let beats = pod.path().join("home/beats");
    let binary = fs::canonicalize(std::env::var("SANDBOXD_BIN").expect("guest binary"))
        .expect("resolve guest binary");
    let request = project().encode_to_vec();
    let command = format!(
        "cd '{}' && PATH='{}' HOME='{}' TMPDIR='{}' GIT_SSL_CAINFO='{}' {}={} exec '{}' \
         prepare-repository",
        sandbox.path().display(),
        github.search_path().to_string_lossy(),
        pod.path().join("home").display(),
        pod.path().join("tmp").display(),
        mirror_sandbox::EGRESS_CA_PATH,
        REQUEST_BYTES_VARIABLE,
        request.len(),
        binary.display(),
    );
    let (host, guest) = tokio::io::duplex(64 * 1024);
    let (input, output) = tokio::io::split(guest);
    let served = tokio::spawn(crate::serve(
        input,
        output,
        crate::RunnerCommand {
            program: "/bin/false".into(),
            args: Vec::new(),
        },
    ));
    let (cancel, cancelled) = tokio::sync::oneshot::channel::<()>();
    let running = tokio::spawn(mirror_sandbox::run_sandbox_command(
        async { Ok(mirror_sandbox::SandboxdStream::split(host)) },
        mirror_sandbox::SandboxStream {
            command,
            workdir: None,
            timeout: Duration::from_secs(120),
            cancellation: Box::pin(async {
                let _ = cancelled.await;
            }),
            stdin: Some(Box::pin(std::io::Cursor::new(request))),
            stdout: Box::pin(tokio::io::sink()),
            stderr: None,
        },
    ));
    tokio::time::timeout(Duration::from_secs(60), async {
        while !beats.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("setup starts");

    cancel.send(()).expect("cancel");
    let result = tokio::time::timeout(mirror_sandbox::STOP_GRACE, running)
        .await
        .expect("the script stops before the command's own grace period ends")
        .expect("command task");

    assert!(
        matches!(&result, Err(mirror_sandbox::SandboxTransportError::Failed(detail)) if detail == "command cancelled; its process group stopped"),
        "{result:?}"
    );
    let stopped_at = fs::metadata(&beats).expect("beats").len();
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(fs::metadata(&beats).expect("beats").len(), stopped_at);
    let child = fs::read_to_string(pod.path().join("home/child")).expect("child");
    assert!(
        !Path::new("/proc").join(child.trim()).exists(),
        "the script's child {child} was left unreaped"
    );
    drop(served);
}

#[test]
fn linked_skill_entrypoints_are_read() {
    let root = fixture();
    let skill = ".agents/skills/review/SKILL.md";
    fs::remove_file(root.path().join(skill)).expect("replace entrypoint");
    write(
        root.path(),
        "shared/review/SKILL.md",
        "---\nname: review\ndescription: Linked review.\n---\nLinked skill body.",
    );
    symlink("../../../shared/review/SKILL.md", root.path().join(skill)).expect("linked entrypoint");
    let resolved = context(root.path()).expect("context");
    assert_eq!(resolved.skills[0].body, "Linked skill body.");
}

#[test]
fn linked_skill_entrypoints_cannot_escape_the_repository() {
    let root = fixture();
    let outside = tempfile::tempdir().expect("outside");
    write(outside.path(), "SKILL.md", "outside instructions");
    let skill = root.path().join(".agents/skills/review/SKILL.md");
    fs::remove_file(&skill).expect("replace entrypoint");
    symlink(outside.path().join("SKILL.md"), skill).expect("outside entrypoint");
    assert!(context(root.path()).is_err());
}

#[tokio::test]
async fn prepare_repository_entrypoint_reads_a_request_and_writes_context_and_environment() {
    let github = LocalGitHub::start();
    let origin = project_origin(&github);
    write(
        &origin,
        ".agents/skills/collection/nested/SKILL.md",
        "---\nname: nested\ndescription: Nested skill.\n---\nEager nested body.",
    );
    seed_git(&origin, &["add", "."]);
    seed_git(&origin, &["commit", "--quiet", "-m", "nested skill"]);
    let commit = seed_git(&origin, &["rev-parse", "HEAD"]);
    let sandbox = tempfile::tempdir().expect("sandbox");

    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(output.status.success());
    let PreparedContext {
        context: prepared,
        environment,
        stages,
    } = serde_json::from_slice(&output.stdout).expect("prepared context");
    assert_eq!(
        stages.map(|stages| (stages.checkout.reused, stages.checkout.seeded)),
        Some((false, false))
    );
    let environment = environment.expect("sandbox environment");
    assert_eq!(
        environment.working_directory,
        fs::canonicalize(sandbox.path())
            .expect("canonical sandbox")
            .join("repository")
    );
    assert!(environment.is_git_repo);
    let snapshot = environment
        .repository_status
        .as_deref()
        .expect("guest Git snapshot");
    assert!(snapshot.contains("nested skill"));
    assert!(snapshot.contains("Status:\n(clean)"));
    assert_eq!(
        seed_git(&sandbox.path().join("repository"), &["rev-parse", "HEAD"]),
        commit
    );
    assert!(prepared
        .instructions
        .iter()
        .any(|(_, text)| text.contains("Seed guidance.")));
    assert!(!prepared
        .instructions
        .iter()
        .any(|(_, text)| text.contains("Eager nested body.")));
    assert_eq!(prepared.skills[0].body, "Eager nested body.");
    assert!(prepared.skills[0]
        .path
        .starts_with(sandbox.path().join("repository")));
    assert_eq!(prepared.skills[0].name, "nested");

    let instructions = sandbox.path().join("repository/AGENTS.md");
    fs::remove_file(&instructions).expect("remove instructions");
    fs::create_dir(&instructions).expect("invalid instruction source");
    let skills_only =
        prepare_in_guest(sandbox.path(), &github, &project(), &["--skills-only"]).await;
    assert!(skills_only.status.success());
    let PreparedContext {
        context: recovered,
        environment: recovered_environment,
        stages: recovered_stages,
    } = serde_json::from_slice(&skills_only.stdout).expect("skills-only context");
    assert!(recovered_stages.expect("stage timings").checkout.reused);
    assert!(recovered.instructions.is_empty());
    assert_eq!(
        recovered_environment,
        Some(mirror_agent::system_prompt::EnvironmentInfo {
            repository_status: None,
            ..environment
        })
    );
    assert_eq!(recovered.skills, prepared.skills);
    let full = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(!full.status.success());
}

#[test]
fn publication_never_replaces_a_concurrent_workspace() {
    use assert_matches::assert_matches;
    let root = tempfile::tempdir().expect("workspace");
    let staging = root.path().join("staging");
    let staged = staging.join("repository");
    let destination = root.path().join("repository");
    write(&staged, ".git/mirror-repository", "selection");
    write(&staged, "src/value", "staged content");
    symlink("src/value", staged.join("alias")).expect("checkout symlink");
    fs::create_dir(&destination).expect("concurrent directory");
    assert_matches!(
        super::repository_checkout::publish_checkout(&staged, &destination),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    );
    fs::remove_dir(&destination).expect("remove empty directory");
    symlink("missing", &destination).expect("concurrent dangling symlink");
    assert_matches!(
        super::repository_checkout::publish_checkout(&staged, &destination),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    );
    assert!(fs::symlink_metadata(&destination)
        .expect("preserved")
        .is_symlink());
    fs::remove_file(&destination).expect("remove symlink");
    super::repository_checkout::publish_checkout(&staged, &destination).expect("publish");
    fs::remove_dir_all(&staging).expect("discard staging");
    assert_eq!(
        fs::read_to_string(destination.join("alias")).expect("published content"),
        "staged content"
    );
    assert_eq!(
        fs::read_link(destination.join("alias")).expect("published symlink"),
        PathBuf::from("src/value")
    );
    assert_eq!(
        fs::read_to_string(destination.join(".git/mirror-repository")).expect("selection"),
        "selection"
    );
}

#[test]
fn renamed_publication_never_claims_a_concurrent_workspace() {
    use assert_matches::assert_matches;
    let root = tempfile::tempdir().expect("workspace");
    let staged = root.path().join("staging/repository");
    let destination = root.path().join("repository");
    write(&staged, ".git/mirror-repository", "selection");
    write(&staged, "src/value", "staged content");
    fs::create_dir(&destination).expect("concurrent directory");
    assert_matches!(
        super::repository_checkout::rename_into_absent(&staged, &destination),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    );
    fs::remove_dir(&destination).expect("remove empty directory");
    symlink("missing", &destination).expect("concurrent dangling symlink");
    assert_matches!(
        super::repository_checkout::rename_into_absent(&staged, &destination),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    );
    fs::remove_file(&destination).expect("remove symlink");
    super::repository_checkout::rename_into_absent(&staged, &destination).expect("publish");
    assert!(fs::symlink_metadata(&staged).is_err());
    assert_eq!(
        fs::read_to_string(destination.join("src/value")).expect("published content"),
        "staged content"
    );
    assert_eq!(
        fs::read_to_string(destination.join(".git/mirror-repository")).expect("selection"),
        "selection"
    );
}

#[tokio::test]
async fn interrupted_linked_publication_is_discarded_and_checked_out_again() {
    let github = LocalGitHub::start();
    project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    write(sandbox.path(), "repository/AGENTS.md", "Partial guidance.");
    write(sandbox.path(), ".mirror-publishing", "");

    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(sandbox.path().join("repository/AGENTS.md")).expect("checkout"),
        "Seed guidance."
    );
    assert!(fs::symlink_metadata(sandbox.path().join(".mirror-publishing")).is_err());
}

#[tokio::test]
async fn unmarked_repository_without_an_unfinished_publication_is_left_alone() {
    let sandbox = tempfile::tempdir().expect("sandbox");
    write(sandbox.path(), "repository/value.txt", "user work");

    assert!(prepare_checkout(sandbox.path(), &project(), &[])
        .await
        .is_err());

    assert_eq!(
        fs::read_to_string(sandbox.path().join("repository/value.txt")).expect("preserved"),
        "user work"
    );
}

#[tokio::test]
async fn prepared_checkout_stays_clean_when_only_change_times_move() {
    let github = LocalGitHub::start();
    project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let output = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(output.status.success());
    let work = sandbox.path().join("repository");
    let alias = sandbox.path().join("value.alias");

    tokio::time::sleep(Duration::from_millis(1100)).await;
    fs::hard_link(work.join("value.txt"), &alias).expect("link a tracked file");
    fs::remove_file(&alias).expect("unlink it again");

    assert_eq!(seed_git(&work, &["diff-files", "--name-only"]), "");
}

#[tokio::test]
async fn retry_rejects_changed_repository_selection_without_modifying_user_work() {
    let github = LocalGitHub::start();
    project_origin(&github);
    let sandbox = tempfile::tempdir().expect("sandbox");
    let definition = project();
    let initial = prepare_in_guest(sandbox.path(), &github, &definition, &[]).await;
    assert!(initial.status.success());
    let work = sandbox.path().join("repository/value.txt");
    fs::write(&work, "user edits").expect("edit checkout");
    let other_repository = RepositoryDefinition {
        url: "https://example.com/different.git".into(),
        ..definition.clone()
    };
    assert!(prepare_checkout(sandbox.path(), &other_repository, &[])
        .await
        .is_err());
    let other_revision = RepositoryDefinition {
        revision: "another-branch".into(),
        ..definition.clone()
    };
    assert!(prepare_checkout(sandbox.path(), &other_revision, &[])
        .await
        .is_err());
    assert_eq!(
        fs::read_to_string(&work).expect("preserved edits"),
        "user edits"
    );
    prepare_checkout(sandbox.path(), &definition, &[])
        .await
        .expect("same selection remains reusable");
}
