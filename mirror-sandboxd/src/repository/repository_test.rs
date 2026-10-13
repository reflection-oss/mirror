use std::ffi::OsString;
use std::fs;
use std::net::{Ipv4Addr, TcpListener};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};

use pretty_assertions::assert_eq;
use prost::Message;
use tokio::io::AsyncWriteExt;

use super::*;
use crate::REQUEST_BYTES_VARIABLE;

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
                "#!/bin/sh\nexec '{}' -c url.git://127.0.0.1:{port}/.insteadOf=https://github.com/ \"$@\"\n",
                git.display()
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
    let binary = fs::canonicalize(std::env::var("SANDBOXD_BIN").expect("guest binary"))
        .expect("resolve guest binary");
    let request = request.encode_to_vec();
    let mut worker = tokio::process::Command::new(&binary)
        .arg("prepare-repository")
        .args(arguments)
        .current_dir(sandbox)
        .env("PATH", github.search_path())
        .env(REQUEST_BYTES_VARIABLE, request.len().to_string())
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
    let error = prepare_checkout(sandbox.path(), &request)
        .await
        .expect_err("credential URL")
        .to_string();
    assert!(!error.contains("TOP_SECRET"));
    write(sandbox.path(), "repository/user-work", "untouched");
    let request = RepositoryDefinition {
        url: "https://example.com/repo.git".to_string(),
        ..request
    };
    assert!(prepare_checkout(sandbox.path(), &request).await.is_err());
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
    } = serde_json::from_slice(&output.stdout).expect("prepared context");
    let environment = environment.expect("sandbox environment");
    assert_eq!(
        environment.working_directory,
        fs::canonicalize(sandbox.path())
            .expect("canonical sandbox")
            .join("repository")
    );
    assert!(environment.is_git_repo);
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
    } = serde_json::from_slice(&skills_only.stdout).expect("skills-only context");
    assert!(recovered.instructions.is_empty());
    assert_eq!(
        recovered_environment.map(|environment| environment.working_directory),
        Some(environment.working_directory)
    );
    assert_eq!(recovered.skills, prepared.skills);
    let full = prepare_in_guest(sandbox.path(), &github, &project(), &[]).await;
    assert!(!full.status.success());
}

#[test]
fn publication_never_replaces_a_concurrent_workspace() {
    use assert_matches::assert_matches;
    let root = tempfile::tempdir().expect("workspace");
    let staged = root.path().join("staged");
    let destination = root.path().join("repository");
    fs::create_dir(&staged).expect("staging");
    fs::create_dir(&destination).expect("concurrent directory");
    assert_matches!(
        super::repository_checkout::publish_checkout(&staged, &destination),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    );
    fs::remove_dir(&destination).expect("remove empty directory");
    symlink(&staged, &destination).expect("concurrent symlink");
    assert_matches!(
        super::repository_checkout::publish_checkout(&staged, &destination),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists
    );
    assert!(fs::symlink_metadata(&destination)
        .expect("preserved")
        .is_symlink());
    fs::remove_file(&destination).expect("remove symlink");
    super::repository_checkout::publish_checkout(&staged, &destination).expect("publish");
    assert!(destination.is_dir());
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
    assert!(prepare_checkout(sandbox.path(), &other_repository)
        .await
        .is_err());
    let other_revision = RepositoryDefinition {
        revision: "another-branch".into(),
        ..definition.clone()
    };
    assert!(prepare_checkout(sandbox.path(), &other_revision)
        .await
        .is_err());
    assert_eq!(
        fs::read_to_string(&work).expect("preserved edits"),
        "user edits"
    );
    prepare_checkout(sandbox.path(), &definition)
        .await
        .expect("same selection remains reusable");
}
