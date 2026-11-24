use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use prost::Message;
use rustix::fs::{renameat_with, RenameFlags, CWD};
use rustix::io::Errno;
use tokio::process::Command;

use super::repository_seed::RepositorySeed;
use super::workspace_script::{read_default_branch, resume_after_wake, WorkspaceScript};
use super::{validate_relative, validate_repository, RepositoryDefinition, ScriptRun};

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CheckoutStages {
    pub reused: bool,
    pub seeded: bool,
    pub seed_lag_seconds: Option<i64>,
    /// The seed's working tree became this checkout's, so only changed files were written.
    pub tree_adopted: bool,
    pub fetch_ms: u64,
    pub checkout_ms: u64,
    pub publish_ms: u64,
    pub default_commit: Option<String>,
    pub setup: Option<ScriptRun>,
    pub resume: Option<ScriptRun>,
}

/// Creates the session checkout under `parent`, or reuses the one already there.
///
/// `disposable_paths` are workspace-relative paths the caller vouches hold nothing but what
/// the sandbox image recreates. A seed at one of them is consumed: its working tree becomes
/// the checkout's and the rest of it is deleted. Any other seed is only read.
pub async fn prepare_checkout(
    parent: &Path,
    requested: &RepositoryDefinition,
    disposable_paths: &[String],
) -> io::Result<CheckoutStages> {
    validate_directory(parent, "")?;
    validate_repository(requested)?;
    let destination = parent.join("repository");
    discard_unfinished_publication(&destination)?;
    discard_unfinished_setup(&destination)?;
    let mut stages = if fs::symlink_metadata(&destination).is_ok() {
        reuse_checkout(&destination, requested)?
    } else {
        create_checkout(parent, &destination, requested, disposable_paths).await?
    };
    stages.resume = resume_after_wake(&destination).await?;
    Ok(stages)
}

fn reuse_checkout(
    destination: &Path,
    requested: &RepositoryDefinition,
) -> io::Result<CheckoutStages> {
    validate_directory(destination, ".git")?;
    validate_directory(destination, &requested.directory)?;
    let expected = requested.encode_to_vec();
    let mut stored = Vec::new();
    fs::File::open(destination.join(".git/mirror-repository"))?
        .take(expected.len() as u64 + 1)
        .read_to_end(&mut stored)?;
    if stored != expected {
        return Err(invalid(
            "existing checkout belongs to a different repository request",
        ));
    }
    Ok(CheckoutStages {
        reused: true,
        ..CheckoutStages::default()
    })
}

async fn create_checkout(
    parent: &Path,
    destination: &Path,
    requested: &RepositoryDefinition,
    disposable_paths: &[String],
) -> io::Result<CheckoutStages> {
    let staging = tempfile::Builder::new()
        .prefix(".mirror-prepare-")
        .tempdir_in(parent)?;
    let checkout = staging.path().join("repository");
    fs::create_dir(&checkout)?;
    git(&checkout, &["init", "--quiet"]).await?;
    git(&checkout, &["config", "core.trustctime", "false"]).await?;
    git(&checkout, &["remote", "add", "origin", &requested.url]).await?;
    let seed = RepositorySeed::find(parent, &requested.url).await;
    let fetching = Instant::now();
    let default_branch_fetched = fetch(&checkout, requested, seed.as_ref()).await?;
    let fetch_ms = elapsed_ms(fetching);
    let seed_lag_seconds = match &seed {
        Some(seed) => seed_lag(&checkout, seed.commit()).await,
        None => None,
    };
    let default_branch = if default_branch_fetched {
        read_default_branch(&checkout).await
    } else {
        None
    };
    let mut stages = match &default_branch {
        Some(branch) if branch.has_setup => {
            set_up_then_check_out(&checkout, destination, requested, &branch.commit).await?
        }
        _ => {
            let disposable_seed = seed
                .as_ref()
                .filter(|seed| disposable_paths.iter().any(|path| path == seed.name()));
            check_out(
                staging.path(),
                &checkout,
                destination,
                requested,
                disposable_seed,
            )
            .await?
        }
    };
    stages.seeded = seed.is_some();
    stages.seed_lag_seconds = seed_lag_seconds;
    stages.fetch_ms = fetch_ms;
    stages.default_commit = default_branch.map(|branch| branch.commit);
    Ok(stages)
}

async fn check_out(
    staging: &Path,
    checkout: &Path,
    destination: &Path,
    requested: &RepositoryDefinition,
    disposable_seed: Option<&RepositorySeed>,
) -> io::Result<CheckoutStages> {
    let checking_out = Instant::now();
    let tree_adopted = match disposable_seed {
        Some(seed) => checkout_over_seed_tree(seed, staging, checkout).await?,
        None => {
            git(checkout, &CHECKOUT).await?;
            false
        }
    };
    let checkout_ms = elapsed_ms(checking_out);
    validate_directory(checkout, &requested.directory)?;
    fs::write(
        checkout.join(".git/mirror-repository"),
        requested.encode_to_vec(),
    )?;
    let publishing = Instant::now();
    publish_checkout(checkout, destination)?;
    Ok(CheckoutStages {
        tree_adopted,
        checkout_ms,
        publish_ms: elapsed_ms(publishing),
        ..CheckoutStages::default()
    })
}

async fn set_up_then_check_out(
    checkout: &Path,
    destination: &Path,
    requested: &RepositoryDefinition,
    default_commit: &str,
) -> io::Result<CheckoutStages> {
    let revision = git(checkout, &["rev-parse", "--verify", "FETCH_HEAD^{commit}"]).await?;
    let checking_out = Instant::now();
    git(
        checkout,
        &["checkout", "--quiet", "--detach", default_commit],
    )
    .await?;
    fs::write(checkout.join(SETUP_PENDING), b"")?;
    fs::write(
        checkout.join(".git/mirror-repository"),
        requested.encode_to_vec(),
    )?;
    let default_checkout_ms = elapsed_ms(checking_out);
    let publishing = Instant::now();
    publish_checkout(checkout, destination)?;
    let publish_ms = elapsed_ms(publishing);
    let setup = WorkspaceScript::Setup.run(destination).await?;
    let checking_out = Instant::now();
    git(
        destination,
        &["checkout", "--quiet", "--force", "--detach", &revision],
    )
    .await?;
    validate_directory(destination, &requested.directory)?;
    fs::remove_file(destination.join(SETUP_PENDING))?;
    Ok(CheckoutStages {
        checkout_ms: default_checkout_ms + elapsed_ms(checking_out),
        publish_ms,
        setup: Some(setup),
        ..CheckoutStages::default()
    })
}

const SETUP_PENDING: &str = ".git/mirror-setup-pending";

fn discard_unfinished_setup(destination: &Path) -> io::Result<()> {
    if fs::symlink_metadata(destination.join(SETUP_PENDING)).is_ok() {
        fs::remove_dir_all(destination)?;
    }
    Ok(())
}

const CHECKOUT: [&str; 4] = ["checkout", "--quiet", "--detach", "FETCH_HEAD"];
const FORCED_CHECKOUT: [&str; 5] = ["checkout", "--quiet", "--force", "--detach", "FETCH_HEAD"];

async fn checkout_over_seed_tree(
    seed: &RepositorySeed,
    staging: &Path,
    checkout: &Path,
) -> io::Result<bool> {
    let moved = match tracked_entries(checkout, seed.commit()).await {
        Ok(tracked) => seed
            .move_tree(&staging.join("seed"), checkout, &tracked)
            .is_ok(),
        Err(_) => false,
    };
    if moved && advance_tree(checkout, seed.commit()).await.is_ok() {
        return Ok(true);
    }
    let adopted = moved && git(checkout, &FORCED_CHECKOUT).await.is_ok();
    if !adopted {
        match fs::remove_file(checkout.join(".git/index")) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        git(checkout, &FORCED_CHECKOUT).await?;
    }
    git(checkout, &["clean", "--quiet", "-ffdx"]).await?;
    Ok(adopted)
}

async fn tracked_entries(checkout: &Path, commit: &str) -> io::Result<Vec<String>> {
    let listing = git(checkout, &["ls-tree", "--name-only", "-z", commit]).await?;
    Ok(listing
        .split('\0')
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect())
}

async fn advance_tree(checkout: &Path, from: &str) -> io::Result<()> {
    git(checkout, &["read-tree", "-m", "-u", from, "FETCH_HEAD"]).await?;
    git(
        checkout,
        &["update-ref", "--no-deref", "HEAD", "FETCH_HEAD"],
    )
    .await?;
    Ok(())
}

async fn fetch(
    checkout: &Path,
    requested: &RepositoryDefinition,
    seed: Option<&RepositorySeed>,
) -> io::Result<bool> {
    if let Some(seed) = seed {
        seed.share_objects(checkout)?;
    }
    let tip = seed.map(|seed| format!("--negotiation-tip={}", seed.commit()));
    let mut arguments = vec!["fetch", "--quiet", "--no-tags", "--depth=1"];
    arguments.extend(tip.as_deref());
    arguments.extend(["origin", requested.revision.as_str()]);
    let with_default_branch = [arguments.as_slice(), &["HEAD"]].concat();
    match git(checkout, &with_default_branch).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::TimedOut => Err(error),
        Err(error) => {
            let advertised = git(checkout, &["ls-remote", "origin", "HEAD"]).await?;
            if advertised.lines().any(|line| line.ends_with("\tHEAD")) {
                return Err(error);
            }
            git(checkout, &arguments).await?;
            Ok(false)
        }
    }
}

async fn seed_lag(checkout: &Path, seed: &str) -> Option<i64> {
    let fetched = commit_time(checkout, "FETCH_HEAD").await?;
    Some(fetched - commit_time(checkout, seed).await?)
}

async fn commit_time(checkout: &Path, revision: &str) -> Option<i64> {
    git(checkout, &["show", "--no-patch", "--format=%ct", revision])
        .await
        .ok()?
        .parse()
        .ok()
}

pub(crate) fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}

pub(super) const GIT_TIME_LIMIT: Duration = Duration::from_mins(2);

pub(super) fn git_command(directory: &Path, args: &[&str]) -> Command {
    let mut command = Command::new("git");
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("GIT_") && name != "GIT_SSL_CAINFO" {
            command.env_remove(name);
        }
    }
    command
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.file.allow=never",
            "-c",
            "protocol.ext.allow=never",
        ])
        .args(args)
        .current_dir(directory)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    command
}

pub(super) async fn git(directory: &Path, args: &[&str]) -> io::Result<String> {
    let mut command = git_command(directory, args);
    command.stdout(Stdio::piped());
    let output = tokio::time::timeout(GIT_TIME_LIMIT, command.output())
        .await
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::TimedOut,
                "repository Git operation timed out",
            )
        })?
        .map_err(|_| invalid("could not execute Git in the sandbox"))?;
    if !output.status.success() || output.stdout.len() > 4096 {
        return Err(invalid(
            "repository Git operation failed; check the revision; private fetch requires scoped credential/egress execution",
        ));
    }
    String::from_utf8(output.stdout)
        .map(|s| s.trim().to_string())
        .map_err(invalid)
}

pub(super) fn validate_directory(root: &Path, path: &str) -> io::Result<()> {
    validate_relative(path)?;
    if fs::symlink_metadata(root)?.is_symlink() {
        return Err(invalid("workspace root cannot be a symlink"));
    }
    let directory = fs::canonicalize(root)?.join(path);
    if !fs::symlink_metadata(&directory)?.is_dir() || fs::canonicalize(&directory)? != directory {
        return Err(invalid(
            "working directory must be a real directory in the repository",
        ));
    }
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

pub(super) fn publish_checkout(checkout: &Path, destination: &Path) -> io::Result<()> {
    match renameat_with(CWD, checkout, CWD, destination, RenameFlags::NOREPLACE) {
        Ok(()) => Ok(()),
        Err(Errno::INVAL | Errno::NOSYS | Errno::OPNOTSUPP) => {
            rename_into_absent(checkout, destination)
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn rename_into_absent(checkout: &Path, destination: &Path) -> io::Result<()> {
    if fs::symlink_metadata(destination).is_ok() {
        return Err(io::ErrorKind::AlreadyExists.into());
    }
    fs::rename(checkout, destination)
}

fn unfinished_publication(destination: &Path) -> PathBuf {
    destination.with_file_name(".mirror-publishing")
}

fn discard_unfinished_publication(destination: &Path) -> io::Result<()> {
    let unfinished = unfinished_publication(destination);
    if fs::symlink_metadata(&unfinished).is_err() {
        return Ok(());
    }
    let partial = fs::symlink_metadata(destination).is_ok_and(|metadata| metadata.is_dir())
        && fs::symlink_metadata(destination.join(".git/mirror-repository")).is_err();
    if partial {
        fs::remove_dir_all(destination)?;
    }
    fs::remove_file(unfinished)
}
