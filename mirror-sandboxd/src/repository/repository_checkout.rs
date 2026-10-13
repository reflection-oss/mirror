use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use prost::Message;
use rustix::fs::{renameat_with, RenameFlags, CWD};
use tokio::process::Command;

use super::repository_seed::RepositorySeed;
use super::{validate_relative, validate_repository, RepositoryDefinition};

pub async fn prepare_checkout(parent: &Path, requested: &RepositoryDefinition) -> io::Result<()> {
    validate_directory(parent, "")?;
    validate_repository(requested)?;
    let destination = parent.join("repository");
    if fs::symlink_metadata(&destination).is_ok() {
        validate_directory(&destination, ".git")?;
        validate_directory(&destination, &requested.directory)?;
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
        return Ok(());
    }
    let staging = tempfile::Builder::new()
        .prefix(".mirror-prepare-")
        .tempdir_in(parent)?;
    let checkout = staging.path().join("repository");
    fs::create_dir(&checkout)?;
    git(&checkout, &["init", "--quiet"]).await?;
    git(&checkout, &["remote", "add", "origin", &requested.url]).await?;
    fetch(parent, &checkout, requested).await?;
    git(
        &checkout,
        &["checkout", "--quiet", "--detach", "FETCH_HEAD"],
    )
    .await?;
    validate_directory(&checkout, &requested.directory)?;
    fs::write(
        checkout.join(".git/mirror-repository"),
        requested.encode_to_vec(),
    )?;
    publish_checkout(&checkout, &destination)?;
    Ok(())
}
async fn fetch(parent: &Path, checkout: &Path, requested: &RepositoryDefinition) -> io::Result<()> {
    let tip = match RepositorySeed::find(parent, &requested.url).await {
        Some(seed) => Some(format!(
            "--negotiation-tip={}",
            seed.share_objects(checkout)?
        )),
        None => None,
    };
    let mut arguments = vec!["fetch", "--quiet", "--no-tags", "--depth=1"];
    arguments.extend(tip.as_deref());
    arguments.extend(["origin", requested.revision.as_str()]);
    git(checkout, &arguments).await?;
    Ok(())
}

pub(super) async fn git(directory: &Path, args: &[&str]) -> io::Result<String> {
    let mut command = Command::new("git");
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("GIT_") {
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
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_mins(2), command.output())
        .await
        .map_err(|_| invalid("repository Git operation timed out"))?
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
    renameat_with(CWD, checkout, CWD, destination, RenameFlags::NOREPLACE).map_err(io::Error::from)
}
