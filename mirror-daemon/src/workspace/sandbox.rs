//! Private sandbox clones and native VM preparation, independent of either UI package.

use std::path::{Path, PathBuf};
use std::time::Duration;

use mirror_protocol::daemon::AgentConfiguration;
use mirror_protocol::WorkerSandbox;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tonic::transport::Channel;
use tonic::Status;

use crate::{AgentOptions, AgentProcess, AgentStartError};

use super::{
    git, local_revision, prepare_runtime, run_workspace_command, sandbox_command,
    sandbox_configuration_json, text, workspace_command, Local, Process, Workspace,
};

/// A private clone whose tools execute only in the explicitly configured VM.
#[derive(Debug)]
pub struct Sandbox {
    local: Local,
    configuration: WorkerSandbox,
}

impl Sandbox {
    pub fn new(local: Local, configuration: WorkerSandbox) -> Self {
        Self {
            local,
            configuration,
        }
    }
}

#[tonic::async_trait]
impl Workspace for Sandbox {
    async fn start_agent(
        &self,
        session_id: &str,
        configuration: &AgentConfiguration,
    ) -> Result<(Option<AgentProcess>, Channel), AgentStartError> {
        let mut command = sandbox_command(&self.local.options);
        command.current_dir(&self.local.directory);
        AgentProcess::start(
            &self.local.options,
            session_id,
            command,
            configuration,
            Some(self.configuration.clone()),
        )
        .await
        .map(|(process, channel)| (Some(process), channel))
    }

    async fn exec(&self, argv: &[String]) -> Result<Process, Status> {
        Process::spawn_sandbox(
            &self.local.options,
            &self.local.directory,
            &self.configuration,
            argv,
        )
    }

    async fn stop(&self) -> Result<(), Status> {
        prepare_runtime(&self.local.options).await?;
        let mut command = sandbox_command(&self.local.options);
        command.args(["sandbox", "stop", &self.configuration.id]);
        run_workspace_command(command, &[], Duration::from_secs(30))
            .await
            .map(|_| ())
    }
}

/// Create only the clone and metadata; creating a workspace never starts its VM.
pub async fn create_sandbox_workspace(
    options: &AgentOptions,
    id: &str,
    source_directory: Option<&str>,
) -> Result<PathBuf, Status> {
    let name = format!("sandbox-{id}");
    let root = options.home.join(".mirror/sandboxes");
    tokio::fs::create_dir_all(&root)
        .await
        .map_err(|_| Status::failed_precondition("could not create sandbox directory"))?;
    let checked = root.clone();
    tokio::task::spawn_blocking(move || crate::private_directory(&checked))
        .await
        .map_err(|_| Status::internal("sandbox directory validation failed"))?
        .map_err(|_| {
            Status::failed_precondition("sandbox directory must be private and user-owned")
        })?;
    let directory = root.join(&name);
    tokio::fs::create_dir(&directory).await.map_err(|_| {
        Status::already_exists("sandbox directory already exists or cannot be created")
    })?;
    if let Some(source) = source_directory {
        if let Err(error) = clone_repository(options, source, &directory, &name).await {
            let _ = tokio::fs::remove_dir_all(&directory).await;
            return Err(error);
        }
    }
    Ok(directory)
}

/// Prepare an idle VM through the native agent; never prepare a VM with live sessions.
pub async fn prepare_sandbox_workspace(
    options: &AgentOptions,
    directory: &str,
    sandbox: &WorkerSandbox,
) -> Result<(), Status> {
    let directory = clone_directory(options, directory).await?;
    prepare_runtime(options).await?;
    let source = sandbox
        .source
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("sandbox source is required"))?;
    let archive = ripgrep_archive(options, &source.image).await?;
    let mut command = sandbox_command(options);
    command.args(["sandbox", "prepare", "--check-tools"]);
    if archive.is_some() {
        command.arg("--ripgrep-archive");
    }
    command
        .arg(sandbox_configuration_json(sandbox)?)
        .arg(&directory);
    let result = run_workspace_command(
        command,
        archive.as_deref().unwrap_or_default(),
        Duration::from_secs(900),
    )
    .await;
    if result.is_err() {
        let mut quarantine = sandbox_command(options);
        quarantine.args(["sandbox", "quarantine", &sandbox.id]);
        let _ = run_workspace_command(quarantine, &[], Duration::from_secs(90)).await;
    }
    result.map(|_| ())
}

/// Delete a stopped VM before removing its clone; failure retains the clone for recovery.
pub async fn delete_sandbox_workspace(
    options: &AgentOptions,
    directory: &str,
    sandbox: &WorkerSandbox,
) -> Result<(), Status> {
    let directory = clone_directory(options, directory).await?;
    prepare_runtime(options).await?;
    let mut command = sandbox_command(options);
    command.args(["sandbox", "delete", &sandbox.id]);
    run_workspace_command(command, &[], Duration::from_secs(90)).await?;
    tokio::fs::remove_dir_all(directory)
        .await
        .map_err(|_| Status::failed_precondition("could not remove sandbox clone"))
}

async fn clone_directory(options: &AgentOptions, directory: &str) -> Result<PathBuf, Status> {
    let root = options.home.join(".mirror/sandboxes");
    let directory = tokio::fs::canonicalize(directory)
        .await
        .map_err(|_| Status::failed_precondition("sandbox clone is unavailable"))?;
    if directory.parent() != Some(root.as_path()) {
        return Err(Status::failed_precondition(
            "sandbox must be a private managed clone",
        ));
    }
    Ok(directory)
}

async fn clone_repository(
    options: &AgentOptions,
    source: &str,
    directory: &Path,
    name: &str,
) -> Result<(), Status> {
    let (store, revision) = local_revision(options, source).await?;
    // Transport copying avoids hardlinks and alternates back into the host repository.
    clone_git(
        options,
        &options.home,
        &[
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            "--",
            text(&store)?,
            text(directory)?,
        ],
    )
    .await?;
    // Detached worktree commits need not be reachable from the store's refs.
    clone_git(
        options,
        directory,
        &[
            "fetch",
            "--quiet",
            "--no-tags",
            "--",
            text(&store)?,
            &revision,
        ],
    )
    .await?;
    clone_git(
        options,
        directory,
        &["switch", "--quiet", "--detach", &revision],
    )
    .await?;
    clone_git(
        options,
        directory,
        &["switch", "--quiet", "-c", &format!("mirror/{name}")],
    )
    .await?;
    let upstream = git(
        options,
        Path::new(source),
        &["config", "--get", "remote.origin.url"],
    )
    .await
    .ok();
    match upstream
        .as_deref()
        .and_then(|value| without_credentials(value.trim()))
    {
        Some(url) => {
            clone_git(options, directory, &["remote", "set-url", "origin", &url]).await?;
        }
        None => {
            clone_git(options, directory, &["remote", "remove", "origin"]).await?;
        }
    }
    for key in ["user.name", "user.email"] {
        if let Ok(value) = git(options, Path::new(source), &["config", "--get", key]).await {
            clone_git(options, directory, &["config", key, value.trim()]).await?;
        }
    }
    Ok(())
}

async fn clone_git(options: &AgentOptions, root: &Path, args: &[&str]) -> Result<String, Status> {
    let mut command = workspace_command(options, "git", root);
    command
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.ext.allow=never",
        ])
        .args(args);
    run_workspace_command(command, &[], Duration::from_secs(120)).await
}

fn without_credentials(value: &str) -> Option<String> {
    let mut url = if value.contains("://") {
        url::Url::parse(value).ok()?
    } else {
        let (host, path) = value.split_once(':')?;
        let host = host.rsplit('@').next()?;
        url::Url::parse(&format!("ssh://{host}/{path}")).ok()?
    };
    if !matches!(url.scheme(), "http" | "https" | "ssh" | "git") {
        return None;
    }
    url.set_username("").ok()?;
    url.set_password(None).ok()?;
    url.set_query(None);
    url.set_fragment(None);
    Some(url.into())
}

async fn ripgrep_archive(options: &AgentOptions, image: &str) -> Result<Option<Vec<u8>>, Status> {
    // Only the approved base images need seeding; custom images must supply their own tools.
    let (base, archive, checksum, size) = match std::env::consts::ARCH {
        "aarch64" => (
            "docker.io/library/python@sha256:715a063709d47191add11cccf48b4de0cdf220302070ca78d10c249f721f1da6",
            "ripgrep-14.1.1-aarch64-unknown-linux-gnu.tar.gz",
            "c827481c4ff4ea10c9dc7a4022c8de5db34a5737cb74484d62eb94a95841ab2f", 2_047_405),
        "x86_64" => (
            "docker.io/library/python@sha256:edd0b3ec946cc68bd20e39480bd03929016d0fb6254cbf870cd3bc55ea25116d",
            "ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz",
            "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e", 2_566_310),
        _ => return Err(Status::failed_precondition("unsupported sandbox architecture")),
    };
    if image != base {
        return Ok(None);
    }
    let file = tokio::fs::File::open(options.home.join(".mirror/sandbox-assets").join(archive))
        .await
        .map_err(|_| {
            Status::failed_precondition("approved guest ripgrep archive is not installed")
        })?;
    let mut bytes = Vec::new();
    file.take(size + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|_| Status::failed_precondition("could not read guest ripgrep archive"))?;
    if bytes.len() as u64 != size || format!("{:x}", Sha256::digest(&bytes)) != checksum {
        return Err(Status::failed_precondition(
            "guest ripgrep archive failed checksum verification",
        ));
    }
    Ok(Some(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn create_sandbox_copies_detached_worktree_commit_without_alternates() {
        let home = tempfile::tempdir().expect("private fixture home");
        let options = AgentOptions {
            executable: home.path().join("missing-worker"),
            bundle: home.path().join("bundle"),
            database: home.path().join("history.db"),
            home: home.path().to_owned(),
            runtime_directory: home.path().join("runtime"),
            state_directory: home.path().join("state"),
            path: std::env::var("PATH").expect("test tool path"),
        };
        let repository = home.path().join("repository");
        let detached = home.path().join("detached");
        std::fs::create_dir(&repository).unwrap();
        clone_git(&options, &repository, &["init", "--quiet"])
            .await
            .unwrap();
        clone_git(&options, &repository, &["config", "user.name", "Fixture"])
            .await
            .unwrap();
        clone_git(
            &options,
            &repository,
            &["config", "user.email", "fixture@example.invalid"],
        )
        .await
        .unwrap();
        clone_git(
            &options,
            &repository,
            &["commit", "--quiet", "--allow-empty", "-m", "base"],
        )
        .await
        .unwrap();
        clone_git(
            &options,
            &repository,
            &["worktree", "add", "--detach", text(&detached).unwrap()],
        )
        .await
        .unwrap();
        std::fs::write(detached.join("only-in-detached.txt"), "detached content").unwrap();
        clone_git(&options, &detached, &["add", "only-in-detached.txt"])
            .await
            .unwrap();
        clone_git(
            &options,
            &detached,
            &["commit", "--quiet", "-m", "detached"],
        )
        .await
        .unwrap();
        let revision = clone_git(&options, &detached, &["rev-parse", "HEAD"])
            .await
            .unwrap();

        crate::private_directory(&home.path().join(".mirror/sandboxes")).unwrap();
        let clone = create_sandbox_workspace(&options, "fixture", Some(text(&detached).unwrap()))
            .await
            .unwrap();

        assert_eq!(
            clone_git(&options, &clone, &["rev-parse", "HEAD"])
                .await
                .unwrap(),
            revision
        );
        assert_eq!(
            std::fs::read_to_string(clone.join("only-in-detached.txt")).unwrap(),
            "detached content"
        );
        assert!(!clone.join(".git/objects/info/alternates").exists());
    }

    #[tokio::test]
    async fn prepare_sandbox_requires_managed_clone() {
        let home = tempfile::tempdir().expect("private fixture home");
        let options = AgentOptions {
            executable: home.path().join("missing-worker"),
            bundle: home.path().join("bundle"),
            database: home.path().join("history.db"),
            home: home.path().to_owned(),
            runtime_directory: home.path().join("runtime"),
            state_directory: home.path().join("state"),
            path: String::new(),
        };
        let result = prepare_sandbox_workspace(
            &options,
            text(&home.path().join(".mirror/sandboxes/missing")).unwrap(),
            &WorkerSandbox::default(),
        )
        .await;

        assert_matches!(result, Err(status) => {
            assert_eq!(status.code(), tonic::Code::FailedPrecondition);
            assert_eq!(status.message(), "sandbox clone is unavailable");
        });
    }
}
