//! Host Git/Jujutsu checkout discovery and provisioning; guest-writable clones are excluded.

use std::path::{Path, PathBuf};
use std::time::Duration;

use mirror_protocol::daemon::AgentConfiguration;
use sha2::{Digest, Sha256};
use tonic::transport::Channel;
use tonic::Status;
use uuid::Uuid;

use crate::{AgentOptions, AgentProcess, AgentStartError};

use super::{
    host_directory, run_workspace_command, text, workspace_command, Checkout, Process, Repository,
    Workspace,
};

/// A host directory; worktree provisioning is independent of agent execution.
#[derive(Debug)]
pub struct Local {
    pub options: AgentOptions,
    pub directory: PathBuf,
}

impl Local {
    pub fn new(options: AgentOptions, directory: PathBuf) -> Self {
        Self { options, directory }
    }
}

#[tonic::async_trait]
impl Workspace for Local {
    async fn start_agent(
        &self,
        session_id: &str,
        configuration: &AgentConfiguration,
        managed_reference: &str,
    ) -> Result<(Option<AgentProcess>, Channel), AgentStartError> {
        let command = workspace_command(&self.options, &self.options.executable, &self.directory);
        AgentProcess::start(
            &self.options,
            session_id,
            command,
            configuration,
            None,
            managed_reference,
        )
        .await
        .map(|(process, channel)| (Some(process), channel))
    }

    async fn exec(&self, argv: &[String]) -> Result<Process, Status> {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| Status::invalid_argument("process requires an executable"))?;
        let mut command = workspace_command(&self.options, program, &self.directory);
        command.args(args);
        Process::spawn(command)
    }

    async fn stop(&self) -> Result<(), Status> {
        Ok(())
    }
}

/// Identify a linked checkout without spawning a VCS process or scanning other workspaces.
pub async fn is_worktree_directory(directory: &Path) -> Result<bool, Status> {
    if tokio::fs::metadata(directory.join(".jj/repo"))
        .await
        .is_ok_and(|metadata| metadata.is_file())
    {
        return Ok(true);
    }
    let marker = directory.join(".git");
    if !tokio::fs::metadata(&marker)
        .await
        .is_ok_and(|metadata| metadata.is_file())
    {
        return Ok(false);
    }
    let link = tokio::fs::read_to_string(marker)
        .await
        .map_err(|_| Status::failed_precondition("could not read Git directory link"))?;
    let Some(target) = link.trim().strip_prefix("gitdir: ") else {
        return Err(Status::failed_precondition("invalid Git directory link"));
    };
    tokio::fs::try_exists(directory.join(target).join("commondir"))
        .await
        .map_err(|_| Status::failed_precondition("Git directory is unavailable"))
}

/// Return the same repository and provider identities used by existing workspace records.
pub async fn describe_repository(
    options: &AgentOptions,
    directory: &str,
) -> Result<Repository, Status> {
    let directory = host_directory(options, directory).await?;
    let root = repository_root(&directory).await?.unwrap_or(directory);
    let (key, mut checkouts) = if tokio::fs::try_exists(root.join(".jj"))
        .await
        .unwrap_or(false)
    {
        let key = jj_repository(&root).await?;
        (text(&key)?.to_owned(), list_jj(options, &root).await?)
    } else if tokio::fs::try_exists(root.join(".git"))
        .await
        .unwrap_or(false)
    {
        let key = git(
            options,
            &root,
            &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        )
        .await?;
        let key = tokio::fs::canonicalize(key.trim())
            .await
            .map_err(|_| Status::failed_precondition("Git repository is unavailable"))?;
        (
            text(&key)?.to_owned(),
            list_git(options, &root, &key).await?,
        )
    } else {
        (
            text(&root)?.to_owned(),
            vec![checkout(options, root.clone(), "local", None)?],
        )
    };
    let current = checkouts
        .iter()
        .find(|checkout| Path::new(&checkout.directory) == root)
        .cloned()
        .ok_or_else(|| Status::failed_precondition("could not identify the current checkout"))?;
    order(&mut checkouts).await;
    Ok(Repository {
        key,
        current: Some(current),
        checkouts,
    })
}

/// Create a managed checkout from a pinned commit, never from another client's dirty files.
pub async fn create_worktree(
    options: &AgentOptions,
    directory: &str,
    name: Option<&str>,
) -> Result<Checkout, Status> {
    let repository = describe_repository(options, directory).await?;
    let current = repository
        .current
        .ok_or_else(|| Status::failed_precondition("repository has no checkout"))?;
    let source = Path::new(&current.directory);
    let name = checkout_name(name)?;
    match current.provider.as_str() {
        "git" => create_git(options, source, Path::new(&repository.key), &name).await,
        "jj" => create_jj(options, source, Path::new(&repository.key), &name).await,
        _ => Err(Status::failed_precondition(
            "creating a checkout requires a Git or Jujutsu repository",
        )),
    }
}

/// Choose a managed fork destination without creating a checkout.
pub async fn fork_destination(
    options: &AgentOptions,
    directory: &str,
    name: &str,
) -> Result<String, Status> {
    let repository = describe_repository(options, directory).await?;
    let current = repository
        .current
        .ok_or_else(|| Status::failed_precondition("repository has no checkout"))?;
    if !matches!(current.provider.as_str(), "git" | "jj") {
        return Err(Status::failed_precondition(
            "forking requires a Git or Jujutsu repository",
        ));
    }
    let name = checkout_name(Some(name))?;
    let path = destination(
        options,
        Path::new(&repository.key),
        Path::new(&current.directory),
        &name,
    )
    .await?;
    Ok(text(&path)?.to_owned())
}

/// Unregister and delete a checkout without removing its repository's primary workspace.
pub async fn drop_worktree(options: &AgentOptions, expected: &Checkout) -> Result<(), Status> {
    let repository = describe_repository(options, &expected.directory).await?;
    let current = repository
        .current
        .ok_or_else(|| Status::failed_precondition("checkout no longer exists"))?;
    if current.id != expected.id
        || current.provider != expected.provider
        || current.directory != expected.directory
    {
        return Err(Status::failed_precondition("checkout identity changed"));
    }
    if !current.managed {
        return Err(Status::failed_precondition(
            "checkout is not managed by Mirror",
        ));
    }
    let root = Path::new(&current.directory);
    if current.provider == "jj" {
        let name = current
            .name
            .as_deref()
            .filter(|name| *name != "default")
            .ok_or_else(|| {
                Status::failed_precondition("the primary Jujutsu workspace cannot be dropped")
            })?;
        jj(
            options,
            root,
            &["workspace", "forget", "--ignore-working-copy", "--", name],
        )
        .await?;
        tokio::fs::remove_dir_all(root)
            .await
            .map_err(|_| Status::failed_precondition("could not remove forgotten workspace"))?;
    } else if current.provider == "git" {
        if repository
            .checkouts
            .first()
            .is_some_and(|checkout| checkout.id == current.id)
        {
            return Err(Status::failed_precondition(
                "the primary Git worktree cannot be dropped",
            ));
        }
        git_worktree(
            options,
            root,
            Path::new(&repository.key),
            &["worktree", "remove", "--force", "--", &current.directory],
            false,
        )
        .await?;
    } else {
        return Err(Status::failed_precondition(
            "directory is not a VCS checkout",
        ));
    }
    Ok(())
}

/// Resolve a committed local revision for an independent sandbox clone.
pub async fn local_revision(
    options: &AgentOptions,
    source: &str,
) -> Result<(PathBuf, String), Status> {
    let source = host_directory(options, source).await?;
    let root = repository_root(&source).await?.ok_or_else(|| {
        Status::failed_precondition("no repository contains the source directory")
    })?;
    let (store, revision) = if tokio::fs::try_exists(root.join(".jj"))
        .await
        .unwrap_or(false)
    {
        (
            jj(options, &root, &["git", "root", "--ignore-working-copy"]).await?,
            jj_revision(options, &root, "@-").await?,
        )
    } else {
        (
            git(
                options,
                &root,
                &["rev-parse", "--path-format=absolute", "--git-common-dir"],
            )
            .await?,
            git(options, &root, &["rev-parse", "--verify", "HEAD^{commit}"])
                .await?
                .trim()
                .to_owned(),
        )
    };
    validate_revision(&revision)?;
    Ok((PathBuf::from(store.trim()), revision))
}

/// Run Git with bounded output and without inheriting daemon authentication variables.
pub async fn git(options: &AgentOptions, root: &Path, args: &[&str]) -> Result<String, Status> {
    let mut command = workspace_command(options, "git", root);
    command.args(args);
    run_workspace_command(command, &[], Duration::from_secs(120)).await
}

async fn git_worktree(
    options: &AgentOptions,
    root: &Path,
    common: &Path,
    args: &[&str],
    exclusive: bool,
) -> Result<String, Status> {
    let mut command = workspace_command(options, "git", root);
    command.args(args);
    let output = mirror_agent::run_worktree_command(common, command, exclusive)
        .await
        .map_err(|_| {
            Status::failed_precondition("could not complete locked Git worktree command")
        })?;
    if !output.status.success() {
        return Err(Status::failed_precondition("Git worktree command failed"));
    }
    String::from_utf8(output.stdout)
        .map_err(|_| Status::data_loss("Git worktree output is not UTF-8"))
}

async fn jj(options: &AgentOptions, root: &Path, args: &[&str]) -> Result<String, Status> {
    let mut command = workspace_command(options, "jj", root);
    command.args(args);
    run_workspace_command(command, &[], Duration::from_secs(120)).await
}

async fn repository_root(directory: &Path) -> Result<Option<PathBuf>, Status> {
    for root in directory.ancestors() {
        for marker in [".git", ".jj"] {
            if tokio::fs::try_exists(root.join(marker))
                .await
                .map_err(|_| Status::failed_precondition("could not discover repository root"))?
            {
                return Ok(Some(root.to_owned()));
            }
        }
    }
    Ok(None)
}

async fn jj_repository(root: &Path) -> Result<PathBuf, Status> {
    let marker = root.join(".jj/repo");
    let target = if tokio::fs::metadata(&marker)
        .await
        .is_ok_and(|value| value.is_dir())
    {
        marker
    } else {
        let link = tokio::fs::read_to_string(&marker)
            .await
            .map_err(|_| Status::failed_precondition("could not read Jujutsu repository link"))?;
        root.join(".jj").join(link.trim())
    };
    tokio::fs::canonicalize(target)
        .await
        .map_err(|_| Status::failed_precondition("Jujutsu repository is unavailable"))
}

fn checkout(
    options: &AgentOptions,
    root: PathBuf,
    provider: &str,
    name: Option<String>,
) -> Result<Checkout, Status> {
    let directory = text(&root)?.to_owned();
    let id = if provider == "jj" {
        name.clone()
            .ok_or_else(|| Status::data_loss("Jujutsu workspace has no identity"))?
    } else {
        directory.clone()
    };
    Ok(Checkout {
        directory,
        id,
        provider: provider.to_owned(),
        name,
        managed: root.starts_with(options.home.join(".mirror/worktrees")),
    })
}

async fn list_git(
    options: &AgentOptions,
    root: &Path,
    common: &Path,
) -> Result<Vec<Checkout>, Status> {
    let listing = git_worktree(
        options,
        root,
        common,
        &["worktree", "list", "--porcelain", "-z"],
        false,
    )
    .await?;
    let mut checkouts = Vec::new();
    for record in listing.split("\0\0") {
        let mut directory = None;
        let mut name = None;
        let mut prunable = false;
        for field in record.split('\0') {
            if let Some(value) = field.strip_prefix("worktree ") {
                directory = Some(value);
            }
            if let Some(value) = field.strip_prefix("branch refs/heads/") {
                name = Some(value.to_owned());
            }
            prunable |= field.starts_with("prunable");
        }
        if let Some(directory) = directory.filter(|_| !prunable) {
            if let Ok(directory) = tokio::fs::canonicalize(directory).await {
                checkouts.push(checkout(options, directory, "git", name)?);
            }
        }
    }
    Ok(checkouts)
}

async fn list_jj(options: &AgentOptions, root: &Path) -> Result<Vec<Checkout>, Status> {
    let listing = jj(
        options,
        root,
        &[
            "workspace",
            "list",
            "--ignore-working-copy",
            "--template",
            "name ++ \"\\0\" ++ root ++ \"\\0\"",
        ],
    )
    .await?;
    let fields = listing
        .strip_suffix('\0')
        .unwrap_or(&listing)
        .split('\0')
        .collect::<Vec<_>>();
    if fields.len() % 2 != 0 {
        return Err(Status::data_loss("invalid Jujutsu workspace listing"));
    }
    let mut checkouts = Vec::new();
    for pair in fields.chunks_exact(2) {
        let path = Path::new(pair[1]);
        if !path.is_absolute() {
            continue;
        }
        if let Ok(directory) = tokio::fs::canonicalize(path).await {
            checkouts.push(checkout(
                options,
                directory,
                "jj",
                Some(pair[0].to_owned()),
            )?);
        }
    }
    checkouts.sort_by_key(|checkout| checkout.name.as_deref() != Some("default"));
    Ok(checkouts)
}

async fn order(checkouts: &mut Vec<Checkout>) {
    let mut dated = Vec::new();
    for (index, checkout) in checkouts.drain(..).enumerate() {
        let root = Path::new(&checkout.directory);
        let marker = if checkout.provider == "jj" {
            root.join(".jj/repo")
        } else {
            root.join(".git")
        };
        let date = tokio::fs::symlink_metadata(marker)
            .await
            .ok()
            .and_then(|value| value.modified().ok());
        dated.push((index != 0, date, checkout.directory.clone(), checkout));
    }
    dated.sort_by(|left, right| (&left.0, &left.1, &left.2).cmp(&(&right.0, &right.1, &right.2)));
    *checkouts = dated
        .into_iter()
        .map(|(_, _, _, checkout)| checkout)
        .collect();
}

fn checkout_name(name: Option<&str>) -> Result<String, Status> {
    let name = name.unwrap_or("").trim();
    if name.is_empty() {
        return Ok(Uuid::new_v4().simple().to_string()[..8].to_owned());
    }
    if name.len() > 64
        || !name.as_bytes()[0].is_ascii_alphanumeric()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        || name.contains("..")
        || name.ends_with('.')
        || name.ends_with(".lock")
    {
        return Err(Status::invalid_argument("invalid worktree name"));
    }
    Ok(name.to_owned())
}

async fn destination(
    options: &AgentOptions,
    shared: &Path,
    label: &Path,
    name: &str,
) -> Result<PathBuf, Status> {
    let digest = format!("{:x}", Sha256::digest(text(shared)?.as_bytes()));
    let label = label
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repository");
    let parent = options
        .home
        .join(".mirror/worktrees")
        .join(format!("{label}-{}", &digest[..8]));
    tokio::fs::create_dir_all(&parent)
        .await
        .map_err(|_| Status::failed_precondition("could not create managed worktree directory"))?;
    Ok(parent.join(name))
}

async fn create_git(
    options: &AgentOptions,
    source: &Path,
    common: &Path,
    name: &str,
) -> Result<Checkout, Status> {
    let revision = git(options, source, &["rev-parse", "--verify", "HEAD^{commit}"]).await?;
    validate_revision(revision.trim())?;
    let path = destination(options, common, common.parent().unwrap_or(common), name).await?;
    let branch = format!("mirror/{name}");
    git_worktree(
        options,
        source,
        common,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            "--",
            text(&path)?,
            revision.trim(),
        ],
        true,
    )
    .await?;
    let path = tokio::fs::canonicalize(path)
        .await
        .map_err(|_| Status::failed_precondition("created worktree is unavailable"))?;
    checkout(options, path, "git", Some(branch))
}

async fn jj_revision(
    options: &AgentOptions,
    source: &Path,
    revset: &str,
) -> Result<String, Status> {
    Ok(jj(
        options,
        source,
        &[
            "log",
            "--ignore-working-copy",
            "--no-graph",
            "-r",
            revset,
            "-T",
            "commit_id",
        ],
    )
    .await?
    .trim()
    .to_owned())
}

async fn create_jj(
    options: &AgentOptions,
    source: &Path,
    repository: &Path,
    name: &str,
) -> Result<Checkout, Status> {
    let revision = jj_revision(options, source, "@-").await?;
    validate_revision(&revision)?;
    let label = repository
        .parent()
        .and_then(Path::parent)
        .unwrap_or(repository);
    let path = destination(options, repository, label, name).await?;
    jj(
        options,
        source,
        &[
            "workspace",
            "add",
            "--name",
            name,
            "--revision",
            &revision,
            "--",
            text(&path)?,
        ],
    )
    .await?;
    let path = tokio::fs::canonicalize(path)
        .await
        .map_err(|_| Status::failed_precondition("created workspace is unavailable"))?;
    checkout(options, path, "jj", Some(name.to_owned()))
}

fn validate_revision(revision: &str) -> Result<(), Status> {
    if !matches!(revision.len(), 40 | 64) || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(Status::failed_precondition(
            "repository revision must resolve to one commit",
        ));
    }
    Ok(())
}
