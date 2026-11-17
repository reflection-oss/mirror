//! Canonical host directories, excluding guest-writable sandbox clones.

use std::path::{Path, PathBuf};

use mirror_protocol::daemon::ResolveDirectoryResponse;
use tonic::Status;

use crate::AgentOptions;

/// Validate a host directory before any repository-configured program can run.
pub async fn host_directory(options: &AgentOptions, directory: &str) -> Result<PathBuf, Status> {
    if !Path::new(directory).is_absolute() {
        return Err(Status::invalid_argument(
            "workspace directory must be absolute",
        ));
    }
    let directory = tokio::fs::canonicalize(directory)
        .await
        .map_err(|_| Status::invalid_argument("workspace directory is unavailable"))?;
    let sandbox_root = options.home.join(".mirror/sandboxes");
    let sandbox_root = tokio::fs::canonicalize(&sandbox_root)
        .await
        .unwrap_or(sandbox_root);
    if directory.starts_with(sandbox_root) {
        return Err(Status::failed_precondition(
            "host VCS operations cannot run in a sandbox clone",
        ));
    }
    if !tokio::fs::metadata(&directory)
        .await
        .is_ok_and(|value| value.is_dir())
    {
        return Err(Status::invalid_argument(
            "workspace directory is unavailable",
        ));
    }
    Ok(directory)
}

/// Resolve paths on the owning host without creating files or running repository code.
pub async fn resolve_directory(
    options: &AgentOptions,
    entered: &str,
    base: &str,
) -> Result<ResolveDirectoryResponse, Status> {
    let path = if entered == "~" {
        options.home.clone()
    } else if let Some(relative) = entered.strip_prefix("~/") {
        options.home.join(relative)
    } else if Path::new(entered).is_absolute() {
        PathBuf::from(entered)
    } else {
        host_directory(options, base).await?.join(entered)
    };
    match tokio::fs::metadata(&path).await {
        Ok(_) => Ok(ResolveDirectoryResponse {
            directory: text(&host_directory(options, text(&path)?).await?)?.to_owned(),
            exists: true,
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .ok_or_else(|| Status::invalid_argument("directory needs a parent"))?;
            let name = path
                .file_name()
                .ok_or_else(|| Status::invalid_argument("directory needs a name"))?;
            let parent = host_directory(options, text(parent)?).await?;
            Ok(ResolveDirectoryResponse {
                directory: text(&parent.join(name))?.to_owned(),
                exists: false,
            })
        }
        Err(_) => Err(Status::failed_precondition("could not inspect directory")),
    }
}

/// Reject paths the string-based RPC cannot represent without changing identity.
pub fn text(path: &Path) -> Result<&str, Status> {
    path.to_str()
        .ok_or_else(|| Status::invalid_argument("workspace path is not UTF-8"))
}
