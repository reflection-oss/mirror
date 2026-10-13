//! The local worker's workspace: the tools' root on this machine's disk.

use std::io;
use std::path::Path;
use std::path::PathBuf;

use futures_util::future::BoxFuture;

use crate::PathState;
use crate::RepositoryIdentity;
use crate::Workspace;

/// The checkout under the root the local worker's tools are confined to.
#[derive(Debug, Clone)]
pub struct LocalWorkspace {
    root: PathBuf,
    identity: Option<RepositoryIdentity>,
}

impl LocalWorkspace {
    /// Discover the repository around `root`, the same root the tools use; like
    /// them, an unresolvable root is an error. Git runs once here, at assembly.
    pub fn new(root: &Path) -> io::Result<Self> {
        let root = std::fs::canonicalize(root)?;
        let identity = RepositoryIdentity::discover(&root);
        Ok(Self { root, identity })
    }

    /// The directory checkout-relative paths resolve against: the repository
    /// toplevel, or the root itself outside any repository.
    fn checkout(&self) -> &Path {
        self.identity
            .as_ref()
            .map_or(&self.root, RepositoryIdentity::toplevel)
    }
}

impl Workspace for LocalWorkspace {
    fn repository(&self) -> Option<&str> {
        self.identity.as_ref().map(RepositoryIdentity::repository)
    }

    fn relative(&self, path: &str) -> Option<String> {
        self.identity
            .as_ref()?
            .relative(Path::new(path), &self.root)
    }

    fn inspect<'a>(&'a self, paths: &'a [String]) -> BoxFuture<'a, io::Result<Vec<PathState>>> {
        Box::pin(async move {
            let mut states = Vec::with_capacity(paths.len());
            for path in paths {
                states.push(inspect_path(&self.checkout().join(path)).await?);
            }
            Ok(states)
        })
    }
}

async fn inspect_path(path: &Path) -> io::Result<PathState> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) if metadata.is_dir() => Ok(PathState::Directory),
        Ok(_) => Ok(PathState::File),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(PathState::Missing),
        Err(_) => Ok(PathState::Unreadable),
    }
}

#[cfg(test)]
#[path = "local_workspace_test.rs"]
mod tests;
