//! Fixtures the memory tests share: one checkout and a fresh store.

use std::io;
use std::path::Path;
use std::sync::Arc;

use futures_util::future::BoxFuture;

use super::MemoryChange;
use super::MemoryCommit;
use super::MemoryRecord;
use super::MemoryStore;
use super::SqliteMemoryDatabase;
use super::SqliteMemoryStore;
use crate::PathState;
use crate::Workspace;

pub(crate) const REPO: &str = "github.com/example/project";

/// A checkout of `REPO`. Paths become repository-relative under `prefix`
/// (a path that escapes the checkout has no relative form), every `.rs`
/// path is a file, and nothing else exists.
pub(crate) struct TestWorkspace {
    prefix: &'static str,
}

impl TestWorkspace {
    /// The tools' root is the repository root.
    pub(crate) const ROOT: Self = Self { prefix: "" };

    /// The tools' root is `prefix` inside the repository.
    pub(crate) const fn under(prefix: &'static str) -> Self {
        Self { prefix }
    }
}

impl Workspace for TestWorkspace {
    fn repository(&self) -> Option<&str> {
        Some(REPO)
    }

    fn relative(&self, path: &str) -> Option<String> {
        if path.starts_with("..") {
            return None;
        }
        if self.prefix.is_empty() {
            Some(path.to_string())
        } else {
            Some(format!("{}/{path}", self.prefix))
        }
    }

    fn inspect<'a>(&'a self, paths: &'a [String]) -> BoxFuture<'a, io::Result<Vec<PathState>>> {
        Box::pin(async move {
            Ok(paths
                .iter()
                .map(|path| {
                    if path.ends_with(".rs") {
                        PathState::File
                    } else {
                        PathState::Missing
                    }
                })
                .collect())
        })
    }
}

/// An empty store in `directory`.
pub(crate) fn open_store(directory: &Path) -> Arc<dyn MemoryStore> {
    let database = SqliteMemoryDatabase::open(directory).expect("database");
    Arc::new(SqliteMemoryStore::new(database).expect("store"))
}

/// A store in `directory` holding `records`, each committed by its own actor.
pub(crate) async fn seeded_store(
    directory: &Path,
    records: Vec<MemoryRecord>,
) -> Arc<dyn MemoryStore> {
    let store = open_store(directory);
    for record in records {
        store
            .commit(MemoryCommit {
                actor: record.actor,
                session_id: None,
                changes: vec![MemoryChange::Insert(record)],
            })
            .await
            .expect("seed");
    }
    store
}
