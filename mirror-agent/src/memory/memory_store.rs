//! The store contract memory depends on; each runtime supplies one
//! implementation bound to one owner.

use futures_util::future::BoxFuture;

use super::MemoryCommit;
use super::MemoryQuery;
use super::MemoryRecord;
use super::MemoryResult;

/// One owner's memory records. Locally a SQLite file; hosted, rows keyed by
/// the owning principal.
///
/// `commit` checks each change against the version of the record it was made
/// against, applies all of them or none, and writes the events itself.
pub trait MemoryStore: Send + Sync {
    /// Records matching `query`: ranked best first when it has `text`, newest
    /// first otherwise. Ranking and the relevance floor belong to the backend.
    fn search<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, MemoryResult<Vec<MemoryRecord>>>;

    /// One record by id, whatever its status.
    fn get<'a>(&'a self, id: &'a str) -> BoxFuture<'a, MemoryResult<Option<MemoryRecord>>>;

    fn commit<'a>(&'a self, commit: MemoryCommit) -> BoxFuture<'a, MemoryResult<()>>;
}
