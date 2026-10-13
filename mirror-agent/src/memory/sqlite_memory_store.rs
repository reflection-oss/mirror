//! The one owner of a store's SQLite connection within a process. A dedicated
//! thread holds the database; async callers send it work and await the reply.

use futures_util::future::BoxFuture;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use super::MemoryCommit;
use super::MemoryError;
use super::MemoryQuery;
use super::MemoryRecord;
use super::MemoryResult;
use super::MemoryStore;
use super::SqliteMemoryDatabase;

/// A cloneable handle to the thread that owns one store's connection. Every
/// read and write in the process goes through it, one at a time.
#[derive(Debug, Clone)]
pub struct SqliteMemoryStore {
    sender: mpsc::Sender<Job>,
}

type Job = Box<dyn FnOnce(&mut SqliteMemoryDatabase) + Send>;

impl SqliteMemoryStore {
    /// Move `database` onto its own thread; it lives until the last handle drops.
    pub fn new(mut database: SqliteMemoryDatabase) -> MemoryResult<Self> {
        let (sender, mut receiver) = mpsc::channel::<Job>(32);
        std::thread::Builder::new()
            .name("mirror-memory".into())
            .spawn(move || {
                while let Some(job) = receiver.blocking_recv() {
                    job(&mut database);
                }
            })?;
        Ok(Self { sender })
    }

    /// Run `job` on the owning thread and return its result.
    pub async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut SqliteMemoryDatabase) -> MemoryResult<T> + Send + 'static,
    ) -> MemoryResult<T> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(Box::new(move |database: &mut SqliteMemoryDatabase| {
                let _ = reply.send(job(database));
            }))
            .await
            .map_err(unavailable)?;
        result.await.map_err(unavailable)?
    }
}

impl MemoryStore for SqliteMemoryStore {
    fn search<'a>(&'a self, query: MemoryQuery) -> BoxFuture<'a, MemoryResult<Vec<MemoryRecord>>> {
        Box::pin(self.run(move |database| database.search(&query)))
    }

    fn get<'a>(&'a self, id: &'a str) -> BoxFuture<'a, MemoryResult<Option<MemoryRecord>>> {
        let id = id.to_string();
        Box::pin(self.run(move |database| database.get(&id)))
    }

    fn commit<'a>(&'a self, commit: MemoryCommit) -> BoxFuture<'a, MemoryResult<()>> {
        Box::pin(self.run(move |database| database.commit(&commit)))
    }
}

fn unavailable(error: impl std::fmt::Display) -> MemoryError {
    MemoryError::Invalid(format!("memory storage worker unavailable: {error}"))
}

#[cfg(test)]
#[path = "sqlite_memory_store_test.rs"]
mod tests;
