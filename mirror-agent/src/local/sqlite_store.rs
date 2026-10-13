//! SQLite storage for the imperative ReACT service.

use std::fs::File;
use std::fs::OpenOptions;
use std::fs::TryLockError;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use crate::Store;
use crate::StoreError;
use futures_util::future::BoxFuture;
use mirror_protocol::agent_message;
use mirror_protocol::chat_record;
use mirror_protocol::AgentMessage;
use prost::Message;
use ring::digest::{digest, SHA256};
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rusqlite::TransactionBehavior;
use uuid::Uuid;

use super::sqlite_migrations;
use crate::Checkpoints;

/// SQLite-backed ReACT conversation storage.
#[derive(Debug)]
pub struct SQLiteStore {
    connection: Connection,
    _lock: Option<SessionLock>,
    checkpoints: Checkpoints,
    /// Best-effort capture of repository state before persisting user messages.
    pub checkpoints_enabled: bool,
}

/// An exclusive lock on one conversation, released when dropped.
#[derive(Debug)]
struct SessionLock(File);

pub(super) const VIEW: &str =
    "FROM react_session s, json_each(s.lineage) AS l, json_each(s.fork_points) AS f \
     JOIN react_entry e ON e.session_id = l.value AND e.sequence <= f.value \
     WHERE s.id = ? AND l.key = f.key";

impl SQLiteStore {
    /// Open a session, optionally creating it if absent so startup can be retried.
    pub fn open(
        path: &Path,
        session_id: &str,
        workspace: &Path,
        create: bool,
        allow_workspace_mismatch: bool,
    ) -> Result<Self, StoreError> {
        validate_session_id(session_id)?;
        let workspace = std::fs::canonicalize(workspace).map_err(persistence)?;
        let checkpoints = Checkpoints::new(workspace.clone());
        let workspace = workspace.to_string_lossy().into_owned();
        let connection = connect(path)?;
        let lock = lock_session(path, session_id)?;
        if create {
            connection
                .execute(
                    "INSERT INTO react_session(id, workspace, lineage, fork_points) \
                     VALUES (?, ?, json_array(?), json_array(?)) \
                     ON CONFLICT(id) DO NOTHING",
                    params![session_id, workspace, session_id, i64::MAX],
                )
                .map_err(persistence)?;
        }
        let saved_workspace = connection
            .query_row(
                "SELECT workspace FROM react_session WHERE id = ?",
                [session_id],
                |row| row.get::<_, String>(0),
            )
            .map_err(|error| match error {
                rusqlite::Error::QueryReturnedNoRows => {
                    StoreError::Persistence(format!("conversation {session_id:?} does not exist"))
                }
                error => persistence(error),
            })?;
        if !allow_workspace_mismatch && saved_workspace != workspace {
            return Err(StoreError::Persistence(format!(
                "conversation {session_id:?} belongs to workspace {saved_workspace:?}, not {workspace:?}"
            )));
        }
        Ok(Self {
            connection,
            _lock: lock,
            checkpoints,
            checkpoints_enabled: false,
        })
    }

    pub fn delete_session(&mut self, session_id: &str) -> Result<(), StoreError> {
        validate_session_id(session_id)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(persistence)?;
        session_row(&transaction, session_id)?;
        let forked: bool = transaction
            .query_row(
                "SELECT EXISTS (\
                    SELECT 1 FROM react_session s, json_each(s.lineage) AS l \
                    WHERE s.id <> ? AND l.value = ?\
                )",
                params![session_id, session_id],
                |row| row.get(0),
            )
            .map_err(persistence)?;
        if forked {
            return Err(StoreError::SessionHasForks {
                session_id: session_id.to_string(),
            });
        }
        transaction
            .execute("DELETE FROM react_entry WHERE session_id = ?", [session_id])
            .map_err(persistence)?;
        transaction
            .execute("DELETE FROM react_session WHERE id = ?", [session_id])
            .map_err(persistence)?;
        transaction.commit().map_err(persistence)
    }

    fn load_context(
        &self,
        session_id: &str,
        window_only: bool,
        page: Option<(u64, u32)>,
    ) -> Result<(Vec<AgentMessage>, u64), StoreError> {
        validate_session_id(session_id)?;
        // Session deletion must not turn an existing history into an empty view
        // between the existence check and the lineage query.
        let transaction = self
            .connection
            .unchecked_transaction()
            .map_err(persistence)?;
        session_row(&transaction, session_id)?;
        let start: i64 = if window_only {
            transaction.query_row(
                "SELECT COALESCE(MAX((SELECT sequence FROM react_entry \
                 WHERE session_id = l.value AND sequence <= f.value AND system_prompt_hash IS NOT NULL \
                 ORDER BY sequence DESC LIMIT 1)), 0) \
                 FROM react_session s, json_each(s.lineage) l, json_each(s.fork_points) f \
                 WHERE s.id = ? AND l.key = f.key",
                [session_id], |row| row.get(0),
            ).map_err(persistence)?
        } else {
            0
        };
        let total = if page.is_some() {
            Some(
                transaction
                    .query_row(&format!("SELECT COUNT(*) {VIEW}"), [session_id], |row| {
                        row.get::<_, i64>(0).map(|count| count as u64)
                    })
                    .map_err(persistence)?,
            )
        } else {
            None
        };
        let (offset, limit) = page.map_or((0, -1), |(offset, limit)| {
            (i64::try_from(offset).unwrap_or(i64::MAX), i64::from(limit))
        });
        let mut statement = transaction
            .prepare(&format!(
                "SELECT e.sequence, e.payload, \
                 (SELECT content FROM react_system_prompt WHERE hash = e.system_prompt_hash) \
                 {VIEW} AND e.sequence >= ?2 ORDER BY e.sequence LIMIT ?3 OFFSET ?4"
            ))
            .map_err(persistence)?;
        let rows = statement
            .query_map(params![session_id, start, limit, offset], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })
            .map_err(persistence)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(persistence)?;
        drop(statement);
        transaction.commit().map_err(persistence)?;
        let total = total.unwrap_or(rows.len() as u64);
        let entries = rows
            .into_iter()
            .map(|(sequence, payload, content)| {
                let mut entry = decode_entry(sequence, &payload)?;
                if let (Some(agent_message::Payload::ContextWindow(window)), Some(content)) =
                    (&mut entry.payload, content)
                {
                    window.system_prompt = content;
                }
                Ok(entry)
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        Ok((entries, total))
    }

    fn append_context(
        &mut self,
        session_id: &str,
        head: u64,
        entries: &mut [AgentMessage],
    ) -> Result<(), StoreError> {
        if entries.is_empty() {
            return Ok(());
        }
        validate_session_id(session_id)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(persistence)?;
        session_row(&transaction, session_id)?;
        let head = i64::try_from(head).map_err(|_| {
            StoreError::Persistence("entry head exceeds SQLite INTEGER".to_string())
        })?;
        let current_head: i64 = transaction
            .query_row(
                "SELECT COALESCE(\
                    (SELECT MAX(sequence) FROM react_entry WHERE session_id = ?1), \
                    json_extract(fork_points, '$[#-2]'), 0) \
                 FROM react_session WHERE id = ?1",
                [session_id],
                |row| row.get(0),
            )
            .map_err(persistence)?;
        if current_head != head {
            return Err(StoreError::Stale);
        }
        let mut insert = transaction
            .prepare(
                "INSERT INTO react_entry(session_id, sequence, entry_id, payload, system_prompt_hash) \
                 VALUES (?, ?, ?, ?, ?)",
            )
            .map_err(persistence)?;
        for (offset, entry) in entries.iter_mut().enumerate() {
            if entry.id.is_empty() {
                entry.id = Uuid::new_v4().simple().to_string();
            }
            let duplicate: bool = transaction
                .query_row(
                    &format!("SELECT EXISTS (SELECT 1 {VIEW} AND e.entry_id = ?)"),
                    params![session_id, entry.id],
                    |row| row.get(0),
                )
                .map_err(persistence)?;
            if duplicate {
                return Err(StoreError::MessageIdConflict {
                    session_id: session_id.to_string(),
                    entry_id: entry.id.clone(),
                });
            }
            if entry.timestamp.is_none() {
                entry.timestamp = Some(prost_types::Timestamp::from(std::time::SystemTime::now()));
            }
            let offset = i64::try_from(offset).map_err(|_| {
                StoreError::Persistence("entry batch exceeds SQLite INTEGER".to_string())
            })?;
            let sequence = head
                .checked_add(offset + 1)
                .ok_or_else(|| StoreError::Persistence("entry sequence overflow".to_string()))?;
            entry.sequence =
                Some(u64::try_from(sequence).map_err(|_| {
                    StoreError::Persistence("entry sequence is negative".to_string())
                })?);
            let mut stored = entry.clone();
            let mut system_prompt_hash = None;
            if let Some(agent_message::Payload::ContextWindow(window)) = &mut stored.payload {
                let hash = digest(&SHA256, window.system_prompt.as_bytes());
                transaction
                    .execute(
                        "INSERT INTO react_system_prompt(hash, content) \
                         VALUES (?, ?) ON CONFLICT(hash) DO NOTHING",
                        params![hash.as_ref(), window.system_prompt],
                    )
                    .map_err(persistence)?;
                system_prompt_hash = Some(hash.as_ref().to_vec());
                window.system_prompt.clear();
            }
            insert
                .execute(params![
                    session_id,
                    sequence,
                    entry.id,
                    stored.encode_to_vec(),
                    system_prompt_hash,
                ])
                .map_err(persistence)?;
        }
        drop(insert);
        transaction.commit().map_err(persistence)
    }

    pub fn fork(
        database: &Path,
        source: &str,
        at_entry_id: &str,
        new_session: &str,
    ) -> Result<(), StoreError> {
        validate_session_id(source)?;
        validate_session_id(new_session)?;
        let mut connection = connect(database)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(persistence)?;
        let parent = session_row(&transaction, source)?;
        let lineage = crate::Lineage::parse(
            source.to_string(),
            serde_json::from_str(&parent.lineage).map_err(persistence)?,
            serde_json::from_str(&parent.fork_points).map_err(persistence)?,
        )
        .map_err(persistence)?;
        let target: Option<(i64, Vec<u8>)> = transaction
            .query_row(
                &format!("SELECT e.sequence, e.payload {VIEW} AND e.entry_id = ?"),
                params![source, at_entry_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(persistence)?;
        let Some((sequence, payload)) = target else {
            return Err(StoreError::EntryNotFound {
                session_id: source.to_string(),
                entry_id: at_entry_id.to_string(),
            });
        };
        let target = decode_entry(sequence, &payload)?;
        let point =
            crate::ForkPoint::parse(&target).ok_or_else(|| StoreError::InvalidForkTarget {
                entry_id: at_entry_id.to_string(),
            })?;
        let (lineage, fork_points) = lineage
            .fork(new_session.to_string(), point)
            .map_err(persistence)?
            .into_columns();
        transaction
            .execute(
                "INSERT INTO react_session(id, workspace, lineage, fork_points) \
                 VALUES (?, ?, ?, ?)",
                params![
                    new_session,
                    parent.workspace,
                    serde_json::to_string(&lineage).map_err(persistence)?,
                    serde_json::to_string(&fork_points).map_err(persistence)?,
                ],
            )
            .map_err(persistence)?;
        transaction.commit().map_err(persistence)
    }
}

impl Store for SQLiteStore {
    fn load<'a>(
        &'a mut self,
        session_id: &'a str,
        full_history: bool,
    ) -> BoxFuture<'a, Result<Vec<AgentMessage>, StoreError>> {
        let result = self
            .load_context(session_id, !full_history, None)
            .map(|(entries, _)| entries);
        Box::pin(async move { result })
    }

    fn messages_page<'a>(
        &'a mut self,
        session_id: &'a str,
        offset: u64,
        limit: u32,
    ) -> BoxFuture<'a, Result<(Vec<AgentMessage>, u64), StoreError>> {
        let result = self.load_context(session_id, false, Some((offset, limit)));
        Box::pin(async move { result })
    }

    fn append<'a>(
        &'a mut self,
        session_id: &'a str,
        head: u64,
        entries: &'a mut [AgentMessage],
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            let checkpoint = if self.checkpoints_enabled
                && entries.iter().any(|entry| {
                    matches!(entry.chat_message(), Some(chat_record::Message::User(_)))
                }) {
                match self.checkpoints.capture().await {
                    Ok(checkpoint) => checkpoint,
                    Err(error) => {
                        eprintln!(
                            "repository checkpoint skipped for session {session_id:?}: {error}"
                        );
                        None
                    }
                }
            } else {
                None
            };
            for entry in entries.iter_mut() {
                if let Some(agent_message::Payload::Chat(chat)) = &mut entry.payload {
                    if let Some(chat_record::Message::User(user)) = &mut chat.message {
                        user.checkpoint = checkpoint.clone();
                    }
                }
            }
            self.append_context(session_id, head, entries)
        })
    }
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        // The lock belongs to the open file description, which a subprocess
        // being spawned concurrently shares until its exec closes the
        // descriptor, so closing ours alone can leave the conversation locked.
        let _ = self.0.unlock();
    }
}

fn connect(path: &Path) -> Result<Connection, StoreError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(persistence)?;
    }
    let mut connection = Connection::open(path).map_err(persistence)?;
    connection
        .pragma_update(None, "foreign_keys", "ON")
        .map_err(persistence)?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(persistence)?;
    sqlite_migrations::migrate(&mut connection)?;
    Ok(connection)
}

fn lock_session(database: &Path, session_id: &str) -> Result<Option<SessionLock>, StoreError> {
    if database == Path::new(":memory:") {
        return Ok(None);
    }
    let failed = |path: &Path, error: std::io::Error| {
        StoreError::Persistence(format!(
            "could not lock conversation {session_id:?} through {path:?}: {error}"
        ))
    };
    let mut directory = std::fs::canonicalize(database)
        .map_err(|error| failed(database, error))?
        .into_os_string();
    directory.push(".locks");
    let directory = PathBuf::from(directory);
    std::fs::create_dir_all(&directory).map_err(|error| failed(&directory, error))?;
    let path = directory.join(format!("{session_id}.lock"));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|error| failed(&path, error))?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            let mut holder = String::new();
            file.read_to_string(&mut holder)
                .map_err(|error| failed(&path, error))?;
            return Err(StoreError::OpenElsewhere {
                session_id: session_id.to_string(),
                holder: holder.trim().to_string(),
            });
        }
        Err(TryLockError::Error(error)) => return Err(failed(&path, error)),
    }
    let mut lock = SessionLock(file);
    lock.0.set_len(0).map_err(|error| failed(&path, error))?;
    write!(lock.0, "{}", std::process::id()).map_err(|error| failed(&path, error))?;
    Ok(Some(lock))
}

struct SessionRow {
    workspace: String,
    lineage: String,
    fork_points: String,
}

fn session_row(connection: &Connection, session_id: &str) -> Result<SessionRow, StoreError> {
    connection
        .query_row(
            "SELECT workspace, lineage, fork_points \
             FROM react_session WHERE id = ?",
            [session_id],
            |row| {
                Ok(SessionRow {
                    workspace: row.get(0)?,
                    lineage: row.get(1)?,
                    fork_points: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(persistence)?
        .ok_or_else(|| StoreError::SessionNotFound {
            session_id: session_id.to_string(),
        })
}

pub(super) fn decode_entry(sequence: i64, payload: &[u8]) -> Result<AgentMessage, StoreError> {
    let mut entry = AgentMessage::decode(payload)
        .map_err(|error| StoreError::Persistence(format!("invalid protobuf entry: {error}")))?;
    entry.sequence = Some(
        u64::try_from(sequence)
            .map_err(|_| StoreError::Persistence("entry sequence is negative".to_string()))?,
    );
    Ok(entry)
}

fn validate_session_id(session_id: &str) -> Result<(), StoreError> {
    if !session_id.is_empty()
        && session_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "._-".contains(character))
    {
        return Ok(());
    }
    Err(StoreError::Persistence(format!(
        "invalid conversation ID {session_id:?}"
    )))
}

fn persistence(error: impl std::fmt::Display) -> StoreError {
    StoreError::Persistence(error.to_string())
}

#[cfg(test)]
#[path = "sqlite_store_test.rs"]
mod tests;
