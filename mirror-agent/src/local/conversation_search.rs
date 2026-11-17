//! Disposable message-text cache; canonical history is opened read-only and never migrated.

use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use mirror_protocol::daemon::{
    ConversationSearchMatch, SearchConversationsRequest, SearchConversationsResponse,
};
use mirror_protocol::{agent_message, chat_record, AgentMessage, ChatRecord};
use rusqlite::{params, Connection, ErrorCode, OpenFlags, OptionalExtension};

use super::sqlite_migrations;
use super::sqlite_store::decode_entry;
use crate::StoreError;

const CACHE_SCHEMA: &str = "
CREATE TABLE source (identity TEXT NOT NULL);
CREATE TABLE entries (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, sequence INTEGER NOT NULL, entry_id TEXT NOT NULL);
CREATE VIRTUAL TABLE message_fts USING fts5(text, tokenize='unicode61 remove_diacritics 2', prefix='2 3 4');
PRAGMA user_version = 1;
";

/// Index a bounded batch without writing history or taking a worker's session lock.
pub fn index_conversation_history(database: &Path) -> Result<bool, StoreError> {
    let Some(mut connection) = read_connection(database)? else {
        return Ok(false);
    };
    let cache = cache_path(database);
    prepare_cache(&cache, database)?;
    let result = (|| {
        connection.execute("ATTACH DATABASE ? AS search", [cache.to_string_lossy()])?;
        let transaction = connection.transaction()?;
        let cursor = indexed_through(&transaction)?;
        let mut statement = transaction.prepare(
            "SELECT id, session_id, sequence, entry_id, payload FROM main.react_entry WHERE id > ? ORDER BY id LIMIT 32",
        )?;
        let mut rows = statement.query([cursor])?;
        let mut bytes = 0;
        let mut entries = Vec::new();
        while let Some(row) = rows.next()? {
            let payload = row.get_ref(4)?.as_blob()?;
            if bytes > 0 && bytes + payload.len() > 4 * 1024 * 1024 {
                break;
            }
            if payload.len() > 10 * 1024 * 1024 {
                return Err(rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Blob,
                    "conversation message exceeds 10 MiB indexing limit".into(),
                ));
            }
            let sequence: i64 = row.get(2)?;
            let entry = decode_entry(sequence, payload).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Blob,
                    Box::new(error),
                )
            })?;
            entries.push((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                sequence,
                row.get::<_, String>(3)?,
                entry,
            ));
            bytes += payload.len();
        }
        drop(rows);
        drop(statement);
        let idle = entries.is_empty();
        for (id, session, sequence, entry_id, entry) in entries {
            transaction.execute("DELETE FROM search.message_fts WHERE rowid = ?", [id])?;
            transaction.execute(
                "INSERT OR REPLACE INTO search.entries VALUES (?, ?, ?, ?)",
                params![id, session, sequence, entry_id],
            )?;
            if let Some(text) = message_text(&entry).filter(|text| !text.trim().is_empty()) {
                transaction.execute(
                    "INSERT INTO search.message_fts(rowid, text) VALUES (?, ?)",
                    params![id, text],
                )?;
            }
        }
        let pruned = idle && prune_deleted(&transaction)?;
        let pending = index_pending(&transaction)?;
        transaction.commit()?;
        Ok(pending || pruned)
    })();
    drop(connection);
    cache_result(&cache, database, result)
}

fn message_text(entry: &AgentMessage) -> Option<&str> {
    match entry.payload.as_ref() {
        Some(agent_message::Payload::Chat(ChatRecord {
            message,
            display: None,
            ..
        })) => match message {
            Some(chat_record::Message::User(user)) => Some(user.content.as_str()),
            Some(chat_record::Message::Assistant(assistant)) => assistant.content.as_deref(),
            _ => None,
        },
        _ => None,
    }
}

// History entries are immutable after v4; deletions can reuse SQLite row IDs, not message identities.
fn indexed_through(connection: &Connection) -> rusqlite::Result<i64> {
    connection
        .query_row(
            "SELECT c.id FROM search.entries c JOIN main.react_entry e ON e.id = c.id \
         AND e.session_id = c.session_id AND e.sequence = c.sequence AND e.entry_id = c.entry_id \
         WHERE c.id <= (SELECT MAX(id) FROM main.react_entry) ORDER BY c.id DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .map(|id| id.unwrap_or(0))
}

fn index_pending(connection: &Connection) -> rusqlite::Result<bool> {
    connection.query_row(
        "SELECT EXISTS (SELECT 1 FROM main.react_entry WHERE id > ?)",
        [indexed_through(connection)?],
        |row| row.get(0),
    )
}

fn prune_deleted(connection: &Connection) -> rusqlite::Result<bool> {
    let mut statement = connection.prepare(
        "SELECT c.id FROM search.entries c WHERE NOT EXISTS (SELECT 1 FROM main.react_entry e \
         WHERE e.id = c.id AND e.session_id = c.session_id AND e.sequence = c.sequence AND e.entry_id = c.entry_id) LIMIT 32",
    )?;
    let ids = statement
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    for id in &ids {
        connection.execute("DELETE FROM search.message_fts WHERE rowid = ?", [id])?;
        connection.execute("DELETE FROM search.entries WHERE id = ?", [id])?;
    }
    Ok(!ids.is_empty())
}

/// Search committed public text with current history identities and fork boundaries checked before ranking.
pub fn search_conversations(
    database: &Path,
    request: &SearchConversationsRequest,
) -> Result<SearchConversationsResponse, StoreError> {
    if request.query.len() > 1024
        || request.query.chars().count() > 256
        || request.limit > 50
        || request.session_ids.len() > 10_000
    {
        return Err(persistence(
            "conversation search exceeds query, limit or identity bounds",
        ));
    }
    let Some(mut connection) = read_connection(database)? else {
        return Ok(SearchConversationsResponse::default());
    };
    let cache = cache_path(database);
    if !cache.try_exists().map_err(persistence)? {
        return Ok(SearchConversationsResponse {
            indexing: true,
            ..Default::default()
        });
    }
    validate_cache_file(&cache)?;
    let identity = source_identity(database)?;
    let result = (|| {
        connection.execute("ATTACH DATABASE ? AS search", [cache.to_string_lossy()])?;
        let transaction = connection.transaction()?;
        let version =
            transaction.query_row("PRAGMA search.user_version", [], |row| row.get::<_, i64>(0))?;
        if version != 1
            || transaction.query_row("SELECT identity FROM search.source", [], |row| {
                row.get::<_, String>(0)
            })? != identity
        {
            return Ok(SearchConversationsResponse {
                indexing: true,
                ..Default::default()
            });
        }
        search_index(&transaction, request)
    })();
    drop(connection);
    cache_result(&cache, database, result)
}

fn search_index(
    connection: &Connection,
    request: &SearchConversationsRequest,
) -> rusqlite::Result<SearchConversationsResponse> {
    let mut response = SearchConversationsResponse {
        indexing: index_pending(connection)?,
        ..Default::default()
    };
    // Memory recall's OR query is intentionally different: search requires every typed word.
    let query = request
        .query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(|term| format!("\"{term}\"*"))
        .collect::<Vec<_>>()
        .join(" AND ");
    if query.is_empty() {
        return Ok(response);
    }
    let mut statement = connection.prepare(
        "WITH scope AS MATERIALIZED (\
           SELECT s.id, l.value AS source, f.value AS through_sequence \
           FROM main.react_session s, json_each(s.lineage) l, json_each(s.fork_points) f \
           WHERE l.key = f.key AND (?2 = '[]' OR s.id IN (SELECT value FROM json_each(?2)))\
         ), matches AS MATERIALIZED (\
           SELECT e.session_id, e.sequence, e.entry_id, rank, \
             substr(snippet(message_fts, 0, '', '', ' … ', 24), 1, 1024) AS snippet \
           FROM search.message_fts JOIN search.entries c ON c.id = message_fts.rowid \
             JOIN main.react_entry e ON e.id = c.id AND e.session_id = c.session_id \
               AND e.sequence = c.sequence AND e.entry_id = c.entry_id \
           WHERE message_fts MATCH ?1 \
             AND EXISTS (SELECT 1 FROM scope WHERE source = e.session_id AND e.sequence <= through_sequence)\
         ), ranked AS (\
           SELECT scope.id, m.entry_id, m.sequence, m.snippet, m.rank, \
             row_number() OVER (PARTITION BY scope.id ORDER BY m.rank, m.sequence DESC) AS position \
           FROM matches m JOIN scope ON m.session_id = scope.source AND m.sequence <= scope.through_sequence\
         ) SELECT id, entry_id, sequence, snippet FROM ranked \
           WHERE position = 1 ORDER BY rank, sequence DESC, id LIMIT ?3",
    )?;
    let scope = serde_json::to_string(&request.session_ids)
        .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))?;
    let limit = if request.limit == 0 {
        20
    } else {
        request.limit
    };
    response.matches = statement
        .query_map(params![query, scope, limit], |row| {
            let sequence: i64 = row.get(2)?;
            Ok(ConversationSearchMatch {
                session_id: row.get(0)?,
                message_id: row.get(1)?,
                sequence: u64::try_from(sequence)
                    .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(2, sequence))?,
                snippet: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(response)
}

fn read_connection(database: &Path) -> Result<Option<Connection>, StoreError> {
    if !database.try_exists().map_err(persistence)? {
        return Ok(None);
    }
    let mut uri = reqwest::Url::from_file_path(database.canonicalize().map_err(persistence)?)
        .map_err(|_| persistence("history database path cannot be encoded as a file URI"))?;
    uri.set_query(Some("mode=ro"));
    // URI mode protects history; the connection flags allow writes only to the attached cache.
    let connection = Connection::open_with_flags(
        uri.as_str(),
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(persistence)?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(persistence)?;
    sqlite_migrations::validate_reader(&connection)?;
    Ok(Some(connection))
}

fn cache_path(database: &Path) -> PathBuf {
    let mut path = database.as_os_str().to_owned();
    path.push(".search-v1.sqlite");
    PathBuf::from(path)
}

fn validate_cache_file(path: &Path) -> Result<(), StoreError> {
    let metadata = std::fs::symlink_metadata(path).map_err(persistence)?;
    if !metadata.is_file() || metadata.nlink() != 1 || metadata.mode() & 0o077 != 0 {
        return Err(persistence(
            "conversation search cache must be a private regular file",
        ));
    }
    Ok(())
}

fn source_identity(database: &Path) -> Result<String, StoreError> {
    let metadata = std::fs::metadata(database).map_err(persistence)?;
    Ok(format!("{}:{}", metadata.dev(), metadata.ino()))
}

fn prepare_cache(path: &Path, database: &Path) -> Result<(), StoreError> {
    if !path.try_exists().map_err(persistence)? {
        return rebuild_cache(path, database);
    }
    let identity = source_identity(database)?;
    validate_cache_file(path)?;
    let canonical = path.canonicalize().map_err(persistence)?;
    let valid = (|| {
        let connection = Connection::open_with_flags(
            &canonical,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        connection.prepare("SELECT id, session_id, sequence, entry_id FROM entries LIMIT 0")?;
        connection.prepare("SELECT rowid, text FROM message_fts LIMIT 0")?;
        let version =
            connection.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))?;
        let source = connection.query_row("SELECT identity FROM source", [], |row| {
            row.get::<_, String>(0)
        })?;
        Ok(version == 1 && source == identity)
    })();
    match valid {
        Ok(true) => Ok(()),
        Ok(false) => rebuild_cache(path, database),
        Err(error) if rebuildable(&error) => rebuild_cache(path, database),
        Err(error) => Err(persistence(error)),
    }
}

fn rebuild_cache(path: &Path, database: &Path) -> Result<(), StoreError> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file = tempfile::NamedTempFile::new_in(parent).map_err(persistence)?;
    let connection = Connection::open(file.path()).map_err(persistence)?;
    connection
        .execute_batch(CACHE_SCHEMA)
        .map_err(persistence)?;
    connection
        .execute(
            "INSERT INTO source VALUES (?)",
            [source_identity(database)?],
        )
        .map_err(persistence)?;
    drop(connection);
    // Only the daemon search actor owns this cache; old journals must not resurrect its contents.
    for suffix in ["-journal", "-wal", "-shm"] {
        let mut journal = path.as_os_str().to_owned();
        journal.push(suffix);
        match std::fs::remove_file(journal) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(persistence(error)),
        }
    }
    file.persist(path).map_err(persistence)?;
    Ok(())
}

fn rebuildable(error: &rusqlite::Error) -> bool {
    matches!(error, rusqlite::Error::QueryReturnedNoRows)
        || matches!(error, rusqlite::Error::SqliteFailure(error, _) if matches!(error.code, ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase | ErrorCode::Unknown | ErrorCode::SchemaChanged))
}

fn cache_result<T>(
    path: &Path,
    database: &Path,
    result: rusqlite::Result<T>,
) -> Result<T, StoreError> {
    if result.as_ref().is_err_and(rebuildable) {
        rebuild_cache(path, database)?;
    }
    result.map_err(persistence)
}

fn persistence(error: impl std::fmt::Display) -> StoreError {
    StoreError::Persistence(error.to_string())
}

#[cfg(test)]
#[path = "conversation_search_test.rs"]
mod tests;
