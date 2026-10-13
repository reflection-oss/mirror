//! SQLite storage for one owner's memory: record rows, an append-only event log,
//! and an FTS5 index over active records.

use std::path::Path;
use std::time::Duration;

use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OpenFlags;
use rusqlite::OptionalExtension;
use rusqlite::Row;
use rusqlite::Transaction;
use rusqlite::TransactionBehavior;

use super::invalid;
use super::now;
use super::split_compounds;
use super::validate_record;
use super::MemoryActor;
use super::MemoryChange;
use super::MemoryCommit;
use super::MemoryError;
use super::MemoryEventOp;
use super::MemoryQuery;
use super::MemoryRecord;
use super::MemoryResult;
use super::MemoryStatus;
use super::MemoryType;

/// Owns the connection to one private store, independent of any session.
#[derive(Debug)]
pub struct SqliteMemoryDatabase {
    connection: Connection,
}

pub const MEMORY_DATABASE: &str = "memory.sqlite3";

/// One appended change, as replayed from the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEvent {
    pub seq: i64,
    pub memory_id: String,
    pub op: String,
    pub actor: String,
    pub session_id: Option<String>,
    /// The record version this change produced.
    pub version: u64,
    pub at_unix_seconds: u64,
}

/// Maintenance bookkeeping for the periodic consolidation pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MaintenanceState {
    pub last_consolidated_at: u64,
    pub last_event_seq: i64,
}

/// The schema, created on open. Nothing distributed has written the store yet,
/// so there are no migrations; the first schema change after a release adds them.
const SCHEMA: &str = include_str!("schema.sql");

const COLUMNS: &str = "id, key, type, status, version, actor, repository, path_globs, \
                       situation, keywords, content, observed_at, updated_at, source";
/// Where a ranked search's BM25 score lands: right after the record's columns.
const SCORE_COLUMN: usize = 14;

impl SqliteMemoryDatabase {
    /// Open or initialize the store under `directory` with private permissions.
    pub fn open(directory: &Path) -> MemoryResult<Self> {
        use std::os::unix::fs::DirBuilderExt;
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::fs::PermissionsExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(directory)?;
        // NOFOLLOW rejects symlinked path components (macOS `/var`), so resolve the
        // directory first; the database file itself must still not be a symlink.
        let directory = std::fs::canonicalize(directory)?;
        let path = directory.join(MEMORY_DATABASE);
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)?;
        let connection = Connection::open_with_flags(
            &path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        // Tightened on every open, so a store created under a looser umask is
        // private too, not only one whose files this call created.
        std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        connection.busy_timeout(Duration::from_secs(2))?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        require_fts5(&connection)?;
        connection.execute_batch(SCHEMA)?;
        Ok(Self { connection })
    }

    /// Records matching `query`, best first for a lexical search and newest
    /// first for a scan. Scope predicates and exclusions apply before the
    /// limit, so another repository's matches never consume it.
    pub fn search(&self, query: &MemoryQuery) -> MemoryResult<Vec<MemoryRecord>> {
        match &query.text {
            Some(text) => search_text(&self.connection, query, text),
            None => scan(&self.connection, query),
        }
    }

    /// One record by id, whatever its status.
    pub fn get(&self, id: &str) -> MemoryResult<Option<MemoryRecord>> {
        read_record(&self.connection, id)
    }

    /// Apply every change together or none. Each update must carry exactly the
    /// next version of its record, so a change computed against a stale read
    /// fails as `Stale` and changes nothing.
    pub fn commit(&mut self, commit: &MemoryCommit) -> MemoryResult<()> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let at = now();
        for change in &commit.changes {
            match change {
                MemoryChange::Insert(record) => {
                    insert(&transaction, record, commit, at)?;
                }
                MemoryChange::Update { record, operation } => {
                    update(&transaction, record, *operation, commit, at)?;
                }
            }
        }
        transaction.commit()?;
        Ok(())
    }

    /// The log for one record, oldest first.
    pub fn events(&self, memory_id: &str) -> MemoryResult<Vec<MemoryEvent>> {
        let mut statement = self.connection.prepare(
            "SELECT seq, memory_id, op, actor, session_id, version, at \
             FROM memory_event WHERE memory_id = ?1 ORDER BY seq",
        )?;
        let rows = statement.query_map([memory_id], |row| {
            Ok(MemoryEvent {
                seq: row.get(0)?,
                memory_id: row.get(1)?,
                op: row.get(2)?,
                actor: row.get(3)?,
                session_id: row.get(4)?,
                version: row
                    .get::<_, i64>(5)
                    .map(|version| u64::try_from(version).unwrap_or_default())?,
                at_unix_seconds: row
                    .get::<_, i64>(6)
                    .map(|at| u64::try_from(at).unwrap_or_default())?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(MemoryError::from)
    }

    /// Consolidation bookkeeping for the whole store: a pass consolidates
    /// every repository's memories at once.
    pub fn maintenance(&self) -> MemoryResult<MaintenanceState> {
        maintenance(&self.connection)
    }

    /// Record a finished pass: when it ran and the change sequence it read.
    pub fn mark_consolidated(&mut self, now: u64) -> MemoryResult<()> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let state = MaintenanceState {
            last_consolidated_at: now,
            last_event_seq: latest_event_seq(&transaction)?,
        };
        write_maintenance(&transaction, &state)?;
        transaction.commit()?;
        Ok(())
    }
}

fn insert(
    transaction: &Transaction<'_>,
    record: &MemoryRecord,
    commit: &MemoryCommit,
    at: u64,
) -> MemoryResult<()> {
    validate_record(record)?;
    if record.version != 1 {
        return Err(invalid("a new memory record starts at version 1"));
    }
    if record.status != MemoryStatus::Active {
        return Err(invalid("a new memory record is active"));
    }
    if read_record(transaction, &record.id)?.is_some() {
        return Err(invalid("memory record already exists"));
    }
    let stored = MemoryRecord {
        actor: commit.actor,
        updated_at: at,
        ..record.clone()
    };
    write_row(transaction, &stored)?;
    append_event(transaction, &stored, MemoryEventOp::Accept, commit, at)
}

/// Replace the head with `after`, whose version must be exactly one past the
/// stored version; anything else is a stale edit and changes nothing. `op`
/// must be the change that takes the record from its stored status to
/// `after`'s, so the log never misstates what happened.
fn update(
    transaction: &Transaction<'_>,
    after: &MemoryRecord,
    op: MemoryEventOp,
    commit: &MemoryCommit,
    at: u64,
) -> MemoryResult<()> {
    validate_record(after)?;
    let before = read_record(transaction, &after.id)?
        .ok_or_else(|| invalid("memory record does not exist"))?;
    if before.version.checked_add(1) != Some(after.version) {
        return Err(MemoryError::Stale);
    }
    if after.observed_at != before.observed_at {
        return Err(invalid("observed_at is fixed for the life of a record"));
    }
    if !op.describes(before.status, after.status) {
        return Err(invalid(format!(
            "{} does not take a memory record from {} to {}",
            op.as_str(),
            before.status.as_str(),
            after.status.as_str()
        )));
    }
    let stored = MemoryRecord {
        actor: commit.actor,
        updated_at: at,
        ..after.clone()
    };
    write_row(transaction, &stored)?;
    append_event(transaction, &stored, op, commit, at)
}

fn write_row(connection: &Connection, record: &MemoryRecord) -> MemoryResult<()> {
    connection.execute(
        "INSERT INTO memory_record (id, key, type, status, version, actor, repository, \
         path_globs, situation, keywords, content, observed_at, updated_at, source) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14) \
         ON CONFLICT(id) DO UPDATE SET key = excluded.key, type = excluded.type, \
         status = excluded.status, version = excluded.version, actor = excluded.actor, \
         repository = excluded.repository, path_globs = excluded.path_globs, \
         situation = excluded.situation, keywords = excluded.keywords, \
         content = excluded.content, observed_at = excluded.observed_at, \
         updated_at = excluded.updated_at, source = excluded.source",
        params![
            record.id,
            record.key,
            record.r#type.as_str(),
            record.status.as_str(),
            to_i64(record.version),
            record.actor.as_str(),
            record.repository,
            json_list(&record.path_globs),
            record.situation,
            json_list(&record.keywords),
            record.content,
            to_i64(record.observed_at),
            to_i64(record.updated_at),
            record.source,
        ],
    )?;
    connection.execute("DELETE FROM memory_fts WHERE id = ?1", [&record.id])?;
    // Retired records leave every retrieval channel immediately.
    if record.status == MemoryStatus::Active {
        connection.execute(
            "INSERT INTO memory_fts (id, key, situation, keywords, content) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                record.id,
                normalized_lexical_text(&record.key.replace('-', " ")),
                normalized_lexical_text(&record.situation),
                normalized_lexical_text(&record.keywords.join(" ")),
                normalized_lexical_text(&record.content),
            ],
        )?;
    }
    Ok(())
}

fn append_event(
    connection: &Connection,
    record: &MemoryRecord,
    op: MemoryEventOp,
    commit: &MemoryCommit,
    at: u64,
) -> MemoryResult<()> {
    connection.execute(
        "INSERT INTO memory_event (memory_id, op, actor, session_id, version, at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            record.id,
            op.as_str(),
            commit.actor.as_str(),
            commit.session_id,
            to_i64(record.version),
            to_i64(at),
        ],
    )?;
    Ok(())
}

fn read_record(connection: &Connection, id: &str) -> MemoryResult<Option<MemoryRecord>> {
    connection
        .query_row(
            &format!("SELECT {COLUMNS} FROM memory_record WHERE id = ?1"),
            [id],
            record_from_row,
        )
        .optional()
        .map_err(MemoryError::from)
}

fn record_from_row(row: &Row<'_>) -> rusqlite::Result<MemoryRecord> {
    let kind: String = row.get(2)?;
    let status: String = row.get(3)?;
    let actor: String = row.get(5)?;
    let path_globs: String = row.get(7)?;
    let keywords: String = row.get(9)?;
    Ok(MemoryRecord {
        id: row.get(0)?,
        key: row.get(1)?,
        r#type: MemoryType::parse(&kind).ok_or_else(|| column_error(2, &kind))?,
        status: MemoryStatus::parse(&status).ok_or_else(|| column_error(3, &status))?,
        version: row
            .get::<_, i64>(4)
            .map(|version| u64::try_from(version).unwrap_or_default())?,
        actor: MemoryActor::parse(&actor).ok_or_else(|| column_error(5, &actor))?,
        repository: row.get(6)?,
        path_globs: serde_json::from_str(&path_globs).map_err(|_| column_error(7, &path_globs))?,
        situation: row.get(8)?,
        keywords: serde_json::from_str(&keywords).map_err(|_| column_error(9, &keywords))?,
        content: row.get(10)?,
        observed_at: row
            .get::<_, i64>(11)
            .map(|at| u64::try_from(at).unwrap_or_default())?,
        updated_at: row
            .get::<_, i64>(12)
            .map(|at| u64::try_from(at).unwrap_or_default())?,
        source: row.get(13)?,
    })
}

fn column_error(index: usize, value: &str) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        index,
        rusqlite::types::Type::Text,
        format!("unexpected memory column value {value:?}").into(),
    )
}

fn json_list(items: &[String]) -> String {
    serde_json::to_string(items).unwrap_or_else(|_| "[]".to_string())
}

fn to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn latest_event_seq(connection: &Connection) -> MemoryResult<i64> {
    Ok(connection.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM memory_event",
        [],
        |row| row.get(0),
    )?)
}

fn maintenance(connection: &Connection) -> MemoryResult<MaintenanceState> {
    let row = connection
        .query_row(
            "SELECT last_consolidated_at, last_event_seq \
             FROM memory_maintenance WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?;
    Ok(match row {
        Some((at, seq)) => MaintenanceState {
            last_consolidated_at: u64::try_from(at).unwrap_or_default(),
            last_event_seq: seq,
        },
        None => MaintenanceState::default(),
    })
}

fn write_maintenance(connection: &Connection, state: &MaintenanceState) -> MemoryResult<()> {
    connection.execute(
        "INSERT INTO memory_maintenance (singleton, last_consolidated_at, last_event_seq) \
         VALUES (1, ?1, ?2) ON CONFLICT(singleton) DO UPDATE SET \
         last_consolidated_at = excluded.last_consolidated_at, \
         last_event_seq = excluded.last_event_seq",
        params![to_i64(state.last_consolidated_at), state.last_event_seq],
    )?;
    Ok(())
}

/// Fail fast when the linked SQLite lacks the FTS5 extension the memory index needs.
fn require_fts5(connection: &Connection) -> Result<(), rusqlite::Error> {
    let enabled: i64 = connection.query_row(
        "SELECT sqlite_compileoption_used('ENABLE_FTS5')",
        [],
        |row| row.get(0),
    )?;
    if enabled == 1 {
        return Ok(());
    }
    Err(rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
        Some("the bundled SQLite was built without FTS5; memory search is unavailable".into()),
    ))
}

const MAX_QUERY_TERMS: usize = 64;
const QUERY_HEAD_CHARS: usize = 3_072;
const QUERY_TAIL_CHARS: usize = 1_024;

/// Turn a user message into an FTS5 `MATCH` expression: NFKC-folded, head/tail
/// truncated, compound-split, one quoted term per distinct token, OR-joined.
/// Returns `None` when nothing is searchable.
pub fn fts_match_query(text: &str) -> Option<String> {
    let split = normalized_lexical_text(&truncate_query(text));
    let mut seen = std::collections::BTreeSet::new();
    let mut terms = Vec::new();
    for token in split.split(|c: char| !c.is_alphanumeric()) {
        if token.chars().count() < 2 {
            continue;
        }
        let term = token.to_lowercase();
        if seen.insert(term.clone()) {
            terms.push(format!("\"{term}\""));
        }
        if terms.len() >= MAX_QUERY_TERMS {
            break;
        }
    }
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

/// The minimum BM25 strength a lexical hit needs: 0.7 × the IDF a term found in
/// exactly one record would have, so a hit must be at least as distinctive as
/// a rare word. It scales with the store at every size; with no stoplist, a
/// small store would otherwise admit matches on words like "the".
pub fn relevance_floor(indexed: usize) -> f64 {
    let n = indexed as f64;
    0.7 * ((n - 1.0 + 0.5) / (1.0 + 0.5)).ln().max(0.0)
}

fn normalized_lexical_text(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;

    split_compounds(&text.nfkc().collect::<String>())
}

fn truncate_query(text: &str) -> String {
    let count = text.chars().count();
    if count <= QUERY_HEAD_CHARS + QUERY_TAIL_CHARS {
        return text.to_string();
    }
    let head: String = text.chars().take(QUERY_HEAD_CHARS).collect();
    let tail: String = text.chars().skip(count - QUERY_TAIL_CHARS).collect();
    format!("{head} {tail}")
}

/// Records whose indexed text matches `text`, best first, held to the
/// relevance floor. Predicates and exclusions apply before the limit.
fn search_text(
    connection: &Connection,
    query: &MemoryQuery,
    text: &str,
) -> MemoryResult<Vec<MemoryRecord>> {
    let Some(fts_query) = fts_match_query(text) else {
        return Ok(Vec::new());
    };
    let indexed: i64 =
        connection.query_row("SELECT count(*) FROM memory_fts", [], |row| row.get(0))?;
    let floor = relevance_floor(usize::try_from(indexed).unwrap_or_default());
    // Column weights follow the FTS5 column order, including the UNINDEXED
    // id column, which still occupies a position.
    let mut sql = format!(
        "SELECT {}, bm25(memory_fts, 0.0, 3.0, 3.0, 3.0, 1.0) AS score \
         FROM memory_fts JOIN memory_record r ON r.id = memory_fts.id \
         WHERE memory_fts MATCH ?1",
        prefixed_columns()
    );
    let mut values: Vec<rusqlite::types::Value> = vec![fts_query.into()];
    push_filter(&mut sql, &mut values, query);
    sql.push_str(" ORDER BY score, memory_fts.id");
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(rusqlite::params_from_iter(values), |row| {
        Ok((record_from_row(row)?, row.get::<_, f64>(SCORE_COLUMN)?))
    })?;
    let mut hits: Vec<MemoryRecord> = Vec::new();
    // FTS5 cannot apply BM25 ranking inside GROUP BY, so distinct selection
    // walks ranked matches. Keep the best copy and the shortest alternative
    // for each accepted content: a later copy may fit the recall when the
    // first does not, without returning every duplicate.
    let mut representatives: std::collections::HashMap<String, (usize, Option<usize>)> =
        std::collections::HashMap::new();
    for row in rows {
        let (record, score) = row?;
        // FTS5 returns the negated BM25; a non-positive strength means the
        // only matching terms were more common than not.
        let strength = -score;
        if strength <= 0.0 || strength < floor {
            continue;
        }
        if query.distinct_content {
            let content = normalized_content(&record);
            if !representatives.contains_key(&content)
                && query.limit > 0
                && representatives.len() >= query.limit
            {
                continue;
            }
            let size = record.content.len();
            if let Some((shortest, alternative)) = representatives.get_mut(&content) {
                if size < *shortest {
                    *shortest = size;
                    if let Some(index) = alternative {
                        hits[*index] = record;
                    } else {
                        *alternative = Some(hits.len());
                        hits.push(record);
                    }
                }
                continue;
            }
            representatives.insert(content, (size, None));
        }
        hits.push(record);
        if !query.distinct_content && query.limit > 0 && hits.len() >= query.limit {
            break;
        }
    }
    Ok(hits)
}

/// Active records in scope, newest first.
fn scan(connection: &Connection, query: &MemoryQuery) -> MemoryResult<Vec<MemoryRecord>> {
    let mut sql = format!("SELECT {} FROM memory_record r WHERE 1", prefixed_columns());
    let mut values = Vec::new();
    push_filter(&mut sql, &mut values, query);
    if query.with_globs_only {
        sql.push_str(" AND r.path_globs != '[]'");
    }
    sql.push_str(" ORDER BY r.observed_at DESC, r.id");
    if query.limit > 0 {
        values.push(to_i64(query.limit as u64).into());
        sql.push_str(&format!(" LIMIT ?{}", values.len()));
    }
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(rusqlite::params_from_iter(values), record_from_row)?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(MemoryError::from)
}

fn prefixed_columns() -> String {
    COLUMNS
        .split(", ")
        .map(|column| format!("r.{}", column.trim()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Append the scope predicates of `query` against the `memory_record` alias `r`.
fn push_filter(sql: &mut String, values: &mut Vec<rusqlite::types::Value>, query: &MemoryQuery) {
    sql.push_str(" AND r.status = 'active'");
    if !query.types.is_empty() {
        let placeholders: Vec<String> = query
            .types
            .iter()
            .map(|kind| {
                values.push(kind.as_str().to_string().into());
                format!("?{}", values.len())
            })
            .collect();
        sql.push_str(&format!(" AND r.type IN ({})", placeholders.join(", ")));
    }
    match &query.repository {
        Some(repository) => {
            values.push(repository.clone().into());
            sql.push_str(&format!(
                " AND (r.repository = '' OR r.repository = ?{})",
                values.len()
            ));
        }
        None => sql.push_str(" AND r.repository = ''"),
    }
    if !query.exclude.is_empty() {
        let placeholders: Vec<String> = query
            .exclude
            .iter()
            .map(|id| {
                values.push(id.clone().into());
                format!("?{}", values.len())
            })
            .collect();
        sql.push_str(&format!(" AND r.id NOT IN ({})", placeholders.join(", ")));
    }
}

pub(crate) fn normalized_content(record: &MemoryRecord) -> String {
    record.content.trim().to_lowercase()
}

#[cfg(test)]
#[path = "sqlite_memory_database_test.rs"]
mod tests;
