//! Indexed conversation summaries and read-only prompt history; no execution ownership.

use std::path::Path;
use std::time::Duration;

use mirror_protocol::daemon::{
    ConversationSummary, GetConversationMessagesRequest, GetConversationMessagesResponse,
    ListConversationsRequest,
};
use mirror_protocol::{agent_message, chat_record, AgentMessage, ChatRecord};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};

use super::sqlite_migrations;
use super::sqlite_store::{decode_entry, VIEW};
use crate::StoreError;

/// Read a conversation's original directory without creating or locking history.
pub fn conversation_directory(
    database: &Path,
    session_id: &str,
) -> Result<Option<String>, StoreError> {
    let Some(connection) = read_connection(database)? else {
        return Ok(None);
    };
    connection
        .query_row(
            "SELECT workspace FROM react_session WHERE id = ?",
            [session_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(persistence)
}

/// Read an indexed page, including empty conversations, without decoding transcripts.
pub fn list_conversations(
    database: &Path,
    request: &ListConversationsRequest,
) -> Result<Vec<ConversationSummary>, StoreError> {
    let Some(connection) = read_connection(database)? else {
        return Ok(Vec::new());
    };
    let mut query = String::from(
        "SELECT id, workspace, title, model, created_at_seconds, created_at_nanos, \
         updated_at_seconds, updated_at_nanos FROM react_session WHERE 1=1",
    );
    let mut values: Vec<rusqlite::types::Value> = Vec::new();
    if !request.session_ids.is_empty() {
        query.push_str(" AND id IN (SELECT value FROM json_each(?))");
        values.push(
            serde_json::to_string(&request.session_ids)
                .map_err(persistence)?
                .into(),
        );
    }
    if request.recent_first {
        if let Some(before) = &request.before_updated_at {
            query.push_str(" AND (COALESCE(updated_at_seconds, 0), COALESCE(updated_at_nanos, 0), id) < (?, ?, ?)");
            values.extend([
                before.seconds.into(),
                before.nanos.into(),
                request.after_id.clone().into(),
            ]);
        }
        query.push_str(" ORDER BY COALESCE(updated_at_seconds, 0) DESC, COALESCE(updated_at_nanos, 0) DESC, id DESC");
    } else {
        query.push_str(" AND id > ? ORDER BY id");
        values.push(request.after_id.clone().into());
    }
    query.push_str(" LIMIT ?");
    values.push(i64::from(request.limit.min(501)).into());
    let mut statement = connection.prepare(&query).map_err(persistence)?;
    let rows = statement
        .query_map(rusqlite::params_from_iter(values), |row| {
            Ok(ConversationSummary {
                id: row.get(0)?,
                directory: row.get(1)?,
                title: row.get(2)?,
                model: row.get(3)?,
                created_at: row
                    .get::<_, Option<i64>>(4)?
                    .zip(row.get(5)?)
                    .map(|(seconds, nanos)| prost_types::Timestamp { seconds, nanos }),
                updated_at: row
                    .get::<_, Option<i64>>(6)?
                    .zip(row.get(7)?)
                    .map(|(seconds, nanos)| prost_types::Timestamp { seconds, nanos }),
            })
        })
        .map_err(persistence)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(persistence)
}

/// Read an indexed tail across fork boundaries without loading the context window.
pub fn conversation_messages(
    database: &Path,
    request: &GetConversationMessagesRequest,
) -> Result<GetConversationMessagesResponse, StoreError> {
    let mut page = GetConversationMessagesResponse::default();
    if request.before_sequence == Some(0) {
        return Ok(page);
    }
    let Some(mut connection) = read_connection(database)? else {
        return Ok(page);
    };
    let transaction = connection.transaction().map_err(persistence)?;
    let mut lineage = transaction
        .prepare(
            "SELECT l.value, f.value FROM react_session s, json_each(s.lineage) l, \
             json_each(s.fork_points) f WHERE s.id = ? AND l.key = f.key \
             ORDER BY CAST(l.key AS INTEGER) DESC",
        )
        .map_err(persistence)?;
    let segments = lineage
        .query_map([&request.session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })
        .map_err(persistence)?;
    // Bound each indexed lineage lookup instead of sorting the entire fork history.
    let mut entries = transaction
        .prepare(
            "SELECT sequence, payload FROM react_entry \
             WHERE session_id = ?1 AND sequence <= ?2 \
             ORDER BY sequence DESC LIMIT ?3",
        )
        .map_err(persistence)?;
    let before = request
        .before_sequence
        .map(|value| value.saturating_sub(1))
        .unwrap_or(i64::MAX as u64)
        .min(i64::MAX as u64) as i64;
    let limit = if request.limit == 0 {
        32
    } else {
        request.limit.min(128)
    } as usize;
    let mut bytes = 0;
    'lineage: for segment in segments {
        let (id, fork_point) = segment.map_err(persistence)?;
        let mut rows = entries
            .query(params![
                id,
                fork_point.min(before),
                (limit + 1 - page.messages.len()) as i64
            ])
            .map_err(persistence)?;
        while let Some(row) = rows.next().map_err(persistence)? {
            let payload = row
                .get_ref(1)
                .map_err(persistence)?
                .as_blob()
                .map_err(persistence)?;
            if page.messages.len() == limit
                || (!page.messages.is_empty() && bytes + payload.len() > 4 * 1024 * 1024)
            {
                page.next_before_sequence = page.messages.last().and_then(|entry| entry.sequence);
                break 'lineage;
            }
            if payload.len() > 10 * 1024 * 1024 {
                return Err(persistence(
                    "conversation message exceeds 10 MiB read limit",
                ));
            }
            bytes += payload.len();
            page.messages
                .push(decode_entry(row.get(0).map_err(persistence)?, payload)?);
        }
    }
    page.messages.reverse();
    Ok(page)
}

pub(super) fn backfill_summaries(connection: &Connection) -> Result<(), StoreError> {
    let mut statement = connection
        .prepare("SELECT id, workspace FROM react_session")
        .map_err(persistence)?;
    let sessions = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(persistence)?;
    for session in sessions {
        let (id, directory) = session.map_err(persistence)?;
        rebuild_summary(connection, &id, &directory)?;
    }
    Ok(())
}

pub(super) fn rebuild_summary(
    connection: &Connection,
    id: &str,
    directory: &str,
) -> Result<(), StoreError> {
    let summary = conversation_summary(connection, id.to_owned(), directory.to_owned())?;
    connection.execute(
        "UPDATE react_session SET title = ?2, model = ?3, \
         created_at_seconds = COALESCE(?4, created_at_seconds), created_at_nanos = COALESCE(?5, created_at_nanos), \
         updated_at_seconds = COALESCE(?6, updated_at_seconds), updated_at_nanos = COALESCE(?7, updated_at_nanos) WHERE id = ?1",
        params![id, summary.title, summary.model, summary.created_at.as_ref().map(|time| time.seconds), summary.created_at.as_ref().map(|time| time.nanos), summary.updated_at.as_ref().map(|time| time.seconds), summary.updated_at.as_ref().map(|time| time.nanos)],
    ).map_err(persistence)?;
    Ok(())
}

pub(super) fn append_summary(
    connection: &Connection,
    id: &str,
    entries: &[AgentMessage],
) -> Result<(), StoreError> {
    let title = entries
        .iter()
        .find_map(user_prompt)
        .map(|content| {
            content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .chars()
                .take(80)
                .collect::<String>()
        })
        .unwrap_or_default();
    let model = entries
        .iter()
        .rev()
        .filter_map(|entry| entry.inference.as_ref())
        .find(|inference| !inference.model.is_empty())
        .map(|inference| &inference.model);
    let first = entries.first().and_then(|entry| entry.timestamp.as_ref());
    let last = entries.last().and_then(|entry| entry.timestamp.as_ref());
    connection.execute(
        "UPDATE react_session SET title = CASE WHEN title = '' THEN ?2 ELSE title END, model = COALESCE(?3, model), \
         created_at_seconds = COALESCE(created_at_seconds, ?4), created_at_nanos = COALESCE(created_at_nanos, ?5), \
         updated_at_seconds = COALESCE(?6, updated_at_seconds), updated_at_nanos = COALESCE(?7, updated_at_nanos) WHERE id = ?1",
        params![id, title, model, first.map(|time| time.seconds), first.map(|time| time.nanos), last.map(|time| time.seconds), last.map(|time| time.nanos)],
    ).map_err(persistence)?;
    Ok(())
}

fn conversation_summary(
    connection: &Connection,
    id: String,
    directory: String,
) -> Result<ConversationSummary, StoreError> {
    let mut summary = ConversationSummary {
        id,
        directory,
        ..Default::default()
    };
    let mut statement = connection
        .prepare(&format!(
            "SELECT e.sequence, e.payload {VIEW} ORDER BY e.sequence"
        ))
        .map_err(persistence)?;
    let mut rows = statement.query([&summary.id]).map_err(persistence)?;
    while let Some(row) = rows.next().map_err(persistence)? {
        let entry = decode_entry(
            row.get(0).map_err(persistence)?,
            row.get_ref(1)
                .map_err(persistence)?
                .as_blob()
                .map_err(persistence)?,
        )?;
        if summary.created_at.is_none() {
            summary.created_at = entry.timestamp;
        }
        summary.updated_at = entry.timestamp;
        if let Some(inference) = &entry.inference {
            if !inference.model.is_empty() {
                summary.model.clone_from(&inference.model);
            }
        }
        if summary.title.is_empty() {
            if let Some(content) = user_prompt(&entry) {
                summary.title = content
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .chars()
                    .take(80)
                    .collect();
            }
        }
    }
    Ok(summary)
}

/// Prefer lineage prompts, then other inputs in reverse durable insertion order.
pub fn prompt_history(
    database: &Path,
    session_id: &str,
    limit: u32,
) -> Result<Vec<String>, StoreError> {
    let Some(mut connection) = read_connection(database)? else {
        return Ok(Vec::new());
    };
    let transaction = connection.transaction().map_err(persistence)?;
    let mut prompts = Vec::new();
    let mut bytes = 0;
    let queries = [
        format!("SELECT e.sequence, e.payload {VIEW} ORDER BY e.sequence DESC"),
        format!("SELECT sequence, payload FROM react_entry WHERE id NOT IN (SELECT e.id {VIEW}) ORDER BY id DESC"),
    ];
    for query in queries {
        let mut statement = transaction.prepare(&query).map_err(persistence)?;
        let mut rows = statement.query([session_id]).map_err(persistence)?;
        while let Some(row) = rows.next().map_err(persistence)? {
            if prompts.len() >= limit.min(1000) as usize {
                return Ok(prompts);
            }
            let entry = decode_entry(
                row.get(0).map_err(persistence)?,
                row.get_ref(1)
                    .map_err(persistence)?
                    .as_blob()
                    .map_err(persistence)?,
            )?;
            let Some(content) = user_prompt(&entry) else {
                continue;
            };
            if prompts.last().map(String::as_str) == Some(content) {
                continue;
            }
            bytes += content.len();
            if bytes > 4 * 1024 * 1024 {
                return Ok(prompts);
            }
            prompts.push(content.to_owned());
        }
    }
    Ok(prompts)
}

fn user_prompt(entry: &AgentMessage) -> Option<&str> {
    match &entry.payload {
        Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(user)),
            display: None,
            ..
        })) if !user.content.trim().is_empty() => Some(&user.content),
        _ => None,
    }
}

fn read_connection(database: &Path) -> Result<Option<Connection>, StoreError> {
    match database.try_exists() {
        Ok(false) => return Ok(None),
        Ok(true) => {}
        Err(error) => return Err(persistence(error)),
    }
    let connection = Connection::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(persistence)?;
    connection
        .busy_timeout(Duration::from_secs(5))
        .map_err(persistence)?;
    sqlite_migrations::validate_reader(&connection)?;
    Ok(Some(connection))
}

fn persistence(error: impl std::fmt::Display) -> StoreError {
    StoreError::Persistence(error.to_string())
}

#[cfg(test)]
#[path = "sqlite_history_test.rs"]
mod tests;
