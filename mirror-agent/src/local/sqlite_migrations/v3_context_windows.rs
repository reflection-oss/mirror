//! One-off conversion of legacy SQLite histories into context windows.

use crate::StoreError;
use mirror_protocol::{agent_message, chat_record, Compaction, ContextWindowRecord};
use prost::Message;
use ring::digest::{digest, SHA256};
use rusqlite::params;

use super::persistence;
use crate::local::sqlite_store::{decode_entry, VIEW};

#[derive(Clone, Message)]
pub(crate) struct LegacyFields {
    #[prost(message, optional, tag = "5")]
    pub(crate) metadata: Option<LegacyMetadata>,
    #[prost(message, optional, tag = "7")]
    pub(crate) compaction: Option<LegacyCompaction>,
}

#[derive(Clone, Message)]
pub(crate) struct LegacyMetadata {
    #[prost(string, tag = "1")]
    pub(crate) model: String,
    #[prost(string, optional, tag = "2")]
    pub(crate) compaction_model: Option<String>,
    #[prost(message, optional, tag = "5")]
    pub(crate) inference: Option<mirror_protocol::InferenceConfig>,
}

#[derive(Clone, Message)]
pub(crate) struct LegacyCompaction {
    #[prost(string, tag = "1")]
    pub(crate) summary: String,
    #[prost(message, repeated, tag = "2")]
    pub(crate) recent: Vec<mirror_protocol::AgentMessage>,
    #[prost(message, optional, tag = "3")]
    pub(crate) usage: Option<mirror_protocol::Usage>,
}

pub(super) fn migrate(transaction: &rusqlite::Transaction<'_>) -> Result<(), StoreError> {
    // Migrate descendants before rewriting the ancestors they inherit.
    let sessions = transaction
        .prepare("SELECT id FROM react_session ORDER BY json_array_length(lineage) DESC")
        .map_err(persistence)?
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(persistence)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(persistence)?;
    let mut statement = transaction
        .prepare(&format!(
            "SELECT e.sequence, e.payload {VIEW} ORDER BY e.sequence"
        ))
        .map_err(persistence)?;
    for session in sessions {
        let rows = statement
            .query_map([&session], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(persistence)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(persistence)?;
        // Older histories can predate metadata and system-prompt records.
        let mut current = ContextWindowRecord::default();
        let mut inference = None;
        let mut compaction_model = String::new();
        for (sequence, payload) in rows {
            let mut entry = decode_entry(sequence, &payload)?;
            let legacy = if entry.payload.is_none() {
                LegacyFields::decode(payload.as_slice()).map_err(persistence)?
            } else {
                LegacyFields::default()
            };
            if let Some(metadata) = legacy.metadata {
                compaction_model = metadata
                    .compaction_model
                    .unwrap_or_else(|| metadata.model.clone());
                if inference.is_none() {
                    inference = metadata.inference;
                }
                current.metadata = Some(prost_types::Struct {
                    fields: [(
                        "model".to_string(),
                        prost_types::Value {
                            kind: Some(prost_types::value::Kind::StringValue(metadata.model)),
                        },
                    )]
                    .into(),
                });
                transaction
                    .execute(
                        "DELETE FROM react_entry WHERE session_id = ? AND sequence = ?",
                        params![session, sequence],
                    )
                    .map_err(persistence)?;
                continue;
            }
            if let Some(chat_record::Message::System(system)) = entry.chat_message() {
                current.system_prompt.clone_from(&system.content);
                current.compaction = None;
                entry.payload = Some(agent_message::Payload::ContextWindow(current.clone()));
            } else if let Some(mut compaction) = legacy.compaction {
                let window = &mut current;
                // Some older compactors carried the system prompt inside recent.
                compaction.recent.retain(|recent| {
                    if let Some(chat_record::Message::System(system)) = recent.chat_message() {
                        window.system_prompt.clone_from(&system.content);
                        false
                    } else {
                        true
                    }
                });
                if inference.is_none() {
                    inference.clone_from(&entry.inference);
                }
                window.compaction = Some(Compaction {
                    model: compaction_model.clone(),
                    inference: entry.inference.take(),
                    recent: compaction.recent,
                    summary: (!compaction.summary.is_empty()).then_some(compaction.summary),
                    usage: compaction.usage,
                });
                entry.payload = Some(agent_message::Payload::ContextWindow(window.clone()));
            } else if entry.inference.is_some() {
                inference.clone_from(&entry.inference);
            }
            if let Some(agent_message::Payload::ContextWindow(window)) = &mut entry.payload {
                entry.inference.clone_from(&inference);
                let hash = digest(&SHA256, window.system_prompt.as_bytes());
                transaction.execute("INSERT INTO react_system_prompt(hash, content) VALUES (?, ?) ON CONFLICT(hash) DO NOTHING", params![hash.as_ref(), window.system_prompt]).map_err(persistence)?;
                window.system_prompt.clear();
                transaction.execute("UPDATE react_entry SET payload = ?, system_prompt_hash = ? WHERE session_id = ? AND sequence = ?", params![entry.encode_to_vec(), hash.as_ref(), session, sequence]).map_err(persistence)?;
            }
        }
    }
    Ok(())
}
