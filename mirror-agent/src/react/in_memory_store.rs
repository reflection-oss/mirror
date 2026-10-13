//! In-memory storage for imperative ReACT sessions and tests.

use std::collections::{HashMap, HashSet};

use crate::ForkPoint;
use crate::Lineage;
use crate::Store;
use crate::StoreError;
use futures_util::future::BoxFuture;
use mirror_protocol::AgentMessage;
use uuid::Uuid;

#[derive(Debug, Default)]
pub struct InMemoryStore {
    sessions: HashMap<String, Vec<AgentMessage>>,
    lineages: HashMap<String, Lineage<String>>,
}

impl InMemoryStore {
    /// Seed a disposable store without changing committed entry IDs, timestamps, or sequences.
    pub fn from_entries(session_id: String, entries: Vec<AgentMessage>) -> Self {
        Self {
            sessions: HashMap::from([(session_id, entries)]),
            ..Self::default()
        }
    }

    /// Fork before a user entry, or at the current head when no entry is supplied.
    pub fn fork(
        &mut self,
        source: &str,
        before_entry_id: Option<&str>,
        child: &str,
    ) -> Result<(), StoreError> {
        if !self.sessions.contains_key(source) {
            return Err(StoreError::SessionNotFound {
                session_id: source.to_string(),
            });
        }
        if self.sessions.contains_key(child) {
            return Err(StoreError::Persistence(format!(
                "session {child:?} already exists"
            )));
        }
        let entries = self.entries(source);
        let point = match before_entry_id {
            Some(id) => {
                let entry = entries.iter().find(|entry| entry.id == id).ok_or_else(|| {
                    StoreError::EntryNotFound {
                        session_id: source.to_string(),
                        entry_id: id.to_string(),
                    }
                })?;
                ForkPoint::parse(entry).ok_or_else(|| StoreError::InvalidForkTarget {
                    entry_id: id.to_string(),
                })?
            }
            None => ForkPoint::after(entries.last().and_then(|entry| entry.sequence).unwrap_or(0))
                .ok_or_else(|| StoreError::Persistence("fork sequence overflow".to_string()))?,
        };
        let lineage = match self.lineages.get(source) {
            Some(lineage) => lineage.clone(),
            None => Lineage::parse(source.to_string(), vec![source.to_string()], vec![i64::MAX])
                .map_err(|error| StoreError::Persistence(error.to_string()))?,
        };
        let lineage = lineage
            .fork(child.to_string(), point)
            .map_err(|error| StoreError::Persistence(error.to_string()))?;
        self.sessions.insert(child.to_string(), Vec::new());
        self.lineages.insert(child.to_string(), lineage);
        Ok(())
    }

    fn entries(&self, session_id: &str) -> Vec<&AgentMessage> {
        let Some(lineage) = self.lineages.get(session_id) else {
            return self
                .sessions
                .get(session_id)
                .into_iter()
                .flatten()
                .collect();
        };
        let (sessions, limits) = lineage.clone().into_columns();
        sessions
            .into_iter()
            .zip(limits)
            .flat_map(|(session, limit)| {
                self.sessions[&session].iter().filter(move |entry| {
                    entry
                        .sequence
                        .is_some_and(|sequence| sequence <= limit as u64)
                })
            })
            .collect()
    }
}

impl Store for InMemoryStore {
    fn load<'a>(
        &'a mut self,
        session_id: &'a str,
        full_history: bool,
    ) -> BoxFuture<'a, Result<Vec<AgentMessage>, StoreError>> {
        self.sessions.entry(session_id.to_string()).or_default();
        let session = self.entries(session_id);
        let visible = if full_history {
            session
        } else {
            crate::compaction::active_window(session)
        };
        let entries = visible.into_iter().cloned().collect();
        Box::pin(async move { Ok(entries) })
    }

    fn append<'a>(
        &'a mut self,
        session_id: &'a str,
        head: u64,
        entries: &'a mut [AgentMessage],
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        let result = (|| {
            let visible = self.entries(session_id);
            if visible.last().and_then(|entry| entry.sequence).unwrap_or(0) != head {
                return Err(StoreError::Stale);
            }
            let mut identities: HashSet<String> =
                visible.iter().map(|entry| entry.id.clone()).collect();
            for (offset, entry) in entries.iter_mut().enumerate() {
                if entry.id.is_empty() {
                    entry.id = Uuid::new_v4().simple().to_string();
                }
                if !identities.insert(entry.id.clone()) {
                    return Err(StoreError::MessageIdConflict {
                        session_id: session_id.to_string(),
                        entry_id: entry.id.clone(),
                    });
                }
                if entry.timestamp.is_none() {
                    entry.timestamp =
                        Some(prost_types::Timestamp::from(std::time::SystemTime::now()));
                }
                entry.sequence = Some(head.checked_add(offset as u64 + 1).ok_or_else(|| {
                    StoreError::Persistence("entry sequence overflow".to_string())
                })?);
            }
            self.sessions
                .entry(session_id.to_string())
                .or_default()
                .extend_from_slice(entries);
            Ok(())
        })();
        Box::pin(async move { result })
    }
}

#[cfg(test)]
#[path = "in_memory_store_test.rs"]
mod tests;
