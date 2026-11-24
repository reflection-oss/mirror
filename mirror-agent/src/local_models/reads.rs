//! Operations that only read, served beside the actor so they never wait for its long ones.

use std::path::Path;
use std::sync::Arc;

use mirror_protocol::local_models::{
    ExecuteResponse, LocalModelCandidate, LocalModelDownload, LocalModelEntry, LocalModelInfo,
};
use tokio::sync::{watch, OnceCell};

use super::capabilities::device_memory;
use super::host_state::{Access, HostState, ServerRef};
use super::{Catalog, LocalModelError};

#[derive(Clone, Debug)]
pub struct Reads {
    catalog: Arc<Catalog>,
    state: watch::Receiver<HostState>,
    /// Device memory as llama.cpp reports it, asked once; `None` when it did not say.
    device_memory: Arc<OnceCell<Option<u64>>>,
}

impl Reads {
    pub fn new(catalog: Arc<Catalog>, state: watch::Receiver<HostState>) -> Self {
        Self {
            catalog,
            state,
            device_memory: Arc::new(OnceCell::new()),
        }
    }

    pub fn state(&self) -> &watch::Receiver<HostState> {
        &self.state
    }

    pub fn status(&self, reference: ServerRef, access: Access) -> ExecuteResponse {
        self.state.borrow().status(reference, access)
    }

    pub async fn inventory(
        &self,
        reference: ServerRef<'_>,
    ) -> Result<Vec<LocalModelEntry>, LocalModelError> {
        let server = self.state.borrow().server(reference)?;
        Ok(server.inventory().await?)
    }

    pub async fn info(
        &self,
        reference: ServerRef<'_>,
        model: &str,
    ) -> Result<LocalModelInfo, LocalModelError> {
        let server = self.state.borrow().server(reference)?;
        Ok(server.info(model).await?)
    }

    /// Search the catalog, marking what the server `reference` names already holds.
    pub async fn search(
        &self,
        query: &str,
        reference: ServerRef<'_>,
    ) -> Result<Vec<LocalModelCandidate>, LocalModelError> {
        let (found, held) = tokio::join!(self.catalog.search(query), self.held(reference));
        let mut found = found?;
        mark_downloaded(&mut found, &held);
        Ok(found)
    }

    /// A repository's quantizations, marking what the server `reference` names already holds.
    pub async fn variants(
        &self,
        repository: &str,
        reference: ServerRef<'_>,
    ) -> Result<Vec<LocalModelCandidate>, LocalModelError> {
        let memory = self.device_memory().await;
        let (found, held) = tokio::join!(
            self.catalog.variants(repository, memory),
            self.held(reference)
        );
        let mut found = found?;
        mark_downloaded(&mut found, &held);
        Ok(found)
    }

    pub async fn review(&self, model: &str) -> Result<LocalModelDownload, LocalModelError> {
        Ok(self.catalog.inspect(model).await?)
    }

    /// The server's models; none when it cannot say, so the catalog still works without it.
    async fn held(&self, reference: ServerRef<'_>) -> Vec<LocalModelEntry> {
        self.inventory(reference).await.unwrap_or_default()
    }

    async fn device_memory(&self) -> Option<u64> {
        *self
            .device_memory
            .get_or_init(|| async {
                let executable = self.state.borrow().state.executable.clone();
                if executable.is_empty() {
                    return None;
                }
                device_memory(Path::new(&executable)).await
            })
            .await
    }
}

/// Mark what the server holds of each candidate, by the inventory's entries.
///
/// A repository counts its quantizations held in full, and a quantization counts itself; an
/// entry still downloading or left `incomplete` is a partial download.
fn mark_downloaded(candidates: &mut [LocalModelCandidate], inventory: &[LocalModelEntry]) {
    for candidate in candidates {
        let held = inventory.iter().filter(|entry| {
            entry.name == candidate.name
                || entry
                    .name
                    .split_once(':')
                    .is_some_and(|(repository, _)| repository == candidate.name)
        });
        for entry in held {
            match entry.status.as_str() {
                "unloaded" | "loading" | "loaded" | "sleeping" => candidate.downloaded += 1,
                "downloading" | "incomplete" => candidate.partial = true,
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn candidates_count_what_the_server_holds_of_them() {
        let entry = |name: &str, status: &str| LocalModelEntry {
            name: name.into(),
            status: status.into(),
            ..Default::default()
        };
        let candidate = |name: &str| LocalModelCandidate {
            name: name.into(),
            ..Default::default()
        };
        let inventory = [
            entry("org/m:Q4", "unloaded"),
            entry("org/m:Q8", "loaded"),
            entry("org/m:Q2", "incomplete"),
            entry("org/m-other:Q4", "unloaded"),
        ];
        let mut found = [
            candidate("org/m"),
            candidate("org/m:Q4"),
            candidate("org/m:Q2"),
        ];

        mark_downloaded(&mut found, &inventory);

        let marks: Vec<_> = found
            .iter()
            .map(|item| (item.downloaded, item.partial))
            .collect();
        assert_eq!(marks, [(2, true), (1, false), (0, true)]);
    }
}
