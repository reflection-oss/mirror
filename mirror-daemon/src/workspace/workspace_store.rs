//! Atomic storage for workspace records; it neither manages nor authorizes execution.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use mirror_protocol::daemon::WorkspaceState;
use prost::Message;
use tonic::Status;
use uuid::Uuid;

use super::WorkspaceRecord;

/// The registry owns this store and admits all mutations.
#[derive(Debug)]
pub struct WorkspaceStore {
    path: PathBuf,
    records: BTreeMap<String, WorkspaceRecord>,
    ignored_directories: BTreeSet<String>,
    skipped_records: Vec<Vec<u8>>,
}

#[derive(Clone, PartialEq, Message)]
struct SavedWorkspaces {
    // Decode each envelope separately so one incompatible record cannot hide the others.
    #[prost(bytes = "vec", repeated, tag = "1")]
    records: Vec<Vec<u8>>,
    #[prost(string, repeated, tag = "2")]
    ignored_directories: Vec<String>,
    // Keep rejected bytes inert, including duplicates whose accepted identity is later deleted.
    #[prost(bytes = "vec", repeated, tag = "3")]
    skipped_records: Vec<Vec<u8>>,
}

const MAX_STORE_BYTES: usize = 16 * 1024 * 1024;

impl WorkspaceStore {
    pub async fn open(directory: &Path) -> io::Result<Self> {
        // Other Mirror builds own the unversioned catalog; never read or rewrite it.
        let path = directory.join("workspaces-v1.pb");
        tokio::task::spawn_blocking(move || load(path)).await?
    }

    pub fn records(&self) -> impl Iterator<Item = &WorkspaceRecord> {
        self.records.values()
    }

    pub fn get(&self, id: &str) -> Option<&WorkspaceRecord> {
        self.records.get(id)
    }

    pub fn ignores(&self, directory: &str) -> bool {
        self.ignored_directories.contains(directory)
    }

    /// Commit a batch without changing memory if validation or persistence fails.
    pub async fn save(
        &mut self,
        updates: impl IntoIterator<Item = WorkspaceRecord>,
    ) -> Result<(), Status> {
        let mut records = self.records.clone();
        for mut record in updates {
            validate(&record).map_err(|_| Status::invalid_argument("invalid workspace record"))?;
            if let Some(previous) = records.get(&record.workspace.id) {
                if previous.creation != record.creation
                    || previous
                        .sessions()
                        .any(|id| !record.sessions().any(|saved| saved == id))
                {
                    return Err(Status::failed_precondition(
                        "workspace identity is immutable",
                    ));
                }
            }
            if records.values().any(|saved| {
                saved.workspace.id != record.workspace.id
                    && saved
                        .sessions()
                        .any(|id| record.sessions().any(|candidate| candidate == id))
            }) {
                return Err(Status::already_exists(
                    "session already belongs to a workspace",
                ));
            }
            record.workspace.inference = None;
            record.workspace.approval_mode = None;
            records.insert(record.workspace.id.clone(), record);
        }
        self.persist(records, self.ignored_directories.clone())
            .await
    }

    /// Retain a deletion tombstone so discovery cannot recreate the record.
    pub async fn remove(&mut self, id: &str) -> Result<(), Status> {
        let record = self
            .get(id)
            .ok_or_else(|| Status::not_found("workspace not found"))?;
        let mut ignored = self.ignored_directories.clone();
        if !record.workspace.directory.is_empty() {
            ignored.insert(record.workspace.directory.clone());
        }
        let mut records = self.records.clone();
        records.remove(id);
        self.persist(records, ignored).await
    }

    async fn persist(
        &mut self,
        records: BTreeMap<String, WorkspaceRecord>,
        ignored_directories: BTreeSet<String>,
    ) -> Result<(), Status> {
        let saved = SavedWorkspaces {
            records: records.values().map(Message::encode_to_vec).collect(),
            ignored_directories: ignored_directories.iter().cloned().collect(),
            skipped_records: self.skipped_records.clone(),
        };
        if saved.records.len() + saved.skipped_records.len() > 10000
            || saved.encoded_len() > MAX_STORE_BYTES
        {
            return Err(Status::resource_exhausted("workspace store is full"));
        }
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || save(&path, saved))
            .await
            .map_err(|_| Status::internal("workspace persistence task failed"))?
            .map_err(|_| Status::internal("could not persist workspace metadata"))?;
        self.records = records;
        self.ignored_directories = ignored_directories;
        Ok(())
    }
}

fn load(path: PathBuf) -> io::Result<WorkspaceStore> {
    let mut store = WorkspaceStore {
        path,
        records: BTreeMap::new(),
        ignored_directories: BTreeSet::new(),
        skipped_records: Vec::new(),
    };
    let file = match std::fs::File::open(&store.path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(store),
        Err(error) => return Err(error),
    };
    let mut bytes = Vec::new();
    file.take((MAX_STORE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_STORE_BYTES {
        return Err(io::Error::other("workspace store exceeds its size bound"));
    }
    let saved = SavedWorkspaces::decode(bytes.as_slice()).map_err(io::Error::other)?;
    if saved.records.len() + saved.skipped_records.len() > 10000 {
        return Err(io::Error::other("workspace store is full"));
    }
    store.ignored_directories = saved.ignored_directories.into_iter().collect();
    store.skipped_records = saved.skipped_records;
    let mut sessions = BTreeSet::new();
    for bytes in saved.records {
        let record = WorkspaceRecord::decode(bytes.as_slice())
            .ok()
            .filter(|record| {
                validate(record).is_ok()
                    && !store.records.contains_key(&record.workspace.id)
                    && !record.sessions().any(|id| sessions.contains(id))
            });
        let Some(mut record) = record else {
            store.skipped_records.push(bytes);
            continue;
        };
        record.workspace.inference = None;
        record.workspace.approval_mode = None;
        if record.workspace.state == WorkspaceState::Provisioning as i32 {
            record.workspace.state = WorkspaceState::Failed as i32;
            record.workspace.error = Some("provisioning interrupted; no automatic replay".into());
        }
        sessions.extend(record.sessions().map(str::to_owned));
        store.records.insert(record.workspace.id.clone(), record);
    }
    if !store.skipped_records.is_empty() {
        eprintln!(
            "workspace catalog: {} invalid records skipped; original bytes retained",
            store.skipped_records.len()
        );
    }
    Ok(store)
}

fn validate(record: &WorkspaceRecord) -> io::Result<()> {
    let workspace = &record.workspace;
    if Uuid::parse_str(&workspace.id).is_err()
        || record.sessions().any(|id| {
            id.is_empty()
                || !id
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        })
        || record.sessions().collect::<BTreeSet<_>>().len()
            != 1 + workspace.previous_session_ids.len()
        || workspace.id != record.creation.workspace_id
        || (!workspace.directory.is_empty() && !Path::new(&workspace.directory).is_absolute())
    {
        return Err(io::Error::other("invalid workspace identity or directory"));
    }
    Ok(())
}

fn save(path: &Path, workspaces: SavedWorkspaces) -> io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("store needs a parent"))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&workspaces.encode_to_vec())?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| error.error)?;
    std::fs::File::open(parent)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use mirror_protocol::daemon::{CreateWorkspaceRequest, Workspace};
    use pretty_assertions::{assert_eq, assert_ne};

    fn workspace() -> WorkspaceRecord {
        let id = Uuid::new_v4().to_string();
        WorkspaceRecord {
            workspace: Workspace {
                id: id.clone(),
                session_id: id.clone(),
                directory: "/fixture".into(),
                state: WorkspaceState::Available as i32,
                ..Default::default()
            },
            creation: CreateWorkspaceRequest {
                workspace_id: id,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn switch_session_retains_bindings_across_restart_and_resume() {
        let directory = tempfile::tempdir().expect("fixture");
        let original = workspace();
        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        store.save([original.clone()]).await.expect("create");
        let fresh = original
            .switch_session(&mirror_protocol::daemon::SwitchWorkspaceSessionRequest {
                workspace_id: original.workspace.id.clone(),
                expected_session_id: original.workspace.session_id.clone(),
                session_id: None,
            })
            .expect("fresh binding");
        assert_ne!(fresh.workspace.session_id, original.workspace.session_id);
        assert_eq!(fresh.workspace.id, original.workspace.id);
        assert_eq!(fresh.workspace.directory, original.workspace.directory);
        assert_eq!(
            fresh.workspace.previous_session_ids,
            vec![original.workspace.session_id.clone()]
        );
        store.save([fresh.clone()]).await.expect("switch");
        let mut reopened = WorkspaceStore::open(directory.path())
            .await
            .expect("restart");
        let resumed = reopened
            .get(&original.workspace.id)
            .expect("same workspace")
            .switch_session(&mirror_protocol::daemon::SwitchWorkspaceSessionRequest {
                workspace_id: original.workspace.id.clone(),
                expected_session_id: fresh.workspace.session_id.clone(),
                session_id: Some(original.workspace.session_id.clone()),
            })
            .expect("resume");
        reopened
            .save([resumed.clone()])
            .await
            .expect("persist resume");
        assert_eq!(reopened.records().count(), 1);
        assert_eq!(resumed.workspace.session_id, original.workspace.session_id);
        assert_eq!(
            resumed.workspace.previous_session_ids,
            vec![fresh.workspace.session_id]
        );
        assert_eq!(resumed.creation, original.creation);
    }

    #[tokio::test]
    async fn save_rejects_claiming_or_forgetting_a_retained_binding() {
        let directory = tempfile::tempdir().expect("fixture");
        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        let mut original = workspace();
        original
            .workspace
            .previous_session_ids
            .push("retained".into());
        store.save([original.clone()]).await.expect("create");
        let mut other = workspace();
        other.workspace.session_id = "retained".into();
        assert_matches!(store.save([other]).await, Err(error) if error.code() == tonic::Code::AlreadyExists);
        original.workspace.previous_session_ids.clear();
        assert_matches!(store.save([original]).await, Err(error) if error.code() == tonic::Code::FailedPrecondition);
        assert_eq!(store.records().count(), 1);
    }

    #[test]
    fn switch_session_rejects_stale_or_foreign_bindings() {
        let original = workspace();
        let request = mirror_protocol::daemon::SwitchWorkspaceSessionRequest {
            workspace_id: original.workspace.id.clone(),
            expected_session_id: "stale".into(),
            session_id: None,
        };
        assert_matches!(original.switch_session(&request), Err(error) if error.code() == tonic::Code::Aborted);
        let request = mirror_protocol::daemon::SwitchWorkspaceSessionRequest {
            expected_session_id: original.workspace.session_id.clone(),
            session_id: Some("foreign".into()),
            ..request
        };
        assert_matches!(original.switch_session(&request), Err(error) if error.code() == tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn open_skips_invalid_records_and_preserves_them_across_writes() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("workspaces-v1.pb");
        let mut valid = workspace();
        let mut malformed = valid.encode_to_vec();
        malformed.extend_from_slice(&[0x12, 0x01, 0xff]); // Malformed creation payload.
        let mut invalid = workspace();
        invalid.workspace.directory = "relative".into();
        let skipped = vec![malformed, invalid.encode_to_vec()];
        let bytes = SavedWorkspaces {
            records: vec![
                skipped[0].clone(),
                valid.encode_to_vec(),
                skipped[1].clone(),
            ],
            ignored_directories: vec!["/forgotten".into()],
            ..Default::default()
        }
        .encode_to_vec();
        std::fs::write(&path, &bytes).expect("mixed catalog");

        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        assert_eq!(store.records().collect::<Vec<_>>(), vec![&valid]);
        assert_eq!(std::fs::read(&path).expect("unchanged catalog"), bytes);
        valid.workspace.name = "renamed".into();
        store
            .save([valid.clone()])
            .await
            .expect("update valid record");

        let saved =
            SavedWorkspaces::decode(std::fs::read(&path).expect("saved catalog").as_slice())
                .expect("decode envelopes");
        assert_eq!(saved.skipped_records, skipped);
        let reopened = WorkspaceStore::open(directory.path())
            .await
            .expect("reopen");
        assert_eq!(reopened.records().collect::<Vec<_>>(), vec![&valid]);
        assert!(reopened.ignores("/forgotten"));
    }

    #[tokio::test]
    async fn remove_does_not_resurrect_skipped_duplicate_identities() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let valid = workspace();
        let mut duplicate_id = valid.clone();
        duplicate_id.workspace.session_id = "another-session".into();
        let mut duplicate_session = workspace();
        duplicate_session.workspace.session_id = valid.workspace.session_id.clone();
        let skipped = vec![
            duplicate_id.encode_to_vec(),
            duplicate_session.encode_to_vec(),
        ];
        let path = directory.path().join("workspaces-v1.pb");
        std::fs::write(
            &path,
            SavedWorkspaces {
                records: vec![
                    valid.encode_to_vec(),
                    skipped[0].clone(),
                    skipped[1].clone(),
                ],
                ..Default::default()
            }
            .encode_to_vec(),
        )
        .expect("duplicate catalog");

        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        assert_eq!(store.records().collect::<Vec<_>>(), vec![&valid]);
        store
            .remove(&valid.workspace.id)
            .await
            .expect("remove accepted record");
        let reopened = WorkspaceStore::open(directory.path())
            .await
            .expect("reopen");
        assert_eq!(reopened.records().count(), 0);
        assert!(reopened.ignores(&valid.workspace.directory));
        let saved =
            SavedWorkspaces::decode(std::fs::read(&path).expect("saved catalog").as_slice())
                .expect("decode envelopes");
        assert_eq!(saved.skipped_records, skipped);
    }

    #[tokio::test]
    async fn open_and_save_leave_the_unversioned_catalog_untouched() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("workspaces.pb");
        let bytes = b"incompatible catalog";
        std::fs::write(&path, bytes).expect("other build's catalog");
        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        assert_eq!(store.records().count(), 0);
        let record = workspace();
        store
            .save([record.clone()])
            .await
            .expect("save separate catalog");
        let reopened = WorkspaceStore::open(directory.path())
            .await
            .expect("reopen");
        assert_eq!(reopened.records().collect::<Vec<_>>(), vec![&record]);
        assert_eq!(std::fs::read(&path).expect("untouched catalog"), bytes);
    }

    #[tokio::test]
    async fn save_counts_skipped_records_toward_the_catalog_limit() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("workspaces-v1.pb");
        let bytes = SavedWorkspaces {
            skipped_records: vec![Vec::new(); 10000],
            ..Default::default()
        }
        .encode_to_vec();
        std::fs::write(&path, &bytes).expect("full catalog");
        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        assert_matches!(store.save([workspace()]).await, Err(error) if error.code() == tonic::Code::ResourceExhausted);
        assert_eq!(store.records().count(), 0);
        assert_eq!(std::fs::read(&path).expect("unchanged catalog"), bytes);
    }

    #[tokio::test]
    async fn save_counts_skipped_bytes_toward_the_size_limit() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("workspaces-v1.pb");
        let bytes = SavedWorkspaces {
            skipped_records: vec![vec![0xff; MAX_STORE_BYTES - 8]],
            ..Default::default()
        }
        .encode_to_vec();
        std::fs::write(&path, &bytes).expect("nearly full catalog");
        let mut store = WorkspaceStore::open(directory.path()).await.expect("open");
        assert_matches!(store.save([workspace()]).await, Err(error) if error.code() == tonic::Code::ResourceExhausted);
        assert_eq!(store.records().count(), 0);
        assert_eq!(std::fs::read(&path).expect("unchanged catalog"), bytes);
    }

    #[tokio::test]
    async fn open_rejects_truncated_outer_framing_without_rewriting() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("workspaces-v1.pb");
        let bytes = b"\x0a\x02\xff";
        std::fs::write(&path, bytes).expect("truncated catalog");
        assert_matches!(WorkspaceStore::open(directory.path()).await, Err(_));
        assert_eq!(std::fs::read(&path).expect("unchanged catalog"), bytes);
    }

    #[tokio::test]
    async fn open_rejects_an_oversized_catalog() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("workspaces-v1.pb");
        let file = std::fs::File::create(&path).expect("catalog");
        file.set_len((MAX_STORE_BYTES + 1) as u64)
            .expect("oversized catalog");
        assert_matches!(WorkspaceStore::open(directory.path()).await, Err(error) if error.to_string() == "workspace store exceeds its size bound");
        assert_eq!(
            file.metadata().expect("unchanged catalog").len(),
            (MAX_STORE_BYTES + 1) as u64
        );
    }
}
