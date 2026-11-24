//! Background downloads, which belong to the daemon rather than to the request that began them.

use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::OptionFuture;
use futures_util::{Stream, StreamExt};
use mirror_protocol::local_models::{
    LocalModelDownload, LocalModelTransfer, LocalModelTransferPhase,
};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;

use super::catalog::url;
use super::host_state::HostState;
use super::{Catalog, CatalogError};
use crate::providers::{router_events, LlamaCpp, LlmError, RouterEvent};

/// The running downloads and the recently ended ones in the published state.
#[derive(Debug)]
pub struct Downloads {
    tasks: JoinSet<()>,
    /// Running downloads by model name; a cancel request is sent once, so its sender is spent.
    running: HashMap<String, Option<oneshot::Sender<()>>>,
    /// When each ended transfer leaves the published list.
    expiry: HashMap<String, tokio::time::Instant>,
    updates: mpsc::Sender<TransferUpdate>,
    reports: mpsc::Receiver<TransferUpdate>,
}

/// How a transfer task reports back to the actor that owns the state.
#[derive(Debug)]
pub enum TransferUpdate {
    Progress {
        name: String,
        done: u64,
    },
    Ended {
        name: String,
        phase: LocalModelTransferPhase,
        error: String,
    },
    /// An ended transfer has been listed long enough.
    Expired {
        name: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("{0} is already downloading")]
    AlreadyDownloading(String),
    #[error("{0} is not downloading")]
    NotDownloading(String),
    #[error("{0} is already active")]
    AlreadyActive(String),
    #[error("The repository changed; review its files, size and license again")]
    RepositoryChanged,
    #[error("llama.cpp reported a failed download of {0}; review and retry to reuse partial data")]
    RouterFailed(String),
    #[error("Download of {0} ended without confirmation")]
    Unconfirmed(String),
    #[error("The files of {0} changed on Hugging Face during the download; review it again")]
    FilesChanged(String),
    #[error("Router has not confirmed a complete, unloaded cache entry for {0}")]
    NotCached(String),
    #[error("Could not find the files of {0} at the reviewed revision in the model cache")]
    FilesNotFound(String),
    #[error("{} has {found} bytes, not the reviewed {expected}", .path.display())]
    SizeMismatch {
        path: PathBuf,
        expected: u64,
        found: u64,
    },
    #[error("{} is not the reviewed file", .0.display())]
    NotReviewed(PathBuf),
    #[error("Cancelling {0} could not be confirmed; the router may still be downloading")]
    CancelUnconfirmed(String),
    #[error("Invalid download progress")]
    InvalidProgress,
    #[error("Router resolved different files or a different revision; review again")]
    UnreviewedSource,
    #[error("Invalid download byte counts")]
    InvalidCounts,
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error(transparent)]
    Server(#[from] LlmError),
}

/// How a transfer ended without failing.
#[derive(Debug, PartialEq, Eq)]
enum PullOutcome {
    Finished,
    /// Cancelling was requested; partial data is kept.
    Cancelled,
}

/// How long a transfer that ended stays listed, so clients can show how it ended.
const ENDED_VISIBLE: Duration = Duration::from_secs(15);

/// The router reports often; clients only need a steady heartbeat.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(500);

/// How long the router gets to confirm it stopped a transfer.
const CANCEL_TIMEOUT: Duration = Duration::from_secs(15);

impl Default for Downloads {
    fn default() -> Self {
        let (updates, reports) = mpsc::channel(16);
        Self {
            tasks: JoinSet::new(),
            running: HashMap::new(),
            expiry: HashMap::new(),
            updates,
            reports,
        }
    }
}

impl Downloads {
    /// The next report from a transfer task, or the next ended transfer to unlist.
    pub async fn next_update(&mut self) -> TransferUpdate {
        let next = self
            .expiry
            .iter()
            .min_by_key(|(_, due)| **due)
            .map(|(name, due)| (name.clone(), *due));
        let expired = OptionFuture::from(next.map(|(name, due)| async move {
            tokio::time::sleep_until(due).await;
            name
        }));
        tokio::select! {
            Some(update) = self.reports.recv() => update,
            Some(name) = expired => TransferUpdate::Expired { name },
            // `self.updates` keeps the channel open for as long as `self` exists.
            else => std::future::pending().await,
        }
    }

    /// Download `review` into `server`, which `server_name` names as the request did.
    ///
    /// `cache` is the server's model cache when Mirror can read it to check the downloaded
    /// files: the managed server's; a server the user runs, perhaps in a container, keeps its
    /// files elsewhere.
    pub fn begin(
        &mut self,
        catalog: Arc<Catalog>,
        server: LlamaCpp,
        server_name: String,
        cache: Option<PathBuf>,
        review: LocalModelDownload,
        state: &watch::Sender<HostState>,
    ) -> Result<(), DownloadError> {
        let cancel = self.register(&review, server_name, state)?;
        while self.tasks.try_join_next().is_some() {}
        self.tasks.spawn(transfer(
            catalog,
            server,
            review,
            cache,
            cancel,
            self.updates.clone(),
        ));
        Ok(())
    }

    fn register(
        &mut self,
        review: &LocalModelDownload,
        server: String,
        state: &watch::Sender<HostState>,
    ) -> Result<oneshot::Receiver<()>, DownloadError> {
        if self.running.contains_key(&review.name) {
            return Err(DownloadError::AlreadyDownloading(review.name.clone()));
        }
        self.expiry.remove(&review.name);
        state.send_modify(|host| {
            let transfers = &mut host.state.transfers;
            transfers.retain(|transfer| transfer.name != review.name);
            transfers.push(LocalModelTransfer {
                name: review.name.clone(),
                phase: LocalModelTransferPhase::Downloading.into(),
                total_bytes: total_bytes(review),
                server,
                ..Default::default()
            });
            host.state.revision += 1;
        });
        let (cancel, cancelled) = oneshot::channel();
        self.running.insert(review.name.clone(), Some(cancel));
        Ok(cancelled)
    }

    /// Ask a running download to stop, keeping its partial data; the task reports once
    /// llama.cpp confirms it.
    pub fn cancel(&mut self, name: &str) -> Result<(), DownloadError> {
        match self.running.get_mut(name).map(Option::take) {
            Some(Some(cancel)) => {
                let _ = cancel.send(());
                Ok(())
            }
            Some(None) => Ok(()),
            None => Err(DownloadError::NotDownloading(name.into())),
        }
    }

    pub fn apply(&mut self, update: TransferUpdate, state: &watch::Sender<HostState>) {
        match &update {
            TransferUpdate::Ended { name, .. } => {
                self.running.remove(name);
                self.expiry
                    .insert(name.clone(), tokio::time::Instant::now() + ENDED_VISIBLE);
            }
            TransferUpdate::Expired { name } => {
                self.expiry.remove(name);
            }
            TransferUpdate::Progress { .. } => {}
        }
        state.send_modify(|host| {
            let transfers = &mut host.state.transfers;
            match update {
                TransferUpdate::Progress { name, done } => {
                    let downloading = i32::from(LocalModelTransferPhase::Downloading);
                    if let Some(transfer) = transfers
                        .iter_mut()
                        .find(|transfer| transfer.name == name && transfer.phase == downloading)
                    {
                        transfer.done_bytes = done;
                    }
                }
                TransferUpdate::Ended { name, phase, error } => {
                    if let Some(transfer) =
                        transfers.iter_mut().find(|transfer| transfer.name == name)
                    {
                        transfer.phase = phase.into();
                        transfer.error = error;
                        if phase == LocalModelTransferPhase::Finished {
                            transfer.done_bytes = transfer.total_bytes;
                        }
                    }
                }
                TransferUpdate::Expired { name } => {
                    transfers.retain(|transfer| transfer.name != name);
                }
            }
            host.state.revision += 1;
        });
    }
}

/// Download one reviewed model and report to the actor; the model is never loaded.
async fn transfer(
    catalog: Arc<Catalog>,
    server: LlamaCpp,
    review: LocalModelDownload,
    cache: Option<PathBuf>,
    cancel: oneshot::Receiver<()>,
    updates: mpsc::Sender<TransferUpdate>,
) {
    // A dropped sender is not a request to cancel.
    let cancel = async {
        if cancel.await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    let name = review.name.clone();
    let (progress, mut reports) = mpsc::channel(8);
    let pull = pull(
        &catalog,
        &server,
        &review,
        cache.as_deref(),
        cancel,
        &progress,
    );
    tokio::pin!(pull);
    let mut last_report = Instant::now();
    let result = loop {
        tokio::select! {
            result = &mut pull => break result,
            Some(done) = reports.recv() => {
                if last_report.elapsed() >= PROGRESS_INTERVAL {
                    last_report = Instant::now();
                    let _ = updates.send(TransferUpdate::Progress { name: name.clone(), done }).await;
                }
            }
        }
    };
    let (phase, error) = match result {
        Ok(PullOutcome::Finished) => (LocalModelTransferPhase::Finished, String::new()),
        Ok(PullOutcome::Cancelled) => (LocalModelTransferPhase::Cancelled, String::new()),
        Err(error) => (LocalModelTransferPhase::Failed, error.to_string()),
    };
    let _ = updates
        .send(TransferUpdate::Ended { name, phase, error })
        .await;
}

/// Download through the router, stopping it there too if the pull ends before the files are in.
///
/// A cancel before the router is asked to download ends the pull at once. The request that
/// starts the download always runs to its end, so the router has registered the transfer
/// before it is told to stop. Completion never loads or selects the model.
async fn pull(
    catalog: &Catalog,
    server: &LlamaCpp,
    reviewed: &LocalModelDownload,
    cache: Option<&Path>,
    cancel: impl Future<Output = ()>,
    progress: &mpsc::Sender<u64>,
) -> Result<PullOutcome, DownloadError> {
    tokio::pin!(cancel);
    let (sources, events) = tokio::select! {
        biased;
        () = &mut cancel => return Ok(PullOutcome::Cancelled),
        prepared = prepare(catalog, server, reviewed) => prepared?,
    };
    let outcome = match server.download(&reviewed.name).await {
        Ok(()) => tokio::select! {
            biased;
            () = &mut cancel => Ok(PullOutcome::Cancelled),
            followed = follow(events, &reviewed.name, &sources, progress) => {
                followed.map(|()| PullOutcome::Finished)
            }
        },
        Err(error) => Err(error.into()),
    };
    if !matches!(outcome, Ok(PullOutcome::Finished)) {
        let stopped = tokio::time::timeout(CANCEL_TIMEOUT, server.pause(&reviewed.name)).await;
        // A pull that failed keeps its own reason, such as the server having stopped.
        if !matches!(stopped, Ok(Ok(()))) && outcome.is_ok() {
            return Err(DownloadError::CancelUnconfirmed(reviewed.name.clone()));
        }
        return outcome;
    }
    let current = catalog.inspect(&reviewed.name).await?;
    confirm(server, reviewed, &current, cache).await?;
    Ok(PullOutcome::Finished)
}

/// Check that the reviewed files can be downloaded, then subscribe to the router's events.
///
/// Subscribing before the download starts means a cached model's immediate completion is
/// not missed.
async fn prepare(
    catalog: &Catalog,
    server: &LlamaCpp,
    reviewed: &LocalModelDownload,
) -> Result<(HashMap<String, u64>, impl Stream<Item = RouterEvent> + Send), DownloadError> {
    let name = &reviewed.name;
    let inventory = server.inventory().await?;
    if inventory.iter().any(|item| {
        item.name == *name && !matches!(item.status.as_str(), "incomplete" | "unloaded")
    }) {
        return Err(DownloadError::AlreadyActive(name.clone()));
    }
    if catalog.inspect(name).await? != *reviewed {
        return Err(DownloadError::RepositoryChanged);
    }
    let sources = expected_sources(reviewed)?;
    Ok((sources, router_events(server.events().await?)))
}

/// Report progress until the router says the download of `name` finished.
async fn follow(
    events: impl Stream<Item = RouterEvent>,
    name: &str,
    sources: &HashMap<String, u64>,
    progress: &mpsc::Sender<u64>,
) -> Result<(), DownloadError> {
    let mut events = std::pin::pin!(events);
    while let Some(event) = events.next().await {
        if event.model != name {
            continue;
        }
        match event.event.as_str() {
            "download_failed" => return Err(DownloadError::RouterFailed(name.into())),
            "download_finished" => return Ok(()),
            "download_progress" => {
                let _ = progress
                    .send(download_progress(&event.data, sources)?)
                    .await;
            }
            _ => {}
        }
    }
    Err(DownloadError::Unconfirmed(name.into()))
}

/// Confirm a finished download is the reviewed one, complete and unloaded in the cache.
async fn confirm(
    server: &LlamaCpp,
    reviewed: &LocalModelDownload,
    current: &LocalModelDownload,
    cache: Option<&Path>,
) -> Result<(), DownloadError> {
    let name = &reviewed.name;
    // Progress pins the reviewed revision, so later commits matter only if they changed the
    // reviewed files, as for a cached entry the router completed without transferring.
    if current.blobs != reviewed.blobs {
        return Err(DownloadError::FilesChanged(name.clone()));
    }
    let inventory = server.inventory().await?;
    if !inventory.iter().any(|item| {
        item.name == *name
            // A cached model reports `preset` once Mirror's presets.ini names it.
            && matches!(item.source.as_str(), "cache" | "preset")
            && item.status == "unloaded"
    }) {
        return Err(DownloadError::NotCached(name.clone()));
    }
    match cache {
        Some(cache) => check_files(cache, reviewed).await,
        None => Ok(()),
    }
}

/// Require every reviewed file to be in `cache` as the review named it, without hashing
/// gigabytes.
///
/// llama.cpp keeps a repository in Hugging Face's cache layout: the file at `path` in the
/// reviewed revision is `models--<org>--<repo>/snapshots/<revision>/<path>`, a link to
/// `blobs/<sha256>` that pins its digest. Where links cannot be made the file is moved there
/// instead, and only its size is checked.
async fn check_files(cache: &Path, reviewed: &LocalModelDownload) -> Result<(), DownloadError> {
    let repo = reviewed.name.split(':').next().unwrap_or_default();
    let snapshot = cache
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(&reviewed.revision);
    for blob in &reviewed.blobs {
        let path = snapshot.join(&blob.path);
        if let Ok(target) = tokio::fs::read_link(&path).await {
            if target.file_name() != Some(blob.digest.as_ref()) {
                return Err(DownloadError::NotReviewed(path));
            }
        }
        let found = tokio::fs::metadata(&path)
            .await
            .map_err(|_| DownloadError::FilesNotFound(reviewed.name.clone()))?
            .len();
        if found != blob.size {
            return Err(DownloadError::SizeMismatch {
                path,
                expected: blob.size,
                found,
            });
        }
    }
    Ok(())
}

/// Bytes across the reviewed files.
fn total_bytes(download: &LocalModelDownload) -> u64 {
    download
        .blobs
        .iter()
        .fold(0, |total, blob| total.saturating_add(blob.size))
}

/// The size of each reviewed file by the URL the router reports it under.
fn expected_sources(reviewed: &LocalModelDownload) -> Result<HashMap<String, u64>, CatalogError> {
    let repo = reviewed.name.split(':').next().unwrap_or_default();
    reviewed
        .blobs
        .iter()
        .map(|blob| {
            let source = url(&[repo, "resolve", &reviewed.revision, &blob.path])?;
            Ok((source.to_string(), blob.size))
        })
        .collect()
}

/// Bytes done across the reviewed files, refusing anything else the router resolved.
fn download_progress(data: &Value, sources: &HashMap<String, u64>) -> Result<u64, DownloadError> {
    let progress = data["progress"]
        .as_object()
        .ok_or(DownloadError::InvalidProgress)?;
    let mut done = 0u64;
    for (source, progress) in progress {
        let size = sources.get(source).ok_or(DownloadError::UnreviewedSource)?;
        let total = progress["total"].as_u64();
        let count = progress["done"].as_u64();
        match (total, count) {
            (Some(total), Some(count)) if (total == 0 || total == *size) && count <= *size => {
                done = done.saturating_add(count);
            }
            _ => return Err(DownloadError::InvalidCounts),
        }
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use mirror_protocol::local_models::{LocalModelBlob, LocalModelState};
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::tools::web_test_server::{TestResponse, TestServer};

    fn review(name: &str) -> LocalModelDownload {
        LocalModelDownload {
            name: name.into(),
            blobs: vec![LocalModelBlob {
                size: 10,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn state() -> (watch::Sender<HostState>, watch::Receiver<HostState>) {
        let (_, alive) = watch::channel(false);
        watch::channel(HostState {
            state: LocalModelState::default(),
            endpoint: String::new(),
            alive,
        })
    }

    fn ended(name: &str) -> TransferUpdate {
        TransferUpdate::Ended {
            name: name.into(),
            phase: LocalModelTransferPhase::Finished,
            error: String::new(),
        }
    }

    #[test]
    fn a_download_reports_progress_and_completes_with_every_byte() {
        let (sender, state) = state();
        let mut downloads = Downloads::default();
        downloads
            .register(&review("org/m:Q4"), "managed".into(), &sender)
            .expect("registered");

        downloads.apply(
            TransferUpdate::Progress {
                name: "org/m:Q4".into(),
                done: 4,
            },
            &sender,
        );
        assert_eq!(state.borrow().state.transfers[0].done_bytes, 4);
        downloads.apply(ended("org/m:Q4"), &sender);

        let transfer = state.borrow().state.transfers[0].clone();
        assert_eq!(transfer.phase, i32::from(LocalModelTransferPhase::Finished));
        assert_eq!(transfer.done_bytes, 10);
        assert_eq!(transfer.server, "managed");
    }

    #[tokio::test(start_paused = true)]
    async fn an_ended_download_leaves_the_list_after_a_while() {
        let (sender, state) = state();
        let mut downloads = Downloads::default();
        downloads
            .register(&review("org/m:Q4"), "managed".into(), &sender)
            .expect("registered");
        downloads.apply(ended("org/m:Q4"), &sender);
        let ended_at = tokio::time::Instant::now();
        let revision = state.borrow().state.revision;

        let update = downloads.next_update().await;

        assert_matches!(&update, TransferUpdate::Expired { name } if name == "org/m:Q4");
        assert_eq!(ended_at.elapsed(), ENDED_VISIBLE);
        downloads.apply(update, &sender);
        assert!(state.borrow().state.transfers.is_empty());
        assert!(
            state.borrow().state.revision > revision,
            "the removal is published"
        );
    }

    #[tokio::test]
    async fn only_a_running_download_can_be_cancelled_and_not_started_twice() {
        let (sender, _state) = state();
        let mut downloads = Downloads::default();
        let cancelled = downloads
            .register(&review("org/m:Q4"), "managed".into(), &sender)
            .expect("registered");

        assert_matches!(downloads.cancel("org/m:Q4"), Ok(()));
        assert_matches!(cancelled.await, Ok(()), "the task is told to cancel");
        assert_matches!(
            downloads.cancel("other/m:Q4"),
            Err(DownloadError::NotDownloading(_))
        );
        assert_matches!(
            downloads.register(&review("org/m:Q4"), "managed".into(), &sender),
            Err(DownloadError::AlreadyDownloading(_))
        );
    }

    #[test]
    fn download_progress_counts_only_the_reviewed_files_at_the_reviewed_revision() {
        let reviewed = LocalModelDownload {
            name: "org/m-GGUF:Q4".into(),
            revision: "abc".into(),
            blobs: vec![LocalModelBlob {
                path: "m-Q4.gguf".into(),
                size: 1000,
                ..Default::default()
            }],
            ..Default::default()
        };
        let sources = expected_sources(&reviewed).expect("sources");
        let reviewed_source = "https://huggingface.co/org/m-GGUF/resolve/abc/m-Q4.gguf";

        let other_revision = "https://huggingface.co/org/m-GGUF/resolve/other/m-Q4.gguf";
        let stray_file = "https://huggingface.co/org/m-GGUF/resolve/abc/stray.gguf";
        let progress = |source: &str| {
            let event = serde_json::json!({"progress": {source: {"done": 400, "total": 1000}}});
            download_progress(&event, &sources)
        };

        assert_matches!(progress(reviewed_source), Ok(400));
        assert_matches!(
            progress(other_revision),
            Err(DownloadError::UnreviewedSource)
        );
        assert_matches!(progress(stray_file), Err(DownloadError::UnreviewedSource));
    }

    /// A review of `org/m-GGUF:Q4` at revision `abc`, one file of `size` bytes.
    fn reviewed(digest: &str, size: u64) -> LocalModelDownload {
        LocalModelDownload {
            name: "org/m-GGUF:Q4".into(),
            revision: "abc".into(),
            blobs: vec![LocalModelBlob {
                path: "Q4/m-Q4.gguf".into(),
                digest: digest.into(),
                size,
            }],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn a_linked_cache_file_must_be_the_reviewed_revision_and_digest() {
        let cache = tempfile::tempdir().expect("cache");
        let repo = cache.path().join("models--org--m-GGUF");
        let digest = "a".repeat(64);
        std::fs::create_dir_all(repo.join("blobs")).expect("blobs");
        std::fs::write(repo.join("blobs").join(&digest), [0; 3]).expect("blob");
        std::fs::create_dir_all(repo.join("snapshots/abc/Q4")).expect("snapshot");
        std::os::unix::fs::symlink(
            format!("../../../blobs/{digest}"),
            repo.join("snapshots/abc/Q4/m-Q4.gguf"),
        )
        .expect("link");
        let other_revision = LocalModelDownload {
            revision: "def".into(),
            ..reviewed(&digest, 3)
        };

        assert_matches!(
            check_files(cache.path(), &reviewed(&digest, 3)).await,
            Ok(())
        );
        assert_matches!(
            check_files(cache.path(), &reviewed(&"b".repeat(64), 3)).await,
            Err(DownloadError::NotReviewed(_))
        );
        assert_matches!(
            check_files(cache.path(), &other_revision).await,
            Err(DownloadError::FilesNotFound(_))
        );
    }

    #[tokio::test]
    async fn a_moved_cache_file_must_have_its_reviewed_size() {
        let cache = tempfile::tempdir().expect("cache");
        let snapshot = cache.path().join("models--org--m-GGUF/snapshots/abc/Q4");
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::write(snapshot.join("m-Q4.gguf"), [0; 4]).expect("file");
        let digest = "a".repeat(64);

        assert_matches!(
            check_files(cache.path(), &reviewed(&digest, 4)).await,
            Ok(())
        );
        assert_matches!(
            check_files(cache.path(), &reviewed(&digest, 5)).await,
            Err(DownloadError::SizeMismatch {
                expected: 5,
                found: 4,
                ..
            })
        );
    }

    #[tokio::test]
    async fn only_the_managed_servers_files_are_checked_once_a_download_completes() {
        // llama.cpp b12110 lists an unloaded cache entry without its path.
        let cached = || {
            [
                TestResponse::json(200, serde_json::json!({"role": "router"})),
                TestResponse::json(
                    200,
                    serde_json::json!({"data": [{
                        "id": "org/m-GGUF:Q4",
                        "source": "cache",
                        "status": {"value": "unloaded"},
                    }]}),
                ),
            ]
        };
        let server = TestServer::start(cached().into_iter().chain(cached()).collect()).await;
        let router = LlamaCpp::new(&format!("{}/v1", server.base_url), None).expect("router");
        let cache = tempfile::tempdir().expect("cache");
        let reviewed = reviewed(&"a".repeat(64), 3);

        assert_matches!(confirm(&router, &reviewed, &reviewed, None).await, Ok(()));
        assert_matches!(
            confirm(&router, &reviewed, &reviewed, Some(cache.path())).await,
            Err(DownloadError::FilesNotFound(_))
        );
    }
}
