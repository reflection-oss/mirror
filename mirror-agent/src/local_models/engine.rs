//! The engine as the daemon sees it: a credential-free guard process and its lifetime.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStdout, Command};
use tokio::sync::{oneshot, watch};
use tokio::task::JoinError;

use super::settings::{Settings, SettingsError};
use crate::private_directory;
use crate::tools::process_group::ProcessGroup;

#[derive(Debug)]
pub struct Engine {
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<()>,
    output: BufReader<ChildStdout>,
    log: PathBuf,
    /// The model cache llama-server downloads into, its `LLAMA_CACHE`.
    pub models: PathBuf,
    /// True while the guard runs; its sender is dropped once the guard has been reaped.
    pub alive: watch::Receiver<bool>,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error("Cannot prepare {}: {source}", .path.display())]
    Prepare {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Preparing the local model directories did not finish: {0}")]
    Interrupted(#[from] JoinError),
    #[error("Could not start the local model guardian {}: {source}", .worker.display())]
    Spawn {
        worker: PathBuf,
        source: std::io::Error,
    },
    #[error("Could not open the guardian status pipe")]
    NoStatusPipe,
    #[error("llama.cpp did not start within {}s", READY_TIMEOUT.as_secs())]
    ReadyTimedOut,
    #[error("Could not read llama.cpp startup status: {0}")]
    Status(std::io::Error),
    /// The guard's own report of why llama-server could not start.
    #[error("{0}")]
    Guard(String),
    #[error("llama-server exited before listening; see {} for its output", .log.display())]
    ExitedBeforeListening { log: PathBuf },
}

/// The `PATH` of every process Mirror starts for llama.cpp: system tools only.
pub const ENGINE_PATH: &str = "/usr/bin:/bin";

/// How long the guard gets to announce the port llama-server listens on.
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the guard gets to clean up after its stdin closes before its group is killed.
const STOP_TIMEOUT: Duration = Duration::from_secs(8);

impl Engine {
    /// Start the guard, which starts llama-server. Nothing here waits for either.
    pub async fn spawn(
        worker: &Path,
        root: &Path,
        context: u64,
        sleep_idle: u64,
        binary: &Path,
    ) -> Result<Self, EngineError> {
        let prepared = root.to_owned();
        let models = tokio::task::spawn_blocking(move || prepare(&prepared)).await??;
        let mut child = Command::new(worker)
            .arg("local-models-guard")
            .arg(root)
            .arg(context.to_string())
            .arg(sleep_idle.to_string())
            .arg(binary)
            .current_dir(root)
            .env_clear()
            .env("PATH", ENGINE_PATH)
            .env("HOME", root.join("home"))
            .env("XDG_CONFIG_HOME", root.join("home/config"))
            .env("XDG_CACHE_HOME", root.join("home/cache"))
            .env("LLAMA_CACHE", &models)
            .env("TMPDIR", root.join("tmp"))
            .env("LC_ALL", "C")
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|source| EngineError::Spawn {
                worker: worker.to_owned(),
                source,
            })?;
        let output = child.stdout.take().ok_or(EngineError::NoStatusPipe)?;
        let (stop, stopped) = oneshot::channel();
        let (alive, lifetime) = watch::channel(true);
        Ok(Self {
            stop,
            task: tokio::spawn(supervise(child, stopped, alive)),
            output: BufReader::new(output),
            log: root.join("logs/llama-server.log"),
            models,
            alive: lifetime,
        })
    }

    /// Wait for the endpoint llama-server listens on, or the reason it will not.
    pub async fn ready(&mut self) -> Result<String, EngineError> {
        let mut line = String::new();
        tokio::time::timeout(READY_TIMEOUT, self.output.read_line(&mut line))
            .await
            .map_err(|_| EngineError::ReadyTimedOut)?
            .map_err(EngineError::Status)?;
        if let Some(reason) = line.trim().strip_prefix("error: ") {
            return Err(EngineError::Guard(reason.into()));
        }
        let port: u16 = line
            .trim()
            .parse()
            .map_err(|_| EngineError::ExitedBeforeListening {
                log: self.log.clone(),
            })?;
        Ok(format!("http://127.0.0.1:{port}/v1"))
    }

    /// Stop the guard and wait until its process group is gone.
    pub async fn stop(self) {
        let _ = self.stop.send(());
        let _ = self.task.await;
    }
}

/// Create the guard's private directories and find the model cache; this blocks on the
/// filesystem.
fn prepare(root: &Path) -> Result<PathBuf, EngineError> {
    let models = Settings::load(root)?.models_dir()?;
    for directory in [
        root.to_owned(),
        root.join("home"),
        root.join("tmp"),
        root.join("logs"),
    ] {
        private_directory(&directory).map_err(|source| EngineError::Prepare {
            path: directory,
            source,
        })?;
    }
    Ok(models)
}

/// Own the guard until it exits or is told to stop, and never leave its group behind.
async fn supervise(mut child: Child, stop: oneshot::Receiver<()>, alive: watch::Sender<bool>) {
    // The guard treats EOF on stdin as its parent going away. `Child::wait` closes
    // `Child::stdin` before waiting, so hold the pipe here until an explicit stop.
    let stdin = child.stdin.take();
    let mut group = ProcessGroup::new(child);
    // Liveness is published apart from the actor, so status stays responsive during downloads.
    let stopped = tokio::select! {
        _ = group.child().wait() => false,
        _ = stop => true,
    };
    let _ = alive.send(false);
    if stopped {
        drop(stdin);
        let _ = tokio::time::timeout(STOP_TIMEOUT, group.child().wait()).await;
    }
    let _ = group.terminate().await;
}
