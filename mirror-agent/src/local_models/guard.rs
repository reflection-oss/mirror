//! The guard process: the group leader that runs llama-server and reports on it.
//!
//! The daemon starts it with `local-models-guard`. Its own process group outlives a
//! llama-server that exits, so descendants can still be cleaned up.

use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use nix::sys::signal::{killpg, Signal};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::unix::pipe;
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::sync::mpsc;

use super::capabilities::{Capabilities, CapabilityError};
use super::executable::serve_prefix;
use super::settings::{load_preset, SettingsError};
use crate::private_directory;

/// Run llama-server until the daemon closes stdin or a signal arrives.
///
/// Every failure, including those before llama-server starts, goes to the daemon through the
/// status pipe, because the daemon discards this process's stderr.
pub async fn run_guard(arguments: &[String]) -> Result<(), GuardError> {
    let result = guard(arguments).await;
    if let Err(message) = &result {
        let mut output = tokio::io::stdout();
        let _ = output
            .write_all(format!("error: {message}\n").as_bytes())
            .await;
        let _ = output.flush().await;
    }
    result
}

#[derive(Debug, thiserror::Error)]
pub enum GuardError {
    #[error("Expected runtime directory, context, idle delay and executable")]
    Arguments,
    #[error("Guardian requires its own process group")]
    ProcessGroup,
    #[error(transparent)]
    Capabilities(#[from] CapabilityError),
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error("Cannot prepare {}: {source}", .path.display())]
    Prepare {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("Cannot open the log in {}: {source}", .directory.display())]
    Log {
        directory: PathBuf,
        source: std::io::Error,
    },
    #[error("Could not {action}: {source}")]
    Io {
        action: &'static str,
        source: std::io::Error,
    },
    #[error("Could not start {binary}: {source}")]
    Spawn {
        binary: String,
        source: std::io::Error,
    },
}

/// Flags without which the router cannot run or is not isolated.
const REQUIRED_FLAGS: [&str; 6] = [
    "--ctx-size",
    "--parallel",
    "--models-max",
    "--no-models-autoload",
    "--offline",
    "--cors-origins",
];

const LOG_FILE: &str = "llama-server.log";
const PREVIOUS_LOG_FILE: &str = "llama-server.previous.log";

/// The log is moved aside once it reaches this size, so recent output is always kept.
const LOG_LIMIT: usize = 2 * 1024 * 1024;

/// A longer line is read, matched and logged in pieces.
const LINE_LIMIT: u64 = 8192;

/// A `GuardError::Io` for a failed `action`.
fn io_error(action: &'static str) -> impl FnOnce(std::io::Error) -> GuardError {
    move |source| GuardError::Io { action, source }
}

async fn guard(arguments: &[String]) -> Result<(), GuardError> {
    let [root, context, sleep_idle, binary] = arguments else {
        return Err(GuardError::Arguments);
    };
    let root = Path::new(root);
    if nix::unistd::getpgrp() != nix::unistd::getpid() {
        return Err(GuardError::ProcessGroup);
    }
    let capabilities = Capabilities::probe(Path::new(binary)).await?;
    let port = free_port().await?;
    let preset = load_preset(root)?;
    let server_arguments =
        server_arguments(&capabilities, &port, context, sleep_idle, preset.as_deref())?;
    let log = EngineLog::open(root).await?;
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(io_error("watch for SIGTERM"))?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .map_err(io_error("watch for SIGINT"))?;
    // One pipe for both streams keeps the log in the order llama-server wrote it.
    let (reader, writer) = std::io::pipe().map_err(io_error("open the output pipe"))?;
    let mut child = Command::new(binary)
        .args(serve_prefix(Path::new(binary)))
        .args(&server_arguments)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(
            writer
                .try_clone()
                .map_err(io_error("share the output pipe"))?,
        )
        .stderr(writer)
        .spawn()
        .map_err(|source| GuardError::Spawn {
            binary: binary.clone(),
            source,
        })?;
    let reader = pipe::Receiver::from_file(std::fs::File::from(std::os::fd::OwnedFd::from(reader)))
        .map_err(io_error("read the output pipe"))?;
    let (ready, mut announcements) = mpsc::channel(1);
    let observer = tokio::spawn(observe_output(
        reader,
        format!("listening on http://127.0.0.1:{port}"),
        ready,
        log,
    ));
    let parent = parent_closed();
    tokio::pin!(parent);
    loop {
        tokio::select! {
            Some(()) = announcements.recv() => {
                let mut output = tokio::io::stdout();
                if output.write_all(format!("{port}\n").as_bytes()).await.is_err() || output.flush().await.is_err() { break; }
            }
            _ = child.wait() => break,
            _ = &mut parent => break,
            _ = terminate.recv() => break,
            _ = interrupt.recv() => break,
        }
    }
    // The guard leads this group, so the final SIGKILL ends the guard too.
    let group = nix::unistd::getpid();
    let _ = killpg(group, Signal::SIGTERM);
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    // The last output, often the reason for an exit, arrives until the pipe closes.
    let _ = tokio::time::timeout(Duration::from_secs(2), observer).await;
    let _ = killpg(group, Signal::SIGKILL);
    Ok(())
}

/// A port nothing listens on now, for llama-server to bind next.
///
/// Losing the race makes llama-server exit before it listens, which the daemon reports.
async fn free_port() -> Result<String, GuardError> {
    let reservation = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .map_err(io_error("reserve a loopback port"))?;
    let port = reservation
        .local_addr()
        .map_err(io_error("read the reserved port"))?
        .port();
    Ok(port.to_string())
}

/// Resolves when the daemon closes this process's stdin.
async fn parent_closed() {
    let mut input = tokio::io::stdin();
    let mut byte = [0];
    while matches!(input.read(&mut byte).await, Ok(1)) {}
}

/// Mirror's isolation and routing flags, then the user's, which win on repeated flags.
fn server_arguments(
    capabilities: &Capabilities,
    port: &str,
    context: &str,
    sleep_idle: &str,
    preset: Option<&Path>,
) -> Result<Vec<String>, CapabilityError> {
    capabilities.require(&REQUIRED_FLAGS)?;
    let mut arguments: Vec<String> = [
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--offline",
        "--no-models-autoload",
        "--models-max",
        "1",
        "--parallel",
        "1",
        "--cors-origins",
    ]
    .map(String::from)
    .into();
    arguments.push(format!("http://127.0.0.1:{port}"));
    // An explicit size disables llama.cpp's `--fit` for the context, so ask for one only
    // when the user chose it. Otherwise each model gets what fits this machine.
    if context == "0" {
        if capabilities.supports("--fit") {
            arguments.extend(["--fit".into(), "on".into()]);
        }
    } else {
        arguments.extend(["--ctx-size".into(), context.into()]);
    }
    // Hardening and noise reduction that older or newer builds may not list.
    for flag in ["--no-mmproj", "--no-webui", "--no-agent"] {
        if capabilities.supports(flag) {
            arguments.push(flag.into());
        }
    }
    // Command-line flags are inherited by model instances and win over presets. Idle
    // unload is opt-in, and a build that does not list the flag keeps models loaded.
    if sleep_idle != "0" && capabilities.supports("--sleep-idle-seconds") {
        arguments.extend(["--sleep-idle-seconds".into(), sleep_idle.into()]);
    }
    if capabilities.supports("--log-colors") {
        arguments.extend(["--log-colors".into(), "off".into()]);
    }
    if let Some(preset) = preset {
        capabilities.require(&["--models-preset"])?;
        arguments.extend([
            "--models-preset".into(),
            preset.to_string_lossy().into_owned(),
        ]);
    }
    Ok(arguments)
}

/// A private copy of llama-server's output, for diagnosing failed starts.
///
/// The previous run's log is kept beside the current one, because a crashed server is
/// restarted before anyone reads what it said.
struct EngineLog {
    directory: PathBuf,
    file: tokio::fs::File,
    written: usize,
}

impl EngineLog {
    async fn open(root: &Path) -> Result<Self, GuardError> {
        let directory = root.join("logs");
        private_directory(&directory).map_err(|source| GuardError::Prepare {
            path: directory.clone(),
            source,
        })?;
        Self::start_fresh(directory.clone())
            .await
            .map_err(|source| GuardError::Log { directory, source })
    }

    /// Move the current log aside and start an empty one.
    async fn start_fresh(directory: PathBuf) -> std::io::Result<Self> {
        match tokio::fs::rename(directory.join(LOG_FILE), directory.join(PREVIOUS_LOG_FILE)).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.join(LOG_FILE))
            .await?;
        Ok(Self {
            directory,
            file,
            written: 0,
        })
    }

    async fn append(&mut self, line: &[u8]) {
        if self.written + line.len() > LOG_LIMIT {
            // If the log cannot be rotated it keeps growing, which beats losing the output.
            if let Ok(fresh) = Self::start_fresh(self.directory.clone()).await {
                *self = fresh;
            }
        }
        // A line reaches the file before the next is read, so the last one before an exit is
        // never lost.
        if self.file.write_all(line).await.is_ok() && self.file.flush().await.is_ok() {
            self.written += line.len();
        }
    }
}

/// Log the output and announce the line saying the server listens; never forward it.
async fn observe_output(
    output: impl AsyncRead + Unpin,
    marker: String,
    ready: mpsc::Sender<()>,
    mut log: EngineLog,
) {
    let mut output = BufReader::new(output);
    let mut line = Vec::new();
    let mut announced = false;
    loop {
        line.clear();
        match (&mut output)
            .take(LINE_LIMIT)
            .read_until(b'\n', &mut line)
            .await
        {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        log.append(&line).await;
        if !announced && line.trim_ascii_end().ends_with(marker.as_bytes()) {
            announced = true;
            let _ = ready.send(()).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::local_models::capabilities::RECORDED_HELP;

    #[test]
    fn a_recorded_release_runs_isolated_and_fits_an_auto_context() {
        let recorded = Capabilities::from_help(RECORDED_HELP).expect("flags");
        let preset = Path::new("/synthetic/presets.ini");

        let arguments = server_arguments(&recorded, "9000", "0", "900", Some(preset));

        assert_eq!(
            arguments.expect("arguments").join(" "),
            "--host 127.0.0.1 --port 9000 --offline --no-models-autoload --models-max 1 \
            --parallel 1 --cors-origins http://127.0.0.1:9000 --fit on --no-mmproj --no-webui \
            --no-agent --sleep-idle-seconds 900 --log-colors off \
            --models-preset /synthetic/presets.ini"
        );
    }

    #[test]
    fn optional_flags_follow_the_installed_build() {
        let old = Capabilities::from_help(
            "--ctx-size --parallel --models-max --no-models-autoload --offline --cors-origins --[no-]webui",
        )
        .expect("flags");

        let arguments = server_arguments(&old, "9000", "8192", "900", None);

        assert_eq!(
            arguments.expect("arguments").join(" "),
            "--host 127.0.0.1 --port 9000 --offline --no-models-autoload --models-max 1 \
            --parallel 1 --cors-origins http://127.0.0.1:9000 --ctx-size 8192 --no-webui"
        );
    }

    #[test]
    fn a_build_without_router_mode_is_named_not_launched() {
        let old = Capabilities::from_help("--ctx-size --parallel --offline").expect("flags");

        assert_matches!(
            server_arguments(&old, "9000", "8192", "0", None),
            Err(CapabilityError::Missing(missing))
                if missing == ["--models-max", "--no-models-autoload", "--cors-origins"]
        );
    }

    #[tokio::test]
    async fn the_listening_line_is_announced_and_every_line_is_logged() {
        let root = tempfile::tempdir().expect("root");
        let log = EngineLog::open(root.path()).await.expect("log");
        let (ready, mut announcements) = mpsc::channel(1);
        let output = &b"loading\r\nmain: server is listening on http://127.0.0.1:9000\r\nfatal: out of memory"[..];

        observe_output(
            output,
            "listening on http://127.0.0.1:9000".into(),
            ready,
            log,
        )
        .await;

        assert_eq!(announcements.recv().await, Some(()));
        assert_eq!(announcements.recv().await, None, "announced once");
        let logged = std::fs::read_to_string(root.path().join("logs").join(LOG_FILE)).expect("log");
        assert!(
            logged.ends_with("fatal: out of memory"),
            "the last line has no newline: {logged}"
        );
    }

    #[tokio::test]
    async fn a_new_run_keeps_the_previous_runs_log() {
        let root = tempfile::tempdir().expect("root");
        let mut first = EngineLog::open(root.path()).await.expect("first");
        first.append(b"crashed\n").await;

        EngineLog::open(root.path()).await.expect("second");

        let kept = std::fs::read_to_string(root.path().join("logs").join(PREVIOUS_LOG_FILE))
            .expect("previous");
        assert_eq!(kept, "crashed\n");
    }
}
