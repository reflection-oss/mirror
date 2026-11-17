#![forbid(unsafe_code)]

mod agent_options;
mod agent_process;
mod daemon_rpc;
mod display_updates;
mod execution_lease;
mod frontend_defaults;
mod history_search;
mod session;
mod session_projection;
mod unix_socket;
mod workspace;
mod workspace_registry;

pub use agent_options::*;
pub use agent_process::{AgentProcess, AgentStartError};
pub use daemon_rpc::DaemonRpc;
pub use display_updates::*;
pub use execution_lease::*;
pub use frontend_defaults::*;
pub use history_search::*;
pub use session::{AttachmentRecovery, Session};
pub use session_projection::{SessionProjection, MAX_FRAME_BYTES};
pub use unix_socket::{local_channel, local_incoming};
pub use workspace_registry::WorkspaceRegistry;

use std::fs::{self, DirBuilder, File, OpenOptions, TryLockError};
use std::io::{self, Read, Seek};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use clap::Parser;
use hyper_util::rt::TokioIo;
use mirror_protocol::agent_rpc_server::AgentRpcServer;
use mirror_protocol::daemon::daemon_rpc_client::DaemonRpcClient;
use mirror_protocol::daemon::daemon_rpc_server::DaemonRpcServer;
use mirror_protocol::daemon::{GetDaemonInfoRequest, GetDaemonInfoResponse, UpgradeDaemonRequest};
use mirror_protocol::mirror::mirror_rpc_server::MirrorRpcServer;
use nix::sys::signal::{kill, Signal};
use nix::sys::stat::{umask, Mode};
use nix::unistd::{geteuid, setsid, Pid};
use sha2::{Digest, Sha256};
use tokio::net::unix::UCred;
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, watch};
use tonic::transport::{Endpoint, Server};
use tower::service_fn;

/// Shared session supervisor; --ensure discovers or starts an independent daemon.
#[derive(Clone, Debug, Parser)]
struct Options {
    #[arg(long)]
    state_root: PathBuf,
    #[arg(long)]
    database: PathBuf,
    #[arg(long)]
    worker_home: PathBuf,
    #[arg(long, default_value = "/usr/bin:/bin")]
    worker_path: String,
    /// Discover or start, print the ready endpoint, then exit without owning its lifetime.
    #[arg(long, conflicts_with = "connect_only")]
    ensure: bool,
    /// Explicitly restart with this installation, interrupting workers even at the same version.
    #[arg(long, requires = "ensure")]
    replace: bool,
    /// Wait for a separately managed daemon; never start a standalone fallback.
    #[arg(long)]
    connect_only: bool,
    /// Report the running daemon without starting or replacing it.
    #[arg(long, conflicts_with_all = ["ensure", "connect_only", "stop"])]
    status: bool,
    /// Stop this daemon and its workers without deleting saved workspaces or history.
    #[arg(long, conflicts_with_all = ["ensure", "connect_only"])]
    stop: bool,
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..=3600))]
    wait_seconds: u64,
    #[arg(long, hide = true)]
    detached: bool,
    #[arg(long, default_value = env!("CARGO_PKG_VERSION"))]
    application_version: pep440_rs::Version,
}

type Upgrade = (
    UpgradeDaemonRequest,
    oneshot::Sender<Result<(), tonic::Status>>,
);

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Bind creates the socket before chmod; discovery must never see a public-mode endpoint.
    umask(Mode::from_bits_truncate(0o077));
    let options = Options::parse();
    if options.detached {
        setsid()?;
    }
    if !options.state_root.is_absolute() || !options.database.is_absolute() {
        anyhow::bail!("state root and database paths must be absolute");
    }
    if options.status || options.stop {
        return manage(&options).await;
    }
    if options.ensure || options.connect_only {
        let socket = ensure(&options).await?;
        println!("MIRROR_DAEMON_ADDRESS=unix:{}", socket.display());
        return Ok(());
    }
    let result = serve(&options).await;
    if result
        .as_ref()
        .err()
        .and_then(|error| error.downcast_ref::<io::Error>())
        .is_some_and(|error| error.kind() == io::ErrorKind::WouldBlock)
    {
        eprintln!("daemon.lock is held; connect or recover explicitly, never unlink it");
        std::process::exit(75);
    }
    if let Some(request) = result? {
        // The old socket and singleton lock have been released; replacement never adopts workers.
        let replacement = Options {
            application_version: request.application_version.parse()?,
            ..options
        };
        start_detached(&replacement, Path::new(&request.executable))?;
    }
    Ok(())
}

async fn manage(options: &Options) -> anyhow::Result<()> {
    let socket = match socket_path(&options.state_root) {
        Ok(socket) => socket,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("Mirror daemon stopped");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let Some((peer, info)) = probe_daemon(&socket).await? else {
        println!("Mirror daemon stopped");
        return Ok(());
    };
    if options.status {
        let info = info.ok_or_else(|| anyhow::anyhow!("Mirror daemon is unresponsive"))?;
        println!(
            "Mirror daemon running ({}, {})",
            info.application_version, info.instance_id
        );
        return Ok(());
    }
    // Explicit stop waits for cleanup; only explicit replacement escalates to SIGKILL.
    stop_daemon(&socket, peer, &mut None).await?;
    let wait = async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            match probe_daemon(&socket).await? {
                None => return Ok::<(), anyhow::Error>(()),
                Some((current, _)) if current == peer => {}
                Some(_) => {
                    anyhow::bail!("daemon changed while stopping; new daemon was not stopped")
                }
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(options.wait_seconds), wait)
        .await
        .map_err(|_| anyhow::anyhow!("daemon stop timed out; daemon.lock must not be removed"))??;
    println!("Mirror daemon stopped");
    Ok(())
}

async fn ensure(options: &Options) -> anyhow::Result<PathBuf> {
    let socket = socket_directory(&options.state_root)?.join("daemon.sock");
    let wait = async {
        let mut child: Option<(Child, File)> = None;
        let mut stopping: Option<(Pid, Instant)> = None;
        let mut replace = options.replace;
        loop {
            if let Some((peer, info)) = probe_daemon(&socket).await? {
                if stopping
                    .as_ref()
                    .is_some_and(|(pid, _)| peer.pid() != Some(pid.as_raw()))
                {
                    replace = false;
                }
                if replace {
                    stop_daemon(&socket, peer, &mut stopping).await?;
                } else if let Some(info) = info {
                    if !info.upgrading
                        && (!options.ensure || reuse_daemon(options, &socket, &info).await?)
                    {
                        return Ok(());
                    }
                } else if options.ensure {
                    anyhow::bail!(
                        "Mirror daemon is unresponsive; restart it explicitly, not during startup"
                    );
                }
            } else if options.ensure && child.is_none() {
                replace = false;
                child = Some(start_detached(options, &std::env::current_exe()?)?);
            }
            if let Some((candidate, stderr)) = &mut child {
                if let Some(status) = candidate.try_wait()? {
                    if status.code() != Some(75) {
                        stderr.rewind()?;
                        let mut diagnostics = Vec::new();
                        stderr.take(4096).read_to_end(&mut diagnostics)?;
                        anyhow::bail!(
                            "daemon candidate exited with {status}: {}",
                            String::from_utf8_lossy(&diagnostics).trim()
                        );
                    }
                    child = None;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(options.wait_seconds), wait)
        .await
        .map_err(|_| {
            anyhow::anyhow!("daemon readiness timed out; daemon.lock must not be removed")
        })??;
    Ok(socket)
}

async fn reuse_daemon(
    options: &Options,
    socket: &Path,
    info: &GetDaemonInfoResponse,
) -> anyhow::Result<bool> {
    let Ok(version) = info.application_version.parse::<pep440_rs::Version>() else {
        eprintln!("MIRROR_DAEMON_WARNING=Running daemon has an unknown version; keeping it. Use /daemon-restart to replace it explicitly.");
        return Ok(true);
    };
    if version >= options.application_version {
        return Ok(true);
    }
    let channel = local_channel(socket.to_owned()).await?;
    let result = DaemonRpcClient::new(channel)
        .upgrade_daemon(UpgradeDaemonRequest {
            expected_instance_id: info.instance_id.clone(),
            application_version: options.application_version.to_string(),
            executable: std::env::current_exe()?.to_string_lossy().into_owned(),
            worker: agent_executable()?.to_string_lossy().into_owned(),
        })
        .await;
    match result {
        Ok(_) => Ok(false),
        Err(error)
            if matches!(
                error.code(),
                tonic::Code::Aborted | tonic::Code::Unavailable
            ) =>
        {
            Ok(false)
        }
        Err(error)
            if matches!(
                error.code(),
                tonic::Code::FailedPrecondition
                    | tonic::Code::Unimplemented
                    | tonic::Code::ResourceExhausted
            ) =>
        {
            eprintln!("MIRROR_DAEMON_WARNING=Daemon update deferred (running {version}, client {}). Workers may still be alive, or this daemon cannot upgrade safely. Use /daemon-restart to interrupt them and replace it.", options.application_version);
            Ok(true)
        }
        Err(error) => Err(error.into()),
    }
}

async fn stop_daemon(
    socket: &Path,
    credentials: UCred,
    stopping: &mut Option<(Pid, Instant)>,
) -> io::Result<()> {
    let pid = credentials
        .pid()
        .filter(|pid| *pid > 1 && *pid as u32 != std::process::id())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "refusing to replace the launcher process",
            )
        })?;
    // Recheck the kernel peer, never signal a PID taken from saved metadata or an RPC.
    let current = match UnixStream::connect(socket).await {
        Ok(current) => current,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) =>
        {
            return Ok(())
        }
        Err(error) => return Err(error),
    };
    if current.peer_cred()? != credentials {
        return Ok(());
    }
    let pid = Pid::from_raw(pid);
    let signal = match stopping {
        Some((previous, started)) if *previous == pid => {
            if started.elapsed() < Duration::from_secs(5) {
                return Ok(());
            }
            Signal::SIGKILL
        }
        _ => {
            *stopping = Some((pid, Instant::now()));
            Signal::SIGTERM
        }
    };
    match kill(pid, signal) {
        Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn installation_id(worker: &Path) -> io::Result<String> {
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    for path in [std::env::current_exe()?, worker.to_owned()] {
        let mut file = File::open(path)?;
        digest.update(file.metadata()?.len().to_le_bytes());
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn socket_directory(state_root: &Path) -> io::Result<PathBuf> {
    private_directory(state_root)?;
    let socket = socket_path(state_root)?;
    let directory = socket
        .parent()
        .ok_or_else(|| io::Error::other("socket needs a parent"))?;
    private_directory(directory)?;
    Ok(directory.to_owned())
}

fn socket_path(state_root: &Path) -> io::Result<PathBuf> {
    check_private_directory(state_root)?;
    let state = state_root.canonicalize()?;
    // macOS permits 103 path bytes; workers use names no longer than daemon.sock.
    let directory = if state.join("daemon.sock").as_os_str().len() <= 103 {
        state
    } else {
        let hash = format!("{:x}", Sha256::digest(state.as_os_str().as_encoded_bytes()));
        // A state root has one endpoint even when independent clients have different TMPDIRs.
        Path::new("/tmp").join(format!("m-{}-{}", geteuid(), &hash[..16]))
    };
    Ok(directory.join("daemon.sock"))
}

async fn probe_daemon(path: &Path) -> io::Result<Option<(UCred, Option<GetDaemonInfoResponse>)>> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "daemon socket must be absolute",
        ));
    }
    match check_private_directory(
        path.parent()
            .ok_or_else(|| io::Error::other("socket needs a parent"))?,
    ) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    }
    socket_metadata(path)?;
    let stream = match UnixStream::connect(path).await {
        Ok(stream) => stream,
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) =>
        {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    let credentials = stream.peer_cred()?;
    if credentials.uid() != geteuid().as_raw() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "foreign daemon peer",
        ));
    }
    // The response must come from this kernel-identified peer, not a second connection.
    let handshake = async {
        let mut stream = Some(stream);
        let channel = Endpoint::from_static("http://localhost")
            .connect_with_connector(service_fn(move |_| {
                std::future::ready(stream.take().map(TokioIo::new).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::ConnectionAborted,
                        "daemon probe connection closed",
                    )
                }))
            }))
            .await
            .ok()?;
        DaemonRpcClient::new(channel)
            .get_daemon_info(GetDaemonInfoRequest {})
            .await
            .ok()
            .map(tonic::Response::into_inner)
    };
    let info = tokio::time::timeout(Duration::from_secs(2), handshake)
        .await
        .ok()
        .flatten();
    Ok(Some((credentials, info)))
}

fn start_detached(options: &Options, executable: &Path) -> io::Result<(Child, File)> {
    // An unlinked private file surfaces startup failures without a pipe keeping clients alive.
    let stderr = tempfile::tempfile_in(&options.state_root)?;
    let mut command = Command::new(executable);
    command
        .arg("--detached")
        .arg("--state-root")
        .arg(&options.state_root)
        .arg("--database")
        .arg(&options.database)
        .arg("--worker-home")
        .arg(options.worker_home.canonicalize()?)
        .arg("--worker-path")
        .arg(&options.worker_path)
        .arg("--application-version")
        .arg(options.application_version.to_string())
        .current_dir("/")
        .env_clear()
        .env("TMPDIR", std::env::temp_dir())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(stderr.try_clone()?);
    // The child calls setsid; neither this helper nor a UI owns a pipe lifeline to it.
    Ok((command.spawn()?, stderr))
}

async fn serve(options: &Options) -> anyhow::Result<Option<UpgradeDaemonRequest>> {
    #[derive(Debug)]
    struct SocketCleanup<'a> {
        path: &'a Path,
        device: u64,
        inode: u64,
    }

    impl Drop for SocketCleanup<'_> {
        fn drop(&mut self) {
            if fs::symlink_metadata(self.path)
                .is_ok_and(|metadata| metadata.dev() == self.device && metadata.ino() == self.inode)
            {
                let _ = fs::remove_file(self.path);
            }
        }
    }

    let directory = &options.state_root;
    let runtime_directory = socket_directory(directory)?;
    let socket = runtime_directory.join("daemon.sock");
    // The lock must outlive socket cleanup, including on startup failure.
    let _lock = lock_daemon(directory)?;
    remove_stale_socket(&socket).await?;
    let state_directory = directory.canonicalize()?;
    // A client upgrade or source-build cleanup must not remove the daemon's agent executable.
    let executables = tempfile::Builder::new()
        .prefix("bin-")
        .tempdir_in(&state_directory)?;
    let executable = executables.path().join("mirror-agent");
    let source = agent_executable()?;
    let bundle = source
        .parent()
        .ok_or_else(|| io::Error::other("agent bundle needs a parent"))?
        .to_owned();
    let destination = executable.clone();
    let installation_id = tokio::task::spawn_blocking(move || {
        std::fs::copy(source, &destination)?;
        installation_id(&destination)
    })
    .await??;
    let agent = AgentOptions {
        executable,
        bundle,
        database: options.database.clone(),
        home: options.worker_home.canonicalize()?,
        runtime_directory,
        state_directory,
        path: options.worker_path.clone(),
    };
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let database = options.database.clone();
    tokio::task::spawn_blocking(move || mirror_agent::local::prepare_history(&database)).await??;
    let (registry, supervisor) = WorkspaceRegistry::spawn(agent).await?;
    // Publish the endpoint only after initialization so other launchers cannot probe it early.
    let listener = UnixListener::bind(&socket)?;
    let metadata = fs::symlink_metadata(&socket)?;
    let _cleanup = SocketCleanup {
        path: &socket,
        device: metadata.dev(),
        inode: metadata.ino(),
    };
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    let info = GetDaemonInfoResponse {
        application_version: options.application_version.to_string(),
        instance_id: uuid::Uuid::new_v4().to_string(),
        upgrading: false,
        installation_id,
        protocol_revision: mirror_protocol::daemon::PROTOCOL_REVISION,
    };
    let (upgrades, requests) = mpsc::channel(1);
    let (upgrading, upgrade_status) = watch::channel(false);
    let rpc = DaemonRpc::new(
        registry.clone(),
        options.database.clone(),
        info.clone(),
        upgrades,
        upgrade_status,
    );
    let (stop, stopped) = tokio::sync::oneshot::channel();
    println!("MIRROR_DAEMON_ADDRESS=unix:{}", socket.display());
    let server = Server::builder()
        .add_service(
            AgentRpcServer::new(rpc.clone())
                .max_decoding_message_size(MAX_FRAME_BYTES)
                .max_encoding_message_size(4 * MAX_FRAME_BYTES),
        )
        .add_service(
            MirrorRpcServer::new(rpc.clone())
                .max_decoding_message_size(MAX_FRAME_BYTES)
                .max_encoding_message_size(4 * MAX_FRAME_BYTES),
        )
        .add_service(
            DaemonRpcServer::new(rpc)
                .max_decoding_message_size(MAX_FRAME_BYTES)
                .max_encoding_message_size(4 * MAX_FRAME_BYTES),
        )
        .serve_with_incoming_shutdown(local_incoming(listener), async {
            let _ = stopped.await;
        });
    tokio::pin!(server);
    let mut replacement = None;
    let result = tokio::select! {
        result = &mut server => Some(result),
        request = upgrade_daemon(requests, &registry, upgrading) => {
            replacement = request;
            None
        }
        _ = terminate.recv() => None,
        _ = interrupt.recv() => None,
    };
    registry.shutdown();
    supervisor.await?;
    let _ = stop.send(());
    if let Some(result) = result {
        result?;
    } else {
        // An unresponsive UI must not hold shutdown open indefinitely.
        if let Ok(result) = tokio::time::timeout(Duration::from_secs(3), &mut server).await {
            result?;
        }
    }
    Ok(replacement)
}

async fn upgrade_daemon(
    mut requests: mpsc::Receiver<Upgrade>,
    registry: &WorkspaceRegistry,
    upgrading: watch::Sender<bool>,
) -> Option<UpgradeDaemonRequest> {
    while let Some((request, reply)) = requests.recv().await {
        let checked = validate_upgrade(&request);
        let result = match checked {
            Ok(()) => registry.stop_if_idle().await,
            Err(_) => Err(tonic::Status::invalid_argument(
                "replacement must be a user-owned executable installation",
            )),
        };
        match result {
            Ok(()) => {
                upgrading.send_replace(true);
                let _ = reply.send(Ok(()));
                return Some(request);
            }
            Err(error) => {
                let _ = reply.send(Err(error));
            }
        }
    }
    None
}

fn validate_upgrade(request: &UpgradeDaemonRequest) -> anyhow::Result<()> {
    let executable = PathBuf::from(&request.executable);
    let worker = executable.with_file_name("mirror-agent");
    if Path::new(&request.worker) != worker {
        anyhow::bail!("replacement agent must belong to the daemon installation bundle");
    }
    for path in [&executable, &worker] {
        let metadata = fs::symlink_metadata(path)?;
        if !path.is_absolute()
            || !metadata.is_file()
            || metadata.mode() & 0o022 != 0
            || metadata.uid() != geteuid().as_raw()
            || metadata.mode() & 0o111 == 0
        {
            anyhow::bail!(
                "upgrade executables must be user-owned executable files, not writable by others"
            );
        }
    }
    Ok(())
}

fn agent_executable() -> io::Result<PathBuf> {
    Ok(std::env::current_exe()?.with_file_name("mirror-agent"))
}

fn lock_daemon(directory: &Path) -> io::Result<File> {
    // File::open sets close-on-exec: agents must never inherit singleton ownership.
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK)
        .open(directory.join("daemon.lock"))?;
    let metadata = lock.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != geteuid().as_raw()
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "daemon.lock must be a private, user-owned regular file with one link",
        ));
    }
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(TryLockError::WouldBlock) => Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "daemon.lock is held; wait or recover explicitly, never unlink it",
        )),
        Err(TryLockError::Error(error)) => Err(error),
    }
}

fn private_directory(path: &Path) -> io::Result<()> {
    match DirBuilder::new().recursive(true).mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    check_private_directory(path)
}

fn check_private_directory(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.uid() != geteuid().as_raw() || metadata.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "socket directory must be user-owned and private",
        ));
    }
    Ok(())
}

fn socket_metadata(path: &Path) -> io::Result<Option<fs::Metadata>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if !metadata.file_type().is_socket()
        || metadata.uid() != geteuid().as_raw()
        || metadata.mode() & 0o077 != 0
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "daemon endpoint must be a private, user-owned socket, not a symlink",
        ));
    }
    Ok(Some(metadata))
}

async fn remove_stale_socket(path: &Path) -> io::Result<()> {
    let Some(previous) = socket_metadata(path)? else {
        return Ok(());
    };
    match tokio::time::timeout(Duration::from_secs(1), UnixStream::connect(path)).await {
        Ok(Err(error))
            if matches!(
                error.kind(),
                io::ErrorKind::ConnectionRefused | io::ErrorKind::NotFound
            ) => {}
        Ok(Err(error)) => return Err(error),
        Ok(Ok(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "daemon endpoint is accepting connections; refusing replacement",
            ))
        }
        Err(_) => {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "daemon endpoint did not respond; refusing replacement",
            ))
        }
    }
    if let Some(current) = socket_metadata(path)? {
        if (current.dev(), current.ino()) != (previous.dev(), previous.ino()) {
            return Err(io::Error::other("daemon endpoint changed during recovery"));
        }
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn reuse_daemon_keeps_same_newer_and_unknown_versions_without_connecting() {
        let options = Options::try_parse_from([
            "mirror-daemon",
            "--ensure",
            "--state-root",
            "/unused",
            "--database",
            "/unused/state.db",
            "--worker-home",
            "/unused",
            "--application-version",
            "0.1.1",
        ])
        .unwrap();
        let mut info = GetDaemonInfoResponse {
            application_version: "0.1.1".into(),
            installation_id: "different-build".into(),
            ..Default::default()
        };
        let socket = Path::new("/must-not-connect");
        let same = reuse_daemon(&options, socket, &info).await.unwrap();
        info.application_version = "0.2.0".into();
        let newer = reuse_daemon(&options, socket, &info).await.unwrap();
        assert_eq!((same, newer), (true, true));
        info.application_version = "unknown".into();
        assert_matches!(reuse_daemon(&options, socket, &info).await, Ok(true));
    }
}
