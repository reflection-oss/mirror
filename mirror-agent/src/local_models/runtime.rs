//! The sequential owner of the managed server: its lifecycle, settings, loads and downloads.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::future::OptionFuture;
use mirror_protocol::local_models::{
    LocalModelDownload, LocalModelInfo, LocalModelServerPhase, LocalModelState,
};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinSet;
use uuid::Uuid;

use super::engine::Engine;
use super::host_state::{HostState, ServerRef, MANAGED, MANAGED_SCHEME};
use super::reads::Reads;
use super::restart_policy::{Decision, RestartPolicy};
use super::settings::{Settings, SettingsError};
use super::{Catalog, Downloads, LocalModelError};
use crate::private_directory;
use crate::providers::{Eviction, LlamaCpp, ModelEvent};

#[derive(Debug)]
pub struct Runtime {
    root: PathBuf,
    worker: PathBuf,
    /// Written only here; every reader watches it.
    state: watch::Sender<HostState>,
    catalog: Arc<Catalog>,
    downloads: Downloads,
    server: Option<Server>,
    /// Numbers the servers started so far, so a late report about an earlier one is ignored.
    launches: u64,
    exits: mpsc::Receiver<u64>,
    exit_reports: mpsc::Sender<u64>,
    model_events: mpsc::Receiver<(u64, ModelEvent)>,
    model_event_reports: mpsc::Sender<(u64, ModelEvent)>,
    /// Set while an unexpected exit is being recovered from.
    restart: Option<Restart>,
    policy: RestartPolicy,
}

/// Something the actor does to the managed server or its settings.
#[derive(Debug)]
pub enum Command {
    Start,
    Stop,
    Restart,
    Context(u64),
    SleepIdle(u64),
    /// `base_url` names the server: empty for the managed one, else a loopback router.
    Load {
        model: String,
        eviction: Eviction,
        base_url: String,
    },
    Restore(String),
    Download {
        review: LocalModelDownload,
        base_url: String,
    },
    CancelDownload(String),
}

/// A command and where to send its outcome; dropping `result` cancels the command.
#[derive(Debug)]
pub struct CommandRequest {
    pub command: Command,
    pub result: oneshot::Sender<Result<Option<LocalModelInfo>, LocalModelError>>,
}

/// The running engine and the tasks reporting on it, which end with it.
#[derive(Debug)]
struct Server {
    engine: Engine,
    id: u64,
    reporters: JoinSet<()>,
}

/// A scheduled attempt to bring back a server that exited without being asked to.
#[derive(Debug)]
struct Restart {
    /// The model to reload once the server is back; empty when none was active.
    model: String,
    due: tokio::time::Instant,
}

/// Context value meaning "let llama.cpp fit it".
const AUTO_CONTEXT: u64 = 0;

const CONTEXT_WINDOWS: [u64; 6] = [8192, 16384, 32768, 65536, 131072, 262144];

/// Idle delays offered for unloading a model, in seconds; zero means never.
const SLEEP_IDLE_CHOICES: [u64; 5] = [0, 300, 900, 1800, 3600];

/// How long one command, or one model reload after a restart, may take.
const OPERATION_TIMEOUT: Duration = Duration::from_secs(600);

/// How long a server gets to come up, counting the guard's wait for llama-server to listen.
const START_TIMEOUT: Duration = Duration::from_secs(60);

/// How long to wait for the router to say what it holds after an interrupted load.
const RECONCILE_TIMEOUT: Duration = Duration::from_secs(5);

/// How long to wait before following the router's events again after the stream ends.
const RESUBSCRIBE_DELAY: Duration = Duration::from_secs(2);

impl Command {
    /// Whether only the daemon that owns the server may ask for it.
    pub fn owner_only(&self) -> bool {
        matches!(self, Self::Stop | Self::Restart)
    }

    /// Whether the command saves a setting, which must finish even if its caller stops waiting:
    /// a write cut short could otherwise land after a newer one.
    fn saves_setting(&self) -> bool {
        matches!(self, Self::Context(_) | Self::SleepIdle(_))
    }
}

impl Runtime {
    /// A runtime that starts from the saved `settings`, with nothing running.
    pub fn new(
        root: PathBuf,
        port: u16,
        worker: PathBuf,
        binary: Option<PathBuf>,
        settings: Settings,
    ) -> Result<(Self, Reads), LocalModelError> {
        let catalog = Arc::new(Catalog::new()?);
        // The run id keeps the reference valid across restarts of the server, and makes a
        // reference from an earlier daemon on the same port fail to resolve.
        let reference = format!(
            "{MANAGED_SCHEME}127.0.0.1:{port}/{}",
            Uuid::new_v4().simple()
        );
        let (state, observed) = watch::channel(initial_state(
            &root,
            &settings,
            binary.as_deref(),
            reference,
        ));
        let (exit_reports, exits) = mpsc::channel(4);
        let (model_event_reports, model_events) = mpsc::channel(16);
        Ok((
            Self {
                root,
                worker,
                state,
                catalog: catalog.clone(),
                downloads: Downloads::default(),
                server: None,
                launches: 0,
                exits,
                exit_reports,
                model_events,
                model_event_reports,
                restart: None,
                policy: RestartPolicy::default(),
            },
            Reads::new(catalog, observed),
        ))
    }

    pub async fn run(
        mut self,
        mut requests: mpsc::Receiver<CommandRequest>,
        mut shutdown: oneshot::Receiver<()>,
    ) {
        loop {
            let due = self.restart.as_ref().map(|restart| restart.due);
            tokio::select! {
                _ = &mut shutdown => break,
                update = self.downloads.next_update() => self.downloads.apply(update, &self.state),
                Some(id) = self.exits.recv() => self.engine_exited(id),
                Some((id, event)) = self.model_events.recv() => self.model_event(id, &event),
                Some(request) = requests.recv() => {
                    if self.handle(request, &mut shutdown).await { break; }
                }
                Some(()) = OptionFuture::from(due.map(tokio::time::sleep_until)) => {
                    if self.recover(&mut requests, &mut shutdown).await { break; }
                }
            }
        }
        self.stop().await;
    }

    /// Run one command and reply; true when shutdown arrived while it ran.
    async fn handle(
        &mut self,
        request: CommandRequest,
        shutdown: &mut oneshot::Receiver<()>,
    ) -> bool {
        let CommandRequest {
            command,
            mut result,
        } = request;
        let cancellable = !command.saves_setting();
        let outcome = tokio::select! {
            biased;
            _ = &mut *shutdown => return true,
            // The caller stopped waiting, for instance because its RPC was cancelled.
            _ = result.closed(), if cancellable => None,
            outcome = tokio::time::timeout(OPERATION_TIMEOUT, self.execute(command)) => {
                Some(outcome.unwrap_or(Err(LocalModelError::TimedOut)))
            }
        };
        match outcome {
            Some(outcome) => {
                if outcome.is_err() {
                    self.settle().await;
                }
                let _ = result.send(outcome);
            }
            None => self.settle().await,
        }
        false
    }

    async fn execute(
        &mut self,
        command: Command,
    ) -> Result<Option<LocalModelInfo>, LocalModelError> {
        match command {
            Command::Start => {
                self.take_over();
                self.start().await?;
                Ok(None)
            }
            Command::Stop => {
                self.stop().await;
                Ok(None)
            }
            Command::Restart => {
                let model = self.state.borrow().state.active_model.clone();
                self.stop().await;
                self.start().await?;
                self.reload(&model).await
            }
            Command::Context(context) => {
                if context != AUTO_CONTEXT && !CONTEXT_WINDOWS.contains(&context) {
                    return Err(LocalModelError::UnsupportedContext(context));
                }
                self.save_setting(move |root| Settings::save_context(root, context))
                    .await?;
                self.publish(|host| host.state.context = context);
                Ok(None)
            }
            Command::SleepIdle(seconds) => {
                if !SLEEP_IDLE_CHOICES.contains(&seconds) {
                    return Err(LocalModelError::UnsupportedSleepIdle(seconds));
                }
                self.save_setting(move |root| Settings::save_sleep_idle(root, seconds))
                    .await?;
                self.publish(|host| host.state.sleep_idle = seconds);
                Ok(None)
            }
            Command::Load {
                model,
                eviction,
                base_url,
            } => match ServerRef::parse(&base_url) {
                ServerRef::External(address) => {
                    // Another server's models say nothing about the managed server's phase.
                    let server = self.state.borrow().server(ServerRef::External(address))?;
                    Ok(Some(server.load(&model, eviction).await?))
                }
                reference => {
                    self.state.borrow().check_issued(reference)?;
                    self.take_over();
                    self.start().await?;
                    self.load_model(&model, eviction).await.map(Some)
                }
            },
            Command::Restore(model) => {
                self.take_over();
                self.start().await?;
                self.reload(&model).await
            }
            Command::Download { review, base_url } => {
                let reference = ServerRef::parse(&base_url);
                let server = self.state.borrow().server(reference)?;
                let (server_name, cache) = match reference {
                    ServerRef::External(address) => (address.to_owned(), None),
                    ServerRef::Managed | ServerRef::Issued(_) => (
                        MANAGED.to_owned(),
                        self.server
                            .as_ref()
                            .map(|server| server.engine.models.clone()),
                    ),
                };
                self.downloads.begin(
                    self.catalog.clone(),
                    server,
                    server_name,
                    cache,
                    review,
                    &self.state,
                )?;
                Ok(None)
            }
            Command::CancelDownload(name) => {
                self.downloads.cancel(&name)?;
                Ok(None)
            }
        }
    }

    /// Load `model` after a start, or report why that did not work; empty means nothing to load.
    async fn reload(&self, model: &str) -> Result<Option<LocalModelInfo>, LocalModelError> {
        if model.is_empty() {
            return Ok(None);
        }
        let loaded = self.load_model(model, Eviction::Forbidden).await;
        if let Err(error) = &loaded {
            self.publish(|host| {
                host.state.notice = format!("Could not load {model}: {error}");
                host.state.revision += 1;
            });
        }
        loaded.map(Some)
    }

    /// A user's start or load supersedes any recovery under way and what it had failed at.
    fn take_over(&mut self) {
        self.restart = None;
        self.policy.reset();
        self.publish(|host| host.state.notice.clear());
    }

    /// Save a setting in the runtime directory, away from the actor's thread.
    async fn save_setting(
        &self,
        save: impl FnOnce(&Path) -> Result<(), SettingsError> + Send + 'static,
    ) -> Result<(), LocalModelError> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || {
            private_directory(&root).map_err(|source| LocalModelError::Prepare {
                path: root.clone(),
                source,
            })?;
            Ok::<_, LocalModelError>(save(&root)?)
        })
        .await??;
        self.publish(|host| host.state.revision += 1);
        Ok(())
    }

    fn publish(&self, change: impl FnOnce(&mut HostState)) {
        self.state.send_modify(change);
    }

    fn set_phase(&self, phase: LocalModelServerPhase, model: &str) {
        self.publish(|host| host.set_phase(phase, model));
    }

    /// Load a model on the managed server, publishing the phase as it goes.
    ///
    /// A load that fails leaves what the router actually holds published, which is not
    /// necessarily nothing: a refused load must not hide the model that is still serving.
    async fn load_model(
        &self,
        model: &str,
        eviction: Eviction,
    ) -> Result<LocalModelInfo, LocalModelError> {
        let server = self.state.borrow().server(ServerRef::Managed)?;
        self.set_phase(LocalModelServerPhase::Loading, model);
        match server.load(model, eviction).await {
            Ok(info) => {
                self.set_phase(LocalModelServerPhase::Serving, model);
                Ok(info)
            }
            Err(error) => {
                self.reconcile().await;
                Err(error.into())
            }
        }
    }

    /// Publish what the router says it holds.
    ///
    /// A router that cannot say, often because it just exited, leaves the published model
    /// in place, so a restart reloads the one it was loading or serving.
    async fn reconcile(&self) {
        let Ok(server) = self.state.borrow().server(ServerRef::Managed) else {
            return;
        };
        if let Ok(Ok(models)) = tokio::time::timeout(RECONCILE_TIMEOUT, server.inventory()).await {
            self.publish(|host| host.settle(&models));
        }
    }

    /// Undo what a command left half done when it was cancelled or timed out.
    async fn settle(&mut self) {
        let phase = self.state.borrow().state.phase;
        if phase == i32::from(LocalModelServerPhase::Starting) {
            self.release().await;
            self.set_phase(LocalModelServerPhase::Stopped, "");
        } else if phase == i32::from(LocalModelServerPhase::Loading) {
            self.reconcile().await;
        }
    }

    /// Schedule a restart when the server exits without having been stopped.
    fn engine_exited(&mut self, id: u64) {
        if self.server.as_ref().is_none_or(|server| server.id != id) {
            return;
        }
        self.server = None;
        let (phase, model) = {
            let state = &self.state.borrow().state;
            (state.phase, state.active_model.clone())
        };
        let live = [
            LocalModelServerPhase::Ready,
            LocalModelServerPhase::Loading,
            LocalModelServerPhase::Serving,
        ]
        .map(i32::from)
        .contains(&phase);
        if live {
            let decision = self.policy.on_exit(Instant::now());
            self.apply(decision, model);
        }
    }

    /// Follow a model event from the running server; one from an earlier server is stale.
    fn model_event(&self, id: u64, event: &ModelEvent) {
        if self.server.as_ref().is_some_and(|server| server.id == id) {
            self.state
                .send_if_modified(|host| host.apply_model_event(event));
        }
    }

    fn apply(&mut self, decision: Decision, model: String) {
        match decision {
            Decision::RetryAfter(delay) => {
                self.set_phase(LocalModelServerPhase::Restarting, &model);
                self.restart = Some(Restart {
                    model,
                    due: tokio::time::Instant::now() + delay,
                });
            }
            Decision::GiveUp(notice) => {
                self.restart = None;
                self.publish(|host| {
                    host.set_phase(LocalModelServerPhase::Stopped, "");
                    host.state.notice = notice;
                });
            }
        }
    }

    /// Run the restart that is due; true when shutdown arrived while it ran.
    ///
    /// A command interrupts the attempt and runs first. An attempt that had not brought the
    /// server back starts over, unless the command took over from it as a stop, a start or a
    /// load does, and a later attempt replaces whatever server it left half started. One that
    /// had leaves the router loading the model on its own.
    async fn recover(
        &mut self,
        requests: &mut mpsc::Receiver<CommandRequest>,
        shutdown: &mut oneshot::Receiver<()>,
    ) -> bool {
        let Some(Restart { model, .. }) = self.restart.take() else {
            return false;
        };
        tokio::select! {
            biased;
            _ = &mut *shutdown => true,
            () = self.restart_attempt(&model) => false,
            Some(request) = requests.recv() => {
                if self.state.borrow().resolve(ServerRef::Managed).is_ok() {
                    self.settle().await;
                } else {
                    self.apply(Decision::RetryAfter(Duration::ZERO), model);
                }
                self.handle(request, shutdown).await
            }
        }
    }

    /// Bring the server back and reload `model`, the one it was serving.
    async fn restart_attempt(&mut self, model: &str) {
        match self.start().await {
            Ok(()) => {
                // The server is back even if the model is not; `reload` says why.
                let reloaded = tokio::time::timeout(OPERATION_TIMEOUT, self.reload(model)).await;
                if reloaded.is_err() {
                    self.reconcile().await;
                }
            }
            Err(error) => {
                let decision = self.policy.on_attempt_failed(error.to_string());
                self.apply(decision, model.to_owned());
            }
        }
    }

    /// Make sure the server is running, starting it if it is not.
    async fn start(&mut self) -> Result<(), LocalModelError> {
        if self.state.borrow().resolve(ServerRef::Managed).is_ok() {
            return Ok(());
        }
        self.release().await;
        self.set_phase(LocalModelServerPhase::Starting, "");
        let started = tokio::time::timeout(START_TIMEOUT, self.launch())
            .await
            .unwrap_or(Err(LocalModelError::StartTimedOut));
        if started.is_err() {
            self.release().await;
            self.set_phase(LocalModelServerPhase::Stopped, "");
        }
        started
    }

    async fn launch(&mut self) -> Result<(), LocalModelError> {
        let (executable, context, sleep_idle) = {
            let state = &self.state.borrow().state;
            (state.executable.clone(), state.context, state.sleep_idle)
        };
        if executable.is_empty() {
            return Err(LocalModelError::NotInstalled);
        }
        let engine = Engine::spawn(
            &self.worker,
            &self.root,
            context,
            sleep_idle,
            Path::new(&executable),
        )
        .await?;
        self.launches += 1;
        let id = self.launches;
        let mut reporters = JoinSet::new();
        let mut alive = engine.alive.clone();
        let exits = self.exit_reports.clone();
        reporters.spawn(async move {
            let _ = alive.wait_for(|alive| !alive).await;
            let _ = exits.send(id).await;
        });
        self.publish(|host| host.alive = engine.alive.clone());
        let server = self.server.insert(Server {
            engine,
            id,
            reporters,
        });
        let endpoint = server.engine.ready().await?;
        let client = LlamaCpp::new(&endpoint, None)?;
        client.check_server().await?;
        server
            .reporters
            .spawn(follow_models(client, id, self.model_event_reports.clone()));
        self.publish(|host| {
            host.endpoint = endpoint;
            host.set_phase(LocalModelServerPhase::Ready, "");
        });
        self.policy.on_ready(Instant::now());
        self.state.borrow().resolve(ServerRef::Managed).map(|_| ())
    }

    /// Forget the running server and reap its process, leaving the published phase alone.
    async fn release(&mut self) {
        self.publish(|host| {
            host.endpoint.clear();
            host.state.revision += 1;
        });
        if let Some(Server {
            engine, reporters, ..
        }) = self.server.take()
        {
            drop(reporters);
            engine.stop().await;
        }
    }

    /// A stop the user asked for: nothing is restarted afterwards.
    async fn stop(&mut self) {
        self.restart = None;
        self.policy.reset();
        self.publish(|host| {
            host.set_phase(LocalModelServerPhase::Stopped, "");
            host.state.notice.clear();
        });
        self.release().await;
    }
}

/// The state the actor starts from: the saved settings, the reference it hands out for this
/// daemon run, and nothing running.
///
/// An unsupported saved value falls back to its default.
fn initial_state(
    root: &Path,
    settings: &Settings,
    binary: Option<&Path>,
    managed_reference: String,
) -> HostState {
    let path_text = |path: Result<PathBuf, SettingsError>| {
        path.map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let (_, alive) = watch::channel(false);
    HostState {
        state: LocalModelState {
            root: root.to_string_lossy().into_owned(),
            executable: binary
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default(),
            context: settings
                .context_tokens
                .filter(|context| CONTEXT_WINDOWS.contains(context))
                .unwrap_or(AUTO_CONTEXT),
            context_choices: CONTEXT_WINDOWS.to_vec(),
            managed_reference,
            sleep_idle: settings
                .sleep_idle_seconds
                .filter(|seconds| SLEEP_IDLE_CHOICES.contains(seconds))
                .unwrap_or_default(),
            sleep_idle_choices: SLEEP_IDLE_CHOICES.to_vec(),
            models_dir: path_text(settings.models_dir()),
            models_preset: path_text(settings.preset_path(root)),
            ..Default::default()
        },
        endpoint: String::new(),
        alive,
    }
}

/// Report server `id`'s model events to the actor for as long as the engine runs.
async fn follow_models(server: LlamaCpp, id: u64, reports: mpsc::Sender<(u64, ModelEvent)>) {
    loop {
        if let Ok(mut events) = server.model_events().await {
            while let Some(event) = events.next().await {
                if reports.send((id, event)).await.is_err() {
                    return;
                }
            }
        }
        tokio::time::sleep(RESUBSCRIBE_DELAY).await;
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;

    /// A runtime whose guard is `cat`, which runs until its stdin closes, as a real guard does.
    fn runtime_with_guard(root: &Path) -> Runtime {
        let worker = root.join("worker");
        std::fs::write(&worker, "#!/bin/sh\nexec cat\n").expect("worker script");
        std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o755)).expect("mode");
        // A child another test forked while the script was open for writing keeps it busy
        // until that child execs; once one run succeeds, no such child is left.
        while let Err(error) = std::process::Command::new(&worker)
            .stdin(std::process::Stdio::null())
            .status()
        {
            assert_eq!(
                error.kind(),
                std::io::ErrorKind::ExecutableFileBusy,
                "{error}"
            );
            std::thread::yield_now();
        }
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700)).expect("private");
        Runtime::new(root.into(), 0, worker, None, Settings::default())
            .expect("runtime")
            .0
    }

    /// Run the guard and report it serving `model`, without a router behind it.
    async fn serve(runtime: &mut Runtime, model: &str) -> u64 {
        let engine = Engine::spawn(
            &runtime.worker,
            &runtime.root,
            0,
            0,
            Path::new("/synthetic/llama-server"),
        )
        .await
        .expect("guard");
        runtime.publish(|host| host.alive = engine.alive.clone());
        runtime.launches += 1;
        runtime.server = Some(Server {
            engine,
            id: runtime.launches,
            reporters: JoinSet::new(),
        });
        runtime.set_phase(LocalModelServerPhase::Serving, model);
        runtime.launches
    }

    #[tokio::test]
    async fn an_unexpected_exit_schedules_a_restart_that_reloads_the_model() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());
        let id = serve(&mut runtime, "org/m:Q4").await;

        runtime.engine_exited(id + 1);
        assert!(
            runtime.restart.is_none(),
            "an unknown server's exit is ignored"
        );
        runtime.engine_exited(id);

        let restart = runtime.restart.as_ref().expect("restart scheduled");
        assert_eq!(restart.model, "org/m:Q4");
        assert_eq!(
            runtime.state.borrow().state.phase,
            i32::from(LocalModelServerPhase::Restarting)
        );
    }

    #[tokio::test]
    async fn a_stop_the_user_asked_for_is_final() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());
        let id = serve(&mut runtime, "org/m:Q4").await;

        runtime.stop().await;
        runtime.engine_exited(id);

        assert!(runtime.restart.is_none());
        assert_eq!(
            runtime.state.borrow().state.phase,
            i32::from(LocalModelServerPhase::Stopped)
        );
    }

    #[tokio::test]
    async fn a_load_takes_over_from_a_pending_restart_and_starts_the_server() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());
        let id = serve(&mut runtime, "org/m:Q4").await;
        runtime.engine_exited(id);

        let loaded = runtime
            .execute(Command::Load {
                model: "org/m:Q4".into(),
                eviction: Eviction::Forbidden,
                base_url: String::new(),
            })
            .await;

        assert_matches!(loaded, Err(LocalModelError::NotInstalled));
        assert!(runtime.restart.is_none());
        assert_eq!(
            runtime.state.borrow().state.phase,
            i32::from(LocalModelServerPhase::Stopped)
        );
    }

    #[tokio::test]
    async fn a_load_through_an_earlier_daemons_reference_is_refused() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());

        let loaded = runtime
            .execute(Command::Load {
                model: "org/m:Q4".into(),
                eviction: Eviction::Allowed,
                base_url: "managed://127.0.0.1:1/earlier".into(),
            })
            .await;

        assert_matches!(loaded, Err(LocalModelError::StaleReference));
        assert!(runtime.server.is_none(), "nothing was started");
    }

    #[tokio::test]
    async fn a_server_that_exits_during_a_load_restarts_with_that_model() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());
        let id = serve(&mut runtime, "").await;
        let gone = portpicker::pick_unused_port().expect("free port");
        runtime.publish(|host| host.endpoint = format!("http://127.0.0.1:{gone}/v1"));

        let loaded = runtime
            .execute(Command::Load {
                model: "org/m:Q4".into(),
                eviction: Eviction::Allowed,
                base_url: String::new(),
            })
            .await;
        runtime.engine_exited(id);

        assert_matches!(loaded, Err(LocalModelError::Server(_)));
        let restart = runtime.restart.as_ref().expect("restart scheduled");
        assert_eq!(restart.model, "org/m:Q4");
    }

    #[tokio::test]
    async fn a_stop_interrupts_a_restart_that_is_still_starting() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());
        runtime.publish(|host| host.state.executable = "/synthetic/llama-server".into());
        let id = serve(&mut runtime, "org/m:Q4").await;
        runtime.engine_exited(id);
        let mut state = runtime.state.subscribe();
        let (commands, requests) = mpsc::channel(1);
        let (_shutdown, stopped) = oneshot::channel();
        tokio::spawn(runtime.run(requests, stopped));
        state
            .wait_for(|host| host.state.phase == i32::from(LocalModelServerPhase::Starting))
            .await
            .expect("restart attempt");

        let (result, outcome) = oneshot::channel();
        commands
            .send(CommandRequest {
                command: Command::Stop,
                result,
            })
            .await
            .expect("stop sent");

        // The guard never reports ready, so an attempt that ran to the end would hold the stop
        // for the guard's 30 s start-up wait.
        let stopped = tokio::time::timeout(Duration::from_secs(10), outcome).await;
        assert_matches!(stopped, Ok(Ok(Ok(None))));
        assert_eq!(
            state.borrow().state.phase,
            i32::from(LocalModelServerPhase::Stopped)
        );
    }

    #[tokio::test]
    async fn the_context_and_idle_delay_are_validated_and_saved() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());

        assert_matches!(
            runtime.execute(Command::Context(123)).await,
            Err(LocalModelError::UnsupportedContext(123))
        );
        assert_matches!(
            runtime.execute(Command::SleepIdle(123)).await,
            Err(LocalModelError::UnsupportedSleepIdle(123))
        );
        runtime
            .execute(Command::Context(32768))
            .await
            .expect("context");
        runtime
            .execute(Command::SleepIdle(900))
            .await
            .expect("idle delay");

        let saved = Settings::load(root.path()).expect("saved settings");
        let reopened = initial_state(root.path(), &saved, None, String::new());
        assert_eq!(reopened.state.context, 32768);
        assert_eq!(reopened.state.sleep_idle, 900);
    }

    #[tokio::test]
    async fn a_setting_is_saved_even_when_its_caller_stopped_waiting() {
        let root = tempfile::tempdir().expect("root");
        let mut runtime = runtime_with_guard(root.path());
        let (result, outcome) = oneshot::channel();
        drop(outcome);
        let (_shutdown, mut stopped) = oneshot::channel();

        runtime
            .handle(
                CommandRequest {
                    command: Command::Context(32768),
                    result,
                },
                &mut stopped,
            )
            .await;

        assert_eq!(runtime.state.borrow().state.context, 32768);
    }
}
