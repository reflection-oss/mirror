//! Single owner of workspace admission, worker lifetimes and attachment routes.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use mirror_protocol::daemon::{
    create_workspace_request::Source, AgentLaunch, AgentLaunchOutcome, AttachWorkspaceRequest,
    AttachWorkspaceResponse, ConfigureDaemonRequest, CreateWorkspaceRequest,
    DeleteWorkspaceRequest, GetFrontendDefaultsResponse, RemoteSource, ResolveDirectoryRequest,
    ResolveDirectoryResponse, SwitchWorkspaceSessionRequest, UpdateWorkspaceRequest,
    WatchWorkspacesRequest, WatchWorkspacesResponse, Workspace, WorkspaceState,
};
use mirror_protocol::mirror::{
    ForkSessionRequest, ForkSessionResponse, ForkWorkspaceRequest, ForkWorkspaceResponse,
};
use mirror_protocol::{PersonalContext, WorkerSandbox};

use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, Id, JoinError, JoinHandle, JoinSet};
use tonic::Status;
use uuid::Uuid;

use crate::{AgentSpawner, PreparedSpawn, SpawnHandle, SpawnMode, SpawnTask};

use crate::workspace::{
    create_workspace, delete_files, describe_repository, discover_workspaces, from_configuration,
    open_workspace, parse_remote_endpoint, prepare_sandbox_workspace, sandbox_configuration,
    update_workspace, workspace_snapshot, Remote, RemoteConnections, Repository,
    Workspace as ExecutionWorkspace, WorkspaceRecord, WorkspaceStore,
};
use crate::{AgentOptions, ExecutionLease, ExistingSettings, FrontendDefaults, Session};

/// Typed commands keep provisioning outcomes separate from transport dispatch.
#[derive(Clone, Debug)]
pub struct WorkspaceRegistry {
    pub remote_connections: RemoteConnections,
    options: AgentOptions,
    commands: mpsc::Sender<Command>,
    shutdown: watch::Sender<bool>,
    changes: watch::Receiver<WatchWorkspacesResponse>,
    live_sessions: watch::Receiver<BTreeMap<String, Session>>,
}

type Reply<T> = oneshot::Sender<Result<T, Status>>;
type Execution = (Arc<dyn ExecutionWorkspace>, ExecutionLease);

pub enum Command {
    PrepareSpawn(AgentSpawner, String, SpawnTask, Reply<PreparedSpawn>),
    StartSpawn(
        AgentSpawner,
        String,
        Option<String>,
        Reply<AttachWorkspaceResponse>,
    ),
    FinishSpawn(String, AgentLaunchOutcome, Reply<Workspace>),
    Configure(ConfigureDaemonRequest, Reply<()>),
    FrontendDefaults(Reply<GetFrontendDefaultsResponse>),
    HostedEndpoint(Reply<String>),
    Snapshot(
        WatchWorkspacesRequest,
        Option<Repository>,
        Reply<WatchWorkspacesResponse>,
    ),
    Create(CreateWorkspaceRequest, Reply<Workspace>),
    PersonalContext(String, Reply<mirror_protocol::PersonalContext>),
    Update(UpdateWorkspaceRequest, Reply<Workspace>),
    SwitchSession(SwitchWorkspaceSessionRequest, Reply<Workspace>),
    Open(String, Reply<Workspace>),
    Close(String, Reply<()>),
    Delete(DeleteWorkspaceRequest, Reply<()>),
    DeleteHosted(DeleteWorkspaceRequest, Reply<Option<String>>),
    Attach(Box<AttachWorkspaceRequest>, Reply<AttachWorkspaceResponse>),
    Authorize(
        String,
        Option<String>,
        bool,
        Reply<(Session, ExecutionLease)>,
    ),
    Execution(String, Reply<Execution>),
    Fork(String, ForkSessionRequest, Reply<ForkSessionResponse>),
    ForkWorkspace(String, ForkWorkspaceRequest, Reply<ForkWorkspaceResponse>),
    StopIfIdle(Reply<()>),
}

struct SessionRoute {
    instance_id: String,
    host_spawning: bool,
    session: Session,
    workspace: Arc<dyn ExecutionWorkspace>,
    sandbox: Option<WorkerSandbox>,
    remote_endpoint: Option<String>,
}

type RemoteStart = Result<SessionRoute, Status>;

struct PendingRemote {
    endpoint: String,
    task: AbortHandle,
    requests: Vec<(AttachWorkspaceRequest, Reply<AttachWorkspaceResponse>)>,
}

struct RegistryState {
    options: AgentOptions,
    spawns: SpawnHandle,
    defaults: FrontendDefaults,
    local_models: mirror_agent::local_models::Service,
    store: WorkspaceStore,
    changes: watch::Sender<WatchWorkspacesResponse>,
    live_sessions: watch::Sender<BTreeMap<String, Session>>,
    sessions: BTreeMap<String, SessionRoute>,
    pending_remotes: BTreeMap<String, PendingRemote>,
    attachments: BTreeMap<String, String>,
}

impl WorkspaceRegistry {
    pub(crate) async fn promotion_context(&self) -> Result<PersonalContext, Status> {
        Remote::snapshot_context(&self.options.home).await
    }

    pub async fn spawn(
        options: AgentOptions,
        local_models: mirror_agent::local_models::Service,
    ) -> io::Result<(Self, JoinHandle<()>)> {
        let (commands, mut receiver) = mpsc::channel(32);
        let remote_connections = RemoteConnections::default();
        let mut state = RegistryState {
            spawns: SpawnHandle {
                remote_connections: remote_connections.clone(),
                commands: commands.downgrade(),
            },
            defaults: FrontendDefaults::default(),
            store: WorkspaceStore::open(&options.state_directory).await?,
            options: options.clone(),
            local_models,
            changes: watch::channel(WatchWorkspacesResponse::default()).0,
            live_sessions: watch::channel(BTreeMap::new()).0,
            sessions: BTreeMap::new(),
            pending_remotes: BTreeMap::new(),
            attachments: BTreeMap::new(),
        };
        state.publish();
        let changes = state.changes.subscribe();
        let live_sessions = state.live_sessions.subscribe();
        let (shutdown, mut stopping) = watch::channel(false);
        let task = tokio::spawn(async move {
            let mut remote_starts = JoinSet::new();
            let mut cleanup = tokio::time::interval(std::time::Duration::from_millis(100));
            loop {
                let command = tokio::select! {
                    biased;
                    _ = stopping.changed() => break,
                    Some(started) = remote_starts.join_next_with_id(), if !remote_starts.is_empty() => {
                        state.finish_remote(started).await;
                        state.publish();
                        continue;
                    }
                    _ = cleanup.tick() => {
                        if !state.sessions.is_empty() || !state.pending_remotes.is_empty() {
                            state.refresh().await;
                            state.publish();
                        }
                        continue;
                    }
                    command = receiver.recv() => match command { Some(command) => command, None => break },
                };
                match command {
                    Command::PrepareSpawn(caller, call_id, task, reply) => {
                        if reply.is_closed() {
                            continue;
                        }
                        let _ = reply.send(state.prepare_spawn(&caller, call_id, task).await);
                    }
                    Command::StartSpawn(caller, id, endpoint, reply) => {
                        if reply.is_closed() {
                            continue;
                        }
                        match state.spawn_attachment(&caller, &id, endpoint) {
                            Ok(request) if request.remote_endpoint.is_some() => {
                                state.begin_remote(request, reply, &mut remote_starts).await
                            }
                            Ok(request) => {
                                let _ = reply.send(state.attach(request).await);
                            }
                            Err(error) => {
                                let _ = reply.send(Err(error));
                            }
                        }
                    }
                    Command::FinishSpawn(id, outcome, reply) => {
                        let _ = reply.send(state.finish_spawn(&id, outcome).await);
                    }
                    Command::Configure(request, reply) => {
                        let connections = request
                            .configurations
                            .first()
                            .and_then(|config| config.mcp_connections.clone())
                            .unwrap_or_default();
                        let result = async {
                            state.defaults.configure(request)?;
                            for route in state.sessions.values() {
                                route.session.refresh_mcp(connections.clone()).await?;
                            }
                            Ok(())
                        }
                        .await;
                        let _ = reply.send(result);
                    }
                    Command::HostedEndpoint(reply) => {
                        let _ = reply.send(state.defaults.remote_endpoint());
                    }
                    Command::FrontendDefaults(reply) => {
                        let _ = reply.send(state.defaults.public());
                    }
                    Command::Snapshot(request, repository, reply) => {
                        let result = async {
                            if let Some(repository) = &repository {
                                discover_workspaces(
                                    &mut state.store,
                                    request.include_unmanaged,
                                    repository,
                                )
                                .await?;
                            }
                            Ok(state.snapshot(&request, repository.as_ref()))
                        }
                        .await;
                        let _ = reply.send(result);
                    }
                    Command::PersonalContext(id, reply) => {
                        let _ = reply.send(state.personal_context(&id).await);
                    }
                    Command::Create(request, reply) => {
                        let result =
                            create_workspace(&state.options, &mut state.store, request, None)
                                .await
                                .map(|workspace| state.project(workspace));
                        let _ = reply.send(result);
                    }
                    Command::Update(request, reply) => {
                        let result = update_workspace(&mut state.store, request)
                            .await
                            .map(|workspace| state.project(workspace));
                        let _ = reply.send(result);
                    }
                    Command::SwitchSession(request, reply) => {
                        let _ = reply.send(state.switch_session(request).await);
                    }
                    Command::Open(id, reply) => {
                        let result = open_workspace(&mut state.store, &id)
                            .await
                            .map(|workspace| state.project(workspace));
                        let _ = reply.send(result);
                    }
                    Command::Close(id, reply) => {
                        let _ = reply.send(state.close(&id).await);
                    }
                    Command::Delete(request, reply) => {
                        let _ = reply.send(state.delete(request, false).await.map(|_| ()));
                    }
                    Command::DeleteHosted(request, reply) => {
                        let _ = reply.send(state.delete(request, true).await);
                    }
                    Command::Attach(request, reply) => {
                        let request = match state.host_attachment(*request) {
                            Ok(request) => request,
                            Err(error) => {
                                let _ = reply.send(Err(error));
                                continue;
                            }
                        };
                        state.refresh().await;
                        if request.remote_endpoint.is_some()
                            && !state.sessions.contains_key(&request.workspace_id)
                        {
                            state.begin_remote(request, reply, &mut remote_starts).await;
                        } else {
                            let _ = reply.send(state.attach(request).await);
                        }
                    }
                    Command::Authorize(id, session, host, reply) => {
                        let _ = reply.send(state.authorize(&id, session.as_deref(), host).await);
                    }
                    Command::Execution(id, reply) => {
                        let _ = reply.send(state.execution(&id).await);
                    }
                    Command::Fork(id, request, reply) => {
                        let _ = reply.send(state.fork(&id, request).await);
                    }
                    Command::ForkWorkspace(id, request, reply) => {
                        let _ = reply.send(state.fork_workspace(&id, request).await);
                    }
                    Command::StopIfIdle(reply) => {
                        state.refresh().await;
                        if !receiver.is_empty()
                            || !state.sessions.is_empty()
                            || !state.pending_remotes.is_empty()
                        {
                            let _ = reply.send(Err(Status::failed_precondition(
                                "daemon is busy; close every workspace before upgrading",
                            )));
                        } else {
                            receiver.close();
                            let _ = reply.send(Ok(()));
                            return;
                        }
                    }
                }
                state.publish();
            }
            receiver.close();
            remote_starts.shutdown().await;
            state.pending_remotes.clear();
            for route in state.sessions.values() {
                route.session.close().await;
            }
            state.refresh().await;
        });
        Ok((
            Self {
                remote_connections,
                options,
                commands,
                shutdown,
                changes,
                live_sessions,
            },
            task,
        ))
    }

    /// Repository commands run outside the registry so a slow VCS cannot block agent traffic.
    pub async fn snapshot(
        &self,
        request: WatchWorkspacesRequest,
    ) -> Result<WatchWorkspacesResponse, Status> {
        let repository = match &request.directory {
            Some(directory) => Some(describe_repository(&self.options, directory).await?),
            None => None,
        };
        let (reply, result) = oneshot::channel();
        if *self.shutdown.borrow() {
            return Err(unavailable());
        }
        // A change wakes every watch; read-only snapshots must wait for admission.
        self.commands
            .send(Command::Snapshot(request, repository, reply))
            .await
            .map_err(|_| unavailable())?;
        result.await.map_err(|_| unavailable())?
    }

    /// Read-only session lookup must not queue behind provisioning or worker startup.
    pub fn live_session(&self, session_id: &str) -> Option<Session> {
        self.live_sessions
            .borrow()
            .get(session_id)
            .filter(|session| !session.is_closed())
            .cloned()
    }

    pub fn changes(&self) -> watch::Receiver<WatchWorkspacesResponse> {
        self.changes.clone()
    }
    pub fn stopping(&self) -> watch::Receiver<bool> {
        self.shutdown.subscribe()
    }

    pub fn shutdown(&self) {
        self.shutdown.send_replace(true);
    }

    fn submit(&self, command: Command) -> Result<(), Status> {
        if *self.shutdown.borrow() {
            return Err(unavailable());
        }
        self.commands
            .try_send(command)
            .map_err(|_| Status::resource_exhausted("supervisor ingress is full or closed"))
    }

    pub async fn configure(&self, mut request: ConfigureDaemonRequest) -> Result<(), Status> {
        request.directory = crate::workspace::text(
            &crate::workspace::host_directory(&self.options, &request.directory).await?,
        )?
        .to_owned();
        let (reply, result) = oneshot::channel();
        self.submit(Command::Configure(request, reply))?;
        result.await.map_err(|_| unavailable())?
    }

    pub async fn frontend_defaults(&self) -> Result<GetFrontendDefaultsResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::FrontendDefaults(reply))?;
        result.await.map_err(|_| unavailable())?
    }

    /// Hosted lifecycle requests use only the connection configured on this host.
    pub async fn hosted_catalog(&self) -> Result<Remote, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::HostedEndpoint(reply))?;
        let endpoint = result.await.map_err(|_| unavailable())??;
        self.remote_connections.connect(&endpoint, "").await
    }

    pub async fn resolve_directory(
        &self,
        request: ResolveDirectoryRequest,
    ) -> Result<ResolveDirectoryResponse, Status> {
        let base = if request.base_directory.is_empty() {
            self.frontend_defaults().await?.directory
        } else {
            request.base_directory
        };
        crate::workspace::resolve_directory(&self.options, &request.path, &base).await
    }

    pub async fn create(&self, request: CreateWorkspaceRequest) -> Result<Workspace, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Create(request, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn update(&self, request: UpdateWorkspaceRequest) -> Result<Workspace, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Update(request, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn switch_session(
        &self,
        request: SwitchWorkspaceSessionRequest,
    ) -> Result<Workspace, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::SwitchSession(request, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn open(&self, id: String) -> Result<Workspace, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Open(id, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn close(&self, id: String) -> Result<(), Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Close(id, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn delete(&self, request: DeleteWorkspaceRequest) -> Result<(), Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Delete(request, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn delete_hosted(
        &self,
        request: DeleteWorkspaceRequest,
    ) -> Result<Option<String>, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::DeleteHosted(request, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn attach(
        &self,
        mut request: AttachWorkspaceRequest,
    ) -> Result<AttachWorkspaceResponse, Status> {
        if let Some(endpoint) = &mut request.remote_endpoint {
            *endpoint = parse_remote_endpoint(endpoint)?.into();
        }
        let (reply, result) = oneshot::channel();
        self.submit(Command::Attach(Box::new(request), reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn authorize(
        &self,
        id: String,
        session: Option<String>,
        host: bool,
    ) -> Result<(Session, ExecutionLease), Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Authorize(id, session, host, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn execution(&self, id: String) -> Result<Execution, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Execution(id, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn fork(
        &self,
        id: String,
        request: ForkSessionRequest,
    ) -> Result<ForkSessionResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::Fork(id, request, reply))
            .map_err(|error| match error.code() {
                tonic::Code::ResourceExhausted => {
                    Status::aborted(format!("fork was not started: {}", error.message()))
                }
                _ => error,
            })?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn fork_workspace(
        &self,
        id: String,
        request: ForkWorkspaceRequest,
    ) -> Result<ForkWorkspaceResponse, Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::ForkWorkspace(id, request, reply))?;
        result.await.map_err(|_| unavailable())?
    }
    pub async fn stop_if_idle(&self) -> Result<(), Status> {
        let (reply, result) = oneshot::channel();
        self.submit(Command::StopIfIdle(reply))?;
        result.await.map_err(|_| unavailable())?
    }
}

impl RegistryState {
    fn spawn_caller(&self, caller: &AgentSpawner) -> Result<Workspace, Status> {
        let route = self
            .sessions
            .get(&caller.workspace_id)
            .filter(|route| {
                route.instance_id == caller.worker_id
                    && route.host_spawning
                    && !route.session.is_closed()
            })
            .ok_or_else(|| Status::permission_denied("calling host worker is no longer running"))?;
        Ok(route.session.settings())
    }

    async fn prepare_spawn(
        &mut self,
        caller: &AgentSpawner,
        call_id: String,
        task: SpawnTask,
    ) -> Result<PreparedSpawn, Status> {
        let parent = self.spawn_caller(caller)?;
        if call_id.is_empty() || call_id.len() > 256 {
            return Err(Status::invalid_argument("invalid spawn call identity"));
        }
        if let Some(record) = self.store.records().find(|record| {
            record
                .workspace
                .agent_launch
                .as_ref()
                .is_some_and(|launch| {
                    launch.parent_session_id == parent.session_id && launch.call_id == call_id
                })
        }) {
            return Ok(PreparedSpawn {
                workspace: record.workspace.clone(),
                endpoint: None,
                fresh: false,
            });
        }
        let endpoint = (task.mode == SpawnMode::Remote)
            .then(|| self.defaults.remote_endpoint())
            .transpose()?;
        let id = Uuid::new_v4().to_string();
        let source = if task.mode == SpawnMode::Remote {
            Source::Remote(RemoteSource {
                session_id: Some(id.clone()),
                ..Default::default()
            })
        } else {
            let directory = crate::workspace::resolve_directory(
                &self.options,
                task.directory.as_deref().unwrap_or(&parent.directory),
                &parent.directory,
            )
            .await?;
            if !directory.exists {
                return Err(Status::failed_precondition(
                    "child directory does not exist on this host",
                ));
            }
            match task.mode {
                SpawnMode::Folder => Source::Directory(directory.directory),
                _ => Source::WorktreeFrom(directory.directory),
            }
        };
        let mut preferences = parent.preferences.clone().unwrap_or_default();
        preferences.approval_mode = parent.approval_mode;
        self.defaults.configuration(Some(&preferences))?;
        self.spawn_caller(caller)?;
        let result = create_workspace(
            &self.options,
            &mut self.store,
            CreateWorkspaceRequest {
                workspace_id: id.clone(),
                name: task.name,
                preferences: Some(preferences),
                source: Some(source),
                ..Default::default()
            },
            Some(AgentLaunch {
                parent_workspace_id: parent.id,
                parent_session_id: parent.session_id,
                call_id,
                input_id: Uuid::new_v4().to_string(),
                outcome: AgentLaunchOutcome::Unknown as i32,
                session_id: String::new(),
            }),
        )
        .await;
        match result {
            Ok(workspace) => Ok(PreparedSpawn {
                workspace,
                endpoint,
                fresh: true,
            }),
            Err(error) => match self.store.get(&id) {
                // Provisioning may have completed before persistence failed. Never resume it implicitly.
                Some(record) => Ok(PreparedSpawn {
                    workspace: record.workspace.clone(),
                    endpoint: None,
                    fresh: false,
                }),
                None => Err(error),
            },
        }
    }

    fn spawn_attachment(
        &self,
        caller: &AgentSpawner,
        id: &str,
        endpoint: Option<String>,
    ) -> Result<AttachWorkspaceRequest, Status> {
        let parent = self.spawn_caller(caller)?;
        let record = self
            .store
            .get(id)
            .ok_or_else(|| Status::not_found("child workspace not found"))?;
        if !record
            .workspace
            .agent_launch
            .as_ref()
            .is_some_and(|launch| {
                launch.parent_workspace_id == parent.id
                    && launch.parent_session_id == parent.session_id
                    && launch.outcome == AgentLaunchOutcome::Unknown as i32
            })
        {
            return Err(Status::permission_denied(
                "workspace is not an undispatched direct child",
            ));
        }
        self.host_attachment(AttachWorkspaceRequest {
            workspace_id: id.to_owned(),
            client_id: Uuid::new_v4().to_string(),
            expected_session_id: Some(record.workspace.session_id.clone()),
            use_daemon_configuration: true,
            require_new_worker: true,
            remote_endpoint: endpoint,
            ..Default::default()
        })
    }

    async fn finish_spawn(
        &mut self,
        id: &str,
        outcome: AgentLaunchOutcome,
    ) -> Result<Workspace, Status> {
        let mut record = self
            .store
            .get(id)
            .ok_or_else(|| Status::not_found("child workspace not found"))?
            .clone();
        let launch = record
            .workspace
            .agent_launch
            .as_mut()
            .ok_or_else(|| Status::failed_precondition("workspace has no launch receipt"))?;
        if launch.outcome != AgentLaunchOutcome::Unknown as i32 {
            return Err(Status::failed_precondition(
                "launch already has a final receipt",
            ));
        }
        launch.outcome = outcome as i32;
        let workspace = record.workspace.clone();
        self.store.save([record]).await?;
        Ok(workspace)
    }

    async fn personal_context(&mut self, id: &str) -> Result<PersonalContext, Status> {
        let mut record = self
            .store
            .get(id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .clone();
        if !matches!(record.creation.source, Some(Source::Remote(_))) {
            return Err(Status::invalid_argument(
                "personal context requires a remote workspace",
            ));
        }
        if let Some(context) = record.personal_context {
            return Ok(context);
        }
        let context = Remote::snapshot_context(&self.options.home).await?;
        record.personal_context = Some(context.clone());
        self.store.save([record]).await?;
        Ok(context)
    }

    fn host_attachment(
        &self,
        mut request: AttachWorkspaceRequest,
    ) -> Result<AttachWorkspaceRequest, Status> {
        if !request.use_daemon_configuration {
            return Ok(request);
        }
        if request.configuration.is_some() {
            return Err(Status::invalid_argument(
                "choose daemon or frontend configuration",
            ));
        }
        let record = self
            .store
            .get(&request.workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?;
        if matches!(record.creation.source, Some(Source::Remote(_))) {
            let endpoint = self.defaults.remote_endpoint()?;
            if request
                .remote_endpoint
                .as_ref()
                .is_some_and(|supplied| supplied != &endpoint)
            {
                return Err(Status::permission_denied(
                    "hosted connection differs from host configuration",
                ));
            }
            request.remote_endpoint = Some(endpoint);
        }
        Ok(request)
    }

    async fn switch_session(
        &mut self,
        request: SwitchWorkspaceSessionRequest,
    ) -> Result<Workspace, Status> {
        let record = self
            .store
            .get(&request.workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .switch_session(&request)?;
        self.refresh().await;
        if self.sessions.contains_key(&request.workspace_id)
            || self.pending_remotes.contains_key(&request.workspace_id)
        {
            return Err(Status::failed_precondition(
                "close workspace before switching its session",
            ));
        }
        let workspace = record.workspace.clone();
        self.store.save([record]).await?;
        Ok(workspace)
    }

    fn snapshot(
        &self,
        request: &WatchWorkspacesRequest,
        repository: Option<&Repository>,
    ) -> WatchWorkspacesResponse {
        let snapshot = workspace_snapshot(&self.store, request, repository);
        WatchWorkspacesResponse {
            workspaces: snapshot
                .workspaces
                .into_iter()
                .map(|workspace| self.project(workspace))
                .collect(),
            next_after_id: snapshot.next_after_id,
        }
    }

    fn project(&self, mut workspace: Workspace) -> Workspace {
        workspace.can_delete_files = Some(false);
        if let Some(record) = self.store.get(&workspace.id) {
            workspace.is_worktree |=
                matches!(&record.creation.source, Some(Source::WorktreeFrom(_)))
                    || record
                        .checkout
                        .as_ref()
                        .is_some_and(|checkout| checkout.managed);
        }
        workspace.can_delete_files = Some(
            workspace.kind == mirror_protocol::daemon::WorkspaceKind::Sandbox as i32
                || (workspace.is_worktree
                    && Path::new(&workspace.directory)
                        .starts_with(self.options.home.join(".mirror/worktrees"))),
        );
        if let Some(route) = self
            .sessions
            .get(&workspace.id)
            .filter(|route| !route.session.is_closed())
        {
            let settings = route.session.settings();
            workspace.inference = settings.inference;
            workspace.approval_mode = settings.approval_mode;
        }
        workspace
    }

    fn publish(&self) {
        self.live_sessions.send_replace(
            self.sessions
                .values()
                .filter_map(|route| {
                    let session = &route.session;
                    (!session.is_closed()).then(|| (session.settings().session_id, session.clone()))
                })
                .collect(),
        );
        let snapshot = self.snapshot(
            &WatchWorkspacesRequest {
                include_closed: true,
                include_unmanaged: true,
                ..Default::default()
            },
            None,
        );
        self.changes.send_if_modified(|previous| {
            if *previous == snapshot {
                return false;
            }
            *previous = snapshot;
            true
        });
    }

    fn attachment_record(
        &self,
        request: &AttachWorkspaceRequest,
    ) -> Result<WorkspaceRecord, Status> {
        if Uuid::parse_str(&request.client_id).is_err() {
            return Err(Status::invalid_argument("client_id must be a UUID"));
        }
        if request.resume_daemon_id.is_some() {
            return Err(Status::failed_precondition(
                "live upgrade handoffs are unsupported; attach explicitly",
            ));
        }
        let record = self
            .store
            .get(&request.workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .clone();
        if request
            .expected_session_id
            .as_ref()
            .is_some_and(|expected| expected != &record.workspace.session_id)
        {
            return Err(Status::failed_precondition(
                "workspace conversation binding changed; refresh before attaching",
            ));
        }
        let remote = matches!(record.creation.source, Some(Source::Remote(_)));
        if remote != request.remote_endpoint.is_some() {
            return Err(Status::failed_precondition(
                "only remote workspaces require a live remote endpoint",
            ));
        }
        if self
            .sessions
            .get(&request.workspace_id)
            .is_some_and(|route| route.remote_endpoint != request.remote_endpoint)
        {
            return Err(Status::failed_precondition(
                "workspace is attached to a different endpoint",
            ));
        }
        if record.workspace.state == WorkspaceState::Failed as i32
            || (record.workspace.state == WorkspaceState::Provisioning as i32 && !remote)
        {
            return Err(Status::failed_precondition(
                "workspace provisioning did not complete",
            ));
        }
        if let Some(expected) = &request.existing_worker_id {
            if record.workspace.state == WorkspaceState::Closed as i32 {
                return Err(Status::cancelled("workspace closed"));
            }
            if request.require_new_worker
                || !self
                    .sessions
                    .get(&request.workspace_id)
                    .is_some_and(|route| {
                        route.instance_id == *expected && !route.session.is_closed()
                    })
            {
                return Err(Status::failed_precondition(
                    "attachment worker is no longer running",
                ));
            }
        }
        if request.require_new_worker
            && self
                .sessions
                .get(&request.workspace_id)
                .is_some_and(|route| !route.session.is_closed())
        {
            return Err(Status::failed_precondition(
                "close workspace before requesting a new worker",
            ));
        }
        Ok(record)
    }

    async fn begin_remote(
        &mut self,
        request: AttachWorkspaceRequest,
        reply: Reply<AttachWorkspaceResponse>,
        starts: &mut JoinSet<RemoteStart>,
    ) {
        let admitted = (|| {
            let record = self.attachment_record(&request)?;
            let endpoint = request.remote_endpoint.clone().ok_or_else(|| {
                Status::invalid_argument("remote attachment requires an endpoint")
            })?;
            let configuration = if request.use_daemon_configuration {
                self.defaults
                    .configuration(record.workspace.preferences.as_ref())?
            } else {
                request.configuration.clone().ok_or_else(|| {
                    Status::failed_precondition("configuration is required to attach")
                })?
            };
            Ok::<_, Status>((record.workspace, endpoint, configuration))
        })();
        let (workspace, endpoint, configuration) = match admitted {
            Ok(admitted) => admitted,
            Err(error) => {
                let _ = reply.send(Err(error));
                return;
            }
        };
        if let Some(pending) = self.pending_remotes.get_mut(&workspace.id) {
            if pending.endpoint != endpoint || request.require_new_worker {
                let _ = reply.send(Err(Status::failed_precondition(
                    "workspace already has a pending attachment",
                )));
            } else if pending.requests.len() >= 16 {
                let _ = reply.send(Err(Status::resource_exhausted("attachment limit reached")));
            } else {
                pending.requests.push((request, reply));
            }
            return;
        }
        if self.sessions.len() + self.pending_remotes.len() >= 64 {
            let _ = reply.send(Err(Status::resource_exhausted(
                "live session limit reached",
            )));
            return;
        }
        let saved = async {
            let mut record = self
                .store
                .get(&workspace.id)
                .ok_or_else(|| Status::not_found("workspace not found"))?
                .clone();
            record.workspace.state = WorkspaceState::Provisioning as i32;
            record.workspace.error = None;
            self.store.save([record]).await
        }
        .await;
        if let Err(error) = saved {
            let _ = reply.send(Err(error));
            return;
        }
        let existing = if workspace
            .agent_launch
            .as_ref()
            .is_some_and(|launch| launch.outcome == AgentLaunchOutcome::Unknown as i32)
        {
            ExistingSettings::Replace
        } else {
            ExistingSettings::Keep
        };
        let snapshots = self.spawns.clone();
        let client_id = request.client_id.clone();
        let remote_endpoint = endpoint.clone();
        let connections = self.spawns.remote_connections.clone();
        let task = starts.spawn(async move {
            let execution = Arc::new(
                connections
                    .connect(&remote_endpoint, &workspace.session_id)
                    .await?,
            );
            let (agent, channel) = execution
                .start_agent(&workspace.session_id, &configuration, "")
                .await
                .map_err(|error| Status::unavailable(error.reason))?;
            let session = Session::start(
                agent,
                channel,
                workspace,
                configuration,
                client_id,
                true,
                None,
                Some(snapshots),
                existing,
            )
            .await?;
            Ok(SessionRoute {
                instance_id: Uuid::new_v4().to_string(),
                host_spawning: false,
                session,
                workspace: execution,
                sandbox: None,
                remote_endpoint: Some(remote_endpoint),
            })
        });
        self.pending_remotes.insert(
            request.workspace_id.clone(),
            PendingRemote {
                endpoint,
                task,
                requests: vec![(request, reply)],
            },
        );
    }

    async fn finish_remote(&mut self, started: Result<(Id, RemoteStart), JoinError>) {
        let task_id = started
            .as_ref()
            .map(|(id, _)| *id)
            .unwrap_or_else(|error| error.id());
        let workspace_id = self
            .pending_remotes
            .iter()
            .find_map(|(id, pending)| (pending.task.id() == task_id).then(|| id.clone()));
        let Some(pending) = workspace_id.and_then(|id| self.pending_remotes.remove(&id)) else {
            if let Ok((_, Ok(route))) = started {
                route.session.close().await;
            }
            return;
        };
        let result = match started {
            Ok((_, result)) => result,
            Err(_) => Err(Status::unavailable("remote attachment stopped")),
        };
        let saved = async {
            let request = &pending
                .requests
                .first()
                .ok_or_else(|| Status::internal("missing attachment request"))?
                .0;
            let mut record = self
                .store
                .get(&request.workspace_id)
                .ok_or_else(|| Status::not_found("workspace not found"))?
                .clone();
            record.workspace.state = WorkspaceState::Available as i32;
            record.workspace.error = result
                .as_ref()
                .err()
                .map(|_| "Remote attachment failed; retry attachment.".into());
            self.store.save([record]).await
        }
        .await;
        let result = match result {
            Ok(route) => {
                if saved.is_ok() && pending.requests.iter().any(|(_, reply)| !reply.is_closed()) {
                    self.sessions.insert(route.session.settings().id, route);
                } else {
                    route.session.close().await;
                }
                saved
            }
            Err(error) => saved.and(Err(error)),
        };
        for (request, reply) in pending.requests {
            if !reply.is_closed() {
                let response = match &result {
                    Ok(()) => self.reserve_attachment(request).await,
                    Err(error) => Err(error.clone()),
                };
                let _ = reply.send(response);
            }
        }
    }

    fn cancel_remote(&mut self, id: &str) {
        if let Some(pending) = self.pending_remotes.remove(id) {
            pending.task.abort();
            for (_, reply) in pending.requests {
                let _ = reply.send(Err(Status::cancelled("remote attachment cancelled")));
            }
        }
    }

    async fn attach(
        &mut self,
        request: AttachWorkspaceRequest,
    ) -> Result<AttachWorkspaceResponse, Status> {
        self.refresh().await;
        let mut record = self.attachment_record(&request)?;
        if !self.sessions.contains_key(&request.workspace_id) {
            if request.remote_endpoint.is_some() {
                return Err(Status::failed_precondition(
                    "remote attachment is not ready",
                ));
            }
            if self.sessions.len() + self.pending_remotes.len() >= 64 {
                return Err(Status::resource_exhausted("live session limit reached"));
            }
            let mut configuration = if request.use_daemon_configuration {
                if request.configuration.is_some() {
                    return Err(Status::invalid_argument(
                        "choose daemon or frontend configuration",
                    ));
                }
                self.defaults
                    .configuration(record.workspace.preferences.as_ref())?
            } else {
                request.configuration.clone().ok_or_else(|| {
                    Status::failed_precondition("configuration is required to start a worker")
                })?
            };
            if configuration.inference.as_ref().is_none_or(|inference| {
                inference.model.is_empty() || inference.api.as_ref().is_none_or(String::is_empty)
            }) {
                return Err(Status::invalid_argument(
                    "explicit inference model and API are required",
                ));
            }
            let sandbox = sandbox_configuration(&self.options, &record);
            let shared = sandbox.as_ref().is_some_and(|sandbox| {
                self.sessions.values().any(|route| {
                    route
                        .sandbox
                        .as_ref()
                        .is_some_and(|other| other.id == sandbox.id)
                })
            });
            if let Some(sandbox) = sandbox.as_ref().filter(|_| !shared) {
                prepare_sandbox_workspace(&self.options, &record.workspace.directory, sandbox)
                    .await?;
            }
            let directory = tokio::fs::canonicalize(&record.workspace.directory)
                .await
                .map_err(|_| Status::failed_precondition("workspace directory is unavailable"))?;
            if directory != Path::new(&record.workspace.directory) {
                return Err(Status::failed_precondition(
                    "workspace directory identity changed",
                ));
            }
            // A sandboxed worker cannot reach the host's llama.cpp server.
            let restoring = match (&configuration.inference, &sandbox) {
                (Some(inference), None) => self
                    .local_models
                    .prepare(inference)
                    .await
                    .map_err(|error| Status::failed_precondition(error.to_string()))?,
                _ => None,
            };
            let instance_id = Uuid::new_v4().to_string();
            // A sandbox or frontend-configured worker must not inherit host-owned capabilities.
            let host_spawning = request.use_daemon_configuration && sandbox.is_none();
            let spawner = host_spawning.then(|| AgentSpawner {
                registry: self.spawns.clone(),
                workspace_id: record.workspace.id.clone(),
                worker_id: instance_id.clone(),
            });
            if host_spawning {
                configuration.host_tools.push(AgentSpawner::definition());
            }
            let execution = from_configuration(&self.options, directory, sandbox.clone());
            let (agent, channel) = match execution
                .start_agent(
                    &record.workspace.session_id,
                    &configuration,
                    &self.local_models.reference().await,
                )
                .await
            {
                Ok(started) => started,
                Err(error) => {
                    if !shared {
                        execution.stop().await?;
                    }
                    return Ok(AttachWorkspaceResponse {
                        startup_messages: error.messages,
                        startup_error: Some(error.reason.into()),
                        ..Default::default()
                    });
                }
            };
            // The conversation opens now; the saved model fills memory while it is read.
            if let Some(model) = restoring {
                self.local_models.restore_in_background(model);
            }
            let started = Session::start(
                agent,
                channel,
                record.workspace.clone(),
                configuration,
                request.client_id.clone(),
                request.use_daemon_configuration,
                spawner,
                None,
                ExistingSettings::Replace,
            )
            .await;
            let session = match started {
                Ok(session) => session,
                Err(error) => {
                    if !shared {
                        execution.stop().await?;
                    }
                    return Err(error);
                }
            };
            record.workspace.state = WorkspaceState::Available as i32;
            record.workspace.error = None;
            if let Err(error) = self.store.save([record]).await {
                session.close().await;
                if !shared {
                    execution.stop().await?;
                }
                return Err(error);
            }
            self.sessions.insert(
                request.workspace_id.clone(),
                SessionRoute {
                    instance_id,
                    host_spawning,
                    session,
                    workspace: execution,
                    sandbox,
                    remote_endpoint: request.remote_endpoint.clone(),
                },
            );
        }
        self.reserve_attachment(request).await
    }

    async fn reserve_attachment(
        &mut self,
        request: AttachWorkspaceRequest,
    ) -> Result<AttachWorkspaceResponse, Status> {
        let record = self
            .store
            .get(&request.workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?;
        let route = self
            .sessions
            .get(&request.workspace_id)
            .ok_or_else(|| Status::unavailable("worker stopped during attachment"))?;
        let host_tools = request
            .configuration
            .map(|config| config.host_tools)
            .or_else(|| request.use_daemon_configuration.then(Vec::new));
        let mut response = route
            .session
            .reserve(
                request.client_id,
                host_tools,
                request.replaces_attachment_id,
            )
            .await?;
        response.worker_instance_id = route.instance_id.clone();
        // Running sessions retain their startup metadata; the catalog owns later edits.
        response.workspace = Some(self.project(record.workspace.clone()));
        self.attachments
            .insert(response.attachment_id.clone(), request.workspace_id.clone());
        Ok(response)
    }

    async fn authorize(
        &self,
        id: &str,
        session_id: Option<&str>,
        host_only: bool,
    ) -> Result<(Session, ExecutionLease), Status> {
        let workspace_id = self
            .attachments
            .get(id)
            .ok_or_else(|| Status::permission_denied("unknown attachment"))?;
        let route = self
            .sessions
            .get(workspace_id)
            .ok_or_else(|| Status::permission_denied("attachment worker stopped"))?;
        if session_id.is_some_and(|id| id != route.session.settings().session_id) {
            return Err(Status::invalid_argument(
                "session does not match attached workspace",
            ));
        }
        let lease = route.session.authorize(id.to_owned(), host_only).await?;
        Ok((route.session.clone(), lease))
    }

    async fn execution(&self, id: &str) -> Result<Execution, Status> {
        let (_, lease) = self.authorize(id, None, true).await?;
        let route = self
            .attachments
            .get(id)
            .and_then(|id| self.sessions.get(id))
            .ok_or_else(|| Status::permission_denied("attachment worker stopped"))?;
        Ok((route.workspace.clone(), lease))
    }

    async fn close(&mut self, id: &str) -> Result<(), Status> {
        let mut record = self
            .store
            .get(id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .clone();
        self.cancel_remote(id);
        if let Some(route) = self.sessions.get(id) {
            route.session.close().await;
        }
        self.refresh().await;
        if self.sessions.contains_key(id) {
            return Err(Status::unavailable("workspace cleanup has not completed"));
        }
        record.workspace.state = WorkspaceState::Closed as i32;
        self.store.save([record]).await?;
        Ok(())
    }

    async fn delete(
        &mut self,
        request: DeleteWorkspaceRequest,
        hosted: bool,
    ) -> Result<Option<String>, Status> {
        let record = self
            .store
            .get(&request.workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .clone();
        if self
            .sessions
            .get(&request.workspace_id)
            .is_some_and(|route| !route.session.is_closed())
            || self.pending_remotes.contains_key(&request.workspace_id)
        {
            return Err(Status::failed_precondition(
                "close workspace before deleting it",
            ));
        }
        self.refresh().await;
        if self.sessions.contains_key(&request.workspace_id) {
            return Err(Status::failed_precondition("workspace cleanup is pending"));
        }
        if hosted {
            if request.delete_files || !matches!(record.creation.source, Some(Source::Remote(_))) {
                return Err(Status::invalid_argument(
                    "hosted deletion requires a remote workspace",
                ));
            }
            let endpoint = self.defaults.remote_endpoint()?;
            let remote = self
                .spawns
                .remote_connections
                .connect(&endpoint, &record.workspace.session_id)
                .await?;
            tokio::time::timeout(
                std::time::Duration::from_secs(120),
                remote
                    .lifecycle()
                    .delete_session(mirror_protocol::mirror::DeleteSessionRequest {
                        session_id: record.workspace.session_id.clone(),
                    }),
            )
            .await
            .map_err(|_| {
                Status::deadline_exceeded("hosted deletion is unconfirmed; inspect before retrying")
            })??;
        }
        if request.delete_files {
            let directory = Path::new(&record.workspace.directory);
            if let Some(other) = self.store.records().find(|other| {
                other.workspace.id != record.workspace.id
                    && !other.workspace.directory.is_empty()
                    && Path::new(&other.workspace.directory).starts_with(directory)
            }) {
                return Err(Status::failed_precondition(format!(
                    "directory is shared with another workspace: {:?} ({})",
                    other.workspace.name, other.workspace.id,
                )));
            }
            delete_files(&self.options, &record).await?;
        }
        match self.store.remove(&request.workspace_id).await {
            Ok(()) => Ok(None),
            Err(error) if hosted => Ok(Some(format!(
                "Hosted chat deleted, but its local handle could not be removed: {}",
                error.message(),
            ))),
            Err(error) => Err(error),
        }
    }

    async fn fork(
        &mut self,
        id: &str,
        request: ForkSessionRequest,
    ) -> Result<ForkSessionResponse, Status> {
        let (session, lease) = self
            .authorize(id, Some(&request.source_session_id), false)
            .await?;
        let workspace_id = self
            .attachments
            .get(id)
            .ok_or_else(|| Status::permission_denied("attachment expired"))?;
        let source = self
            .store
            .get(workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .clone();
        let remote = matches!(source.creation.source, Some(Source::Remote(_)));
        let response = session.fork(request, lease).await?;
        let id = if remote {
            response.session_id.clone()
        } else {
            Uuid::new_v4().to_string()
        };
        let mut workspace = source.workspace;
        workspace.id = id.clone();
        workspace.session_id = response.session_id.clone();
        workspace.previous_session_ids.clear();
        workspace.state = WorkspaceState::Available as i32;
        let creation = CreateWorkspaceRequest {
            create_directory: false,
            workspace_id: id,
            name: workspace.name.clone(),
            preferences: workspace.preferences.clone(),
            source: if remote {
                Some(Source::Remote(RemoteSource {
                    session_id: Some(response.session_id.clone()),
                    ..Default::default()
                }))
            } else if source.sandbox_id.is_some() {
                source.creation.source
            } else {
                Some(Source::Directory(workspace.directory.clone()))
            },
        };
        self.store
            .save([WorkspaceRecord {
                workspace,
                creation,
                checkout: source.checkout,
                sandbox_id: source.sandbox_id,
                discovered: false,
                personal_context: None,
            }])
            .await
            .map_err(|error| {
                Status::internal(format!(
                    "fork {} was created but could not be saved: {}",
                    response.session_id,
                    error.message()
                ))
            })?;
        Ok(response)
    }

    async fn fork_workspace(
        &mut self,
        id: &str,
        request: ForkWorkspaceRequest,
    ) -> Result<ForkWorkspaceResponse, Status> {
        let (session, lease) = self
            .authorize(id, Some(&request.source_session_id), false)
            .await?;
        let workspace_id = self
            .attachments
            .get(id)
            .ok_or_else(|| Status::permission_denied("attachment expired"))?;
        let source = self
            .store
            .get(workspace_id)
            .ok_or_else(|| Status::not_found("workspace not found"))?
            .clone();
        crate::workspace::fork_workspace(
            &self.options,
            &mut self.store,
            &source,
            session,
            lease,
            request,
        )
        .await
    }

    async fn refresh(&mut self) {
        let abandoned: Vec<_> = self
            .pending_remotes
            .iter()
            .filter(|(_, pending)| pending.requests.iter().all(|(_, reply)| reply.is_closed()))
            .map(|(id, _)| id.clone())
            .collect();
        for id in abandoned {
            self.cancel_remote(&id);
        }
        let active_sandboxes = self
            .sessions
            .values()
            .filter(|route| !route.session.is_closed())
            .filter_map(|route| route.sandbox.as_ref().map(|sandbox| sandbox.id.clone()))
            .collect::<std::collections::BTreeSet<_>>();
        let mut stopped = Vec::new();
        for (id, route) in &mut self.sessions {
            if !route.session.is_closed() {
                continue;
            }
            if route
                .sandbox
                .as_ref()
                .is_none_or(|sandbox| !active_sandboxes.contains(&sandbox.id))
                && route.workspace.stop().await.is_err()
            {
                continue;
            }
            stopped.push(id.clone());
        }
        for id in stopped {
            self.sessions.remove(&id);
        }
        let mut expired = Vec::new();
        for (attachment, workspace) in &self.attachments {
            match self.sessions.get(workspace) {
                Some(route)
                    if route
                        .session
                        .authorize(attachment.clone(), false)
                        .await
                        .is_ok() => {}
                _ => expired.push(attachment.clone()),
            }
        }
        for id in expired {
            self.attachments.remove(&id);
        }
    }
}

fn unavailable() -> Status {
    Status::unavailable("supervisor stopped")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::{poll_fn, Future};
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::task::Poll;

    use assert_matches::assert_matches;
    use mirror_protocol::agent_rpc_server::{AgentRpc, AgentRpcServer};
    use mirror_protocol::daemon::{AgentConfiguration, SandboxSource, WorkspaceKind};
    use mirror_protocol::{
        session_message, ApprovalMode, CallToolRequest, CallToolResponse, ClientMessage,
        GetMessagesRequest, GetMessagesResponse, InferenceConfig, InputAck, SessionMessage,
    };
    use pretty_assertions::assert_eq;
    use tonic::{Request, Response, Streaming};

    async fn isolated_models(
        root: &std::path::Path,
        options: &AgentOptions,
    ) -> mirror_agent::local_models::Host {
        mirror_agent::local_models::Host::bind(
            root.join("models"),
            options.executable.clone(),
            None,
            0,
        )
        .await
        .expect("isolated model service")
    }

    #[tokio::test]
    async fn personal_context_startup_retries_and_initialized_attachments() {
        let home = tempfile::tempdir().expect("home");
        let root = home.path();
        let options = AgentOptions {
            executable: root.join("unused"),
            bundle: root.into(),
            database: root.join("unused.db"),
            home: root.into(),
            runtime_directory: root.into(),
            state_directory: root.into(),
            path: String::new(),
        };
        let models = isolated_models(root, &options).await;
        std::fs::create_dir_all(root.join(".mirror/AGENTS.md")).expect("invalid instructions");
        let (registry, task) = WorkspaceRegistry::spawn(options.clone(), models.service.clone())
            .await
            .expect("registry");
        let id = Uuid::new_v4().to_string();
        let creation = CreateWorkspaceRequest {
            workspace_id: id.clone(),
            source: Some(Source::Remote(RemoteSource {
                session_id: Some(id.clone()),
                ..Default::default()
            })),
            ..Default::default()
        };
        registry
            .create(creation.clone())
            .await
            .expect("metadata registration does not read instructions");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let (inputs, mut received) = mpsc::unbounded_channel();
        std::fs::write(root.join("capture-startup"), "").expect("capture startup");
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(FixtureAgent(root.into(), Some(inputs))))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );
        let attachment = AttachWorkspaceRequest {
            workspace_id: id.clone(),
            client_id: Uuid::new_v4().to_string(),
            remote_endpoint: Some(endpoint),
            configuration: Some(AgentConfiguration {
                inference: Some(InferenceConfig::default()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_matches!(registry.attach(attachment.clone()).await,
            Err(error) if error.code() == tonic::Code::FailedPrecondition);
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        let failed = registry
            .snapshot(Default::default())
            .await
            .expect("failed workspace");
        assert!(failed.workspaces[0].error.is_some());
        std::fs::remove_dir(root.join(".mirror/AGENTS.md")).expect("remove invalid file");
        std::fs::write(root.join(".mirror/AGENTS.md"), "Original guidance").expect("instructions");
        std::fs::write(root.join("fail-startup"), "").expect("fail after snapshot");
        assert_matches!(registry.attach(attachment.clone()).await, Err(_));
        let first = received.recv().await.expect("startup message");
        assert_matches!(
            first.payload,
            Some(mirror_protocol::client_message::Payload::SetInferenceConfig(_))
        );
        assert_eq!(
            first
                .personal_context
                .as_ref()
                .expect("snapshot")
                .personal_instructions[0]
                .content,
            "Original guidance"
        );
        registry.shutdown();
        task.await.expect("stopped");
        std::fs::write(root.join(".mirror/AGENTS.md"), "Replacement")
            .expect("changed instructions");
        std::fs::remove_file(root.join("fail-startup")).expect("allow startup");
        let (registry, task) = WorkspaceRegistry::spawn(options.clone(), models.service.clone())
            .await
            .expect("restart");
        registry
            .attach(attachment.clone())
            .await
            .expect("retry startup");
        let retry = received.recv().await.expect("retry configuration");
        assert_eq!(retry.personal_context, first.personal_context);
        registry.shutdown();
        task.await.expect("stopped");

        // A different daemon must not collect context for an initialized session with no user input.
        std::fs::write(root.join("initialized"), "").expect("initial context window");
        std::fs::remove_file(root.join(".mirror/AGENTS.md")).expect("remove instructions");
        std::fs::create_dir(root.join(".mirror/AGENTS.md")).expect("invalid instructions");
        std::fs::create_dir(root.join("other-daemon")).expect("other daemon state");
        let (registry, task) = WorkspaceRegistry::spawn(
            AgentOptions {
                state_directory: root.join("other-daemon"),
                ..options
            },
            models.service.clone(),
        )
        .await
        .expect("other daemon");
        registry
            .create(creation)
            .await
            .expect("register existing session");
        registry
            .attach(attachment)
            .await
            .expect("attach without discovery");
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        registry.shutdown();
        task.await.expect("stopped");
        server.abort();
    }

    #[tokio::test]
    async fn hosted_deletion_requires_a_host_connection_and_retains_the_record_on_failure() {
        let directory = tempfile::tempdir().expect("isolated registry");
        let root = directory.path();
        let options = AgentOptions {
            executable: root.join("unused-worker"),
            bundle: root.into(),
            database: root.join("history.db"),
            home: root.into(),
            runtime_directory: root.into(),
            state_directory: root.into(),
            path: String::new(),
        };
        let models = isolated_models(root, &options).await;
        let (registry, task) = WorkspaceRegistry::spawn(options, models.service.clone())
            .await
            .expect("registry");
        let id = Uuid::new_v4().to_string();
        registry
            .create(CreateWorkspaceRequest {
                workspace_id: id.clone(),
                source: Some(Source::Remote(RemoteSource {
                    session_id: Some(id.clone()),
                    ..Default::default()
                })),
                ..Default::default()
            })
            .await
            .expect("remote handle");
        registry.close(id.clone()).await.expect("closed attachment");
        assert_matches!(registry.delete_hosted(DeleteWorkspaceRequest {
            workspace_id: id.clone(), delete_files: false,
        }).await, Err(error) if error.code() == tonic::Code::FailedPrecondition);
        let retained = registry
            .snapshot(WatchWorkspacesRequest {
                include_closed: true,
                ..Default::default()
            })
            .await
            .expect("retained catalog");
        assert_matches!(retained.workspaces.as_slice(), [workspace] if workspace.id == id);
        registry.shutdown();
        task.await.expect("shutdown");
        models.close().await;
    }

    #[tokio::test]
    async fn snapshot_waits_for_supervisor_capacity() {
        let directory = tempfile::tempdir().expect("isolated registry");
        let root = directory.path();
        let options = AgentOptions {
            executable: root.join("unused-worker"),
            bundle: root.into(),
            database: root.join("unused.db"),
            home: root.into(),
            runtime_directory: root.into(),
            state_directory: root.into(),
            path: String::new(),
        };
        let models = isolated_models(root, &options).await;
        let (registry, task) = WorkspaceRegistry::spawn(options, models.service.clone())
            .await
            .expect("registry");
        let capacity = registry
            .commands
            .reserve_many(registry.commands.max_capacity())
            .await
            .expect("saturate ingress");
        let snapshot = registry.snapshot(WatchWorkspacesRequest::default());
        tokio::pin!(snapshot);
        assert_matches!(
            poll_fn(|context| Poll::Ready(snapshot.as_mut().poll(context))).await,
            Poll::Pending
        );
        drop(capacity);
        assert_eq!(
            snapshot.await.expect("snapshot after capacity is released"),
            WatchWorkspacesResponse::default()
        );
        registry.shutdown();
        assert_matches!(
            registry.snapshot(WatchWorkspacesRequest::default()).await,
            Err(status) if status.code() == tonic::Code::Unavailable
        );
        task.await.expect("registry stopped");
    }

    struct FixtureAgent(PathBuf, Option<mpsc::UnboundedSender<ClientMessage>>);

    #[tonic::async_trait]
    impl AgentRpc for FixtureAgent {
        type SessionStream = std::pin::Pin<
            Box<dyn futures_core::Stream<Item = Result<SessionMessage, Status>> + Send>,
        >;

        async fn session(
            &self,
            request: Request<Streaming<ClientMessage>>,
        ) -> Result<Response<Self::SessionStream>, Status> {
            let mut incoming = request.into_inner();
            let inputs = self.1.clone();
            let root = self.0.clone();
            Ok(Response::new(Box::pin(async_stream::try_stream! {
                while let Some(message) = incoming.message().await? {
                    if root.join("capture-startup").exists()
                        && matches!(message.payload, Some(mirror_protocol::client_message::Payload::SetInferenceConfig(_))) {
                        if let Some(inputs) = &inputs { let _ = inputs.send(message.clone()); }
                        if root.join("fail-startup").exists() {
                            Err::<(), _>(Status::unavailable("fixture startup failure"))?;
                        }
                    }
                    if matches!(message.payload, Some(mirror_protocol::client_message::Payload::Message(_))) {
                        if let Some(inputs) = &inputs { let _ = inputs.send(message.clone()); }
                        if root.join("lose-prompt-ack").exists() { continue; }
                    }
                    if matches!(message.payload, Some(mirror_protocol::client_message::Payload::InspectSettings(()))) {
                        yield SessionMessage {
                            payload: Some(session_message::Payload::SessionSettings(Default::default())),
                            ..Default::default()
                        };
                    }
                    yield SessionMessage {
                        payload: Some(session_message::Payload::InputAck(InputAck {
                            message_id: message.message_id,
                        })),
                        ..Default::default()
                    };
                }
            })))
        }

        async fn get_messages(
            &self,
            _: Request<GetMessagesRequest>,
        ) -> Result<Response<GetMessagesResponse>, Status> {
            if self.0.join("fail-session").exists() {
                return Err(Status::failed_precondition("fixture session failure"));
            }
            let messages = if self.0.join("initialized").exists() {
                vec![mirror_protocol::AgentMessage {
                    sequence: Some(1),
                    payload: Some(mirror_protocol::agent_message::Payload::ContextWindow(
                        Default::default(),
                    )),
                    ..Default::default()
                }]
            } else {
                Vec::new()
            };
            Ok(Response::new(GetMessagesResponse {
                total: messages.len() as u64,
                messages,
                ..Default::default()
            }))
        }

        async fn call_tool(
            &self,
            _: Request<CallToolRequest>,
        ) -> Result<Response<CallToolResponse>, Status> {
            Err(Status::unimplemented("fixture has no tools"))
        }
    }

    #[tokio::test]
    async fn host_spawns_use_host_defaults_and_never_replay_an_unknown_prompt() {
        let home = tempfile::tempdir().expect("host");
        let root = home.path().canonicalize().expect("host path");
        let options = AgentOptions {
            executable: root.join("worker"),
            bundle: root.clone(),
            database: root.join("history.db"),
            home: root.clone(),
            runtime_directory: root.join("runtime"),
            state_directory: root.join("state"),
            path: "/usr/bin:/bin".into(),
        };
        crate::private_directory(&options.runtime_directory).expect("runtime");
        crate::private_directory(&options.state_directory).expect("state");
        std::fs::write(
            &options.executable,
            r#"#!/bin/sh
set -- $(dd bs=1 count=4 2>/dev/null | od -An -tu1)
dd bs=1 count=$((($1 << 24) | ($2 << 16) | ($3 << 8) | $4)) of=/dev/null 2>/dev/null
for directory in "$HOME"/runtime/a*; do
    if [ ! -e "$directory/rpc" ]; then
        ln -s "$HOME/fixture.sock" "$directory/rpc"
        echo "MIRROR_RPC_ADDRESS=unix:$directory/rpc"
        exec cat > /dev/null
    fi
done
exit 1
"#,
        )
        .expect("worker fixture");
        std::fs::set_permissions(&options.executable, std::fs::Permissions::from_mode(0o700))
            .expect("executable");
        let listener = tokio::net::UnixListener::bind(root.join("fixture.sock")).expect("listener");
        let (inputs, mut received) = mpsc::unbounded_channel();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(FixtureAgent(
                    root.clone(),
                    Some(inputs),
                )))
                .serve_with_incoming(crate::local_incoming(listener)),
        );
        let models = isolated_models(&root, &options).await;
        let (registry, task) = WorkspaceRegistry::spawn(options.clone(), models.service.clone())
            .await
            .expect("registry");
        let mut configuration = ConfigureDaemonRequest {
            directory: root.to_string_lossy().into_owned(),
            configurations: vec![AgentConfiguration {
                inference: Some(InferenceConfig {
                    model: "host-model".into(),
                    api: Some("host-api".into()),
                    ..Default::default()
                }),
                api_key: Some("synthetic-host-secret".into()),
                approval_mode: ApprovalMode::Bypass as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        registry
            .configure(configuration.clone())
            .await
            .expect("host configuration");
        let parent = registry
            .create(CreateWorkspaceRequest {
                workspace_id: Uuid::new_v4().to_string(),
                source: Some(Source::Directory(root.to_string_lossy().into_owned())),
                ..Default::default()
            })
            .await
            .expect("parent");
        let attachment = registry
            .attach(AttachWorkspaceRequest {
                workspace_id: parent.id.clone(),
                client_id: Uuid::new_v4().to_string(),
                use_daemon_configuration: true,
                ..Default::default()
            })
            .await
            .expect("host worker");
        configuration.configurations[0].approval_mode = ApprovalMode::Hitl as i32;
        registry
            .configure(configuration)
            .await
            .expect("changed host defaults");
        let spawner = AgentSpawner {
            registry: SpawnHandle {
                remote_connections: registry.remote_connections.clone(),
                commands: registry.commands.downgrade(),
            },
            workspace_id: parent.id.clone(),
            worker_id: attachment.worker_instance_id,
        };
        let (lifetime, observed) = watch::channel(());
        let call = mirror_protocol::ToolCall {
            id: "launch-one".into(),
            function: Some(mirror_protocol::FunctionCall {
                name: "spawn_agent".into(),
                arguments: r#"{"name":"child","message":"one task","mode":"folder"}"#.into(),
            }),
            ..Default::default()
        };
        let result = spawner.call(call.clone(), observed.clone()).await;
        assert!(result.success, "{}", result.content);
        let input = received.recv().await.expect("initial prompt");
        assert_eq!(
            spawner.call(call.clone(), observed.clone()).await.content,
            result.content
        );
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        let catalog = registry
            .snapshot(Default::default())
            .await
            .expect("catalog");
        let child = catalog
            .workspaces
            .iter()
            .find(|workspace| workspace.session_id == input.session_id)
            .expect("child");
        assert_eq!(child.directory, parent.directory);
        assert_eq!(
            child.inference.as_ref().expect("inference").model,
            "host-model"
        );
        assert_eq!(child.approval_mode, Some(ApprovalMode::Bypass as i32));
        let launch = child.agent_launch.as_ref().expect("provenance");
        assert_eq!(launch.parent_session_id, parent.session_id);
        assert_eq!(launch.input_id, input.message_id);
        assert_eq!(launch.outcome(), AgentLaunchOutcome::Accepted);
        assert!(!format!("{catalog:?}").contains("synthetic-host-secret"));

        std::fs::write(root.join("lose-prompt-ack"), "").expect("lost response fixture");
        let interrupted = mirror_protocol::ToolCall {
            id: "launch-two".into(),
            ..call.clone()
        };
        let running = {
            let spawner = spawner.clone();
            let interrupted = interrupted.clone();
            let observed = observed.clone();
            tokio::spawn(async move { spawner.call(interrupted, observed).await })
        };
        let unconfirmed = received
            .recv()
            .await
            .expect("prompt dispatched before response loss");
        lifetime.send_replace(());
        assert!(running
            .await
            .expect("cancelled call")
            .content
            .contains("unknown"));
        let (_new_lifetime, new_observed) = watch::channel(());
        let recovered = spawner.call(interrupted, new_observed).await;
        assert!(recovered.content.contains("unknown"));
        assert!(recovered.content.contains(&unconfirmed.session_id));
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        let hosted = mirror_protocol::ToolCall {
            id: "hosted-without-connection".into(),
            function: Some(mirror_protocol::FunctionCall {
                name: "spawn_agent".into(),
                arguments: r#"{"name":"hosted","message":"task","mode":"remote"}"#.into(),
            }),
            ..Default::default()
        };
        let (_live, observed) = watch::channel(());
        assert!(spawner
            .call(hosted, observed.clone())
            .await
            .content
            .contains("not configured on the daemon host"));
        registry.close(parent.id).await.expect("revoke caller");
        let denied = spawner
            .call(
                mirror_protocol::ToolCall {
                    id: "after-close".into(),
                    ..call
                },
                observed,
            )
            .await;
        assert!(denied
            .content
            .contains("calling host worker is no longer running"));
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        registry.shutdown();
        task.await.expect("shutdown");
        let (restarted, task) = WorkspaceRegistry::spawn(options, models.service.clone())
            .await
            .expect("restart");
        let catalog = restarted
            .snapshot(WatchWorkspacesRequest {
                include_closed: true,
                ..Default::default()
            })
            .await
            .expect("persisted receipts");
        assert_eq!(catalog.workspaces.len(), 3);
        assert_eq!(
            catalog
                .workspaces
                .iter()
                .find(|workspace| workspace.session_id == unconfirmed.session_id)
                .expect("uncertain child")
                .agent_launch
                .as_ref()
                .expect("receipt")
                .outcome(),
            AgentLaunchOutcome::Unknown
        );
        assert_matches!(received.try_recv(), Err(mpsc::error::TryRecvError::Empty));
        restarted.shutdown();
        task.await.expect("shutdown");
        server.abort();
    }

    #[tokio::test]
    async fn attach_failed_sandbox_fork_preserves_source_vm_until_last_session_closes() {
        let home = tempfile::tempdir().expect("isolated sandbox home");
        let root = home.path().canonicalize().unwrap();
        let options = AgentOptions {
            executable: root.join("worker"),
            bundle: root.clone(),
            database: root.join("history.db"),
            home: root.clone(),
            runtime_directory: root.join("runtime"),
            state_directory: root.join("state"),
            path: "/usr/bin:/bin".into(),
        };
        crate::private_directory(&options.runtime_directory).unwrap();
        crate::private_directory(&options.state_directory).unwrap();
        let directory = root.join(".mirror/sandboxes/shared");
        crate::private_directory(&directory).unwrap();
        let platform = if cfg!(target_os = "macos") {
            "darwin"
        } else {
            "linux"
        };
        let assets = root.join(format!(
            ".mirror/sandbox-assets/microsandbox-0.7.7-{platform}-{}",
            std::env::consts::ARCH
        ));
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("msb"), "fixture only").unwrap();
        std::fs::write(assets.join("libkrunfw.so"), "fixture only").unwrap();
        std::fs::write(assets.join("libkrunfw.dylib"), "fixture only").unwrap();
        std::fs::write(&options.executable, r#"#!/bin/sh
if [ "$1" = sandbox ]; then
    printf '%s\n' "$2" >> "$HOME/lifecycle"
    exit 0
fi
# Consume the framed startup request before replying, as the native worker does.
set -- $(dd bs=1 count=4 2>/dev/null | od -An -tu1)
dd bs=1 count=$((($1 << 24) | ($2 << 16) | ($3 << 8) | $4)) of=/dev/null 2>/dev/null
if [ -e "$HOME/fail-agent" ]; then
    echo 'MIRROR_RPC_ERROR=agent failed: invalid agent configuration: Invalid prompt budget: fixture'
    exit 1
fi
for directory in "$HOME"/runtime/a*; do
    if [ ! -e "$directory/rpc" ]; then
        ln -s "$HOME/fixture.sock" "$directory/rpc"
        echo "MIRROR_RPC_ADDRESS=unix:$directory/rpc"
        exec cat > /dev/null
    fi
done
exit 1
"#).unwrap();
        std::fs::set_permissions(&options.executable, std::fs::Permissions::from_mode(0o700))
            .unwrap();
        let source_id = Uuid::new_v4().to_string();
        let fork_id = Uuid::new_v4().to_string();
        let record = |id: &str| WorkspaceRecord {
            workspace: Workspace {
                id: id.into(),
                session_id: id.into(),
                directory: directory.to_str().unwrap().into(),
                kind: WorkspaceKind::Sandbox as i32,
                state: WorkspaceState::Available as i32,
                ..Default::default()
            },
            creation: CreateWorkspaceRequest {
                workspace_id: id.into(),
                source: Some(Source::Sandbox(SandboxSource {
                    image: "fixture-custom-image".into(),
                    ..Default::default()
                })),
                ..Default::default()
            },
            sandbox_id: Some(source_id.clone()),
            ..Default::default()
        };
        // A conversation fork retains its source's clone and VM identity.
        WorkspaceStore::open(&options.state_directory)
            .await
            .unwrap()
            .save([record(&source_id), record(&fork_id)])
            .await
            .unwrap();
        let listener = tokio::net::UnixListener::bind(root.join("fixture.sock")).unwrap();
        let server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(AgentRpcServer::new(FixtureAgent(root.clone(), None)))
                .serve_with_incoming(crate::local_incoming(listener)),
        );
        let models = isolated_models(&root, &options).await;
        let (registry, task) = WorkspaceRegistry::spawn(options, models.service.clone())
            .await
            .unwrap();
        let request = |id: &str| AttachWorkspaceRequest {
            workspace_id: id.into(),
            client_id: Uuid::new_v4().to_string(),
            configuration: Some(AgentConfiguration {
                inference: Some(InferenceConfig {
                    model: "fixture".into(),
                    api: Some("fixture".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let source = registry.attach(request(&source_id)).await.unwrap();
        assert!(!source.attachment_id.is_empty());
        let mut reconnect = request(&source_id);
        reconnect.existing_worker_id = Some(source.worker_instance_id.clone());
        let attached = registry.attach(reconnect.clone()).await.unwrap();
        assert_eq!(attached.worker_instance_id, source.worker_instance_id);
        let mut stale = request(&source_id);
        stale.existing_worker_id = Some(Uuid::new_v4().to_string());
        assert_matches!(registry.attach(stale).await,
            Err(status) if status.code() == tonic::Code::FailedPrecondition);
        assert_matches!(registry.stop_if_idle().await,
            Err(status) if status.code() == tonic::Code::FailedPrecondition);
        std::fs::write(root.join("fail-agent"), "").unwrap();
        let failed = registry.attach(request(&fork_id)).await.unwrap();
        assert_eq!(
            failed.startup_error.as_deref(),
            Some("Invalid prompt budget")
        );
        assert!(failed.attachment_id.is_empty());
        assert_eq!(
            std::fs::read_to_string(root.join("lifecycle")).unwrap(),
            "prepare\n"
        );
        assert_matches!(
            registry
                .authorize(source.attachment_id.clone(), None, false)
                .await,
            Ok(_)
        );

        std::fs::remove_file(root.join("fail-agent")).unwrap();
        std::fs::write(root.join("fail-session"), "").unwrap();
        assert_matches!(registry.attach(request(&fork_id)).await,
            Err(status) if status.message() == "fixture session failure");
        assert_eq!(
            std::fs::read_to_string(root.join("lifecycle")).unwrap(),
            "prepare\n"
        );
        std::fs::remove_file(root.join("fail-session")).unwrap();
        let catalog = root.join("state/workspaces-v1.pb");
        std::fs::rename(&catalog, root.join("saved-catalog")).unwrap();
        std::fs::create_dir(&catalog).unwrap();
        assert_matches!(registry.attach(request(&fork_id)).await,
            Err(status) if status.message() == "could not persist workspace metadata");
        assert_eq!(
            std::fs::read_to_string(root.join("lifecycle")).unwrap(),
            "prepare\n"
        );
        std::fs::remove_dir(&catalog).unwrap();
        std::fs::rename(root.join("saved-catalog"), catalog).unwrap();
        registry.close(source_id).await.unwrap();
        assert_matches!(registry.attach(reconnect).await,
            Err(status) if status.code() == tonic::Code::Cancelled && status.message() == "workspace closed");
        assert_eq!(
            std::fs::read_to_string(root.join("lifecycle")).unwrap(),
            "prepare\nstop\n"
        );

        std::fs::write(root.join("fail-agent"), "").unwrap();
        let failed = registry.attach(request(&fork_id)).await.unwrap();
        assert_eq!(
            failed.startup_error.as_deref(),
            Some("Invalid prompt budget")
        );
        assert_eq!(
            std::fs::read_to_string(root.join("lifecycle")).unwrap(),
            "prepare\nstop\nprepare\nstop\n"
        );
        registry.stop_if_idle().await.unwrap();
        assert_matches!(registry.attach(request(&fork_id)).await, Err(_));
        task.await.unwrap();
        server.abort();
    }
}
