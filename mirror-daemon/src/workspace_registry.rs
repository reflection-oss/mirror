//! Single owner of workspace admission, worker lifetimes and attachment routes.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use mirror_protocol::daemon::{
    create_workspace_request::Source, AttachWorkspaceRequest, AttachWorkspaceResponse,
    ConfigureDaemonRequest, CreateWorkspaceRequest, DeleteWorkspaceRequest,
    GetFrontendDefaultsResponse, RemoteSource, ResolveDirectoryRequest, ResolveDirectoryResponse,
    SwitchWorkspaceSessionRequest, UpdateWorkspaceRequest, WatchWorkspacesRequest,
    WatchWorkspacesResponse, Workspace, WorkspaceState,
};
use mirror_protocol::mirror::{
    ForkSessionRequest, ForkSessionResponse, ForkWorkspaceRequest, ForkWorkspaceResponse,
};
use mirror_protocol::WorkerSandbox;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::{AbortHandle, Id, JoinError, JoinHandle, JoinSet};
use tonic::Status;
use uuid::Uuid;

use crate::workspace::{
    create_workspace, delete_files, describe_repository, discover_workspaces, from_configuration,
    open_workspace, parse_remote_endpoint, prepare_sandbox_workspace, sandbox_configuration,
    update_workspace, workspace_snapshot, Remote, Repository, Workspace as ExecutionWorkspace,
    WorkspaceRecord, WorkspaceStore,
};
use crate::{AgentOptions, ExecutionLease, FrontendDefaults, Session};

/// Typed commands keep provisioning outcomes separate from transport dispatch.
#[derive(Clone, Debug)]
pub struct WorkspaceRegistry {
    options: AgentOptions,
    commands: mpsc::Sender<Command>,
    shutdown: watch::Sender<bool>,
    changes: watch::Receiver<WatchWorkspacesResponse>,
}

type Reply<T> = oneshot::Sender<Result<T, Status>>;
type Execution = (Arc<dyn ExecutionWorkspace>, ExecutionLease);

enum Command {
    Configure(ConfigureDaemonRequest, Reply<()>),
    FrontendDefaults(Reply<GetFrontendDefaultsResponse>),
    Snapshot(
        WatchWorkspacesRequest,
        Option<Repository>,
        Reply<WatchWorkspacesResponse>,
    ),
    Create(CreateWorkspaceRequest, Reply<Workspace>),
    Update(UpdateWorkspaceRequest, Reply<Workspace>),
    SwitchSession(SwitchWorkspaceSessionRequest, Reply<Workspace>),
    Open(String, Reply<Workspace>),
    Close(String, Reply<()>),
    Delete(DeleteWorkspaceRequest, Reply<()>),
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
    defaults: FrontendDefaults,
    store: WorkspaceStore,
    changes: watch::Sender<WatchWorkspacesResponse>,
    sessions: BTreeMap<String, SessionRoute>,
    pending_remotes: BTreeMap<String, PendingRemote>,
    attachments: BTreeMap<String, String>,
}

impl WorkspaceRegistry {
    pub async fn spawn(options: AgentOptions) -> io::Result<(Self, JoinHandle<()>)> {
        let mut state = RegistryState {
            defaults: FrontendDefaults::default(),
            store: WorkspaceStore::open(&options.state_directory).await?,
            options: options.clone(),
            changes: watch::channel(WatchWorkspacesResponse::default()).0,
            sessions: BTreeMap::new(),
            pending_remotes: BTreeMap::new(),
            attachments: BTreeMap::new(),
        };
        state.publish();
        let changes = state.changes.subscribe();
        let (commands, mut receiver) = mpsc::channel(32);
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
                    Command::Configure(request, reply) => {
                        let _ = reply.send(state.defaults.configure(request));
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
                    Command::Create(request, reply) => {
                        let result = create_workspace(&state.options, &mut state.store, request)
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
                        let _ = reply.send(state.delete(request).await);
                    }
                    Command::Attach(request, reply) => {
                        state.refresh().await;
                        if request.remote_endpoint.is_some()
                            && !state.sessions.contains_key(&request.workspace_id)
                        {
                            state.begin_remote(*request, reply, &mut remote_starts);
                        } else {
                            let _ = reply.send(state.attach(*request).await);
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
                options,
                commands,
                shutdown,
                changes,
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
        self.submit(Command::Fork(id, request, reply))?;
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
            || record.workspace.state == WorkspaceState::Provisioning as i32
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

    fn begin_remote(
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
            let configuration = request.configuration.clone().ok_or_else(|| {
                Status::failed_precondition("configuration is required to attach")
            })?;
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
        let client_id = request.client_id.clone();
        let remote_endpoint = endpoint.clone();
        let task = starts.spawn(async move {
            let execution =
                Arc::new(Remote::connect(&remote_endpoint, &workspace.session_id).await?);
            let (agent, channel) = execution
                .start_agent(&workspace.session_id, &configuration)
                .await
                .map_err(|error| Status::unavailable(error.reason))?;
            let session =
                Session::start(agent, channel, workspace, configuration, client_id).await?;
            Ok(SessionRoute {
                instance_id: Uuid::new_v4().to_string(),
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
        let result = match result {
            Ok(route) => {
                let admitted = async {
                    let mut record = self
                        .store
                        .get(&route.session.settings().id)
                        .ok_or_else(|| Status::not_found("workspace not found"))?
                        .clone();
                    record.workspace.state = WorkspaceState::Available as i32;
                    record.workspace.error = None;
                    self.store.save([record]).await
                }
                .await;
                if admitted.is_ok() && pending.requests.iter().any(|(_, reply)| !reply.is_closed())
                {
                    self.sessions.insert(route.session.settings().id, route);
                } else {
                    route.session.close().await;
                }
                admitted
            }
            Err(error) => Err(error),
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
            let configuration = if request.use_daemon_configuration {
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
            let execution = from_configuration(&self.options, directory, sandbox.clone());
            let (agent, channel) = match execution
                .start_agent(&record.workspace.session_id, &configuration)
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
            let started = Session::start(
                agent,
                channel,
                record.workspace.clone(),
                configuration,
                request.client_id.clone(),
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
                    instance_id: Uuid::new_v4().to_string(),
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
        let mut response = route.session.reserve(request.client_id, host_tools).await?;
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

    async fn delete(&mut self, request: DeleteWorkspaceRequest) -> Result<(), Status> {
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
        self.store.remove(&request.workspace_id).await
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
        if remote && (!request.after_entry_id.is_empty() || !request.session_id.is_empty()) {
            return Err(Status::unimplemented(
                "response forks require a host workspace",
            ));
        }
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
            }])
            .await?;
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
        session_message, CallToolRequest, CallToolResponse, ClientMessage, GetMessagesRequest,
        GetMessagesResponse, InferenceConfig, InputAck, SessionMessage,
    };
    use pretty_assertions::assert_eq;
    use tonic::{Request, Response, Streaming};

    #[tokio::test]
    async fn snapshot_waits_for_supervisor_capacity() {
        let directory = tempfile::tempdir().expect("isolated registry");
        let root = directory.path();
        let (registry, task) = WorkspaceRegistry::spawn(AgentOptions {
            executable: root.join("unused-worker"),
            bundle: root.into(),
            database: root.join("unused.db"),
            home: root.into(),
            runtime_directory: root.into(),
            state_directory: root.into(),
            path: String::new(),
        })
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

    struct FixtureAgent(PathBuf);

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
            Ok(Response::new(Box::pin(async_stream::try_stream! {
                while let Some(message) = incoming.message().await? {
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
            Ok(Response::new(GetMessagesResponse::default()))
        }

        async fn call_tool(
            &self,
            _: Request<CallToolRequest>,
        ) -> Result<Response<CallToolResponse>, Status> {
            Err(Status::unimplemented("fixture has no tools"))
        }
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
                .add_service(AgentRpcServer::new(FixtureAgent(root.clone())))
                .serve_with_incoming(crate::local_incoming(listener)),
        );
        let (registry, task) = WorkspaceRegistry::spawn(options).await.unwrap();
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
