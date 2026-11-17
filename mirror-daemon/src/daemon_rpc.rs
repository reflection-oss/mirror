//! Canonical DaemonRPC transport; attachment metadata is the execution boundary.

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use futures_core::Stream;
use mirror_protocol::agent_rpc_server::AgentRpc;
use mirror_protocol::daemon::{
    self, daemon_rpc_server::DaemonRpc as DaemonService, exec_response, ExecRequest, ExecResponse,
};
use mirror_protocol::mirror::{
    mirror_rpc_server::MirrorRpc, DeleteSessionRequest, DeleteSessionResponse, ForkSessionRequest,
    ForkSessionResponse, ForkWorkspaceRequest, ForkWorkspaceResponse, GetSessionRequest,
    GetSessionResponse, ListSessionsRequest, ListSessionsResponse, StartSessionRequest,
    StartSessionResponse, WatchSessionsRequest, WatchSessionsResponse,
};
use mirror_protocol::{
    CallToolRequest, CallToolResponse, ClientMessage, GetMessagesRequest, GetMessagesResponse,
    SessionMessage,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use crate::workspace::Workspace;
use crate::{AttachmentRecovery, ExecutionLease, HistorySearch, Upgrade, WorkspaceRegistry};

/// No frontend adapter owns worker configuration, history or persisted identity.
#[derive(Clone, Debug)]
pub struct DaemonRpc {
    registry: WorkspaceRegistry,
    database: PathBuf,
    search: HistorySearch,
    info: daemon::GetDaemonInfoResponse,
    upgrades: mpsc::Sender<Upgrade>,
    upgrading: watch::Receiver<bool>,
}

impl DaemonRpc {
    fn check_identity<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if let Some(expected) = request.metadata().get("mirror-daemon-id") {
            if expected.to_str().ok() != Some(self.info.instance_id.as_str()) {
                return Err(Status::failed_precondition(
                    "daemon changed; reconnect explicitly",
                ));
            }
        }
        Ok(())
    }

    pub fn new(
        registry: WorkspaceRegistry,
        database: PathBuf,
        info: daemon::GetDaemonInfoResponse,
        upgrades: mpsc::Sender<Upgrade>,
        upgrading: watch::Receiver<bool>,
    ) -> Self {
        Self {
            search: HistorySearch::start(database.clone(), registry.stopping()),
            registry,
            database,
            info,
            upgrades,
            upgrading,
        }
    }
}

#[tonic::async_trait]
impl DaemonService for DaemonRpc {
    async fn get_daemon_info(
        &self,
        _: Request<daemon::GetDaemonInfoRequest>,
    ) -> Result<Response<daemon::GetDaemonInfoResponse>, Status> {
        let mut info = self.info.clone();
        info.upgrading = *self.upgrading.borrow();
        Ok(Response::new(info))
    }

    async fn upgrade_daemon(
        &self,
        request: Request<daemon::UpgradeDaemonRequest>,
    ) -> Result<Response<daemon::UpgradeDaemonResponse>, Status> {
        self.check_identity(&request)?;
        let request = request.into_inner();
        if request.expected_instance_id != self.info.instance_id {
            return Err(Status::aborted("daemon changed before upgrade"));
        }
        let desired = request
            .application_version
            .parse::<pep440_rs::Version>()
            .map_err(|_| Status::invalid_argument("invalid application version"))?;
        let current = self
            .info
            .application_version
            .parse::<pep440_rs::Version>()
            .map_err(|_| Status::internal("invalid running application version"))?;
        if desired <= current {
            return Ok(Response::new(daemon::UpgradeDaemonResponse {}));
        }
        let (reply, result) = oneshot::channel();
        self.upgrades
            .try_send((request, reply))
            .map_err(|_| Status::unavailable("daemon upgrade already in progress"))?;
        result
            .await
            .map_err(|_| Status::unavailable("daemon upgrade stopped"))??;
        Ok(Response::new(daemon::UpgradeDaemonResponse {}))
    }

    async fn configure_daemon(
        &self,
        request: Request<daemon::ConfigureDaemonRequest>,
    ) -> Result<Response<daemon::ConfigureDaemonResponse>, Status> {
        self.registry.configure(request.into_inner()).await?;
        Ok(Response::new(daemon::ConfigureDaemonResponse {}))
    }

    async fn get_frontend_defaults(
        &self,
        request: Request<daemon::GetFrontendDefaultsRequest>,
    ) -> Result<Response<daemon::GetFrontendDefaultsResponse>, Status> {
        self.check_identity(&request)?;
        Ok(Response::new(self.registry.frontend_defaults().await?))
    }

    async fn resolve_directory(
        &self,
        request: Request<daemon::ResolveDirectoryRequest>,
    ) -> Result<Response<daemon::ResolveDirectoryResponse>, Status> {
        self.check_identity(&request)?;
        Ok(Response::new(
            self.registry
                .resolve_directory(request.into_inner())
                .await?,
        ))
    }

    async fn list_conversations(
        &self,
        request: Request<daemon::ListConversationsRequest>,
    ) -> Result<Response<daemon::ListConversationsResponse>, Status> {
        self.check_identity(&request)?;
        let mut request = request.into_inner();
        if request.limit > 500 || request.session_ids.len() > 500 {
            return Err(Status::invalid_argument(
                "conversation limit and identity count must not exceed 500",
            ));
        }
        if (request.recent_first
            && request.before_updated_at.is_some() == request.after_id.is_empty())
            || (!request.recent_first && request.before_updated_at.is_some())
        {
            return Err(Status::invalid_argument(
                "activity pagination requires both timestamp and identity",
            ));
        }
        let limit = if request.limit == 0 {
            100
        } else {
            request.limit
        };
        let recent_first = request.recent_first;
        request.limit = limit + 1;
        let database = self.database.clone();
        let mut conversations = tokio::task::spawn_blocking(move || {
            mirror_agent::local::list_conversations(&database, &request)
        })
        .await
        .map_err(|error| Status::internal(error.to_string()))?
        .map_err(|error| Status::failed_precondition(error.to_string()))?;
        let more = conversations.len() > limit as usize;
        conversations.truncate(limit as usize);
        let last = conversations.last().filter(|_| more);
        let next_after_id = last.map(|entry| entry.id.clone()).unwrap_or_default();
        let next_before_updated_at = last
            .filter(|_| recent_first)
            .map(|entry| entry.updated_at.unwrap_or_default());
        Ok(Response::new(daemon::ListConversationsResponse {
            conversations,
            next_after_id,
            next_before_updated_at,
        }))
    }

    async fn search_conversations(
        &self,
        request: Request<daemon::SearchConversationsRequest>,
    ) -> Result<Response<daemon::SearchConversationsResponse>, Status> {
        let request = request.into_inner();
        if request.query.len() > 1024
            || request.query.chars().count() > 256
            || request.limit > 50
            || request.session_ids.len() > 10_000
        {
            return Err(Status::invalid_argument(
                "conversation search exceeds query, limit or identity bounds",
            ));
        }
        Ok(Response::new(self.search.search(request).await?))
    }

    async fn get_prompt_history(
        &self,
        request: Request<daemon::GetPromptHistoryRequest>,
    ) -> Result<Response<daemon::GetPromptHistoryResponse>, Status> {
        self.check_identity(&request)?;
        let request = request.into_inner();
        if request.limit > 1000 {
            return Err(Status::invalid_argument("prompt limit exceeds 1000"));
        }
        let limit = if request.limit == 0 {
            1000
        } else {
            request.limit
        };
        let database = self.database.clone();
        let prompts = tokio::task::spawn_blocking(move || {
            mirror_agent::local::prompt_history(&database, &request.session_id, limit)
        })
        .await
        .map_err(|error| Status::internal(error.to_string()))?
        .map_err(|error| Status::failed_precondition(error.to_string()))?;
        Ok(Response::new(daemon::GetPromptHistoryResponse { prompts }))
    }

    async fn get_conversation_messages(
        &self,
        request: Request<daemon::GetConversationMessagesRequest>,
    ) -> Result<Response<daemon::GetConversationMessagesResponse>, Status> {
        self.check_identity(&request)?;
        let request = request.into_inner();
        if request.session_id.is_empty() || request.limit > 128 {
            return Err(Status::invalid_argument(
                "session identity is required; message limit must not exceed 128",
            ));
        }
        let database = self.database.clone();
        // History reads must not queue behind the registry's worker startup.
        let page = tokio::task::spawn_blocking(move || {
            mirror_agent::local::conversation_messages(&database, &request)
        })
        .await
        .map_err(|error| Status::internal(error.to_string()))?
        .map_err(|error| Status::failed_precondition(error.to_string()))?;
        Ok(Response::new(page))
    }

    type WatchWorkspacesStream =
        Pin<Box<dyn Stream<Item = Result<daemon::WatchWorkspacesResponse, Status>> + Send>>;
    async fn watch_workspaces(
        &self,
        request: Request<daemon::WatchWorkspacesRequest>,
    ) -> Result<Response<Self::WatchWorkspacesStream>, Status> {
        self.check_identity(&request)?;
        let request = request.into_inner();
        if request
            .directory
            .as_ref()
            .is_some_and(|directory| !Path::new(directory).is_absolute())
        {
            return Err(Status::invalid_argument(
                "directory filter must be absolute",
            ));
        }
        if request.limit > 500 {
            return Err(Status::invalid_argument("workspace limit exceeds 500"));
        }
        let registry = self.registry.clone();
        let mut changes = registry.changes();
        let mut stopping = registry.stopping();
        let stream = async_stream::try_stream! {
            let mut refresh = tokio::time::interval(std::time::Duration::from_secs(2));
            refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            refresh.tick().await;
            let mut previous = None;
            loop {
                changes.borrow_and_update();
                if *stopping.borrow() { Err(Status::cancelled("daemon stopped"))?; }
                let snapshot = tokio::select! {
                    biased;
                    _ = stopping.changed() => Err(Status::cancelled("daemon stopped")),
                    snapshot = registry.snapshot(request.clone()) => snapshot,
                }?;
                if previous.as_ref() != Some(&snapshot) { previous = Some(snapshot.clone()); yield snapshot; }
                tokio::select! {
                    biased;
                    _ = stopping.changed() => Err(Status::cancelled("daemon stopped")),
                    changed = changes.changed() => {
                        if changed.is_err() { break; }
                        Ok(())
                    },
                    _ = refresh.tick(), if request.directory.is_some() => Ok(()),
                }?;
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn create_workspace(
        &self,
        request: Request<daemon::CreateWorkspaceRequest>,
    ) -> Result<Response<daemon::CreateWorkspaceResponse>, Status> {
        self.check_identity(&request)?;
        Ok(Response::new(daemon::CreateWorkspaceResponse {
            workspace: Some(self.registry.create(request.into_inner()).await?),
        }))
    }
    async fn update_workspace(
        &self,
        request: Request<daemon::UpdateWorkspaceRequest>,
    ) -> Result<Response<daemon::UpdateWorkspaceResponse>, Status> {
        self.check_identity(&request)?;
        Ok(Response::new(daemon::UpdateWorkspaceResponse {
            workspace: Some(self.registry.update(request.into_inner()).await?),
        }))
    }
    async fn switch_workspace_session(
        &self,
        request: Request<daemon::SwitchWorkspaceSessionRequest>,
    ) -> Result<Response<daemon::SwitchWorkspaceSessionResponse>, Status> {
        self.check_identity(&request)?;
        Ok(Response::new(daemon::SwitchWorkspaceSessionResponse {
            workspace: Some(self.registry.switch_session(request.into_inner()).await?),
        }))
    }
    async fn open_workspace(
        &self,
        request: Request<daemon::OpenWorkspaceRequest>,
    ) -> Result<Response<daemon::OpenWorkspaceResponse>, Status> {
        self.check_identity(&request)?;
        Ok(Response::new(daemon::OpenWorkspaceResponse {
            workspace: Some(
                self.registry
                    .open(request.into_inner().workspace_id)
                    .await?,
            ),
        }))
    }
    async fn close_workspace(
        &self,
        request: Request<daemon::CloseWorkspaceRequest>,
    ) -> Result<Response<daemon::CloseWorkspaceResponse>, Status> {
        self.check_identity(&request)?;
        self.registry
            .close(request.into_inner().workspace_id)
            .await?;
        Ok(Response::new(daemon::CloseWorkspaceResponse {}))
    }
    async fn delete_workspace(
        &self,
        request: Request<daemon::DeleteWorkspaceRequest>,
    ) -> Result<Response<daemon::DeleteWorkspaceResponse>, Status> {
        self.check_identity(&request)?;
        self.registry.delete(request.into_inner()).await?;
        Ok(Response::new(daemon::DeleteWorkspaceResponse {}))
    }
    async fn attach_workspace(
        &self,
        request: Request<daemon::AttachWorkspaceRequest>,
    ) -> Result<Response<daemon::AttachWorkspaceResponse>, Status> {
        if request
            .get_ref()
            .expected_daemon_id
            .as_ref()
            .is_some_and(|expected| expected != &self.info.instance_id)
        {
            return Err(Status::failed_precondition(
                "daemon changed; reconnect explicitly",
            ));
        }
        let mut response = self.registry.attach(request.into_inner()).await?;
        response.daemon_instance_id = self.info.instance_id.clone();
        Ok(Response::new(response))
    }

    type ExecStream = ReceiverStream<Result<ExecResponse, Status>>;
    async fn exec(
        &self,
        request: Request<ExecRequest>,
    ) -> Result<Response<Self::ExecStream>, Status> {
        let id = attachment_id(&request)?;
        self.check_identity(&request)?;
        let request = request.into_inner();
        if request.argv.is_empty()
            || request.argv.len() > 1024
            || request.argv.iter().map(String::len).sum::<usize>() > 65536
        {
            return Err(Status::invalid_argument(
                "invalid process argument count or size",
            ));
        }
        let (workspace, lease) = self.registry.execution(id).await?;
        let (output, stream) = mpsc::channel(16);
        tokio::spawn(async move {
            let result = run_process(workspace.as_ref(), lease, request, &output).await;
            if let Err(error) = result {
                let _ = output.send(Err(error)).await;
            }
        });
        Ok(Response::new(ReceiverStream::new(stream)))
    }
}

#[tonic::async_trait]
impl AgentRpc for DaemonRpc {
    type SessionStream = Pin<Box<dyn Stream<Item = Result<SessionMessage, Status>> + Send>>;
    async fn session(
        &self,
        request: Request<Streaming<ClientMessage>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let id = attachment_id(&request)?;
        let (session, lease) = self.registry.authorize(id.clone(), None, false).await?;
        let recovery = match request
            .metadata()
            .get("mirror-resynchronize")
            .and_then(|value| value.to_str().ok())
        {
            Some("1") => AttachmentRecovery::InBand,
            Some("history") => AttachmentRecovery::InBandWithHistory,
            _ => AttachmentRecovery::Reconnect,
        };
        Ok(Response::new(Box::pin(
            session
                .attach(id, request.into_inner(), lease, recovery)
                .await?,
        )))
    }
    async fn get_messages(
        &self,
        request: Request<GetMessagesRequest>,
    ) -> Result<Response<GetMessagesResponse>, Status> {
        let id = attachment_id(&request)?;
        let (session, _lease) = self
            .registry
            .authorize(id, Some(request.get_ref().session_id.clone()), false)
            .await?;
        session
            .get_messages(request.into_inner())
            .await
            .map(Response::new)
    }
    async fn call_tool(
        &self,
        request: Request<CallToolRequest>,
    ) -> Result<Response<CallToolResponse>, Status> {
        let id = attachment_id(&request)?;
        let (session, lease) = self
            .registry
            .authorize(id, Some(request.get_ref().session_id.clone()), true)
            .await?;
        session
            .call_tool(request.into_inner(), lease)
            .await
            .map(Response::new)
    }
}

#[tonic::async_trait]
impl MirrorRpc for DaemonRpc {
    async fn start_session(
        &self,
        _request: Request<StartSessionRequest>,
    ) -> Result<Response<StartSessionResponse>, Status> {
        Err(Status::unimplemented("use DaemonRPC workspace lifecycle"))
    }

    async fn list_sessions(
        &self,
        _request: Request<ListSessionsRequest>,
    ) -> Result<Response<ListSessionsResponse>, Status> {
        Err(Status::unimplemented("use DaemonRPC workspace lifecycle"))
    }

    async fn get_session(
        &self,
        _request: Request<GetSessionRequest>,
    ) -> Result<Response<GetSessionResponse>, Status> {
        Err(Status::unimplemented("use DaemonRPC workspace lifecycle"))
    }

    async fn delete_session(
        &self,
        _request: Request<DeleteSessionRequest>,
    ) -> Result<Response<DeleteSessionResponse>, Status> {
        Err(Status::unimplemented("use DaemonRPC workspace lifecycle"))
    }

    type WatchSessionsStream =
        Pin<Box<dyn Stream<Item = Result<WatchSessionsResponse, Status>> + Send>>;

    async fn watch_sessions(
        &self,
        _request: Request<WatchSessionsRequest>,
    ) -> Result<Response<Self::WatchSessionsStream>, Status> {
        Err(Status::unimplemented("use DaemonRPC workspace lifecycle"))
    }

    async fn fork_workspace(
        &self,
        request: Request<ForkWorkspaceRequest>,
    ) -> Result<Response<ForkWorkspaceResponse>, Status> {
        self.registry
            .fork_workspace(attachment_id(&request)?, request.into_inner())
            .await
            .map(Response::new)
    }

    async fn fork_session(
        &self,
        request: Request<ForkSessionRequest>,
    ) -> Result<Response<ForkSessionResponse>, Status> {
        self.registry
            .fork(attachment_id(&request)?, request.into_inner())
            .await
            .map(Response::new)
    }
}

async fn run_process(
    workspace: &dyn Workspace,
    mut lease: ExecutionLease,
    request: ExecRequest,
    output: &mpsc::Sender<Result<ExecResponse, Status>>,
) -> Result<(), Status> {
    if output.is_closed() || lease.is_closed() {
        return Err(Status::cancelled("process attachment closed before launch"));
    }
    let mut process = workspace.exec(&request.argv).await?;
    let stdin = process
        .stdin
        .take()
        .ok_or_else(|| Status::internal("process stdin unavailable"))?;
    let stdout = process
        .stdout
        .take()
        .ok_or_else(|| Status::internal("process stdout unavailable"))?;
    let stderr = process
        .stderr
        .take()
        .ok_or_else(|| Status::internal("process stderr unavailable"))?;
    let feeding = async {
        let mut stdin = stdin;
        if let Err(error) = stdin.write_all(request.stdin.as_bytes()).await {
            if error.kind() != std::io::ErrorKind::BrokenPipe {
                return Err(Status::failed_precondition("process stdin failed"));
            }
        }
        drop(stdin);
        Ok::<_, Status>(())
    };
    let result = {
        let streaming = async {
            let (_, _, _, code) = tokio::try_join!(
                feeding,
                pipe_process_output(stdout, output, true),
                pipe_process_output(stderr, output, false),
                process.wait(),
            )?;
            Ok::<_, Status>(code)
        };
        tokio::select! {
            biased;
            _ = output.closed() => Err(Status::cancelled("process client disconnected")),
            _ = lease.closed() => Err(Status::cancelled("session closed")),
            result = streaming => result,
            _ = tokio::time::sleep(Duration::from_secs(24 * 60 * 60)) => Err(Status::deadline_exceeded("workspace process timed out")),
        }
    };
    let code = match result {
        Ok(code) => code,
        Err(error) => {
            process.cancel().await?;
            return Err(error);
        }
    };
    output
        .send(Ok(ExecResponse {
            payload: Some(exec_response::Payload::ExitCode(code)),
        }))
        .await
        .map_err(|_| Status::cancelled("process client disconnected"))
}

async fn pipe_process_output(
    mut reader: impl AsyncRead + Unpin,
    output: &mpsc::Sender<Result<ExecResponse, Status>>,
    stdout: bool,
) -> Result<(), Status> {
    let mut buffer = vec![0; 65536];
    loop {
        let size = reader
            .read(&mut buffer)
            .await
            .map_err(|_| Status::internal("process output failed"))?;
        if size == 0 {
            return Ok(());
        }
        let bytes = buffer[..size].to_vec();
        let payload = if stdout {
            exec_response::Payload::Stdout(bytes)
        } else {
            exec_response::Payload::Stderr(bytes)
        };
        output
            .send(Ok(ExecResponse {
                payload: Some(payload),
            }))
            .await
            .map_err(|_| Status::cancelled("process client disconnected"))?;
    }
}

fn attachment_id<T>(request: &Request<T>) -> Result<String, Status> {
    request
        .metadata()
        .get("mirror-attachment-id")
        .and_then(|value| value.to_str().ok())
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| Status::unauthenticated("mirror-attachment-id metadata is required"))
}
