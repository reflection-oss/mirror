use std::path::PathBuf;

use mirror_protocol::mirror::mirror_rpc_server;
use mirror_protocol::mirror::DeleteSessionRequest;
use mirror_protocol::mirror::DeleteSessionResponse;
use mirror_protocol::mirror::ForkSessionRequest;
use mirror_protocol::mirror::ForkSessionResponse;
use mirror_protocol::mirror::GetSessionRequest;
use mirror_protocol::mirror::GetSessionResponse;
use mirror_protocol::mirror::ListSessionsRequest;
use mirror_protocol::mirror::ListSessionsResponse;
use mirror_protocol::mirror::SessionScope;
use mirror_protocol::mirror::StartSessionRequest;
use mirror_protocol::mirror::StartSessionResponse;
use mirror_protocol::mirror::WatchSessionsRequest;
use mirror_protocol::mirror::WatchSessionsResponse;
use mirror_protocol::mirror::{ForkWorkspaceRequest, ForkWorkspaceResponse};
use tonic::Request;
use tonic::Response;
use tonic::Status;
use uuid::Uuid;

use crate::local::SQLiteStore;
use crate::rpc::fork_status;
use crate::ReactAgentServiceHandle;

#[derive(Clone)]
pub struct LocalMirrorRpc {
    service: ReactAgentServiceHandle,
    database: PathBuf,
    workspace: Option<PathBuf>,
}

impl LocalMirrorRpc {
    #[must_use]
    pub fn new(
        service: ReactAgentServiceHandle,
        database: PathBuf,
        workspace: Option<PathBuf>,
    ) -> Self {
        Self {
            service,
            database,
            workspace,
        }
    }
}

#[tonic::async_trait]
impl mirror_rpc_server::MirrorRpc for LocalMirrorRpc {
    async fn start_session(
        &self,
        _request: Request<StartSessionRequest>,
    ) -> Result<Response<StartSessionResponse>, Status> {
        Err(single_session())
    }

    async fn list_sessions(
        &self,
        _request: Request<ListSessionsRequest>,
    ) -> Result<Response<ListSessionsResponse>, Status> {
        Err(single_session())
    }

    async fn delete_session(
        &self,
        _request: Request<DeleteSessionRequest>,
    ) -> Result<Response<DeleteSessionResponse>, Status> {
        Err(single_session())
    }

    async fn get_session(
        &self,
        _request: Request<GetSessionRequest>,
    ) -> Result<Response<GetSessionResponse>, Status> {
        Err(single_session())
    }

    type WatchSessionsStream = futures_util::stream::Empty<Result<WatchSessionsResponse, Status>>;

    async fn watch_sessions(
        &self,
        _request: Request<WatchSessionsRequest>,
    ) -> Result<Response<Self::WatchSessionsStream>, Status> {
        Err(single_session())
    }

    async fn fork_workspace(
        &self,
        request: Request<ForkWorkspaceRequest>,
    ) -> Result<Response<ForkWorkspaceResponse>, Status> {
        let request = request.into_inner();
        if request.source_session_id != self.service.session_id() {
            return Err(Status::not_found(
                "source session is not served by this worker",
            ));
        }
        Uuid::parse_str(&request.session_id)
            .map_err(|_| Status::invalid_argument("child session ID must be a UUID"))?;
        let workspace = self.workspace.clone().ok_or_else(|| {
            Status::failed_precondition("workspace forking is unavailable in a sandbox")
        })?;
        let service = self.service.clone();
        let database = self.database.clone();
        tokio::spawn(async move {
            let (head, workspace) = service
                .fork_workspace(
                    crate::Checkpoints::new(workspace),
                    PathBuf::from(request.directory),
                    request.name,
                )
                .await
                .map_err(|error| Status::failed_precondition(error.to_string()))?;
            tokio::task::spawn_blocking(move || {
                SQLiteStore::fork_at_head(
                    &database,
                    &request.source_session_id,
                    head,
                    &request.session_id,
                    &workspace,
                )
                .map_err(fork_status)?;
                Ok::<_, Status>(Response::new(ForkWorkspaceResponse {
                    workspace: workspace.to_string_lossy().into_owned(),
                }))
            })
            .await
            .map_err(|error| Status::internal(format!("local fork task failed: {error}")))?
        })
        .await
        .map_err(|error| Status::internal(format!("local fork task failed: {error}")))?
    }

    async fn fork_session(
        &self,
        request: Request<ForkSessionRequest>,
    ) -> Result<Response<ForkSessionResponse>, Status> {
        let request = request.into_inner();
        if request.source_session_id != self.service.session_id() {
            return Err(Status::not_found(format!(
                "this worker serves session {:?}, not {:?}",
                self.service.session_id(),
                request.source_session_id
            )));
        }
        match SessionScope::try_from(request.scope) {
            Ok(SessionScope::Unspecified | SessionScope::Private) => {}
            Ok(SessionScope::Public) => {
                return Err(Status::unimplemented(
                    "public sessions require a hosted Mirror server",
                ));
            }
            Err(_) => return Err(Status::invalid_argument("unknown session scope")),
        }
        if request.at_entry_id.is_empty() == request.after_entry_id.is_empty() {
            return Err(Status::invalid_argument(
                "specify exactly one fork boundary",
            ));
        }
        let session_id = if request.session_id.is_empty() {
            Uuid::new_v4().simple().to_string()
        } else {
            Uuid::parse_str(&request.session_id)
                .map_err(|_| Status::invalid_argument("child session ID must be a UUID"))?;
            request.session_id.clone()
        };
        let database = self.database.clone();
        let session_id = tokio::task::spawn_blocking(move || {
            if request.after_entry_id.is_empty() {
                SQLiteStore::fork(
                    &database,
                    &request.source_session_id,
                    &request.at_entry_id,
                    &session_id,
                )
            } else {
                SQLiteStore::fork_after_response(
                    &database,
                    &request.source_session_id,
                    &request.after_entry_id,
                    &session_id,
                )
            }
            .map_err(fork_status)?;
            Ok::<_, Status>(session_id)
        })
        .await
        .map_err(|error| Status::internal(format!("local fork task failed: {error}")))??;
        Ok(Response::new(ForkSessionResponse { session_id }))
    }
}

fn single_session() -> Status {
    Status::unimplemented("the local worker serves one session; use the session's AgentRPC")
}

#[cfg(test)]
#[path = "mirror_rpc_test.rs"]
mod tests;
