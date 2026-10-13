use std::path::PathBuf;

use mirror_protocol::mirror::mirror_rpc_server;
use mirror_protocol::mirror::DeleteSessionRequest;
use mirror_protocol::mirror::DeleteSessionResponse;
use mirror_protocol::mirror::ForkSessionRequest;
use mirror_protocol::mirror::ForkSessionResponse;
use mirror_protocol::mirror::ListSessionsRequest;
use mirror_protocol::mirror::ListSessionsResponse;
use mirror_protocol::mirror::SessionScope;
use mirror_protocol::mirror::StartSessionRequest;
use mirror_protocol::mirror::StartSessionResponse;
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
}

impl LocalMirrorRpc {
    #[must_use]
    pub fn new(service: ReactAgentServiceHandle, database: PathBuf) -> Self {
        Self { service, database }
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
        let database = self.database.clone();
        let session_id = tokio::task::spawn_blocking(move || {
            let session_id = Uuid::new_v4().simple().to_string();
            SQLiteStore::fork(
                &database,
                &request.source_session_id,
                &request.at_entry_id,
                &session_id,
            )
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
