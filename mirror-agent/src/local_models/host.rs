//! Host-wide admission, independent of daemon profiles and frontend attachments.

use std::io;
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::time::Duration;

use mirror_protocol::local_models::{
    local_models_service_client::LocalModelsServiceClient,
    local_models_service_server::LocalModelsServiceServer, WatchRequest,
};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Status;

use super::host_state::Access;
use super::runtime::Runtime;
use super::service::Service;
use super::settings::Settings;
use super::LocalModelError;

/// The daemon owns this host lease; other profiles only receive peer access.
#[derive(Debug)]
pub struct Host {
    pub service: Service,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

/// All daemon profiles on a host share one model router.
pub const CONTROL_PORT: u16 = 18433;

/// How long a joining daemon waits for the owner's first state.
const JOIN_TIMEOUT: Duration = Duration::from_secs(5);

impl Host {
    /// Bind or join management without starting an engine or inspecting models.
    pub async fn bind(
        root: PathBuf,
        worker: PathBuf,
        binary: Option<PathBuf>,
        port: u16,
    ) -> Result<Self, LocalModelError> {
        if !root.is_absolute() || !worker.is_absolute() {
            return Err(LocalModelError::RelativePaths);
        }
        match TcpListener::bind((Ipv4Addr::LOCALHOST, port)).await {
            Ok(listener) => Self::own(listener, root, worker, binary).await,
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => Self::join(root, port).await,
            Err(source) => Err(LocalModelError::Bind { port, source }),
        }
    }

    /// Attach to the daemon already serving `port`, if it manages the same runtime directory.
    async fn join(root: PathBuf, port: u16) -> Result<Self, LocalModelError> {
        let mut peer = connect(port)
            .await
            .map_err(|_| LocalModelError::Unavailable)?;
        let first_state = async {
            peer.watch(WatchRequest {})
                .await
                .map_err(|_| LocalModelError::ForeignPort(port))?
                .into_inner()
                .message()
                .await
                .map_err(|_| LocalModelError::Unavailable)?
                .and_then(|reply| reply.state)
                .ok_or(LocalModelError::Unavailable)
        };
        let state = tokio::time::timeout(JOIN_TIMEOUT, first_state)
            .await
            .map_err(|_| LocalModelError::Unavailable)??;
        if state.root != root.to_string_lossy() {
            return Err(LocalModelError::ForeignPort(port));
        }
        Ok(Self {
            service: Service::peer(peer),
            shutdown: None,
            task: None,
        })
    }

    async fn own(
        listener: TcpListener,
        root: PathBuf,
        worker: PathBuf,
        binary: Option<PathBuf>,
    ) -> Result<Self, LocalModelError> {
        let saved = root.clone();
        // An unreadable saved file falls back to the defaults; starting the server still
        // reports it.
        let settings = tokio::task::spawn_blocking(move || Settings::load(&saved))
            .await?
            .unwrap_or_default();
        let port = listener
            .local_addr()
            .map_err(|source| LocalModelError::Bind { port: 0, source })?
            .port();
        let (runtime, reads) = Runtime::new(root, port, worker, binary, settings)?;
        let (commands, requests) = mpsc::channel(16);
        let service = Service::local(commands.clone(), reads.clone(), Access::Owner);
        let peer = Service::local(commands, reads, Access::Peer);
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            let (stop_actor, stopped_actor) = oneshot::channel();
            let mut actor = tokio::spawn(runtime.run(requests, stopped_actor));
            tokio::select! {
                _ = stopped => {},
                _ = tonic::transport::Server::builder()
                    .add_service(LocalModelsServiceServer::new(peer))
                    .serve_with_incoming(TcpListenerStream::new(listener)) => {},
                _ = &mut actor => return,
            }
            let _ = stop_actor.send(());
            let _ = actor.await;
        });
        Ok(Self {
            service,
            shutdown: Some(shutdown),
            task: Some(task),
        })
    }

    /// Reap an owned engine; closing a peer never stops another daemon's router.
    pub async fn close(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

/// A control client for the daemon serving `port` on loopback.
pub(super) async fn connect(
    port: u16,
) -> Result<LocalModelsServiceClient<tonic::transport::Channel>, Status> {
    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
        .map_err(|_| Status::invalid_argument("Invalid local control address"))?
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .map_err(|_| Status::unavailable("Local runtime is unavailable"))?;
    Ok(LocalModelsServiceClient::new(channel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use mirror_protocol::local_models::local_models_service_server::LocalModelsService;
    use mirror_protocol::local_models::{execute_request::Operation, ExecuteRequest};
    use pretty_assertions::assert_eq;

    /// The owner of a fresh runtime directory on a free port, and that port.
    async fn owner() -> (tempfile::TempDir, u16, Host) {
        let root = tempfile::tempdir().expect("private root");
        let port = portpicker::pick_unused_port().expect("free port");
        let host = Host::bind(root.path().into(), "/synthetic/worker".into(), None, port)
            .await
            .expect("owner");
        (root, port, host)
    }

    #[tokio::test]
    async fn another_profile_attaches_as_a_peer_who_cannot_stop_the_owner() {
        let (root, port, owner) = owner().await;
        let peer = Host::bind(root.path().into(), "/synthetic/worker".into(), None, port)
            .await
            .expect("second profile");
        let mut client = connect(port).await.expect("client");

        let state = client
            .watch(WatchRequest {})
            .await
            .expect("watch")
            .into_inner()
            .message()
            .await
            .expect("message")
            .expect("snapshot")
            .state
            .expect("state");
        let denied = client
            .execute(ExecuteRequest {
                operation: Some(Operation::Stop(())),
                ..Default::default()
            })
            .await
            .expect("stop response")
            .into_inner();

        assert!(!state.owned);
        assert_eq!(denied.error, LocalModelError::OwnerOnly.to_string());
        peer.close().await;
        assert_matches!(
            client.watch(WatchRequest {}).await,
            Ok(_),
            "a peer closing keeps the owner"
        );
        owner.close().await;
    }

    #[tokio::test]
    async fn the_owner_closing_stops_its_runtime_without_creating_models() {
        let (root, _port, owner) = owner().await;
        let service = owner.service.clone();

        owner.close().await;

        let started = LocalModelsService::execute(
            &service,
            tonic::Request::new(ExecuteRequest {
                operation: Some(Operation::Start(())),
                ..Default::default()
            }),
        )
        .await
        .expect("a local reply")
        .into_inner();
        assert_eq!(started.error, LocalModelError::Stopped.to_string());
        assert!(!root.path().join("models").exists());
    }

    #[tokio::test]
    async fn a_port_serving_another_runtime_directory_is_not_joined() {
        let (_root, port, _owner) = owner().await;
        let other = tempfile::tempdir().expect("other root");

        let joined = Host::bind(other.path().into(), "/synthetic/worker".into(), None, port).await;

        assert_matches!(joined, Err(LocalModelError::ForeignPort(joined_port)) if joined_port == port);
    }
}
