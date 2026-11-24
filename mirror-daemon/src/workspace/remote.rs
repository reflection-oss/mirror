//! A live server connection; remote commands never execute on the daemon host.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mirror_protocol::daemon::AgentConfiguration;
use mirror_protocol::PersonalContext;
use mirror_sandbox::{
    run_command, sandboxd_client::SandboxdClient, SandboxStream, SandboxTransportError,
    MAX_MESSAGE_BYTES,
};
use tokio::sync::OnceCell;
use tonic::metadata::MetadataMap;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use tonic::Status;

use crate::{AgentProcess, AgentStartError};

use super::{Process, Workspace};

/// Server authority remains server-side; this workspace owns only its client connection.
#[derive(Debug)]
pub struct Remote {
    channel: Channel,
    metadata: MetadataMap,
}

impl Remote {
    pub fn lifecycle(
        &self,
    ) -> mirror_protocol::mirror::mirror_rpc_client::MirrorRpcClient<Channel> {
        mirror_protocol::mirror::mirror_rpc_client::MirrorRpcClient::new(self.channel.clone())
    }

    pub fn runtime(
        &self,
    ) -> mirror_protocol::mirror::mirror_runtime_client::MirrorRuntimeClient<Channel> {
        mirror_protocol::mirror::mirror_runtime_client::MirrorRuntimeClient::new(
            self.channel.clone(),
        )
    }

    /// Snapshot only instruction files on this host; never forward frontend configuration.
    pub async fn snapshot_context(home: &Path) -> Result<PersonalContext, Status> {
        let directory = home.join(".mirror");
        let personal_instructions = tokio::task::spawn_blocking(move || {
            mirror_agent::system_prompt::load_global_instructions(&directory)
        })
        .await
        .map_err(|_| Status::internal("instruction discovery failed"))?
        .map_err(|_| {
            Status::failed_precondition("Cannot load ~/.mirror instructions (maximum 64 KiB).")
        })?;
        Ok(PersonalContext {
            personal_instructions,
        })
    }
}

/// Keep the most recently used origin warm. Other live sessions own their channel clones.
#[derive(Clone, Debug, Default)]
pub struct RemoteConnections {
    current: Arc<Mutex<Option<Arc<RemoteOrigin>>>>,
}

#[derive(Debug)]
struct RemoteOrigin {
    url: url::Url,
    channel: OnceCell<Channel>,
}

impl RemoteConnections {
    /// Only the channel is shared; routing metadata belongs to the individual session.
    pub async fn connect(&self, endpoint: &str, session_id: &str) -> Result<Remote, Status> {
        let url = parse_remote_endpoint(endpoint)?;
        let mut metadata = MetadataMap::new();
        metadata.insert(
            "x-mirror-session-id",
            session_id
                .parse()
                .map_err(|_| Status::invalid_argument("invalid remote session ID"))?,
        );
        let origin = {
            let mut current = self.current.lock().expect("remote connection cache");
            if let Some(origin) = current.as_ref().filter(|origin| origin.url == url) {
                Arc::clone(origin)
            } else {
                let origin = Arc::new(RemoteOrigin {
                    url,
                    channel: OnceCell::new(),
                });
                *current = Some(Arc::clone(&origin));
                origin
            }
        };
        let connect = origin.channel.get_or_try_init(|| async {
            // The cached channel outlives network changes and sleep; keepalive pings make a
            // silently dead connection fail its calls and redial instead of hanging them.
            let mut endpoint = Endpoint::from_shared(origin.url.to_string())
                .map_err(|_| Status::invalid_argument("invalid remote endpoint"))?
                .connect_timeout(Duration::from_secs(30))
                .http2_keep_alive_interval(Duration::from_secs(20))
                .keep_alive_timeout(Duration::from_secs(10))
                .keep_alive_while_idle(true);
            if origin.url.scheme() == "https" {
                endpoint = endpoint
                    .tls_config(ClientTlsConfig::new().with_webpki_roots())
                    .map_err(|_| Status::invalid_argument("invalid remote TLS configuration"))?;
            }
            endpoint
                .connect()
                .await
                .map_err(|_| Status::unavailable("remote server connection failed"))
        });
        // The budget includes waiting behind another caller dialing this origin.
        let channel = tokio::time::timeout(Duration::from_secs(30), connect)
            .await
            .map_err(|_| Status::unavailable("remote server connection timed out"))??
            .clone();
        Ok(Remote { channel, metadata })
    }
}

/// Validate live origins and canonicalize host case, default ports and the root path.
pub fn parse_remote_endpoint(endpoint: &str) -> Result<url::Url, Status> {
    let url = url::Url::parse(endpoint)
        .map_err(|_| Status::invalid_argument("invalid remote endpoint"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err(Status::invalid_argument(
            "remote endpoint must be an HTTP(S) origin",
        ));
    }
    Ok(url)
}

#[tonic::async_trait]
impl Workspace for Remote {
    async fn start_agent(
        &self,
        _session_id: &str,
        _configuration: &AgentConfiguration,
        _managed_reference: &str,
    ) -> Result<(Option<AgentProcess>, Channel), AgentStartError> {
        Ok((None, self.channel.clone()))
    }

    async fn exec(&self, argv: &[String]) -> Result<Process, Status> {
        if argv.is_empty() {
            return Err(Status::invalid_argument("process requires an executable"));
        }
        let command = shlex::try_join(argv.iter().map(String::as_str))
            .map_err(|_| Status::invalid_argument("process arguments contain NUL"))?;
        let client = SandboxdClient::new(self.channel.clone())
            .max_encoding_message_size(MAX_MESSAGE_BYTES)
            .max_decoding_message_size(MAX_MESSAGE_BYTES);
        let metadata = self.metadata.clone();
        Ok(Process::run(move |pipes| async move {
            let result = run_command(
                async { Ok(client) },
                SandboxStream {
                    command,
                    workdir: None,
                    timeout: Duration::from_secs(24 * 60 * 60),
                    stdin: Some(Box::pin(pipes.stdin)),
                    stdout: Box::pin(pipes.stdout),
                    stderr: Some(Box::pin(pipes.stderr)),
                    cancellation: Box::pin(async {
                        let _ = pipes.cancellation.await;
                    }),
                },
                metadata,
                Duration::from_secs(60),
            )
            .await;
            match result {
                Ok(outcome) => Ok(outcome.exit_code),
                Err(SandboxTransportError::OutcomeUnknown(_)) => Err(Status::aborted(
                    "remote process completion is unknown; it was not replayed",
                )),
                Err(_) => Err(Status::failed_precondition("remote process failed")),
            }
        }))
    }

    async fn stop(&self) -> Result<(), Status> {
        // Closing a client workspace must not terminate the server-owned agent.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};

    #[tokio::test]
    async fn concurrent_sessions_share_a_socket_only_on_the_same_origin() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio_stream::{wrappers::TcpListenerStream, StreamExt};

        async fn server() -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let accepted = Arc::new(AtomicUsize::new(0));
            let count = accepted.clone();
            let incoming = TcpListenerStream::new(listener).map(move |socket| {
                if socket.is_ok() {
                    count.fetch_add(1, Ordering::Relaxed);
                }
                socket
            });
            let server = tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_routes(tonic::service::Routes::default())
                    .serve_with_incoming(incoming)
                    .await
                    .unwrap();
            });
            (origin, accepted, server)
        }

        let (first, first_count, first_server) = server().await;
        let (second, second_count, second_server) = server().await;
        let connections = RemoteConnections::default();
        let canonical = format!("{first}/");
        let (catalog, a) = tokio::join!(
            connections.connect(&first, ""),
            connections.connect(&canonical, "a"),
        );
        let catalog = catalog.unwrap();
        let a = a.unwrap();
        let b = connections.connect(&first, "b").await.unwrap();
        assert_eq!(a.metadata.get("x-mirror-session-id").unwrap(), "a");
        assert_eq!(b.metadata.get("x-mirror-session-id").unwrap(), "b");
        // An RPC round trip proves the server accepted the socket before we count it.
        async fn round_trip(remote: &Remote) {
            let status = remote
                .lifecycle()
                .list_sessions(mirror_protocol::mirror::ListSessionsRequest::default())
                .await
                .unwrap_err();
            assert_eq!(status.code(), tonic::Code::Unimplemented);
        }
        round_trip(&catalog).await;
        let other = connections.connect(&second, "c").await.unwrap();
        round_trip(&other).await;
        round_trip(&a).await;
        assert_eq!(first_count.load(Ordering::Relaxed), 1);
        assert_eq!(second_count.load(Ordering::Relaxed), 1);
        first_server.abort();
        second_server.abort();
    }

    #[test]
    fn remote_origins_share_default_ports_but_not_other_authorities() {
        let desktop = parse_remote_endpoint("https://fixture.invalid").unwrap();
        assert_eq!(
            parse_remote_endpoint("https://FIXTURE.invalid:443/").unwrap(),
            desktop
        );
        assert_ne!(
            parse_remote_endpoint("https://fixture.invalid:8443").unwrap(),
            desktop
        );
        assert_ne!(
            parse_remote_endpoint("http://fixture.invalid").unwrap(),
            desktop
        );
        assert_ne!(
            parse_remote_endpoint("https://other.invalid").unwrap(),
            desktop
        );
        assert_eq!(
            parse_remote_endpoint("http://fixture.invalid:80").unwrap(),
            parse_remote_endpoint("http://fixture.invalid/").unwrap()
        );
    }

    #[test]
    fn remote_origins_reject_credentials_and_non_origin_components() {
        assert_matches!(
            parse_remote_endpoint("https://user:synthetic@fixture.invalid"),
            Err(_)
        );
        assert_matches!(
            parse_remote_endpoint("https://fixture.invalid/path"),
            Err(_)
        );
        assert_matches!(
            parse_remote_endpoint("https://fixture.invalid/?token=synthetic"),
            Err(_)
        );
        assert_matches!(
            parse_remote_endpoint("https://fixture.invalid/#fragment"),
            Err(_)
        );
        assert_matches!(parse_remote_endpoint("file:///tmp/fixture"), Err(_));
    }
}
