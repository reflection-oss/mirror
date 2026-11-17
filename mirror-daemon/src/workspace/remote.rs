//! A live server connection; remote commands never execute on the daemon host.

use std::time::Duration;

use mirror_protocol::daemon::AgentConfiguration;
use mirror_sandbox::{
    run_command, sandboxd_client::SandboxdClient, SandboxStream, SandboxTransportError,
    MAX_MESSAGE_BYTES,
};
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
    /// Attach only to an explicit live endpoint, never one recovered from workspace metadata.
    pub async fn connect(endpoint: &str, session_id: &str) -> Result<Self, Status> {
        let url = parse_remote_endpoint(endpoint)?;
        let mut endpoint = Endpoint::from_shared(endpoint.to_owned())
            .map_err(|_| Status::invalid_argument("invalid remote endpoint"))?
            .connect_timeout(Duration::from_secs(30));
        if url.scheme() == "https" {
            endpoint = endpoint
                .tls_config(ClientTlsConfig::new().with_webpki_roots())
                .map_err(|_| Status::invalid_argument("invalid remote TLS configuration"))?;
        }
        let mut metadata = MetadataMap::new();
        metadata.insert(
            "x-mirror-session-id",
            session_id
                .parse()
                .map_err(|_| Status::invalid_argument("invalid remote session ID"))?,
        );
        let channel = endpoint
            .connect()
            .await
            .map_err(|_| Status::unavailable("remote server connection failed"))?;
        Ok(Self { channel, metadata })
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
