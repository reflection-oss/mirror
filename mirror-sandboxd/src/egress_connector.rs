use std::io;
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::task::{JoinHandle, JoinSet};
use tokio_rustls::TlsConnector;

use crate::java_truststore::write_java_truststore;
use mirror_sandbox::{EgressProxyConfig, MAX_EGRESS_IDENTITY_BYTES};

pub struct EgressConnector {
    address: SocketAddr,
    ca: tempfile::NamedTempFile,
    identity: watch::Sender<Option<Arc<ExecutionToken>>>,
    task: JoinHandle<()>,
}

struct ExecutionToken {
    value: String,
    expires_at: u64,
}

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_CONNECTIONS: usize = 64;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

impl EgressConnector {
    pub async fn start(config: EgressProxyConfig, java_truststore: &Path) -> io::Result<Self> {
        let proxy_port = u16::try_from(config.port).map_err(|_| invalid_config())?;
        if !valid_host(&config.host) || config.port == 0 || config.ca_pem.len() > 64 * 1024 {
            return Err(invalid_config());
        }
        let name = ServerName::try_from(config.host.clone()).map_err(|_| invalid_config())?;
        let certificates = rustls_pemfile::certs(&mut config.ca_pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid_config())?;
        let mut roots = RootCertStore::empty();
        for cert in &certificates {
            roots.add(cert.clone()).map_err(|_| invalid_config())?;
        }
        if roots.is_empty() {
            return Err(invalid_config());
        }
        write_java_truststore(java_truststore, &certificates)?;
        let tls = TlsConnector::from(Arc::new(
            ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        ));
        let mut ca = tempfile::Builder::new()
            .prefix("mirror-egress-ca-")
            .suffix(".pem")
            .tempfile()?;
        ca.write_all(config.ca_pem.as_bytes())?;
        ca.flush()?;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
        let address = listener.local_addr()?;
        let (identity, latest) = watch::channel(None);
        let task = tokio::spawn(async move {
            let mut connections = JoinSet::new();
            loop {
                tokio::select! {
                    connection = listener.accept(), if connections.len() < MAX_CONNECTIONS => {
                        let Ok((client, _)) = connection else { break };
                        let token = latest.borrow().clone();
                        let tls = tls.clone();
                        let name = name.clone();
                        let host = config.host.clone();
                        connections.spawn(async move {
                            let _ = relay(client, token, tls, name, &host, proxy_port).await;
                        });
                    }
                    _ = connections.join_next(), if !connections.is_empty() => {}
                }
            }
        });
        Ok(Self {
            address,
            ca,
            identity,
            task,
        })
    }

    pub fn install(&self, expires_at: u64, bytes: &[u8]) -> bool {
        if bytes.is_empty()
            || bytes.len() > MAX_EGRESS_IDENTITY_BYTES
            || !bytes
                .iter()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
            || remaining(expires_at).is_none_or(|lifetime| lifetime > Duration::from_mins(1))
        {
            return false;
        }
        let Ok(value) = String::from_utf8(bytes.to_vec()) else {
            return false;
        };
        self.identity
            .send_replace(Some(Arc::new(ExecutionToken { value, expires_at })));
        true
    }

    pub fn revoke(&self) {
        self.identity.send_replace(None);
        self.task.abort();
    }

    pub fn configure_command(&self, command: &mut Command) {
        let proxy = format!("http://{}", self.address);
        for key in [
            "HTTPS_PROXY",
            "https_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
        ] {
            command.env(key, &proxy);
        }
        command
            .env("GRPC_PROXY_EXP", self.address.to_string())
            .env("NO_PROXY", "")
            .env("no_proxy", "");
        for key in [
            "SSL_CERT_FILE",
            "GIT_SSL_CAINFO",
            "CURL_CA_BUNDLE",
            "REQUESTS_CA_BUNDLE",
            "NODE_EXTRA_CA_CERTS",
        ] {
            command.env(key, self.ca.path());
        }
        command
            .env("GH_TOKEN", "mirror-managed")
            .env("GH_ENTERPRISE_TOKEN", "mirror-managed")
            .env("BUILD_BUDDY_API_KEY", "mirror-managed");
    }
}

impl Drop for EgressConnector {
    fn drop(&mut self) {
        self.revoke();
    }
}

async fn relay(
    mut client: TcpStream,
    token: Option<Arc<ExecutionToken>>,
    tls: TlsConnector,
    name: ServerName<'static>,
    proxy_host: &str,
    proxy_port: u16,
) -> io::Result<()> {
    let handshake = async {
        let header = read_header(&mut client).await?;
        let target = connect_target(&header).ok_or_else(invalid_request)?;
        let token = token.ok_or_else(invalid_request)?;
        let socket = TcpStream::connect((proxy_host, proxy_port)).await?;
        let mut upstream = tls.connect(name, socket).await?;
        remaining(token.expires_at).ok_or_else(invalid_request)?;
        let request = format!(
            "CONNECT {target} HTTP/1.1\r\nHost: {target}\r\nProxy-Authorization: Bearer {}\r\n\r\n",
            token.value,
        );
        upstream.write_all(request.as_bytes()).await?;
        let response = read_header(&mut upstream).await?;
        if !response.starts_with(b"HTTP/1.1 200 ") && !response.starts_with(b"HTTP/1.0 200 ") {
            return Err(invalid_request());
        }
        client
            .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            .await?;
        Ok::<_, io::Error>(upstream)
    };
    let Ok(Ok(mut upstream)) = tokio::time::timeout(HANDSHAKE_TIMEOUT, handshake).await else {
        let _ = tokio::time::timeout(
            HANDSHAKE_TIMEOUT,
            client.write_all(
                b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            ),
        )
        .await;
        return Err(invalid_request());
    };
    let _ = tokio::time::timeout(
        Duration::from_hours(6),
        tokio::io::copy_bidirectional(&mut client, &mut upstream),
    )
    .await;
    Ok(())
}

async fn read_header(reader: &mut (impl AsyncRead + Unpin)) -> io::Result<Vec<u8>> {
    let mut header = Vec::new();
    while header.len() < MAX_HEADER_BYTES {
        header.push(reader.read_u8().await?);
        if header.ends_with(b"\r\n\r\n") {
            return Ok(header);
        }
    }
    Err(invalid_request())
}

fn connect_target(header: &[u8]) -> Option<&str> {
    let text = std::str::from_utf8(header).ok()?;
    let mut lines = text.split("\r\n");
    let line = lines.next()?;
    let target = line.strip_prefix("CONNECT ")?.strip_suffix(" HTTP/1.1")?;
    let (host, port) = target.rsplit_once(':')?;
    if !valid_host(host) || port != "443" {
        return None;
    }
    for line in lines.take_while(|line| !line.is_empty()) {
        let (key, value) = line.split_once(':')?;
        if key.is_empty()
            || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || value.bytes().any(|b| b.is_ascii_control() && b != b'\t')
            || key.eq_ignore_ascii_case("transfer-encoding")
            || (key.eq_ignore_ascii_case("content-length") && value.trim() != "0")
            || (key.eq_ignore_ascii_case("host") && ![target, host].contains(&value.trim()))
        {
            return None;
        }
    }
    Some(target)
}

fn valid_host(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        })
}

fn remaining(expires_at: u64) -> Option<Duration> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
    Duration::from_secs(expires_at)
        .checked_sub(now)
        .filter(|remaining| !remaining.is_zero())
}

fn invalid_config() -> io::Error {
    io::Error::other("invalid sandbox egress transport configuration")
}
fn invalid_request() -> io::Error {
    io::Error::other("sandbox egress connection denied")
}

#[cfg(test)]
#[path = "egress_connector_tests.rs"]
mod tests;
