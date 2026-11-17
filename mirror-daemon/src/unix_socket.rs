//! Private local listeners and connections; existing endpoints are never replaced.

use std::io;
use std::path::PathBuf;

use hyper_util::rt::TokioIo;
use nix::unistd::geteuid;
use tokio::net::{UnixListener, UnixStream};
use tonic::transport::{Channel, Endpoint};
use tower::service_fn;

/// Admit only the current user's peers, including on platforms without socket mode checks.
pub fn local_incoming(
    listener: UnixListener,
) -> impl futures_core::Stream<Item = io::Result<UnixStream>> {
    async_stream::try_stream! {
        loop {
            let (stream, _) = listener.accept().await?;
            if stream.peer_cred().is_ok_and(|peer| peer.uid() == geteuid().as_raw()) {
                yield stream;
            }
        }
    }
}

/// Connect to a local worker without permitting HTTP proxy or TCP fallback.
pub async fn local_channel(path: PathBuf) -> Result<Channel, tonic::transport::Error> {
    Endpoint::from_static("http://localhost")
        .connect_with_connector(service_fn(move |_| {
            let path = path.clone();
            async move {
                let stream = UnixStream::connect(path).await?;
                if stream.peer_cred()?.uid() != geteuid().as_raw() {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "foreign socket peer",
                    ));
                }
                Ok::<_, io::Error>(TokioIo::new(stream))
            }
        }))
        .await
}
