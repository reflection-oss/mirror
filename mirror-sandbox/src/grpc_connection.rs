//! gRPC over a provider-owned byte stream, including exec stdin/stdout.
use crate::{
    sandboxd_client::SandboxdClient,
    sandboxd_server::{Sandboxd, SandboxdServer},
    MAX_MESSAGE_BYTES,
};
use hyper_util::rt::TokioIo;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::watch;
use tonic::transport::{Channel, Endpoint, Server};

struct GrpcConnection<R, W> {
    input: R,
    output: W,
    closing: watch::Sender<bool>,
}
impl<R, W> Drop for GrpcConnection<R, W> {
    fn drop(&mut self) {
        self.closing.send_replace(true);
    }
}
impl<R, W> tonic::transport::server::Connected for GrpcConnection<R, W> {
    type ConnectInfo = ();
    fn connect_info(&self) {}
}
impl<R: AsyncRead + Unpin, W: Unpin> AsyncRead for GrpcConnection<R, W> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if buffer.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        let before = buffer.filled().len();
        let result = Pin::new(&mut self.input).poll_read(cx, buffer);
        if result.is_ready()
            && (matches!(result, Poll::Ready(Err(_))) || buffer.filled().len() == before)
        {
            self.closing.send_replace(true);
        }
        result
    }
}
impl<R: Unpin, W: AsyncWrite + Unpin> AsyncWrite for GrpcConnection<R, W> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        let result = Pin::new(&mut self.output).poll_write(cx, buffer);
        if matches!(result, Poll::Ready(Err(_))) {
            self.closing.send_replace(true);
        }
        result
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.output).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.output).poll_shutdown(cx)
    }
}

/// The connector can consume this stream once; it never reconnects or replays work.
pub async fn connect<R, W>(
    input: R,
    output: W,
    closing: watch::Sender<bool>,
) -> Result<SandboxdClient<Channel>, tonic::transport::Error>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let mut stream = Some(GrpcConnection {
        input,
        output,
        closing,
    });
    let channel = Endpoint::from_static("http://sandboxd")
        .connect_with_connector(tower::service_fn(move |_| {
            let stream = stream.take();
            async move {
                stream.map(TokioIo::new).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotConnected,
                        "sandboxd exec stream cannot reconnect",
                    )
                })
            }
        }))
        .await?;
    Ok(SandboxdClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_BYTES)
        .max_encoding_message_size(MAX_MESSAGE_BYTES))
}

/// Serve exactly one provider connection; losing it stops the service.
pub async fn serve_connection<R, W, S>(
    input: R,
    output: W,
    service: S,
    closing: watch::Sender<bool>,
) -> io::Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
    S: Sandboxd,
{
    use futures_util::StreamExt;
    let mut closed = closing.subscribe();
    let mut shutdown = closing.subscribe();
    let stream = GrpcConnection {
        input,
        output,
        closing,
    };
    let incoming = futures_util::stream::once(async { Ok::<_, io::Error>(stream) })
        .chain(futures_util::stream::pending());
    let server = Server::builder()
        .add_service(
            SandboxdServer::new(service)
                .max_decoding_message_size(MAX_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_MESSAGE_BYTES),
        )
        .serve_with_incoming_shutdown(incoming, async move {
            let _ = shutdown.wait_for(|closed| *closed).await;
        });
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result.map_err(io::Error::other),
        () = async { let _ = closed.wait_for(|closed| *closed).await; } => {
            tokio::time::timeout(std::time::Duration::from_secs(5), server).await
                .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "sandboxd gRPC shutdown timed out"))?
                .map_err(io::Error::other)
        }
    }
}
