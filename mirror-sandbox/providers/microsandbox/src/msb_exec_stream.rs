//! The standard input and output of one `msb exec --stream` process.

use std::io;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use mirror_sandbox::SandboxTransportError;
use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;
use tokio::io::ReadBuf;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;

/// Reads the guest command's output and writes its input. Dropping the stream
/// kills `msb exec`, which ends the guest command.
pub(crate) struct MsbExecStream {
    input: ChildStdin,
    output: ChildStdout,
    /// Held for its `kill_on_drop`.
    _process: Child,
}

impl MsbExecStream {
    pub fn new(mut process: Child) -> Result<Self, SandboxTransportError> {
        let (Some(input), Some(output)) = (process.stdin.take(), process.stdout.take()) else {
            return Err(SandboxTransportError::Failed(
                "msb exec was started without piped standard streams".to_string(),
            ));
        };
        Ok(Self {
            input,
            output,
            _process: process,
        })
    }
}

impl AsyncRead for MsbExecStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.output).poll_read(cx, buffer)
    }
}

impl AsyncWrite for MsbExecStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.input).poll_write(cx, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.input).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.input).poll_shutdown(cx)
    }
}
