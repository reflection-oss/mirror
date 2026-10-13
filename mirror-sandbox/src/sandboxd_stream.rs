//! An open byte stream to a sandbox's `sandboxd`.

use std::fmt;

use tokio::io::AsyncRead;
use tokio::io::AsyncWrite;

/// The two directions of one stream to `sandboxd serve`. Dropping both halves
/// ends the stream, which `sandboxd` treats as end of input.
pub struct SandboxdStream {
    pub reader: Box<dyn AsyncRead + Send + Unpin>,
    pub writer: Box<dyn AsyncWrite + Send + Unpin>,
}

impl SandboxdStream {
    #[must_use]
    pub fn split<S>(stream: S) -> Self
    where
        S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
    {
        let (reader, writer) = tokio::io::split(stream);
        Self {
            reader: Box::new(reader),
            writer: Box::new(writer),
        }
    }
}

impl fmt::Debug for SandboxdStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SandboxdStream")
    }
}
