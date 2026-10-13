//! Where a captured stream's kept text lives.

use std::fs::File;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;

/// A captured stream's kept text: in an unlinked file, or in memory when no
/// file can be written, so a command still runs on a read-only `/tmp`.
#[derive(Debug)]
pub(super) enum CaptureStorage {
    /// An unlinked file, and how many bytes of it were written whole; a write
    /// that failed partway may leave more.
    File {
        file: File,
        written: u64,
    },
    Memory(Vec<u8>),
    /// Kept text that was lost: its file failed and could not be read back.
    Failed {
        kind: std::io::ErrorKind,
        message: String,
    },
}

impl CaptureStorage {
    /// Storage in a new unlinked file in the temporary directory, or in memory
    /// when that cannot be created.
    pub(super) fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("mirror-bash-{}.txt", uuid::Uuid::new_v4().simple()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path);
        match file {
            Ok(file) => {
                // Unlinked at once, so the file goes with its handle however
                // the command ends.
                let _ = std::fs::remove_file(&path);
                Self::File { file, written: 0 }
            }
            Err(_) => Self::Memory(Vec::new()),
        }
    }

    /// Append `bytes`, moving to memory if the file fails.
    pub(super) fn write(&mut self, bytes: &[u8]) {
        match self {
            Self::File { file, written } => match file.write_all(bytes) {
                Ok(()) => *written += bytes.len() as u64,
                Err(_) => self.fall_back_to_memory(bytes),
            },
            Self::Memory(memory) => memory.extend_from_slice(bytes),
            Self::Failed { .. } => {}
        }
    }

    /// Move to memory the bytes written whole, then `bytes`, the write that
    /// failed; whatever of `bytes` reached the file is not read back.
    fn fall_back_to_memory(&mut self, bytes: &[u8]) {
        let state = match self.read_all() {
            Ok(mut kept) => {
                kept.extend_from_slice(bytes);
                Self::Memory(kept)
            }
            Err(error) => Self::Failed {
                kind: error.kind(),
                message: error.to_string(),
            },
        };
        *self = state;
    }

    /// Drop the kept text.
    pub(super) fn clear(&mut self) {
        match self {
            Self::File { file, written } => {
                if *written == 0 {
                    return;
                }
                if file.set_len(0).and_then(|()| file.rewind()).is_ok() {
                    *written = 0;
                } else {
                    *self = Self::Memory(Vec::new());
                }
            }
            Self::Memory(memory) => memory.clear(),
            // Nothing that was lost is needed any more.
            Self::Failed { .. } => *self = Self::Memory(Vec::new()),
        }
    }

    /// The kept text.
    ///
    /// # Errors
    ///
    /// Returns the error that lost kept text, or a failure to read the file.
    pub(super) fn read_all(&mut self) -> std::io::Result<Vec<u8>> {
        match self {
            Self::File { file, written } => {
                let mut bytes = Vec::new();
                file.seek(SeekFrom::Start(0))?;
                Read::by_ref(file).take(*written).read_to_end(&mut bytes)?;
                file.seek(SeekFrom::End(0))?;
                Ok(bytes)
            }
            Self::Memory(memory) => Ok(memory.clone()),
            Self::Failed { kind, message } => Err(std::io::Error::new(
                *kind,
                format!("captured output was lost: {message}"),
            )),
        }
    }
}

#[cfg(test)]
#[path = "capture_storage_test.rs"]
mod tests;
