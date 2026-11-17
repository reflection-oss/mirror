//! Cancellation and cleanup ownership for an attachment-authorized operation.

use tokio::sync::watch;

/// Either attachment loss or worker shutdown cancels execution; dropping releases cleanup.
#[derive(Debug)]
pub struct ExecutionLease {
    pub attachment: watch::Receiver<()>,
    pub worker: watch::Receiver<()>,
}

impl ExecutionLease {
    pub fn is_closed(&self) -> bool {
        self.attachment.has_changed().is_err() || self.worker.has_changed().unwrap_or(true)
    }

    pub async fn closed(&mut self) {
        tokio::select! {
            _ = self.attachment.changed() => {},
            _ = self.worker.changed() => {},
        }
    }
}
