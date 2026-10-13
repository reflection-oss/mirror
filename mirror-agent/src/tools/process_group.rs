//! Unix subprocess-group lifecycle helpers.

use std::process::ExitStatus;

use nix::sys::signal;
use nix::unistd::Pid;
use tokio::process::Child;

/// A subprocess group killed and reaped unless disarmed after completion.
pub(super) struct ProcessGroup {
    process_id: Option<u32>,
    child: Option<Child>,
    runtime: tokio::runtime::Handle,
}

impl ProcessGroup {
    pub(super) fn new(child: Child) -> Self {
        Self {
            process_id: child.id(),
            child: Some(child),
            runtime: tokio::runtime::Handle::current(),
        }
    }

    pub(super) fn child(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("child is only taken during drop")
    }

    pub(super) fn disarm(mut self) {
        self.process_id = None;
    }

    /// Kill the group and reap its leader.
    pub(super) async fn terminate(mut self) -> std::io::Result<ExitStatus> {
        kill_group(self.process_id);
        let status = self.child().wait().await;
        // On macOS a member forked while the group was being signalled can miss
        // the signal. Its membership keeps the group ID reserved, so signal the
        // group again now that the leader is gone.
        kill_group(self.process_id);
        self.disarm();
        status
    }
}

impl Drop for ProcessGroup {
    fn drop(&mut self) {
        kill_group(self.process_id);
        if let Some(mut child) = self.child.take().filter(|child| child.id().is_some()) {
            let process_id = self.process_id;
            // Tokio's orphan queue only offers best-effort reaping after cancellation.
            self.runtime.spawn(async move {
                let _ = child.wait().await;
                // Catch members that missed the first signal, as in `terminate`.
                kill_group(process_id);
            });
        }
    }
}

/// Send SIGKILL to every process in the group led by `process_id`.
fn kill_group(process_id: Option<u32>) {
    if let Some(process_id) = process_id {
        let _ = signal::kill(Pid::from_raw(-(process_id as i32)), signal::Signal::SIGKILL);
    }
}

#[cfg(unix)]
pub(super) fn return_code(status: ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;

    status
        .code()
        .or_else(|| status.signal().map(|signal| 128 + signal))
        .unwrap_or(2)
}

#[cfg(not(unix))]
pub(super) fn return_code(status: ExitStatus) -> i32 {
    status.code().unwrap_or(2)
}

#[cfg(test)]
#[path = "process_group_test.rs"]
mod tests;
