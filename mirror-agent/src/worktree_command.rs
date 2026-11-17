//! Cross-process Git worktree admission shared with the CLI's repository lock.

use std::fs::{File, TryLockError};
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::process::{Output, Stdio};
use std::time::Duration;

use nix::fcntl::OFlag;
use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::oneshot;

/// Serialize additions against additions, listings and removals in every daemon profile.
pub async fn run_worktree_command(
    common: &Path,
    mut command: Command,
    exclusive: bool,
) -> io::Result<Output> {
    let file = acquire(common, exclusive).await?;
    // Worktree commands take no input; stdin keeps the lock alive even if the parent exits.
    let mut child = command
        .stdin(Stdio::from(file.try_clone()?))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true)
        .spawn()?;
    let (cancel, cancelled) = oneshot::channel::<()>();
    // The supervisor, not the caller, owns the file until Git and its children are reaped.
    let task = tokio::spawn(async move {
        let _file = file;
        let group = child.id().map(|id| Pid::from_raw(id as i32));
        let result = tokio::select! {
            _ = cancelled => Err(io::Error::new(io::ErrorKind::Interrupted, "worktree command cancelled")),
            result = tokio::time::timeout(Duration::from_secs(120), output(&mut child)) => {
                result.unwrap_or_else(|_| Err(io::Error::new(io::ErrorKind::TimedOut, "worktree command timed out")))
            }
        };
        if let Some(group) = group {
            let _ = killpg(group, Signal::SIGKILL);
        }
        child.kill().await?;
        result
    });
    let result = task.await.map_err(io::Error::other)?;
    drop(cancel);
    result
}

async fn acquire(common: &Path, exclusive: bool) -> io::Result<File> {
    let path = common.join("mirror-worktrees.lock");
    let file = tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .custom_flags((OFlag::O_NOFOLLOW | OFlag::O_NONBLOCK).bits())
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Err(io::Error::other("worktree lock is not a regular file"));
        }
        Ok::<_, io::Error>(file)
    })
    .await??;
    // Nonblocking attempts keep cancellation from stranding a blocking-pool thread.
    tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            let result = if exclusive {
                file.try_lock()
            } else {
                file.try_lock_shared()
            };
            match result {
                Ok(()) => return Ok(file),
                Err(TryLockError::Error(error)) => return Err(error),
                Err(TryLockError::WouldBlock) => {
                    tokio::time::sleep(Duration::from_millis(20)).await
                }
            }
        }
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "worktree lock timed out"))?
}

async fn output(child: &mut Child) -> io::Result<Output> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("missing Git stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("missing Git stderr"))?;
    let mut stdout = stdout.take(1024 * 1024 + 1);
    let mut stderr = stderr.take(1024 * 1024 + 1);
    let mut out = Vec::new();
    let mut err = Vec::new();
    tokio::try_join!(stdout.read_to_end(&mut out), stderr.read_to_end(&mut err))?;
    if out.len() > 1024 * 1024 || err.len() > 1024 * 1024 {
        return Err(io::Error::other(
            "worktree command output exceeded its bound",
        ));
    }
    Ok(Output {
        status: child.wait().await?,
        stdout: out,
        stderr: err,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use std::io::{Read, Write};
    use tokio::io::AsyncWriteExt;
    use tokio::net::UnixListener;

    fn probe(common: &Path, exclusive: bool) -> Command {
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .args([
                "--exact",
                "worktree_command::tests::lock_probe",
                "--nocapture",
            ])
            .env("MIRROR_LOCK_TEST_DIRECTORY", common)
            .env("MIRROR_LOCK_TEST_EXCLUSIVE", exclusive.to_string());
        command
    }

    #[test]
    fn lock_probe() {
        let Some(directory) = std::env::var_os("MIRROR_LOCK_TEST_DIRECTORY") else {
            return;
        };
        let directory = std::path::PathBuf::from(directory);
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(directory.join("mirror-worktrees.lock"))
            .expect("repository lock");
        assert_matches!(file.try_lock(), Err(TryLockError::WouldBlock));
        if std::env::var("MIRROR_LOCK_TEST_EXCLUSIVE").expect("mode") == "true" {
            assert_matches!(file.try_lock_shared(), Err(TryLockError::WouldBlock));
        } else {
            file.try_lock_shared()
                .expect("concurrent listings are allowed");
        }
        let mut socket = std::os::unix::net::UnixStream::connect(directory.join("ready"))
            .expect("parent socket");
        socket.write_all(b"ready").expect("signal admission");
        let mut release = [0];
        socket.read_exact(&mut release).expect("parent release");
    }

    #[tokio::test]
    async fn shared_commands_admit_other_readers_but_exclude_additions() {
        let directory = tempfile::tempdir().expect("fixture");
        let listener = UnixListener::bind(directory.path().join("ready")).expect("socket");
        let common = directory.path().to_owned();
        let command = probe(&common, false);
        let task = tokio::spawn(async move { run_worktree_command(&common, command, false).await });
        let (mut socket, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
            .await
            .expect("child admission")
            .expect("child connects");
        let mut ready = [0; 5];
        socket
            .read_exact(&mut ready)
            .await
            .expect("child has checked the lock");
        assert_eq!(&ready, b"ready");
        socket.write_all(b"x").await.expect("release child");
        assert!(task
            .await
            .expect("supervisor")
            .expect("command")
            .status
            .success());
        let file = File::options()
            .append(true)
            .open(directory.path().join("mirror-worktrees.lock"))
            .expect("lock file");
        file.try_lock().expect("command released the lock");
    }

    #[tokio::test]
    async fn cancelled_addition_reaps_child_before_releasing_the_lock() {
        let directory = tempfile::tempdir().expect("fixture");
        let listener = UnixListener::bind(directory.path().join("ready")).expect("socket");
        let common = directory.path().to_owned();
        let command = probe(&common, true);
        let task = tokio::spawn(async move { run_worktree_command(&common, command, true).await });
        let (mut socket, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
            .await
            .expect("child admission")
            .expect("child connects");
        let mut ready = [0; 5];
        socket
            .read_exact(&mut ready)
            .await
            .expect("child has checked the lock");
        assert_eq!(&ready, b"ready");
        task.abort();
        assert_matches!(task.await, Err(error) if error.is_cancelled());
        assert_eq!(socket.read(&mut ready).await.expect("child is stopped"), 0);
        let mut command = Command::new(std::env::current_exe().expect("test executable"));
        command
            .args(["--exact", "worktree_command::tests::lock_probe"])
            .env_remove("MIRROR_LOCK_TEST_DIRECTORY");
        assert!(run_worktree_command(directory.path(), command, true)
            .await
            .expect("lock is released after reaping")
            .status
            .success());
    }
}
