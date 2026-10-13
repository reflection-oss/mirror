//! Private Git and Jujutsu working-tree snapshots without changing checkout state.

use std::ffi::OsStr;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;

use mirror_protocol::Checkpoint;
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Captures repository state as unreferenced commits eligible for Git garbage collection.
#[derive(Debug)]
pub struct Checkpoints {
    workspace: PathBuf,
}

/// A repository could not be snapshotted safely.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointError {
    #[error("repository checkpoint I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("repository checkpoint Git {operation:?} failed: {detail}")]
    Git { operation: String, detail: String },
    #[error("repository checkpoint Jujutsu failed: {0}")]
    Jujutsu(String),
    #[error("repository checkpoint {0:?} timed out")]
    Timeout(String),
    #[error("repository checkpoint returned non-UTF-8 output")]
    Encoding(#[from] std::string::FromUtf8Error),
}

impl Checkpoints {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }

    /// Capture repository files, or return None outside Git and Jujutsu.
    pub async fn capture(&self) -> Result<Option<Checkpoint>, CheckpointError> {
        let workspace = tokio::fs::canonicalize(&self.workspace).await?;
        let jj_root = tokio::task::spawn_blocking(move || {
            workspace
                .ancestors()
                .find(|directory| crate::is_repository_root(directory))
                .filter(|directory| directory.join(".jj").is_dir())
                .map(Path::to_path_buf)
        })
        .await
        .map_err(io::Error::other)?;
        if let Some(root) = jj_root {
            let head_commit = jj_snapshot(&root, false).await?;
            let snapshot_commit = jj_snapshot(&root, true).await?;
            return Ok(Some(Checkpoint {
                repository_root: root.to_string_lossy().into_owned(),
                head_commit: Some(head_commit),
                snapshot_commit,
            }));
        }
        let git = Git {
            root: &self.workspace,
            index: None,
            objects: None,
        };
        let discovered = git.output(&["rev-parse", "--show-toplevel"], None).await?;
        if !discovered.status.success() {
            let error = String::from_utf8_lossy(&discovered.stderr);
            if error.contains("not a git repository")
                || error.contains("must be run in a work tree")
            {
                return Ok(None);
            }
        }
        let root = PathBuf::from(text("rev-parse", discovered)?);
        let git = Git {
            root: &root,
            index: None,
            objects: None,
        };
        let head = git.revision("HEAD").await?;
        let temporary = tempfile::tempdir()?;
        let index = temporary.path().join("index");
        git.copy_index(&index).await?;
        let destination = PathBuf::from(
            git.run(&[
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "objects",
            ])
            .await?,
        );
        let objects = temporary.path().join("objects");
        tokio::fs::create_dir_all(objects.join("info")).await?;
        let alternate = serde_json::to_string(&destination).map_err(io::Error::other)?;
        tokio::fs::write(objects.join("info/alternates"), format!("{alternate}\n")).await?;
        let snapshot = Git {
            root: &root,
            index: Some(&index),
            objects: Some(&objects),
        };
        snapshot.clear_index_flags().await?;
        snapshot
            .run(&["add", "--all", "--sparse", "--", "."])
            .await?;
        let tree = snapshot.run(&["write-tree"]).await?;
        let mut args = vec!["commit-tree", tree.as_str(), "-m", "Repository checkpoint"];
        if let Some(head) = &head {
            args.extend(["-p", head.as_str()]);
        }
        let commit = snapshot.run(&args).await?;
        tokio::task::spawn_blocking(move || promote_objects(&objects, &destination))
            .await
            .map_err(io::Error::other)??;
        Ok(Some(Checkpoint {
            repository_root: root.to_string_lossy().into_owned(),
            head_commit: head,
            snapshot_commit: commit,
        }))
    }
}

// Native snapshots preserve jj's sparse checkout, ignores, and conflict representation.
async fn jj_snapshot(root: &Path, snapshot: bool) -> Result<String, CheckpointError> {
    let mut command = Command::new("jj");
    #[cfg(test)]
    if let Some(binary) = std::env::var_os("JJ_TEST_BIN") {
        command = Command::new(std::fs::canonicalize(binary)?);
    }
    command
        .current_dir(root)
        // Keep the snapshot operation uncommitted: do not update @, bookmarks, or checkout state.
        .arg(if snapshot {
            "--no-integrate-operation"
        } else {
            "--ignore-working-copy"
        })
        .args([
            "--color=never",
            "--no-pager",
            "--config",
            "snapshot.max-new-file-size=0",
            "--config",
            "snapshot.auto-track=\"all()\"",
            "log",
            "--no-graph",
            "-r",
            "@",
            "-T",
            "commit_id",
        ])
        .env("LC_ALL", "C")
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(60), command.output())
        .await
        .map_err(|_| CheckpointError::Timeout("jj snapshot".to_string()))??;
    if !output.status.success() {
        return Err(CheckpointError::Jujutsu(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

struct Git<'a> {
    root: &'a Path,
    index: Option<&'a Path>,
    objects: Option<&'a Path>,
}

impl Git<'_> {
    async fn copy_index(&self, destination: &Path) -> Result<(), CheckpointError> {
        let source = PathBuf::from(
            self.run(&["rev-parse", "--path-format=absolute", "--git-path", "index"])
                .await?,
        );
        let destination = destination.to_path_buf();
        tokio::task::spawn_blocking(move || -> io::Result<()> {
            let metadata = match std::fs::metadata(&source) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error),
            };
            std::fs::copy(source, &destination)?;
            // Preserve the index timestamp so Git still detects racily clean entries.
            std::fs::File::options()
                .write(true)
                .open(destination)?
                .set_modified(metadata.modified()?)
        })
        .await
        .map_err(io::Error::other)?
        .map_err(Into::into)
    }

    async fn clear_index_flags(&self) -> Result<(), CheckpointError> {
        let tracked = self.output(&["ls-files", "-v", "-z"], None).await?;
        check("ls-files", &tracked)?;
        let root = self.root.to_path_buf();
        let (assumed, skipped) = tokio::task::spawn_blocking(move || -> io::Result<_> {
            let mut assumed = Vec::new();
            let mut skipped = Vec::new();
            for entry in tracked
                .stdout
                .split(|byte| *byte == 0)
                .filter(|entry| !entry.is_empty())
            {
                let path = &entry[2..];
                if entry[0].is_ascii_lowercase() {
                    assumed.extend_from_slice(path);
                    assumed.push(0);
                }
                if entry[0].eq_ignore_ascii_case(&b'S') {
                    // Missing sparse-checkout files are not deletions; materialized files may be edited.
                    match std::fs::symlink_metadata(root.join(OsStr::from_bytes(path))) {
                        Ok(_) => {
                            skipped.extend_from_slice(path);
                            skipped.push(0);
                        }
                        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error),
                    }
                }
            }
            Ok((assumed, skipped))
        })
        .await
        .map_err(io::Error::other)??;
        for (flag, paths) in [
            ("--no-assume-unchanged", assumed),
            ("--no-skip-worktree", skipped),
        ] {
            if !paths.is_empty() {
                let cleared = self
                    .output(&["update-index", flag, "-z", "--stdin"], Some(&paths))
                    .await?;
                check("update-index", &cleared)?;
            }
        }
        Ok(())
    }

    async fn output(&self, args: &[&str], input: Option<&[u8]>) -> Result<Output, CheckpointError> {
        let mut command = Command::new("git");
        command
            .current_dir(self.root)
            .args([
                "-c",
                "core.splitIndex=false",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
            ])
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_COMMON_DIR")
            .env_remove("GIT_NAMESPACE")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("GIT_AUTHOR_NAME", "Mirror")
            .env("GIT_AUTHOR_EMAIL", "mirror@localhost")
            .env("GIT_COMMITTER_NAME", "Mirror")
            .env("GIT_COMMITTER_EMAIL", "mirror@localhost")
            .env("LC_ALL", "C")
            .kill_on_drop(true);
        if let Some(index) = self.index {
            command.env("GIT_INDEX_FILE", index);
        }
        if let Some(objects) = self.objects {
            command.env("GIT_OBJECT_DIRECTORY", objects);
        }
        let operation = async {
            let Some(input) = input else {
                return command.output().await;
            };
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| io::Error::other("missing Git stdin"))?;
            let write = async move {
                let result = stdin.write_all(input).await;
                drop(stdin);
                result
            };
            let (written, output) = tokio::join!(write, child.wait_with_output());
            written?;
            output
        };
        tokio::time::timeout(Duration::from_secs(60), operation)
            .await
            .map_err(|_| CheckpointError::Timeout(args.join(" ")))?
            .map_err(Into::into)
    }

    async fn run(&self, args: &[&str]) -> Result<String, CheckpointError> {
        text(args[0], self.output(args, None).await?)
    }

    async fn revision(&self, name: &str) -> Result<Option<String>, CheckpointError> {
        let output = self
            .output(&["rev-parse", "--verify", "--quiet", name], None)
            .await?;
        if output.status.code() == Some(1) {
            return Ok(None);
        }
        text("rev-parse", output).map(Some)
    }
}

// Failed staging stays in temporary storage; publish only after every Git operation succeeds.
fn promote_objects(source: &Path, destination: &Path) -> io::Result<()> {
    for directory in std::fs::read_dir(source)? {
        let directory = directory?;
        if directory.file_name() == "info" {
            continue;
        }
        let target = destination.join(directory.file_name());
        std::fs::create_dir_all(&target)?;
        let mut files = std::fs::read_dir(directory.path())?.collect::<Result<Vec<_>, _>>()?;
        // A pack must be complete before its index makes it discoverable by other Git processes.
        files.sort_by_key(|file| file.path().extension() == Some(OsStr::new("idx")));
        for file in files {
            let mut input = std::fs::File::open(file.path())?;
            let mut output = tempfile::NamedTempFile::new_in(&target)?;
            io::copy(&mut input, &mut output)?;
            output
                .as_file()
                .set_permissions(input.metadata()?.permissions())?;
            match output.persist_noclobber(target.join(file.file_name())) {
                Ok(_) => {}
                Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.error),
            }
        }
    }
    Ok(())
}

fn check(operation: &str, output: &Output) -> Result<(), CheckpointError> {
    if !output.status.success() {
        return Err(CheckpointError::Git {
            operation: operation.to_string(),
            detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    Ok(())
}

fn text(operation: &str, output: Output) -> Result<String, CheckpointError> {
    check(operation, &output)?;
    Ok(String::from_utf8(output.stdout)?
        .trim_end_matches('\n')
        .to_string())
}

#[cfg(test)]
#[path = "checkpoints_test.rs"]
mod tests;
