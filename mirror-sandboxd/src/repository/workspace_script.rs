use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use mirror_agent::system_prompt::EnvironmentInfo;
use nix::sys::signal::{killpg, Signal};
use nix::unistd::Pid;
use tokio::process::Child;
use tokio::signal::unix::{signal, SignalKind};

use super::repository_checkout::{elapsed_ms, git_command, GIT_TIME_LIMIT};
use super::{CheckoutStages, DISPOSABLE_PATHS_VARIABLE};
use crate::connection::process_supervision::{
    supervise_process, ProcessInterrupt, ProcessSupervision,
};
use crate::tool_runner::REQUEST_BYTES_VARIABLE;

#[derive(Debug, Clone, Copy)]
pub(crate) enum WorkspaceScript {
    Setup,
    Resume,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScriptRun {
    pub status: ScriptStatus,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScriptStatus {
    Exited(i32),
    TimedOut,
    Unfinished,
}

pub(crate) struct DefaultBranch {
    pub(crate) commit: String,
    pub(crate) has_setup: bool,
}

pub const SETUP_TIME_LIMIT: Duration = Duration::from_mins(20);
pub const RESUME_TIME_LIMIT: Duration = Duration::from_mins(1);

const WAKE_MARKER: &str = "mirror-workspace-resumed";

impl ScriptRun {
    pub fn outcome(&self) -> &'static str {
        match self.status {
            ScriptStatus::Exited(0) => "ok",
            ScriptStatus::TimedOut => "timed_out",
            ScriptStatus::Exited(_) | ScriptStatus::Unfinished => "failed",
        }
    }
}

impl WorkspaceScript {
    fn name(self) -> &'static str {
        match self {
            Self::Setup => "setup",
            Self::Resume => "resume",
        }
    }

    fn limit(self) -> Duration {
        match self {
            Self::Setup => SETUP_TIME_LIMIT,
            Self::Resume => RESUME_TIME_LIMIT,
        }
    }

    fn stored(self, checkout: &Path) -> PathBuf {
        checkout.join(format!(".git/mirror-{}", self.name()))
    }

    fn log(self) -> PathBuf {
        std::env::var_os("HOME")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!(".cache/mirror/{}.log", self.name()))
    }

    async fn store(self, checkout: &Path, commit: &str) -> bool {
        let stored = self.stored(checkout);
        let written = async {
            let file = fs::File::create(&stored)?;
            file.set_permissions(fs::Permissions::from_mode(0o755))?;
            let object = format!("{commit}:.mirror/{}", self.name());
            let mut command = git_command(checkout, &["cat-file", "blob", &object]);
            command.stdout(file);
            let status = tokio::time::timeout(GIT_TIME_LIMIT, command.status())
                .await
                .map_err(|_| io::Error::from(io::ErrorKind::TimedOut))??;
            Ok::<_, io::Error>(status.success())
        };
        let stored_script = written.await.unwrap_or(false);
        if !stored_script {
            let _ = fs::remove_file(&stored);
        }
        stored_script
    }

    pub(crate) async fn run(self, checkout: &Path) -> io::Result<ScriptRun> {
        let started = Instant::now();
        let status = match self.spawn(checkout) {
            Ok(child) => self.supervise(child).await?,
            Err(_) => ScriptStatus::Unfinished,
        };
        Ok(ScriptRun {
            status,
            elapsed_ms: elapsed_ms(started),
        })
    }

    fn spawn(self, checkout: &Path) -> io::Result<Child> {
        let log = self.log();
        if let Some(directory) = log.parent() {
            fs::create_dir_all(directory)?;
        }
        let output = fs::File::create(log)?;
        let mut command = std::process::Command::new(self.stored(checkout));
        command
            .current_dir(checkout)
            .env("MIRROR_SANDBOX", "1")
            .env_remove(REQUEST_BYTES_VARIABLE)
            .env_remove(DISPOSABLE_PATHS_VARIABLE)
            .stdin(Stdio::null())
            .stdout(output.try_clone()?)
            .stderr(output)
            .process_group(0);
        tokio::process::Command::from(command)
            .kill_on_drop(true)
            .spawn()
    }

    async fn supervise(self, mut child: Child) -> io::Result<ScriptStatus> {
        let Some(group) = child
            .id()
            .and_then(|pid| i32::try_from(pid).ok())
            .map(Pid::from_raw)
        else {
            child.kill().await?;
            return Ok(ScriptStatus::Unfinished);
        };
        let mut terminate = signal(SignalKind::terminate())?;
        let terminated = async {
            terminate.recv().await;
            let _ = killpg(group, Signal::SIGKILL);
        };
        match supervise_process(group, self.limit(), terminated, |_| child.wait()).await {
            ProcessSupervision::Completed(status) => Ok(status
                .ok()
                .and_then(|status| status.code())
                .map_or(ScriptStatus::Unfinished, ScriptStatus::Exited)),
            ProcessSupervision::Interrupted {
                reason: ProcessInterrupt::Deadline,
                ..
            } => Ok(ScriptStatus::TimedOut),
            ProcessSupervision::Interrupted {
                reason: ProcessInterrupt::Cancel,
                ..
            } => Err(io::Error::new(
                io::ErrorKind::Interrupted,
                format!(".mirror/{} was stopped by SIGTERM", self.name()),
            )),
        }
    }

    fn notice(self, run: &ScriptRun, commit: &str) -> Option<String> {
        let outcome = match run.status {
            ScriptStatus::Exited(0) => return None,
            ScriptStatus::TimedOut => format!(
                "was stopped at its {}-minute limit",
                self.limit().as_secs() / 60
            ),
            ScriptStatus::Exited(code) => format!("exited with status {code}"),
            ScriptStatus::Unfinished => "did not finish".to_string(),
        };
        Some(format!(
            "- `.mirror/{}` from the default branch at {} {outcome} after {} s. The session \
             started anyway. Its output is in {}.",
            self.name(),
            commit.get(..12).unwrap_or(commit),
            run.elapsed_ms / 1000,
            self.log().display(),
        ))
    }
}

pub(crate) async fn read_default_branch(checkout: &Path) -> Option<DefaultBranch> {
    let fetched = fs::read_to_string(checkout.join(".git/FETCH_HEAD")).ok()?;
    let commit = fetched.lines().last()?.split('\t').next()?;
    if commit.is_empty() || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    WorkspaceScript::Resume.store(checkout, commit).await;
    Some(DefaultBranch {
        commit: commit.to_string(),
        has_setup: WorkspaceScript::Setup.store(checkout, commit).await,
    })
}

pub(crate) async fn resume_after_wake(checkout: &Path) -> io::Result<Option<ScriptRun>> {
    let resume = WorkspaceScript::Resume;
    let marker = std::env::temp_dir().join(WAKE_MARKER);
    if fs::symlink_metadata(resume.stored(checkout)).is_err()
        || fs::symlink_metadata(&marker).is_ok()
    {
        return Ok(None);
    }
    let run = resume.run(checkout).await?;
    let _ = fs::write(marker, b"");
    Ok(Some(run))
}

pub(crate) fn report_failures(environment: &mut EnvironmentInfo, stages: &CheckoutStages) {
    let Some(commit) = stages.default_commit.as_deref() else {
        return;
    };
    let notices = [
        (WorkspaceScript::Setup, &stages.setup),
        (WorkspaceScript::Resume, &stages.resume),
    ]
    .into_iter()
    .filter_map(|(script, run)| script.notice(run.as_ref()?, commit))
    .collect::<Vec<_>>();
    if notices.is_empty() {
        return;
    }
    let status = environment
        .repository_status
        .get_or_insert_with(String::new);
    if !status.is_empty() {
        status.push_str("\n\n");
    }
    status.push_str("# Workspace scripts\n");
    status.push_str(&notices.join("\n"));
}
