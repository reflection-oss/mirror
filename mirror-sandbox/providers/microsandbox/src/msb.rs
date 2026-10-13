//! The `msb` command its user installs; Mirror does not ship microsandbox.

use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

/// A checked `msb` executable.
#[derive(Clone, Debug)]
pub struct Msb {
    program: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum MsbSetupError {
    #[error(
        "msb was not found on PATH; install microsandbox {MINIMUM_MSB_VERSION} or newer, \
         or set {MSB_VARIABLE} to its path"
    )]
    NotFound,
    #[error("{program:?} --version failed: {detail}")]
    Unusable { program: PathBuf, detail: String },
    #[error("msb {found} is older than {MINIMUM_MSB_VERSION}, the oldest Mirror supports")]
    TooOld { found: String },
    #[error(
        "microsandbox needs read and write access to /dev/kvm ({0}); \
         add your user to the kvm group and log in again"
    )]
    Kvm(std::io::Error),
}

/// Why an `msb` command did not succeed.
#[derive(Debug)]
pub(crate) enum MsbFailure {
    /// The named sandbox does not exist.
    SandboxMissing,
    /// `create` found a sandbox with the name already.
    SandboxExists,
    /// The command was killed after its timeout; its effect is unknown.
    TimedOut(String),
    Failed(String),
}

/// The environment variable that names the `msb` to use instead of `PATH`.
pub const MSB_VARIABLE: &str = "MIRROR_MSB";
pub const MINIMUM_MSB_VERSION: MsbVersion = MsbVersion(0, 7, 7);
/// How `msb` reports a command on a sandbox that does not exist.
const MISSING_SANDBOX: &str = "sandbox not found";
/// How `msb create` reports a name that is taken, even with `--quiet`.
const EXISTING_SANDBOX: &str = "sandbox already exists";
const VERSION_TIMEOUT: Duration = Duration::from_secs(10);

/// A `major.minor.patch` release of microsandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MsbVersion(pub u64, pub u64, pub u64);

impl Msb {
    /// Uses `$MIRROR_MSB` or the first `msb` on `PATH`, after checking that
    /// this host can run microVMs.
    pub async fn locate() -> Result<Self, MsbSetupError> {
        check_virtualization()?;
        let program = match std::env::var_os(MSB_VARIABLE) {
            Some(program) => PathBuf::from(program),
            None => find_on_path("msb").ok_or(MsbSetupError::NotFound)?,
        };
        Self::at(program).await
    }

    /// Uses `program` after checking its version. Unlike [`Msb::locate`], this
    /// does not check the host's virtualization support.
    pub async fn at(program: PathBuf) -> Result<Self, MsbSetupError> {
        let msb = Self { program };
        let output = msb
            .run(vec!["--version".into()], VERSION_TIMEOUT)
            .await
            .map_err(|failure| MsbSetupError::Unusable {
                program: msb.program.clone(),
                detail: failure.to_string(),
            })?;
        let reported = String::from_utf8_lossy(&output);
        let found = reported.split_whitespace().nth(1).unwrap_or_default();
        let Some(version) = MsbVersion::parse(found) else {
            return Err(MsbSetupError::Unusable {
                program: msb.program,
                detail: format!("unrecognized version output {:?}", reported.trim()),
            });
        };
        // A pre-release precedes its release, so `0.7.7-rc.1` is older than
        // 0.7.7; build metadata after `+` does not change the order.
        let prerelease = found.split('+').next().unwrap_or_default().contains('-');
        if version < MINIMUM_MSB_VERSION || (prerelease && version == MINIMUM_MSB_VERSION) {
            return Err(MsbSetupError::TooOld {
                found: found.to_string(),
            });
        }
        Ok(msb)
    }

    /// Runs one command and returns its standard output.
    pub(crate) async fn run(
        &self,
        arguments: Vec<OsString>,
        timeout: Duration,
    ) -> Result<Vec<u8>, MsbFailure> {
        let subcommand = arguments
            .first()
            .map(|argument| argument.to_string_lossy().into_owned())
            .unwrap_or_default();
        let child = self
            .command()
            .args(arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                MsbFailure::Failed(format!("{:?} could not start: {error}", self.program))
            })?;
        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Err(_) => {
                return Err(MsbFailure::TimedOut(format!(
                    "msb {subcommand} did not finish within {timeout:?}"
                )))
            }
            Ok(Err(error)) => {
                return Err(MsbFailure::Failed(format!(
                    "msb {subcommand} could not be awaited: {error}"
                )))
            }
            Ok(Ok(output)) => output,
        };
        if output.status.success() {
            return Ok(output.stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains(MISSING_SANDBOX) {
            return Err(MsbFailure::SandboxMissing);
        }
        if stderr.contains(EXISTING_SANDBOX) {
            return Err(MsbFailure::SandboxExists);
        }
        Err(MsbFailure::Failed(format!(
            "msb {subcommand} exited with {}: {}",
            output.status,
            stderr.trim()
        )))
    }

    /// A command for `msb` that is killed when its handle is dropped.
    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.kill_on_drop(true);
        command
    }
}

impl fmt::Display for MsbFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SandboxMissing => f.write_str(MISSING_SANDBOX),
            Self::SandboxExists => f.write_str(EXISTING_SANDBOX),
            Self::TimedOut(detail) | Self::Failed(detail) => f.write_str(detail),
        }
    }
}

impl MsbVersion {
    /// Parses `major.minor.patch`, ignoring any pre-release or build suffix.
    pub fn parse(version: &str) -> Option<Self> {
        let release = version.split(['-', '+']).next()?;
        let mut parts = release.split('.').map(|part| part.parse::<u64>().ok());
        let version = Self(parts.next()??, parts.next()??, parts.next()??);
        parts.next().is_none().then_some(version)
    }
}

impl fmt::Display for MsbVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

/// microsandbox uses KVM on Linux; on macOS it uses the Hypervisor framework,
/// whose entitlement `msb` carries itself.
fn check_virtualization() -> Result<(), MsbSetupError> {
    if cfg!(target_os = "linux") {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kvm")
            .map_err(MsbSetupError::Kvm)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "msb_test.rs"]
mod tests;
