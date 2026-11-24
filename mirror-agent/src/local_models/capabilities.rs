//! What the installed llama-server reports about itself: its flags and its devices.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

use super::engine::ENGINE_PATH;
use super::executable::serve_prefix;

/// Flags read from `llama-server --help`.
///
/// llama.cpp adds and renames flags between releases. Probing the installed
/// build, rather than assuming one, lets Mirror drop optional hardening flags a
/// build lacks and name the missing feature when a required one is absent.
#[derive(Debug)]
pub struct Capabilities {
    flags: HashSet<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum CapabilityError {
    #[error("{} {flag} timed out", .binary.display())]
    TimedOut { binary: PathBuf, flag: &'static str },
    #[error("Could not run {} {flag}: {source}", .binary.display())]
    Unrunnable {
        binary: PathBuf,
        flag: &'static str,
        source: std::io::Error,
    },
    #[error("llama-server --help lists no flags")]
    NoFlags,
    #[error("Incompatible llama-server (missing {}). Update llama.cpp or Mirror.", .0.join(", "))]
    Missing(Vec<String>),
}

/// How long `llama-server --help` may take.
const REPORT_TIMEOUT: Duration = Duration::from_secs(10);

impl Capabilities {
    /// Ask the build what it supports.
    pub async fn probe(binary: &Path) -> Result<Self, CapabilityError> {
        Self::from_help(&run_report(binary, "--help").await?)
    }

    /// The flags `help` lists; help that lists none was not read.
    pub fn from_help(help: &str) -> Result<Self, CapabilityError> {
        let mut flags = HashSet::new();
        for token in help.split(|c: char| c.is_whitespace() || c == ',') {
            let Some(name) = token.strip_prefix("--") else {
                continue;
            };
            // A negatable flag is documented once as `--[no-]name`.
            let (negatable, name) = match name.strip_prefix("[no-]") {
                Some(name) => (true, name),
                None => (false, name),
            };
            if !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            {
                flags.insert(format!("--{name}"));
                if negatable {
                    flags.insert(format!("--no-{name}"));
                }
            }
        }
        if flags.is_empty() {
            return Err(CapabilityError::NoFlags);
        }
        Ok(Self { flags })
    }

    /// Whether the build lists `flag`.
    pub fn supports(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }

    /// Name every missing flag, since each marks a feature the build lacks.
    pub fn require(&self, flags: &[&str]) -> Result<(), CapabilityError> {
        let missing: Vec<String> = flags
            .iter()
            .filter(|flag| !self.supports(flag))
            .map(|flag| (*flag).to_owned())
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        Err(CapabilityError::Missing(missing))
    }
}

/// The largest device memory llama.cpp reports, in bytes.
///
/// Asking llama.cpp keeps Mirror from guessing at GPUs, unified memory or offload limits.
pub async fn device_memory(binary: &Path) -> Option<u64> {
    parse_device_memory(&run_report(binary, "--list-devices").await.ok()?)
}

/// What `binary` prints for an informational flag, run without this process's environment.
///
/// Releases differ on whether it goes to stdout or stderr, so both are returned.
async fn run_report(binary: &Path, flag: &'static str) -> Result<String, CapabilityError> {
    let output = tokio::time::timeout(
        REPORT_TIMEOUT,
        Command::new(binary)
            .args(serve_prefix(binary))
            .arg(flag)
            .env_clear()
            .env("PATH", ENGINE_PATH)
            .env("LC_ALL", "C")
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| CapabilityError::TimedOut {
        binary: binary.to_owned(),
        flag,
    })?
    .map_err(|source| CapabilityError::Unrunnable {
        binary: binary.to_owned(),
        flag,
        source,
    })?;
    Ok([output.stdout, output.stderr]
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .join("\n"))
}

/// Read `NAME: description (TOTAL MiB, FREE MiB free)` lines and keep the largest total.
fn parse_device_memory(listing: &str) -> Option<u64> {
    listing
        .lines()
        .filter_map(|line| {
            let (_, rest) = line.rsplit_once('(')?;
            rest.split_whitespace().next()?.parse::<u64>().ok()
        })
        .filter(|mebibytes| *mebibytes > 0)
        .max()
        .map(|mebibytes| mebibytes << 20)
}

/// Lines copied from `llama-server --help` of release 0.9.0 (b12110), the flags Mirror reads.
#[cfg(test)]
pub(super) const RECORDED_HELP: &str = "\
-c,    --ctx-size N                     size of the prompt context (default: 0, 0 = loaded from model)
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit in device memory
-np,   --parallel N                     number of server slots (default: -1, -1 = auto)
--mmproj-auto, --no-mmproj, --no-mmproj-auto
-ag,   --agent, -no-ag, --no-agent      whether to enable CORS proxy and all built-in tools
--ui,  --webui, --no-ui, --no-webui     whether to enable the Web UI (default: enabled)
--log-colors [on|off|auto]              Set colored logging ('on', 'off', or 'auto', default: 'auto')
--offline                               Offline mode: forces use of cache, prevents network access
--cors-origins ORIGINS                  comma-separated list of allowed origins for CORS (default: *)
--models-preset PATH                    path to INI file containing model presets for the router server
--models-max N                          for router server, maximum number of models to load simultaneously
--models-autoload, --no-models-autoload
--sleep-idle-seconds SECONDS            number of seconds of idleness after which the server will sleep
";

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;

    use super::*;

    #[test]
    fn device_memory_keeps_the_largest_device() {
        let listing = "\
Available devices:
  BLAS: Accelerate (0 MiB, 0 MiB free)
  MTL0: Apple M5 Max (28753 MiB, 28753 MiB free)
";

        assert_matches!(parse_device_memory(listing), Some(bytes) if bytes == 28753 << 20);
        assert_matches!(parse_device_memory("no devices"), None);
    }
}
