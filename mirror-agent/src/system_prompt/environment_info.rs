//! Runtime environment facts for the system prompt.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use crate::is_repository_root;

/// Runtime environment facts surfaced to the agent.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentInfo {
    pub working_directory: PathBuf,
    pub is_git_repo: bool,
    pub platform: String,
    pub os_version: String,
    pub shell: String,
}

/// A bash script that prints the facts [`EnvironmentInfo::from_probe`] reads:
/// exactly two lines, each empty when its fact is unavailable.
pub const ENVIRONMENT_PROBE: &str =
    r#"printf '%s\n' "$(uname -sr 2>/dev/null)" "${BASH_VERSION%%(*}""#;

impl EnvironmentInfo {
    /// Environment facts from [`ENVIRONMENT_PROBE`]'s output on a `platform`
    /// machine whose `working_directory` is also readable here.
    pub fn from_probe(working_directory: &Path, platform: &str, probe_output: &str) -> Self {
        let mut lines = probe_output.lines().map(str::trim);
        let mut next = || lines.next().filter(|line| !line.is_empty());
        let os_version = next().map_or_else(|| platform.to_string(), str::to_string);
        let shell = next().map_or_else(|| "bash".to_string(), |version| format!("bash {version}"));
        Self {
            working_directory: working_directory.components().collect(),
            is_git_repo: is_git_checkout(working_directory),
            platform: platform.to_string(),
            os_version,
            shell,
        }
    }
}

/// Gather environment facts for a selected working directory on this machine.
pub fn gather_environment_info(working_directory: &Path) -> EnvironmentInfo {
    let platform = match std::env::consts::OS {
        "linux" => "Linux",
        "macos" => "Darwin",
        "windows" => "Windows",
        other => other,
    };
    EnvironmentInfo::from_probe(working_directory, platform, &run_probe(ENVIRONMENT_PROBE))
}

/// This machine's local date as `YYYY-MM-DD`.
pub(super) fn local_date() -> Option<String> {
    Some(run_probe("date +%Y-%m-%d").trim().to_string()).filter(|date| !date.is_empty())
}

fn is_git_checkout(directory: &Path) -> bool {
    directory
        .ancestors()
        .find(|ancestor| is_repository_root(ancestor))
        .is_some_and(|root| root.join(".git").exists())
}

fn run_probe(script: &str) -> String {
    let inherited = ["PATH", "TZ"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name, value)));
    Command::new("/bin/bash")
        .args(["-c", script])
        .env_clear()
        .envs(inherited)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn gather_describes_the_selected_directory() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("repository marker");

        let environment = gather_environment_info(workspace.path());

        assert_eq!(environment.working_directory, workspace.path());
        assert!(environment.is_git_repo);
        assert!(environment.os_version.starts_with(&environment.platform));
        assert!(environment.shell.starts_with("bash "));
    }

    #[test]
    fn gather_reports_a_directory_outside_any_repository() {
        let workspace = tempfile::tempdir().expect("temporary workspace");

        assert!(!gather_environment_info(workspace.path()).is_git_repo);
    }

    #[test]
    fn jujutsu_workspace_inside_a_git_checkout_is_not_a_git_repository() {
        let checkout = tempfile::tempdir().expect("temporary checkout");
        fs::create_dir(checkout.path().join(".git")).expect("git marker");
        let workspace = checkout.path().join("workspaces/feature");
        fs::create_dir_all(workspace.join(".jj")).expect("jujutsu marker");

        assert!(!gather_environment_info(&workspace).is_git_repo);
    }
}
