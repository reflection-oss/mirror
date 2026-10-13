//! How every sandbox from one provider is configured.

use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;

use mirror_sandbox::SANDBOXD_PATH;
use uuid::Uuid;

use crate::EgressHost;

/// The image, resources, network allowlist and host paths shared by the
/// sandboxes one provider creates. Each sandbox mounts `workspace` at the same
/// path, so a path names the same file on the host and in the guest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MicrosandboxConfig {
    /// An OCI image reference, such as `debian:bookworm-slim`.
    pub image: String,
    /// An absolute host directory, mounted read-write.
    pub workspace: PathBuf,
    /// A Linux `mirror-sandboxd` for the host's architecture, mounted
    /// read-only at [`SANDBOXD_PATH`].
    pub sandboxd: PathBuf,
    /// The only hosts the sandbox may reach, on TCP port 443.
    pub egress: Vec<EgressHost>,
    /// Virtual CPUs, or `msb`'s default.
    pub cpus: Option<u8>,
    /// Memory in MiB, or `msb`'s default.
    pub memory_mib: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum InvalidMicrosandboxConfig {
    #[error("the {role} path {path:?} must be absolute and must not contain `:`")]
    Path { role: &'static str, path: PathBuf },
    #[error("the sandbox image reference is empty")]
    EmptyImage,
}

/// Marks every sandbox Mirror creates, so `msb ls --label` can find them.
pub const WORKSPACE_LABEL: &str = "mirror.workspace";
/// The provisioning request a sandbox was created for.
pub const REQUEST_LABEL: &str = "mirror.request";
/// The Mirror session a sandbox was created for.
pub const SESSION_LABEL: &str = "mirror.session";

impl MicrosandboxConfig {
    pub fn validate(&self) -> Result<(), InvalidMicrosandboxConfig> {
        mount_source("workspace", &self.workspace)?;
        mount_source("sandboxd", &self.sandboxd)?;
        if self.image.is_empty() {
            return Err(InvalidMicrosandboxConfig::EmptyImage);
        }
        Ok(())
    }

    /// The `msb create` arguments for one sandbox: no network beyond the
    /// HTTPS allowlist, the workspace at its host path, and `sandboxd`.
    pub(crate) fn create_arguments(
        &self,
        name: &str,
        request_id: Uuid,
        session_id: Uuid,
    ) -> Vec<OsString> {
        let workspace = self.workspace.as_os_str();
        let mut volume = workspace.to_os_string();
        volume.push(":");
        volume.push(workspace);
        let mut sandboxd = self.sandboxd.as_os_str().to_os_string();
        sandboxd.push(format!(":{SANDBOXD_PATH}:ro"));
        let mut arguments: Vec<OsString> = vec![
            "create".into(),
            "--quiet".into(),
            "--name".into(),
            name.into(),
            "--label".into(),
            WORKSPACE_LABEL.into(),
            "--label".into(),
            format!("{REQUEST_LABEL}={request_id}").into(),
            "--label".into(),
            format!("{SESSION_LABEL}={session_id}").into(),
            "--no-net".into(),
            "--tls-intercept".into(),
        ];
        for host in &self.egress {
            arguments.push("--net-rule".into());
            arguments.push(format!("allow@{host}:tcp:443").into());
        }
        arguments.extend([
            "--volume".into(),
            volume,
            "--mount-file".into(),
            sandboxd,
            "--workdir".into(),
            workspace.to_os_string(),
        ]);
        if let Some(cpus) = self.cpus {
            arguments.extend(["--cpus".into(), cpus.to_string().into()]);
        }
        if let Some(memory_mib) = self.memory_mib {
            arguments.extend(["--memory".into(), format!("{memory_mib}M").into()]);
        }
        arguments.extend(["--".into(), self.image.clone().into()]);
        arguments
    }
}

/// `msb` mount specifications are `SOURCE:DEST[:OPTIONS]`, so a `:` in a path
/// would change what is mounted.
fn mount_source(role: &'static str, path: &Path) -> Result<(), InvalidMicrosandboxConfig> {
    if path.is_absolute() && !path.to_string_lossy().contains(':') {
        return Ok(());
    }
    Err(InvalidMicrosandboxConfig::Path {
        role,
        path: path.to_path_buf(),
    })
}

#[cfg(test)]
#[path = "microsandbox_config_test.rs"]
mod tests;
