//! Configuration shared by the worker and its workspace process launcher.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use mirror_agent::local::{LocalSandbox, LocalSandboxIdentity};
use mirror_sandbox_microsandbox::{EgressHost, MicrosandboxConfig, MicrosandboxProvider, Msb};
use serde::Deserialize;
use uuid::Uuid;

/// The same ID and clone select the same VM for tools and extensions.
#[derive(Debug, Deserialize)]
pub struct SandboxConfig {
    pub id: Uuid,
    pub image: String,
    pub sandboxd: PathBuf,
    #[serde(default)]
    pub egress: Vec<String>,
    pub cpus: Option<u8>,
    pub memory_mib: Option<u32>,
}

/// Host-owned state must never live under the guest-writable clone.
pub fn quarantine_path(id: Uuid) -> Result<PathBuf, Box<dyn Error>> {
    let home = std::env::var_os("HOME").ok_or("HOME is required for sandbox state")?;
    Ok(PathBuf::from(home)
        .join(".mirror/sandbox-state")
        .join(id.to_string()))
}

impl TryFrom<&mirror_agent::WorkerSandbox> for SandboxConfig {
    type Error = Box<dyn Error>;

    fn try_from(sandbox: &mirror_agent::WorkerSandbox) -> Result<Self, Self::Error> {
        let source = sandbox
            .source
            .as_ref()
            .ok_or("sandbox source is required")?;
        Ok(Self {
            id: Uuid::parse_str(&sandbox.id)?,
            image: source.image.clone(),
            sandboxd: sandbox.sandboxd.clone().into(),
            egress: source.egress.clone(),
            cpus: Some(u8::try_from(source.cpus)?),
            memory_mib: Some(source.memory_mib),
        })
    }
}

impl SandboxConfig {
    pub async fn provider(&self, workspace: &Path) -> Result<MicrosandboxProvider, Box<dyn Error>> {
        let egress = self
            .egress
            .iter()
            .map(|host| EgressHost::parse(host))
            .collect::<Result<Vec<_>, _>>()?;
        let quarantine = quarantine_path(self.id)?;
        let parent = quarantine.parent().ok_or("quarantine path has no parent")?;
        std::fs::create_dir_all(parent)?;
        Ok(MicrosandboxProvider::new(
            Msb::locate().await?,
            MicrosandboxConfig {
                image: self.image.clone(),
                workspace: std::fs::canonicalize(workspace)?,
                sandboxd: self.sandboxd.clone(),
                egress,
                cpus: self.cpus,
                memory_mib: self.memory_mib,
            },
        )?
        .with_quarantine_file(quarantine))
    }

    pub async fn start(
        &self,
        session_id: Uuid,
        workspace: &Path,
    ) -> Result<Arc<LocalSandbox>, Box<dyn Error>> {
        let workspace = std::fs::canonicalize(workspace)?;
        let guest_workspace = workspace
            .to_str()
            .ok_or_else(|| format!("workspace {workspace:?} is not UTF-8"))?
            .to_string();
        Ok(Arc::new(LocalSandbox::start(
            Arc::new(self.provider(&workspace).await?),
            LocalSandboxIdentity {
                request_id: self.id,
                session_id,
            },
            guest_workspace,
        )))
    }
}
