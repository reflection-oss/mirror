//! What `msb inspect --format json` reports about one sandbox.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use mirror_sandbox::SANDBOXD_PATH;
use serde::Deserialize;

use crate::EgressHost;

/// The parts of a sandbox's inspection that Mirror relies on. Unknown fields
/// are ignored; a missing field fails to parse, so a changed format can only
/// make a sandbox unusable, never unchecked.
#[derive(Debug, Deserialize)]
pub(crate) struct SandboxInspection {
    pub status: SandboxStatus,
    config: InspectedConfig,
    /// The configuration the running VM booted with; absent while stopped.
    active_config: Option<InspectedConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub(crate) enum SandboxStatus {
    Starting,
    Running,
    Draining,
    Paused,
    Stopped,
    Crashed,
    #[serde(other)]
    Unrecognized,
}

#[derive(Debug, Deserialize)]
struct InspectedConfig {
    #[serde(default)]
    labels: BTreeMap<String, String>,
    network: InspectedNetwork,
    mounts: Vec<InspectedMount>,
}

#[derive(Debug, Deserialize)]
struct InspectedMount {
    host: PathBuf,
    guest: PathBuf,
    options: MountOptions,
}

#[derive(Debug, Deserialize)]
struct MountOptions {
    readonly: bool,
}

#[derive(Debug, Deserialize)]
struct InspectedNetwork {
    policy: NetworkPolicy,
    ports: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct NetworkPolicy {
    default_egress: PolicyAction,
    default_ingress: PolicyAction,
    rules: Vec<NetworkRule>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PolicyAction {
    Allow,
    Deny,
    #[serde(other)]
    Unrecognized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RuleDirection {
    Egress,
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct NetworkRule {
    action: PolicyAction,
    direction: RuleDirection,
    destination: RuleDestination,
    ports: Vec<PortRange>,
    protocols: Vec<String>,
}

/// Exactly one field is set: a domain, a domain suffix, or another target
/// kind such as an address range or group.
#[derive(Debug, Deserialize)]
struct RuleDestination {
    domain: Option<String>,
    domain_suffix: Option<String>,
    #[serde(flatten)]
    other: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
struct PortRange {
    start: u16,
    end: u16,
}

const HTTPS_PORT: u16 = 443;

impl SandboxInspection {
    pub fn label(&self, key: &str) -> Option<&str> {
        self.config.labels.get(key).map(String::as_str)
    }

    /// Confirms that both the stored and the running configuration deny all
    /// traffic except HTTPS to exactly `egress`. Like [`Self::verify_mounts`],
    /// the problem reads as a predicate of the sandbox.
    pub fn verify_network(&self, egress: &[EgressHost]) -> Result<(), String> {
        let expected: BTreeSet<(bool, &str)> = egress
            .iter()
            .map(|host| (host.includes_subdomains(), host.name()))
            .collect();
        self.verify_each_config(|config| config.verify_network(&expected))
    }

    /// Confirms that both the stored and the running configuration mount
    /// `workspace` read-write at the same path and one `sandboxd` file
    /// read-only, and nothing else from the host.
    pub fn verify_mounts(&self, workspace: &Path) -> Result<(), String> {
        self.verify_each_config(|config| config.verify_mounts(workspace))
    }

    fn verify_each_config(
        &self,
        verify: impl Fn(&InspectedConfig) -> Result<(), String>,
    ) -> Result<(), String> {
        verify(&self.config)?;
        match &self.active_config {
            Some(active) => verify(active).map_err(|problem| format!("booted a VM that {problem}")),
            None => Ok(()),
        }
    }
}

impl InspectedConfig {
    fn verify_network(&self, expected: &BTreeSet<(bool, &str)>) -> Result<(), String> {
        let policy = &self.network.policy;
        if policy.default_egress != PolicyAction::Deny
            || policy.default_ingress != PolicyAction::Deny
        {
            return Err(format!(
                "does not deny network traffic by default: egress {:?}, ingress {:?}",
                policy.default_egress, policy.default_ingress
            ));
        }
        if !self.network.ports.is_empty() {
            return Err(format!("publishes host ports {:?}", self.network.ports));
        }
        let allowed = policy
            .rules
            .iter()
            .map(NetworkRule::https_host)
            .collect::<Result<BTreeSet<_>, _>>()?;
        if &allowed != expected {
            return Err(format!(
                "allows HTTPS to {allowed:?} rather than {expected:?}"
            ));
        }
        Ok(())
    }

    /// `sandboxd`'s host path may change when Mirror is reinstalled, so it
    /// need only be a file; a directory there would show its files to the
    /// guest.
    fn verify_mounts(&self, workspace: &Path) -> Result<(), String> {
        let mut mounts_workspace = false;
        let mut mounts_sandboxd = false;
        for mount in &self.mounts {
            let is_workspace =
                mount.host == workspace && mount.guest == workspace && !mount.options.readonly;
            let is_sandboxd = mount.guest == Path::new(SANDBOXD_PATH)
                && mount.options.readonly
                && mount.host.is_file();
            if is_workspace && !mounts_workspace {
                mounts_workspace = true;
            } else if is_sandboxd && !mounts_sandboxd {
                mounts_sandboxd = true;
            } else {
                return Err(format!(
                    "mounts {:?} at {:?} besides the workspace and sandboxd",
                    mount.host, mount.guest
                ));
            }
        }
        if !mounts_workspace {
            return Err(format!(
                "does not mount the workspace {workspace:?} read-write at the same path"
            ));
        }
        if !mounts_sandboxd {
            return Err(format!(
                "does not mount a sandboxd file read-only at {SANDBOXD_PATH}"
            ));
        }
        Ok(())
    }
}

impl NetworkRule {
    /// The `(includes subdomains, name)` this rule allows on HTTPS, or why the
    /// rule is something else.
    fn https_host(&self) -> Result<(bool, &str), String> {
        let https_only = self.action == PolicyAction::Allow
            && self.direction == RuleDirection::Egress
            && self.protocols == ["tcp"]
            && self.ports
                == [PortRange {
                    start: HTTPS_PORT,
                    end: HTTPS_PORT,
                }];
        let destination = &self.destination;
        match (&destination.domain, &destination.domain_suffix) {
            (Some(domain), None) if https_only && destination.other.is_empty() => {
                Ok((false, domain))
            }
            (None, Some(suffix)) if https_only && destination.other.is_empty() => {
                Ok((true, suffix))
            }
            _ => Err(format!(
                "has a network rule other than an HTTPS host allowance: {self:?}"
            )),
        }
    }
}

#[cfg(test)]
#[path = "sandbox_inspection_test.rs"]
mod tests;
