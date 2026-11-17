use std::collections::HashMap;
use std::fmt::Write;

use crate::{GuestFile, GuestProfile, EGRESS_CA_PATH};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KubernetesGuestConfig {
    pub proxy_host: String,
    pub clusters: Vec<KubernetesGuestCluster>,
    pub local_proxy_port: u16,
    pub placeholder_token: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KubernetesGuestCluster {
    pub name: String,
    pub contexts: Vec<String>,
}

pub const KUBECONFIG_PATH: &str = "/tmp/mirror-kubeconfig.yaml";
const KUBECONFIG_USER: &str = "mirror";

impl KubernetesGuestConfig {
    pub fn profile(&self) -> Result<GuestProfile, String> {
        self.validate()?;
        Ok(GuestProfile {
            name: "kubernetes".into(),
            environment: HashMap::from([("KUBECONFIG".to_owned(), KUBECONFIG_PATH.to_owned())]),
            files: vec![GuestFile {
                path: KUBECONFIG_PATH.into(),
                mode: 0o600,
                contents: self.kubeconfig().into_bytes(),
            }],
            endpoints: Vec::new(),
        })
    }

    fn validate(&self) -> Result<(), String> {
        if !plain_name(&self.proxy_host) {
            return Err(format!(
                "proxy host {:?} is not a host name",
                self.proxy_host
            ));
        }
        if self.local_proxy_port == 0 {
            return Err("the sandbox local proxy port is unset".into());
        }
        if !plain_scalar(&self.placeholder_token) {
            return Err(format!(
                "placeholder token {:?} is not a plain YAML scalar",
                self.placeholder_token
            ));
        }
        if self.clusters.is_empty() {
            return Err("the kubeconfig would list no clusters".into());
        }
        let mut contexts = Vec::new();
        for cluster in &self.clusters {
            for name in std::iter::once(&cluster.name).chain(&cluster.contexts) {
                if !plain_name(name) {
                    return Err(format!(
                        "cluster or context name {name:?} is not a DNS label"
                    ));
                }
                if contexts.contains(name) {
                    return Err(format!("context {name:?} is listed twice"));
                }
                contexts.push(name.clone());
            }
        }
        Ok(())
    }

    fn kubeconfig(&self) -> String {
        let mut text = String::from("apiVersion: v1\nkind: Config\npreferences: {}\nclusters:\n");
        for cluster in &self.clusters {
            let _ = write!(
                text,
                "- name: \"{name}\"\n  cluster:\n    server: \"https://{host}/clusters/{name}\"\n    certificate-authority: \"{ca}\"\n    proxy-url: \"http://127.0.0.1:{port}\"\n",
                name = cluster.name,
                host = self.proxy_host,
                ca = EGRESS_CA_PATH,
                port = self.local_proxy_port,
            );
        }
        let _ = write!(
            text,
            "users:\n- name: \"{KUBECONFIG_USER}\"\n  user:\n    token: \"{}\"\ncontexts:\n",
            self.placeholder_token
        );
        for cluster in &self.clusters {
            for context in std::iter::once(&cluster.name).chain(&cluster.contexts) {
                let _ = write!(
                    text,
                    "- name: \"{context}\"\n  context:\n    cluster: \"{name}\"\n    user: \"{KUBECONFIG_USER}\"\n",
                    name = cluster.name
                );
            }
        }
        let _ = writeln!(text, "current-context: \"{}\"", self.clusters[0].name);
        text
    }
}

fn plain_scalar(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn plain_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 253
        && name.split('.').all(|label| {
            !label.is_empty()
                && !label.starts_with('-')
                && !label.ends_with('-')
                && label
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        })
}

#[cfg(test)]
#[path = "kubernetes_guest_config_tests.rs"]
mod tests;
