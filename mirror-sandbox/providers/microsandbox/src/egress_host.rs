//! A host a sandbox may reach over HTTPS.

use std::fmt;

/// A DNS name a sandbox may reach on TCP port 443. A leading `*.` allows every
/// subdomain of the rest of the name, which must have at least two labels.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EgressHost {
    name: String,
    subdomains: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("egress host {0:?} must be a DNS name such as `api.github.com` or `*.github.com`")]
pub struct InvalidEgressHost(String);

const MAX_NAME_BYTES: usize = 253;
const MAX_LABEL_BYTES: usize = 63;

impl EgressHost {
    /// Lowercases the name, as DNS names are case-insensitive.
    pub fn parse(host: &str) -> Result<Self, InvalidEgressHost> {
        let lowered = host.to_ascii_lowercase();
        let (name, subdomains) = match lowered.strip_prefix("*.") {
            Some(name) => (name, true),
            None => (lowered.as_str(), false),
        };
        let labels: Vec<&str> = name.split('.').collect();
        let valid = name.len() <= MAX_NAME_BYTES
            && labels.len() >= 2
            && labels.iter().all(|label| valid_label(label));
        if !valid {
            return Err(InvalidEgressHost(host.to_string()));
        }
        Ok(Self {
            name: name.to_string(),
            subdomains,
        })
    }

    /// The name without any `*.` prefix.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn includes_subdomains(&self) -> bool {
        self.subdomains
    }
}

impl fmt::Display for EgressHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.subdomains {
            f.write_str("*.")?;
        }
        f.write_str(&self.name)
    }
}

fn valid_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= MAX_LABEL_BYTES
        && !label.starts_with('-')
        && !label.ends_with('-')
        && label
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

#[cfg(test)]
#[path = "egress_host_test.rs"]
mod tests;
