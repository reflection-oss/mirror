use std::collections::HashMap;

use crate::{GuestFile, GuestProfile};

pub const EGRESS_CA_PATH: &str = "/tmp/mirror-egress-ca.pem";
const PLACEHOLDER: &str = "mirror-managed";
const CA_VARIABLES: [&str; 7] = [
    "SSL_CERT_FILE",
    "GIT_SSL_CAINFO",
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
];

pub fn egress_profile(
    ca_pem: &str,
    placeholder_variables: &[String],
) -> Result<GuestProfile, String> {
    let mut environment: HashMap<String, String> = CA_VARIABLES
        .iter()
        .map(|name| ((*name).to_owned(), EGRESS_CA_PATH.to_owned()))
        .collect();
    for name in placeholder_variables {
        if !environment_variable_name(name)
            || environment
                .insert(name.clone(), PLACEHOLDER.to_owned())
                .is_some()
        {
            return Err(format!(
                "placeholder variable {name:?} is not a variable name, is repeated or names a CA \
                 variable"
            ));
        }
    }
    Ok(GuestProfile {
        name: "egress".into(),
        environment,
        files: vec![GuestFile {
            path: EGRESS_CA_PATH.into(),
            mode: 0o644,
            contents: ca_pem.as_bytes().to_vec(),
        }],
        endpoints: Vec::new(),
    })
}

pub fn environment_variable_name(name: &str) -> bool {
    name.bytes()
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

#[cfg(test)]
#[path = "guest_profile_tests.rs"]
mod tests;
