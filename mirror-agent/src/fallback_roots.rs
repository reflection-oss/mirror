use std::sync::OnceLock;

use reqwest::Certificate;
use reqwest::ClientBuilder;
use rustls::pki_types::CertificateDer;
use rustls::RootCertStore;

pub fn with_fallback_roots(builder: ClientBuilder) -> ClientBuilder {
    builder.tls_certs_merge(fallback_roots().iter().cloned())
}

fn fallback_roots() -> &'static [Certificate] {
    static ROOTS: OnceLock<Vec<Certificate>> = OnceLock::new();
    ROOTS.get_or_init(|| {
        if cfg!(all(
            unix,
            not(target_os = "android"),
            not(target_vendor = "apple")
        )) {
            roots_to_merge(&rustls_native_certs::load_native_certs().certs)
        } else {
            Vec::new()
        }
    })
}

fn roots_to_merge<Der: AsRef<[u8]>>(system_certs: &[Der]) -> Vec<Certificate> {
    let mut system_store = RootCertStore::empty();
    let (parsed, _ignored) = system_store.add_parsable_certificates(
        system_certs
            .iter()
            .map(|certificate| CertificateDer::from(certificate.as_ref())),
    );
    if parsed > 0 {
        return Vec::new();
    }
    webpki_root_certs::TLS_SERVER_ROOT_CERTS
        .iter()
        .filter_map(|certificate| Certificate::from_der(certificate).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use std::process::Command;

    #[test]
    fn system_roots_take_precedence_over_bundled_roots() {
        assert_eq!(
            roots_to_merge(&webpki_root_certs::TLS_SERVER_ROOT_CERTS[..1]).len(),
            0
        );
    }

    #[test]
    fn unparsable_system_roots_fall_back_to_bundled_roots() {
        assert_eq!(
            roots_to_merge(&[b"not a certificate"]).len(),
            webpki_root_certs::TLS_SERVER_ROOT_CERTS.len()
        );
        assert_eq!(
            roots_to_merge::<Vec<u8>>(&[]).len(),
            webpki_root_certs::TLS_SERVER_ROOT_CERTS.len()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn empty_system_store_probe() {
        if std::env::var_os("MIRROR_EMPTY_SYSTEM_STORE_PROBE").is_none() {
            return;
        }
        let _ = rustls::crypto::ring::default_provider().install_default();
        let without_fallback = reqwest::Client::builder()
            .build()
            .expect_err("the probe's SSL_CERT_FILE and SSL_CERT_DIR leave the system store empty");
        assert_matches!(
            std::error::Error::source(&without_fallback).map(ToString::to_string),
            Some(message) if message.contains("No CA certificates were loaded from the system")
        );
        assert_eq!(
            fallback_roots().len(),
            webpki_root_certs::TLS_SERVER_ROOT_CERTS.len()
        );
        assert_matches!(
            with_fallback_roots(reqwest::Client::builder()).build(),
            Ok(_)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn empty_system_store_falls_back_to_bundled_roots() {
        let directory = tempfile::tempdir().expect("fixture");
        let empty_bundle = directory.path().join("empty.pem");
        std::fs::write(&empty_bundle, "").expect("empty CA bundle");
        let output = Command::new(std::env::current_exe().expect("test executable"))
            .args([
                "--exact",
                "fallback_roots::tests::empty_system_store_probe",
                "--nocapture",
            ])
            .env("MIRROR_EMPTY_SYSTEM_STORE_PROBE", "1")
            .env("SSL_CERT_FILE", &empty_bundle)
            .env("SSL_CERT_DIR", directory.path())
            .output()
            .expect("probe runs");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success()
                && stdout.contains("fallback_roots::tests::empty_system_store_probe ... ok"),
            "probe failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
