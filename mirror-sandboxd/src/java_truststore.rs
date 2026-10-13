use std::io;
use std::io::Write;
use std::path::Path;

use ring::digest::{Context, SHA1_FOR_LEGACY_USE_ONLY};
use rustls::pki_types::CertificateDer;

pub(crate) const JAVA_TRUSTSTORE: &str = "/tmp/mirror-egress-truststore.jks";

const MAGIC: u32 = 0xFEED_FEED;
const VERSION: u32 = 2;
const TRUSTED_CERTIFICATE: u32 = 2;

pub(crate) fn write_java_truststore(
    path: &Path,
    certificates: &[CertificateDer<'_>],
) -> io::Result<()> {
    let directory = path
        .parent()
        .ok_or_else(|| io::Error::other("Java truststore path has no directory"))?;
    let mut file = tempfile::Builder::new()
        .prefix(".mirror-egress-truststore-")
        .tempfile_in(directory)?;
    file.write_all(&java_truststore(certificates)?)?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn java_truststore(certificates: &[CertificateDer<'_>]) -> io::Result<Vec<u8>> {
    let mut store = Vec::new();
    store.extend_from_slice(&MAGIC.to_be_bytes());
    store.extend_from_slice(&VERSION.to_be_bytes());
    store.extend_from_slice(&length(certificates.len())?.to_be_bytes());
    for (index, certificate) in certificates.iter().enumerate() {
        store.extend_from_slice(&TRUSTED_CERTIFICATE.to_be_bytes());
        java_string(&mut store, &format!("mirror-egress-{index}"));
        store.extend_from_slice(&0u64.to_be_bytes());
        java_string(&mut store, "X.509");
        store.extend_from_slice(&length(certificate.len())?.to_be_bytes());
        store.extend_from_slice(certificate);
    }
    let mut digest = Context::new(&SHA1_FOR_LEGACY_USE_ONLY);
    digest.update(b"Mighty Aphrodite");
    digest.update(&store);
    store.extend_from_slice(digest.finish().as_ref());
    Ok(store)
}

fn java_string(store: &mut Vec<u8>, value: &str) {
    store.extend_from_slice(&(value.len() as u16).to_be_bytes());
    store.extend_from_slice(value.as_bytes());
}

fn length(value: usize) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| io::Error::other("Java truststore entry is too large"))
}

#[cfg(test)]
#[path = "java_truststore_tests.rs"]
mod tests;
