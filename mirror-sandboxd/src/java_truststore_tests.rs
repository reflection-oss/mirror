use pretty_assertions::assert_eq;
use rustls::pki_types::CertificateDer;

use super::write_java_truststore;

#[test]
fn each_certificate_becomes_a_trusted_entry_under_an_empty_password_digest() {
    let directory = tempfile::tempdir().expect("directory");
    let path = directory.path().join("truststore.jks");
    let certificate = CertificateDer::from(vec![0x30, 0x03, 0x02, 0x01, 0x01]);

    write_java_truststore(&path, &[certificate]).expect("truststore");

    let mut expected = vec![0xFE, 0xED, 0xFE, 0xED, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2];
    expected.extend_from_slice(b"\x00\x0fmirror-egress-0");
    expected.extend_from_slice(&[0; 8]);
    expected.extend_from_slice(b"\x00\x05X.509");
    expected.extend_from_slice(&[0, 0, 0, 5, 0x30, 0x03, 0x02, 0x01, 0x01]);
    expected.extend_from_slice(&[
        0x9a, 0xaa, 0x50, 0xa4, 0xf4, 0xf1, 0xea, 0x97, 0xfd, 0x9a, 0x4d, 0x0b, 0x53, 0xb6, 0x3d,
        0xae, 0x94, 0xc7, 0xa0, 0xdc,
    ]);
    assert_eq!(std::fs::read(&path).expect("read truststore"), expected);
}
