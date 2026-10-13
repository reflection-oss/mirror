use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;
use crate::FakeMsb;

#[test]
fn version_parse_ignores_a_prerelease_suffix() {
    assert_eq!(MsbVersion::parse("0.8.0-rc.1"), Some(MsbVersion(0, 8, 0)));
    assert_eq!(MsbVersion::parse("0.7"), None);
}

#[tokio::test]
async fn at_accepts_the_minimum_version() {
    let fake = FakeMsb::new();

    assert_matches!(Msb::at(fake.program()).await, Ok(_));
    assert_eq!(fake.calls(), ["--version"]);
}

#[tokio::test]
async fn at_rejects_an_older_msb() {
    let fake = FakeMsb::new();
    fake.write("version", "msb 0.7.6\n");

    assert_matches!(
        Msb::at(fake.program()).await,
        Err(MsbSetupError::TooOld { found }) if found == "0.7.6"
    );
}

#[tokio::test]
async fn at_rejects_a_prerelease_of_the_minimum_version() {
    let fake = FakeMsb::new();
    fake.write("version", "msb 0.7.7-rc.1\n");

    assert_matches!(
        Msb::at(fake.program()).await,
        Err(MsbSetupError::TooOld { found }) if found == "0.7.7-rc.1"
    );
}

#[tokio::test]
async fn at_rejects_a_program_that_is_not_msb() {
    let fake = FakeMsb::new();
    fake.write("version", "something else\n");

    assert_matches!(
        Msb::at(fake.program()).await,
        Err(MsbSetupError::Unusable { detail, .. }) if detail.contains("something else")
    );
}

#[tokio::test]
async fn run_reports_a_missing_sandbox() {
    let fake = FakeMsb::new();
    let msb = fake.msb().await;

    let inspected = msb
        .run(
            vec![
                "inspect".into(),
                "--format".into(),
                "json".into(),
                "absent".into(),
            ],
            Duration::from_secs(10),
        )
        .await;

    assert_matches!(inspected, Err(MsbFailure::SandboxMissing));
}
