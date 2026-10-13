use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;

fn config() -> MicrosandboxConfig {
    MicrosandboxConfig {
        image: "debian:bookworm-slim".to_string(),
        workspace: PathBuf::from("/Users/me/project"),
        sandboxd: PathBuf::from("/opt/mirror/linux/mirror-sandboxd"),
        egress: vec![EgressHost::parse("api.github.com").expect("host")],
        cpus: Some(2),
        memory_mib: Some(2048),
    }
}

#[test]
fn create_arguments_isolate_the_sandbox_and_mount_its_workspace_in_place() {
    let request_id = Uuid::from_u128(1);
    let session_id = Uuid::from_u128(2);

    let arguments = config().create_arguments("mirror-ws-test", request_id, session_id);

    assert_eq!(
        arguments,
        [
            "create",
            "--quiet",
            "--name",
            "mirror-ws-test",
            "--label",
            "mirror.workspace",
            "--label",
            "mirror.request=00000000-0000-0000-0000-000000000001",
            "--label",
            "mirror.session=00000000-0000-0000-0000-000000000002",
            "--no-net",
            "--tls-intercept",
            "--net-rule",
            "allow@api.github.com:tcp:443",
            "--volume",
            "/Users/me/project:/Users/me/project",
            "--mount-file",
            "/opt/mirror/linux/mirror-sandboxd:/usr/local/bin/mirror-sandboxd:ro",
            "--workdir",
            "/Users/me/project",
            "--cpus",
            "2",
            "--memory",
            "2048M",
            "--",
            "debian:bookworm-slim",
        ]
        .map(OsString::from)
    );
}

#[test]
fn validate_rejects_a_workspace_path_that_would_change_the_mount() {
    let mut config = config();
    config.workspace = PathBuf::from("/tmp/a:/etc");

    assert_matches!(
        config.validate(),
        Err(InvalidMicrosandboxConfig::Path {
            role: "workspace",
            ..
        })
    );
}

#[test]
fn validate_rejects_a_relative_sandboxd_path() {
    let mut config = config();
    config.sandboxd = PathBuf::from("mirror-sandboxd");

    assert_matches!(
        config.validate(),
        Err(InvalidMicrosandboxConfig::Path {
            role: "sandboxd",
            ..
        })
    );
}
