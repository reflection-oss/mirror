use std::path::Path;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use serde_json::json;
use serde_json::Value;

use super::*;
use crate::sandboxd_file;

const WORKSPACE: &str = "/home/ada/repo";

fn https_rule(destination: Value) -> Value {
    json!({
        "action": "allow",
        "direction": "egress",
        "destination": destination,
        "ports": [{"start": 443, "end": 443}],
        "protocols": ["tcp"],
    })
}

fn mount(host: &str, guest: &str, readonly: bool) -> Value {
    json!({"host": host, "guest": guest, "options": {"readonly": readonly}, "type": "Bind"})
}

fn mounts() -> Vec<Value> {
    vec![
        mount(WORKSPACE, WORKSPACE, false),
        mount(&sandboxd_file(), "/usr/local/bin/mirror-sandboxd", true),
    ]
}

fn config_with(default_egress: &str, rules: Vec<Value>, mounts: Vec<Value>) -> Value {
    json!({
        "labels": {"mirror.request": "request"},
        "network": {
            "policy": {
                "default_egress": default_egress,
                "default_ingress": "deny",
                "rules": rules,
            },
            "ports": [],
        },
        "mounts": mounts,
    })
}

fn config(default_egress: &str, rules: Vec<Value>) -> Value {
    config_with(default_egress, rules, mounts())
}

fn inspection(status: &str, config: Value, active_config: Value) -> SandboxInspection {
    serde_json::from_value(json!({
        "status": status,
        "config": config,
        "active_config": active_config,
        "name": "mirror-ws-test",
    }))
    .expect("inspection parses")
}

fn github() -> Vec<EgressHost> {
    vec![
        EgressHost::parse("api.github.com").expect("host"),
        EgressHost::parse("*.githubusercontent.com").expect("host"),
    ]
}

fn github_rules() -> Vec<Value> {
    vec![
        https_rule(json!({"domain": "api.github.com"})),
        https_rule(json!({"domain_suffix": "githubusercontent.com"})),
    ]
}

#[test]
fn verify_network_confirms_the_requested_allowlist() {
    let running = inspection(
        "Running",
        config("deny", github_rules()),
        config("deny", github_rules()),
    );

    assert_eq!(running.status, SandboxStatus::Running);
    assert_eq!(running.label("mirror.request"), Some("request"));
    assert_matches!(running.verify_network(&github()), Ok(()));
}

#[test]
fn verify_network_checks_a_stopped_sandbox_from_its_stored_configuration() {
    let stopped = inspection("Stopped", config("deny", vec![]), Value::Null);

    assert_matches!(stopped.verify_network(&[]), Ok(()));
}

#[test]
fn verify_network_refuses_a_running_vm_that_booted_with_another_policy() {
    let running = inspection(
        "Running",
        config("deny", github_rules()),
        config("allow", github_rules()),
    );

    assert_matches!(
        running.verify_network(&github()),
        Err(problem) if problem.starts_with("booted a VM that does not deny network traffic")
    );
}

#[test]
fn verify_network_refuses_a_rule_beyond_the_allowlist() {
    let mut rules = github_rules();
    rules.push(json!({
        "action": "allow",
        "direction": "egress",
        "destination": {"group": "public"},
        "ports": [],
        "protocols": [],
    }));
    let running = inspection("Running", config("deny", rules), Value::Null);

    assert_matches!(
        running.verify_network(&github()),
        Err(problem) if problem.contains("other than an HTTPS host allowance")
    );
}

#[test]
fn verify_network_refuses_a_missing_allowance() {
    let running = inspection("Running", config("deny", vec![]), Value::Null);

    assert_matches!(
        running.verify_network(&github()),
        Err(problem) if problem.starts_with("allows HTTPS to {}")
    );
}

#[test]
fn verify_mounts_accepts_the_workspace_and_sandboxd_from_any_install() {
    let elsewhere = tempfile::NamedTempFile::new().expect("reinstalled sandboxd");
    let mut reinstalled = mounts();
    reinstalled[1] = mount(
        &elsewhere.path().display().to_string(),
        "/usr/local/bin/mirror-sandboxd",
        true,
    );
    let running = inspection(
        "Running",
        config_with("deny", vec![], mounts()),
        config_with("deny", vec![], reinstalled),
    );

    assert_matches!(running.verify_mounts(Path::new(WORKSPACE)), Ok(()));
}

#[test]
fn verify_mounts_refuses_a_sandbox_of_another_workspace() {
    let stopped = inspection("Stopped", config("deny", vec![]), Value::Null);

    assert_matches!(
        stopped.verify_mounts(Path::new("/home/ada/other")),
        Err(problem) if problem.starts_with("mounts \"/home/ada/repo\"")
    );
}

#[test]
fn verify_mounts_refuses_a_writable_sandboxd() {
    let mut mounts = mounts();
    mounts[1] = mount("/tmp/sandboxd", "/usr/local/bin/mirror-sandboxd", false);
    let stopped = inspection("Stopped", config_with("deny", vec![], mounts), Value::Null);

    assert_matches!(
        stopped.verify_mounts(Path::new(WORKSPACE)),
        Err(problem) if problem.contains("besides the workspace and sandboxd")
    );
}

#[test]
fn verify_mounts_refuses_a_sandbox_without_sandboxd() {
    let stopped = inspection(
        "Stopped",
        config_with("deny", vec![], vec![mount(WORKSPACE, WORKSPACE, false)]),
        Value::Null,
    );

    assert_matches!(
        stopped.verify_mounts(Path::new(WORKSPACE)),
        Err(problem) if problem.starts_with("does not mount a sandboxd file")
    );
}

#[test]
fn verify_mounts_refuses_a_directory_at_sandboxds_path() {
    let directory = tempfile::tempdir().expect("directory");
    let mut mounts = mounts();
    mounts[1] = mount(
        &directory.path().display().to_string(),
        "/usr/local/bin/mirror-sandboxd",
        true,
    );
    let stopped = inspection("Stopped", config_with("deny", vec![], mounts), Value::Null);

    assert_matches!(
        stopped.verify_mounts(Path::new(WORKSPACE)),
        Err(problem) if problem.contains("besides the workspace and sandboxd")
    );
}

#[test]
fn verify_mounts_refuses_a_read_only_workspace() {
    let stopped = inspection(
        "Stopped",
        config_with("deny", vec![], vec![mount(WORKSPACE, WORKSPACE, true)]),
        Value::Null,
    );

    assert_matches!(
        stopped.verify_mounts(Path::new(WORKSPACE)),
        Err(problem) if problem.contains("besides the workspace and sandboxd")
    );
}

#[test]
fn unknown_status_parses_as_unrecognized() {
    let migrating = inspection("Migrating", config("deny", vec![]), Value::Null);

    assert_eq!(migrating.status, SandboxStatus::Unrecognized);
}

#[test]
fn inspection_without_a_network_policy_does_not_parse() {
    let parsed = serde_json::from_value::<SandboxInspection>(json!({
        "status": "Running",
        "config": {"labels": {}, "network": {"ports": []}, "mounts": []},
        "active_config": null,
    }));

    assert_matches!(parsed, Err(_));
}
