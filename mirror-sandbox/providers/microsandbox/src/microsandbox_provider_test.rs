use std::path::PathBuf;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use serde_json::json;
use serde_json::Value;

use super::*;
use crate::sandboxd_file;
use crate::EgressHost;
use crate::FakeMsb;

const REQUEST: Uuid = Uuid::from_u128(1);
const SESSION: Uuid = Uuid::from_u128(2);
const NAME: &str = "mirror-ws-00000000000000000000000000000001";

/// An `msb inspect` document whose status the fake fills in.
fn document(request: Uuid, default_egress: &str) -> Value {
    json!({
        "status": "__STATUS__",
        "config": {
            "labels": {"mirror.workspace": "", "mirror.request": request.to_string()},
            "network": {
                "policy": {
                    "default_egress": default_egress,
                    "default_ingress": "deny",
                    "rules": [{
                        "action": "allow",
                        "direction": "egress",
                        "destination": {"domain": "api.github.com"},
                        "ports": [{"start": 443, "end": 443}],
                        "protocols": ["tcp"],
                    }],
                },
                "ports": [],
            },
            "mounts": [
                {"host": "/workspace", "guest": "/workspace", "options": {"readonly": false}},
                {
                    "host": sandboxd_file(),
                    "guest": "/usr/local/bin/mirror-sandboxd",
                    "options": {"readonly": true},
                },
            ],
        },
        "active_config": null,
    })
}

async fn provider(fake: &FakeMsb) -> MicrosandboxProvider {
    let config = MicrosandboxConfig {
        image: "debian:bookworm-slim".to_string(),
        workspace: PathBuf::from("/workspace"),
        sandboxd: PathBuf::from("/opt/mirror/mirror-sandboxd"),
        egress: vec![EgressHost::parse("api.github.com").expect("host")],
        cpus: None,
        memory_mib: None,
    };
    MicrosandboxProvider::new(fake.msb().await, config).expect("valid config")
}

fn creates(fake: &FakeMsb) -> usize {
    fake.calls()
        .iter()
        .filter(|call| call.starts_with("create "))
        .count()
}

#[tokio::test]
async fn create_makes_one_sandbox_per_request() {
    let fake = FakeMsb::new();
    fake.set_template(&document(REQUEST, "deny"));
    let provider = provider(&fake).await;

    let first = provider.create(REQUEST, SESSION).await.expect("create");
    let retried = provider.create(REQUEST, SESSION).await.expect("retry");

    assert_eq!(first, NAME);
    assert_eq!(retried, NAME);
    assert_eq!(creates(&fake), 1);
}

#[tokio::test]
async fn create_removes_a_sandbox_whose_network_policy_is_not_the_allowlist() {
    let fake = FakeMsb::new();
    fake.set_template(&document(REQUEST, "allow"));
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.ends_with("it was removed")
    );
    assert_eq!(fake.status(NAME), None);
}

#[tokio::test]
async fn create_will_not_adopt_a_sandbox_made_for_another_request() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document(Uuid::from_u128(9), "deny"));
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.contains("not created for request")
    );
    assert_eq!(fake.status(NAME).as_deref(), Some("Running"));
}

#[tokio::test]
async fn create_removes_a_sandbox_it_made_with_the_wrong_mounts() {
    let fake = FakeMsb::new();
    let mut template = document(REQUEST, "deny");
    template["config"]["mounts"][0]["options"]["readonly"] = json!(true);
    fake.set_template(&template);
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.ends_with("it was removed")
    );
    assert_eq!(fake.status(NAME), None);
}

#[tokio::test]
async fn create_removes_a_bad_sandbox_its_failed_create_made() {
    let fake = FakeMsb::new();
    let mut template = document(REQUEST, "deny");
    template["config"]["mounts"][0]["options"]["readonly"] = json!(true);
    fake.set_template(&template);
    fake.write("create_fails", "");
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.ends_with("it was removed")
    );
    assert_eq!(fake.status(NAME), None);
}

#[tokio::test]
async fn create_leaves_a_sandbox_another_create_made_first() {
    let fake = FakeMsb::new();
    let mut template = document(REQUEST, "deny");
    template["config"]["mounts"][0] = json!({
        "host": "/elsewhere",
        "guest": "/elsewhere",
        "options": {"readonly": false},
    });
    fake.set_template(&template);
    fake.write("create_races", "");
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.contains("\"/elsewhere\"")
    );
    assert_eq!(fake.status(NAME).as_deref(), Some("Running"));
}

#[tokio::test]
async fn create_removes_another_workspaces_sandbox_whose_network_is_open() {
    let fake = FakeMsb::new();
    let mut document = document(REQUEST, "allow");
    document["config"]["mounts"][0] = json!({
        "host": "/elsewhere",
        "guest": "/elsewhere",
        "options": {"readonly": false},
    });
    fake.add_sandbox(NAME, "Running", &document);
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.ends_with("it was removed")
    );
    assert_eq!(fake.status(NAME), None);
}

#[tokio::test]
async fn create_will_not_adopt_or_remove_a_sandbox_of_another_workspace() {
    let fake = FakeMsb::new();
    let mut document = document(REQUEST, "deny");
    document["config"]["mounts"][0] = json!({
        "host": "/elsewhere",
        "guest": "/elsewhere",
        "options": {"readonly": false},
    });
    fake.add_sandbox(NAME, "Running", &document);
    let provider = provider(&fake).await;

    let created = provider.create(REQUEST, SESSION).await;

    assert_matches!(
        created,
        Err(SandboxProviderError::Rejected(detail)) if detail.contains("\"/elsewhere\"")
    );
    assert_eq!(fake.status(NAME).as_deref(), Some("Running"));
}

#[tokio::test]
async fn liveness_follows_the_vm_status() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document(REQUEST, "deny"));
    let provider = provider(&fake).await;

    assert_matches!(provider.liveness(NAME).await, Ok(SandboxLiveness::Ready));
    fake.set_status(NAME, "Crashed");
    assert_matches!(
        provider.liveness(NAME).await,
        Ok(SandboxLiveness::Suspended)
    );
    fake.set_status(NAME, "Starting");
    assert_matches!(
        provider.liveness(NAME).await,
        Err(SandboxProviderError::Retryable(_))
    );
    provider.delete(NAME).await.expect("delete");
    assert_matches!(provider.liveness(NAME).await, Ok(SandboxLiveness::Gone));
}

#[tokio::test]
async fn suspend_stops_the_vm_and_resume_starts_it_again() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document(REQUEST, "deny"));
    let provider = provider(&fake).await;

    provider.suspend(NAME).await.expect("suspend");
    provider
        .wait_until_suspended(NAME)
        .await
        .expect("suspended");
    assert_eq!(fake.status(NAME).as_deref(), Some("Stopped"));

    provider.resume(NAME).await.expect("resume");
    provider.wait_until_ready(NAME).await.expect("ready");
    assert!(fake.calls().contains(&format!("start {NAME}")));
}

#[tokio::test]
async fn resume_unpauses_a_paused_vm() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Paused", &document(REQUEST, "deny"));
    let provider = provider(&fake).await;

    provider.resume(NAME).await.expect("resume");

    assert_eq!(fake.calls().last(), Some(&format!("resume {NAME}")));
}

#[tokio::test]
async fn wait_until_ready_rejects_a_vm_that_crashed() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Crashed", &document(REQUEST, "deny"));
    let provider = provider(&fake).await;

    assert_matches!(
        provider.wait_until_ready(NAME).await,
        Err(SandboxProviderError::Rejected(detail)) if detail.contains("crashed")
    );
}

#[tokio::test]
async fn delete_is_idempotent() {
    let fake = FakeMsb::new();
    fake.add_sandbox(NAME, "Running", &document(REQUEST, "deny"));
    let provider = provider(&fake).await;

    provider.delete(NAME).await.expect("delete");
    provider.delete(NAME).await.expect("delete again");

    assert_eq!(fake.status(NAME), None);
}
