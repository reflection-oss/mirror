use std::fs;
use std::os::unix::fs::PermissionsExt;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use serde_json::json;

use super::*;

#[test]
fn registers_stdio_paths_and_environment_references_without_overwriting() {
    let source = tempfile::tempdir().expect("source");
    let connections = tempfile::tempdir().expect("connections");
    fs::create_dir(source.path().join(".claude-plugin")).expect("manifest directory");
    fs::write(
        source.path().join(".claude-plugin/plugin.json"),
        r#"{"name":"design"}"#,
    )
    .expect("manifest");
    fs::write(source.path().join(".mcp.json"), r#"{"mcpServers":{"assets":{"command":"node","args":["${CLAUDE_PLUGIN_ROOT}/server.js"],"env":{"ASSETS_TOKEN":"${MY_ASSETS_TOKEN}"}}}}"#).expect("servers");
    let plugin = Plugin::load(source.path()).expect("plugin");
    let loaded = PluginMcpConnection::load(&plugin).expect("connections");
    assert_matches!(loaded.as_slice(), [connection] if connection.name == "plugin_design_assets");
    assert_matches!(
        loaded[0].register(connections.path(), false, &["UNRELATED".into()]),
        Err(PluginError::Mcp(_))
    );
    assert_eq!(fs::read_dir(connections.path()).expect("list").count(), 0);
    loaded[0]
        .register(connections.path(), false, &["MY_ASSETS_TOKEN".into()])
        .expect("register");
    let path = connections.path().join("plugin_design_assets.json");
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).expect("read")).expect("json");
    assert_eq!(
        config,
        json!({"command":"node","args":[format!("{}/server.js", plugin.root.display())],"env":{"ASSETS_TOKEN":"MY_ASSETS_TOKEN"}})
    );
    assert_eq!(
        fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
        0o600
    );
    fs::write(&path, "keep user settings").expect("edit");
    assert_matches!(loaded[0].register(connections.path(), false, &["MY_ASSETS_TOKEN".into()]), Err(PluginError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists);
    assert_eq!(
        fs::read_to_string(path).expect("read"),
        "keep user settings"
    );
}

#[test]
fn codex_inline_server_overrides_default_and_can_use_oauth() {
    let source = tempfile::tempdir().expect("source");
    let connections = tempfile::tempdir().expect("connections");
    fs::create_dir(source.path().join(".codex-plugin")).expect("manifest directory");
    fs::write(source.path().join(".codex-plugin/plugin.json"), r#"{"name":"design","mcpServers":{"docs":{"type":"http","url":"https://docs.example.com/mcp"}}}"#).expect("manifest");
    fs::write(
        source.path().join(".mcp.json"),
        r#"{"mcpServers":{"docs":{"command":"old-server"}}}"#,
    )
    .expect("defaults");
    let loaded = PluginMcpConnection::load(&Plugin::load(source.path()).expect("plugin"))
        .expect("connections");
    loaded[0]
        .register(connections.path(), true, &[])
        .expect("register");
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(connections.path().join("plugin_design_docs.json")).expect("read"),
    )
    .expect("json");
    assert_eq!(
        config,
        json!({"url":"https://docs.example.com/mcp","auth":{"type":"oauth"}})
    );
}

#[test]
fn portable_bearer_header_becomes_a_credential_reference() {
    let source = tempfile::tempdir().expect("source");
    let connections = tempfile::tempdir().expect("connections");
    fs::write(source.path().join("plugin.json"), r#"{"name":"design"}"#).expect("manifest");
    fs::write(source.path().join("mcp.json"), r#"{"mcpServers":{"unsupported":{"unexpected":true},"docs":{"type":"streamable-http","url":"https://docs.example.com/mcp","headers":{"Authorization":"Bearer ${DOCS_TOKEN}"}}}}"#).expect("servers");
    let loaded =
        PluginMcpConnection::selected(&Plugin::load(source.path()).expect("plugin"), "docs")
            .expect("connections");
    assert_matches!(
        loaded.register(connections.path(), false, &["UNRELATED".into()]),
        Err(PluginError::Mcp(_))
    );
    assert_eq!(fs::read_dir(connections.path()).expect("list").count(), 0);
    loaded
        .register(connections.path(), false, &["DOCS_TOKEN".into()])
        .expect("register");
    let config: serde_json::Value = serde_json::from_slice(
        &fs::read(connections.path().join("plugin_design_docs.json")).expect("read"),
    )
    .expect("json");
    assert_eq!(
        config,
        json!({"url":"https://docs.example.com/mcp","auth":{"type":"bearer","tokenEnvVar":"DOCS_TOKEN"}})
    );
}

#[test]
fn rejects_literal_credentials_without_copying_them_into_errors() {
    let source = tempfile::tempdir().expect("source");
    fs::write(source.path().join("plugin.json"), r#"{"name":"design"}"#).expect("manifest");
    fs::write(source.path().join("mcp.json"), r#"{"mcpServers":{"docs":{"url":"https://docs.example.com/mcp","headers":{"Authorization":"Bearer SYNTHETIC_PRIVATE_TOKEN"}}}}"#).expect("servers");
    let error = PluginMcpConnection::load(&Plugin::load(source.path()).expect("plugin"))
        .expect_err("literal credential");
    assert_matches!(error, PluginError::Mcp(_));
    assert!(!error.to_string().contains("SYNTHETIC_PRIVATE_TOKEN"));
}

#[test]
fn rejects_an_mcp_file_outside_the_plugin() {
    let source = tempfile::tempdir().expect("source");
    fs::create_dir(source.path().join(".claude-plugin")).expect("manifest directory");
    fs::write(
        source.path().join(".claude-plugin/plugin.json"),
        r#"{"name":"design","mcpServers":"../servers.json"}"#,
    )
    .expect("manifest");
    assert_matches!(
        PluginMcpConnection::load(&Plugin::load(source.path()).expect("plugin")),
        Err(PluginError::Path)
    );
}
