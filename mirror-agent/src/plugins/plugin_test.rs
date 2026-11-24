use std::fs;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use super::*;

#[test]
fn loads_codex_metadata_without_executing_components() {
    let source = tempfile::tempdir().expect("source");
    fs::create_dir(source.path().join(".codex-plugin")).expect("manifest directory");
    fs::write(
        source.path().join(".codex-plugin/plugin.json"),
        r#"{"name":"design","description":"Design workflows","hooks":{"command":"exit 1"}}"#,
    )
    .expect("manifest");
    let plugin = Plugin::load(source.path()).expect("load");
    assert_eq!(plugin.name, "design");
    assert_eq!(plugin.description, "Design workflows");
    assert_eq!(plugin.format, PluginFormat::Codex);
}

#[test]
fn portable_manifest_takes_precedence() {
    let source = tempfile::tempdir().expect("source");
    fs::create_dir(source.path().join(".claude-plugin")).expect("manifest directory");
    fs::write(
        source.path().join(".claude-plugin/plugin.json"),
        r#"{"name":"legacy"}"#,
    )
    .expect("legacy manifest");
    fs::write(source.path().join("plugin.json"), r#"{"name":"portable"}"#)
        .expect("portable manifest");
    let plugin = Plugin::load(source.path()).expect("load");
    assert_eq!(plugin.name, "portable");
    assert_eq!(plugin.format, PluginFormat::Portable);
}

#[cfg(unix)]
#[test]
fn registers_and_removes_a_claude_folder_without_deleting_it() {
    let source = tempfile::tempdir().expect("source");
    let registry = tempfile::tempdir().expect("registry");
    fs::create_dir(source.path().join(".claude-plugin")).expect("manifest directory");
    fs::write(
        source.path().join(".claude-plugin/plugin.json"),
        r#"{"name":"review"}"#,
    )
    .expect("manifest");
    add_plugin(registry.path(), source.path()).expect("register");
    assert_matches!(load_plugins(registry.path()).expect("load").as_slice(), [plugin] if plugin.name == "review");
    assert_matches!(add_plugin(registry.path(), source.path()), Err(PluginError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists);
    let ordinary = registry.path().join("notes");
    fs::write(&ordinary, "keep").expect("regular file");
    assert_matches!(
        remove_plugin(registry.path(), "notes"),
        Err(PluginError::NotRegistration)
    );
    assert_eq!(fs::read_to_string(ordinary).expect("preserved"), "keep");
    assert_matches!(load_plugins(registry.path()).expect("list").as_slice(), [plugin] if plugin.name == "review");
    remove_plugin(registry.path(), "review").expect("remove");
    assert_eq!(load_plugins(registry.path()).expect("load").len(), 0);
    assert_eq!(
        Plugin::load(source.path()).expect("source preserved").name,
        "review"
    );
}

#[cfg(unix)]
#[test]
fn rejects_manifest_symlinks_outside_the_plugin() {
    let source = tempfile::tempdir().expect("source");
    let outside = tempfile::NamedTempFile::new().expect("outside");
    fs::write(outside.path(), r#"{"name":"escape"}"#).expect("manifest");
    std::os::unix::fs::symlink(outside.path(), source.path().join("plugin.json")).expect("link");
    assert_matches!(Plugin::load(source.path()), Err(PluginError::Path));
}

#[test]
fn rejects_names_that_escape_the_registry() {
    let source = tempfile::tempdir().expect("source");
    fs::write(source.path().join("plugin.json"), r#"{"name":"../escape"}"#).expect("manifest");
    assert_matches!(Plugin::load(source.path()), Err(PluginError::Name));
}

#[cfg(unix)]
#[test]
fn rejects_a_named_pipe_manifest_before_opening_it() {
    let source = tempfile::tempdir().expect("source");
    nix::unistd::mkfifo(
        &source.path().join("plugin.json"),
        nix::sys::stat::Mode::S_IRUSR,
    )
    .expect("pipe");
    assert_matches!(Plugin::load(source.path()), Err(PluginError::Path));
}
