use std::fs;

use assert_matches::assert_matches;
use pretty_assertions::assert_eq;

use crate::plugins::add_plugin;
use crate::react::ContextHooks;
use crate::{agent_message, chat_record, AgentMessage, ChatRecord, MentionedSkills, UserMessage};

use super::*;

#[tokio::test]
async fn claude_commands_expand_arguments_as_text() {
    let source = tempfile::tempdir().expect("source");
    let registry = tempfile::tempdir().expect("registry");
    fs::create_dir(source.path().join(".claude-plugin")).expect("manifest directory");
    fs::create_dir(source.path().join("commands")).expect("commands");
    fs::write(
        source.path().join(".claude-plugin/plugin.json"),
        r#"{"name":"review"}"#,
    )
    .expect("manifest");
    fs::write(
        source.path().join("commands/check.md"),
        "Review $ARGUMENTS using ${CLAUDE_PLUGIN_ROOT}/guide.md",
    )
    .expect("command");
    add_plugin(registry.path(), source.path()).expect("register");
    let skills = load_plugin_skills(registry.path()).expect("skills");
    assert_matches!(skills.as_slice(), [skill] if skill.name == "review:check" && skill.disable_model_invocation);
    let input = AgentMessage {
        payload: Some(agent_message::Payload::Chat(ChatRecord {
            message: Some(chat_record::Message::User(UserMessage {
                content: "$review:check $(touch never-run)".into(),
                ..UserMessage::default()
            })),
            ..ChatRecord::default()
        })),
        ..AgentMessage::default()
    };
    let context = MentionedSkills::new(skills).on_input(&[], &[input]).await;
    assert_matches!(context[0].chat_message(), Some(chat_record::Message::User(user))
        if user.content.contains("Review $(touch never-run) using")
        && user.content.contains(&format!("{}/guide.md", source.path().canonicalize().expect("root").display())));
}

#[test]
fn codex_skill_paths_are_namespaced_and_deduplicated() {
    let source = tempfile::tempdir().expect("source");
    let registry = tempfile::tempdir().expect("registry");
    fs::create_dir(source.path().join(".codex-plugin")).expect("manifest directory");
    fs::create_dir_all(source.path().join("skills/plan")).expect("skill directory");
    fs::write(
        source.path().join(".codex-plugin/plugin.json"),
        r#"{"name":"design","skills":"./skills"}"#,
    )
    .expect("manifest");
    fs::write(
        source.path().join("skills/plan/SKILL.md"),
        "---\nname: plan\ndescription: Plan a change.\n---\nMake a plan.",
    )
    .expect("skill");
    add_plugin(registry.path(), source.path()).expect("register");
    let skills = load_plugin_skills(registry.path()).expect("skills");
    assert_matches!(skills.as_slice(), [skill] if skill.name == "design:plan" && !skill.disable_model_invocation);
    assert_eq!(skills[0].body, "Make a plan.");
}

#[test]
fn portable_custom_commands_extend_the_default_directory() {
    let source = tempfile::tempdir().expect("source");
    let registry = tempfile::tempdir().expect("registry");
    fs::create_dir(source.path().join("commands")).expect("commands");
    fs::write(
        source.path().join("plugin.json"),
        r#"{"name":"review","commands":["./check.md"]}"#,
    )
    .expect("manifest");
    fs::write(source.path().join("commands/other.md"), "Other command").expect("default command");
    fs::write(source.path().join("check.md"), "Review the change").expect("command");
    add_plugin(registry.path(), source.path()).expect("register");
    assert_matches!(load_plugin_skills(registry.path()).expect("skills").as_slice(), [default, custom] if default.name == "review:other" && custom.name == "review:check");
}

#[test]
fn component_paths_cannot_escape_the_plugin() {
    let source = tempfile::tempdir().expect("source");
    let registry = tempfile::tempdir().expect("registry");
    fs::create_dir(source.path().join(".claude-plugin")).expect("manifest directory");
    fs::write(
        source.path().join(".claude-plugin/plugin.json"),
        r#"{"name":"review","commands":"../outside.md"}"#,
    )
    .expect("manifest");
    add_plugin(registry.path(), source.path()).expect("register");
    let invalid = Plugin::load(source.path()).expect("plugin");
    assert_matches!(plugin_skills(&invalid), Err(PluginError::Path));
    let healthy = tempfile::tempdir().expect("healthy");
    fs::write(
        healthy.path().join("plugin.json"),
        r#"{"name":"valid","commands":"check.md"}"#,
    )
    .expect("manifest");
    fs::write(healthy.path().join("check.md"), "Check").expect("command");
    add_plugin(registry.path(), healthy.path()).expect("register healthy plugin");
    assert_matches!(load_plugin_skills(registry.path()).expect("catalog").as_slice(), [skill] if skill.name == "valid:check");
}
