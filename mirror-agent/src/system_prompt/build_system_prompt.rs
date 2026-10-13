//! System-prompt composition.

use mirror_protocol::system_prompt_report::Part;
use mirror_protocol::SystemPromptReport;

use crate::compaction::estimate_chars;

use super::EnvironmentInfo;
use super::Skill;
use super::SystemPromptConfig;

/// Assemble the prompt and metadata-only provenance with cumulative token estimates.
pub fn build_system_prompt(config: &SystemPromptConfig) -> (String, SystemPromptReport) {
    let parts = prompt_parts(config);
    let mut prompt = String::new();
    let mut report = SystemPromptReport::default();
    for (source, text) in parts {
        prompt.push_str(&text);
        let estimated_tokens = estimate_chars(prompt.len()) as u64;
        report.parts.push(Part {
            source,
            estimated_tokens: estimated_tokens - report.estimated_tokens,
        });
        report.estimated_tokens = estimated_tokens;
    }
    (prompt, report)
}

fn prompt_parts(config: &SystemPromptConfig) -> Vec<(String, String)> {
    let mut parts = context_parts(config);
    if let Some(prompt) = config
        .custom_prompt
        .as_ref()
        .filter(|text| !text.is_empty())
    {
        if let Some((_, text)) = parts.first_mut() {
            text.insert_str(0, "\n\n");
        }
        parts.insert(0, ("Custom prompt".into(), prompt.clone()));
    }
    if let Some(policy) = config
        .prompt_injection_policy
        .as_ref()
        .filter(|text| !text.is_empty())
    {
        parts.insert(0, ("Prompt injection policy".into(), policy.clone()));
    }
    // Memory supplied the path, so the pinned bullets follow whatever base the
    // other flags chose.
    if let Some(mut pinned) = config
        .pinned_memories
        .as_deref()
        .and_then(super::load_pinned_memories)
    {
        if !parts.is_empty() {
            pinned.1.insert_str(0, "\n\n");
        }
        parts.push(pinned);
    }
    parts
}

fn context_parts(config: &SystemPromptConfig) -> Vec<(String, String)> {
    let resolved = super::resolve_prompt_instructions(config);
    let mut parts = Vec::new();
    if let Some(environment) = &config.environment {
        parts.push((
            "Environment".into(),
            environment_section(environment, super::local_date().as_deref()),
        ));
    }
    if config.include_project_instructions && !resolved.instructions.is_empty() {
        parts.push(("Project instructions framing".into(), "# Project instructions\n\nThe following instructions come from AGENTS.md / CLAUDE.md files found in the project.".into()));
        parts.extend(resolved.instructions);
    }
    // Top-level separators belong to the following part, including rounding carry.
    for (_, text) in parts.iter_mut().skip(1) {
        text.insert_str(0, "\n\n");
    }
    if config.include_skills {
        let mut catalog = skills_section(&resolved.skills);
        if !parts.is_empty() {
            if let Some((_, text)) = catalog.first_mut() {
                text.insert_str(0, "\n\n");
            }
        }
        parts.extend(catalog);
    }
    parts
}

fn environment_section(environment: &EnvironmentInfo, date: Option<&str>) -> String {
    let mut section = format!(
        "# Environment\n- Working directory: {}\n- Git repository: {}\n- Platform: {}\n- OS version: {}\n- Shell: {}",
        environment.working_directory.display(),
        if environment.is_git_repo { "yes" } else { "no" },
        environment.platform,
        environment.os_version,
        environment.shell,
    );
    if let Some(date) = date {
        section.push_str(&format!("\n- Current date: {date}"));
    }
    section
}

fn skills_section(skills: &[Skill]) -> Vec<(String, String)> {
    let visible = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect::<Vec<_>>();
    if visible.is_empty() {
        return Vec::new();
    }
    let mut parts = vec![(
        "Skills framing".into(),
        "# Skills\n\nThe following skills provide specialized instructions for specific tasks. When a task matches a skill's description, read its SKILL.md file in full before acting. Paths referenced inside a skill are relative to that skill's directory.\n\n<available_skills>".into(),
    )];
    for skill in visible {
        parts.push((
            skill.path.display().to_string(),
            format!(
                "\n  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    <path>{}</path>\n  </skill>",
                escape_xml(&skill.name),
                escape_xml(&skill.description),
                escape_xml(&skill.path.display().to_string()),
            ),
        ));
    }
    parts.push(("Skills framing".into(), "\n</available_skills>".into()));
    parts
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::system_prompt::{
        mentioned_skills, resolve_prompt_instructions, ResolvedInstructions,
    };

    fn environment(path: PathBuf) -> EnvironmentInfo {
        EnvironmentInfo {
            working_directory: path,
            is_git_repo: true,
            platform: "Darwin".to_string(),
            os_version: "Darwin 25.5.0".to_string(),
            shell: "bash 3.2.57".to_string(),
        }
    }

    #[test]
    fn environment_section_lists_the_facts_and_the_date() {
        assert_eq!(
            environment_section(&environment(PathBuf::from("/work/repo")), Some("2026-10-03")),
            "# Environment\n- Working directory: /work/repo\n- Git repository: yes\n- Platform: Darwin\n- OS version: Darwin 25.5.0\n- Shell: bash 3.2.57\n- Current date: 2026-10-03",
        );
        let scratch = EnvironmentInfo {
            is_git_repo: false,
            ..environment(PathBuf::from("/scratch"))
        };
        let section = environment_section(&scratch, None);
        assert!(section.contains("- Git repository: no\n"));
        assert!(section.ends_with("- Shell: bash 3.2.57"));
    }

    #[test]
    fn environment_section_states_the_day_the_prompt_is_built() {
        let config = SystemPromptConfig {
            environment: Some(environment(PathBuf::from("/work/repo"))),
            ..SystemPromptConfig::default()
        };

        let before = crate::system_prompt::local_date().expect("local date");
        let (prompt, report) = build_system_prompt(&config);
        let after = crate::system_prompt::local_date().expect("local date");

        let stated = [before, after].map(|date| format!("\n- Current date: {date}"));
        assert!(stated.iter().any(|line| prompt.ends_with(line)), "{prompt}");
        assert_eq!(report.parts.len(), 1);
        assert_eq!(report.parts[0].source, "Environment");
    }

    #[test]
    fn environment_precedes_project_context() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");
        fs::write(workspace.path().join("AGENTS.md"), "Follow project rules.")
            .expect("write instructions");
        let config = SystemPromptConfig {
            working_directory: Some(workspace.path().into()),
            environment: Some(environment(workspace.path().into())),
            global_instructions_dir: Some(workspace.path().join("no-global")),
            global_skills_dir: Some(workspace.path().join("no-global-skills")),
            ..SystemPromptConfig::default()
        };

        let (prompt, report) = build_system_prompt(&config);
        assert_eq!(report.estimated_tokens, estimate_chars(prompt.len()) as u64);
        assert_eq!(
            report
                .parts
                .iter()
                .map(|part| part.source.as_str())
                .take(2)
                .collect::<Vec<_>>(),
            vec!["Environment", "Project instructions framing"],
        );
        assert!(prompt.starts_with("# Environment\n"));
        assert!(prompt.contains("\n\n# Project instructions\n"));
    }

    #[test]
    fn default_prompt_is_empty_without_repository_context() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("repository marker");
        let config = SystemPromptConfig {
            working_directory: Some(workspace.path().into()),
            global_instructions_dir: Some(workspace.path().join("no-global")),
            global_skills_dir: Some(workspace.path().join("no-global-skills")),
            ..SystemPromptConfig::default()
        };
        let (prompt, report) = build_system_prompt(&config);
        assert_eq!(prompt, "");
        assert_eq!(report.estimated_tokens, 0);
        assert!(report.parts.is_empty());
    }

    #[test]
    fn discovered_context_without_environment_facts_has_no_environment_section() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        fs::create_dir(workspace.path().join(".git")).expect("create repository marker");
        fs::write(workspace.path().join("AGENTS.md"), "Follow project rules.")
            .expect("write instructions");
        let skill_dir = workspace.path().join(".agents/skills/review");
        fs::create_dir_all(&skill_dir).expect("create skill directory");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: review\ndescription: Review A & B.\n---\nbody",
        )
        .expect("write skill");
        let config = SystemPromptConfig {
            working_directory: Some(workspace.path().into()),
            global_instructions_dir: Some(workspace.path().join("no-global")),
            global_skills_dir: Some(workspace.path().join("no-global-skills")),
            ..SystemPromptConfig::default()
        };

        let (prompt, report) = build_system_prompt(&config);
        assert_eq!(report.estimated_tokens, estimate_chars(prompt.len()) as u64);
        assert_eq!(
            report
                .parts
                .iter()
                .map(|part| part.estimated_tokens)
                .sum::<u64>(),
            report.estimated_tokens,
        );
        assert_eq!(
            report
                .parts
                .iter()
                .map(|part| part.source.as_str())
                .collect::<Vec<_>>(),
            vec![
                "Project instructions framing",
                workspace
                    .path()
                    .join("AGENTS.md")
                    .to_str()
                    .expect("instruction path"),
                "Skills framing",
                skill_dir.join("SKILL.md").to_str().expect("skill path"),
                "Skills framing",
            ],
        );
        let metadata = format!("{report:?}");
        assert!(!metadata.contains("Follow project rules."));
        assert!(!metadata.contains("Review A"));
        assert!(!metadata.contains("body"));
        assert!(prompt.starts_with("# Project instructions\n"));
        assert!(!prompt.contains("# Environment"));
        assert!(prompt.contains("Follow project rules."));
        assert!(prompt.contains("<description>Review A &amp; B.</description>"));

        let (additional, report) = build_system_prompt(&SystemPromptConfig {
            custom_prompt: Some("Interactive instructions.".into()),
            ..config
        });
        assert_eq!(additional, format!("Interactive instructions.\n\n{prompt}"));
        assert_eq!(report.parts[0].source, "Custom prompt");
        assert_eq!(
            report.estimated_tokens,
            estimate_chars(additional.len()) as u64
        );
    }

    #[test]
    fn policy_is_a_separate_part_from_agent_identity() {
        let (prompt, report) = build_system_prompt(&SystemPromptConfig {
            custom_prompt: Some("Agent identity.".into()),
            prompt_injection_policy: Some("Shared safety policy.\n\n".into()),
            ..Default::default()
        });
        assert_eq!(prompt, "Shared safety policy.\n\nAgent identity.");
        assert_eq!(report.parts[0].source, "Prompt injection policy");
        assert_eq!(report.parts[1].source, "Custom prompt");
    }

    #[test]
    fn custom_prompt_without_context_is_preserved() {
        let config = SystemPromptConfig {
            custom_prompt: Some("Only this.".to_string()),
            ..SystemPromptConfig::default()
        };

        assert_eq!(build_system_prompt(&config).0, "Only this.");
    }

    #[test]
    fn supplied_sources_bypass_server_files() {
        let workspace = tempfile::tempdir().expect("workspace");
        fs::write(
            workspace.path().join("AGENTS.md"),
            "Server-only instructions.",
        )
        .expect("server file");
        let config = SystemPromptConfig {
            working_directory: Some(workspace.path().into()),
            global_instructions_dir: Some(workspace.path().into()),
            resolved: Some(crate::system_prompt::ResolvedInstructions {
                instructions: vec![(
                    "Sandbox instructions".into(),
                    "Accepted sandbox instructions.".into(),
                )],
                skills: Vec::new(),
            }),
            ..SystemPromptConfig::default()
        };
        let (prompt, _) = build_system_prompt(&config);
        assert!(prompt.contains("Accepted sandbox instructions."));
        assert!(!prompt.contains("Server-only instructions."));
        let empty = SystemPromptConfig {
            resolved: Some(crate::system_prompt::ResolvedInstructions::default()),
            ..config
        };
        let (prompt, _) = build_system_prompt(&empty);
        assert!(!prompt.contains("Accepted sandbox instructions."));
        assert!(!prompt.contains("Server-only instructions."));
    }

    #[test]
    fn custom_prompt_composes_with_opted_in_context() {
        for custom_prompt in [None, Some("Custom base.".into())] {
            for include_skills in [false, true] {
                for include_project_instructions in [false, true] {
                    let config = SystemPromptConfig {
                        custom_prompt: custom_prompt.clone(),
                        include_skills,
                        include_project_instructions,
                        resolved: Some(crate::system_prompt::ResolvedInstructions {
                            instructions: vec![(
                                "Repository instructions".into(),
                                "Repository-specific guidance.".into(),
                            )],
                            skills: vec![Skill {
                                name: "review".into(),
                                description: "Repository-specific skill.".into(),
                                body: "Skill body.".into(),
                                path: PathBuf::from("/skills/review/SKILL.md"),
                                disable_model_invocation: false,
                            }],
                        }),
                        ..SystemPromptConfig::default()
                    };
                    let (prompt, _) = build_system_prompt(&config);
                    if let Some(custom) = &custom_prompt {
                        assert!(prompt.starts_with(custom));
                    }
                    assert_eq!(
                        prompt.contains("Repository-specific guidance."),
                        include_project_instructions
                    );
                    assert_eq!(
                        prompt.contains("Repository-specific skill."),
                        include_skills
                    );
                }
            }
        }
    }

    #[test]
    fn pinned_memories_follow_any_base_whenever_memory_names_the_file() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let pinned = workspace.path().join("PINNED_MEMORIES.md");
        fs::write(&pinned, "- prefer bazelisk\n").expect("write pinned");
        let section = format!(
            "Pinned memories from: {}\n- prefer bazelisk\n",
            pinned.display()
        );
        let config = SystemPromptConfig {
            working_directory: Some(workspace.path().into()),
            global_instructions_dir: Some(workspace.path().join("no-global")),
            include_project_instructions: false,
            include_skills: false,
            pinned_memories: Some(pinned.clone()),
            ..SystemPromptConfig::default()
        };

        let (prompt, report) = build_system_prompt(&config);
        assert_eq!(prompt, section);
        assert_eq!(report.parts[0].source, pinned.display().to_string());

        let (prompt, _) = build_system_prompt(&SystemPromptConfig {
            custom_prompt: Some("Only this.".to_string()),
            ..config.clone()
        });
        assert_eq!(prompt, format!("Only this.\n\n{section}"));

        let (prompt, _) = build_system_prompt(&SystemPromptConfig {
            pinned_memories: None,
            ..config
        });
        assert!(!prompt.contains("Pinned memories"));
    }

    #[test]
    fn client_skills_share_catalog_and_explicit_invocation_with_local_skills() {
        let workspace = tempfile::tempdir().expect("temporary workspace");
        let extension = Skill {
            name: "mirror-extensions".into(),
            description: "Build Mirror widgets.".into(),
            body: "Extension instructions.".into(),
            path: PathBuf::from("/extension/SKILL.md"),
            disable_model_invocation: false,
        };
        let config = SystemPromptConfig {
            working_directory: Some(workspace.path().into()),
            global_instructions_dir: Some(workspace.path().join("no-global")),
            global_skills_dir: Some(workspace.path().join("no-global")),
            additional_skills: vec![extension.clone()],
            ..SystemPromptConfig::default()
        };
        let (prompt, report) = build_system_prompt(&config);
        assert!(prompt.contains("Build Mirror widgets."));
        assert!(!prompt.contains("Extension instructions."));
        assert!(report
            .parts
            .iter()
            .any(|part| part.source == "/extension/SKILL.md"));
        let resolved = resolve_prompt_instructions(&config);
        assert_eq!(resolved.skills, vec![extension.clone()]);
        let invoked = mentioned_skills("Use $mirror-extensions", &resolved.skills);
        assert_eq!(
            invoked[0].invocation_content(),
            extension.invocation_content()
        );

        let directory = workspace.path().join(".agents/skills/mirror-extensions");
        fs::create_dir_all(&directory).expect("skill directory");
        fs::write(
            directory.join("SKILL.md"),
            "---\nname: mirror-extensions\ndescription: Project override.\n---\nUser override.",
        )
        .expect("project skill");
        let local = resolve_prompt_instructions(&config).skills;
        assert_eq!(local.len(), 1);
        assert_eq!(local[0].body, "User override.");
        assert_eq!(local[0].path, directory.join("SKILL.md"));
        assert_matches!(local.as_slice(), [Skill { name, .. }] if name == "mirror-extensions");
        let disabled = SystemPromptConfig {
            include_skills: false,
            ..config.clone()
        };
        assert!(resolve_prompt_instructions(&disabled).skills.is_empty());
        assert_eq!(build_system_prompt(&disabled).0, "");

        let supplied = ResolvedInstructions::default();
        let pre_resolved = SystemPromptConfig {
            resolved: Some(supplied.clone()),
            ..config
        };
        assert_eq!(resolve_prompt_instructions(&pre_resolved), supplied);
    }

    #[test]
    fn manual_only_skills_are_hidden() {
        let section = skills_section(&[Skill {
            name: "manual".to_string(),
            description: "Manual only.".to_string(),
            body: "body".to_string(),
            path: PathBuf::from("/skills/manual/SKILL.md"),
            disable_model_invocation: true,
        }]);

        assert_eq!(section, Vec::<(String, String)>::new());
        assert_matches!(section.as_slice(), []);
    }
}
