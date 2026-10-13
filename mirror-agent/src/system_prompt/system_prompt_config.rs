//! System-prompt configuration.

use std::path::PathBuf;

use super::{EnvironmentInfo, Skill};

/// Inputs controlling system-prompt composition.
#[derive(Debug, Clone)]
pub struct SystemPromptConfig {
    /// Instructions added before opted-in project context and skills.
    pub custom_prompt: Option<String>,
    /// Shared safety policy, independent of agent identity and task instructions.
    pub prompt_injection_policy: Option<String>,
    pub resolved: Option<ResolvedInstructions>,
    /// Where project instructions and skills are discovered when `resolved`
    /// is unset.
    pub working_directory: Option<PathBuf>,
    /// Facts about the client-selected workspace, rendered as the
    /// `# Environment` section only when present.
    pub environment: Option<EnvironmentInfo>,
    pub include_project_instructions: bool,
    pub include_skills: bool,
    /// Client-provided skills; user and project skills take precedence by name.
    pub additional_skills: Vec<Skill>,
    pub global_instructions_dir: Option<PathBuf>,
    /// The user's `PINNED_MEMORIES.md`, present only when the session has
    /// memory. Its bullets are loaded into the prompt at every snapshot, hand
    /// edits included.
    pub pinned_memories: Option<PathBuf>,
    pub global_skills_dir: Option<PathBuf>,
}

impl Default for SystemPromptConfig {
    fn default() -> Self {
        Self {
            custom_prompt: None,
            prompt_injection_policy: None,
            resolved: None,
            working_directory: None,
            environment: None,
            include_project_instructions: true,
            include_skills: true,
            additional_skills: Vec::new(),
            global_instructions_dir: None,
            pinned_memories: None,
            global_skills_dir: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ResolvedInstructions {
    pub instructions: Vec<(String, String)>,
    pub skills: Vec<Skill>,
}
