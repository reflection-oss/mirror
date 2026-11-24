//! Plugin skills and Markdown commands in the shared skill catalog.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::plugins::{load_plugins, Plugin, PluginError};

use super::load_instructions::ContextFiles;
use super::skill::{parse_command, read_skills};
use super::Skill;

/// Read namespaced skills and commands from registered folders.
pub fn load_plugin_skills(directory: &Path) -> Result<Vec<Skill>, PluginError> {
    let mut skills = Vec::new();
    for plugin in load_plugins(directory)? {
        match plugin_skills(&plugin) {
            Ok(loaded) => skills.extend(loaded),
            Err(error) => eprintln!("Skipping skills for plugin {}: {error}", plugin.name),
        }
    }
    Ok(skills)
}

#[derive(Default, Deserialize)]
struct Components {
    skills: Option<Paths>,
    commands: Option<Paths>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Paths {
    One(String),
    Many(Vec<String>),
}

impl Paths {
    fn resolve(self, plugin: &Plugin) -> Result<Vec<PathBuf>, PluginError> {
        let paths = match self {
            Self::One(path) => vec![path],
            Self::Many(paths) => paths,
        };
        paths.iter().map(|path| plugin.path(path)).collect()
    }
}

fn plugin_skills(plugin: &Plugin) -> Result<Vec<Skill>, PluginError> {
    let components: Components =
        serde_json::from_slice(&plugin.read(plugin.format.manifest_path())?)
            .map_err(|_| PluginError::Manifest)?;
    let mut roots = default_path(plugin, "skills")?;
    if let Some(paths) = components.skills {
        roots.extend(paths.resolve(plugin)?);
    }
    let mut files = ContextFiles::rooted(plugin.root.clone());
    let mut skills = read_skills(&roots, &mut files)?;
    let mut commands = default_path(plugin, "commands")?;
    if let Some(paths) = components.commands {
        commands.extend(paths.resolve(plugin)?);
    }
    for command in read_commands(commands, &mut files)? {
        if !skills.iter().any(|skill| skill.name == command.name) {
            skills.push(command);
        }
    }
    for skill in &mut skills {
        skill.name = format!("{}:{}", plugin.name, skill.name);
        skill.body = skill
            .body
            .replace("${CLAUDE_PLUGIN_ROOT}", &plugin.root.to_string_lossy())
            .replace("${PLUGIN_ROOT}", &plugin.root.to_string_lossy());
    }
    Ok(skills)
}

fn read_commands(paths: Vec<PathBuf>, files: &mut ContextFiles) -> Result<Vec<Skill>, PluginError> {
    let mut commands = Vec::new();
    for path in paths {
        let paths = if path.is_dir() {
            let mut entries = fs::read_dir(path)?
                .take(1025)
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()?;
            if entries.len() > 1024 {
                return Err(PluginError::Size);
            }
            entries.sort();
            entries
        } else {
            vec![path]
        };
        for path in paths
            .into_iter()
            .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        {
            if let Some(bytes) = files.read(&path)? {
                let raw = std::str::from_utf8(&bytes).map_err(|_| PluginError::Manifest)?;
                if let Some(command) = parse_command(&path, raw) {
                    commands.push(command);
                }
            }
        }
    }
    Ok(commands)
}

fn default_path(plugin: &Plugin, name: &str) -> Result<Vec<PathBuf>, PluginError> {
    match plugin.path(name) {
        Ok(path) => Ok(vec![path]),
        Err(PluginError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(Vec::new())
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "plugin_skills_test.rs"]
mod tests;
