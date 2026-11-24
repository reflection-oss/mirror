//! Claude Code, Codex, and portable plugin directories.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

/// A plugin loaded from a user-selected directory.
#[derive(Debug, Clone)]
pub struct Plugin {
    pub root: PathBuf,
    pub name: String,
    pub description: String,
    pub version: Option<String>,
    pub format: PluginFormat,
}

/// The manifest used to interpret a plugin's components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PluginFormat {
    Portable,
    Codex,
    Claude,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    name: String,
    #[serde(default)]
    description: String,
    version: Option<String>,
}

/// A plugin could not be loaded or registered.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PluginError {
    #[error("plugin file operation failed: {0}")]
    Io(#[from] io::Error),
    #[error("expected plugin.json, .codex-plugin/plugin.json, or .claude-plugin/plugin.json")]
    MissingManifest,
    #[error("not a registered plugin link")]
    NotRegistration,
    #[error("invalid plugin manifest")]
    Manifest,
    #[error("plugin names must start with a letter or digit and contain only letters, digits, underscores, or dashes")]
    Name,
    #[error("plugin component must be a relative path inside the plugin directory")]
    Path,
    #[error("plugin file exceeds 256 KiB")]
    Size,
    #[error("plugin registration name differs from its manifest")]
    Identity,
    #[error("plugin MCP configuration: {0}")]
    Mcp(&'static str),
}

impl Plugin {
    /// Read metadata without executing any plugin code.
    pub fn load(directory: &Path) -> Result<Self, PluginError> {
        let root = directory.canonicalize()?;
        let format = [
            PluginFormat::Portable,
            PluginFormat::Codex,
            PluginFormat::Claude,
        ]
        .into_iter()
        .find(|format| root.join(format.manifest_path()).exists())
        .ok_or(PluginError::MissingManifest)?;
        let manifest: Manifest = serde_json::from_slice(&read_file(&root, format.manifest_path())?)
            .map_err(|_| PluginError::Manifest)?;
        validate_name(&manifest.name)?;
        Ok(Self {
            root,
            name: manifest.name,
            description: manifest.description,
            version: manifest.version,
            format,
        })
    }

    /// Load a registration by its validated name, not an arbitrary path.
    pub(crate) fn registered(directory: &Path, name: &str) -> Result<Self, PluginError> {
        validate_name(name)?;
        let plugin = Self::load(&directory.join(name))?;
        if plugin.name != name {
            return Err(PluginError::Identity);
        }
        Ok(plugin)
    }

    /// Resolve a component inside the plugin, including symlink targets.
    pub fn path(&self, relative: &str) -> Result<PathBuf, PluginError> {
        component_path(&self.root, relative)
    }

    /// Read a bounded, regular file contained in this plugin.
    pub fn read(&self, relative: &str) -> Result<Vec<u8>, PluginError> {
        read_file(&self.root, relative)
    }
}

/// The shared plugin directory on the current execution host.
pub fn plugin_directory() -> Result<PathBuf, PluginError> {
    std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join(".mirror/plugins"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set").into())
}

/// Load registered plugin folders in stable name order.
pub fn load_plugins(directory: &Path) -> Result<Vec<Plugin>, PluginError> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut paths = entries
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let mut plugins = Vec::new();
    for path in paths {
        let Some(name) = path.file_name() else {
            continue;
        };
        if name.as_encoded_bytes().starts_with(b".") {
            continue;
        }
        match Plugin::load(&path) {
            Ok(plugin) if Some(plugin.name.as_str()) == name.to_str() => plugins.push(plugin),
            Ok(_) => eprintln!("Skipping plugin {name:?}: {}", PluginError::Identity),
            Err(error) => eprintln!("Skipping plugin {name:?}: {error}"),
        }
    }
    Ok(plugins)
}

/// Register a live folder; an existing registration is never replaced.
#[cfg(unix)]
pub fn add_plugin(directory: &Path, source: &Path) -> Result<Plugin, PluginError> {
    let plugin = Plugin::load(source)?;
    fs::create_dir_all(directory)?;
    std::os::unix::fs::symlink(&plugin.root, directory.join(&plugin.name))?;
    Ok(plugin)
}

/// Remove a registration while preserving the source directory.
pub fn remove_plugin(directory: &Path, name: &str) -> Result<(), PluginError> {
    validate_name(name)?;
    let path = directory.join(name);
    if !fs::symlink_metadata(&path)?.is_symlink() {
        return Err(PluginError::NotRegistration);
    }
    fs::remove_file(path)?;
    Ok(())
}

fn validate_name(name: &str) -> Result<(), PluginError> {
    if name.len() <= 128
        && name.starts_with(|character: char| character.is_ascii_alphanumeric())
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        Ok(())
    } else {
        Err(PluginError::Name)
    }
}

fn component_path(root: &Path, relative: &str) -> Result<PathBuf, PluginError> {
    let relative = Path::new(relative);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(PluginError::Path);
    }
    let path = root.join(relative).canonicalize()?;
    if !path.starts_with(root) {
        return Err(PluginError::Path);
    }
    Ok(path)
}

fn read_file(root: &Path, relative: &str) -> Result<Vec<u8>, PluginError> {
    let path = component_path(root, relative)?;
    if !fs::metadata(&path)?.is_file() {
        return Err(PluginError::Path);
    }
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(256 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 256 * 1024 {
        return Err(PluginError::Size);
    }
    Ok(bytes)
}

impl PluginFormat {
    /// Manifest path relative to the plugin root.
    pub fn manifest_path(self) -> &'static str {
        match self {
            Self::Portable => "plugin.json",
            Self::Codex => ".codex-plugin/plugin.json",
            Self::Claude => ".claude-plugin/plugin.json",
        }
    }
}

#[cfg(test)]
#[path = "plugin_test.rs"]
mod tests;
