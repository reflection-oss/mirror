//! Resolve only the plugin hooks explicitly trusted by the worker operator.

use std::path::Path;
use std::time::Duration;

use regex::Regex;
use serde::Deserialize;

use super::pre_tool_hook::MAX_HOOK_BYTES;
use super::{PreToolError, PreToolHook};
use crate::plugins::{Plugin, PluginFormat};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UserConfig {
    #[serde(default)]
    plugins: Vec<String>,
    #[serde(rename = "description")]
    _description: Option<String>,
    hooks: Option<HookEvents>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HookFile {
    #[serde(rename = "description")]
    _description: Option<String>,
    hooks: HookEvents,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HookEvents {
    #[serde(rename = "PreToolUse")]
    pre_tool_use: Vec<HookGroup>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HookGroup {
    matcher: Option<String>,
    hooks: Vec<HookCommand>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum HookCommand {
    Command {
        command: String,
        timeout: Option<u64>,
        #[serde(rename = "statusMessage")]
        _status_message: Option<String>,
    },
}

pub(super) fn load(
    bytes: &[u8],
    path: &Path,
    cwd: &Path,
) -> Result<Vec<PreToolHook>, PreToolError> {
    let config: UserConfig = serde_json::from_slice(bytes).map_err(config_error)?;
    if config.hooks.is_none() && config.plugins.is_empty() {
        return Err(PreToolError::Configuration(
            "expected hooks.PreToolUse or trusted plugins",
        ));
    }
    let mut commands = Vec::new();
    if let Some(events) = config.hooks {
        commands.extend(compile(events, cwd, None)?);
    }
    let registry = path.with_file_name("plugins");
    let mut loaded = std::collections::HashSet::new();
    for name in config.plugins {
        if !loaded.insert(name.clone()) {
            continue;
        }
        let plugin = Plugin::registered(&registry, &name)
            .map_err(|_| PreToolError::Configuration("cannot load trusted plugin"))?;
        commands.extend(load_plugin(&plugin, cwd)?);
    }
    Ok(commands)
}

fn load_plugin(plugin: &Plugin, cwd: &Path) -> Result<Vec<PreToolHook>, PreToolError> {
    #[derive(Deserialize)]
    struct Manifest {
        hooks: Option<String>,
    }
    const DEFAULT: &str = "hooks/hooks.json";
    let bytes = plugin
        .read(plugin.format.manifest_path())
        .map_err(|_| PreToolError::Configuration("cannot read plugin manifest"))?;
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|_| {
        PreToolError::Configuration("expected a single relative hook file in plugin manifest")
    })?;
    let source = manifest.hooks.as_deref().unwrap_or(DEFAULT);
    let mut commands = Vec::new();
    let mut append = |path: &str, optional: bool| {
        let bytes = match plugin.read(path) {
            Ok(bytes) => bytes,
            Err(crate::plugins::PluginError::Io(error))
                if optional && error.kind() == std::io::ErrorKind::NotFound =>
            {
                return Ok(())
            }
            Err(_) => {
                return Err(PreToolError::Configuration(
                    "cannot read hook file inside plugin",
                ))
            }
        };
        if bytes.len() as u64 > MAX_HOOK_BYTES {
            return Err(PreToolError::OutputTooLarge);
        }
        let file: HookFile = serde_json::from_slice(&bytes).map_err(config_error)?;
        commands.extend(compile(file.hooks, cwd, Some(&plugin.root))?);
        Ok(())
    };
    if plugin.format == PluginFormat::Claude
        && Path::new(source.strip_prefix("./").unwrap_or(source)) != Path::new(DEFAULT)
    {
        append(DEFAULT, true)?;
    }
    append(source, false)?;
    Ok(commands)
}

fn config_error(error: serde_json::Error) -> PreToolError {
    // Serde diagnostics can quote secret-bearing values.
    PreToolError::Configuration(if error.is_data() {
        "expected only hooks.PreToolUse groups with matcher and command hooks (type, command, timeout); other events, fields and hook types are unsupported"
    } else {
        "expected valid JSON"
    })
}

fn compile(
    events: HookEvents,
    cwd: &Path,
    root: Option<&Path>,
) -> Result<Vec<PreToolHook>, PreToolError> {
    let mut commands = Vec::new();
    for group in events.pre_tool_use {
        let matcher = match group.matcher.as_deref() {
            None | Some("" | "*") => None,
            Some(pattern) => Some(
                Regex::new(pattern).map_err(|_| PreToolError::Configuration("invalid matcher"))?,
            ),
        };
        for HookCommand::Command {
            command, timeout, ..
        } in group.hooks
        {
            let timeout = timeout.unwrap_or(600);
            if command.trim().is_empty() || !(1..=600).contains(&timeout) {
                return Err(PreToolError::Configuration(
                    "empty command or timeout outside 1..600",
                ));
            }
            commands.push(PreToolHook::new(
                matcher.clone(),
                command,
                Duration::from_secs(timeout),
                cwd,
                root,
            ));
        }
    }
    Ok(commands)
}
