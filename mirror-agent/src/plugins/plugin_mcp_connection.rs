//! Import plugin MCP declarations into Mirror's existing connection files.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::os::unix::fs::DirBuilderExt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::mcp::{valid_mcp_server_name, valid_mcp_url};

use super::{Plugin, PluginError, PluginFormat};

/// One connection ready for explicit registration by the user.
#[derive(Debug)]
pub struct PluginMcpConnection {
    pub server: String,
    pub name: String,
    config: Config,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum Config {
    Http {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        auth: Option<Auth>,
    },
    Stdio {
        command: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Auth {
    Bearer {
        #[serde(rename = "tokenEnvVar")]
        variable: String,
    },
    Oauth,
}

#[derive(Default, Deserialize)]
struct Manifest {
    #[serde(rename = "mcpServers")]
    servers: Option<Source>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Source {
    File(String),
    Servers(BTreeMap<String, serde_json::Value>),
    Many(Vec<Source>),
}

#[derive(Deserialize)]
struct ServerFile {
    #[serde(rename = "mcpServers")]
    servers: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Server {
    #[serde(rename = "type")]
    transport: Option<String>,
    command: Option<String>,
    url: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
}

impl PluginMcpConnection {
    /// Read and validate definitions without connecting or resolving secrets.
    pub fn load(plugin: &Plugin) -> Result<Vec<Self>, PluginError> {
        definitions(plugin)?
            .into_iter()
            .map(|(server, value)| Self::from_definition(plugin, server, value))
            .collect()
    }

    /// Validate only the chosen server, independently of its siblings.
    pub fn selected(plugin: &Plugin, server: &str) -> Result<Self, PluginError> {
        let value = definitions(plugin)?
            .remove(server)
            .ok_or(PluginError::Mcp("server not found"))?;
        Self::from_definition(plugin, server.to_owned(), value)
    }

    fn from_definition(
        plugin: &Plugin,
        server: String,
        value: serde_json::Value,
    ) -> Result<Self, PluginError> {
        let config = serde_json::from_value(value)
            .map_err(|_| PluginError::Mcp("unsupported server configuration"))?;
        let name = format!("plugin_{}_{}", plugin.name, server);
        if !valid_mcp_server_name(&name) {
            return Err(PluginError::Mcp("use lowercase MCP names without double underscores, at most 128 characters including the plugin prefix"));
        }
        Ok(Self {
            server,
            name,
            config: convert(plugin, config)?,
        })
    }

    /// Atomically register one connection, preserving existing settings.
    pub fn register(
        &self,
        directory: &Path,
        oauth: bool,
        allowed_env: &[String],
    ) -> Result<(), PluginError> {
        if !valid_mcp_server_name(&self.name) {
            return Err(PluginError::Mcp("invalid connection name"));
        }
        let approved = |name: &String| allowed_env.contains(name);
        let authorized = match &self.config {
            Config::Http {
                auth: Some(Auth::Bearer { variable }),
                ..
            } => approved(variable),
            Config::Stdio { env, .. } => env.values().all(approved),
            Config::Http { .. } => true,
        };
        if !authorized {
            return Err(PluginError::Mcp("review the selected server and explicitly grant each required host variable with --allow-env NAME"));
        }
        let mut config = self.config.clone();
        if oauth {
            match &mut config {
                Config::Http { auth, .. } if auth.is_none() => *auth = Some(Auth::Oauth),
                _ => {
                    return Err(PluginError::Mcp(
                        "OAuth requires an HTTP server without a bearer header",
                    ))
                }
            }
        }
        let content = serde_json::to_vec_pretty(&config).map_err(|_| PluginError::Manifest)?;
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(directory)?;
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        temporary.write_all(&content)?;
        temporary.write_all(b"\n")?;
        temporary
            .persist_noclobber(directory.join(format!("{}.json", self.name)))
            .map_err(|error| PluginError::Io(error.error))?;
        Ok(())
    }
}

fn definitions(plugin: &Plugin) -> Result<BTreeMap<String, serde_json::Value>, PluginError> {
    let mut servers = BTreeMap::new();
    let default = if plugin.format == PluginFormat::Portable {
        "mcp.json"
    } else {
        ".mcp.json"
    };
    match plugin.read(default) {
        Ok(bytes) => servers.extend(parse_file(&bytes)?),
        Err(PluginError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    if plugin.format != PluginFormat::Portable {
        let manifest: Manifest =
            serde_json::from_slice(&plugin.read(plugin.format.manifest_path())?)
                .map_err(|_| PluginError::Manifest)?;
        if let Some(source) = manifest.servers {
            extend_servers(plugin, source, &mut servers)?;
        }
    }
    Ok(servers)
}

fn parse_file(bytes: &[u8]) -> Result<BTreeMap<String, serde_json::Value>, PluginError> {
    serde_json::from_slice::<ServerFile>(bytes)
        .map(|file| file.servers)
        .map_err(|_| {
            PluginError::Mcp("expected an mcpServers object with stdio or HTTP server definitions")
        })
}

fn extend_servers(
    plugin: &Plugin,
    source: Source,
    servers: &mut BTreeMap<String, serde_json::Value>,
) -> Result<(), PluginError> {
    match source {
        Source::File(path) => servers.extend(parse_file(&plugin.read(&path)?)?),
        Source::Servers(inline) => servers.extend(inline),
        Source::Many(sources) => {
            for source in sources {
                extend_servers(plugin, source, servers)?;
            }
        }
    }
    Ok(())
}

fn convert(plugin: &Plugin, server: Server) -> Result<Config, PluginError> {
    match (server.command, server.url, server.transport.as_deref()) {
        (Some(command), None, None | Some("stdio"))
            if !command.trim().is_empty() && server.headers.is_empty() =>
        {
            let env = server
                .env
                .into_iter()
                .map(|(key, value)| {
                    if !variable_name(&key) {
                        return Err(PluginError::Mcp("invalid environment variable name"));
                    }
                    Ok((key, environment_reference(&value)?.to_string()))
                })
                .collect::<Result<_, PluginError>>()?;
            let command = expand_root(plugin, &command);
            let command = if command.starts_with("./") {
                plugin.path(&command)?.to_string_lossy().into_owned()
            } else {
                command
            };
            Ok(Config::Stdio {
                command,
                args: server
                    .args
                    .iter()
                    .map(|value| expand_root(plugin, value))
                    .collect(),
                env,
            })
        }
        (None, Some(url), None | Some("http" | "streamable-http"))
            if valid_mcp_url(&url) && server.env.is_empty() && server.args.is_empty() =>
        {
            let auth = match server.headers.into_iter().collect::<Vec<_>>().as_slice() {
                [] => None,
                [(key, value)] if key.eq_ignore_ascii_case("authorization") => {
                    let value = value.strip_prefix("Bearer ").ok_or(PluginError::Mcp(
                        "use a Bearer ${VARIABLE} authorization header",
                    ))?;
                    Some(Auth::Bearer {
                        variable: environment_reference(value)?.to_string(),
                    })
                }
                _ => {
                    return Err(PluginError::Mcp(
                        "use OAuth or a Bearer ${VARIABLE} authorization header",
                    ))
                }
            };
            Ok(Config::Http { url, auth })
        }
        _ => Err(PluginError::Mcp(
            "expected a stdio command or HTTPS URL (HTTP is available on loopback)",
        )),
    }
}

fn expand_root(plugin: &Plugin, value: &str) -> String {
    value
        .replace("${CLAUDE_PLUGIN_ROOT}", &plugin.root.to_string_lossy())
        .replace("${PLUGIN_ROOT}", &plugin.root.to_string_lossy())
}

fn environment_reference(value: &str) -> Result<&str, PluginError> {
    value
        .strip_prefix("${")
        .and_then(|value| value.strip_suffix('}'))
        .filter(|name| variable_name(name))
        .ok_or(PluginError::Mcp(
            "use ${VARIABLE} references for environment values and bearer tokens",
        ))
}

fn variable_name(name: &str) -> bool {
    name.starts_with(|character: char| character.is_ascii_alphabetic() || character == '_')
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

#[cfg(test)]
#[path = "plugin_mcp_connection_test.rs"]
mod tests;
