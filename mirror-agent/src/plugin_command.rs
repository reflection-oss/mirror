//! Manage user-selected agent plugin folders on this host.

use std::error::Error;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use mirror_agent::plugins::{
    add_plugin, load_plugins, plugin_directory, remove_plugin, Plugin, PluginMcpConnection,
};
use mirror_agent::system_prompt::load_plugin_skills;

#[derive(Parser)]
#[command(
    name = "mirror plugins",
    about = "Use plugins built for Claude Code or Codex"
)]
struct PluginCommand {
    /// Plugin registrations; defaults to ~/.mirror/plugins.
    #[arg(long, global = true)]
    directory: Option<PathBuf>,
    #[command(subcommand)]
    action: PluginAction,
}

#[derive(Subcommand)]
enum PluginAction {
    /// Register a reviewed local plugin folder, shared by desktop and CLI.
    Add { path: PathBuf },
    /// List registered plugin folders.
    List,
    /// List the namespaced skills and commands available to conversations.
    Skills,
    /// List MCP connections provided by a plugin.
    Connections { name: String },
    /// Register a reviewed plugin MCP server in shared connection settings.
    Connect {
        name: String,
        server: String,
        /// Host variable this server is permitted to receive (repeat per variable).
        #[arg(long)]
        allow_env: Vec<String>,
        /// Configure browser sign-in for an HTTP server.
        #[arg(long)]
        oauth: bool,
        /// Connection directory; defaults to MIRROR_MCP_DIR or ~/.mirror/mcp.
        #[arg(long)]
        mcp_directory: Option<PathBuf>,
    },
    /// Unregister a plugin, keeping its source files.
    Remove { name: String },
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let command = PluginCommand::parse_from(
        std::iter::once("mirror plugins".to_string()).chain(std::env::args().skip(2)),
    );
    let directory = command.directory.map(Ok).unwrap_or_else(plugin_directory)?;
    match command.action {
        PluginAction::Add { path } => {
            let plugin = add_plugin(&directory, &path)?;
            println!("Added {}", plugin.name);
        }
        PluginAction::List => {
            for plugin in load_plugins(&directory)? {
                println!("{} · {:?}", plugin.name, plugin.root);
            }
        }
        PluginAction::Skills => {
            for skill in load_plugin_skills(&directory)? {
                println!("${} · {}", skill.name, skill.description.escape_default());
            }
        }
        PluginAction::Connections { name } => {
            for connection in PluginMcpConnection::load(&find_plugin(&directory, &name)?)? {
                println!("{} · {}", connection.server, connection.name);
            }
        }
        PluginAction::Connect {
            name,
            server,
            oauth,
            allow_env,
            mcp_directory,
        } => {
            let connection =
                PluginMcpConnection::selected(&find_plugin(&directory, &name)?, &server)?;
            let destination = mcp_directory
                .or_else(|| std::env::var_os("MIRROR_MCP_DIR").map(PathBuf::from))
                .map(Ok)
                .unwrap_or_else(|| {
                    plugin_directory().map(|directory| directory.with_file_name("mcp"))
                })?;
            connection.register(&destination, oauth, &allow_env)?;
            println!("Registered {} in Connections", connection.name);
        }
        PluginAction::Remove { name } => {
            remove_plugin(&directory, &name)?;
            println!("Removed {name}");
        }
    }
    Ok(())
}

fn find_plugin(directory: &Path, name: &str) -> Result<Plugin, Box<dyn Error>> {
    load_plugins(directory)?
        .into_iter()
        .find(|plugin| plugin.name == name)
        .ok_or_else(|| "Plugin not found; use mirror plugins list".into())
}
