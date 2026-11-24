"""Local MCP declarations for the Rust runtime; no client-owned tool execution."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

from mirror import config
from mirror.cli import mcp_configuration
from mirror.cli.extensions.connections import configuration as mcp

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class McpConnections:
  """Configured endpoints for the connection picker."""

  connections: tuple[mcp_configuration.McpConnection, ...] = ()

  @property
  def connected_servers(self) -> set[str]:
    return {connection.name for connection in self.connections}


def local_connections(
  *, on_auth_required: Callable[[str], None]
) -> McpConnections:
  """Load local declarations independently of desktop placement preferences."""
  connections: list[mcp_configuration.McpConnection] = []
  for name, configured in mcp.load_servers().items():
    if mcp.requires_login(name, configured):
      on_auth_required(name)
      continue
    connections.append(
      mcp_configuration.McpConnection(
        name, configured, str(mcp.config_dir() / 'credentials')
      )
    )
  return McpConnections(tuple(connections))


def daemon_connections() -> tuple[mcp_configuration.McpConnection, ...]:
  """Resolve local MCPs on the daemon host, including over SSH."""
  settings = config.Config(config.default_path()).read()
  extension = config.resolve_extension(
    config.extension_overrides(settings.get('extensions'), 'core/connections')
  )
  if extension.enabled is False:
    return ()
  return local_connections(
    on_auth_required=lambda name: _LOGGER.warning(
      'Local MCP %s needs sign-in', name
    )
  ).connections
