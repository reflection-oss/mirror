"""MCP declarations for the Rust runtime; no client-owned tool execution."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable, Mapping

import grpc

from mirror.cli import mcp_configuration
from mirror.cli.extensions.connections import configuration as mcp
from mirror.cli.extensions.connections import (
  gateway_transport,
  server_connections,
)

_LOGGER = logging.getLogger(__name__)
_PARALLEL_SEARCH_URL = 'https://search.parallel.ai/mcp'


@dataclasses.dataclass
class McpConnections:
  """Configured endpoints and their service identities for the connection picker."""

  connections: tuple[mcp_configuration.McpConnection, ...] = ()
  services: Mapping[str, str] = dataclasses.field(
    default_factory=dict[str, str]
  )

  @property
  def connected_servers(self) -> set[str]:
    return {connection.name for connection in self.connections}

  @property
  def connected_services(self) -> set[str]:
    return {self.service(name) for name in self.connected_servers}

  def service(self, name: str) -> str:
    return self.services.get(name, name)


def local_connections(
  *, on_auth_required: Callable[[str], None]
) -> McpConnections:
  """Load secret-free declarations; the runtime discovers and invokes tools."""
  connections: list[mcp_configuration.McpConnection] = []
  servers = mcp.load_servers()
  if not os.environ.get('FIRECRAWL_API_KEY', '').strip() and not any(
    server.get('url', '').rstrip('/') == _PARALLEL_SEARCH_URL
    for server in servers.values()
    if isinstance(server.get('url'), str)
  ):
    servers.setdefault(
      'parallel',
      {
        'url': _PARALLEL_SEARCH_URL,
        'enabledTools': ['web_search', 'web_fetch'],
      },
    )
  for name, configured in servers.items():
    if configured.get('enabledTools') == []:
      continue
    if mcp.requires_login(name, configured):
      on_auth_required(name)
      continue
    connections.append(
      mcp_configuration.McpConnection(
        name, configured, str(mcp.config_dir() / 'credentials')
      )
    )
  return McpConnections(tuple(connections))


async def gateway_connections(
  gateway: gateway_transport.Gateway,
  *,
  on_connection_required: Callable[[str], None],
) -> McpConnections:
  """Resolve the authorized catalog into static gateway endpoints."""
  try:
    listed = await gateway_transport.list_servers(gateway)
  except (grpc.RpcError, TimeoutError) as error:
    _LOGGER.warning('MCP gateway catalog unavailable: %s', error)
    return McpConnections()
  try:
    saved = (
      await server_connections.load(gateway)
      if any(server.HasField('provider') for server in listed)
      else None
    )
  except server_connections.ERRORS:
    saved = None
  connections: list[mcp_configuration.McpConnection] = []
  services: dict[str, str] = {}
  for server in listed:
    service = server.provider if server.HasField('provider') else server.name
    services[server.name] = service
    if server.HasField('provider') and not any(
      server_connections.usable(account)
      for account in (saved.for_provider(service) if saved is not None else ())
    ):
      on_connection_required(service)
      continue
    connections.append(
      mcp_configuration.McpConnection(
        server.name, {'url': gateway.server_url(server.name)}
      )
    )
  return McpConnections(tuple(connections), services)
