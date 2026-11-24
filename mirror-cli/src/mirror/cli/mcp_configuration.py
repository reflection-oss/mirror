"""Worker-startup MCP configuration; discovery and execution belong to Rust."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import cast

from google.protobuf import json_format
from mirror.protocol import agent_pb2


@dataclasses.dataclass(frozen=True)
class McpConnection:
  """An MCP endpoint with credential references, never credential values."""

  name: str
  config: Mapping[str, object]
  credential_directory: str = ''


def runtime_connections(
  connections: Sequence[McpConnection], *, reconnect: bool = False
) -> agent_pb2.McpConnections:
  """Translate external MCP declarations to the private runtime contract."""
  result = agent_pb2.McpConnections(reconnect=reconnect)
  for connection in connections:
    config = dict(connection.config)
    auth = config.pop('auth', None)
    if isinstance(auth, Mapping):
      auth = cast(Mapping[object, object], auth)
      if auth.get('type') == 'bearer':
        config['bearerEnvVar'] = auth.get('tokenEnvVar')
      elif auth.get('type') == 'oauth':
        config['oauth'] = {'credential': auth.get('credential')}
      else:
        raise ValueError('Invalid MCP authentication type')
    if 'enabledTools' in config:
      config['enabledTools'] = {'names': config['enabledTools']}
    target = result.connections.add(
      name=connection.name, credential_directory=connection.credential_directory
    )
    json_format.ParseDict(config, target.config)
  return result
