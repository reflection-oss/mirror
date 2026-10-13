"""MCP client access to the Mirror server's MCP gateway."""

from __future__ import annotations

import asyncio
import dataclasses
import urllib.parse
from collections.abc import Awaitable
from typing import Protocol, cast

import grpc.aio
from mirror.protocol import (
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import remote_rpc_client

_LIST_TIMEOUT_SECONDS = 5


class _McpGatewayStub(Protocol):
  def ListMcpServers(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListMcpServersRequest
  ) -> Awaitable[mirror_pb2.ListMcpServersResponse]: ...


@dataclasses.dataclass(frozen=True)
class Gateway:
  """A Mirror server whose MCP gateway serves this client."""

  endpoint: str
  secure: bool = True

  def server_url(self, server: str) -> str:
    """Return the Streamable HTTP URL of one gateway server."""
    scheme = 'https' if self.secure else 'http'
    return (
      f'{scheme}://{self.endpoint}/mcp/{urllib.parse.quote(server, safe="")}'
    )


async def list_servers(gateway: Gateway) -> tuple[mirror_pb2.McpServer, ...]:
  """Return the MCP servers in the gateway's catalog."""
  transport = remote_rpc_client.Transport(
    gateway.endpoint, secure=gateway.secure
  )
  try:
    async with asyncio.timeout(_LIST_TIMEOUT_SECONDS):
      channel = await transport.connect()
      response = await _stub(channel).ListMcpServers(
        mirror_pb2.ListMcpServersRequest()
      )
  finally:
    await transport.close()
  return tuple(response.servers)


def _stub(channel: grpc.aio.Channel) -> _McpGatewayStub:
  return cast(_McpGatewayStub, mirror_pb2_grpc.McpGatewayServiceStub(channel))
