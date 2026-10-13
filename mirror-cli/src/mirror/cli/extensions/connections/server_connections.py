"""Connections the Mirror server stores for the signed-in user."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncGenerator, Awaitable
from typing import Protocol, cast

import grpc
from mirror.protocol import (
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import remote_rpc_client
from mirror.cli.extensions.connections import gateway_transport

ERRORS = (grpc.RpcError, TimeoutError, RuntimeError, OSError)
_LIST_TIMEOUT_SECONDS = 5
_REVOKED = 'revoked'


class _MirrorRuntimeStub(Protocol):
  def ListProviders(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListProvidersRequest
  ) -> Awaitable[mirror_pb2.ListProvidersResponse]: ...

  def ListConnections(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListConnectionsRequest
  ) -> Awaitable[mirror_pb2.ListConnectionsResponse]: ...

  def CreateConnection(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.CreateConnectionRequest
  ) -> Awaitable[mirror_pb2.CreateConnectionResponse]: ...

  def RevokeConnection(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.RevokeConnectionRequest
  ) -> Awaitable[mirror_pb2.RevokeConnectionResponse]: ...

  def SetDefaultConnection(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.SetDefaultConnectionRequest
  ) -> Awaitable[mirror_pb2.SetDefaultConnectionResponse]: ...

  def BeginOAuthConnection(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.BeginOAuthConnectionRequest
  ) -> Awaitable[mirror_pb2.BeginOAuthConnectionResponse]: ...

  def WaitOAuthConnection(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.WaitOAuthConnectionRequest
  ) -> Awaitable[mirror_pb2.WaitOAuthConnectionResponse]: ...


@dataclasses.dataclass(frozen=True)
class ServerConnections:
  """The Mirror server's providers and the caller's unrevoked connections."""

  providers: tuple[mirror_pb2.Provider, ...]
  connections: tuple[mirror_pb2.Connection, ...]

  def provider(self, name: str) -> mirror_pb2.Provider | None:
    """Return the named provider when the server offers it."""
    return next(
      (provider for provider in self.providers if provider.name == name), None
    )

  def for_provider(self, provider: str) -> tuple[mirror_pb2.Connection, ...]:
    """Return the caller's connections for one provider."""
    return tuple(
      connection
      for connection in self.connections
      if connection.provider == provider
    )


def usable(connection: mirror_pb2.Connection) -> bool:
  """Return whether the server can use a connection's credential."""
  return connection.status == 'active' and (
    not connection.HasField('expires_at') or connection.expires_at > time.time()
  )


@contextlib.asynccontextmanager
async def _runtime(
  gateway: gateway_transport.Gateway,
) -> AsyncGenerator[_MirrorRuntimeStub]:
  transport = remote_rpc_client.Transport(
    gateway.endpoint, secure=gateway.secure
  )
  try:
    channel = await transport.connect()
    yield cast(_MirrorRuntimeStub, mirror_pb2_grpc.MirrorRuntimeStub(channel))
  finally:
    await transport.close()


async def load(gateway: gateway_transport.Gateway) -> ServerConnections:
  """Return the server's providers and the caller's connections."""
  async with asyncio.timeout(_LIST_TIMEOUT_SECONDS), _runtime(gateway) as stub:
    providers = await stub.ListProviders(mirror_pb2.ListProvidersRequest())
    connections: list[mirror_pb2.Connection] = []
    offset = 0
    while True:
      page = await stub.ListConnections(
        mirror_pb2.ListConnectionsRequest(offset=offset)
      )
      connections.extend(page.connections)
      if not page.HasField('next_offset'):
        break
      if page.next_offset <= offset:
        raise RuntimeError('connection pagination did not advance')
      offset = page.next_offset
  return ServerConnections(
    tuple(providers.providers),
    tuple(
      connection for connection in connections if connection.status != _REVOKED
    ),
  )


async def create(
  gateway: gateway_transport.Gateway, provider: str, token: str
) -> mirror_pb2.CreateConnectionResponse:
  """Store a token as the caller's connection for `provider`."""
  async with _runtime(gateway) as stub:
    return await stub.CreateConnection(
      mirror_pb2.CreateConnectionRequest(
        provider=provider, static_token=mirror_pb2.StaticToken(token=token)
      )
    )


async def begin_sign_in(
  gateway: gateway_transport.Gateway, provider: str
) -> mirror_pb2.BeginOAuthConnectionResponse:
  """Start an OAuth sign-in that the Mirror server completes itself."""
  async with _runtime(gateway) as stub:
    return await stub.BeginOAuthConnection(
      mirror_pb2.BeginOAuthConnectionRequest(provider=provider)
    )


async def wait_for_sign_in(
  gateway: gateway_transport.Gateway, authorization_id: str
) -> mirror_pb2.WaitOAuthConnectionResponse:
  """Wait until a begun sign-in connects or fails."""
  async with _runtime(gateway) as stub:
    return await stub.WaitOAuthConnection(
      mirror_pb2.WaitOAuthConnectionRequest(authorization_id=authorization_id)
    )


async def revoke(
  gateway: gateway_transport.Gateway, connection_id: str
) -> None:
  """Revoke one of the caller's connections."""
  async with _runtime(gateway) as stub:
    await stub.RevokeConnection(
      mirror_pb2.RevokeConnectionRequest(connection_id=connection_id)
    )


async def set_default(
  gateway: gateway_transport.Gateway, connection_id: str
) -> None:
  """Make one connection the default for its provider."""
  async with _runtime(gateway) as stub:
    await stub.SetDefaultConnection(
      mirror_pb2.SetDefaultConnectionRequest(connection_id=connection_id)
    )
