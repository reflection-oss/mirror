"""Remote AgentRPC worker transport."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from typing import Protocol, cast

import grpc.aio
from mirror.protocol import (
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import host_tools as _host_tools
from mirror.cli import rpc_client

_MAX_SESSION_PAGE = 1000


class _MirrorRPCStub(Protocol):
  """The typed surface used from generated MirrorRPC stubs."""

  def StartSession(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.StartSessionRequest
  ) -> Awaitable[mirror_pb2.StartSessionResponse]: ...

  def DeleteSession(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.DeleteSessionRequest
  ) -> Awaitable[mirror_pb2.DeleteSessionResponse]: ...

  def ListSessions(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListSessionsRequest
  ) -> Awaitable[mirror_pb2.ListSessionsResponse]: ...

  def ForkSession(  # noqa: N802
    self, request: mirror_pb2.ForkSessionRequest
  ) -> Awaitable[mirror_pb2.ForkSessionResponse]: ...


class Transport:
  """Open a channel without owning the externally managed worker."""

  def __init__(self, endpoint: str, *, secure: bool = True) -> None:
    self._endpoint = endpoint
    self._secure = secure
    self._channel: grpc.aio.Channel | None = None

  async def connect(
    self, host_tools: Sequence[_host_tools.ToolDefinition] = ()
  ) -> grpc.aio.Channel:
    """Return a ready channel to the configured remote endpoint."""
    del host_tools
    if self._secure:
      self._channel = grpc.aio.secure_channel(
        self._endpoint,
        grpc.ssl_channel_credentials(),
        options=rpc_client.GRPC_CHANNEL_OPTIONS,
      )
    else:
      self._channel = grpc.aio.insecure_channel(
        self._endpoint, options=rpc_client.GRPC_CHANNEL_OPTIONS
      )
    try:
      await asyncio.wait_for(self._channel.channel_ready(), timeout=30)
      return self._channel
    except TimeoutError as error:
      await self.close()
      raise RuntimeError(
        f'AgentRPC endpoint {self._endpoint!r} was not ready within 30 seconds'
      ) from error

  async def close(self) -> None:
    """Close the client channel without stopping the remote worker."""
    if self._channel is not None:
      await self._channel.close(grace=None)
      self._channel = None


async def start_remote_session(
  endpoint: str,
  session_id: str,
  *,
  secure: bool = True,
  workspace: mirror_pb2.Workspace | None = None,
) -> None:
  """Provision a session through a remote Mirror server."""
  transport = Transport(endpoint, secure=secure)
  try:
    channel = await transport.connect()
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(channel))
    await stub.StartSession(
      mirror_pb2.StartSessionRequest(session_id=session_id, workspace=workspace)
    )
  finally:
    await transport.close()


async def fork_remote_session(
  endpoint: str,
  source_session_id: str,
  at_entry_id: str,
  *,
  secure: bool = True,
) -> str:
  transport = Transport(endpoint, secure=secure)
  try:
    channel = await transport.connect()
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(channel))
    response = await stub.ForkSession(
      mirror_pb2.ForkSessionRequest(
        source_session_id=source_session_id,
        at_entry_id=at_entry_id,
      )
    )
    return response.session_id
  finally:
    await transport.close()


async def delete_remote_session(
  endpoint: str, session_id: str, *, secure: bool = True
) -> None:
  """Delete an owned session through a remote Mirror server."""
  transport = Transport(endpoint, secure=secure)
  try:
    channel = await transport.connect()
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(channel))
    await stub.DeleteSession(
      mirror_pb2.DeleteSessionRequest(session_id=session_id)
    )
  finally:
    await transport.close()


async def list_remote_sessions(
  endpoint: str, *, secure: bool = True
) -> list[str]:
  """Return every session visible through a remote Mirror server."""
  transport = Transport(endpoint, secure=secure)
  try:
    channel = await transport.connect()
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(channel))
    session_ids: list[str] = []
    offset = 0
    total = 1
    while offset < total:
      page = await stub.ListSessions(
        mirror_pb2.ListSessionsRequest(
          offset=offset,
          limit=_MAX_SESSION_PAGE,
        )
      )
      total = page.total
      session_ids.extend(session.id for session in page.sessions)
      offset += len(page.sessions)
      if not page.sessions:
        break
    return session_ids
  finally:
    await transport.close()
