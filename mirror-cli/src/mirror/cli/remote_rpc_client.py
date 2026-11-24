"""Remote AgentRPC worker transport."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Sequence
from typing import Protocol, cast

import grpc.aio
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  daemon_pb2_grpc,
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import host_tools as _host_tools
from mirror.cli import daemon_client, rpc_client

_MAX_SESSION_PAGE = 1000


class _SessionStub(Protocol):
  def StartSession(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.StartSessionRequest
  ) -> Awaitable[mirror_pb2.StartSessionResponse]: ...

  def ListSessions(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListSessionsRequest
  ) -> Awaitable[mirror_pb2.ListSessionsResponse]: ...


class _MirrorRPCStub(_SessionStub, Protocol):
  def DeleteSession(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.DeleteSessionRequest
  ) -> Awaitable[mirror_pb2.DeleteSessionResponse]: ...

  def ForkSession(  # noqa: N802
    self, request: mirror_pb2.ForkSessionRequest
  ) -> Awaitable[mirror_pb2.ForkSessionResponse]: ...


class _MirrorRuntimeStub(Protocol):
  def EscalateSession(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.EscalateSessionRequest
  ) -> Awaitable[mirror_pb2.EscalateSessionResponse]: ...


_interceptors: list[grpc.aio.ClientInterceptor] = []


def add_interceptor(interceptor: grpc.aio.ClientInterceptor) -> None:
  """Apply an interceptor to future remote channels.

  Interceptors that understand the local daemon wrappers and only add W3C
  trace context may declare supports_daemon_sessions = True.
  """
  _interceptors.append(interceptor)


def remove_interceptor(interceptor: grpc.aio.ClientInterceptor) -> None:
  """Stop applying `interceptor` to remote channels opened from now on."""
  if interceptor in _interceptors:
    _interceptors.remove(interceptor)


class Transport:
  """Open a channel without owning the externally managed worker."""

  def __init__(self, endpoint: str, *, secure: bool = True) -> None:
    self._endpoint = endpoint
    self._secure = secure
    self._channel: grpc.aio.Channel | None = None

  async def connect(
    self,
    host_tools: Sequence[_host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> rpc_client.Connection:
    """Return a ready channel to the configured remote endpoint."""
    del host_tools
    if self._secure:
      self._channel = grpc.aio.secure_channel(
        self._endpoint,
        grpc.ssl_channel_credentials(),
        options=rpc_client.GRPC_CHANNEL_OPTIONS,
        interceptors=tuple(_interceptors),
      )
    else:
      self._channel = grpc.aio.insecure_channel(
        self._endpoint,
        options=rpc_client.GRPC_CHANNEL_OPTIONS,
        interceptors=tuple(_interceptors),
      )
    try:
      await asyncio.wait_for(self._channel.channel_ready(), timeout=30)
      return rpc_client.Connection(self._channel, attachment)
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


class _RemoteDaemonStub(Protocol):
  def GetDaemonInfo(  # noqa: N802, generated RPC name
    self, request: daemon_pb2.GetDaemonInfoRequest, *, timeout: float
  ) -> Awaitable[daemon_pb2.GetDaemonInfoResponse]: ...

  def StartRemoteSession(  # noqa: N802, generated RPC name
    self, request: daemon_pb2.StartRemoteSessionRequest
  ) -> Awaitable[daemon_pb2.StartRemoteSessionResponse]: ...

  def ListRemoteSessions(  # noqa: N802, generated RPC name
    self, request: daemon_pb2.ListRemoteSessionsRequest
  ) -> Awaitable[daemon_pb2.ListRemoteSessionsResponse]: ...


class _DaemonSessions:
  def __init__(self, stub: _RemoteDaemonStub, origin: str) -> None:
    self._stub = stub
    self._origin = origin

  async def StartSession(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.StartSessionRequest
  ) -> mirror_pb2.StartSessionResponse:
    result = await self._stub.StartRemoteSession(
      daemon_pb2.StartRemoteSessionRequest(
        remote_endpoint=self._origin, request=request
      )
    )
    return result.response

  async def ListSessions(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListSessionsRequest
  ) -> mirror_pb2.ListSessionsResponse:
    result = await self._stub.ListRemoteSessions(
      daemon_pb2.ListRemoteSessionsRequest(
        remote_endpoint=self._origin, request=request
      )
    )
    return result.response


@contextlib.asynccontextmanager
async def _session_stub(
  endpoint: str, secure: bool, daemon_address: str | None
) -> AsyncGenerator[_SessionStub, None]:
  """Yield a session stub over the local daemon or a direct server channel.

  The transport is chosen before the stub is yielded, so a failed mutation is
  never replayed over the other path.

  Args:
    endpoint: The server's AgentRPC address.
    secure: Whether the server is reached over TLS.
    daemon_address: The local daemon socket, or None to connect directly.

  Yields:
    A stub for the server's session lifecycle RPCs.
  """
  # Custom gRPC trust roots, proxies and arbitrary Python interceptors belong
  # to the original client transport.
  if (
    daemon_address is not None
    # Resolver and Unix-socket targets (including showcase) belong to gRPC, as
    # do plaintext targets without a port, which gRPC dials on 443 but an
    # `http://` origin would dial on 80.
    and re.fullmatch(
      r'(?:[\w.-]+|\[[0-9a-fA-F:.%]+\])(?::[0-9]+)' + ('?' if secure else ''),
      endpoint,
    )
    and not endpoint.startswith((
      'unix:',
      'unix-abstract:',
      'dns:',
      'ipv4:',
      'ipv6:',
      'xds:',
    ))
    and all(
      getattr(i, 'supports_daemon_sessions', False) for i in _interceptors
    )
    and not any(
      os.environ.get(key)
      for key in (
        'grpc_proxy',
        'https_proxy',
        'http_proxy',
        'GRPC_DEFAULT_SSL_ROOTS_FILE_PATH',
        'GRPC_SSL_CIPHER_SUITES',
      )
    )
  ):
    async with grpc.aio.insecure_channel(
      daemon_address,
      options=rpc_client.GRPC_CHANNEL_OPTIONS + daemon_client.CHANNEL_OPTIONS,
      interceptors=tuple(_interceptors),
    ) as channel:
      daemon = cast(_RemoteDaemonStub, daemon_pb2_grpc.DaemonRPCStub(channel))
      info = await daemon.GetDaemonInfo(
        daemon_pb2.GetDaemonInfoRequest(), timeout=10
      )
      # An active older daemon may have deferred its upgrade. Choose before dispatch:
      # an error from StartSession must never replay the mutation over a second path.
      if info.supports_remote_sessions:
        yield _DaemonSessions(
          daemon, f'{"https" if secure else "http"}://{endpoint}'
        )
        return
  transport = Transport(endpoint, secure=secure)
  try:
    channel = (await transport.connect()).channel
    yield cast(_SessionStub, mirror_pb2_grpc.MirrorRPCStub(channel))
  finally:
    await transport.close()


async def start_remote_session(
  endpoint: str,
  session_id: str,
  *,
  secure: bool = True,
  workspace: mirror_pb2.Workspace | None = None,
  parent_session_id: str = '',
  daemon_address: str | None = None,
) -> str:
  """Provision a session through a remote Mirror server.

  Args:
    endpoint: The server's AgentRPC address.
    session_id: The new session's ID.
    secure: Whether to connect over TLS.
    daemon_address: Local daemon socket shared with the attachment path.
    workspace: The repository to check out; None starts a scratch session.
    parent_session_id: An owned session whose tree the new session joins.

  Returns:
    The ID of the tree's root, or an empty string for a session started
    without a parent or by a server that does not support trees.
  """
  async with _session_stub(endpoint, secure, daemon_address) as stub:
    response = await stub.StartSession(
      mirror_pb2.StartSessionRequest(
        session_id=session_id,
        workspace=workspace,
        parent_session_id=parent_session_id,
      )
    )
    return response.root_session_id


ESCALATION_NOTICE = (
  'Session escalated; it can use your write access until the session ends.'
)


async def escalate_remote_session(
  endpoint: str, session_id: str, *, secure: bool = True
) -> None:
  """Escalate an owned session as the caller for the rest of the session."""
  transport = Transport(endpoint, secure=secure)
  try:
    connection = await transport.connect()
    stub = cast(
      _MirrorRuntimeStub, mirror_pb2_grpc.MirrorRuntimeStub(connection.channel)
    )
    await stub.EscalateSession(
      mirror_pb2.EscalateSessionRequest(session_id=session_id, escalated=True)
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
    channel = (await transport.connect()).channel
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
    channel = (await transport.connect()).channel
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(channel))
    await stub.DeleteSession(
      mirror_pb2.DeleteSessionRequest(session_id=session_id)
    )
  finally:
    await transport.close()


async def remote_session_page(
  endpoint: str,
  *,
  offset: int,
  limit: int,
  secure: bool = True,
  daemon_address: str | None = None,
) -> mirror_pb2.ListSessionsResponse:
  """Return one page of the caller's sessions on a remote Mirror server."""
  async with _session_stub(endpoint, secure, daemon_address) as stub:
    return await stub.ListSessions(
      mirror_pb2.ListSessionsRequest(offset=offset, limit=limit)
    )


async def list_remote_sessions(
  endpoint: str, *, secure: bool = True, daemon_address: str | None = None
) -> list[str]:
  """Return every session visible through a remote Mirror server."""
  async with _session_stub(endpoint, secure, daemon_address) as stub:
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
