"""Daemon workspace, history, and process RPCs."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from typing import Protocol, cast

import grpc
import grpc.aio
from google.protobuf import field_mask_pb2
from mirror.protocol import (
  daemon_pb2,
  daemon_pb2_grpc,
)

from mirror.cli import shell


@dataclasses.dataclass(frozen=True)
class ConversationInfo:
  """Descriptive history metadata, never execution configuration."""

  id: str
  title: str | None
  cwd: str
  model: str
  created_at: datetime.datetime | None
  updated_at: datetime.datetime | None


class _SnapshotCall(Protocol):
  def __next__(self) -> daemon_pb2.WatchWorkspacesResponse: ...

  def cancel(self) -> bool: ...


class _Stub(Protocol):
  def GetFrontendDefaults(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.GetFrontendDefaultsRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.GetFrontendDefaultsResponse: ...

  def ResolveDirectory(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.ResolveDirectoryRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.ResolveDirectoryResponse: ...

  def ListConversations(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.ListConversationsRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.ListConversationsResponse: ...

  def GetPromptHistory(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.GetPromptHistoryRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.GetPromptHistoryResponse: ...

  def WatchWorkspaces(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.WatchWorkspacesRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> _SnapshotCall: ...

  def CreateWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.CreateWorkspaceRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.CreateWorkspaceResponse: ...

  def DeleteWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.DeleteWorkspaceRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.DeleteWorkspaceResponse: ...

  def OpenWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.OpenWorkspaceRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.OpenWorkspaceResponse: ...

  def CloseWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.CloseWorkspaceRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.CloseWorkspaceResponse: ...

  def SwitchWorkspaceSession(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.SwitchWorkspaceSessionRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.SwitchWorkspaceSessionResponse: ...

  def UpdateWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.UpdateWorkspaceRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> daemon_pb2.UpdateWorkspaceResponse: ...


class _AsyncStub(Protocol):
  def Exec(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.ExecRequest,
    *,
    metadata: Sequence[tuple[str, str]],
  ) -> grpc.aio.UnaryStreamCall[
    daemon_pb2.ExecRequest, daemon_pb2.ExecResponse
  ]: ...

  def WatchWorkspaces(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.WatchWorkspacesRequest,
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> AsyncIterator[daemon_pb2.WatchWorkspacesResponse]: ...


class Client:
  """Send explicit metadata mutations; listing never opens a worker."""

  def __init__(
    self,
    address: str,
    *,
    remote_host: str | None = None,
    daemon_id: str | None = None,
  ) -> None:
    self._address = address
    self._metadata = (('mirror-daemon-id', daemon_id),) if daemon_id else ()
    self.remote_host = remote_host
    self.connected = True
    self._options = (
      ('grpc.enable_http_proxy', 0),
      ('grpc.default_authority', 'localhost'),
      ('grpc.max_receive_message_length', 16 * 1024 * 1024),
    )
    self._channel = grpc.insecure_channel(address, options=self._options)
    self._stub = cast(_Stub, daemon_pb2_grpc.DaemonRPCStub(self._channel))

  def conversations(
    self, *, limit: int | None = None, session_ids: Sequence[str] = ()
  ) -> list[ConversationInfo]:
    """Read indexed, activity-ordered pages, stopping at the requested limit."""
    if limit is not None and limit < 1:
      raise ValueError('Conversation limit must be positive')
    result: list[ConversationInfo] = []
    request = daemon_pb2.ListConversationsRequest(
      limit=min(limit or 500, 500), recent_first=True, session_ids=session_ids
    )
    previous: tuple[int, int, str] | None = None
    while True:
      page = self._stub.ListConversations(
        request, metadata=self._metadata, timeout=10
      )
      result.extend(
        ConversationInfo(
          id=item.id,
          title=item.title or None,
          cwd=item.directory,
          model=item.model,
          created_at=item.created_at.ToDatetime(tzinfo=datetime.UTC)
          if item.HasField('created_at')
          else None,
          updated_at=item.updated_at.ToDatetime(tzinfo=datetime.UTC)
          if item.HasField('updated_at')
          else None,
        )
        for item in page.conversations
      )
      if not page.next_after_id or (limit is not None and len(result) >= limit):
        return result
      cursor = (
        page.next_before_updated_at.seconds,
        page.next_before_updated_at.nanos,
        page.next_after_id,
      )
      if not page.HasField('next_before_updated_at') or (
        previous is not None and cursor >= previous
      ):
        raise ValueError('Daemon conversation cursor did not advance')
      previous = cursor
      request.after_id = page.next_after_id
      request.before_updated_at.CopyFrom(page.next_before_updated_at)
      request.limit = (
        min(limit - len(result), 500) if limit is not None else 500
      )

  def conversation_details(
    self, session_ids: Sequence[str]
  ) -> list[ConversationInfo]:
    """Read exact identities in bounded batches, without walking unrelated history."""
    result: list[ConversationInfo] = []
    for start in range(0, len(session_ids), 500):
      result.extend(
        self.conversations(session_ids=session_ids[start : start + 500])
      )
    return result

  def prompt_history(self, conversation_id: str) -> list[str]:
    """Read canonical prompts without attaching to an agent."""
    return list(
      self._stub.GetPromptHistory(
        daemon_pb2.GetPromptHistoryRequest(session_id=conversation_id),
        metadata=self._metadata,
        timeout=10,
      ).prompts
    )

  def snapshot(
    self, request: daemon_pb2.WatchWorkspacesRequest
  ) -> daemon_pb2.WatchWorkspacesResponse:
    """A one-shot listing is exactly the initial watch snapshot."""
    call = self._stub.WatchWorkspaces(
      request, metadata=self._metadata, timeout=130
    )
    try:
      return next(call)
    finally:
      call.cancel()

  async def watch(
    self, request: daemon_pb2.WatchWorkspacesRequest
  ) -> AsyncGenerator[daemon_pb2.WatchWorkspacesResponse | None]:
    """Report disconnects and reconnect read-only subscriptions, never mutations."""
    async with grpc.aio.insecure_channel(
      self._address, options=self._options
    ) as channel:
      stub = cast(_AsyncStub, daemon_pb2_grpc.DaemonRPCStub(channel))
      while True:
        try:
          async for snapshot in stub.WatchWorkspaces(
            request, metadata=self._metadata
          ):
            self.connected = True
            yield snapshot
        except grpc.aio.AioRpcError as error:
          self.connected = False
          if error.code() != grpc.StatusCode.UNAVAILABLE and not (
            error.code() == grpc.StatusCode.CANCELLED
            and error.details() == 'daemon stopped'
          ):
            raise
          yield None
        await asyncio.sleep(0.25)

  def frontend_defaults(self) -> daemon_pb2.GetFrontendDefaultsResponse:
    """Read public execution defaults, never the daemon's private configuration."""
    return self._stub.GetFrontendDefaults(
      daemon_pb2.GetFrontendDefaultsRequest(),
      metadata=self._metadata,
      timeout=10,
    )

  def resolve_directory(
    self, path: str, *, base: str = ''
  ) -> daemon_pb2.ResolveDirectoryResponse:
    """Resolve a path on the daemon host without creating it."""
    return self._stub.ResolveDirectory(
      daemon_pb2.ResolveDirectoryRequest(path=path, base_directory=base),
      metadata=self._metadata,
      timeout=10,
    )

  def create(
    self, request: daemon_pb2.CreateWorkspaceRequest
  ) -> daemon_pb2.Workspace:
    return self._stub.CreateWorkspace(
      request, metadata=self._metadata, timeout=910
    ).workspace

  def drop(self, request: daemon_pb2.DeleteWorkspaceRequest) -> None:
    self._stub.DeleteWorkspace(request, metadata=self._metadata, timeout=130)

  def open_workspace(self, workspace_id: str) -> daemon_pb2.Workspace:
    """Reopen shared membership without starting or authorizing a worker."""
    return self._stub.OpenWorkspace(
      daemon_pb2.OpenWorkspaceRequest(workspace_id=workspace_id),
      metadata=self._metadata,
      timeout=130,
    ).workspace

  def close_workspace(self, workspace_id: str) -> None:
    self._stub.CloseWorkspace(
      daemon_pb2.CloseWorkspaceRequest(workspace_id=workspace_id),
      metadata=self._metadata,
      timeout=130,
    )

  def switch_session(
    self,
    workspace_id: str,
    expected_session_id: str,
    *,
    session_id: str | None = None,
  ) -> daemon_pb2.Workspace:
    """Rebind once; callers must inspect unknown outcomes rather than retrying."""
    return self._stub.SwitchWorkspaceSession(
      daemon_pb2.SwitchWorkspaceSessionRequest(
        workspace_id=workspace_id,
        expected_session_id=expected_session_id,
        session_id=session_id,
      ),
      metadata=self._metadata,
      timeout=10,
    ).workspace

  def update(
    self, workspace: daemon_pb2.Workspace, fields: Sequence[str]
  ) -> daemon_pb2.Workspace:
    return self._stub.UpdateWorkspace(
      daemon_pb2.UpdateWorkspaceRequest(
        workspace_id=workspace.id,
        name=workspace.name,
        preferences=workspace.preferences
        if workspace.HasField('preferences')
        else None,
        update_mask=field_mask_pb2.FieldMask(paths=fields),
      ),
      metadata=self._metadata,
      timeout=10,
    ).workspace

  def close(self) -> None:
    self._channel.close()


@dataclasses.dataclass(frozen=True)
class Shell(shell.Shell):
  """Run processes through the current workspace attachment."""

  connection: Callable[[], tuple[grpc.aio.Channel, str]] | None = None

  @contextlib.asynccontextmanager
  async def open(
    self, argv: Sequence[str], *, data: str = ''
  ) -> AsyncGenerator[shell.Process]:
    if not argv:
      raise ValueError('Workspace process requires an executable')
    if self.connection is None:
      raise RuntimeError('Workspace is not attached')
    channel, attachment_id = self.connection()
    call = cast(_AsyncStub, daemon_pb2_grpc.DaemonRPCStub(channel)).Exec(
      daemon_pb2.ExecRequest(argv=argv, stdin=data),
      metadata=(('mirror-attachment-id', attachment_id),),
    )
    process = _Process(call)
    reading = asyncio.create_task(process.read())
    try:
      yield process
    finally:
      # Transport loss cancels and reaps in the daemon; never replay execution.
      call.cancel()
      reading.cancel()
      await asyncio.gather(reading, return_exceptions=True)
      if process.result.done() and not process.result.cancelled():
        process.result.exception()


class _Process:
  def __init__(
    self,
    call: grpc.aio.UnaryStreamCall[
      daemon_pb2.ExecRequest, daemon_pb2.ExecResponse
    ],
  ) -> None:
    self._call = call
    self._stdout: asyncio.Queue[bytes] = asyncio.Queue(16)
    self._stderr: asyncio.Queue[bytes] = asyncio.Queue(16)
    self.result: asyncio.Future[int] = (
      asyncio.get_running_loop().create_future()
    )

  async def read(self) -> None:
    try:
      async for response in self._call:
        match response.WhichOneof('payload'):
          case 'stdout':
            await self._stdout.put(response.stdout)
          case 'stderr':
            await self._stderr.put(response.stderr)
          case 'exit_code':
            self.result.set_result(response.exit_code)
            break
          case _:
            self.result.set_exception(
              shell.ProcessError('Invalid process response')
            )
            break
      if not self.result.done():
        self.result.set_exception(
          shell.ProcessError(
            'Process completion is unknown; it was not replayed'
          )
        )
    except grpc.aio.AioRpcError as error:
      self.result.set_exception(
        shell.ProcessError(error.details() or 'Workspace process failed')
      )
    # Cancellation must not block on abandoned readers.
    await self._stdout.put(b'')
    await self._stderr.put(b'')

  async def read_stdout(self) -> bytes:
    return await self._stdout.get()

  async def read_stderr(self) -> bytes:
    return await self._stderr.get()

  async def wait(self) -> int:
    return await asyncio.shield(self.result)
