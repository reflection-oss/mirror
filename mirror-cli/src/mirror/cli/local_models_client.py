"""Observe and control the daemon's local model service."""

from __future__ import annotations

import asyncio
import time
import types
from collections.abc import Awaitable, Sequence
from typing import Protocol, Self, cast

import grpc.aio
from mirror.protocol import (
  local_models_pb2,
  local_models_pb2_grpc,
)

# Backoff while the daemon is unreachable, so an idle observer stays quiet.
_RETRY_SECONDS = (2, 5, 15, 30)
# A watch that lived this long was healthy, so its end restarts the backoff.
_STABLE_SECONDS = 30


class _Stub(Protocol):
  def Execute(  # noqa: N802, generated RPC name
    self, request: local_models_pb2.ExecuteRequest
  ) -> Awaitable[local_models_pb2.ExecuteResponse]: ...

  def Watch(  # noqa: N802, generated RPC name
    self, request: local_models_pb2.WatchRequest
  ) -> grpc.aio.UnaryStreamCall[
    local_models_pb2.WatchRequest, local_models_pb2.WatchResponse
  ]: ...


def _explain(error: grpc.aio.AioRpcError) -> str:
  """Turn a transport failure into what happened and what the user can do."""
  if error.code() == grpc.StatusCode.UNAVAILABLE:
    return 'The Mirror daemon stopped answering. Run /daemon-restart.'
  details = error.details() or ''
  return f'Local model request failed ({error.code().name}): {details}'


class Client:
  """The daemon's model state, kept current by a watch, and its operations.

  Use as an async context manager to keep the state current for its lifetime.
  Operations work without it, reconnecting on demand.
  """

  def __init__(
    self, address: str, options: Sequence[tuple[str, int | str]] = ()
  ) -> None:
    self._address = address
    self._options = options
    self.state = local_models_pb2.LocalModelState()
    self._lock = asyncio.Lock()
    self._channel: grpc.aio.Channel | None = None
    self._stub: _Stub | None = None
    self._watcher: asyncio.Task[None] | None = None
    self._observer: asyncio.Task[None] | None = None
    self._changed = asyncio.Event()

  async def __aenter__(self) -> Self:
    self._observer = asyncio.create_task(self._observe())
    return self

  async def __aexit__(
    self,
    exception_type: type[BaseException] | None,
    exception: BaseException | None,
    traceback: types.TracebackType | None,
  ) -> None:
    if self._observer is not None:
      self._observer.cancel()
      await asyncio.gather(self._observer, return_exceptions=True)
    await self.close()

  async def changed(self, since: int) -> None:
    """Wait until the daemon reports state newer than revision `since`."""
    while self.state.revision == since:
      await self._changed.wait()

  def _apply(self, state: local_models_pb2.LocalModelState) -> None:
    if state.revision < self.state.revision:
      return
    self.state = state
    previous, self._changed = self._changed, asyncio.Event()
    previous.set()

  async def execute(
    self, request: local_models_pb2.ExecuteRequest
  ) -> local_models_pb2.ExecuteResponse:
    """Run one operation; cancelling the caller cancels it on the daemon."""
    stub = await self._connected()
    try:
      reply = await stub.Execute(request)
    except grpc.aio.AioRpcError as error:
      raise ValueError(_explain(error)) from None
    if reply.HasField('state'):
      self._apply(reply.state)
    if reply.error:
      raise ValueError(reply.error)
    return reply

  async def close(self) -> None:
    """Close this connection without stopping the daemon or its model service."""
    async with self._lock:
      await self._disconnect()

  async def _connected(self) -> _Stub:
    """The stub of a live connection, replacing one whose watch has ended."""
    async with self._lock:
      if self._watcher is not None and self._watcher.done():
        await self._disconnect()
      if self._stub is None:
        try:
          await self._connect()
        except (OSError, TimeoutError, grpc.aio.AioRpcError):
          await self._disconnect()
          raise ValueError(
            'Cannot reach the local model service. Run /daemon-restart.'
          ) from None
      assert self._stub is not None
      return self._stub

  async def _connect(self) -> None:
    async with asyncio.timeout(10):
      self._channel = grpc.aio.insecure_channel(
        self._address, options=self._options
      )
      self._stub = cast(
        _Stub, local_models_pb2_grpc.LocalModelsServiceStub(self._channel)
      )
      watch = self._stub.Watch(local_models_pb2.WatchRequest())
      first = await watch.read()
    if not isinstance(first, local_models_pb2.WatchResponse):
      raise OSError('Local model service closed before reporting its state.')
    self.state = local_models_pb2.LocalModelState()
    self._apply(first.state)
    self._watcher = asyncio.create_task(self._watch(watch))

  async def _disconnect(self) -> None:
    if self._watcher is not None:
      self._watcher.cancel()
      await asyncio.gather(self._watcher, return_exceptions=True)
    if self._channel is not None:
      await self._channel.close()
    self._watcher = self._channel = self._stub = None

  async def _watch(
    self,
    watch: grpc.aio.UnaryStreamCall[
      local_models_pb2.WatchRequest, local_models_pb2.WatchResponse
    ],
  ) -> None:
    try:
      while isinstance(
        response := await watch.read(), local_models_pb2.WatchResponse
      ):
        self._apply(response.state)
    except grpc.aio.AioRpcError:
      pass

  async def _observe(self) -> None:
    """Keep watching, so background downloads are noticed while nothing else asks."""
    failures = 0
    while True:
      try:
        await self._connected()
      except ValueError:
        pass
      else:
        started = time.monotonic()
        if self._watcher is not None:
          await asyncio.wait([self._watcher])
        if time.monotonic() - started > _STABLE_SECONDS:
          failures = 0
      await asyncio.sleep(
        _RETRY_SECONDS[min(failures, len(_RETRY_SECONDS) - 1)]
      )
      failures += 1
