"""An in-process local model service for the CLI's tests; no engine or HTTP."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import AsyncIterator
from typing import override

import grpc.aio
from mirror.protocol import (
  local_models_pb2,
  local_models_pb2_grpc,
)


_RUNNING = (
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_READY,
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING,
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_SERVING,
)


class LocalService(local_models_pb2_grpc.LocalModelsServiceServicer):
  """Answers the operations the CLI sends; tests set what it answers with.

  Like the daemon, it refuses managed reads while the server is down and
  stopping a server another daemon owns.
  """

  def __init__(self, root: pathlib.Path) -> None:
    self.state = local_models_pb2.LocalModelState(
      root=str(root),
      executable='/usr/bin/llama-server',
      context=65536,
      owned=True,
      models_preset=str(root / 'presets.ini'),
      phase=local_models_pb2.LOCAL_MODEL_SERVER_PHASE_STOPPED,
    )
    self.models: list[local_models_pb2.LocalModelEntry] = []
    self.candidates: list[local_models_pb2.LocalModelCandidate] = []
    self.manifest = local_models_pb2.LocalModelDownload()
    self.requests: list[local_models_pb2.ExecuteRequest] = []
    # `info` and `load` wait for this, so a test can hold an operation open.
    self.ready = asyncio.Event()
    self.ready.set()
    self.requested = asyncio.Event()
    self.cancelled = asyncio.Event()
    self._updates: asyncio.Queue[local_models_pb2.LocalModelState] = (
      asyncio.Queue()
    )

  def publish(self) -> None:
    """Tell every watcher about the changes made to `state`."""
    self.state.revision += 1
    self._updates.put_nowait(
      local_models_pb2.LocalModelState.FromString(
        self.state.SerializeToString()
      )
    )

  def _set_phase(
    self,
    phase: local_models_pb2.LocalModelServerPhase.ValueType,
    model: str = '',
  ) -> None:
    self.state.phase = phase
    self.state.active_model = model
    self.publish()

  async def _wait(self) -> None:
    self.requested.set()
    try:
      await self.ready.wait()
    except asyncio.CancelledError:
      self.cancelled.set()
      raise

  @override
  async def Watch(
    self,
    request: local_models_pb2.WatchRequest,
    context: grpc.aio.ServicerContext[
      local_models_pb2.WatchRequest, local_models_pb2.WatchResponse
    ],
  ) -> AsyncIterator[local_models_pb2.WatchResponse]:
    del request, context
    yield local_models_pb2.WatchResponse(state=self.state)
    while True:
      yield local_models_pb2.WatchResponse(state=await self._updates.get())

  @override
  async def Execute(
    self,
    request: local_models_pb2.ExecuteRequest,
    context: grpc.aio.ServicerContext[
      local_models_pb2.ExecuteRequest, local_models_pb2.ExecuteResponse
    ],
  ) -> local_models_pb2.ExecuteResponse:
    del context
    self.requests.append(request)
    reply = local_models_pb2.ExecuteResponse()
    operation = request.WhichOneof('operation')
    if (
      operation in ('inventory', 'info')
      and request.base_url in ('', 'managed')
      and self.state.phase not in _RUNNING
    ):
      reply.error = 'llama.cpp is not running'
      operation = None
    if operation == 'stop' and not self.state.owned:
      reply.error = 'This server belongs to another Mirror daemon'
      operation = None
    match operation:
      case 'start':
        self._set_phase(local_models_pb2.LOCAL_MODEL_SERVER_PHASE_READY)
      case 'load':
        await self._wait()
        self._set_phase(
          local_models_pb2.LOCAL_MODEL_SERVER_PHASE_SERVING, request.load.model
        )
      case 'stop':
        self._set_phase(local_models_pb2.LOCAL_MODEL_SERVER_PHASE_STOPPED)
      case 'context':
        self.state.context = request.context
        self.publish()
      case 'inventory':
        reply.models.extend(self.models)
      case 'catalog' | 'variants':
        reply.candidates.extend(self.candidates)
      case 'review':
        reply.download.CopyFrom(self.manifest)
      case 'download':
        self.state.transfers.add(
          name=request.download.name,
          phase=local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_DOWNLOADING,
          server=request.base_url or 'managed',
        )
        self.publish()
      case 'info':
        await self._wait()
        reply.info.efforts.extend(['none', 'on', 'off'])
        reply.info.default_effort = 'none'
      case _:
        pass
    reply.state.CopyFrom(self.state)
    return reply
