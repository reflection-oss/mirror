"""AgentRPC client for the TUI."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import types
import uuid
from collections.abc import (
  AsyncIterator,
  Awaitable,
  Callable,
  Mapping,
  Sequence,
)
from typing import Protocol, cast

import grpc.aio
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import host_tools as _host_tools
from mirror.cli import safety, system_prompt

_MAX_MESSAGE_SIZE = 10 * 1024 * 1024
GRPC_CHANNEL_OPTIONS = (
  ('grpc.max_send_message_length', _MAX_MESSAGE_SIZE),
  ('grpc.max_receive_message_length', _MAX_MESSAGE_SIZE),
)
type MessageHandler = Callable[[agent_pb2.SessionMessage], None]
type ErrorHandler = Callable[[str], Awaitable[None]]


@dataclasses.dataclass(frozen=True, init=False)
class WorkerConfiguration:
  """An immutable worker-process launch configuration."""

  _values: Mapping[str, object]
  _cwd: pathlib.Path
  _custom_prompt: str | None

  def __init__(
    self, values: Mapping[str, object], *, cwd: pathlib.Path
  ) -> None:
    values = dict(values)
    custom_prompt = cast(str | None, values.get('system_prompt'))
    prompt = system_prompt.base_prompt(cwd)
    if custom_prompt:
      prompt += f'\n\n{custom_prompt}'
    values['system_prompt'] = prompt
    values['prompt_injection_policy'] = safety.PROMPT_INJECTION_POLICY
    values.setdefault('include_environment', True)
    object.__setattr__(self, '_values', types.MappingProxyType(values))
    object.__setattr__(self, '_cwd', cwd)
    object.__setattr__(self, '_custom_prompt', custom_prompt)

  @property
  def values(self) -> Mapping[str, object]:
    """The read-only launch values."""
    return self._values

  def updated(self, updates: Mapping[str, object]) -> WorkerConfiguration:
    """Return a configuration with `updates` overlaid."""
    return WorkerConfiguration(
      dict(self._values)
      | {'system_prompt': self._custom_prompt}
      | dict(updates),
      cwd=self._cwd,
    )

  def inference_config(
    self, model: str, reasoning_effort: str, *, remote: bool
  ) -> agent_pb2.InferenceConfig:
    """Translate launch settings into a provider inference configuration."""
    model = str(self.values.get('model', model))
    inference = agent_pb2.InferenceConfig(
      model='' if model == 'server default' else model,
      reasoning_effort=reasoning_effort,
    )
    if not remote:
      model_base_url = self.values.get('model_base_url')
      api = self.values.get('model_api') or self.values.get('api')
      if model_base_url and api == 'openai-codex-responses':
        api = 'openai-chat-completions'
      if api:
        inference.api = str(api)
      base_url = model_base_url or self.values.get('base_url')
      # An explicit empty URL clears the worker's previous endpoint selection.
      inference.base_url = str(base_url or '')
    elif remote_api := self.values.get('remote_model_api'):
      inference.api = str(remote_api)
    return inference

  @property
  def remote_workspace(self) -> mirror_pb2.Workspace | None:
    """The workspace a new hosted session starts in, if one was selected."""
    workspace = self._values.get('remote_workspace')
    return workspace if isinstance(workspace, mirror_pb2.Workspace) else None

  def after_start(self) -> WorkerConfiguration:
    """Return the configuration for restarting an opened session."""
    return self.updated({
      'create_session': False,
      'allow_workspace_mismatch': True,
    })


class _AgentRPCStub(Protocol):
  """The typed surface used from generated AgentRPC stubs."""

  def Session(  # noqa: N802, generated RPC name
    self, requests: AsyncIterator[agent_pb2.ClientMessage]
  ) -> AsyncIterator[agent_pb2.SessionMessage]: ...

  def GetMessages(  # noqa: N802, generated RPC name
    self, request: agent_pb2.GetMessagesRequest
  ) -> Awaitable[agent_pb2.GetMessagesResponse]: ...

  def CallTool(  # noqa: N802, generated RPC name
    self, request: agent_pb2.CallToolRequest
  ) -> Awaitable[agent_pb2.CallToolResponse]: ...


class _MirrorRPCStub(Protocol):
  def ForkSession(  # noqa: N802
    self, request: mirror_pb2.ForkSessionRequest
  ) -> Awaitable[mirror_pb2.ForkSessionResponse]: ...


async def _requests(
  queue: asyncio.Queue[agent_pb2.ClientMessage | None],
  ready: asyncio.Event | None = None,
  attachment_id: str | None = None,
) -> AsyncIterator[agent_pb2.ClientMessage]:
  while (request := await queue.get()) is not None:
    if ready is not None and request.message_id != attachment_id:
      await ready.wait()
    yield request


class Transport(Protocol):
  """Own one AgentRPC channel lifecycle."""

  async def connect(
    self, host_tools: Sequence[_host_tools.ToolDefinition] = ()
  ) -> grpc.aio.Channel: ...

  async def close(self) -> None: ...


class Client:
  """Exchange typed TUI messages over a transport-owned AgentRPC channel."""

  def __init__(
    self,
    transport: Transport,
    conversation_id: str,
    *,
    on_message: MessageHandler,
    on_error: ErrorHandler,
    host_tools_registry: _host_tools.HostToolRegistry | None = None,
    inference_config: agent_pb2.InferenceConfig | None = None,
  ) -> None:
    self._transport = transport
    self._conversation_id = conversation_id
    self._on_message = on_message
    self._on_error = on_error
    self._host_tools_registry = host_tools_registry
    self._channel: grpc.aio.Channel | None = None
    # Accept commands before the worker is connected. The session stream drains
    # this outbox in order once startup and replay have completed.
    self._request_queue: asyncio.Queue[agent_pb2.ClientMessage | None] = (
      asyncio.Queue()
    )
    self._tasks: set[asyncio.Task[None]] = set()
    self._closing = False
    self._attachment_id: str | None = None
    self._attached = asyncio.Event()
    self._attachment_error: str | None = None
    self._replay_buffer: list[agent_pb2.SessionMessage] | None = None
    if inference_config is not None:
      self._attachment_id = uuid.uuid4().hex
      self.send(
        agent_pb2.ClientMessage(
          session_id=conversation_id,
          message_id=self._attachment_id,
          set_inference_config=inference_config,
        )
      )

  async def __aenter__(self) -> Client:
    """Start a client and close it when its owning scope exits."""
    await self.start(replay=False)
    return self

  async def __aexit__(
    self,
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    traceback: types.TracebackType | None,
  ) -> None:
    del exc_type, exc_value, traceback
    await self.close()

  @property
  def transport(self) -> Transport:
    """The transport that owns this client's channel lifecycle."""
    return self._transport

  async def start(
    self, *, replay: bool, wait_for_attachment: bool = False
  ) -> None:
    """Open the channel and start the bidirectional session."""
    try:
      self._channel = await self._transport.connect(
        self._host_tools_registry.tools()
        if self._host_tools_registry is not None
        else ()
      )
      stub = cast(_AgentRPCStub, agent_pb2_grpc.AgentRPCStub(self._channel))
      ready = asyncio.Event() if replay and wait_for_attachment else None
      responses = stub.Session(
        _requests(self._request_queue, ready, self._attachment_id)
      )
      if replay and wait_for_attachment:
        if self._attachment_id is None:
          raise ValueError(
            'TUI replay requires its startup inference configuration'
          )
        self._replay_buffer = []
        self._tasks.add(
          asyncio.create_task(
            self._read_session(responses, self._request_queue)
          )
        )
        await asyncio.wait_for(self._attached.wait(), timeout=30)
        if self._attachment_error is not None:
          raise RuntimeError(self._attachment_error)
        await self._replay_messages(stub)
        buffered, self._replay_buffer = self._replay_buffer, None
        for response in buffered:
          self._on_message(response)
        if ready is not None:
          ready.set()
        return
      if replay:
        await self._replay_messages(stub)
      self._tasks.add(
        asyncio.create_task(self._read_session(responses, self._request_queue))
      )
    except BaseException:
      await self.close()
      raise

  def send(self, request: agent_pb2.ClientMessage) -> None:
    """Queue one typed request for the ordered AgentRPC session stream."""
    if self._closing:
      raise RuntimeError('AgentRPC client is closed')
    self._request_queue.put_nowait(request)

  def enqueue(self, message: agent_pb2.AgentMessage) -> None:
    """Submit one durable record to the active session."""
    self.send(
      agent_pb2.ClientMessage(
        session_id=self._conversation_id,
        message_id=message.id,
        message=message,
      )
    )

  async def messages(self) -> list[agent_pb2.AgentMessage]:
    """Return the active worker's complete typed message history."""
    if self._channel is None:
      raise RuntimeError('AgentRPC client is not connected')
    stub = cast(_AgentRPCStub, agent_pb2_grpc.AgentRPCStub(self._channel))
    return await self._get_messages(stub)

  async def call_tool(
    self, name: str, arguments: Mapping[str, object]
  ) -> _host_tools.ToolResult:
    """Call any tool for an extension, locally or through the Rust worker."""
    if self._closing:
      raise RuntimeError('AgentRPC client is not connected')
    registry = self._host_tools_registry
    if registry is not None and registry.has(name):
      context = _host_tools.ExecutionContext(
        origin=_host_tools.CallOrigin.EXTENSION,
      )
      return await registry.execute(
        name, json.dumps(arguments), context=context
      )
    return await self._call_worker_tool(name, arguments)

  async def fork_session(self, entry_id: str) -> str:
    if self._channel is None:
      raise RuntimeError('AgentRPC client is not connected')
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(self._channel))
    response = await stub.ForkSession(
      mirror_pb2.ForkSessionRequest(
        source_session_id=self._conversation_id,
        at_entry_id=entry_id,
      )
    )
    return response.session_id

  async def close(self) -> None:
    """Close the RPC session and its transport-owned channel."""
    self._closing = True
    self._request_queue.put_nowait(None)
    await self._transport.close()
    self._channel = None
    if self._tasks:
      _, pending = await asyncio.wait(self._tasks, timeout=1)
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
    self._tasks.clear()
    self._host_tools_registry = None

  async def _call_worker_tool(
    self, name: str, arguments: Mapping[str, object]
  ) -> _host_tools.ToolResult:
    if self._channel is None:
      raise RuntimeError('AgentRPC client is not connected')
    stub = cast(_AgentRPCStub, agent_pb2_grpc.AgentRPCStub(self._channel))
    response = await stub.CallTool(
      agent_pb2.CallToolRequest(
        session_id=self._conversation_id,
        name=name,
        arguments=json.dumps(arguments),
      )
    )
    return _host_tools.ToolResult(
      response.content,
      response.success,
      response.display if response.HasField('display') else None,
    )

  async def _get_messages(
    self, stub: _AgentRPCStub
  ) -> list[agent_pb2.AgentMessage]:
    records: list[agent_pb2.AgentMessage] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
      page = await stub.GetMessages(
        agent_pb2.GetMessagesRequest(
          session_id=self._conversation_id,
          offset=offset,
          limit=1000 if total is None else min(1000, total - offset),
        )
      )
      if total is None:
        total = page.total
      records.extend(page.messages)
      offset += len(page.messages)
      if not page.messages:
        if offset < total:
          raise RuntimeError(
            f'Incomplete conversation history: {offset} of {total} records'
          )
        break
    return records

  async def _replay_messages(self, stub: _AgentRPCStub) -> None:
    records = await self._get_messages(stub)
    for index, record in enumerate(records):
      self._on_message(agent_pb2.SessionMessage(agent_message=record))
      if index % 100 == 99:
        await asyncio.sleep(0)
    self._on_message(
      agent_pb2.SessionMessage(
        replay_done=agent_pb2.ReplayDone(
          conversation_id=self._conversation_id,
          has_user_message=any(
            record.WhichOneof('payload') == 'chat'
            and record.chat.WhichOneof('message') == 'user'
            for record in records
          ),
        )
      ),
    )

  async def _read_session(
    self,
    responses: AsyncIterator[agent_pb2.SessionMessage],
    requests: asyncio.Queue[agent_pb2.ClientMessage | None],
  ) -> None:
    try:
      async for response in responses:
        if (
          response.HasField('input_ack')
          and response.input_ack.message_id == self._attachment_id
        ):
          self._attached.set()
        if (
          response.HasField('input_nack')
          and response.input_nack.message_id == self._attachment_id
        ):
          raise RuntimeError(
            f'TUI attachment rejected: {response.input_nack.error}'
          )
        if response.WhichOneof('payload') == 'host_tool_call':
          task = asyncio.create_task(
            self._execute_host_tool(response.host_tool_call, requests)
          )
          self._tasks.add(task)
          task.add_done_callback(self._tasks.discard)
          continue
        if self._replay_buffer is not None:
          self._replay_buffer.append(response)
        else:
          self._on_message(response)
    except grpc.aio.AioRpcError as error:
      self._attachment_error = (
        f'agent worker RPC failed: {error.code().name}: {error.details()}'
      )
      self._attached.set()
      if not self._closing:
        await self._on_error(
          f'agent worker RPC failed: {error.code().name}: {error.details()}'
        )
      return
    except Exception as error:  # noqa: BLE001 - transport isolation boundary.
      self._attachment_error = (
        f'agent worker protocol failed: {type(error).__name__}: {error}'
      )
      self._attached.set()
      if not self._closing:
        await self._on_error(
          f'agent worker protocol failed: {type(error).__name__}: {error}'
        )
      return
    if not self._closing:
      self._attachment_error = 'the agent worker exited unexpectedly.'
      self._attached.set()
      await self._on_error('the agent worker exited unexpectedly.')

  async def _execute_host_tool(
    self,
    call: agent_pb2.ToolCall,
    requests: asyncio.Queue[agent_pb2.ClientMessage | None],
  ) -> None:
    host_tools_registry = self._host_tools_registry
    if host_tools_registry is None or not call.HasField('function'):
      result = _host_tools.ToolResult('host tool session is unavailable', False)
    else:
      result = await host_tools_registry.execute(
        call.function.name, call.function.arguments
      )
    await requests.put(
      agent_pb2.ClientMessage(
        session_id=self._conversation_id,
        message_id=f'host-tool-result:{call.id}',
        host_tool_result=agent_pb2.HostToolResult(
          call_id=call.id,
          content=result.content,
          display=result.display,
          success=result.success,
        ),
      )
    )
