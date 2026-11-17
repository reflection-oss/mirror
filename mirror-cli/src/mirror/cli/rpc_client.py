"""AgentRPC client for the TUI."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import pathlib
import types
import uuid
from collections.abc import (
  AsyncIterator,
  Awaitable,
  Callable,
  Coroutine,
  Mapping,
  Sequence,
)
from typing import Protocol, cast

import grpc.aio
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
  daemon_pb2,
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror import config
from mirror.cli import host_tools as _host_tools
from mirror.cli import models, safety, session_continuity, system_prompt

_MAX_MESSAGE_SIZE = 10 * 1024 * 1024
GRPC_CHANNEL_OPTIONS = (
  ('grpc.max_send_message_length', _MAX_MESSAGE_SIZE),
  ('grpc.max_receive_message_length', _MAX_MESSAGE_SIZE),
)
type MessageHandler = Callable[
  [agent_pb2.SessionMessage], Awaitable[None] | None
]
type ErrorHandler = Callable[[str], Awaitable[None]]
type SettingsHandler = Callable[[daemon_pb2.Workspace], None]
type SettingsWatch = Callable[[], AsyncIterator[daemon_pb2.Workspace]]

_logger = logging.getLogger(__name__)

RESTARTING = 'server is restarting; reconnect'
REATTACH_DELAYS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) + (15.0,) * 40
_TRANSIENT_CODES = frozenset({
  grpc.StatusCode.UNAVAILABLE,
  grpc.StatusCode.DEADLINE_EXCEEDED,
})
_CORE_FAILURE = 'Internal error from Core'
_SETTINGS = frozenset({
  'set_inference_config',
  'set_approval_mode',
  'set_web_provider',
})


class AttachmentLost(RuntimeError):
  """The frontend lost its subscription, not necessarily its worker."""


class WorkspaceClosed(RuntimeError):
  """The daemon deliberately closed the workspace."""


def _transient(error: grpc.aio.AioRpcError) -> bool:
  return (
    error.code()
    in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)
    and error.details()
    != 'session stopped; no automatic restart or replay of commands'
  ) or (error.code(), error.details()) in (
    (
      grpc.StatusCode.RESOURCE_EXHAUSTED,
      'attachment lost events; reconnect to catch up',
    ),
    (
      grpc.StatusCode.RESOURCE_EXHAUSTED,
      'supervisor ingress is full or closed',
    ),
    (grpc.StatusCode.ALREADY_EXISTS, 'client already attached'),
    (grpc.StatusCode.CANCELLED, 'daemon stopped'),
  )


def _error_message(error: Exception) -> str:
  if not isinstance(error, grpc.aio.AioRpcError):
    return str(error) or type(error).__name__
  if (error.code(), error.details()) == (
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    'attachment lost events; reconnect to catch up',
  ):
    return 'Daemon event buffer overflowed; the frontend missed updates.'
  return f'AgentRPC {error.code().name}: {error.details()}'


def inference_speed(inference: agent_pb2.InferenceConfig) -> str:
  """Read the processing speed from a shared provider configuration."""
  if not models.supports_fast_mode(inference.api):
    return 'normal'
  try:
    extra_body: object = json.loads(inference.extra_body or '{}')
  except json.JSONDecodeError:
    return 'normal'
  if not isinstance(extra_body, dict):
    return 'normal'
  key, value = (
    ('speed', 'fast')
    if inference.api == 'anthropic-messages'
    else ('service_tier', 'priority')
  )
  fields = cast(dict[str, object], extra_body)
  return 'fast' if fields.get(key) == value else 'normal'


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
    prompt = (
      '' if values.get('daemon_address') else system_prompt.base_prompt(cwd)
    )
    if custom_prompt:
      prompt += f'\n\n{custom_prompt}'
    values['system_prompt'] = prompt
    values['prompt_injection_policy'] = safety.PROMPT_INJECTION_POLICY
    values.setdefault('include_environment', True)
    if values.get('sandboxed'):
      values['memory_dir'] = None
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
    self,
    model: str,
    reasoning_effort: str,
    *,
    remote: bool,
    speed: str = 'normal',
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
    if speed == 'fast' and models.supports_fast_mode(inference.api):
      extra_body = (
        {'speed': 'fast'}
        if inference.api == 'anthropic-messages'
        else {'service_tier': 'priority'}
      )
      inference.extra_body = json.dumps(extra_body)
    return inference

  @property
  def web_provider(self) -> str | None:
    """The local retrieval selection, independent of inference settings."""
    return config.web_provider(self._values)

  @property
  def remote_workspace(self) -> mirror_pb2.Workspace | None:
    """The workspace a new hosted session starts in, if one was selected."""
    workspace = self._values.get('remote_workspace')
    return workspace if isinstance(workspace, mirror_pb2.Workspace) else None

  def after_start(self) -> WorkerConfiguration:
    """Return the configuration for restarting an opened session."""
    return self.updated({
      'create_session': False,
      'escalate_session': False,
      'allow_workspace_mismatch': True,
      'require_new_worker': False,
    })


class _AgentRPCStub(Protocol):
  """The typed surface used from generated AgentRPC stubs."""

  def Session(  # noqa: N802, generated RPC name
    self,
    requests: AsyncIterator[agent_pb2.ClientMessage],
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> AsyncIterator[agent_pb2.SessionMessage]: ...

  def GetMessages(  # noqa: N802, generated RPC name
    self,
    request: agent_pb2.GetMessagesRequest,
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> Awaitable[agent_pb2.GetMessagesResponse]: ...

  def CallTool(  # noqa: N802, generated RPC name
    self,
    request: agent_pb2.CallToolRequest,
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> Awaitable[agent_pb2.CallToolResponse]: ...


class _MirrorRPCStub(Protocol):
  def ForkWorkspace(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ForkWorkspaceRequest,
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> Awaitable[mirror_pb2.ForkWorkspaceResponse]: ...

  def ForkSession(  # noqa: N802
    self,
    request: mirror_pb2.ForkSessionRequest,
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> Awaitable[mirror_pb2.ForkSessionResponse]: ...


def _core_failure(error: grpc.aio.AioRpcError) -> bool:
  return (
    error.code() == grpc.StatusCode.INTERNAL
    and error.details() == _CORE_FAILURE
  )


def _retryable(error: grpc.aio.AioRpcError) -> bool:
  return error.code() in _TRANSIENT_CODES or _core_failure(error)


async def _requests(
  queue: asyncio.Queue[agent_pb2.ClientMessage | None],
  ready: asyncio.Event | None = None,
  attachment: agent_pb2.ClientMessage | None = None,
) -> AsyncIterator[agent_pb2.ClientMessage]:
  if attachment is not None:
    yield attachment
  while True:
    if ready is not None:
      await ready.wait()
    request = await queue.get()
    if request is None:
      return
    yield request


async def deliver_message(
  handler: MessageHandler, message: agent_pb2.SessionMessage
) -> None:
  """Finish one ordered callback, yielding even when transport data is ready."""
  result = handler(message)
  if result is not None:
    await result
  await asyncio.sleep(0)


def _renewed(
  setting: agent_pb2.ClientMessage, session_id: str
) -> agent_pb2.ClientMessage:
  renewed = agent_pb2.ClientMessage()
  renewed.CopyFrom(setting)
  renewed.session_id = session_id
  renewed.message_id = uuid.uuid4().hex
  return renewed


class _Restarted(RuntimeError):
  """The server ended the session stream to restart; the session continues."""


class _AttemptFailed(Exception):
  """A reattach attempt's replacement stream ended before it took over."""

  def __init__(self, message: str, *, retryable: bool) -> None:
    super().__init__(message)
    self.retryable = retryable


@dataclasses.dataclass(frozen=True)
class Connection:
  """A channel and its transport-specific, non-overwriting attachment handshake."""

  channel: grpc.aio.Channel
  attachment: agent_pb2.ClientMessage | None
  initial_settings: daemon_pb2.Workspace | None = None
  settings: SettingsWatch | None = None
  daemon_attachment_id: str | None = None
  execution_host: bool = True
  # Manual retries may rediscover the daemon, but must still require this worker.
  reconnect: Callable[[bool], Awaitable[Connection]] | None = None


class Transport(Protocol):
  """Own one AgentRPC channel lifecycle."""

  async def connect(
    self,
    host_tools: Sequence[_host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> Connection: ...

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
    on_settings: SettingsHandler | None = None,
    on_settings_error: Callable[[str], None] = _logger.warning,
    on_closed: Callable[[], Awaitable[None]] | None = None,
    on_reconnect: Callable[[str | None], None] | None = None,
    on_disconnect: ErrorHandler | None = None,
    resynchronize: bool = False,
    web_provider: str | None = None,
    reattach_delays: Sequence[float] = REATTACH_DELAYS,
    on_reattaching: Callable[[bool], None] | None = None,
  ) -> None:
    self._transport = transport
    self._on_reattaching = on_reattaching
    self._web_provider = web_provider
    self._conversation_id = conversation_id
    self._on_message = on_message
    self._on_error = on_error
    self._on_settings = on_settings
    self._on_settings_error = on_settings_error
    self._on_closed = on_closed
    self._on_reconnect = on_reconnect
    self._on_disconnect = on_disconnect or on_error
    # Opt-in consumers handle replay_start/state_restored as display replacement.
    self._resynchronize = resynchronize
    self._reconnect: Callable[[bool], Awaitable[Connection]] | None = None
    self._run_task: asyncio.Task[None] | None = None
    self._session_task: asyncio.Task[None] | None = None
    self._disconnect_reason: str | None = None
    self._host_tools_registry = host_tools_registry
    self._channel: grpc.aio.Channel | None = None
    self._metadata: tuple[tuple[str, str], ...] = ()
    self._execution_host = False
    # Accept commands before the worker is connected. The session stream drains
    # this outbox in order once startup and replay have completed.
    self._request_queue: asyncio.Queue[agent_pb2.ClientMessage | None] = (
      asyncio.Queue()
    )
    self._tasks: set[asyncio.Task[None]] = set()
    self._closing = False
    self._pending: dict[str, agent_pb2.ClientMessage] = {}
    self._settings: dict[str, agent_pb2.ClientMessage] = {}
    self._attachment_id: str | None = None
    self._attached = asyncio.Event()
    self._attachment_error: Exception | None = None
    self._stream_retryable = False
    self._replay_buffer: list[agent_pb2.SessionMessage] | None = None
    self._live_requests: (
      asyncio.Queue[agent_pb2.ClientMessage | None] | None
    ) = None
    self._settings_watch: asyncio.Task[None] | None = None
    self._continuity = session_continuity.Continuity()
    self._reattach_delays = tuple(reattach_delays)
    self.inference_config = inference_config
    self.shared_settings: daemon_pb2.Workspace | None = None
    self._attachment = (
      agent_pb2.ClientMessage(
        session_id=conversation_id,
        message_id=uuid.uuid4().hex,
        set_inference_config=inference_config,
      )
      if inference_config is not None
      else None
    )

    if web_provider is not None:
      self.set_web_provider(web_provider)

  def set_web_provider(self, provider: str) -> None:
    """Apply retrieval selection through the ordered session control stream."""
    if provider not in ('you', 'tavily', 'firecrawl'):
      raise ValueError('web_provider must be you, tavily, or firecrawl')
    self.send(
      agent_pb2.ClientMessage(
        session_id=self._conversation_id,
        message_id=uuid.uuid4().hex,
        set_web_provider=agent_pb2.SetWebProvider(provider=provider),
      )
    )
    self._web_provider = provider

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
    self,
    *,
    replay: bool,
    wait_for_attachment: bool = False,
    on_attached: Callable[[], Awaitable[None]] | None = None,
  ) -> None:
    """Open the channel and start the bidirectional session."""
    if self._closing or (
      self._run_task is not None and not self._run_task.done()
    ):
      raise RuntimeError('AgentRPC client is closed or already started')
    started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    self._run_task = asyncio.create_task(
      self._run(
        started,
        replay,
        wait_for_attachment or on_attached is not None,
        on_attached,
      )
    )
    try:
      done, _ = await asyncio.wait(
        (started, self._run_task), return_when=asyncio.FIRST_COMPLETED
      )
      if started not in done:
        await self._run_task
        raise RuntimeError('AgentRPC client stopped before attachment')
      await started
    except BaseException:
      await self.close()
      raise

  async def reconnect(self) -> bool:
    """Retry the same worker; return False only when reopening is necessary."""
    if self._closing:
      raise WorkspaceClosed('Workspace closed.')
    if self._reconnect is None:
      return False
    if self._run_task is not None and not self._run_task.done():
      return True
    if self._on_reconnect is not None:
      self._on_reconnect(self._disconnect_reason or 'Connection interrupted.')
    started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    self._run_task = asyncio.create_task(
      self._run(started, True, True, None, manual=True)
    )
    try:
      done, _ = await asyncio.wait(
        (started, self._run_task), return_when=asyncio.FIRST_COMPLETED
      )
      if started not in done:
        await self._run_task
        raise AttachmentLost('AgentRPC client stopped before attachment')
      await started
    except grpc.aio.AioRpcError as error:
      if (error.code(), error.details()) == (
        grpc.StatusCode.FAILED_PRECONDITION,
        'attachment worker is no longer running',
      ):
        return False
      raise
    except asyncio.CancelledError:
      await self.close()
      raise
    return True

  async def _run(
    self,
    started: asyncio.Future[None] | None,
    replay: bool,
    wait_for_attachment: bool,
    on_attached: Callable[[], Awaitable[None]] | None,
    *,
    manual: bool = False,
  ) -> None:
    attempts = 0
    loop = asyncio.get_running_loop()
    connected_at = loop.time()
    deadline: float | None = (
      loop.time() + 45 if self._disconnect_reason else None
    )

    async def attached_once() -> None:
      nonlocal on_attached
      callback, on_attached = on_attached, None
      if callback is not None:
        await callback()

    while not self._closing:
      try:
        async with asyncio.timeout_at(
          deadline if self._disconnect_reason else None
        ):
          connection = (
            await self._reconnect(manual)
            if self._disconnect_reason and self._reconnect is not None
            else await self._transport.connect(
              self._host_tools_registry.tools()
              if self._host_tools_registry is not None
              else (),
              attachment=self._attachment,
            )
          )
        # Synchronizing history is backpressured by rendering, not a connection timeout.
        ready = await self._start_connection(
          connection,
          replay,
          wait_for_attachment,
          attached_once if on_attached is not None else None,
        )
        if self._disconnect_reason:
          self._disconnect_reason = None
          if self._on_reconnect is not None:
            self._on_reconnect(None)
        if ready is not None:
          ready.set()
        if started is not None and not started.done():
          started.set_result(None)
        manual = False
        connected_at = loop.time()
        await self._follow()
        return
      except (
        Exception
      ) as error:  # The subscription supervisor owns terminal reporting.
        await self._disconnect()
        if (
          isinstance(error, grpc.aio.AioRpcError)
          and self._metadata
          and error.code() == grpc.StatusCode.CANCELLED
          and error.details() == 'workspace closed'
        ):
          error = WorkspaceClosed('Workspace closed.')
        if manual and started is not None and not started.done():
          started.set_exception(error)
          return
        transient = isinstance(error, AttachmentLost) or (
          self._disconnect_reason is not None
          and (
            isinstance(error, TimeoutError)
            or isinstance(error, grpc.aio.AioRpcError)
            and _transient(error)
          )
        )
        if transient:
          if not self._disconnect_reason and loop.time() - connected_at >= 30:
            attempts, deadline = 0, None
          # Keep the original disconnect cause if reattachment also fails.
          self._disconnect_reason = self._disconnect_reason or _error_message(
            error
          )
          if deadline is None:
            deadline = loop.time() + 45
          # Pending daemon reservations expire after 30s; a reconnect must outlast them.
          if (
            self._reconnect is not None
            and attempts < 10
            and loop.time() < deadline
          ):
            if self._on_reconnect is not None:
              self._on_reconnect(self._disconnect_reason)
            await asyncio.sleep(min(0.25 * 2**attempts, 5))
            attempts += 1
            replay, wait_for_attachment = True, True
            continue
        if started is not None and not started.done():
          started.set_exception(
            AttachmentLost(self._disconnect_reason) if transient else error
          )
        elif isinstance(error, WorkspaceClosed):
          self._closing = True
          if self._on_closed is not None:
            await self._on_closed()
          else:
            await self._on_error(str(error))
        elif transient:
          await self._on_disconnect(
            self._disconnect_reason or _error_message(error)
          )
        else:
          await self._on_error(_error_message(error))
        return

  async def _start_connection(
    self,
    connection: Connection,
    replay: bool,
    wait_for_attachment: bool,
    on_attached: Callable[[], Awaitable[None]] | None = None,
  ) -> asyncio.Event | None:
    stub = self._adopt(connection)
    ready = asyncio.Event() if wait_for_attachment else None
    self._live_requests = self._request_queue
    responses = stub.Session(
      _requests(self._request_queue, ready, self._attachment),
      metadata=self._metadata,
    )
    if ready is not None:
      if self._attachment_id is None:
        raise ValueError(
          'TUI replay requires its startup inference configuration'
        )
      # Daemon replay is already ordered before its attachment ack; consume it
      # with callback backpressure rather than growing an unbounded second queue.
      self._replay_buffer = [] if not self._metadata else None
      self._session_task = self._spawn(
        self._read_session(responses, self._request_queue)
      )
      try:
        await asyncio.wait_for(
          self._attached.wait(), timeout=None if self._metadata else 30
        )
      except TimeoutError as error:
        raise AttachmentLost('Agent attachment timed out') from error
      if self._attachment_error is not None:
        raise self._attachment_error
      if on_attached is not None:
        # Frontend hooks need a live stream, not ownership of host-tool execution.
        ready.set()
        await on_attached()
      await self._load_history(stub, replay)
      await self._flush_replay_buffer()
      if self._attachment_error is not None:
        raise self._attachment_error
      return ready
    await self._load_history(stub, replay)
    self._session_task = self._spawn(
      self._read_session(responses, self._request_queue)
    )
    return None

  def _adopt(self, connection: Connection) -> _AgentRPCStub:
    self._channel = connection.channel
    self._metadata = (
      (('mirror-attachment-id', connection.daemon_attachment_id),)
      if connection.daemon_attachment_id is not None
      else ()
    )
    if self._metadata and self._resynchronize:
      self._metadata += (
        (
          'mirror-resynchronize',
          'history' if self._disconnect_reason else '1',
        ),
      )
    self._execution_host = connection.execution_host
    self._attachment = connection.attachment
    self._attachment_id = (
      self._attachment.message_id if self._attachment is not None else None
    )
    self._reconnect = (
      connection.reconnect if self._on_reconnect is not None else None
    )
    self._attached.clear()
    self._attachment_error = None
    self._stream_retryable = False
    if connection.initial_settings is not None:
      self._receive_settings(connection.initial_settings)
    if connection.settings is not None:
      if self._settings_watch is not None:
        self._settings_watch.cancel()
      self._settings_watch = self._spawn(
        self._read_settings(connection.settings)
      )
    return cast(
      _AgentRPCStub,
      agent_pb2_grpc.AgentRPCStub(self._channel),
    )

  def _spawn(self, work: Coroutine[object, object, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(work)
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)
    return task

  async def _load_history(self, stub: _AgentRPCStub, replay: bool) -> None:
    if self._metadata:
      return
    if replay:
      await self._replay_messages(stub)
    elif self._attachment_id is not None:
      try:
        head = await stub.GetMessages(
          agent_pb2.GetMessagesRequest(
            session_id=self._conversation_id, offset=0, limit=1
          ),
          metadata=self._metadata,
        )
      except grpc.aio.AioRpcError as error:
        if error.code() != grpc.StatusCode.UNIMPLEMENTED:
          raise
        return
      self._continuity.advance(head.total)
      if head.records_turn_ends:
        self._continuity.expect_turn_end_records()

  async def _flush_replay_buffer(self) -> None:
    while self._replay_buffer:
      buffered, self._replay_buffer = self._replay_buffer, []
      for response in buffered:
        await self._deliver(response)
    self._replay_buffer = None

  def _receive_settings(self, settings: daemon_pb2.Workspace) -> None:
    self.inference_config = settings.inference
    self.shared_settings = settings
    if self._on_settings is not None:
      self._on_settings(settings)

  async def _read_settings(self, settings: SettingsWatch) -> None:
    delay = 0.25
    reported: str | None = None
    while not self._closing:
      try:
        async for snapshot in settings():
          self._receive_settings(snapshot)
          delay = 0.25
          reported = None
        detail, retry = 'stream ended', True
      except grpc.aio.AioRpcError as error:
        detail = f'{error.code().name}: {error.details()}'
        retry = error.code() == grpc.StatusCode.UNAVAILABLE or (
          error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
          and error.details() == 'supervisor ingress is full or closed'
        )
      if self._closing:
        return
      self.shared_settings = None
      recovery = 'Retrying.' if retry else 'Reconnect to refresh settings.'
      message = (
        f'Shared settings unavailable ({detail}). '
        f'Displaying last known settings. {recovery}'
      )
      # Settings observations never report a worker failure or detach its host.
      if reported != message:
        self._on_settings_error(message)
        reported = message
      if not retry:
        return
      await asyncio.sleep(delay)
      delay = min(delay * 2, 5)

  def send(self, request: agent_pb2.ClientMessage) -> None:
    """Queue one typed request for the ordered AgentRPC session stream."""
    if self._closing:
      raise RuntimeError('AgentRPC client is closed')
    if self._disconnect_reason and self._channel is None:
      raise RuntimeError('AgentRPC client is disconnected')
    self._pending[request.message_id] = request
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
    stub = cast(
      _AgentRPCStub,
      agent_pb2_grpc.AgentRPCStub(self._channel),
    )
    return await self._get_messages(stub)

  @property
  def execution_host(self) -> bool:
    """Whether this attachment owns the worker's client-local execution routes."""
    return self._execution_host

  async def call_tool(
    self, name: str, arguments: Mapping[str, object]
  ) -> _host_tools.ToolResult:
    """Call any tool for an extension, locally or through the Rust worker."""
    if self._closing:
      raise RuntimeError('AgentRPC client is not connected')
    if self._metadata and not self._execution_host:
      return _host_tools.ToolResult(
        'Only the execution host may invoke tools', False
      )
    registry = self._host_tools_registry
    if registry is not None and registry.has(name):
      context = _host_tools.ExecutionContext(
        origin=_host_tools.CallOrigin.EXTENSION,
      )
      return await registry.execute(
        name, json.dumps(arguments), context=context
      )
    return await self._call_worker_tool(name, arguments)

  async def fork_workspace(
    self, session_id: str, directory: pathlib.Path | None, name: str
  ) -> pathlib.Path:
    """Fork into a checkout while retaining a caller-known recovery identity."""
    if self._channel is None:
      raise RuntimeError('AgentRPC client is not connected')
    stub = cast(_MirrorRPCStub, mirror_pb2_grpc.MirrorRPCStub(self._channel))
    response = await stub.ForkWorkspace(
      mirror_pb2.ForkWorkspaceRequest(
        source_session_id=self._conversation_id,
        session_id=session_id,
        directory=str(directory) if directory is not None else '',
        name=name,
      ),
      metadata=self._metadata,
    )
    return pathlib.Path(response.workspace)

  async def fork_session(self, entry_id: str) -> str:
    if self._channel is None:
      raise RuntimeError('AgentRPC client is not connected')
    stub = cast(
      _MirrorRPCStub,
      mirror_pb2_grpc.MirrorRPCStub(self._channel),
    )
    response = await stub.ForkSession(
      mirror_pb2.ForkSessionRequest(
        source_session_id=self._conversation_id,
        at_entry_id=entry_id,
      ),
      metadata=self._metadata,
    )
    return response.session_id

  async def close(self) -> None:
    """Close the RPC session and its transport-owned channel."""
    self._closing = True
    self._pending.clear()
    if self._run_task is not None:
      self._run_task.cancel()
      await asyncio.gather(self._run_task, return_exceptions=True)
    await self._disconnect()
    self._host_tools_registry = None

  async def _disconnect(self) -> None:
    for task in self._tasks:
      task.cancel()
    await asyncio.gather(*self._tasks, return_exceptions=True)
    self._tasks.clear()
    # Never drain an old stream's outbox into a new attachment.
    self._request_queue = asyncio.Queue()
    self._pending.clear()
    await self._transport.close()
    self._channel = None
    self._execution_host = False

  async def _call_worker_tool(
    self, name: str, arguments: Mapping[str, object]
  ) -> _host_tools.ToolResult:
    if self._channel is None:
      raise RuntimeError('AgentRPC client is not connected')
    stub = cast(
      _AgentRPCStub,
      agent_pb2_grpc.AgentRPCStub(self._channel),
    )
    response = await stub.CallTool(
      agent_pb2.CallToolRequest(
        session_id=self._conversation_id,
        name=name,
        arguments=json.dumps(arguments),
        web_provider=self._web_provider,
      ),
      metadata=self._metadata,
    )
    return _host_tools.ToolResult(
      response.content,
      response.success,
      response.display if response.HasField('display') else None,
    )

  async def _get_messages(
    self, stub: _AgentRPCStub, *, offset: int = 0
  ) -> list[agent_pb2.AgentMessage]:
    records: list[agent_pb2.AgentMessage] = []
    total: int | None = None
    while total is None or offset < total:
      page = await stub.GetMessages(
        agent_pb2.GetMessagesRequest(
          session_id=self._conversation_id,
          offset=offset,
          limit=1000 if total is None else min(1000, total - offset),
        ),
        metadata=self._metadata,
      )
      if total is None:
        total = page.total
      if page.records_turn_ends:
        self._continuity.expect_turn_end_records()
      records.extend(page.messages)
      for record in page.messages:
        self._pending.pop(record.id, None)
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
    for record in records:
      await self._deliver(agent_pb2.SessionMessage(agent_message=record))
    await self._deliver(
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

  async def _replay_prefix(self, count: int) -> None:
    if self._channel is None:
      raise AttachmentLost('Session disconnected during synchronization')
    stub = cast(_AgentRPCStub, agent_pb2_grpc.AgentRPCStub(self._channel))
    offset = 0
    while offset < count:
      page = await stub.GetMessages(
        agent_pb2.GetMessagesRequest(
          session_id=self._conversation_id,
          offset=offset,
          limit=min(128, count - offset),
        ),
        metadata=self._metadata,
      )
      if not page.messages or len(page.messages) > count - offset:
        raise AttachmentLost('Conversation changed during synchronization')
      for record in page.messages:
        await self._emit(agent_pb2.SessionMessage(agent_message=record))
      offset += len(page.messages)

  async def _read_session(
    self,
    responses: AsyncIterator[agent_pb2.SessionMessage],
    requests: asyncio.Queue[agent_pb2.ClientMessage | None],
  ) -> None:
    retryable = False
    try:
      await self._receive(responses, requests)
    except grpc.aio.AioRpcError as error:
      retryable = _retryable(error)
      ending = self._stream_failure(error)
    except AttachmentLost as error:
      ending = error
    except Exception as error:  # noqa: BLE001 - transport isolation boundary.
      ending = RuntimeError(
        f'agent worker protocol failed: {type(error).__name__}: {error}'
      )
    else:
      ending = (
        AttachmentLost('Agent session stream ended')
        if self._metadata
        else RuntimeError('the agent worker exited unexpectedly.')
      )
    if self._retired(requests):
      return
    self._attachment_error = ending
    self._stream_retryable = retryable
    self._attached.set()

  async def _receive(
    self,
    responses: AsyncIterator[agent_pb2.SessionMessage],
    requests: asyncio.Queue[agent_pb2.ClientMessage | None],
  ) -> None:
    async for response in responses:
      if self._retired(requests):
        return
      if response.HasField('replay_start'):
        self._continuity = self._continuity.replaced()
        await self._emit(response)
        await self._replay_prefix(response.replay_start.history_prefix)
        continue
      self._settle_input(response)
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
        self._spawn(self._execute_host_tool(response.host_tool_call, requests))
        await asyncio.sleep(0)
        continue
      if self._replay_buffer is not None:
        self._replay_buffer.append(response)
        await asyncio.sleep(0)
      else:
        await self._deliver(response)

  def _stream_failure(self, error: grpc.aio.AioRpcError) -> Exception:
    if self._metadata and (error.code(), error.details()) == (
      grpc.StatusCode.CANCELLED,
      'workspace closed',
    ):
      return WorkspaceClosed('Workspace closed.')
    if self._resumable(error):
      return _Restarted(_error_message(error))
    if self._metadata and _transient(error):
      return AttachmentLost(_error_message(error))
    return RuntimeError(_error_message(error))

  def _retired(
    self, requests: asyncio.Queue[agent_pb2.ClientMessage | None]
  ) -> bool:
    return (
      self._live_requests is not None and requests is not self._live_requests
    )

  def _settle_input(self, response: agent_pb2.SessionMessage) -> None:
    kind = response.WhichOneof('payload')
    if kind == 'input_nack':
      self._pending.pop(response.input_nack.message_id, None)
      return
    if kind != 'input_ack':
      return
    request = self._pending.get(response.input_ack.message_id)
    if request is None:
      return
    payload = request.WhichOneof('payload')
    if payload in _SETTINGS:
      self._settings[payload] = request
    if payload != 'message':
      del self._pending[request.message_id]

  def _resumable(self, error: grpc.aio.AioRpcError) -> bool:
    return self._attachment_id is not None and (
      (
        error.code() == grpc.StatusCode.UNAVAILABLE
        and error.details() == RESTARTING
      )
      or _core_failure(error)
    )

  async def _deliver(self, response: agent_pb2.SessionMessage) -> None:
    for message in self._continuity.continuation(response):
      await self._emit(message)

  async def _emit(self, message: agent_pb2.SessionMessage) -> None:
    self._continuity.observe(message)
    if message.HasField('agent_message'):
      self._pending.pop(message.agent_message.id, None)
    await deliver_message(self._on_message, message)

  async def _follow(self) -> None:
    while self._session_task is not None:
      await self._session_task
      ending = self._attachment_error
      if not isinstance(ending, _Restarted):
        break
      await self._reattach(ending)
    if self._attachment_error is not None:
      raise self._attachment_error

  async def _reattach(self, restart: _Restarted) -> None:
    """Resume a session whose server restarted, or raise why it cannot."""
    self._report_reattaching(True)
    for message in self._continuity.resume():
      await self._emit(message)
    failure = str(restart)
    for delay in self._reattach_delays:
      await asyncio.sleep(delay)
      try:
        await self._reattach_once()
      except _AttemptFailed as error:
        failure = str(error)
        if not error.retryable:
          break
      except grpc.aio.AioRpcError as error:
        failure = _error_message(error)
        if not _retryable(error):
          break
      except (OSError, RuntimeError) as error:
        failure = f'{type(error).__name__}: {error}'
      else:
        self._report_reattaching(False)
        return
    self._report_reattaching(False)
    raise RuntimeError(f'{restart}; reattaching failed: {failure}')

  def _report_reattaching(self, reattaching: bool) -> None:
    if self._on_reattaching is not None:
      self._on_reattaching(reattaching)

  async def _reattach_once(self) -> None:
    await self._transport.close()
    connection = await self._transport.connect(
      self._host_tools_registry.tools()
      if self._host_tools_registry is not None
      else (),
      attachment=self._reattachment(),
    )
    stub = self._adopt(connection)
    if self._attachment_id is None:
      raise _AttemptFailed('reattached without an attachment', retryable=False)
    requests: asyncio.Queue[agent_pb2.ClientMessage | None] = asyncio.Queue()
    ready = asyncio.Event()
    self._live_requests = requests
    self._replay_buffer = []
    self._session_task = self._spawn(
      self._read_session(
        stub.Session(
          _requests(requests, ready, self._attachment),
          metadata=self._metadata,
        ),
        requests,
      )
    )
    try:
      await asyncio.wait_for(self._attached.wait(), timeout=30)
    except TimeoutError as error:
      raise _AttemptFailed('attachment timed out', retryable=True) from error
    self._raise_if_ended()
    for record in await self._get_messages(
      stub, offset=self._continuity.sequence or 0
    ):
      await self._deliver(agent_pb2.SessionMessage(agent_message=record))
    self._replay_buffer = session_continuity.after_history(
      self._replay_buffer or [], self._continuity.sequence or 0
    )
    await self._flush_replay_buffer()
    for message in self._continuity.settle(self._pending.values()):
      await self._emit(message)
    self._raise_if_ended()
    self._take_over(requests)
    ready.set()

  def _reattachment(self) -> agent_pb2.ClientMessage | None:
    selected = self._settings.get('set_inference_config')
    inference = (
      selected.set_inference_config
      if selected is not None
      else self.inference_config
    )
    if inference is None:
      return None
    return agent_pb2.ClientMessage(
      session_id=self._conversation_id,
      message_id=uuid.uuid4().hex,
      set_inference_config=inference,
    )

  def _raise_if_ended(self) -> None:
    if self._attachment_error is not None:
      raise _AttemptFailed(
        str(self._attachment_error), retryable=self._stream_retryable
      )

  def _take_over(
    self, requests: asyncio.Queue[agent_pb2.ClientMessage | None]
  ) -> None:
    previous, self._request_queue = self._request_queue, requests
    previous.put_nowait(None)
    restored = [
      _renewed(setting, self._conversation_id)
      for kind, setting in self._settings.items()
      if kind != 'set_inference_config'
      and setting.message_id not in self._pending
    ]
    for request in [*restored, *self._pending.values()]:
      requests.put_nowait(request)
    for request in restored:
      self._pending[request.message_id] = request

  async def _execute_host_tool(
    self,
    call: agent_pb2.ToolCall,
    requests: asyncio.Queue[agent_pb2.ClientMessage | None],
  ) -> None:
    host_tools_registry = self._host_tools_registry
    if (
      not self._execution_host
      or host_tools_registry is None
      or not call.HasField('function')
    ):
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
