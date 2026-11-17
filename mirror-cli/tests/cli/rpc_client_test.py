"""Tests for client-side AgentRPC stream handling."""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import cast, override
from unittest import mock

import freezegun
import grpc
import pytest
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
)

from mirror.cli import host_tools, rpc_client, safety, showcase


@pytest.mark.asyncio
async def test_call_tool_uses_connected_session_and_preserves_result(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  requests: list[agent_pb2.CallToolRequest] = []

  class Stub:
    async def CallTool(
      self,
      request: agent_pb2.CallToolRequest,
      *,
      metadata: Sequence[tuple[str, str]] = (),
    ) -> agent_pb2.CallToolResponse:  # noqa: N802
      assert metadata == ()
      requests.append(request)
      return agent_pb2.CallToolResponse(
        content='output', success=False, display='failed'
      )

  channel = object()

  def stub_factory(actual: object) -> Stub:
    assert actual is channel
    return Stub()

  monkeypatch.setattr(agent_pb2_grpc, 'AgentRPCStub', stub_factory)
  registry = host_tools.HostToolRegistry()
  registry.register('core/connections', [_OriginTool()], namespace='mcp')
  client = rpc_client.Client(
    cast(rpc_client.Transport, object()),
    'target-session',
    on_message=lambda _: None,
    on_error=mock.AsyncMock(),
    host_tools_registry=registry,
  )
  with pytest.raises(RuntimeError, match='not connected'):
    await client.call_tool('bash', {'command': 'pwd'})
  setattr(client, '_channel', channel)
  result = await client.call_tool('bash', {'command': 'pwd'})
  assert result == host_tools.ToolResult('output', False, 'failed')
  assert len(requests) == 1
  assert requests[0].session_id == 'target-session'
  assert requests[0].name == 'bash'
  assert json.loads(requests[0].arguments) == {'command': 'pwd'}


@pytest.mark.asyncio
async def test_showcase_accepts_agent_message_input() -> None:
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(
    showcase.ShowcaseAgent(delay=0), server
  )
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  finished = asyncio.Event()
  message = agent_pb2.AgentMessage(
    id='user-showcase',
    chat=agent_pb2.ChatRecord(
      user=agent_pb2.UserMessage(content='play demo'), success=True
    ),
  )

  async def requests() -> AsyncIterator[agent_pb2.ClientMessage]:
    yield agent_pb2.ClientMessage(
      session_id='showcase', message_id=message.id, message=message
    )
    await finished.wait()

  try:
    async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
      stream = agent_pb2_grpc.AgentRPCStub(channel).Session(requests())
      ack = await asyncio.wait_for(stream.read(), timeout=2)
      echoed = await asyncio.wait_for(stream.read(), timeout=2)
      assert ack.input_ack.message_id == message.id
      assert echoed.agent_message.id == message.id
      assert echoed.agent_message.chat == message.chat
      stream.cancel()
  finally:
    finished.set()
    await server.stop(None)


class _EchoHostTool:
  definition = host_tools.ToolDefinition('echo', 'Echo arguments', {})

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    return host_tools.ToolResult(f'echo: {arguments}', True)


class _BlockingHostTool(_EchoHostTool):
  def __init__(self) -> None:
    self.started = asyncio.Event()
    self.release = asyncio.Event()

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    self.started.set()
    await self.release.wait()
    return await super().execute(arguments)


class _ReplayStub:
  def __init__(self, records: Sequence[agent_pb2.AgentMessage]) -> None:
    self._records = records

  async def GetMessages(  # noqa: N802, generated RPC name
    self,
    request: agent_pb2.GetMessagesRequest,
    *,
    metadata: Sequence[tuple[str, str]] = (),
  ) -> agent_pb2.GetMessagesResponse:
    assert metadata == ()
    offset = int(request.offset)
    limit = min(request.limit, 1)
    return agent_pb2.GetMessagesResponse(
      messages=self._records[offset : offset + limit],
      total=len(self._records),
    )


class _HostToolStub:
  def __init__(self, release: asyncio.Event) -> None:
    self._release = release

  async def _responses(self) -> AsyncIterator[agent_pb2.SessionMessage]:
    yield agent_pb2.SessionMessage(
      host_tool_call=agent_pb2.ToolCall(
        id='call-1',
        type='function',
        function=agent_pb2.FunctionCall(name='test__echo', arguments='{}'),
      )
    )
    await self._release.wait()


def test_initial_inference_precedes_unchanged_queued_input() -> None:
  async def on_error(_: str) -> None:
    pass

  client = rpc_client.Client(
    cast(rpc_client.Transport, object()),
    'conversation',
    on_message=lambda _: None,
    on_error=cast(Callable[[str], Awaitable[None]], on_error),
    inference_config=agent_pb2.InferenceConfig(
      model='test-model', reasoning_effort='high'
    ),
  )
  execute = agent_pb2.AgentMessage(
    id='extension-1',
    extension=agent_pb2.ExtensionRecord(extension='todo', kind='state'),
  )

  client.enqueue(execute)
  requests = cast(
    asyncio.Queue[agent_pb2.ClientMessage | None],
    getattr(client, '_request_queue'),
  )
  inference_request = cast(
    agent_pb2.ClientMessage, getattr(client, '_attachment')
  )
  assert inference_request is not None
  assert inference_request.session_id == 'conversation'
  assert inference_request.message_id
  assert inference_request.set_inference_config == agent_pb2.InferenceConfig(
    model='test-model', reasoning_effort='high'
  )
  execute_request = requests.get_nowait()

  assert execute_request is not None
  assert execute_request.session_id == 'conversation'
  assert execute_request.message == execute
  assert execute_request.message_id == execute.id


@pytest.mark.asyncio
async def test_replay_delivers_the_complete_trace_from_short_pages() -> None:
  async def on_error(_: str) -> None:
    pass

  received: list[agent_pb2.SessionMessage] = []
  client = rpc_client.Client(
    cast(rpc_client.Transport, object()),
    'conversation',
    on_message=received.append,
    on_error=cast(Callable[[str], Awaitable[None]], on_error),
  )
  records = (
    agent_pb2.AgentMessage(id='first'),
    agent_pb2.AgentMessage(id='last'),
  )
  replay = cast(
    Callable[[object], Awaitable[None]], getattr(client, '_replay_messages')
  )

  await replay(_ReplayStub(records))

  assert [message.agent_message.id for message in received[:-1]] == [
    'first',
    'last',
  ]
  assert received[-1].HasField('replay_done')
  assert received[-1].replay_done.dropped_entries == 0


@pytest.mark.asyncio
async def test_host_tool_result_returns_through_client_stream() -> None:
  async def on_error(_: str) -> None:
    pass

  client = rpc_client.Client(
    cast(rpc_client.Transport, object()),
    'conversation',
    on_message=lambda _: None,
    on_error=cast(Callable[[str], Awaitable[None]], on_error),
  )
  registry = host_tools.HostToolRegistry()
  registry.register('test', [_EchoHostTool()])
  setattr(client, '_host_tools_registry', registry)
  setattr(client, '_execution_host', True)
  requests: asyncio.Queue[agent_pb2.ClientMessage | None] = asyncio.Queue()

  execute_host_tool = cast(
    Callable[
      [
        agent_pb2.ToolCall,
        asyncio.Queue[agent_pb2.ClientMessage | None],
      ],
      Awaitable[None],
    ],
    getattr(client, '_execute_host_tool'),
  )
  await execute_host_tool(
    agent_pb2.ToolCall(
      id='call-1',
      type='function',
      function=agent_pb2.FunctionCall(
        name='test__echo', arguments='{"text":"hi"}'
      ),
    ),
    requests,
  )

  response = await requests.get()
  assert response is not None
  assert response.message_id == 'host-tool-result:call-1'
  assert response.host_tool_result.call_id == 'call-1'
  assert response.host_tool_result.content == 'echo: {"text":"hi"}'
  assert response.host_tool_result.success


@pytest.mark.asyncio
async def test_completed_host_tool_task_is_pruned() -> None:
  async def on_error(_: str) -> None:
    pass

  client = rpc_client.Client(
    cast(rpc_client.Transport, object()),
    'conversation',
    on_message=lambda _: None,
    on_error=cast(Callable[[str], Awaitable[None]], on_error),
  )
  tools = _BlockingHostTool()
  registry = host_tools.HostToolRegistry()
  registry.register('test', [tools])
  setattr(client, '_host_tools_registry', registry)
  setattr(client, '_execution_host', True)
  requests: asyncio.Queue[agent_pb2.ClientMessage | None] = asyncio.Queue()
  stream_release = asyncio.Event()
  read_session = cast(
    Callable[
      [
        AsyncIterator[agent_pb2.SessionMessage],
        asyncio.Queue[agent_pb2.ClientMessage | None],
      ],
      Awaitable[None],
    ],
    getattr(client, '_read_session'),
  )
  reader = asyncio.create_task(
    read_session(_HostToolStub(stream_release)._responses(), requests)
  )

  await tools.started.wait()
  tasks = cast(set[asyncio.Task[None]], getattr(client, '_tasks'))
  assert len(tasks) == 1
  tools.release.set()
  await requests.get()
  await asyncio.sleep(0)

  assert not tasks
  setattr(client, '_closing', True)
  stream_release.set()
  await reader


def test_worker_configuration_renders_the_interactive_prompt_verbatim() -> None:
  configuration = rpc_client.WorkerConfiguration(
    {}, cwd=pathlib.Path('/selected/workspace')
  )
  values = configuration.values
  assert (
    values['system_prompt']
    == """You are Mirror, an expert coding assistant operating inside a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.
<tools>
- read: Read file contents
- bash: Execute bash commands (ls, rg, etc.)
- glob: Find files by name or glob pattern
- edit: Make precise file edits with exact text replacement
- write: Create or overwrite files
In addition to the tools above, you may have access to other custom tools depending on the project.
</tools>
<rules>
- Use glob to find files by name or pattern. Use bash for other file operations like ls and rg.
- Use read to examine files instead of cat or sed.
- Use edit for precise changes (old_string must match exactly)
- Keep old_string as small as possible while still being unique in the file. Do not pad with large unchanged regions.
- Use write only for new files or complete rewrites.
- Be concise in your responses
- Show file paths clearly when working with files
</rules>
<cwd>
/selected/workspace
</cwd>"""
  )
  assert values['prompt_injection_policy'] == safety.PROMPT_INJECTION_POLICY
  assert (
    configuration.after_start().values['system_prompt']
    == values['system_prompt']
  )


def test_worker_configuration_composes_custom_instructions_once() -> None:
  configuration = rpc_client.WorkerConfiguration(
    {'system_prompt': 'Agent identity.'}, cwd=pathlib.Path('/workspace')
  )
  prompt = str(configuration.values['system_prompt'])
  assert prompt.endswith('<cwd>\n/workspace\n</cwd>\n\nAgent identity.')
  assert configuration.after_start().values['system_prompt'] == prompt
  child = configuration.updated({'system_prompt': 'Child identity.'})
  child_prompt = str(child.values['system_prompt'])
  assert child_prompt.endswith('<cwd>\n/workspace\n</cwd>\n\nChild identity.')
  assert child_prompt.count('You are Mirror,') == 1
  assert 'Agent identity.' not in child_prompt
  cleared = configuration.updated({'system_prompt': ''})
  assert str(cleared.values['system_prompt']).endswith(
    '<cwd>\n/workspace\n</cwd>'
  )


def test_worker_configuration_keeps_policy_separate_from_identity() -> None:
  configuration = rpc_client.WorkerConfiguration(
    {'system_prompt': 'Agent identity.'}, cwd=pathlib.Path('/workspace')
  )
  assert (
    configuration.values['prompt_injection_policy']
    == safety.PROMPT_INJECTION_POLICY
  )
  assert str(configuration.values['system_prompt']).endswith(
    '\n\nAgent identity.'
  )
  child = configuration.updated({'system_prompt': 'Child identity.'})
  assert (
    child.values['prompt_injection_policy'] == safety.PROMPT_INJECTION_POLICY
  )
  assert str(child.values['system_prompt']).endswith('\n\nChild identity.')


def test_worker_configuration_asks_the_worker_to_describe_its_workspace() -> (
  None
):
  configuration = rpc_client.WorkerConfiguration(
    {}, cwd=pathlib.Path('/workspace')
  )
  assert configuration.values['include_environment'] is True
  assert configuration.after_start().values['include_environment'] is True
  opted_out = configuration.updated({'include_environment': False})
  assert opted_out.values['include_environment'] is False


class _UnusedTransport:
  async def connect(
    self,
    host_tools: Sequence[host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> rpc_client.Connection:
    pytest.fail('Client-owned tools should not need a worker connection')

  async def close(self) -> None:
    pass


class _OriginTool:
  definition = host_tools.ToolDefinition(
    'origin', 'Return execution origin', {}
  )

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    return host_tools.ToolResult(context.origin.value, True)


@pytest.mark.asyncio
async def test_service_call_context_is_not_taken_from_tool_arguments() -> None:
  registry = host_tools.HostToolRegistry()
  registry.register('core/connections', [_OriginTool()], namespace='mcp')
  client = rpc_client.Client(
    _UnusedTransport(),
    'session',
    on_message=lambda _: None,
    on_error=lambda _: asyncio.sleep(0),
    host_tools_registry=registry,
  )
  try:
    result = await client.call_tool('mcp__origin', {'origin': 'model'})
    assert result == host_tools.ToolResult('extension', True)
    model = await registry.execute('mcp__origin', '{"origin":"extension"}')
    assert model == host_tools.ToolResult('model', True)
  finally:
    await registry.close()


class _ChannelTransport:
  def __init__(
    self,
    channel: grpc.aio.Channel,
    daemon_attachment_id: str | None = None,
    reconnect: Callable[[bool], Awaitable[rpc_client.Connection]] | None = None,
  ) -> None:
    self.channel = channel
    self.daemon_attachment_id = daemon_attachment_id
    self.reconnect = reconnect

  async def connect(
    self,
    host_tools: Sequence[host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> rpc_client.Connection:
    return rpc_client.Connection(
      self.channel,
      attachment,
      daemon_attachment_id=self.daemon_attachment_id,
      reconnect=self.reconnect,
    )

  async def close(self) -> None:
    await self.channel.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('attachment_id', 'code', 'details', 'expected'),
  [
    ('fixture', grpc.StatusCode.CANCELLED, 'workspace closed', 'closed'),
    ('fixture', grpc.StatusCode.UNAVAILABLE, 'workspace closed', 'UNAVAILABLE'),
    (
      'fixture',
      grpc.StatusCode.CANCELLED,
      'unexpected failure',
      'unexpected failure',
    ),
    (None, grpc.StatusCode.CANCELLED, 'workspace closed', 'CANCELLED'),
  ],
)
async def test_session_end_only_recognizes_intentional_daemon_close(
  attachment_id: str | None,
  code: grpc.StatusCode,
  details: str,
  expected: str,
) -> None:
  result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

  async def closed() -> None:
    result.set_result('closed')

  async def failed(message: str) -> None:
    result.set_result(message)

  class Agent(agent_pb2_grpc.AgentRPCServicer):
    @override
    async def Session(  # noqa: N802, generated RPC name
      self,
      request_iterator: AsyncIterator[agent_pb2.ClientMessage],
      context: grpc.aio.ServicerContext[
        agent_pb2.ClientMessage, agent_pb2.SessionMessage
      ],
    ) -> AsyncIterator[agent_pb2.SessionMessage]:
      del request_iterator
      yield agent_pb2.SessionMessage(replay_done=agent_pb2.ReplayDone())
      await context.abort(code, details)

  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(Agent(), server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
  client = rpc_client.Client(
    _ChannelTransport(channel, attachment_id),
    'session',
    on_message=lambda _: None,
    on_closed=closed,
    on_error=failed,
  )
  try:
    async with client:
      assert expected in await asyncio.wait_for(result, 5)
    assert channel.get_state() == grpc.ChannelConnectivity.SHUTDOWN
  finally:
    await client.close()
    await server.stop(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('code', 'details', 'outcome'),
  [
    (
      grpc.StatusCode.FAILED_PRECONDITION,
      'attachment worker is no longer running',
      contextlib.nullcontext(),
    ),
    (
      grpc.StatusCode.CANCELLED,
      'workspace closed',
      pytest.raises(rpc_client.WorkspaceClosed),
    ),
    (
      grpc.StatusCode.FAILED_PRECONDITION,
      'workspace conversation binding changed; refresh before attaching',
      pytest.raises(grpc.aio.AioRpcError),
    ),
  ],
)
async def test_reconnect_only_allows_reopening_a_missing_worker(
  code: grpc.StatusCode,
  details: str,
  outcome: contextlib.AbstractContextManager[object],
) -> None:
  disconnected: asyncio.Queue[str] = asyncio.Queue()
  attempts: list[bool] = []

  async def reconnect(manual: bool) -> rpc_client.Connection:
    attempts.append(manual)
    if not manual:
      raise RuntimeError('automatic recovery failed')
    raise grpc.aio.AioRpcError(code, details=details)

  class Agent(agent_pb2_grpc.AgentRPCServicer):
    @override
    async def Session(  # noqa: N802, generated RPC name
      self,
      request_iterator: AsyncIterator[agent_pb2.ClientMessage],
      context: grpc.aio.ServicerContext[
        agent_pb2.ClientMessage, agent_pb2.SessionMessage
      ],
    ) -> AsyncIterator[agent_pb2.SessionMessage]:
      del request_iterator
      yield agent_pb2.SessionMessage(replay_done=agent_pb2.ReplayDone())
      await context.abort(grpc.StatusCode.UNAVAILABLE, 'disconnected')

  loop = asyncio.get_running_loop()
  with (
    tempfile.TemporaryDirectory(prefix='mr-') as directory,
    freezegun.freeze_time('2000-01-01', ignore=['grpc']) as clock,
  ):
    address = f'unix:{directory}/agent.sock'
    server = grpc.aio.server()
    agent_pb2_grpc.add_AgentRPCServicer_to_server(Agent(), server)
    server.add_insecure_port(address)
    await server.start()

    def retrying(reason: str | None) -> None:
      del reason
      loop.call_soon(clock.tick, 1)

    client = rpc_client.Client(
      _ChannelTransport(
        grpc.aio.insecure_channel(address), 'fixture', reconnect
      ),
      'session',
      on_message=lambda _: None,
      on_error=disconnected.put,
      on_reconnect=retrying,
    )
    try:
      await client.start(replay=False)
      assert await disconnected.get() == 'automatic recovery failed'
      assert attempts == [False]
      with outcome:
        assert await client.reconnect() is False
      assert attempts == [False, True]
      assert disconnected.empty()
    finally:
      await client.close()
      await server.stop(None)


@pytest.mark.asyncio
@pytest.mark.parametrize('reattach', [False, True])
async def test_daemon_resync_reads_missing_history_before_restoring_live_state(
  reattach: bool,
) -> None:
  resync, approval, release, restored = (asyncio.Event() for _ in range(4))
  received: list[str] = []
  pages: list[int] = []
  errors: list[str] = []
  rejected: asyncio.Future[str] = asyncio.get_running_loop().create_future()

  class Agent(agent_pb2_grpc.AgentRPCServicer):
    connections = 0

    @override
    async def GetMessages(
      self,
      request: agent_pb2.GetMessagesRequest,
      context: grpc.aio.ServicerContext[
        agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
      ],
    ) -> agent_pb2.GetMessagesResponse:
      del context
      pages.append(request.offset)
      return agent_pb2.GetMessagesResponse(
        messages=[
          agent_pb2.AgentMessage(id=str(index))
          for index in range(
            request.offset, request.offset + min(64, request.limit)
          )
        ],
        total=130,
      )

    @override
    async def Session(
      self,
      request_iterator: AsyncIterator[agent_pb2.ClientMessage],
      context: grpc.aio.ServicerContext[
        agent_pb2.ClientMessage, agent_pb2.SessionMessage
      ],
    ) -> AsyncIterator[agent_pb2.SessionMessage]:
      self.connections += 1
      recovery = '1' if self.connections == 1 else 'history'
      assert ('mirror-resynchronize', recovery) in context.invocation_metadata()
      request = await anext(request_iterator)
      if self.connections == 1:
        yield agent_pb2.SessionMessage(
          replay_done=agent_pb2.ReplayDone(conversation_id='session')
        )
        yield agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id=request.message_id)
        )
        await resync.wait()
        if reattach:
          await context.abort(
            grpc.StatusCode.UNAVAILABLE, 'synthetic disconnect'
          )
      yield agent_pb2.SessionMessage(
        replay_start=agent_pb2.ReplayStart(history_prefix=129)
      )
      yield agent_pb2.SessionMessage(
        agent_message=agent_pb2.AgentMessage(id='129')
      )
      yield agent_pb2.SessionMessage(
        replay_done=agent_pb2.ReplayDone(
          conversation_id='session', dropped_entries=129
        )
      )
      yield agent_pb2.SessionMessage(
        hitl_request=agent_pb2.HITLRequest(
          call=agent_pb2.ToolCall(id='pending-approval')
        )
      )
      yield agent_pb2.SessionMessage(state_restored={})
      if reattach:
        yield agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id=request.message_id)
        )
      request = await anext(request_iterator)
      yield agent_pb2.SessionMessage(
        input_nack=agent_pb2.InputNack(
          message_id=request.message_id, error='declined'
        )
      )
      await asyncio.Event().wait()

  async def receive(message: agent_pb2.SessionMessage) -> None:
    if message.HasField('input_ack'):
      return
    received.append(
      message.agent_message.id or message.WhichOneof('payload') or ''
    )
    if message.HasField('hitl_request'):
      approval.set()
      await release.wait()
    if message.HasField('state_restored'):
      restored.set()
    if message.HasField('input_nack'):
      rejected.set_result(message.input_nack.message_id)

  async def error(message: str) -> None:
    errors.append(message)

  loop = asyncio.get_running_loop()
  with (
    tempfile.TemporaryDirectory(prefix='mr-') as directory,
    freezegun.freeze_time('2000-01-01', ignore=['grpc']) as clock,
  ):
    address = f'unix:{directory}/agent.sock'
    server = grpc.aio.server()
    peer = Agent()
    agent_pb2_grpc.add_AgentRPCServicer_to_server(peer, server)
    server.add_insecure_port(address)
    await server.start()

    async def reconnect(manual: bool) -> rpc_client.Connection:
      assert not manual
      transport.channel = grpc.aio.insecure_channel(address)
      return rpc_client.Connection(
        transport.channel,
        agent_pb2.ClientMessage(session_id='session', message_id='reattached'),
        daemon_attachment_id='reattached',
        reconnect=reconnect,
      )

    def retrying(reason: str | None) -> None:
      del reason
      loop.call_soon(clock.tick, 1)

    transport = _ChannelTransport(
      grpc.aio.insecure_channel(address), 'attachment', reconnect
    )
    client = rpc_client.Client(
      transport,
      'session',
      on_message=receive,
      on_error=error,
      on_reconnect=retrying,
      resynchronize=True,
      inference_config=agent_pb2.InferenceConfig(model='synthetic'),
    )
    try:
      await client.start(replay=True, wait_for_attachment=True)
      received.clear()
      resync.set()
      await approval.wait()
      assert not restored.is_set(), (
        'Approval rendering is part of synchronization'
      )
      release.set()
      await restored.wait()
      assert pages == [0, 64, 128]
      assert received == [
        'replay_start',
        *map(str, range(130)),
        'replay_done',
        'hitl_request',
        'state_restored',
      ]
      client.enqueue(agent_pb2.AgentMessage(id='new-input'))
      assert await rejected == 'new-input'
      assert peer.connections == 1 + reattach
      assert client.execution_host
      assert not errors
    finally:
      await client.close()
      await server.stop(None)


@pytest.mark.asyncio
async def test_host_callback_cannot_claim_extension_origin() -> None:
  response: asyncio.Future[agent_pb2.ClientMessage] = (
    asyncio.get_running_loop().create_future()
  )

  class Agent(agent_pb2_grpc.AgentRPCServicer):
    @override
    async def Session(  # noqa: N802, generated RPC name
      self,
      request_iterator: AsyncIterator[agent_pb2.ClientMessage],
      context: grpc.aio.ServicerContext[
        agent_pb2.ClientMessage, agent_pb2.SessionMessage
      ],
    ) -> AsyncIterator[agent_pb2.SessionMessage]:
      yield agent_pb2.SessionMessage(
        host_tool_call=agent_pb2.ToolCall(
          id='call',
          function=agent_pb2.FunctionCall(
            name='mcp__origin', arguments='{"origin":"extension"}'
          ),
        )
      )
      response.set_result(await anext(request_iterator))
      await asyncio.Event().wait()

  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(Agent(), server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  registry = host_tools.HostToolRegistry()
  registry.register('core/connections', [_OriginTool()], namespace='mcp')
  client = rpc_client.Client(
    _ChannelTransport(grpc.aio.insecure_channel(f'127.0.0.1:{port}')),
    'session',
    on_message=lambda _: None,
    on_error=lambda _: asyncio.sleep(0),
    host_tools_registry=registry,
  )
  try:
    async with client:
      result = await asyncio.wait_for(response, 2)
      assert result.host_tool_result.call_id == 'call'
      assert result.host_tool_result.content == 'model'
      assert result.host_tool_result.success
  finally:
    await registry.close()
    await server.stop(None)


@pytest.mark.parametrize('api', ['openai-responses', 'openai-codex-responses'])
@pytest.mark.parametrize('remote', [False, True])
def test_fast_speed_requests_priority_without_changing_reasoning(
  api: str, remote: bool
) -> None:
  configuration = rpc_client.WorkerConfiguration(
    {'model': 'model', 'model_api': api, 'remote_model_api': api},
    cwd=pathlib.Path('/workspace'),
  )

  inference = configuration.inference_config(
    'model', 'high', remote=remote, speed='fast'
  )

  assert json.loads(inference.extra_body) == {'service_tier': 'priority'}
  assert inference.reasoning_effort == 'high'
  assert rpc_client.inference_speed(inference) == 'fast'


@pytest.mark.parametrize('remote', [False, True])
def test_anthropic_fast_speed_uses_native_speed_without_changing_reasoning(
  remote: bool,
) -> None:
  configuration = rpc_client.WorkerConfiguration(
    {
      'model': 'claude-opus-4-8',
      'model_api': 'anthropic-messages',
      'remote_model_api': 'anthropic-messages',
    },
    cwd=pathlib.Path('/workspace'),
  )

  inference = configuration.inference_config(
    'claude-opus-4-8', 'high', remote=remote, speed='fast'
  )

  assert json.loads(inference.extra_body) == {'speed': 'fast'}
  assert inference.reasoning_effort == 'high'
  assert rpc_client.inference_speed(inference) == 'fast'


@pytest.mark.parametrize('speed', ['normal', 'fast'])
def test_speed_never_adds_priority_to_unsupported_protocols(speed: str) -> None:
  configuration = rpc_client.WorkerConfiguration(
    {'model': 'model', 'model_api': 'reflection'},
    cwd=pathlib.Path('/workspace'),
  )

  inference = configuration.inference_config(
    'model', 'high', remote=False, speed=speed
  )

  assert not inference.HasField('extra_body')


@pytest.mark.parametrize(
  'api', ['openai-responses', 'openai-codex-responses', 'anthropic-messages']
)
def test_normal_speed_clears_fast_processing(api: str) -> None:
  configuration = rpc_client.WorkerConfiguration(
    {'model': 'model', 'model_api': api},
    cwd=pathlib.Path('/workspace'),
  )

  fast = configuration.inference_config(
    'model', 'high', remote=False, speed='fast'
  )
  normal = configuration.inference_config(
    'model', 'high', remote=False, speed='normal'
  )

  assert fast.HasField('extra_body')
  assert not normal.HasField('extra_body')
  assert rpc_client.inference_speed(normal) == 'normal'
  assert not configuration.inference_config(
    'model', 'high', remote=False
  ).HasField('extra_body')


def test_speed_respects_codex_custom_endpoint_protocol_override() -> None:
  configuration = rpc_client.WorkerConfiguration(
    {
      'model': 'model',
      'model_api': 'openai-codex-responses',
      'model_base_url': 'http://localhost:8000/v1',
    },
    cwd=pathlib.Path('/workspace'),
  )

  inference = configuration.inference_config(
    'model', 'high', remote=False, speed='fast'
  )

  assert inference.api == 'openai-chat-completions'
  assert not inference.HasField('extra_body')


def test_remote_inference_sends_only_the_server_protocol() -> None:
  configuration = rpc_client.WorkerConfiguration(
    {
      'model': 'claude-opus-4-8',
      'model_api': 'openai-chat-completions',
      'model_base_url': 'http://localhost:8000/v1',
      'remote_model_api': 'anthropic-messages',
    },
    cwd=pathlib.Path('/workspace'),
  )

  assert configuration.inference_config(
    'server default', 'medium', remote=True
  ) == agent_pb2.InferenceConfig(
    model='claude-opus-4-8', api='anthropic-messages', reasoning_effort='medium'
  )


@pytest.mark.asyncio
async def test_deliver_message_yields_after_synchronous_callback() -> None:
  order: list[str] = []
  asyncio.get_running_loop().call_soon(order.append, 'input')
  await rpc_client.deliver_message(
    lambda _: order.append('message'), agent_pb2.SessionMessage()
  )
  assert order == ['message', 'input']


@pytest.mark.asyncio
async def test_deliver_message_waits_for_async_callback_and_propagates_failure() -> (
  None
):
  async def receive(message: agent_pb2.SessionMessage) -> None:
    del message
    await asyncio.sleep(0)
    raise ValueError('failed callback')

  with pytest.raises(ValueError, match='failed callback'):
    await rpc_client.deliver_message(receive, agent_pb2.SessionMessage())


@pytest.mark.asyncio
async def test_deliver_message_cancellation_reaches_async_callback() -> None:
  started = asyncio.Event()
  stopped = asyncio.Event()

  async def receive(message: agent_pb2.SessionMessage) -> None:
    del message
    started.set()
    try:
      await asyncio.Future[None]()
    finally:
      stopped.set()

  delivering = asyncio.create_task(
    rpc_client.deliver_message(receive, agent_pb2.SessionMessage())
  )
  await started.wait()
  delivering.cancel()
  with pytest.raises(asyncio.CancelledError):
    await delivering
  assert stopped.is_set()


@pytest.mark.parametrize(
  'extra_body', ['', '{', 'null', '[]', '{"speed": "fast"}']
)
def test_shared_openai_speed_requires_priority(extra_body: str) -> None:
  assert (
    rpc_client.inference_speed(
      agent_pb2.InferenceConfig(api='openai-responses', extra_body=extra_body)
    )
    == 'normal'
  )
