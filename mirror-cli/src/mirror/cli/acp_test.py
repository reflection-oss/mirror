"""Tests for the ACP stdio server over a fake agent worker."""

from __future__ import annotations

import asyncio
import json
import pathlib
import socket
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import cast, override

import grpc
import pytest
import pytest_asyncio
from google.protobuf import timestamp_pb2
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
  daemon_pb2,
  mirror_pb2,
)

from mirror.cli import acp, host_tools, inference_choices, models, rpc_client

# What the protocols with credentials and `models.json` offer.
_OFFERED = (
  models.ModelOption(
    'gpt-6-astra', 'GPT-6-Astra', 'gpt-6-astra', api='openai-codex-responses'
  ),
  models.ModelOption(
    'claude-opus-4-8', 'Claude Opus 4.8', api='anthropic-messages'
  ),
  models.ModelOption(models.GEMINI_MODEL, 'Gemini 4 Argon', api='gemini'),
  # A `models.json` entry that names no protocol.
  models.ModelOption('beam-mini', 'Beam Mini'),
)


def _delta(**fields: object) -> agent_pb2.SessionMessage:
  """A runtime delta carrying one streamed field."""
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(**fields)  # pyright: ignore[reportArgumentType]
  )


class _FakeAgent(agent_pb2_grpc.AgentRPCServicer):
  """Scripted worker: a thinking delta, an approved tool call, streamed text."""

  def __init__(self) -> None:
    self.approvals: list[agent_pb2.HITLResponse] = []
    self.interrupts = 0
    self.user_messages: list[str] = []
    self.configs: list[agent_pb2.InferenceConfig] = []
    self.approval_in_flight = False
    self.attached: set[str] = set()
    self.history: list[agent_pb2.AgentMessage] = []

  @override
  async def GetMessages(  # noqa: N802
    self,
    request: agent_pb2.GetMessagesRequest,
    context: grpc.aio.ServicerContext[
      agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
    ],
  ) -> agent_pb2.GetMessagesResponse:
    del context
    return agent_pb2.GetMessagesResponse(
      messages=self.history[request.offset : request.offset + request.limit],
      total=len(self.history),
    )

  @override
  async def Session(  # noqa: N802
    self,
    request_iterator: AsyncIterator[agent_pb2.ClientMessage],
    context: grpc.aio.ServicerContext[
      agent_pb2.ClientMessage, agent_pb2.SessionMessage
    ],
  ) -> AsyncIterator[agent_pb2.SessionMessage]:
    attaching = True
    async for request in request_iterator:
      if attaching:
        # Like the daemon, an attachment first replays the conversation.
        attaching = False
        if request.session_id in self.attached:
          yield agent_pb2.SessionMessage(
            agent_message=agent_pb2.AgentMessage(
              id='replayed-1',
              chat=agent_pb2.ChatRecord(
                assistant=agent_pb2.AssistantMessage(content='Earlier answer'),
                success=True,
              ),
              inference=agent_pb2.InferenceConfig(model='reflection/beam'),
            )
          )
        self.attached.add(request.session_id)
        yield agent_pb2.SessionMessage(
          replay_done=agent_pb2.ReplayDone(conversation_id=request.session_id)
        )
      payload = request.WhichOneof('payload')
      if payload == 'set_inference_config':
        self.configs.append(request.set_inference_config)
        yield agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id=request.message_id)
        )
        continue
      if payload == 'interrupt':
        self.interrupts += 1
        if self.approval_in_flight:
          # The approval request crossed the interrupt on the wire.
          self.approval_in_flight = False
          yield agent_pb2.SessionMessage(
            hitl_request=agent_pb2.HITLRequest(
              call=agent_pb2.ToolCall(
                id='call-3',
                type='function',
                function=agent_pb2.FunctionCall(name='bash', arguments='{}'),
              )
            )
          )
        # Like the worker, an interrupted turn still ends with a completion.
        yield _delta(turn_interrupted=agent_pb2.RuntimeDelta().turn_interrupted)
        yield _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed)
        continue
      if payload == 'hitl_response':
        self.approvals.append(request.hitl_response)
        yield agent_pb2.SessionMessage(
          hitl_resolved=agent_pb2.HITLResolved(
            call_id=request.hitl_response.call_id,
            approved=request.hitl_response.approved,
          )
        )
        if not request.hitl_response.approved:
          yield _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed)
          continue
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='tool-1',
            chat=agent_pb2.ChatRecord(
              tool=agent_pb2.ToolMessage(
                tool_call_id='call-1', content='README.md\nmain.go', name='bash'
              ),
              success=True,
            ),
          )
        )
        for word in ('Hello ', 'world'):
          yield _delta(stream_content=agent_pb2.StreamContent(content=word))
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='assistant-2',
            chat=agent_pb2.ChatRecord(
              assistant=agent_pb2.AssistantMessage(content='Hello world'),
              usage=agent_pb2.Usage(prompt_tokens=30, completion_tokens=4),
              success=True,
            ),
            inference=agent_pb2.InferenceConfig(model='reflection/beam'),
          ),
          stream_final=True,
        )
        yield _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed)
        continue
      if payload != 'message':
        continue
      content = request.message.chat.user.content
      self.user_messages.append(content)
      yield agent_pb2.SessionMessage(
        input_ack=agent_pb2.InputAck(message_id=request.message_id)
      )
      yield agent_pb2.SessionMessage(agent_message=request.message)
      if content.startswith('notice'):
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='notice-1',
            chat=agent_pb2.ChatRecord(
              assistant=agent_pb2.AssistantMessage(
                content='Model request failed; check credentials.'
              ),
              success=True,
            ),
          )
        )
        yield _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed)
        continue
      if content.startswith('unstreamed'):
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='assistant-3',
            chat=agent_pb2.ChatRecord(
              assistant=agent_pb2.AssistantMessage(content='Whole answer'),
              success=True,
            ),
            inference=agent_pb2.InferenceConfig(model='reflection/beam'),
          )
        )
        yield _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed)
        continue
      if content.startswith('hang'):
        # Stay mid-turn until interrupted.
        self.approval_in_flight = content.startswith('hang-approval')
        yield _delta(stream_content=agent_pb2.StreamContent(content='working'))
        continue
      if content.startswith('drop-during-approval'):
        yield agent_pb2.SessionMessage(
          hitl_request=agent_pb2.HITLRequest(
            call=agent_pb2.ToolCall(
              id='call-4',
              type='function',
              function=agent_pb2.FunctionCall(name='bash', arguments='{}'),
            )
          )
        )
        # The worker dies while the client is deciding.
        return
      if content.startswith('toolfail'):
        # A successful tool round, then the next model call fails and the
        # worker commits a notice instead of an answer.
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='assistant-4',
            chat=agent_pb2.ChatRecord(
              assistant=agent_pb2.AssistantMessage(
                tool_calls=[
                  agent_pb2.ToolCall(
                    id='call-2',
                    type='function',
                    function=agent_pb2.FunctionCall(
                      name='bash', arguments='{"command": "pwd"}'
                    ),
                  )
                ]
              ),
              success=True,
            ),
            inference=agent_pb2.InferenceConfig(model='reflection/beam'),
          )
        )
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='tool-2',
            chat=agent_pb2.ChatRecord(
              tool=agent_pb2.ToolMessage(
                tool_call_id='call-2', content='/repo', name='bash'
              ),
              success=True,
            ),
          )
        )
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id='notice-2',
            chat=agent_pb2.ChatRecord(
              assistant=agent_pb2.AssistantMessage(
                content='Provider returned 429; giving up.'
              ),
              success=True,
            ),
          )
        )
        yield _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed)
        continue
      yield _delta(stream_reasoning=agent_pb2.StreamReasoning(reasoning='Hmm.'))
      call = agent_pb2.ToolCall(
        id='call-1',
        type='function',
        function=agent_pb2.FunctionCall(
          name='bash', arguments='{"command": "ls"}'
        ),
      )
      yield agent_pb2.SessionMessage(
        agent_message=agent_pb2.AgentMessage(
          id='assistant-1',
          chat=agent_pb2.ChatRecord(
            assistant=agent_pb2.AssistantMessage(tool_calls=[call]),
            usage=agent_pb2.Usage(prompt_tokens=20, completion_tokens=6),
            success=True,
          ),
          inference=agent_pb2.InferenceConfig(model='reflection/beam'),
        )
      )
      yield agent_pb2.SessionMessage(
        hitl_request=agent_pb2.HITLRequest(call=call)
      )


class _ChannelTransport:
  """A transport over an already running fake worker.

  Given a session ID it behaves like the daemon's transport for a hosted
  session: the launch settings are an offer to whoever starts the session,
  kept in `offered`, and the stream opens with a handshake that sets nothing.
  """

  def __init__(
    self,
    port: int,
    ready: asyncio.Event | None = None,
    settings: daemon_pb2.Workspace | None = None,
    refusal: Exception | None = None,
    session_id: str | None = None,
  ) -> None:
    self._port = port
    self._ready = ready
    self._settings = settings
    self._refusal = refusal
    self._session_id = session_id
    self.offered: agent_pb2.InferenceConfig | None = None
    self._channel: grpc.aio.Channel | None = None
    self.closed = False

  async def connect(
    self,
    host_tools: Sequence[host_tools.ToolDefinition] = (),
    *,
    attachment: agent_pb2.ClientMessage | None = None,
  ) -> rpc_client.Connection:
    del host_tools
    if self._ready is not None:
      # A slow worker start.
      await self._ready.wait()
    if self._refusal is not None:
      raise self._refusal
    self._channel = grpc.aio.insecure_channel(f'127.0.0.1:{self._port}')
    if self._session_id is not None:
      if attachment is not None:
        self.offered = attachment.set_inference_config
      attachment = agent_pb2.ClientMessage(
        session_id=self._session_id, message_id='attachment'
      )
    return rpc_client.Connection(
      self._channel, attachment, initial_settings=self._settings
    )

  async def close(self) -> None:
    self.closed = True
    if self._channel is not None:
      await self._channel.close()
      self._channel = None


class _Lines:
  """Collect protocol output as decoded messages."""

  def __init__(self) -> None:
    self.messages: asyncio.Queue[dict[str, object]] = asyncio.Queue()

  def write(self, data: bytes) -> None:
    for line in data.decode('utf-8').splitlines():
      if line.strip():
        self.messages.put_nowait(json.loads(line))

  def flush(self) -> None:
    return None


class _Client:
  """Drive the server the way an ACP client does."""

  def __init__(
    self, reader: asyncio.StreamReader, output: _Lines, server: acp.Server
  ) -> None:
    self._reader = reader
    self._output = output
    self._next_id = 0
    self.server = server
    self.transports: list[_ChannelTransport] = []
    self.model_fetches: list[Sequence[models.ModelOption]] = []

  def inject_worker_events(
    self, session_id: str, events: Sequence[agent_pb2.SessionMessage | None]
  ) -> None:
    """Queue worker events as if they had arrived, `None` being a disconnect."""
    session = self.server._sessions[session_id]  # pyright: ignore[reportPrivateUsage]
    for event in events:
      session.events.put_nowait(event)
      if event is None:
        session.failed = True

  def request(self, method: str, params: dict[str, object]) -> int:
    self._next_id += 1
    self._write({
      'jsonrpc': '2.0',
      'id': self._next_id,
      'method': method,
      'params': params,
    })
    return self._next_id

  def notify(self, method: str, params: dict[str, object]) -> None:
    self._write({'jsonrpc': '2.0', 'method': method, 'params': params})

  def respond(self, request_id: object, result: dict[str, object]) -> None:
    self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

  def close(self) -> None:
    self._reader.feed_eof()

  def request_then_cancel(self, session_id: str, text: str) -> int:
    """Deliver a prompt and its cancel in one read, as a fast client can."""
    self._next_id += 1
    lines = [
      {
        'jsonrpc': '2.0',
        'id': self._next_id,
        'method': 'session/prompt',
        'params': {
          'sessionId': session_id,
          'prompt': [{'type': 'text', 'text': text}],
        },
      },
      {
        'jsonrpc': '2.0',
        'method': 'session/cancel',
        'params': {'sessionId': session_id},
      },
    ]
    self._reader.feed_data(
      b''.join(json.dumps(line).encode('utf-8') + b'\n' for line in lines)
    )
    return self._next_id

  def cancel_then_respond(self, session_id: str, request_id: object) -> None:
    """Deliver a cancel and the permission answer in one read."""
    lines = [
      {
        'jsonrpc': '2.0',
        'method': 'session/cancel',
        'params': {'sessionId': session_id},
      },
      {
        'jsonrpc': '2.0',
        'id': request_id,
        'result': {'outcome': {'outcome': 'cancelled'}},
      },
    ]
    self._reader.feed_data(
      b''.join(json.dumps(line).encode('utf-8') + b'\n' for line in lines)
    )

  def _write(self, message: dict[str, object]) -> None:
    self._reader.feed_data(json.dumps(message).encode('utf-8') + b'\n')

  async def next(self) -> dict[str, object]:
    return await asyncio.wait_for(self._output.messages.get(), timeout=5)

  async def wait_response(
    self, request_id: int
  ) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return the response and the server messages that preceded it."""
    preceding: list[dict[str, object]] = []
    while True:
      message = await self.next()
      if message.get('id') == request_id and 'method' not in message:
        return message, preceding
      preceding.append(message)


@pytest_asyncio.fixture
async def fake_agent() -> AsyncIterator[tuple[_FakeAgent, int]]:
  agent = _FakeAgent()
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(  # pyright: ignore[reportUnknownMemberType]
    agent, server
  )
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  try:
    yield agent, port
  finally:
    await server.stop(None)


async def _start(
  port: int,
  tmp_path: pathlib.Path,
  second_worker_ready: asyncio.Event | None = None,
  *,
  offered: Sequence[models.ModelOption] = _OFFERED,
  failed_restarts: int = 0,
) -> tuple[_Client, asyncio.Task[None], list[Mapping[str, object]]]:
  configs: list[Mapping[str, object]] = []
  transports: list[_ChannelTransport] = []
  fetches: list[Sequence[models.ModelOption]] = []
  restarts_to_fail = failed_restarts

  def worker_config(conversation_id: str) -> Mapping[str, object]:
    config = {
      'conversation_id': conversation_id,
      'model': 'reflection/beam',
      'api': 'openai-chat-completions',
      'base_url': 'http://127.0.0.1:8000/v1',
      'approval_mode': 'approve',
    }
    configs.append(config)
    return config

  async def transport(
    config: Mapping[str, object],
    workspace: pathlib.Path,
    conversation_id: str,
    on_message: rpc_client.MessageHandler,
  ) -> tuple[rpc_client.Transport, str]:
    del config, workspace, on_message
    nonlocal restarts_to_fail
    if transports and restarts_to_fail:
      # The daemon cannot stop the conversation's worker.
      restarts_to_fail -= 1
      raise grpc.RpcError()
    transport = _ChannelTransport(
      port, second_worker_ready if len(transports) == 1 else None
    )
    transports.append(transport)
    return transport, conversation_id

  async def model_source() -> Sequence[models.ModelOption]:
    fetches.append(offered)
    return offered

  server = acp.Server(
    worker_config=worker_config,
    transport=transport,
    reasoning_effort='high',
    model_source=model_source,
  )
  reader = asyncio.StreamReader()
  output = _Lines()
  task = asyncio.create_task(server.serve(reader, output))
  client = _Client(reader, output, server)
  client.transports = transports
  client.model_fetches = fetches

  response, _ = await client.wait_response(
    client.request('initialize', {'protocolVersion': 1})
  )
  assert response['result'] == {
    'protocolVersion': 1,
    'agentCapabilities': {
      'loadSession': False,
      'promptCapabilities': {
        'image': False,
        'audio': False,
        'embeddedContext': True,
      },
    },
    'authMethods': [],
  }
  response, _ = await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  result = response['result']
  assert isinstance(result, dict)
  assert result['sessionId'] == configs[0]['conversation_id']
  return client, task, configs


async def _finish(client: _Client, task: asyncio.Task[None]) -> None:
  client.close()
  await asyncio.wait_for(task, timeout=5)


def _content_text(update: dict[str, object]) -> object:
  """The text of an update's single content block, if it has one."""
  content = update.get('content')
  if not isinstance(content, dict):
    return None
  return cast(dict[str, object], content).get('text')


def _chunk_texts(messages: Sequence[dict[str, object]]) -> list[object]:
  """The text of every streamed chunk among the server's notifications."""
  chunks: list[object] = []
  for message in messages:
    if message.get('method') != 'session/update':
      continue
    params = cast(dict[str, object], message['params'])
    update = cast(dict[str, object], params['update'])
    content = cast(dict[str, object], update['content'])
    chunks.append(content['text'])
  return chunks


@pytest.mark.asyncio
async def test_prompt_streams_turn_and_forwards_permission(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])

  prompt_id = client.request(
    'session/prompt',
    {
      'sessionId': session_id,
      'prompt': [
        {'type': 'text', 'text': 'list files'},
        {
          'type': 'resource',
          'resource': {'uri': 'file:///x.go', 'text': 'package x'},
        },
      ],
    },
  )
  updates: list[dict[str, object]] = []
  while True:
    message = await client.next()
    if message.get('method') == 'session/request_permission':
      params = cast(dict[str, object], message['params'])
      assert params['sessionId'] == session_id
      tool_call = cast(dict[str, object], params['toolCall'])
      assert tool_call['toolCallId'] == 'call-1'
      assert tool_call['rawInput'] == {'command': 'ls'}
      client.respond(
        message['id'],
        {'outcome': {'outcome': 'selected', 'optionId': 'allow'}},
      )
      continue
    if message.get('id') == prompt_id:
      break
    params = cast(dict[str, object], message['params'])
    assert params['sessionId'] == session_id
    updates.append(cast(dict[str, object], params['update']))

  assert message['result'] == {
    'stopReason': 'end_turn',
    'usage': {'inputTokens': 50, 'outputTokens': 10, 'totalTokens': 60},
  }
  assert agent.user_messages == [
    'list files\n\n<file uri="file:///x.go">\npackage x\n</file>'
  ]
  assert [approval.approved for approval in agent.approvals] == [True]
  assert agent.configs[0].model == 'reflection/beam'
  assert agent.configs[0].reasoning_effort == 'high'
  assert [
    (
      update['sessionUpdate'],
      update.get('toolCallId'),
      update.get('status'),
      _content_text(update),
    )
    for update in updates
  ] == [
    ('agent_thought_chunk', None, None, 'Hmm.'),
    ('tool_call', 'call-1', 'in_progress', None),
    ('tool_call_update', 'call-1', 'completed', None),
    ('agent_message_chunk', None, None, 'Hello '),
    ('agent_message_chunk', None, None, 'world'),
  ]
  tool_update = updates[2]
  assert tool_update['content'] == [
    {
      'type': 'content',
      'content': {'type': 'text', 'text': 'README.md\nmain.go'},
    }
  ]
  await _finish(client, task)


@pytest.mark.asyncio
async def test_rejected_permission_denies_the_call(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'go'}]},
  )
  while True:
    message = await client.next()
    if message.get('method') == 'session/request_permission':
      client.respond(message['id'], {'outcome': {'outcome': 'cancelled'}})
      continue
    if message.get('id') == prompt_id:
      break
  assert message['result'] == {
    'stopReason': 'end_turn',
    'usage': {'inputTokens': 20, 'outputTokens': 6, 'totalTokens': 26},
  }
  assert [approval.approved for approval in agent.approvals] == [False]
  await _finish(client, task)


@pytest.mark.asyncio
async def test_cancel_interrupts_the_running_turn(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'hang'}]},
  )
  while True:
    message = await client.next()
    if 'working' in json.dumps(message):
      break
  client.notify('session/cancel', {'sessionId': session_id})
  response, _ = await client.wait_response(prompt_id)
  assert response['result'] == {'stopReason': 'cancelled'}
  assert agent.interrupts == 1
  # The interrupted turn's completion must not end the next prompt early.
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_cancel_before_the_turn_starts_is_not_lost(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request_then_cancel(session_id, 'list files')
  response, _ = await client.wait_response(prompt_id)
  assert response['result'] == {'stopReason': 'cancelled'}
  assert agent.user_messages == []
  assert agent.interrupts == 0
  await _finish(client, task)


@pytest.mark.asyncio
async def test_cancel_while_permission_is_pending(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'go'}]},
  )
  while True:
    message = await client.next()
    if message.get('method') == 'session/request_permission':
      break
  # The client dismisses its approval card without answering.
  client.notify('session/cancel', {'sessionId': session_id})
  response, _ = await client.wait_response(prompt_id)
  assert response['result'] == {
    'stopReason': 'cancelled',
    'usage': {'inputTokens': 20, 'outputTokens': 6, 'totalTokens': 26},
  }
  assert agent.interrupts == 1
  assert agent.approvals == []
  # A late answer to the dismissed request is ignored, and the session works.
  client.respond(
    message['id'], {'outcome': {'outcome': 'selected', 'optionId': 'allow'}}
  )
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  assert agent.approvals == []
  await _finish(client, task)


@pytest.mark.asyncio
async def test_cancel_and_permission_answer_in_one_read(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'go'}]},
  )
  while True:
    message = await client.next()
    if message.get('method') == 'session/request_permission':
      break
  client.cancel_then_respond(session_id, message['id'])
  response, _ = await client.wait_response(prompt_id)
  assert response['result'] == {
    'stopReason': 'cancelled',
    'usage': {'inputTokens': 20, 'outputTokens': 6, 'totalTokens': 26},
  }
  assert agent.interrupts == 1
  assert agent.approvals == []
  # The server survived the already-settled answer and still serves.
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_cancel_before_a_queued_permission_request(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {
      'sessionId': session_id,
      'prompt': [{'type': 'text', 'text': 'hang-approval'}],
    },
  )
  while True:
    message = await client.next()
    if 'working' in json.dumps(message):
      break
  client.notify('session/cancel', {'sessionId': session_id})
  response, preceding = await client.wait_response(prompt_id)
  assert response['result'] == {'stopReason': 'cancelled'}
  # The approval that arrived after the cancel was never put to the client.
  assert [m.get('method') for m in preceding] == []
  assert agent.interrupts == 1
  assert agent.approvals == []
  await _finish(client, task)


@pytest.mark.asyncio
async def test_worker_loss_during_approval_fails_prompt_and_session(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {
      'sessionId': session_id,
      'prompt': [{'type': 'text', 'text': 'drop-during-approval'}],
    },
  )
  # The permission request is outstanding when the worker goes away; the
  # client never answers it.
  response, preceding = await client.wait_response(prompt_id)
  assert [m.get('method') for m in preceding] == ['session/request_permission']
  assert response['error'] == {
    'code': -32603,
    'message': 'agent connection failed; check provider configuration',
  }
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'again'}]},
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': 'agent connection failed; start a new session',
  }
  await _finish(client, task)


@pytest.mark.asyncio
async def test_queued_approval_after_worker_loss_is_not_asked(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'hang'}]},
  )
  while True:
    message = await client.next()
    if 'working' in json.dumps(message):
      break
  # The worker asks for an approval and dies before the turn reads it.
  client.inject_worker_events(
    session_id,
    [
      agent_pb2.SessionMessage(
        hitl_request=agent_pb2.HITLRequest(
          call=agent_pb2.ToolCall(
            id='call-5',
            type='function',
            function=agent_pb2.FunctionCall(name='bash', arguments='{}'),
          )
        )
      ),
      None,
    ],
  )
  response, preceding = await client.wait_response(prompt_id)
  assert [m.get('method') for m in preceding] == []
  assert response['error'] == {
    'code': -32603,
    'message': 'agent connection failed; check provider configuration',
  }
  await _finish(client, task)


@pytest.mark.asyncio
async def test_shutdown_during_session_start_closes_the_worker(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  ready = asyncio.Event()
  client, task, configs = await _start(port, tmp_path, ready)
  new_id = client.request(
    'session/new', {'cwd': str(tmp_path), 'mcpServers': []}
  )

  async def second_worker_starting() -> None:
    while len(configs) < 2:
      await asyncio.sleep(0)

  await asyncio.wait_for(second_worker_starting(), timeout=5)
  # The ACP client goes away while the second worker is still starting.
  await _finish(client, task)
  assert [transport.closed for transport in client.transports] == [True, True]
  # No response was written for the abandoned request.
  leftover: list[dict[str, object]] = []
  output = client._output  # pyright: ignore[reportPrivateUsage]
  while not output.messages.empty():
    leftover.append(output.messages.get_nowait())
  assert all(m.get('id') != new_id for m in leftover)


@pytest.mark.asyncio
async def test_slow_session_start_does_not_block_other_sessions(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  ready = asyncio.Event()
  client, task, configs = await _start(port, tmp_path, ready)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'hang'}]},
  )
  while True:
    message = await client.next()
    if 'working' in json.dumps(message):
      break
  # A second session whose worker is slow to start...
  new_id = client.request(
    'session/new', {'cwd': str(tmp_path), 'mcpServers': []}
  )

  async def second_worker_starting() -> None:
    while len(configs) < 2:
      await asyncio.sleep(0)

  await asyncio.wait_for(second_worker_starting(), timeout=5)
  # ...must not keep the first session's cancel from being served.
  client.notify('session/cancel', {'sessionId': session_id})
  response, preceding = await client.wait_response(prompt_id)
  assert response['result'] == {'stopReason': 'cancelled'}
  assert agent.interrupts == 1
  assert all(m.get('id') != new_id for m in preceding)
  ready.set()
  response, _ = await client.wait_response(new_id)
  result = response['result']
  assert isinstance(result, dict)
  assert result['sessionId'] == configs[1]['conversation_id']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_failure_notice_after_tool_round_fails_the_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'toolfail'}],
      },
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': 'Provider returned 429; giving up.',
  }
  updates = [
    cast(dict[str, object], cast(dict[str, object], m['params'])['update'])
    for m in preceding
    if m.get('method') == 'session/update'
  ]
  assert [(u['sessionUpdate'], u.get('status')) for u in updates] == [
    ('tool_call', 'in_progress'),
    ('tool_call_update', 'completed'),
  ]
  await _finish(client, task)


def _raw_input(arguments: str) -> object:
  """The `rawInput` of the tool card for a call with these arguments."""
  updates, _ = acp.Translator().translate(
    agent_pb2.SessionMessage(
      agent_message=agent_pb2.AgentMessage(
        id='assistant-6',
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(
            tool_calls=[
              agent_pb2.ToolCall(
                id='call-6',
                type='function',
                function=agent_pb2.FunctionCall(
                  name='bash', arguments=arguments
                ),
              )
            ]
          ),
          success=True,
        ),
        inference=agent_pb2.InferenceConfig(model='reflection/beam'),
      )
    )
  )
  return updates[0]['rawInput']


def test_tool_call_raw_input_preserves_empty_and_invalid_arguments() -> None:
  assert _raw_input('{}') == {}
  assert _raw_input('') == {}
  assert _raw_input('{"command": "ls"}') == {'command': 'ls'}
  assert _raw_input('[1, 2]') == {'arguments': [1, 2]}
  assert _raw_input('not json') == {'arguments': 'not json'}


def test_streamed_refusal_is_not_repeated() -> None:
  translator = acp.Translator()
  updates, outcome = translator.translate(
    agent_pb2.SessionMessage(
      runtime_delta=agent_pb2.RuntimeDelta(
        stream_refusal=agent_pb2.StreamRefusal(refusal='No.')
      )
    )
  )
  assert outcome is None
  assert [u['content'] for u in updates] == [{'type': 'text', 'text': 'No.'}]
  updates, outcome = translator.translate(
    agent_pb2.SessionMessage(
      agent_message=agent_pb2.AgentMessage(
        id='assistant-5',
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(refusal='No.'), success=True
        ),
        inference=agent_pb2.InferenceConfig(model='reflection/beam'),
      )
    )
  )
  assert (updates, outcome) == ([], None)


def test_committed_answer_supplies_text_the_stream_did_not_deliver() -> None:
  translator = acp.Translator()
  translator.translate(
    _delta(stream_content=agent_pb2.StreamContent(content='The command'))
  )
  updates, outcome = translator.translate(
    agent_pb2.SessionMessage(
      agent_message=agent_pb2.AgentMessage(
        id='assistant-6',
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(
            content='The command printed 42.'
          ),
          success=True,
        ),
        inference=agent_pb2.InferenceConfig(model='reflection/beam'),
      ),
      stream_final=True,
    )
  )
  assert outcome is None
  assert [u['content'] for u in updates] == [
    {'type': 'text', 'text': ' printed 42.'}
  ]


@pytest.mark.asyncio
async def test_rejects_unknown_sessions_methods_and_bad_lines(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start(port, tmp_path)
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': 'nope', 'prompt': [{'type': 'text', 'text': 'x'}]},
    )
  )
  assert response['error'] == {'code': -32602, 'message': 'unknown session'}
  response, _ = await client.wait_response(
    client.request('session/load', {'sessionId': 'nope'})
  )
  error = response['error']
  assert isinstance(error, dict)
  assert error['code'] == -32601
  client._reader.feed_data(b'not json\n')  # pyright: ignore[reportPrivateUsage]
  message = await client.next()
  error = message['error']
  assert isinstance(error, dict)
  assert error['code'] == -32700
  await _finish(client, task)


def test_prompt_text_and_tool_kinds() -> None:
  assert acp.prompt_text([
    {'type': 'text', 'text': 'a'},
    {'type': 'resource_link', 'uri': 'file:///b', 'name': 'b'},
    {'type': 'image', 'data': 'x'},
    {'type': 'text', 'text': ''},
  ]) == (
    'a\n\nReferenced resource: b (file:///b)\n\n'
    '[image attachment omitted: not supported by mirror acp]'
  )
  assert acp.tool_kind('bash') == 'execute'
  assert acp.tool_kind('Read') == 'read'
  assert acp.tool_kind('mirror_memory') == 'other'


@pytest.mark.asyncio
async def test_notice_fails_prompt_and_unstreamed_answer_is_forwarded(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'notice'}]},
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': 'Model request failed; check credentials.',
  }
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_new_session_offers_models_by_protocol_and_efforts(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start(port, tmp_path)
  response, _ = await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  result = cast(dict[str, object], response['result'])
  assert result['configOptions'] == [
    {
      'id': 'model',
      'name': 'Model',
      'category': 'model',
      'type': 'select',
      'currentValue': 'openai-chat-completions/reflection/beam',
      'options': [
        {
          'group': 'openai-codex-responses',
          'name': 'ChatGPT plan billing (Codex login)',
          'options': [
            {
              'value': 'openai-codex-responses/gpt-6-astra',
              'name': 'GPT-6-Astra',
              'description': 'gpt-6-astra',
            }
          ],
        },
        {
          'group': 'openai-chat-completions',
          'name': 'OpenRouter',
          'options': [
            {
              'value': 'openai-chat-completions/reflection/beam',
              'name': 'reflection/beam',
            },
            {
              'value': 'openai-chat-completions/beam-mini',
              'name': 'Beam Mini',
            },
          ],
        },
        {
          'group': 'anthropic-messages',
          'name': 'Anthropic Messages API',
          'options': [
            {
              'value': 'anthropic-messages/claude-opus-4-8',
              'name': 'Claude Opus 4.8',
            }
          ],
        },
        {
          'group': 'gemini',
          'name': 'Google Gemini Developer API',
          'options': [
            {'value': 'gemini/gemini-4-argon', 'name': 'Gemini 4 Argon'}
          ],
        },
      ],
    },
    {
      'id': 'effort',
      'name': 'Reasoning effort',
      'category': 'thought_level',
      'type': 'select',
      'currentValue': 'high',
      'options': [
        {'value': 'none', 'name': 'provider default'},
        {'value': 'low', 'name': 'low'},
        {'value': 'medium', 'name': 'medium'},
        {'value': 'high', 'name': 'high'},
        {'value': 'xhigh', 'name': 'xhigh'},
        {'value': 'max', 'name': 'max'},
      ],
    },
  ]
  legacy = cast(dict[str, object], result['models'])
  assert legacy['currentModelId'] == 'openai-chat-completions/reflection/beam'
  assert [
    cast(dict[str, object], model)['modelId']
    for model in cast(list[object], legacy['availableModels'])
  ] == [
    'openai-chat-completions/reflection/beam',
    'openai-codex-responses/gpt-6-astra',
    'anthropic-messages/claude-opus-4-8',
    'gemini/gemini-4-argon',
    'openai-chat-completions/beam-mini',
  ]
  # Both sessions offer the models listed once.
  assert len(client.model_fetches) == 1
  await _finish(client, task)


@pytest.mark.asyncio
async def test_switch_on_the_same_provider_reconfigures_the_worker(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  offered = (
    # A catalog names the starting model, without the session's endpoint.
    models.ModelOption(
      'reflection/beam', 'Beam', api='openai-chat-completions'
    ),
    *_OFFERED,
    # A later entry for an offered model does not replace it.
    models.ModelOption(
      'beam-mini',
      'Beam Mini elsewhere',
      api='openai-chat-completions',
      base_url='http://127.0.0.1:9/v1',
    ),
  )
  client, task, configs = await _start(port, tmp_path, offered=offered)
  session_id = str(configs[0]['conversation_id'])
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {'sessionId': session_id, 'configId': 'effort', 'value': 'low'},
    )
  )
  result = cast(dict[str, object], response['result'])
  assert [
    (option['id'], option['currentValue'])
    for option in cast(list[dict[str, object]], result['configOptions'])
  ] == [('model', 'openai-chat-completions/reflection/beam'), ('effort', 'low')]
  response, _ = await client.wait_response(
    client.request(
      'session/set_model',
      {'sessionId': session_id, 'modelId': 'openai-chat-completions/beam-mini'},
    )
  )
  assert response['result'] == {}
  # The worker reads its configuration before the next prompt.
  await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert [
    (config.model, config.reasoning_effort) for config in agent.configs
  ] == [
    ('reflection/beam', 'high'),
    ('reflection/beam', 'low'),
    ('beam-mini', 'low'),
  ]
  assert {(config.api, config.base_url) for config in agent.configs} == {
    ('openai-chat-completions', 'http://127.0.0.1:8000/v1')
  }
  assert len(client.transports) == 1
  await _finish(client, task)


@pytest.mark.asyncio
async def test_switch_to_another_provider_restarts_the_worker(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'notice'}]},
    )
  )
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {
        'sessionId': session_id,
        'configId': 'model',
        'value': 'openai-codex-responses/gpt-6-astra',
      },
    )
  )
  result = cast(dict[str, object], response['result'])
  options = cast(list[dict[str, object]], result['configOptions'])
  assert options[0]['currentValue'] == 'openai-codex-responses/gpt-6-astra'
  assert [transport.closed for transport in client.transports] == [True, False]
  # The replacement's replay of the conversation is not shown again.
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  restarted = agent.configs[-1]
  assert (
    restarted.model,
    restarted.api,
    restarted.base_url,
    restarted.reasoning_effort,
  ) == ('gpt-6-astra', 'openai-codex-responses', '', 'high')
  await _finish(client, task)


@pytest.mark.asyncio
async def test_switch_to_another_provider_keeps_the_model_when_it_fails(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, configs = await _start(port, tmp_path, failed_restarts=1)
  session_id = str(configs[0]['conversation_id'])
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {
        'sessionId': session_id,
        'configId': 'model',
        'value': 'openai-codex-responses/gpt-6-astra',
      },
    )
  )
  assert cast(dict[str, object], response['error'])['code'] == -32603
  # A worker on the previous settings took over, and the session goes on.
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': session_id,
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  restored = agent.configs[-1]
  assert (
    restored.model,
    restored.api,
    restored.base_url,
    restored.reasoning_effort,
  ) == (
    'reflection/beam',
    'openai-chat-completions',
    'http://127.0.0.1:8000/v1',
    'high',
  )
  assert [transport.closed for transport in client.transports] == [True, False]
  await _finish(client, task)


@pytest.mark.asyncio
async def test_switch_to_another_provider_fails_the_session_if_nothing_starts(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path, failed_restarts=2)
  session_id = str(configs[0]['conversation_id'])
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {
        'sessionId': session_id,
        'configId': 'model',
        'value': 'openai-codex-responses/gpt-6-astra',
      },
    )
  )
  assert cast(dict[str, object], response['error'])['code'] == -32603
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'notice'}]},
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': 'agent connection failed; start a new session',
  }
  await _finish(client, task)


@pytest.mark.asyncio
async def test_switch_to_another_provider_refuses_requests_meanwhile(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  ready = asyncio.Event()
  client, task, configs = await _start(port, tmp_path, ready)
  session_id = str(configs[0]['conversation_id'])
  switch_id = client.request(
    'session/set_config_option',
    {
      'sessionId': session_id,
      'configId': 'model',
      'value': 'openai-codex-responses/gpt-6-astra',
    },
  )

  async def replacement_starting() -> None:
    while len(client.transports) < 2:
      await asyncio.sleep(0)

  await asyncio.wait_for(replacement_starting(), timeout=5)
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'notice'}]},
    )
  )
  assert response['error'] == {
    'code': -32600,
    'message': 'the session is switching providers',
  }
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {'sessionId': session_id, 'configId': 'effort', 'value': 'low'},
    )
  )
  assert response['error'] == {
    'code': -32600,
    'message': 'the session is already switching',
  }
  ready.set()
  response, _ = await client.wait_response(switch_id)
  result = cast(dict[str, object], response['result'])
  options = cast(list[dict[str, object]], result['configOptions'])
  assert [option['currentValue'] for option in options] == [
    'openai-codex-responses/gpt-6-astra',
    'high',
  ]
  await _finish(client, task)


@pytest.mark.asyncio
async def test_switch_to_another_provider_waits_for_the_running_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'hang'}]},
  )
  while True:
    message = await client.next()
    if 'working' in json.dumps(message):
      break
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {
        'sessionId': session_id,
        'configId': 'model',
        'value': 'openai-codex-responses/gpt-6-astra',
      },
    )
  )
  assert response['error'] == {
    'code': -32600,
    'message': 'finish or cancel the running prompt before switching providers',
  }
  client.notify('session/cancel', {'sessionId': session_id})
  response, _ = await client.wait_response(prompt_id)
  assert response['result'] == {'stopReason': 'cancelled'}
  assert len(client.transports) == 1
  await _finish(client, task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('value', 'credential'),
  [
    ('gemini/gemini-4-argon', 'GEMINI_API_KEY'),
    ('anthropic-messages/claude-opus-4-8', 'ANTHROPIC_API_KEY'),
  ],
)
async def test_switch_without_credentials_keeps_the_model(
  fake_agent: tuple[_FakeAgent, int],
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  value: str,
  credential: str,
) -> None:
  # Neither the environment nor a saved key file offers credentials.
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('GEMINI_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {'sessionId': session_id, 'configId': 'model', 'value': value},
    )
  )
  error = cast(dict[str, object], response['error'])
  assert error['code'] == -32602
  assert credential in str(error['message'])
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {'sessionId': session_id, 'configId': 'effort', 'value': 'high'},
    )
  )
  result = cast(dict[str, object], response['result'])
  options = cast(list[dict[str, object]], result['configOptions'])
  assert options[0]['currentValue'] == 'openai-chat-completions/reflection/beam'
  assert len(client.transports) == 1
  await _finish(client, task)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('params', 'message'),
  [
    ({'sessionId': 'nope', 'configId': 'model'}, 'unknown session'),
    ({'configId': 'mode', 'value': 'ask'}, 'unknown config option: mode'),
    (
      {'configId': 'model', 'value': 'gemini/nope'},
      'unknown model: gemini/nope',
    ),
    (
      {'configId': 'effort', 'value': 'extreme'},
      'unknown reasoning effort: extreme',
    ),
  ],
)
async def test_set_config_option_rejects_unknown_targets(
  fake_agent: tuple[_FakeAgent, int],
  tmp_path: pathlib.Path,
  params: dict[str, object],
  message: str,
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {'sessionId': str(configs[0]['conversation_id']), 'value': 'x'} | params,
    )
  )
  assert response['error'] == {'code': -32602, 'message': message}
  await _finish(client, task)


@pytest.mark.asyncio
async def test_offered_models_list_models_json_before_catalogs(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # Gemini has a key, an OpenAI endpoint cannot list its models, a llama.cpp
  # endpoint has no catalog here, no other protocol has credentials, and
  # `models.json` names Gemini's model too.
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
  monkeypatch.setenv('MIRROR_LLAMA_CPP_BASE_URL', 'http://127.0.0.1:9090/v1')
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  monkeypatch.delenv('OPENAI_API_KEY', raising=False)
  monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
  monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  monkeypatch.delenv('AI_GATEWAY_API_KEY', raising=False)
  config = tmp_path / 'models.json'
  config.write_text(
    json.dumps([{'id': models.GEMINI_MODEL, 'label': 'Argon', 'api': 'gemini'}])
  )
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  with socket.socket() as endpoint:
    # Bound but not listening, so the catalog request is refused.
    endpoint.bind(('127.0.0.1', 0))
    monkeypatch.setenv(
      'OPENAI_BASE_URL', f'http://127.0.0.1:{endpoint.getsockname()[1]}/v1'
    )
    offered = await acp.offered_models()
  default = models.default_model('openai-responses')
  catalog = inference_choices.GEMINI_CHOICES[0]
  assert offered == (
    models.ModelOption(models.GEMINI_MODEL, 'Argon', api='gemini'),
    models.ModelOption(default, default, default, api='openai-responses'),
    models.ModelOption(
      catalog.value, catalog.label, catalog.detail, api='gemini'
    ),
  )


_SERVER_MODELS = (
  models.ModelOption('beam-2', 'Beam 2', api='reflection'),
  models.ModelOption(
    'claude-opus-4-8', 'Claude Opus 4.8', api='anthropic-messages'
  ),
)
_KEPT = (
  mirror_pb2.Session(
    id='kept-1',
    title='Fix the build',
    updated_at=timestamp_pb2.Timestamp(seconds=1_800_000_000),
  ),
  mirror_pb2.Session(id='kept-2'),
  mirror_pb2.Session(id='kept-3', title='Write the docs'),
)
_KEPT_PER_PAGE = 2


async def _start_hosted(
  port: int,
  *,
  settings: daemon_pb2.Workspace | None = None,
  ready: asyncio.Event | None = None,
  refusal: Exception | None = None,
  models_listed: Sequence[Sequence[models.ModelOption]] = (_SERVER_MODELS,),
  attach_seconds: float = 30,
) -> tuple[_Client, asyncio.Task[None], list[Mapping[str, object]]]:
  """Serve sessions a server keeps; the transport reports what it was asked."""
  asked: list[Mapping[str, object]] = []
  transports: list[_ChannelTransport] = []

  def worker_config(conversation_id: str) -> Mapping[str, object]:
    return {
      'conversation_id': conversation_id,
      'create_session': True,
      'model': acp.SERVER_DEFAULT,
      'approval_mode': 'approve',
    }

  async def transport(
    config: Mapping[str, object],
    workspace: pathlib.Path,
    conversation_id: str,
    on_message: rpc_client.MessageHandler,
  ) -> tuple[rpc_client.Transport, str]:
    del workspace, on_message
    asked.append(config)
    created = bool(config.get('create_session', True))
    session_id = 'hosted-1' if created else conversation_id
    transports.append(
      _ChannelTransport(
        port, ready, settings=settings, refusal=refusal, session_id=session_id
      )
    )
    return transports[-1], session_id

  listings = list(models_listed)

  async def model_source() -> Sequence[models.ModelOption]:
    return listings.pop(0) if len(listings) > 1 else listings[0]

  async def session_pages(
    offset: int, limit: int
  ) -> tuple[Sequence[mirror_pb2.Session], int]:
    del limit
    return _KEPT[offset : offset + _KEPT_PER_PAGE], len(_KEPT)

  server = acp.Server(
    worker_config=worker_config,
    transport=transport,
    reasoning_effort='high',
    model_source=model_source,
    session_pages=session_pages,
    hosted_attach_seconds=attach_seconds,
  )
  reader = asyncio.StreamReader()
  output = _Lines()
  task = asyncio.create_task(server.serve(reader, output))
  client = _Client(reader, output, server)
  client.transports = transports
  response, _ = await client.wait_response(
    client.request('initialize', {'protocolVersion': 1})
  )
  assert response['result'] == {
    'protocolVersion': 1,
    'agentCapabilities': {
      'loadSession': True,
      'promptCapabilities': {
        'image': False,
        'audio': False,
        'embeddedContext': True,
      },
      'sessionCapabilities': {'list': {}, 'resume': {}},
    },
    'authMethods': [],
  }
  return client, task, asked


def _offered_models(result: object) -> list[object]:
  """Every value the model option offers in a session response, in order."""
  options = cast(dict[str, list[dict[str, object]]], result)['configOptions']
  groups = cast(list[dict[str, list[dict[str, object]]]], options[0]['options'])
  return [entry['value'] for group in groups for entry in group['options']]


def _current_values(result: object) -> list[tuple[object, object]]:
  """Each config option's id and current value in a session response."""
  options = cast(dict[str, list[dict[str, object]]], result)['configOptions']
  return [(option['id'], option['currentValue']) for option in options]


@pytest.mark.asyncio
async def test_hosted_session_switches_server_models_on_the_same_attachment(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, asked = await _start_hosted(port)
  response, _ = await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  result = cast(dict[str, object], response['result'])
  assert result['sessionId'] == 'hosted-1'
  assert asked[0]['create_session'] is True
  assert _current_values(result) == [
    ('model', '/server default'),
    ('effort', 'high'),
  ]
  response, _ = await client.wait_response(
    client.request(
      'session/set_config_option',
      {
        'sessionId': 'hosted-1',
        'configId': 'model',
        'value': 'anthropic-messages/claude-opus-4-8',
      },
    )
  )
  assert _current_values(response['result']) == [
    ('model', 'anthropic-messages/claude-opus-4-8'),
    ('effort', 'high'),
  ]
  await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': 'hosted-1',
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert [(config.model, config.api) for config in agent.configs] == [
    ('claude-opus-4-8', 'anthropic-messages'),
  ]
  assert client.transports[0].offered == agent_pb2.InferenceConfig(
    reasoning_effort='high'
  )
  assert len(client.transports) == 1
  await _finish(client, task)


@pytest.mark.asyncio
async def test_list_sessions_pages_through_what_the_server_keeps(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(port)
  response, _ = await client.wait_response(
    client.request('session/list', {'cwd': str(tmp_path)})
  )
  assert response['result'] == {
    'sessions': [
      {
        'sessionId': 'kept-1',
        'cwd': str(tmp_path),
        'title': 'Fix the build',
        'updatedAt': '2027-01-15T08:00:00+00:00',
      },
      {'sessionId': 'kept-2', 'cwd': str(tmp_path)},
    ],
    'nextCursor': '2',
  }
  response, _ = await client.wait_response(
    client.request('session/list', {'cwd': str(tmp_path), 'cursor': '2'})
  )
  assert response['result'] == {
    'sessions': [
      {'sessionId': 'kept-3', 'cwd': str(tmp_path), 'title': 'Write the docs'}
    ]
  }
  response, _ = await client.wait_response(
    client.request('session/list', {'cursor': 'later'})
  )
  assert response['error'] == {'code': -32602, 'message': 'unknown cursor'}
  await _finish(client, task)


@pytest.mark.asyncio
async def test_load_rejoins_a_kept_session_and_replays_its_conversation(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  agent.attached.add('kept-1')
  agent.history = [
    agent_pb2.AgentMessage(
      id='user-1',
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='List the files'), success=True
      ),
    ),
    agent_pb2.AgentMessage(
      id='assistant-1',
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(
          content='Looking.',
          tool_calls=[
            agent_pb2.ToolCall(
              id='call-1',
              type='function',
              function=agent_pb2.FunctionCall(
                name='bash', arguments='{"command": "ls"}'
              ),
            )
          ],
        ),
        success=True,
      ),
      inference=agent_pb2.InferenceConfig(model='beam-2'),
    ),
    agent_pb2.AgentMessage(
      id='tool-1',
      chat=agent_pb2.ChatRecord(
        tool=agent_pb2.ToolMessage(
          tool_call_id='call-1', content='README.md', name='bash'
        ),
        success=True,
      ),
    ),
  ]
  client, task, asked = await _start_hosted(port)
  response, preceding = await client.wait_response(
    client.request(
      'session/load',
      {'sessionId': 'kept-1', 'cwd': str(tmp_path), 'mcpServers': []},
    )
  )
  assert _current_values(response['result']) == [
    ('model', '/server default'),
    ('effort', 'high'),
  ]
  assert (asked[0]['conversation_id'], asked[0]['create_session']) == (
    'kept-1',
    False,
  )
  updates = [
    cast(dict[str, dict[str, object]], message['params'])['update']
    for message in preceding
  ]
  assert [
    (update['sessionUpdate'], update.get('toolCallId'), _content_text(update))
    for update in updates
  ] == [
    ('user_message_chunk', None, 'List the files'),
    ('agent_message_chunk', None, 'Looking.'),
    ('tool_call', 'call-1', None),
    ('tool_call_update', 'call-1', None),
  ]
  assert agent.configs == []
  await _finish(client, task)


@pytest.mark.asyncio
async def test_resume_rejoins_a_kept_session_without_replaying_it(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  agent.attached.add('kept-1')
  client, task, _ = await _start_hosted(
    port,
    settings=daemon_pb2.Workspace(
      inference=agent_pb2.InferenceConfig(
        model='claude-opus-4-8',
        api='anthropic-messages',
        reasoning_effort='low',
      )
    ),
  )
  response, preceding = await client.wait_response(
    client.request(
      'session/resume',
      {'sessionId': 'kept-1', 'cwd': str(tmp_path), 'mcpServers': []},
    )
  )
  assert preceding == []
  assert _current_values(response['result']) == [
    ('model', 'anthropic-messages/claude-opus-4-8'),
    ('effort', 'low'),
  ]
  assert set(_offered_models(response['result'])) == {
    'reflection/beam-2',
    'anthropic-messages/claude-opus-4-8',
  }
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': 'kept-1',
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  assert agent.user_messages == ['unstreamed']
  assert agent.configs == []
  assert client.transports[0].offered == agent_pb2.InferenceConfig(
    reasoning_effort='high'
  )
  await _finish(client, task)


@pytest.mark.asyncio
async def test_a_turn_run_for_another_client_does_not_end_the_next_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(port)
  await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  client.inject_worker_events(
    'hosted-1',
    [
      _delta(stream_content=agent_pb2.StreamContent(content='For the web')),
      _delta(turn_completed=agent_pb2.RuntimeDelta().turn_completed),
    ],
  )
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': 'hosted-1',
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_resume_of_a_session_the_server_cannot_open_reports_its_reason(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(
    port, refusal=RuntimeError('repository preparation failed')
  )
  response, _ = await client.wait_response(
    client.request(
      'session/resume', {'sessionId': 'kept-1', 'cwd': str(tmp_path)}
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': (
      'could not open the session on the server (repository preparation failed)'
    ),
  }
  await _finish(client, task)


@pytest.mark.asyncio
async def test_new_hosted_session_reaches_its_server_at_the_first_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  client, task, _ = await _start_hosted(port)
  await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  await client.wait_response(
    client.request('session/list', {'cwd': str(tmp_path)})
  )
  attached_before_prompting = set(agent.attached)
  await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': 'hosted-1',
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert (attached_before_prompting, agent.attached) == (set(), {'hosted-1'})
  await _finish(client, task)


@pytest.mark.asyncio
async def test_first_prompt_is_sent_once_the_server_has_prepared_the_session(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  prepared = asyncio.Event()
  client, task, _ = await _start_hosted(port, ready=prepared)
  response, _ = await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  assert cast(dict[str, object], response['result'])['sessionId'] == 'hosted-1'
  prompt_id = client.request(
    'session/prompt',
    {
      'sessionId': 'hosted-1',
      'prompt': [{'type': 'text', 'text': 'unstreamed'}],
    },
  )
  await client.wait_response(
    client.request('session/list', {'cwd': str(tmp_path)})
  )
  sent_while_preparing = list(agent.user_messages)
  prepared.set()
  response, preceding = await client.wait_response(prompt_id)
  assert sent_while_preparing == []
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_cancelling_the_first_prompt_while_preparing_sends_nothing(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  agent, port = fake_agent
  prepared = asyncio.Event()
  client, task, _ = await _start_hosted(port, ready=prepared)
  await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  prompt_id = client.request(
    'session/prompt',
    {'sessionId': 'hosted-1', 'prompt': [{'type': 'text', 'text': 'hang'}]},
  )
  await client.wait_response(
    client.request('session/list', {'cwd': str(tmp_path)})
  )
  client.notify('session/cancel', {'sessionId': 'hosted-1'})
  response, _ = await client.wait_response(prompt_id)
  assert response['result'] == {'stopReason': 'cancelled'}
  prepared.set()
  response, preceding = await client.wait_response(
    client.request(
      'session/prompt',
      {
        'sessionId': 'hosted-1',
        'prompt': [{'type': 'text', 'text': 'unstreamed'}],
      },
    )
  )
  assert response['result'] == {'stopReason': 'end_turn'}
  assert _chunk_texts(preceding) == ['Whole answer']
  assert agent.user_messages == ['unstreamed']
  await _finish(client, task)


@pytest.mark.asyncio
async def test_a_session_the_server_never_prepares_fails_its_prompt_in_time(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(
    port, ready=asyncio.Event(), attach_seconds=0.05
  )
  await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': 'hosted-1', 'prompt': [{'type': 'text', 'text': 'hello'}]},
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': 'could not open the session on the server (timed out)',
  }
  assert client.transports[0].closed
  await _finish(client, task)


@pytest.mark.asyncio
async def test_any_failure_to_prepare_a_new_session_fails_every_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(
    port, refusal=ValueError('unreadable origin')
  )
  await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  prompt: dict[str, object] = {
    'sessionId': 'hosted-1',
    'prompt': [{'type': 'text', 'text': 'hello'}],
  }
  first, _ = await client.wait_response(
    client.request('session/prompt', prompt)
  )
  second, _ = await client.wait_response(
    client.request('session/prompt', prompt)
  )
  failure = {
    'code': -32603,
    'message': 'could not open the session on the server (unreadable origin)',
  }
  assert (first['error'], second['error']) == (failure, failure)
  await _finish(client, task)


@pytest.mark.asyncio
async def test_new_hosted_session_the_server_cannot_prepare_fails_its_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(
    port, refusal=RuntimeError('repository preparation failed')
  )
  await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': 'hosted-1', 'prompt': [{'type': 'text', 'text': 'hello'}]},
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': (
      'could not open the session on the server (repository preparation failed)'
    ),
  }
  await _finish(client, task)


@pytest.mark.asyncio
async def test_an_empty_model_listing_is_read_again_for_the_next_session(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, _ = await _start_hosted(
    port, models_listed=((), (), _SERVER_MODELS)
  )
  first, _ = await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  second, _ = await client.wait_response(
    client.request('session/new', {'cwd': str(tmp_path), 'mcpServers': []})
  )
  assert _offered_models(first['result']) == ['/server default']
  assert set(_offered_models(second['result'])) == {
    'reflection/beam-2',
    'anthropic-messages/claude-opus-4-8',
    '/server default',
  }
  await _finish(client, task)


@pytest.mark.asyncio
async def test_a_connection_lost_between_prompts_fails_the_next_prompt(
  fake_agent: tuple[_FakeAgent, int], tmp_path: pathlib.Path
) -> None:
  _, port = fake_agent
  client, task, configs = await _start(port, tmp_path)
  session_id = str(configs[0]['conversation_id'])
  client.server._sessions[session_id].events.put_nowait(None)  # pyright: ignore[reportPrivateUsage]
  response, _ = await client.wait_response(
    client.request(
      'session/prompt',
      {'sessionId': session_id, 'prompt': [{'type': 'text', 'text': 'hello'}]},
    )
  )
  assert response['error'] == {
    'code': -32603,
    'message': 'agent connection failed; check provider configuration',
  }
  await _finish(client, task)
