"""Session streams that survive a server restart through a real channel."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Callable, Sequence
from typing import override

import grpc
import grpc.aio
import hypothesis
import pytest
from google.protobuf import empty_pb2
from hypothesis import strategies
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
)

from mirror.cli import remote_rpc_client, rpc_client


class _End:
  def __init__(self, code: grpc.StatusCode, details: str) -> None:
    self.code = code
    self.details = details


type _Outgoing = agent_pb2.SessionMessage | _End


class _Server(agent_pb2_grpc.AgentRPCServicer):
  """A hosted session whose history outlives any one Session stream."""

  def __init__(
    self, *, starts_turns: bool = False, durable: bool = False
  ) -> None:
    self.history: list[agent_pb2.AgentMessage] = []
    self.inputs: list[str] = []
    self.records_inputs = True
    self.starts_turns = starts_turns
    self.durable = durable
    self.open_turn: str | None = None
    self.answered = False
    self.on_attach: list[agent_pb2.AgentMessage] = []
    self.attachments = 0
    self.refusals = 0
    self.refused = 0
    self._outbox: asyncio.Queue[_Outgoing] | None = None
    self._changed = asyncio.Condition()

  def commit(self, record: agent_pb2.AgentMessage) -> None:
    record.sequence = len(self.history) + 1
    self.history.append(record)
    self.emit(agent_pb2.SessionMessage(agent_message=record))

  def emit(self, *messages: agent_pb2.SessionMessage) -> None:
    if self._outbox is not None:
      for message in messages:
        self._outbox.put_nowait(message)

  @property
  def connected(self) -> bool:
    return self._outbox is not None

  def end(
    self,
    code: grpc.StatusCode = grpc.StatusCode.UNAVAILABLE,
    details: str = rpc_client.RESTARTING,
    *,
    refusals: int = 0,
  ) -> None:
    assert self._outbox is not None
    self._outbox.put_nowait(_End(code, details))
    self._outbox = None
    self.refusals = refusals

  async def wait_for(self, condition: Callable[[], bool]) -> None:
    async with self._changed:
      await asyncio.wait_for(self._changed.wait_for(condition), timeout=5)

  async def _notify(self) -> None:
    async with self._changed:
      self._changed.notify_all()

  @override
  async def GetMessages(
    self,
    request: agent_pb2.GetMessagesRequest,
    context: grpc.aio.ServicerContext[
      agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
    ],
  ) -> agent_pb2.GetMessagesResponse:
    del context
    after = [
      record for record in self.history if record.sequence > request.offset
    ]
    return agent_pb2.GetMessagesResponse(
      messages=after[: request.limit],
      total=len(self.history),
      records_turn_ends=self.durable,
    )

  @override
  async def Session(
    self,
    request_iterator: AsyncIterator[agent_pb2.ClientMessage],
    context: grpc.aio.ServicerContext[
      agent_pb2.ClientMessage, agent_pb2.SessionMessage
    ],
  ) -> AsyncIterator[agent_pb2.SessionMessage]:
    if self.refusals:
      self.refusals -= 1
      self.refused += 1
      await self._notify()
      await context.abort(grpc.StatusCode.UNAVAILABLE, rpc_client.RESTARTING)
    outbox: asyncio.Queue[_Outgoing] = asyncio.Queue()
    self._outbox = outbox
    reading = asyncio.create_task(self._read(request_iterator, outbox))
    try:
      while True:
        outgoing = await outbox.get()
        if isinstance(outgoing, _End):
          await context.abort(outgoing.code, outgoing.details)
        yield outgoing
    finally:
      reading.cancel()

  async def _read(
    self,
    requests: AsyncIterator[agent_pb2.ClientMessage],
    outbox: asyncio.Queue[_Outgoing],
  ) -> None:
    async for request in requests:
      outbox.put_nowait(
        agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id=request.message_id)
        )
      )
      if request.HasField('set_inference_config'):
        self.attachments += 1
        for record in self.on_attach:
          self.commit(record)
        self.on_attach = []
      if request.HasField('message'):
        self.inputs.append(request.message_id)
        if self.records_inputs and request.message_id not in {
          record.id for record in self.history
        }:
          self.record_input(request.message)
      await self._notify()

  def record_input(self, message: agent_pb2.AgentMessage) -> None:
    if not self.starts_turns:
      self.commit(message)
      return
    record = agent_pb2.AgentMessage()
    record.CopyFrom(message)
    self.answered = False
    if self.open_turn is None:
      self.open_turn = f'turn-{message.id}'
      self.emit(_turn_started(self.open_turn))
    record.turn_id = self.open_turn
    self.commit(record)

  def end_turn(
    self, outcome: agent_pb2.TurnOutcome.ValueType, error: str = ''
  ) -> None:
    turn_id = self.open_turn
    assert turn_id is not None
    self.open_turn = None
    if outcome == agent_pb2.TURN_OUTCOME_FAILED:
      self.emit(_turn_error(error))
    if self.durable:
      self.commit(_turn_end(turn_id, outcome, error))
    if outcome == agent_pb2.TURN_OUTCOME_INTERRUPTED:
      self.emit(_turn_interrupted())
    self.emit(_turn_completed())


class _Caller:
  """Record what a session client shows, in order."""

  def __init__(self) -> None:
    self.errors: list[str] = []
    self.reconnecting: list[bool] = []
    self._reconnected = asyncio.Event()
    self._shown: list[str] = []
    self._changed = asyncio.Condition()

  async def show(self, message: agent_pb2.SessionMessage) -> None:
    if message.HasField('agent_message'):
      self._shown.append(message.agent_message.id)
    elif message.HasField('runtime_delta'):
      self._shown.append(message.runtime_delta.WhichOneof('payload') or '')
    async with self._changed:
      self._changed.notify_all()

  def reconnect(self, reconnecting: bool) -> None:
    self.reconnecting.append(reconnecting)
    if not reconnecting:
      self._reconnected.set()

  async def wait_until_reconnected(self) -> None:
    await asyncio.wait_for(self._reconnected.wait(), timeout=5)

  async def wait_until_reconnected_after(self, mark: int) -> None:
    async with asyncio.timeout(5):
      while False not in self.reconnecting[mark:]:
        await asyncio.sleep(0.001)

  async def fail(self, message: str) -> None:
    self.errors.append(message)
    async with self._changed:
      self._changed.notify_all()

  @property
  def shown(self) -> list[str]:
    return list(self._shown)

  async def wait_for(self, entry: str) -> None:
    async with self._changed:
      await asyncio.wait_for(
        self._changed.wait_for(lambda: entry in self._shown), timeout=5
      )

  async def settle(self, expected: list[str]) -> None:
    async with self._changed:
      with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(
          self._changed.wait_for(lambda: self._shown == expected), timeout=5
        )

  async def wait_for_error(self) -> None:
    async with self._changed:
      await asyncio.wait_for(
        self._changed.wait_for(lambda: bool(self.errors)), timeout=5
      )


def _record(
  record_id: str, role: str, turn_id: str | None = None
) -> agent_pb2.AgentMessage:
  chat = (
    agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content=record_id))
    if role == 'user'
    else agent_pb2.ChatRecord(
      assistant=agent_pb2.AssistantMessage(content=record_id)
    )
  )
  return agent_pb2.AgentMessage(id=record_id, turn_id=turn_id, chat=chat)


def _turn_started(turn_id: str) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      turn_id=turn_id, turn_started=empty_pb2.Empty()
    )
  )


def _turn_completed() -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(turn_completed=empty_pb2.Empty())
  )


def _turn_interrupted() -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(turn_interrupted=empty_pb2.Empty())
  )


def _turn_error(message: str) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      turn_error=agent_pb2.TurnError(message=message)
    )
  )


def _turn_end(
  turn_id: str, outcome: agent_pb2.TurnOutcome.ValueType, error: str = ''
) -> agent_pb2.AgentMessage:
  end = agent_pb2.TurnEndRecord(outcome=outcome)
  if error:
    end.error = error
  return agent_pb2.AgentMessage(
    id=f'{turn_id}-end', turn_id=turn_id, turn_end=end
  )


async def _serve(server: _Server) -> tuple[grpc.aio.Server, int]:
  listener = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(server, listener)  # pyright: ignore[reportUnknownMemberType] - protoc registration has no parameter annotations.
  port = listener.add_insecure_port('127.0.0.1:0')
  await listener.start()
  return listener, port


def _client(
  port: int, caller: _Caller, delays: tuple[float, ...] = (0.01,) * 50
) -> rpc_client.Client:
  return rpc_client.Client(
    remote_rpc_client.Transport(f'127.0.0.1:{port}', secure=False),
    'session',
    on_message=caller.show,
    on_error=caller.fail,
    inference_config=agent_pb2.InferenceConfig(model='synthetic'),
    reattach_delays=delays,
    on_reattaching=caller.reconnect,
  )


@pytest.mark.asyncio
async def test_reattach_restarting_delivers_missed_records_once() -> None:
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.commit(_record('first', 'assistant'))
    await caller.wait_for('first')
    server.end(refusals=2)
    server.commit(_record('second', 'assistant'))
    server.commit(_record('third', 'assistant'))
    server.on_attach = [_record('fourth', 'assistant')]
    await caller.wait_for('fourth')
    server.commit(_record('fifth', 'assistant'))
    await caller.wait_for('fifth')

    assert caller.shown == ['first', 'second', 'third', 'fourth', 'fifth']
    assert caller.errors == []
    assert server.refused == 2
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'late',
  [(), (_turn_completed(),)],
  ids=['silent', 'announced'],
)
async def test_reattach_restarting_completes_a_turn_that_ended_in_the_gap(
  late: tuple[agent_pb2.SessionMessage, ...],
) -> None:
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.emit(_turn_started('turn'))
    server.commit(_record('ask', 'user', 'turn'))
    await caller.wait_for('ask')
    server.end()
    server.commit(_record('answer', 'assistant', 'turn'))
    await caller.wait_for('turn_completed')
    server.emit(*late)
    server.commit(_record('after', 'assistant'))
    await caller.wait_for('after')

    assert caller.shown == [
      'turn_started',
      'ask',
      'answer',
      'turn_completed',
      'after',
    ]
    assert caller.errors == []
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_completes_a_gap_turn_before_a_later_record() -> (
  None
):
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.emit(_turn_started('turn'))
    server.commit(_record('ask', 'user', 'turn'))
    server.commit(_record('answer', 'assistant', 'turn'))
    await caller.wait_for('answer')
    server.end()
    server.commit(_record('note', 'assistant'))
    await caller.wait_for('note')

    assert caller.shown == [
      'turn_started',
      'ask',
      'answer',
      'turn_completed',
      'note',
    ]
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_reports_a_turn_that_failed_in_the_gap() -> (
  None
):
  server = _Server(durable=True)
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.emit(_turn_started('turn'))
    server.commit(_record('ask', 'user', 'turn'))
    await caller.wait_for('ask')
    server.end()
    server.commit(
      _turn_end('turn', agent_pb2.TURN_OUTCOME_FAILED, 'model offline')
    )
    await caller.wait_for('turn_completed')
    server.emit(_turn_completed())
    server.commit(_record('after', 'assistant'))
    await caller.wait_for('after')

    assert caller.shown == [
      'turn_started',
      'ask',
      'turn_error',
      'turn-end',
      'turn_completed',
      'after',
    ]
    assert caller.reconnecting == [True, False]
    assert caller.errors == []
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_keeps_a_turn_open_for_another_clients_input() -> (
  None
):
  server = _Server(durable=True)
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.emit(_turn_started('turn'))
    server.commit(_record('ask', 'user', 'turn'))
    server.commit(_record('answer', 'assistant', 'turn'))
    await caller.wait_for('answer')
    server.end()
    await caller.wait_until_reconnected()
    server.commit(_record('other', 'user', 'turn'))
    server.commit(_record('reply', 'assistant', 'turn'))
    server.commit(_turn_end('turn', agent_pb2.TURN_OUTCOME_COMPLETED))
    server.emit(_turn_completed())
    await caller.wait_for('turn_completed')

    assert caller.shown == [
      'turn_started',
      'ask',
      'answer',
      'other',
      'reply',
      'turn-end',
      'turn_completed',
    ]
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_starts_and_completes_a_turn_run_in_the_gap() -> (
  None
):
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.commit(_record('before', 'assistant'))
    await caller.wait_for('before')
    server.end()
    server.commit(_record('ask', 'user', 'turn'))
    server.commit(_record('answer', 'assistant', 'turn'))
    await caller.wait_for('turn_completed')

    assert caller.shown == [
      'before',
      'turn_started',
      'ask',
      'answer',
      'turn_completed',
    ]
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_resends_only_unrecorded_inputs() -> None:
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    client.enqueue(_record('recorded', 'user'))
    await caller.wait_for('recorded')
    server.records_inputs = False
    client.enqueue(_record('acknowledged', 'user'))
    await server.wait_for(lambda: 'acknowledged' in server.inputs)
    server.records_inputs = True
    server.end()
    await caller.wait_for('acknowledged')

    assert server.inputs == ['recorded', 'acknowledged', 'acknowledged']
    assert [record.id for record in server.history] == [
      'recorded',
      'acknowledged',
    ]
    assert caller.errors == []
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_close_while_reattaching_stops_without_an_error() -> None:
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller, delays=(0.01,) * 10_000)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.end(refusals=10_000)
    await server.wait_for(lambda: server.refused > 0)
    await asyncio.wait_for(client.close(), timeout=2)

    assert caller.errors == []
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_reports_the_failure_when_retries_run_out() -> (
  None
):
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller, delays=(0.0, 0.0))
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.end(refusals=2)
    await caller.wait_for_error()

    assert caller.errors == [
      'AgentRPC UNAVAILABLE: server is restarting; reconnect; '
      'reattaching failed: AgentRPC UNAVAILABLE: '
      'server is restarting; reconnect'
    ]
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_other_unavailable_errors_are_reported_without_reattaching() -> (
  None
):
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    server.end(details='connection reset')
    await caller.wait_for_error()

    assert caller.errors == ['AgentRPC UNAVAILABLE: connection reset']
    assert server.attachments == 1
  finally:
    await client.close()
    await listener.stop(None)


@pytest.mark.asyncio
async def test_reattach_restarting_delivers_an_input_sent_as_the_stream_ends() -> (
  None
):
  server = _Server()
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    await asyncio.sleep(0.01)
    server.end()
    await asyncio.sleep(0)
    client.enqueue(_record('racing', 'user'))
    await caller.wait_for('racing')

    assert [record.id for record in server.history] == ['racing']
    assert caller.errors == []
  finally:
    await client.close()
    await listener.stop(None)


_FUZZ_STEPS = (
  'note',
  'input',
  'unrecorded_input',
  'foreign_input',
  'answer',
  'complete',
  'fail',
  'interrupt',
  'restart',
  'refused_restart',
  'reattached',
  'pause',
)
_TURN_EVENTS = frozenset({
  'turn_started',
  'turn_error',
  'turn_interrupted',
  'turn_completed',
})


class _Fuzz:
  """Drive one random script of server events, inputs and restarts.

  Servers that commit turn-end records also fail and interrupt turns and
  record other clients' inputs, inside a turn or as a turn of their own.
  """

  def __init__(
    self, server: _Server, client: rpc_client.Client, caller: _Caller
  ) -> None:
    self._server = server
    self._client = client
    self._caller = caller
    self._restarted: int | None = None
    self._notes = 0
    self._answers = 0
    self._foreign = 0
    self._inputs: list[str] = []

  def _idle(self) -> bool:
    recorded = {record.id for record in self._server.history}
    return self._server.open_turn is None and all(
      input_id in recorded for input_id in self._inputs
    )

  async def step(self, step: str) -> None:
    server = self._server
    match step:
      case 'note' if self._idle():
        self._notes += 1
        server.commit(_record(f'note-{self._notes}', 'assistant'))
      case 'input' | 'unrecorded_input' if self._idle():
        input_id = f'input-{len(self._inputs) + 1}'
        self._inputs.append(input_id)
        server.records_inputs = step == 'input'
        self._client.enqueue(_record(input_id, 'user'))
        await server.wait_for(lambda: input_id in server.inputs)
        server.records_inputs = True
      case 'foreign_input' if server.durable and (
        self._idle() or server.open_turn is not None
      ):
        self._foreign += 1
        server.record_input(_record(f'foreign-{self._foreign}', 'user'))
      case 'answer' if server.open_turn is not None and not server.answered:
        server.answered = True
        self._answers += 1
        server.commit(
          _record(
            f'{server.open_turn}-answer-{self._answers}',
            'assistant',
            server.open_turn,
          )
        )
      case 'complete' if server.open_turn is not None and server.answered:
        server.end_turn(agent_pb2.TURN_OUTCOME_COMPLETED)
      case 'fail' if server.durable and server.open_turn is not None:
        server.end_turn(agent_pb2.TURN_OUTCOME_FAILED, 'model offline')
      case 'interrupt' if server.durable and server.open_turn is not None:
        server.end_turn(agent_pb2.TURN_OUTCOME_INTERRUPTED)
      case 'restart' if server.connected:
        self._restarted = len(self._caller.reconnecting)
        server.end()
      case 'refused_restart' if server.connected:
        self._restarted = len(self._caller.reconnecting)
        server.end(refusals=2)
      case 'reattached' if self._restarted is not None:
        await self._caller.wait_until_reconnected_after(self._restarted)
      case 'pause':
        await asyncio.sleep(0.005)
      case _:
        pass

  async def finish(self) -> None:
    server = self._server
    if server.connected:
      server.end()
    await server.wait_for(
      lambda: all(
        input_id in {record.id for record in server.history}
        for input_id in self._inputs
      )
    )
    await self.step('answer')
    await self.step('complete')

  def expected(self) -> list[str]:
    shown: list[str] = []
    started: set[str] = set()
    for record in self._server.history:
      if record.turn_id and record.turn_id not in started:
        started.add(record.turn_id)
        shown.append('turn_started')
      shown.extend(self._shown_for(record))
    return shown

  def _shown_for(self, record: agent_pb2.AgentMessage) -> list[str]:
    if record.HasField('turn_end'):
      outcome = record.turn_end.outcome
      return [
        *(['turn_error'] if outcome == agent_pb2.TURN_OUTCOME_FAILED else []),
        record.id,
        *(
          ['turn_interrupted']
          if outcome == agent_pb2.TURN_OUTCOME_INTERRUPTED
          else []
        ),
        'turn_completed',
      ]
    if not self._server.durable and '-answer-' in record.id:
      return [record.id, 'turn_completed']
    return [record.id]


async def _fuzz_reattach(steps: Sequence[str], *, durable: bool) -> None:
  server = _Server(starts_turns=True, durable=durable)
  listener, port = await _serve(server)
  caller = _Caller()
  client = _client(port, caller)
  fuzz = _Fuzz(server, client, caller)
  try:
    await client.start(replay=False, wait_for_attachment=True)
    for step in steps:
      await fuzz.step(step)
    await fuzz.finish()
    expected = fuzz.expected()
    await caller.settle(expected)

    assert caller.errors == []
    assert caller.shown == expected
    assert [record.id for record in server.history] == [
      record_id for record_id in expected if record_id not in _TURN_EVENTS
    ]
  finally:
    await client.close()
    await listener.stop(None)


@hypothesis.settings(
  max_examples=int(os.environ.get('MIRROR_REATTACH_FUZZ_EXAMPLES', '25')),
  deadline=None,
  database=None,
  suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
@hypothesis.given(
  strategies.lists(
    strategies.sampled_from(_FUZZ_STEPS), min_size=8, max_size=60
  ),
  strategies.booleans(),
)
def test_reattach_restarting_shows_every_record_and_turn_once(
  steps: list[str], durable: bool
) -> None:
  asyncio.run(_fuzz_reattach(steps, durable=durable))
