"""Continue a session's message stream across a reattach exactly once."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from google.protobuf import empty_pb2
from mirror.protocol import agent_pb2

_STREAMING = frozenset({
  'stream_content',
  'stream_reasoning',
  'stream_refusal',
  'stream_tool_call',
})
_ENDINGS = frozenset({'turn_error', 'turn_interrupted', 'turn_completed'})
_LIFECYCLE = _STREAMING | _ENDINGS | {'stream_aborted', 'turn_started'}


def _runtime(delta: agent_pb2.RuntimeDelta) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(runtime_delta=delta)


def _turn_started(turn_id: str | None) -> agent_pb2.SessionMessage:
  return _runtime(
    agent_pb2.RuntimeDelta(turn_id=turn_id, turn_started=empty_pb2.Empty())
  )


def _turn_completed() -> agent_pb2.SessionMessage:
  return _runtime(agent_pb2.RuntimeDelta(turn_completed=empty_pb2.Empty()))


def _turn_interrupted() -> agent_pb2.SessionMessage:
  return _runtime(agent_pb2.RuntimeDelta(turn_interrupted=empty_pb2.Empty()))


def _turn_error(message: str) -> agent_pb2.SessionMessage:
  return _runtime(
    agent_pb2.RuntimeDelta(turn_error=agent_pb2.TurnError(message=message))
  )


def _awaits_turn(request: agent_pb2.ClientMessage) -> bool:
  return (
    request.WhichOneof('payload') == 'message'
    and request.message.WhichOneof('payload') == 'chat'
    and request.message.chat.WhichOneof('message') == 'user'
  )


def _turn_of(record: agent_pb2.AgentMessage) -> str | None:
  return record.turn_id if record.HasField('turn_id') else None


def _covered(message: agent_pb2.SessionMessage, sequence: int) -> bool:
  return (
    message.HasField('agent_message')
    and message.agent_message.HasField('sequence')
    and message.agent_message.sequence <= sequence
  )


def after_history(
  buffered: Sequence[agent_pb2.SessionMessage], sequence: int
) -> list[agent_pb2.SessionMessage]:
  """Drop live turn and preview events that delivered history already covers.

  An event the live stream sent before a committed record the caller already
  has happened before that record, so the history it describes was shown.
  """
  cut = max(
    (
      index
      for index, message in enumerate(buffered)
      if _covered(message, sequence)
    ),
    default=-1,
  )
  return [
    message
    for index, message in enumerate(buffered)
    if index > cut
    or message.runtime_delta.WhichOneof('payload') not in _LIFECYCLE
  ]


class Continuity:
  """Track what the caller was shown so a reattached stream continues it once.

  Until `resume` is called every message passes through unchanged. Afterwards,
  committed records the caller already has are dropped, a second replay is
  suppressed, and turn boundaries that fell in the gap are rebuilt from the
  committed turn-end records. History from servers that commit no turn-end
  records falls back to inferring a turn's end from its final reply.
  """

  def __init__(self) -> None:
    self.sequence: int | None = None
    self._resumed = False
    self._durable_ends = False
    self._last_chat: agent_pb2.AgentMessage | None = None
    self._streaming = False
    self._turn_open = False
    self._turn_id: str | None = None
    self._error_shown = False
    self._interrupt_shown = False
    self._end: agent_pb2.TurnEndRecord | None = None
    self._ended_turn: str | None = None
    self._ended_turns: set[str] = set()
    self._records_since_end = False
    self._superseded: set[str] = set()

  def replaced(self) -> Continuity:
    """Return a tracker for a caller that replaced its display with a snapshot."""
    fresh = Continuity()
    fresh.sequence = self.sequence
    fresh._durable_ends = self._durable_ends
    return fresh

  def advance(self, sequence: int) -> None:
    """Record that the caller has every committed record through `sequence`."""
    self.sequence = max(self.sequence or 0, sequence)

  def expect_turn_end_records(self) -> None:
    """Rely on the server's turn-end records instead of inferring turn ends."""
    self._durable_ends = True

  def observe(self, response: agent_pb2.SessionMessage) -> None:
    """Update the shown state with one message delivered to the caller."""
    kind = response.WhichOneof('payload')
    if kind == 'agent_message':
      self._observe_record(response)
    elif kind == 'runtime_delta':
      self._observe_delta(response.runtime_delta)

  def resume(self) -> list[agent_pb2.SessionMessage]:
    """Start filtering for a replacement stream; finish what the gap cut off."""
    self._resumed = True
    shown: list[agent_pb2.SessionMessage] = []
    if self._streaming:
      shown.append(
        _runtime(agent_pb2.RuntimeDelta(stream_aborted=empty_pb2.Empty()))
      )
    if self._turn_open and self._end is not None:
      shown.extend(
        self._ending(
          self._end, None, shown=(self._error_shown, self._interrupt_shown)
        )
      )
    return shown

  def continuation(
    self, response: agent_pb2.SessionMessage
  ) -> list[agent_pb2.SessionMessage]:
    """Return the messages to show the caller in place of `response`."""
    if not self._resumed:
      return [response]
    kind = response.WhichOneof('payload')
    if kind == 'replay_done':
      return []
    if kind == 'agent_message':
      return self._record_continuation(response)
    if kind == 'runtime_delta':
      return self._delta_continuation(response)
    return [response]

  def settle(
    self, pending: Iterable[agent_pb2.ClientMessage]
  ) -> list[agent_pb2.SessionMessage]:
    """Complete a turn that history without turn-end records shows ended."""
    if not self._turn_open or (
      self._durable_ends and self._turn_id is not None
    ):
      return []
    if not self._settled(pending):
      return []
    self._superseded = {'turn_completed'}
    return [_turn_completed()]

  def _observe_record(self, response: agent_pb2.SessionMessage) -> None:
    record = response.agent_message
    if record.HasField('sequence'):
      self.advance(record.sequence)
    payload = record.WhichOneof('payload')
    if payload == 'turn_end':
      self._durable_ends = True
      if self._turn_open:
        self._end = record.turn_end
      return
    if payload == 'chat':
      self._last_chat = record
      if record.chat.WhichOneof('message') == 'assistant':
        self._streaming = False
    if response.stream_final:
      self._streaming = False
    turn_id = _turn_of(record)
    if not self._turn_open and turn_id is not None:
      self._records_since_end |= turn_id == self._ended_turn

  def _observe_delta(self, delta: agent_pb2.RuntimeDelta) -> None:
    change = delta.WhichOneof('payload')
    if change in _STREAMING:
      self._streaming = True
    elif change == 'stream_aborted':
      self._streaming = False
    elif change == 'turn_started':
      self._turn_open = True
      self._turn_id = delta.turn_id if delta.HasField('turn_id') else None
      self._error_shown = False
      self._interrupt_shown = False
      self._end = None
      self._superseded = set()
    elif change == 'turn_error':
      self._error_shown = True
    elif change == 'turn_interrupted':
      self._interrupt_shown = True
    elif change == 'turn_completed':
      self._ended_turn = self._turn_id
      if self._turn_id is not None:
        self._ended_turns.add(self._turn_id)
      self._turn_open = False
      self._turn_id = None
      self._end = None
      self._records_since_end = False
      self._streaming = False

  def _record_continuation(
    self, response: agent_pb2.SessionMessage
  ) -> list[agent_pb2.SessionMessage]:
    record = response.agent_message
    if (
      record.HasField('sequence')
      and self.sequence is not None
      and record.sequence <= self.sequence
    ):
      return []
    if record.WhichOneof('payload') == 'turn_end':
      return self._turn_end_continuation(response)
    turn_id = _turn_of(record)
    if turn_id is None:
      if not self._durable_ends and self._turn_open and self._answered():
        self._superseded = {'turn_completed'}
        return [_turn_completed(), response]
      return [response]
    if not self._turn_open:
      if turn_id == self._ended_turn:
        return [response]
      return [_turn_started(turn_id), response]
    if self._turn_id is None:
      self._turn_id = turn_id
      return [response]
    if self._turn_id != turn_id:
      return [_turn_completed(), _turn_started(turn_id), response]
    return [response]

  def _turn_end_continuation(
    self, response: agent_pb2.SessionMessage
  ) -> list[agent_pb2.SessionMessage]:
    record = response.agent_message
    turn_id = _turn_of(record)
    if not self._turn_open:
      if turn_id == self._ended_turn and not self._records_since_end:
        return [response]
      return [
        _turn_started(turn_id),
        *self._ending(record.turn_end, response, shown=(False, False)),
      ]
    if self._turn_id is not None and turn_id not in (None, self._turn_id):
      return [
        _turn_completed(),
        _turn_started(turn_id),
        *self._ending(record.turn_end, response, shown=(False, False)),
      ]
    return self._ending(
      record.turn_end,
      response,
      shown=(self._error_shown, self._interrupt_shown),
    )

  def _ending(
    self,
    end: agent_pb2.TurnEndRecord,
    record: agent_pb2.SessionMessage | None,
    *,
    shown: tuple[bool, bool],
  ) -> list[agent_pb2.SessionMessage]:
    error_shown, interrupt_shown = shown
    failed = end.outcome == agent_pb2.TURN_OUTCOME_FAILED
    interrupted = end.outcome == agent_pb2.TURN_OUTCOME_INTERRUPTED
    ending = [_turn_error(end.error)] if failed and not error_shown else []
    ending.extend([record] if record is not None else [])
    ending.extend(
      [_turn_interrupted()] if interrupted and not interrupt_shown else []
    )
    ending.append(_turn_completed())
    self._superseded = {'turn_completed'}
    if failed:
      self._superseded.add('turn_error')
    if interrupted:
      self._superseded.add('turn_interrupted')
    return ending

  def _delta_continuation(
    self, response: agent_pb2.SessionMessage
  ) -> list[agent_pb2.SessionMessage]:
    delta = response.runtime_delta
    change = delta.WhichOneof('payload')
    if change in _ENDINGS:
      return self._ending_continuation(response, change)
    if change == 'turn_started':
      return self._start_continuation(response)
    return [response]

  def _ending_continuation(
    self, response: agent_pb2.SessionMessage, change: str
  ) -> list[agent_pb2.SessionMessage]:
    if self._durable_ends:
      if self._turn_open and self._turn_id is None:
        return [response]
      return []
    if self._turn_open or change not in self._superseded:
      return [response]
    self._superseded.discard(change)
    if change == 'turn_completed':
      self._superseded.clear()
    return []

  def _start_continuation(
    self, response: agent_pb2.SessionMessage
  ) -> list[agent_pb2.SessionMessage]:
    delta = response.runtime_delta
    turn_id = delta.turn_id if delta.HasField('turn_id') else None
    if self._durable_ends and self._stale_start(turn_id):
      return []
    if not self._turn_open:
      return [response]
    if self._turn_id is None or turn_id is None or turn_id == self._turn_id:
      if self._turn_id is None:
        self._turn_id = turn_id
      return []
    return [_turn_completed(), response]

  def _stale_start(self, turn_id: str | None) -> bool:
    if turn_id is None or turn_id not in self._ended_turns:
      return False
    return turn_id != self._ended_turn or not self._records_since_end

  def _settled(self, pending: Iterable[agent_pb2.ClientMessage]) -> bool:
    return self._answered() and not any(
      _awaits_turn(request) for request in pending
    )

  def _answered(self) -> bool:
    record = self._last_chat
    if record is None or record.chat.WhichOneof('message') != 'assistant':
      return False
    if (
      self._turn_id is not None
      and record.HasField('turn_id')
      and record.turn_id != self._turn_id
    ):
      return False
    return not (record.chat.assistant.tool_calls or record.chat.truncated)
