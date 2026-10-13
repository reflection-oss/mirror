"""Product events observed by one CLI session, without conversation content.

The worker stream already supplies lifecycle events and durable record identity;
this observer keeps analytics out of rendering and the pure state reducer.
"""

from __future__ import annotations

import time

from mirror.protocol import agent_pb2

from mirror.cli.statsig.events import client, event


class ConversationMetrics:
  """Count live activity once per attached conversation, excluding replay."""

  def __init__(self, *, remote: bool) -> None:
    self._workspace_kind = 'remote' if remote else 'local'
    self._conversation_id = ''
    self._turn_active = False
    self._seen_entries: set[str] = set()
    self._compaction_started: float | None = None
    self._tool_results: dict[str, bool | None] = {}

  def observe(
    self,
    message: agent_pb2.SessionMessage,
    *,
    conversation_id: str,
    replaying: bool,
  ) -> None:
    """Observe authoritative client events before their UI state changes."""
    if conversation_id != self._conversation_id:
      self._seen_entries.clear()
    if conversation_id != self._conversation_id or replaying:
      self._turn_active = False
      self._compaction_started = None
      self._tool_results.clear()
    self._conversation_id = conversation_id
    if message.HasField('agent_message'):
      entry = message.agent_message
      if not self._new_entry(entry) or replaying:
        return
      self._entry(entry)
    if replaying:
      return
    if message.WhichOneof('payload') != 'runtime_delta':
      return
    kind = message.runtime_delta.WhichOneof('payload')
    if kind == 'turn_error':
      self._compaction_started = None
    if kind == 'compaction_started' and self._compaction_started is None:
      self._compaction_started = time.monotonic()
    if kind == 'turn_started' and not self._turn_active:
      self._turn_active = True
      self._tool_results.clear()
      self._emit('mirror_conversation_turn_started')
    elif kind in {'turn_completed', 'turn_interrupted'}:
      if self._turn_active:
        self._emit('mirror_turn_tool_call_count', len(self._tool_results))
        self._emit(
          'mirror_turn_tool_success_count',
          sum(result is True for result in self._tool_results.values()),
        )
        self._emit(
          'mirror_turn_tool_failure_count',
          sum(result is False for result in self._tool_results.values()),
        )
        self._tool_results.clear()
      if kind == 'turn_interrupted' and self._turn_active:
        self._emit('mirror_agent_turn_interrupted')
      self._turn_active = False
      self._compaction_started = None

  def _new_entry(self, entry: agent_pb2.AgentMessage) -> bool:
    if entry.id:
      if entry.id in self._seen_entries:
        return False
      self._seen_entries.add(entry.id)
    return True

  def _entry(self, entry: agent_pb2.AgentMessage) -> None:
    if entry.HasField('context_window') and entry.context_window.HasField(
      'compaction'
    ):
      self._emit('mirror_compaction_completed')
      if self._compaction_started is not None:
        self._emit(
          'mirror_compaction_duration_seconds',
          time.monotonic() - self._compaction_started,
        )
      self._compaction_started = None
    if entry.WhichOneof('payload') != 'chat':
      return
    chat = entry.chat
    if chat.WhichOneof('message') == 'user' and not chat.HasField('display'):
      self._emit('mirror_conversation_user_message')
    elif chat.WhichOneof('message') == 'assistant':
      self._emit('mirror_conversation_assistant_message')
      if self._turn_active:
        for call in chat.assistant.tool_calls:
          self._tool_results.setdefault(call.id, None)
    elif chat.WhichOneof('message') == 'tool':
      call_id = chat.tool.tool_call_id
      if call_id in self._tool_results:
        self._tool_results[call_id] = chat.success

  def _emit(self, name: str, value: int | float = 1) -> None:
    client.get().log_event(
      event.Event(
        name,
        value=value,
        metadata={
          'conversation_id': self._conversation_id,
          'workspace_kind': self._workspace_kind,
        },
      )
    )
