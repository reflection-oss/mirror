"""Conversation entries on screen and the history they paint."""

from __future__ import annotations

import dataclasses
import functools
import io
import itertools
from collections.abc import Callable, Iterable, Iterator

from mirror.protocol import agent_pb2
from rich import console as _console

from mirror.cli import actions as _actions
from mirror.cli import rendering as _rendering
from mirror.cli import tool_groups as _tool_groups


@dataclasses.dataclass(frozen=True)
class Block:
  """One stable presentation block, or a revised final tool group."""

  key: str
  revision: tuple[int, int]
  renderable: _console.RenderableType
  is_user: bool = False
  position: int = 0
  welcome: bool = False


@dataclasses.dataclass
class _Entry:
  source: Transcript
  record: agent_pb2.AgentMessage

  @functools.cached_property
  def renderable(self) -> _console.RenderableType | None:
    return self.source.project(self.record, {})

  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    rendered = self.renderable
    if rendered is not None:
      yield from console.render(rendered, options)


def index_tool_calls(
  message: agent_pb2.AssistantMessage, hints: _rendering.CallHints
) -> None:
  """Keep each call's name and arguments for the result that will cite it."""
  for call in message.tool_calls:
    if call.type != 'function':
      continue
    name = call.function.name
    hints[call.id] = (
      name,
      call.function.arguments,
      _rendering.activity_kind(name),
    )


class Transcript:
  """Own the conversation entries and render them to a line budget."""

  def __init__(
    self,
    *,
    view_mode: Callable[[], _actions.ViewMode],
    skill_names: frozenset[str],
    get_rewind_entry_id: Callable[[], str | None],
  ) -> None:
    self._view_mode = view_mode
    self._skill_names = skill_names
    self._get_rewind_entry_id = get_rewind_entry_id
    self._entries: list[agent_pb2.AgentMessage] = []
    self._replayed: list[agent_pb2.AgentMessage] = []
    self._decorations: list[tuple[int, _console.RenderableType, bool]] = []
    self._group_breaks: set[int] = set()
    self._blocks: list[Block] = []
    self._group_continues = False
    self._projected = 0
    self._projected_decorations = 0
    self._presentation: tuple[_actions.ViewMode, int | None] | None = None
    self._presentation_revision = 0
    self._call_hints: _rendering.CallHints = {}
    self._block_snapshot: tuple[Block, ...] = ()
    self._record_ids: dict[str, agent_pb2.AgentMessage] = {}
    self._replay_ids: dict[str, agent_pb2.AgentMessage] = {}

  @property
  def has_conversation(self) -> bool:
    """Whether live or replayed entries contain conversation, not setup notices."""
    return any(
      (
        entry.HasField('chat')
        and entry.chat.WhichOneof('message') in ('user', 'assistant', 'tool')
      )
      or entry.HasField('task_result')
      or (
        entry.HasField('context_window')
        and entry.context_window.HasField('compaction')
      )
      for entry in itertools.chain(self._entries, self._replayed)
    )

  def welcome_notices(self) -> tuple[_console.RenderableType, ...]:
    """Notices eligible for display below the welcome banner."""
    return tuple(block for _, block, welcome in self._decorations if welcome)

  def break_group(self) -> None:
    """Retain a presentation boundary such as an approval or turn ending."""
    self._group_breaks.add(len(self._entries))

  def _compact(self) -> bool:
    return self._view_mode() is _actions.ViewMode.COMPACT

  def _extra_compact(self) -> bool:
    return self._view_mode() is _actions.ViewMode.EXTRA_COMPACT

  def append(self, entry: agent_pb2.AgentMessage) -> bool:
    """Append one durable entry to the active transcript."""
    if not self._remember(entry, self._record_ids):
      return False
    self._entries.append(entry)
    return True

  def extend(self, entries: Iterable[agent_pb2.AgentMessage]) -> None:
    """Append durable entries to the active transcript."""
    for entry in entries:
      self.append(entry)

  def append_replayed(self, entry: agent_pb2.AgentMessage) -> None:
    """Append one entry to the pending replay buffer."""
    if self._remember(entry, self._replay_ids):
      self._replayed.append(entry)

  def _remember(
    self,
    entry: agent_pb2.AgentMessage,
    identities: dict[str, agent_pb2.AgentMessage],
  ) -> bool:
    if not entry.id:
      return True
    previous = identities.get(entry.id)
    if previous is not None:
      if previous != entry:
        raise ValueError(f'Conflicting durable message {entry.id!r}')
      return False
    identities[entry.id] = entry
    return True

  def contains(
    self, entry: agent_pb2.AgentMessage, *, replayed: bool = False
  ) -> bool:
    """Recognize a repeated durable record without accepting conflicting content."""
    if not entry.id:
      return False
    identities = self._replay_ids if replayed else self._record_ids
    previous = identities.get(entry.id)
    if previous is not None and previous != entry:
      raise ValueError(f'Conflicting durable message {entry.id!r}')
    return previous is not None

  def clear_replayed(self) -> None:
    """Discard entries buffered for a stale replay."""
    self._replayed.clear()
    self._replay_ids.clear()

  def replayed(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """The complete trace buffered for the pending session hydration."""
    return tuple(self._replayed)

  def entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """Return the current durable transcript."""
    return tuple(self._entries)

  def adopt_replayed(self) -> None:
    """Replace the active transcript with the completed replay."""
    self._entries, self._replayed = self._replayed, []
    self._record_ids, self._replay_ids = self._replay_ids, {}
    self._group_breaks.clear()
    self._presentation = None

  def clear_history(self) -> None:
    """Discard all active transcript entries and decorations."""
    self._entries.clear()
    self._record_ids.clear()
    self._decorations.clear()
    self._group_breaks.clear()
    self._presentation = None

  def clear_decorations(self) -> None:
    """Discard UI-only transcript events."""
    self._decorations.clear()
    self._presentation = None

  def blocks(self) -> tuple[Block, ...]:
    """Project only new records; retain settled blocks until presentation changes."""
    presentation = (self._view_mode(), self.rewind_cut())
    changed = False
    if presentation != self._presentation:
      self._blocks.clear()
      self._group_continues = False
      self._call_hints.clear()
      self._projected = self._projected_decorations = 0
      self._presentation = presentation
      self._presentation_revision += 1
      changed = True
    stop = len(self._entries) if presentation[1] is None else presentation[1]
    while self._projected <= stop:
      position = self._projected
      while self._projected_decorations < len(self._decorations):
        at, renderable, welcome = self._decorations[self._projected_decorations]
        if at > position:
          break
        self._blocks.append(
          Block(
            f'decoration:{self._projected_decorations}',
            (self._presentation_revision, 0),
            renderable,
            position=at,
            welcome=welcome,
          )
        )
        self._group_continues = isinstance(renderable, _tool_groups.ToolGroup)
        self._projected_decorations += 1
        changed = True
      if position in self._group_breaks:
        self._group_continues = False
      if position == stop:
        break
      entry = self._entries[position]
      renderable = self.project(entry, self._call_hints, lazy=True)
      if renderable is not None:
        previous = self._blocks[-1] if self._blocks else None
        if (
          self._group_continues
          and isinstance(renderable, _tool_groups.ToolGroup)
          and previous is not None
          and isinstance(previous.renderable, _tool_groups.ToolGroup)
          and previous.renderable.key == renderable.key
        ):
          self._blocks[-1] = Block(
            previous.key,
            (previous.revision[0], previous.revision[1] + 1),
            previous.renderable.merge(renderable),
            position=position + 1,
          )
        else:
          self._blocks.append(
            Block(
              entry.id or f'entry:{position}',
              (self._presentation_revision, 0),
              renderable,
              position=position + 1,
              is_user=(
                entry.WhichOneof('payload') == 'chat'
                and entry.chat.WhichOneof('message') == 'user'
                and not entry.chat.display
              ),
            )
          )
        self._group_continues = isinstance(renderable, _tool_groups.ToolGroup)
      self._projected += 1
      changed = True
    if changed:
      unanswered = (
        ()
        if self._extra_compact()
        else tuple(
          Block(
            f'unanswered:{call_id}',
            (self._presentation_revision, 0),
            _rendering.unanswered_call(
              (name, _rendering.compact_args(arguments, tool_name=name), kind),
              compact=self._compact(),
            ),
          )
          for call_id, (name, arguments, kind) in self._call_hints.items()
        )
      )
      self._block_snapshot = (*self._blocks, *unanswered)
    return self._block_snapshot

  def _visible_entry(self, entry: agent_pb2.AgentMessage) -> bool:
    """Classify record visibility before either lazy or immediate projection."""
    match entry.WhichOneof('payload'):
      case 'log':
        return True
      case 'context_window':
        return entry.context_window.HasField('compaction')
      case 'task_result':
        return not self._extra_compact()
      case 'chat':
        match entry.chat.WhichOneof('message'):
          case 'user':
            return bool(entry.chat.user.content.strip())
          case 'tool':
            return not self._extra_compact()
          case 'assistant':
            message = entry.chat.assistant
            return bool(
              message.content.strip()
              or message.refusal.strip()
              or (
                message.reasoning_content.strip()
                and not self._compact()
                and not self._extra_compact()
              )
            )
          case _:
            return False
      case _:
        return False

  def project(
    self,
    entry: agent_pb2.AgentMessage,
    hints: _rendering.CallHints,
    *,
    elapsed: str | None = None,
    lazy: bool = False,
    include_reasoning: bool = True,
    include_content: bool = True,
  ) -> _console.RenderableType | None:
    """One renderable for an entry, or None for entries the UI does not show.

    Projection reaches for nothing but the entry and `hints`, the display hints
    a tool result borrows from its call, so the same entry draws the same block
    whether it arrives from the running turn or from stored history. The caller
    owns that map: a turn pairs against the calls it is running, and a drawing
    of the history pairs against a reading of its own.
    """
    if entry.HasField('chat') and entry.chat.HasField('assistant'):
      index_tool_calls(entry.chat.assistant, hints)
    if not self._visible_entry(entry):
      return None
    if lazy and not entry.chat.HasField('tool'):
      return _Entry(self, entry)
    kind = entry.WhichOneof('payload')
    if kind == 'log':
      return _rendering.log_text(entry.log.level, entry.log.text)
    if kind == 'context_window':
      return _rendering.compaction_text(
        entry.context_window.compaction,
        elapsed=elapsed,
        show_detail=not self._extra_compact(),
      )
    if kind == 'task_result':
      task_result = entry.task_result
      return _rendering.task_result_renderable(
        task_result.tool_name,
        task_result.task_id,
        task_result.content,
        display=task_result.display
        if task_result.HasField('display')
        else None,
        success=task_result.success,
        compact=self._compact(),
      )
    if kind != 'chat':
      return None  # Metadata and future record kinds have no rendering.
    message = entry.chat
    role = message.WhichOneof('message')
    if role == 'system':
      return None
    if role == 'tool':
      return self.resolve_tool_call(message, hints)
    if role == 'user':
      content = message.user.content
      if message.HasField('display') and message.display:
        return _rendering.skill_loaded_text(message.display)
      return _rendering.user_text(content, self._skill_names)
    if role == 'assistant':
      assistant = message.assistant
      return _rendering.assistant_turn_renderable(
        (
          assistant.reasoning_content
          if include_reasoning
          and not self._extra_compact()
          and assistant.HasField('reasoning_content')
          else None
        ),
        (
          assistant.content
          if include_content and assistant.HasField('content')
          else None
        ),
        assistant.refusal if assistant.HasField('refusal') else None,
        skill_names=self._skill_names,
        compact_reasoning=self._compact(),
      )
    return None

  def resolve_tool_call(
    self, message: agent_pb2.ChatRecord, hints: _rendering.CallHints
  ) -> _console.RenderableType | None:
    """Combine a tool result with its parked call into one block."""
    tool = message.tool
    hint = hints.pop(tool.tool_call_id, None)
    name = hint[0] if hint else (tool.name if tool.HasField('name') else '')
    arguments_json = hint[1] if hint else ''
    args = _rendering.compact_args(arguments_json, tool_name=name)
    activity_kind = hint[2] if hint else 'tool'
    display = message.display if message.HasField('display') else None
    if (
      message.success
      and (skill := _rendering.skill_display_for_tool(name, args)) is not None
    ):
      return _rendering.skill_loaded_text(skill)
    if self._compact():
      group = _tool_groups.result(
        _rendering.strip_ansi(name),
        _rendering.strip_ansi(args),
        success=message.success,
        operation=_rendering.tool_operation(arguments_json),
      )
      if group is not None:
        return group
    return _rendering.tool_result_renderable(
      name,
      args,
      tool.content,
      display,
      compact=self._compact(),
      activity_kind=activity_kind,
      error=not message.success,
    )

  def transcript_index(self, entry_id: str) -> int | None:
    """The transcript position of `entry_id`, or None when it is absent."""
    return next(
      (
        index
        for index, entry in enumerate(self._entries)
        if entry.id == entry_id
      ),
      None,
    )

  def rewind_cut(self) -> int | None:
    """Transcript index a staged rewind hides from, or None when unstaged.

    A missing target cuts at 0, so the preview shows the divider and nothing
    else.
    """
    entry_id = self._get_rewind_entry_id()
    if entry_id is None:
      return None
    index = self.transcript_index(entry_id)
    return index if index is not None else 0

  def add_decoration(
    self, renderable: _console.RenderableType, *, welcome: bool = False
  ) -> str:
    """Keep a UI-only event at its current position in this transcript."""
    key = f'decoration:{len(self._decorations)}'
    self._decorations.append((len(self._entries), renderable, welcome))
    return key

  def truncate(self, stop: int) -> None:
    """Discard entries and UI-only events from `stop` onward."""
    del self._entries[stop:]
    self._record_ids = {entry.id: entry for entry in self._entries if entry.id}
    self._group_breaks = {pos for pos in self._group_breaks if pos <= stop}
    self._decorations = [
      (position, block, welcome)
      for position, block, welcome in self._decorations
      if position <= stop
    ]
    self._presentation = None

  def newest_first(
    self, *, include_welcome_notices: bool = True
  ) -> Iterator[tuple[int, _console.RenderableType]]:
    """Read the same stable projection used by the owned conversation viewport."""
    stop = self.rewind_cut()
    if stop is None:
      stop = len(self._entries)
    for block in reversed(self.blocks()):
      if include_welcome_notices or not block.welcome:
        position = (
          stop if block.key.startswith('unanswered:') else block.position
        )
        yield position, block.renderable

  def render_history(
    self,
    header: list[_console.RenderableType],
    trailer: list[_console.RenderableType],
    *,
    console_width: int,
    scrollback_limit: int,
    cropped_history_hint: str | None = None,
    include_welcome_notices: bool = True,
    theme_name: str = 'dark',
  ) -> str:
    """The pinned rows, then the newest history the line budget allows.

    Entries are drawn from the end backwards and stop at the budget, so the
    ones the screen has no room for are never projected at all. One Console
    is reused across blocks to avoid per-block initialization.
    """
    width = max(1, console_width)
    buf = io.StringIO()
    console = _rendering.ansi_console(buf, width, theme_name=theme_name)

    def render(block: _console.RenderableType) -> str:
      buf.seek(0)
      buf.truncate()
      console.print(block, end='')
      return _rendering.sanitize_ansi(buf.getvalue()).rstrip()

    limit = scrollback_limit
    tail: list[str] = []
    lines = 0

    def fits(ansi: str) -> bool:
      """Take a block if the budget allows; the newest one always does."""
      nonlocal lines
      # +2: the block's own lines plus the blank row separating blocks.
      count = ansi.count('\n') + 2
      if limit and tail and lines + count > limit:
        return False
      tail.append(ansi)
      lines += count
      return True

    dropped = 0
    for left_behind, block in self.newest_first(
      include_welcome_notices=include_welcome_notices
    ):
      if not fits(render(block)):
        dropped += left_behind
        break

    tail.reverse()
    head = [render(row) for row in header]
    if dropped:
      head.append(
        render(_rendering.cropped_history_notice(dropped, cropped_history_hint))
      )
    below = [render(row) for row in trailer]
    blocks = [*head, *tail, *below]
    return '\n\n'.join(blocks) + '\n\n' if blocks else ''
