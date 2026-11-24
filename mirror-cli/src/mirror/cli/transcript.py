"""Conversation entries on screen and the history they paint."""

from __future__ import annotations

import copy
import dataclasses
import functools
import io
import itertools
from collections.abc import Callable, Iterable, Iterator

from mirror.protocol import agent_pb2
from rich import console as _console

from mirror.cli import actions as _actions
from mirror.cli import rendering as _rendering
from mirror.cli import themes
from mirror.cli import tool_groups as _tool_groups


@dataclasses.dataclass(frozen=True)
class Block:
  """A read-only presentation snapshot; revisions replace rather than mutate it."""

  key: str
  revision: tuple[int, int]
  renderable: _console.RenderableType
  position: int = 0
  welcome: bool = False

  @functools.cached_property
  def source(self) -> _console.RenderableType | None:
    """Resolve lazy presentation without performing terminal layout."""
    return (
      self.renderable.renderable
      if isinstance(self.renderable, _Entry)
      else self.renderable
    )

  @functools.cached_property
  def parts(self) -> tuple[_console.RenderableType, ...]:
    """Expose stable Markdown parts for independent viewport layout."""
    if isinstance(self.renderable, (_Entry, _rendering.Answer)):
      return self.renderable.parts
    return (self.renderable,)

  @functools.cached_property
  def expanded(self) -> _console.RenderableType | None:
    """The full output a click swaps in, when the row collapses one."""
    rendered = self.renderable
    if isinstance(rendered, _Entry):
      rendered = rendered.renderable
    if isinstance(rendered, _rendering.Expandable):
      return rendered.expanded
    return None


@dataclasses.dataclass(frozen=True)
class _Entry:
  record: agent_pb2.AgentMessage
  view_mode: _actions.ViewMode
  skill_names: frozenset[str]

  @functools.cached_property
  def renderable(self) -> _console.RenderableType | None:
    return _render_entry(self.record, self.view_mode, self.skill_names)

  @functools.cached_property
  def parts(self) -> tuple[_console.RenderableType, ...]:
    rendered = self.renderable
    if isinstance(rendered, _rendering.Answer):
      return rendered.parts
    if isinstance(rendered, _console.Group) and self.record.chat.HasField(
      'assistant'
    ):
      answer = rendered.renderables[-1]
      if isinstance(answer, _rendering.Answer):
        return (rendered.renderables[0], *answer.parts)
    return (self,)

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


def _render_entry(
  entry: agent_pb2.AgentMessage,
  view_mode: _actions.ViewMode,
  skill_names: frozenset[str],
  *,
  elapsed: str | None = None,
  include_reasoning: bool = True,
  include_content: bool = True,
) -> _console.RenderableType | None:
  """Project a record using its captured presentation settings."""
  compact = view_mode is _actions.ViewMode.DEFAULT
  focus = view_mode is _actions.ViewMode.FOCUS
  kind = entry.WhichOneof('payload')
  if kind == 'log':
    return _rendering.log_text(entry.log.level, entry.log.text)
  if kind == 'fork':
    return _rendering.context_display_text(
      f'Forked from {entry.fork.parent_session_id}'
    )
  if kind == 'context_window':
    return _rendering.compaction_text(
      entry.context_window.compaction,
      elapsed=elapsed,
      show_detail=not focus,
    )
  if kind == 'task_result':
    task_result = entry.task_result
    if compact and task_result.tool_name == 'bash':
      return _rendering.bash_result(
        f'background task {task_result.task_id[:8]}',
        task_result.content,
        task_result.display if task_result.HasField('display') else None,
        error=not task_result.success,
        exit_code=_exit_code(entry),
        expandable=True,
      )
    return _rendering.task_result_renderable(
      task_result.tool_name,
      task_result.task_id,
      task_result.content,
      display=task_result.display if task_result.HasField('display') else None,
      success=task_result.success,
      compact=compact,
    )
  if kind != 'chat':
    return None
  message = entry.chat
  role = message.WhichOneof('message')
  if role == 'user':
    if message.HasField('display') and message.display:
      return _rendering.context_display_text(message.display)
    return _rendering.user_text(message.user, skill_names)
  if role == 'assistant':
    assistant = message.assistant
    return _rendering.assistant_turn_renderable(
      (
        assistant.reasoning_content
        if include_reasoning
        and view_mode.shows_reasoning
        and assistant.HasField('reasoning_content')
        else None
      ),
      assistant.content
      if include_content and assistant.HasField('content')
      else None,
      assistant.refusal if assistant.HasField('refusal') else None,
      skill_names=skill_names,
    )
  return None


def _exit_code(entry: agent_pb2.AgentMessage) -> int | None:
  metadata = entry.metadata
  return metadata.exit_code if metadata.HasField('exit_code') else None


class Transcript:
  """Own read-only record snapshots; replace presentation blocks, never edit them."""

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
    self._active_tools: tuple[
      tuple[str, ...], _tool_groups.ToolGroup | None
    ] = ((), None)
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
    return self._view_mode() is _actions.ViewMode.DEFAULT

  def _focus(self) -> bool:
    return self._view_mode() is _actions.ViewMode.FOCUS

  def append(self, entry: agent_pb2.AgentMessage) -> bool:
    """Append one durable entry to the active transcript."""
    snapshot = self._remember(entry, self._record_ids)
    if snapshot is None:
      return False
    self._entries.append(snapshot)
    return True

  def extend(self, entries: Iterable[agent_pb2.AgentMessage]) -> None:
    """Append durable entries to the active transcript."""
    for entry in entries:
      self.append(entry)

  def append_replayed(self, entry: agent_pb2.AgentMessage) -> None:
    """Append one entry to the pending replay buffer."""
    snapshot = self._remember(entry, self._replay_ids)
    if snapshot is not None:
      self._replayed.append(snapshot)

  def _remember(
    self,
    entry: agent_pb2.AgentMessage,
    identities: dict[str, agent_pb2.AgentMessage],
  ) -> agent_pb2.AgentMessage | None:
    previous = identities.get(entry.id) if entry.id else None
    if previous is not None:
      if previous != entry:
        raise ValueError(f'Conflicting durable message {entry.id!r}')
      return None
    snapshot = copy.deepcopy(entry)
    if snapshot.id:
      identities[snapshot.id] = snapshot
    return snapshot

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
    """Detached copies of the trace buffered for pending session hydration."""
    return tuple(copy.deepcopy(entry) for entry in self._replayed)

  def entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """Return detached copies so callers cannot mutate retained records."""
    return tuple(copy.deepcopy(entry) for entry in self._entries)

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

  def blocks(
    self,
    *,
    active_call_ids: tuple[str, ...] = (),
    active_group: _tool_groups.ToolGroup | None = None,
  ) -> tuple[Block, ...]:
    """Project records, omitting tools already displayed in the live pane."""
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
            )
          )
        self._group_continues = isinstance(renderable, _tool_groups.ToolGroup)
      self._projected += 1
      changed = True
    active_tools = (active_call_ids, active_group)
    if changed or active_tools != self._active_tools:
      self._active_tools = active_tools
      settled = self._blocks
      if (
        active_group is not None
        and settled
        and settled[-1].renderable == active_group
      ):
        settled = settled[:-1]
      unanswered = (
        ()
        if self._focus()
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
          if call_id not in active_call_ids
        )
      )
      self._block_snapshot = (*settled, *unanswered)
    return self._block_snapshot

  def _visible_entry(self, entry: agent_pb2.AgentMessage) -> bool:
    """Classify record visibility before either lazy or immediate projection."""
    match entry.WhichOneof('payload'):
      case 'log' | 'fork':
        return True
      case 'context_window':
        return entry.context_window.HasField('compaction')
      case 'task_result':
        return not self._focus()
      case 'chat':
        match entry.chat.WhichOneof('message'):
          case 'user':
            return bool(entry.chat.user.content.strip())
          case 'tool':
            return not self._focus()
          case 'assistant':
            message = entry.chat.assistant
            return bool(
              message.content.strip()
              or message.refusal.strip()
              or (
                message.reasoning_content.strip()
                and self._view_mode().shows_reasoning
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
    """Project a record; lazy rendering captures the current presentation mode."""
    if entry.HasField('chat') and entry.chat.HasField('assistant'):
      index_tool_calls(entry.chat.assistant, hints)
    if not self._visible_entry(entry):
      return None
    if entry.chat.HasField('tool'):
      return self.resolve_tool_call(
        entry.chat, hints, exit_code=_exit_code(entry)
      )
    if lazy and not entry.chat.HasField('user'):
      return _Entry(entry, self._view_mode(), self._skill_names)
    return _render_entry(
      entry,
      self._view_mode(),
      self._skill_names,
      elapsed=elapsed,
      include_reasoning=include_reasoning,
      include_content=include_content,
    )

  def resolve_tool_call(
    self,
    message: agent_pb2.ChatRecord,
    hints: _rendering.CallHints,
    *,
    exit_code: int | None = None,
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
    if self._compact() and name == 'bash':
      return _rendering.bash_result(
        args,
        tool.content,
        display,
        error=not message.success,
        exit_code=exit_code,
        expandable=True,
      )
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
    theme_name: str | themes.Theme = 'dark',
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
