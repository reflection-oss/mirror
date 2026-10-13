"""Live stream pane: accumulate partial deltas and render a bounded preview."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Collection, Sequence

from mirror.protocol import agent_pb2
from prompt_toolkit import data_structures as _data_structures
from prompt_toolkit.formatted_text import ansi as _ansi
from rich import console as _console
from rich import text as _text

from mirror.cli import actions as _actions
from mirror.cli import rendering as _rendering
from mirror.cli import state as _state
from mirror.cli import terminal as _terminal_mod
from mirror.cli import themes

_STREAMED_ARGUMENT_PREFIX_LIMIT = 4096
# Explains why only the latest portion remains visible in a full live pane.
_STREAM_OVERFLOW_MESSAGE = 'Earlier output temporarily hidden'
# The blank row the live stream pane keeps between its text and the rows below.
STREAM_PANE_MARGIN = 1


def render_stream_preview(
  preview: _state.StreamPreview,
  *,
  width: int,
  max_lines: int,
  skill_names: Collection[str] = (),
  compact_reasoning: bool = False,
  hide_reasoning: bool = False,
  theme_name: str = 'dark',
) -> tuple[_ansi.ANSI, _data_structures.Point]:
  """Render and truncate a live stream for a fixed viewport."""
  width = max(1, width)
  max_lines = max(1, max_lines)
  renderable = _rendering.assistant_turn_renderable(
    None if hide_reasoning else (preview.reasoning or None),
    preview.content or None,
    preview.refusal or None,
    # prompt_toolkit's ANSI parser does not understand OSC-8 hyperlinks.
    hyperlinks=False,
    skill_names=skill_names,
    compact_reasoning=compact_reasoning and not hide_reasoning,
  )
  rendered_ansi = (
    _rendering.renderable_to_ansi(
      renderable, width, theme_name=theme_name
    ).rstrip()
    if renderable is not None
    else ''
  )
  rendered_lines = rendered_ansi.splitlines(keepends=True)
  if len(rendered_lines) > max_lines:
    marker = _text.Text(
      f'{_STREAM_OVERFLOW_MESSAGE.center(width)}\n{"…".center(width)}\n',
      style='ui.info',
    )
    marker_lines = _rendering.renderable_to_ansi(
      marker, width, theme_name=theme_name
    ).splitlines(keepends=True)
    tail_lines = max(1, max_lines - len(marker_lines))
    rendered_ansi = ''.join([
      *marker_lines[: max_lines - tail_lines],
      *rendered_lines[-tail_lines:],
    ])

  ansi = _ansi.ANSI(rendered_ansi)
  plain = ''.join(
    fragment[1]
    for fragment in ansi.__pt_formatted_text__()
    if fragment[0] != '[ZeroWidthEscape]'
  )
  lines = plain.split('\n')
  return ansi, _data_structures.Point(x=len(lines[-1]), y=len(lines) - 1)


class StreamPane:
  """Accumulate partial deltas and render a viewport-bounded live preview."""

  def __init__(
    self,
    terminal: _terminal_mod.TerminalOutput,
    *,
    get_state: Callable[[], _state.State],
    dispatch: Callable[[Sequence[_actions.Action]], object],
    skill_names: frozenset[str],
    get_columns: Callable[[], int],
    get_max_lines: Callable[[], int],
  ) -> None:
    self._terminal = terminal
    self._get_state = get_state
    self._dispatch = dispatch
    self._skill_names = skill_names
    self._get_columns = get_columns
    self._get_max_lines = get_max_lines
    self._emit_task: asyncio.Task[None] | None = None
    self._high_water = 0
    # Rich rendering is cached because prompt_toolkit requests text and cursor
    # position independently during each redraw.
    self._render_key: tuple[str, str, str, int, int, bool, bool, str] | None = (
      None
    )
    self._ansi_cache = _ansi.ANSI('')
    self._cursor_cache = _data_structures.Point(x=0, y=0)

  def _compact(self) -> bool:
    return self._get_state().status_bar.view_mode is _actions.ViewMode.COMPACT

  def _extra_compact(self) -> bool:
    return (
      self._get_state().status_bar.view_mode is _actions.ViewMode.EXTRA_COMPACT
    )

  @property
  def preview(self) -> _state.StreamPreview | None:
    """The immutable active stream preview."""
    return self._get_state().stream.preview

  def reset_height(self) -> None:
    """Release rows reserved by previously visible stream text."""
    self._high_water = 0

  def on_partial(self, delta: agent_pb2.RuntimeDelta) -> None:
    """Accumulate a stream fragment and refresh its live preview."""
    kind = delta.WhichOneof('payload')
    if kind == 'stream_tool_call':
      self.on_tool_call(delta.stream_tool_call)
      return
    preview = self.preview
    if preview is None:
      self._emit_task = None
    preview = preview or _state.StreamPreview()
    content = delta.stream_content.content if kind == 'stream_content' else ''
    reasoning = (
      delta.stream_reasoning.reasoning if kind == 'stream_reasoning' else ''
    )
    refusal = delta.stream_refusal.refusal if kind == 'stream_refusal' else ''
    actions: list[_actions.Action] = [
      _actions.ObserveStreamText(
        content=content,
        reasoning=reasoning,
        refusal=refusal,
      )
    ]
    # Streamed text can precede silent reasoning within the same response.
    if reasoning or (
      (content or refusal) and not self._get_state().stream.pending
    ):
      actions.append(_actions.SetStatus('thinking…'))
    committed_reasoning = (
      preview.reasoning
      if (content or refusal)
      and preview.reasoning
      and preview.committed_reasoning is None
      else None
    )
    if committed_reasoning is not None:
      actions.append(_actions.CommitStreamReasoning())
    self._dispatch(actions)
    if committed_reasoning is not None:
      self._high_water = 0
      self._emit_reasoning(committed_reasoning)

  def _emit_reasoning(self, reasoning: str) -> None:
    if self._extra_compact() or self._compact():
      return
    renderable = _rendering.reasoning_text(reasoning)
    self._emit_task = self._terminal.create_task(
      self._terminal.emit([renderable])
    )

  def _emit_content(self, content: str) -> None:
    renderable = _rendering.assistant_turn_renderable(
      None,
      content,
      None,
      skill_names=self._skill_names,
      compact_reasoning=self._compact(),
    )
    if renderable is None:
      return
    if self._emit_task is not None:
      self._emit_task = self._terminal.create_task(
        self._terminal.emit_after(self._emit_task, [renderable])
      )
    else:
      self._emit_task = self._terminal.create_task(
        self._terminal.emit([renderable])
      )

  async def wait_for_emits(self) -> None:
    """Drain queued stream writes before replacing their transcript on screen."""
    while self._emit_task is not None and not self._emit_task.done():
      await self._emit_task

  def committed_blocks(self) -> list[_console.RenderableType]:
    """Return stream prefixes already moved from the live pane to scrollback."""
    preview = self.preview
    if preview is None:
      return []
    blocks: list[_console.RenderableType] = []
    if preview.committed_reasoning is not None and not (
      self._extra_compact() or self._compact()
    ):
      blocks.append(_rendering.reasoning_text(preview.committed_reasoning))
    if preview.committed_content is not None:
      content = _rendering.assistant_turn_renderable(
        None,
        preview.committed_content,
        None,
        skill_names=self._skill_names,
        compact_reasoning=self._compact(),
      )
      if content is not None:
        blocks.append(content)
    return blocks

  def on_tool_call(self, tool_call: agent_pb2.StreamToolCall) -> None:
    """Refresh a pending row as a tool call assembles during the stream."""
    stream = self._get_state().stream
    preview = stream.preview
    reasoning = (
      preview.reasoning
      if preview is not None
      and preview.reasoning
      and preview.committed_reasoning is None
      else None
    )
    content = (
      preview.content
      if preview is not None
      and preview.content
      and preview.committed_content is None
      else None
    )
    index = tool_call.index
    existing = next(
      (call for call in stream.tool_calls if call.index == index),
      _actions.StreamToolCall(index),
    )
    name = existing.name + tool_call.name
    remaining = _STREAMED_ARGUMENT_PREFIX_LIMIT - len(existing.arguments)
    arguments = existing.arguments
    if remaining > 0:
      arguments += tool_call.arguments[:remaining]
    previous = next(
      (call for call in stream.pending if call.index == index), None
    )
    display_name = name or (previous.name if previous else '…')
    args = _rendering.compact_args(arguments, tool_name=name)
    if not args and previous is not None:
      args = previous.args
    pending = (
      _actions.PendingToolCall(
        index,
        display_name,
        args,
        _rendering.activity_kind(name),
      )
      if args
      else None
    )
    actions: list[_actions.Action] = []
    if reasoning is not None:
      actions.append(_actions.CommitStreamReasoning())
    if content is not None:
      actions.append(_actions.CommitStreamContent())
    actions.append(
      _actions.ObserveStreamToolCall(
        _actions.StreamToolCall(index, name, arguments), pending
      )
    )
    if pending is not None:
      actions.append(_actions.SetStatus('running tool…'))
    self._dispatch(actions)
    if reasoning is not None or content is not None:
      self._high_water = 0
    if reasoning is not None:
      self._emit_reasoning(reasoning)
    if content is not None:
      self._emit_content(content)

  def clear_pending(self) -> None:
    """Drop pending rows created while tool calls were streaming."""
    if self._get_state().stream.pending:
      self._dispatch((_actions.ClearStreamPending(),))

  def close(self) -> _state.StreamPreview | None:
    """Clear and return the active preview."""
    preview = self.preview
    stream = self._get_state().stream
    if preview is None and not stream.pending and not stream.tool_calls:
      return None
    self._dispatch((_actions.CloseStream(),))
    self._high_water = 0
    self._render_key = None
    return preview

  def handoff(self, rendered: _console.RenderableType | None) -> None:
    """Atomically replace the live pane with durable scrollback output."""
    preview = self.close()
    if rendered is None:
      return
    if preview is not None and self._emit_task is not None:
      self._emit_task = self._terminal.create_task(
        self._terminal.emit_after(self._emit_task, [rendered])
      )
    else:
      self._emit_task = self._terminal.create_task(
        self._terminal.emit([rendered])
      )

  def abort(self) -> None:
    """Preserve generated text when a stream ends without a final message."""
    preview = self.preview
    rendered = (
      None
      if preview is None
      else _rendering.assistant_turn_renderable(
        None if self._extra_compact() else (preview.reasoning or None),
        preview.content or None,
        preview.refusal or None,
        skill_names=self._skill_names,
        compact_reasoning=self._compact(),
      )
    )
    self.handoff(rendered)

  @property
  def pane_height(self) -> int:
    """The whole live stream pane: its text plus one blank row, or nothing.

    One number decides both whether the pane is on screen and how tall it is,
    so the blank row under the stream cannot outlive the text above it.
    """
    rows = self.pane_rows
    return rows + STREAM_PANE_MARGIN if rows else 0

  @property
  def pane_rows(self) -> int:
    """The stream text's rows: grows with the stream, released on close."""
    if self.preview is not None:
      self._refresh_render()
    return min(self._high_water, self._get_max_lines())

  def _refresh_render(self) -> None:
    """Refresh the cached, viewport-bounded live rendering when needed."""
    width = max(1, self._get_columns())
    max_lines = self._get_max_lines()
    preview = self.preview or _state.StreamPreview()
    compact_reasoning = self._compact()
    hide_reasoning = self._extra_compact()
    render_key = (
      preview.reasoning,
      preview.content,
      preview.refusal,
      width,
      max_lines,
      compact_reasoning,
      hide_reasoning,
      themes.current(self._get_state().extensions),
    )
    if render_key == self._render_key:
      return
    self._render_key = render_key
    self._ansi_cache, self._cursor_cache = render_stream_preview(
      preview,
      width=width,
      max_lines=max_lines,
      skill_names=self._skill_names,
      compact_reasoning=compact_reasoning,
      hide_reasoning=hide_reasoning,
      theme_name=themes.current(self._get_state().extensions),
    )
    if self.preview is not None:
      # An emptied preview reserves nothing: its text moved to scrollback when
      # tool calls began, and the rows would stay blank above the prompt.
      rendered = (
        self._cursor_cache.y + 1 if self._ansi_cache.value.strip() else 0
      )
      self._high_water = max(self._high_water, rendered)

  def ansi(self) -> _ansi.ANSI:
    """Return the cached live stream text for prompt_toolkit."""
    self._refresh_render()
    return self._ansi_cache

  def cursor_position(self) -> _data_structures.Point:
    """Return the end position required by the formatted-text control."""
    self._refresh_render()
    return self._cursor_cache
