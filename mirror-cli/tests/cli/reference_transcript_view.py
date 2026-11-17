"""Python viewport reference retained for behavioral comparisons, never shipped."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import functools
import io
import itertools
import re
import sys
import time
from collections.abc import Callable, Generator, Sequence
from typing import override

from prompt_toolkit import formatted_text, mouse_events, utils
from prompt_toolkit.application import current
from prompt_toolkit.layout import controls
from rich import cells, color, console, styled, text

from mirror.cli import (
  actions,
  code_layout,
  hyperlinks,
  rendering,
  screen_selection,
  themes,
  transcript,
  workspace_application,
)


def _summarize(content: str, width: int) -> str:
  """Normalize and clip a prompt without splitting terminal graphemes."""
  content = ' '.join(content.split())
  summary = rendering.icon_prefix(rendering.PROMPT) + rendering.strip_ansi(
    content
  )
  used = end = 0
  for start, stop, size in cells.split_graphemes(summary)[0]:
    # Fit both renderers: their cell widths differ for some emoji sequences.
    used += max(size, utils.get_cwidth(summary[start:stop]))
    if used <= width - 1:
      end = stop
    if used > width:
      return summary[:end].rstrip() + '…'
  return summary


@dataclasses.dataclass(frozen=True)
class Position:
  block: str
  row: int
  part: int = 0


type RowPosition = tuple[str, tuple[int, int], int, int]

type _Row = tuple[formatted_text.OneStyleAndTextTuple, ...]
type _Rows = tuple[_Row, ...]
type _LayoutKey = tuple[
  str, tuple[int, int], int, int, str | themes.Theme, int, bool
]
type _LayoutIdentity = tuple[str, int, int, str | themes.Theme, int, bool]
type _Layout = tuple[tuple[int, int], _Rows, int]


_PREFETCH_IDLE = 0.1
"""Seconds without scrolling or painting before history is formatted ahead."""

_PREFETCH_SCREENS = 3
"""Viewport heights of history formatted ahead on each side of the viewport."""


def _presentation(blocks: Sequence[transcript.Block]) -> int | None:
  """The view mode and rewind cut that produced a settled snapshot."""
  return blocks[0].revision[0] if blocks else None


def _plain(fragments: _Row) -> str:
  return (
    formatted_text
    .fragment_list_to_text(list(fragments))
    .strip()
    .removeprefix(rendering.icon_prefix(rendering.ASSISTANT_ICON))
  )


def _layout_bytes(key: _LayoutKey, rows: _Rows) -> int:
  """Account for retained Python objects, not just their character counts."""
  size = sys.getsizeof(key) + sys.getsizeof(rows) + sys.getsizeof((rows, 0))
  size += sum(map(sys.getsizeof, key)) + sum(map(sys.getsizeof, key[1]))
  size += sys.getsizeof(0)
  strings: set[int] = set()
  for row in rows:
    size += sys.getsizeof(row)
    for fragment in row:
      size += sys.getsizeof(fragment)
      for value in fragment:
        if id(value) not in strings:
          strings.add(id(value))
          size += sys.getsizeof(value)
          if isinstance(value, functools.partial):
            size += sys.getsizeof(value.args) + sys.getsizeof(value.keywords)
            size += sys.getsizeof(value.func)
            for argument in value.args:
              if isinstance(argument, str) and id(argument) not in strings:
                strings.add(id(argument))
                size += sys.getsizeof(argument)
  return size


def _split_rows(
  fragments: formatted_text.StyleAndTextTuples, padding: int
) -> _Rows:
  """Coalesce spans without dropping their actions or retaining terminal fill."""
  while fragments and fragments[-1][0] == '':
    last = fragments.pop()
    if text := last[1].rstrip():
      fragments.append(('', text))
      break
  # ANSI decoding emits one fragment per character; split style runs instead.
  runs: formatted_text.StyleAndTextTuples = []
  for _, group in itertools.groupby(
    fragments, key=lambda fragment: (fragment[0], fragment[2:])
  ):
    parts = list(group)
    first = parts[0]
    text = ''.join(part[1] for part in parts)
    runs.append(
      (first[0], text, first[2]) if len(first) == 3 else (first[0], text)
    )
  rows = [tuple(line) for line in formatted_text.split_lines(runs)]
  while rows and not any(text for _, text, *_ in rows[-1]):
    rows.pop()
  return tuple(rows) + ((),) * padding


type _Handler = Callable[[mouse_events.MouseEvent], None]


def _toggle_header(row: _Row, expanded: bool, handler: _Handler) -> _Row:
  """Mark a clickable tool header with a trailing chevron.

  The chevron follows the header text rather than leading the row so tool
  rows keep the transcript's shared gutter alignment.
  """
  fragments = [(style, value) for style, value, *_ in row]
  while fragments and not fragments[-1][1].strip():
    fragments.pop()
  if fragments:
    style, value = fragments[-1]
    fragments[-1] = (style, value.rstrip())
  fragments.append(('class:ui.muted', ' ▾' if expanded else ' ▸'))
  return _clickable(tuple(fragments), handler)


def _clickable(row: _Row, handler: _Handler) -> _Row:
  """Route clicks on a row to `handler` without displacing existing actions."""
  return tuple(
    (f'{fragment[0]} class:mouse-pointer', fragment[1], handler)
    if len(fragment) == 2
    else fragment
    for fragment in row
  )


class TranscriptView(screen_selection.HoverControl):
  """Paint visible rows with bounded retained layouts and stable reading positions."""

  def __init__(
    self,
    source: Callable[[], tuple[transcript.Block, ...]],
    live: Callable[[], Sequence[console.RenderableType]],
    *,
    navigate: Callable[[actions.NavigateHistory], object] | None = None,
    pending: Callable[[], Sequence[transcript.Block]] = lambda: (),
    theme_name: Callable[[], str | themes.Theme] = lambda: 'dark',
    max_cache_bytes: int = 8 * 1024 * 1024,
  ) -> None:
    if max_cache_bytes < 0:
      raise ValueError('max_cache_bytes must be nonnegative')
    self._max_cache_bytes = max_cache_bytes
    self._code_layout = code_layout.CodeLayout()
    self._source = source
    self._pending = pending
    self._navigate = navigate or self.navigate
    self._live = live
    self._theme_name = theme_name
    self._anchor: Position | None = None
    self._anchor_index = 0
    self._pending_scroll: collections.deque[int] = collections.deque()
    self._wheel_rows = 0
    self._wheel_flush: asyncio.Handle | None = None
    # Heights outlive the byte-bounded row cache: navigation needs only these.
    self._heights: dict[_LayoutIdentity, tuple[tuple[int, int], int]] = {}
    self._estimates: dict[_LayoutIdentity, tuple[tuple[int, int], int]] = {}
    self._width = 0
    self._height = 0
    self._blocks: Sequence[transcript.Block] = ()
    self._settled_snapshot: tuple[transcript.Block, ...] = ()
    self._pending_snapshot: tuple[transcript.Block, ...] = ()
    self._live_source: Sequence[console.RenderableType] = ()
    self._live_snapshot: tuple[transcript.Block, ...] = ()
    self._live_revision = 0
    self._settling_line: tuple[str, int, int, bool] | None = None
    self._cache: collections.OrderedDict[_LayoutIdentity, _Layout] = (
      collections.OrderedDict()
    )
    self._expanded: set[str] = set()
    self._copy_ids = itertools.count()
    self._hovered_style: str | None = None
    self._hovered_action = False
    self._copied_style: str | None = None
    self._copy_label = 'Copied'
    self._copy_until = 0.0
    self._visible: _Rows = ()
    self._prepared_positions: tuple[RowPosition, ...] = ()
    self._visible_positions: tuple[RowPosition, ...] = ()
    self._cache_bytes = 0
    self._render_task: asyncio.Task[None] | None = None
    self._prefetch_task: asyncio.Task[None] | None = None
    self._prefetch_timer: asyncio.TimerHandle | None = None
    self._last_activity = 0.0
    self._painted_span: tuple[int, int, int, int] | None = None
    self._render_needed: tuple[transcript.Block, _LayoutKey] | None = None
    self._layout_ready = True
    self._generation = 0
    # Paint-local layouts survive eviction from the byte-bounded history cache.
    self._layouts: dict[_LayoutIdentity, _Layout] = {}
    self._used_layouts: set[_LayoutIdentity] = set()
    self._header_height = 0
    self._user_position: tuple[int, int, int | None] | None = None
    self._summary: tuple[str, int, str] | None = None

  def _remember_line(
    self,
    index: int,
    part: int,
    row: int,
    first: int | None = None,
    *,
    from_end: bool = False,
  ) -> None:
    """Locate reading content across rewrapping and preview settlement."""
    rows = self._rows(index, part)
    target_row = next(
      (i for i in range(row, len(rows)) if _plain(rows[i])), row
    )
    pattern = r'\s+'.join(
      re.escape(word) for word in _plain(rows[target_row]).split()
    )
    if not pattern:
      return
    preceding: list[str] = []
    if first is not None:
      for block_index in range(first, index + 1):
        stop = (
          part if block_index == index else len(self._blocks[block_index].parts)
        )
        for part_index in range(stop):
          preceding.extend(map(_plain, self._rows(block_index, part_index)))
    preceding.extend(map(_plain, rows[:target_row]))
    occurrence = (
      sum(1 for _ in re.finditer(pattern, '\n'.join(preceding)))
      if pattern
      else 0
    )
    if from_end:
      following = '\n'.join(map(_plain, rows[target_row + 1 :]))
      occurrence = -1 - sum(1 for _ in re.finditer(pattern, following))
    self._settling_line = (
      pattern,
      occurrence,
      target_row - row,
      first is not None,
    )

  def settle_live(self, identities: str | Sequence[str]) -> None:
    """Transfer the preview anchor without laying out the completed answer."""
    if self._anchor is None or not self._anchor.block.startswith('live:'):
      return
    with self._layout_pass():
      index, part, row = self._start()
      if not isinstance(identities, str):
        live_index = int(self._anchor.block.removeprefix('live:'))
        if live_index < len(identities):
          self._anchor = Position(identities[live_index], row, part)
        return
      first = len(self._settled_snapshot) + len(self._pending_snapshot)
      current_part = self._blocks[index].parts[part]
      final = next(
        (
          block for block in reversed(self._source()) if block.key == identities
        ),
        None,
      )
      # Unmapped answer roots cannot share an unambiguous source position.
      roots = sum(
        isinstance(block.renderable, rendering.Answer)
        and not block.renderable.continuation
        and block.renderable.source_offset == 0
        for block in self._live_snapshot
      )
      if (
        isinstance(current_part, rendering.Answer)
        and final is not None
        and roots == 1
      ):
        final_part = next(
          (
            i
            for i, value in enumerate(final.parts)
            if isinstance(value, rendering.Answer)
            and value.source_offset == current_part.source_offset
            and value.source.startswith(current_part.source)
          ),
          None,
        )
        if final_part is not None:
          self._remember_line(index, part, row)
          self._anchor = Position(identities, row, final_part)
          return
      self._remember_line(index, part, row, first)
      self._anchor = Position(identities, row)

  def follow(self) -> None:
    """Resume following the latest conversation output."""
    self._pending_scroll.clear()
    self._anchor = None
    self._settling_line = None

  def first(self) -> None:
    """Reach the oldest entry without formatting the intervening history."""
    self._pending_scroll.clear()
    if self._blocks:
      self._anchor_index = 0
      self._settling_line = None
      self._anchor = Position(self._blocks[0].key, 0)

  @property
  def reading(self) -> bool:
    """Whether incoming output should leave the reading anchor in place."""
    return self._anchor is not None

  def clear(self) -> None:
    """Release layouts when their conversation changes."""
    self.clear_hover()
    self._copied_style = None
    self._visible = ()
    self._prepared_positions = self._visible_positions = ()
    self._anchor = None
    self._anchor_index = 0
    self._pending_scroll.clear()
    self._heights.clear()
    self._estimates.clear()
    self._settling_line = None
    self._expanded.clear()
    self._cache.clear()
    self._cache_bytes = 0
    self._summary = None
    self._generation += 1
    self._layouts.clear()
    self._used_layouts.clear()
    self._blocks = self._settled_snapshot = self._pending_snapshot = ()
    self._live_snapshot = ()
    self._live_source = ()
    self._user_position = None

  async def close(self) -> None:
    """Release pending layouts when the conversation is closed."""
    self.clear()
    if self._prefetch_timer is not None:
      self._prefetch_timer.cancel()
      self._prefetch_timer = None
    tasks = [
      task
      for task in (self._render_task, self._prefetch_task)
      if task is not None
    ]
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

  @contextlib.contextmanager
  def _layout_pass(self) -> Generator[None]:
    """Retain only layouts still needed by the current paint or its retries."""
    self._layout_ready = True
    self._used_layouts.clear()
    try:
      yield
    finally:
      self._layouts = {
        identity: layout
        for identity, layout in self._layouts.items()
        if identity in self._used_layouts
      }
      self._used_layouts.clear()

  def _layout_key(self, index: int, part: int) -> _LayoutKey:
    block = self._blocks[index]
    return (
      block.key,
      block.revision,
      part,
      self._width,
      self._theme_name(),
      1,  # Match inline scrollback's empty row between blocks or Markdown parts.
      block.key in self._expanded,
    )

  def _row_count(self, index: int, part: int, *, exact: bool = True) -> int:
    """Return a part's height, estimating it when navigation only passes over it.

    Scrolling far must not lay out every part it skips: only the part the
    viewport lands in needs its rows. Skipped parts use their last measured
    height, or an estimate from their source, so a long burst resolves in one
    pass and only the destination is formatted.
    """
    if not self._pending_scroll:
      return len(self._rows(index, part))
    key = self._layout_key(index, part)
    identity = (key[0], *key[2:])
    known = self._known_rows(key)
    if known is not None:
      return known
    if not exact and self._defers_layout(index, part):
      return self._estimated_rows(index, part, key)
    height = len(self._rows(index, part))
    # Navigation heights must not pin rows the byte-bounded cache evicted.
    if self._layout_ready and self._render_needed is None:
      self._layouts.pop(identity, None)
    return height

  def _retained_rows(self, key: _LayoutKey) -> _Rows | None:
    identity = (key[0], *key[2:])
    cached = self._layouts.get(identity) or self._cache.get(identity)
    return cached[1] if cached is not None and cached[0] == key[1] else None

  def _known_rows(self, key: _LayoutKey) -> int | None:
    """A part's measured height for this revision, without formatting it."""
    rows = self._retained_rows(key)
    if rows is not None:
      return len(rows)
    known = self._heights.get((key[0], *key[2:]))
    if known is not None and known[0] == key[1]:
      return known[1]
    return None

  def _defers_layout(self, index: int, part: int) -> bool:
    """Whether laying out this part would wait for a background render."""
    app = current.get_app_or_none()
    return (
      app is not None
      and app.is_running
      and not isinstance(
        self._blocks[index].parts[part], (rendering.UserText, text.Text)
      )
    )

  def _estimated_rows(self, index: int, part: int, key: _LayoutKey) -> int:
    """Approximate a part's wrapped height without formatting it."""
    identity = (key[0], *key[2:])
    estimate = self._estimates.get(identity)
    if estimate is not None and estimate[0] == key[1]:
      return estimate[1]
    renderable = self._blocks[index].parts[part]
    padding = key[5]
    if isinstance(renderable, rendering.Answer):
      width = max(1, key[3] - rendering.GUTTER_WIDTH)
      height = padding + sum(
        max(1, -(-cells.cell_len(line) // width))
        for line in renderable.source.splitlines() or ('',)
      )
    else:
      # Tool rows usually collapse to a header; their real height replaces this.
      height = 1 + padding
    self._estimates[identity] = (key[1], height)
    return height

  def _rows(self, index: int, part: int) -> _Rows:
    block = self._blocks[index]
    renderable = block.parts[part]
    key = self._layout_key(index, part)
    padding = key[5]
    identity = (key[0], *key[2:])
    self._used_layouts.add(identity)
    cached = self._layouts.get(identity) or self._cache.get(identity)
    if cached is not None:
      self._layouts[identity] = cached
      if identity in self._cache:
        self._cache.move_to_end(identity)
      if cached[0] == block.revision:
        return cached[1]
    app = current.get_app_or_none()
    # Input geometry and plain status decorations must be available immediately.
    if (
      app is not None
      and app.is_running
      and not isinstance(renderable, (rendering.UserText, text.Text))
    ):
      self._render_needed = self._render_needed or (block, key)
      # Keep the last painted revision while formatting the newest snapshot.
      if cached is not None:
        return cached[1]
      self._render_needed = (block, key)
      self._layout_ready = False
      return ((),) * padding
    rows, size = self._render_block(block, key, next(self._copy_ids))
    self._remember_rows(key, rows, size)
    return rows

  def refresh(self) -> bool:
    """Format changed visible content before requesting its next paint."""
    app = current.get_app_or_none()
    if app is None or not app.is_running or not self._width or not self._height:
      return False
    with self._layout_pass():
      visible, header_height = self._prepare_content(self._width, self._height)
    if self._render_needed is not None:
      self._schedule_render()
    elif self._layout_ready and (
      visible != self._visible or header_height != self._header_height
    ):
      workspace_application.invalidate_transcript()
    return True

  def _schedule_render(self) -> None:
    # One render at a time; prioritize unseen parts over refreshing live text.
    if self._render_task is None and self._render_needed is not None:
      block, key = self._render_needed
      self._render_task = current.get_app().create_background_task(
        self._render(block, key, self._generation)
      )

  async def _render(
    self, block: transcript.Block, key: _LayoutKey, generation: int
  ) -> None:
    try:
      # Rich layout must not hold up input or RPC cancellation.
      for _ in range(max(1, self._height)):
        rows, size = await asyncio.to_thread(
          self._render_block, block, key, next(self._copy_ids)
        )
        if generation != self._generation:
          break
        self._remember_rows(key, rows, size)
        # Discover the next miss without waiting for a repaint of unchanged rows.
        with self._layout_pass():
          visible, header_height = self._prepare_content(
            self._width, self._height
          )
        if self._layout_ready and (
          visible != self._visible or header_height != self._header_height
        ):
          workspace_application.invalidate_transcript()
        if self._render_needed is None:
          break
        block, key = self._render_needed
    finally:
      self._render_task = None
      workspace_application.invalidate_transcript()

  def _schedule_prefetch(self) -> None:
    """Format nearby history once input and painting pause.

    Scrolling into history that was never formatted pays for Rich layout and
    cell conversion on the way; idle time is free to do the former ahead.
    """
    app = current.get_app_or_none()
    if (
      app is None
      or not app.is_running
      or app.loop is None
      or self._prefetch_timer is not None
      or self._prefetch_task is not None
    ):
      return
    self._prefetch_timer = app.loop.call_later(
      _PREFETCH_IDLE, self._start_prefetch
    )

  def _start_prefetch(self) -> None:
    self._prefetch_timer = None
    app = current.get_app_or_none()
    if app is None or not app.is_running or app.loop is None:
      return
    idle = time.monotonic() - self._last_activity
    if idle < _PREFETCH_IDLE:
      self._prefetch_timer = app.loop.call_later(
        _PREFETCH_IDLE - idle, self._start_prefetch
      )
      return
    # Visible misses have priority; their paint schedules the next attempt.
    if self._render_task is not None or self._render_needed is not None:
      return
    targets = self._prefetch_targets()
    if targets:
      self._prefetch_task = app.create_background_task(
        self._prefetch(targets, self._generation)
      )

  def _prefetch_targets(self) -> list[tuple[transcript.Block, _LayoutKey]]:
    """Unformatted parts within a few screens of the painted rows, nearest first."""
    span = self._painted_span
    # The transcript may have shrunk, for example by a rewind, since the paint.
    if span is None or not all(
      index < len(self._blocks) and part < len(self._blocks[index].parts)
      for index, part in (span[:2], span[2:])
    ):
      return []
    above = self._nearby(span[0], span[1], -1)
    below = self._nearby(span[2], span[3], 1)
    return [
      target
      for pair in itertools.zip_longest(above, below)
      for target in pair
      if target is not None
    ]

  def _nearby(
    self, index: int, part: int, direction: int
  ) -> list[tuple[transcript.Block, _LayoutKey]]:
    budget = _PREFETCH_SCREENS * self._height
    targets: list[tuple[transcript.Block, _LayoutKey]] = []
    while budget > 0:
      position = self._step(index, part, direction)
      if position is None:
        break
      index, part = position
      block = self._blocks[index]
      key = self._layout_key(index, part)
      height = self._known_rows(key)
      if height is None:
        height = self._estimated_rows(index, part, key)
      budget -= height
      if (
        not block.key.startswith('live:')
        and self._defers_layout(index, part)
        and self._retained_rows(key) is None
      ):
        targets.append((block, key))
    return targets

  def _step(
    self, index: int, part: int, direction: int
  ) -> tuple[int, int] | None:
    if direction < 0:
      if part > 0:
        return index, part - 1
      if index > 0:
        return index - 1, len(self._blocks[index - 1].parts) - 1
      return None
    if part + 1 < len(self._blocks[index].parts):
      return index, part + 1
    if index + 1 < len(self._blocks):
      return index + 1, 0
    return None

  async def _prefetch(
    self, targets: list[tuple[transcript.Block, _LayoutKey]], generation: int
  ) -> None:
    started = self._last_activity
    try:
      for block, key in targets:
        # Any input or paint means the user is active again: yield to it.
        if (
          generation != self._generation
          or self._last_activity != started
          or self._render_task is not None
        ):
          break
        rows, size = await asyncio.to_thread(
          self._render_block, block, key, next(self._copy_ids)
        )
        if generation != self._generation or key[3] != self._width:
          break
        self._remember_rows(key, rows, size, painting=False)
    finally:
      self._prefetch_task = None

  def _remember_rows(
    self, key: _LayoutKey, rows: _Rows, size: int, *, painting: bool = True
  ) -> None:
    identity = (key[0], *key[2:])
    self._heights[identity] = (key[1], len(rows))
    layout = (key[1], rows, size)
    # Only the current paint may hold layouts outside the byte bound.
    if painting:
      self._layouts[identity] = layout
    if size > self._max_cache_bytes:
      return
    # Retain only the latest revision per shape, including its stale-paint fallback.
    previous = self._cache.pop(identity, None)
    if previous is not None:
      self._cache_bytes -= previous[2]
    self._cache[identity] = layout
    self._cache_bytes += size
    while self._cache and (
      self._cache_bytes + sys.getsizeof(self._cache) > self._max_cache_bytes
    ):
      _, (_, _, removed_size) = self._cache.popitem(last=False)
      self._cache_bytes -= removed_size
    if not self._cache:
      self._cache.clear()

  def _render_block(
    self, block: transcript.Block, key: _LayoutKey, copy_id: int
  ) -> tuple[_Rows, int]:
    """Render captured geometry and copy IDs without mutating viewport state."""
    _, _, part, width, theme, padding, expanded = key
    content = block.parts[part]
    if block.expanded is not None:
      width = max(1, width - 2)  # Room for the trailing chevron.
      if expanded:
        content = block.expanded
    painter = rendering.ansi_console(
      io.StringIO(),
      width,
      theme_name=theme,
      interactive=not block.key.startswith('live:'),
    )
    if block.key.startswith('live:') and isinstance(content, rendering.Answer):
      retained = self._code_layout.render(
        content, painter, width, padding, theme == 'mono'
      )
      if retained is not None:
        return retained
    fragments: formatted_text.StyleAndTextTuples = []
    marker = ''
    handlers: dict[bool, Callable[[mouse_events.MouseEvent], None]] = {}
    # Direct rendering bypasses the console's default text style.
    for index, segment in enumerate(
      painter.render(styled.Styled(content, 'ui.text'))
    ):
      rich_style = segment.style
      value = segment.text
      metadata = rich_style.meta if rich_style else {}
      source = metadata.get('copy_code')
      if isinstance(source, str):
        marker = f' class:code-copy-{copy_id}-{index} '
        handlers = {
          False: functools.partial(self._code_event, source, marker, False),
          True: functools.partial(self._code_event, source, marker, True),
        }
      if rich_style and theme == 'mono':
        rich_style = rich_style.without_color
      decoded = hyperlinks.from_rich(
        value,
        rich_style,
        color.ColorSystem.STANDARD
        if theme == 'terminal'
        else color.ColorSystem.TRUECOLOR,
      )
      if marker and any(
        key in metadata
        for key in ('copy_code', 'code_block', 'code_action', 'code_hint')
      ):
        action = bool(metadata.get('code_action'))
        role = ' class:code-action' if action else ''
        if metadata.get('code_hint'):
          role = ' class:code-hint'
        handler = handlers[action]
        fragments.extend(
          (style + marker + role, text, handler) for style, text, *_ in decoded
        )
      else:
        fragments.extend(decoded)
    rows = _split_rows(fragments, padding)
    if block.expanded is not None and rows:
      # The whole row toggles: the header, the preview, and the expanded output.
      handler = functools.partial(self._tool_event, block.key)
      rows = (
        _toggle_header(rows[0], expanded, handler),
        *(_clickable(row, handler) for row in rows[1:]),
      )
    return rows, _layout_bytes(key, rows)

  def _tool_event(self, key: str, event: mouse_events.MouseEvent) -> None:
    if (
      event.event_type == mouse_events.MouseEventType.MOUSE_UP
      and event.button == mouse_events.MouseButton.LEFT
      and not event.modifiers
    ):
      if self._anchor is None:
        index, part, row = self._content_start()
        self._anchor = Position(self._blocks[index].key, row, part)
      if key in self._expanded:
        self._expanded.remove(key)
      else:
        self._expanded.add(key)
        # At the bottom edge, opening would otherwise show only the preview.
        if event.position.y >= self._height - 3:
          self._anchor = Position(key, 0)

  @override
  def clear_hover(self) -> None:
    if self._hovered_style is not None:
      self._hovered_style = None
      self._hovered_action = False
      workspace_application.invalidate_transcript()

  def _code_event(
    self, text: str, style: str, action: bool, event: mouse_events.MouseEvent
  ) -> None:
    if (
      event.event_type == mouse_events.MouseEventType.MOUSE_MOVE
      and event.button == mouse_events.MouseButton.NONE
      and not event.modifiers
    ):
      self._hovered_style = style
      self._hovered_action = action
    elif (
      action
      and self._hovered_style == style
      and event.event_type == mouse_events.MouseEventType.MOUSE_UP
      and event.button == mouse_events.MouseButton.LEFT
      and not event.modifiers
    ):
      try:
        workspace_application.copy_text(text)
      except ValueError:
        self._copy_label = 'Too large'
      else:
        self._copy_label = 'Copied'
      self._copied_style = style
      self._copy_until = time.monotonic() + 1.5
      app = current.get_app()
      if app.loop is not None:
        app.loop.call_later(1.5, app.invalidate)

  def _move(
    self, index: int, part: int, row: int, amount: int
  ) -> tuple[int, int, int]:
    if not self._layout_ready:
      return index, part, row
    row += amount
    while row < 0 and (index > 0 or part > 0):
      if part > 0:
        part -= 1
      else:
        index -= 1
        part = len(self._blocks[index].parts) - 1
      height = self._row_count(index, part, exact=False)
      # Parts within a screen of the destination are placed exactly, so moves
      # up to a screen, such as paging, never skip rows; only history jumped
      # further past may use an estimate.
      if row + height >= -self._height:
        height = self._row_count(index, part)
        if not self._layout_ready:
          return index, part, 0
      row += height
    while self._layout_ready:
      height = self._row_count(index, part, exact=False)
      if row < height + self._height:
        height = self._row_count(index, part)
        if row < height or not self._layout_ready:
          break
      if part + 1 < len(self._blocks[index].parts):
        row -= height
        part += 1
      elif index + 1 < len(self._blocks):
        row -= height
        index, part = index + 1, 0
      else:
        break
    return index, part, max(0, min(row, self._row_count(index, part) - 1))

  def _start(self) -> tuple[int, int, int]:
    if self._anchor is None:
      index = len(self._blocks) - 1
      part = len(self._blocks[index].parts) - 1
      return self._move(
        index, part, self._row_count(index, part) - 1, -self._height + 1
      )
    index = self._anchor_index
    if (
      index >= len(self._blocks)
      or self._blocks[index].key != self._anchor.block
    ):
      index = next(
        (
          i
          for i, block in enumerate(self._blocks)
          if block.key == self._anchor.block
        ),
        0,
      )
      self._anchor_index = index
    part = min(self._anchor.part, len(self._blocks[index].parts) - 1)
    if self._settling_line is not None:
      pattern, occurrence, offset, whole_record = self._settling_line
      parts = range(len(self._blocks[index].parts)) if whole_record else (part,)
      rows = [self._rows(index, i) for i in parts]
      if not self._layout_ready:
        return index, part, 0
      self._settling_line = None
      text = '\n'.join(map(_plain, itertools.chain.from_iterable(rows)))
      matches = list(re.finditer(pattern, text)) if pattern else []
      match = (
        matches[occurrence]
        if -len(matches) <= occurrence < len(matches)
        else None
      )
      if match is not None:
        row = max(0, text.count('\n', 0, match.start()) - offset)
        for part_index, chunk in zip(parts, rows, strict=True):
          if row < len(chunk):
            part = part_index
            self._anchor = Position(self._blocks[index].key, row, part)
            break
          row -= len(chunk)
    return index, part, min(self._anchor.row, self._row_count(index, part) - 1)

  def _owner(
    self, index: int, part: int, row: int
  ) -> rendering.UserText | None:
    """The user message whose box has crossed the viewport's top edge."""
    user_index = self._user_before(index)
    if user_index is None or (user_index == index and part == 0 and row == 0):
      return None
    block = self._blocks[user_index]
    if user_index == index and len(block.parts) == 1:
      # The summary stands in for text that has scrolled off; while the first
      # text row is still painted it would only repeat it.
      rows = self._rows(index, part)
      if self._layout_ready and not any(map(_plain, rows[:row])):
        return None
    message = block.renderable
    return message if isinstance(message, rendering.UserText) else None

  def _header(self, owner: rendering.UserText) -> _Rows:
    """Summarize the original message in one full-width shaded row."""
    cached = self._summary
    # Identity avoids hashing or comparing multi-megabyte prompts on redraw.
    if (
      cached is None
      or cached[0] is not owner.content
      or cached[1] != self._width
    ):
      summary = _summarize(owner.content, self._width)
      padding = self._width - max(
        cells.cell_len(summary), utils.get_cwidth(summary)
      )
      cached = self._summary = (
        owner.content,
        self._width,
        summary + ' ' * padding,
      )
    return ((('class:ui.message', cached[2]),),)

  def _content_start(self) -> tuple[int, int, int]:
    """The first conversation row painted below any pinned header."""
    index, part, row = self._start()
    if self._anchor is not None:
      return index, part, row
    owner = self._owner(index, part, row)
    if owner is None:
      return index, part, row
    return self._move(index, part, row, len(self._header(owner)))

  def _shows_last_row(self, index: int, part: int, row: int) -> bool:
    """Whether a viewport whose content starts here reaches the final row."""
    owner = self._owner(index, part, row)
    header_height = 0 if owner is None else len(self._header(owner))
    end_index, end_part, end_row = self._move(
      index, part, row, self._height - header_height - 1
    )
    return (
      end_index == len(self._blocks) - 1
      and end_part == len(self._blocks[end_index].parts) - 1
      and end_row == self._row_count(end_index, end_part) - 1
    )

  def scroll(self, rows: int) -> None:
    """Move by terminal rows, resuming following when the final row is visible."""
    self.clear_hover()
    self._last_activity = time.monotonic()
    if not self._blocks or (
      rows > 0 and self._anchor is None and not self._pending_scroll
    ):
      return
    # Movement waiting on the landing part's layout absorbs newer input: only
    # the latest target is ever laid out, never each intermediate one.
    if self._pending_scroll:
      self._pending_scroll[-1] += rows
    else:
      self._pending_scroll.append(rows)
    with self._layout_pass():
      self._apply_scroll()

  def _apply_scroll(self) -> None:
    """Retry navigation, never treating a missing layout as one row."""
    if not self._blocks:
      self._pending_scroll.clear()
    while self._pending_scroll:
      index, part, row = self._move(
        *self._content_start(), self._pending_scroll[0]
      )
      if not self._layout_ready:
        return
      follows = self._shows_last_row(index, part, row)
      if not self._layout_ready:
        return
      self._pending_scroll.popleft()
      if follows:
        self._anchor = None
        self._settling_line = None
      else:
        self._anchor_index = index
        self._anchor = Position(self._blocks[index].key, row, part)

  def page(self, direction: int) -> None:
    """Scroll one viewport, retaining a row of context."""
    self.scroll(direction * max(1, self._height - self._header_height - 1))

  def navigate(self, action: actions.NavigateHistory) -> None:
    """Apply reducer-selected navigation without taking editor focus."""
    match action.direction:
      case 'page_up':
        self.page(-1)
      case 'page_down':
        self.page(1)
      case 'oldest':
        self.first()
      case 'latest':
        self.follow()
      case 'up':
        self.scroll(-action.rows)
      case 'down':
        self.scroll(action.rows)

  def _refresh_blocks(self) -> None:
    settled = self._source()
    pending = tuple(self._pending())
    live_source = self._live()
    if (
      isinstance(live_source, tuple)
      and live_source is self._live_source
      and settled is self._settled_snapshot
      and pending == self._pending_snapshot
    ):
      return
    live_blocks: list[transcript.Block] = []
    for index, renderable in enumerate(live_source):
      previous = (
        self._live_snapshot[index] if index < len(self._live_snapshot) else None
      )
      if previous is not None and previous.renderable is renderable:
        live_blocks.append(previous)
      else:
        self._live_revision += 1
        live_blocks.append(
          transcript.Block(
            f'live:{index}', (0, self._live_revision), renderable
          )
        )
    self._live_source = live_source
    if settled is self._settled_snapshot and pending == self._pending_snapshot:
      if len(live_blocks) == len(self._live_snapshot) and all(
        block is previous
        for block, previous in zip(
          live_blocks, self._live_snapshot, strict=True
        )
      ):
        return
    else:
      self._user_position = None
      if _presentation(settled) != _presentation(self._settled_snapshot):
        self._expanded.clear()
    if self._user_position is not None:
      first, last, user = self._user_position
      last = min(last, len(settled) + len(pending) - 1)
      self._user_position = (first, last, user) if first <= last else None
    if self._anchor is not None and self._settling_line is None:
      for previous, replacement in zip(self._live_snapshot, live_blocks):
        before, after = previous.renderable, replacement.renderable
        if (
          previous.key == self._anchor.block
          and isinstance(before, rendering.Answer)
          and before.truncated
          and isinstance(after, rendering.Answer)
          and not after.truncated
        ):
          index = next(
            i for i, block in enumerate(self._blocks) if block is previous
          )
          self._remember_line(
            index, self._anchor.part, self._anchor.row, from_end=True
          )
    self._live_snapshot = tuple(live_blocks)
    self._settled_snapshot = settled
    self._pending_snapshot = pending
    self._blocks = settled + pending + self._live_snapshot

  def _user_before(self, index: int) -> int | None:
    if self._user_position is not None:
      first, last, user = self._user_position
      if first <= index <= last:
        return user
      if index > last:
        for position in range(last + 1, index + 1):
          if isinstance(self._blocks[position].renderable, rendering.UserText):
            first = user = position
        self._user_position = (first, index, user)
        return user
    user = next(
      (
        i
        for i in range(index, -1, -1)
        if isinstance(self._blocks[i].renderable, rendering.UserText)
      ),
      None,
    )
    self._user_position = (user or 0, index, user)
    return user

  def _prepare_content(self, width: int, height: int) -> tuple[_Rows, int]:
    """Discover missing layouts without publishing unpainted rows to the mouse."""
    if (
      width != self._width
      and self._anchor is not None
      and self._blocks
      and self._settling_line is None
    ):
      self._remember_line(*self._start())
    if (width, height) != (self._width, self._height):
      self.clear_hover()
    self._render_needed = None
    self._layout_ready = True
    self._width, self._height = max(1, width), max(1, height)
    self._refresh_blocks()
    self._apply_scroll()
    scroll_needed = self._render_needed
    # Paint the resolved anchor while further navigation awaits its geometry.
    self._render_needed = None
    self._layout_ready = True
    visible: list[_Row] = []
    positions: list[RowPosition] = []
    header_height = 0
    self._painted_span = None
    if self._blocks:
      index, part, row = self._start()
      first = last = index, part
      owner = self._owner(index, part, row)
      header = () if owner is None else self._header(owner)
      header_height = len(header)
      if self._anchor is None:
        index, part, row = self._move(index, part, row, len(header))
      visible.extend(header)
      while len(visible) < height and index < len(self._blocks):
        rows = self._rows(index, part)
        while row < len(rows) and len(visible) < height:
          visible.append(rows[row])
          block = self._blocks[index]
          positions.append((block.key, block.revision, part, row))
          row += 1
        last = index, part
        if part + 1 < len(self._blocks[index].parts):
          part, row = part + 1, 0
        else:
          index, part, row = index + 1, 0, 0
      self._painted_span = (*first, *last)
    # Live revisions must not starve the geometry needed by queued navigation.
    self._render_needed = scroll_needed or self._render_needed
    self._prepared_positions = tuple(positions)
    return tuple(visible), header_height

  def scrolling_positions(self) -> tuple[RowPosition, ...]:
    """Identify painted body rows without comparing their terminal cells."""
    return self._visible_positions

  def scrolling_range(self) -> range:
    """Return movable painted rows, even while their replacement is prepared."""
    return range(self._header_height, len(self._visible))

  @override
  def create_content(self, width: int, height: int) -> controls.UIContent:
    with self._layout_pass():
      visible, header_height = self._prepare_content(width, height)
      if self._layout_ready:
        self._header_height = header_height
        self._visible_positions = self._prepared_positions
      snapshot = visible if self._layout_ready else self._visible
      if snapshot != self._visible:
        self.clear_hover()
      self._visible = snapshot
      if self._layout_ready and self._render_needed is None:
        self._layouts.clear()
      self._schedule_render()
      self._last_activity = time.monotonic()
      self._schedule_prefetch()
      return controls.UIContent(
        get_line=lambda index: self._display_row(snapshot[index]),
        line_count=len(snapshot),
        show_cursor=False,
      )

  def _display_row(self, fragments: _Row) -> formatted_text.StyleAndTextTuples:
    if self._hovered_style is None:
      return list(fragments)
    if time.monotonic() >= self._copy_until:
      self._copied_style = None
    row: formatted_text.StyleAndTextTuples = []
    for fragment in fragments:
      style, text = fragment[:2]
      if self._hovered_style and self._hovered_style in style:
        copied = self._copied_style == self._hovered_style
        if 'class:code-action' in style:
          style += self._hovered_style.replace('code-copy-', 'click-target-')
          style += ' class:mouse-pointer'
          if self._hovered_action:
            style += ' class:code-action-hover'
          icon = (
            ('✓' if self._copy_label == 'Copied' else '!') if copied else '⧉'
          )
          text = icon.center(len(text))
        elif 'class:code-hint' in style and (copied or self._hovered_action):
          label = self._copy_label if copied else 'click to copy'
          if len(label) <= len(text):
            text = label.rjust(len(text))
        if copied and (
          'class:code-action' in style or 'class:code-hint' in style
        ):
          style += (
            ' class:code-copied'
            if self._copy_label == 'Copied'
            else ' class:code-copy-error'
          )
      row.append(
        (style, text, fragment[2]) if len(fragment) == 3 else (style, text)
      )
    return row

  def _queue_wheel(self, rows: int) -> None:
    """Coalesce wheel input: apply only the net movement once per input batch.

    Intermediate scroll targets are never laid out or painted; opposite wheel
    ticks cancel, so a burst costs one navigation however many events arrived.
    """
    # Following output, downward ticks are no-ops and must not cancel upward ones.
    if rows > 0 and self._anchor is None and not self._wheel_rows:
      return
    self._wheel_rows += rows
    if self._wheel_flush is not None:
      return
    app = current.get_app_or_none()
    loop = app.loop if app is not None else None
    if loop is None:
      self._flush_wheel()
    else:
      self._wheel_flush = loop.call_soon(self._flush_wheel)

  def _flush_wheel(self) -> None:
    """Apply the accumulated wheel movement as a single navigation."""
    if self._wheel_flush is not None:
      self._wheel_flush.cancel()
      self._wheel_flush = None
    rows, self._wheel_rows = self._wheel_rows, 0
    if rows:
      self._navigate(
        actions.NavigateHistory('down' if rows > 0 else 'up', rows=abs(rows))
      )
      workspace_application.invalidate_transcript()

  @override
  def mouse_handler(self, mouse_event: mouse_events.MouseEvent) -> None:
    if mouse_event.event_type == mouse_events.MouseEventType.MOUSE_MOVE:
      self.clear_hover()
    if mouse_event.event_type == mouse_events.MouseEventType.SCROLL_UP:
      self._queue_wheel(-1)
    elif mouse_event.event_type == mouse_events.MouseEventType.SCROLL_DOWN:
      self._queue_wheel(1)
    elif 0 <= mouse_event.position.y < len(self._visible):
      column = 0
      for fragment in self._visible[mouse_event.position.y]:
        if '[ZeroWidthEscape]' in fragment[0]:
          continue
        end = column + len(fragment[1])
        if column <= mouse_event.position.x < end and len(fragment) == 3:
          fragment[2](mouse_event)
          break
        column = end
    workspace_application.invalidate_transcript()
