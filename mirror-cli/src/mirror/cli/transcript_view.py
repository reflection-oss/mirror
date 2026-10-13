"""An anchored, lazily rendered conversation inside the terminal screen."""

from __future__ import annotations

import collections
import dataclasses
import itertools
import re
from collections.abc import Callable, Sequence
from typing import override

from prompt_toolkit import formatted_text, mouse_events
from prompt_toolkit.application import current
from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import controls
from rich import console

from mirror.cli import actions, rendering, transcript


@dataclasses.dataclass(frozen=True)
class Position:
  block: str
  row: int


class TranscriptView(controls.UIControl):
  """Paint visible rows while keeping reading positions independent of arrivals."""

  def __init__(
    self,
    source: transcript.Transcript,
    live: Callable[[], Sequence[console.RenderableType]],
    *,
    navigate: Callable[[actions.NavigateHistory], object] | None = None,
    theme_name: Callable[[], str] = lambda: 'dark',
  ) -> None:
    self._source = source
    self._navigate = navigate or self.navigate
    self._live = live
    self._theme_name = theme_name
    self._anchor: Position | None = None
    self._width = 0
    self._height = 0
    self._blocks: Sequence[transcript.Block] = ()
    self._visible: list[formatted_text.StyleAndTextTuples] = []
    self._live_snapshot: tuple[transcript.Block, ...] = ()
    self._live_revision = 0
    self._settling_line: tuple[str, int, int] | None = None
    self._cache: collections.OrderedDict[
      tuple[str, tuple[int, int], int, str],
      list[formatted_text.StyleAndTextTuples],
    ] = collections.OrderedDict()
    self._cache_bytes = 0
    self._header_height = 0

  def _remember_line(self, index: int, row: int, first: int) -> None:
    """Locate reading content across rewrapping and preview settlement."""

    def plain(fragments: formatted_text.StyleAndTextTuples) -> str:
      return (
        ''.join(text for _, text, *_ in fragments).strip().removeprefix('● ')
      )

    rows = self._rows(index)
    target_row = next((i for i in range(row, len(rows)) if plain(rows[i])), row)
    pattern = r'\s+'.join(
      re.escape(word) for word in plain(rows[target_row]).split()
    )
    preceding = '\n'.join(
      plain(fragments)
      for i in range(first, index + 1)
      for fragments in (
        self._rows(i)[:target_row] if i == index else self._rows(i)
      )
    )
    occurrence = (
      sum(1 for _ in re.finditer(pattern, preceding)) if pattern else 0
    )
    self._settling_line = (pattern, occurrence, target_row - row)

  def settle_live(self, identities: str | Sequence[str]) -> None:
    """Transfer the preview anchor to one final record or retained preview parts."""
    if self._anchor is None or not self._anchor.block.startswith('live:'):
      return
    index, row = self._start()
    if isinstance(identities, str):
      first = next(
        i
        for i, block in enumerate(self._blocks)
        if block.key.startswith('live:')
      )
      self._remember_line(index, row, first)
      self._anchor = Position(
        identities, sum(len(self._rows(i)) for i in range(first, index)) + row
      )
    else:
      part = int(self._anchor.block.removeprefix('live:'))
      if part < len(identities):
        self._anchor = Position(identities[part], row)

  def follow(self) -> None:
    """Resume following the latest conversation output."""
    self._anchor = None

  def first(self) -> None:
    """Reach the oldest entry without formatting the intervening history."""
    if self._blocks:
      self._anchor = Position(self._blocks[0].key, 0)

  @property
  def reading(self) -> bool:
    """Whether incoming output should leave the reading anchor in place."""
    return self._anchor is not None

  def clear(self) -> None:
    """Release layouts when their conversation changes."""
    self._anchor = None
    self._settling_line = None
    self._cache.clear()
    self._cache_bytes = 0
    self._blocks = ()
    self._visible.clear()
    self._live_snapshot = ()

  def _rows(self, index: int) -> list[formatted_text.StyleAndTextTuples]:
    block = self._blocks[index]
    theme_name = self._theme_name()
    key = (block.key, block.revision, self._width, theme_name)
    cached = self._cache.get(key)
    if cached is not None:
      self._cache.move_to_end(key)
      return cached
    value = rendering.renderable_to_ansi(
      block.renderable, self._width, theme_name=theme_name
    )
    value = rendering.without_hyperlinks(value)
    fragments = formatted_text.to_formatted_text(ansi.ANSI(value.rstrip()))
    rows: list[formatted_text.StyleAndTextTuples] = []
    for line in formatted_text.split_lines(fragments):
      row: formatted_text.StyleAndTextTuples = [
        (style, ''.join(fragment[1] for fragment in group))
        for style, group in itertools.groupby(
          line, key=lambda fragment: fragment[0]
        )
      ]
      rows.append(row)
    rows.extend([[], []])
    self._cache[key] = rows
    self._cache_bytes += sum(len(text) for row in rows for _, text, *_ in row)
    while len(self._cache) > 1 and (
      len(self._cache) > 64 or self._cache_bytes > 8 * 1024 * 1024
    ):
      _, removed = self._cache.popitem(last=False)
      self._cache_bytes -= sum(
        len(text) for row in removed for _, text, *_ in row
      )
    return rows

  def _move(self, index: int, row: int, amount: int) -> tuple[int, int]:
    row += amount
    while row < 0 and index > 0:
      index -= 1
      row += len(self._rows(index))
    while index < len(self._blocks) - 1 and row >= len(self._rows(index)):
      row -= len(self._rows(index))
      index += 1
    return index, max(0, min(row, len(self._rows(index)) - 1))

  def _start(self) -> tuple[int, int]:
    if self._anchor is not None:
      index = next(
        (
          i
          for i, block in enumerate(self._blocks)
          if block.key == self._anchor.block
        ),
        0,
      )
      rows = self._rows(index)
      if self._settling_line is not None:
        (pattern, occurrence, offset), self._settling_line = (
          self._settling_line,
          None,
        )
        if pattern:
          text = '\n'.join(
            ''
            .join(fragment[1] for fragment in fragments)
            .strip()
            .removeprefix('● ')
            for fragments in rows
          )
          match = next(
            itertools.islice(re.finditer(pattern, text), occurrence, None), None
          )
          if match is not None:
            row = max(0, text.count('\n', 0, match.start()) - offset)
            self._anchor = Position(self._blocks[index].key, row)
      return index, min(self._anchor.row, len(rows) - 1)
    index = len(self._blocks) - 1
    return self._move(index, len(self._rows(index)) - 1, -self._height + 1)

  def scroll(self, rows: int) -> None:
    """Move by terminal rows without moving the composer or changing focus."""
    if not self._blocks:
      return
    index, row = self._start()
    index, row = self._move(index, row, rows)
    self._anchor = Position(self._blocks[index].key, row)
    if rows > 0:
      end_index, end_row = self._move(
        index, row, self._height - self._header_height - 1
      )
      if (
        end_index == len(self._blocks) - 1
        and end_row == len(self._rows(end_index)) - 1
      ):
        self.follow()

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
        self.scroll(-3)
      case 'down':
        self.scroll(3)

  @override
  def create_content(self, width: int, height: int) -> controls.UIContent:
    if width != self._width and self._anchor is not None and self._blocks:
      index, row = self._start()
      self._remember_line(index, row, index)
    self._width, self._height = max(1, width), max(1, height)
    settled = self._source.blocks()
    live_blocks: list[transcript.Block] = []
    for index, renderable in enumerate(self._live()):
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
    live = self._live_snapshot = tuple(live_blocks)
    self._blocks = settled + live if live else settled
    self._visible = []
    self._header_height = 0
    if self._blocks:
      index, row = self._start()
      user_index = next(
        (i for i in range(index, -1, -1) if self._blocks[i].is_user),
        None,
      )
      if user_index is not None and (user_index < index or row > 0):
        # The message owns this section once its box crosses the top edge.
        # Reserve only a bounded header, leaving room for the conversation.
        header = self._rows(user_index)[:-2][: max(1, height // 3)]
        self._header_height = len(header)
        if self._anchor is None:
          index, row = self._move(index, row, len(header))
        self._visible.extend(header)
        if index == user_index:
          row = max(row, len(header))
      while len(self._visible) < height and index < len(self._blocks):
        rows = self._rows(index)
        while row < len(rows) and len(self._visible) < height:
          self._visible.append(rows[row])
          row += 1
        index, row = index + 1, 0
    return controls.UIContent(
      get_line=lambda index: self._visible[index],
      line_count=len(self._visible),
      show_cursor=False,
    )

  @override
  def mouse_handler(self, mouse_event: mouse_events.MouseEvent) -> None:
    if mouse_event.event_type == mouse_events.MouseEventType.SCROLL_UP:
      self._navigate(actions.NavigateHistory('up'))
    elif mouse_event.event_type == mouse_events.MouseEventType.SCROLL_DOWN:
      self._navigate(actions.NavigateHistory('down'))
    current.get_app().invalidate()
