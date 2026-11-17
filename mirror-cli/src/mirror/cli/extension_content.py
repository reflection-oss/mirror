"""Render extension source hooks as Python-owned chrome, not transcript rows."""

from __future__ import annotations

import dataclasses
import re
from typing import override

from mirror_transcript import content, model
from prompt_toolkit import formatted_text
from prompt_toolkit.layout import containers, controls
from rich import (
  console as rich_console,
  measure,
  padding,
  panel,
  rule,
  segment,
  style,
  syntax,
  table,
  text,
)

from mirror.cli import rendering, themes


def window(
  source: rich_console.RenderableType, theme: str | themes.Theme
) -> containers.Window:
  """Keep logical widgets on the existing Python layout and fault boundary."""
  return containers.Window(
    _Content(source, theme), wrap_lines=False, dont_extend_height=True
  )


@dataclasses.dataclass
class _Content(controls.UIControl):
  source: rich_console.RenderableType
  theme: str | themes.Theme

  def _control(self, width: int) -> controls.FormattedTextControl:
    return controls.FormattedTextControl(
      formatted_text.ANSI(
        rendering.render_blocks(
          [self.source], max(1, width), theme_name=self.theme
        )
      ),
      show_cursor=False,
    )

  @override
  def preferred_width(self, max_available_width: int) -> int:
    return self._control(max_available_width).preferred_width(
      max_available_width
    )

  @override
  def preferred_height(
    self,
    width: int,
    max_available_height: int,
    wrap_lines: bool,
    get_line_prefix: controls.GetLinePrefixCallable | None,
  ) -> int | None:
    return self._control(width).preferred_height(
      width, max_available_height, wrap_lines, get_line_prefix
    )

  @override
  def create_content(self, width: int, height: int) -> controls.UIContent:
    return self._control(width).create_content(width, height)


def _style(sgr: str) -> style.Style:
  if not re.fullmatch(r'[0-9;]*', sgr):
    raise ValueError('Widget style must contain SGR parameters only')
  decoded = text.Text.from_ansi(f'\x1b[{sgr}m ')
  return decoded.get_style_at_offset(rich_console.Console(), 0)


def _text(value: str, sgr: str = '') -> text.Text:
  return text.Text(content.source(value), style=_style(sgr))


@dataclasses.dataclass(frozen=True)
class Native:
  """Compatibility for native source hooks while Python retains widget layout."""

  block: model.Block

  def __rich_console__(
    self, console: rich_console.Console, options: rich_console.ConsoleOptions
  ) -> rich_console.RenderResult:
    block = self.block
    if block.height == 0:
      return
    if block.collapsed:
      yield from console.render(
        rich_console.Group(*map(Native, block.collapsed)), options
      )
      return
    value = self._source(console, options)
    if block.title and block.kind not in (model.Kind.PANEL, model.Kind.RULE):
      value = rich_console.Group(_text(block.title, block.title_sgr), value)
    if block.padding:
      value = padding.Padding(value, (0, block.padding))
    lines = console.render_lines(
      value, options, style=_style(block.sgr), pad=False
    )
    if block.collapse and len(lines) > block.collapse:
      lines = lines[: block.collapse] + [[segment.Segment('…')]]
    if block.height is not None:
      lines = lines[: block.height] + [[]] * max(0, block.height - len(lines))
    if block.trim_end:
      while lines and not ''.join(part.text for part in lines[-1]).strip():
        lines.pop()
    for line in lines:
      yield from line
      yield segment.Segment.line()

  def __rich_measure__(
    self, console: rich_console.Console, options: rich_console.ConsoleOptions
  ) -> measure.Measurement:
    return measure.Measurement.get(
      console, options, self._source(console, options)
    )

  def _source(
    self, console: rich_console.Console, options: rich_console.ConsoleOptions
  ) -> rich_console.RenderableType:
    block = self.block
    children = [Native(child) for child in block.children]
    match block.kind:
      case model.Kind.MARKDOWN:
        return rendering.MarkdownSource(block.text)
      case model.Kind.CODE:
        return syntax.Syntax(
          content.source(block.text), block.language, background_color='default'
        )
      case model.Kind.DIFF:
        return rendering.DiffPanel(
          block.text, show_file_headers=block.show_file_headers
        )
      case model.Kind.RULE:
        return rule.Rule(
          _text(block.title, block.title_sgr),
          style=_style(block.sgr),
          align='left'
          if block.alignment is model.Alignment.LEFT
          else 'right'
          if block.alignment is model.Alignment.RIGHT
          else 'center',
        )
      case model.Kind.PANEL:
        return panel.Panel(
          rich_console.Group(*children),
          title=_text(block.title, block.title_sgr),
          border_style=_style(block.border_sgr),
          padding=0,
        )
      case model.Kind.FRAME:
        if len(block.insets) != 4:
          raise ValueError('Widget frame needs four insets')
        top, right, bottom, left = block.insets
        body: rich_console.RenderableType = rich_console.Group(*children)
        if block.marker is not None:
          row = table.Table.grid(padding=0)
          row.add_column(width=max(1, left))
          row.add_column(ratio=1)
          row.add_row(_text(block.marker.text, block.marker.sgr), body)
          body, left = row, 0
        return padding.Padding(body, (top, right, bottom, left))
      case model.Kind.TABLE | model.Kind.GRID:
        grid = (
          table.Table(box=None, show_header=False, padding=(0, 1))
          if block.kind is model.Kind.TABLE
          else table.Table.grid(padding=0)
        )
        for row in block.children:
          grid.add_row(*map(Native, row.children))
        return grid
      case model.Kind.ROW:
        row = table.Table.grid(padding=0, expand=block.spread)
        for child in block.children:
          row.add_column(ratio=child.ratio or None)
        row.add_row(*children)
        return row
      case model.Kind.INLINE:
        value = text.Text()
        for child in children:
          for part in console.render(child, options):
            value.append(part.text.replace('\n', ''), part.style)
        return value
      case model.Kind.COLUMN:
        values: list[rich_console.RenderableType] = []
        for child in children:
          if values and block.gap:
            values.append(text.Text('\n' * (block.gap - 1)))
          values.append(child)
        return rich_console.Group(*values)
      case model.Kind.TEXT:
        value = _text(block.text, block.sgr)
        if block.lines:
          value = text.Text('\n').join(
            text.Text.assemble(*(_text(span.text, span.sgr) for span in line))
            for line in block.lines
          )
        value.no_wrap = block.no_wrap or block.clip
        value.overflow = (
          'crop'
          if block.clip or block.overflow is model.Overflow.CROP
          else 'ellipsis'
          if block.overflow is model.Overflow.ELLIPSIS
          else 'fold'
        )
        return value
