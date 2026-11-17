"""Optional Rich source adapter; no measuring, wrapping or console rendering."""

import dataclasses
import itertools
import re
from collections.abc import Callable, Mapping

from mirror_transcript import content, model
from pygments import token
from rich import (
  color,
  columns,
  console,
  default_styles,
  errors,
  markdown,
  padding,
  panel,
  rule,
  style,
  syntax,
  table,
  text,
)


def _color_codes(value: color.Color, background: bool) -> str:
  return ';'.join(value.get_ansi_codes(foreground=not background))


def encode_style(
  value: style.Style,
  *,
  encode_color: Callable[[color.Color, bool], str] = _color_codes,
  colors: bool = True,
  background: bool = True,
) -> str:
  """Encode a logical style, quantizing colors independently of Rich's ANSI cache."""
  attributes = value.without_color.clear_meta_and_links().render('x')
  codes = (
    [attributes[2:].partition('m')[0]] if attributes.startswith('\x1b[') else []
  )
  if colors:
    if value.color is not None:
      codes.append(encode_color(value.color, False))
    if background and value.bgcolor is not None:
      codes.append(encode_color(value.bgcolor, True))
  return ';'.join(codes)


def syntax_palette(
  name: str,
  *,
  encode_color: Callable[[color.Color, bool], str] = _color_codes,
  colors: bool = True,
) -> dict[str, str]:
  """Extract token colors only; the native renderer parses and highlights source."""
  theme = syntax.Syntax.get_theme(name)
  return {
    role: encode_style(
      theme.get_style_for_token(kind),
      encode_color=encode_color,
      colors=colors,
      background=False,
    )
    for role, kind in (
      ('text', token.Text),
      ('comment', token.Comment),
      ('string', token.String),
      ('escape', token.String.Escape),
      ('keyword', token.Keyword),
      ('operator', token.Operator),
      ('number', token.Number),
      ('constant', token.Keyword.Constant),
      ('function', token.Name.Function),
      ('class', token.Name.Class),
      ('builtin', token.Name.Builtin),
      ('name', token.Name),
    )
  }


@dataclasses.dataclass(frozen=True)
class Projection:
  """Resolve Rich sources with optional application types and palette encoding."""

  styles: Mapping[str, str]
  encode_color: Callable[[color.Color, bool], str] = _color_codes
  custom: Callable[[console.RenderableType], list[model.Block] | None] = (
    lambda _: None
  )

  def style_codes(self, value: str | style.Style) -> str:
    """Resolve attributes and colors without laying out content."""
    if isinstance(value, str) and value.removeprefix('ui.') in self.styles:
      return self.styles[value.removeprefix('ui.')]
    codes: list[str] = []
    if isinstance(value, str):
      names = [token for token in value.split() if token.startswith('ui.')]
      codes.extend(
        self.styles.get(name.removeprefix('ui.'), '') for name in names
      )
      try:
        parsed = style.Style.parse(
          ' '.join(token for token in value.split() if token not in names)
        )
      except errors.StyleSyntaxError:
        if value not in default_styles.DEFAULT_STYLES:
          return self.styles['text']
        parsed = default_styles.DEFAULT_STYLES[value]
    else:
      parsed = value
    colored = any(
      re.search(r'(?:^|;)(?:[349][0-9]|10[0-7])(?:;|$)', sgr)
      for sgr in self.styles.values()
    )
    codes.append(
      encode_style(parsed, encode_color=self.encode_color, colors=colored)
    )
    return ';'.join(filter(None, codes))

  def text_lines(self, value: text.Text) -> list[model.Line]:
    """Preserve overlapping logical styles without wrapping their text."""
    boundaries = sorted({
      0,
      len(value),
      *(min(len(value), span.start) for span in value.spans),
      *(min(len(value), span.end) for span in value.spans),
    })
    spans: list[model.Span] = []
    base = self.style_codes(value.style)
    for start, end in itertools.pairwise(boundaries):
      codes = ';'.join(
        filter(
          None,
          [
            base,
            *(
              self.style_codes(span.style)
              for span in value.spans
              if span.start <= start < span.end
            ),
          ],
        )
      )
      spans.append(model.Span(content.source(value.plain[start:end]), codes))
    return content.lines(spans)

  def cell(self, value: console.RenderableType) -> model.Block:
    """Preserve source overflow policy; the native renderer measures the cell."""
    return model.Block(
      kind=model.Kind.FRAME,
      insets=(0, 0, 0, 0),
      sgr=self.style_codes(value.style) if isinstance(value, text.Text) else '',
      overflow=model.Overflow[(value.overflow or 'fold').upper()]
      if isinstance(value, text.Text)
      else model.Overflow.FOLD,
      no_wrap=isinstance(value, text.Text) and value.no_wrap is True,
      children=self.blocks(value),
    )

  def blocks(self, value: console.RenderableType) -> list[model.Block]:
    """Extract source trees, including application types nested inside Rich ones."""
    custom = self.custom(value)
    if custom is not None:
      return custom
    match value:
      case str():
        return [model.Block(content.source(value), sgr=self.styles['text'])]
      case text.Text():
        return [model.Block(lines=self.text_lines(value))]
      case markdown.Markdown():
        return [
          model.Block(
            content.source(value.markup),
            kind=model.Kind.MARKDOWN,
            sgr=self.styles['text'],
          )
        ]
      case syntax.Syntax():
        lexer = value.lexer
        return [
          model.Block(
            content.source(value.code),
            kind=model.Kind.CODE,
            sgr=self.styles['text'],
            language=lexer.aliases[0]
            if lexer is not None and lexer.aliases
            else '',
          )
        ]
      case console.Group():
        result: list[model.Block] = []
        for child in value.renderables:
          result.extend(self.blocks(child))
        return result
      case padding.Padding():
        return [
          model.Block(
            kind=model.Kind.FRAME,
            insets=(value.top, value.right, value.bottom, value.left),
            sgr=self.style_codes(value.style),
            children=self.blocks(value.renderable),
          )
        ]
      case panel.Panel():
        title = value.title
        if isinstance(title, str):
          title = text.Text.from_markup(title)
        return [
          model.Block(
            kind=model.Kind.PANEL,
            insets=padding.Padding.unpack(value.padding),
            sgr=self.style_codes(value.style),
            title=content.source(title.plain) if title else '',
            title_sgr=self.style_codes(title.style) if title else '',
            border_sgr=self.style_codes(value.border_style),
            children=self.blocks(value.renderable),
          )
        ]
      case columns.Columns():
        return [
          model.Block(
            kind=model.Kind.ROW,
            children=[
              model.Block(kind=model.Kind.COLUMN, children=self.blocks(child))
              for child in value.renderables
            ],
          )
        ]
      case table.Table():
        return [self._table(value)]
      case rule.Rule():
        title = (
          value.title.copy()
          if isinstance(value.title, text.Text)
          else text.Text.from_markup(value.title)
        )
        title.plain = title.plain.replace('\n', ' ')
        return [
          model.Block(
            content.source(value.characters),
            kind=model.Kind.RULE,
            lines=self.text_lines(title),
            sgr=self.style_codes(value.style),
            alignment=model.Alignment[value.align.upper()],
          )
        ]
      case _:
        return [
          model.Block(
            f'No logical presentation for {type(value).__name__}',
            sgr=self.styles['warning'],
          )
        ]

  def _table(self, value: table.Table) -> model.Block:
    rows = list(
      itertools.zip_longest(
        *(column.cells for column in value.columns), fillvalue=''
      )
    )
    if value.show_header:
      rows.insert(0, tuple(column.header for column in value.columns))
    return model.Block(
      kind=model.Kind.TABLE if value.show_header else model.Kind.GRID,
      spread=value.expand,
      children=[
        model.Block(
          kind=model.Kind.ROW,
          children=[
            model.Block(
              kind=model.Kind.FRAME,
              ratio=column.ratio or 0,
              no_wrap=column.no_wrap,
              overflow=model.Overflow[column.overflow.upper()],
              insets=(
                value.padding[0],
                value.padding[1]
                if value.pad_edge or index + 1 < len(row)
                else 0,
                value.padding[2],
                value.padding[3] if value.pad_edge and index == 0 else 0,
              ),
              sgr=self.style_codes(column.style),
              children=self.blocks(cell),
            )
            for index, (column, cell) in enumerate(
              zip(value.columns, row, strict=True)
            )
          ],
        )
        for row in rows
      ],
    )
