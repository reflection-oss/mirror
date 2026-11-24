"""Extract logical transcript sources; never measure, wrap, or paint in Python."""

from __future__ import annotations

import functools
import io
from collections.abc import Mapping

from mirror_transcript import compat_rich, content, model
from prompt_toolkit import data_structures
from prompt_toolkit.output import color_depth, vt100
from prompt_toolkit.styles import base
from rich import color, console, text

from mirror.cli import rendering, themes, tool_groups


@functools.lru_cache(maxsize=512)
def _color_codes(
  value: color.Color, depth: color_depth.ColorDepth, background: bool
) -> str:
  if value.type in (color.ColorType.DEFAULT, color.ColorType.STANDARD):
    return ';'.join(value.get_ansi_codes(foreground=not background))
  if depth == color_depth.ColorDepth.DEPTH_1_BIT:
    return ''
  buffer = io.StringIO()
  encoder = vt100.Vt100_Output(
    buffer, lambda: data_structures.Size(rows=1, columns=1)
  )
  rgb = value.get_truecolor().hex.lstrip('#')
  attrs = (
    base.DEFAULT_ATTRS._replace(bgcolor=rgb)
    if background
    else base.DEFAULT_ATTRS._replace(color=rgb)
  )
  encoder.set_attributes(attrs, depth)
  encoder.flush()
  return buffer.getvalue().removeprefix('\x1b[0;').removesuffix('m')


@functools.lru_cache(maxsize=128)
def palette(
  name: str | themes.Theme, depth: color_depth.ColorDepth
) -> dict[str, str]:
  """Use the same palette and terminal color quantization as Python chrome."""
  resolved = color_depth.ColorDepth.DEPTH_4_BIT if name == 'terminal' else depth

  def encode(value: color.Color, background: bool) -> str:
    return _color_codes(value, resolved, background)

  colors = name != 'mono' and depth != color_depth.ColorDepth.DEPTH_1_BIT
  result = {
    role.removeprefix('ui.'): compat_rich.encode_style(
      value, encode_color=encode, colors=colors
    )
    for role, value in themes.rich_theme(name).styles.items()
    if role.startswith('ui.')
  }
  for prefix, syntax in {
    **themes.syntax_names(name),
    'shell': 'monokai',
  }.items():
    result.update({
      f'syntax.{prefix}.{role}': codes
      for role, codes in compat_rich.syntax_palette(
        syntax, encode_color=encode, colors=colors
      ).items()
    })
  return result


class Projection:
  """Adapt the existing Python membership projection without rebuilding policy."""

  def __init__(
    self, styles: Mapping[str, str], depth: color_depth.ColorDepth
  ) -> None:
    self.styles = styles
    self.rich = compat_rich.Projection(
      styles,
      encode_color=lambda value, background: _color_codes(
        value, depth, background
      ),
      custom=self._custom,
    )

  def blocks(self, value: console.RenderableType) -> list[model.Block]:
    """Project an existing source, including nested application renderables."""
    return self.rich.blocks(value)

  def _tool(
    self, title: model.Block, body: console.RenderableType | None, error: bool
  ) -> list[model.Block]:
    children = [title]
    if body is not None:
      children.append(
        model.Block(
          kind=model.Kind.FRAME,
          insets=(0, 0, 0, 2),
          marker=model.Span('└', '2'),
          children=self.blocks(body),
        )
      )
    return [
      model.Block(
        kind=model.Kind.FRAME,
        insets=(0, 0, 0, rendering.GUTTER_WIDTH),
        marker=model.Span('✗', self.styles['error'])
        if error
        else model.Span('·', '2'),
        children=children,
      )
    ]

  def _custom(self, value: console.RenderableType) -> list[model.Block] | None:
    match value:
      case tool_groups.ToolGroup():
        return self.blocks(value.summary())
      case rendering.ToolBlock():
        return self._tool(self.rich.cell(value.title), value.body, value.error)
      case rendering.BashResult():
        title = rendering.ToolLabel('Ran', error=value.error).render()
        title.append(' ')
        suffix = text.Text(
          value.suffix, style='ui.error' if value.error else 'ui.muted'
        )
        return self._tool(
          model.Block(
            kind=model.Kind.FRAME,
            insets=(0, 0, 0, 0),
            overflow=model.Overflow.FOLD
            if value.wrap
            else model.Overflow.ELLIPSIS,
            no_wrap=not value.wrap,
            children=[
              model.Block(
                kind=model.Kind.INLINE,
                children=[
                  *self.blocks(title),
                  model.Block(
                    content.source(value.command),
                    kind=model.Kind.CODE,
                    language='bash',
                    syntax='shell',
                    sgr='49',
                  ),
                  *self.blocks(suffix),
                ],
              )
            ],
          ),
          value.body,
          value.error,
        )
      case rendering.CommandApproval():
        title = rendering.ToolLabel(value.name).render()
        title.append(' ')
        code = model.Block(
          content.source(value.command),
          kind=model.Kind.CODE,
          language='bash',
          syntax='shell',
          sgr='49',
        )
        return self._tool(
          model.Block(
            kind=model.Kind.INLINE, children=[*self.blocks(title), code]
          ),
          rendering.approval_arguments(value.arguments)
          if value.arguments
          else None,
          False,
        )
      case rendering.DiffPanel():
        return [
          model.Block(
            content.source(value.diff),
            kind=model.Kind.DIFF,
            show_file_headers=value.show_file_headers,
          )
        ]
      case rendering.UserText():
        return [
          model.Block(
            kind=model.Kind.FRAME,
            insets=(1, 1, 1, rendering.GUTTER_WIDTH),
            marker=model.Span(rendering.PROMPT, '1;2'),
            sgr=self.styles['message'],
            children=self.blocks(value.body()),
          )
        ]
      case rendering.Answer():
        return [
          model.Block(
            kind=model.Kind.FRAME,
            insets=(0, 0, 0, rendering.GUTTER_WIDTH),
            marker=model.Span(
              '' if value.continuation else rendering.ASSISTANT_ICON,
              self.styles['accent'],
            ),
            children=[
              model.Block(
                content.source(value.source).strip(), kind=model.Kind.MARKDOWN
              )
            ],
          )
        ]
      case rendering.MarkdownSource():
        return [
          model.Block(
            content.source(value.content),
            kind=model.Kind.MARKDOWN,
            sgr=self.rich.style_codes(value.style),
          )
        ]
      case rendering.Expandable():
        return [
          model.Block(
            kind=model.Kind.COLUMN,
            children=self.blocks(value.expanded),
            collapsed=self.blocks(value.collapsed),
          )
        ]
      case _:
        return None
