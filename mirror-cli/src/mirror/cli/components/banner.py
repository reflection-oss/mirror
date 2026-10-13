"""Live welcome presentation for an empty conversation."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from prompt_toolkit.application import current
from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import (
  containers,
  controls,
  dimension,
  mouse_handlers as layout_mouse_handlers,
  screen as layout_screen,
)
from rich import align, console, text

from mirror.cli import rendering, state, themes
from mirror.cli.components import component


# Metacarbon's exact cell artwork; no bitmap generator is needed at runtime.
_WORDMARK = (
  '                                          ██                                    ',
  '  ▄▄████▄▄       ▄▄████▄▄      ██      ██                                       ',
  ' ██▀    ▀██     ██████████     ████  ████ ██ ██  ████ ██  ████   ████   ██  ████',
  '██        ██   ████████████    ██  ██  ██ ██ ████     ████     ██    ██ ████    ',
  '██        ██   ████████████    ██      ██ ██ ██       ██       ██    ██ ██      ',
  ' ██▄    ▄██     ██████████     ██      ██ ██ ██       ██       ██    ██ ██      ',
  '  ▀▀████▀▀       ▀▀████▀▀      ██      ██ ██ ██       ██         ████   ██      ',
)
_REFLECTION = (
  '  ▄▄████▄▄       ▄▄████▄▄      ██      ██ ██ ██       ██         ████   ██      ',
  ' █▀▀    ▀█▀     ▀█▀█▀█▀█▀█     █▀      █▀ ▀█ █▀       ▀█       █▀    █▀ ▀█      ',
  ' ▀█        ▀█   █▀█▀█▀█▀█▀█▀    █▀      █▀ ▀█ █▀       ▀█       █▀    █▀ ▀█      ',
)
_MARK = ('▗▞▀▀▚▖  ▗▟██▙▖', '█    █  ██████', '▝▚▄▄▞▘  ▝▜██▛▘')
_MARK_REFLECTION = ('▗▝▀▀▚   ▗▐█▜▙ ', ' ▚    ▜  ▚▜▚▜▚▜')


def _reflected(
  logo: tuple[str, ...], reflection: tuple[str, ...], *, gap: int = 0
) -> text.Text:
  image = text.Text('\n'.join(logo), 'ui.welcome-logo')
  image.append('\n' * (gap + 1) + '┈' * len(logo[0]), 'ui.welcome-axis')
  image.append('\n' * (gap + 1) + reflection[0], 'ui.welcome-reflection')
  image.append('\n' + '\n'.join(reflection[1:]), 'ui.welcome-reflection-deep')
  return image


def artwork(
  columns: int,
  rows: int,
  encoding: str | None,
  *,
  available_rows: int | None = None,
) -> text.Text:
  """Choose Metacarbon's logo, dropping whole parts as the composer grows."""
  choices: list[text.Text] = []
  if columns >= 84 and rows >= 30:
    choices.extend((
      _reflected(_WORDMARK, _REFLECTION, gap=1),
      text.Text('\n'.join(_WORDMARK), 'ui.welcome-logo'),
    ))
  if columns >= 20 and rows >= 22:
    choices.extend((
      _reflected(_MARK, _MARK_REFLECTION),
      text.Text('\n'.join(_MARK), 'ui.welcome-logo'),
    ))
  image = next(
    (
      image
      for image in choices
      if available_rows is None or len(image.split()) + 1 <= available_rows
    ),
    text.Text(),
  )
  try:
    image.plain.encode(encoding or 'ascii')
  except UnicodeEncodeError:
    return text.Text('Mirror', 'ui.welcome-logo.bold')
  return image


def input_width(columns: int) -> int:
  """Keep the welcome composer at most 80 cells wide, with side gutters."""
  return columns if columns < 30 else min(80, columns - 6)


def centered(
  content: containers.AnyContainer, columns: int
) -> containers.VSplit:
  """Center a fixed-width welcome column without changing its controls."""
  width = input_width(columns)
  return containers.VSplit([
    containers.Window(width=(columns - width) // 2),
    containers.HSplit([content], width=dimension.Dimension.exact(width)),
    containers.Window(),
  ])


class Background(containers.HSplit):
  """Metacarbon's quiet dot grid, kept clear of content and colored surfaces."""

  def __init__(self, content: containers.AnyContainer, rows: int) -> None:
    super().__init__(
      [content],
      height=dimension.Dimension.exact(rows),
      align=containers.VerticalAlign.CENTER,
    )

  @override
  def write_to_screen(
    self,
    screen: layout_screen.Screen,
    mouse_handlers: layout_mouse_handlers.MouseHandlers,
    write_position: layout_screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    super().write_to_screen(
      screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
    )
    # Match the source grid (4×2) and its content margin (3×1).
    x0, y0 = write_position.xpos, write_position.ypos
    width, height = write_position.width, write_position.height
    spans: list[tuple[int, int]] = []
    for y in range(height):
      row = screen.data_buffer[y0 + y]
      busy = [
        x
        for x in range(width)
        if row[x0 + x].char != ' ' or 'class:composer' in row[x0 + x].style
      ]
      spans.append((busy[0] - 3, busy[-1] + 3) if busy else (width, -1))
    for y in range(0, height, 2):
      for x in range(0, width, 4):
        if any(
          left <= x <= right for left, right in spans[max(0, y - 1) : y + 2]
        ):
          continue
        screen.data_buffer[y0 + y][x0 + x] = layout_screen.Char(
          '⠂', 'class:welcome-dots'
        )


class Banner(component.Component):
  """The current model and workspace, owned by the prompt rather than scrollback."""

  def __init__(
    self,
    directory: str,
    occupied_rows: Callable[[int, int], int],
  ) -> None:
    self._directory = directory
    self._occupied_rows = occupied_rows

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    # Hidden containers are still walked for key bindings and focus discovery.
    return containers.HSplit([
      containers.Window(
        controls.FormattedTextControl(lambda: self._text(state)),
        dont_extend_height=True,
        wrap_lines=False,
      ),
      containers.Window(height=1),
    ])

  def _text(self, state: state.State) -> ansi.ANSI:
    output = current.get_app().output
    columns, rows = state.layout.columns, output.get_size().rows
    width = input_width(columns)
    details = rendering.render_banner(state.status_bar.model, self._directory)
    occupied = self._occupied_rows(width, rows)
    theme_name = themes.current(state.extensions)
    details_ansi = rendering.renderable_to_ansi(
      details, width, theme_name=theme_name
    )
    # On short, narrow terminals the facts need the side gutters more than art.
    if len(details_ansi.splitlines()) + occupied + 1 > rows:
      details_ansi = rendering.renderable_to_ansi(
        details, columns, theme_name=theme_name
      )
    logo = artwork(
      columns,
      rows,
      output.encoding(),
      available_rows=rows - len(details_ansi.splitlines()) - occupied - 1,
    )
    details_text = text.Text.from_ansi(details_ansi.rstrip())
    welcome = (
      console.Group(
        align.Align.center(logo),
        text.Text(''),
        align.Align.center(details_text),
      )
      if logo
      else align.Align.center(details_text)
    )
    return ansi.ANSI(
      rendering.renderable_to_ansi(
        welcome, columns, theme_name=theme_name
      ).rstrip()
    )
