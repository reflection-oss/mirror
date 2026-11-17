"""Workspace-browser artwork and compact conversation identity."""

from __future__ import annotations

import itertools
from typing import override

from prompt_toolkit import formatted_text
from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import containers, controls
from rich import text

from mirror.cli import rendering, state, themes
from mirror.cli.components import component

# Metacarbon's compact mark keeps its cell geometry at every terminal width.
_MARK = ('▗▞▀▀▚▖  ▗▟██▙▖', '█    █  ██████', '▝▚▄▄▞▘  ▝▜██▛▘')


def artwork(
  columns: int,
  rows: int,
  encoding: str | None,
  *,
  available_rows: int | None = None,
) -> text.Text:
  """Keep the browser's identity compact, yielding space to work and input."""
  if rows < 12 or columns < 6:
    return text.Text()
  compact = text.Text('Mirror', 'ui.welcome-logo.bold')
  choices = [compact]
  if columns >= 26 and rows >= 24:
    mark = text.Text(_MARK[0] + '\n' + _MARK[1], 'ui.welcome-logo')
    mark.append('   Mirror', 'bold')
    mark.append('\n' + _MARK[2])
    reflected = mark.copy()
    reflected.append('\n' + '┈' * 14, 'ui.welcome-axis')
    reflected.append('\n▗▝▀▀▚   ▗▐█▜▙', 'ui.welcome-reflection-deep')
    choices = [reflected, mark, compact]
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


class Banner(component.Component):
  """The current model and workspace, owned by the prompt rather than scrollback."""

  def __init__(self, directory: str) -> None:
    self._directory = directory

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

  def _text(self, state: state.State) -> formatted_text.StyleAndTextTuples:
    details = rendering.render_banner(state.status_bar.model, self._directory)
    fragments = ansi.ANSI(
      rendering.renderable_to_ansi(
        details,
        state.layout.columns,
        theme_name=themes.current(state.extensions, state.extension_themes),
      ).rstrip()
    ).__pt_formatted_text__()
    # ANSI emits one fragment per character; layout needs only the style runs.
    return [
      (style, ''.join(fragment[1] for fragment in run))
      for style, run in itertools.groupby(fragments, key=lambda part: part[0])
    ]
