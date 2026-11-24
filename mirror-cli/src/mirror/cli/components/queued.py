"""Stateless queued-instruction component."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import quotes, rendering, state, themes
from mirror.cli.components import component


class Queued(component.Component):
  """Local submissions not yet observed in authoritative history."""

  @override
  def cache_key(self, state: state.State) -> object:
    return (
      state.turn.pending_instructions,
      state.quotes,
      self._skip_first(),
      state.layout.columns,
      themes.current(state.extensions, state.extension_themes),
    )

  def __init__(self, *, skip_first: Callable[[], bool] = lambda: False) -> None:
    self._skip_first = skip_first

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    instructions = state.turn.pending_instructions
    if self._skip_first():
      instructions = instructions[1:]
    text = rendering.render_blocks(
      [
        rendering.user_text(quotes.message(instruction, state.quotes))
        for instruction in instructions
      ],
      state.layout.columns,
      theme_name=themes.current(state.extensions, state.extension_themes),
    )
    return containers.Window(
      controls.FormattedTextControl(ansi.ANSI(text)),
      height=None if text else dimension.Dimension.exact(0),
      wrap_lines=False,
      dont_extend_height=True,
    )
