"""Stateless queued-instruction component."""

from __future__ import annotations

from typing import override

from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import rendering, state, themes
from mirror.cli.components import component


class Queued(component.Component):
  """Local submissions not yet observed in authoritative history."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    text = rendering.render_blocks(
      [
        rendering.user_text(instruction)
        for instruction in state.turn.pending_instructions
      ],
      state.layout.columns,
      theme_name=themes.current(state.extensions),
    )
    return containers.Window(
      controls.FormattedTextControl(ansi.ANSI(text)),
      height=None if text else dimension.Dimension.exact(0),
      wrap_lines=False,
      dont_extend_height=True,
    )
