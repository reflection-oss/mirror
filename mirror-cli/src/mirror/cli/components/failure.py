"""Stateless worker-failure component."""

from __future__ import annotations

from typing import override

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import state
from mirror.cli.components import component


class Failure(component.Component):
  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    fragments: StyleAndTextTuples = (
      []
      if state.lifecycle.failure is None
      else [
        ('class:worktree-failed', 'Agent failed to load. '),
        ('class:menu-hint', 'Press Enter to try loading again'),
      ]
    )
    return containers.Window(
      controls.FormattedTextControl(fragments),
      height=dimension.Dimension.exact(1 if fragments else 0),
    )
