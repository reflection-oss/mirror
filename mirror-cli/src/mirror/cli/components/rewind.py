"""Stateless conversation-rewind components."""

from __future__ import annotations

from typing import override

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import state
from mirror.cli.components import component
from mirror.cli.components import menu


class RewindPicker(component.Component):
  """Rewind choices and fork confirmation."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    panel = _confirm_panel(state)
    fragments = panel.render() if panel is not None else []
    height = panel.height if panel is not None else 0
    return containers.Window(
      controls.FormattedTextControl(fragments),
      height=dimension.Dimension.exact(height if state.rewind.active else 0),
    )


class RewindBanner(component.Component):
  """Standing reminder that the composer targets an earlier turn."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    staged = state.rewind.staged
    fragments: StyleAndTextTuples = []
    if staged is not None:
      fragments.append((
        'class:rewind-banner',
        (
          f' ↩ rewound past {staged.discards} turn(s) · '
          'Esc returns to the latest turn '
        ),
      ))
    return containers.Window(
      controls.FormattedTextControl(fragments),
      height=dimension.Dimension.exact(1 if staged is not None else 0),
      style='class:rewind-banner',
    )


def _confirm_panel(state: state.State) -> menu.ConfirmPanel | None:
  staged = state.rewind.staged
  if staged is None or not staged.confirming:
    return None
  return menu.ConfirmPanel(
    title='Continue in a new conversation?',
    lines=(
      f'Continuing from before: {staged.target.preview(40)}',
      f'{staged.discards} later turn(s) stay in the original conversation.',
      'Files changed during those turns are not restored.',
    ),
    hint='↵/y fork and send · n/Esc keep editing',
  )
