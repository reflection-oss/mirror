"""Static status-bar component for the terminal UI."""

from __future__ import annotations

from typing import override

from prompt_toolkit.formatted_text import (
  StyleAndTextTuples as _StyleAndTextTuples,
)
from prompt_toolkit.layout import containers, controls

from mirror.cli import state as _state
from mirror.cli.components import component


class StatusBar(component.Component):
  """The static status bar below transient activity."""

  def __init__(
    self,
    extension_left: component.Component,
    extension_right: component.Component,
  ) -> None:
    self._extension_left = extension_left
    self._extension_right = extension_right

  @override
  def render(self, state: _state.State) -> containers.AnyContainer:
    """Render the complete status-bar layout from application state."""
    return containers.VSplit([
      containers.Window(
        controls.FormattedTextControl(_left(state)),
        style='class:footer',
        dont_extend_height=True,
        dont_extend_width=True,
      ),
      self._extension_left.render(state),
      containers.Window(),
      self._extension_right.render(state),
    ])


def _left(state: _state.State) -> _StyleAndTextTuples:
  if state.session.pending_switch is not None:
    return []
  approval = (
    ('class:footer-approval-manual', '🛠  manual')
    if state.status_bar.approval_mode == 'approve'
    else ('class:footer-approval-auto', '🛠  auto')
  )
  segments = [('class:footer class:footer-model', state.status_bar.model)]
  if state.status_bar.reasoning_effort != 'none':
    segments.append(('class:footer', state.status_bar.reasoning_effort))
  if state.status_bar.resize_dirty:
    segments.insert(0, ('class:footer', 'Ctrl+L to redraw'))
  fragments: _StyleAndTextTuples = [
    ('class:footer', ' '),
    approval,
  ]
  for index, segment in enumerate(segments):
    fragments.append(('class:footer-divider', ' | ' if index == 0 else ' · '))
    fragments.append(segment)
  fragments.append(('class:footer', ' '))
  return fragments
