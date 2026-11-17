"""Static status-bar component for the terminal UI."""

from __future__ import annotations

from typing import override

from prompt_toolkit.formatted_text import (
  StyleAndTextTuples as _StyleAndTextTuples,
)
from prompt_toolkit.layout import containers, controls

from mirror.cli import state as _state
from mirror.cli.components import component, layout


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
  def cache_key(self, state: _state.State) -> object | None:
    left = self._extension_left.cache_key(state)
    right = self._extension_right.cache_key(state)
    if left is None or right is None:
      return None
    return (
      state.status_bar.model,
      state.status_bar.reasoning_effort,
      state.status_bar.speed,
      state.status_bar.approval_mode,
      state.session.pending_switch,
      state.session.read_only,
      left,
      right,
    )

  @override
  def render(self, state: _state.State) -> containers.AnyContainer:
    """Render the complete status-bar layout from application state."""
    return layout.Columns(
      [
        containers.Window(
          controls.FormattedTextControl(fragments(state)),
          style='class:footer',
          dont_extend_height=True,
          dont_extend_width=True,
        ),
        self._extension_left.render(state),
        containers.Window(),
        self._extension_right.render(state),
      ],
      retain=self.cache_key(state) is not None,
    )


def fragments(state: _state.State) -> _StyleAndTextTuples:
  """Render model and approval settings as status-bar fragments."""
  if state.session.read_only:
    return [('class:footer', ' Read-only · Esc parent · PgUp/PgDn scroll ')]
  if state.session.pending_switch is not None:
    return []
  approval = {
    'approve': ('class:footer-approval-manual', '🛠  manual'),
    'bypass': ('class:footer-approval-bypass', '🛠  bypass'),
  }.get(
    state.status_bar.approval_mode,
    ('class:footer-approval-auto', '🛠  auto'),
  )
  segments = [('class:footer class:footer-model', state.status_bar.model)]
  if state.status_bar.reasoning_effort != 'none':
    segments.append(('class:footer', state.status_bar.reasoning_effort))
  if state.status_bar.speed == 'fast':
    segments.append(('class:footer', 'fast'))
  fragments: _StyleAndTextTuples = [
    ('class:footer', ' '),
    approval,
  ]
  for index, segment in enumerate(segments):
    fragments.append(('class:footer-divider', ' | ' if index == 0 else ' · '))
    fragments.append(segment)
  fragments.append(('class:footer', ' '))
  return fragments
