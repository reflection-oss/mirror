"""Stateless worker-failure and connection-recovery component."""

from __future__ import annotations

from typing import override

from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import state
from mirror.cli.components import component


class Failure(component.Component):
  @override
  def cache_key(self, state: state.State) -> object:
    return (
      state.lifecycle.failure,
      state.lifecycle.disconnect_reason,
      state.lifecycle.synchronizing,
    )

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    error = state.lifecycle.failure
    reason = state.lifecycle.disconnect_reason
    if reason is not None:
      status = (
        'Synchronizing session state…'
        if state.lifecycle.synchronizing
        else 'Connection lost. Reconnecting…'
      )
      if error is not None:
        status = "Couldn't reconnect. Press Enter to retry."
      return containers.Window(
        controls.FormattedTextControl([
          (
            'class:menu-hint',
            status,
          )
        ]),
        height=dimension.Dimension(max=2),
        wrap_lines=True,
      )
    elif error is not None:
      message = f'Agent error: {error}'
      hint = 'Press Enter to retry.'
    else:
      return containers.Window(height=0)
    return containers.HSplit([
      containers.Window(
        controls.FormattedTextControl([
          (
            'class:worktree-failed',
            message,
          )
        ]),
        height=dimension.Dimension(max=3),
        wrap_lines=True,
      ),
      containers.Window(
        controls.FormattedTextControl([('class:menu-hint', hint)]),
        height=dimension.Dimension(max=2),
        wrap_lines=True,
      ),
    ])
