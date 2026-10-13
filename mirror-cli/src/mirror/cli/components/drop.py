"""Stateless worktree-drop confirmation component."""

from __future__ import annotations

from typing import override

from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import state
from mirror.cli.components import component, menu


class DropConfirmation(component.Component):
  """Confirmation prompt for deleting an unmanaged worktree."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    confirmation = state.drop_confirmation
    panel = (
      None
      if confirmation is None
      else menu.ConfirmPanel(
        title=f'Delete unmanaged worktree {confirmation.label}?',
        lines=(
          confirmation.path,
          'The checkout and all uncommitted changes will be deleted.',
        ),
        hint='y delete · n/Esc cancel',
      )
    )
    return containers.Window(
      controls.FormattedTextControl([] if panel is None else panel.render()),
      height=dimension.Dimension.exact(0 if panel is None else panel.height),
    )
