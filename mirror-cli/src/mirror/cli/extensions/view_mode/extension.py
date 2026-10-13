"""View controls and a color-coded mode badge at the right of the status bar."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import override

from prompt_toolkit.layout import containers, controls

from mirror.cli import actions, commands, extension, keymap, state

_MODES = {
  'detail': actions.ViewMode.TRANSCRIPT,
  'compact': actions.ViewMode.COMPACT,
  'focus': actions.ViewMode.EXTRA_COMPACT,
}
_BADGES = {
  actions.ViewMode.COMPACT: ('COMPACT', 'class:ui.view-compact'),
  actions.ViewMode.EXTRA_COMPACT: ('FOCUS', 'class:ui.view-focus'),
  actions.ViewMode.TRANSCRIPT: ('DETAIL', 'class:ui.view-detail'),
}


@dataclasses.dataclass(frozen=True)
class _State:
  pass


def _view(context: commands.CommandContext, args: str) -> actions.Action:
  del context
  token = '-'.join(args.strip().lower().replace('_', ' ').split())
  if not token:
    return actions.CycleViewMode()
  mode = _MODES.get(token)
  if mode is None:
    # Keep existing /view invocations and keybinding configurations compatible.
    try:
      mode = actions.ViewMode(token)
    except ValueError:
      return actions.ShowText(
        f'Unknown view mode: {args}. Modes: {", ".join(_MODES)}',
        style='ui.warning',
      )
  return actions.SetViewMode(mode)


class ViewModeExtension(extension.Extension[_State, None, None]):
  """Own the view command, shortcut and status badge for local and remote sessions."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/view-mode'

  @staticmethod
  @override
  def initial_state() -> _State:
    return _State()

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_RIGHT

  @override
  def render_order(self) -> int:
    return 100  # After the throughput and context widgets.

  @override
  def status_separator(self) -> str:
    return ' '

  @override
  def tips(self) -> tuple[str, ...]:
    return ('Use Ctrl+T to cycle through detail, compact, and focus views',)

  @override
  def keybindings(self) -> Sequence[keymap.Binding]:
    return (keymap.Binding(('c-t',), actions.CycleViewMode()),)

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'view',
        'Cycle view: detail (reasoning), compact, focus; Ctrl+T',
        _view,
        id=commands.CommandId.VIEW,
        arguments=commands.OptionalArguments('[mode]'),
        available_when_busy=True,
      ),
    )

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    del extension_state
    if global_state.session.pending_switch is not None:
      return None
    label, style = _BADGES[global_state.status_bar.view_mode]
    return containers.Window(
      controls.FormattedTextControl([(style, f' {label} ')]),
      dont_extend_height=True,
      dont_extend_width=True,
    )
