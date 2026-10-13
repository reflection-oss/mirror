"""Vi-style prompt editing using prompt-toolkit."""

from typing import override

from prompt_toolkit import (
  application,
  cursor_shapes,
  enums,
)
from prompt_toolkit.key_binding import vi_state
from prompt_toolkit.layout import containers, controls

from mirror.cli import extension, state
from mirror.cli.composer import editing

_CURSOR = cursor_shapes.ModalCursorShapeConfig()
_MODE_STYLES = {
  'NORMAL': 'class:ui.info bold reverse',
  'INSERT': 'class:ui.accent bold reverse',
  'VISUAL': 'class:ui.selected',
  'REPLACE': 'class:ui.error bold reverse',
  'OPERATOR': 'class:ui.warning bold reverse',
}


def _status() -> list[tuple[str, str]]:
  mode = indicator(application.get_app())
  return [(_MODE_STYLES[mode], f' {mode} ')] if mode else []


class VimExtension(extension.Extension[tuple[()], None, None]):
  """A stateless widget; prompt-toolkit owns the actual editing state."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/vim'

  @override
  def default_enabled(self) -> bool:
    return False

  @override
  def prompt_editor(self) -> editing.Editor:
    return editing.Editor(
      enums.EditingMode.VI,
      cursor=_CURSOR,
    )

  @staticmethod
  @override
  def initial_state() -> tuple[()]:
    return ()

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_LEFT

  @override
  def render(
    self,
    global_state: state.State,
    extension_state: tuple[()],
  ) -> containers.AnyContainer | None:
    if global_state.session.pending_switch is not None:
      return None
    return containers.Window(
      controls.FormattedTextControl(_status),
      style='class:footer',
      dont_extend_height=True,
      dont_extend_width=True,
    )


def indicator(app: application.Application[str | None]) -> str:
  """Return the live native mode for the status widget."""
  if app.editing_mode != enums.EditingMode.VI:
    return ''
  if app.current_buffer.selection_state is not None:
    return 'VISUAL'
  if app.vi_state.operator_func is not None:
    return 'OPERATOR'
  match app.vi_state.input_mode:
    case vi_state.InputMode.NAVIGATION:
      return 'NORMAL'
    case vi_state.InputMode.REPLACE | vi_state.InputMode.REPLACE_SINGLE:
      return 'REPLACE'
    case _:
      return 'INSERT'
