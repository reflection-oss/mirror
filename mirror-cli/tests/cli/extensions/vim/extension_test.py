"""The Vim extension owns its mode widget and native editor appearance."""

import pytest
from prompt_toolkit import application, buffer, cursor_shapes, enums, layout
from prompt_toolkit.input import base as input_base
from prompt_toolkit.key_binding import vi_state
from prompt_toolkit.layout import containers, controls
from prompt_toolkit.output import base as output_base

from mirror.cli import state, themes
from mirror.cli.extensions.vim import extension as vim_extension


def _state() -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('test', 'medium', 'auto'),
    session=state.SessionState('test'),
    lifecycle=state.LifecycleState(active=True),
  )


@pytest.mark.parametrize(
  ('mode', 'label', 'role', 'shape'),
  [
    (
      vi_state.InputMode.INSERT,
      ' INSERT ',
      'accent',
      cursor_shapes.CursorShape.BEAM,
    ),
    (
      vi_state.InputMode.NAVIGATION,
      ' NORMAL ',
      'info',
      cursor_shapes.CursorShape.BLOCK,
    ),
    (
      vi_state.InputMode.REPLACE,
      ' REPLACE ',
      'error',
      cursor_shapes.CursorShape.UNDERLINE,
    ),
  ],
)
def test_widget_and_cursor_follow_live_mode(
  mode: vi_state.InputMode,
  label: str,
  role: str,
  shape: cursor_shapes.CursorShape,
) -> None:
  plugin = vim_extension.VimExtension()
  app = application.Application[str | None](
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
    editing_mode=enums.EditingMode.VI,
  )
  app.vi_state.input_mode = mode
  editor = plugin.prompt_editor()
  widget = plugin.render(_state(), plugin.initial_state())
  assert isinstance(widget, containers.Window)
  assert isinstance(widget.content, controls.FormattedTextControl)
  with application.set_app(app):
    fragments = widget.content.create_content(80, 1).get_line(0)
    assert fragments[0] == (f'class:ui.{role} bold reverse', label)
    assert editor.cursor is not None
    assert editor.cursor.get_cursor_shape(app) == shape
    app.editing_mode = enums.EditingMode.EMACS
    assert callable(widget.content.text)
    assert widget.content.text() == []


def test_visual_selection_uses_colored_background_not_cursor_reverse() -> None:
  plugin = vim_extension.VimExtension()
  composer = buffer.Buffer()
  composer.text = 'hello'
  composer.start_selection()
  app = application.Application[str | None](
    layout=layout.Layout(layout.Window(layout.BufferControl(composer))),
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
    editing_mode=enums.EditingMode.VI,
  )
  app.vi_state.input_mode = vi_state.InputMode.NAVIGATION
  editor = plugin.prompt_editor()
  assert editor.style is None
  attrs = themes.prompt_style().get_attrs_for_style_str('class:selected')
  assert attrs.bgcolor == '1e3207'
  assert attrs.color == 'ffffff'
  assert not attrs.reverse
  widget = plugin.render(_state(), plugin.initial_state())
  assert isinstance(widget, containers.Window)
  assert isinstance(widget.content, controls.FormattedTextControl)
  with application.set_app(app):
    assert widget.content.create_content(80, 1).get_line(0)[0] == (
      'class:ui.selected',
      ' VISUAL ',
    )
