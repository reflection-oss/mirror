from __future__ import annotations

from collections.abc import Callable
from typing import Never, override

import pytest
from prompt_toolkit.layout import containers, controls, mouse_handlers, screen
from prompt_toolkit.layout import screen as _screen

from mirror.cli import extension_faults, extension_rendering


def _fail() -> Never:
  raise ValueError('broken widget callback')


def _invalid_style() -> containers.Window:
  return containers.Window(
    controls.FormattedTextControl([('fg:ansibrightwhite', 'invalid')]), height=1
  )


@pytest.mark.parametrize(
  'make_container',
  [
    pytest.param(lambda: containers.Window(width=_fail), id='width'),
    pytest.param(lambda: containers.Window(height=_fail), id='height'),
    pytest.param(
      lambda: containers.Window(controls.FormattedTextControl(_fail)),
      id='lazy-content',
    ),
    pytest.param(_invalid_style, id='inline-style'),
    pytest.param(
      lambda: containers.FloatContainer(
        containers.Window(controls.FormattedTextControl('partial output')),
        [containers.Float(_invalid_style(), top=0, left=0, width=7, height=1)],
      ),
      id='deferred-float',
    ),
  ],
)
def test_failed_widget_never_contaminates_shared_screen(
  make_container: Callable[[], containers.Container],
) -> None:
  errors: list[Exception] = []
  boundary = extension_faults.Boundary(errors.append)
  widget = extension_rendering.ExtensionContainer(make_container(), boundary)
  sibling = extension_rendering.ExtensionContainer(
    containers.Window(controls.FormattedTextControl('sibling')), boundary
  )
  target = screen.Screen()
  target.data_buffer[0][0] = screen.Char('P', 'class:prompt')
  handlers = mouse_handlers.MouseHandlers()
  position = screen.WritePosition(2, 0, 20, 1)
  widget.preferred_width(20)
  widget.preferred_height(20, 1)
  widget.write_to_screen(target, handlers, position, '', True, None)
  widget.write_to_screen(target, handlers, position, '', True, None)
  sibling.write_to_screen(target, handlers, position, '', True, None)
  assert len(errors) == 1
  assert dict(target.data_buffer[0]) == {0: screen.Char('P', 'class:prompt')}


def test_valid_widget_preserves_its_content_and_styles() -> None:
  errors: list[Exception] = []
  widget = extension_rendering.ExtensionContainer(
    containers.Window(
      controls.FormattedTextControl([('fg:ansiwhite', 'OK')]), height=1
    ),
    extension_faults.Boundary(errors.append),
  )
  target = screen.Screen()
  widget.write_to_screen(
    target,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, 2, 1),
    '',
    True,
    None,
  )
  assert target.data_buffer[0][0].char + target.data_buffer[0][1].char == 'OK'
  assert 'fg:ansiwhite' in target.data_buffer[0][0].style
  assert errors == []


class _Overflow(containers.Window):
  @override
  def write_to_screen(
    self,
    screen: screen.Screen,
    mouse_handlers: mouse_handlers.MouseHandlers,
    write_position: screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    super().write_to_screen(
      screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
    )
    screen.data_buffer[0][0] = _screen.Char('X')
    screen.data_buffer[1000][1000] = _screen.Char('X')
    screen.zero_width_escapes[1000][1000] = '\x1b[0m'
    screen.width = screen.height = 10000


def test_widget_overflow_does_not_overwrite_or_expand_the_terminal() -> None:
  errors: list[Exception] = []
  child = _Overflow(controls.FormattedTextControl('OK'))
  widget = extension_rendering.ExtensionContainer(
    child, extension_faults.Boundary(errors.append)
  )
  output = screen.Screen(initial_width=20, initial_height=30)
  output.data_buffer[0][0] = _screen.Char('G')
  widget.write_to_screen(
    output,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(xpos=4, ypos=8, width=3, height=2),
    '',
    False,
    None,
  )
  assert output.width == 20
  assert output.height == 30
  assert output.data_buffer[0][0].char == 'G'
  assert output.data_buffer[8][4].char == 'O'
  assert 1000 not in output.data_buffer
  assert 1000 not in output.zero_width_escapes
  assert not errors
