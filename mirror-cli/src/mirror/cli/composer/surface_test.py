"""Composer framing preserves its input while reserving padding and footer rows."""

import pytest
from prompt_toolkit import buffer
from prompt_toolkit.layout import containers, controls, mouse_handlers, screen

from mirror.cli import themes
from mirror.cli.composer import surface


@pytest.mark.asyncio
@pytest.mark.parametrize('menu_visible', [False, True])
async def test_wrap_input_frames_multiline_input(menu_visible: bool) -> None:
  input_buffer = buffer.Buffer()
  input_buffer.text = 'first\nsecond'
  input_control = controls.BufferControl(buffer=input_buffer)
  input_window = containers.Window(
    input_control,
    dont_extend_height=True,
    wrap_lines=True,
  )
  framed = surface.wrap_input(
    input_window,
    lambda: menu_visible,
    containers.Window(controls.FormattedTextControl(' footer'), height=1),
  )
  height = 5 if menu_visible else 6
  target = screen.Screen()
  framed.write_to_screen(
    target,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, 12, height),
    '',
    True,
    None,
  )
  rows = [
    ''.join(target.data_buffer[row][column].char for column in range(12))
    for row in range(height)
  ]
  assert rows == ([] if menu_visible else [' ' * 12]) + [
    ' first      ',
    ' second     ',
    ' ' * 12,
    ' footer     ',
    '▀' * 12,
  ]
  assert framed.preferred_height(12, 20).preferred == height
  assert input_window.content is input_control
  assert input_buffer.text == 'first\nsecond'
  sheet = themes.prompt_style()
  assert (
    sheet.get_attrs_for_style_str(
      target.data_buffer[height - 5][0].style
    ).bgcolor
    == '282828'
  )
  assert (
    sheet.get_attrs_for_style_str(
      target.data_buffer[height - 1][0].style
    ).bgcolor
    == ''
  )


def test_wrap_input_reserves_horizontal_padding_when_wrapping() -> None:
  framed = surface.wrap_input(
    containers.Window(
      controls.FormattedTextControl('x' * 25),
      dont_extend_height=True,
      wrap_lines=True,
    ),
    lambda: False,
    containers.Window(height=1),
  )
  assert framed.preferred_height(10, 20).preferred == 8


def test_composer_styles_do_not_recolor_unfilled_content() -> None:
  sheet = themes.prompt_style()
  assert sheet.get_attrs_for_style_str('').bgcolor == ''
  assert sheet.get_attrs_for_style_str('class:composer').color == 'd3e3da'
  assert (
    sheet.get_attrs_for_style_str('class:composer class:footer').color
    == sheet.get_attrs_for_style_str('class:ui.muted').color
  )
  assert (
    sheet.get_attrs_for_style_str('class:composer class:footer-divider').color
    == sheet.get_attrs_for_style_str('class:ui.faint').color
  )
  assert (
    sheet.get_attrs_for_style_str(
      'class:composer class:skill-mention-input'
    ).color
    == sheet.get_attrs_for_style_str('class:ui.info').color
  )


def test_wrap_input_does_not_stretch_when_numbered_selector_hides_input() -> (
  None
):
  framed = surface.wrap_input(
    containers.Window(height=0),
    lambda: True,
    containers.Window(controls.FormattedTextControl(' footer'), height=1),
  )
  assert framed.preferred_height(80, 40).max == 3
  target = screen.Screen()
  framed.write_to_screen(
    target,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, 80, 40),
    '',
    True,
    None,
  )
  assert target.data_buffer[1][1].char == 'f'
  assert target.data_buffer[2][0].char == '▀'
