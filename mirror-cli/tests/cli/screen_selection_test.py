"""Screen selection preserves clicks, displayed text and Unicode across widgets."""

from collections.abc import Iterator

import pytest
from prompt_toolkit import data_structures, mouse_events
from prompt_toolkit.application import application, current
from prompt_toolkit.input import defaults
from prompt_toolkit.output import base
from prompt_toolkit.layout import (
  containers,
  controls,
  layout,
  mouse_handlers,
  screen,
)

from mirror.cli import screen_selection


@pytest.fixture(autouse=True)
def _application_context() -> Iterator[None]:
  with defaults.create_pipe_input() as pipe:
    app = application.Application[str | None](
      input=pipe, output=base.DummyOutput()
    )
    with current.set_app(app):
      yield


def _paint(
  selection: screen_selection.ScreenSelection, width: int = 12
) -> screen.Screen:
  app = current.get_app()
  app.layout = layout.Layout(selection)
  app.render_counter += 1
  app.layout.update_parents_relations()
  frame = screen.Screen()
  selection.write_to_screen(
    frame,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, width, 3),
    '',
    False,
    None,
  )
  frame.draw_all_floats()
  return frame


def _event(
  kind: mouse_events.MouseEventType,
  x: int,
  y: int,
  button: mouse_events.MouseButton = mouse_events.MouseButton.LEFT,
) -> mouse_events.MouseEvent:
  return mouse_events.MouseEvent(
    data_structures.Point(x, y), kind, button, frozenset()
  )


@pytest.mark.parametrize('reverse', [False, True])
def test_drag_copies_displayed_unicode_across_regions(reverse: bool) -> None:
  copied: list[str] = []
  selection = screen_selection.ScreenSelection(
    containers.HSplit([
      containers.Window(controls.FormattedTextControl('alpha界🙂'), height=1),
      containers.Window(controls.FormattedTextControl('composer'), height=1),
      containers.Window(controls.FormattedTextControl('git main'), height=1),
    ]),
    copied.append,
  )
  _paint(selection)
  start, end = ((7, 2), (0, 0)) if reverse else ((0, 0), (7, 2))
  selection.mouse_handler(
    _event(mouse_events.MouseEventType.MOUSE_DOWN, *start)
  )
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_MOVE, *end))
  selected = _paint(selection)
  assert 'reverse' in selected.data_buffer[1][2].style
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_UP, *end))
  assert copied == ['alpha界🙂\ncomposer\ngit main']


def test_stationary_click_reaches_existing_handler_without_copying() -> None:
  clicks: list[mouse_events.MouseEventType] = []
  copied: list[str] = []

  def click(event: mouse_events.MouseEvent) -> None:
    clicks.append(event.event_type)

  selection = screen_selection.ScreenSelection(
    containers.Window(controls.FormattedTextControl([('', 'button', click)])),
    copied.append,
  )
  _paint(selection)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_DOWN, 2, 0))
  assert not clicks
  _paint(selection)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_UP, 2, 0))
  assert clicks == [
    mouse_events.MouseEventType.MOUSE_DOWN,
    mouse_events.MouseEventType.MOUSE_UP,
  ]
  assert not copied


def test_wheel_retains_existing_handler_and_clears_selection() -> None:
  events: list[mouse_events.MouseEventType] = []
  copied: list[str] = []

  def record(event: mouse_events.MouseEvent) -> None:
    events.append(event.event_type)

  selection = screen_selection.ScreenSelection(
    containers.Window(controls.FormattedTextControl([('', 'history', record)])),
    copied.append,
  )
  _paint(selection)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_DOWN, 0, 0))
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_MOVE, 3, 0))
  selection.mouse_handler(
    _event(
      mouse_events.MouseEventType.SCROLL_DOWN,
      2,
      0,
      mouse_events.MouseButton.NONE,
    )
  )
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_UP, 3, 0))
  assert events == [mouse_events.MouseEventType.SCROLL_DOWN]
  assert not copied


def test_drag_returning_to_its_origin_does_not_activate_a_button() -> None:
  clicks: list[mouse_events.MouseEventType] = []
  copied: list[str] = []

  def click(event: mouse_events.MouseEvent) -> None:
    clicks.append(event.event_type)

  selection = screen_selection.ScreenSelection(
    containers.Window(controls.FormattedTextControl([('', 'button', click)])),
    copied.append,
  )
  _paint(selection)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_DOWN, 2, 0))
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_MOVE, 4, 0))
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_UP, 2, 0))
  assert not clicks
  assert not copied


@pytest.mark.parametrize('resize', [False, True])
def test_changed_frame_cancels_selection_without_copying(resize: bool) -> None:
  text = ['original']
  copied: list[str] = []
  selection = screen_selection.ScreenSelection(
    containers.Window(controls.FormattedTextControl(lambda: text[0])),
    copied.append,
  )
  _paint(selection)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_DOWN, 0, 0))
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_MOVE, 3, 0))
  text[0] = 'original' if resize else 'changed'
  _paint(selection, width=10 if resize else 12)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_UP, 3, 0))
  assert not copied


def test_selection_uses_completed_floating_menu_instead_of_covered_text() -> (
  None
):
  copied: list[str] = []
  selection = screen_selection.ScreenSelection(
    containers.FloatContainer(
      containers.Window(controls.FormattedTextControl('underlying')),
      floats=[
        containers.Float(
          containers.Window(controls.FormattedTextControl('POPUP')),
          top=1,
          left=1,
          width=5,
          height=1,
        )
      ],
    ),
    copied.append,
  )
  _paint(selection)
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_DOWN, 1, 1))
  selection.mouse_handler(_event(mouse_events.MouseEventType.MOUSE_UP, 5, 1))
  assert copied == ['POPUP']
