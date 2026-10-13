"""Read-only drag selection across a completed fullscreen frame."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterator
from typing import override

from prompt_toolkit import data_structures, mouse_events
from prompt_toolkit.key_binding import key_bindings
from prompt_toolkit.layout import containers
from prompt_toolkit.layout import mouse_handlers as _mouse_handlers
from prompt_toolkit.layout import screen as _screen


class ScreenSelection(containers.DynamicContainer):
  """Select displayed cells while retaining stationary clicks and wheel handlers."""

  def __init__(
    self, content: containers.Container, copy: Callable[[str], None]
  ) -> None:
    super().__init__(lambda: content)
    self._copy = copy
    self._screen: _screen.Screen | None = None
    self._handlers = _mouse_handlers.MouseHandlers()
    self._position = _screen.WritePosition(0, 0, 0, 0)
    self._snapshot: _screen.Screen | None = None
    self._pressed: mouse_events.MouseEvent | None = None
    self._dragged = False
    self._start: data_structures.Point | None = None
    self._end: data_structures.Point | None = None

  def clear(self) -> None:
    """End selection without changing the terminal clipboard."""
    self._snapshot = None
    self._pressed = None
    self._dragged = False
    self._start = self._end = None

  @override
  def reset(self) -> None:
    self.clear()
    super().reset()

  @override
  def write_to_screen(
    self,
    screen: _screen.Screen,
    mouse_handlers: _mouse_handlers.MouseHandlers,
    write_position: _screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    child_handlers = _mouse_handlers.MouseHandlers()
    super().write_to_screen(
      screen, child_handlers, write_position, parent_style, erase_bg, z_index
    )
    # Include floating menus before capturing text or installing mouse routing.
    screen.draw_with_z_index(
      sys.maxsize,
      functools.partial(
        self._finish_frame,
        screen,
        mouse_handlers,
        child_handlers,
        write_position,
      ),
    )

  def _finish_frame(
    self,
    target: _screen.Screen,
    handlers: _mouse_handlers.MouseHandlers,
    child_handlers: _mouse_handlers.MouseHandlers,
    position: _screen.WritePosition,
  ) -> None:
    if (position.width, position.height) != (
      self._position.width,
      self._position.height,
    ):
      self.clear()
    self._position = position
    self._screen = target
    self._handlers = child_handlers
    snapshot = self._snapshot
    if snapshot is not None:
      # A changed cell must not silently turn an old selection into new output.
      if any(
        target.data_buffer[y][x].char != snapshot.data_buffer[y][x].char
        for x, y in self._cells()
      ):
        self.clear()
      elif self._start != self._end:
        for x, y in self._cells():
          cell = target.data_buffer[y][x]
          target.data_buffer[y][x] = _screen.Char(
            cell.char, cell.style + ' reverse'
          )
    handlers.set_mouse_handler_for_range(
      position.xpos,
      position.xpos + position.width,
      position.ypos,
      position.ypos + position.height,
      self.mouse_handler,
    )

  def _cells(self) -> Iterator[tuple[int, int]]:
    if self._start is None or self._end is None:
      return
    start, end = sorted((self._start, self._end), key=lambda p: (p.y, p.x))
    for y in range(start.y, end.y + 1):
      first = start.x if y == start.y else self._position.xpos
      last = (
        end.x + 1 if y == end.y else self._position.xpos + self._position.width
      )
      for x in range(first, last):
        yield x, y

  def _point(self, point: data_structures.Point) -> data_structures.Point:
    position = self._position
    x = max(position.xpos, min(point.x, position.xpos + position.width - 1))
    y = max(position.ypos, min(point.y, position.ypos + position.height - 1))
    # A wide glyph's continuation cell contains no text of its own.
    if self._snapshot is not None:
      while x > position.xpos and not self._snapshot.data_buffer[y][x].char:
        x -= 1
    return data_structures.Point(x, y)

  def _text(self) -> str:
    snapshot = self._snapshot
    if snapshot is None:
      return ''
    lines: dict[int, str] = {}
    for x, y in self._cells():
      lines[y] = lines.get(y, '') + snapshot.data_buffer[y][x].char
    return '\n'.join(line.rstrip() for line in lines.values())

  def mouse_handler(
    self, event: mouse_events.MouseEvent
  ) -> key_bindings.NotImplementedOrNone:
    """Defer a left click until release; a moved gesture selects instead."""
    if event.modifiers or event.button != mouse_events.MouseButton.LEFT:
      if event.event_type != mouse_events.MouseEventType.MOUSE_MOVE:
        self.clear()
      return self._handlers.mouse_handlers[event.position.y][event.position.x](
        event
      )
    if event.event_type == mouse_events.MouseEventType.MOUSE_DOWN:
      self.clear()
      self._snapshot = self._screen
      self._pressed = event
      self._start = self._end = self._point(event.position)
    elif self._pressed is not None:
      self._end = self._point(event.position)
      self._dragged |= self._start != self._end
      if event.event_type == mouse_events.MouseEventType.MOUSE_UP:
        pressed = self._pressed
        self._pressed = None
        if self._start == self._end:
          dragged = self._dragged
          self.clear()
          if dragged:
            return None
          handler = self._handlers.mouse_handlers[pressed.position.y][
            pressed.position.x
          ]
          handler(pressed)
          return handler(event)
        text = self._text()
        if text:
          self._copy(text)
    return None
