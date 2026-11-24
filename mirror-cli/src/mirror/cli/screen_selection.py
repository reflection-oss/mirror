"""Read-only drag selection across a completed fullscreen frame."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterator
from typing import override

from prompt_toolkit import data_structures, mouse_events
from prompt_toolkit.application import current
from prompt_toolkit.key_binding import key_bindings
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.layout import mouse_handlers as _mouse_handlers
from prompt_toolkit.layout import screen as _screen

from mirror.cli import row_renderer, terminal


def _cell_changed(
  before: _screen.Char, after: _screen.Char, *, clicking: bool
) -> bool:
  if clicking:
    identity = next(
      (
        token
        for token in before.style.split()
        if token.startswith('class:click-target-')
      ),
      None,
    )
    # Feedback can change a button's glyph or color without replacing its action.
    if identity is not None and identity in after.style.split():
      return False
  return before.char != after.char or before.style.removesuffix(
    ' reverse'
  ) != after.style.removesuffix(' reverse')


class HoverControl(controls.UIControl):
  """A control notified when the pointer leaves, which prompt-toolkit omits."""

  def clear_hover(self) -> None:
    """Hide hover feedback without changing selection or content."""


class SelectionRegion(containers.DynamicContainer):
  """A rectangular boundary for mouse selection."""

  def __init__(self, content: containers.AnyContainer) -> None:
    container = containers.to_container(content)
    super().__init__(lambda: container)
    self._screen: _screen.Screen | None = None
    self._position: _screen.WritePosition | None = None

  def position_for(
    self, screen: _screen.Screen
  ) -> _screen.WritePosition | None:
    """Return the region's allocation only if it was drawn in this frame."""
    return self._position if self._screen is screen else None

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
    super().write_to_screen(
      screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
    )
    self._screen = screen
    self._position = write_position


def _same_position(
  first: _screen.WritePosition | None, second: _screen.WritePosition | None
) -> bool:
  return (
    first is not None
    and second is not None
    and (first.xpos, first.ypos, first.width, first.height)
    == (second.xpos, second.ypos, second.width, second.height)
  )


class ScreenSelection(containers.DynamicContainer):
  """Select displayed cells while retaining stationary clicks and wheel handlers."""

  scroll_handler: (
    Callable[[mouse_events.MouseEvent], key_bindings.NotImplementedOrNone]
    | None
  ) = None

  def __init__(
    self, content: containers.Container, copy: Callable[[str], None]
  ) -> None:
    super().__init__(lambda: content)
    self._copy = copy
    self._screen: _screen.Screen | None = None
    self._handlers = _mouse_handlers.MouseHandlers()
    self._position = _screen.WritePosition(0, 0, 0, 0)
    self._region: SelectionRegion | None = None
    self._selection_position: _screen.WritePosition | None = None
    self._snapshot: _screen.Screen | None = None
    self._pressed: mouse_events.MouseEvent | None = None
    self._dragged = False
    self._start: data_structures.Point | None = None
    self._end: data_structures.Point | None = None
    self._mouse_position: data_structures.Point | None = None
    self._pointer = False
    self._hover_controls: tuple[HoverControl, ...] = ()

  def _update_pointer(self) -> None:
    position = self._mouse_position
    active = bool(
      position is not None
      and self._screen is not None
      and 'class:mouse-pointer'
      in self._screen.data_buffer[position.y][position.x].style.split()
    )
    if active != self._pointer:
      self._pointer = active
      if sequence := terminal.mouse_pointer_sequence(active):
        output = current.get_app().output
        output.write_raw(sequence)
        output.flush()

  def _clear_hover(self) -> None:
    for control in self._hover_controls:
      control.clear_hover()

  def clear(self, *, keep_hover: bool = False) -> None:
    """End selection without changing the terminal clipboard."""
    if not keep_hover:
      self._clear_hover()
      self._mouse_position = None
      self._update_pointer()
    self._snapshot = None
    self._region = None
    self._selection_position = None
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
    child_handlers = row_renderer.MouseHandlers()
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
    if not _same_position(position, self._position) or (
      self._region is not None
      and not _same_position(
        self._region.position_for(target), self._selection_position
      )
    ):
      self.clear()
    self._position = position
    self._screen = target
    self._handlers = child_handlers
    hover_controls = tuple(
      window.content
      for window in target.visible_windows
      if isinstance(window.content, HoverControl)
    )
    for control in self._hover_controls:
      if control not in hover_controls:
        control.clear_hover()
    self._hover_controls = hover_controls
    snapshot = self._snapshot
    if snapshot is not None:
      # Styles also identify click targets whose labels can remain unchanged.
      if any(
        _cell_changed(
          snapshot.data_buffer[y][x],
          target.data_buffer[y][x],
          clicking=self._pressed is not None and not self._dragged,
        )
        for x, y in self._cells()
      ):
        self.clear()
      elif self._start != self._end:
        for x, y in self._cells():
          cell = target.data_buffer[y][x]
          target.data_buffer[y][x] = _screen.Char(
            cell.char, cell.style + ' reverse'
          )
    self._update_pointer()
    handlers.set_mouse_handler_for_range(
      position.xpos,
      position.xpos + position.width,
      position.ypos,
      position.ypos + position.height,
      self.mouse_handler,
    )

  def _capture(self) -> _screen.Screen | None:
    # Freeze the clicked area before later stream updates move the content.
    # The renderer can then reuse its buffers without changing selected text.
    if self._screen is None:
      return None
    snapshot = _screen.Screen()
    position = self._position
    for y in range(position.ypos, position.ypos + position.height):
      snapshot.data_buffer[y].update({
        x: cell
        for x, cell in self._screen.data_buffer[y].items()
        if position.xpos <= x < position.xpos + position.width
      })
    return snapshot

  def _cells(self) -> Iterator[tuple[int, int]]:
    if self._start is None or self._end is None:
      return
    start, end = sorted((self._start, self._end), key=lambda p: (p.y, p.x))
    position = self._selection_position or self._position
    for y in range(start.y, end.y + 1):
      first = start.x if y == start.y else position.xpos
      last = end.x + 1 if y == end.y else position.xpos + position.width
      for x in range(first, last):
        yield x, y

  def _point(self, point: data_structures.Point) -> data_structures.Point:
    position = self._selection_position or self._position
    x = max(position.xpos, min(point.x, position.xpos + position.width - 1))
    y = max(position.ypos, min(point.y, position.ypos + position.height - 1))
    # A wide glyph's continuation cell contains no text of its own.
    if self._snapshot is not None:
      while x > position.xpos and not self._snapshot.data_buffer[y][x].char:
        x -= 1
    return data_structures.Point(x, y)

  def text(self) -> str:
    """Return the text currently selected, or an empty string."""
    return self._text() if self._start != self._end else ''

  def _text(self) -> str:
    snapshot = self._snapshot
    if snapshot is None:
      return ''
    lines: dict[int, str] = {}
    for x, y in self._cells():
      lines[y] = lines.get(y, '') + snapshot.data_buffer[y][x].char
    return '\n'.join(line.rstrip() for line in lines.values())

  def _begin_selection(self, event: mouse_events.MouseEvent) -> None:
    """Anchor the drag to its starting region."""
    self.clear(keep_hover=True)
    self._snapshot = self._capture()
    self._pressed = event
    if self._screen is not None:
      for container in layout.walk(self):
        if not isinstance(container, SelectionRegion):
          continue
        position = container.position_for(self._screen)
        if (
          position is not None
          and position.xpos <= event.position.x < position.xpos + position.width
          and position.ypos
          <= event.position.y
          < position.ypos + position.height
        ):
          self._region = container
          self._selection_position = position
          break
    self._start = self._end = self._point(event.position)

  def mouse_handler(
    self, event: mouse_events.MouseEvent
  ) -> key_bindings.NotImplementedOrNone:
    """Defer a left click until release; a moved gesture selects instead."""
    if self.scroll_handler is not None and event.event_type in (
      mouse_events.MouseEventType.SCROLL_UP,
      mouse_events.MouseEventType.SCROLL_DOWN,
    ):
      self.clear()
      if self.scroll_handler(event) is not NotImplemented:
        # The scroll handler scheduled its damage; suppress the binding's redraw.
        return NotImplemented
    self._mouse_position = event.position
    if (
      event.modifiers
      or (self._pressed is not None and self._dragged)
      or (
        event.event_type == mouse_events.MouseEventType.MOUSE_MOVE
        and event.button != mouse_events.MouseButton.NONE
      )
    ):
      self._mouse_position = None
    self._update_pointer()
    if event.modifiers or event.button != mouse_events.MouseButton.LEFT:
      self._clear_hover()
      if event.event_type != mouse_events.MouseEventType.MOUSE_MOVE:
        self.clear()
      return self._handlers.mouse_handlers[event.position.y][event.position.x](
        event
      )
    if event.event_type == mouse_events.MouseEventType.MOUSE_DOWN:
      self._begin_selection(event)
    elif self._pressed is not None:
      self._end = self._point(event.position)
      self._dragged |= event.position != self._pressed.position
      if event.event_type == mouse_events.MouseEventType.MOUSE_UP:
        pressed = self._pressed
        self._pressed = None
        if self._start == self._end:
          dragged = self._dragged
          self.clear(keep_hover=True)
          if dragged:
            self._clear_hover()
            return None
          handler = self._handlers.mouse_handlers[pressed.position.y][
            pressed.position.x
          ]
          handler(pressed)
          return handler(event)
        text = self._text()
        if text:
          self._copy(text)
        self._clear_hover()
    return None
