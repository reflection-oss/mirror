"""Catch failures in deferred extension painting and mouse callbacks."""

from __future__ import annotations

import functools
from typing import override

from prompt_toolkit import styles
from prompt_toolkit.layout import containers, dimension
from prompt_toolkit.layout import mouse_handlers as _mouse_handlers
from prompt_toolkit.layout import screen as _screen

from mirror.cli import extension_faults


class ExtensionContainer(containers.Container):
  """Keep broken widgets out of the shared terminal screen."""

  def __init__(
    self, child: containers.AnyContainer, boundary: extension_faults.Boundary
  ) -> None:
    self._child = containers.to_container(child)
    self._boundary = boundary

  @override
  def reset(self) -> None:
    self._boundary.call(self._child.reset, None)

  @override
  def preferred_width(self, max_available_width: int) -> dimension.Dimension:
    return self._boundary.call(
      self._child.preferred_width,
      dimension.Dimension.exact(0),
      max_available_width,
    )

  @override
  def preferred_height(
    self, width: int, max_available_height: int
  ) -> dimension.Dimension:
    return self._boundary.call(
      self._child.preferred_height,
      dimension.Dimension.exact(0),
      width,
      max_available_height,
    )

  @override
  def get_children(self) -> list[containers.Container]:
    # Layout traversal must not invoke the child's unguarded callbacks.
    return []

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
    if not self._boundary.active():
      return
    painted = _screen.Screen(
      initial_width=screen.width, initial_height=screen.height
    )
    handlers = _mouse_handlers.MouseHandlers()
    try:
      self._child.write_to_screen(
        painted, handlers, write_position, parent_style, erase_bg, z_index
      )
      painted.draw_all_floats()
      # Validate before copying: the terminal renderer parses inline colors later.
      validator = styles.Style([])
      checked: set[str] = set()
      for row in painted.data_buffer.values():
        for cell in row.values():
          if cell.style not in checked:
            validator.get_attrs_for_style_str(cell.style)
            checked.add(cell.style)
    except (
      Exception
    ) as error:  # A broken widget must not break the terminal renderer.
      self._boundary.fail(error)
      return
    top, bottom = (
      write_position.ypos,
      write_position.ypos + write_position.height,
    )
    left, right = (
      write_position.xpos,
      write_position.xpos + write_position.width,
    )
    for y in range(top, bottom):
      screen.data_buffer[y].update({
        x: cell
        for x, cell in painted.data_buffer[y].items()
        if left <= x < right
      })
      screen.zero_width_escapes[y].update({
        x: value
        for x, value in painted.zero_width_escapes[y].items()
        if left <= x < right
      })
      for x, handler in handlers.mouse_handlers[y].items():
        if left <= x < right:
          mouse_handlers.mouse_handlers[y][x] = functools.partial(
            self._boundary.call, handler, NotImplemented
          )
    screen.cursor_positions.update(painted.cursor_positions)
    screen.menu_positions.update(painted.menu_positions)
    screen.visible_windows_to_write_positions.update(
      painted.visible_windows_to_write_positions
    )
    screen.width = max(
      screen.width,
      min(painted.width, right),
    )
    screen.height = max(
      screen.height,
      min(painted.height, bottom),
    )
