"""Fullscreen workspace columns with extension sidebars and shared navigation."""

import dataclasses
from collections.abc import Callable

from prompt_toolkit import data_structures, filters
from prompt_toolkit.layout import containers

from mirror.cli import extension, screen_selection, state
from mirror.cli.components import component

_MAX_SIDEBAR_PERCENT = 80


class Frame:
  """Lay out extension sidebars, the conversation, and workspace navigation."""

  def __init__(
    self,
    extensions: extension.ExtensionView,
    get_state: Callable[[], state.State],
    navigation: containers.AnyContainer,
    *,
    get_size: Callable[[], data_structures.Size],
  ) -> None:
    host = component.ComponentHost(get_state)
    self._left = containers.to_container(
      host.mount(extensions.component(extension.RenderRegion.SIDEBAR_LEFT))
    )
    self._right = containers.to_container(
      host.mount(extensions.component(extension.RenderRegion.SIDEBAR_RIGHT))
    )
    self._navigation = containers.to_container(navigation)
    self._get_state = get_state
    self._get_size = get_size
    self._source: state.State | None = None
    self._projected: state.State | None = None

  def _widths(self, columns: int) -> tuple[int, int]:
    available = columns * _MAX_SIDEBAR_PERCENT // 100
    if available <= 0:
      return 0, 0
    left = self._left.preferred_width(columns).preferred
    right = self._right.preferred_width(columns).preferred
    total = left + right
    if total > available:
      left = available * left // total
      right = available - left
    return left, right

  def _left_width(self) -> int:
    return self._widths(self._get_size().columns)[0]

  def _right_width(self) -> int:
    return self._widths(self._get_size().columns)[1]

  def content_size(self) -> data_structures.Size:
    """Return the center's allocation, excluding sidebars and navigation."""
    size = self._get_size()
    navigation = self._navigation.preferred_height(
      size.columns, size.rows
    ).preferred
    return data_structures.Size(
      rows=max(0, size.rows - navigation),
      columns=self._content_columns(size.columns),
    )

  def _content_columns(self, columns: int) -> int:
    left, right = self._widths(columns)
    return max(1, columns - left - right)

  def content_state(self) -> state.State:
    """Project layout dimensions without changing the shared state."""
    source = self._get_state()
    columns = self._content_columns(source.layout.columns)
    if columns == source.layout.columns:
      return source
    if (
      source is not self._source
      or self._projected is None
      or self._projected.layout.columns != columns
    ):
      self._source = source
      self._projected = dataclasses.replace(
        source, layout=dataclasses.replace(source.layout, columns=columns)
      )
    return self._projected

  def wrap(self, content: containers.AnyContainer) -> containers.Container:
    """Place sidebars around the conversation and navigation below."""
    columns = containers.VSplit(
      [
        _sidebar(self._left, self._left_width),
        screen_selection.SelectionRegion(
          containers.HSplit(
            [content], width=lambda: self.content_size().columns
          )
        ),
        _sidebar(self._right, self._right_width),
      ],
      height=lambda: self.content_size().rows,
    )
    return containers.HSplit(
      [columns, screen_selection.SelectionRegion(self._navigation)],
      style='class:theme-root',
    )


def _sidebar(
  content: containers.Container, width: Callable[[], int]
) -> containers.ConditionalContainer:
  return containers.ConditionalContainer(
    screen_selection.SelectionRegion(containers.HSplit([content], width=width)),
    filter=filters.Condition(lambda: width() > 0),
  )
