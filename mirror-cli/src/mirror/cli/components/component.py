"""Declarative component bridge for prompt_toolkit."""

from __future__ import annotations

import abc
import types
from collections.abc import Callable, Mapping

from prompt_toolkit.application import current
from prompt_toolkit.layout import containers

from mirror.cli import state


class Component(abc.ABC):
  """A stateless terminal component rendered from application state."""

  @abc.abstractmethod
  def render(self, state: state.State) -> containers.AnyContainer:
    """Render the component from one immutable state snapshot."""


class ComponentCollection:
  """The fixed set of declarative components owned by one REPL."""

  def __init__(self, items: Mapping[str, Component]) -> None:
    self._items = types.MappingProxyType(dict(items))

  def __getitem__(self, name: str) -> Component:
    return self._items[name]


class ComponentHost:
  """Mount stateless components into prompt_toolkit's retained layout."""

  def __init__(self, get_state: Callable[[], state.State]) -> None:
    self._get_state = get_state

  def mount(self, component: Component) -> containers.AnyContainer:
    """Adapt one declarative component to a retained container."""
    rendered_state: state.State | None = None
    render_counter = -1
    rendered_app: object | None = None
    rendered: containers.Container | None = None

    def get_container() -> containers.Container:
      nonlocal rendered_state, render_counter, rendered, rendered_app
      snapshot = self._get_state()
      app = current.get_app()
      counter = app.render_counter
      if not app.is_running and counter == 0:
        return containers.to_container(component.render(snapshot))
      # Layout walks, measurement and painting must share one component tree.
      # State changes also invalidate between frames (e.g. modal key routing).
      if (
        rendered is None
        or app is not rendered_app
        or snapshot is not rendered_state
        or counter != render_counter
      ):
        rendered = containers.to_container(component.render(snapshot))
        rendered_state, render_counter, rendered_app = snapshot, counter, app
      return rendered

    return containers.DynamicContainer(get_container)
