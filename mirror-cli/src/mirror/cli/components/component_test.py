"""A component tree is shared by layout walks and painting within a frame."""

import cProfile
import dataclasses
from typing import override

from prompt_toolkit import application
from prompt_toolkit.input import base as input_base
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import base as output_base

from mirror.cli import actions, rendering, state
from mirror.cli.components import banner, component


class CountingComponent(component.Component):
  def __init__(self) -> None:
    self.calls = 0

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    self.calls += 1
    return containers.Window(controls.FormattedTextControl(state.composer.text))


def _state() -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('fixture', 'medium', 'auto'),
    session=state.SessionState('fixture'),
    lifecycle=state.LifecycleState(active=True),
  )


def test_mount_reuses_tree_until_state_or_frame_changes() -> None:
  snapshot = _state()
  widget = CountingComponent()
  mounted = containers.to_container(
    component.ComponentHost(lambda: snapshot).mount(widget)
  )
  app = application.Application[None](
    layout=layout.Layout(mounted),
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  widget.calls = 0
  with application.set_app(app):
    app.render_counter += 1
    first = mounted.get_children()[0]
    mounted.preferred_width(120)
    mounted.preferred_height(120, 40)
    list(app.layout.walk())
    assert mounted.get_children()[0] is first
    assert widget.calls == 1

    # Even an equal-but-distinct snapshot invalidates without a deep comparison.
    snapshot = dataclasses.replace(snapshot)
    assert mounted.get_children()[0] is not first
    assert widget.calls == 2

    # Clock-driven activity and extension widgets refresh without a state update.
    app.render_counter += 1
    mounted.preferred_height(120, 40)
    assert widget.calls == 3


def test_mount_updates_composer_snapshot_before_next_frame() -> None:
  snapshot = _state()
  mounted = containers.to_container(
    component.ComponentHost(lambda: snapshot).mount(CountingComponent())
  )
  first = mounted.get_children()[0]
  snapshot = dataclasses.replace(
    snapshot, composer=actions.ComposerState('new draft', 9)
  )
  updated = mounted.get_children()[0]
  assert updated is not first
  assert isinstance(updated, containers.Window)
  assert updated.content.create_content(120, 40).get_line(0) == [
    ('', 'new draft')
  ]


def test_hidden_banner_layout_walk_does_not_render_rich() -> None:
  mounted = containers.to_container(
    component.ComponentHost(_state).mount(
      banner.Banner('/synthetic', lambda _width, _height: 0)
    )
  )
  hidden = containers.ConditionalContainer(mounted, filter=False)
  with cProfile.Profile() as profile:
    list(layout.walk(hidden))
    hidden.preferred_height(120, 40)
  calls = [
    entry
    for entry in profile.getstats()
    if entry.code == rendering.render_banner.__code__
  ]
  assert calls == [], (
    'Hidden banners must not render during focus/key-binding walks'
  )


def test_mount_does_not_share_a_cached_tree_between_applications() -> None:
  snapshot = _state()
  widget = CountingComponent()
  mounted = containers.to_container(
    component.ComponentHost(lambda: snapshot).mount(widget)
  )
  first = application.Application[None](
    input=input_base.DummyInput(), output=output_base.DummyOutput()
  )
  second = application.Application[None](
    input=input_base.DummyInput(), output=output_base.DummyOutput()
  )
  first.render_counter = second.render_counter = 1
  with application.set_app(first):
    tree = mounted.get_children()[0]
  with application.set_app(second):
    assert mounted.get_children()[0] is not tree


def test_mount_outside_an_application_keeps_external_diagnostics_fresh() -> (
  None
):
  widget = CountingComponent()
  snapshot = _state()
  mounted = containers.to_container(
    component.ComponentHost(lambda: snapshot).mount(widget)
  )
  first = mounted.get_children()[0]
  assert mounted.get_children()[0] is not first
