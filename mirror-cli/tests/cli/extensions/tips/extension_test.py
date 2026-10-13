from __future__ import annotations

import dataclasses
import time

import pytest
from freezegun import api as freezegun_api
from prompt_toolkit.layout import containers, mouse_handlers, screen

from mirror.cli import actions, extension_state, reducer, state, themes
from mirror.cli.extensions.tips import extension as tips_extension


def _application(
  plugin: tips_extension.TipsExtension,
  tips: tuple[str, ...] = ('First tip', 'Second tip'),
) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'approve'),
    session=state.SessionState('conversation', replaying=False),
    lifecycle=state.LifecycleState(active=True),
    extensions=extension_state.ExtensionStates({
      plugin.id(): plugin.initial_state()
    }),
    extension_tips=tips,
  )


def _dispatch(
  plugin: tips_extension.TipsExtension,
  current: state.State,
  action: actions.Action,
) -> state.State:
  """Apply extension and core reducers in the same order as the CLI."""
  updated, emitted = plugin.reduce(current, action)
  assert emitted == ()
  current, _ = reducer.reduce(current, action, observed_at=time.monotonic())
  return dataclasses.replace(
    current, extensions=current.extensions.updated(plugin.id(), updated)
  )


def _render(
  plugin: tips_extension.TipsExtension, current: state.State
) -> str | None:
  """Paint the public widget API without depending on its container structure."""
  rendered = plugin.render(current, plugin.require_state(current))
  if rendered is None:
    return None
  widget = containers.to_container(rendered)
  painted = screen.Screen()
  widget.write_to_screen(
    painted,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, 80, 1),
    '',
    True,
    None,
  )
  return ''.join(painted.data_buffer[0][i].char for i in range(80)).rstrip()


@pytest.mark.parametrize(
  'start', [actions.TurnStarted, actions.AgentTurnStarted]
)
def test_render_tip_waits_fifteen_seconds(
  start: type[actions.TurnStarted] | type[actions.AgentTurnStarted],
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _application(plugin, ('Only tip',))
  assert _render(plugin, current) is None
  current = _dispatch(plugin, current, start(observed_at=time.monotonic()))
  assert _render(plugin, current) is None

  frozen_ui_time.tick(14)
  assert _render(plugin, current) is None
  frozen_ui_time.tick(1)
  assert _render(plugin, current) == '  └ Tip: Only tip'


@pytest.mark.parametrize('name', themes.NAMES)
def test_render_tip_uses_the_selected_theme_muted_role(
  name: str, frozen_ui_time: freezegun_api.FrozenDateTimeFactory
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin, ('Only tip',)),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  frozen_ui_time.tick(15)
  rendered = plugin.render(current, plugin.require_state(current))
  assert rendered is not None
  painted = screen.Screen()
  containers.to_container(rendered).write_to_screen(
    painted,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, 80, 1),
    '',
    True,
    None,
  )
  sheet = themes.prompt_style(name)
  assert sheet.get_attrs_for_style_str(painted.data_buffer[0][6].style) == (
    sheet.get_attrs_for_style_str('class:ui.muted')
  )


def test_render_tip_stays_fixed_through_tool_steps(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  current = _dispatch(
    plugin, current, actions.UserMessageObserved('First prompt')
  )
  frozen_ui_time.tick(15)
  first = _render(plugin, current)
  assert first in ('  └ Tip: First tip', '  └ Tip: Second tip')

  frozen_ui_time.tick(60)
  current = _dispatch(
    plugin, current, actions.AgentTurnStarted(observed_at=time.monotonic())
  )
  current = _dispatch(plugin, current, actions.SetStatus('running tool…'))
  assert _render(plugin, current) == first
  frozen_ui_time.tick(600)
  assert _render(plugin, current) == first


def test_render_tip_finishes_without_a_completion_tip(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin, ('Only tip',)),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  frozen_ui_time.tick(15)
  assert _render(plugin, current) == '  └ Tip: Only tip'

  current = _dispatch(plugin, current, actions.SubmitInstruction('Next prompt'))
  current = _dispatch(plugin, current, actions.TurnFinished())
  assert current.activity.busy
  assert _render(plugin, current) is None
  frozen_ui_time.tick(60)
  assert _render(plugin, current) is None


def test_render_tip_changes_on_next_turn_without_repeating(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  frozen_ui_time.tick(15)
  first = _render(plugin, current)
  assert first is not None
  current = _dispatch(plugin, current, actions.TurnFinished())
  current = _dispatch(
    plugin, current, actions.AgentTurnStarted(observed_at=time.monotonic())
  )
  current = _dispatch(
    plugin, current, actions.UserMessageObserved('Next prompt')
  )
  assert _render(plugin, current) is None

  frozen_ui_time.tick(15)
  second = _render(plugin, current)
  assert second in ('  └ Tip: First tip', '  └ Tip: Second tip')
  assert second != first


def test_render_tip_restarts_delay_when_queued_input_is_consumed(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  current = _dispatch(
    plugin, current, actions.UserMessageObserved('First prompt')
  )
  frozen_ui_time.tick(15)
  first = _render(plugin, current)
  current = _dispatch(plugin, current, actions.SubmitInstruction('Next prompt'))
  assert _render(plugin, current) == first

  current = _dispatch(
    plugin, current, actions.UserMessageObserved('Next prompt')
  )
  current = _dispatch(
    plugin, current, actions.AgentTurnStarted(observed_at=time.monotonic())
  )
  assert _render(plugin, current) is None
  frozen_ui_time.tick(15)
  second = _render(plugin, current)
  assert second in ('  └ Tip: First tip', '  └ Tip: Second tip')
  assert second != first


@pytest.mark.parametrize(
  ('tips', 'expected'),
  [((), None), (('Only tip',), '  └ Tip: Only tip')],
)
def test_render_tip_handles_small_catalogs(
  tips: tuple[str, ...],
  expected: str | None,
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin, tips),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  frozen_ui_time.tick(15)
  assert _render(plugin, current) == expected
  current = _dispatch(plugin, current, actions.TurnFinished())
  current = _dispatch(
    plugin, current, actions.TurnStarted(observed_at=time.monotonic())
  )
  frozen_ui_time.tick(15)
  assert _render(plugin, current) == expected


def test_render_tip_hides_removed_contributions_until_next_turn(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(
    plugin,
    _application(plugin, ('Old tip',)),
    actions.TurnStarted(observed_at=time.monotonic()),
  )
  frozen_ui_time.tick(15)
  assert _render(plugin, current) == '  └ Tip: Old tip'

  current = dataclasses.replace(current, extension_tips=('New tip',))
  assert _render(plugin, current) is None
  current = _dispatch(plugin, current, actions.TurnFinished())
  current = _dispatch(
    plugin, current, actions.TurnStarted(observed_at=time.monotonic())
  )
  frozen_ui_time.tick(15)
  assert _render(plugin, current) == '  └ Tip: New tip'


def test_render_tip_ignores_replayed_turns(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  plugin = tips_extension.TipsExtension()
  current = _dispatch(plugin, _application(plugin), actions.SetReplaying(True))
  current = _dispatch(
    plugin, current, actions.AgentTurnStarted(observed_at=time.monotonic())
  )
  current = _dispatch(
    plugin, current, actions.UserMessageObserved('Old prompt')
  )
  frozen_ui_time.tick(15)
  assert _render(plugin, current) is None
  current = _dispatch(
    plugin, current, actions.ReplayFinished(has_user_message=True)
  )
  assert _render(plugin, current) is None
