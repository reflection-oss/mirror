from __future__ import annotations

import asyncio
import dataclasses
import pathlib
from collections.abc import AsyncIterator
from typing import override
from unittest import mock

import pytest
import pytest_asyncio

from mirror import config
from mirror.cli import (
  actions,
  effects,
  extension,
  reducer,
  state,
  user_extensions,
  workspace_controller,
)
from mirror.cli.statsig import config as statsig_config
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event

_SOURCE = """from mirror.cli import extension
class Widget(extension.Extension[None, None, None]):
  @classmethod
  def id(cls):
    return 'clock'
"""


class _RecordingAnalytics(statsig_client.EventClient):
  """Capture events without transport, using the SDK's consent lifecycle."""

  def __init__(self) -> None:
    super().__init__(statsig_config.StatsigConfig('test-user', 'test-launch'))
    self.events: list[statsig_event.Event] = []

  @override
  def log_event(self, event: statsig_event.Event) -> None:
    if self.enabled:
      self.events.append(event)


@pytest_asyncio.fixture
async def analytics(
  monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[_RecordingAnalytics]:
  async with _RecordingAnalytics() as analytics:
    # The singleton is the capture boundary; no clients enter extension contexts.
    monkeypatch.setattr(statsig_client, 'get', lambda: analytics)
    await analytics.enable()
    yield analytics


@pytest.fixture
def settings(tmp_path: pathlib.Path) -> config.Config:
  return config.Config(tmp_path / 'config.json')


@pytest_asyncio.fixture
async def manager(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  settings: config.Config,
) -> AsyncIterator[user_extensions.UserExtensions]:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  bundled = tmp_path / 'bundled'
  for name, default_enabled in (('clock', True), ('dormant', False)):
    (bundled / name).mkdir(parents=True)
    (bundled / name / 'extension.py').write_text(
      _SOURCE.replace("'clock'", repr(name))
      + f'  def default_enabled(self): return {default_enabled!r}\n'
    )
  root = tmp_path / 'extensions'
  (root / 'mine').mkdir(parents=True)
  source = root / 'mine' / 'extension.py'
  source.write_text(_SOURCE.replace("'clock'", "'mine'"))
  settings.set_extension(str(source.resolve()), True)
  registry = extension.ExtensionRegistry()
  try:
    yield user_extensions.UserExtensions(
      registry,
      root,
      bundled=bundled,
      packaged=tmp_path / 'packaged',
      settings=settings,
    )
  finally:
    await registry.close()


@pytest.fixture
def context(tmp_path: pathlib.Path) -> extension.Context:
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'low', 'auto'),
    session=state.SessionState('session'),
    lifecycle=state.LifecycleState(active=True),
  )
  return extension.Context(
    extension.Workspace(tmp_path, remote=False),
    lambda _: None,
    lambda: current,
    lambda _: None,
    lambda _: asyncio.sleep(0),
    lambda name, arguments: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    # Inventory operations never use the agent-owning workspace service.
    workspace_controller=mock.Mock(
      spec_set=workspace_controller.WorkspaceController
    ),
    create_agent=lambda options, on_message, on_error: pytest.fail(
      'Unexpected agent creation'
    ),
  )


def _counts(
  enabled: int,
  disabled: int,
  hidden: int,
  external_enabled: int,
  external_disabled: int,
) -> list[statsig_event.Event]:
  return [
    statsig_event.Event('mirror_extension_enabled_count', value=enabled),
    statsig_event.Event('mirror_extension_disabled_count', value=disabled),
    statsig_event.Event('mirror_extension_hidden_count', value=hidden),
    statsig_event.Event(
      'mirror_extension_external_enabled_count', value=external_enabled
    ),
    statsig_event.Event(
      'mirror_extension_external_disabled_count', value=external_disabled
    ),
  ]


def _state(key: str, value: str) -> statsig_event.Event:
  return statsig_event.Event(
    'mirror_extension_state', value=key, metadata={'state': value}
  )


@pytest.mark.asyncio
async def test_start_reports_enablement_naming_only_trusted_extensions(
  manager: user_extensions.UserExtensions, analytics: _RecordingAnalytics
) -> None:
  await manager.start()

  assert analytics.events == [
    statsig_event.Event('mirror_extension_enabled_count', value=2),
    statsig_event.Event('mirror_extension_disabled_count', value=1),
    statsig_event.Event('mirror_extension_hidden_count', value=0),
    statsig_event.Event('mirror_extension_external_enabled_count', value=1),
    statsig_event.Event('mirror_extension_external_disabled_count', value=0),
    statsig_event.Event(
      'mirror_extension_state',
      value='builtin/clock',
      metadata={'state': 'enabled'},
    ),
    statsig_event.Event(
      'mirror_extension_state',
      value='builtin/dormant',
      metadata={'state': 'disabled'},
    ),
  ]


@pytest.mark.asyncio
async def test_enablement_is_not_reported_while_analytics_is_off(
  manager: user_extensions.UserExtensions, analytics: _RecordingAnalytics
) -> None:
  await analytics.disable()
  await manager.start()

  assert analytics.events == []


@pytest.mark.asyncio
async def test_set_enabled_reports_disabled_and_enabled_inventory(
  manager: user_extensions.UserExtensions,
  analytics: _RecordingAnalytics,
  tmp_path: pathlib.Path,
) -> None:
  await manager.start()
  analytics.events.clear()

  await manager.set_enabled(manager.root / 'mine', False)
  assert analytics.events == _counts(1, 2, 0, 0, 1) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
  ]
  analytics.events.clear()

  await manager.set_enabled(tmp_path / 'bundled' / 'dormant', True)
  assert analytics.events == _counts(2, 1, 0, 0, 1) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'enabled'),
  ]


@pytest.mark.asyncio
async def test_apply_settings_reports_packaged_and_external_hidden_inventory(
  manager: user_extensions.UserExtensions,
  analytics: _RecordingAnalytics,
  tmp_path: pathlib.Path,
) -> None:
  packaged = tmp_path / 'packaged' / 'spark'
  packaged.mkdir(parents=True)
  (packaged / 'extension.py').write_text(_SOURCE.replace('clock', 'spark'))
  await manager.start()
  assert analytics.events == _counts(3, 1, 0, 1, 0) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
    _state('packaged/spark', 'enabled'),
  ]
  analytics.events.clear()

  await manager.set_enabled(packaged, False)
  assert analytics.events == _counts(2, 2, 0, 1, 0) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
    _state('packaged/spark', 'disabled'),
  ]
  analytics.events.clear()

  await manager.apply_settings({
    'builtin/clock': {'enabled': 'hidden'},
    'packaged/spark': {'enabled': 'hidden'},
    str(manager.root / 'mine' / 'extension.py'): {'enabled': 'hidden'},
  })
  assert analytics.events == _counts(0, 1, 3, 0, 0) + [
    _state('builtin/clock', 'hidden'),
    _state('builtin/dormant', 'disabled'),
    _state('packaged/spark', 'hidden'),
  ]
  analytics.events.clear()

  await manager.apply_settings({'packaged/spark': True})
  assert analytics.events == _counts(2, 2, 0, 0, 1) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
    _state('packaged/spark', 'enabled'),
  ]


@pytest.mark.asyncio
async def test_change_reports_toggle_effect_not_workspace_focus(
  manager: user_extensions.UserExtensions,
  analytics: _RecordingAnalytics,
  context: extension.Context,
) -> None:
  await manager.start()
  analytics.events.clear()
  await manager.on_foreground(context)
  await manager.on_background(context)
  await manager.on_foreground(
    dataclasses.replace(
      context,
      workspace=extension.Workspace(
        context.workspace.cwd / 'other', remote=False
      ),
    )
  )
  assert analytics.events == []

  _, emitted = reducer.reduce(
    context.get_state(), actions.ToggleExtension('builtin/clock')
  )
  [effect] = emitted
  assert isinstance(effect, effects.ChangeExtension)
  await manager.change(context, effect.request)
  assert analytics.events == _counts(1, 2, 0, 1, 0) + [
    _state('builtin/clock', 'disabled'),
    _state('builtin/dormant', 'disabled'),
  ]


@pytest.mark.asyncio
async def test_change_reports_extension_consent(
  manager: user_extensions.UserExtensions,
  analytics: _RecordingAnalytics,
  settings: config.Config,
  context: extension.Context,
) -> None:
  settings.update(lambda values: values.update(extensions={}))
  await manager.start()
  assert analytics.events == _counts(1, 2, 0, 0, 1) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
  ]
  analytics.events.clear()

  await manager.change(
    context,
    actions.ExtensionConsent(str(manager.root / 'mine' / 'extension.py'), True),
  )
  assert analytics.events == _counts(2, 1, 0, 1, 0) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
  ]


@pytest.mark.asyncio
async def test_apply_settings_reports_current_inventory_when_analytics_reenabled(
  manager: user_extensions.UserExtensions,
  analytics: _RecordingAnalytics,
  settings: config.Config,
) -> None:
  await analytics.disable()
  await manager.start()
  await manager.set_enabled(manager.root / 'mine', False)
  assert analytics.events == []

  await analytics.enable()
  await manager.apply_settings(settings.read().get('extensions'))
  snapshot = _counts(1, 2, 0, 0, 1) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
  ]
  assert analytics.events == snapshot
  analytics.events.clear()

  await analytics.disable()
  await manager.apply_settings(settings.read().get('extensions'))
  assert analytics.events == []
  await analytics.enable()
  await manager.apply_settings(settings.read().get('extensions'))
  assert analytics.events == snapshot


@pytest.mark.asyncio
async def test_set_enabled_failure_does_not_pollute_later_inventory(
  manager: user_extensions.UserExtensions,
  analytics: _RecordingAnalytics,
  tmp_path: pathlib.Path,
) -> None:
  await manager.start()
  analytics.events.clear()
  dormant = tmp_path / 'bundled' / 'dormant'
  (dormant / 'extension.py').write_text('raise ValueError("broken extension")')
  with pytest.raises(extension.ExtensionLoadError, match='broken extension'):
    await manager.set_enabled(dormant, True)
  assert analytics.events == []

  await manager.set_enabled(manager.root / 'mine', False)
  assert analytics.events == _counts(1, 2, 0, 0, 1) + [
    _state('builtin/clock', 'enabled'),
    _state('builtin/dormant', 'disabled'),
  ]
