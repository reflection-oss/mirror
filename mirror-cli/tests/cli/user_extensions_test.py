from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import pathlib
import sys
from unittest import mock

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers

from mirror import config
from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  state,
  user_extensions,
)
from mirror.cli.extensions.user_extensions import extension as user_extension


@pytest.fixture(autouse=True)
def enable_test_extensions(
  disable_extensions: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  del disable_extensions
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')


_SOURCE = """from mirror.cli import extension
class Widget(extension.Extension[None, None, None]):
  @classmethod
  def id(cls):
    return 'clock'
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('default_enabled', 'override', 'enabled', 'prefixes'),
  [
    (True, None, True, ['●']),
    (False, None, False, ['○']),
    (False, True, True, ['●']),
    (True, False, False, ['○']),
    ('hidden', None, False, []),
    ('hidden', True, True, ['●']),
    ('hidden', False, False, ['○']),
  ],
)
@pytest.mark.parametrize('source_kind', ['builtin', 'packaged'])
async def test_trusted_default_enabled_and_user_override(
  tmp_path: pathlib.Path,
  source_kind: str,
  default_enabled: bool | str,
  override: bool | None,
  enabled: bool,
  prefixes: list[str],
) -> None:
  bundled = tmp_path / 'bundled'
  directory = bundled / 'clock'
  directory.mkdir(parents=True)
  marker = tmp_path / 'loaded'
  (directory / 'extension.py').write_text(
    _SOURCE
    + f'  def default_enabled(self): return {default_enabled!r}\n'
    + '  async def on_load(self, context):\n'
    + '    from pathlib import Path\n'
    + f'    Path({str(marker)!r}).touch()\n'
  )
  settings = config.Config(tmp_path / 'config.json')
  if override is not None:
    settings.set_extension(f'{source_kind}/clock', override)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry,
    tmp_path / 'extensions',
    bundled=bundled if source_kind == 'builtin' else None,
    packaged=bundled if source_kind == 'packaged' else None,
    settings=settings,
  )
  try:
    await manager.start()
    assert bool(registry.extension_ids()) is enabled
    assert marker.exists() is enabled
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert [choice.prefix for choice in opened.selector.choices] == prefixes
    assert settings.path.exists() is (override is not None)
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('default_enabled', 'override', 'enabled', 'prefixes'),
  [
    (False, {}, False, ['○']),
    (True, {}, True, ['●']),
    (True, {'enabled': False}, False, ['○']),
    ('hidden', {}, False, []),
  ],
)
async def test_bundled_enablement_override_changes_apply_immediately(
  tmp_path: pathlib.Path,
  default_enabled: bool | str,
  override: config.JsonObject,
  enabled: bool,
  prefixes: list[str],
) -> None:
  bundled = tmp_path / 'bundled'
  directory = bundled / 'clock'
  directory.mkdir(parents=True)
  marker = tmp_path / 'unloaded'
  (directory / 'extension.py').write_text(
    _SOURCE
    + f'  def default_enabled(self): return {default_enabled!r}\n'
    + '  async def on_unload(self, context):\n'
    + '    from pathlib import Path\n'
    + f'    Path({str(marker)!r}).touch()\n'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.set_extension('builtin/clock', True)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry, tmp_path / 'extensions', bundled=bundled, settings=settings
  )
  try:
    await manager.start()
    assert registry.extension_ids() == ('clock',)
    settings.path.write_text(
      json.dumps({'extensions': {'builtin/clock': override}})
    )
    await manager.apply_settings(settings.read().get('extensions'))
    assert bool(registry.extension_ids()) is enabled
    assert marker.exists() is not enabled
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert [choice.prefix for choice in opened.selector.choices] == prefixes
    await manager.apply_settings(settings.read().get('extensions'))
    assert bool(registry.extension_ids()) is enabled
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_discovery_registers_and_unloads_opt_in_children(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  source = directory / 'extension.py'
  source.write_text(
    _SOURCE
    + '  def extensions(self): return (Child(),)\n'
    + 'class Child(extension.Extension[None, None, None]):\n'
    + '  global_ = False\n'
    + '  @classmethod\n'
    + '  def id(cls): return "review/indicator"\n'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.set_extension(str(source.resolve()), True)
  registry = extension.ExtensionRegistry()
  try:
    await user_extensions.UserExtensions(
      registry, root, settings=settings
    ).start()
    assert registry.extension_ids() == ('clock', 'review/indicator')
    await registry.unload('clock')
    assert registry.extension_ids() == ()
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('choice', 'enabled'), [('enable', True), ('disable', False)]
)
@pytest.mark.parametrize('foreground_first', [False, True])
async def test_start_remembers_selector_consent_without_prompting_again(
  tmp_path: pathlib.Path,
  capsys: pytest.CaptureFixture[str],
  choice: str,
  enabled: bool,
  foreground_first: bool,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  source = directory / 'extension.py'
  source.write_text(_SOURCE)
  dispatched: list[actions.Action] = []
  context = dataclasses.replace(_context(tmp_path), dispatch=dispatched.append)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  try:
    if foreground_first:
      await manager.on_foreground(context)
      assert not dispatched
    await manager.start()
    assert not registry.extension_ids()
    if not foreground_first:
      assert not dispatched
      await manager.on_foreground(context)
    opened = dispatched.pop()
    assert isinstance(opened, actions.OpenSelector)
    assert opened.selector.selected == 'disable'
    assert opened.selector.numbered
    assert str(source) in opened.selector.title
    assert 'unsandboxed' in opened.selector.title
    request = opened.selector.on_select(choice)
    assert isinstance(request, actions.ExtensionConsent)
    await manager.change(context, request)
    assert bool(registry.extension_ids()) is enabled
  finally:
    await registry.close()
  assert capsys.readouterr().err == ''
  assert config.extension_choices(
    config.Config(tmp_path / 'config.json').read().get('extensions')
  ) == {str(source): enabled}
  second = extension.ExtensionRegistry()
  try:
    manager = user_extensions.UserExtensions(second, root)
    await manager.start()
    dispatched.clear()
    await manager.on_foreground(context)
    assert bool(second.extension_ids()) is enabled
    assert dispatched == []
  finally:
    await second.close()


@pytest.mark.asyncio
async def test_escape_leaves_consent_undecided_and_prompts_next_extension(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'extensions'
  first = root / 'a'
  second = root / 'b'
  first.mkdir(parents=True)
  second.mkdir()
  (first / 'extension.py').write_text(
    'raise AssertionError("unapproved import")'
  )
  (second / 'extension.py').write_text(
    'raise AssertionError("unapproved import")'
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  dispatched: list[actions.Action] = []
  context = dataclasses.replace(_context(tmp_path), dispatch=dispatched.append)
  try:
    await manager.start()
    await manager.on_foreground(context)
    opened = dispatched.pop()
    assert isinstance(opened, actions.OpenSelector)
    assert str(first / 'extension.py') in opened.selector.title
    assert opened.selector.cancel_action is not None
    assert isinstance(opened.selector.cancel_action, actions.ExtensionConsent)
    await manager.change(context, opened.selector.cancel_action)
    opened = dispatched.pop()
    assert isinstance(opened, actions.OpenSelector)
    assert str(second / 'extension.py') in opened.selector.title
    assert not registry.extension_ids()
    assert not (tmp_path / 'config.json').exists()
    restarted = user_extensions.UserExtensions(registry, root)
    await restarted.start()
    await restarted.on_foreground(context)
    opened = dispatched.pop()
    assert isinstance(opened, actions.OpenSelector)
    assert str(first / 'extension.py') in opened.selector.title
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, 'hidden'])
async def test_disabled_and_hidden_external_settings_do_not_prompt(
  tmp_path: pathlib.Path,
  enabled: bool | str,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  (directory / 'extension.py').write_text(
    'raise AssertionError("disabled import")'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.path.write_text(
    json.dumps({
      'extensions': {
        str(directory / 'extension.py'): {'enabled': enabled},
        'builtin/clock': {'enabled': enabled},
      }
    })
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  dispatched: list[actions.Action] = []
  context = dataclasses.replace(_context(tmp_path), dispatch=dispatched.append)
  try:
    await manager.start()
    await manager.on_foreground(context)
    assert dispatched == []
    assert not registry.extension_ids()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_pending_consent_does_not_override_a_new_saved_decision(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  source = directory / 'extension.py'
  source.write_text('raise AssertionError("disabled import")')
  settings = config.Config(tmp_path / 'config.json')
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root, settings=settings)
  dispatched: list[actions.Action] = []
  context = dataclasses.replace(_context(tmp_path), dispatch=dispatched.append)
  try:
    await manager.start()
    await manager.on_foreground(context)
    opened = dispatched.pop()
    assert isinstance(opened, actions.OpenSelector)
    settings.set_extension(str(source), False)
    request = opened.selector.on_select('enable')
    assert isinstance(request, actions.ExtensionConsent)
    await manager.change(context, request)
    assert config.extension_choices(settings.read().get('extensions')) == {
      str(source): False
    }
    assert not registry.extension_ids()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_description_uses_docstring_without_import_then_loaded_override(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  marker = tmp_path / 'imported'
  (directory / 'extension.py').write_text(
    '"""Show the local clock."""\n'
    + _SOURCE
    + '  def description(self): return "Clock with live settings"\n'
    + f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  try:
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert opened.selector.choices[0].detail == 'Show the local clock.'
    assert not marker.exists()
    assert (
      await user_extension.UserExtensionsTool(manager).execute(
        '{"action":"load","name":"clock"}'
      )
    ).success
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert opened.selector.choices[0].detail == 'Clock with live settings'
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_bundled_extension_can_be_disabled_before_import(
  tmp_path: pathlib.Path,
) -> None:
  bundled = tmp_path / 'bundled'
  directory = bundled / 'clock'
  directory.mkdir(parents=True)
  (directory / 'extension.py').write_text(
    'raise AssertionError("disabled builtin was imported")'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.set_extension('builtin/clock', False)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry, tmp_path / 'extensions', bundled=bundled
  )
  await manager.start(trust_extensions=True)
  assert registry.extension_ids() == ()
  opened = manager.selector()
  assert isinstance(opened, actions.OpenSelector)
  assert opened.selector.choices[0].value == 'builtin/clock'
  assert opened.selector.choices[0].label == 'clock'
  assert opened.selector.choices[0].prefix == '○'


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('enabled', 'loaded', 'visible'),
  [(True, True, True), (False, False, True), ('hidden', False, False)],
)
@pytest.mark.parametrize('bundled', [False, True])
async def test_extension_states_control_execution_and_discovery(
  tmp_path: pathlib.Path,
  enabled: bool | str,
  loaded: bool,
  visible: bool,
  bundled: bool,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  marker = tmp_path / 'imported'
  (directory / 'extension.py').write_text(
    f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
    'from mirror.cli import actions, commands\n'
    + _SOURCE
    + '  def slash_commands(self):\n'
    '    return (commands.CommandDef("clock", "Clock", '
    'lambda context, args: actions.ShowText("tick")),)\n'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.path.write_text(
    json.dumps({
      'extensions': {
        str(directory / 'extension.py'): {'enabled': enabled},
        'builtin/clock': {'enabled': enabled},
      }
    })
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry, root, bundled=root if bundled else None
  )
  host = registry.create_host(_context(tmp_path))
  try:
    # Trust bypasses consent, never disabled/hidden defaults.
    await manager.start(trust_extensions=True)
    await host.start()
    assert marker.exists() is loaded
    assert bool(registry.extension_ids()) is loaded
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert bool(opened.selector.choices) is visible
    assert bool(commands.menu_commands('/', host.slash_commands)) is loaded
    assert (
      commands.command_named('clock', host.slash_commands) is not None
    ) is loaded
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('initial', 'updated', 'loaded', 'visible'),
  [
    (True, False, False, True),
    (True, 'hidden', False, False),
    (False, True, True, True),
    (False, 'hidden', False, False),
    ('hidden', True, True, True),
    ('hidden', False, False, True),
  ],
)
async def test_user_override_transitions_between_extension_states(
  tmp_path: pathlib.Path,
  initial: bool | str,
  updated: bool | str,
  loaded: bool,
  visible: bool,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  source = directory / 'extension.py'
  source.write_text(_SOURCE)
  settings = config.Config(tmp_path / 'config.json')
  settings.path.write_text(
    json.dumps({'extensions': {str(source): {'enabled': initial}}})
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root, settings=settings)
  try:
    await manager.start()
    settings.path.write_text(
      json.dumps({'extensions': {str(source): {'enabled': updated}}})
    )
    await manager.apply_settings(settings.read().get('extensions'))
    assert bool(registry.extension_ids()) is loaded
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert bool(opened.selector.choices) is visible
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_bundled_hidden_setting_disables_without_import(
  tmp_path: pathlib.Path,
) -> None:
  bundled = tmp_path / 'bundled'
  directory = bundled / 'clock'
  directory.mkdir(parents=True)
  (directory / 'extension.py').write_text(
    'raise AssertionError("disabled extension imported")'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.path.write_text(
    '{"extensions":{"builtin/clock":{"enabled":"hidden"}}}'
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry, tmp_path / 'extensions', bundled=bundled
  )
  await manager.start()
  assert registry.extension_ids() == ()


@pytest.mark.asyncio
async def test_noninteractive_unknown_extensions_are_not_approved(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  (directory / 'extension.py').write_text(_SOURCE)
  monkeypatch.setattr(sys, 'stdin', io.StringIO(''))
  registry = extension.ExtensionRegistry()
  await user_extensions.UserExtensions(registry, root).start()
  assert registry.extension_ids() == ()
  assert not config.Config(tmp_path / 'config.json').read().get('extensions')


@pytest.mark.asyncio
async def test_extensions_command_toggles_and_persists_without_importing_off_code(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  source = directory / 'extension.py'
  source.write_text(_SOURCE)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  dispatched: list[actions.Action] = []
  context = dataclasses.replace(_context(tmp_path), dispatch=dispatched.append)
  try:
    opened = manager.selector()
    assert isinstance(opened, actions.OpenSelector)
    assert opened.selector.choices[0].label == 'clock'
    assert opened.selector.choices[0].prefix == '○'
    action = opened.selector.on_select(str(source))
    assert isinstance(action, actions.ToggleExtension)
    await manager.change(context, action)
    assert registry.source_directory('clock') == directory
    refreshed = dispatched[-1]
    assert isinstance(refreshed, actions.OpenSelector)
    assert refreshed.selector.choices[0].label == 'clock'
    assert refreshed.selector.choices[0].prefix == '●'
    action = refreshed.selector.on_select(str(source))
    assert isinstance(action, actions.ToggleExtension)
    await manager.change(context, action)
    assert registry.extension_ids() == ()
    assert (
      config.extension_choices(
        config.Config(tmp_path / 'config.json').read().get('extensions')
      )[str(source)]
      is False
    )
    source.write_text('raise AssertionError("disabled code was imported")')
    await manager.start(trust_extensions=True)
    assert registry.extension_ids() == ()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_create_returns_a_working_live_widget(
  tmp_path: pathlib.Path,
) -> None:
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path / 'extensions')
  result = await user_extension.UserExtensionsTool(manager).execute(
    '{"action":"create","name":"clock"}'
  )
  path = tmp_path / 'extensions' / 'clock' / 'extension.py'
  assert result.success
  assert str(path) in result.content
  assert path.read_text() in result.content
  assert 'from typing import override' in path.read_text()
  assert path.read_text().count('@override') == 6
  assert registry.extension_ids() == ()
  assert (
    await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"load","name":"clock"}'
    )
  ).success
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'low', 'auto'),
    session=state.SessionState('session'),
    lifecycle=state.LifecycleState(active=True),
  )
  updated = asyncio.Event()

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    restored, emitted = host.reduce(current, action)
    current = dataclasses.replace(current, extensions=restored)
    for effect in emitted:
      assert isinstance(effect, effects.RunExtensionEffect)
      host.handle_effect(effect)
    if isinstance(action, actions.RoutedExtensionAction):
      updated.set()

  host = registry.create_host(
    extension.Context(
      extension.Workspace(tmp_path, remote=False),
      dispatch,
      lambda: current,
      lambda _: None,
      registry.refresh_tools,
      lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  current = dataclasses.replace(current, extensions=host.initial_states)
  try:
    await host.start()
    await asyncio.wait_for(updated.wait(), timeout=2)
    assert current.extensions['clock'] == 'Widget'
    widget = containers.to_container(
      host.render(current, extension.RenderRegion.WIDGET_BELOW_PROMPT)
    )
    assert widget.preferred_height(80, 20).preferred == 1
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_create_never_overwrites_existing_source(
  tmp_path: pathlib.Path,
) -> None:
  manager = user_extensions.UserExtensions(
    extension.ExtensionRegistry(), tmp_path
  )
  assert (
    await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"create","name":"clock"}'
    )
  ).success
  path = tmp_path / 'clock' / 'extension.py'
  path.write_text('existing user code')
  result = await user_extension.UserExtensionsTool(manager).execute(
    '{"action":"create","name":"clock"}'
  )
  assert not result.success
  assert path.read_text() == 'existing user code'


@pytest.mark.asyncio
async def test_user_extensions_start_reload_unload(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  source = directory / 'extension.py'
  definition = _SOURCE
  source.write_text(definition)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path)
  try:
    await manager.start(trust_extensions=True)
    listed = await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"list"}'
    )
    assert json.loads(listed.content) == {
      'clock': {'directory': str(directory), 'workspaces': []}
    }
    assert user_extension.UserExtensionsTool(manager).definition.needs_approval
    source.write_text(
      definition.replace('class Widget', 'class ReloadedWidget')
    )
    assert (
      await user_extension.UserExtensionsTool(manager).execute(
        '{"action":"reload","extension_id":"clock"}'
      )
    ).success
    assert registry.source_directory('clock') == directory
    assert (
      await user_extension.UserExtensionsTool(manager).execute(
        '{"action":"unload","extension_id":"clock"}'
      )
    ).success
    assert registry.extension_ids() == ()
    assert source.exists()
  finally:
    await registry.close()


def _context(cwd: pathlib.Path, *, remote: bool = False) -> extension.Context:
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'low', 'auto'),
    session=state.SessionState('session'),
    lifecycle=state.LifecycleState(active=True),
  )
  return extension.Context(
    extension.Workspace(cwd, remote=remote),
    lambda _: None,
    lambda: current,
    lambda _: None,
    lambda _: asyncio.sleep(0),
    lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )


@pytest.mark.asyncio
async def test_list_reads_workspace_health_and_reload_clears_failure(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  source = directory / 'extension.py'
  source.write_text(
    _SOURCE
    + """
  def handle_message(self, context, message):
    if context.workspace.cwd.name == 'broken':
      raise ValueError('quote failed')
"""
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path)
  await manager.start(trust_extensions=True)
  broken = registry.create_host(_context(tmp_path / 'broken'))
  healthy = registry.create_host(_context(tmp_path / 'healthy'))
  paused = registry.create_host(_context(tmp_path / 'paused'))
  remote = registry.create_host(_context(tmp_path / 'remote', remote=True))
  try:
    await broken.start()
    await healthy.start()
    await remote.start()
    broken.handle_message(agent_pb2.SessionMessage())
    listed = await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"list"}'
    )
    assert json.loads(listed.content) == {
      'clock': {
        'directory': str(directory),
        'workspaces': [
          {
            'cwd': str(tmp_path / 'broken'),
            'remote': False,
            'status': 'disabled',
            'error': 'ValueError: quote failed',
          },
          {
            'cwd': str(tmp_path / 'healthy'),
            'remote': False,
            'status': 'active',
            'error': None,
          },
          {
            'cwd': str(tmp_path / 'paused'),
            'remote': False,
            'status': 'not started',
            'error': None,
          },
          {
            'cwd': str(tmp_path / 'remote'),
            'remote': True,
            'status': 'active',
            'error': None,
          },
        ],
      },
    }
    source.write_text(_SOURCE)
    assert (
      await user_extension.UserExtensionsTool(manager).execute(
        '{"action":"reload","extension_id":"clock"}'
      )
    ).success
    reloaded = json.loads(
      (
        await user_extension.UserExtensionsTool(manager).execute(
          '{"action":"list"}'
        )
      ).content
    )
    assert reloaded['clock']['workspaces'][0] == {
      'cwd': str(tmp_path / 'broken'),
      'remote': False,
      'status': 'active',
      'error': None,
    }
    await broken.close()
    await healthy.close()
    remaining = json.loads(
      (
        await user_extension.UserExtensionsTool(manager).execute(
          '{"action":"list"}'
        )
      ).content
    )
    assert remaining['clock']['workspaces'] == [
      {
        'cwd': str(tmp_path / 'paused'),
        'remote': False,
        'status': 'not started',
        'error': None,
      },
      {
        'cwd': str(tmp_path / 'remote'),
        'remote': True,
        'status': 'active',
        'error': None,
      },
    ]
  finally:
    await asyncio.gather(
      broken.close(), healthy.close(), paused.close(), remote.close()
    )
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('replacement', 'diagnostic'),
  [
    ('invalid python!', 'SyntaxError: invalid syntax (extension.py, line 1)'),
    (
      _SOURCE.replace('clock', 'time'),
      "ValueError: Extension ID must match folder name 'clock'; got 'time'",
    ),
  ],
)
async def test_invalid_reload_preserves_loaded_extension(
  tmp_path: pathlib.Path,
  replacement: str,
  diagnostic: str,
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  source = directory / 'extension.py'
  source.write_text(_SOURCE)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path)
  try:
    await manager.start(trust_extensions=True)
    source.write_text(replacement)
    result = await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"reload","extension_id":"clock"}'
    )
    assert not result.success
    assert diagnostic in result.content
    assert registry.extension_ids() == ('clock',)
    assert registry.source_directory('clock') == directory
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('broken_source', 'diagnostic'),
  [
    ('invalid python!', 'SyntaxError: invalid syntax (extension.py, line 1)'),
    ('raise RuntimeError("broken import")', 'RuntimeError: broken import'),
    (
      _SOURCE
      + '  async def on_load(self, context): raise RuntimeError("broken setup")\n',
      'RuntimeError: broken setup',
    ),
  ],
)
async def test_load_returns_diagnostic_and_can_retry(
  tmp_path: pathlib.Path, broken_source: str, diagnostic: str
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  source = directory / 'extension.py'
  source.write_text(broken_source)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path)
  try:
    result = await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"load","name":"clock"}'
    )
    assert not result.success
    assert diagnostic in result.content
    assert not registry.extension_ids()

    source.write_text(_SOURCE)
    result = await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"load","name":"clock"}'
    )
    assert result.success
    assert registry.extension_ids() == ('clock',)
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['../outside', '/outside', '..'])
async def test_user_extensions_reject_external_paths(
  tmp_path: pathlib.Path, name: str
) -> None:
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path)
  result = await user_extension.UserExtensionsTool(manager).execute(
    json.dumps({'action': 'load', 'name': name})
  )
  assert not result.success
  assert registry.extension_ids() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('arguments', 'success'),
  [
    ('{"action":"load","name":"clock"}', True),
    ('{"action":"reload","extension_id":"clock"}', False),
    ('{"action":"unload","extension_id":"clock"}', False),
  ],
)
async def test_user_extensions_can_override_but_not_manage_bundled_source(
  tmp_path: pathlib.Path,
  arguments: str,
  success: bool,
) -> None:
  bundled = tmp_path / 'bundled' / 'clock'
  bundled.mkdir(parents=True)
  (bundled / 'extension.py').write_text(_SOURCE)
  root = tmp_path / 'user'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  (directory / 'extension.py').write_text(_SOURCE)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  try:
    await registry.load(bundled)
    result = await user_extension.UserExtensionsTool(manager).execute(arguments)
    assert result.success is success
    assert registry.extension_ids() == ('clock',)
    assert registry.source_directory('clock') == (
      directory if success else bundled
    )
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_user_extensions_reject_source_symlink(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'extensions'
  directory = root / 'clock'
  directory.mkdir(parents=True)
  outside = tmp_path / 'outside.py'
  outside.write_text(_SOURCE)
  (directory / 'extension.py').symlink_to(outside)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  await manager.start()
  assert registry.extension_ids() == ()
  assert not (
    await user_extension.UserExtensionsTool(manager).execute(
      '{"action":"load","name":"clock"}'
    )
  ).success


@pytest.mark.asyncio
async def test_packaged_respects_extensions_disabled(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  packaged = tmp_path / 'packaged'
  (packaged / 'clock').mkdir(parents=True)
  (packaged / 'clock' / 'extension.py').write_text(
    _SOURCE
    + '  def extensions(self): return (Child(),)\n'
    + 'class Child(extension.Extension[None, None, None]):\n'
    + '  global_ = False\n'
    + '  @classmethod\n'
    + '  def id(cls): return "clock/indicator"\n'
  )
  settings = config.Config(tmp_path / 'config.json')
  settings.set_extension('packaged/clock', True)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry,
    tmp_path / 'user',
    packaged=packaged,
    settings=settings,
  )
  try:
    await manager.start()
    assert registry.extension_ids() == ('clock', 'clock/indicator')
    monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'false')
    await manager.apply_settings(settings.read().get('extensions'))
    assert manager.selector().selector.choices == ()
    assert registry.extension_ids() == ()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_folder_precedence_is_resolved_before_import(
  tmp_path: pathlib.Path,
) -> None:
  packaged = tmp_path / 'packaged'
  user = tmp_path / 'user'
  repo = tmp_path / 'repo'
  cwd = repo / 'nested'
  cwd.mkdir(parents=True)
  (repo / '.jj').mkdir()
  ancestor = repo / '.mirror' / 'extensions'
  nearest = cwd / '.mirror' / 'extensions'
  marker = tmp_path / 'shadowed-import'
  # Even import-time effects from losing sources must never run.
  for root in (packaged, user, ancestor):
    (root / 'clock').mkdir(parents=True)
    (root / 'clock' / 'extension.py').write_text(
      f'from pathlib import Path\nPath({str(marker)!r}).touch()\n'
    )
  selected = nearest / 'clock'
  selected.mkdir(parents=True)
  (selected / 'extension.py').write_text(_SOURCE)
  settings = config.Config(tmp_path / 'config.json')
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry,
    user,
    cwd=cwd,
    packaged=packaged,
    settings=settings,
  )
  try:
    await manager.start()
    assert not marker.exists()
    assert registry.extension_ids() == ()
    assert [choice.value for choice in manager.selector().selector.choices] == [
      str(selected / 'extension.py')
    ]
    await manager.set_enabled(selected, True)
    assert registry.source_directory('clock') == selected
    with pytest.raises(ValueError, match='shadowed'):
      await manager.set_enabled(user / 'clock', True)
    assert registry.source_directory('clock') == selected
    await manager.set_enabled(selected, False)
    await manager.start(trust_extensions=True)
    assert registry.extension_ids() == ()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_local_override_replaces_packaged_before_activation_and_can_be_removed(
  tmp_path: pathlib.Path,
) -> None:
  packaged = tmp_path / 'packaged'
  (packaged / 'clock').mkdir(parents=True)
  (packaged / 'clock' / 'extension.py').write_text(_SOURCE)
  root = tmp_path / 'user'
  settings = config.Config(tmp_path / 'config.json')
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry,
    root,
    packaged=packaged,
    settings=settings,
  )
  try:
    await manager.start()
    assert registry.source_directory('clock') == packaged / 'clock'
    local = root / 'clock'
    local.mkdir(parents=True)
    (local / 'extension.py').write_text(_SOURCE)
    with pytest.raises(ValueError, match='shadowed'):
      await manager.reload('clock')
    await manager.apply_settings(settings.read().get('extensions'))
    assert registry.extension_ids() == ()
    await manager.set_enabled(local, True)
    assert registry.source_directory('clock') == local
    (local / 'extension.py').unlink()
    await manager.apply_settings(settings.read().get('extensions'))
    assert registry.source_directory('clock') == packaged / 'clock'
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_folder_id_cannot_impersonate_another_extension(
  tmp_path: pathlib.Path,
) -> None:
  clock = tmp_path / 'clock'
  other = tmp_path / 'other'
  clock.mkdir()
  other.mkdir()
  (clock / 'extension.py').write_text(_SOURCE)
  marker = tmp_path / 'activated'
  (other / 'extension.py').write_text(
    _SOURCE
    + '  async def on_load(self, context):\n'
    + '    from pathlib import Path\n'
    + f'    Path({str(marker)!r}).touch()\n'
  )
  registry = extension.ExtensionRegistry()
  try:
    await registry.load(clock)
    with pytest.raises(
      extension.ExtensionLoadError, match='must match folder name'
    ):
      await registry.load(other)
    assert registry.source_directory('clock') == clock
    assert not marker.exists()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_extension_skills_follow_enablement_reload_and_workspace(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  (directory / 'extension.py').write_text(
    'import pathlib\n'
    + _SOURCE
    + '  global_ = False\n'
    + '  def default_enabled(self): return False\n'
    + '  def skills(self):\n'
    + '    return (pathlib.Path(__file__).with_name("SKILL.md"),)\n'
  )
  path = directory / 'SKILL.md'
  path.write_text('---\nname: clock\ndescription: Clock widget.\n---\nOriginal')
  registry = extension.ExtensionRegistry()
  local = registry.create_host(_context(tmp_path), extensions=('clock',))
  unselected = registry.create_host(_context(tmp_path))
  remote = registry.create_host(
    _context(tmp_path, remote=True), extensions=('clock',)
  )
  try:
    assert await registry.load(directory) is None
    assert local.skills == ()
    await registry.load(directory, overrides={'enabled': True})
    assert len(local.skills) == 1
    assert local.skills[0].path == path.resolve()
    assert local.skills[0].body == 'Original'
    assert unselected.skills == ()
    assert remote.skills == ()

    path.write_text(
      '---\nname: clock\ndescription: Clock widget.\n---\nUpdated'
    )
    await registry.reload('clock')
    assert local.skills[0].body == 'Updated'
    await registry.configure('clock', {'enabled': False})
    assert local.skills == ()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_invalid_extension_skill_fails_loading(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  (directory / 'extension.py').write_text(
    'import pathlib\n'
    + _SOURCE
    + '  def skills(self):\n'
    + '    return (pathlib.Path(__file__).with_name("SKILL.md"),)\n'
  )
  registry = extension.ExtensionRegistry()
  try:
    with pytest.raises(
      extension.ExtensionLoadError, match='Invalid extension skill'
    ):
      await registry.load(directory)
    assert registry.extension_ids() == ()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_authoring_extension_contributes_its_packaged_skill_only_when_enabled(
  tmp_path: pathlib.Path,
) -> None:
  registry = extension.ExtensionRegistry()
  host = registry.create_host(_context(tmp_path))
  try:
    assert not await registry.register(user_extension.UserExtensions(), None)
    assert host.skills == ()
    await registry.register(
      user_extension.UserExtensions(), None, overrides={'enabled': True}
    )
    assert len(host.skills) == 1
    skill = host.skills[0]
    assert skill.name == 'mirror-extensions'
    assert 'Mirror CLI' in skill.description
    assert 'user-extensions__extensions' in skill.body
    assert skill.path.is_file()
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_discovered_extension_skills_follow_source_precedence_and_consent(
  tmp_path: pathlib.Path,
) -> None:
  packaged = tmp_path / 'packaged'
  bundled = packaged / 'clock'
  bundled.mkdir(parents=True)
  source = (
    'import pathlib\n'
    + _SOURCE
    + '  def skills(self):\n'
    + '    return (pathlib.Path(__file__).with_name("SKILL.md"),)\n'
  )
  (bundled / 'extension.py').write_text(source)
  (bundled / 'SKILL.md').write_text(
    '---\nname: clock\ndescription: Packaged clock.\n---\nPackaged'
  )
  root = tmp_path / 'extensions'
  settings = config.Config(tmp_path / 'config.json')
  registry = extension.ExtensionRegistry()
  host = registry.create_host(_context(tmp_path))
  manager = user_extensions.UserExtensions(
    registry, root, packaged=packaged, settings=settings
  )
  try:
    await manager.start()
    assert len(host.skills) == 1
    assert host.skills[0].body == 'Packaged'

    user = root / 'clock'
    user.mkdir(parents=True)
    (user / 'extension.py').write_text(source)
    (user / 'SKILL.md').write_text(
      '---\nname: clock\ndescription: User clock.\n---\nUser'
    )
    await manager.start()
    assert host.skills == ()
    settings.set_extension(str(user / 'extension.py'), True)
    await manager.apply_settings(settings.read().get('extensions'))
    assert len(host.skills) == 1
    assert host.skills[0].path == (user / 'SKILL.md').resolve()
    assert host.skills[0].body == 'User'

    settings.set_extension(str(user / 'extension.py'), False)
    await manager.apply_settings(settings.read().get('extensions'))
    assert host.skills == ()
  finally:
    await registry.close()
