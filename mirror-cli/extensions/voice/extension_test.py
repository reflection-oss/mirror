from __future__ import annotations

import pathlib
from unittest import mock

import pytest

from mirror import config
from mirror.cli import (
  actions,
  extension,
  extension_sources,
  keymap,
  user_extensions,
)
from mirror.cli.extensions.voice import extension as voice


@pytest.mark.asyncio
async def test_installed_voice_is_optional_and_loading_never_installs(
  tmp_path: pathlib.Path,
) -> None:
  bundled = pathlib.Path(voice.__file__).resolve().parent.parent
  project = bundled / 'voice'
  assert (project / 'uv.lock').is_file()
  assert (project / 'pyproject.toml').is_file()
  assert (project / 'worker.py').is_file()
  settings = config.Config(tmp_path / 'config.json')
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry, tmp_path / 'extensions', bundled=bundled, settings=settings
  )
  with mock.patch(
    'asyncio.create_subprocess_exec',
    side_effect=AssertionError('Unexpected subprocess'),
  ):
    try:
      await manager.start()
      assert 'core/voice' not in registry.extension_ids()
      settings.set_extension('builtin/voice', True)
      await manager.apply_settings(settings.read().get('extensions'))
      assert 'core/voice' in registry.extension_ids()
    finally:
      await registry.close()


@pytest.mark.asyncio
async def test_first_use_requests_separate_consent_without_installing() -> None:
  context = mock.Mock(
    spec=extension.Context, get_state=mock.Mock(), dispatch=mock.Mock()
  )
  context.get_state.return_value.lifecycle.active = True
  widget = voice.Voice()
  with mock.patch(
    'asyncio.create_subprocess_exec',
    side_effect=AssertionError('Unexpected subprocess'),
  ):
    await widget.handle_effect(context, 'toggle')
  opened = context.dispatch.call_args.args[0]
  assert isinstance(opened, actions.OpenSelector)
  assert opened.selector.id == 'voice-consent'
  assert opened.selector.selected == 'cancel'
  assert isinstance(
    opened.selector.on_select('cancel'), actions.RerenderRequested
  )
  assert isinstance(
    opened.selector.on_select('prepare'), actions.RoutedExtensionAction
  )
  assert 'install and run locked audio dependencies' in opened.selector.title


def test_rebinding_preserves_extension_filter_and_replay() -> None:
  widget = voice.Voice()
  bindings = widget.keybindings()
  wildcard = bindings[2]
  assert wildcard.action is not None
  rebound = keymap.resolve(
    (bindings,),
    (keymap.Remapping(wildcard.action, (('f7',),)),),
  )
  binding = next(item for item in rebound if item.keys == ('f7',))
  assert binding.filter is wildcard.filter
  assert binding.replay


@pytest.mark.asyncio
async def test_github_source_acquisition_does_not_install_dependencies(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  source = tmp_path / 'voice-example'
  source.mkdir()
  (source / 'extension.py').write_text('raise AssertionError("Not approved")\n')
  (source / 'pyproject.toml').write_text('must not be resolved\n')
  settings = config.Config(tmp_path / 'config.json')
  settings.set_extension('https://github.com/example/voice-example', False)
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry, tmp_path / 'extensions', settings=settings
  )
  with (
    mock.patch.object(
      extension_sources.Repository, 'install', return_value=source
    ),
    mock.patch(
      'asyncio.create_subprocess_exec',
      side_effect=AssertionError('Unexpected subprocess'),
    ),
    mock.patch(
      'subprocess.run', side_effect=AssertionError('Unexpected subprocess')
    ),
  ):
    try:
      await manager.start(trust_extensions=True)
      assert not registry.extension_ids()
    finally:
      await registry.close()
