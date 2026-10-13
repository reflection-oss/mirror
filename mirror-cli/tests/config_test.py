from __future__ import annotations

import asyncio
import json
import os
import pathlib

import pytest

from typing import cast

from mirror import config


def test_extension_config_merges_defaults_without_mutating_them() -> None:
  defaults: config.JsonObject = {
    'display': {'color': 'blue', 'limit': 10},
    'models': ['default'],
    'timeout': 5,
  }
  overrides: config.JsonObject = {
    'enabled': True,
    'config': {
      'display': {'limit': 0},
      'models': [],
      'timeout': None,
      'unknown': False,
    },
  }
  effective = config.resolve_extension(overrides, defaults)
  assert effective.enabled is True
  assert effective.config == {
    'display': {'color': 'blue', 'limit': 0},
    'models': [],
    'timeout': None,
    'unknown': False,
  }
  assert defaults['timeout'] == 5
  assert defaults['display'] == {'color': 'blue', 'limit': 10}
  defaults['new_setting'] = 42
  assert (
    config.resolve_extension(overrides, defaults).config['new_setting'] == 42
  )
  assert 'new_setting' not in overrides


@pytest.mark.parametrize(
  'overrides',
  [
    {'enabled': 'false'},
    {'enabled': 1},
    {'enabled': None},
    {'enabled': []},
    {'hidden': True},
    {'config': []},
  ],
)
def test_extension_config_rejects_invalid_core_settings(
  overrides: config.JsonObject,
) -> None:
  with pytest.raises(ValueError):
    config.resolve_extension(overrides)


def test_config_comparison_preserves_types_and_ignores_object_key_order() -> (
  None
):
  assert not config.same_config({'value': True}, {'value': 1})
  assert config.same_config({'a': 1, 'b': 2}, {'b': 2, 'a': 1})


def test_extension_config_snapshots_do_not_share_nested_values() -> None:
  defaults: config.JsonObject = {'nested': {'list': [1]}}
  merged = config.merge_config(defaults, {})
  nested = merged['nested']
  assert isinstance(nested, dict)
  nested['list'] = [2]
  assert defaults == {'nested': {'list': [1]}}


@pytest.mark.parametrize(
  ('value', 'expected'),
  [
    ({}, False),
    ({'checkpoints_enabled': True}, True),
    ({'checkpoints_enabled': False}, False),
    ({'checkpoints_enabled': None}, False),
    ({'checkpoints_enabled': 'true'}, False),
    ({'checkpoints_enabled': 'false'}, False),
    ({'checkpoints_enabled': 1}, False),
    ({'checkpoints_enabled': 0}, False),
    ({'checkpoints_enabled': True, 'unknown': 1, 'extensions': {}}, True),
  ],
)
def test_checkpoints_enabled(value: dict[str, object], expected: bool) -> None:
  assert config.checkpoints_enabled(value) is expected


def test_extension_choices_are_persisted_and_shared(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / '.mirror' / 'config.json'
  first = config.Config(path)
  second = config.Config(path)
  assert not first.read().get('extensions')
  first.set_extension('/extensions/clock/extension.py', True)
  second.set_extension('builtin/tokens_per_second', False)
  assert config.extension_choices(first.read().get('extensions')) == {
    '/extensions/clock/extension.py': True,
    'builtin/tokens_per_second': False,
  }
  assert json.loads(path.read_text())['extensions'][
    'builtin/tokens_per_second'
  ] == {'enabled': False}


def test_legacy_choices_remain_readable_without_rewriting_other_entries(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text('{"extensions":{"clock":true,"timer":false,"new":{}}}')
  settings = config.Config(path)
  assert config.extension_choices(settings.read().get('extensions')) == {
    'clock': True,
    'timer': False,
  }
  settings.set_extension('clock', False)
  assert json.loads(path.read_text()) == {
    'extensions': {
      'clock': {'enabled': False},
      'timer': False,
      'new': {},
    }
  }


def test_cached_reads_are_independent_and_notice_atomic_replacement(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text('{"extensions":{"a":{"enabled":true}}}')
  settings = config.Config(path)
  snapshot = settings.read()
  cast(dict[str, object], snapshot['extensions'])['a'] = {'enabled': False}
  assert config.extension_choices(settings.read().get('extensions')) == {
    'a': True
  }
  stat = path.stat()
  replacement = tmp_path / 'replacement.json'
  replacement.write_text('{"extensions":{"b":{"enabled":true}}}')
  os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
  replacement.replace(path)
  assert config.extension_choices(settings.read().get('extensions')) == {
    'b': True
  }


@pytest.mark.asyncio
async def test_polling_keeps_last_good_config_through_invalid_edits(
  tmp_path: pathlib.Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  path = tmp_path / 'config.json'
  settings = config.Config(path)
  updates: asyncio.Queue[dict[str, bool]] = asyncio.Queue()

  async def apply_config(value: dict[str, object]) -> None:
    updates.put_nowait(config.extension_choices(value.get('extensions')))

  task = asyncio.create_task(settings.watch(apply_config, interval=0.01))
  try:
    assert await asyncio.wait_for(updates.get(), 2) == {}
    path.write_text('{"extensions":{"clock":{"enabled":true}}}')
    assert await asyncio.wait_for(updates.get(), 2) == {'clock': True}
    path.write_text('{"extensions":')
    with pytest.raises(TimeoutError):
      await asyncio.wait_for(updates.get(), 0.1)
    assert 'Ignoring invalid config' in caplog.text
    with pytest.raises(json.JSONDecodeError):
      settings.set_extension('clock', False)
    assert path.read_text() == '{"extensions":'
    path.write_text('{"extensions":{"clock":{"enabled":false}}}')
    assert await asyncio.wait_for(updates.get(), 2) == {'clock': False}
  finally:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task


def test_unknown_settings_survive_updates_and_bad_flags_are_ignored(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text(
    json.dumps({
      'future-setting': {'anything': [1, 'two']},
      'extensions': {
        'clock': {'enabled': True, 'color': 'blue'},
        'timer': {'enabled': False},
        'typo': {'enabled': 'yes'},
        'number': {'enabled': 1},
        'broken': None,
      },
    })
  )
  settings = config.Config(path)
  assert config.extension_choices(settings.read().get('extensions')) == {
    'clock': True,
    'timer': False,
  }
  settings.set_extension('clock', False)
  saved = json.loads(path.read_text())
  assert saved['future-setting'] == {'anything': [1, 'two']}
  assert saved['extensions']['clock'] == {'enabled': False, 'color': 'blue'}
  assert saved['extensions']['typo'] == {'enabled': 'yes'}
  assert saved['extensions']['broken'] is None


@pytest.mark.parametrize('extensions', [None, 'invalid', ['clock']])
def test_bad_extensions_section_is_ignored(extensions: object) -> None:
  assert config.extension_choices(extensions) == {}


def test_invalid_config_is_not_overwritten(tmp_path: pathlib.Path) -> None:
  path = tmp_path / 'config.json'
  path.write_text('not json')
  with pytest.raises(json.JSONDecodeError):
    config.Config(path).set_extension('clock', True)
  assert path.read_text() == 'not json'


def test_extension_config_updates_keep_other_settings(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text(json.dumps({'extensions': {'demo': True}, 'theme': 'dark'}))
  settings = config.Config(path)

  settings.set_extension_config('builtin/memory', {'enabled': True})
  settings.set_extension_config('demo', {'clock': '24h'})

  assert json.loads(path.read_text()) == {
    'extensions': {
      'builtin/memory': {'config': {'enabled': True}},
      'demo': {'enabled': True, 'config': {'clock': '24h'}},
    },
    'theme': 'dark',
  }
