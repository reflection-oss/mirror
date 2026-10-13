import pathlib

import pytest

from mirror import config
from mirror.cli import workspace_tags


def test_definitions_persist_and_keep_configured_order(
  tmp_path: pathlib.Path,
) -> None:
  settings = config.Config(tmp_path / 'config.json')
  settings.update(
    lambda current: current.update({'analytics': False, 'workspace_tags': []})
  )
  blocked = workspace_tags.WorkspaceTag('blocked', 'Blocked', '#ff0000')
  feature = workspace_tags.WorkspaceTag('feature', 'Feature')
  renamed = workspace_tags.WorkspaceTag('blocked', 'Waiting', '#ffffff')
  workspace_tags.save(settings, blocked)
  workspace_tags.save(settings, feature)
  workspace_tags.save(settings, renamed)
  restored = config.Config(settings.path)
  assert workspace_tags.definitions(restored.read()) == (renamed, feature)
  assert restored.read()['analytics'] is False
  workspace_tags.delete(restored, 'blocked')
  assert workspace_tags.definitions(settings.read()) == (feature,)


@pytest.mark.parametrize('color', ['red', '#fff', 'bg:#123456', '#ffffff\x1b'])
def test_workspace_tag_rejects_invalid_colors(color: str) -> None:
  with pytest.raises(ValueError, match='color'):
    workspace_tags.WorkspaceTag('blocked', 'Blocked', color)


def test_explicitly_empty_definitions_stay_empty() -> None:
  assert workspace_tags.definitions({'workspace_tags': []}) == ()


def test_deleting_defaults_does_not_restore_them(
  tmp_path: pathlib.Path,
) -> None:
  settings = config.Config(tmp_path / 'config.json')
  workspace_tags.delete(settings, 'blocked')
  assert (
    workspace_tags.definitions(settings.read())
    == workspace_tags.DEFAULT_TAGS[1:]
  )
  workspace_tags.delete(settings, 'in-progress')
  assert workspace_tags.definitions(config.Config(settings.path).read()) == ()


def test_duplicate_ids_are_rejected() -> None:
  with pytest.raises(ValueError, match='duplicate'):
    workspace_tags.definitions({
      'workspace_tags': [
        {'id': 'blocked', 'name': 'Blocked'},
        {'id': 'blocked', 'name': 'Waiting'},
      ]
    })


@pytest.mark.parametrize('icon', ['★', '🚀'])
def test_custom_icons_persist(tmp_path: pathlib.Path, icon: str) -> None:
  settings = config.Config(tmp_path / 'config.json')
  tag = workspace_tags.WorkspaceTag('custom', 'Anything', icon=icon)
  workspace_tags.save(settings, tag)
  assert (
    workspace_tags.definitions(config.Config(settings.path).read())[-1] == tag
  )


@pytest.mark.parametrize('icon', ['', ' ', '\n', '\x1b', 'long'])
def test_invalid_icons_are_rejected(icon: str) -> None:
  with pytest.raises(ValueError, match='tag icon'):
    workspace_tags.WorkspaceTag('custom', 'Anything', icon=icon)


def test_old_definitions_default_to_square() -> None:
  assert workspace_tags.definitions({
    'workspace_tags': [{'id': 'old', 'name': 'Old'}]
  }) == (workspace_tags.WorkspaceTag('old', 'Old', icon='■'),)
