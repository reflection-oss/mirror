"""Explicit acknowledgments must precede interactive agent startup."""

from __future__ import annotations

import io
import json
import pathlib

import pytest

from mirror.cli import onboarding


def test_first_use_combines_both_notices_and_remembers_acceptance(
  tmp_path: pathlib.Path,
) -> None:
  prompts: list[str] = []

  def accept(message: str) -> bool:
    prompts.append(message)
    return True

  path = tmp_path / 'config.json'
  path.write_text(json.dumps({'extensions': {'example': False}}))
  directory = tmp_path / 'project'
  onboarding.ensure_consent(directory, path=path, confirm=accept)
  assert len(prompts) == 1
  assert onboarding.USAGE_WARNING in prompts[0]
  assert onboarding.FOLDER_WARNING in prompts[0]
  assert repr(str(directory)) in prompts[0]
  assert json.loads(path.read_text()) == {
    'extensions': {'example': False},
    'usage_warning_acknowledged': True,
    'trusted_directories': [str(directory)],
  }

  prompts.clear()
  onboarding.ensure_consent(directory, path=path, confirm=accept)
  assert prompts == []

  onboarding.ensure_consent(directory / 'child', path=path, confirm=accept)
  assert len(prompts) == 1
  assert onboarding.FOLDER_WARNING in prompts[0]
  assert onboarding.USAGE_WARNING not in prompts[0]


def test_declining_first_use_records_neither_acknowledgment(
  tmp_path: pathlib.Path,
) -> None:
  prompts: list[str] = []

  def decline(message: str) -> bool:
    prompts.append(message)
    return False

  path = tmp_path / 'config.json'
  with pytest.raises(SystemExit, match='no agent was started'):
    onboarding.ensure_consent(tmp_path, path=path, confirm=decline)
  assert len(prompts) == 1
  assert onboarding.USAGE_WARNING in prompts[0]
  assert onboarding.FOLDER_WARNING in prompts[0]
  assert not path.exists()

  prompts.clear()
  with pytest.raises(SystemExit, match='no agent was started'):
    onboarding.ensure_consent(tmp_path, path=path, confirm=decline)
  assert len(prompts) == 1
  assert onboarding.USAGE_WARNING in prompts[0]
  assert onboarding.FOLDER_WARNING in prompts[0]
  assert not path.exists()


def test_declining_directory_preserves_only_usage_acknowledgment(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text(json.dumps({'usage_warning_acknowledged': True}))
  with pytest.raises(SystemExit, match='no agent was started'):
    onboarding.ensure_consent(tmp_path, path=path, confirm=lambda _: False)
  assert json.loads(path.read_text()) == {'usage_warning_acknowledged': True}

  prompts: list[str] = []

  def accept(message: str) -> bool:
    prompts.append(message)
    return True

  onboarding.ensure_consent(tmp_path, path=path, confirm=accept)
  assert len(prompts) == 1
  assert onboarding.FOLDER_WARNING in prompts[0]


def test_symlink_uses_resolved_directory_consent(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'project'
  directory.mkdir()
  alias = tmp_path / 'alias'
  alias.symlink_to(directory, target_is_directory=True)
  path = tmp_path / 'config.json'
  onboarding.ensure_consent(directory, path=path, confirm=lambda _: True)
  onboarding.ensure_consent(
    alias, path=path, confirm=lambda _: pytest.fail('already trusted')
  )


def test_noninteractive_input_cannot_accept_notice(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr('sys.stdin', io.StringIO('yes\nyes\n'))
  with pytest.raises(SystemExit, match='open Mirror in a terminal'):
    onboarding.ensure_consent(tmp_path, path=tmp_path / 'config.json')


def test_acceptance_preserves_settings_changed_while_prompt_is_open(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text(json.dumps({'usage_warning_acknowledged': True}))

  def accept(message: str) -> bool:
    assert onboarding.FOLDER_WARNING in message
    path.write_text(
      json.dumps({
        'usage_warning_acknowledged': True,
        'trusted_directories': ['/another/project'],
        'extensions': {'example': False},
      })
    )
    return True

  onboarding.ensure_consent(tmp_path, path=path, confirm=accept)
  assert json.loads(path.read_text()) == {
    'usage_warning_acknowledged': True,
    'trusted_directories': ['/another/project', str(tmp_path)],
    'extensions': {'example': False},
  }


def test_ensure_install_id_creates_install_id_once(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text(json.dumps({'analytics': False}))

  onboarding.ensure_install_id(path=path)
  first = json.loads(path.read_text())
  onboarding.ensure_install_id(path=path)

  assert json.loads(path.read_text()) == first
  assert first == {'analytics': False, 'install_id': first['install_id']}
  assert len(first['install_id']) == 32


def test_take_analytics_notice_returns_notice_once(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'config.json'
  path.write_text(json.dumps({'analytics': False}))

  first = onboarding.take_analytics_notice(path=path)
  second = onboarding.take_analytics_notice(path=path)

  assert first is not None
  assert 'anonymous feature usage' in first.plain
  assert second is None
  assert json.loads(path.read_text()) == {
    'analytics': False,
    'analytics_notice_shown': True,
  }


def test_directory_trust_check_does_not_grant_consent(
  tmp_path: pathlib.Path,
) -> None:
  settings = tmp_path / 'config.json'
  assert not onboarding.is_trusted_directory(tmp_path, path=settings)
  assert not settings.exists()
  settings.write_text(json.dumps({'trusted_directories': [str(tmp_path)]}))
  before = settings.read_bytes()
  assert onboarding.is_trusted_directory(tmp_path, path=settings)
  assert not onboarding.is_trusted_directory(tmp_path / 'child', path=settings)
  assert settings.read_bytes() == before
