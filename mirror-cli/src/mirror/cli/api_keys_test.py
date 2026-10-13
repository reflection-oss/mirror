"""Saved Anthropic credentials are private and environment overrides win."""

from __future__ import annotations

import os
import pathlib
import stat

import pytest

from mirror.cli import api_keys


def test_saved_anthropic_key_is_private_and_environment_overrides_it(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  assert not api_keys.has_anthropic_key()
  api_keys.save_key(api_keys.anthropic_key_path(), 'synthetic-saved-key')
  assert stat.S_IMODE(api_keys.anthropic_key_path().stat().st_mode) == 0o600
  assert api_keys.anthropic_key() == 'synthetic-saved-key'
  assert 'ANTHROPIC_API_KEY' not in os.environ
  monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic-environment-key')
  assert api_keys.anthropic_key() == 'synthetic-environment-key'


@pytest.mark.parametrize('contents', ['', '  \n'])
def test_empty_saved_key_is_not_configured(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  path = api_keys.anthropic_key_path()
  path.parent.mkdir()
  path.write_text(contents)
  assert not api_keys.has_anthropic_key()


@pytest.mark.parametrize('home', [None, '', 'relative'])
def test_invalid_home_never_uses_current_directory(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, home: str | None
) -> None:
  monkeypatch.chdir(tmp_path)
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  if home is None:
    monkeypatch.delenv('HOME', raising=False)
  else:
    monkeypatch.setenv('HOME', home)
  assert not api_keys.has_anthropic_key()
  with pytest.raises(OSError, match='Set HOME to an absolute path'):
    api_keys.anthropic_key_path()
  assert not (tmp_path / '.mirror').exists()


def test_custom_endpoint_never_reads_saved_production_key(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  api_keys.save_key(api_keys.anthropic_key_path(), 'synthetic-production-key')
  monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://custom.example')
  assert api_keys.anthropic_key() == ''
  assert not api_keys.has_anthropic_key()
  monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic-explicit-key')
  assert api_keys.anthropic_key() == 'synthetic-explicit-key'


@pytest.mark.parametrize('corrupt', [False, True])
def test_unreadable_saved_key_does_not_crash_picker(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  path = api_keys.anthropic_key_path()
  path.parent.mkdir()
  if corrupt:
    path.write_bytes(b'\xff')
  else:
    path.mkdir()
  assert not api_keys.has_anthropic_key()
  with pytest.raises(OSError, match='Could not read saved Anthropic API key'):
    api_keys.anthropic_key()
