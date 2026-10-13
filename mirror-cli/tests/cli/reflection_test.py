"""Reflection credential storage and model routing."""

from __future__ import annotations

import pathlib
import stat

import pytest

from mirror.cli import main, models, reflection
from mirror.cli.extensions.inference_controls import extension


@pytest.fixture(autouse=True)
def isolated_home(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  monkeypatch.delenv('MIRROR_MODELS_CONFIG', raising=False)


def test_reflection_uses_public_production_urls() -> None:
  assert reflection.KEY_URL == 'https://platform.reflection.ai/api-keys'
  assert reflection.BASE_URL == 'https://api.reflection.ai/openai/v1'


def test_save_key_is_private_and_does_not_change_environment() -> None:
  reflection.save_key('synthetic-reflection-key')
  path = reflection.key_path()
  assert path.read_text() == 'synthetic-reflection-key'
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  assert reflection.has_key()
  assert 'REFLECTION_API_KEY' not in reflection.os.environ
  assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize('key', ['', 'two words', 'line\nbreak'])
def test_invalid_key_is_not_saved(key: str) -> None:
  with pytest.raises(ValueError, match='nonempty and contain no whitespace'):
    reflection.save_key(key)
  assert not reflection.key_path().exists()


@pytest.mark.parametrize('arguments', [[], ['--api', 'reflection']])
def test_reflection_default_and_restart_routing(
  arguments: list[str],
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
  monkeypatch.setattr(main.sys, 'argv', ['mirror', *arguments])
  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]
  assert args.api == 'reflection'
  assert args.model == 'Beam-501B-A23B'
  models.save_last_model(args.model)
  monkeypatch.setattr(main.sys, 'argv', ['mirror'])
  restarted = main._parse_args()  # pyright: ignore[reportPrivateUsage]
  assert restarted.model_api == 'reflection'
  assert models.find_model(args.model).api == 'reflection'


def test_picker_resolves_reflection_without_credentials() -> None:
  action = extension._select_inference('reflection')  # pyright: ignore[reportPrivateUsage]
  assert isinstance(action, extension.actions.ModelResolved)
  assert action.option.api == 'reflection'


@pytest.mark.parametrize('environment', [False, True])
@pytest.mark.parametrize('saved', [False, True])
def test_remove_key_preserves_environment(
  environment: bool, saved: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
  if environment:
    monkeypatch.setenv('REFLECTION_API_KEY', 'synthetic-env-key')
  if saved:
    reflection.save_key('synthetic-saved-key')

  assert reflection.remove_key() is saved
  assert not reflection.key_path().exists()
  assert reflection.has_environment_key() is environment
  assert reflection.has_key() is environment
  assert reflection.remove_key() is False


def test_remove_key_sanitizes_filesystem_errors() -> None:
  reflection.key_path().mkdir(parents=True)
  with pytest.raises(
    OSError, match='^Could not remove saved Reflection API key$'
  ):
    reflection.remove_key()
