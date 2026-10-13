"""Fixtures shared by the mirror test suite."""

from __future__ import annotations

from collections.abc import Iterator


import freezegun
import pytest
from freezegun import api as freezegun_api
from freezegun import config as freezegun_config


@pytest.fixture
def frozen_ui_time() -> Iterator[freezegun_api.FrozenDateTimeFactory]:
  """Freeze animation frames while thread and asyncio scheduling stay live."""
  ignored = freezegun_config.settings.default_ignore_list
  # Freezegun otherwise uses real time anywhere prompt_toolkit is on the stack.
  freezegun.configure(
    default_ignore_list=[name for name in ignored if name != 'prompt_toolkit']
  )
  try:
    with freezegun.freeze_time('2000-01-01', real_asyncio=True) as clock:
      assert isinstance(clock, freezegun_api.FrozenDateTimeFactory)
      yield clock
  finally:
    freezegun.configure(default_ignore_list=ignored)


@pytest.fixture(autouse=True)
def disable_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Do not execute external extensions in public harness tests."""
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'false')


@pytest.fixture(autouse=True)
def isolate_models_config(
  tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Keep the model config, and the last-used model beside it, out of $HOME."""
  directory = tmp_path_factory.mktemp('models-config')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(directory / 'models.json'))


@pytest.fixture(autouse=True)
def isolate_state_database(
  tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Keep the agent and CLI databases out of $HOME."""
  directory = tmp_path_factory.mktemp('state')
  monkeypatch.setenv('MIRROR_STATE_DB', str(directory / 'state.db'))
  monkeypatch.setenv('MIRROR_CLI_DB', str(directory / 'cli.db'))
