"""Run the installed worker with inert audio modules; corrupt models fail closed."""

from __future__ import annotations

import pathlib
import runpy
import sys
from unittest import mock

import pytest

from mirror.cli.extensions.voice import extension


@pytest.mark.parametrize(
  'arguments', [[], ['--prepare'], ['--prepare', '--offline']]
)
def test_worker_checks_model_integrity_before_loading(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
  (tmp_path / 'config.json').write_text('corrupt model configuration')
  downloader = mock.Mock(return_value=str(tmp_path))
  model = mock.Mock()
  modules = {
    'faster_whisper': mock.Mock(
      utils=mock.Mock(download_model=downloader), WhisperModel=model
    ),
    'numpy': mock.Mock(),
    'sounddevice': mock.Mock(),
  }
  worker = pathlib.Path(extension.__file__).with_name('worker.py')
  monkeypatch.setattr(sys, 'argv', [str(worker), *arguments])
  with mock.patch.dict(sys.modules, modules):
    with pytest.raises(ValueError, match='integrity check failed: config.json'):
      runpy.run_path(str(worker), run_name='__main__')
  model.assert_not_called()
  downloader.assert_called_once_with(
    'Systran/faster-whisper-base',
    revision='ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66',
    local_files_only='--prepare' not in arguments or '--offline' in arguments,
    use_auth_token=False,
  )
