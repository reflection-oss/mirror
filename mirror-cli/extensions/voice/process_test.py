"""Exercise the installed helper with synthetic uv and an audio-free worker."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

from mirror.cli import extension_dependencies
from mirror.cli.extensions.voice import process


@pytest.fixture
def project(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
  monkeypatch.setenv('HOME', str(tmp_path / 'home'))
  root = tmp_path / 'project'
  root.mkdir()
  (root / 'pyproject.toml').write_text('[project]\nname = "test-voice"\n')
  (root / 'uv.lock').write_text('version = 1\n')
  (root / 'worker.py').write_text(
    'import json, os, sys\n'
    'print(json.dumps({"env": dict(os.environ), "args": sys.argv[1:]}))\n'
  )
  monkeypatch.setattr(process, '__file__', str(root / 'process.py'))
  uv = tmp_path / 'uv'
  uv.write_text(
    f'#!{sys.executable}\n'
    'import json, os, pathlib, sys\n'
    'if sys.argv[1:] == ["--version"]:\n'
    '  print("uv 0.10.10")\n'
    '  sys.exit(0)\n'
    'root = pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"])\n'
    'with (root.parent / "syncs").open("a") as calls:\n'
    '  calls.write(json.dumps(sys.argv[1:]) + "\\n")\n'
    '(root / "bin").mkdir(parents=True, exist_ok=True)\n'
    f'(root / "bin" / "python").symlink_to({sys.executable!r})\n'
  )
  uv.chmod(0o700)
  monkeypatch.setenv('PATH', str(tmp_path))
  return root


@pytest.mark.asyncio
async def test_setup_reuse_and_environment(
  project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-host-secret')
  monkeypatch.setenv('HF_TOKEN', 'synthetic-hf-secret')
  monkeypatch.setenv('UV_INDEX_URL', 'https://synthetic-secret@example.invalid')
  monkeypatch.setenv('PYTHONPATH', '/synthetic/host/path')
  async with process.worker('--prepare') as worker:
    output, _ = await worker.communicate()
  root = extension_dependencies.directory(project)
  assert (root / 'ready').exists()
  assert 'synthetic' not in output.decode()
  payload = json.loads(output)
  assert payload['env']['HOME'] == str(root / 'home')
  assert payload['env']['HF_HOME'] == str(root / 'cache' / 'huggingface')
  assert payload['args'] == ['--prepare']
  command = json.loads((root / 'syncs').read_text())
  assert '--locked' in command
  assert '--no-python-downloads' in command
  assert '--no-build' in command
  assert command[-1] == sys.executable
  monkeypatch.setenv('PATH', '')
  async with process.worker('--prepare') as worker:
    output, _ = await worker.communicate()
  assert json.loads(output)['args'] == ['--prepare', '--offline']
  async with process.worker() as worker:
    output, _ = await worker.communicate()
  assert json.loads(output)['args'] == ['--offline']
  assert len((root / 'syncs').read_text().splitlines()) == 1


@pytest.mark.asyncio
async def test_failed_model_setup_not_ready(project: pathlib.Path) -> None:
  (project / 'worker.py').write_text(
    'raise RuntimeError("model unavailable")\n'
  )
  with pytest.raises(RuntimeError, match='model unavailable'):
    async with process.worker('--prepare') as worker:
      await worker.communicate()
  assert not list(
    extension_dependencies.directory(project).glob('model-*.ready')
  )


@pytest.mark.asyncio
async def test_cancelled_worker_is_reaped(project: pathlib.Path) -> None:
  (project / 'worker.py').write_text(
    'import time\nprint("started", flush=True)\ntime.sleep(60)\n'
  )
  with pytest.raises(asyncio.CancelledError):
    async with process.worker('--prepare') as worker:
      assert worker.stdout is not None
      assert await worker.stdout.readline() == b'started\n'
      raise asyncio.CancelledError
  assert worker.returncode is not None
  assert not list(
    extension_dependencies.directory(project).glob('model-*.ready')
  )


@pytest.mark.asyncio
async def test_changed_worker_requires_model_setup(
  project: pathlib.Path,
) -> None:
  async with process.worker('--prepare') as worker:
    await worker.communicate()
  path = project / 'worker.py'
  path.write_text(path.read_text() + '\n# Changed model revision\n')
  with pytest.raises(RuntimeError, match='model changed or is missing'):
    async with process.worker():
      pytest.fail('Capture must not download a new model')
  async with process.worker('--prepare') as worker:
    output, _ = await worker.communicate()
  assert json.loads(output)['args'] == ['--prepare']
  assert (
    len(
      (extension_dependencies.directory(project) / 'syncs')
      .read_text()
      .splitlines()
    )
    == 1
  )
