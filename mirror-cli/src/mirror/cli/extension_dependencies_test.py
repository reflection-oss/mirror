"""Exercise the installed helper with synthetic uv and worker processes."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

from mirror.cli import extension_dependencies as process


@pytest.fixture
def project(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
  monkeypatch.setenv('HOME', str(tmp_path / 'home'))
  root = tmp_path / 'project'
  root.mkdir()
  (root / 'pyproject.toml').write_text('[project]\nname = "test-extension"\n')
  (root / 'uv.lock').write_text('version = 1\n')
  (root / 'worker.py').write_text(
    'import json, os, sys\n'
    'print(json.dumps({"env": dict(os.environ), "args": sys.argv[1:]}))\n'
  )
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
  await process.prepare(project)
  async with process.run(project, 'worker.py', '--echo') as worker:
    output, _ = await worker.communicate()
  root = process.directory(project)
  assert (root / 'ready').exists()
  assert 'synthetic' not in output.decode()
  payload = json.loads(output)
  assert payload['env']['HOME'] == str(root / 'home')
  assert payload['args'] == ['--echo']
  command = json.loads((root / 'syncs').read_text())
  assert '--locked' in command
  assert '--no-python-downloads' in command
  assert '--no-build' in command
  assert command[-1] == sys.executable
  monkeypatch.setenv('PATH', '')
  await process.prepare(project)
  async with process.run(project, 'worker.py', '--echo') as worker:
    output, _ = await worker.communicate()
  assert json.loads(output)['args'] == ['--echo']
  async with process.run(project, 'worker.py') as worker:
    output, _ = await worker.communicate()
  assert json.loads(output)['args'] == []
  assert len((root / 'syncs').read_text().splitlines()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['pyproject.toml', 'uv.lock'])
async def test_changed_dependency_requires_setup(
  project: pathlib.Path, name: str
) -> None:
  await process.prepare(project)
  async with process.run(project, 'worker.py', '--echo') as worker:
    await worker.communicate()
  previous = process.directory(project)
  path = project / name
  path.write_text(path.read_text() + '\n# Changed\n')
  assert process.directory(project) != previous
  with pytest.raises(RuntimeError, match='changed or is missing'):
    async with process.run(project, 'worker.py'):
      pytest.fail('Must not start or implicitly install')
  await process.prepare(project)
  async with process.run(project, 'worker.py', '--echo') as worker:
    await worker.communicate()
  assert (process.directory(project) / 'ready').exists()
  assert (previous / 'ready').exists()


@pytest.mark.asyncio
async def test_missing_uv(
  project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('PATH', '')
  with pytest.raises(RuntimeError, match='requires uv on PATH'):
    await process.prepare(project)
  assert not (process.directory(project) / 'ready').exists()


@pytest.mark.asyncio
async def test_failed_resolution(project: pathlib.Path) -> None:
  uv = project.parent / 'uv'
  uv.write_text(
    uv.read_text().replace(
      'root = pathlib.Path',
      'sys.stderr.write("offline or locked resolution failed")\nsys.exit(1)\nroot = pathlib.Path',
    )
  )
  with pytest.raises(RuntimeError, match='offline or locked resolution failed'):
    await process.prepare(project)
  assert not (process.directory(project) / 'ready').exists()


@pytest.mark.asyncio
async def test_concurrent_use_fails_without_mutating_environment(
  project: pathlib.Path,
) -> None:
  await process.prepare(project)
  async with process.run(project, 'worker.py', '--echo') as worker:
    with pytest.raises(RuntimeError, match='already in use'):
      await process.prepare(project)
    await worker.communicate()
  assert (
    len((process.directory(project) / 'syncs').read_text().splitlines()) == 1
  )


@pytest.mark.asyncio
async def test_cancelled_worker_is_reaped(project: pathlib.Path) -> None:
  (project / 'worker.py').write_text(
    'import time\nprint("started", flush=True)\ntime.sleep(60)\n'
  )
  await process.prepare(project)
  child: asyncio.subprocess.Process | None = None
  with pytest.raises(asyncio.CancelledError):
    async with process.run(project, 'worker.py') as worker:
      child = worker
      assert worker.stdout is not None
      assert await worker.stdout.readline() == b'started\n'
      raise asyncio.CancelledError
  assert child is not None
  assert child.returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('version', ['0.9.0', 'not-a-version'])
async def test_unsupported_uv_fails_before_sync(
  project: pathlib.Path, version: str
) -> None:
  uv = project.parent / 'uv'
  uv.write_text(uv.read_text().replace('uv 0.10.10', f'uv {version}'))
  with pytest.raises(RuntimeError, match='requires uv >=0.10.10'):
    await process.prepare(project)
  assert not (process.directory(project) / 'syncs').exists()


@pytest.mark.asyncio
async def test_run_never_bootstraps_an_environment(
  project: pathlib.Path,
) -> None:
  with pytest.raises(RuntimeError, match='changed or is missing'):
    async with process.run(project, 'worker.py'):
      pytest.fail('Capture must not bootstrap')
  assert not process.directory(project).exists()


@pytest.mark.asyncio
async def test_cancelled_setup_is_not_ready(project: pathlib.Path) -> None:
  uv = project.parent / 'uv'
  uv.write_text(
    uv.read_text().replace(
      '(root / "bin").mkdir',
      'import time\ntime.sleep(60)\n(root / "bin").mkdir',
    )
  )
  root = process.directory(project)
  setup = asyncio.create_task(process.prepare(project))
  try:
    async with asyncio.timeout(5):
      while not (root / 'syncs').exists():
        await asyncio.sleep(0.01)
  finally:
    setup.cancel()
    with pytest.raises(asyncio.CancelledError):
      await setup
  assert not (root / 'ready').exists()


@pytest.mark.asyncio
async def test_run_only_forwards_explicit_environment(
  project: pathlib.Path,
) -> None:
  await process.prepare(project)
  async with process.run(
    project, 'worker.py', environment={'EXTENSION_SETTING': 'explicit'}
  ) as worker:
    output, _ = await worker.communicate()
  assert json.loads(output)['env']['EXTENSION_SETTING'] == 'explicit'
