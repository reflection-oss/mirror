"""Explicit setup and subprocess execution for locked extension dependencies."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import os
import pathlib
import shutil
import signal
import sys
import tempfile
from collections.abc import AsyncGenerator, Generator, Mapping


def directory(project: pathlib.Path) -> pathlib.Path:
  """Keep each project/dependency/interpreter revision in its own environment."""
  digest = hashlib.sha256()
  for name in ('pyproject.toml', 'uv.lock'):
    digest.update((project / name).read_bytes())
  digest.update(str(project.resolve()).encode())
  digest.update(f'{sys.executable}:{sys.version}:{sys.platform}'.encode())
  return (
    pathlib.Path.home()
    / '.mirror'
    / 'extension-environments'
    / digest.hexdigest()
  )


def _environment(root: pathlib.Path) -> dict[str, str]:
  return {
    'PATH': os.defpath,
    'HOME': str(root / 'home'),
    'XDG_CONFIG_HOME': str(root / 'home' / '.config'),
    'XDG_CACHE_HOME': str(root / 'cache'),
    'UV_CACHE_DIR': str(root / 'cache' / 'uv'),
    'UV_PROJECT_ENVIRONMENT': str(root / '.venv'),
  }


@contextlib.contextmanager
def _lock(root: pathlib.Path) -> Generator[None, None, None]:
  root.mkdir(parents=True, exist_ok=True, mode=0o700)
  with (root / 'lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
      raise RuntimeError('Extension environment is already in use.') from error
    yield


@contextlib.asynccontextmanager
async def _run(
  command: tuple[str, ...], root: pathlib.Path, environment: Mapping[str, str]
) -> AsyncGenerator[asyncio.subprocess.Process, None]:
  with tempfile.TemporaryFile() as errors:
    process = await asyncio.create_subprocess_exec(
      *command,
      cwd=root,
      env=dict(environment),
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=errors,
      start_new_session=True,
    )
    try:
      yield process
      code = await process.wait()
      if code:
        errors.seek(max(0, errors.seek(0, os.SEEK_END) - 8192))
        detail = errors.read().decode(errors='replace').strip()
        raise RuntimeError(f'Extension subprocess exited with {code}: {detail}')
    finally:
      with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
      try:
        async with asyncio.timeout(2):
          await process.communicate()
      except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
          os.killpg(process.pid, signal.SIGKILL)
        await process.communicate()


async def _sync(project: pathlib.Path, root: pathlib.Path) -> None:
  uv = shutil.which('uv')
  if uv is None:
    raise RuntimeError(
      'Extension setup requires uv on PATH. Install uv and retry setup.'
    )
  async with _run((uv, '--version'), root, _environment(root)) as check:
    output, _ = await check.communicate()
  try:
    version = tuple(int(part) for part in output.decode().split()[1].split('.'))
  except (ValueError, IndexError) as error:
    raise RuntimeError(
      'Could not determine uv version; extension setup requires uv >=0.10.10.'
    ) from error
  if version < (0, 10, 10):
    raise RuntimeError(
      'Extension setup requires uv >=0.10.10. Upgrade uv and retry setup.'
    )
  async with _run(
    (
      uv,
      '--no-config',
      '--no-python-downloads',
      'sync',
      '--project',
      str(project),
      '--locked',
      '--no-dev',
      '--no-install-project',
      '--no-build',
      '--python',
      sys.executable,
    ),
    root,
    _environment(root),
  ) as setup:
    await setup.communicate()


async def prepare(project: pathlib.Path) -> None:
  """Install locked wheels after user consent; reuse successful environments."""
  project = project.resolve()
  root = directory(project)
  with _lock(root):
    ready = root / 'ready'
    if ready.is_file() and (root / '.venv' / 'bin' / 'python').is_file():
      return
    ready.unlink(missing_ok=True)
    async with asyncio.timeout(300):
      await _sync(project, root)
    ready.touch()


@contextlib.asynccontextmanager
async def run(
  project: pathlib.Path,
  script: str,
  *arguments: str,
  environment: Mapping[str, str] | None = None,
) -> AsyncGenerator[asyncio.subprocess.Process, None]:
  """Run in a prepared environment, forwarding only explicit environment values."""
  project = project.resolve()
  root = directory(project)
  python = root / '.venv' / 'bin' / 'python'
  if not ((root / 'ready').is_file() and python.is_file()):
    raise RuntimeError(
      'Extension environment changed or is missing; run setup again.'
    )
  with _lock(root):
    command = (str(python), '-I', str(project / script), *arguments)
    async with _run(
      command, root, _environment(root) | dict(environment or {})
    ) as process:
      yield process
