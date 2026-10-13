"""Prepare the local model after consent, then capture without downloading."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import pathlib
from collections.abc import AsyncGenerator

from mirror.cli import extension_dependencies


@contextlib.asynccontextmanager
async def worker(
  *arguments: str,
) -> AsyncGenerator[asyncio.subprocess.Process, None]:
  """Own voice's model readiness separately from its Python dependencies."""
  project = pathlib.Path(__file__).resolve().parent
  root = extension_dependencies.directory(project)
  revision = hashlib.sha256((project / 'worker.py').read_bytes()).hexdigest()
  ready = root / f'model-{revision}.ready'
  preparing = arguments == ('--prepare',)
  if preparing:
    await extension_dependencies.prepare(project)
  elif not ready.is_file():
    raise RuntimeError('Voice model changed or is missing; retry /voice setup.')
  environment = {
    'HF_HOME': str(root / 'cache' / 'huggingface'),
    'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    **{
      name: os.environ[name]
      for name in ('XDG_RUNTIME_DIR',)
      if name in os.environ
    },
  }
  offline = ('--offline',) if ready.is_file() else ()
  async with extension_dependencies.run(
    project, 'worker.py', *arguments, *offline, environment=environment
  ) as process:
    yield process
  if preparing:
    ready.touch()
