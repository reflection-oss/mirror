"""Bounded UTF-8 file operations through the workspace's process transport."""

from __future__ import annotations

import hashlib
import pathlib
from importlib import resources

from mirror.cli import shell

# Only trusted code is executed; paths are arguments and file contents use stdin.
_SCRIPT = (
  resources
  .files('mirror.cli')
  .joinpath('workspace_files_script.py')
  .read_text(encoding='utf-8')
)


async def read(
  backend: shell.Shell, path: str | pathlib.PurePath, *, max_bytes: int
) -> str:
  """Read strict UTF-8 without newline conversion or symlink traversal."""
  result = await _run(backend, 'read', path, max_bytes=max_bytes)
  return result.decode('utf-8')


async def write(
  backend: shell.Shell,
  path: str | pathlib.PurePath,
  content: str,
  *,
  max_bytes: int,
  expected: str | None,
) -> None:
  """Replace a UTF-8 file, optionally checking its previously read contents."""
  data = content.encode('utf-8')
  if len(data) > max_bytes:
    raise ValueError(
      f'File {str(path)!r} exceeds byte limit ({max_bytes} bytes)'
    )
  digest = (
    hashlib.sha256(expected.encode('utf-8')).hexdigest()
    if expected is not None
    else ''
  )
  await _run(
    backend, 'write', path, max_bytes=max_bytes, data=data, expected=digest
  )


async def _run(
  backend: shell.Shell,
  operation: str,
  path: str | pathlib.PurePath,
  *,
  max_bytes: int,
  data: bytes = b'',
  expected: str = '',
) -> bytes:
  if max_bytes <= 0:
    raise ValueError('File byte limit must be positive')
  result = await backend.run(
    (
      'python3',
      '-I',
      '-c',
      _SCRIPT,
      operation,
      str(path),
      str(max_bytes),
      expected,
    ),
    data=data,
    limit=max(max_bytes, 65536),
  )
  if result.returncode:
    detail = f'{str(path)!r}: {result.stderr.decode(errors="replace").strip()}'
    if result.returncode == 3:
      raise FileNotFoundError(detail)
    raise ValueError(detail)
  return result.stdout
