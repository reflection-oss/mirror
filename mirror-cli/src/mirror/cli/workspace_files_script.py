"""Stdlib-only file operations sent to the workspace's Python interpreter."""

from __future__ import annotations

import fcntl
import hashlib
import os
import pathlib
import secrets
import stat
import sys


def _version(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
  return (
    info.st_dev,
    info.st_ino,
    info.st_mode,
    info.st_size,
    info.st_mtime_ns,
    info.st_ctime_ns,
  )


def _read(
  directory: int, name: str, limit: int
) -> tuple[bytes, os.stat_result]:
  """Read a bounded regular file and reject concurrent changes."""
  descriptor = os.open(
    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
  )
  with os.fdopen(descriptor, 'rb') as source:
    before = os.fstat(source.fileno())
    if not stat.S_ISREG(before.st_mode):
      raise ValueError('path is not a regular file')
    content = source.read(limit + 1)
    if len(content) > limit:
      raise ValueError(f'file exceeds byte limit ({limit} bytes)')
    if _version(os.fstat(source.fileno())) != _version(before):
      raise ValueError('file changed while being read')
  return content, before


def _write(directory: int, name: str, limit: int, expected: str) -> None:
  """Stage a replacement, checking the original when a digest is provided."""
  content = sys.stdin.buffer.read(limit + 1)
  if len(content) > limit:
    raise ValueError(f'file exceeds byte limit ({limit} bytes)')
  if expected and expected != 'missing':
    original, before = _read(directory, name, limit)
    if hashlib.sha256(original).hexdigest() != expected:
      raise ValueError('file changed since it was read')
  else:
    try:
      before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
      before = None
    if expected == 'missing' and before is not None:
      raise ValueError('file changed since it was read')
    if before is not None and not stat.S_ISREG(before.st_mode):
      raise ValueError('path is not a regular file')
  staged = '.mirror-write-' + secrets.token_hex(16)
  descriptor = os.open(
    staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory
  )
  try:
    with os.fdopen(descriptor, 'wb') as target:
      target.write(content)
      if before is not None:
        os.fchmod(target.fileno(), stat.S_IMODE(before.st_mode))
    if (
      expected
      and before is not None
      and _version(os.stat(name, dir_fd=directory, follow_symlinks=False))
      != _version(before)
    ):
      raise ValueError('file changed while preparing the write')
    if expected == 'missing':
      os.link(staged, name, src_dir_fd=directory, dst_dir_fd=directory)
    else:
      os.replace(staged, name, src_dir_fd=directory, dst_dir_fd=directory)
  finally:
    try:
      os.unlink(staged, dir_fd=directory)
    except FileNotFoundError:
      pass


def _locked_write(
  directory: int, name: str, limit: int, expected: str, lock: str
) -> None:
  """Serialize participating writers using their chosen sibling lock file."""
  if not lock:
    _write(directory, name, limit, expected)
    return
  if pathlib.Path(lock).name != lock or lock in ('.', '..', name):
    raise ValueError('lock must name a separate sibling file')
  descriptor = os.open(
    lock,
    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
    0o600,
    dir_fd=directory,
  )
  try:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
      raise ValueError('lock must be a regular file with no hard links')
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _write(directory, name, limit, expected)
  finally:
    os.close(descriptor)


def main() -> int:
  """Run one file operation using argv, stdin, stdout and an exit status."""
  operation, name, limit_text, expected = sys.argv[1:5]
  lock = sys.argv[5] if len(sys.argv) > 5 else ''
  limit = int(limit_text)
  path = pathlib.Path(
    os.path.relpath(pathlib.Path(name).expanduser().absolute())
  )
  try:
    if not path.name:
      raise ValueError('path is not a file')
    directory = os.open('.', os.O_RDONLY | os.O_DIRECTORY)
    try:
      for part in path.parts[:-1]:
        if operation == 'write' and expected in ('', 'missing'):
          try:
            os.mkdir(part, dir_fd=directory)
          except FileExistsError:
            pass
        child = os.open(
          part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
        )
        os.close(directory)
        directory = child
      if operation == 'read':
        sys.stdout.buffer.write(_read(directory, path.name, limit)[0])
      else:
        _locked_write(directory, path.name, limit, expected, lock)
    finally:
      os.close(directory)
  except FileNotFoundError as error:
    print(str(error), file=sys.stderr)
    return 3
  except (OSError, ValueError) as error:
    print(str(error), file=sys.stderr)
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main())
