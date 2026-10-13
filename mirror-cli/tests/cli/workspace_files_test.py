from __future__ import annotations

import os
import pathlib
import stat

import pytest

from mirror.cli import extension, shell


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> extension.Workspace:
  return extension.Workspace(tmp_path, False, shell=shell.Shell(tmp_path))


@pytest.mark.asyncio
async def test_read_write_use_bound_workspace_and_preserve_utf8(
  tmp_path: pathlib.Path,
) -> None:
  target = tmp_path / 'target'
  target.mkdir()
  workspace = extension.Workspace(tmp_path, False, shell=shell.Shell(target))
  text = '\ufeffcafé\r\n日本語\n'
  path = "nested/a'; echo unsafe"

  await workspace.write(path, text)

  assert await workspace.read(path) == text
  assert (target / path).read_bytes() == text.encode('utf-8')
  assert not (tmp_path / 'nested').exists()


@pytest.mark.asyncio
async def test_read_write_accept_absolute_and_ancestor_paths(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  directory = tmp_path / 'nested'
  directory.mkdir()
  nested = extension.Workspace(directory, False, shell=shell.Shell(directory))

  await nested.write('../config', 'value')

  assert await workspace.read(tmp_path / 'config') == 'value'


@pytest.mark.asyncio
async def test_read_write_enforce_byte_not_character_limits(
  workspace: extension.Workspace,
) -> None:
  await workspace.write('text', 'éé', max_bytes=4)
  assert await workspace.read('text', max_bytes=4) == 'éé'
  with pytest.raises(ValueError, match='byte limit'):
    await workspace.read('text', max_bytes=3)
  with pytest.raises(ValueError, match='byte limit'):
    await workspace.write('text', 'ééé', max_bytes=4)
  assert await workspace.read('text') == 'éé'


@pytest.mark.asyncio
async def test_read_rejects_invalid_utf8(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  (tmp_path / 'binary').write_bytes(b'\xff')
  with pytest.raises(UnicodeDecodeError):
    await workspace.read('binary')


@pytest.mark.asyncio
async def test_read_missing_file_has_specific_error(
  workspace: extension.Workspace,
) -> None:
  with pytest.raises(FileNotFoundError, match='missing'):
    await workspace.read('missing')


@pytest.mark.asyncio
async def test_read_write_without_backend_never_touch_host(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'file'
  path.write_text('host')
  workspace = extension.Workspace(tmp_path, False)
  with pytest.raises(RuntimeError, match='unavailable'):
    await workspace.read(path)
  with pytest.raises(RuntimeError, match='unavailable'):
    await workspace.write(path, 'changed')
  assert path.read_text() == 'host'


@pytest.mark.asyncio
async def test_write_replaces_file_preserving_mode(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  path = tmp_path / 'file'
  path.write_text('old')
  path.chmod(0o751)
  previous = tmp_path / 'previous'
  previous.hardlink_to(path)

  await workspace.write('file', 'new')

  assert path.read_text() == 'new'
  assert previous.read_text() == 'old'
  assert stat.S_IMODE(path.stat().st_mode) == 0o751
  assert not tuple(tmp_path.glob('.mirror-write-*'))


@pytest.mark.asyncio
async def test_write_expected_checks_even_empty_original(
  workspace: extension.Workspace,
) -> None:
  await workspace.write('file', '')
  await workspace.write('file', 'edited', expected='')
  with pytest.raises(ValueError, match='changed since'):
    await workspace.write('file', 'stale edit', expected='')
  assert await workspace.read('file') == 'edited'


@pytest.mark.asyncio
async def test_write_expected_missing_file_does_not_create_directories(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  with pytest.raises(FileNotFoundError):
    await workspace.write('missing/file', 'edited', expected='original')
  assert not (tmp_path / 'missing').exists()


@pytest.mark.asyncio
async def test_read_write_reject_symlink_files(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  target = tmp_path / 'target'
  target.write_text('untouched')
  (tmp_path / 'link').symlink_to(target)
  with pytest.raises(ValueError):
    await workspace.read('link')
  with pytest.raises(ValueError):
    await workspace.write('link', 'changed')
  assert target.read_text() == 'untouched'
  assert (tmp_path / 'link').is_symlink()


@pytest.mark.asyncio
async def test_read_write_reject_symlink_directories(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  target = tmp_path / 'target'
  target.mkdir()
  (target / 'file').write_text('untouched')
  (tmp_path / 'link').symlink_to(target, target_is_directory=True)
  with pytest.raises(ValueError):
    await workspace.read('link/file')
  with pytest.raises(ValueError):
    await workspace.write('link/file', 'changed')
  assert (target / 'file').read_text() == 'untouched'


@pytest.mark.asyncio
async def test_read_write_reject_fifo_without_blocking(
  tmp_path: pathlib.Path, workspace: extension.Workspace
) -> None:
  os.mkfifo(tmp_path / 'pipe')
  with pytest.raises(ValueError, match='regular file'):
    await workspace.read('pipe')
  with pytest.raises(ValueError, match='regular file'):
    await workspace.write('pipe', 'changed')
