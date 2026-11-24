from __future__ import annotations

import pathlib
import subprocess
import urllib.parse

import pytest
from mirror.protocol import mirror_pb2

from mirror.cli import hosted_workspace


_WITH_CREDENTIALS = urllib.parse.urlunsplit((
  'https',
  '@'.join((':'.join(('synthetic-user', 'synthetic-password')), 'github.com')),
  '/acme/widgets.git',
  '',
  '',
))


def _git(repository: pathlib.Path, *arguments: str) -> None:
  subprocess.run(
    (
      'git',
      '-C',
      str(repository),
      '-c',
      'user.name=Test',
      '-c',
      'user.email=test@example.invalid',
      '-c',
      'commit.gpgsign=false',
      *arguments,
    ),
    check=True,
    capture_output=True,
  )


@pytest.fixture
def clone(tmp_path: pathlib.Path) -> pathlib.Path:
  """A committed repository on `main` with a tracked `package` directory."""
  repository = tmp_path / 'widgets'
  (repository / 'package').mkdir(parents=True)
  _git(repository, 'init', '--initial-branch=main')
  (repository / 'package' / 'module.py').write_text('value = 1\n')
  _git(repository, 'add', 'package')
  _git(repository, 'commit', '--message', 'initial')
  return repository


@pytest.mark.parametrize(
  ('remote', 'expected'),
  [
    ('git@github.com:Acme/Widgets.git', 'https://github.com/Acme/Widgets'),
    ('ssh://git@github.com/acme/widgets', 'https://github.com/acme/widgets'),
    ('https://GitHub.com/acme/widgets/', 'https://github.com/acme/widgets'),
  ],
)
def test_repository_url_normalizes_clone_addresses(
  remote: str, expected: str
) -> None:
  assert hosted_workspace.repository_url(remote) == expected


@pytest.mark.parametrize(
  'remote',
  [
    'http://github.com/acme/widgets',
    _WITH_CREDENTIALS,
    'https://github.com/',
    'https://github.com/acme/../widgets',
  ],
)
def test_repository_url_rejects_what_a_server_cannot_clone(remote: str) -> None:
  with pytest.raises(ValueError, match='Workspace|Repository'):
    hosted_workspace.repository_url(remote)


@pytest.mark.asyncio
async def test_for_directory_uses_the_tracked_origin_branch_and_directory(
  clone: pathlib.Path,
) -> None:
  _git(clone, 'remote', 'add', 'origin', 'git@github.com:acme/widgets.git')
  _git(clone, 'config', 'branch.main.remote', 'origin')
  _git(clone, 'config', 'branch.main.merge', 'refs/heads/release')
  _git(clone, 'update-ref', 'refs/remotes/origin/release', 'HEAD')
  untracked = clone / 'package' / 'scratch'
  untracked.mkdir()

  workspace = await hosted_workspace.for_directory(untracked)

  assert workspace == mirror_pb2.Workspace(
    name='widgets',
    repository_url='https://github.com/acme/widgets',
    revision='release',
    directory='package',
  )


@pytest.mark.asyncio
async def test_for_directory_leaves_out_a_directory_the_revision_lacks(
  clone: pathlib.Path,
) -> None:
  _git(clone, 'remote', 'add', 'origin', 'git@github.com:acme/widgets.git')
  _git(clone, 'config', 'branch.main.remote', 'origin')
  _git(clone, 'config', 'branch.main.merge', 'refs/heads/main')
  _git(clone, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
  unpushed = clone / 'unpushed'
  unpushed.mkdir()
  (unpushed / 'module.py').write_text('value = 2\n')
  _git(clone, 'add', 'unpushed')
  _git(clone, 'commit', '--message', 'local only')

  workspace = await hosted_workspace.for_directory(unpushed)

  assert workspace == mirror_pb2.Workspace(
    name='widgets',
    repository_url='https://github.com/acme/widgets',
    revision='main',
  )


@pytest.mark.asyncio
async def test_for_directory_skips_a_path_that_is_a_file_in_the_revision(
  clone: pathlib.Path,
) -> None:
  _git(clone, 'remote', 'add', 'origin', 'git@github.com:acme/widgets.git')
  _git(clone, 'config', 'branch.main.remote', 'origin')
  _git(clone, 'config', 'branch.main.merge', 'refs/heads/main')
  _git(clone, 'update-ref', 'refs/remotes/origin/main', 'HEAD')
  _git(clone, 'rm', 'package/module.py')
  _git(clone, 'commit', '--message', "a directory takes the file's place")
  replacement = clone / 'package' / 'module.py'
  replacement.mkdir(parents=True)

  workspace = await hosted_workspace.for_directory(replacement)

  assert workspace == mirror_pb2.Workspace(
    name='widgets',
    repository_url='https://github.com/acme/widgets',
    revision='main',
    directory='package',
  )


@pytest.mark.asyncio
async def test_for_directory_reads_the_copy_a_custom_fetch_mapping_keeps(
  clone: pathlib.Path,
) -> None:
  _git(clone, 'remote', 'add', 'origin', 'git@github.com:acme/widgets.git')
  _git(
    clone,
    'config',
    'remote.origin.fetch',
    '+refs/heads/*:refs/remotes/mirrored/*',
  )
  _git(clone, 'config', 'branch.main.remote', 'origin')
  _git(clone, 'config', 'branch.main.merge', 'refs/heads/main')
  _git(clone, 'update-ref', 'refs/remotes/mirrored/main', 'HEAD')

  workspace = await hosted_workspace.for_directory(clone / 'package')

  assert workspace == mirror_pb2.Workspace(
    name='widgets',
    repository_url='https://github.com/acme/widgets',
    revision='main',
    directory='package',
  )


@pytest.mark.asyncio
async def test_for_directory_falls_back_to_the_default_branch(
  clone: pathlib.Path,
) -> None:
  _git(clone, 'remote', 'add', 'origin', _WITH_CREDENTIALS)
  _git(clone, 'switch', '--create', 'unpushed')

  workspace = await hosted_workspace.for_directory(clone)

  assert workspace == mirror_pb2.Workspace(
    name='widgets',
    repository_url='https://github.com/acme/widgets',
    revision='HEAD',
  )


@pytest.mark.asyncio
async def test_for_directory_without_a_usable_origin_is_none(
  clone: pathlib.Path, tmp_path: pathlib.Path
) -> None:
  outside = tmp_path / 'plain'
  outside.mkdir()

  assert await hosted_workspace.for_directory(clone) is None
  assert await hosted_workspace.for_directory(outside) is None
