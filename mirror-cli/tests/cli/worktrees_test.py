from __future__ import annotations

import concurrent.futures
import os
import pathlib
import shutil
import subprocess
import threading
from collections.abc import Sequence
from unittest import mock

import pytest

from mirror.cli import worktrees


def _git(cwd: pathlib.Path, *arguments: str) -> str:
  completed = subprocess.run(
    ('git', '-C', str(cwd), *arguments),
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _jj_change_id(cwd: pathlib.Path) -> str:
  completed = subprocess.run(
    (
      'jj',
      '--repository',
      str(cwd),
      'log',
      '-r',
      '@',
      '--no-graph',
      '-T',
      'change_id',
    ),
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _repository(path: pathlib.Path) -> pathlib.Path:
  path.mkdir()
  remote = path.with_name(f'{path.name}-origin.git')
  subprocess.run(
    ('git', 'init', '--bare', str(remote)),
    check=True,
    capture_output=True,
    text=True,
  )
  _git(path, 'init', '--initial-branch=main')
  _git(path, 'config', 'user.email', 'mirror@example.com')
  _git(path, 'config', 'user.name', 'Mirror Test')
  (path / 'tracked.txt').write_text('base\n')
  _git(path, 'add', 'tracked.txt')
  _git(path, 'commit', '-m', 'base')
  _git(path, 'remote', 'add', 'origin', str(remote))
  _git(path, 'push', '--set-upstream', 'origin', 'main')
  return path


def test_list_worktrees_includes_linked_checkout(
  tmp_path: pathlib.Path,
) -> None:
  repository = _repository(tmp_path / 'repository')
  linked = tmp_path / 'linked'
  _git(repository, 'worktree', 'add', '-b', 'feature', str(linked))

  found = worktrees.list_worktrees(repository)
  default_branch = _git(repository, 'branch', '--show-current')

  assert [(item.path, item.name) for item in found] == [
    (repository.resolve(), default_branch),
    (linked.resolve(), 'feature'),
  ]

  shutil.rmtree(linked)
  assert worktrees.list_worktrees(repository) == (
    worktrees.Worktree(
      repository.resolve(),
      default_branch,
      worktrees.WorktreeIdentity('git', str(repository.resolve())),
    ),
  )


def test_list_worktrees_ignores_locked_missing_checkout(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repository = tmp_path / 'repository'
  repository.mkdir()
  missing = tmp_path / 'missing'
  output = f"""worktree {repository}
HEAD 1111111111111111111111111111111111111111
branch refs/heads/main

worktree {missing}
HEAD 2222222222222222222222222222222222222222
detached
locked initializing
"""

  def fake_git(*_: object, **__: object) -> str:
    return output

  monkeypatch.setattr(worktrees, '_git', fake_git)

  assert worktrees.list_worktrees(repository) == (
    worktrees.Worktree(
      repository.resolve(),
      'main',
      worktrees.WorktreeIdentity('git', str(repository.resolve())),
    ),
  )


def test_create_worktree_uses_latest_main_without_current_changes(
  tmp_path: pathlib.Path,
  monkeypatch,
) -> None:
  repository = _repository(tmp_path / 'repository')
  updater = tmp_path / 'updater'
  subprocess.run(
    (
      'git',
      'clone',
      '--branch',
      'main',
      _git(repository, 'remote', 'get-url', 'origin'),
      updater,
    ),
    check=True,
    capture_output=True,
    text=True,
  )
  _git(updater, 'config', 'user.email', 'mirror@example.com')
  _git(updater, 'config', 'user.name', 'Mirror Test')
  _git(updater, 'checkout', '--orphan', 'replacement-main')
  (updater / 'tracked.txt').write_text('latest main\n')
  _git(updater, 'add', 'tracked.txt')
  _git(updater, 'commit', '-m', 'replace main')
  latest_main = _git(updater, 'rev-parse', 'HEAD')
  _git(updater, 'push', '--force', 'origin', 'HEAD:main')
  mirror_home = tmp_path / 'home'
  mirror_home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: mirror_home)
  (repository / 'tracked.txt').write_text('current work\n')
  (repository / 'new.txt').write_text('new\n')
  source_head = _git(repository, 'rev-parse', 'HEAD')

  created = worktrees.create_worktree(repository, 'release-prep')

  assert created.name == 'mirror/release-prep'
  assert created.path.name == 'release-prep'
  assert created.label == 'release-prep'
  assert (created.path / 'tracked.txt').read_text() == 'latest main\n'
  assert not (created.path / 'new.txt').exists()
  assert (repository / 'tracked.txt').read_text() == 'current work\n'
  assert _git(repository, 'rev-parse', 'HEAD') == source_head
  assert _git(created.path, 'rev-parse', 'HEAD') == latest_main


def test_overlapping_git_creations_use_their_own_fetched_revision(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  repository = _repository(tmp_path / 'repository')
  original = _git(repository, 'rev-parse', 'HEAD')
  (repository / 'tracked.txt').write_text('updated main\n')
  _git(repository, 'commit', '-am', 'update')
  updated = _git(repository, 'rev-parse', 'HEAD')
  _git(repository, 'push', 'origin', 'HEAD:future-main')
  remote = pathlib.Path(_git(repository, 'remote', 'get-url', 'origin'))
  (repository / 'tracked.txt').write_text('local work\n')
  home = tmp_path / 'home'
  home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: home)
  first_fetched = threading.Event()
  second_fetched = threading.Event()
  run_git = worktrees._git

  def git(cwd: pathlib.Path, arguments: Sequence[str]) -> str:
    output = run_git(cwd, arguments)
    if arguments[0] == 'fetch':
      if not first_fetched.is_set():
        first_fetched.set()
        assert second_fetched.wait(timeout=5)
      else:
        second_fetched.set()
    return output

  monkeypatch.setattr(worktrees, '_git', git)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    first = executor.submit(worktrees.create_worktree, repository, 'first')
    try:
      assert first_fetched.wait(timeout=5)
      _git(remote, 'update-ref', 'refs/heads/main', updated)
      second = executor.submit(worktrees.create_worktree, repository, 'second')
      first_created = first.result(timeout=5)
      second_created = second.result(timeout=5)
    finally:
      second_fetched.set()

  assert _git(first_created.path, 'rev-parse', 'HEAD') == original
  assert _git(second_created.path, 'rev-parse', 'HEAD') == updated
  assert first_created.path != second_created.path
  assert first_created.id != second_created.id
  assert (repository / 'tracked.txt').read_text() == 'local work\n'
  assert (
    _git(repository, 'for-each-ref', '--format=%(refname)', 'refs/mirror/')
    == ''
  )


def test_failed_creation_preserves_existing_worktree_and_cleans_fetch_ref(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  repository = _repository(tmp_path / 'repository')
  home = tmp_path / 'home'
  home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: home)
  created = worktrees.create_worktree(repository, 'review')
  (created.path / 'tracked.txt').write_text('keep this work\n')

  with pytest.raises(worktrees.WorktreeError):
    worktrees.create_worktree(repository, 'review')

  assert (created.path / 'tracked.txt').read_text() == 'keep this work\n'
  assert created in worktrees.list_worktrees(repository)
  assert (
    _git(repository, 'for-each-ref', '--format=%(refname)', 'refs/mirror/')
    == ''
  )


def test_create_worktree_blank_name_uses_generated_folder(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repository = _repository(tmp_path / 'repository')
  mirror_home = tmp_path / 'home'
  mirror_home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: mirror_home)

  created = worktrees.create_worktree(repository, '   ')

  assert len(created.path.name) == 8
  assert created.name == f'mirror/{created.path.name}'


def test_create_worktree_rejects_path_name(
  tmp_path: pathlib.Path,
) -> None:
  repository = _repository(tmp_path / 'repository')

  with pytest.raises(worktrees.WorktreeError, match='worktree name'):
    worktrees.create_worktree(repository, '../outside')


def test_drop_worktree_removes_managed_git_checkout(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repository = _repository(tmp_path / 'repository')
  mirror_home = tmp_path / 'home'
  mirror_home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: mirror_home)
  created = worktrees.create_worktree(repository)
  (created.path / 'uncommitted.txt').write_text('discarded\n')

  worktrees.drop_worktree(created)

  assert not created.path.exists()
  assert created not in worktrees.list_worktrees(repository)


def test_drop_worktree_requires_confirmation_for_unmanaged_checkout(
  tmp_path: pathlib.Path,
) -> None:
  repository = _repository(tmp_path / 'repository')
  linked = tmp_path / 'linked'
  _git(repository, 'worktree', 'add', '-b', 'feature', str(linked))
  worktree = worktrees.Worktree(linked, 'feature')

  with pytest.raises(worktrees.WorktreeError, match='requires confirmation'):
    worktrees.drop_worktree(worktree)

  assert linked.exists()
  worktrees.drop_worktree(worktree, confirmed=True)
  assert not linked.exists()


def test_drop_worktree_refuses_primary_checkout_after_confirmation(
  tmp_path: pathlib.Path,
) -> None:
  repository = _repository(tmp_path / 'repository')
  worktree = worktrees.Worktree(repository, 'main')

  with pytest.raises(worktrees.WorktreeError, match='primary Git worktree'):
    worktrees.drop_worktree(worktree, confirmed=True)

  assert repository.exists()


def test_list_worktrees_orders_jj_workspaces_by_creation(
  tmp_path: pathlib.Path,
  monkeypatch,
) -> None:
  source = tmp_path / 'source'
  older = tmp_path / 'z-older'
  newer = tmp_path / 'a-newer'
  for workspace in (source, older, newer):
    marker = workspace / '.jj' / 'repo'
    marker.mkdir(parents=True)
  os.utime(older / '.jj' / 'repo', (1, 1))
  os.utime(newer / '.jj' / 'repo', (2, 2))
  calls: list[tuple[pathlib.Path, tuple[str, ...]]] = []

  def fake_jj(cwd: pathlib.Path, arguments: tuple[str, ...]) -> str:
    calls.append((cwd, arguments))
    if arguments[:2] == ('workspace', 'list'):
      return 'a-newer\0default\0z-older\0'
    return str(
      {'default': source, 'z-older': older, 'a-newer': newer}[arguments[-1]]
    )

  monkeypatch.setattr(worktrees, '_jj', fake_jj)

  found = worktrees.list_worktrees(source)

  assert found == (
    worktrees.Worktree(
      source.resolve(),
      'default',
      worktrees.WorktreeIdentity('jj', 'default'),
    ),
    worktrees.Worktree(
      older.resolve(),
      'z-older',
      worktrees.WorktreeIdentity('jj', 'z-older'),
    ),
    worktrees.Worktree(
      newer.resolve(),
      'a-newer',
      worktrees.WorktreeIdentity('jj', 'a-newer'),
    ),
  )
  assert calls[0][1][:2] == ('workspace', 'list')


@pytest.mark.skipif(shutil.which('jj') is None, reason='jj is not installed')
def test_create_jj_workspace_recovers_stale_source(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  source = _repository(tmp_path / 'source')
  subprocess.run(
    ('jj', 'git', 'init', '--colocate', str(source)),
    check=True,
    capture_output=True,
    text=True,
  )
  (source / 'tracked.txt').write_text('stale checkout\n')
  subprocess.run(
    ('jj', '--repository', str(source), 'status'),
    check=True,
    capture_output=True,
    text=True,
  )
  subprocess.run(
    (
      'jj',
      '--repository',
      str(source),
      'new',
      'main@origin',
      '--ignore-working-copy',
    ),
    check=True,
    capture_output=True,
    text=True,
  )
  stale = subprocess.run(
    ('jj', '--repository', str(source), 'log', '-r', '@'),
    check=False,
    capture_output=True,
    text=True,
  )
  assert stale.returncode != 0
  assert 'working copy is stale' in stale.stderr.lower()
  mirror_home = tmp_path / 'home'
  mirror_home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: mirror_home)

  created = worktrees.create_worktree(source, 'release-prep')

  assert _jj_change_id(source)
  assert (created.path / 'tracked.txt').read_text() == 'base\n'
  assert created in worktrees.list_worktrees(source)


@pytest.mark.skipif(shutil.which('jj') is None, reason='jj is not installed')
def test_drop_jj_workspace_ignores_stale_checkout(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  source = _repository(tmp_path / 'source')
  subprocess.run(
    ('jj', 'git', 'init', '--colocate', str(source)),
    check=True,
    capture_output=True,
    text=True,
  )
  mirror_home = tmp_path / 'home'
  mirror_home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: mirror_home)
  created = worktrees.create_worktree(source, 'release-prep')
  subprocess.run(
    (
      'jj',
      '--repository',
      str(created.path),
      'new',
      '--ignore-working-copy',
    ),
    check=True,
    capture_output=True,
    text=True,
  )

  worktrees.drop_worktree(created)

  assert not created.path.exists()
  assert created not in worktrees.list_worktrees(source)


@pytest.mark.skipif(shutil.which('jj') is None, reason='jj is not installed')
def test_create_jj_workspace_uses_main_without_current_changes(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  source = _repository(tmp_path / 'source')
  subprocess.run(
    ('jj', 'git', 'init', '--colocate', str(source)),
    check=True,
    capture_output=True,
    text=True,
  )
  (source / 'tracked.txt').write_text('current work\n')
  (source / 'untracked.txt').write_text('excluded\n')
  mirror_home = tmp_path / 'home'
  mirror_home.mkdir()
  monkeypatch.setattr(pathlib.Path, 'home', lambda: mirror_home)

  created = worktrees.create_worktree(source, 'release-prep')

  assert created.name == 'release-prep'
  assert created.path.name == 'release-prep'
  assert (created.path / 'tracked.txt').read_text() == 'base\n'
  assert not (created.path / 'untracked.txt').exists()
  assert created in worktrees.list_worktrees(source)
  assert _jj_change_id(source) != _jj_change_id(created.path)

  worktrees.drop_worktree(created)

  assert not created.path.exists()
  assert created not in worktrees.list_worktrees(source)


def test_current_worktree_resolves_nested_symlink_without_discovery(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repository = _repository(tmp_path / 'repository')
  nested = repository / 'package'
  nested.mkdir()
  link = tmp_path / 'link'
  link.symlink_to(nested, target_is_directory=True)
  monkeypatch.setattr(
    worktrees,
    'list_worktrees',
    mock.Mock(side_effect=AssertionError('must not enumerate worktrees')),
  )

  assert worktrees.current_worktree(link) == worktrees.Worktree(
    repository.resolve(),
    'main',
    worktrees.WorktreeIdentity('git', str(repository.resolve())),
  )
  _git(repository, 'checkout', '--detach')
  assert worktrees.current_worktree(link).name is None


@pytest.mark.skipif(shutil.which('jj') is None, reason='jj is not installed')
def test_current_worktree_ignores_missing_jj_sibling(
  tmp_path: pathlib.Path,
) -> None:
  source = tmp_path / 'source'
  subprocess.run(
    ('jj', 'git', 'init', '--colocate', str(source)),
    check=True,
    capture_output=True,
  )
  missing = tmp_path / 'deleted'
  subprocess.run(
    (
      'jj',
      '--repository',
      str(source),
      'workspace',
      'add',
      '--name',
      'deleted',
      str(missing),
    ),
    check=True,
    capture_output=True,
  )
  shutil.rmtree(missing)

  assert worktrees.current_worktree(source) == worktrees.Worktree(
    source.resolve(),
    'default',
    worktrees.WorktreeIdentity('jj', 'default'),
  )


def test_current_worktree_outside_repository(tmp_path: pathlib.Path) -> None:
  assert worktrees.current_worktree(tmp_path) == worktrees.Worktree(
    tmp_path.resolve(), None
  )


@pytest.mark.skipif(shutil.which('jj') is None, reason='jj is not installed')
def test_current_worktree_uses_jj_identity_for_colocated_and_linked_workspaces(
  tmp_path: pathlib.Path,
) -> None:
  source = tmp_path / 'source'
  subprocess.run(
    ('jj', 'git', 'init', '--colocate', str(source)),
    check=True,
    capture_output=True,
  )
  linked = tmp_path / 'linked'
  subprocess.run(
    (
      'jj',
      '--repository',
      str(source),
      'workspace',
      'add',
      '--name',
      'feature',
      str(linked),
    ),
    check=True,
    capture_output=True,
  )
  nested = linked / 'package'
  nested.mkdir()
  assert worktrees.current_worktree(source) == worktrees.Worktree(
    source.resolve(),
    'default',
    worktrees.WorktreeIdentity('jj', 'default'),
  )
  assert worktrees.current_worktree(nested) == worktrees.Worktree(
    linked.resolve(),
    'feature',
    worktrees.WorktreeIdentity('jj', 'feature'),
  )
