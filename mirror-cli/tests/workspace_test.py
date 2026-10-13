"""Tests for conversation workspace identity helpers."""

from __future__ import annotations

import pathlib
import subprocess

from mirror import workspace


def _git(cwd: pathlib.Path, *args: str) -> None:
  subprocess.run(
    ['git', '-C', str(cwd), *args],
    check=True,
    capture_output=True,
    text=True,
  )


def test_git_workspace_key_unifies_linked_worktrees(
  tmp_path: pathlib.Path,
) -> None:
  workspace.git_workspace_key.cache_clear()
  main = tmp_path / 'main'
  linked = tmp_path / 'linked'
  main.mkdir()
  _git(main, 'init')
  _git(main, 'config', 'user.email', 'test@example.com')
  _git(main, 'config', 'user.name', 'Test')
  (main / 'README').write_text('hi\n', encoding='utf-8')
  _git(main, 'add', 'README')
  _git(main, 'commit', '-m', 'init')
  _git(main, 'worktree', 'add', str(linked), '-b', 'feature')
  (main / '.jj').mkdir()  # An incomplete Jujutsu marker still permits Git.

  main_key = workspace.git_workspace_key(str(main))
  linked_key = workspace.git_workspace_key(str(linked))
  assert main_key is not None
  assert main_key == linked_key
  assert workspace.git_workspace_key(str(tmp_path / 'missing')) is None


def test_git_workspace_key_unifies_jujutsu_workspaces(
  tmp_path: pathlib.Path,
) -> None:
  workspace.git_workspace_key.cache_clear()
  main = tmp_path / 'main'
  linked = tmp_path / 'linked'
  (main / '.jj' / 'repo').mkdir(parents=True)
  (linked / '.jj').mkdir(parents=True)
  (linked / '.jj' / 'repo').write_text('../../main/.jj/repo')

  main_key = workspace.git_workspace_key(str(main))
  linked_key = workspace.git_workspace_key(str(linked))

  assert main_key is not None
  assert main_key == linked_key


def test_session_scope_cycles_through_every_scope() -> None:
  assert (
    workspace.next_session_scope(workspace.SessionScope.CWD)
    is workspace.SessionScope.GIT
  )
  assert (
    workspace.next_session_scope(workspace.SessionScope.GIT)
    is workspace.SessionScope.ALL
  )
  assert (
    workspace.next_session_scope(workspace.SessionScope.ALL)
    is workspace.SessionScope.CWD
  )
