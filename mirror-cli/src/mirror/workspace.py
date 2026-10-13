"""Project-directory identity helpers for conversation listing."""

from __future__ import annotations

import enum
import functools
import pathlib
import shutil
import subprocess

from mirror import project

_GIT = shutil.which('git') or 'git'


class SessionScope(enum.Enum):
  """How `/resume` chooses which conversations to show."""

  CWD = 'cwd'
  GIT = 'git'
  ALL = 'all'


_SCOPE_ORDER = (SessionScope.CWD, SessionScope.GIT, SessionScope.ALL)


def next_session_scope(scope: SessionScope) -> SessionScope:
  """Cycle cwd → git → all → cwd."""
  return _SCOPE_ORDER[(_SCOPE_ORDER.index(scope) + 1) % len(_SCOPE_ORDER)]


def resolve_cwd(cwd: str | pathlib.Path) -> str:
  """Return one absolute form of a project directory path."""
  return str(pathlib.Path(cwd).resolve())


@functools.lru_cache(maxsize=256)
def git_workspace_key(cwd: str) -> str | None:
  """Stable identity shared by a repository's isolated checkouts."""
  resolved = resolve_cwd(cwd)
  try:
    jj_repository = project.jj_repository(pathlib.Path(resolved))
  except OSError:
    jj_repository = None
  if jj_repository is not None:
    return str(jj_repository)
  try:
    completed = subprocess.run(
      (
        _GIT,
        '-C',
        resolved,
        'rev-parse',
        '--path-format=absolute',
        '--git-common-dir',
      ),
      check=False,
      capture_output=True,
      text=True,
      close_fds=False,
    )
  except OSError:
    return None
  if completed.returncode != 0:
    return None
  common_dir = completed.stdout.strip()
  if not common_dir:
    return None
  return str(pathlib.Path(common_dir).resolve())
