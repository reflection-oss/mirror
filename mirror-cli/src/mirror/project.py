"""Project-root discovery shared by instruction and skill loading."""

from __future__ import annotations

import pathlib

ROOT_MARKERS = ('.git', '.jj')
GLOBAL_DIR = pathlib.Path.home() / '.mirror'
PROJECT_DIR = pathlib.Path('.mirror')


def _is_root(directory: pathlib.Path) -> bool:
  return any((directory / marker).exists() for marker in ROOT_MARKERS)


def repository_root(cwd: pathlib.Path) -> pathlib.Path | None:
  """Return the nearest repository working-copy root containing `cwd`."""
  return next(
    (directory for directory in (cwd, *cwd.parents) if _is_root(directory)),
    None,
  )


def jj_workspace_root(cwd: pathlib.Path) -> pathlib.Path | None:
  """Return the nearest Jujutsu workspace root containing `cwd`."""
  return next(
    (
      directory
      for directory in (cwd, *cwd.parents)
      if (directory / '.jj').is_dir()
    ),
    None,
  )


def jj_repository(cwd: pathlib.Path) -> pathlib.Path | None:
  """Return the shared Jujutsu repository backing the nearest workspace."""
  root = jj_workspace_root(cwd)
  if root is None:
    return None
  marker = root / '.jj' / 'repo'
  if marker.is_dir():
    return marker.resolve()
  target = marker.read_text().strip()
  return (marker.parent / target).resolve()


def directories(cwd: pathlib.Path) -> list[pathlib.Path]:
  """Return directories from the nearest project root through `cwd`."""
  walked: list[pathlib.Path] = []
  for directory in (cwd, *cwd.parents):
    walked.append(directory)
    if _is_root(directory):
      return list(reversed(walked))
  return [cwd]


def resource_directories(
  cwd: pathlib.Path,
  name: str,
  global_dir: pathlib.Path | None = None,
  *,
  project_dirs: tuple[pathlib.Path, ...] | None = None,
  include_global: bool = True,
) -> list[pathlib.Path]:
  """Return global and project-ancestor directories for one resource."""
  roots = (
    [global_dir if global_dir is not None else GLOBAL_DIR / name]
    if include_global
    else []
  )
  relative_dirs = project_dirs or (PROJECT_DIR / name,)
  roots.extend(
    directory / relative
    for directory in directories(cwd)
    for relative in relative_dirs
  )
  return roots


def is_repository(cwd: pathlib.Path) -> bool:
  """Whether `cwd` or an ancestor contains the project-root marker."""
  return repository_root(cwd) is not None
