"""Tests for shared project-root discovery."""

from __future__ import annotations

import pathlib

from mirror import project


def test_directories_returns_root_through_working_directory(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'repo'
  cwd = root / 'src' / 'package'
  cwd.mkdir(parents=True)
  (root / '.git').mkdir()

  assert project.directories(cwd) == [root, root / 'src', cwd]
  assert project.is_repository(cwd)


def test_directories_recognizes_jujutsu_workspace(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'repo'
  cwd = root / 'src' / 'package'
  cwd.mkdir(parents=True)
  (root / '.jj').mkdir()

  assert project.directories(cwd) == [root, root / 'src', cwd]
  assert project.is_repository(cwd)


def test_directories_stops_at_nearest_nested_root(
  tmp_path: pathlib.Path,
) -> None:
  outer = tmp_path / 'outer'
  inner = outer / 'inner'
  cwd = inner / 'package'
  cwd.mkdir(parents=True)
  (outer / '.git').mkdir()
  (inner / '.git').write_text('gitdir: elsewhere')

  assert project.directories(cwd) == [inner, cwd]


def test_directories_without_root_returns_only_working_directory(
  tmp_path: pathlib.Path,
) -> None:
  cwd = tmp_path / 'untracked' / 'package'
  cwd.mkdir(parents=True)

  assert project.directories(cwd) == [cwd]
  assert not project.is_repository(cwd)


def test_resource_directories_include_global_and_project_ancestors(
  tmp_path: pathlib.Path,
) -> None:
  root = tmp_path / 'repo'
  cwd = root / 'nested'
  cwd.mkdir(parents=True)
  (root / '.git').mkdir()
  global_dir = tmp_path / 'global'

  assert project.resource_directories(cwd, 'extensions', global_dir) == [
    global_dir,
    root / '.mirror' / 'extensions',
    cwd / '.mirror' / 'extensions',
  ]
