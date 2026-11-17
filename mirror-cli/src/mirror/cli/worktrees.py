"""Client-side checkout descriptions and managed-path classification."""

from __future__ import annotations

import dataclasses
import pathlib

from mirror import project


class WorktreeError(RuntimeError):
  """An isolated workspace operation failed."""


@dataclasses.dataclass(frozen=True)
class WorktreeIdentity:
  """Opaque provider-owned identity independent of worktree location."""

  provider: str
  id: str


@dataclasses.dataclass(frozen=True)
class Worktree:
  """An isolated checkout described by the daemon."""

  path: pathlib.Path
  name: str | None
  identity: WorktreeIdentity | None = None
  warning: str | None = dataclasses.field(default=None, compare=False)

  @property
  def provider(self) -> str:
    """The provider namespace that owns `id`."""
    return self.identity.provider if self.identity is not None else 'local'

  @property
  def id(self) -> str:
    """The immutable identifier used by the underlying VCS."""
    if self.identity is not None:
      return self.identity.id
    return self.name or str(self.path.resolve())

  @property
  def label(self) -> str:
    """The compact name shown in the worktree switcher."""
    if self.name is not None and self.name.startswith('mirror/'):
      return self.path.name
    return self.name or self.path.name or str(self.path)


def is_managed(worktree: Worktree) -> bool:
  """Whether `worktree` lives under Mirror's managed checkout directory."""
  managed = (pathlib.Path.home() / '.mirror' / 'worktrees').resolve()
  return worktree.path.resolve().is_relative_to(managed)


def is_managed_directory(path: pathlib.Path) -> bool:
  """Whether the directory is a checkout root under Mirror's managed directory."""
  path = path.resolve()
  managed = (pathlib.Path.home() / '.mirror' / 'worktrees').resolve()
  return path.is_relative_to(managed) and project.repository_root(path) == path
