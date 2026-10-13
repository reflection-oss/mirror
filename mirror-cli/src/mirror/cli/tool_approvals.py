"""Versioned additive tool-call approvals, cached for one CLI lifetime."""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import cast

from mirror import config


@dataclasses.dataclass(frozen=True)
class ToolApproval:
  """An exact tool name and argument string in one execution scope."""

  scope: str
  tool: str
  arguments_sha256: str


@dataclasses.dataclass(frozen=True)
class ArgvPrefixApproval:
  """A literal argument prefix for one tool in one execution scope."""

  scope: str
  tool: str
  argv: tuple[str, ...]


type ApprovalRule = ToolApproval | ArgvPrefixApproval


def from_call(name: str, arguments: str, scope: str) -> ToolApproval | None:
  """Hash arguments verbatim, without interpreting tool-specific syntax."""
  if not name or not scope:
    return None
  try:
    digest = hashlib.sha256(arguments.encode('utf-8')).hexdigest()
  except UnicodeError:
    return None
  return ToolApproval(scope, name, digest)


class ToolApprovals:
  """A startup snapshot with atomic, merge-on-write explicit additions."""

  def __init__(self, path: pathlib.Path | None = None) -> None:
    self._config = config.Config(
      path
      if path is not None
      else config.default_path().with_name('tool_approvals.json')
    )
    self._entries: set[ApprovalRule] | None = None
    self.available = True

  def contains(self, approval: ApprovalRule) -> bool:
    """Load once, then consult memory only; a failed load disables remembering."""
    if self._entries is None:
      self._entries = set()
      try:
        self._entries = _read_entries(self._config.read())
      except (OSError, ValueError, RecursionError):
        self.available = False
        raise
    return approval in self._entries

  def remember(self, approval: ApprovalRule) -> None:
    """Persist before caching; duplicate approvals perform no file operations."""
    if self.contains(approval):
      return
    if not self.available:
      raise ValueError(
        'Remembered tool approvals are unavailable until restart'
      )
    entries: set[ApprovalRule] = set()

    def change(settings: dict[str, object]) -> None:
      nonlocal entries
      entries = _read_entries(settings)
      if approval not in entries:
        # Preserve unknown grant kinds from the locked snapshot, not our cache.
        raw_entries = cast(list[object], settings.setdefault('entries', []))
        raw_entries.append({
          'kind': 'exact_arguments'
          if isinstance(approval, ToolApproval)
          else 'argv_prefix',
          **dataclasses.asdict(approval),
        })
        entries.add(approval)
      settings['version'] = 1

    # Config merges other CLIs' additions under its existing lock and atomic replace.
    try:
      self._config.update(change)
    except (OSError, ValueError, RecursionError):
      self._entries = set()
      self.available = False
      raise
    self._entries = entries


def _read_entries(settings: dict[str, object]) -> set[ApprovalRule]:
  if not settings:
    return set()
  if (
    set(settings) != {'version', 'entries'}
    or type(settings['version']) is not int
    or settings['version'] != 1
  ):
    raise ValueError('Unsupported tool approvals format')
  raw_entries = settings['entries']
  if not isinstance(raw_entries, list):
    raise ValueError('Tool approvals entries must be a list')
  entries: set[ApprovalRule] = set()
  for raw in cast(list[object], raw_entries):
    if (entry := _read_entry(raw)) is not None:
      entries.add(entry)
  return entries


def _read_entry(raw: object) -> ApprovalRule | None:
  """Validate the shared envelope and decode only supported grant kinds."""
  if not isinstance(raw, dict):
    raise ValueError('Invalid tool approval entry')
  entry = cast(dict[str, object], raw)
  # Untagged v1 entries grant exact argument matching only.
  if set(entry) == {'scope', 'tool', 'arguments_sha256'}:
    entry = {'kind': 'exact_arguments', **entry}
  kind, scope, tool = entry.get('kind'), entry.get('scope'), entry.get('tool')
  if (
    not isinstance(kind, str)
    or not kind
    or not isinstance(scope, str)
    or not scope
    or not isinstance(tool, str)
    or not tool
  ):
    raise ValueError('Invalid tool approval kind, scope or name')
  if kind == 'argv_prefix':
    return _read_prefix(entry, scope, tool)
  # Unknown kinds are inert additive grants; restrictions require a version bump.
  if kind != 'exact_arguments':
    return None
  return _read_exact(entry, scope, tool)


def _read_exact(
  entry: dict[str, object], scope: str, tool: str
) -> ToolApproval:
  if set(entry) != {'kind', 'scope', 'tool', 'arguments_sha256'}:
    raise ValueError('Invalid exact tool approval entry')
  digest = entry['arguments_sha256']
  if (
    not isinstance(digest, str)
    or len(digest) != 64
    or any(character not in '0123456789abcdef' for character in digest)
  ):
    raise ValueError('Invalid tool approval argument digest')
  return ToolApproval(scope, tool, digest)


def _read_prefix(
  entry: dict[str, object], scope: str, tool: str
) -> ArgvPrefixApproval:
  if set(entry) != {'kind', 'scope', 'tool', 'argv'}:
    raise ValueError('Invalid argv prefix approval entry')
  raw = entry['argv']
  if not isinstance(raw, list) or not raw:
    raise ValueError('Tool approval argv must be a nonempty list')
  argv: list[str] = []
  for word in cast(list[object], raw):
    if not isinstance(word, str) or any(
      not char.isprintable() for char in word
    ):
      raise ValueError('Tool approval argv must contain printable strings')
    argv.append(word)
  if not argv[0]:
    raise ValueError('Tool approval argv must name an executable')
  return ArgvPrefixApproval(scope, tool, tuple(argv))
