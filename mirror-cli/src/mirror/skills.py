"""Discovery and loading of `SKILL.md` agent skills."""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Sequence
from typing import cast
from xml.sax import saxutils

import yaml

from mirror import project

SKILL_ROOT_DIRS: tuple[pathlib.Path, ...] = (
  pathlib.Path('.agents') / 'skills',
  pathlib.Path('.claude') / 'skills',
)

SKILL_FILENAME = 'SKILL.md'

_MAX_SCAN_DEPTH = 6
_MAX_SCAN_DIRS = 2000
_FRONTMATTER_FENCE = '---'
_FRONTMATTER_ENTRY = re.compile(r'^([A-Za-z_][A-Za-z0-9_-]*:\s*)(.*)$')
_VALID_SKILL_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')


@dataclasses.dataclass(frozen=True)
class Skill:
  """A single loaded skill."""

  name: str
  description: str
  body: str
  path: pathlib.Path
  disable_model_invocation: bool = False

  @property
  def directory(self) -> pathlib.Path:
    """The skill's base directory; relative references resolve against it."""
    return self.path.parent

  def invocation_content(self) -> str:
    """The contextual user fragment injected for an explicit invocation."""
    name = saxutils.escape(self.name)
    path = saxutils.escape(str(self.path))
    return f'<skill>\n<name>{name}</name>\n<path>{path}</path>\n{self.body}\n</skill>'

  @classmethod
  def from_file(cls, path: pathlib.Path) -> Skill | None:
    """Load a skill from a `SKILL.md` file; `None` if unreadable or undescribed."""
    try:
      raw = path.read_text('utf-8')
    except (OSError, UnicodeError):
      return None
    frontmatter, body = _split_frontmatter(raw)
    description_value = frontmatter.get('description')
    if not isinstance(description_value, str) or not description_value.strip():
      return None
    name_value = frontmatter.get('name')
    name = (
      name_value.strip()
      if isinstance(name_value, str) and name_value.strip()
      else path.parent.name
    )
    if _VALID_SKILL_NAME.fullmatch(name) is None:
      return None
    return cls(
      name=name,
      description=' '.join(description_value.split()),
      body=body,
      path=path,
      disable_model_invocation=(
        frontmatter.get('disable-model-invocation') is True
      ),
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, object], str]:
  """Extract and parse leading YAML frontmatter from a Markdown document."""
  normalized = (
    raw.removeprefix('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
  )
  lines = normalized.split('\n')
  if not lines or lines[0].strip() != _FRONTMATTER_FENCE:
    return {}, normalized.strip()
  end = next(
    (
      index
      for index, line in enumerate(lines[1:], start=1)
      if line.strip() == _FRONTMATTER_FENCE
    ),
    None,
  )
  if end is None:
    return {}, normalized.strip()
  block = '\n'.join(lines[1:end])
  body = '\n'.join(lines[end + 1 :]).strip()
  return _parse_frontmatter(block), body


def _parse_frontmatter(block: str) -> dict[str, object]:
  """Parse YAML, repairing common unquoted colon-containing scalars once."""
  try:
    parsed: object = yaml.safe_load(block)
  except yaml.YAMLError:
    repaired = _repair_unquoted_colons(block)
    if repaired == block:
      return {}
    try:
      parsed = yaml.safe_load(repaired)
    except yaml.YAMLError:
      return {}
  if not isinstance(parsed, dict):
    return {}
  values = cast(dict[object, object], parsed)
  return {key: value for key, value in values.items() if isinstance(key, str)}


def _repair_unquoted_colons(block: str) -> str:
  """Quote top-level plain scalars containing YAML-invalid `: ` sequences."""
  repaired: list[str] = []
  for line in block.splitlines():
    match = _FRONTMATTER_ENTRY.fullmatch(line)
    if match is None:
      repaired.append(line)
      continue
    prefix, raw_value = match.groups()
    scalar, comment = _split_yaml_comment(raw_value)
    value = scalar.strip()
    if (
      ': ' not in value
      or not value
      or value.startswith(('"', "'", '|', '>', '[', '{'))
    ):
      repaired.append(line)
      continue
    escaped = value.replace("'", "''")
    repaired.append(f"{prefix}'{escaped}'{comment}")
  return '\n'.join(repaired)


def _split_yaml_comment(value: str) -> tuple[str, str]:
  """Separate an inline YAML comment from a plain scalar."""
  for index, character in enumerate(value):
    if character == '#' and (index == 0 or value[index - 1].isspace()):
      comment_start = len(value[:index].rstrip())
      return value[:comment_start], value[comment_start:]
  return value, ''


def _skills_in_dir(directory: pathlib.Path) -> list[Skill]:
  """Recursively load skills under `directory`, within bounded traversal."""
  if not directory.is_dir():
    return []
  try:
    canonical_root = directory.resolve()
  except OSError:
    return []
  loaded: list[Skill] = []
  queue: list[tuple[pathlib.Path, int]] = [(directory, 0)]
  visited: set[pathlib.Path] = set()
  index = 0
  while index < len(queue) and len(visited) < _MAX_SCAN_DIRS:
    current, depth = queue[index]
    index += 1
    try:
      canonical = current.resolve()
    except OSError:
      continue
    if not canonical.is_relative_to(canonical_root) or canonical in visited:
      continue
    visited.add(canonical)

    skill_file = current / SKILL_FILENAME
    if skill_file.is_file():
      try:
        skill_path = skill_file.resolve()
      except OSError:
        continue
      if skill_path.is_relative_to(canonical_root):
        if (skill := Skill.from_file(skill_file)) is not None:
          loaded.append(skill)
      continue  # Resources nested inside a skill are not separate skills.
    if depth >= _MAX_SCAN_DEPTH:
      continue
    try:
      entries = sorted(current.iterdir())
    except OSError:
      continue
    for entry in entries:
      if entry.name.startswith('.') or entry.name == 'node_modules':
        continue
      try:
        if entry.is_dir():
          queue.append((entry, depth + 1))
      except OSError:
        continue
  return loaded


def load_skills(
  cwd: pathlib.Path,
  global_dir: pathlib.Path | None = None,
  *,
  include_global: bool = True,
) -> list[Skill]:
  """Return skills discoverable from `cwd`, walking up to the project root."""
  scan_roots = project.resource_directories(
    cwd,
    'skills',
    global_dir,
    project_dirs=SKILL_ROOT_DIRS,
    include_global=include_global,
  )

  by_name: dict[str, Skill] = {}
  seen_files: set[pathlib.Path] = set()
  for root in scan_roots:
    for skill in _skills_in_dir(root):
      try:
        canonical = skill.path.resolve()
      except OSError:
        canonical = skill.path.absolute()
      if canonical in seen_files:
        continue
      seen_files.add(canonical)
      by_name[skill.name] = skill
  return list(by_name.values())


def mentioned_skill_names(text: str, available: Sequence[Skill]) -> list[str]:
  """Return exact whitespace-delimited `$name` lookups, without repeats."""
  known = {skill.name for skill in available}
  seen: set[str] = set()
  mentioned: list[str] = []
  for token in text.split():
    if not token.startswith('$'):
      continue
    name = token[1:]
    if name in known and name not in seen:
      seen.add(name)
      mentioned.append(name)
  return mentioned
