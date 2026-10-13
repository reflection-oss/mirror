"""Presentation-only summaries of consecutive tool calls."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from rich import console, text

_MCP_TOOLS = frozenset({'mcp_search', 'mcp_describe', 'mcp_call'})
_EXPLORATION_TOOLS = frozenset({
  'read',
  'grep',
  'find',
  'glob',
  'ls',
  'web_search',
  'web_browse',
})


def group_key(name: str) -> str | None:
  """The presentation group a tool belongs to, if any."""
  if name in _MCP_TOOLS:
    return 'tool:mcp'
  if name in _EXPLORATION_TOOLS:
    return 'exploration'
  parts = name.split('__', 2)
  if len(parts) == 3 and all(parts):
    return f'{parts[0]}:{parts[1]}'
  if not name or name in {'bash', 'edit', 'write', 'subagent'}:
    return None
  return f'tool:{name}'


def display_name(name: str) -> str:
  """Hide tool namespaces in human-facing labels."""
  if name in _MCP_TOOLS:
    return name.removeprefix('mcp_')
  parts = name.split('__', 2)
  if len(parts) == 3 and all(parts):
    return parts[2]
  return name


def status_label(key: str) -> str:
  return (
    'Exploring' if key == 'exploration' else f'Using {key.partition(":")[2]}'
  )


@dataclasses.dataclass(frozen=True)
class ToolGroup:
  """Completed tool attempts belonging to one semantic group."""

  key: str = 'exploration'
  reads: int = 0
  searches: int = 0
  listings: int = 0
  web_searches: int = 0
  web_browses: int = 0
  tool_counts: tuple[tuple[str, int], ...] = ()
  latest: str = ''

  def merge(self, other: ToolGroup) -> ToolGroup:
    counts = dict(self.tool_counts)
    for name, count in other.tool_counts:
      counts[name] = counts.get(name, 0) + count
    return ToolGroup(
      key=self.key,
      reads=self.reads + other.reads,
      searches=self.searches + other.searches,
      listings=self.listings + other.listings,
      web_searches=self.web_searches + other.web_searches,
      web_browses=self.web_browses + other.web_browses,
      tool_counts=tuple(counts.items()),
      latest=other.latest or self.latest,
    )

  def __rich_console__(
    self, console: console.Console, options: console.ConsoleOptions
  ) -> Iterator[text.Text]:
    del console, options
    yield self.summary()

  def summary(self, *, active: bool = False) -> text.Text:
    """A bounded header shared by the live panel and committed transcript."""
    label = self.key.partition(':')[2]
    if self.key == 'exploration':
      label = 'Exploring' if active else 'Explored'
    parts: list[str] = []
    for count, singular, plural in (
      (self.reads, 'read', 'reads'),
      (self.searches, 'search', 'searches'),
      (self.listings, 'directory scan', 'directory scans'),
      (self.web_searches, 'web search', 'web searches'),
      (self.web_browses, 'web browse', 'web browses'),
    ):
      if count:
        parts.append(f'{count} {singular if count == 1 else plural}')
    parts.extend(f'{count} {name}' for name, count in self.tool_counts)
    summary = text.Text(
      style='ui.tool-heading', no_wrap=True, overflow='ellipsis'
    )
    summary.append('· ', style='dim')
    summary.append(label, style='bold')
    if parts:
      summary.append(' · ' + ', '.join(parts))
    return summary


def result(
  name: str, args: str = '', *, success: bool, operation: str = ''
) -> ToolGroup | None:
  """Summarize a grouped attempt without retaining its output payload."""
  key = group_key(name)
  if key is None:
    return None
  verb = {
    'read': 'Read',
    'ls': 'Scanned',
    'grep': 'Searched',
    'glob': 'Searched',
    'find': 'Searched',
    'web_search': 'Web searched',
    'web_browse': 'Web browsed',
  }.get(name, display_name(name).capitalize())
  if not success:
    verb = f'{display_name(name)} failed'
  counted_name = (
    operation if key.startswith('tool:') and operation else display_name(name)
  )
  return ToolGroup(
    key=key,
    reads=int(success and name == 'read'),
    searches=int(success and name in {'grep', 'find', 'glob'}),
    listings=int(success and name == 'ls'),
    web_searches=int(name == 'web_search'),
    web_browses=int(name == 'web_browse'),
    tool_counts=((counted_name, 1),) if key != 'exploration' else (),
    latest=f'{verb} {args}'.rstrip(),
  )
