"""Extension agent options and the built-in tool catalog."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Options:
  """Additional instructions for a new agent, preserving workspace context."""

  system_prompt: str | None = None


def builtins() -> tuple[str, ...]:
  """Return the built-in names; integration tests enforce parity with Rust.

  Memory and MCP tools are present only when their session capabilities are
  available; the worker drops them otherwise.
  """
  return (
    'read',
    'bash',
    'write',
    'edit',
    'grep',
    'glob',
    'ls',
    'task',
    'web_search',
    'web_browse',
    'subagent',
    'search_memory',
    'save_memory',
    'mcp_search',
    'mcp_describe',
    'mcp_call',
  )
