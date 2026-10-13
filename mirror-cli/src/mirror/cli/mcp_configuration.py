"""Worker-startup MCP configuration; discovery and execution belong to Rust."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class McpConnection:
  """An MCP endpoint with credential references, never credential values."""

  name: str
  config: Mapping[str, object]
  credential_directory: str = ''
