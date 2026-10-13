"""Literal Bash argv extraction for remembered command prefixes."""

from __future__ import annotations

import json
import re
from typing import cast

import tree_sitter
import tree_sitter_bash
from mirror.cli import tool_approvals


def prefixes(
  arguments: str, scope: str
) -> tuple[tool_approvals.ArgvPrefixApproval, ...]:
  """Return literal argv prefixes, full command first, for saved-grant matching."""
  try:
    # Preserve duplicate keys so ambiguous JSON cannot acquire a prefix grant.
    raw: object = json.loads(arguments, object_pairs_hook=tuple)
  except (ValueError, RecursionError):
    return ()
  if not isinstance(raw, tuple):
    return ()
  pairs = cast(tuple[tuple[str, object], ...], raw)
  if len(pairs) != 1:
    return ()
  name, command = pairs[0]
  if name != 'command' or not isinstance(command, str):
    return ()
  argv = _literal_argv(command)
  return tuple(
    tool_approvals.ArgvPrefixApproval(scope, 'bash', argv[:length])
    for length in range(len(argv), 0, -1)
  )


def suggested_rule(
  full: tool_approvals.ArgvPrefixApproval,
) -> tool_approvals.ArgvPrefixApproval:
  """Suggest a command family for file viewers; keep other prefixes narrow."""
  # These are suggestions requiring user approval, not automatic allowances.
  if full.argv[0] in {'cat', 'head', 'tail', 'wc'}:
    return tool_approvals.ArgvPrefixApproval(
      full.scope, full.tool, full.argv[:1]
    )
  return full


def _literal_argv(command: str) -> tuple[str, ...]:
  """Fail closed unless the entire script is one command of literal words."""
  if not command or not command.isprintable():
    return ()
  source = command.encode('utf-8')
  parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_bash.language()))
  root = parser.parse(source).root_node
  if root.has_error or len(root.children) != 1:
    return ()
  node = root.children[0]
  if node.type != 'command':
    return ()
  words: list[str] = []
  for child in node.children:
    word = _literal_word(child, source)
    if word is None:
      return ()
    words.append(word)
  return tuple(words) if words and words[0] else ()


def _literal_word(node: tree_sitter.Node, source: bytes) -> str | None:
  """Decode only literal syntax; neither evaluate expansions nor guess escapes."""
  text = source[node.start_byte : node.end_byte].decode('utf-8')
  if node.type in ('word', 'number'):
    return text if re.fullmatch(r'[\w./:@%+,=-]+', text) else None
  if node.type == 'raw_string':
    return text[1:-1]
  if node.type == 'string':
    return text[1:-1] if not any(char in text for char in '\\$`') else None
  if node.type in ('command_name', 'concatenation'):
    parts: list[str] = []
    for child in node.children:
      part = _literal_word(child, source)
      if part is None:
        return None
      parts.append(part)
    return ''.join(parts)
  return None
