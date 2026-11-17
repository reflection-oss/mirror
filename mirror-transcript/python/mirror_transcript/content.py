"""Display-copy limits shared by every logical block-tree entry point."""

import dataclasses
import re
from collections.abc import Iterable, Sequence

from mirror_transcript import model


_ESCAPES = re.compile(
  r'\x1b(?:\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]'
  r'|\][\s\S]*?(?:\x07|\x1b\\)|[PX^_][\s\S]*?(?:\x1b\\)'
  r'|[\x20-\x2f]+[\x30-\x7e]|[\x30-\x7e])'
  r'|\x9b[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]'
  r'|\x9d[\s\S]*?(?:\x07|\x9c)|[\x80-\x9f]',
)
_CONTROLS = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')


def source(value: str) -> str:
  """Normalize a display copy without executing embedded terminal controls."""
  return _CONTROLS.sub('', _ESCAPES.sub('', value)).replace('\t', '    ')


def lines(spans: Iterable[model.Span]) -> list[model.Line]:
  """Split styled source spans on explicit newlines, without wrapping text."""
  result: list[list[model.Span]] = [[]]
  for span in spans:
    for index, part in enumerate(span.text.split('\n')):
      if index:
        result.append([])
      result[-1].append(model.Span(part, span.sgr))
  return list(result)


def bounded(blocks: Sequence[model.Block]) -> list[model.Block]:
  """Clip display copies to wire budgets without changing application sources."""
  remaining_lines = 8191  # Reserve one block for the truncation notice.
  remaining_bytes = 1024 * 1024
  truncated = False

  def text_prefix(value: str, limit: int) -> str:
    nonlocal remaining_bytes, truncated
    prefix = value.encode()[: min(limit, remaining_bytes)].decode(
      'utf-8', 'ignore'
    )
    remaining_bytes -= len(prefix.encode())
    truncated |= prefix != value
    return prefix

  def visit(values: Sequence[model.Block]) -> list[model.Block]:
    nonlocal remaining_lines, truncated
    result: list[model.Block] = []
    for value in values:
      if remaining_lines == 0 or remaining_bytes == 0:
        truncated = True
        break
      remaining_lines -= 1
      prefix = text_prefix(value.text, 1024 * 1024)
      parts = prefix.split('\n', remaining_lines + 1)
      text = '\n'.join(
        part.encode()[:65536].decode('utf-8', 'ignore')
        for part in parts[: remaining_lines + 1]
      )
      truncated |= text != prefix
      remaining_lines -= text.count('\n')
      lines: list[model.Line] = []
      for line in value.lines:
        if remaining_lines == 0:
          truncated = True
          break
        remaining_lines -= 1
        spans: list[model.Span] = []
        available = 65536
        for span in line:
          overhead = len(span.sgr.encode()) + 48
          if available < overhead:
            truncated = True
            break
          text_value = text_prefix(span.text, available - overhead)
          available -= len(text_value.encode()) + overhead
          spans.append(dataclasses.replace(span, text=text_value))
        lines.append(spans)
      result.append(
        dataclasses.replace(
          value,
          text=text,
          lines=lines,
          collapsed=visit(value.collapsed),
          children=visit(value.children),
        )
      )
    return result

  result = visit(blocks)
  if truncated:
    result.append(model.Block('… content truncated for display', sgr='2'))
  return result
