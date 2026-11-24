"""Quoted selections: composer chips for text the user quotes.

The composer holds each quote as a chip such as `▌first line… +4 lines`. The
chip's text is also its key into the quote table, so copies, edits, undo and
recalled prompts keep resolving to the same quote. Zero-width word joiners end
every chip, so neither the text typed after a chip nor text that merely looks
like one reads as a chip; quotes whose chips would read alike get one more. A sent
message carries its quotes as `UserMessage.parts`, and the agent decides how
the model sees them.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Iterator, Sequence

from mirror.protocol import agent_pb2

_RAIL = '▌'
# Zero-width and never typed, so it marks where a chip ends without showing.
_END = '\u2060'
_DETAIL = re.compile(r'(?: \+\d+ lines?)?$')
_PREVIEW_WIDTH = 40


@dataclasses.dataclass(frozen=True)
class Quote:
  """One quoted selection and the chip label that stands for it."""

  label: str
  text: str


def label(text: str) -> str:
  """A chip for text: a rail, its first line, ellipsized, and a line count."""
  lines = text.splitlines() or ['']
  first = lines[0].replace(_END, '').strip()
  if len(first) > _PREVIEW_WIDTH:
    first = first[: _PREVIEW_WIDTH - 1].rstrip() + '…'
  elif len(lines) > 1:
    first += '…'
  extra = len(lines) - 1
  detail = f' +{extra} line{"s" if extra != 1 else ""}' if extra else ''
  return f'{_RAIL}{first}{detail}{_END}'


def split_chip(chip: str) -> tuple[str, str, str]:
  """Split a chip into its rail, preview and line-count detail, without its end."""
  chip = chip.rstrip(_END)
  match = _DETAIL.search(chip, len(_RAIL))
  start = match.start() if match else len(chip)
  return chip[: len(_RAIL)], chip[len(_RAIL) : start], chip[start:]


def normalize(text: str) -> str:
  """Trim trailing spaces and surrounding blank lines from selected text."""
  return '\n'.join(line.rstrip() for line in text.splitlines()).strip('\n')


def add(text: str, quotes: Sequence[Quote]) -> tuple[Quote, tuple[Quote, ...]]:
  """The quote for text, added under a label no other quote in `quotes` uses."""
  taken = {quote.label: quote for quote in quotes}
  base = label(text).removesuffix(_END)
  chip = base + _END
  while (existing := taken.get(chip)) is not None:
    if existing.text == text:
      return existing, tuple(quotes)
    chip += _END
  quote = Quote(chip, text)
  return quote, (*quotes, quote)


def _labels(quotes: Iterable[Quote]) -> str:
  # Longest first, so a chip quoting another chip is not read as the inner one.
  ordered = sorted({quote.label for quote in quotes}, key=len, reverse=True)
  return '|'.join(map(re.escape, ordered))


def chips(text: str, quotes: Sequence[Quote]) -> Iterator[tuple[int, Quote]]:
  """Yield the start offset and quote of each chip in text order."""
  known = {quote.label: quote for quote in quotes}
  if not known:
    return
  for match in re.finditer(_labels(quotes), text):
    yield match.start(), known[match[0]]


def chip_before(text: str, cursor: int, quotes: Sequence[Quote]) -> str | None:
  """The longest chip ending at the cursor, with one following space."""
  before = text[:cursor]
  # A quote of a chip makes a label that ends with the quoted chip's label.
  for quote in sorted(quotes, key=lambda quote: len(quote.label), reverse=True):
    for chip in (f'{quote.label} ', quote.label):
      if before.endswith(chip):
        return chip
  return None


def step_over_end(text: str, cursor: int, previous: int) -> int:
  """Move a cursor off a chip's end, the way it was going; it would rest unseen."""
  if text[cursor : cursor + 1] != _END:
    return cursor
  if cursor >= previous:
    return len(text) - len(text[cursor:].lstrip(_END))
  return max(len(text[:cursor].rstrip(_END)) - 1, 0)


def message(text: str, quotes: Sequence[Quote]) -> agent_pb2.UserMessage:
  """The user message for composer text, with each chip as a quoted part.

  An end marker left by a chip that no longer resolves is dropped.
  """
  parts: list[agent_pb2.UserMessagePart] = []
  offset = 0
  for start, quote in chips(text, quotes):
    if plain := text[offset:start].replace(_END, ''):
      parts.append(agent_pb2.UserMessagePart(text=plain))
    parts.append(agent_pb2.UserMessagePart(quoted_selection=quote.text))
    offset = start + len(quote.label)
  if offset == 0:
    return agent_pb2.UserMessage(content=text.replace(_END, ''))
  if plain := text[offset:].replace(_END, ''):
    parts.append(agent_pb2.UserMessagePart(text=plain))
  return agent_pb2.UserMessage(parts=parts)


def display(
  user: agent_pb2.UserMessage, quotes: Sequence[Quote] = ()
) -> tuple[str, tuple[Quote, ...]]:
  """Composer text for a user message, adding its quotes to `quotes`."""
  if not user.parts:
    return user.content, tuple(quotes)
  pieces: list[str] = []
  found = tuple(quotes)
  for part in user.parts:
    if part.WhichOneof('part') == 'quoted_selection':
      quote, found = add(part.quoted_selection, found)
      pieces.append(quote.label)
    else:
      pieces.append(part.text)
  return ''.join(pieces), found
