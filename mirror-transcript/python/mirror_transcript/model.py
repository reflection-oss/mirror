"""Width-independent transcript content; Python remains its ordering authority."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class Span:
  text: str
  sgr: str = ''


type Line = Sequence[Span]


class Kind(enum.IntEnum):
  TEXT = 0
  MARKDOWN = 1
  CODE = 2
  DIFF = 3
  COLUMN = 4
  ROW = 5
  TABLE = 6
  FRAME = 7
  PANEL = 8
  GRID = 9
  INLINE = 10
  RULE = 11


class Alignment(enum.IntEnum):
  CENTER = 0
  LEFT = 1
  RIGHT = 2


class Overflow(enum.IntEnum):
  FOLD = 0
  CROP = 1
  ELLIPSIS = 2


@dataclasses.dataclass(frozen=True)
class Block:
  text: str = ''
  kind: Kind = Kind.TEXT
  lines: Sequence[Line] = ()
  children: Sequence[Block] = ()
  sgr: str = ''
  title: str = ''
  padding: int = 0
  collapse: int = 0
  language: str = ''
  title_sgr: str = ''
  spread: bool = False
  insets: Sequence[int] = ()
  marker: Span | None = None
  clip: bool = False
  trim_end: bool = False
  ratio: int = 0
  border_sgr: str = ''
  overflow: Overflow = Overflow.FOLD
  show_file_headers: bool = True
  collapsed: Sequence[Block] = ()
  no_wrap: bool = False
  gap: int = 0
  height: int | None = None
  syntax: str = ''
  alignment: Alignment = Alignment.CENTER
  word_wrap: bool = True


@dataclasses.dataclass(frozen=True)
class Element:
  id: int
  lines: Sequence[Line] = ()
  blocks: Sequence[Block] = ()
  sticky: Span | None = None
