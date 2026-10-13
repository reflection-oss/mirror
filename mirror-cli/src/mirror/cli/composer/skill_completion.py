"""Typeahead completion UI for `$skill-name` mentions."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Collection, Sequence
from typing import override

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import processors
from rich import console as rich_console
from rich import segment, style, text

from mirror import skills
from mirror.cli import selection


@dataclasses.dataclass(frozen=True)
class TextEdit:
  """Replacement text and cursor position after a completion action."""

  text: str
  cursor: int


@dataclasses.dataclass(frozen=True)
class Query:
  """The active `$skill` token around the input cursor."""

  start: int
  end: int
  value: str


def query(text: str, cursor: int) -> Query | None:
  """Return the active `$skill` token when it is eligible for typeahead."""
  before = text[:cursor]
  start = max(before.rfind(' '), before.rfind('\n'), before.rfind('\t')) + 1
  token = before[start:]
  if not token.startswith('$'):
    return None
  value = token[1:]
  end = cursor
  while end < len(text) and not text[end].isspace():
    end += 1
  return Query(start=start, end=end, value=value)


def _mention_pattern(names: Collection[str]) -> str:
  alternatives = '|'.join(
    re.escape(name) for name in sorted(names, key=len, reverse=True)
  )
  return rf'(?<!\S)\$(?:{alternatives})(?!\S)'


def highlight_mentions(
  rendered: text.Text, names: Collection[str], mention_style: str = 'ui.info'
) -> None:
  """Tint exact whitespace-delimited skill mentions in a Rich text value."""
  if names:
    rendered.highlight_regex(_mention_pattern(names), style=mention_style)


@dataclasses.dataclass(frozen=True)
class MentionHighlighted:
  """A Rich renderable with exact skill mentions tinted after rendering."""

  renderable: rich_console.RenderableType
  names: frozenset[str]

  def __rich_console__(
    self,
    console: rich_console.Console,
    options: rich_console.ConsoleOptions,
  ) -> rich_console.RenderResult:
    tint = console.get_style('ui.info')
    pattern = re.compile(_mention_pattern(self.names))
    for rendered_segment in console.render(self.renderable, options):
      if rendered_segment.control:
        yield rendered_segment
        continue
      offset = 0
      for match in pattern.finditer(rendered_segment.text):
        if match.start() > offset:
          yield segment.Segment(
            rendered_segment.text[offset : match.start()],
            rendered_segment.style,
          )
        combined = style.Style.combine((
          rendered_segment.style or style.Style(),
          tint,
        ))
        yield segment.Segment(match.group(), combined)
        offset = match.end()
      if offset < len(rendered_segment.text):
        yield segment.Segment(
          rendered_segment.text[offset:], rendered_segment.style
        )


def highlight_renderable(
  renderable: rich_console.RenderableType, names: Collection[str]
) -> rich_console.RenderableType:
  """Apply mention tinting to any Rich renderable."""
  return (
    MentionHighlighted(renderable, frozenset(names)) if names else renderable
  )


class SkillMentionProcessor(processors.Processor):
  """Tint exact `$name` tokens that resolve to available skills."""

  def __init__(self, available: Sequence[skills.Skill]) -> None:
    self.update(available)

  def update(self, available: Sequence[skills.Skill]) -> None:
    """Refresh mention highlighting after the skill catalog changes."""
    self._names = frozenset(skill.name for skill in available)

  @override
  def apply_transformation(
    self, transformation_input: processors.TransformationInput
  ) -> processors.Transformation:
    line = transformation_input.document.lines[transformation_input.lineno]
    spans = tuple(
      match.span()
      for match in re.finditer(r'\S+', line)
      if match.group().startswith('$') and match.group()[1:] in self._names
    )
    if not spans:
      return processors.Transformation(transformation_input.fragments)

    fragments: StyleAndTextTuples = []
    offset = 0
    for fragment in transformation_input.fragments:
      original_style, text = fragment[:2]
      end = offset + len(text)
      boundaries = sorted({
        boundary
        for span in spans
        for boundary in span
        if offset < boundary < end
      })
      starts = [offset, *boundaries]
      stops = [*boundaries, end]
      for start, stop in zip(starts, stops, strict=True):
        style = (
          'class:skill-mention-input'
          if any(
            span_start <= start < span_end for span_start, span_end in spans
          )
          else original_style
        )
        transformed = text[start - offset : stop - offset]
        if len(fragment) == 2:
          fragments.append((style, transformed))
        else:
          fragments.append((style, transformed, fragment[2]))
      offset = end
    return processors.Transformation(fragments)


@dataclasses.dataclass(frozen=True, init=False)
class SkillCompletion:
  """Pure matching and text edits for repository skill mentions."""

  available: tuple[skills.Skill, ...]

  def __init__(self, available: Sequence[skills.Skill]) -> None:
    object.__setattr__(self, 'available', tuple(available))

  def matches(self, text: str, cursor: int) -> tuple[skills.Skill, ...]:
    """Skills matching the mention under `cursor`."""
    active_query = query(text, cursor)
    if active_query is None:
      return ()
    return selection.fuzzy_matches(
      self.available, active_query.value, lambda skill: (skill.name,)
    )

  def visible(self, text: str, cursor: int) -> bool:
    return bool(self.matches(text, cursor))

  def complete(
    self, text: str, cursor: int, skill: skills.Skill
  ) -> TextEdit | None:
    """Replace the active mention with `skill`."""
    active_query = query(text, cursor)
    if active_query is None:
      return None
    suffix = text[active_query.end :]
    separator = '' if suffix[:1] in {' ', '\t'} else ' '
    prefix = text[: active_query.start]
    completed = f'{prefix}${skill.name}{separator}{suffix}'
    completed_cursor = len(prefix) + len(skill.name) + 2
    return TextEdit(completed, completed_cursor)

  def cancel(self, text: str, cursor: int) -> TextEdit | None:
    """Remove the active incomplete mention."""
    active_query = query(text, cursor)
    if active_query is None:
      return None
    return TextEdit(
      text[: active_query.start] + text[active_query.end :],
      active_query.start,
    )
