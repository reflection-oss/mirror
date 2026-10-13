"""Prompt-toolkit processors that restyle the input line without changing it."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from prompt_toolkit.formatted_text import (
  StyleAndTextTuples as _StyleAndTextTuples,
)
from prompt_toolkit.layout import processors as _processors

from mirror.cli import commands as _commands


class SlashInputProcessor(_processors.Processor):
  """Accent only `/` strongly; keep command text subtle and args plain."""

  @override
  def apply_transformation(
    self, transformation_input: _processors.TransformationInput
  ) -> _processors.Transformation:
    if (
      transformation_input.lineno != 0
      or not _commands.looks_like_slash_command(
        transformation_input.document.text
      )
    ):
      return _processors.Transformation(transformation_input.fragments)
    line = transformation_input.document.lines[0]
    command_end = next(
      (index for index, character in enumerate(line) if character.isspace()),
      len(line),
    )
    fragments: _StyleAndTextTuples = []
    offset = 0
    for fragment in transformation_input.fragments:
      original_style, text = fragment[:2]
      end = offset + len(text)
      boundaries = [
        boundary for boundary in (1, command_end) if offset < boundary < end
      ]
      starts = [offset, *boundaries]
      stops = [*boundaries, end]
      for start, stop in zip(starts, stops, strict=True):
        style = (
          'class:slash-input-prefix'
          if start == 0
          else 'class:slash-input-command'
          if start < command_end
          else original_style
        )
        transformed_text = text[start - offset : stop - offset]
        if len(fragment) == 2:
          fragments.append((style, transformed_text))
        else:
          fragments.append((style, transformed_text, fragment[2]))
      offset = end
    return _processors.Transformation(fragments)


class CompletionPreviewProcessor(_processors.Processor):
  """Draw the active completion preview at the input cursor."""

  def __init__(self, preview: Callable[[], _StyleAndTextTuples | None]) -> None:
    self._preview = preview

  @override
  def apply_transformation(
    self, transformation_input: _processors.TransformationInput
  ) -> _processors.Transformation:
    preview = self._preview()
    document = transformation_input.document
    if (
      transformation_input.lineno != document.cursor_position_row
      or preview is None
    ):
      return _processors.Transformation(transformation_input.fragments)

    column = document.cursor_position_col
    fragments = self._insert_at_cursor(
      transformation_input.fragments, preview, column
    )
    preview_length = sum(len(text) for _, text, *_ in preview)
    return _processors.Transformation(
      fragments,
      source_to_display=lambda index: (
        index if index <= column else index + preview_length
      ),
      display_to_source=lambda index: (
        index
        if index <= column
        else column
        if index <= column + preview_length
        else index - preview_length
      ),
    )

  @staticmethod
  def _insert_at_cursor(
    fragments: _StyleAndTextTuples,
    preview: _StyleAndTextTuples,
    column: int,
  ) -> _StyleAndTextTuples:
    transformed: _StyleAndTextTuples = []
    offset = 0
    for index, fragment in enumerate(fragments):
      text = fragment[1]
      end = offset + len(text)
      if offset <= column <= end:
        before = text[: column - offset]
        after = text[column - offset :]
        if len(fragment) == 2:
          before_fragment = (fragment[0], before)
          after_fragment = (fragment[0], after)
        else:
          before_fragment = (fragment[0], before, fragment[2])
          after_fragment = (fragment[0], after, fragment[2])
        if before:
          transformed.append(before_fragment)
        transformed.extend(preview)
        if after:
          transformed.append(after_fragment)
        transformed.extend(fragments[index + 1 :])
        return transformed
      transformed.append(fragment)
      offset = end
    transformed.extend(preview)
    return transformed
