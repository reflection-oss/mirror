"""Reusable text component with a sweeping highlight."""

from __future__ import annotations

import time

from prompt_toolkit import formatted_text


def render(text: str, *, style: str = '') -> formatted_text.StyleAndTextTuples:
  """Render a right-to-left highlight followed by a one-second rest.

  Args:
    text: Text to highlight.
    style: Base prompt_toolkit style applied to the text.
  """
  position = (1 - (time.monotonic() % 3.4) / 2.4) * (len(text) + 4) - 2
  fragments: formatted_text.StyleAndTextTuples = []
  for index, character in enumerate(text):
    highlight = round(127 * max(0, 1 - abs(index - position) / 2))
    fragments.append((f'class:ui.highlight-{highlight}', character))
  return formatted_text.to_formatted_text(fragments, style=style)
