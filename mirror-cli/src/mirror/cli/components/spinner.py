"""A two-cell Crossfire spinner with a per-turn star accent."""

from __future__ import annotations

import enum

from prompt_toolkit import formatted_text
from prompt_toolkit.formatted_text import utils

from mirror.cli.components import animation


class Color(enum.Enum):
  """Per-turn accent choices; shared theme roles determine their colors."""

  PALE_PINK = 'ui.accent'
  NEON_YELLOW = 'ui.warning'
  LIGHT_GREEN = 'ui.success'
  BLUE = 'ui.info'
  PURPLE = 'ui.heading'


_ANIMATION = animation.Animation(
  ('*+', '+*', ':*', '.+', '+.', '*:'), interval=1.4 / 2.4 / 6
)


def render(
  color: Color, *, style: str = ''
) -> formatted_text.StyleAndTextTuples:
  """Render two Crossfire cells at 2.4x, accenting only stars with color."""
  pose = utils.fragment_list_to_text(_ANIMATION.render())
  accent = f'class:{color.value}'
  fragments: formatted_text.StyleAndTextTuples = [
    (accent if character == '*' else '', character) for character in pose
  ]
  return formatted_text.to_formatted_text(fragments, style=style)
