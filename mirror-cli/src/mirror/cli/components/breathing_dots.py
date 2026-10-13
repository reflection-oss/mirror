"""Reusable two-dot loading indicator rendered as inline text fragments."""

from __future__ import annotations

from prompt_toolkit import formatted_text

from mirror.cli.components import animation


def render(*, style: str = '') -> formatted_text.StyleAndTextTuples:
  """Render two dots that trade size over a shared 1.2-second cycle.

  Args:
    style: Base prompt_toolkit style applied to both dots.
  """
  pairs = (
    '·●',
    '·●',
    '·●',
    '•●',
    '••',
    '●•',
    '●·',
    '●·',
    '●·',
    '●•',
    '••',
    '•●',
  )
  sizes = {'·': 0, '•': 1, '●': 2}
  frames: list[formatted_text.AnyFormattedText] = []
  for left, right in pairs:
    left_is_smaller = sizes[left] <= sizes[right]
    frame: formatted_text.StyleAndTextTuples = [
      ('class:activity-spinner-highlight' if left_is_smaller else '', left),
      ('' if left_is_smaller else 'class:activity-spinner-highlight', right),
    ]
    frames.append(frame)
  return formatted_text.to_formatted_text(
    animation.Animation(frames, interval=0.1),
    style=style,
  )
