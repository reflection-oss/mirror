"""A looping sequence of inline text frames with a declarative cadence."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from prompt_toolkit import formatted_text


class Animation:
  """Formatted text frames selected from a shared monotonic timeline.

  Animation does not schedule redraws or start playback at construction time.
  The host controls refresh cadence, and all instances share an absolute phase.
  """

  def __init__(
    self, frames: Sequence[formatted_text.AnyFormattedText], *, interval: float
  ) -> None:
    if not frames:
      raise ValueError('An animation requires at least one frame')
    if not math.isfinite(interval) or interval <= 0:
      raise ValueError(
        f'Animation interval must be finite and positive: {interval!r}'
      )
    self._frames = tuple(
      tuple(formatted_text.to_formatted_text(frame)) for frame in frames
    )
    self._interval = interval

  def render(self) -> formatted_text.StyleAndTextTuples:
    """Return the frame selected by the current monotonic time."""
    index = int(time.monotonic() / self._interval) % len(self._frames)
    return list(self._frames[index])

  def __pt_formatted_text__(self) -> formatted_text.StyleAndTextTuples:
    return self.render()
