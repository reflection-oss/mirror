"""Buffer Space taps until another key or terminal repeats resolve the gesture."""

from __future__ import annotations

import dataclasses

from mirror.cli import actions

RELEASE_SECONDS = 0.25
INITIAL_REPEAT_SECONDS = 1.5


@dataclasses.dataclass(frozen=True)
class SpaceHold:
  """Pending spaces that have not changed the composer."""

  original: actions.ComposerState
  started: float
  last: float
  count: int = 1
  repeats: int = 0

  def press(self, now: float) -> SpaceHold:
    """Accumulate a tap or repeat without inserting text."""
    return dataclasses.replace(
      self,
      last=now,
      count=self.count + 1,
      repeats=self.repeats + 1 if now - self.last <= 0.15 else 0,
    )

  @property
  def confirmed(self) -> bool:
    # Rapid pasted bursts and ordinary double spaces should remain text.
    return self.repeats >= 3 and self.last - self.started >= 0.25

  @property
  def deadline(self) -> float:
    return self.last + (
      INITIAL_REPEAT_SECONDS if self.count == 1 else RELEASE_SECONDS
    )
