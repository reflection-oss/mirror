from __future__ import annotations

import dataclasses
import pathlib

from mirror.cli.statsig import constants


@dataclasses.dataclass(frozen=True)
class StatsigConfig:
  # The user ID.
  user_id: str
  # The CLI launch's session ID, as shown by `/status`.
  session_id: str
  # Where undelivered events are saved when the client is turned off and
  # resent from when it is turned on. None disables saving.
  unsent_events_directory: pathlib.Path | None = None

  def __post_init__(self) -> None:
    if not 1 <= len(self.user_id) <= constants.MAX_USER_ID_LENGTH:
      raise ValueError(
        'Statsig requires an opaque user ID of '
        f'1–{constants.MAX_USER_ID_LENGTH} characters'
      )
