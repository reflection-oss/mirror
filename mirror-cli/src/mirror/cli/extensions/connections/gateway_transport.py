"""Mirror endpoint for explicit remote-session connection setup."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Gateway:
  """A Mirror server used to prepare remote sessions, not local MCPs."""

  endpoint: str
  secure: bool = True
