"""Local MCP availability for the CLI connection picker."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from mirror.cli.extensions.connections import configuration as mcp

_LOCAL = 'this machine'


class Availability(enum.Enum):
  """Where a service can be used, in picker order."""

  LOCAL = 'on this machine only'
  LOCAL_SIGN_IN = 'needs sign-in on this machine'
  LOCAL_UNAVAILABLE = 'unavailable on this machine — retry'
  LOCAL_AVAILABLE = 'available on this machine — sign in'

  @property
  def symbol(self) -> str:
    """The picker marker for this availability."""
    return '◐' if self is Availability.LOCAL else '○'


@dataclasses.dataclass(frozen=True)
class Local:
  """This machine's MCP servers."""

  catalog: Sequence[mcp.CatalogServer]
  configured: Mapping[str, Mapping[str, Any]]
  connected: Collection[str]

  def needs_sign_in(self, name: str) -> bool:
    """Return whether a configured server needs an interactive sign-in."""
    try:
      return mcp.requires_login(name, self.configured[name])
    except (OSError, ValueError):
      return True


@dataclasses.dataclass(frozen=True)
class Offer:
  """One service `/connect` can sign in to."""

  name: str
  display_name: str
  availability: Availability


@dataclasses.dataclass(frozen=True)
class Row:
  """One connection shown by `/connections`."""

  service: str
  where: str
  account: str
  status: str
  default: bool = False


def display_names(local: Local) -> dict[str, str]:
  """Return local service display names, keyed by service name."""
  names = {entry.name: entry.display_name for entry in local.catalog}
  for name in local.configured:
    names.setdefault(name, name)
  return names


def offers(local: Local) -> tuple[Offer, ...]:
  """Return one offer per local service."""
  order = list(Availability)
  found = [
    Offer(
      name,
      label,
      _local_availability(name, local)
      if name in local.configured
      else Availability.LOCAL_AVAILABLE,
    )
    for name, label in display_names(local).items()
  ]
  return tuple(
    sorted(
      found,
      key=lambda offer: (
        order.index(offer.availability),
        offer.display_name.casefold(),
      ),
    )
  )


def _local_availability(name: str, local: Local) -> Availability:
  if local.needs_sign_in(name):
    return Availability.LOCAL_SIGN_IN
  if name in local.connected:
    return Availability.LOCAL
  return Availability.LOCAL_UNAVAILABLE


def rows(local: Local) -> tuple[Row, ...]:
  """Return every configured local MCP."""
  names = display_names(local)
  found = [
    Row(names.get(name, name), _LOCAL, '—', _local_status(name, local))
    for name in local.configured
  ]
  return tuple(sorted(found, key=lambda row: row.service.casefold()))


def _local_status(name: str, local: Local) -> str:
  if local.needs_sign_in(name):
    return 'needs sign-in'
  return 'connected' if name in local.connected else 'unavailable'


def format_rows(found: Sequence[Row]) -> str:
  """Format connections as an aligned table."""
  if not found:
    return 'No connections. Run /connect to add one.'
  table = [('Service', 'Where', 'Account', 'Status')] + [
    (
      row.service,
      row.where,
      row.account,
      f'{row.status} (default)' if row.default else row.status,
    )
    for row in found
  ]
  widths = [max(len(line[column]) for line in table) for column in range(3)]
  return '\n'.join([
    'Connections:',
    *(
      '  '
      + '  '.join(
        cell.ljust(width) for cell, width in zip(line[:3], widths, strict=True)
      )
      + f'  {line[3]}'
      for line in table
    ),
  ])
