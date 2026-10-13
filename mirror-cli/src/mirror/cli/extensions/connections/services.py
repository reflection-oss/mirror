"""One entry per service across the Mirror server and this machine."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from mirror.cli.extensions.connections import configuration as mcp
from mirror.cli.extensions.connections import server_connections

_SERVER = 'Mirror server'
_LOCAL = 'this machine'


class Availability(enum.Enum):
  """Where a service can be used, in picker order."""

  SERVER = 'connected on Mirror server'
  LOCAL = 'on this machine only'
  LOCAL_SIGN_IN = 'needs sign-in on this machine'
  LOCAL_UNAVAILABLE = 'unavailable on this machine — retry'
  SERVER_EXPIRED = 'expired on Mirror server — sign in'
  SERVER_AVAILABLE = 'available — sign in'
  LOCAL_AVAILABLE = 'available on this machine — sign in'

  @property
  def symbol(self) -> str:
    """The picker marker for this availability."""
    if self is Availability.SERVER:
      return '●'
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
  accounts: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Row:
  """One connection shown by `/connections`."""

  service: str
  where: str
  account: str
  status: str
  default: bool = False


def display_names(
  server: server_connections.ServerConnections | None, local: Local
) -> dict[str, str]:
  """Return every known service's display name, keyed by service name."""
  names = {entry.name: entry.display_name for entry in local.catalog}
  for name in local.configured:
    names.setdefault(name, name)
  for provider in server.providers if server is not None else ():
    names[provider.name] = provider.display_name or names.get(
      provider.name, provider.name
    )
  return names


def offers(
  server: server_connections.ServerConnections | None,
  gateway_services: Collection[str],
  local: Local,
) -> tuple[Offer, ...]:
  """Return one offer per service, preferring the Mirror server."""
  order = list(Availability)
  found = [
    _offer(name, display_name, server, gateway_services, local)
    for name, display_name in display_names(server, local).items()
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


def _offer(
  name: str,
  display_name: str,
  server: server_connections.ServerConnections | None,
  gateway_services: Collection[str],
  local: Local,
) -> Offer:
  provider = server.provider(name) if server is not None else None
  connections = server.for_provider(name) if server is not None else ()
  accounts = tuple(
    connection.account_identifier
    for connection in connections
    if server_connections.usable(connection)
  )
  if (provider is not None and accounts) or name in gateway_services:
    return Offer(name, display_name, Availability.SERVER, accounts)
  if provider is not None:
    availability = (
      Availability.SERVER_EXPIRED
      if connections
      else Availability.SERVER_AVAILABLE
    )
  elif name in local.configured:
    availability = _local_availability(name, local)
  else:
    availability = Availability.LOCAL_AVAILABLE
  return Offer(name, display_name, availability)


def _local_availability(name: str, local: Local) -> Availability:
  if local.needs_sign_in(name):
    return Availability.LOCAL_SIGN_IN
  if name in local.connected:
    return Availability.LOCAL
  return Availability.LOCAL_UNAVAILABLE


def rows(
  server: server_connections.ServerConnections | None,
  gateway_services: Collection[str],
  local: Local,
) -> tuple[Row, ...]:
  """Return every connection on the Mirror server and this machine."""
  names = display_names(server, local)
  found = [
    Row(
      names.get(connection.provider, connection.provider),
      _SERVER,
      connection.account_identifier,
      'connected' if server_connections.usable(connection) else 'expired',
      connection.is_default,
    )
    for connection in (server.connections if server is not None else ())
  ]
  found.extend(
    Row(names.get(service, service), _SERVER, '—', 'connected')
    for service in gateway_services
    if server is None
    or not any(
      server_connections.usable(connection)
      for connection in server.for_provider(service)
    )
  )
  found.extend(
    Row(
      names.get(name, name),
      _LOCAL,
      '—',
      _local_status(name, gateway_services, local),
    )
    for name in local.configured
  )
  return tuple(
    sorted(found, key=lambda row: (row.service.casefold(), row.where))
  )


def _local_status(
  name: str, gateway_services: Collection[str], local: Local
) -> str:
  if local.needs_sign_in(name):
    return 'needs sign-in'
  if name in gateway_services:
    return 'not used — Mirror server connected'
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
