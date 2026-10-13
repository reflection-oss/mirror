from __future__ import annotations

from collections.abc import Collection

from mirror.protocol import mirror_pb2

from mirror.cli.extensions.connections import server_connections, services

_GITHUB = mirror_pb2.Provider(
  name='github',
  display_name='GitHub',
  sign_in=mirror_pb2.PROVIDER_SIGN_IN_TOKEN,
)


def _local_github(connected: Collection[str]) -> services.Local:
  return services.Local((), {'github': {'command': 'github-mcp'}}, connected)


def test_offers_local_server_that_failed_to_start_as_unavailable() -> None:
  [offer] = services.offers(None, (), _local_github(connected=()))
  assert offer.availability is services.Availability.LOCAL_UNAVAILABLE


def test_offers_mirror_server_sign_in_when_the_server_has_the_provider() -> (
  None
):
  server = server_connections.ServerConnections((_GITHUB,), ())
  [offer] = services.offers(server, (), _local_github(connected=('github',)))
  assert offer.availability is services.Availability.SERVER_AVAILABLE


def test_rows_keep_a_connected_gateway_beside_an_expired_account() -> None:
  expired = mirror_pb2.Connection(
    id='connection-1',
    provider='github',
    account_identifier='octocat',
    status='expired',
  )
  server = server_connections.ServerConnections((_GITHUB,), (expired,))
  assert services.rows(server, {'github'}, services.Local((), {}, ())) == (
    services.Row('GitHub', 'Mirror server', 'octocat', 'expired'),
    services.Row('GitHub', 'Mirror server', '—', 'connected'),
  )
