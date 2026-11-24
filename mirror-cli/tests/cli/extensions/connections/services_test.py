from __future__ import annotations

from mirror.cli.extensions.connections import configuration, services


def test_offers_local_server_that_failed_to_start_as_unavailable() -> None:
  local = services.Local((), {'github': {'command': 'github-mcp'}}, ())
  [offer] = services.offers(local)
  assert offer.availability is services.Availability.LOCAL_UNAVAILABLE


def test_offers_unconfigured_catalog_service_for_local_sign_in() -> None:
  local = services.Local(
    (
      configuration.CatalogServer(
        'linear', 'Linear', 'https://local.test/mcp', None
      ),
    ),
    {},
    (),
  )
  [offer] = services.offers(local)
  assert offer.availability is services.Availability.LOCAL_AVAILABLE


def test_rows_report_configured_local_connections() -> None:
  local = services.Local((), {'github': {'command': 'github-mcp'}}, ('github',))
  assert services.rows(local) == (
    services.Row('github', 'this machine', '—', 'connected'),
  )
