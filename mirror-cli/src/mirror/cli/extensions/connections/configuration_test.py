"""Tests for MCP server configuration and credentials."""

from __future__ import annotations

import asyncio
import json
import pathlib
import stat
from collections.abc import Awaitable, Callable

import pytest
from mcp.client import auth as mcp_auth
from mcp.shared import auth as mcp_shared_auth

from mirror.cli.extensions.connections import configuration as mcp


@pytest.mark.asyncio
@pytest.mark.parametrize('fail', [False, True])
async def test_concurrent_authentication_shares_success_and_failure(
  monkeypatch: pytest.MonkeyPatch,
  fail: bool,
) -> None:
  server = mcp.load_catalog()[0]
  started = asyncio.Event()
  release = asyncio.Event()
  calls = 0

  async def authenticate(
    server: mcp.CatalogServer, *, show_url: Callable[[str], Awaitable[None]]
  ) -> None:
    del server, show_url
    nonlocal calls
    calls += 1
    started.set()
    await release.wait()
    if fail:
      raise ValueError('login failed')

  async def show_url(_: str) -> None:
    pass

  monkeypatch.setattr(mcp, '_authenticate', authenticate)
  first = asyncio.create_task(mcp.authenticate(server, show_url=show_url))
  await started.wait()
  second = asyncio.create_task(mcp.authenticate(server, show_url=show_url))
  await asyncio.sleep(0)
  release.set()
  results = await asyncio.gather(first, second, return_exceptions=True)
  assert calls == 1
  assert (
    all(isinstance(result, ValueError) for result in results)
    if fail
    else results == [None, None]
  )


@pytest.mark.asyncio
async def test_authentication_survives_one_cancelled_waiter_and_stops_after_the_last(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  server = mcp.load_catalog()[0]
  started = asyncio.Event()
  stopped = asyncio.Event()

  async def authenticate(
    server: mcp.CatalogServer, *, show_url: Callable[[str], Awaitable[None]]
  ) -> None:
    del server, show_url
    started.set()
    try:
      await asyncio.Event().wait()
    finally:
      stopped.set()

  async def show_url(_: str) -> None:
    pass

  monkeypatch.setattr(mcp, '_authenticate', authenticate)
  first = asyncio.create_task(mcp.authenticate(server, show_url=show_url))
  await started.wait()
  second = asyncio.create_task(mcp.authenticate(server, show_url=show_url))
  await asyncio.sleep(0)
  first.cancel()
  await asyncio.gather(first, return_exceptions=True)
  assert not stopped.is_set()
  second.cancel()
  await asyncio.gather(second, return_exceptions=True)
  assert stopped.is_set()


def test_add_list_and_remove_server(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))

  path = mcp.add_server(mcp.load_catalog()[0])

  assert json.loads(path.read_text()) == {
    'auth': {'credential': 'notion', 'type': 'oauth'},
    'url': 'https://mcp.notion.com/mcp',
  }
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  assert mcp.requires_login('notion', mcp.load_servers()['notion'])

  mcp.remove_server('notion')

  assert not path.exists()
  assert mcp.load_servers() == {}


def test_catalog_contains_supported_servers() -> None:
  catalog = mcp.load_catalog()
  assert [server.name for server in catalog] == [
    'notion',
    'datadog',
    'linear',
    'slack',
  ]
  linear = next(server for server in catalog if server.name == 'linear')
  assert linear.oauth == mcp.CatalogOAuth()
  slack = next(server for server in catalog if server.name == 'slack')
  assert slack.oauth == mcp.CatalogOAuth(
    client_id='6696120515462.11886240096129',
    redirect_host='localhost',
    redirect_port=8765,
  )


def test_rejects_unsafe_name(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))

  server = mcp.CatalogServer(
    '../outside', 'Outside', 'https://example.com/mcp', mcp.CatalogOAuth()
  )
  with pytest.raises(ValueError, match='server name'):
    mcp.add_server(server)


@pytest.mark.asyncio
async def test_oauth_storage_round_trips_sdk_records(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  storage = mcp.OAuthStorage('notion')
  tokens = mcp_shared_auth.OAuthToken(
    access_token='access',
    token_type='Bearer',
    refresh_token='refresh',
    expires_in=3600,
  )
  client_info = mcp_shared_auth.OAuthClientInformationFull.model_validate({
    'client_id': 'registered-client',
    'redirect_uris': ['http://127.0.0.1:1234/oauth/callback'],
    'token_endpoint_auth_method': 'none',
  })

  await storage.set_tokens(tokens)
  await storage.set_client_info(client_info)

  assert await mcp.OAuthStorage('notion').get_tokens() == tokens
  assert await mcp.OAuthStorage('notion').get_client_info() == client_info
  provider = mcp.oauth_provider('https://example.com/mcp', 'notion')
  assert provider.context.token_expiry_time == storage.token_expiry_time()
  assert (
    stat.S_IMODE((tmp_path / 'credentials' / 'notion.json').stat().st_mode)
    == 0o600
  )


@pytest.mark.asyncio
async def test_slack_provider_uses_fixed_client(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  slack = next(
    server for server in mcp.load_catalog() if server.name == 'slack'
  )
  assert slack.oauth is not None
  provider = mcp.oauth_provider(
    slack.url,
    slack.name,
    redirect_uri='http://localhost:8765/oauth/callback',
    client_id=slack.oauth.client_id,
  )

  assert isinstance(provider, mcp_auth.OAuthClientProvider)
  client_info = await provider.context.storage.get_client_info()
  assert client_info is not None
  assert client_info.redirect_uris is not None
  assert client_info.client_id == '6696120515462.11886240096129'
  assert str(client_info.redirect_uris[0]) == (
    'http://localhost:8765/oauth/callback'
  )


@pytest.mark.asyncio
async def test_failed_authentication_leaves_no_configuration(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  server = mcp.load_catalog()[0]

  async def fail_authentication(
    _: mcp.CatalogServer,
    *,
    show_url: Callable[[str], Awaitable[None]],
  ) -> None:
    del show_url
    await mcp.OAuthStorage(server.name).set_tokens(
      mcp_shared_auth.OAuthToken(access_token='access', token_type='Bearer')
    )
    raise ValueError('cancelled')

  monkeypatch.setattr(mcp, 'authenticate', fail_authentication)

  async def show_url(_: str) -> None:
    pass

  with pytest.raises(ValueError, match='cancelled'):
    await mcp.install_server(server, show_url=show_url)

  assert mcp.load_servers() == {}
  assert not (tmp_path / 'credentials' / 'notion.json').exists()


@pytest.mark.asyncio
async def test_requires_login_distinguishes_expired_and_live_credentials(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  config = {'url': 'https://mcp.notion.com/mcp', 'auth': {'type': 'oauth'}}
  storage = mcp.OAuthStorage('notion')
  await storage.set_tokens(
    mcp_shared_auth.OAuthToken(
      access_token='expired', token_type='Bearer', expires_in=0
    )
  )
  assert mcp.requires_login('notion', config)
  await storage.set_tokens(
    mcp_shared_auth.OAuthToken(
      access_token='live', token_type='Bearer', expires_in=3600
    )
  )
  assert not mcp.requires_login('notion', config)
