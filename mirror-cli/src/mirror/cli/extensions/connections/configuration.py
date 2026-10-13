"""MCP server configuration and OAuth credentials."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import re
import secrets
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

import httpx2
from mcp import client as mcp_client
from mcp.client import auth as mcp_auth
from mcp.client import streamable_http as mcp_http
from mcp.shared import auth as mcp_shared_auth

_LOGGER = logging.getLogger(__name__)
_NAME = re.compile(r'^[A-Za-z0-9_-]+$')


@dataclasses.dataclass(frozen=True)
class CatalogOAuth:
  """Curated OAuth details that cannot be discovered from the server."""

  client_id: str | None = None
  redirect_host: str = '127.0.0.1'
  redirect_port: int = 0


@dataclasses.dataclass(frozen=True)
class CatalogServer:
  """One curated MCP server offered by `/connect`."""

  name: str
  display_name: str
  url: str
  oauth: CatalogOAuth | None


_CATALOG = (
  CatalogServer(
    'notion', 'Notion', 'https://mcp.notion.com/mcp', CatalogOAuth()
  ),
  CatalogServer(
    'datadog', 'Datadog', 'https://mcp.datadoghq.com/v1/mcp', CatalogOAuth()
  ),
  CatalogServer(
    'linear', 'Linear', 'https://mcp.linear.app/mcp', CatalogOAuth()
  ),
  CatalogServer(
    'slack',
    'Slack',
    'https://mcp.slack.com/mcp',
    CatalogOAuth(
      client_id='6696120515462.11886240096129',
      redirect_host='localhost',
      redirect_port=8765,
    ),
  ),
)


def load_catalog() -> tuple[CatalogServer, ...]:
  """Return the curated MCP server catalog bundled with Mirror."""
  return _CATALOG


def config_dir() -> pathlib.Path:
  """Return the global MCP configuration directory."""
  configured = os.environ.get('MIRROR_MCP_DIR')
  return (
    pathlib.Path(configured).expanduser()
    if configured
    else pathlib.Path.home() / '.mirror' / 'mcp'
  )


def _server_path(name: str) -> pathlib.Path:
  if not _NAME.fullmatch(name):
    raise ValueError(
      'MCP server name may contain only letters, numbers, underscores, and dashes'
    )
  return config_dir() / f'{name}.json'


def _credentials_path(name: str) -> pathlib.Path:
  if not _NAME.fullmatch(name):
    raise ValueError(
      'MCP credential name may contain only letters, numbers, underscores, and dashes'
    )
  return config_dir() / 'credentials' / f'{name}.json'


def _read_json(path: pathlib.Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text())
  except json.JSONDecodeError as error:
    raise ValueError(f'invalid JSON in {path}: {error}') from error
  if not isinstance(value, dict):
    raise ValueError(f'{path} must contain a JSON object')
  return cast(dict[str, Any], value)


def load_servers() -> dict[str, dict[str, Any]]:
  """Load and minimally validate all configured MCP servers."""
  directory = config_dir()
  if not directory.exists():
    return {}
  servers: dict[str, dict[str, Any]] = {}
  for path in sorted(directory.glob('*.json')):
    try:
      name = path.stem
      _server_path(name)
      config = _read_json(path)
      has_url = isinstance(config.get('url'), str)
      has_command = isinstance(config.get('command'), str)
      if has_url == has_command:
        raise ValueError(
          f'{path} must contain exactly one of "url" or "command"'
        )
    except (OSError, UnicodeError, ValueError) as error:
      _LOGGER.warning('Ignoring invalid MCP server configuration: %s', error)
      continue
    servers[name] = config
  return servers


def add_server(server: CatalogServer) -> pathlib.Path:
  """Atomically add one HTTP MCP server definition."""
  path = _server_path(server.name)
  if path.exists():
    raise ValueError(f'MCP server {server.name!r} already exists')
  config: dict[str, Any] = {'url': server.url}
  if server.oauth is not None:
    config['auth'] = {'type': 'oauth', 'credential': server.name}
  _write_private_json(path, config)
  return path


async def install_server(
  server: CatalogServer,
  *,
  show_url: Callable[[str], Awaitable[None]],
) -> pathlib.Path:
  """Authenticate and atomically expose one catalog server."""
  path = _server_path(server.name)
  if path.exists():
    config = _read_json(path)
    if not requires_login(server.name, config):
      raise ValueError(f'MCP server {server.name!r} already exists')
    auth = cast(dict[str, object], config['auth'])
    credential = auth.get('credential', server.name)
    url = config.get('url')
    client_id = auth.get('clientId')
    if (
      not isinstance(credential, str)
      or not isinstance(url, str)
      or (client_id is not None and not isinstance(client_id, str))
    ):
      raise ValueError(
        f'MCP server {server.name!r} has invalid OAuth configuration'
      )
    oauth = server.oauth or CatalogOAuth()
    if client_id is not None:
      oauth = dataclasses.replace(oauth, client_id=client_id)
    await authenticate(
      dataclasses.replace(server, name=credential, url=url, oauth=oauth),
      show_url=show_url,
    )
    return path
  try:
    if server.oauth is not None:
      await authenticate(server, show_url=show_url)
    return add_server(server)
  except BaseException:
    _credentials_path(server.name).unlink(missing_ok=True)
    raise


def remove_server(name: str) -> None:
  """Remove one server definition and its credentials."""
  path = _server_path(name)
  if not path.exists():
    raise ValueError(f'unknown MCP server: {name}')
  path.unlink()
  _credentials_path(name).unlink(missing_ok=True)


def requires_login(name: str, config: Mapping[str, Any]) -> bool:
  """Return whether an OAuth server needs an explicit sign-in."""
  raw_auth = config.get('auth')
  if not isinstance(raw_auth, dict):
    return False
  auth = cast(dict[str, object], raw_auth)
  if auth.get('type') != 'oauth':
    return False
  credential = auth.get('credential', name)
  if not isinstance(credential, str):
    raise ValueError(f'MCP server {name!r} has an invalid credential name')
  path = _credentials_path(credential)
  saved = _read_json(path) if path.exists() else {}
  raw_tokens = saved.get('tokens')
  tokens = (
    cast(dict[str, object], raw_tokens) if isinstance(raw_tokens, dict) else {}
  )
  expiry = saved.get('expires_at')
  return not tokens.get('access_token') or (
    isinstance(expiry, (int, float))
    and expiry <= time.time()
    and not tokens.get('refresh_token')
  )


def _write_private_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
  path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.{secrets.token_hex(6)}')
  try:
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as output:
      json.dump(value, output, indent=2, sort_keys=True)
      output.write('\n')
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _write_credentials(name: str, updates: Mapping[str, Any]) -> None:
  path = _credentials_path(name)
  credentials = _read_json(path) if path.exists() else {}
  credentials.update(updates)
  _write_private_json(path, credentials)


class OAuthStorage:
  """Persist the official MCP SDK's OAuth records in one private file."""

  def __init__(
    self,
    name: str,
    default_client_info: mcp_shared_auth.OAuthClientInformationFull
    | None = None,
  ) -> None:
    self._name = name
    self._default_client_info = default_client_info

  def _read(self) -> dict[str, Any]:
    path = _credentials_path(self._name)
    return _read_json(path) if path.exists() else {}

  async def get_tokens(self) -> mcp_shared_auth.OAuthToken | None:
    value = self._read().get('tokens')
    return (
      mcp_shared_auth.OAuthToken.model_validate(value)
      if isinstance(value, dict)
      else None
    )

  async def set_tokens(self, tokens: mcp_shared_auth.OAuthToken) -> None:
    expires_at = (
      time.time() + tokens.expires_in if tokens.expires_in is not None else None
    )
    _write_credentials(
      self._name,
      {
        'tokens': tokens.model_dump(mode='json', exclude_none=True),
        'expires_at': expires_at,
      },
    )

  async def clear_tokens(self, *, clear_client_info: bool = False) -> None:
    path = _credentials_path(self._name)
    if not path.exists():
      return
    credentials = _read_json(path)
    credentials.pop('tokens', None)
    credentials.pop('expires_at', None)
    if clear_client_info:
      credentials.pop('client_info', None)
    _write_private_json(path, credentials)

  def token_expiry_time(self) -> float | None:
    value = self._read().get('expires_at')
    return float(value) if isinstance(value, (int, float)) else None

  async def get_client_info(
    self,
  ) -> mcp_shared_auth.OAuthClientInformationFull | None:
    value = self._read().get('client_info')
    if isinstance(value, dict):
      return mcp_shared_auth.OAuthClientInformationFull.model_validate(value)
    if self._default_client_info is not None:
      await self.set_client_info(self._default_client_info)
    return self._default_client_info

  async def set_client_info(
    self, client_info: mcp_shared_auth.OAuthClientInformationFull
  ) -> None:
    _write_credentials(
      self._name,
      {'client_info': client_info.model_dump(mode='json', exclude_none=True)},
    )

  def redirect_uri(self) -> str:
    value = self._read().get('client_info')
    if isinstance(value, dict):
      client_info = mcp_shared_auth.OAuthClientInformationFull.model_validate(
        value
      )
      if client_info.redirect_uris:
        return str(client_info.redirect_uris[0])
    raise ValueError(f'OAuth credential {self._name!r} requires authentication')


def oauth_provider(
  server_url: str,
  credential: str,
  *,
  redirect_uri: str | None = None,
  client_id: str | None = None,
  redirect_handler: Callable[[str], Awaitable[None]] | None = None,
  callback_handler: Callable[
    [], Awaitable[mcp_shared_auth.AuthorizationCodeResult]
  ]
  | None = None,
) -> mcp_auth.OAuthClientProvider:
  """Build the official SDK OAuth provider for one persisted credential."""
  storage = OAuthStorage(credential)
  if redirect_uri is None:
    redirect_uri = storage.redirect_uri()
  if client_id is not None:
    storage = OAuthStorage(
      credential,
      mcp_shared_auth.OAuthClientInformationFull.model_validate({
        'client_id': client_id,
        'redirect_uris': [redirect_uri],
        'token_endpoint_auth_method': 'none',
      }),
    )
  metadata = mcp_shared_auth.OAuthClientMetadata.model_validate({
    'client_name': 'Mirror',
    'redirect_uris': [redirect_uri],
    'token_endpoint_auth_method': 'none',
  })
  provider = mcp_auth.OAuthClientProvider(
    server_url,
    metadata,
    storage,
    redirect_handler=redirect_handler,
    callback_handler=callback_handler,
  )
  # The SDK stores `expires_in`, so restore its absolute deadline across restarts.
  provider.context.token_expiry_time = storage.token_expiry_time()
  return provider


@dataclasses.dataclass
class _Authentication:
  task: asyncio.Task[None]
  waiters: int = 0


_authentications: dict[str, _Authentication] = {}


async def authenticate(
  server: CatalogServer,
  *,
  show_url: Callable[[str], Awaitable[None]],
) -> None:
  """Share one interactive OAuth flow across callers using the same credential."""
  pending = _authentications.get(server.name)
  if pending is None:
    pending = _Authentication(
      asyncio.create_task(_authenticate(server, show_url=show_url))
    )
    _authentications[server.name] = pending
  pending.waiters += 1
  try:
    await asyncio.shield(pending.task)
  finally:
    pending.waiters -= 1
    if pending.waiters == 0:
      del _authentications[server.name]
      if not pending.task.done():
        pending.task.cancel()
      await asyncio.gather(pending.task, return_exceptions=True)


async def _authenticate(
  server: CatalogServer,
  *,
  show_url: Callable[[str], Awaitable[None]],
) -> None:
  """Authenticate through the official SDK before persisting configuration."""
  oauth = server.oauth
  if oauth is None:
    raise ValueError(f'MCP server {server.name!r} does not use OAuth')
  callback = OAuthCallback(oauth.redirect_host, oauth.redirect_port)
  await callback.start()
  try:
    redirect_uri = (
      f'http://{oauth.redirect_host}:{callback.port}/oauth/callback'
    )
    provider = oauth_provider(
      server.url,
      server.name,
      redirect_uri=redirect_uri,
      client_id=oauth.client_id,
      redirect_handler=show_url,
      callback_handler=callback.wait_for_code,
    )
    async with httpx2.AsyncClient(
      auth=provider, follow_redirects=True
    ) as http_client:
      transport = mcp_http.streamable_http_client(
        server.url, http_client=http_client
      )
      async with mcp_client.Client(transport, mode='legacy'):
        pass
  finally:
    await callback.close()


class OAuthCallback:
  """Single-use loopback OAuth callback server."""

  def __init__(self, host: str, port: int) -> None:
    self._host = host
    self._requested_port = port
    self._server: asyncio.Server | None = None
    self._result: asyncio.Future[dict[str, str]] | None = None
    self.port = port

  async def start(self) -> None:
    loop = asyncio.get_running_loop()
    self._result = loop.create_future()
    self._server = await asyncio.start_server(
      self._handle, self._host, self._requested_port
    )
    socket = self._server.sockets[0]
    self.port = cast(tuple[str, int], socket.getsockname())[:2][1]

  async def _handle(
    self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
  ) -> None:
    try:
      line = await reader.readline()
      parts = line.decode(errors='replace').split()
      query = (
        urllib.parse.parse_qs(urllib.parse.urlparse(parts[1]).query)
        if len(parts) >= 2
        else {}
      )
      values = {key: items[0] for key, items in query.items() if items}
      if self._result is not None and not self._result.done():
        self._result.set_result(values)
      body = b'Authentication complete. You may close this window.'
      writer.write(
        b'HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\n'
        + f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n'.encode()
        + body
      )
      await writer.drain()
    finally:
      writer.close()
      await writer.wait_closed()

  async def wait_for_code(self) -> mcp_shared_auth.AuthorizationCodeResult:
    if self._result is None:
      raise RuntimeError('OAuth callback server is not running')
    values = await asyncio.wait_for(self._result, timeout=300)
    if 'error' in values:
      raise ValueError(f'OAuth authorization failed: {values["error"]}')
    code = values.get('code')
    if not code:
      raise ValueError('OAuth callback omitted authorization code')
    return mcp_shared_auth.AuthorizationCodeResult(
      code=code,
      state=values.get('state'),
      iss=values.get('iss'),
    )

  async def close(self) -> None:
    if self._server is not None:
      self._server.close()
      await self._server.wait_closed()
