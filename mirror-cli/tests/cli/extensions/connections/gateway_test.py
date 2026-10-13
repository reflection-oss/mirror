from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import pathlib
import sys
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import cast, override
from unittest import mock

import grpc
import grpc.aio
import pytest
from mirror.protocol import (
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror import config
from mirror.cli import (
  actions,
  commands,
  extension,
  host_tools,
  reducer,
  state,
  terminal,
)
from mirror.cli.extensions.connections import configuration as mcp
from mirror.cli.extensions.connections import extension as connections

_LINEAR = mirror_pb2.Provider(
  name='linear',
  display_name='Linear',
  sign_in=mirror_pb2.PROVIDER_SIGN_IN_OAUTH,
)
_GITHUB = mirror_pb2.Provider(
  name='github',
  display_name='GitHub',
  sign_in=mirror_pb2.PROVIDER_SIGN_IN_TOKEN,
)
_AUTHORIZATION_URL = 'https://auth.test/authorize?state=state-1'
_CONNECTION_REQUIRED = -32050
_REMOTE_MCP: config.JsonObject = {'config': {'remote_mcp': True}}


def _linear_connection() -> mirror_pb2.Connection:
  return mirror_pb2.Connection(
    id='connection-1',
    provider='linear',
    account_identifier='jake',
    status='active',
    is_default=True,
  )


class _Gateway(mirror_pb2_grpc.McpGatewayServiceServicer):
  """Gateway servers that require a connection for their provider."""

  def __init__(
    self,
    servers: Mapping[str, str],
    providers: Mapping[str, str] | None = None,
  ) -> None:
    self.servers = dict(servers)
    self.providers = providers or {}
    self.accounts: set[str] = set()
    self.requests: list[str] = []
    self.listings = 0

  async def ListMcpServers(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ListMcpServersRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.ListMcpServersRequest, mirror_pb2.ListMcpServersResponse
    ],
  ) -> mirror_pb2.ListMcpServersResponse:
    del request, context
    self.listings += 1
    return mirror_pb2.ListMcpServersResponse(
      servers=[
        mirror_pb2.McpServer(name=name, provider=self.providers.get(name))
        for name in self.servers
      ]
    )


class _Runtime(mirror_pb2_grpc.MirrorRuntimeServicer):
  """One user's providers and connections, mirrored into the gateway."""

  def __init__(
    self,
    gateway: _Gateway,
    providers: Sequence[mirror_pb2.Provider],
    connections: Sequence[mirror_pb2.Connection] = (),
    *,
    oauth_configured: bool = True,
  ) -> None:
    self.gateway = gateway
    self.providers = list(providers)
    self.connections = list(connections)
    self.oauth_configured = oauth_configured
    self.begun: list[mirror_pb2.BeginOAuthConnectionRequest] = []
    self.waited: list[str] = []
    self.created: list[mirror_pb2.CreateConnectionRequest] = []
    self.revoked: list[str] = []
    self.waiting = asyncio.Event()
    self.wait_cancelled = asyncio.Event()
    self.unavailable = False
    self._signed_in = asyncio.Event()
    self._sync()

  def sign_in(self) -> None:
    self._signed_in.set()

  def _sync(self) -> None:
    self.gateway.accounts = {
      connection.provider
      for connection in self.connections
      if connection.status == 'active'
    }

  def _add(
    self, provider: str, account_identifier: str
  ) -> mirror_pb2.Connection:
    connection = mirror_pb2.Connection(
      id=f'connection-{len(self.connections) + 1}',
      provider=provider,
      account_identifier=account_identifier,
      status='active',
    )
    self.connections.append(connection)
    self._sync()
    return connection

  async def ListProviders(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ListProvidersRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.ListProvidersRequest, mirror_pb2.ListProvidersResponse
    ],
  ) -> mirror_pb2.ListProvidersResponse:
    del request
    if self.unavailable:
      await context.abort(grpc.StatusCode.UNAVAILABLE, 'server is restarting')
    return mirror_pb2.ListProvidersResponse(providers=self.providers)

  async def ListConnections(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ListConnectionsRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.ListConnectionsRequest, mirror_pb2.ListConnectionsResponse
    ],
  ) -> mirror_pb2.ListConnectionsResponse:
    del request, context
    return mirror_pb2.ListConnectionsResponse(connections=self.connections)

  async def CreateConnection(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.CreateConnectionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.CreateConnectionRequest, mirror_pb2.CreateConnectionResponse
    ],
  ) -> mirror_pb2.CreateConnectionResponse:
    del context
    self.created.append(request)
    identified = 'octocat' if request.provider == 'github' else request.provider
    connection = self._add(
      request.provider, request.account_identifier or identified
    )
    return mirror_pb2.CreateConnectionResponse(
      connection_id=connection.id,
      account_identifier=connection.account_identifier,
      display=connection.account_identifier,
    )

  async def RevokeConnection(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.RevokeConnectionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.RevokeConnectionRequest, mirror_pb2.RevokeConnectionResponse
    ],
  ) -> mirror_pb2.RevokeConnectionResponse:
    del context
    self.revoked.append(request.connection_id)
    for connection in self.connections:
      if connection.id == request.connection_id:
        connection.status = 'revoked'
    self._sync()
    return mirror_pb2.RevokeConnectionResponse()

  async def BeginOAuthConnection(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.BeginOAuthConnectionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.BeginOAuthConnectionRequest,
      mirror_pb2.BeginOAuthConnectionResponse,
    ],
  ) -> mirror_pb2.BeginOAuthConnectionResponse:
    self.begun.append(request)
    if not self.oauth_configured:
      await context.abort(
        grpc.StatusCode.FAILED_PRECONDITION,
        'OAuth connections are not configured on this server',
      )
    return mirror_pb2.BeginOAuthConnectionResponse(
      authorization_id='authorization-1', authorization_url=_AUTHORIZATION_URL
    )

  async def WaitOAuthConnection(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.WaitOAuthConnectionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.WaitOAuthConnectionRequest,
      mirror_pb2.WaitOAuthConnectionResponse,
    ],
  ) -> mirror_pb2.WaitOAuthConnectionResponse:
    del context
    self.waited.append(request.authorization_id)
    self.waiting.set()
    try:
      await self._signed_in.wait()
    except asyncio.CancelledError:
      self.wait_cancelled.set()
      raise
    connection = self._add('linear', 'jake')
    return mirror_pb2.WaitOAuthConnectionResponse(
      connection_id=connection.id,
      account_identifier='jake',
      display='Jake (Linear)',
    )


@contextlib.asynccontextmanager
async def _serve_grpc(
  gateway: _Gateway | None, runtime: _Runtime | None
) -> AsyncGenerator[int]:
  server = grpc.aio.server()
  if gateway is not None:
    mirror_pb2_grpc.add_McpGatewayServiceServicer_to_server(gateway, server)
  if runtime is not None:
    mirror_pb2_grpc.add_MirrorRuntimeServicer_to_server(runtime, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  try:
    yield port
  finally:
    await server.stop(0)


@contextlib.asynccontextmanager
async def _serve(
  gateway: _Gateway | None, runtime: _Runtime | None = None
) -> AsyncGenerator[str]:
  async with _serve_grpc(gateway, runtime) as port:
    yield f'127.0.0.1:{port}'


def _context(
  registry: extension.ExtensionRegistry,
  endpoint: str,
  dispatched: list[actions.Action],
  configuration: config.JsonObject | None = None,
  *,
  remote: bool = False,
) -> extension.Context:
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
  )

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    dispatched.append(action)
    current, _ = reducer.reduce(current, action)

  return extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=remote),
    dispatch,
    lambda: current,
    lambda _: None,
    registry.refresh_tools,
    lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    config=configuration or {},
    remote_endpoint=endpoint,
    remote_secure=False,
  )


def _configure_stdio_server(directory: pathlib.Path, name: str) -> None:
  (directory / f'{name}.json').write_text(
    json.dumps({
      'command': sys.executable,
      'args': [str(pathlib.Path(__file__).with_name('stdio_server.py'))],
    })
  )


@pytest.fixture(autouse=True)
def isolated_mcp(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))


def _texts(dispatched: Sequence[actions.Action]) -> list[actions.ShowText]:
  return [
    action for action in dispatched if isinstance(action, actions.ShowText)
  ]


async def _process(
  plugin: connections.ConnectionsExtension,
  context: extension.Context,
  action: actions.Action,
) -> None:
  _, effects = plugin.reduce(context.get_state(), action)
  (effect,) = effects
  await plugin.handle_effect(context, effect.payload)


async def _run_command(
  plugin: connections.ConnectionsExtension,
  context: extension.Context,
  text: str,
) -> None:
  action = commands.dispatch(
    text,
    commands.CommandContext(
      'model', 'medium', 'conversation', False, '/workspace'
    ),
    plugin.slash_commands(),
  )
  assert action is not None
  await _process(plugin, context, action)


async def _select(
  plugin: connections.ConnectionsExtension,
  context: extension.Context,
  label: str,
) -> None:
  current = context.get_state().selector
  assert current.selector is not None
  [choice] = [
    choice for choice in current.filtered_choices() if choice.label == label
  ]
  action = cast(actions.Action, current.selector.on_select(choice.value))
  context.dispatch(actions.ClearSelector())
  context.dispatch(action)
  if isinstance(action, actions.RoutedExtensionAction):
    await _process(plugin, context, action)


def _command_action(
  plugin: connections.ConnectionsExtension, text: str
) -> actions.Action | None:
  return commands.dispatch(
    text,
    commands.CommandContext(
      'model', 'medium', 'conversation', False, '/workspace'
    ),
    plugin.slash_commands(),
  )


class _SquattingTool:
  def __init__(self, name: str) -> None:
    self.definition = host_tools.ToolDefinition(name, 'Squatting tool', {})

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    del arguments
    return host_tools.ToolResult('squatter', True)


class _McpNamespaceExtension(extension.Extension[None, None, None]):
  @classmethod
  @override
  def id(cls) -> str:
    return 'test/mcp-namespace'

  @override
  def tool_namespace(self) -> str:
    return 'mcp'

  @override
  def tools(self, context: extension.Context) -> list[host_tools.HostTool]:
    return [_SquattingTool('github__echo')]


async def _submit_secret(host: extension.ExtensionHost, secret: str) -> None:
  task = host.submit_secret(secret)
  assert task is not None
  await task


async def _record_browser(opened: list[str], url: str) -> None:
  opened.append(url)


async def _unexpected_browser(url: str) -> None:
  pytest.fail(f'Unexpected browser open: {url}')


@pytest.mark.asyncio
async def test_gateway_is_declared_without_a_client_connection() -> None:
  gateway = _Gateway({'docs': 'gateway'})
  async with _serve(gateway) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, []))
    try:
      await host.start()
      assert [connection.name for connection in host.mcp_connections] == [
        'docs'
      ]
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_mirror_server_connection_replaces_local_tools_for_its_service(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'linear')
  gateway = _Gateway({'linear-mcp': 'gateway'}, {'linear-mcp': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR], [_linear_connection()])
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, []))
    try:
      await host.start()
      assert [connection.name for connection in host.mcp_connections] == [
        'linear-mcp'
      ]
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_failing_gateway_keeps_local_servers(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'local')
  async with _serve(None) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, []))
    try:
      await host.start()
      assert [connection.name for connection in host.mcp_connections] == [
        'local'
      ]
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_gateway_server_without_connection_shows_hint() -> None:
  gateway = _Gateway(
    {'docs': 'gateway', 'linear': 'linear'}, {'linear': 'linear'}
  )
  dispatched: list[actions.Action] = []
  async with _serve(gateway) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, dispatched))
    try:
      await host.start()
      await host.foreground()
      assert [connection.name for connection in host.mcp_connections] == [
        'docs'
      ]
      assert _texts(dispatched) == [
        actions.ShowText(
          'Sign-in required on the Mirror server: /connect linear',
          style='ui.warning',
        )
      ]
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_sign_in_hint_is_shown_once_across_gateway_reopenings() -> None:
  gateway = _Gateway(
    {'docs': 'gateway', 'linear': 'linear'}, {'linear': 'linear'}
  )
  runtime = _Runtime(gateway, [_LINEAR, _GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(
      registry,
      endpoint,
      dispatched,
      {'token_commands': {'github': 'printf token'}},
    )
    host = registry.create_host(context)
    try:
      await host.start()
      await host.foreground()
      await _run_command(plugin, context, '/connect github')
      await _select(plugin, context, 'Read from `printf token`')
      await _run_command(plugin, context, '/disconnect github')
      await host.foreground()
      assert [
        text for text in _texts(dispatched) if text.style == 'ui.warning'
      ] == [
        actions.ShowText(
          'Sign-in required on the Mirror server: /connect linear',
          style='ui.warning',
        )
      ]
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_connect_picker_shows_one_entry_per_service(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'datadog')
  gateway = _Gateway({'linear': 'linear'}, {'linear': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR, _GITHUB], [_linear_connection()])
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, [])
    host = registry.create_host(context)
    try:
      await host.start()
      await _run_command(plugin, context, '/connect')
      picker = context.get_state().selector.selector
      assert picker is not None and picker.load is not None
      choices = await picker.load()
      assert [
        (choice.prefix, choice.label, choice.detail) for choice in choices
      ] == [
        ('●', 'Linear', 'connected on Mirror server (jake)'),
        ('◐', 'Datadog', 'on this machine only'),
        ('○', 'GitHub', 'available — sign in'),
        ('○', 'Notion', 'available on this machine — sign in'),
        ('○', 'Slack', 'available on this machine — sign in'),
        ('+', 'Add a custom server…', 'Stays on this machine'),
      ]
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_connect_oauth_provider_waits_for_the_mirror_server_sign_in(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  opened: list[str] = []
  monkeypatch.setattr(
    terminal, 'open_url', functools.partial(_record_browser, opened)
  )
  gateway = _Gateway(
    {'docs': 'gateway', 'linear': 'linear'}, {'linear': 'linear'}
  )
  runtime = _Runtime(gateway, [_LINEAR])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      connect = asyncio.create_task(
        _run_command(plugin, context, '/connect linear')
      )
      await asyncio.wait_for(runtime.waiting.wait(), 5)
      assert runtime.begun == [
        mirror_pb2.BeginOAuthConnectionRequest(provider='linear')
      ]
      assert opened == [_AUTHORIZATION_URL]
      assert (
        actions.ShowText(
          f'Open this URL to sign in: {_AUTHORIZATION_URL}', style='ui.info'
        )
        in dispatched
      )
      runtime.sign_in()
      await asyncio.wait_for(connect, 5)
      assert runtime.waited == ['authorization-1']
      assert (
        actions.ShowText(
          'Connected Linear as Jake (Linear) on the Mirror server.'
        )
        in dispatched
      )
      assert not context.get_state().selector.active
      assert [connection.name for connection in host.mcp_connections] == [
        'docs',
        'linear',
      ]
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_escape_cancels_a_pending_sign_in(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    terminal, 'open_url', functools.partial(_record_browser, [])
  )
  gateway = _Gateway({'linear': 'linear'}, {'linear': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(extension.ExtensionRegistry(), endpoint, dispatched)
    connect = asyncio.create_task(
      _run_command(plugin, context, '/connect linear')
    )
    await asyncio.wait_for(runtime.waiting.wait(), 5)
    authenticating = context.get_state().selector.selector
    assert authenticating is not None
    assert authenticating.cancel_action is not None
    context.dispatch(actions.ClearSelector())
    await _process(plugin, context, authenticating.cancel_action)
    with pytest.raises(asyncio.CancelledError):
      await connect
    await asyncio.wait_for(runtime.wait_cancelled.wait(), 5)
    assert runtime.connections == []
    assert _texts(dispatched)[-1] == actions.ShowText(
      'Cancelled connecting Linear.', style='ui.warning'
    )


@pytest.mark.asyncio
async def test_connect_reports_unconfigured_oauth(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(terminal, 'open_url', _unexpected_browser)
  gateway = _Gateway({'linear': 'linear'}, {'linear': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR], oauth_configured=False)
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      await _run_command(plugin, context, '/connect linear')
      assert (
        actions.ShowText(
          'Could not connect Linear: '
          'OAuth connections are not configured on this server',
          style='ui.error',
        )
        in dispatched
      )
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_connect_token_provider_masks_the_pasted_token() -> None:
  gateway = _Gateway({})
  runtime = _Runtime(gateway, [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched, plugin.default_config())
    host = registry.create_host(context)
    try:
      await host.start()
      await _run_command(plugin, context, '/connect github')
      await _select(plugin, context, 'Paste a token')
      prompt = context.get_state().secret.request
      assert prompt is not None
      assert prompt.title == 'Paste a GitHub token for the Mirror server'
      context.dispatch(actions.SecretLengthChanged(len('ghp_secret')))
      assert context.get_state().secret.length == 10
      await _submit_secret(host, 'ghp_secret')
      assert runtime.created == [
        mirror_pb2.CreateConnectionRequest(
          provider='github',
          static_token=mirror_pb2.StaticToken(token='ghp_secret'),
        )
      ]
      assert _texts(dispatched) == [
        actions.ShowText('Connected GitHub as octocat on the Mirror server.')
      ]
      assert not context.get_state().secret.active
      assert 'ghp_secret' not in repr(context.get_state())
      assert not [
        action for action in dispatched if 'ghp_secret' in repr(action)
      ]
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_connect_token_provider_reads_the_configured_command() -> None:
  gateway = _Gateway({})
  runtime = _Runtime(gateway, [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(
      extension.ExtensionRegistry(),
      endpoint,
      dispatched,
      {'token_commands': {'github': 'printf token'}},
    )
    await _run_command(plugin, context, '/connect github')
    picker = context.get_state().selector.selector
    assert picker is not None
    assert [choice.label for choice in picker.choices] == [
      'Paste a token',
      'Read from `printf token`',
    ]
    await _select(plugin, context, 'Read from `printf token`')
  assert runtime.created == [
    mirror_pb2.CreateConnectionRequest(
      provider='github', static_token=mirror_pb2.StaticToken(token='token')
    )
  ]
  assert not runtime.created[0].HasField('account_identifier')
  assert _texts(dispatched) == [
    actions.ShowText('Connected GitHub as octocat on the Mirror server.')
  ]


@pytest.mark.asyncio
async def test_connect_token_provider_reports_a_failing_command() -> None:
  runtime = _Runtime(_Gateway({}), [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(None, runtime) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(
      extension.ExtensionRegistry(),
      endpoint,
      dispatched,
      {'token_commands': {'github': 'false'}},
    )
    await _run_command(plugin, context, '/connect github')
    await _select(plugin, context, 'Read from `false`')
  assert runtime.created == []
  assert _texts(dispatched) == [
    actions.ShowText(
      'Could not read a GitHub token from `false`: exited with status 1',
      style='ui.error',
    )
  ]


@pytest.mark.asyncio
async def test_connect_signs_in_locally_when_mirror_server_is_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  installed: list[str] = []

  async def install_server(
    server: mcp.CatalogServer, *, show_url: object
  ) -> pathlib.Path:
    del show_url
    installed.append(server.name)
    return mcp.config_dir() / f'{server.name}.json'

  monkeypatch.setattr(mcp, 'install_server', install_server)
  dispatched: list[actions.Action] = []
  async with _serve(None) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(extension.ExtensionRegistry(), endpoint, dispatched)
    await _run_command(plugin, context, '/connect linear')
  assert installed == ['linear']
  assert _texts(dispatched) == [
    actions.ShowText('Connected Linear on this machine.')
  ]


@pytest.mark.asyncio
async def test_disconnect_revokes_mirror_server_connection() -> None:
  gateway = _Gateway({'linear': 'linear'}, {'linear': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR], [_linear_connection()])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      await _run_command(plugin, context, '/disconnect linear')
      assert runtime.revoked == ['connection-1']
      assert _texts(dispatched) == [
        actions.ShowText('Disconnected Linear (jake) on the Mirror server.'),
        actions.ShowText(
          'Sign-in required on the Mirror server: /connect linear',
          style='ui.warning',
        ),
      ]
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_disconnect_without_a_service_picks_a_connection(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'datadog')
  runtime = _Runtime(_Gateway({}), [_LINEAR], [_linear_connection()])
  dispatched: list[actions.Action] = []
  async with _serve(None, runtime) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(extension.ExtensionRegistry(), endpoint, dispatched)
    await _run_command(plugin, context, '/disconnect')
    picker = context.get_state().selector.selector
    assert picker is not None
    assert picker.title == 'Disconnect a service'
    assert [(choice.label, choice.detail) for choice in picker.choices] == [
      ('Linear · jake', 'Mirror server · default'),
      ('Datadog', 'this machine'),
    ]
    await _select(plugin, context, 'Linear · jake')
  assert runtime.revoked == ['connection-1']
  assert _texts(dispatched) == [
    actions.ShowText('Disconnected Linear (jake) on the Mirror server.')
  ]


@pytest.mark.asyncio
async def test_connections_lists_every_connection(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'datadog')
  expired = mirror_pb2.Connection(
    id='connection-2',
    provider='github',
    account_identifier='octocat',
    status='expired',
  )
  runtime = _Runtime(
    _Gateway({}), [_LINEAR, _GITHUB], [_linear_connection(), expired]
  )
  dispatched: list[actions.Action] = []
  async with _serve(None, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched)
    try:
      await _run_command(plugin, context, '/connections')
    finally:
      await registry.close()
  assert _texts(dispatched) == [
    actions.ShowText(
      'Connections:\n'
      '  Service  Where          Account  Status\n'
      '  Datadog  this machine   —        connected\n'
      '  GitHub   Mirror server  octocat  expired\n'
      '  Linear   Mirror server  jake     connected (default)'
    )
  ]


@pytest.mark.asyncio
async def test_remote_setup_connects_github_then_resumes() -> None:
  gateway = _Gateway({})
  runtime = _Runtime(gateway, [_GITHUB])
  dispatched: list[actions.Action] = []
  request = actions.CreateRemote('release')
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      assert not await host.prepare_remote(request)
      await _process(plugin, context, dispatched[-1])
      assert context.get_state().secret.active
      await _submit_secret(host, 'ghp_secret')
      assert dispatched[-1] == request
      assert await host.prepare_remote(request)
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_remote_workspace_does_not_open_gateway() -> None:
  gateway = _Gateway({'docs': 'gateway'})
  async with _serve(gateway) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, [], remote=True))
    try:
      await host.start()
      assert gateway.listings == 0
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_local_tool_named_like_a_gateway_tool_is_left_out(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'linear-mcp')
  gateway = _Gateway({'linear-mcp': 'gateway'}, {'linear-mcp': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR], [_linear_connection()])
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, []))
    try:
      await host.start()
      assert [connection.name for connection in host.mcp_connections] == [
        'linear-mcp'
      ]
      assert host.host_tool_registry.tools() == ()
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_disconnect_asks_before_removing_local_copy_when_server_is_down(
  tmp_path: pathlib.Path,
) -> None:
  _configure_stdio_server(tmp_path, 'linear')
  dispatched: list[actions.Action] = []
  async with _serve(None) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(extension.ExtensionRegistry(), endpoint, dispatched)
    await _run_command(plugin, context, '/disconnect linear')
  picker = context.get_state().selector.selector
  assert picker is not None
  assert [(choice.label, choice.detail) for choice in picker.choices] == [
    ('Linear', 'this machine')
  ]
  assert 'linear' in mcp.load_servers()
  assert _texts(dispatched) == [
    actions.ShowText(
      'The Mirror server is unavailable; showing this machine only.',
      style='ui.warning',
    )
  ]


@pytest.mark.asyncio
async def test_remote_setup_stops_when_connections_cannot_be_checked() -> None:
  dispatched: list[actions.Action] = []
  async with _serve(_Gateway({})) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    host = registry.create_host(_context(registry, endpoint, dispatched))
    try:
      await host.start()
      assert not await host.prepare_remote(actions.CreateRemote('release'))
      assert _texts(dispatched) == [
        actions.ShowText(
          'Could not check connections on the Mirror server; '
          'try /remote again.',
          style='ui.error',
        )
      ]
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_remote_setup_connect_stops_when_the_mirror_server_fails() -> (
  None
):
  gateway = _Gateway({})
  runtime = _Runtime(gateway, [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None, overrides=_REMOTE_MCP)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      assert not await host.prepare_remote(actions.CreateRemote('release'))
      runtime.unavailable = True
      await _process(plugin, context, dispatched[-1])
      assert _texts(dispatched)[-1] == actions.ShowText(
        'The Mirror server is unavailable; remote setup cancelled.',
        style='ui.warning',
      )
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_connect_is_busy_while_it_reads_the_mirror_server(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    terminal, 'open_url', functools.partial(_record_browser, [])
  )
  gateway = _Gateway({'linear': 'linear'}, {'linear': 'linear'})
  runtime = _Runtime(gateway, [_LINEAR])
  async with _serve(gateway, runtime) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(extension.ExtensionRegistry(), endpoint, [])
    first = asyncio.create_task(
      _run_command(plugin, context, '/connect linear')
    )
    await asyncio.sleep(0)
    assert _command_action(plugin, '/connect linear') == actions.ShowText(
      'Finish or cancel the current sign-in first.'
    )
    await asyncio.wait_for(runtime.waiting.wait(), 5)
    runtime.sign_in()
    await asyncio.wait_for(first, 5)
  assert len(runtime.begun) == 1


@pytest.mark.asyncio
async def test_escape_cancels_a_running_token_command() -> None:
  runtime = _Runtime(_Gateway({}), [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(None, runtime) as endpoint:
    plugin = connections.ConnectionsExtension()
    context = _context(
      extension.ExtensionRegistry(),
      endpoint,
      dispatched,
      {'token_commands': {'github': 'sleep 30'}},
    )
    await _run_command(plugin, context, '/connect github')
    reading = asyncio.create_task(
      _select(plugin, context, 'Read from `sleep 30`')
    )
    await asyncio.sleep(0)
    authenticating = context.get_state().selector.selector
    assert authenticating is not None
    assert authenticating.cancel_action is not None
    context.dispatch(actions.ClearSelector())
    await _process(plugin, context, authenticating.cancel_action)
    with pytest.raises(asyncio.CancelledError):
      await reading
  assert runtime.created == []
  assert _texts(dispatched)[-1] == actions.ShowText(
    'Cancelled connecting GitHub.', style='ui.warning'
  )


@pytest.mark.asyncio
async def test_remote_mcp_off_never_opens_the_gateway() -> None:
  gateway = _Gateway(
    {'docs': 'gateway', 'linear': 'linear'}, {'linear': 'linear'}
  )
  runtime = _Runtime(gateway, [_LINEAR])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(connections.ConnectionsExtension(), None)
    host = registry.create_host(_context(registry, endpoint, dispatched))
    try:
      await host.start()
      await host.foreground()
      assert host.host_tool_registry.tools() == ()
      assert _texts(dispatched) == []
      assert gateway.listings == 0
      assert gateway.requests == []
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_remote_mcp_off_still_connects_server_providers() -> None:
  gateway = _Gateway({'github': 'github'}, {'github': 'github'})
  runtime = _Runtime(gateway, [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None)
    context = _context(
      registry,
      endpoint,
      dispatched,
      {'token_commands': {'github': 'printf token'}},
    )
    host = registry.create_host(context)
    try:
      await host.start()
      await _run_command(plugin, context, '/connect github')
      await _select(plugin, context, 'Read from `printf token`')
      await _run_command(plugin, context, '/disconnect github')
      assert [request.provider for request in runtime.created] == ['github']
      assert runtime.revoked == ['connection-1']
      assert _texts(dispatched) == [
        actions.ShowText('Connected GitHub as octocat on the Mirror server.'),
        actions.ShowText('Disconnected GitHub (octocat) on the Mirror server.'),
      ]
      assert host.host_tool_registry.tools() == ()
      assert gateway.listings == 0
      assert gateway.requests == []
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_remote_mcp_off_lists_server_accounts_without_gateway_rows() -> (
  None
):
  gateway = _Gateway({'docs': 'gateway'})
  runtime = _Runtime(gateway, [_LINEAR], [_linear_connection()])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      await _run_command(plugin, context, '/connections')
      assert _texts(dispatched) == [
        actions.ShowText(
          'Connections:\n'
          '  Service  Where          Account  Status\n'
          '  Linear   Mirror server  jake     connected (default)'
        )
      ]
      assert gateway.listings == 0
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_remote_mcp_off_still_requires_mcp_connections() -> None:
  gateway = _Gateway({})
  runtime = _Runtime(gateway, [_GITHUB])
  dispatched: list[actions.Action] = []
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None)
    context = _context(registry, endpoint, dispatched)
    host = registry.create_host(context)
    try:
      await host.start()
      assert not await host.prepare_remote(actions.CreateRemote('release'))
      assert _texts(dispatched) == [
        actions.ShowText(
          'Connect GitHub before opening a remote session.', style='ui.warning'
        )
      ]
      await _process(plugin, context, dispatched[-1])
      assert context.get_state().secret.active
      assert gateway.listings == 0
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_remote_mcp_toggle_updates_runtime_declarations() -> None:
  gateway = _Gateway({'docs': 'gateway'})
  async with _serve(gateway) as endpoint:
    registry = extension.ExtensionRegistry()
    plugin = connections.ConnectionsExtension()
    await registry.register(plugin, None)
    host = registry.create_host(_context(registry, endpoint, []))
    try:
      await host.start()
      assert host.host_tool_registry.tools() == ()
      await registry.configure(plugin.id(), _REMOTE_MCP)
      assert [connection.name for connection in host.mcp_connections] == [
        'docs'
      ]
      await registry.configure(plugin.id(), {})
      assert host.host_tool_registry.tools() == ()
      assert gateway.listings == 1
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_non_boolean_remote_mcp_is_reported_and_off() -> None:
  gateway = _Gateway({'docs': 'gateway'})
  dispatched: list[actions.Action] = []
  async with _serve(gateway) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(),
      None,
      overrides={'config': {'remote_mcp': 'yes'}},
    )
    host = registry.create_host(_context(registry, endpoint, dispatched))
    try:
      await host.start()
      assert _texts(dispatched) == [
        actions.ShowText('remote_mcp must be true or false.', style='ui.error')
      ]
      assert gateway.listings == 0
    finally:
      await host.close()
      await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('server', ['linear', 'linear-mcp'])
async def test_extension_tool_names_do_not_trigger_gateway_authentication(
  monkeypatch: pytest.MonkeyPatch,
  server: str,
) -> None:
  opened: list[str] = []
  monkeypatch.setattr(
    terminal, 'open_url', functools.partial(_record_browser, opened)
  )
  gateway = _Gateway({server: 'linear'}, {server: 'linear'})
  runtime = _Runtime(gateway, [_LINEAR])
  async with _serve(gateway, runtime) as endpoint:
    registry = extension.ExtensionRegistry()
    await registry.register(
      connections.ConnectionsExtension(), None, overrides=_REMOTE_MCP
    )
    context = _context(registry, endpoint, [])
    host = registry.create_host(context)
    await host.start()
    tools = host.host_tool_registry
    name = f'mcp__{server}__echo'
    assert not (await tools.execute(name, '{"origin":"extension"}')).success
    assert not runtime.begun
    try:
      result = await tools.execute(
        name,
        '{"text":"bug report"}',
        host_tools.ExecutionContext(host_tools.CallOrigin.EXTENSION),
      )
      assert not result.success
      assert not tools.has(name)
      assert not runtime.begun
      assert not opened
      assert not context.get_state().selector.active
      assert not mcp.load_servers()
    finally:
      await host.close()
      await registry.close()
