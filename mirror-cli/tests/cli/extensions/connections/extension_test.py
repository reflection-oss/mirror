from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import sys
from collections.abc import Awaitable, Callable
from unittest import mock

import httpx2
import pytest

from mirror.cli import (
  actions,
  commands,
  extension,
  host_tools,
  mcp_configuration,
  reducer,
  state,
)
from mirror.cli.extensions.connections import configuration as mcp
from mirror.cli.extensions.connections import extension as connections_extension
from mirror.cli.extensions.connections import transport as mcp_extensions


def _catalog() -> tuple[mcp.CatalogServer, ...]:
  return (
    mcp.CatalogServer('notion', 'Notion', 'https://notion.test/mcp', None),
    mcp.CatalogServer('datadog', 'Datadog', 'https://datadog.test/mcp', None),
  )


@pytest.fixture(autouse=True)
def isolated_mcp(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  monkeypatch.setenv('FIRECRAWL_API_KEY', 'configured-firecrawl-key')
  monkeypatch.setattr(mcp, 'load_catalog', _catalog)


def _context(
  registry: extension.ExtensionRegistry, dispatched: list[actions.Action]
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
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
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
  )


async def _run(
  plugin: connections_extension.ConnectionsExtension,
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
  _, effects = plugin.reduce(context.get_state(), action)
  await plugin.handle_effect(context, effects[0].payload)


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', [False, True])
async def test_unloading_mcp_removes_runtime_connections(
  monkeypatch: pytest.MonkeyPatch,
  remote: bool,
) -> None:
  connections = mcp_extensions.McpConnections((
    mcp_configuration.McpConnection(
      'test', {'url': 'https://example.test/mcp'}
    ),
  ))

  def open_connections(
    *, on_auth_required: Callable[[str], None] | None = None
  ) -> mcp_extensions.McpConnections:
    return connections

  monkeypatch.setattr(mcp_extensions, 'local_connections', open_connections)
  registry = extension.ExtensionRegistry()
  plugin = connections_extension.ConnectionsExtension()
  await registry.register(plugin, None)
  context = _context(registry, [])
  host = registry.create_host(
    dataclasses.replace(
      context, workspace=dataclasses.replace(context.workspace, remote=remote)
    )
  )
  try:
    await host.start()
    assert {command.name for command in host.slash_commands} == {
      'connect',
      'connections',
      'disconnect',
    }
    assert host.host_tool_registry.tools() == ()
    assert [connection.name for connection in host.mcp_connections] == (
      [] if remote else ['test']
    )
    await registry.unload(plugin.id())
    assert host.mcp_connections == ()
    assert host.slash_commands == ()
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_default_search_connection_reaches_the_runtime(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('FIRECRAWL_API_KEY')
  registry = extension.ExtensionRegistry()
  await registry.register(connections_extension.ConnectionsExtension(), None)
  host = registry.create_host(_context(registry, []))
  try:
    await host.start()
    assert [connection.name for connection in host.mcp_connections] == [
      'parallel'
    ]
    assert host.mcp_connections[0].config['enabledTools'] == [
      'web_search',
      'web_fetch',
    ]
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_extension_declares_stdio_without_starting_a_client(
  tmp_path: pathlib.Path,
) -> None:
  (tmp_path / 'test.json').write_text(
    json.dumps({
      'command': sys.executable,
      'args': [str(pathlib.Path(__file__).with_name('stdio_server.py'))],
    })
  )
  registry = extension.ExtensionRegistry()
  await registry.register(connections_extension.ConnectionsExtension(), None)
  host = registry.create_host(_context(registry, []))
  try:
    await host.start()
    assert host.host_tool_registry.tools() == ()
    assert host.mcp_connections[0].name == 'test'
    assert host.mcp_connections[0].config['command'] == sys.executable
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_install_reports_saved_configuration_when_activation_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry = extension.ExtensionRegistry()
  plugin = connections_extension.ConnectionsExtension()
  await registry.register(plugin, None)
  dispatched: list[actions.Action] = []
  context = _context(registry, dispatched)
  host = registry.create_host(context)
  await host.start()
  replacement = mcp_extensions.McpConnections((
    mcp_configuration.McpConnection(
      'notion', {'url': 'https://notion.test/mcp'}
    ),
  ))

  def open_connections(
    *, on_auth_required: Callable[[str], None] | None = None
  ) -> mcp_extensions.McpConnections:
    return replacement

  async def fail_refresh(extension_id: str) -> None:
    raise ValueError(f'Cannot refresh {extension_id}')

  monkeypatch.setattr(mcp_extensions, 'local_connections', open_connections)
  context = dataclasses.replace(context, refresh_tools=fail_refresh)
  try:
    await _run(plugin, context, '/connect notion')
    assert 'notion' in mcp.load_servers()
    assert isinstance(dispatched[-1], actions.ShowText)
    assert 'was saved, but could not be activated' in dispatched[-1].text
    assert host.mcp_connections == ()
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_only_read_only_connections_command_is_available_during_turns() -> (
  None
):
  plugin = connections_extension.ConnectionsExtension()
  available = plugin.slash_commands()
  assert [
    command.name for command in available if command.is_available(busy=True)
  ] == ['connections']
  dispatched: list[actions.Action] = []
  await _run(
    plugin,
    _context(extension.ExtensionRegistry(), dispatched),
    '/connections',
  )
  assert dispatched == [
    actions.ShowText('No connections. Run /connect to add one.')
  ]


@pytest.mark.asyncio
async def test_authentication_uses_a_cancellable_selector(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  server = mcp.CatalogServer(
    'notion', 'Notion', 'https://notion.test', mcp.CatalogOAuth()
  )
  monkeypatch.setattr(mcp, 'load_catalog', lambda: (server,))
  started = asyncio.Event()

  async def install_server(
    server: mcp.CatalogServer, *, show_url: Callable[[str], Awaitable[None]]
  ) -> None:
    del server, show_url
    started.set()
    await asyncio.Event().wait()

  monkeypatch.setattr(mcp, 'install_server', install_server)
  registry = extension.ExtensionRegistry()
  plugin = connections_extension.ConnectionsExtension()
  context = _context(registry, [])
  install = asyncio.create_task(_run(plugin, context, '/connect notion'))
  try:
    await asyncio.wait_for(started.wait(), 2)
    request = context.get_state().selector.selector
    assert request is not None
    assert request.title == 'Authenticating Notion'
    assert request.cancel_action is not None
    _, effects = plugin.reduce(context.get_state(), request.cancel_action)
    await plugin.handle_effect(context, effects[0].payload)
    with pytest.raises(asyncio.CancelledError):
      await install
    assert not context.get_state().selector.active
  finally:
    install.cancel()
    await asyncio.gather(install, return_exceptions=True)


@pytest.mark.asyncio
async def test_connect_and_disconnect_manage_local_configuration(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def open_connections(
    *, on_auth_required: Callable[[str], None] | None = None
  ) -> mcp_extensions.McpConnections:
    return mcp_extensions.McpConnections()

  monkeypatch.setattr(mcp_extensions, 'local_connections', open_connections)
  registry = extension.ExtensionRegistry()
  plugin = connections_extension.ConnectionsExtension()
  context = _context(registry, [])
  await _run(plugin, context, '/connect notion')
  assert 'notion' in mcp.load_servers()
  await _run(plugin, context, '/disconnect notion')
  assert 'notion' not in mcp.load_servers()


@pytest.mark.asyncio
async def test_grouped_disconnect_selector_still_deletes_local_mcp() -> None:
  mcp.add_server(_catalog()[0])
  registry = extension.ExtensionRegistry()
  plugin = connections_extension.ConnectionsExtension()
  context = _context(registry, [])
  await _run(plugin, context, '/disconnect')
  picker = context.get_state().selector.selector
  assert picker is not None
  assert picker.choices[0].section == 'This machine'
  action = picker.on_select(picker.choices[0].value)
  assert isinstance(action, actions.RoutedExtensionAction)
  _, effects = plugin.reduce(context.get_state(), action)
  await plugin.handle_effect(context, effects[0].payload)
  assert 'notion' not in mcp.load_servers()


@pytest.mark.asyncio
async def test_install_reports_network_errors(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def install_server(
    server: mcp.CatalogServer, *, show_url: Callable[[str], Awaitable[None]]
  ) -> None:
    del server, show_url
    raise httpx2.HTTPError('offline')

  monkeypatch.setattr(mcp, 'install_server', install_server)
  registry = extension.ExtensionRegistry()
  dispatched: list[actions.Action] = []
  await _run(
    connections_extension.ConnectionsExtension(),
    _context(registry, dispatched),
    '/connect notion',
  )
  assert isinstance(dispatched[-1], actions.ShowText)
  assert 'Could not add Notion' in dispatched[-1].text


@pytest.mark.asyncio
async def test_startup_warns_once_for_missing_credentials_across_workspaces() -> (
  None
):
  mcp.add_server(dataclasses.replace(_catalog()[0], oauth=mcp.CatalogOAuth()))
  mcp.add_server(dataclasses.replace(_catalog()[1], oauth=mcp.CatalogOAuth()))
  expected = actions.ShowText(
    'Sign-in required on this machine: /connect datadog, /connect notion',
    style='ui.warning',
  )
  registry = extension.ExtensionRegistry()
  await registry.register(connections_extension.ConnectionsExtension(), None)
  dispatched: list[actions.Action] = []
  context = _context(registry, dispatched)
  host = registry.create_host(context)
  second = registry.create_host(
    dataclasses.replace(
      context, workspace=extension.Workspace(pathlib.Path('/second'), False)
    )
  )
  try:
    await host.start()
    await host.foreground()
    assert [
      action for action in dispatched if isinstance(action, actions.ShowText)
    ] == [expected]
    dispatched.clear()
    await host.background()
    await second.start()
    await second.foreground()
    await second.background()
    await host.foreground()
    assert not any(
      isinstance(action, actions.ShowText) for action in dispatched
    )
  finally:
    await second.close()
    await host.close()
    await registry.close()

  restarted = extension.ExtensionRegistry()
  await restarted.register(connections_extension.ConnectionsExtension(), None)
  dispatched.clear()
  new_host = restarted.create_host(_context(restarted, dispatched))
  try:
    await new_host.start()
    await new_host.foreground()
    assert [
      action for action in dispatched if isinstance(action, actions.ShowText)
    ] == [expected]
  finally:
    await new_host.close()
    await restarted.close()


@pytest.mark.asyncio
async def test_connect_reauthenticates_without_removing_configuration(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  server = mcp.CatalogServer(
    'notion', 'Notion', 'https://notion.test/mcp', mcp.CatalogOAuth()
  )
  monkeypatch.setattr(mcp, 'load_catalog', lambda: (server,))
  path = mcp.add_server(server)
  before = path.read_text()
  authenticated: list[str] = []

  async def authenticate(
    server: mcp.CatalogServer, *, show_url: Callable[[str], Awaitable[None]]
  ) -> None:
    authenticated.append(server.name)

  monkeypatch.setattr(mcp, 'authenticate', authenticate)
  await _run(
    connections_extension.ConnectionsExtension(),
    _context(extension.ExtensionRegistry(), []),
    '/connect notion',
  )
  assert authenticated == ['notion']
  assert path.read_text() == before


@pytest.mark.asyncio
async def test_connect_signs_in_with_a_custom_server_client_id(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  (tmp_path / 'custom.json').write_text(
    json.dumps({
      'url': 'https://custom.test/mcp',
      'auth': {'type': 'oauth', 'clientId': 'client-1'},
    })
  )
  authenticated: list[mcp.CatalogServer] = []

  async def authenticate(
    server: mcp.CatalogServer, *, show_url: Callable[[str], Awaitable[None]]
  ) -> None:
    authenticated.append(server)

  monkeypatch.setattr(mcp, 'authenticate', authenticate)
  await _run(
    connections_extension.ConnectionsExtension(),
    _context(extension.ExtensionRegistry(), []),
    '/connect custom',
  )
  [server] = authenticated
  assert server.url == 'https://custom.test/mcp'
  assert server.oauth == mcp.CatalogOAuth(client_id='client-1')


@pytest.mark.asyncio
@pytest.mark.parametrize('origin', list(host_tools.CallOrigin))
async def test_missing_tools_never_start_authentication(
  monkeypatch: pytest.MonkeyPatch,
  origin: host_tools.CallOrigin,
) -> None:
  authenticate = mock.AsyncMock(side_effect=AssertionError('Unexpected OAuth'))
  monkeypatch.setattr(mcp, 'authenticate', authenticate)
  registry = extension.ExtensionRegistry()
  await registry.register(connections_extension.ConnectionsExtension(), None)
  host = registry.create_host(_context(registry, []))
  try:
    await host.start()
    tools = host.host_tool_registry
    assert not tools.has('mcp__linear__save_issue')
    result = await tools.execute(
      'mcp__linear__save_issue', '{}', host_tools.ExecutionContext(origin)
    )
    assert result == host_tools.ToolResult(
      'Unknown host tool: mcp__linear__save_issue', False
    )
    authenticate.assert_not_called()
    assert tools.tools() == ()
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_disconnected_model_tool_does_not_start_authentication(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def open_connections(
    *,
    on_auth_required: Callable[[str], None] | None = None,
  ) -> mcp_extensions.McpConnections:
    return mcp_extensions.McpConnections((
      mcp_configuration.McpConnection(
        'linear', {'url': 'https://linear.test/mcp'}
      ),
    ))

  async def authenticate(
    server: mcp.CatalogServer,
    *,
    show_url: Callable[[str], Awaitable[None]],
  ) -> None:
    pytest.fail('Model calls must not start OAuth')

  monkeypatch.setattr(mcp, 'authenticate', authenticate)
  monkeypatch.setattr(mcp_extensions, 'local_connections', open_connections)
  registry = extension.ExtensionRegistry()
  await registry.register(connections_extension.ConnectionsExtension(), None)
  host = registry.create_host(_context(registry, []))
  try:
    await host.start()
    assert not (
      await host.host_tool_registry.execute('mcp__linear__echo', '{}')
    ).success
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_failed_service_connection_does_not_execute_a_tool(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def open_connections(
    *,
    on_auth_required: Callable[[str], None] | None = None,
  ) -> mcp_extensions.McpConnections:
    return mcp_extensions.McpConnections()

  monkeypatch.setattr(mcp_extensions, 'local_connections', open_connections)
  mcp.add_server(_catalog()[0])
  registry = extension.ExtensionRegistry()
  await registry.register(connections_extension.ConnectionsExtension(), None)
  host = registry.create_host(_context(registry, []))
  try:
    await host.start()
    result = await host.host_tool_registry.execute(
      'mcp__notion__echo',
      '{}',
      host_tools.ExecutionContext(host_tools.CallOrigin.EXTENSION),
    )
    assert not result.success
    assert 'Unknown host tool' in result.content
    assert not host.host_tool_registry.tools()
  finally:
    await host.close()
    await registry.close()
