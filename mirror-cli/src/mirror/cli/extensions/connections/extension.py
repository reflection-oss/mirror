"""Service connections on the Mirror server or this machine, and their tools."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import logging
import shlex
from collections.abc import (
  Awaitable,
  Callable,
  Generator,
  Iterable,
  Mapping,
  Sequence,
)
from typing import cast, override

import grpc
import httpx2
from mirror.protocol import mirror_pb2

from mirror import config
from mirror.cli import (
  actions,
  commands,
  extension,
  mcp_configuration,
  selection,
  state,
  terminal,
)
from mirror.cli.extensions.connections import configuration as mcp
from mirror.cli.extensions.connections import (
  gateway_transport,
  local_tokens,
  server_connections,
  services,
)
from mirror.cli.extensions.connections import transport as mcp_extensions

_LOGGER = logging.getLogger(__name__)
_AUTHENTICATION = 'connect-auth'
_TOKEN = 'connect-token:'
_PASTE = 'paste'
_COMMAND = 'command'
_CUSTOM = '+custom'
_LOCAL = 'local:'
_REQUIRED = 'required_remote_connections'
_TOKEN_COMMANDS = 'token_commands'
_REMOTE_MCP = 'remote_mcp'
_BUSY = 'Finish or cancel the current sign-in first.'
_STYLES = {
  services.Availability.SERVER: 'class:extension-on',
  services.Availability.LOCAL: 'class:extension-off',
}
_UNAVAILABLE = 'The Mirror server is unavailable; showing this machine only.'


@dataclasses.dataclass(frozen=True)
class _Pick(actions.ExtensionAction):
  pass


@dataclasses.dataclass(frozen=True)
class _Connect(actions.ExtensionAction):
  service: str
  resume: actions.CreateRemote | None = None


@dataclasses.dataclass(frozen=True)
class _SaveToken(actions.ExtensionAction):
  service: str
  label: str
  token: str = dataclasses.field(repr=False)
  resume: actions.CreateRemote | None = None


@dataclasses.dataclass(frozen=True)
class _ReadToken(actions.ExtensionAction):
  service: str
  label: str
  command: str
  resume: actions.CreateRemote | None = None


@dataclasses.dataclass(frozen=True)
class _List(actions.ExtensionAction):
  pass


@dataclasses.dataclass(frozen=True)
class _Disconnect(actions.ExtensionAction):
  service: str


@dataclasses.dataclass(frozen=True)
class _Revoke(actions.ExtensionAction):
  connection_id: str
  label: str


@dataclasses.dataclass(frozen=True)
class _RemoveLocal(actions.ExtensionAction):
  name: str
  label: str


@dataclasses.dataclass(frozen=True)
class _SetDefault(actions.ExtensionAction):
  connection_id: str
  label: str
  resume: actions.CreateRemote


@dataclasses.dataclass(frozen=True)
class _Cancel(actions.ExtensionAction):
  pass


@dataclasses.dataclass(frozen=True)
class _AuthenticationFailed(actions.ExtensionAction):
  server: str


type _Effect = (
  _Pick
  | _Connect
  | _SaveToken
  | _ReadToken
  | _List
  | _Disconnect
  | _Revoke
  | _RemoveLocal
  | _SetDefault
  | _Cancel
  | _AuthenticationFailed
)
_ROUTED = (
  _Pick,
  _Connect,
  _ReadToken,
  _List,
  _Disconnect,
  _Revoke,
  _RemoveLocal,
  _SetDefault,
  _Cancel,
  _AuthenticationFailed,
)


class ConnectionsExtension(extension.Extension[None, _Effect, None]):
  """Own service connections, their commands, and their MCP tools."""

  def __init__(self) -> None:
    self._local = mcp_extensions.McpConnections()
    self._gateway_connections = mcp_extensions.McpConnections()
    self._gateway: gateway_transport.Gateway | None = None
    self._gateway_lock = asyncio.Lock()
    self._pending_connections: set[str] = set()
    self._authentication: asyncio.Task[object] | None = None
    self._context: extension.Context | None = None
    self._pending_auth: set[str] = set()
    self._failed_servers: set[str] = set()
    self._hinted: set[str] = set()
    self._startup_pending = True

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/connections'

  @override
  def default_config(self) -> config.JsonObject:
    return {
      _REQUIRED: ['github'],
      _TOKEN_COMMANDS: {'github': 'gh auth token'},
      _REMOTE_MCP: False,
    }

  @override
  async def on_load(self, context: None) -> None:
    self._local = mcp_extensions.local_connections(
      on_auth_required=self._auth_required
    )

  @override
  async def on_config(self, context: extension.Context) -> None:
    gateway = _gateway(context)
    if gateway is None or context.workspace.remote:
      return
    wanted = gateway if _remote_mcp(context) else None
    async with self._gateway_lock:
      if wanted == self._gateway:
        return
      await self._open_gateway(context, wanted)
    if self._context is not None:
      self._report_connections(self._context)

  async def _open_gateway(
    self, context: extension.Context, gateway: gateway_transport.Gateway | None
  ) -> None:
    self._pending_connections.clear()
    connections = (
      await mcp_extensions.gateway_connections(
        gateway, on_connection_required=self._pending_connections.add
      )
      if gateway is not None
      else mcp_extensions.McpConnections()
    )
    await self._swap(
      context,
      self._gateway_connections,
      connections,
      self._use_gateway_connections,
    )
    self._gateway = gateway

  def _use_gateway_connections(
    self, connections: mcp_extensions.McpConnections
  ) -> None:
    self._gateway_connections = connections

  def _use_local(self, connections: mcp_extensions.McpConnections) -> None:
    self._local = connections

  async def _swap(
    self,
    context: extension.Context,
    previous: mcp_extensions.McpConnections,
    connections: mcp_extensions.McpConnections,
    use: Callable[[mcp_extensions.McpConnections], None],
  ) -> None:
    use(connections)
    try:
      await context.refresh_tools(self.id())
    except ValueError:
      use(previous)
      await context.refresh_tools(self.id())
      raise

  async def _refresh_gateway(
    self, context: extension.Context, gateway: gateway_transport.Gateway
  ) -> None:
    async with self._gateway_lock:
      if gateway == self._gateway:
        await self._open_gateway(context, gateway)

  def _report_connections(self, context: extension.Context) -> None:
    pending = self._pending_connections - self._local.connected_services
    unhinted = sorted(pending - self._hinted)
    self._hinted = (self._hinted & pending) | set(unhinted)
    if unhinted:
      context.dispatch(
        actions.ShowText(
          f'Sign-in required on the Mirror server: {_commands(unhinted)}',
          style='ui.warning',
        )
      )

  def _auth_required(self, server: str) -> None:
    if self._context is None:
      self._pending_auth.add(server)
    else:
      self._context.dispatch(
        actions.RoutedExtensionAction(self.id(), _AuthenticationFailed(server))
      )

  @override
  async def on_foreground(self, context: extension.Context) -> None:
    self._context = context
    if self._startup_pending:
      self._startup_pending = False
      pending = sorted(
        self._pending_auth - self._gateway_connections.connected_services
      )
      if pending:
        context.dispatch(
          actions.ShowText(
            f'Sign-in required on this machine: {_commands(pending)}',
            style='ui.warning',
          )
        )
      unavailable = sorted(
        mcp.load_servers().keys()
        - self._local.connected_servers
        - self._pending_auth
        - self._gateway_connections.connected_services
      )
      if unavailable:
        context.dispatch(
          actions.ShowText(
            f'MCP servers unavailable: {", ".join(unavailable)}. '
            f'Their tools were not loaded. Retry with {_commands(unavailable)}.',
            style='ui.warning',
          )
        )
      self._failed_servers.update(self._pending_auth)
      self._pending_auth.clear()
    for server in sorted(self._pending_auth):
      self._auth_required(server)
    self._pending_auth.clear()
    self._report_connections(context)

  @override
  async def on_background(self, context: extension.Context) -> None:
    if self._context is context:
      self._context = None

  @override
  def mcp_connections(
    self, context: extension.Context
  ) -> Sequence[mcp_configuration.McpConnection]:
    if context.workspace.remote:
      return ()
    gateway = (
      self._gateway_connections.connections
      if context.config.get(_REMOTE_MCP) is True
      else ()
    )
    services: set[str] = (
      self._gateway_connections.connected_services if gateway else set()
    )
    names = {connection.name for connection in gateway}
    return (
      *gateway,
      *(
        connection
        for connection in self._local.connections
        if connection.name not in names
        and self._local.service(connection.name) not in services
      ),
    )

  @override
  def tips(self) -> tuple[str, ...]:
    return ('/connect links services to Mirror for extra tools',)

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'connect',
        'Connect a service on the Mirror server or this machine',
        self._connect_command,
        arguments=commands.OptionalArguments('[service]'),
      ),
      commands.CommandDef(
        'connections',
        'Show connected services',
        lambda context, arguments: self._route(_List()),
        available_when_busy=True,
      ),
      commands.CommandDef(
        'disconnect',
        'Disconnect a service',
        lambda context, arguments: self._route(_Disconnect(arguments.strip())),
        arguments=commands.OptionalArguments('[service]'),
      ),
    )

  def _route(self, action: actions.ExtensionAction) -> actions.Action:
    return actions.RoutedExtensionAction(self.id(), action)

  def _connect_command(
    self, context: commands.CommandContext, arguments: str
  ) -> actions.Action:
    if self._authentication is not None:
      return actions.ShowText(_BUSY)
    service = arguments.strip()
    return self._route(_Connect(service) if service else _Pick())

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[None, tuple[extension.PrivateEffect[_Effect], ...]]:
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, _ROUTED)
    ):
      return None, (self.effect(action.action),)
    return None, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _Effect
  ) -> None:
    match effect:
      case _Pick():
        context.dispatch(self._picker(context))
      case _Connect(service, resume):
        await self._connect(context, service, resume)
      case _SaveToken(service, label, token, resume):
        await self._save_token(context, service, label, token, resume)
      case _ReadToken():
        await self._read_token(context, effect)
      case _List():
        await self._list(context)
      case _Disconnect(service):
        await self._disconnect(context, service)
      case _Revoke() | _RemoveLocal():
        await self._remove(context, effect)
      case _SetDefault(connection_id, label, resume):
        await self._set_default(context, connection_id, label, resume)
      case _Cancel():
        if self._authentication is not None:
          self._authentication.cancel()
      case _AuthenticationFailed(server):
        if server in self._failed_servers:
          return
        self._failed_servers.add(server)
        context.dispatch(
          actions.ShowText(
            f'{server} needs sign-in on this machine. Run /connect {server}.',
            style='ui.warning',
          )
        )

  def _local_state(self) -> services.Local:
    return services.Local(
      mcp.load_catalog(), mcp.load_servers(), self._local.connected_servers
    )

  def _picker(self, context: extension.Context) -> actions.Action:
    def on_select(value: str) -> actions.Action:
      if value == _CUSTOM:
        return actions.ShowText(_custom_help())
      return self._route(_Connect(value))

    return actions.OpenSelector(
      selection.Selector(
        'connect',
        'Connect a service',
        (),
        on_select,
        extension_id=self.id(),
        empty='Loading services…',
        load=functools.partial(self._choices, context),
      )
    )

  async def _choices(
    self, context: extension.Context
  ) -> tuple[selection.Choice, ...]:
    server = await _server(_gateway(context))
    offers = services.offers(
      server, self._gateway_connections.connected_services, self._local_state()
    )
    return (
      *(
        selection.Choice(
          offer.name,
          offer.display_name,
          offer.availability.value
          + (f' ({", ".join(offer.accounts)})' if offer.accounts else ''),
          prefix=offer.availability.symbol,
          prefix_style=_STYLES.get(offer.availability, ''),
          section=offer.availability.value.capitalize(),
        )
        for offer in offers
      ),
      selection.Choice(
        _CUSTOM,
        'Add a custom server…',
        'Stays on this machine',
        prefix='+',
        section='Custom',
      ),
    )

  async def _connect(
    self,
    context: extension.Context,
    service: str,
    resume: actions.CreateRemote | None,
  ) -> None:
    if self._authentication is not None:
      context.dispatch(actions.ShowText(_BUSY))
      return
    task = asyncio.current_task()
    self._authentication = task
    try:
      await self._connect_service(context, service, resume)
    finally:
      if self._authentication is task:
        self._authentication = None

  async def _connect_service(
    self,
    context: extension.Context,
    service: str,
    resume: actions.CreateRemote | None,
  ) -> None:
    gateway = _gateway(context)
    server = await _server(gateway)
    provider = server.provider(service) if server is not None else None
    if resume is not None and (gateway is None or server is None):
      context.dispatch(
        actions.ShowText(
          'The Mirror server is unavailable; remote setup cancelled.',
          style='ui.warning',
        )
      )
    elif gateway is None or provider is None:
      await self._connect_locally(context, service, server is not None)
    elif provider.sign_in == mirror_pb2.PROVIDER_SIGN_IN_OAUTH:
      await self._sign_in(context, gateway, provider, resume)
    elif provider.sign_in == mirror_pb2.PROVIDER_SIGN_IN_TOKEN:
      self._choose_token_source(context, provider, resume)
    else:
      context.dispatch(
        actions.ShowText(
          f'{provider.display_name} cannot be connected from this client.',
          style='ui.error',
        )
      )

  async def _sign_in(
    self,
    context: extension.Context,
    gateway: gateway_transport.Gateway,
    provider: mirror_pb2.Provider,
    resume: actions.CreateRemote | None,
  ) -> None:
    connection = await self._authenticate(
      context,
      provider.display_name,
      lambda: self._authorize(context, gateway, provider),
    )
    if connection is not None:
      await self._connected(
        context,
        gateway,
        provider.name,
        provider.display_name,
        connection.connection_id,
        connection.display or connection.account_identifier,
        resume,
      )

  async def _authenticate[T](
    self,
    context: extension.Context,
    label: str,
    sign_in: Callable[[], Awaitable[T]],
  ) -> T | None:
    with self._cancellable(context, label, 'Starting sign-in…'):
      try:
        return await sign_in()
      except (*server_connections.ERRORS, ValueError) as error:
        _show_failure(context, label, error)
        return None

  @contextlib.contextmanager
  def _cancellable(
    self, context: extension.Context, label: str, message: str
  ) -> Generator[None]:
    self._authentication = asyncio.current_task()
    self._show_authentication(context, label, message)
    try:
      yield
    except asyncio.CancelledError:
      context.dispatch(
        actions.ShowText(f'Cancelled connecting {label}.', style='ui.warning')
      )
      raise
    finally:
      self._finish_authentication(context)

  async def _authorize(
    self,
    context: extension.Context,
    gateway: gateway_transport.Gateway,
    provider: mirror_pb2.Provider,
  ) -> mirror_pb2.WaitOAuthConnectionResponse:
    authorization = await server_connections.begin_sign_in(
      gateway, provider.name
    )
    instruction = f'Open this URL to sign in: {authorization.authorization_url}'
    context.dispatch(actions.ShowText(instruction, style='ui.info'))
    self._show_authentication(context, provider.display_name, f'{instruction}.')
    await terminal.open_url(authorization.authorization_url)
    return await server_connections.wait_for_sign_in(
      gateway, authorization.authorization_id
    )

  def _choose_token_source(
    self,
    context: extension.Context,
    provider: mirror_pb2.Provider,
    resume: actions.CreateRemote | None,
  ) -> None:
    configured = context.config.get(_TOKEN_COMMANDS, {})
    if not isinstance(configured, dict) or not all(
      isinstance(command, str) for command in configured.values()
    ):
      context.dispatch(
        actions.ShowText(
          f'{_TOKEN_COMMANDS} must map provider names to commands.',
          style='ui.error',
        )
      )
      return
    cancelled = (
      actions.ShowText('Remote setup cancelled.')
      if resume is not None
      else None
    )
    paste = actions.RequestSecret(
      actions.SecretRequest(
        f'{_TOKEN}{provider.name}',
        f'Paste a {provider.display_name} token for the Mirror server',
        self.id(),
        lambda token: _SaveToken(
          provider.name, provider.display_name, token, resume
        ),
        cancel_action=cancelled,
      )
    )
    command = cast(dict[str, str], configured).get(provider.name, '').strip()
    if not command:
      context.dispatch(paste)
      return
    read = self._route(
      _ReadToken(provider.name, provider.display_name, command, resume)
    )
    context.dispatch(
      actions.OpenSelector(
        selection.Selector(
          f'{_TOKEN}{provider.name}',
          f'Connect {provider.display_name} on the Mirror server',
          (
            selection.Choice(_PASTE, 'Paste a token'),
            selection.Choice(
              _COMMAND, f'Read from `{command}`', 'Runs on this machine'
            ),
          ),
          lambda value: paste if value == _PASTE else read,
          extension_id=self.id(),
          cancel_action=cancelled,
        )
      )
    )

  async def _read_token(
    self, context: extension.Context, effect: _ReadToken
  ) -> None:
    if self._authentication is not None:
      context.dispatch(actions.ShowText(_BUSY))
      return
    with self._cancellable(
      context, effect.label, 'Reading a token on this machine.'
    ):
      try:
        token = await local_tokens.capture(shlex.split(effect.command))
      except (local_tokens.TokenCommandError, ValueError) as error:
        context.dispatch(
          actions.ShowText(
            f'Could not read a {effect.label} token from `{effect.command}`: '
            f'{error}',
            style='ui.error',
          )
        )
        return
    await self._save_token(
      context, effect.service, effect.label, token, effect.resume
    )

  async def _save_token(
    self,
    context: extension.Context,
    service: str,
    label: str,
    token: str,
    resume: actions.CreateRemote | None,
  ) -> None:
    gateway = _gateway(context)
    if gateway is None:
      context.dispatch(
        actions.ShowText('No Mirror server is configured.', style='ui.warning')
      )
      return
    token = token.strip()
    if not token:
      context.dispatch(
        actions.ShowText(f'No {label} token was entered.', style='ui.warning')
      )
      return
    try:
      created = await server_connections.create(gateway, service, token)
    except server_connections.ERRORS as error:
      _show_failure(context, label, error)
      return
    await self._connected(
      context,
      gateway,
      service,
      label,
      created.connection_id,
      created.display or created.account_identifier,
      resume,
    )

  async def _connected(
    self,
    context: extension.Context,
    gateway: gateway_transport.Gateway,
    service: str,
    label: str,
    connection_id: str,
    account: str,
    resume: actions.CreateRemote | None,
  ) -> None:
    await _prefer(gateway, service, connection_id)
    await self._refresh_gateway(context, gateway)
    context.dispatch(
      actions.ShowText(f'Connected {label} as {account} on the Mirror server.')
    )
    self._report_connections(context)
    if resume is not None:
      context.dispatch(resume)

  async def _connect_locally(
    self, context: extension.Context, service: str, reachable: bool
  ) -> None:
    catalog = {entry.name: entry for entry in mcp.load_catalog()}
    local = self._local_state()
    entry = catalog.get(service)
    label = entry.display_name if entry is not None else service
    if service not in local.configured:
      if entry is not None:
        await self._install(context, entry)
        return
      context.dispatch(
        actions.ShowText(
          f'Unknown service {service!r}. Run /connect to see what you can connect.'
          if reachable
          else f'The Mirror server is unavailable and {service!r} is not '
          'available on this machine.',
          style='ui.warning',
        )
      )
    elif local.needs_sign_in(service):
      await self._install(
        context,
        entry or mcp.CatalogServer(service, service, '', mcp.CatalogOAuth()),
      )
    elif service in self._local.connected_servers:
      context.dispatch(
        actions.ShowText(f'{label} is already connected on this machine.')
      )
    else:
      await self._reload(context)
      context.dispatch(
        actions.ShowText(f'Connected {label} on this machine.')
        if service in self._local.connected_servers
        else actions.ShowText(
          f'Could not start {label}; check {mcp.config_dir()}/{service}.json.',
          style='ui.error',
        )
      )

  async def _list(self, context: extension.Context) -> None:
    gateway = _gateway(context)
    server = await _server(gateway)
    text = services.format_rows(
      services.rows(
        server,
        self._gateway_connections.connected_services,
        self._local_state(),
      )
    )
    if gateway is not None and server is None:
      text += f'\n{_UNAVAILABLE}'
    context.dispatch(actions.ShowText(text))

  async def _disconnect(self, context: extension.Context, service: str) -> None:
    gateway = _gateway(context)
    server = await _server(gateway)
    local = self._local_state()
    names = services.display_names(server, local)
    label = names.get(service, service)
    options = _removals(server, local, names, service)
    if not options:
      context.dispatch(
        actions.ShowText(
          f'{label} is not connected. Run /connections to see connections.'
          if service
          else 'Nothing is connected. Run /connect to add a service.',
          style='ui.warning',
        )
      )
    elif (
      service and len(options) == 1 and (server is not None or gateway is None)
    ):
      [(_, removal)] = options.values()
      await self._remove(context, removal)
    else:
      if gateway is not None and server is None:
        context.dispatch(actions.ShowText(_UNAVAILABLE, style='ui.warning'))
      context.dispatch(
        actions.OpenSelector(
          selection.Selector(
            'disconnect',
            f'Disconnect {label}' if service else 'Disconnect a service',
            tuple(choice for choice, _ in options.values()),
            lambda value: self._route(options[value][1]),
            extension_id=self.id(),
          )
        )
      )

  async def _remove(
    self, context: extension.Context, removal: _Revoke | _RemoveLocal
  ) -> None:
    if isinstance(removal, _RemoveLocal):
      mcp.remove_server(removal.name)
      await self._reload(context)
      context.dispatch(
        actions.ShowText(f'Disconnected {removal.label} on this machine.')
      )
      return
    gateway = _gateway(context)
    if gateway is None:
      return
    try:
      await server_connections.revoke(gateway, removal.connection_id)
    except server_connections.ERRORS as error:
      _show_failure(context, removal.label, error, verb='disconnect')
      return
    await self._refresh_gateway(context, gateway)
    context.dispatch(
      actions.ShowText(f'Disconnected {removal.label} on the Mirror server.')
    )
    self._report_connections(context)

  @override
  async def on_remote_setup(
    self, context: extension.Context, request: actions.CreateRemote
  ) -> bool:
    required = context.config.get(_REQUIRED, [])
    if not isinstance(required, list) or not all(
      isinstance(name, str) for name in required
    ):
      context.dispatch(
        actions.ShowText(
          f'{_REQUIRED} must be a list of provider names.', style='ui.error'
        )
      )
      return False
    gateway = _gateway(context)
    if not required or gateway is None:
      return True
    server = await _server(gateway)
    if server is None:
      context.dispatch(
        actions.ShowText(
          'Could not check connections on the Mirror server; '
          'try /remote again.',
          style='ui.error',
        )
      )
      return False
    for name in cast(list[str], required):
      provider = server.provider(name)
      if provider is not None and not self._remote_ready(
        context, server, provider, request
      ):
        return False
    return True

  def _remote_ready(
    self,
    context: extension.Context,
    server: server_connections.ServerConnections,
    provider: mirror_pb2.Provider,
    request: actions.CreateRemote,
  ) -> bool:
    usable = [
      connection
      for connection in server.for_provider(provider.name)
      if server_connections.usable(connection)
    ]
    if not usable:
      context.dispatch(
        actions.ShowText(
          f'Connect {provider.display_name} before opening a remote session.',
          style='ui.warning',
        )
      )
      context.dispatch(self._route(_Connect(provider.name, request)))
      return False
    if len(usable) > 1 and not any(
      connection.is_default for connection in usable
    ):
      context.dispatch(self._default_picker(provider, usable, request))
      return False
    return True

  def _default_picker(
    self,
    provider: mirror_pb2.Provider,
    usable: Iterable[mirror_pb2.Connection],
    request: actions.CreateRemote,
  ) -> actions.Action:
    labels = {
      connection.id: f'{provider.display_name} ({connection.account_identifier})'
      for connection in usable
    }
    return actions.OpenSelector(
      selection.Selector(
        'connect-default',
        f'Choose the {provider.display_name} account for remote sessions',
        tuple(
          selection.Choice(connection_id, label)
          for connection_id, label in labels.items()
        ),
        lambda value: self._route(_SetDefault(value, labels[value], request)),
        extension_id=self.id(),
        cancel_action=actions.ShowText('Remote setup cancelled.'),
      )
    )

  async def _set_default(
    self,
    context: extension.Context,
    connection_id: str,
    label: str,
    resume: actions.CreateRemote,
  ) -> None:
    gateway = _gateway(context)
    if gateway is None:
      return
    try:
      await server_connections.set_default(gateway, connection_id)
    except server_connections.ERRORS as error:
      _show_failure(context, label, error, verb='select')
      return
    context.dispatch(actions.ShowText(f'{label} is now the default.'))
    context.dispatch(resume)

  async def _reload(self, context: extension.Context) -> None:
    connections = mcp_extensions.local_connections(
      on_auth_required=self._auth_required
    )
    await self._swap(context, self._local, connections, self._use_local)
    self._failed_servers.difference_update(connections.connected_servers)

  def _show_authentication(
    self, context: extension.Context, name: str, message: str
  ) -> None:
    context.dispatch(
      actions.OpenSelector(
        selection.Selector(
          _AUTHENTICATION,
          f'Authenticating {name}',
          (),
          lambda _: actions.ClearSelector(),
          extension_id=self.id(),
          empty=f'{message} Esc cancels.',
          cancel_action=self._route(_Cancel()),
        )
      )
    )

  def _finish_authentication(self, context: extension.Context) -> None:
    self._authentication = None
    request = context.get_state().selector.selector
    if request is not None and request.id == _AUTHENTICATION:
      context.dispatch(actions.ClearSelector())

  async def _install(
    self, context: extension.Context, server: mcp.CatalogServer
  ) -> None:
    self._authentication = asyncio.current_task()
    saved = False
    if server.oauth is not None:
      self._show_authentication(
        context, server.display_name, 'Complete sign-in in your browser.'
      )
    try:
      await mcp.install_server(
        server, show_url=lambda url: _show_url(context, url)
      )
      saved = True
      await self._reload(context)
      context.dispatch(
        actions.ShowText(f'Connected {server.display_name} on this machine.')
      )
    except asyncio.CancelledError:
      context.dispatch(
        actions.ShowText(
          f'{server.display_name} was saved, but activation was cancelled.'
          if saved
          else f'Cancelled adding {server.display_name}.',
          style='ui.warning',
        )
      )
      raise
    except (httpx2.HTTPError, TimeoutError, ValueError, OSError) as error:
      context.dispatch(
        actions.ShowText(
          f'{server.display_name} was saved, but could not be activated: {error}'
          if saved
          else f'Could not add {server.display_name}; no changes were saved: {error}',
          style='ui.error',
        )
      )
    finally:
      self._finish_authentication(context)


def _remote_mcp(context: extension.Context) -> bool:
  enabled = context.config.get(_REMOTE_MCP, False)
  if isinstance(enabled, bool):
    return enabled
  context.dispatch(
    actions.ShowText(f'{_REMOTE_MCP} must be true or false.', style='ui.error')
  )
  return False


def _gateway(context: extension.Context) -> gateway_transport.Gateway | None:
  if context.remote_endpoint is None:
    return None
  return gateway_transport.Gateway(
    context.remote_endpoint, context.remote_secure
  )


async def _server(
  gateway: gateway_transport.Gateway | None,
) -> server_connections.ServerConnections | None:
  if gateway is None:
    return None
  try:
    return await server_connections.load(gateway)
  except server_connections.ERRORS as error:
    _LOGGER.warning(
      'Mirror server connections at %r are unavailable: %s',
      gateway.endpoint,
      _describe(error),
    )
    return None


async def _prefer(
  gateway: gateway_transport.Gateway, service: str, connection_id: str
) -> None:
  try:
    server = await server_connections.load(gateway)
    usable = [
      connection
      for connection in server.for_provider(service)
      if server_connections.usable(connection)
    ]
    if len(usable) > 1 and not any(
      connection.id == connection_id and connection.is_default
      for connection in usable
    ):
      await server_connections.set_default(gateway, connection_id)
  except server_connections.ERRORS as error:
    _LOGGER.warning(
      'Could not make %r the default %s connection: %s',
      connection_id,
      service,
      _describe(error),
    )


def _removals(
  server: server_connections.ServerConnections | None,
  local: services.Local,
  names: Mapping[str, str],
  service: str,
) -> dict[str, tuple[selection.Choice, _Revoke | _RemoveLocal]]:
  options: dict[str, tuple[selection.Choice, _Revoke | _RemoveLocal]] = {}
  for connection in server.connections if server is not None else ():
    if service and connection.provider != service:
      continue
    label = names.get(connection.provider, connection.provider)
    options[connection.id] = (
      selection.Choice(
        connection.id,
        f'{label} · {connection.account_identifier}',
        'Mirror server' + (' · default' if connection.is_default else ''),
        section='Mirror server',
      ),
      _Revoke(connection.id, f'{label} ({connection.account_identifier})'),
    )
  for name in local.configured:
    if service and name != service:
      continue
    label = names.get(name, name)
    options[f'{_LOCAL}{name}'] = (
      selection.Choice(
        f'{_LOCAL}{name}', label, 'this machine', section='This machine'
      ),
      _RemoveLocal(name, label),
    )
  return options


def _commands(services_needing_sign_in: Iterable[str]) -> str:
  return ', '.join(f'/connect {name}' for name in services_needing_sign_in)


def _custom_help() -> str:
  return '\n'.join((
    'Custom MCP servers stay on this machine. Save one JSON file per server '
    f'as {mcp.config_dir()}/<name>.json, for example:',
    '  {"url":"https://example.com/mcp","auth":{"type":"oauth"}}',
    '  {"url":"https://example.com/mcp","auth":{"type":"bearer","tokenEnvVar":"EXAMPLE_TOKEN"}}',
    '  {"command":"npx","args":["-y","example-mcp"],"env":{"API_KEY":"EXAMPLE_API_KEY"}}',
    'Then run /connect <name>.',
  ))


def _describe(error: BaseException) -> str:
  if isinstance(error, grpc.aio.AioRpcError):
    return error.details() or error.code().name
  return str(error)


def _show_failure(
  context: extension.Context,
  label: str,
  error: BaseException,
  *,
  verb: str = 'connect',
) -> None:
  context.dispatch(
    actions.ShowText(
      f'Could not {verb} {label}: {_describe(error)}', style='ui.error'
    )
  )


async def _show_url(context: extension.Context, url: str) -> None:
  context.dispatch(
    actions.ShowText(f'Open this URL to authenticate:\n{url}', style='ui.info')
  )
  await terminal.open_url(url)
