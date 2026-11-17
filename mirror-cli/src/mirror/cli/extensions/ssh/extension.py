"""Connect this frontend to a devserver daemon with /ssh."""

from __future__ import annotations

import dataclasses
import re
import uuid
from collections.abc import Sequence
from typing import override

from prompt_toolkit import application
from prompt_toolkit.layout import containers, controls

from mirror import config
from mirror.cli import actions, commands, extension, frontend, selection, state
from mirror.cli.extensions.ssh import transport


@dataclasses.dataclass(frozen=True)
class SavedConnection:
  """A stable local UI scope, independent of daemon and tunnel lifetimes."""

  id: str
  host: str


def _select(value: str) -> actions.Action:
  if value == 'new':
    return actions.RestorePrompt('/ssh ')
  return actions.RunFrontendCommand(
    'ssh', '--local' if value == 'local' else f'--saved {value}'
  )


@dataclasses.dataclass(frozen=True)
class _State:
  pass


class SshExtension(extension.Extension[_State, None, None]):
  """Own saved connections, connection menus and the persistent host indicator."""

  def __init__(self, settings: config.Config | None = None) -> None:
    self._settings = settings or config.Config(config.default_path())
    self._connecting = ''
    self._failed = ''

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/ssh'

  @staticmethod
  @override
  def initial_state() -> _State:
    return _State()

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_RIGHT

  @override
  def render_order(self) -> int:
    return 2

  @override
  def cache_key(
    self, global_state: state.State, extension_state: _State
  ) -> object:
    del global_state, extension_state
    return (
      self.frontend_status(self.frontend_context)
      if self.frontend_context is not None
      else ''
    )

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    del global_state, extension_state
    if self.frontend_context is None:
      return None
    status = self.frontend_status(self.frontend_context)
    if not status:
      return None
    return containers.Window(
      controls.FormattedTextControl([('class:footer', f' {status}')]),
      dont_extend_height=True,
      dont_extend_width=True,
    )

  @override
  def default_config(self) -> config.JsonObject:
    return {'connections': []}

  @override
  def frontend_commands(self) -> Sequence[frontend.Command]:
    return (
      frontend.Command(
        'ssh',
        'Connect to a devserver or return to the local daemon',
        self.open,
        arguments=commands.OptionalArguments('[user@host or SSH alias]'),
      ),
    )

  @override
  def frontend_status(self, context: frontend.Context) -> str:
    if self._connecting:
      return f'SSH · {self._connecting} · connecting…'
    location = ''
    if context.connection is not None:
      location = f'SSH · {context.connection.host}'
      if not context.metadata.connected:
        location += ' · disconnected'
    if self._failed:
      location += (' · ' if location else '') + f'SSH to {self._failed} failed'
    return location

  def configuration_values(self) -> config.JsonObject:
    """Read the ordinary bundled-extension settings without copying credentials."""
    overrides = config.extension_overrides(
      self._settings.read().get('extensions'), 'builtin/ssh'
    )
    return config.resolve_extension(overrides, self.default_config()).config

  def connections(self) -> tuple[SavedConnection, ...]:
    """Validate non-secret connection identities before showing or using them."""
    values = self.configuration_values().get('connections')
    if not isinstance(values, list):
      raise ValueError('SSH connections must be a list')
    result: list[SavedConnection] = []
    for value in values:
      if (
        not isinstance(value, dict)
        or not isinstance(identity := value.get('id'), str)
        or not re.fullmatch('[0-9a-f]{32}', identity)
        or not isinstance(host := value.get('host'), str)
      ):
        raise ValueError('Invalid saved SSH connection')
      result.append(SavedConnection(identity, transport.host_name(host)))
    if len({item.id for item in result}) != len(result):
      raise ValueError('Duplicate saved SSH connection identity')
    return tuple(result)

  def picker(self, context: frontend.Context) -> actions.OpenSelector:
    """Use the same searchable selector as the other bundled commands."""
    choices = tuple(
      selection.Choice(item.id, item.host) for item in self.connections()
    ) + (selection.Choice('new', 'New connection…'),)
    if context.connection:
      choices += (selection.Choice('local', 'Return to local daemon'),)
    return actions.OpenSelector(
      selection.Selector(
        'ssh',
        'SSH connections',
        choices,
        _select,
        extension_id=self.id(),
        current=context.connection.id if context.connection else None,
        pinned_count=2 if context.connection else 1,
      )
    )

  async def open(
    self, context: frontend.Context, arguments: str
  ) -> actions.Action | None:
    """Open the selector, or connect to a host entered in the normal composer."""
    if not arguments:
      return self.picker(context)
    if arguments == '--local':
      context.switch(frontend.Change(None))
      return None
    saved = self.connections()
    if arguments.startswith('--saved '):
      identity = arguments.removeprefix('--saved ')
      target = next((item for item in saved if item.id == identity), None)
      if target is None:
        raise ValueError('The selected SSH connection is no longer available.')
    else:
      host = transport.host_name(arguments.strip())
      target = next(
        (item for item in saved if item.host == host),
        SavedConnection(uuid.uuid4().hex, host),
      )
    connected: frontend.Connection | None = None
    self._connecting = target.host
    self._failed = ''
    application.get_app().invalidate()
    try:
      # Authentication and the remote launcher's progress use the terminal.
      async with application.in_terminal():
        connected = await transport.connect(context, target.id, target.host)
      if target not in saved:
        self._settings.set_extension_config(
          'builtin/ssh',
          {
            'connections': [
              {'id': item.id, 'host': item.host} for item in (*saved, target)
            ]
          },
        )
      context.switch(frontend.Change(connected))
    except BaseException:
      self._failed = target.host
      if connected is not None:
        await connected.close()
      raise
    finally:
      self._connecting = ''
      application.get_app().invalidate()
