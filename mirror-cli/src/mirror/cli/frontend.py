"""Frontend-wide extension commands and explicit daemon handoffs."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Sequence
from typing import TextIO

from prompt_toolkit import application

from mirror.cli import actions, commands, daemon_client


@dataclasses.dataclass(frozen=True)
class Connection:
  """One SSH connection; its owner closes only the tunnel, never the daemon."""

  id: str
  host: str
  address: str
  daemon_id: str
  close: Callable[[], Awaitable[None]] = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class Change:
  """A requested handoff, with None selecting the local daemon."""

  connection: Connection | None


@dataclasses.dataclass(frozen=True)
class Command:
  """A command that operates on the frontend using the ordinary UI actions."""

  name: str
  description: str
  run: Callable[[Context, str], Awaitable[actions.Action | None]]
  arguments: commands.ArgumentSpec | None = None

  def definition(self) -> commands.CommandDef:
    return commands.CommandDef(
      self.name,
      self.description,
      lambda _context, arguments: actions.RunFrontendCommand(
        self.name, arguments
      ),
      arguments=self.arguments,
      available_when_busy=True,
      shortcut_action=actions.RunFrontendCommand(self.name),
    )


@dataclasses.dataclass
class Context:
  """Services for frontend extensions; switching transfers tunnel ownership."""

  connection: Connection | None
  metadata: daemon_client.Client
  terminal: TextIO
  switch: Callable[[Change], None]
  commands: Callable[[], Sequence[Command]]
  statuses: Callable[[Context], Sequence[str]]
  _running: bool = dataclasses.field(default=False, init=False)

  async def run(
    self,
    name: str,
    arguments: str = '',
    *,
    dispatch: Callable[[actions.Action], object],
  ) -> None:
    if self._running:
      return
    command = next(
      (item for item in self.commands() if item.name == name), None
    )
    if command is None:
      raise ValueError(f'Frontend command unavailable: /{name}')
    self._running = True
    try:
      try:
        action = await command.run(self, arguments)
      except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        message = str(error) or (
          f'/{name} timed out.'
          if isinstance(error, TimeoutError)
          else f'/{name} failed.'
        )
        action = actions.ShowText(message, style='ui.error')
      if action is not None:
        dispatch(action)
    finally:
      self._running = False

  def status(self) -> str:
    """Render only local connection metadata, never remote diagnostics."""
    return ' · '.join(self.statuses(self))

  def start(
    self,
    name: str,
    arguments: str = '',
    *,
    dispatch: Callable[[actions.Action], object],
  ) -> asyncio.Task[None]:
    """Run a command and route its result back through the invoking view's UI."""
    return application.get_app().create_background_task(
      self.run(name, arguments, dispatch=dispatch)
    )
