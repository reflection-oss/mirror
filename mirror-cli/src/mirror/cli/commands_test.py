"""Dispatch telemetry includes the real extension fault boundary."""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator
from typing import override
from unittest import mock

import pytest
import pytest_asyncio

from mirror.cli import actions, commands, extension, workspace_controller
from mirror.cli.statsig.events import client, event


@pytest.fixture
def emitted_events(monkeypatch: pytest.MonkeyPatch) -> list[event.Event]:
  # The client has no transport hook; capture its public emission boundary
  # without enabling collection, constructing an identity, or opening a socket.
  emitted: list[event.Event] = []
  monkeypatch.setattr(client.get(), 'log_event', emitted.append)
  return emitted


def _context(*, busy: bool = False) -> commands.CommandContext:
  return commands.CommandContext('model', 'low', 'conversation', busy, '/test')


class _Contribution(extension.Extension[None, None, None]):
  def __init__(self) -> None:
    self.result: actions.Action | None = actions.ShowText('synthetic result')
    self.calls: list[str] = []

  @classmethod
  @override
  def id(cls) -> str:
    return 'synthetic/private-extension-name'

  @override
  def slash_commands(self) -> tuple[commands.CommandDef, ...]:
    return (
      commands.CommandDef(
        'quit',
        'Override a builtin',
        self._handle,
        id=commands.CommandId.QUIT,
        arguments=commands.RequiredArguments('<synthetic argument>'),
      ),
    )

  def _handle(
    self, context: commands.CommandContext, arguments: str
  ) -> actions.Action | None:
    del context
    self.calls.append(arguments)
    if arguments == 'raise':
      raise ValueError('synthetic failure')
    return self.result


@pytest.fixture
def contribution() -> _Contribution:
  return _Contribution()


@pytest_asyncio.fixture
async def host(
  contribution: _Contribution, tmp_path: pathlib.Path
) -> AsyncIterator[extension.ExtensionHost]:
  registry = extension.ExtensionRegistry()
  await registry.register(contribution, None)
  instance = registry.create_host(
    extension.Context(
      workspace=extension.Workspace(tmp_path, remote=False),
      dispatch=lambda _: None,
      get_state=lambda: pytest.fail('Unexpected state access'),
      enqueue=lambda _: pytest.fail('Unexpected agent message'),
      refresh_tools=registry.refresh_tools,
      call_tool=lambda name, arguments: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      create_agent=lambda options, on_message, on_error: pytest.fail(
        'Unexpected agent creation'
      ),
      builtins=tuple,
      extension_tools=tuple,
      # Constructing a real controller discovers worktrees and creates agents.
      workspace_controller=mock.Mock(
        spec_set=workspace_controller.WorkspaceController
      ),
    )
  )
  try:
    await instance.start()
    yield instance
  finally:
    await instance.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('text', 'result', 'command_id'),
  [
    ('/exit', actions.QuitApp(), 'quit'),
    (
      '/rename synthetic-private-title',
      actions.RenameWorktree('synthetic-private-title'),
      'rename',
    ),
    (
      '/rename',
      actions.ShowText('Usage: /rename <alias>', style='ui.warning'),
      'rename',
    ),
  ],
)
async def test_dispatch_builtin_normal_return(
  host: extension.ExtensionHost,
  emitted_events: list[event.Event],
  text: str,
  result: actions.Action,
  command_id: str,
) -> None:
  available = commands.resolve_commands((
    *commands.COMMANDS,
    *host.slash_commands,
  ))
  assert commands.dispatch(text, _context(), available) == result
  assert emitted_events == [
    event.Event(
      'mirror_slash_command_dispatched',
      value=1,
      metadata={'command': command_id},
    )
  ]


def test_dispatch_normal_return_without_action(
  emitted_events: list[event.Event],
) -> None:
  command = commands.CommandDef(
    'synthetic-noop', 'No action', lambda context, arguments: None
  )
  assert commands.dispatch('/synthetic-noop', _context(), (command,)) is None
  assert emitted_events == [
    event.Event(
      'mirror_slash_command_dispatched', value=1, metadata={'command': 'custom'}
    )
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('text', 'busy'),
  [
    ('/synthetic/path', False),
    ('/', False),
    ('/synthetic-unknown', False),
    ('/status synthetic-private-argument', False),
    ('/quit synthetic-private-argument', True),
  ],
)
async def test_dispatch_rejected(
  host: extension.ExtensionHost,
  contribution: _Contribution,
  emitted_events: list[event.Event],
  text: str,
  busy: bool,
) -> None:
  available = commands.resolve_commands((
    *commands.COMMANDS,
    *host.slash_commands,
  ))
  commands.dispatch(text, _context(busy=busy), available)
  assert contribution.calls == []
  assert emitted_events == []


@pytest.mark.asyncio
@pytest.mark.parametrize('result', [actions.ShowText('synthetic result'), None])
async def test_dispatch_contributed_override_normal_return(
  host: extension.ExtensionHost,
  contribution: _Contribution,
  emitted_events: list[event.Event],
  result: actions.Action | None,
) -> None:
  contribution.result = result
  available = commands.resolve_commands((
    *commands.COMMANDS,
    *host.slash_commands,
  ))
  assert (
    commands.dispatch('/quit synthetic-private-argument', _context(), available)
    == result
  )
  assert contribution.calls == ['synthetic-private-argument']
  assert emitted_events == [
    event.Event(
      'mirror_slash_command_dispatched', value=1, metadata={'command': 'custom'}
    )
  ]


@pytest.mark.asyncio
async def test_dispatch_throwing_and_faulted_contribution(
  host: extension.ExtensionHost,
  contribution: _Contribution,
  emitted_events: list[event.Event],
) -> None:
  available = commands.resolve_commands((
    *commands.COMMANDS,
    *host.slash_commands,
  ))
  assert commands.dispatch(
    '/quit raise', _context(), available
  ) == actions.ShowText(
    f'Extension {contribution.id()} disabled: ValueError: synthetic failure. '
    'Fix and reload.',
    style='ui.error',
  )
  assert emitted_events == []
  assert commands.dispatch(
    '/quit retry', _context(), available
  ) == actions.ShowText(
    f'Extension {contribution.id()} is unavailable. Fix and reload.'
  )
  assert contribution.calls == ['raise']
  assert emitted_events == []


@pytest.mark.asyncio
async def test_dispatch_inactive_contribution(
  host: extension.ExtensionHost,
  contribution: _Contribution,
  emitted_events: list[event.Event],
) -> None:
  available = commands.resolve_commands((
    *commands.COMMANDS,
    *host.slash_commands,
  ))
  await host.close()
  assert commands.dispatch(
    '/quit unused', _context(), available
  ) == actions.ShowText(
    f'Extension {contribution.id()} is unavailable. Fix and reload.'
  )
  assert contribution.calls == []
  assert emitted_events == []


def test_folder_and_close_commands(emitted_events: list[event.Event]) -> None:
  del emitted_events
  assert commands.dispatch(
    '/folder ../notes', _context()
  ) == actions.OpenFolder('../notes')
  assert commands.dispatch('/close', _context()) == actions.CloseWorkspace()
