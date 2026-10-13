from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import sys
from collections.abc import Mapping
from unittest import mock

import pytest

from mirror.cli import actions, commands, extension, state
from mirror.cli.extensions.memory import extension as memory


@dataclasses.dataclass
class _Lifecycle:
  """The extension lifecycle a context carries, pointed at a scratch home."""

  root: pathlib.Path
  registry: extension.ExtensionRegistry

  async def set_enabled(
    self, directory: pathlib.Path, enabled: bool
  ) -> str | None:
    del directory, enabled
    return None

  async def reload(self, extension_id: str) -> str | None:
    del extension_id
    return None


@dataclasses.dataclass
class _Harness:
  context: extension.Context
  dispatched: list[actions.Action]
  restarts: list[Mapping[str, object]]
  settings: pathlib.Path


def _harness(tmp_path: pathlib.Path) -> _Harness:
  dispatched: list[actions.Action] = []
  restarts: list[Mapping[str, object]] = []

  async def restart(updates: Mapping[str, object]) -> bool:
    restarts.append(updates)
    return True

  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
  )
  context = extension.Context(
    extension.Workspace(tmp_path / 'project', remote=False),
    dispatch=dispatched.append,
    get_state=lambda: current,
    enqueue=lambda _message: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    restart_worker=restart,
    worker_command=(sys.executable,),
    worker_cwd=tmp_path,
    extensions=_Lifecycle(
      tmp_path / '.mirror' / 'extensions', extension.ExtensionRegistry()
    ),
  )
  return _Harness(
    context, dispatched, restarts, tmp_path / '.mirror' / 'config.json'
  )


def _command_context() -> commands.CommandContext:
  return commands.CommandContext(
    model='gpt-test',
    reasoning_effort='medium',
    conversation_id='conv-abc',
    busy=False,
    cwd='/tmp/project',
  )


def _available() -> tuple[commands.CommandDef, ...]:
  return (*commands.COMMANDS, *memory.MemoryExtension().slash_commands())


def test_memory_command_routes_its_arguments_to_the_extension() -> None:
  routed = actions.RoutedExtensionAction(
    memory.MemoryExtension.id(), memory.Run(('status',))
  )
  assert (
    commands.dispatch('/memory', _command_context(), _available()) == routed
  )
  assert (
    commands.dispatch('/memory status', _command_context(), _available())
    == routed
  )
  assert commands.dispatch(
    '/memory import', _command_context(), _available()
  ) == actions.RoutedExtensionAction(
    memory.MemoryExtension.id(), memory.Run(('import',))
  )
  assert commands.dispatch(
    '/memory import codex=~/.codex/memories', _command_context(), _available()
  ) == actions.RoutedExtensionAction(
    memory.MemoryExtension.id(),
    memory.Run(('import', 'codex=~/.codex/memories')),
  )
  usage = commands.dispatch('/memory maybe', _command_context(), _available())
  assert isinstance(usage, actions.ShowText)
  assert usage.text == 'Usage: /memory [status | import [sources…]]'
  assert (
    commands.dispatch('/memory on', _command_context(), _available()) == usage
  )
  assert (
    commands.dispatch('/memory off', _command_context(), _available()) == usage
  )
  extra = commands.dispatch(
    '/memory status please', _command_context(), _available()
  )
  assert isinstance(extra, actions.ShowText)


def test_memory_extension_starts_off() -> None:
  plugin = memory.MemoryExtension()
  assert plugin.default_enabled() is True
  assert plugin.default_config() == {'enabled': False}


def test_memory_is_off_unless_the_setting_is_the_boolean_true() -> None:
  assert not memory.enabled({})
  assert not memory.enabled({'memory': True})
  assert not memory.enabled({
    'extensions': {'builtin/memory': {'config': {'enabled': 'yes'}}}
  })
  assert not memory.enabled({
    'extensions': {'builtin/memory': {'config': {'enabled': 1}}}
  })
  assert not memory.enabled({
    'extensions': {
      'builtin/memory': {'enabled': False, 'config': {'enabled': True}}
    }
  })
  assert memory.enabled({
    'extensions': {'builtin/memory': {'config': {'enabled': True}}}
  })
  assert memory.worker_config({}) == {'memory_dir': None}
  assert memory.worker_config({
    'extensions': {'builtin/memory': {'config': {'enabled': True}}}
  }) == {'memory_dir': str(memory.STORE)}


@pytest.mark.asyncio
async def test_set_enabled_persists_the_setting_and_restarts_the_worker(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)

  assert await memory.set_enabled(harness.context, True)

  assert json.loads(harness.settings.read_text()) == {
    'extensions': {'builtin/memory': {'config': {'enabled': True}}}
  }
  assert harness.restarts == [{'memory_dir': str(memory.STORE)}]
  assert harness.dispatched == [
    actions.ExtensionConfigChanged('core/memory', {'enabled': True})
  ]

  assert await memory.set_enabled(harness.context, False)

  assert json.loads(harness.settings.read_text()) == {
    'extensions': {'builtin/memory': {'config': {'enabled': False}}}
  }
  assert harness.restarts == [
    {'memory_dir': str(memory.STORE)},
    {'memory_dir': None},
  ]
  assert harness.dispatched == [
    actions.ExtensionConfigChanged('core/memory', {'enabled': True}),
    actions.ExtensionConfigChanged('core/memory', {'enabled': False}),
  ]


@pytest.mark.asyncio
async def test_status_reads_the_saved_setting(tmp_path: pathlib.Path) -> None:
  harness = _harness(tmp_path)
  plugin = memory.MemoryExtension()

  await plugin.handle_effect(harness.context, memory.Run(('status',)))
  assert harness.dispatched == [
    actions.ShowText('◆ memory off', style='ui.info')
  ]

  harness.settings.parent.mkdir(parents=True)
  harness.settings.write_text(
    json.dumps({
      'extensions': {'builtin/memory': {'config': {'enabled': True}}}
    })
  )
  await plugin.handle_effect(harness.context, memory.Run(('status',)))
  assert harness.dispatched[-1] == actions.ShowText(
    '◆ memory on', style='ui.info'
  )
  assert harness.restarts == []


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', [False, True])
async def test_import_uses_host_worker_without_workspace_execution(
  tmp_path: pathlib.Path, remote: bool
) -> None:
  harness = _harness(tmp_path)
  worker = tmp_path / 'worker.py'
  worker.write_text(
    'import pathlib, sys\n'
    'assert sys.argv[1:4] == ["memory", "import", "--store"]\n'
    'assert sys.argv[5:] == ["notes=notes"]\n'
    'print(pathlib.Path("notes").read_text())\n'
  )
  (tmp_path / 'notes').write_text('host memories imported')
  context = dataclasses.replace(
    harness.context,
    workspace=extension.Workspace(tmp_path / 'guest', remote=remote),
    worker_command=(sys.executable, str(worker)),
    worker_cwd=tmp_path,
  )

  await memory.MemoryExtension().handle_effect(
    context, memory.Run(('import', 'notes=notes'))
  )

  assert harness.dispatched == [
    actions.ShowText('◆ host memories imported', style='ui.info')
  ]
  assert not (tmp_path / 'guest').exists()


@pytest.mark.asyncio
async def test_import_without_host_worker_reports_unavailable(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)

  await memory.MemoryExtension().handle_effect(
    dataclasses.replace(harness.context, worker_command=()),
    memory.Run(('import',)),
  )

  assert harness.dispatched == [
    actions.ShowText(
      'Memory is unavailable in this workspace.', style='ui.warning'
    )
  ]


@pytest.mark.asyncio
async def test_import_reports_host_worker_failure(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  context = dataclasses.replace(
    harness.context,
    worker_command=(
      sys.executable,
      '-c',
      'print("import rejected"); raise SystemExit(2)',
    ),
    worker_cwd=tmp_path,
  )

  await memory.MemoryExtension().handle_effect(context, memory.Run(('import',)))

  assert harness.dispatched == [
    actions.ShowText('import rejected', style='ui.error')
  ]
