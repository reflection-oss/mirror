from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import subprocess
from typing import cast
from unittest import mock

import pytest
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls

from mirror.cli import (
  actions,
  extension,
  extension_state,
  shell,
  state,
)
from mirror.cli import (
  effects as _effects,
)
from mirror.cli.extensions.repository_status import (
  extension as repository_status,
)


def _git(cwd: pathlib.Path, *arguments: str) -> str:
  return subprocess.run(
    ('git', '-C', str(cwd), *arguments),
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _repository(path: pathlib.Path) -> pathlib.Path:
  path.mkdir()
  _git(path, 'init', '--initial-branch=main')
  _git(path, 'config', 'user.email', 'mirror@example.com')
  _git(path, 'config', 'user.name', 'Mirror Test')
  (path / 'tracked.txt').write_text('base\n')
  _git(path, 'add', 'tracked.txt')
  _git(path, 'commit', '-m', 'base')
  return path


def _application(
  extension_states: extension_state.ExtensionStates,
) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    extensions=extension_states,
  )


def _fragments(value: containers.AnyContainer) -> StyleAndTextTuples:
  window = containers.to_container(value)
  assert isinstance(window, containers.Window)
  assert isinstance(window.content, controls.FormattedTextControl)
  assert not callable(window.content.text)
  return cast(StyleAndTextTuples, window.content.text)


async def _observe_repository_status(
  workspace: pathlib.Path,
  backend: shell.Shell | None = None,
) -> repository_status.RepositoryStatus:
  current: state.State
  changed = asyncio.Event()

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    extension_states, effects = host.reduce(current, action)
    current = dataclasses.replace(current, extensions=extension_states)
    for effect in effects:
      assert isinstance(effect, _effects.RunExtensionEffect)
      host.handle_effect(effect)
    if isinstance(action, actions.RoutedExtensionAction):
      changed.set()

  context = extension.Context(
    extension.Workspace(
      workspace, remote=False, shell=backend or shell.Shell(workspace)
    ),
    dispatch=dispatch,
    get_state=lambda: current,
    enqueue=lambda _message: None,
    refresh_tools=lambda name: registry.refresh_tools(name),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(repository_status.RepositoryStatusExtension(), None)
  host = registry.create_host(context)
  current = _application(host.initial_states)
  await host.start()
  await asyncio.wait_for(changed.wait(), timeout=2)
  await host.close()
  observed = current.extensions['core/status']
  assert isinstance(observed, repository_status.RepositoryStatus)
  return observed


@pytest.mark.asyncio
async def test_repository_status_shows_git_branch_and_changes(
  tmp_path: pathlib.Path,
) -> None:
  repository = _repository(tmp_path / 'repository')
  (repository / 'tracked.txt').write_text('changed\n')
  (repository / 'new.txt').write_text('new\n')

  observed = await _observe_repository_status(repository)
  assert observed == repository_status.RepositoryStatus('git', 'main', 1, 1, 1)
  rendered = repository_status.RepositoryStatusExtension().render(
    _application(extension_state.ExtensionStates()), observed
  )
  assert isinstance(rendered, containers.VSplit)
  assert (
    ''.join(text for _, text in _fragments(rendered.children[-1]))
    == '1 change · +1/-1 '
  )


@pytest.mark.asyncio
async def test_repository_status_reads_bound_workspace_not_frontend_path(
  tmp_path: pathlib.Path,
) -> None:
  host = tmp_path / 'host'
  host.mkdir()
  target = _repository(tmp_path / 'target')
  observed = await _observe_repository_status(host, shell.Shell(target))
  assert observed == repository_status.RepositoryStatus('git', 'main', 0, 0, 0)


@pytest.mark.asyncio
async def test_repository_status_without_backend_fails_closed(
  tmp_path: pathlib.Path,
) -> None:
  repository = _repository(tmp_path / 'host')
  plugin = repository_status.RepositoryStatusExtension()
  current = _application(
    extension_state.ExtensionStates({plugin.id(): plugin.initial_state()})
  )
  context = extension.Context(
    extension.Workspace(repository, remote=True),
    dispatch=lambda _: pytest.fail('Unexpected host repository status'),
    get_state=lambda: current,
    enqueue=lambda _: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  _, emitted = plugin.reduce(current, actions.ExtensionLoaded(plugin.id()))
  assert len(emitted) == 1
  effect = emitted[0]
  assert isinstance(effect, extension.PrivateEffect)

  with pytest.raises(
    RuntimeError, match='Workspace process execution is unavailable'
  ):
    await plugin.handle_effect(context, effect.payload)


@pytest.mark.asyncio
async def test_replay_preserves_live_repository_status(
  tmp_path: pathlib.Path,
) -> None:
  observed = repository_status.RepositoryStatus('jj', 'revision', 4, 2, 1)
  current = _application(
    extension_state.ExtensionStates({'core/status': observed})
  )
  registry = extension.ExtensionRegistry()
  await registry.register(repository_status.RepositoryStatusExtension(), None)
  host = registry.create_host(
    extension.Context(
      extension.Workspace(tmp_path, remote=False, shell=shell.Shell(tmp_path)),
      dispatch=lambda _: None,
      get_state=lambda: current,
      enqueue=lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  restored, effects = host.reduce(current, actions.HydrateExtensions(()))
  assert restored['core/status'] == observed
  assert effects == ()
  await host.close()
  await registry.close()


@pytest.mark.asyncio
async def test_repository_status_handles_unborn_branch(
  tmp_path: pathlib.Path,
) -> None:
  _git(tmp_path, 'init', '--initial-branch=main')
  assert await _observe_repository_status(
    tmp_path
  ) == repository_status.RepositoryStatus('git', 'main', 0, 0, 0)
  (tmp_path / 'staged.txt').write_text('staged\n')
  _git(tmp_path, 'add', 'staged.txt')
  (tmp_path / 'staged.txt').write_text('staged\nunstaged\n')
  (tmp_path / 'untracked').mkdir()
  (tmp_path / 'untracked' / 'first.txt').write_text('untracked\n')
  (tmp_path / 'untracked' / 'second.txt').write_text('untracked\n')
  assert await _observe_repository_status(
    tmp_path
  ) == repository_status.RepositoryStatus('git', 'main', 2, 0, 1)


@pytest.mark.asyncio
async def test_repository_status_shows_jj_change_id_from_stale_workspace(
  tmp_path: pathlib.Path,
) -> None:
  workspace = tmp_path / 'workspace'
  workspace.mkdir()
  (workspace / '.jj').mkdir()
  binary = tmp_path / 'jj'
  binary.write_text(
    '#!/bin/sh\n'
    'printf "%s\\n" "$*" >> "$CALLS"\n'
    'case "$*" in\n'
    '  *--ignore-working-copy*) printf "%s\\0" rtrmtuvm bookmark 2 3; printf 1;;\n'
    '  *) exit 1;;\n'
    'esac\n'
  )
  binary.chmod(0o755)
  calls = tmp_path / 'calls'
  backend = shell.Shell(
    workspace,
    environment={
      'PATH': f'{tmp_path}:/usr/bin:/bin',
      'CALLS': str(calls),
    },
  )
  assert await _observe_repository_status(workspace, backend) == (
    repository_status.RepositoryStatus('jj', 'rtrmtuvm', 3, 1, 2, ('bookmark',))
  )
  first, second = calls.read_text().splitlines()
  assert '--ignore-working-copy' not in first
  assert '--ignore-working-copy' in second
  assert 'self.diff().files().len()' in second
  assert 'self.diff().stat().total_added()' in second
  assert 'self.diff().stat().total_removed()' in second


@pytest.mark.asyncio
async def test_worktree_activation_keeps_last_status(
  tmp_path: pathlib.Path,
) -> None:
  current: state.State
  context = extension.Context(
    extension.Workspace(tmp_path, remote=False, shell=shell.Shell(tmp_path)),
    dispatch=lambda _: None,
    get_state=lambda: current,
    enqueue=lambda _message: None,
    refresh_tools=lambda name: registry.refresh_tools(name),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(repository_status.RepositoryStatusExtension(), None)
  host = registry.create_host(context)
  status_extension = repository_status.RepositoryStatusExtension()
  current = _application(host.initial_states)
  loading = status_extension.initial_state()

  rendered = containers.to_container(status_extension.render(current, loading))
  assert isinstance(rendered, containers.Window)
  assert rendered.preferred_height(80, 100).preferred == 1
  assert _fragments(rendered) == [('class:footer', '   loading repository… ')]

  observed = repository_status.RepositoryStatus(
    'jj', 'rtrmtuvm', 3, 1, 2, ('feature',)
  )
  rendered = containers.to_container(status_extension.render(current, observed))
  assert isinstance(rendered, containers.VSplit)
  assert _fragments(rendered.children[0]) == [
    ('class:footer', '   rtrmtuvm feature ')
  ]
  assert _fragments(rendered.children[2]) == [
    ('class:footer', '2 changes · '),
    ('class:ui.success', '+3'),
    ('class:footer', '/'),
    ('class:ui.error', '-1 '),
  ]

  current = dataclasses.replace(
    current,
    extensions=current.extensions.updated(status_extension.id(), observed),
  )
  active_states, effects = host.reduce(current, actions.SetActive(value=True))
  assert active_states == current.extensions
  assert effects == ()

  refreshed = dataclasses.replace(observed, changes=3)
  current = dataclasses.replace(current, extensions=active_states)
  refreshed_states, effects = host.reduce(
    current,
    actions.RoutedExtensionAction(
      status_extension.id(), repository_status.StatusChanged(refreshed)
    ),
  )
  assert refreshed_states[status_extension.id()] == refreshed
  assert effects == ()


@pytest.mark.asyncio
async def test_mutating_tools_trigger_one_debounced_refresh_and_deactivation_stops_it(
  tmp_path: pathlib.Path,
) -> None:
  (tmp_path / '.jj').mkdir()
  calls = tmp_path / 'calls'
  calls.write_text('0')
  binary = tmp_path / 'jj'
  binary.write_text(
    '#!/bin/sh\n'
    'count=$(cat "$CALLS")\n'
    'count=$((count + 1))\n'
    'printf %s "$count" > "$CALLS"\n'
    'printf "%s\\0" revision "" "$count" 0; printf 0\n'
  )
  binary.chmod(0o755)
  backend = shell.Shell(
    tmp_path,
    environment={
      'PATH': f'{tmp_path}:/usr/bin:/bin',
      'CALLS': str(calls),
    },
  )
  changed = asyncio.Event()
  current: state.State

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    extension_states, effects = host.reduce(current, action)
    current = dataclasses.replace(current, extensions=extension_states)
    for effect in effects:
      assert isinstance(effect, _effects.RunExtensionEffect)
      host.handle_effect(effect)
    if isinstance(action, actions.RoutedExtensionAction):
      changed.set()

  context = extension.Context(
    extension.Workspace(tmp_path, remote=False, shell=backend),
    dispatch=dispatch,
    get_state=lambda: current,
    enqueue=lambda _message: None,
    refresh_tools=lambda name: registry.refresh_tools(name),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(repository_status.RepositoryStatusExtension(), None)
  host = registry.create_host(context)
  current = _application(host.initial_states)

  await host.start()
  await asyncio.wait_for(changed.wait(), timeout=2)
  assert calls.read_text() == '1'
  changed.clear()

  current = dataclasses.replace(
    current,
    stream=state.StreamState(
      running=(actions.RunningToolCall('edit', 'edit', '', 'tool', '{}'),)
    ),
  )
  _, effects = host.reduce(current, actions.FinishRunningToolCall('edit'))
  assert len(effects) == 1
  refresh = effects[0]
  assert isinstance(refresh, _effects.RunExtensionEffect)

  await host.handle_effect(refresh)
  await host.handle_effect(refresh)
  await asyncio.wait_for(changed.wait(), timeout=2)
  await asyncio.sleep(0.05)
  assert calls.read_text() == '2'

  await host.close()
  await host.handle_effect(refresh)
  await asyncio.sleep(0.05)
  assert calls.read_text() == '2'


@pytest.mark.asyncio
async def test_extension_loads_without_a_location_declaration(
  tmp_path: pathlib.Path,
) -> None:
  context = extension.Context(
    extension.Workspace(tmp_path, remote=True),
    dispatch=lambda _: None,
    get_state=lambda: _application(extension_state.ExtensionStates()),
    enqueue=lambda _message: None,
    refresh_tools=lambda name: registry.refresh_tools(name),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  entrypoint = pathlib.Path(cast(str, repository_status.__file__)).parent

  assert await registry.load(entrypoint) == 'core/status'
  host = registry.create_host(context)
  assert 'core/status' in host.initial_states
  await host.close()
  await registry.close()
