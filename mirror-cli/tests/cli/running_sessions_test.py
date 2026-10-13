"""Running-session telemetry through real dispatch, roster, and consent changes."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
from collections.abc import AsyncIterator
from typing import override

import pytest
import pytest_asyncio
from prompt_toolkit.application import current
from prompt_toolkit.input import base as input_base
from prompt_toolkit.output import base as output_base

from mirror import config as config_lib
from mirror.cli import (
  actions,
  cli,
  extension,
  user_extensions,
  workspace_controller,
  workspaces,
  worktrees,
)
from mirror.cli.statsig.events import client
from mirror.cli.statsig.events import event as statsig_event


class _RecordingClient(client.EventClient):
  """Replace only the telemetry boundary; never construct an HTTP client."""

  def __init__(self) -> None:
    super().__init__()
    self.collecting = True
    self.events: list[statsig_event.Event] = []

  @property
  @override
  def enabled(self) -> bool:
    return self.collecting

  @override
  async def enable(self) -> None:
    self.collecting = True

  @override
  async def disable(self) -> None:
    self.collecting = False

  @override
  def log_event(self, event: statsig_event.Event) -> None:
    assert self.enabled, 'Opt-out must prevent collection, not just delivery'
    if event.name == 'mirror_running_sessions':
      self.events.append(event)


class _Agent(cli._Repl):  # pyright: ignore[reportPrivateUsage] - terminal agent integration
  """Use real dispatch while exposing completion of backend shutdown."""

  def __init__(
    self,
    spec: workspaces.AgentSpec,
    controller: workspace_controller.WorkspaceController,
    *,
    registry: extension.ExtensionRegistry,
    directory: pathlib.Path,
    settings: config_lib.Config,
  ) -> None:
    super().__init__(
      [],
      spec.config,
      session_id='test-session',
      workspace=spec.workspace,
      database=directory / 'cli.db',
      active=spec.active,
      extension_registry=registry,
      workspace_controller=controller,
      user_extensions=user_extensions.UserExtensions(
        registry, directory / 'extensions', settings=settings
      ),
    )
    self.close_started = asyncio.Event()
    self.close_allowed = asyncio.Event()
    self.close_allowed.set()
    self.closed = asyncio.Event()

  @override
  async def close(self) -> None:
    self.close_started.set()
    await self.close_allowed.wait()
    await super().close()
    self.closed.set()


@dataclasses.dataclass
class _Roster:
  controller: workspace_controller.WorkspaceController
  first: _Agent
  second: _Agent
  remote: _Agent
  first_workspace: workspaces.AgentWorkspace
  second_workspace: workspaces.AgentWorkspace
  remote_workspace: workspaces.AgentWorkspace
  recorder: _RecordingClient
  deletion: asyncio.Future[None]
  discovered: list[worktrees.Worktree]


@pytest_asyncio.fixture
async def roster(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Roster]:
  # The transport boundary is replaced by a recording EventClient; reducer,
  # terminal dispatch, roster changes and consent toggling remain real.
  recorder = _RecordingClient()
  monkeypatch.setattr(client, 'get', lambda: recorder)
  registry = extension.ExtensionRegistry()
  settings = config_lib.Config(tmp_path / 'config.json')
  config = {
    'model': 'test-model',
    'conversation_id': 'test-conversation',
    'create_session': True,
    'model_selection_override': True,
  }

  def create_agent(
    spec: workspaces.AgentSpec,
    controller: workspace_controller.WorkspaceController,
  ) -> _Agent:
    return _Agent(
      spec, controller, registry=registry, directory=tmp_path, settings=settings
    )

  second_path = tmp_path / 'second'
  second_path.mkdir()
  first_worktree = worktrees.current_worktree(tmp_path)
  second_worktree = worktrees.Worktree(second_path, 'second')
  discovered = [first_worktree, second_worktree]
  deletion: asyncio.Future[None] = asyncio.get_running_loop().create_future()

  async def delete_remote_session(session_id: str) -> None:
    assert session_id == 'remote-session'
    await deletion

  with current.create_app_session(
    input=input_base.DummyInput(), output=output_base.DummyOutput()
  ):
    controller = workspace_controller.WorkspaceController(
      config,
      workspace=tmp_path,
      reasoning_effort='medium',
      create_agent=create_agent,
      database=tmp_path / 'cli.db',
      settings=settings,
      discover=lambda _: tuple(discovered),
      display_unmanaged_worktrees=True,
      delete_remote_session=delete_remote_session,
    )
    first = controller.selected
    assert isinstance(first, workspaces.AgentWorkspace)
    assert isinstance(first.agent, _Agent)
    first.agent.dispatch(actions.WorkerStarted(), actions.ReplayFinished(False))
    second = create_agent(
      workspaces.AgentSpec(config, second_path, 'medium', False, False),
      controller,
    )
    second.dispatch(actions.WorkerStarted(), actions.ReplayFinished(False))
    second_workspace = workspaces.AgentWorkspace(
      'second',
      workspaces.Directory('second', second_path, 'second'),
      second,
      False,
      discovered=True,
    )
    remote = create_agent(
      workspaces.AgentSpec(config, tmp_path, 'medium', False, True), controller
    )
    remote.dispatch(actions.WorkerStarted(), actions.ReplayFinished(False))
    remote_workspace = workspaces.AgentWorkspace(
      'remote', workspaces.RemoteSession('remote-session'), remote, False
    )
    controller.workspaces.extend((second_workspace, remote_workspace))
    try:
      yield _Roster(
        controller,
        first.agent,
        second,
        remote,
        first,
        second_workspace,
        remote_workspace,
        recorder,
        deletion,
        discovered,
      )
    finally:
      first.agent.close_allowed.set()
      second.close_allowed.set()
      remote.close_allowed.set()
      if not deletion.done():
        deletion.set_result(None)
      await controller.close()


async def _finish_drop(
  controller: workspace_controller.WorkspaceController,
) -> None:
  # There is no public completion handle for background deletion.
  await asyncio.gather(*controller._workspace_deletion_tasks)  # pyright: ignore[reportPrivateUsage]
  await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_dispatch_idle_history_replacement_emits_nothing(
  roster: _Roster,
) -> None:
  roster.first.dispatch(actions.RerenderStarted())
  assert roster.first.worktree_state is actions.WorktreeAgentState.IDLE
  roster.first.dispatch(actions.RepaintFinished(True))
  roster.first.dispatch(
    actions.SessionSwitchStarted(actions.PendingSwitch('saved', False))
  )
  roster.first.dispatch(actions.ReplayFinished(True))
  assert roster.first.worktree_state is actions.WorktreeAgentState.IDLE
  roster.first.dispatch(actions.RepaintFinished(True))
  assert roster.recorder.events == []


@pytest.mark.asyncio
async def test_dispatch_live_turn_during_repaint_stays_running(
  roster: _Roster,
) -> None:
  roster.first.dispatch(actions.RerenderStarted())
  roster.first.dispatch(actions.AgentTurnStarted())
  assert roster.first.worktree_state is actions.WorktreeAgentState.RUNNING
  roster.first.dispatch(actions.RepaintFinished(True))
  assert roster.first.worktree_state is actions.WorktreeAgentState.RUNNING
  roster.first.dispatch(actions.TurnFinished())
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]


@pytest.mark.asyncio
async def test_dispatch_counts_visible_turns_and_deduplicates(
  roster: _Roster,
) -> None:
  roster.first.dispatch(actions.TurnStarted(), actions.TurnFinished())
  assert roster.recorder.events == []
  roster.first.dispatch(actions.TurnStarted())
  roster.second.dispatch(actions.TurnStarted())
  roster.second.dispatch(actions.TurnStarted())
  roster.controller.running_sessions_changed()
  roster.first.dispatch(actions.TurnFinished())
  roster.second.dispatch(actions.TurnFinished())
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=2),
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]


@pytest.mark.asyncio
async def test_drop_running_tab_updates_count_before_worker_closes(
  roster: _Roster,
) -> None:
  roster.controller.selected = roster.remote_workspace
  roster.remote.dispatch(actions.TurnStarted())
  assert await roster.controller.drop(confirmed=True)
  assert roster.remote.worktree_state is actions.WorktreeAgentState.RUNNING
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]
  roster.deletion.set_result(None)
  await _finish_drop(roster.controller)
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]


@pytest.mark.asyncio
async def test_drop_failure_restores_running_tab_count(roster: _Roster) -> None:
  roster.controller.selected = roster.remote_workspace
  roster.remote.dispatch(actions.TurnStarted())
  assert await roster.controller.drop(confirmed=True)
  assert roster.recorder.events[-1] == statsig_event.Event(
    'mirror_running_sessions', value=0
  )
  roster.deletion.set_exception(RuntimeError('synthetic deletion failure'))
  await _finish_drop(roster.controller)
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
    statsig_event.Event('mirror_running_sessions', value=1),
  ]


@pytest.mark.asyncio
async def test_apply_analytics_config_enabling_samples_only_current_count(
  roster: _Roster,
) -> None:
  await cli.apply_analytics_config({'analytics': False}, roster.controller)
  roster.first.dispatch(actions.TurnStarted())
  roster.second.dispatch(actions.TurnStarted())
  roster.second.dispatch(actions.TurnFinished())
  assert roster.recorder.events == []
  await cli.apply_analytics_config(
    {'analytics_notice_shown': True}, roster.controller
  )
  await cli.apply_analytics_config(
    {'analytics_notice_shown': True}, roster.controller
  )
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1)
  ]


@pytest.mark.asyncio
async def test_apply_analytics_config_reenabling_without_turn_change_resamples(
  roster: _Roster,
) -> None:
  roster.first.dispatch(actions.TurnStarted())
  await cli.apply_analytics_config({'analytics': False}, roster.controller)
  await cli.apply_analytics_config(
    {'analytics_notice_shown': True}, roster.controller
  )
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=1),
  ]


@pytest.mark.asyncio
async def test_apply_analytics_config_without_notice_remains_disabled(
  roster: _Roster,
) -> None:
  await cli.apply_analytics_config({'analytics': True}, roster.controller)
  roster.first.dispatch(actions.TurnStarted())
  assert not roster.recorder.enabled
  assert roster.recorder.events == []


@pytest.mark.asyncio
async def test_close_workspace_updates_count_before_worker_shutdown(
  roster: _Roster,
) -> None:
  roster.first.dispatch(actions.TurnStarted())
  roster.first.close_allowed.clear()
  closing = asyncio.create_task(
    roster.controller.close_workspace(roster.first_workspace.id)
  )
  await roster.first.close_started.wait()
  assert roster.first.worktree_state is actions.WorktreeAgentState.RUNNING
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]
  roster.first.close_allowed.set()
  assert await closing
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]


@pytest.mark.asyncio
async def test_reconcile_hidden_running_workspace_updates_count_before_close(
  roster: _Roster,
) -> None:
  roster.second.dispatch(actions.TurnStarted())
  roster.second.close_allowed.clear()
  roster.controller.set_display_unmanaged_worktrees(False)
  await roster.second.close_started.wait()
  assert roster.second.worktree_state is actions.WorktreeAgentState.RUNNING
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]
  roster.second.close_allowed.set()
  await roster.controller.initialize()
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]


@pytest.mark.asyncio
async def test_reconcile_missing_directory_updates_count_without_turn_change(
  roster: _Roster,
) -> None:
  roster.second_workspace.discovered = False
  roster.discovered.pop()
  roster.second.dispatch(actions.TurnStarted())
  roster.second_workspace.path.rmdir()
  await roster.controller.initialize()
  assert roster.second.worktree_state is actions.WorktreeAgentState.RUNNING
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
  ]
  roster.second_workspace.path.mkdir()
  await roster.controller.initialize()
  assert roster.recorder.events == [
    statsig_event.Event('mirror_running_sessions', value=1),
    statsig_event.Event('mirror_running_sessions', value=0),
    statsig_event.Event('mirror_running_sessions', value=1),
  ]
