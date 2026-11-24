"""Workspace metrics use daemon snapshots and the real telemetry consent lifecycle."""

from __future__ import annotations

import asyncio
import concurrent.futures
import pathlib
import queue
from collections.abc import AsyncIterator
from typing import override

import grpc
import pytest
import pytest_asyncio
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
)

from mirror.cli import (
  actions,
  cli,
  daemon_client,
  models,
  workspace_controller,
  workspaces,
)
from mirror.cli.statsig import config as statsig_config
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event


class _Agent:
  """Inert view; never starts a worker or executes workspace code."""

  def __init__(self, spec: workspaces.AgentSpec) -> None:
    self.workspace = spec.workspace
    self.reasoning_effort = spec.reasoning_effort
    self.approval_mode = str(spec.config['approval_mode'])
    self.model = str(spec.config['model'])
    self.model_selection = models.ModelOption(self.model, self.model)
    self.conversation_id = str(spec.config['conversation_id'])
    self.activity_status = ''
    self.queued_message_count = 0
    self.worktree_state = actions.WorktreeAgentState.NOT_STARTED
    self.restarting = False
    self.errors: list[str] = []
    self.error_received = asyncio.Event()

  def composer_state(self) -> actions.ComposerState:
    return actions.ComposerState()

  async def switch_workspace_session(
    self, session_id: str | None = None
  ) -> bool:
    raise AssertionError(f'Unexpected session switch: {session_id!r}')

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    return ()

  def submit_instruction(self, instruction: str) -> None:
    raise AssertionError(f'Unexpected instruction: {instruction!r}')

  def submit_message(self, message: agent_pb2.UserMessage) -> None:
    raise AssertionError(f'Unexpected message: {message!r}')

  def escalate(self) -> None:
    raise AssertionError('Unexpected escalation')

  def show_error(self, message: str) -> None:
    self.errors.append(message)
    self.error_received.set()

  async def fork_workspace(
    self, session_id: str, directory: pathlib.Path | None, name: str
  ) -> pathlib.Path:
    raise AssertionError('Unexpected workspace fork')

  async def start(self) -> None:
    pass

  async def close(self) -> None:
    pass


def _create_agent(
  spec: workspaces.AgentSpec,
  controller: workspace_controller.WorkspaceController,
) -> _Agent:
  del controller
  return _Agent(spec)


class _WorkspaceCounts(statsig_client.EventClient):
  """Capture snapshots with real enable/disable consent, without delivery."""

  def __init__(self) -> None:
    super().__init__(
      statsig_config.StatsigConfig('workspace-count-test', 'test-launch')
    )
    self.events: asyncio.Queue[statsig_event.Event] = asyncio.Queue()

  @override
  def log_event(self, event: statsig_event.Event) -> None:
    if self.enabled and event.name == 'mirror_workspace_count':
      self.events.put_nowait(event)

  async def wait_for(self, *, local: int, remote: int) -> None:
    """Require the next snapshot, without skipping duplicates or wrong counts."""
    async with asyncio.timeout(5):
      local_event = await self.events.get()
      remote_event = await self.events.get()
    assert local_event.name == remote_event.name == 'mirror_workspace_count'
    assert local_event.metadata == {'location': 'local'}
    assert remote_event.metadata == {'location': 'remote'}
    assert (local_event.value, remote_event.value) == (local, remote)
    await asyncio.sleep(0)

  def assert_empty(self) -> None:
    assert self.events.empty(), 'Unexpected workspace count snapshot'


@pytest_asyncio.fixture
async def workspace_counts(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
) -> AsyncIterator[_WorkspaceCounts]:
  monkeypatch.setenv('HOME', str(tmp_path))
  async with _WorkspaceCounts() as counts:
    monkeypatch.setattr(statsig_client, 'get', lambda: counts)
    await counts.enable()
    yield counts


async def _remote_sessions() -> tuple[str, str]:
  return ('remote-one', 'remote-two')


async def _delete_remote(session_id: str) -> None:
  del session_id


@pytest_asyncio.fixture
async def controller(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_counts: _WorkspaceCounts,
) -> AsyncIterator[workspace_controller.WorkspaceController]:
  del workspace_counts
  initial = workspace_client.create(
    daemon_pb2.CreateWorkspaceRequest(
      workspace_id='00000000-0000-0000-0000-000000000001',
      directory=str(tmp_path),
      name='Launch',
    )
  )
  controller = workspace_controller.WorkspaceController(
    {'conversation_id': '', 'model': 'm-test', 'approval_mode': 'auto'},
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_create_agent,
    session_id=initial.session_id,
    remote_enabled=True,
    list_remote_sessions=_remote_sessions,
    delete_remote_session=_delete_remote,
    is_trusted_directory=lambda _: True,
  )
  try:
    yield controller
  finally:
    await controller.close()


async def _refresh(
  controller: workspace_controller.WorkspaceController,
) -> None:
  controller.refresh()
  task = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
  assert task is not None
  await task
  await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_workspace_count_local_and_remote_rosters(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  feature = workspace_snapshot.workspaces.add(
    id='feature',
    session_id='feature-session',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  await _refresh(controller)
  workspace_counts.assert_empty()
  await controller.open_session(feature.session_id)
  await workspace_counts.wait_for(local=2, remote=0)
  controller.load_remote_sessions()
  discovery = controller._remote_session_task  # pyright: ignore[reportPrivateUsage] - wait for metadata discovery.
  assert discovery is not None
  await discovery
  workspace_counts.assert_empty()
  await controller.open_session('remote-one')
  await workspace_counts.wait_for(local=2, remote=1)
  await controller.open_session('remote-two')
  await workspace_counts.wait_for(local=2, remote=2)
  await controller.create_remote()
  await workspace_counts.wait_for(local=2, remote=3)
  feature.state = daemon_pb2.WORKSPACE_STATE_UNAVAILABLE
  await _refresh(controller)
  await workspace_counts.wait_for(local=1, remote=3)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_excludes_catalog_until_explicitly_opened(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_counts: _WorkspaceCounts,
) -> None:
  workspace_snapshot.workspaces.add(
    id='first',
    session_id='first-session',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  workspace_snapshot.workspaces.add(
    id='second',
    session_id='second-session',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  controller = workspace_controller.WorkspaceController(
    {'conversation_id': '', 'model': 'm-test', 'approval_mode': 'auto'},
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_create_agent,
    is_trusted_directory=lambda _: True,
  )
  try:
    await workspace_counts.wait_for(local=0, remote=0)
    assert isinstance(controller.selected, workspaces.EmptyWorkspace)
    await controller.open_session('first-session')
    await workspace_counts.wait_for(local=1, remote=0)
    await controller.open_session('second-session')
    await workspace_counts.wait_for(local=2, remote=0)
    assert await controller.close_workspace('second')
    await workspace_counts.wait_for(local=1, remote=0)
    assert tmp_path.is_dir()
    workspace_counts.assert_empty()
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_workspace_count_tracks_duplicate_folder_handles_and_close(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  assert isinstance(controller.selected, workspaces.AgentWorkspace)
  first = await controller.open_folder(
    controller.selected.id, tmp_path, name='Notes'
  )
  await workspace_counts.wait_for(local=2, remote=0)
  second = await controller.open_folder(first.id, tmp_path, name='Review')
  await workspace_counts.wait_for(local=3, remote=0)
  assert await controller.close_workspace(first.id)
  await workspace_counts.wait_for(local=2, remote=0)
  assert await controller.close_workspace(second.id)
  await workspace_counts.wait_for(local=1, remote=0)
  assert tmp_path.is_dir()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_ignores_pre_creation_snapshot(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
  ],
  workspace_queries: queue.Queue[daemon_pb2.WatchWorkspacesRequest],
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  workspace_queries.get_nowait()
  stale = daemon_pb2.WatchWorkspacesResponse()
  stale.CopyFrom(workspace_snapshot)
  response: concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse] = (
    concurrent.futures.Future()
  )
  workspace_responses.put(response)
  controller.refresh()
  await asyncio.to_thread(workspace_queries.get, True, 5)
  try:
    assert isinstance(controller.selected, workspaces.AgentWorkspace)
    await controller.open_folder(controller.selected.id, tmp_path, name='New')
    await workspace_counts.wait_for(local=2, remote=0)
  finally:
    response.set_result(stale)
  await _refresh(controller)
  workspace_counts.assert_empty()
  assert len(controller.visible_workspaces()) == 2


@pytest.mark.asyncio
async def test_workspace_count_excludes_pending_creation(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_creations: queue.Queue[
    tuple[
      daemon_pb2.CreateWorkspaceRequest,
      concurrent.futures.Future[daemon_pb2.Workspace],
    ]
  ],
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  (tmp_path / '.git').mkdir()
  assert await controller.create('feature')
  request, result = await asyncio.to_thread(workspace_creations.get, True, 5)
  assert isinstance(
    controller.visible_workspaces()[1], workspaces.PendingWorkspace
  )
  await _refresh(controller)
  workspace_counts.assert_empty()
  result.set_result(
    workspace_snapshot.workspaces.add(
      id=request.workspace_id,
      session_id='new-session',
      directory=str(tmp_path),
      name='feature',
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    )
  )
  await workspace_counts.wait_for(local=2, remote=0)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_restores_workspace_after_failed_drop(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_deletions: queue.Queue[
    tuple[daemon_pb2.DeleteWorkspaceRequest, concurrent.futures.Future[None]]
  ],
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  assert isinstance(controller.selected, workspaces.AgentWorkspace)
  directory = tmp_path / '.mirror' / 'worktrees' / 'feature'
  (directory / '.git').mkdir(parents=True)
  opened = await controller.open_folder(controller.selected.id, directory)
  workspace_snapshot.workspaces[1].is_worktree = True
  await workspace_counts.wait_for(local=2, remote=0)
  await _refresh(controller)
  assert await controller.select_workspace(opened.id)
  selected = controller.selected
  assert isinstance(selected, workspaces.AgentWorkspace)
  selected.draft = actions.ComposerState('unsent draft', 4)
  assert await controller.drop()
  await workspace_counts.wait_for(local=1, remote=0)
  request, result = await asyncio.to_thread(workspace_deletions.get, True, 5)
  assert request.workspace_id == opened.id
  result.set_exception(ValueError('synthetic deletion failure'))
  assert isinstance(controller.selected, workspaces.AgentWorkspace)
  agent = controller.selected.agent
  assert isinstance(agent, _Agent)
  await asyncio.wait_for(agent.error_received.wait(), 5)
  await _refresh(controller)
  assert selected not in controller.visible_workspaces()
  assert 'synthetic deletion failure' in agent.errors[-1]
  await workspace_counts.wait_for(local=2, remote=0)
  assert await controller.select_workspace(opened.id)
  assert isinstance(controller.selected, workspaces.AgentWorkspace)
  assert controller.selected.draft == selected.draft
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_updates_with_display_settings(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_queries: queue.Queue[daemon_pb2.WatchWorkspacesRequest],
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  assert not workspace_queries.get_nowait().include_unmanaged
  unmanaged = workspace_snapshot.workspaces.add(
    id='unmanaged',
    session_id='unmanaged-session',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  controller.set_display_unmanaged_worktrees(True)
  await _refresh(controller)
  workspace_counts.assert_empty()
  assert workspace_queries.get_nowait().include_unmanaged
  await controller.open_session(unmanaged.session_id)
  await workspace_counts.wait_for(local=2, remote=0)
  assert workspace_queries.get_nowait().include_unmanaged
  workspace_snapshot.workspaces.remove(unmanaged)
  controller.set_display_unmanaged_worktrees(False)
  await _refresh(controller)
  await workspace_counts.wait_for(local=1, remote=0)
  assert not workspace_queries.get_nowait().include_unmanaged
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_navigation_and_unchanged_reconciliation(
  tmp_path: pathlib.Path,
  controller: workspace_controller.WorkspaceController,
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  assert isinstance(controller.selected, workspaces.AgentWorkspace)
  second = await controller.open_folder(controller.selected.id, tmp_path)
  await workspace_counts.wait_for(local=2, remote=0)
  assert await controller.select_workspace(second.id)
  assert await controller.select(0)
  await controller.cycle(1)
  await _refresh(controller)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_failed_reconciliation_keeps_last_snapshot(
  controller: workspace_controller.WorkspaceController,
  workspace_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
  ],
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  failed: concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse] = (
    concurrent.futures.Future()
  )
  failed.set_exception(RuntimeError('synthetic discovery failure'))
  workspace_responses.put(failed)
  with pytest.raises(grpc.RpcError):
    await _refresh(controller)
  selected = controller.selected
  assert isinstance(selected, workspaces.AgentWorkspace)
  assert isinstance(selected.agent, _Agent)
  await asyncio.wait_for(selected.agent.error_received.wait(), timeout=5)
  assert 'synthetic discovery failure' in selected.agent.errors[-1]
  controller.report_workspace_count()
  workspace_counts.assert_empty()
  await _refresh(controller)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_reports_zero_when_daemon_marks_paths_unavailable(
  controller: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  workspace_snapshot.workspaces[
    0
  ].state = daemon_pb2.WORKSPACE_STATE_UNAVAILABLE
  await _refresh(controller)
  await workspace_counts.wait_for(local=0, remote=0)
  await _refresh(controller)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_cancelled_reconciliation(
  controller: workspace_controller.WorkspaceController,
  workspace_queries: queue.Queue[daemon_pb2.WatchWorkspacesRequest],
  workspace_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
  ],
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  workspace_queries.get_nowait()
  blocked: concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse] = (
    concurrent.futures.Future()
  )
  workspace_responses.put(blocked)
  controller.refresh()
  await asyncio.to_thread(workspace_queries.get, True, 5)
  try:
    await controller.close()
  finally:
    blocked.set_result(daemon_pb2.WatchWorkspacesResponse())
  await asyncio.sleep(0)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_analytics_settings_resync_without_replay(
  controller: workspace_controller.WorkspaceController,
  workspace_counts: _WorkspaceCounts,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  await workspace_counts.disable()
  with monkeypatch.context() as disabled:
    disabled.setattr(
      workspaces.AgentWorkspace,
      'available',
      property(lambda _: pytest.fail('Visibility inspected while opted out')),
    )
    controller.report_workspace_count()
  await workspace_counts.enable()
  controller.report_workspace_count()
  await workspace_counts.wait_for(local=1, remote=0)
  controller.report_workspace_count()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_analytics_config_samples_current_roster(
  controller: workspace_controller.WorkspaceController,
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  await cli.apply_analytics_config({'analytics': False}, controller)
  await controller.create_remote()
  await controller.create_remote()
  workspace_counts.assert_empty()
  await cli.apply_analytics_config({'analytics_notice_shown': True}, controller)
  await workspace_counts.wait_for(local=1, remote=2)
  await cli.apply_analytics_config({'analytics_notice_shown': True}, controller)
  workspace_counts.assert_empty()
  await cli.apply_analytics_config({'analytics': False}, controller)
  await controller.create_remote()
  workspace_counts.assert_empty()
  await cli.apply_analytics_config({'analytics_notice_shown': True}, controller)
  await workspace_counts.wait_for(local=1, remote=3)
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_successful_drop_emits_once(
  controller: workspace_controller.WorkspaceController,
  workspace_counts: _WorkspaceCounts,
) -> None:
  await workspace_counts.wait_for(local=1, remote=0)
  assert await controller.create_remote() == 1
  await workspace_counts.wait_for(local=1, remote=1)
  assert await controller.select(1)
  assert await controller.drop()
  await workspace_counts.wait_for(local=1, remote=0)
  await _refresh(controller)
  workspace_counts.assert_empty()
