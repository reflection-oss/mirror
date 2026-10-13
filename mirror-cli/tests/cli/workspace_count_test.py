"""Workspace metric behavior with synthetic backends and no telemetry delivery."""

from __future__ import annotations

import asyncio
import concurrent.futures
import pathlib
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import override

import pytest
import pytest_asyncio
from mirror.protocol import agent_pb2

from mirror import workspace as workspace_identity
from mirror.cli import (
  actions,
  cli,
  cli_store,
  models,
  workspace_controller,
  workspaces,
  worktrees,
)
from mirror.cli.statsig import config as statsig_config
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event


class _Agent:
  """Inert implementation of the agent protocol; never starts a worker or UI."""

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
    self.errors: list[str] = []
    self.error_received = asyncio.Event()

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    return ()

  def submit_instruction(self, instruction: str) -> None:
    raise AssertionError(f'Unexpected instruction: {instruction!r}')

  def show_error(self, message: str) -> None:
    self.errors.append(message)
    self.error_received.set()

  async def start(self) -> None:
    pass

  async def close(self) -> None:
    pass


def _create_agent(
  spec: workspaces.AgentSpec,
  controller: workspace_controller.WorkspaceController,
) -> _Agent:
  del controller  # Unused.
  return _Agent(spec)


class _WorktreeCreator:
  """Hold the blocking creation backend until the test completes its result."""

  def __init__(self) -> None:
    self.loop = asyncio.get_running_loop()
    self.requested = asyncio.Event()
    self.result: concurrent.futures.Future[worktrees.Worktree] = (
      concurrent.futures.Future()
    )

  def __call__(
    self, source: pathlib.Path, name: str | None
  ) -> worktrees.Worktree:
    del source, name  # Unused.
    self.loop.call_soon_threadsafe(self.requested.set)
    return self.result.result(timeout=5)


class _DiscoveredWorktrees(list[worktrees.Worktree]):
  """Signal consumption of a discovery result, including unchanged rosters."""

  def __init__(self, items: Sequence[worktrees.Worktree]) -> None:
    super().__init__(items)
    self.read = asyncio.Event()

  @override
  def __iter__(self) -> Iterator[worktrees.Worktree]:
    yield from super().__iter__()
    asyncio.get_running_loop().call_soon(self.read.set)

  async def wait_until_read(self) -> None:
    await asyncio.wait_for(self.read.wait(), timeout=5)
    self.read.clear()
    # Flush completion callbacks before the next UI action.
    await asyncio.sleep(0)


def _config() -> Mapping[str, object]:
  return {
    'conversation_id': 'conv-1',
    'model': 'm-test',
    'approval_mode': 'auto',
    'create_session': True,
  }


@pytest.fixture(autouse=True)
def isolate_workspace(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Keep persistence and checkout identity independent of the host and VCS."""
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'false')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(tmp_path / 'models.json'))
  monkeypatch.setenv('MIRROR_STATE_DB', str(tmp_path / 'state.db'))
  monkeypatch.setenv('MIRROR_CLI_DB', str(tmp_path / 'cli.db'))

  def current_worktree(path: pathlib.Path) -> worktrees.Worktree:
    return worktrees.Worktree(path, 'main')

  def repository_key(cwd: str) -> str:
    return cwd

  monkeypatch.setattr(worktrees, 'current_worktree', current_worktree)
  monkeypatch.setattr(workspace_identity, 'git_workspace_key', repository_key)


class _WorkspaceCounts(statsig_client.EventClient):
  """Capture snapshots with the SDK's real consent lifecycle, without delivery."""

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
    # Let completion callbacks run before the next user action.
    await asyncio.sleep(0)

  def assert_empty(self) -> None:
    assert self.events.empty(), 'Unexpected workspace count snapshot'


@pytest_asyncio.fixture
async def workspace_counts(
  monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[_WorkspaceCounts]:
  async with _WorkspaceCounts() as counts:
    # Replace the shared client boundary so earlier instance method overrides
    # cannot bypass capture; collection still follows enable/disable consent.
    monkeypatch.setattr(statsig_client, 'get', lambda: counts)
    await counts.enable()
    yield counts


@pytest.mark.asyncio
async def test_workspace_count_local_and_remote_rosters(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  feature = tmp_path / 'feature'
  feature.mkdir()

  async def remote_sessions() -> Sequence[str]:
    return ('remote-one', 'remote-two')

  async def delete_remote(session_id: str) -> None:
    del session_id  # Unused.

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: (
      worktrees.Worktree(tmp_path, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
    display_unmanaged_worktrees=True,
    remote_enabled=True,
    list_remote_sessions=remote_sessions,
    delete_remote_session=delete_remote,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    controller.refresh()
    await workspace_counts.wait_for(local=2, remote=0)
    controller.load_remote_sessions()
    await workspace_counts.wait_for(local=2, remote=1)
    await workspace_counts.wait_for(local=2, remote=2)
    await controller.create_remote()
    await workspace_counts.wait_for(local=2, remote=3)
    feature.rmdir()
    controller.refresh()
    await workspace_counts.wait_for(local=1, remote=3)
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_initial_snapshot_includes_restored_folders(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  folder = tmp_path / 'saved-folder'
  folder.mkdir()
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.save_workspace(str(tmp_path), 'saved-folder', str(folder), 'Notes')
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    database=database,
    is_trusted_directory=lambda _: True,
  )
  try:
    await workspace_counts.wait_for(local=2, remote=0)
    workspace_counts.assert_empty()
    assert await controller.close_workspace('saved-folder')
    await workspace_counts.wait_for(local=1, remote=0)
    assert folder.is_dir()
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_tracks_duplicate_folder_handles_and_close(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    is_trusted_directory=lambda _: True,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    first = await controller.open_folder(tmp_path, tmp_path, name='Notes')
    await workspace_counts.wait_for(local=2, remote=0)
    second = await controller.open_folder(first.id, tmp_path, name='Review')
    await workspace_counts.wait_for(local=3, remote=0)
    assert await controller.close_workspace(first.id)
    await workspace_counts.wait_for(local=2, remote=0)
    assert await controller.close_workspace(second.id)
    await workspace_counts.wait_for(local=1, remote=0)
    assert tmp_path.is_dir()
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_excludes_pending_creation(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  creator = _WorktreeCreator()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
    create=creator,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    assert await controller.create('feature')
    await asyncio.wait_for(creator.requested.wait(), timeout=5)
    assert isinstance(
      controller.visible_workspaces()[1], workspaces.PendingWorkspace
    )
    controller.refresh()
    workspace_counts.assert_empty()
    feature = tmp_path / 'feature'
    feature.mkdir()
    creator.result.set_result(worktrees.Worktree(feature, 'feature'))
    await workspace_counts.wait_for(local=2, remote=0)
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_excludes_dropping_and_restores_failed_drop(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  finish_deletion = asyncio.Event()

  async def fail_deletion(session_id: str) -> None:
    del session_id  # Unused.
    await finish_deletion.wait()
    raise RuntimeError('synthetic deletion failure')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
    remote_enabled=True,
    delete_remote_session=fail_deletion,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    assert await controller.create_remote() == 1
    assert await controller.select(1)
    await workspace_counts.wait_for(local=1, remote=1)
    assert await controller.drop()
    await workspace_counts.wait_for(local=1, remote=0)
    finish_deletion.set()
    await workspace_counts.wait_for(local=1, remote=1)
  finally:
    finish_deletion.set()
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_updates_with_display_settings(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  unmanaged = tmp_path / 'external'
  unmanaged.mkdir()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: (
      worktrees.Worktree(tmp_path, 'main'),
      worktrees.Worktree(unmanaged, 'external'),
    ),
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    controller.set_display_unmanaged_worktrees(True)
    await workspace_counts.wait_for(local=2, remote=0)
    controller.set_display_unmanaged_worktrees(False)
    await workspace_counts.wait_for(local=1, remote=0)
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_navigation_and_unchanged_reconciliation(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  feature = tmp_path / 'feature'
  feature.mkdir()
  discovered = _DiscoveredWorktrees([
    worktrees.Worktree(tmp_path, 'main'),
    worktrees.Worktree(feature, 'feature'),
  ])
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: discovered,
    display_unmanaged_worktrees=True,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    controller.refresh()
    await discovered.wait_until_read()
    await workspace_counts.wait_for(local=2, remote=0)
    workspace_counts.assert_empty()

    assert await controller.select(1)
    await discovered.wait_until_read()
    workspace_counts.assert_empty()
    assert await controller.select_workspace(controller.workspaces[0].identity)
    await discovered.wait_until_read()
    workspace_counts.assert_empty()
    await controller.cycle(1)
    await discovered.wait_until_read()
    workspace_counts.assert_empty()
    controller.refresh()
    await discovered.wait_until_read()
    workspace_counts.assert_empty()
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_failed_reconciliation(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  feature = tmp_path / 'feature'
  feature.mkdir()
  discovered = _DiscoveredWorktrees([
    worktrees.Worktree(tmp_path, 'main'),
    worktrees.Worktree(feature, 'feature'),
  ])
  fail = False

  def discover(path: pathlib.Path) -> Sequence[worktrees.Worktree]:
    del path  # Unused.
    if fail:
      raise RuntimeError('synthetic discovery failure')
    return discovered

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=discover,
    display_unmanaged_worktrees=True,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    controller.refresh()
    await discovered.wait_until_read()
    await workspace_counts.wait_for(local=2, remote=0)
    fail = True
    feature.rmdir()
    controller.refresh()
    selected = controller.selected
    assert isinstance(selected, workspaces.AgentWorkspace)
    assert isinstance(selected.agent, _Agent)
    await workspace_counts.wait_for(local=1, remote=0)
    await asyncio.wait_for(selected.agent.error_received.wait(), timeout=5)
    workspace_counts.assert_empty()

    selected.agent.error_received.clear()
    await controller.initialize()
    await asyncio.wait_for(selected.agent.error_received.wait(), timeout=5)
    workspace_counts.assert_empty()

    fail = False
    controller.refresh()
    await discovered.wait_until_read()
    workspace_counts.assert_empty()
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_reports_zero_when_all_local_paths_disappear(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  launch = tmp_path / 'main'
  launch.mkdir()

  def discover(path: pathlib.Path) -> Sequence[worktrees.Worktree]:
    del path  # Unused.
    pytest.fail('Discovery should not run without an available local path')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=launch,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=discover,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    launch.rmdir()
    await controller.initialize()
    await workspace_counts.wait_for(local=0, remote=0)
    await controller.initialize()
    workspace_counts.assert_empty()
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_cancelled_reconciliation(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  launch = tmp_path / 'main'
  launch.mkdir()
  loop = asyncio.get_running_loop()
  requested = asyncio.Event()
  result: concurrent.futures.Future[Sequence[worktrees.Worktree]] = (
    concurrent.futures.Future()
  )

  def discover(path: pathlib.Path) -> Sequence[worktrees.Worktree]:
    del path  # Unused.
    loop.call_soon_threadsafe(requested.set)
    return result.result(timeout=5)

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=launch,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=discover,
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    controller.refresh()
    await asyncio.wait_for(requested.wait(), timeout=5)
    launch.rmdir()
    await controller.close()
    workspace_counts.assert_empty()
  finally:
    result.set_result((worktrees.Worktree(launch, 'main'),))
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_analytics_settings_resync_without_replay(
  tmp_path: pathlib.Path,
  workspace_counts: _WorkspaceCounts,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  await workspace_counts.disable()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
    remote_enabled=True,
    delete_remote_session=lambda _: asyncio.sleep(0),
  )
  try:
    workspace_counts.assert_empty()
    # apply_config calls this hook after enabling or disabling the SDK.
    await workspace_counts.enable()
    controller.report_workspace_count()
    await workspace_counts.wait_for(local=1, remote=0)
    controller.report_workspace_count()
    workspace_counts.assert_empty()

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

    await workspace_counts.disable()
    controller.report_workspace_count()
    await controller.create_remote()
    await controller.create_remote()
    workspace_counts.assert_empty()
    await workspace_counts.enable()
    controller.report_workspace_count()
    await workspace_counts.wait_for(local=1, remote=2)
    controller.report_workspace_count()
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_analytics_config_samples_current_roster(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  await workspace_counts.disable()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    remote_enabled=True,
    delete_remote_session=lambda _: asyncio.sleep(0),
  )
  try:
    await controller.create_remote()
    await controller.create_remote()
    workspace_counts.assert_empty()
    await cli.apply_analytics_config(
      {'analytics_notice_shown': True}, controller
    )
    await workspace_counts.wait_for(local=1, remote=2)
    await cli.apply_analytics_config(
      {'analytics_notice_shown': True}, controller
    )
    workspace_counts.assert_empty()

    await cli.apply_analytics_config({'analytics': False}, controller)
    await controller.create_remote()
    workspace_counts.assert_empty()
    await cli.apply_analytics_config(
      {'analytics_notice_shown': True}, controller
    )
    await workspace_counts.wait_for(local=1, remote=3)
  finally:
    await controller.close()
  workspace_counts.assert_empty()


@pytest.mark.asyncio
async def test_workspace_count_successful_drop_emits_once(
  tmp_path: pathlib.Path, workspace_counts: _WorkspaceCounts
) -> None:
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_create_agent,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
    remote_enabled=True,
    delete_remote_session=lambda _: asyncio.sleep(0),
  )
  try:
    await workspace_counts.wait_for(local=1, remote=0)
    assert await controller.create_remote() == 1
    await workspace_counts.wait_for(local=1, remote=1)
    assert await controller.select(1)
    assert await controller.drop()
    await workspace_counts.wait_for(local=1, remote=0)
  finally:
    await controller.close()
  workspace_counts.assert_empty()
