"""Frontend selection, drafts and trust; not daemon lifecycle coverage."""

from __future__ import annotations

import asyncio
import concurrent.futures
import pathlib
import queue
import uuid

import pytest
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  mirror_pb2,
)

from mirror import config
from mirror.cli import (
  actions,
  daemon_client,
  models,
  rpc_client,
  workspace_controller,
  workspaces,
)


class _Agent:
  def __init__(self, spec: workspaces.AgentSpec) -> None:
    self.spec = spec
    self.workspace = spec.workspace
    self.reasoning_effort = spec.reasoning_effort
    self.approval_mode = str(spec.config.get('approval_mode', 'auto'))
    self.model = str(spec.config['model'])
    self.conversation_id = str(spec.config['conversation_id'])
    self.activity_status = ''
    self.queued_message_count = 0
    self.worktree_state = actions.WorktreeAgentState.NOT_STARTED
    self.restarting = False
    self.draft = actions.ComposerState()
    self.started = False
    self.closed = False
    self.closed_event = asyncio.Event()
    self.errors: list[str] = []
    self.submitted: list[str] = []
    self.escalations = 0

  @property
  def model_selection(self) -> models.ModelOption:
    inference = rpc_client.WorkerConfiguration(
      self.spec.config, cwd=self.spec.workspace
    ).inference_config(
      self.model, self.reasoning_effort, remote=self.spec.remote
    )
    return models.ModelOption(
      self.model, self.model, api=inference.api, base_url=inference.base_url
    )

  def show_error(self, message: str) -> None:
    self.errors.append(message)

  def composer_state(self) -> actions.ComposerState:
    return self.draft

  async def switch_workspace_session(
    self, session_id: str | None = None
  ) -> bool:
    raise NotImplementedError('this fixture does not own native workers')

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    return ()

  def submit_instruction(self, instruction: str) -> None:
    self.submitted.append(instruction)

  def escalate(self) -> None:
    self.escalations += 1

  async def fork_workspace(
    self, session_id: str, directory: pathlib.Path | None, name: str
  ) -> pathlib.Path:
    raise NotImplementedError('this fixture does not fork workspaces')

  async def start(self) -> None:
    self.started = True

  async def close(self) -> None:
    self.closed = True
    self.closed_event.set()


class _AgentFactory:
  def __init__(self) -> None:
    self.agents: list[_Agent] = []

  def __call__(
    self,
    spec: workspaces.AgentSpec,
    controller: workspace_controller.WorkspaceController,
  ) -> _Agent:
    del controller
    agent = _Agent(spec)
    self.agents.append(agent)
    return agent


async def _refresh(
  controller: workspace_controller.WorkspaceController,
) -> None:
  controller.refresh()
  task = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
  assert task is not None
  await task
  await asyncio.sleep(0)


def _register_workspace(
  client: daemon_client.Client, directory: pathlib.Path
) -> daemon_pb2.Workspace:
  return client.create(
    daemon_pb2.CreateWorkspaceRequest(
      workspace_id=str(uuid.uuid4()),
      directory=str(directory),
      name=directory.name,
    )
  )


async def _open_saved_workspace(
  controller: workspace_controller.WorkspaceController, session_id: str
) -> None:
  selected = controller.selected.identity
  await controller.open_session(session_id)
  await controller.select_workspace(selected)


def _config() -> dict[str, object]:
  return {
    'conversation_id': 'conv-1',
    'model': 'm-test',
    'approval_mode': 'auto',
    'create_session': True,
  }


@pytest.mark.asyncio
@pytest.mark.parametrize('speed', ['normal', 'fast'])
async def test_workspace_speed_survives_preferences_update_and_reopen(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_address: str,
  speed: str,
) -> None:
  record = _register_workspace(workspace_client, tmp_path)
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    store=workspace_client,
    is_trusted_directory=lambda _: True,
  )
  try:
    await controller.open_session(record.session_id)
    controller.save_preferences('high', 'auto', 'fast')
    controller.save_preferences('high', 'auto', speed)
    controller.save_preferences('low', 'approve')
  finally:
    await controller.close()

  restored = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    store=daemon_client.Client(workspace_address),
    is_trusted_directory=lambda _: True,
  )
  try:
    await restored.open_session(record.session_id)
    assert factory.agents[-1].spec.speed == speed
    assert factory.agents[-1].reasoning_effort == 'low'
    assert factory.agents[-1].approval_mode == 'approve'
  finally:
    await restored.close()


@pytest.mark.asyncio
async def test_controller_escalates_only_the_remote_sessions_it_is_asked_to(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse]
  ],
) -> None:
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    store=workspace_client,
    remote_enabled=True,
  )
  try:
    await controller.initialize()
    response: concurrent.futures.Future[daemon_pb2.WatchWorkspacesResponse] = (
      concurrent.futures.Future()
    )
    workspace_responses.put(response)
    asyncio.get_running_loop().call_soon(
      response.set_result, workspace_snapshot
    )
    assert await controller.create_remote('Release', escalated=True) == 0
    assert (
      await controller.create_remote(
        'aaaaaaaa-0000-0000-0000-000000000000', escalated=True
      )
      == 1
    )
    assert await controller.create_remote('Plain') == 2
    assert (
      await controller.create_remote(
        'aaaaaaaa-0000-0000-0000-000000000000', escalated=True
      )
      == 1
    )
    assert factory.agents[0].spec.config['escalate_session'] is True
    assert factory.agents[1].spec.config['escalate_session'] is True
    assert factory.agents[2].spec.config['escalate_session'] is False
    assert [agent.escalations for agent in factory.agents] == [0, 1, 0]
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_startup_leaves_running_agents_untouched_until_explicit_open(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  stopped = workspace_snapshot.workspaces.add(
    id='stopped',
    session_id='saved',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  running = workspace_snapshot.workspaces.add(
    id='running',
    session_id='live',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    inference=agent_pb2.InferenceConfig(
      model='live-model',
      api='openai-responses',
      base_url='http://127.0.0.1:1',
      reasoning_effort='high',
      extra_body='{"service_tier": "priority"}',
    ),
    approval_mode=agent_pb2.APPROVAL_MODE_HITL,
    preferences=daemon_pb2.WorkspacePreferences(
      approval_mode=agent_pb2.APPROVAL_MODE_AUTO,
      definition=mirror_pb2.AgentDefinition(model='stale-model'),
    ),
  )
  untrusted = workspace_snapshot.workspaces.add(
    id='untrusted',
    session_id='untrusted-live',
    directory=str(tmp_path / 'untrusted'),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    inference=running.inference,
  )
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config()
    | {'model_selection_override': True, 'approval_mode_override': True},
    workspace=tmp_path,
    reasoning_effort='low',
    create_agent=factory,
    store=workspace_client,
    is_trusted_directory=lambda path: path == pathlib.Path(running.directory),
  )
  try:
    before = workspace_snapshot.SerializeToString()
    await controller.initialize()
    await _refresh(controller)
    assert isinstance(controller.selected, workspaces.EmptyWorkspace)
    assert controller.visible_workspaces() == ()
    assert factory.agents == []
    assert workspace_snapshot.SerializeToString() == before
    assert {record.id for record in controller.workspace_catalog()} == {
      stopped.id,
      running.id,
      untrusted.id,
    }
    await controller.open_session(running.session_id)
    assert isinstance(controller.selected, workspaces.AgentWorkspace)
    assert controller.selected.path == pathlib.Path(running.directory)
    (active,) = factory.agents
    assert not active.started
    await (await controller.selected.start())
    assert active.started
    assert active.model == 'live-model'
    assert active.reasoning_effort == 'high'
    assert active.spec.speed == 'fast'
    assert active.approval_mode == 'approve'
    assert active.spec.config['model_base_url'] == 'http://127.0.0.1:1'
    assert active.spec.config['attach_running'] is True
    assert active.spec.config['create_session'] is False
  finally:
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('name', 'label'), [(None, 'Creating…'), ('review', 'review')]
)
async def test_spawn_publishes_pending_tab_until_creation_finishes(
  tmp_path: pathlib.Path,
  name: str | None,
  label: str,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_creations: queue.Queue[
    tuple[
      daemon_pb2.CreateWorkspaceRequest,
      concurrent.futures.Future[daemon_pb2.Workspace],
    ]
  ],
) -> None:
  registered = _register_workspace(workspace_client, tmp_path)
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    session_id=registered.session_id,
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=factory,
    is_trusted_directory=lambda _: True,
    settings=config.Config(tmp_path / 'config.json'),
  )
  (tmp_path / '.git').mkdir()
  task = asyncio.create_task(controller.spawn_workspace(tmp_path, name, None))
  try:
    request, response = await asyncio.to_thread(
      workspace_creations.get, True, 5
    )
    pending = controller.visible_workspaces()[-1]
    assert isinstance(pending, workspaces.PendingWorkspace)
    assert pending.label == label
    assert pending.state is actions.WorktreeAgentState.CREATING
    assert await controller.select(1)
    pending.draft = actions.ComposerState('pending draft', 7)
    record = workspace_snapshot.workspaces.add(
      id=request.workspace_id,
      session_id=str(uuid.uuid4()),
      name=name or 'generated',
      directory=str(tmp_path / 'created'),
      preferences=request.preferences,
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    )
    # A watch snapshot can publish the ready tab before CreateWorkspace returns.
    await _refresh(controller)
    ready = controller.selected
    assert isinstance(ready, workspaces.AgentWorkspace)
    assert ready.draft == actions.ComposerState('pending draft', 7)
    assert not ready.resume_pending
    ready.draft = actions.ComposerState('edited after snapshot', 12)
    response.set_result(record)
    result = await task
    assert config.Config(tmp_path / 'config.json').read()[
      'trusted_directories'
    ] == [str((tmp_path / 'created').resolve())]
    assert result.workspace.id == record.id
    assert len(controller.visible_workspaces()) == 2
    assert len(factory.agents) == 2
    assert controller.selected is ready
    assert ready.identity == pending.identity
    assert ready.draft == actions.ComposerState('edited after snapshot', 12)
    assert ready.state is not actions.WorktreeAgentState.CREATING
  finally:
    await controller.close()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', ['.git', '.jj'])
async def test_worktree_choice_names_the_launch_repository_and_uses_its_subdirectory(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_creations: queue.Queue[
    tuple[
      daemon_pb2.CreateWorkspaceRequest,
      concurrent.futures.Future[daemon_pb2.Workspace],
    ]
  ],
  marker: str,
) -> None:
  (tmp_path / marker).mkdir()
  nested = tmp_path / 'nested'
  nested.mkdir()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=nested,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    is_trusted_directory=lambda _: True,
    settings=config.Config(tmp_path / 'config.json'),
  )
  try:
    choice = controller.new_workspace_selector().choices[2]
    assert choice.detail == f'From {tmp_path}'
    await controller.create_workspace('worktree')
    request, response = await asyncio.to_thread(
      workspace_creations.get, True, 5
    )
    response.set_exception(ValueError('fixture does not provision checkouts'))
    assert request.worktree_from == str(nested)
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_worktree_choice_uses_selected_local_source_then_first_open_local(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_creations: queue.Queue[
    tuple[
      daemon_pb2.CreateWorkspaceRequest,
      concurrent.futures.Future[daemon_pb2.Workspace],
    ]
  ],
) -> None:
  repository = tmp_path / 'other-project'
  repository.mkdir()
  (repository / '.git').mkdir()
  registered = _register_workspace(workspace_client, repository)
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    is_trusted_directory=lambda _: True,
    settings=config.Config(tmp_path / 'config.json'),
  )
  try:
    await controller.open_session(registered.session_id)
    assert (
      controller.new_workspace_selector().choices[2].detail
      == f'From {repository}'
    )
    controller.selected = workspaces.EmptyWorkspace()
    assert (
      controller.new_workspace_selector().choices[2].detail
      == f'From {repository}'
    )
    await controller.create_workspace('worktree')
    request, response = await asyncio.to_thread(
      workspace_creations.get, True, 5
    )
    response.set_exception(ValueError('fixture does not provision checkouts'))
    assert request.worktree_from == str(repository)
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_plain_directory_allows_new_but_rejects_worktree_before_pending_entry(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  # The filesystem root has no repository ancestors; the RPC fixture never executes there.
  directory = pathlib.Path('/')
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=directory,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    is_trusted_directory=lambda _: True,
    settings=config.Config(tmp_path / 'config.json'),
  )
  try:
    assert (
      controller.new_workspace_selector().choices[2].detail
      == 'Requires Git or Jujutsu · /'
    )
    with pytest.raises(ValueError, match='Worktrees require Git or Jujutsu'):
      await controller.create_workspace('worktree')
    assert controller.visible_workspaces() == ()
    assert not workspace_snapshot.workspaces
    identity = await controller.create_workspace('normal')
    assert controller.visible_workspaces()[0].identity == identity
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_worktree_does_not_fall_back_from_a_selected_sandbox(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  registered = _register_workspace(workspace_client, tmp_path)
  workspace_snapshot.workspaces[0].kind = daemon_pb2.WORKSPACE_KIND_SANDBOX
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    is_trusted_directory=lambda _: True,
    settings=config.Config(tmp_path / 'config.json'),
  )
  try:
    await controller.open_session(registered.session_id)
    before = workspace_snapshot.SerializeToString()
    assert (
      controller.new_workspace_selector().choices[2].detail
      == 'Requires a host repository, not a sandbox'
    )
    with pytest.raises(ValueError, match='host repository, not a sandbox'):
      await controller.create_workspace('worktree')
    assert len(controller.visible_workspaces()) == 1
    assert workspace_snapshot.SerializeToString() == before
  finally:
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('is_worktree', [False, True])
async def test_drop_unmanaged_directory_is_rejected(
  tmp_path: pathlib.Path,
  is_worktree: bool,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_deletions: queue.Queue[
    tuple[daemon_pb2.DeleteWorkspaceRequest, concurrent.futures.Future[None]]
  ],
  workspace_deleted: queue.Queue[str],
) -> None:
  folder = _register_workspace(workspace_client, tmp_path)
  workspace_snapshot.workspaces[0].is_worktree = is_worktree
  other = _register_workspace(workspace_client, tmp_path / 'other')
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    is_trusted_directory=lambda _: True,
    settings=config.Config(tmp_path / 'config.json'),
  )
  try:
    await controller.open_session(other.session_id)
    await controller.open_session(folder.session_id)
    with pytest.raises(ValueError, match='Only managed worktrees'):
      controller.drop_confirmation()
    assert not await controller.drop()
    assert controller.selected.id == folder.id
    assert workspace_deleted.empty()
    assert workspace_deletions.empty()
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_drop_only_failed_creation_returns_to_landing_with_draft(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_creations: queue.Queue[
    tuple[
      daemon_pb2.CreateWorkspaceRequest,
      concurrent.futures.Future[daemon_pb2.Workspace],
    ]
  ],
) -> None:
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    store=workspace_client,
    settings=config.Config(tmp_path / 'config.json'),
    reasoning_effort='medium',
    create_agent=factory,
    is_trusted_directory=lambda _: True,
  )
  try:
    (tmp_path / '.git').mkdir()
    identity = await controller.create_workspace('worktree')
    request, response = await asyncio.to_thread(
      workspace_creations.get, True, 5
    )
    assert await controller.select_workspace(identity)
    pending = controller.selected
    assert isinstance(pending, workspaces.PendingWorkspace)
    pending.draft = actions.ComposerState('unsent draft', 4)
    workspace_snapshot.workspaces.add(
      id=request.workspace_id,
      name='failed',
      directory=str(tmp_path),
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_FAILED,
      error='checkout failed',
    )
    response.set_exception(ValueError('checkout failed'))
    creations = controller._creation_tasks  # pyright: ignore[reportPrivateUsage] - await the rejected creation.
    await asyncio.gather(*tuple(creations), return_exceptions=True)
    await _refresh(controller)

    assert await controller.drop()
    await _refresh(controller)
    assert controller.visible_workspaces() == ()
    assert isinstance(controller.selected, workspaces.EmptyWorkspace)
    assert controller.selected.draft == actions.ComposerState('unsent draft', 4)
    assert not workspace_snapshot.workspaces
    assert factory.agents == []
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_picker_identity_and_tab_positions_survive_discovery_reordering(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  primary = workspace_client.create(
    daemon_pb2.CreateWorkspaceRequest(
      workspace_id=str(uuid.uuid4()), name='main', directory=str(main)
    )
  )
  feature_record = workspace_client.create(
    daemon_pb2.CreateWorkspaceRequest(
      workspace_id=str(uuid.uuid4()), name='feature', directory=str(feature)
    )
  )
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config()
    | {'conversation_id': primary.session_id, 'create_session': False},
    session_id=primary.session_id,
    store=workspace_client,
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    is_trusted_directory=lambda _: True,
  )
  try:
    await _refresh(controller)
    await _open_saved_workspace(controller, feature_record.session_id)
    identity = controller.visible_workspaces()[1].identity
    workspace_snapshot.workspaces.reverse()
    await _refresh(controller)

    assert await controller.select_workspace(identity)
    assert controller.selected is controller.visible_workspaces()[1]
    assert controller.visible_workspaces()[1].label == 'feature'

    del workspace_snapshot.workspaces[0]
    await _refresh(controller)
    assert not await controller.select_workspace(identity)
    assert 'no longer available' in factory.agents[0].errors[-1]
    assert factory.agents[1].closed
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_shared_session_switch_preserves_separate_drafts_and_resumes_old_history(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
) -> None:
  registered = _register_workspace(workspace_client, tmp_path)
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    session_id=registered.session_id,
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    store=workspace_client,
    is_trusted_directory=lambda _: True,
  )
  factory.agents[0].draft = actions.ComposerState('old draft', 3)
  fresh = workspace_client.switch_session(registered.id, registered.session_id)
  try:
    await _refresh(controller)
    await asyncio.wait_for(factory.agents[0].closed_event.wait(), 5)
    assert isinstance(controller.selected, workspaces.AgentWorkspace)
    assert controller.selected.id == registered.id
    assert controller.selected.agent.conversation_id == fresh.session_id
    assert controller.selected.draft is None
    factory.agents[-1].draft = actions.ComposerState('new draft', 4)
    await controller.open_session(registered.session_id)
    assert isinstance(controller.selected, workspaces.AgentWorkspace)
    assert controller.selected.id == registered.id
    assert controller.selected.agent.conversation_id == registered.session_id
    assert controller.selected.draft == actions.ComposerState('old draft', 3)
    assert len(controller.visible_workspaces()) == 1
    assert not factory.agents[-1].started
    await controller.open_session(fresh.session_id)
    assert isinstance(controller.selected, workspaces.AgentWorkspace)
    assert controller.selected.draft == actions.ComposerState('new draft', 4)
    assert factory.agents[-1].submitted == []
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_shared_last_close_preserves_each_draft_and_reopens_identity(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_address: str,
) -> None:
  registered = _register_workspace(workspace_client, tmp_path)
  first_factory = _AgentFactory()
  second_factory = _AgentFactory()
  first = workspace_controller.WorkspaceController(
    _config(),
    session_id=registered.session_id,
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=first_factory,
    store=daemon_client.Client(workspace_address),
    is_trusted_directory=lambda _: True,
  )
  second = workspace_controller.WorkspaceController(
    _config(),
    session_id=registered.session_id,
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=second_factory,
    store=daemon_client.Client(workspace_address),
    is_trusted_directory=lambda _: True,
  )
  original = first.selected
  assert isinstance(original, workspaces.AgentWorkspace)
  first_factory.agents[0].draft = actions.ComposerState('first draft', 3)
  second_factory.agents[0].draft = actions.ComposerState('second draft', 6)
  try:
    await first.initialize()
    await second.initialize()
    assert await first.close_workspace(original.id)
    await asyncio.wait_for(second_factory.agents[0].closed_event.wait(), 5)
    assert first.visible_workspaces() == second.visible_workspaces() == ()
    assert isinstance(first.selected, workspaces.EmptyWorkspace)
    assert isinstance(second.selected, workspaces.EmptyWorkspace)
    assert first.selected.draft == actions.ComposerState('first draft', 3)
    assert second.selected.draft == actions.ComposerState('second draft', 6)
    first.selected.draft = actions.ComposerState('edited first', 5)
    second.selected.draft = actions.ComposerState('edited second', 4)

    await first.open_session(original.agent.conversation_id)
    await _refresh(second)
    assert second.visible_workspaces() == ()
    assert isinstance(second.selected, workspaces.EmptyWorkspace)
    assert second.selected.draft == actions.ComposerState('edited second', 4)
    await second.open_session(original.agent.conversation_id)

    assert isinstance(first.selected, workspaces.AgentWorkspace)
    assert isinstance(second.selected, workspaces.AgentWorkspace)
    assert first.selected.id == second.selected.id == original.id
    assert (
      first.selected.agent.conversation_id == original.agent.conversation_id
    )
    assert first.selected.draft == actions.ComposerState('edited first', 5)
    assert second.selected.draft == actions.ComposerState('edited second', 4)
    assert (
      first_factory.agents[-1].submitted
      == second_factory.agents[-1].submitted
      == []
    )
    assert not second_factory.agents[-1].started
  finally:
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_open_folder_validation_and_trust(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
) -> None:
  registered = _register_workspace(workspace_client, tmp_path)
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    session_id=registered.session_id,
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    store=workspace_client,
    is_trusted_directory=lambda _: False,
  )
  folder = tmp_path / 'untrusted'
  folder.mkdir()
  with pytest.raises(ValueError, match='not trusted'):
    await controller.open_folder(tmp_path, folder)
  (tmp_path / '.git').mkdir()
  with pytest.raises(ValueError, match='Trust the workspace directory'):
    await controller.create_worktree_workspace(tmp_path)
  with pytest.raises(FileNotFoundError):
    await controller.open_folder(tmp_path, pathlib.Path('missing'))
  (tmp_path / 'file.txt').write_text('not a directory')
  with pytest.raises(ValueError, match='not a directory'):
    await controller.open_folder(tmp_path, pathlib.Path('file.txt'))
  assert len(factory.agents) == 1
  await controller.close()
