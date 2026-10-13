from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import datetime
import functools
import pathlib
import sqlite3
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import cast, override
from unittest import mock

import pytest
from google.protobuf import timestamp_pb2
from mirror.protocol import (
  agent_pb2,
  client_pb2,
  mirror_pb2,
)
from prompt_toolkit.application import current
from prompt_toolkit.input import defaults
from prompt_toolkit.layout import containers, controls
from prompt_toolkit.output import base

from mirror import config, project
from mirror.cli import (
  actions,
  cli,
  cli_store,
  effects,
  extension,
  models,
  rpc_client,
  workspace_controller,
  workspace_tags,
  workspaces,
  worktrees,
)
from mirror.cli.components import drop, failure, worktree_tabs


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
    self.action: str | None = None
    self.started = False
    self.closed = False
    self.errors: list[str] = []
    self.entries: tuple[agent_pb2.AgentMessage, ...] = ()
    self.submitted: list[str] = []
    self.instruction_submitted = asyncio.Event()
    self.activation_finished = asyncio.Event()

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

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    return self.entries

  def submit_instruction(self, instruction: str) -> None:
    self.submitted.append(instruction)
    self.instruction_submitted.set()

  async def start(self) -> None:
    self.started = True

  async def activate(self, *, show_context: bool) -> str | None:
    del show_context
    await asyncio.sleep(0)  # The real prompt application yields while active.
    self.activation_finished.set()
    return self.action

  async def close(self) -> None:
    self.closed = True


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


@dataclasses.dataclass
class _Creation:
  source: pathlib.Path
  name: str | None
  result: concurrent.futures.Future[worktrees.Worktree]


class _WorktreeCreator:
  """Expose blocking backend calls as independently completable async requests."""

  def __init__(self) -> None:
    self.loop = asyncio.get_running_loop()
    self.requests: asyncio.Queue[_Creation] = asyncio.Queue()

  def __call__(
    self, source: pathlib.Path, name: str | None
  ) -> worktrees.Worktree:
    result: concurrent.futures.Future[worktrees.Worktree] = (
      concurrent.futures.Future()
    )
    self.loop.call_soon_threadsafe(
      self.requests.put_nowait, _Creation(source, name, result)
    )
    return result.result(timeout=5)

  async def next(self) -> _Creation:
    return await asyncio.wait_for(self.requests.get(), timeout=5)


class _RecordingOutput(base.DummyOutput):
  """Signal rendered terminal output without polling or sleeping."""

  def __init__(self) -> None:
    self.frames: asyncio.Queue[str] = asyncio.Queue()
    self.parts: list[str] = []

  @override
  def write(self, data: str) -> None:
    self.parts.append(data)

  @override
  def flush(self) -> None:
    if self.parts:
      self.frames.put_nowait(''.join(self.parts))
      self.parts.clear()

  async def wait_for_text(self, text: str) -> None:
    rendered = ''
    async with asyncio.timeout(5):
      while text not in rendered:
        rendered += await self.frames.get()


def _fixture_worktree(path: pathlib.Path) -> worktrees.Worktree:
  root = project.repository_root(path.resolve()) or path.resolve()
  return worktrees.Worktree(root, root.name)


@pytest.fixture(autouse=True)
def current_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
  """Give synthetic checkouts their fixture identity without invoking VCS."""
  monkeypatch.setattr(worktrees, 'current_worktree', _fixture_worktree)


@pytest.fixture
def managed_root(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
  monkeypatch.setenv('HOME', str(tmp_path))
  root = tmp_path / '.mirror' / 'worktrees'
  root.mkdir(parents=True)
  return root


async def _refresh(
  controller: workspace_controller.WorkspaceController,
) -> None:
  controller.refresh()
  task = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
  assert task is not None
  await task
  await asyncio.sleep(0)


async def _delete_remote(_: str) -> None:
  pass


def _config() -> Mapping[str, object]:
  return {
    'conversation_id': 'conv-1',
    'model': 'm-test',
    'approval_mode': 'auto',
    'create_session': True,
  }


def _repl() -> cli._Repl:
  return cli._Repl(
    [sys.executable],
    _config(),
    session_id='test-session',
    workspace_effects=asyncio.Queue(),
    workspace_controller=mock.Mock(
      spec=workspace_controller.WorkspaceController
    ),
    extension_registry=extension.ExtensionRegistry(),
  )


def _render(
  controller: workspace_controller.WorkspaceController,
) -> list[tuple[str, str]]:
  return worktree_tabs.render_tabs(
    controller.visible_workspaces(), controller.selected.identity
  )


async def _dispatch_and_wait(
  repl: cli._Repl,
  action: actions.Action,
  controller: workspace_controller.WorkspaceController,
) -> None:
  await asyncio.gather(*repl.dispatch(action))
  queue = repl._workspace_effects
  assert queue is not None
  while not queue.empty():
    agent, effect = queue.get_nowait()
    await cli.handle_workspace_effect(controller, agent, effect)


async def _run(controller: workspace_controller.WorkspaceController) -> None:
  async def activate(
    workspace: workspaces.Workspace, show_context: bool
  ) -> str | None:
    if isinstance(workspace, workspaces.PendingWorkspace):
      return await cli._activate_pending(workspace, containers.Window(height=0))
    assert isinstance(workspace, workspaces.AgentWorkspace)
    await workspace.start()
    agent = workspace.agent
    assert isinstance(agent, _Agent)
    return await agent.activate(show_context=show_context)

  await cli.run_workspaces(controller, activate, asyncio.Queue())


def _failure_text(repl: cli._Repl) -> str:
  window = failure.Failure().render(repl._state)
  assert isinstance(window.content, controls.FormattedTextControl)
  assert not callable(window.content.text)
  return ''.join(text for _, text in window.content.text)


@pytest.mark.asyncio
async def test_creation_keeps_repl_available(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  observed: list[tuple[bool, bool, str]] = []
  repl: cli._Repl

  class Controller:
    async def create(self, name: str | None = None) -> bool:
      assert name == 'feature'
      observed.append((
        repl._state.activity.busy,
        repl._input.read_only(),
        repl._state.activity.status,
      ))
      return True

  repl = _repl()
  repl.dispatch(actions.WorkerStarted())
  exits: list[str | None] = []
  monkeypatch.setattr(
    repl._app, 'exit', lambda result=None: exits.append(result)
  )

  await _dispatch_and_wait(
    repl,
    actions.CreateWorktree('feature'),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert observed == [(False, False, '')]
  assert exits == []
  assert not repl._state.activity.busy
  assert not repl._input.read_only()
  assert repl.worktree_state is actions.WorktreeAgentState.IDLE


@pytest.mark.asyncio
async def test_creation_during_turn_preserves_running_state(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class Controller:
    async def create(self, name: str | None = None) -> bool:
      assert name == 'feature'
      assert repl._state.activity.busy
      assert repl._state.activity.status == 'thinking…'
      return True

  repl = _repl()
  repl.dispatch(actions.WorkerStarted(), actions.TurnStarted())
  exits: list[str | None] = []
  monkeypatch.setattr(
    repl._app, 'exit', lambda result=None: exits.append(result)
  )

  await _dispatch_and_wait(
    repl,
    actions.CreateWorktree('feature'),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert exits == []
  assert repl._state.activity.busy
  assert repl._state.activity.status == 'thinking…'
  assert repl.worktree_state is actions.WorktreeAgentState.RUNNING


@pytest.mark.asyncio
async def test_failed_creation_keeps_repl_available() -> None:
  class Controller:
    async def create(self, name: str | None = None) -> bool:
      del name
      return False

  repl = _repl()

  await _dispatch_and_wait(
    repl,
    actions.CreateWorktree(),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert not repl._state.activity.busy
  assert not repl._input.read_only()
  assert repl._state.activity.status == ''


@pytest.mark.asyncio
async def test_remote_command_selects_new_agent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  workspace = mirror_pb2.Workspace(
    name='project', repository_url='https://github.com/acme/project'
  )

  class Controller:
    async def create_remote(
      self,
      conversation_id: str | None = None,
      remote_workspace: mirror_pb2.Workspace | None = None,
    ) -> int | None:
      assert conversation_id == 'remote-conversation'
      assert remote_workspace == workspace
      return 1

    def can_select(self, index: int) -> bool:
      return index == 1

  repl = _repl()
  exits: list[str | None] = []
  monkeypatch.setattr(
    repl._app, 'exit', lambda result=None: exits.append(result)
  )

  await _dispatch_and_wait(
    repl,
    actions.CreateRemote('remote-conversation', workspace),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert not repl._app.erase_when_done
  assert exits == ['select:1']


@pytest.mark.asyncio
async def test_drop_command_exits_to_selected_remaining_worktree(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  dropped = False

  class Controller:
    def drop_confirmation(
      self,
    ) -> actions.DropConfirmation | None:
      return None

    async def drop(self, *, confirmed: bool = False) -> bool:
      nonlocal dropped
      assert not confirmed
      dropped = True
      return True

  repl = _repl()
  exits: list[str | None] = []
  monkeypatch.setattr(
    repl._app, 'exit', lambda result=None: exits.append(result)
  )

  await _dispatch_and_wait(
    repl,
    actions.Drop(),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert dropped
  assert not repl._app.erase_when_done
  assert exits == ['dropped']


@pytest.mark.asyncio
async def test_drop_command_confirms_unmanaged_worktree(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[bool] = []
  confirmation = actions.DropConfirmation('feature', str(tmp_path))

  class Controller:
    def drop_confirmation(self) -> actions.DropConfirmation:
      return confirmation

    async def drop(self, *, confirmed: bool = False) -> bool:
      calls.append(confirmed)
      return True

  repl = _repl()
  exits: list[str | None] = []
  monkeypatch.setattr(
    repl._app, 'exit', lambda result=None: exits.append(result)
  )

  await _dispatch_and_wait(
    repl,
    actions.Drop(),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert calls == []
  assert repl._input.read_only()
  window = drop.DropConfirmation().render(repl._state)
  assert isinstance(window.content, controls.FormattedTextControl)
  assert not callable(window.content.text)
  panel = ''.join(text for _, text in window.content.text)
  assert 'Delete unmanaged worktree feature?' in panel
  assert str(tmp_path) in panel
  assert 'uncommitted changes will be deleted' in panel

  repl.dispatch(actions.ClearWorktreeDrop())
  assert not repl._input.read_only()

  await _dispatch_and_wait(
    repl,
    actions.Drop(),
    cast(workspace_controller.WorkspaceController, Controller()),
  )
  await _dispatch_and_wait(
    repl,
    actions.ConfirmWorktreeDrop(),
    cast(workspace_controller.WorkspaceController, Controller()),
  )

  assert calls == [True]
  assert exits == ['dropped']


@pytest.mark.asyncio
async def test_failed_agent_keeps_draft_editable_and_restarts_on_submit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._input.text = 'keep this draft'
  repl.dispatch(actions.WorkerFailed('conversation is already running'))
  calls: list[str] = []

  async def shutdown() -> None:
    calls.append('shutdown')

  async def start(*, replay: bool) -> None:
    assert not replay
    calls.append('start')
    message_id, instruction = repl._state.turn.submitted_inputs[0]
    repl._send_instruction(instruction, message_id=message_id)
    repl.dispatch(actions.WorkerStarted())

  def send(instruction: str, *, message_id: str | None = None) -> None:
    assert instruction == 'keep this draft'
    assert message_id == repl._state.turn.input_order[0]
    calls.append('send')

  monkeypatch.setattr(repl, '_shutdown_worker', shutdown)
  monkeypatch.setattr(repl, '_start_worker', start)
  monkeypatch.setattr(repl, '_send_instruction', send)

  assert not repl._input.read_only()
  assert 'Press Enter to try loading again' in _failure_text(repl)
  repl.dispatch(actions.SubmitRequested(None, None))
  await asyncio.gather(*tuple(repl._app._background_tasks))

  assert calls == ['shutdown', 'start', 'send']
  assert not repl._input.text
  assert not repl._input.read_only()


@pytest.mark.asyncio
async def test_controller_starts_only_new_remote_sessions_in_the_workspace(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)
  workspace = mirror_pb2.Workspace(
    name='project', repository_url='https://github.com/acme/project'
  )

  assert await controller.create_remote('Release', workspace) == 1
  assert (
    await controller.create_remote(
      'aaaaaaaa-0000-0000-0000-000000000000', workspace
    )
    == 2
  )

  assert factory.agents[1].spec.config['remote_workspace'] == workspace
  assert factory.agents[2].spec.config['remote_workspace'] is None


@pytest.mark.asyncio
async def test_controller_starts_remote_agent_without_creating_worktree(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()
  local_creations: list[str | None] = []
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=lambda _path, name: (
      local_creations.append(name) or worktrees.Worktree(main, 'unused')
    ),
  )
  await _refresh(controller)

  assert await controller.create_remote() == 1

  remote = factory.agents[1]
  assert remote.spec.remote
  assert remote.spec.workspace == main
  assert remote.spec.config['create_session'] is True
  assert remote.spec.config['model'] == 'server default'
  assert remote.spec.config['approval_mode'] == 'auto'
  assert remote.started
  assert local_creations == []
  assert controller.visible_workspaces()[-1].label.startswith('remote-')

  await controller.cycle(1)
  reconcile = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
  assert reconcile is not None
  await reconcile
  assert not remote.closed
  assert 'remote-' in ''.join(text for _, text, *_ in _render(controller))


@pytest.mark.asyncio
async def test_controller_creates_local_worktree_from_remote_tab(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'feature'
  main.mkdir()
  created.mkdir()
  started = threading.Event()
  release = threading.Event()
  creation_paths: list[pathlib.Path] = []

  def create(path: pathlib.Path, name: str | None) -> worktrees.Worktree:
    creation_paths.append(path)
    started.set()
    release.wait()
    return worktrees.Worktree(created, name)

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=create,
  )
  await _refresh(controller)
  factory.agents[0].reasoning_effort = 'low'
  factory.agents[0].approval_mode = 'approve'
  remote_tab = await controller.create_remote()
  assert remote_tab is not None
  assert await controller.select(remote_tab)
  factory.agents[1].reasoning_effort = 'xhigh'
  factory.agents[1].approval_mode = 'auto'

  assert await controller.create('feature')
  tasks = tuple(
    controller._worktree_creation_tasks  # pyright: ignore[reportPrivateUsage]
  )
  try:
    assert await asyncio.to_thread(started.wait, 1)
    pending = controller.visible_workspaces()
    assert pending[1].label == 'feature'
    assert pending[1].state is actions.WorktreeAgentState.CREATING
    assert pending[2].remote
  finally:
    release.set()
    await asyncio.gather(*tasks)

  assert creation_paths == [main]
  assert factory.agents[2].spec.reasoning_effort == 'low'
  assert factory.agents[2].spec.config['approval_mode'] == 'approve'
  rendered = _render(controller)
  text = ''.join(fragment for _, fragment, *_ in rendered)
  assert controller.visible_workspaces()[1].label == 'feature'
  assert controller.visible_workspaces()[2].remote
  assert '@remote-' in text
  assert any(
    style == 'class:worktree-tab-selected' and '@remote-' in fragment
    for style, fragment, *_ in rendered
  )


@pytest.mark.asyncio
async def test_controller_drops_remote_workspace(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  deleted: list[str] = []

  async def delete_remote(session_id: str) -> None:
    deleted.append(session_id)

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    remote_enabled=True,
    delete_remote_session=delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)
  tab = await controller.create_remote('aaaaaaaa-0000-0000-0000-000000000000')
  assert tab == 1
  assert await controller.select(tab)

  assert controller.drop_confirmation() is None
  assert await controller.drop()
  tasks = tuple(
    controller._workspace_deletion_tasks  # pyright: ignore[reportPrivateUsage]
  )
  await asyncio.gather(*tasks)
  await asyncio.sleep(0)

  assert deleted == ['aaaaaaaa-0000-0000-0000-000000000000']
  assert factory.agents[1].closed
  assert 'remote-' not in ''.join(text for _, text, *_ in _render(controller))


@pytest.mark.asyncio
async def test_controller_does_not_discover_remotes_on_startup(
  tmp_path: pathlib.Path,
) -> None:
  async def list_remote_sessions() -> Sequence[str]:
    pytest.fail('Remote discovery must be explicitly requested')

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'cli.db',
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    list_remote_sessions=list_remote_sessions,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
  )
  await _refresh(controller)
  await _run(controller)
  assert len(factory.agents) == 1
  assert factory.agents[0].errors == []


@pytest.mark.asyncio
async def test_controller_loads_remote_sessions_as_lazy_tabs(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_alias(
      str(main.resolve()),
      'aaaaaaaa-0000-0000-0000-000000000000',
      'Named remote',
    )

  listed: list[None] = []

  async def list_remote_sessions() -> Sequence[str]:
    listed.append(None)
    return (
      'aaaaaaaa-0000-0000-0000-000000000000',
      'bbbbbbbb-0000-0000-0000-000000000000',
    )

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=database,
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    list_remote_sessions=list_remote_sessions,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)

  repl = _repl()
  await _dispatch_and_wait(repl, actions.LoadRemoteSessions(), controller)
  await _dispatch_and_wait(repl, actions.LoadRemoteSessions(), controller)
  await asyncio.sleep(0)
  await _dispatch_and_wait(repl, actions.LoadRemoteSessions(), controller)
  await _run(controller)

  assert listed == [None]
  assert len(factory.agents) == 3
  assert not factory.agents[1].started
  assert not factory.agents[2].started
  assert factory.agents[1].spec.config['create_session'] is False
  assert factory.agents[2].spec.config['create_session'] is False
  assert controller.visible_workspaces()[1].label == 'Named remote'
  assert controller.visible_workspaces()[2].label == 'remote-bbbbbbbb'
  assert ''.join(text for _, text, *_ in _render(controller)).split('\n')[
    -1
  ] == (' main ·   @Named remote ↻   @remote-bbbbbbbb ↻   Tab / Shift+Tab')


@pytest.mark.asyncio
async def test_rewound_remote_tab_keeps_preferences_and_deletes_the_child(
  tmp_path: pathlib.Path,
) -> None:
  parent = 'aaaaaaaa-0000-0000-0000-000000000000'
  child = 'bbbbbbbb-0000-0000-0000-000000000000'
  deleted: list[str] = []

  async def delete_remote(session_id: str) -> None:
    deleted.append(session_id)

  factory = _AgentFactory()
  database = tmp_path / 'cli.db'
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    database=database,
    remote_enabled=True,
    delete_remote_session=delete_remote,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
  )
  await _refresh(controller)
  assert await controller.create_remote(parent) == 1
  assert await controller.select(1)
  controller.rename('My work')
  controller.save_preferences('high', 'approve')
  remote = factory.agents[1]
  # A rewind can finish after the user has selected another tab.
  assert await controller.select(0)
  controller.retarget_remote(remote, child)
  assert await controller.create_remote(child) == 1
  assert len(factory.agents) == 2
  assert controller.visible_workspaces()[1].label == 'My work'
  with cli_store.Store(database) as store:
    assert store.worktree_overlays(str(tmp_path)) == {
      tmp_path.name: cli_store.WorktreeOverlay(
        preferences=client_pb2.WorkspacePreferences(
          model='m-test', model_api='', model_base_url=''
        ),
      ),
      child: cli_store.WorktreeOverlay(
        alias='My work',
        preferences=client_pb2.WorkspacePreferences(
          reasoning_effort='high',
          approval_mode=agent_pb2.APPROVAL_MODE_HITL,
        ),
      ),
    }
  assert await controller.create_remote(parent) == 2
  assert await controller.select(1)
  assert await controller.drop()
  await _run(controller)
  assert deleted == [child]
  assert remote.closed


@pytest.mark.asyncio
async def test_reopened_remote_tab_keeps_its_server_protocol(
  tmp_path: pathlib.Path,
) -> None:
  session = 'aaaaaaaa-0000-0000-0000-000000000000'
  database = tmp_path / 'cli.db'
  before = _AgentFactory()
  after = _AgentFactory()

  def controller(
    factory: _AgentFactory,
  ) -> workspace_controller.WorkspaceController:
    return workspace_controller.WorkspaceController(
      _config(),
      workspace=tmp_path,
      reasoning_effort='medium',
      create_agent=factory,
      database=database,
      remote_enabled=True,
      delete_remote_session=_delete_remote,
      discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
    )

  first = controller(before)
  await _refresh(first)
  assert await first.create_remote(session) == 1
  first.save_model(
    before.agents[1],
    models.ModelOption(
      'claude-opus-4-8', 'claude-opus-4-8', api='anthropic-messages'
    ),
  )
  second = controller(after)
  await _refresh(second)
  assert await second.create_remote(session) == 1

  assert after.agents[1].spec.config['model'] == 'claude-opus-4-8'
  assert after.agents[1].spec.config['remote_model_api'] == (
    'anthropic-messages'
  )


@pytest.mark.asyncio
async def test_controller_creates_named_remote_conversation(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'cli.db',
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)

  assert await controller.create_remote('Remote conversation') == 1

  remote = factory.agents[1]
  session_id = str(remote.spec.config['conversation_id'])
  assert str(uuid.UUID(session_id)) == session_id
  assert remote.spec.config['create_session'] is True
  assert controller.visible_workspaces()[1].label == 'Remote conversation'
  with cli_store.Store(tmp_path / 'cli.db') as store:
    assert store.worktree_overlays(str(main.resolve()))[session_id].alias == (
      'Remote conversation'
    )


@pytest.mark.asyncio
async def test_controller_renders_agent_states(tmp_path: pathlib.Path) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)

  factory.agents[0].worktree_state = actions.WorktreeAgentState.RUNNING
  factory.agents[1].worktree_state = actions.WorktreeAgentState.ATTENTION
  rendered = _render(controller)

  assert ''.join(text for _, text, *_ in rendered).split('\n')[-1] == (
    ' main ●   feature !   Tab / Shift+Tab'
  )
  assert rendered[0] == ('class:worktree-tab-selected', ' main ')
  assert '\n' not in ''.join(text for _, text, *_ in rendered)
  assert (
    controller.visible_workspaces()[1].state
    is actions.WorktreeAgentState.ATTENTION
  )

  feature.rmdir()
  assert _render(controller) == []


@pytest.mark.asyncio
async def test_rename_aliases_default_and_persists_in_footer(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  default = tmp_path / 'default'
  moved_default = tmp_path / 'moved-default'
  feature = tmp_path / 'feature'
  default.mkdir()
  moved_default.mkdir()
  feature.mkdir()
  database = tmp_path / 'state.db'
  monkeypatch.setattr(
    workspace_controller.workspace_identity,
    'git_workspace_key',
    lambda _: 'stable-repository',
  )
  discovered = (
    worktrees.Worktree(
      default,
      'default',
      worktrees.WorktreeIdentity('jj', 'default'),
    ),
    worktrees.Worktree(
      feature,
      'feature',
      worktrees.WorktreeIdentity('jj', 'feature'),
    ),
  )
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=default,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=database,
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: discovered,
  )
  await _refresh(controller)

  controller.save_workspace_tag(
    workspace_tags.WorkspaceTag('blocked', 'Blocked')
  )
  controller.set_workspace_tag('default', 'blocked')
  assert controller.rename('  Primary   work  ') == 'Primary work'
  controller.save_preferences('high', 'approve')
  assert controller.visible_workspaces()[0].label == 'Primary work'
  assert ''.join(text for _, text, *_ in _render(controller)).split('\n')[
    -1
  ] == (' Primary work ·   feature ·   Tab / Shift+Tab')

  monkeypatch.setattr(
    worktrees,
    'current_worktree',
    mock.Mock(
      return_value=worktrees.Worktree(
        moved_default, 'default', worktrees.WorktreeIdentity('jj', 'default')
      )
    ),
  )
  restored = _AgentFactory()
  restored_controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=moved_default,
    reasoning_effort='medium',
    create_agent=restored,
    database=database,
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (
      worktrees.Worktree(
        moved_default,
        'default',
        worktrees.WorktreeIdentity('jj', 'default'),
      ),
      discovered[1],
    ),
  )
  await _refresh(restored_controller)
  assert restored_controller.workspace_tag(
    'default'
  ) == workspace_tags.WorkspaceTag('blocked', 'Blocked')
  assert restored_controller.visible_workspaces()[0].label == 'Primary work'
  assert restored.agents[0].spec.reasoning_effort == 'high'
  assert restored.agents[0].spec.config['approval_mode'] == 'approve'
  assert restored.agents[0].spec.config['conversation_id'] == 'conv-1'


@pytest.mark.asyncio
async def test_rename_rejects_duplicate_aliases_and_worktree_ids(
  tmp_path: pathlib.Path,
) -> None:
  default = tmp_path / 'default'
  feature = tmp_path / 'feature'
  default.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=default,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=tmp_path / 'state.db',
    discover=lambda _: (
      worktrees.Worktree(default, 'default'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)
  controller.rename('Shared alias')
  assert await controller.select(1)

  with pytest.raises(ValueError, match='already in use'):
    controller.rename('shared ALIAS')
  with pytest.raises(ValueError, match='already in use'):
    controller.rename('default')


@pytest.mark.asyncio
async def test_reconcile_keeps_overlay_for_externally_removed_worktree(
  tmp_path: pathlib.Path,
) -> None:
  default = tmp_path / 'default'
  feature = tmp_path / 'feature'
  default.mkdir()
  feature.mkdir()
  database = tmp_path / 'state.db'
  repository_key = str(default.resolve())
  with cli_store.Store(database) as store:
    store.set_worktree_alias(
      repository_key,
      str(feature.resolve()),
      'Reusable alias',
    )
  available = [
    worktrees.Worktree(default, 'default'),
    worktrees.Worktree(feature, 'feature'),
  ]
  discovery_fails = False

  def discover(_: pathlib.Path) -> Sequence[worktrees.Worktree]:
    if discovery_fails:
      raise worktrees.WorktreeError('temporary failure')
    return available

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=default,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=database,
    discover=discover,
  )
  await _refresh(controller)
  assert controller.visible_workspaces()[1].label == 'Reusable alias'
  expected_overlays = {
    'default': cli_store.WorktreeOverlay(
      preferences=client_pb2.WorkspacePreferences(
        model='m-test', model_api='', model_base_url=''
      ),
    ),
    'feature': cli_store.WorktreeOverlay(
      alias='Reusable alias',
      preferences=client_pb2.WorkspacePreferences(
        model='m-test', model_api='', model_base_url=''
      ),
    ),
  }
  with cli_store.Store(database) as store:
    assert store.worktree_overlays(repository_key) == expected_overlays

  discovery_fails = True
  await controller._reconcile()  # pyright: ignore[reportPrivateUsage]
  with cli_store.Store(database) as store:
    assert store.worktree_overlays(repository_key) == expected_overlays

  discovery_fails = False
  available.pop()
  feature.rmdir()
  await controller._reconcile()  # pyright: ignore[reportPrivateUsage]

  assert factory.agents[1].closed
  with cli_store.Store(database) as store:
    assert store.worktree_overlays(repository_key) == expected_overlays

  restored = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=default,
    reasoning_effort='medium',
    create_agent=restored,
    database=database,
    discover=lambda _: available,
  )
  await _refresh(controller)
  assert len(restored.agents) == 1


@pytest.mark.asyncio
async def test_rename_during_turn_updates_footer_without_touching_turn(
  tmp_path: pathlib.Path,
) -> None:
  default = tmp_path / 'default'
  feature = tmp_path / 'feature'
  default.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=default,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=tmp_path / 'state.db',
    discover=lambda _: (
      worktrees.Worktree(default, 'default'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)
  repl = _repl()
  repl.dispatch(actions.WorkerStarted(), actions.TurnStarted())
  await _dispatch_and_wait(
    repl, actions.RenameWorktree('Running work'), controller
  )

  assert repl._state.activity.busy
  assert repl._state.activity.status == 'thinking…'
  assert not repl._state.turn.queued
  assert 'Running work' in ''.join(text for _, text, *_ in _render(controller))


@pytest.mark.asyncio
async def test_continue_binds_and_lazily_starts_each_worktree_session(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'main-saved',
      entry_id='main-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='main work',
      cwd=str(main),
      model='m-test',
    )
    store.record_user_message(
      'feature-saved',
      entry_id='feature-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='feature work',
      cwd=str(feature),
      model='m-test',
    )
  config = dict(_config()) | {
    'conversation_id': 'main-saved',
    'create_session': False,
  }
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    config,
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=database,
    continue_recent=True,
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)

  assert ''.join(text for _, text, *_ in _render(controller)).split('\n')[
    -1
  ] == (' main ↻   feature ↻   Tab / Shift+Tab')
  assert factory.agents[0].spec.active
  assert not factory.agents[1].spec.active
  assert factory.agents[1].spec.config['conversation_id'] == 'feature-saved'
  assert factory.agents[1].spec.config['create_session'] is False
  assert not controller.can_select(0)
  assert controller.can_select(1)
  assert not controller.can_select(2)
  factory.agents[0].action = 'select:1'

  await _run(controller)

  assert factory.agents[0].started
  assert factory.agents[1].started


@pytest.mark.asyncio
async def test_continue_uses_cli_store(tmp_path: pathlib.Path) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'feature-saved',
      entry_id='entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='feature work',
      cwd=str(feature),
      model='m-test',
    )
  factory = _AgentFactory()

  controller = workspace_controller.WorkspaceController(
    dict(_config()) | {'conversation_id': 'main', 'create_session': True},
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=database,
    continue_recent=True,
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)

  assert factory.agents[1].spec.config['conversation_id'] == 'feature-saved'
  assert factory.agents[1].spec.config['create_session'] is False


@pytest.mark.asyncio
async def test_controller_reconciles_a_deleted_selected_worktree(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  deleted = tmp_path / 'deleted'
  main.mkdir()
  deleted.mkdir()
  discovered = [
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(deleted, 'deleted'),
  ]
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=deleted,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: tuple(discovered),
  )
  await _refresh(controller)
  deleted_agent = factory.agents[0]
  deleted.rmdir()
  discovered.pop()

  await controller.cycle(1)
  task = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
  assert task is not None
  await task

  assert deleted_agent.closed
  assert _render(controller) == []


@pytest.mark.asyncio
async def test_controller_cycles_before_reconciliation_finishes(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  items = (
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(feature, 'feature'),
  )
  started = threading.Event()
  release = threading.Event()
  calls = 0

  def discover(_: pathlib.Path) -> Sequence[worktrees.Worktree]:
    nonlocal calls
    calls += 1
    if calls > 1:
      started.set()
      release.wait()
    return items

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=discover,
  )
  await _refresh(controller)

  task: asyncio.Task[None] | None = None
  try:
    await asyncio.wait_for(controller.cycle(1), timeout=0.1)
    assert any(
      style == 'class:worktree-tab-selected' and 'feature' in text
      for style, text, *_ in _render(controller)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
    assert task is not None and not task.done()
  finally:
    release.set()
  await task


@pytest.mark.asyncio
async def test_controller_does_not_cycle_or_reconcile_one_worktree(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)

  assert not controller.can_cycle()
  await controller.cycle(1)

  assert controller._reconcile_task is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_controller_refresh_discovers_external_worktree(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  available = [worktrees.Worktree(main, 'main')]
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: available,
  )
  await _refresh(controller)
  available.append(worktrees.Worktree(feature, 'feature'))

  controller.refresh()
  task = controller._reconcile_task  # pyright: ignore[reportPrivateUsage]
  assert task is not None
  await task

  assert controller.can_cycle()
  assert controller.visible_workspaces()[1].label == 'feature'
  assert len(factory.agents) == 2


@pytest.mark.asyncio
async def test_controller_creates_a_worktree_from_selected_agent_settings(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  workspace = main / 'package'
  created = tmp_path / 'created'
  workspace.mkdir(parents=True)
  (main / '.git').mkdir()
  created.mkdir()
  started = threading.Event()
  release = threading.Event()

  def create(_: pathlib.Path, name: str | None) -> worktrees.Worktree:
    assert name == 'release-prep'
    started.set()
    release.wait()
    return worktrees.Worktree(created, name)

  def discover(_: pathlib.Path) -> Sequence[worktrees.Worktree]:
    items = [worktrees.Worktree(main, 'main')]
    if started.is_set():
      items.append(worktrees.Worktree(created, 'release-prep'))
    return items

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=workspace,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    discover=discover,
    create=create,
  )
  await _refresh(controller)
  factory.agents[0].approval_mode = 'approve'
  factory.agents[0].reasoning_effort = 'high'
  reconcile = asyncio.create_task(asyncio.Event().wait())
  controller._reconcile_task = reconcile  # pyright: ignore[reportPrivateUsage]

  assert await controller.create('release-prep')
  tasks = tuple(
    controller._worktree_creation_tasks  # pyright: ignore[reportPrivateUsage]
  )
  try:
    assert await asyncio.to_thread(started.wait, 1)
    assert reconcile.cancelled()
    await controller.cycle(1)
    assert controller._reconcile_task is None  # pyright: ignore[reportPrivateUsage]
    assert len(factory.agents) == 1
    pending = controller.visible_workspaces()
    assert pending[1].label == 'release-prep'
    assert pending[1].state is actions.WorktreeAgentState.CREATING
  finally:
    release.set()
    await asyncio.gather(*tasks)

  created_spec = factory.agents[1].spec
  assert factory.agents[1].started
  assert created_spec.workspace == created / 'package'
  assert any(
    tab.label == 'release-prep' for tab in controller.visible_workspaces()
  )
  assert created_spec.reasoning_effort == 'high'
  assert created_spec.config['approval_mode'] == 'approve'
  assert created_spec.config['create_session'] is True
  assert created_spec.config['allow_workspace_mismatch'] is False
  assert created_spec.config['conversation_id'] != 'conv-1'
  with cli_store.Store(tmp_path / 'state.db') as store:
    assert store.worktree_overlays(str(main.resolve())) == {
      'main': cli_store.WorktreeOverlay(
        preferences=client_pb2.WorkspacePreferences(
          model='m-test', model_api='', model_base_url=''
        ),
      ),
      'release-prep': cli_store.WorktreeOverlay(
        preferences=client_pb2.WorkspacePreferences(
          reasoning_effort='high',
          approval_mode=agent_pb2.APPROVAL_MODE_HITL,
          model='m-test',
          model_api='',
          model_base_url='',
        ),
      ),
    }
  rendered = _render(controller)
  assert [tab.label for tab in controller.visible_workspaces()].count(
    'release-prep'
  ) == 1
  assert any(
    style == 'class:worktree-tab-selected' and 'release-prep' in text
    for style, text, *_ in rendered
  )
  assert controller.visible_workspaces()[1].identity == pending[1].identity


@pytest.mark.asyncio
async def test_controller_labels_unnamed_creation_as_creating(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  release = threading.Event()

  def create(_: pathlib.Path, __: str | None) -> worktrees.Worktree:
    release.wait()
    return worktrees.Worktree(created, 'generated')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=create,
  )
  await _refresh(controller)

  assert await controller.create()
  try:
    assert controller.visible_workspaces()[1].label == 'creating…'
    assert (
      controller.visible_workspaces()[1].state
      is actions.WorktreeAgentState.CREATING
    )
  finally:
    release.set()
    await asyncio.gather(
      *tuple(
        controller._worktree_creation_tasks  # pyright: ignore[reportPrivateUsage]
      )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('name', 'label'), [(None, 'creating…'), ('review', 'review')]
)
async def test_spawn_publishes_pending_tab_until_creation_finishes(
  tmp_path: pathlib.Path, name: str | None, label: str
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  started = threading.Event()
  release = threading.Event()

  def create(_: pathlib.Path, name: str | None) -> worktrees.Worktree:
    started.set()
    release.wait()
    return worktrees.Worktree(created, name or 'generated')

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=create,
  )
  await _refresh(controller)
  task = asyncio.create_task(controller.spawn_workspace(main, name, None))
  try:
    assert await asyncio.to_thread(started.wait, 1)
    assert controller.visible_workspaces()[-1].label == label
    assert (
      controller.visible_workspaces()[-1].state
      is actions.WorktreeAgentState.CREATING
    )
  finally:
    release.set()
    await task
  assert len(controller.visible_workspaces()) == 2
  assert (
    controller.visible_workspaces()[-1].state
    is not actions.WorktreeAgentState.CREATING
  )


@pytest.mark.asyncio
async def test_overlapping_spawns_keep_results_and_preferences_with_their_requests(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  sibling = tmp_path / 'sibling'
  first_path = tmp_path / 'first'
  second_path = tmp_path / 'second'
  main.mkdir()
  sibling.mkdir()
  first_path.mkdir()
  second_path.mkdir()
  factory = _AgentFactory()
  creator = _WorktreeCreator()
  roster = [
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(sibling, 'sibling'),
  ]
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    display_unmanaged_worktrees=True,
    discover=lambda _: roster,
    create=creator,
  )
  await _refresh(controller)
  factory.agents[1].model = 'sibling-model'
  factory.agents[1].reasoning_effort = 'high'
  factory.agents[1].approval_mode = 'approve'
  preferences = client_pb2.WorkspacePreferences(system_prompt='Review only.')
  first = asyncio.create_task(
    controller.spawn_workspace(
      main, None, 'First task', preferences=preferences
    )
  )
  first_request = await creator.next()
  second = asyncio.create_task(
    controller.spawn_workspace(
      sibling, None, 'Second task', preferences=preferences
    )
  )
  second_request = await creator.next()
  pending_first, pending_second = controller.visible_workspaces()[2:]
  assert pending_first.label == pending_second.label == 'creating…'
  assert pending_first.identity != pending_second.identity
  assert controller.can_select(2) and controller.can_select(3)
  assert first_request.source == main
  assert second_request.source == sibling
  assert first_request.name is second_request.name is None
  assert controller.list_workspaces(main).creating == 'creating…, creating…'
  assert await controller.select(1)
  factory.agents[0].reasoning_effort = 'low'

  roster.append(worktrees.Worktree(second_path, 'second'))
  second_request.result.set_result(roster[-1])
  second_result = await second

  assert second_result.workspace.id == 'second'
  assert second_result.workspace.path == second_path
  assert second_result.workspace.agent.model == 'sibling-model'
  assert second_result.workspace.agent.reasoning_effort == 'high'
  assert controller.visible_workspaces()[2] == pending_first
  assert controller.visible_workspaces()[3].identity == pending_second.identity
  assert not first.done()
  assert controller.selected is controller.visible_workspaces()[1]
  assert factory.agents[2].approval_mode == 'approve'
  await asyncio.wait_for(
    factory.agents[2].instruction_submitted.wait(), timeout=5
  )
  assert factory.agents[2].submitted == [
    "Message from the agent in workspace 'sibling'. Complete this request in your "
    'workspace and leave the response here for the sending agent to inspect.\n\nSecond task'
  ]

  roster.append(worktrees.Worktree(first_path, 'first'))
  first_request.result.set_result(roster[-1])
  first_result = await first

  assert first_result.workspace.id == 'first'
  assert first_result.workspace.path == first_path
  assert first_result.workspace.agent.model == 'm-test'
  assert first_result.workspace.agent.reasoning_effort == 'medium'
  assert factory.agents[3].approval_mode == 'auto'
  await asyncio.wait_for(
    factory.agents[3].instruction_submitted.wait(), timeout=5
  )
  assert factory.agents[3].submitted == [
    "Message from the agent in workspace 'main'. Complete this request in your "
    'workspace and leave the response here for the sending agent to inspect.\n\nFirst task'
  ]
  assert preferences == client_pb2.WorkspacePreferences(
    system_prompt='Review only.'
  )
  assert controller.list_workspaces(main).creating is None
  assert controller.selected is controller.visible_workspaces()[1]
  assert len(controller.visible_workspaces()) == 4
  assert len({tab.identity for tab in controller.visible_workspaces()}) == 4
  assert all(
    tab.state is not actions.WorktreeAgentState.CREATING
    for tab in controller.visible_workspaces()
  )

  await _refresh(controller)
  assert [tab.label for tab in controller.visible_workspaces()] == [
    'main',
    'sibling',
    'first',
    'second',
  ]
  assert controller.visible_workspaces()[2].identity == pending_first.identity
  assert controller.visible_workspaces()[3].identity == pending_second.identity


@pytest.mark.asyncio
async def test_overlapping_failure_preserves_pending_tab_and_remote_selection(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  factory = _AgentFactory()
  creator = _WorktreeCreator()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=creator,
  )
  first = asyncio.create_task(controller.spawn_workspace(main, 'review', None))
  first_request = await creator.next()
  second = asyncio.create_task(controller.spawn_workspace(main, 'review', None))
  second_request = await creator.next()
  first_tab, second_tab = controller.visible_workspaces()[1:]
  assert first_tab.identity != second_tab.identity
  assert await controller.create_remote() == 3
  assert controller.can_select(1)
  assert await controller.select(2)
  assert controller.selected is controller.visible_workspaces()[2]
  assert controller.can_select(3)
  assert await controller.select(3)
  remote_identity = controller.visible_workspaces()[3].identity
  assert (
    worktree_tabs.workspace_choices(controller.visible_workspaces())[-1].value
    == remote_identity
  )
  await controller.cycle(-1)
  assert controller.selected is controller.visible_workspaces()[2]
  await controller.cycle(1)
  assert controller.selected is controller.visible_workspaces()[3]

  second_request.result.set_exception(
    worktrees.WorktreeError('name already exists')
  )
  with pytest.raises(worktrees.WorktreeError, match='name already exists'):
    await second
  assert controller.visible_workspaces()[1] == first_tab
  assert controller.visible_workspaces()[2].identity == second_tab.identity
  assert (
    controller.visible_workspaces()[2].state
    is actions.WorktreeAgentState.FAILED
  )
  assert controller.visible_workspaces()[3].identity == remote_identity
  assert controller.selected is controller.visible_workspaces()[3]
  assert controller.list_workspaces(main).creating == 'review'

  first_request.result.set_result(worktrees.Worktree(created, 'review'))
  await first

  assert [tab.label for tab in controller.visible_workspaces()[:2]] == [
    'main',
    'review',
  ]
  assert controller.visible_workspaces()[1].identity == first_tab.identity
  assert controller.visible_workspaces()[2].identity == second_tab.identity
  assert controller.visible_workspaces()[3].identity == remote_identity
  assert controller.selected is controller.visible_workspaces()[3]
  assert factory.agents[0].errors == []
  assert factory.agents[1].errors == []
  assert factory.agents[2].started
  assert [
    workspace.label for workspace in controller.visible_workspaces()
  ].count('review') == 2


@pytest.mark.asyncio
async def test_cancelled_spawn_does_not_block_creation_and_shutdown_waits_for_both(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  first_path = tmp_path / 'first'
  second_path = tmp_path / 'second'
  main.mkdir()
  first_path.mkdir()
  second_path.mkdir()
  creator = _WorktreeCreator()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=creator,
  )
  first = asyncio.create_task(controller.spawn_workspace(main, 'first', None))
  first_request = await creator.next()
  first.cancel()
  with pytest.raises(asyncio.CancelledError):
    await first
  first_tab = controller.visible_workspaces()[1]
  assert await controller.create(' second ')
  second_request = await creator.next()
  assert second_request.name == 'second'
  assert controller.visible_workspaces()[1] == first_tab
  assert controller.visible_workspaces()[2].label == 'second'
  assert 'second' in ''.join(text for _, text in _render(controller))
  assert controller.can_cycle()
  controller.refresh()
  assert len(factory.agents) == 1

  shutdown = asyncio.create_task(_run(controller))
  await factory.agents[0].activation_finished.wait()
  assert not shutdown.done()
  second_request.result.set_result(worktrees.Worktree(second_path, 'second'))
  first_request.result.set_result(worktrees.Worktree(first_path, 'first'))
  await shutdown

  assert {tab.label for tab in controller.visible_workspaces()} == {
    'main',
    'first',
    'second',
  }
  assert all(
    tab.state is not actions.WorktreeAgentState.CREATING
    for tab in controller.visible_workspaces()
  )
  assert controller.selected is controller.visible_workspaces()[0]
  assert factory.agents[0].closed
  assert factory.agents[1].closed
  assert factory.agents[2].closed
  assert controller.list_workspaces(main).creating is None


@pytest.mark.asyncio
async def test_pending_and_failed_tabs_do_not_allow_dropping_the_last_workspace(
  managed_root: pathlib.Path,
) -> None:
  main = managed_root / 'main'
  main.mkdir()
  factory = _AgentFactory()
  creator = _WorktreeCreator()
  dropped: list[worktrees.Worktree] = []
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=managed_root / 'state.db',
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=creator,
    drop=lambda item, _: dropped.append(item),
  )
  creation = asyncio.create_task(
    controller.spawn_workspace(main, 'first', None)
  )
  request = await creator.next()
  try:
    assert not await controller.drop(confirmed=True)
    assert controller.selected is controller.visible_workspaces()[0]
  finally:
    request.result.set_exception(worktrees.WorktreeError('fetch failed'))
    with pytest.raises(worktrees.WorktreeError, match='fetch failed'):
      await creation
  assert not await controller.drop(confirmed=True)
  assert dropped == []
  assert factory.agents[0].errors == [
    'cannot drop the only available worktree',
    'cannot drop the only available worktree',
  ]
  assert await controller.select(1)
  assert await controller.drop()
  assert controller.selected is controller.visible_workspaces()[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['\t', '\x1b[Z', '\x1b[49;5u'])
@pytest.mark.usefixtures('frozen_ui_time')
async def test_creation_view_can_navigate_away_before_checkout_finishes(
  tmp_path: pathlib.Path, key: str
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  factory = _AgentFactory()
  creator = _WorktreeCreator()
  output = _RecordingOutput()
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    controller = workspace_controller.WorkspaceController(
      _config(),
      workspace=main,
      reasoning_effort='medium',
      create_agent=factory,
      database=tmp_path / 'state.db',
      discover=lambda _: (worktrees.Worktree(main, 'main'),),
      create=creator,
    )
    creation = asyncio.create_task(
      controller.spawn_workspace(main, 'feature', None)
    )
    request = await creator.next()
    assert await controller.select(1)
    running = asyncio.create_task(_run(controller))
    try:
      await output.wait_for_text('·● Creating feature…')
      pipe.send_text(key)
      await asyncio.wait_for(factory.agents[0].activation_finished.wait(), 5)
      assert controller.selected is controller.visible_workspaces()[0]
      assert (
        controller.visible_workspaces()[1].state
        is actions.WorktreeAgentState.CREATING
      )
      assert not creation.done()
      assert not running.done()
    finally:
      request.result.set_result(worktrees.Worktree(created, 'feature'))
      await creation
      await asyncio.wait_for(running, timeout=5)
  assert factory.agents[1].closed
  assert not factory.agents[1].activation_finished.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('key', 'selected', 'count'), [('\t', 2, 3), ('\r', 1, 2)]
)
async def test_selected_failure_remains_visible_while_another_tab_finishes(
  tmp_path: pathlib.Path,
  key: str,
  selected: int,
  count: int,
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  factory = _AgentFactory()
  creator = _WorktreeCreator()
  output = _RecordingOutput()
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    controller = workspace_controller.WorkspaceController(
      _config(),
      workspace=main,
      reasoning_effort='medium',
      create_agent=factory,
      database=tmp_path / 'state.db',
      discover=lambda _: (worktrees.Worktree(main, 'main'),),
      create=creator,
    )
    first = asyncio.create_task(controller.spawn_workspace(main, 'first', None))
    first_request = await creator.next()
    second = asyncio.create_task(
      controller.spawn_workspace(main, 'second', None)
    )
    second_request = await creator.next()
    identities = tuple(tab.identity for tab in controller.visible_workspaces())
    assert await controller.select(1)
    running = asyncio.create_task(_run(controller))
    try:
      await output.wait_for_text('Creating first')
      second_request.result.set_result(worktrees.Worktree(created, 'second'))
      await second
      assert controller.selected is controller.visible_workspaces()[1]
      assert (
        controller.visible_workspaces()[2].state
        is not actions.WorktreeAgentState.CREATING
      )
      first_request.result.set_exception(
        worktrees.WorktreeError('fetch failed')
      )
      with pytest.raises(worktrees.WorktreeError, match='fetch failed'):
        await first
      await output.wait_for_text('fetch failed')
      assert not running.done()
      assert (
        controller.visible_workspaces()[1].state
        is actions.WorktreeAgentState.FAILED
      )
      assert (
        tuple(tab.identity for tab in controller.visible_workspaces())
        == identities
      )
      assert factory.agents[0].errors == []
      pipe.send_text(key)
      await asyncio.wait_for(running, timeout=5)
    finally:
      running.cancel()
      await asyncio.gather(running, return_exceptions=True)

  assert controller.selected is controller.visible_workspaces()[selected]
  assert len(controller.visible_workspaces()) == count
  assert factory.agents[1].activation_finished.is_set()
  assert not factory.agents[0].activation_finished.is_set()


@pytest.mark.parametrize(
  'legacy_preferences',
  [
    b'',
    client_pb2.WorkspacePreferences(
      reasoning_effort='high'
    ).SerializeToString(),
  ],
)
@pytest.mark.asyncio
async def test_legacy_overlay_preserves_default_tools(
  tmp_path: pathlib.Path, legacy_preferences: bytes
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_alias(str(main.resolve()), 'main', 'Main alias')
  with sqlite3.connect(database) as connection:
    connection.execute(
      'UPDATE cli_worktree_overlays SET preferences = ?', (legacy_preferences,)
    )
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config() | {'tools': ('bash',)},
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=database,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)
  assert factory.agents[0].spec.config['tools'] == ('bash',)
  assert factory.agents[0].spec.tools is None
  assert controller.visible_workspaces()[0].label == 'Main alias'
  controller.save_preferences('high', 'auto')
  with cli_store.Store(database) as store:
    overlay = store.worktree_overlays(str(main.resolve()))['main']
    assert not overlay.preferences.HasField('tools_allowlist')
  reopened = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config() | {'tools': ('bash', 'read')},
    workspace=main,
    reasoning_effort='medium',
    create_agent=reopened,
    database=database,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)
  assert reopened.agents[0].spec.config['tools'] == ('bash', 'read')
  assert reopened.agents[0].spec.tools is None


@pytest.mark.parametrize(
  ('tools', 'builtins', 'extension_tools'),
  [
    (None, ('bash',), None),
    (
      client_pb2.WorkspacePreferences.ToolAllowlist(
        names=['mcp_search', 'mcp_describe', 'mcp_call', 'mcp__extension_tool']
      ),
      ('mcp_search', 'mcp_describe', 'mcp_call'),
      ('mcp__extension_tool',),
    ),
    (client_pb2.WorkspacePreferences.ToolAllowlist(), (), ()),
    (
      client_pb2.WorkspacePreferences.ToolAllowlist(names=['read']),
      ('read',),
      (),
    ),
    (
      client_pb2.WorkspacePreferences.ToolAllowlist(names=['checklist']),
      (),
      ('checklist',),
    ),
    (
      client_pb2.WorkspacePreferences.ToolAllowlist(
        names=['read', 'checklist']
      ),
      ('read',),
      ('checklist',),
    ),
  ],
)
@pytest.mark.asyncio
async def test_workspace_tool_selection_restores_exactly(
  tmp_path: pathlib.Path,
  tools: client_pb2.WorkspacePreferences.ToolAllowlist | None,
  builtins: tuple[str, ...],
  extension_tools: tuple[str, ...] | None,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_preferences(
      str(main.resolve()),
      'main',
      client_pb2.WorkspacePreferences(tools_allowlist=tools),
    )
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config() | {'tools': ('bash',)},
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=database,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)
  assert factory.agents[0].spec.config['tools'] == builtins
  assert factory.agents[0].spec.tools == extension_tools


@pytest.mark.asyncio
async def test_spawn_survives_caller_cancellation_and_tab_switch(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  sibling = tmp_path / 'sibling'
  created = tmp_path / 'created'
  main.mkdir()
  sibling.mkdir()
  created.mkdir()
  started = threading.Event()
  release = threading.Event()

  def create(_: pathlib.Path, name: str | None) -> worktrees.Worktree:
    started.set()
    release.wait()
    return worktrees.Worktree(created, name)

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=tmp_path / 'state.db',
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(sibling, 'sibling'),
    ),
    create=create,
  )
  await _refresh(controller)
  caller = asyncio.create_task(
    controller.spawn_workspace(
      main,
      'custom',
      None,
      preferences=client_pb2.WorkspacePreferences(
        system_prompt='Review only.',
        tools_allowlist=client_pb2.WorkspacePreferences.ToolAllowlist(
          names=('read', 'test_checklist')
        ),
        extensions=('test/indicator',),
      ),
    )
  )
  try:
    assert await asyncio.to_thread(started.wait, 1)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
      await caller
    await controller.cycle(1)
    assert controller.visible_workspaces()[-1].label == 'custom'
    assert (
      controller.visible_workspaces()[-1].state
      is actions.WorktreeAgentState.CREATING
    )
    assert len(factory.agents) == 2
  finally:
    release.set()
    await asyncio.gather(
      *tuple(controller._worktree_creation_tasks)  # pyright: ignore[reportPrivateUsage]
    )
  spec = factory.agents[2].spec
  assert spec.config['system_prompt'] == 'Review only.'
  assert spec.config['tools'] == ('read',)
  assert spec.extensions == ('test/indicator',)
  assert spec.tools == ('test_checklist',)
  assert factory.agents[2].started
  assert len(controller.visible_workspaces()) == 3
  assert (
    controller.visible_workspaces()[-1].state
    is not actions.WorktreeAgentState.CREATING
  )
  with cli_store.Store(tmp_path / 'state.db') as store:
    saved = store.worktree_overlays(str(main.resolve()))['custom'].preferences
    assert saved.system_prompt == 'Review only.'
    assert saved.extensions == ['test/indicator']
    assert saved.tools_allowlist.names == ['read', 'test_checklist']


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('source_mode', 'requested', 'expected'),
  [
    ('auto', agent_pb2.APPROVAL_MODE_HITL, 'approve'),
    ('approve', agent_pb2.APPROVAL_MODE_AUTO, 'auto'),
  ],
)
async def test_spawn_preserves_explicit_approval_mode(
  tmp_path: pathlib.Path, source_mode: str, requested: int, expected: str
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=lambda *_: worktrees.Worktree(created, 'custom'),
  )
  await _refresh(controller)
  factory.agents[0].approval_mode = source_mode
  await controller.spawn_workspace(
    main,
    'custom',
    None,
    preferences=client_pb2.WorkspacePreferences(approval_mode=requested),
  )
  assert factory.agents[1].approval_mode == expected


@pytest.mark.asyncio
async def test_persistence_failure_keeps_created_tab_and_requested_configuration(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  database = tmp_path / 'state.db'
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=database,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=lambda *_: worktrees.Worktree(created, 'review'),
  )
  await _refresh(controller)
  preferences = client_pb2.WorkspacePreferences(
    model='review-model',
    system_prompt='Review only.',
    approval_mode=agent_pb2.APPROVAL_MODE_HITL,
    tools_allowlist=client_pb2.WorkspacePreferences.ToolAllowlist(
      names=['read']
    ),
  )
  with monkeypatch.context() as patch:
    patch.setattr(
      cli_store.Store,
      'set_worktree_preferences',
      mock.Mock(side_effect=sqlite3.OperationalError('disk full')),
    )
    with pytest.raises(worktrees.WorktreeError, match='could not save'):
      await controller.spawn_workspace(
        main, 'review', None, preferences=preferences
      )
  assert controller.visible_workspaces()[1].label == 'review'
  target = factory.agents[-1]
  assert not target.started
  assert target.spec.config['tools'] == ('read',)
  assert target.spec.tools == ()
  assert target.spec.config['model'] == 'review-model'
  assert target.spec.config['system_prompt'] == 'Review only.'
  assert target.approval_mode == 'approve'
  assert await controller.select(1)
  controller.save_preferences('medium', 'approve')
  with cli_store.Store(database) as store:
    saved = store.worktree_overlays(str(main.resolve()))['review'].preferences
    assert saved.tools_allowlist == preferences.tools_allowlist
    assert saved.system_prompt == 'Review only.'


@pytest.mark.asyncio
async def test_spawn_retains_failed_tab_and_accepts_another_creation(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()

  def fail(_: pathlib.Path, __: str | None) -> worktrees.Worktree:
    raise worktrees.WorktreeError('fetch failed')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=fail,
  )
  await _refresh(controller)
  with pytest.raises(worktrees.WorktreeError, match='fetch failed'):
    await controller.spawn_workspace(main, 'review', None)
  assert len(controller.visible_workspaces()) == 2
  failed = controller.visible_workspaces()[1]
  assert failed.state is actions.WorktreeAgentState.FAILED
  with pytest.raises(worktrees.WorktreeError, match='fetch failed'):
    await controller.spawn_workspace(main, 'review', None)
  assert controller.visible_workspaces()[1] == failed
  assert controller.visible_workspaces()[2].identity != failed.identity


@pytest.mark.asyncio
async def test_controller_keeps_failed_creation_on_its_tab_until_dismissed(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()

  def fail(_: pathlib.Path, __: str | None) -> worktrees.Worktree:
    raise worktrees.WorktreeError('fetch failed')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=fail,
  )
  await _refresh(controller)

  assert await controller.create('release-prep')
  with pytest.raises(worktrees.WorktreeError, match='fetch failed'):
    await asyncio.gather(
      *tuple(
        controller._worktree_creation_tasks  # pyright: ignore[reportPrivateUsage]
      )
    )

  assert (
    controller.visible_workspaces()[1].state
    is actions.WorktreeAgentState.FAILED
  )
  assert factory.agents[0].errors == []
  assert await controller.select(1)
  assert controller.drop_confirmation() is None
  assert await controller.drop()
  assert len(controller.visible_workspaces()) == 1
  assert controller.selected is controller.visible_workspaces()[0]
  assert _render(controller) == []


@pytest.mark.asyncio
async def test_dismissing_failed_creation_skips_a_deleting_workspace(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  factory = _AgentFactory()
  deletion_started = asyncio.Event()
  deletion_released = asyncio.Event()

  async def delete_remote(_: str) -> None:
    deletion_started.set()
    await deletion_released.wait()

  def fail(_: pathlib.Path, __: str | None) -> worktrees.Worktree:
    raise worktrees.WorktreeError('fetch failed')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    remote_enabled=True,
    delete_remote_session=delete_remote,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=fail,
  )
  with pytest.raises(worktrees.WorktreeError, match='fetch failed'):
    await controller.spawn_workspace(main, 'feature', None)
  assert await controller.create_remote() == 2
  assert await controller.select(2)
  assert await controller.drop()
  try:
    await asyncio.wait_for(deletion_started.wait(), timeout=5)
    assert controller.selected is controller.visible_workspaces()[1]
    assert await controller.drop()
    assert len(controller.visible_workspaces()) == 1
    assert controller.selected is controller.visible_workspaces()[0]
    assert controller.visible_workspaces()[0].label == 'main'
  finally:
    deletion_released.set()
    await _run(controller)
  assert factory.agents[1].closed


@pytest.mark.asyncio
async def test_controller_drops_selected_worktree_and_closes_agent(
  managed_root: pathlib.Path,
) -> None:
  main = managed_root / 'main'
  feature = managed_root / 'feature'
  main.mkdir()
  feature.mkdir()
  items = (
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(feature, 'feature'),
  )
  database = managed_root / 'state.db'
  dropped: list[tuple[worktrees.Worktree, bool]] = []
  started = threading.Event()
  release = threading.Event()

  def drop(item: worktrees.Worktree, confirmed: bool) -> None:
    dropped.append((item, confirmed))
    started.set()
    release.wait()

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=feature,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=database,
    discover=lambda _: items,
    drop=drop,
  )
  await _refresh(controller)
  assert controller.rename('Feature alias') == 'Feature alias'
  reconcile = asyncio.create_task(asyncio.Event().wait())
  controller._reconcile_task = reconcile  # pyright: ignore[reportPrivateUsage]

  assert controller.drop_confirmation() is None
  assert await controller.drop(confirmed=True)

  assert reconcile.cancelled()
  assert await asyncio.to_thread(started.wait, 1)
  assert dropped == [(items[1], True)]
  assert not factory.agents[0].closed
  assert _render(controller) == []
  assert not await controller.drop()
  assert factory.agents[1].errors == ['cannot drop the only available worktree']

  tasks = tuple(
    controller._workspace_deletion_tasks  # pyright: ignore[reportPrivateUsage]
  )
  release.set()
  await asyncio.gather(*tasks)
  await asyncio.sleep(0)

  assert factory.agents[0].closed
  assert _render(controller) == []
  with cli_store.Store(database) as store:
    assert store.worktree_overlays(str(feature)) == {
      'main': cli_store.WorktreeOverlay(
        preferences=client_pb2.WorkspacePreferences(
          model='m-test', model_api='', model_base_url=''
        ),
      ),
    }


@pytest.mark.asyncio
async def test_controller_hides_dropping_worktree_and_restores_on_failure(
  managed_root: pathlib.Path,
) -> None:
  main = managed_root / 'main'
  feature = managed_root / 'feature'
  main.mkdir()
  feature.mkdir()
  items = (
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(feature, 'feature'),
  )
  started = threading.Event()
  release = threading.Event()

  def fail_drop(_: worktrees.Worktree, __: bool) -> None:
    started.set()
    release.wait()
    raise worktrees.WorktreeError('delete failed')

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=feature,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: items,
    drop=fail_drop,
  )
  await _refresh(controller)

  assert await controller.drop(confirmed=True)
  try:
    assert await asyncio.to_thread(started.wait, 1)
    assert _render(controller) == []
  finally:
    release.set()

  tasks = tuple(
    controller._workspace_deletion_tasks  # pyright: ignore[reportPrivateUsage]
  )
  await asyncio.gather(*tasks)
  await asyncio.sleep(0)
  rendered = _render(controller)
  assert controller.visible_workspaces()[1].label == 'feature'
  assert any(
    style == 'class:worktree-tab-selected' and 'main' in text
    for style, text, *_ in rendered
  )
  assert factory.agents[0].errors == ['delete failed']


@pytest.mark.asyncio
async def test_controller_owns_concurrent_workspace_deletions(
  managed_root: pathlib.Path,
) -> None:
  first = managed_root / 'first'
  second = managed_root / 'second'
  third = managed_root / 'third'
  first.mkdir()
  second.mkdir()
  third.mkdir()
  items = (
    worktrees.Worktree(first, 'first'),
    worktrees.Worktree(second, 'second'),
    worktrees.Worktree(third, 'third'),
  )
  first_started = threading.Event()
  second_started = threading.Event()
  first_release = threading.Event()
  second_release = threading.Event()
  started = {first: first_started, second: second_started}
  releases = {first: first_release, second: second_release}

  def drop(item: worktrees.Worktree, _: bool) -> None:
    started[item.path].set()
    releases[item.path].wait()

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=first,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: items,
    drop=drop,
  )
  await _refresh(controller)

  assert await controller.drop()
  assert await asyncio.to_thread(first_started.wait, 1)
  first_task = next(
    iter(
      controller._workspace_deletion_tasks  # pyright: ignore[reportPrivateUsage]
    )
  )
  assert await controller.drop()
  assert await asyncio.to_thread(second_started.wait, 1)
  second_task = next(
    task
    for task in controller._workspace_deletion_tasks  # pyright: ignore[reportPrivateUsage]
    if task is not first_task
  )
  assert not await controller.drop()
  assert factory.agents[2].errors == ['cannot drop the only available worktree']

  first_release.set()
  await first_task
  await asyncio.sleep(0)
  assert factory.agents[0].closed
  assert not second_task.done()
  assert not factory.agents[1].closed

  second_release.set()
  await second_task
  await asyncio.sleep(0)
  assert factory.agents[1].closed
  assert not factory.agents[2].closed
  assert not controller.can_cycle()


@pytest.mark.asyncio
async def test_orchestrator_lists_sends_and_inspects_local_agents(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=tmp_path / 'state.db',
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)
  target = factory.agents[1]
  target.entries = (
    agent_pb2.AgentMessage(
      id='before-send',
      timestamp=timestamp_pb2.Timestamp(seconds=100, nanos=500_000_000),
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(content='Existing answer.')
      ),
    ),
  )

  listed = controller.list_workspaces(main)
  sent = await controller.send_message(main, 'feature', 'Review the parser.')
  await asyncio.gather(
    *controller._message_delivery_tasks  # pyright: ignore[reportPrivateUsage]
  )

  assert [item.id for item in listed.workspaces] == ['main', 'feature']
  assert listed.workspaces[0].current is True
  assert listed.workspaces[0].path == main
  assert sent.cursor == 100.5
  assert target.started
  assert target.submitted == [
    "Message from the agent in workspace 'main'. Complete this request in "
    'your workspace and leave the response here for the sending agent to '
    'inspect.\n\nReview the parser.'
  ]

  target.entries = (
    *target.entries,
    agent_pb2.AgentMessage(
      id='delegated-request',
      timestamp=timestamp_pb2.Timestamp(seconds=101),
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='Review the parser.')
      ),
    ),
    agent_pb2.AgentMessage(
      id='answer-1',
      timestamp=timestamp_pb2.Timestamp(seconds=102),
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(
          content='The parser is correct.', reasoning_content='private'
        )
      ),
    ),
  )
  target.worktree_state = actions.WorktreeAgentState.UNREAD

  inspected = controller.inspect_workspace(main, 'feature')

  assert inspected.messages == (
    workspace_controller.MessageSnapshot(
      100.5, 'assistant', 'Existing answer.'
    ),
    workspace_controller.MessageSnapshot(101.0, 'user', 'Review the parser.'),
    workspace_controller.MessageSnapshot(
      102.0, 'assistant', 'The parser is correct.'
    ),
  )
  assert inspected.workspace.agent.state is actions.WorktreeAgentState.UNREAD
  with pytest.raises(worktrees.WorktreeError, match='this workspace'):
    await controller.send_message(main, 'main', 'Loop back.')


@pytest.mark.asyncio
async def test_workspace_lookup_accepts_current_names_and_prefers_ids(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    display_unmanaged_worktrees=True,
    database=tmp_path / 'state.db',
    discover=lambda _: (
      worktrees.Worktree(
        main, 'duplicate', worktrees.WorktreeIdentity('jj', 'main')
      ),
      worktrees.Worktree(
        feature, 'duplicate', worktrees.WorktreeIdentity('jj', 'feature')
      ),
    ),
  )
  await _refresh(controller)
  with pytest.raises(worktrees.WorktreeError, match='ambiguous'):
    controller.inspect_workspace(main, 'duplicate')
  assert controller.inspect_workspace(main, 'feature').workspace.id == 'feature'
  controller.rename('Review work')
  assert (
    controller.inspect_workspace(main, 'REVIEW WORK').workspace.id == 'main'
  )
  controller.rename('New name')
  with pytest.raises(worktrees.WorktreeError, match='not found'):
    controller.inspect_workspace(main, 'Review work')
  result = controller.inspect_workspace(main, 'main')
  assert result.workspace.name == 'New name'
  assert result.workspace.id == 'main'


@pytest.mark.asyncio
async def test_orchestrator_send_reports_startup_failure_to_sender(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=tmp_path / 'state.db',
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)

  async def fail_start() -> None:
    raise RuntimeError('startup failed')

  monkeypatch.setattr(factory.agents[1], 'start', fail_start)
  sent = await controller.send_message(main, 'feature', 'Investigate.')
  assert sent.status == 'accepted'
  await asyncio.gather(
    *controller._message_delivery_tasks  # pyright: ignore[reportPrivateUsage]
  )
  assert factory.agents[1].submitted == []
  assert factory.agents[0].submitted == [
    "Message delivery to workspace 'feature' failed: startup failed"
  ]


@pytest.mark.asyncio
async def test_orchestrator_send_returns_before_agent_startup(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)
  started = asyncio.Event()
  release = asyncio.Event()

  async def slow_start() -> None:
    started.set()
    await release.wait()

  target = factory.agents[1]
  monkeypatch.setattr(target, 'start', slow_start)
  sent = await asyncio.wait_for(
    controller.send_message(main, 'feature', 'Investigate.'),
    timeout=0.1,
  )
  try:
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not target.submitted
  finally:
    release.set()
  await asyncio.gather(
    *controller._message_delivery_tasks  # pyright: ignore[reportPrivateUsage]
  )

  assert sent.cursor is None
  assert target.submitted == [
    "Message from the agent in workspace 'main'. Complete this request in "
    'your workspace and leave the response here for the sending agent to '
    'inspect.\n\nInvestigate.'
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['feature', None, '   '])
async def test_orchestrator_spawns_and_messages_a_named_worktree(
  tmp_path: pathlib.Path,
  name: str | None,
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'state.db',
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=lambda _source, name: worktrees.Worktree(created, name or 'feature'),
  )
  await _refresh(controller)

  result = await controller.spawn_workspace(main, name, 'Implement the parser.')
  await asyncio.gather(
    *controller._message_delivery_tasks  # pyright: ignore[reportPrivateUsage]
  )

  assert result.workspace.id == 'feature'
  assert result.cursor is None
  assert result.status == 'accepted'
  assert factory.agents[1].started
  assert factory.agents[1].submitted[-1].endswith('Implement the parser.')
  assert controller.visible_workspaces()[-1].label == 'feature'


@pytest.mark.asyncio
async def test_controller_activates_before_agent_start_finishes(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  started = asyncio.Event()
  release = asyncio.Event()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
  )
  await _refresh(controller)

  async def slow_start() -> None:
    started.set()
    await release.wait()

  monkeypatch.setattr(factory.agents[0], 'start', slow_start)
  try:
    await asyncio.wait_for(_run(controller), timeout=5.0)
  finally:
    release.set()

  assert started.is_set()


@pytest.mark.asyncio
async def test_controller_closes_agents(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: (
      worktrees.Worktree(main, 'main'),
      worktrees.Worktree(feature, 'feature'),
    ),
  )
  await _refresh(controller)
  await _run(controller)

  assert factory.agents[0].started
  assert factory.agents[0].closed
  assert factory.agents[1].closed


class _RemoteSetup(extension.Extension[None, None, None]):
  def __init__(self) -> None:
    self.ready = False
    self.fail = False
    self.calls: list[tuple[str | None, str | None]] = []

  @classmethod
  def id(cls) -> str:
    return 'test/remote-setup'

  async def on_remote_setup(
    self, context: extension.Context, request: actions.CreateRemote
  ) -> bool:
    self.calls.append((context.remote_endpoint, request.title))
    if self.fail:
      raise RuntimeError('setup unavailable')
    return self.ready


@pytest.mark.asyncio
async def test_remote_creation_waits_for_setup_from_local_workspace(
  tmp_path: pathlib.Path,
) -> None:
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    database=tmp_path / 'cli.db',
    remote_enabled=True,
    delete_remote_session=_delete_remote,
    discover=lambda _: (worktrees.Worktree(tmp_path, 'main'),),
  )
  await _refresh(controller)
  registry = extension.ExtensionRegistry()
  setup = _RemoteSetup()
  await registry.register(setup, None)
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=base.DummyOutput()),
  ):
    repl = cli._Repl(
      [sys.executable],
      _config(),
      session_id='test-session',
      workspace=tmp_path,
      workspace_effects=asyncio.Queue(),
      workspace_controller=mock.Mock(
        spec=workspace_controller.WorkspaceController
      ),
      extension_registry=registry,
      configured_remote_endpoint='server.test:443',
    )
    ready = asyncio.Event()
    app_task = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    try:
      await _dispatch_and_wait(
        repl, actions.CreateRemote('Release'), controller
      )
      assert len(factory.agents) == 1
      assert setup.calls == [('server.test:443', 'Release')]
      setup.ready = True
      await _dispatch_and_wait(
        repl, actions.CreateRemote('Release'), controller
      )
      assert len(factory.agents) == 2
      assert factory.agents[1].started
      assert controller.visible_workspaces()[1].label == 'Release'
      assert await app_task == 'select:1'
      setup.fail = True
      await _dispatch_and_wait(
        repl, actions.CreateRemote('Another'), controller
      )
      assert len(factory.agents) == 2
      # A disabled hook must not silently allow the next launch.
      await _dispatch_and_wait(
        repl, actions.CreateRemote('Another'), controller
      )
      assert len(factory.agents) == 2
      assert len(setup.calls) == 3
    finally:
      app_task.cancel()
      await asyncio.gather(app_task, return_exceptions=True)
      await registry.close()


@pytest.mark.asyncio
async def test_picker_identity_and_tab_positions_survive_discovery_reordering(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  main.mkdir()
  feature.mkdir()
  roster = [
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(feature, 'feature'),
  ]
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    discover=lambda _: roster,
  )
  await _refresh(controller)
  identity = controller.visible_workspaces()[1].identity
  roster.reverse()
  await _refresh(controller)

  assert await controller.select_workspace(identity)
  assert controller.selected is controller.visible_workspaces()[1]
  assert controller.visible_workspaces()[1].label == 'feature'

  feature.rmdir()
  assert not await controller.select_workspace(identity)
  assert 'no longer available' in factory.agents[1].errors[-1]
  await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize('continue_recent', [False, True])
async def test_initial_discovery_preserves_active_agent_and_preferences(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  continue_recent: bool,
) -> None:
  main = tmp_path / 'main'
  feature = tmp_path / 'feature'
  cwd = main / 'package'
  cwd.mkdir(parents=True)
  (feature / 'package').mkdir(parents=True)
  link = tmp_path / 'link'
  link.symlink_to(cwd, target_is_directory=True)
  initial = worktrees.Worktree(
    main, 'default', worktrees.WorktreeIdentity('jj', 'default')
  )
  sibling = worktrees.Worktree(
    feature, 'feature', worktrees.WorktreeIdentity('jj', 'feature')
  )
  monkeypatch.setattr(
    worktrees, 'current_worktree', mock.Mock(return_value=initial)
  )
  monkeypatch.setattr(
    workspace_controller.workspace_identity,
    'git_workspace_key',
    mock.Mock(return_value='repository'),
  )
  database = tmp_path / 'state.db'
  preferences = client_pb2.WorkspacePreferences(
    model='saved-model',
    tools_allowlist=client_pb2.WorkspacePreferences.ToolAllowlist(
      names=['bash', 'extension-tool']
    ),
    reasoning_effort='high',
  )
  with cli_store.Store(database) as store:
    store.set_worktree_alias('repository', initial.id, 'Primary')
    store.set_worktree_preferences('repository', initial.id, preferences)
    store.set_worktree_alias('repository', str(feature), 'Sibling')
    store.set_worktree_preferences('repository', str(feature), preferences)
    store.record_user_message(
      'sibling-saved',
      entry_id='entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved work',
      cwd=str(feature / 'package'),
      model='saved-model',
    )
  started = threading.Event()
  release = threading.Event()
  activated = asyncio.Event()
  quit_agent = asyncio.Event()

  def discover(path: pathlib.Path) -> Sequence[worktrees.Worktree]:
    assert path in (main, feature)
    started.set()
    assert release.wait(5)
    return (sibling, initial)

  async def activate(*, show_context: bool) -> str | None:
    assert not show_context
    assert [tab.label for tab in controller.visible_workspaces()] == [
      'Sibling',
      'Primary',
    ]
    assert controller.selected is controller.visible_workspaces()[1]
    activated.set()
    await quit_agent.wait()
    return None

  factory = _AgentFactory()
  config = dict(_config()) | {'create_session': not continue_recent}
  controller = workspace_controller.WorkspaceController(
    config,
    workspace=link,
    reasoning_effort='medium',
    create_agent=factory,
    display_unmanaged_worktrees=True,
    database=database,
    continue_recent=continue_recent,
    discover=discover,
  )
  assert len(factory.agents) == 1
  active = factory.agents[0]
  monkeypatch.setattr(active, 'activate', activate)
  assert active.workspace == cwd
  assert controller.launch_workspace == cwd
  assert active.spec.config['model'] == 'saved-model'
  assert active.spec.config['tools'] == ('bash',)
  assert active.spec.tools == ('extension-tool',)
  assert active.reasoning_effort == 'high'
  assert controller.visible_workspaces()[0].label == 'Primary'
  task = asyncio.create_task(_run(controller))
  try:
    assert await asyncio.to_thread(started.wait, 1)
    assert not activated.is_set()
    assert not active.started
    release.set()
    await asyncio.wait_for(activated.wait(), 1)
    assert active.started
    active.worktree_state = actions.WorktreeAgentState.RUNNING
    active.submitted.append('keep this turn')
    await _refresh(controller)
    assert len(factory.agents) == 2
    assert factory.agents[0] is active
    assert active.conversation_id == 'conv-1'
    assert active.submitted == ['keep this turn']
    assert not active.closed
    assert [tab.label for tab in controller.visible_workspaces()] == [
      'Sibling',
      'Primary',
    ]
    assert controller.selected is controller.visible_workspaces()[1]
    assert (
      controller.visible_workspaces()[1].state
      is actions.WorktreeAgentState.RUNNING
    )
    discovered_agent = factory.agents[1]
    assert discovered_agent.workspace == feature / 'package'
    assert discovered_agent.spec.config['model'] == 'saved-model'
    assert discovered_agent.spec.config['tools'] == ('bash',)
    assert discovered_agent.spec.tools == ('extension-tool',)
    assert not discovered_agent.started
    assert discovered_agent.spec.config['create_session'] is not continue_recent
    if continue_recent:
      assert discovered_agent.conversation_id == 'sibling-saved'
    with cli_store.Store(database) as store:
      overlays = store.worktree_overlays('repository')
      assert str(feature) not in overlays
      assert overlays['feature'].alias == 'Sibling'
      assert overlays['feature'].preferences == preferences
  finally:
    release.set()
    quit_agent.set()
    await task
  assert all(agent.closed for agent in factory.agents)


@pytest.mark.asyncio
async def test_initial_roster_includes_discovered_and_saved_workspaces(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  sibling = tmp_path / 'sibling'
  folder = tmp_path / 'folder'
  main.mkdir()
  sibling.mkdir()
  folder.mkdir()
  with cli_store.Store(tmp_path / 'state.db') as store:
    store.save_workspace(str(main), 'saved-folder', str(folder), 'folder')
  restored = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=tmp_path / 'state.db',
    display_unmanaged_worktrees=True,
    is_trusted_directory=lambda _: True,
    discover=lambda _: (
      worktrees.Worktree(sibling, 'sibling'),
      worktrees.Worktree(main, 'main'),
    ),
  )

  async def activate(
    selected: workspaces.Workspace, show_context: bool
  ) -> str | None:
    assert not show_context
    assert [tab.label for tab in restored.visible_workspaces()] == [
      'sibling',
      'main',
      'folder',
    ]
    assert selected is restored.visible_workspaces()[1]
    return None

  await cli.run_workspaces(restored, activate, asyncio.Queue())


@pytest.mark.asyncio
async def test_initial_discovery_failure_keeps_launch_workspace(
  tmp_path: pathlib.Path,
) -> None:
  def discover(_: pathlib.Path) -> Sequence[worktrees.Worktree]:
    raise worktrees.WorktreeError('discovery unavailable')

  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    discover=discover,
  )

  async def activate(
    selected: workspaces.Workspace, show_context: bool
  ) -> str | None:
    assert not show_context
    assert controller.visible_workspaces() == (selected,)
    assert selected.path == tmp_path
    return None

  await cli.run_workspaces(controller, activate, asyncio.Queue())


@pytest.mark.asyncio
async def test_shutdown_during_initial_discovery(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  started = threading.Event()
  release = threading.Event()
  finished = threading.Event()
  activated = asyncio.Event()
  quit_agent = asyncio.Event()

  def discover(path: pathlib.Path) -> Sequence[worktrees.Worktree]:
    started.set()
    try:
      assert release.wait(5)
      return (
        worktrees.Worktree(path, 'main'),
        worktrees.Worktree(path / 'sibling', 'sibling'),
      )
    finally:
      finished.set()

  async def activate(*, show_context: bool) -> str | None:
    del show_context
    activated.set()
    await quit_agent.wait()
    return None

  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=factory,
    discover=discover,
  )
  monkeypatch.setattr(factory.agents[0], 'activate', activate)
  task = asyncio.create_task(_run(controller))
  try:
    assert await asyncio.to_thread(started.wait, 1)
    assert not activated.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, 1)
    assert not release.is_set()
    assert factory.agents[0].closed
    assert controller._reconcile_task is None  # pyright: ignore[reportPrivateUsage]
  finally:
    release.set()
    quit_agent.set()
    await asyncio.gather(task, return_exceptions=True)
    assert await asyncio.to_thread(finished.wait, 1)
  await asyncio.sleep(0)
  assert len(factory.agents) == 1


@pytest.mark.asyncio
async def test_managed_roster_preserves_launch_checkout_and_updates_preference(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  launch = tmp_path / 'repository'
  launch.mkdir()
  managed = tmp_path / '.mirror' / 'worktrees' / 'feature'
  managed.mkdir(parents=True)
  unmanaged = tmp_path / 'external'
  unmanaged.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=launch,
    reasoning_effort='medium',
    create_agent=factory,
    discover=lambda _: (
      worktrees.Worktree(launch, 'repository'),
      worktrees.Worktree(managed, 'feature'),
      worktrees.Worktree(unmanaged, 'external'),
    ),
  )
  await _refresh(controller)
  assert [tab.label for tab in controller.visible_workspaces()] == [
    'repository',
    'feature',
  ]
  assert len(factory.agents) == 2

  assert await controller.select(1)
  await _refresh(controller)
  assert [tab.label for tab in controller.visible_workspaces()] == [
    'repository',
    'feature',
  ]
  assert not factory.agents[0].closed
  assert await controller.select(0)
  await _refresh(controller)

  controller.set_display_unmanaged_worktrees(True)
  await _refresh(controller)
  assert [tab.label for tab in controller.visible_workspaces()] == [
    'repository',
    'feature',
    'external',
  ]

  controller.set_display_unmanaged_worktrees(False)
  await _refresh(controller)
  assert [tab.label for tab in controller.visible_workspaces()] == [
    'repository',
    'feature',
  ]
  assert factory.agents[2].closed
  assert not factory.agents[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['git', 'jj'])
@pytest.mark.parametrize('launch_primary', [False, True])
async def test_managed_roster_keeps_primary_checkout(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
  launch_primary: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  primary = tmp_path / 'repository'
  primary.mkdir()
  managed = tmp_path / '.mirror' / 'worktrees' / 'feature'
  managed.mkdir(parents=True)
  unmanaged = tmp_path / 'external'
  unmanaged.mkdir()
  primary_id = 'default' if provider == 'jj' else str(primary)
  primary_name = 'default' if provider == 'jj' else 'main'
  primary_worktree = worktrees.Worktree(
    primary, primary_name, worktrees.WorktreeIdentity(provider, primary_id)
  )
  feature = worktrees.Worktree(
    managed, 'feature', worktrees.WorktreeIdentity(provider, 'feature')
  )
  launch = primary if launch_primary else managed
  monkeypatch.setattr(
    worktrees,
    'current_worktree',
    lambda _: primary_worktree if launch_primary else feature,
  )
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=launch,
    reasoning_effort='medium',
    create_agent=factory,
    discover=lambda _: (
      primary_worktree,
      feature,
      worktrees.Worktree(unmanaged, 'external'),
    ),
  )
  await _refresh(controller)
  primary_tab, feature_tab = controller.visible_workspaces()
  assert primary_tab.identity == f'{provider}:{primary_id}'
  assert primary_tab.label == primary_name
  assert primary_tab.path == primary
  assert feature_tab.identity == f'{provider}:feature'
  assert [
    item.path for item in controller.list_workspaces(launch).workspaces
  ] == [
    primary,
    managed,
  ]
  await controller.select_workspace(primary_tab.identity)
  assert controller.selected is primary_tab
  await _refresh(controller)
  assert controller.visible_workspaces() == (primary_tab, feature_tab)
  assert len(factory.agents) == 2
  assert await controller.select_workspace(feature_tab.identity)
  await _refresh(controller)
  assert controller.visible_workspaces() == (primary_tab, feature_tab)
  await controller.close()


class _AgentView(cli._Repl):
  """Real terminal UI with an immediately ready in-process backend."""

  @override
  async def start(self) -> None:
    self.dispatch(actions.WorkerStarted())


@pytest.mark.asyncio
async def test_live_roster_picker_handoff_and_inactive_agent_effects(
  tmp_path: pathlib.Path,
) -> None:
  name = 'feature-zzzz'  # Fuzzy search also matches the temporary path.
  main = tmp_path / 'main'
  feature = tmp_path / name
  main.mkdir()
  feature.mkdir()
  creator = _WorktreeCreator()
  registry = extension.ExtensionRegistry()
  queue: asyncio.Queue[tuple[cli._Repl, effects.Effect]] = asyncio.Queue()
  views: list[tuple[_AgentView, asyncio.Event]] = []
  output = _RecordingOutput()

  def create_agent(
    spec: workspaces.AgentSpec,
    controller: workspace_controller.WorkspaceController,
  ) -> _AgentView:
    agent = _AgentView(
      [sys.executable],
      spec.config,
      session_id='test-session',
      workspace=spec.workspace,
      reasoning_effort=spec.reasoning_effort,
      active=spec.active,
      extension_registry=registry,
      workspace_effects=queue,
      workspace_controller=controller,
    )
    rendered = asyncio.Event()
    agent._app.after_render += lambda _: rendered.set()
    views.append((agent, rendered))
    return agent

  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    controller = workspace_controller.WorkspaceController(
      _config(),
      workspace=main,
      reasoning_effort='medium',
      create_agent=create_agent,
      database=tmp_path / 'state.db',
      discover=lambda _: (worktrees.Worktree(main, 'main'),),
      create=creator,
    )
    running = asyncio.create_task(
      cli.run_workspaces(
        controller,
        functools.partial(cli._activate_workspace, controller),
        queue,
      )
    )
    agent, rendered = views[0]
    await asyncio.wait_for(rendered.wait(), timeout=5)
    single_row_height = agent._bottom_pane.preferred_height(80, 100).preferred
    agent.dispatch(actions.OpenWorkspaces())
    await asyncio.wait_for(queue.join(), timeout=5)
    agent.dispatch(actions.SelectorSearchChanged(name))
    assert agent.selector.filtered_choices() == ()
    creating = asyncio.create_task(
      agent._extension_context.workspace_controller.spawn_workspace(
        main, name, None
      )
    )
    request = await creator.next()
    try:
      await output.wait_for_text('creating ·')
      pending = controller.visible_workspaces()[1]
      assert isinstance(pending, workspaces.PendingWorkspace)
      assert agent.selector.query == name
      assert [choice.value for choice in agent.selector.filtered_choices()] == [
        pending.identity
      ]
      pipe.send_text('\r')
      await output.wait_for_text(f'Creating {name}')
      assert controller.selected is pending
      assert not await controller.drop()
      assert not agent.selector.active

      agent.dispatch(
        actions.SelectWorktreeRequested(0),
        actions.SetReasoning('high'),
        actions.SetApprovalMode('approve'),
      )
      await asyncio.wait_for(queue.join(), timeout=5)
      assert controller.selected is pending
      with cli_store.Store(tmp_path / 'state.db') as store:
        preferences = store.worktree_overlays(str(main))['main'].preferences
        assert preferences.reasoning_effort == 'high'
        assert preferences.approval_mode == agent_pb2.APPROVAL_MODE_HITL

      request.result.set_result(worktrees.Worktree(feature, name))
      submission = await creating
      assert submission.workspace.path == feature
      created, ready = views[1]
      await asyncio.wait_for(ready.wait(), timeout=5)
      assert controller.selected.identity == pending.identity
      assert isinstance(controller.selected, workspaces.AgentWorkspace)
      assert controller.selected.agent is created
      assert controller.selected.state is actions.WorktreeAgentState.IDLE
      assert (
        created._bottom_pane.preferred_height(80, 100).preferred
        == single_row_height + 1
      )
      assert not agent._state.lifecycle.active
      pipe.send_text('\x04')
      await asyncio.wait_for(running, timeout=5)
      assert agent._state.lifecycle.quitting
      assert created._state.lifecycle.quitting
    finally:
      if not request.result.done():
        request.result.set_result(worktrees.Worktree(feature, name))
      running.cancel()
      await asyncio.gather(running, creating, return_exceptions=True)
      await registry.close()


@pytest.mark.asyncio
async def test_overlapping_drops_do_not_delete_the_replacement_selection(
  managed_root: pathlib.Path,
) -> None:
  first = managed_root / 'first'
  second = managed_root / 'second'
  first.mkdir()
  second.mkdir()
  deleted: list[pathlib.Path] = []
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=managed_root,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    display_unmanaged_worktrees=True,
    discover=lambda _: (
      worktrees.Worktree(managed_root, 'main'),
      worktrees.Worktree(first, 'first'),
      worktrees.Worktree(second, 'second'),
    ),
    drop=lambda worktree, _: deleted.append(worktree.path),
  )
  await _refresh(controller)
  controller.refresh()
  results = await asyncio.gather(
    controller.drop(confirmed=True),
    controller.drop(confirmed=True),
  )
  await controller.close()
  assert results.count(True) == 1
  assert deleted == [managed_root]
  assert controller.selected.path == first


@pytest.mark.asyncio
async def test_workspace_tags_cycle_without_switching_and_survive_restart(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'default'
  main.mkdir()
  config.Config(tmp_path / 'config.json').update(
    lambda current: current.update({'workspace_tags': []})
  )
  database = tmp_path / 'cli.db'
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=database,
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (),
  )
  selected = controller.selected
  assert controller.cycle_workspace_tag() is None
  with pytest.raises(ValueError, match='unavailable'):
    controller.cycle_workspace_tag(workspace_id='missing')
  blocked = workspace_tags.WorkspaceTag('blocked', 'Blocked', '#ff0000')
  feature = workspace_tags.WorkspaceTag('feature', 'Feature')
  controller.save_workspace_tag(blocked)
  controller.save_workspace_tag(feature)
  assert controller.cycle_workspace_tag() == blocked
  assert controller.cycle_workspace_tag() == feature
  assert controller.cycle_workspace_tag() is None
  controller.set_workspace_tag('default', 'blocked')
  controller.rename('Renamed')
  assert controller.selected is selected
  assert controller.list_workspaces(main).workspaces[0].tag == blocked
  assert controller.list_workspaces(main).workspaces[0].tag_id == 'blocked'
  restored = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=database,
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (),
  )
  assert restored.workspace_tag('default') == blocked
  renamed = workspace_tags.WorkspaceTag('blocked', 'Waiting', '#00ff00')
  restored.save_workspace_tag(renamed)
  assert restored.workspace_tag('default') == renamed
  assert restored.list_workspace_tags() == (renamed, feature)
  restored.delete_workspace_tag('blocked')
  assert restored.workspace_tag('default') is None
  with pytest.raises(ValueError, match='unknown'):
    restored.set_workspace_tag('default', 'deleted')
  with pytest.raises(ValueError, match='unavailable'):
    restored.set_workspace_tag('Renamed', 'feature')
  await controller.close()
  await restored.close()


@pytest.mark.asyncio
async def test_workspace_tag_commands_apply_to_current_workspace(
  tmp_path: pathlib.Path,
) -> None:
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=tmp_path / 'cli.db',
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (),
  )
  repl = _repl()
  await _dispatch_and_wait(
    repl,
    actions.WorkspaceTagCommand('define blocked #ff0000 Waiting on review'),
    controller,
  )
  await _dispatch_and_wait(repl, actions.WorkspaceTagCommand(), controller)
  assert controller.workspace_tag(tmp_path.name) == workspace_tags.WorkspaceTag(
    'blocked', 'Waiting on review', '#ff0000'
  )
  await _dispatch_and_wait(
    repl, actions.WorkspaceTagCommand('clear'), controller
  )
  assert controller.workspace_tag(tmp_path.name) is None
  await _dispatch_and_wait(
    repl, actions.WorkspaceTagCommand('assign blocked'), controller
  )
  assert controller.list_workspaces(tmp_path).workspaces[0].tag_id == 'blocked'
  await _dispatch_and_wait(
    repl, actions.WorkspaceTagCommand('delete blocked'), controller
  )
  assert controller.list_workspace_tags() == workspace_tags.DEFAULT_TAGS[1:]
  assert controller.list_workspaces(tmp_path).workspaces[0].tag_id is None
  await controller.close()


@pytest.mark.asyncio
async def test_unknown_persisted_workspace_tag_is_not_reassigned(
  tmp_path: pathlib.Path,
) -> None:
  config.Config(tmp_path / 'config.json').update(
    lambda current: current.update({'workspace_tags': []})
  )
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_alias(str(tmp_path.resolve()), tmp_path.name, 'Main')
  with cli_store.Store(database) as store:
    store.set_workspace_tag(str(tmp_path.resolve()), tmp_path.name, 'unknown')
  workspace_tags.save(
    config.Config(tmp_path / 'config.json'),
    workspace_tags.WorkspaceTag('feature', 'Feature'),
  )
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=database,
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (),
  )
  assert controller.list_workspaces(tmp_path).workspaces[0].tag_id == 'unknown'
  assert controller.cycle_workspace_tag() is None
  assert controller.list_workspaces(tmp_path).workspaces[0].tag_id is None
  assert controller.cycle_workspace_tag() == workspace_tags.WorkspaceTag(
    'feature', 'Feature'
  )
  await controller.close()


@pytest.mark.asyncio
async def test_tags_follow_provider_ids_not_names_or_positions(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'default'
  main.mkdir()
  feature = tmp_path / 'checkout'
  feature.mkdir()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=tmp_path / 'cli.db',
    settings=config.Config(tmp_path / 'config.json'),
    display_unmanaged_worktrees=True,
    discover=lambda _: (
      worktrees.Worktree(main, 'default'),
      worktrees.Worktree(
        feature, 'mirror/feature', worktrees.WorktreeIdentity('jj', 'stable-id')
      ),
    ),
  )
  await _refresh(controller)
  before = controller.visible_workspaces()
  tag = workspace_tags.WorkspaceTag('feature', 'Feature')
  controller.save_workspace_tag(tag)
  controller.set_workspace_tag('stable-id', 'feature')
  assert controller.visible_workspaces() == before
  assert controller.selected is before[0]
  assert controller.list_workspaces(main).workspaces[0].tag is None
  assert controller.list_workspaces(main).workspaces[1].tag == tag
  assert controller.tab_tags() == {'jj:stable-id': tag}
  controller.set_workspace_tag('stable-id', None)
  assert controller.tab_tags() == {}
  first = workspace_tags.DEFAULT_TAGS[0]
  assert controller.cycle_workspace_tag(workspace_id='stable-id') == first
  assert controller.tab_tags() == {'jj:stable-id': first}
  assert controller.visible_workspaces() == before
  assert controller.selected is before[0]
  await controller.close()


@pytest.mark.asyncio
async def test_fresh_workspace_cycles_defaults_without_writing_config(
  tmp_path: pathlib.Path,
) -> None:
  settings = config.Config(tmp_path / 'config.json')
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=tmp_path / 'cli.db',
    settings=settings,
    discover=lambda _: (),
  )
  selected = controller.selected
  assert controller.cycle_workspace_tag() == workspace_tags.DEFAULT_TAGS[0]
  assert controller.cycle_workspace_tag() == workspace_tags.DEFAULT_TAGS[1]
  assert controller.cycle_workspace_tag() is None
  assert controller.selected is selected
  assert not settings.path.exists()
  await controller.close()


class _TagFeedbackRepl(cli._Repl):
  def __init__(
    self,
    path: pathlib.Path,
    controller: workspace_controller.WorkspaceController,
  ) -> None:
    self.feedback: list[actions.ShowText] = []
    super().__init__(
      [sys.executable],
      _config(),
      session_id='tag-feedback',
      workspace=path,
      workspace_controller=controller,
      database=path / 'cli.db',
      extension_registry=extension.ExtensionRegistry(),
    )

  @override
  def dispatch(
    self, *requested: actions.Action
  ) -> tuple[asyncio.Task[None], ...]:
    self.feedback.extend(
      action for action in requested if isinstance(action, actions.ShowText)
    )
    return super().dispatch(*requested)


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['cycle', 'assign blocked', 'clear'])
async def test_tag_assignment_is_silent(
  tmp_path: pathlib.Path, command: str
) -> None:
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=tmp_path / 'cli.db',
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (),
  )
  repl = _TagFeedbackRepl(tmp_path, controller)
  repl.feedback.clear()
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand(command)
  )
  assert repl.feedback == []
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand('list')
  )
  assert repl.feedback == []
  assert repl.selector.selector is not None
  with pytest.raises(ValueError, match='unknown workspace tag'):
    await cli.handle_workspace_effect(
      controller, repl, effects.WorkspaceTagCommand('assign missing')
    )
  await controller.close()


@pytest.mark.asyncio
async def test_tag_picker_custom_icon_lifecycle_is_silent(
  tmp_path: pathlib.Path,
) -> None:
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=tmp_path / 'cli.db',
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (),
  )
  repl = _TagFeedbackRepl(tmp_path, controller)
  repl.feedback.clear()
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand('menu')
  )
  assert repl.selector.selector is not None
  assert repl.selector.selector.current == 'clear'
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand('new 🚀')
  )
  assert repl.selector.selector is not None
  assert repl.selector.selector.id == 'workspace-tag-color'
  assert len(controller.list_workspace_tags()) == 2
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand('create - 🚀')
  )
  created = controller.list_workspaces(tmp_path).workspaces[0].tag
  assert created is not None
  assert created.icon == '🚀'
  assert created.color is None
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand(f'symbol {created.id} ★')
  )
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand(f'color {created.id} #af87ff')
  )
  await cli.handle_workspace_effect(
    controller,
    repl,
    effects.WorkspaceTagCommand(f'rename {created.id} My feature'),
  )
  assert controller.workspace_tag(tmp_path.name) == workspace_tags.WorkspaceTag(
    created.id, 'My feature', '#af87ff', '★'
  )
  await cli.handle_workspace_effect(
    controller,
    repl,
    effects.WorkspaceTagCommand(f'edit {created.id}'),
  )
  assert controller.workspace_tag(tmp_path.name) is not None
  assert repl.selector.selector is not None
  deletion = repl.selector.selector.on_select(
    repl.selector.selector.choices[3].value
  )
  assert isinstance(deletion, actions.WorkspaceTagCommand)
  await cli.handle_workspace_effect(
    controller, repl, effects.WorkspaceTagCommand(deletion.args)
  )
  assert repl.selector.selector is not None
  assert len(controller.list_workspace_tags()) == 2
  assert controller.workspace_tag(tmp_path.name) is None
  assert repl.feedback == []
  await controller.close()


@pytest.mark.asyncio
async def test_spawn_workspace_persists_custom_tag(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  created = tmp_path / 'created'
  main.mkdir()
  created.mkdir()
  database = tmp_path / 'cli.db'
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
    database=database,
    settings=config.Config(tmp_path / 'config.json'),
    discover=lambda _: (worktrees.Worktree(main, 'main'),),
    create=lambda *_: worktrees.Worktree(created, 'feature'),
  )
  tag = workspace_tags.WorkspaceTag('review', 'Review', '#af87ff', '★')
  controller.save_workspace_tag(tag)
  result = await controller.spawn_workspace(
    main,
    'feature',
    None,
    preferences=client_pb2.WorkspacePreferences(tag_id=tag.id),
  )
  assert result.workspace.tag_id == tag.id
  assert result.workspace.tag == tag
  assert controller.workspace_tag('main') is None
  with cli_store.Store(database) as store:
    assert (
      store.worktree_overlays(str(main.resolve()))['feature'].tag_id == tag.id
    )
  await controller.close()


@pytest.mark.asyncio
async def test_list_workspaces_unavailable_source_identifies_directory(
  tmp_path: pathlib.Path,
) -> None:
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=_AgentFactory(),
  )
  source = tmp_path / 'unopened'
  try:
    with pytest.raises(worktrees.WorktreeError, match='unavailable') as error:
      controller.list_workspaces(source)
    assert str(source) in str(error.value)
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_duplicate_folder_handles_route_tasks_independently(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    database=database,
    is_trusted_directory=lambda _: True,
  )
  first = await controller.open_folder(
    tmp_path, pathlib.Path('.'), name='Notes'
  )
  second = await controller.open_folder(
    first.id, pathlib.Path('.'), name='Review'
  )
  await controller.select_workspace(second.id)
  await controller.start_task(first.id, 'first task')
  factory.agents[1].instruction_submitted.clear()
  await controller.send_message(second.id, first.id, 'follow up')
  await factory.agents[1].instruction_submitted.wait()

  assert factory.agents[0].submitted == []
  assert factory.agents[1].submitted[0] == 'first task'
  assert factory.agents[1].submitted[1].endswith('follow up')
  assert factory.agents[2].submitted == []
  with pytest.raises(worktrees.WorktreeError, match='ambiguous'):
    controller.list_workspaces(tmp_path)

  await controller.close()
  with cli_store.Store(database) as store:
    assert store.saved_workspaces(str(tmp_path)) == [
      (first.id, str(tmp_path), 'Notes'),
      (second.id, str(tmp_path), 'Review'),
    ]


@pytest.mark.asyncio
async def test_close_restored_folder_preserves_files_and_persists_closure(
  tmp_path: pathlib.Path,
) -> None:
  folder = tmp_path / 'folder'
  folder.mkdir()
  (folder / 'keep.txt').write_text('keep')
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.save_workspace(str(tmp_path), 'saved-folder', str(folder), 'Notes')
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    database=database,
    is_trusted_directory=lambda _: True,
  )
  assert await controller.close_workspace('saved-folder')
  assert factory.agents[1].closed
  assert (folder / 'keep.txt').read_text() == 'keep'
  with cli_store.Store(database) as store:
    assert store.saved_workspaces(str(tmp_path)) == []
    assert store.closed_workspaces(str(tmp_path)) == {'saved-folder'}
  await controller.close()


@pytest.mark.asyncio
async def test_open_folder_validation_and_trust(tmp_path: pathlib.Path) -> None:
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    database=tmp_path / 'cli.db',
    is_trusted_directory=lambda _: False,
  )
  folder = tmp_path / 'untrusted'
  folder.mkdir()
  with pytest.raises(ValueError, match='not trusted'):
    await controller.open_folder(tmp_path, folder)
  with pytest.raises(FileNotFoundError):
    await controller.open_folder(tmp_path, pathlib.Path('missing'))
  with pytest.raises(ValueError, match='not a directory'):
    await controller.open_folder(tmp_path, pathlib.Path('cli.db'))
  assert len(factory.agents) == 1
  await controller.close()


@pytest.mark.asyncio
async def test_missing_folder_retains_handle_but_cannot_start(
  tmp_path: pathlib.Path,
) -> None:
  folder = tmp_path / 'folder'
  folder.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    database=tmp_path / 'cli.db',
    is_trusted_directory=lambda _: True,
  )
  opened = await controller.open_folder(tmp_path, folder)
  folder.rmdir()
  with pytest.raises(worktrees.WorktreeError, match='not found'):
    await controller.start_task(opened.id, 'do not run')
  assert await controller.close_workspace(opened.id)
  assert not factory.agents[1].started
  await controller.close()


@pytest.mark.asyncio
async def test_close_worktree_suppresses_discovery_without_deleting(
  tmp_path: pathlib.Path,
) -> None:
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=tmp_path,
    reasoning_effort='high',
    create_agent=factory,
    database=tmp_path / 'cli.db',
    is_trusted_directory=lambda _: True,
  )
  initial = controller.list_workspaces(tmp_path).workspaces[0]
  opened = await controller.open_folder(initial.id, tmp_path)
  assert await controller.close_workspace(initial.id)
  await _refresh(controller)
  assert controller.list_workspaces(opened.id).workspaces[0].id == opened.id
  assert tmp_path.is_dir()
  await controller.close()


@pytest.mark.asyncio
async def test_create_worktree_workspace_returns_selectable_task_handle(
  managed_root: pathlib.Path,
) -> None:
  main = managed_root / 'main'
  main.mkdir()
  checkout = managed_root / 'checkout'
  checkout.mkdir()
  factory = _AgentFactory()
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='high',
    create_agent=factory,
    database=managed_root / 'cli.db',
    create=lambda source, name: worktrees.Worktree(
      checkout, name or source.name
    ),
  )
  handle = await controller.create_worktree_workspace(main, 'review')
  assert handle.kind == 'worktree'
  assert handle.cwd == checkout
  assert await controller.select_workspace(handle.id)
  await controller.start_task(handle.id, 'review this')
  assert factory.agents[1].submitted == ['review this']
  assert await controller.close_workspace(handle.id)
  assert checkout.is_dir()
  await controller.close()


@pytest.mark.asyncio
async def test_closed_worktree_is_not_rediscovered_after_restart(
  tmp_path: pathlib.Path,
) -> None:
  main = tmp_path / 'main'
  main.mkdir()
  other = tmp_path / 'other'
  other.mkdir()
  roster = (
    worktrees.Worktree(main, 'main'),
    worktrees.Worktree(other, 'other'),
  )
  database = tmp_path / 'cli.db'
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='high',
    create_agent=_AgentFactory(),
    database=database,
    discover=lambda _: roster,
    display_unmanaged_worktrees=True,
  )
  await _refresh(controller)
  assert await controller.close_workspace('other')
  await _refresh(controller)
  assert len(controller.list_workspaces(main).workspaces) == 1
  assert other.is_dir()
  await controller.close()
  restored = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='high',
    create_agent=_AgentFactory(),
    database=database,
    discover=lambda _: roster,
    display_unmanaged_worktrees=True,
  )
  await _refresh(restored)
  assert len(restored.list_workspaces(main).workspaces) == 1
  assert other.is_dir()
  await restored.close()


@pytest.mark.asyncio
async def test_open_directory_classifies_only_managed_roots_as_worktrees(
  managed_root: pathlib.Path,
) -> None:
  main = managed_root.parent / 'unmanaged'
  main.mkdir()
  (main / '.git').mkdir()
  checkout = managed_root / 'checkout'
  checkout.mkdir()
  (checkout / '.git').mkdir()
  child = checkout / 'child'
  child.mkdir()
  dropped: list[pathlib.Path] = []
  controller = workspace_controller.WorkspaceController(
    _config(),
    workspace=main,
    reasoning_effort='high',
    create_agent=_AgentFactory(),
    database=managed_root / 'cli.db',
    is_trusted_directory=lambda _: True,
    drop=lambda backing, _: dropped.append(backing.path),
  )
  assert controller.list_workspaces(main).workspaces[0].kind == 'folder'
  unmanaged = await controller.open_folder(main, main)
  managed = await controller.open_folder(unmanaged.id, checkout)
  subdirectory = await controller.open_folder(unmanaged.id, child)
  assert unmanaged.kind == subdirectory.kind == 'folder'
  assert managed.kind == 'worktree'
  assert await controller.select_workspace(unmanaged.id)
  assert not await controller.drop(confirmed=True)
  assert dropped == []
  assert await controller.select_workspace(managed.id)
  assert await controller.drop()
  await controller.close()
  assert dropped == [checkout]
