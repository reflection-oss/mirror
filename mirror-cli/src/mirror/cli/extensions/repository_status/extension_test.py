"""Repository reads follow host visibility even when tabs share a directory."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import subprocess
import tempfile
from collections.abc import AsyncIterator, Sequence
from typing import override
from unittest import mock

import grpc.aio
import pytest
import pytest_asyncio
from mirror.protocol import (
  daemon_pb2,
  daemon_pb2_grpc,
)
from prompt_toolkit.layout import containers, controls

from mirror.cli import (
  actions,
  daemon_client,
  effects,
  extension,
  local_rpc_client,
  shell,
  state,
)
from mirror.cli.extensions.repository_status import (
  extension as repository_status,
)


class _Tab:
  def __init__(
    self,
    registry: extension.ExtensionRegistry,
    directory: pathlib.Path,
    backend: shell.Shell | None = None,
  ) -> None:
    self.state: state.State
    self.reads: list[None] = []
    self.changed = asyncio.Event()
    self.updates: list[repository_status.StatusChanged] = []
    self.failed = asyncio.Event()
    self.next_error: Exception | None = None
    context = extension.Context(
      extension.Workspace(
        directory,
        remote=False,
        shell=backend or shell.Shell(directory, validate=self.read),
      ),
      dispatch=self.dispatch,
      get_state=lambda: self.state,
      enqueue=lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda _name, _arguments: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda _options, _message, _error: pytest.fail(
        'Unexpected agent creation'
      ),
    )
    self.host = registry.create_host(context)
    self.state = state.State(
      activity=state.ActivityState(),
      status_bar=state.StatusBarState('model', 'medium', 'auto'),
      session=state.SessionState('conversation'),
      lifecycle=state.LifecycleState(active=True),
      workspace=str(directory),
      extensions=self.host.initial_states,
    )

  def read(self) -> None:
    self.reads.append(None)
    error, self.next_error = self.next_error, None
    if error is not None:
      self.failed.set()
      raise error

  def dispatch(self, action: actions.Action) -> None:
    states, pending = self.host.reduce(self.state, action)
    self.state = dataclasses.replace(self.state, extensions=states)
    for effect in pending:
      assert isinstance(effect, effects.RunExtensionEffect)
      self.host.handle_effect(effect)
    if isinstance(action, actions.RoutedExtensionAction):
      assert isinstance(action.action, repository_status.StatusChanged)
      self.updates.append(action.action)
      self.changed.set()

  async def wait_for_update(self) -> None:
    await asyncio.wait_for(self.changed.wait(), timeout=3)
    self.changed.clear()

  def finish_edit(self) -> None:
    self.state = dataclasses.replace(
      self.state,
      stream=state.StreamState(
        running=(actions.RunningToolCall('edit', 'edit', '', 'tool', '{}'),)
      ),
    )
    self.dispatch(actions.FinishRunningToolCall('edit'))


@pytest_asyncio.fixture
async def registry() -> extension.ExtensionRegistry:
  registry = extension.ExtensionRegistry()
  await registry.register(repository_status.RepositoryStatusExtension(), None)
  return registry


@pytest.fixture
def repository(tmp_path: pathlib.Path) -> pathlib.Path:
  # A nested Git repository must not inherit its parent's Jujutsu identity.
  (tmp_path / '.jj').mkdir()
  tmp_path = tmp_path / 'repository'
  tmp_path.mkdir()
  subprocess.run(
    ('git', 'init', '--initial-branch=main', str(tmp_path)),
    check=True,
    capture_output=True,
  )
  (tmp_path / 'tracked').write_text('initial\n')
  subprocess.run(('git', '-C', str(tmp_path), 'add', 'tracked'), check=True)
  return tmp_path


@pytest_asyncio.fixture
async def transport(
  repository: pathlib.Path,
) -> AsyncIterator[local_rpc_client.Transport]:
  class Daemon(daemon_pb2_grpc.DaemonRPCServicer):
    attachment = 0

    @override
    def AttachWorkspace(  # noqa: N802, generated RPC name
      self,
      request: daemon_pb2.AttachWorkspaceRequest,
      context: grpc.ServicerContext,
    ) -> daemon_pb2.AttachWorkspaceResponse:
      del context
      self.attachment += 1
      return daemon_pb2.AttachWorkspaceResponse(
        attachment_id=str(self.attachment),
        worker_instance_id='worker',
        workspace=daemon_pb2.Workspace(
          id=request.workspace_id, session_id=request.expected_session_id
        ),
      )

    @override
    async def Exec(  # noqa: N802, generated RPC name
      self,
      request: daemon_pb2.ExecRequest,
      context: grpc.aio.ServicerContext[
        daemon_pb2.ExecRequest, daemon_pb2.ExecResponse
      ],
    ) -> AsyncIterator[daemon_pb2.ExecResponse]:
      metadata = context.invocation_metadata()
      assert metadata is not None
      assert ('mirror-attachment-id', str(self.attachment)) in metadata
      result = await shell.Shell(repository).run(
        request.argv, data=request.stdin
      )
      yield daemon_pb2.ExecResponse(stdout=result.stdout)
      yield daemon_pb2.ExecResponse(stderr=result.stderr)
      yield daemon_pb2.ExecResponse(exit_code=result.returncode)

  with tempfile.TemporaryDirectory(prefix='mr-status-') as root:
    address = f'unix:{root}/daemon.sock'
    server = grpc.aio.server()
    daemon_pb2_grpc.add_DaemonRPCServicer_to_server(Daemon(), server)  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
    server.add_insecure_port(address)
    await server.start()
    transport = local_rpc_client.Transport(
      [str(repository / 'mirror-agent')],
      {
        'daemon_address': address,
        'workspace_id': 'workspace',
        'conversation_id': 'session',
      },
      workspace=repository,
      log_dir=repository,
    )
    try:
      yield transport
    finally:
      await transport.close()
      await server.stop(None)


@pytest.mark.asyncio
async def test_status_recovers_after_attachment_is_lost(
  registry: extension.ExtensionRegistry,
  repository: pathlib.Path,
  transport: local_rpc_client.Transport,
) -> None:
  tab = _Tab(
    registry,
    repository,
    daemon_client.Shell(repository, connection=transport.workspace_connection),
  )
  try:
    # Foreground hooks before startup must not try to attach or launch a worker.
    assert not await tab.host.foreground()
    connection = await transport.connect()
    await tab.host.start()
    await tab.host.foreground()
    await tab.wait_for_update()
    assert isinstance(
      tab.state.extensions['core/status'], repository_status.RepositoryStatus
    )

    # The transport is closed before the reconnect callback reaches the frontend.
    await transport.close()
    tab.finish_edit()
    await tab.wait_for_update()
    status = tab.host.status('core/status')
    assert status is not None and status.error is None
    assert not isinstance(
      tab.state.extensions['core/status'], repository_status.RepositoryStatus
    )

    assert connection.reconnect is not None
    await connection.reconnect(False)
    (repository / 'tracked').write_text('reconnected\nextra\n')
    tab.dispatch(actions.WorkerStarted())
    await tab.wait_for_update()
    observed = tab.state.extensions['core/status']
    assert isinstance(observed, repository_status.RepositoryStatus)
    assert observed.additions == 2
  finally:
    await tab.host.close()
    await registry.close()


@dataclasses.dataclass(frozen=True)
class _BlockedDiff(shell.Shell):
  entered: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
  release: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

  @override
  async def run(
    self,
    argv: Sequence[str],
    *,
    data: str = '',
    timeout: float = 30,
    limit: int = 2 * 1024 * 1024,
  ) -> shell.Result:
    result = await super().run(argv, data=data, timeout=timeout, limit=limit)
    if argv[-1].startswith('git diff ') and not self.entered.is_set():
      self.entered.set()
      await self.release.wait()
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'transition',
  [actions.WorkerReconnecting('disconnected'), actions.WorkerStopped()],
)
async def test_status_discards_poll_started_before_reattachment(
  registry: extension.ExtensionRegistry,
  repository: pathlib.Path,
  transition: actions.Action,
) -> None:
  backend = _BlockedDiff(repository)
  tab = _Tab(registry, repository, backend)
  try:
    await tab.host.start()
    await tab.host.foreground()
    await asyncio.wait_for(backend.entered.wait(), timeout=3)
    tab.dispatch(transition)
    plugin = repository_status.RepositoryStatusExtension()
    rendered = plugin.render(tab.state, plugin.require_state(tab.state))
    assert isinstance(rendered, containers.VSplit)
    details = rendered.children[-1]
    assert isinstance(details, containers.Window)
    assert isinstance(details.content, controls.FormattedTextControl)
    assert details.content.text == [('class:footer', 'repository unavailable ')]

    (repository / 'tracked').write_text('new attachment\nextra\n')
    tab.dispatch(actions.WorkerStarted())
    backend.release.set()
    await tab.wait_for_update()
    [update] = tab.updates
    assert isinstance(update.value, repository_status.RepositoryStatus)
    assert update.value.additions == 2
  finally:
    await tab.host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_status_keeps_genuine_faults_disabled_after_reconnection(
  registry: extension.ExtensionRegistry, repository: pathlib.Path
) -> None:
  tab = _Tab(registry, repository)
  try:
    await tab.host.start()
    await tab.host.foreground()
    await tab.wait_for_update()
    tab.next_error = RuntimeError('status bug')
    tab.finish_edit()
    await asyncio.wait_for(tab.failed.wait(), timeout=3)
    tab.dispatch(actions.WorkerStarted())
    status = tab.host.status('core/status')
    assert status is not None and status.status == 'disabled'
    assert status.error == 'RuntimeError: status bug'
  finally:
    await tab.host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_shared_folder_tabs_keep_visibility_and_refreshes_independent(
  registry: extension.ExtensionRegistry, repository: pathlib.Path
) -> None:
  first = _Tab(registry, repository)
  second = _Tab(registry, repository)
  try:
    await first.host.start()
    await second.host.start()
    await first.host.foreground()
    await first.wait_for_update()
    await asyncio.sleep(0.3)
    assert second.reads == []

    await first.host.background()
    await second.host.foreground()
    await second.wait_for_update()
    await first.host.close()
    (repository / 'tracked').write_text('edited\nextra\n')
    second.finish_edit()
    await second.wait_for_update()
    observed = second.state.extensions['core/status']
    assert isinstance(observed, repository_status.RepositoryStatus)
    assert observed.additions == 2

    await second.host.background()
    (repository / 'tracked').write_text('edited again\nextra\nthird\n')
    await second.host.foreground()
    await second.wait_for_update()
    observed = second.state.extensions['core/status']
    assert isinstance(observed, repository_status.RepositoryStatus)
    assert observed.additions == 3
  finally:
    await first.host.close()
    await second.host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_status_outside_repository_shows_workspace_path(
  registry: extension.ExtensionRegistry, tmp_path: pathlib.Path
) -> None:
  tab = _Tab(registry, tmp_path)
  try:
    await tab.host.start()
    await tab.host.foreground()
    await tab.wait_for_update()
    plugin = repository_status.RepositoryStatusExtension()
    rendered = plugin.render(tab.state, plugin.require_state(tab.state))
    assert isinstance(rendered, containers.VSplit)
    path, _, status = rendered.children
    assert isinstance(path, containers.Window)
    assert isinstance(path.content, controls.FormattedTextControl)
    assert path.content.text == [('class:footer', f'   {tmp_path} ')]
    assert isinstance(status, containers.Window)
    assert status.align == containers.WindowAlign.RIGHT
    assert isinstance(status.content, controls.FormattedTextControl)
    assert status.content.text == [
      ('class:footer', 'not a Git/Jujutsu repository ')
    ]
    assert tab.host.status('core/status') == extension.ExtensionStatus(
      extension.Workspace(tmp_path, remote=False), 'active', None
    )
  finally:
    await tab.host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_status_timeout_preserves_counts_and_allows_the_next_refresh(
  registry: extension.ExtensionRegistry, repository: pathlib.Path
) -> None:
  tab = _Tab(registry, repository)
  try:
    await tab.host.start()
    await tab.host.foreground()
    await tab.wait_for_update()
    previous = tab.state.extensions['core/status']

    tab.next_error = TimeoutError()
    tab.finish_edit()
    await asyncio.wait_for(tab.failed.wait(), timeout=3)
    assert tab.state.extensions['core/status'] == previous

    (repository / 'tracked').write_text('edited\nextra\n')
    tab.finish_edit()
    await tab.wait_for_update()
    observed = tab.state.extensions['core/status']
    assert isinstance(observed, repository_status.RepositoryStatus)
    assert observed.additions == 2
  finally:
    await tab.host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_hiding_during_debounce_defers_reads_until_next_foreground(
  registry: extension.ExtensionRegistry, repository: pathlib.Path
) -> None:
  tab = _Tab(registry, repository)
  try:
    await tab.host.start()
    await tab.host.foreground()
    await asyncio.sleep(0.05)
    await tab.host.background()
    tab.finish_edit()
    await asyncio.sleep(0.3)
    assert tab.reads == []
    await tab.host.foreground()
    await tab.wait_for_update()
    assert tab.reads
  finally:
    await tab.host.close()
    await registry.close()
