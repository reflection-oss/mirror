from __future__ import annotations

import asyncio
import contextlib
import io
import pathlib
import sys
from collections.abc import Callable, Iterator
from unittest import mock

import pyte
import pytest
from mirror.protocol import daemon_pb2
from prompt_toolkit import buffer, data_structures
from prompt_toolkit.application import current, dummy
from prompt_toolkit.input import base as input_base
from prompt_toolkit.input import defaults
from prompt_toolkit.layout import containers, mouse_handlers, screen
from prompt_toolkit.output import base, vt100

from mirror.cli import (
  actions,
  cli,
  daemon_client,
  extension,
  pending_view,
  rendering,
  state,
  workspace_application,
  workspace_controller,
  workspaces,
)
from mirror.cli.components import spinner
from mirror.cli.extensions.repository_status import (
  extension as repository_status,
)
from mirror.cli.sandbox import setup as sandbox_setup
from tests import terminal_io


@contextlib.contextmanager
def _pending_session(
  output: base.Output, rendered: Callable[[], None] = lambda: None
) -> Iterator[input_base.PipeInput]:
  """Open a piped app session with one application current until the view runs.

  The CLI activates a pending view on its running workspace screen. Without an
  application current, each `get_app()` while the view builds its layout
  constructs a fresh `DummyApplication` with every default key binding, and
  the few hundred calls spend seconds of a test's bound before a key is read.
  """
  with (
    terminal_io.input(output, rendered)
    if isinstance(output, vt100.Vt100_Output)
    else defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
    current.set_app(dummy.DummyApplication()),
  ):
    yield pipe


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('keys', 'expected'),
  [
    ('\t', 'next'),
    ('\x1b[Z', 'previous'),
    ('\x1b', 'cancel-sandbox'),
    ('\x1b[27u', 'cancel-sandbox'),
    ('\x1bb\r', 'install-sandbox'),
    ('\r', 'install-sandbox'),
  ],
)
async def test_pending_sandbox_owns_approval_and_navigation(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  keys: str,
  expected: str,
) -> None:
  monkeypatch.setattr(sandbox_setup.Options, 'runtime_ready', lambda _: False)
  monkeypatch.setattr(sandbox_setup.platform, 'system', lambda: 'Darwin')
  monkeypatch.setattr(sandbox_setup.platform, 'machine', lambda: 'arm64')
  source = workspaces.AgentWorkspace(
    'source',
    daemon_pb2.Workspace(
      id='source',
      directory=str(tmp_path),
      name='main',
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    mock.Mock(
      spec=workspaces.Agent,
      model='test-model',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    False,
  )
  pending = workspaces.SandboxWorkspace('sandbox', '◇ test', source=source)
  run = workspace_application.WorkspaceApplication.run_async

  async def without_sequence_timeout(
    app: workspace_application.WorkspaceApplication,
  ) -> str | None:
    app.timeoutlen = None  # Standalone Escape must not wait for an Alt chord.
    return await run(app)

  monkeypatch.setattr(
    workspace_application.WorkspaceApplication,
    'run_async',
    without_sequence_timeout,
  )
  with _pending_session(base.DummyOutput()) as pipe:
    pipe.send_text(keys)
    result = await asyncio.wait_for(
      cli._activate_pending(pending, containers.Window(height=0)), 3
    )
  assert result == expected
  assert pending.available


@pytest.mark.asyncio
@pytest.mark.parametrize('failed', [False, True])
async def test_pending_sandbox_can_leave_running_or_failed_setup(
  tmp_path: pathlib.Path, failed: bool
) -> None:
  source = workspaces.AgentWorkspace(
    'source',
    daemon_pb2.Workspace(
      id='source',
      directory=str(tmp_path),
      name='main',
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    mock.Mock(
      spec=workspaces.Agent,
      model='test-model',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    False,
  )
  pending = workspaces.SandboxWorkspace(
    'sandbox',
    '◇ test',
    source=source,
    task=None if failed else mock.Mock(spec=asyncio.Task, done=lambda: False),
    phase='Starting sandbox…',
    error='startup failed' if failed else None,
    draft=actions.ComposerState('unfinished draft', 4),
  )
  with _pending_session(base.DummyOutput()) as pipe:
    pipe.send_text('\t')
    assert (
      await asyncio.wait_for(
        cli._activate_pending(pending, containers.Window(height=0)), 3
      )
      == 'next'
    )
  assert pending.available
  assert not pending.finished
  assert pending.draft == actions.ComposerState('unfinished draft', 4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('columns', 'rows'),
  [(40, 12), (100, 24), (120, 42)],
)
async def test_pending_sandbox_renders_live_phase_and_spinner(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  columns: int,
  rows: int,
) -> None:
  source = workspaces.AgentWorkspace(
    'source',
    daemon_pb2.Workspace(
      id='source',
      directory=str(tmp_path),
      name='main',
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    mock.Mock(
      spec=workspaces.Agent,
      model='test-model',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    False,
  )
  pending = workspaces.SandboxWorkspace(
    'sandbox',
    '◇ test',
    source=source,
    task=mock.Mock(spec=asyncio.Task, done=lambda: False),
    phase='Starting sandbox…',
  )
  registry = extension.ExtensionRegistry()
  await registry.register(repository_status.RepositoryStatusExtension(), None)
  terminal = io.BytesIO()
  output = vt100.Vt100_Output(
    io.TextIOWrapper(terminal, encoding='utf-8'),
    lambda: data_structures.Size(rows=rows, columns=columns),
    term='xterm-256color',
    enable_cpr=False,
  )
  frame = asyncio.Event()
  monkeypatch.setattr(spinner, 'render', lambda *args, **kwargs: [('', '*+')])

  def presented() -> None:
    snapshot = pyte.Screen(columns, rows)
    pyte.Stream(snapshot).feed(terminal.getvalue().decode())
    if f'*+ {pending.phase}' in '\n'.join(snapshot.display):
      frame.set()

  with _pending_session(output, presented) as pipe:
    task = asyncio.create_task(
      cli._activate_pending(
        pending, containers.Window(height=0), extension_registry=registry
      )
    )
    try:
      await asyncio.wait_for(frame.wait(), 3)
      screen = pyte.Screen(columns, rows)
      pyte.Stream(screen).feed(terminal.getvalue().decode())
      displayed = '\n'.join(screen.display)
      assert 'test-model' in screen.display[-3]
      assert 'loading repository…' in screen.display[-1]
      activity_row = next(
        i
        for i, line in enumerate(screen.display)
        if '*+ Starting sandbox' in line
      )
      assert not screen.display[activity_row + 1].strip(' ⠂')
      assert screen.display[activity_row + 3].strip() == rendering.PROMPT
      assert rendering.PROMPT in displayed
      assert 'test-model' in displayed
      assert 'git main' not in displayed
      assert 'Esc to cancel' in displayed
      assert 'Workspace not ready' not in displayed
      frame.clear()
      pending.phase = 'Loading workspace…'
      await asyncio.wait_for(frame.wait(), 3)
      screen = pyte.Screen(columns, rows)
      pyte.Stream(screen).feed(terminal.getvalue().decode())
      assert '*+ Loading workspace' in '\n'.join(screen.display)
      pipe.send_text('draft for the sandbox\t')
      assert await asyncio.wait_for(task, 3) == 'next'
      assert pending.draft == actions.ComposerState('draft for the sandbox', 21)
    finally:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('keys', 'expected'), [('\r', 'start-sandbox'), ('2', 'copy-sandbox')]
)
async def test_cached_sandbox_offers_empty_or_repository_copy(
  keys: str,
  expected: str,
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(sandbox_setup.Options, 'runtime_ready', lambda _: True)
  monkeypatch.setattr(sandbox_setup.Options, 'image_ready', lambda _: True)
  source = workspaces.AgentWorkspace(
    'source',
    daemon_pb2.Workspace(
      id='source',
      directory=str(tmp_path),
      name='main',
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    mock.Mock(
      spec=workspaces.Agent,
      model='test-model',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    False,
  )
  pending = workspaces.SandboxWorkspace('sandbox', '◇ cached', source=source)
  with _pending_session(base.DummyOutput()) as pipe:
    pipe.send_text(keys)
    assert (
      await asyncio.wait_for(
        cli._activate_pending(pending, containers.Window(height=0)), 3
      )
      == expected
    )


def test_ready_sandbox_renders_the_standard_selector_surface(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('TERM', 'xterm-256color')
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setattr(sandbox_setup.Options, 'runtime_ready', lambda _: True)
  monkeypatch.setattr(sandbox_setup.Options, 'image_ready', lambda _: True)
  source = workspaces.AgentWorkspace(
    'source',
    daemon_pb2.Workspace(
      id='source',
      directory=str(tmp_path),
      name='main',
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    mock.Mock(
      spec=workspaces.Agent,
      model='test-model',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    False,
  )
  pending = workspaces.SandboxWorkspace('sandbox', '◇ test', source=source)
  output = vt100.Vt100_Output(
    io.TextIOWrapper(io.BytesIO(), encoding='utf-8'),
    lambda: data_structures.Size(rows=42, columns=120),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    app = workspace_application.WorkspaceApplication(
      layout=pending_view.layout_for(
        pending,
        lambda: state.MenuState(active=True, selected=2, max_rows=3),
        containers.Window(height=0),
        buffer.Buffer(),
        lambda: 'dark',
        extension.ExtensionRegistry().create_view(),
      )
    )
    rendered = screen.Screen(initial_width=120, initial_height=42)
    with current.set_app(app):
      app.layout.container.write_to_screen(
        rendered,
        mouse_handlers.MouseHandlers(),
        screen.WritePosition(xpos=0, ypos=0, width=120, height=42),
        '',
        False,
        None,
      )
  lines = [
    ''.join(rendered.data_buffer[row][column].char for column in range(120))
    for row in range(42)
  ]
  heading = next(i for i, line in enumerate(lines) if 'Sandbox  (3/3)' in line)
  assert lines[heading - 1] == '▄' * 120
  assert 'class:menu-surface' in rendered.data_buffer[heading][20].style
  assert '1. Empty workspace' in lines[heading + 1]
  assert '2. Copy current repository' in lines[heading + 2]
  assert '3. Cancel' in lines[heading + 3]
  assert '⠂' not in '\n'.join(lines[heading - 1 : heading + 5])


@pytest.mark.asyncio
async def test_ready_sandbox_restores_draft_without_submitting(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
) -> None:
  controller = mock.Mock(spec=workspace_controller.WorkspaceController)
  agent = cli._Repl(
    [sys.executable],
    {'model': 'test-model', 'conversation_id': 'test'},
    session_id='test',
    metadata=workspace_client,
    workspace=tmp_path,
    workspace_controller=controller,
    extension_registry=extension.ExtensionRegistry(),
  )
  workspace = workspaces.AgentWorkspace(
    'sandbox',
    daemon_pb2.Workspace(
      id='sandbox',
      directory=str(tmp_path),
      name='sandbox',
      kind=daemon_pb2.WORKSPACE_KIND_SANDBOX,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    agent,
    False,
    draft=actions.ComposerState('review this code when ready', 7),
  )
  controller.workspaces = [workspace]
  controller.visible_workspaces.return_value = (workspace,)
  started = asyncio.create_task(asyncio.sleep(0))
  await started
  with (
    mock.patch.object(
      workspace, 'start', new_callable=mock.AsyncMock, return_value=started
    ),
    mock.patch.object(
      agent, 'activate', new_callable=mock.AsyncMock, return_value='next'
    ),
    mock.patch.object(agent, 'submit_instruction') as submit,
  ):
    assert await cli._activate_workspace(controller, workspace, False) == 'next'
    assert agent._input.text == 'review this code when ready'
    assert agent._input.cursor_position == 7
    assert workspace.draft is None
    submit.assert_not_called()
  await agent.close()
