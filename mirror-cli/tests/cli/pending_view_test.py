"""Creation uses the real terminal application and native writer without a worker."""

import asyncio
import contextlib
import dataclasses
import io
from collections.abc import AsyncGenerator
from typing import override

import pyte
import pytest
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  mirror_pb2,
)
from prompt_toolkit import buffer, data_structures, mouse_events
from prompt_toolkit.application import current
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import vt100

from mirror.cli import (
  actions,
  cli,
  extension,
  state,
  workspace_application,
  workspaces,
)
from mirror.cli.components import component
from mirror.cli.composer import frame, pane, surface
from mirror.cli.extensions.repository_status import (
  extension as repository_status,
)
from tests import terminal_io


@pytest.fixture(autouse=True)
def terminal_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv('TERM', 'xterm-256color')
  monkeypatch.delenv('NO_COLOR', raising=False)


class _Widget(extension.Extension[str, None, None]):
  def __init__(self, region: extension.RenderRegion) -> None:
    self.region = region
    self.configured = False
    self.clicked = False

  @classmethod
  @override
  def id(cls) -> str:
    return 'test/widget'

  @staticmethod
  @override
  def initial_state() -> str:
    return 'EXTENSION WIDGET'

  @override
  def render_region(self) -> extension.RenderRegion:
    return self.region

  @override
  async def on_config(self, context: extension.Context) -> None:
    del context
    self.configured = True

  def click(self, event: mouse_events.MouseEvent) -> None:
    del event
    self.clicked = True

  @override
  def render(
    self, global_state: state.State, extension_state: str
  ) -> containers.AnyContainer:
    del global_state
    return containers.Window(
      controls.FormattedTextControl([('', extension_state, self.click)]),
      height=1,
    )


class _Terminal:
  def __init__(self, rows: int, columns: int) -> None:
    self.size = data_structures.Size(rows=rows, columns=columns)
    self.input: terminal_io.Input | None = None
    self.extensions = extension.ExtensionRegistry()
    self.output = io.StringIO()
    self.screen = pyte.Screen(columns, rows)
    self.stream = pyte.Stream(self.screen)
    self.frames: asyncio.Queue[list[str]] = asyncio.Queue()
    self.terminal_output = vt100.Vt100_Output(
      self.output, lambda: self.size, term='xterm-256color', enable_cpr=False
    )

  def rendered(self) -> None:
    self.stream.feed(self.output.getvalue())
    self.output.seek(0)
    self.output.truncate()
    self.frames.put_nowait(self.screen.display.copy())

  async def containing(self, text: str) -> list[str]:
    try:
      async with asyncio.timeout(5):
        while True:
          frame = await self.frames.get()
          if len(frame) == self.size.rows and text in '\n'.join(frame):
            return frame
    except TimeoutError:
      raise AssertionError(
        f'Missing {text!r}:\n' + '\n'.join(self.screen.display)
      ) from None

  def resize(self, rows: int, columns: int) -> None:
    self.size = data_structures.Size(rows=rows, columns=columns)
    self.screen.resize(lines=rows, columns=columns)
    if self.input is not None:
      self.input.resize(self.size)

  async def activate(
    self,
    pending: workspaces.PendingWorkspace,
    app: workspace_application.WorkspaceApplication,
  ) -> str | None:
    return await cli._activate_pending(
      pending,
      _tabs(),
      app,
      theme_name=lambda: 'dark',
      extension_registry=self.extensions,
    )


@contextlib.asynccontextmanager
async def _running(
  terminal: _Terminal,
) -> AsyncGenerator[
  tuple[workspace_application.WorkspaceApplication, terminal_io.Input]
]:
  await terminal.extensions.register(
    repository_status.RepositoryStatusExtension(), None
  )
  with (
    terminal_io.input(terminal.terminal_output, terminal.rendered) as pipe,
    current.create_app_session(input=pipe, output=terminal.terminal_output),
  ):
    terminal.input = pipe
    app = workspace_application.WorkspaceApplication(
      full_screen=True,
      mouse_support=True,
      refresh_interval=0.01,
    )
    app.enable_views()
    started = asyncio.Event()
    running = asyncio.create_task(app.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 5)
    try:
      yield app, pipe
    finally:
      app.exit()
      await asyncio.wait_for(running, 5)
      app.finish_views()
      await terminal.extensions.close()


def _tabs() -> containers.Window:
  return containers.Window(
    controls.FormattedTextControl('workspace tabs\n main  feature'), height=2
  )


def _pending(text: str = '') -> workspaces.PendingWorkspace:
  return workspaces.PendingWorkspace(
    'feature',
    'feature',
    draft=actions.ComposerState(text, len(text)),
    preferences=daemon_pb2.WorkspacePreferences(
      definition=mirror_pb2.AgentDefinition(
        model='test-model', reasoning_effort='medium'
      ),
      approval_mode=agent_pb2.APPROVAL_MODE_AUTO,
    ),
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('frozen_ui_time')
@pytest.mark.parametrize('rows, columns', [(8, 40), (24, 80), (42, 120)])
@pytest.mark.parametrize(
  'text', ['draft', 'long draft\n' * 100 + 'draft'], ids=['short', 'long']
)
async def test_pending_composer_bottom_alignment_and_bounded_input(
  rows: int, columns: int, text: str
) -> None:
  terminal = _Terminal(rows, columns)
  pending = _pending(text)
  async with _running(terminal) as (app, pipe):
    view = asyncio.create_task(terminal.activate(pending, app))
    try:
      frame = await terminal.containing('test-model')
      assert 'main  feature' in frame[-1]
      assert 'workspace tabs' in frame[-2]
      assert 'loading repository…' in frame[-3]
      assert frame[-4] == '▀' * columns
      assert 'test-model' in frame[-5]
      assert not frame[-6].strip()
      assert 'draft' in frame[-7]
      assert 'Workspace not ready' not in '\n'.join(frame)
      assert app.layout.current_buffer is not None
      assert app.layout.current_buffer.text == text
      assert app.layout.current_buffer.cursor_position == len(text)
      pipe.send_text('\t')
      assert await asyncio.wait_for(view, 5) == 'next'
    finally:
      view.cancel()
      await asyncio.gather(view, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.usefixtures('frozen_ui_time')
async def test_pending_status_uses_activity_spinner_and_resizes() -> None:
  terminal = _Terminal(24, 80)
  async with _running(terminal) as (app, pipe):
    view = asyncio.create_task(terminal.activate(_pending(), app))
    try:
      frame = await terminal.containing('Creating feature')
      assert frame[-10].startswith('.+ Creating feature')
      pipe.send_text('editing\x1b[D')
      await terminal.containing('editing')
      terminal.resize(10, 40)
      app.invalidate()
      small = await terminal.containing('loading repository')
      assert len(small) == 10
      assert 'test-model' in small[-5]
      assert 'editing' in small[-7]
      terminal.resize(35, 100)
      app.invalidate()
      large = await terminal.containing('Creating feature')
      assert len(large) == 35
      assert 'test-model' in large[-5]
      assert 'editing' in large[-7]
      assert app.layout.current_buffer is not None
      assert app.layout.current_buffer.cursor_position == 6
      pipe.send_text('\t')
      assert await asyncio.wait_for(view, 5) == 'next'
    finally:
      view.cancel()
      await asyncio.gather(view, return_exceptions=True)


@pytest.mark.asyncio
async def test_pending_draft_focus_and_unavailable_submission_survive_failure() -> (
  None
):
  terminal = _Terminal(24, 80)
  pending = _pending()
  async with _running(terminal) as (app, pipe):
    first = asyncio.create_task(terminal.activate(pending, app))
    await terminal.containing('Creating feature')
    pipe.send_text('draft\x1b[D\r')
    frame = await terminal.containing('› draft')
    assert 'loading repository…' in frame[-3]
    assert 'Workspace not ready' not in '\n'.join(frame)
    assert pending.notice is None
    assert pending.draft == actions.ComposerState('draft', 4)
    pipe.send_text('\t')
    assert await asyncio.wait_for(first, 5) == 'next'
    pending.error = 'checkout failed'
    second = asyncio.create_task(terminal.activate(pending, app))
    try:
      await terminal.containing('checkout failed')
      assert app.layout.current_buffer is not None
      assert app.layout.current_buffer.text == 'draft'
      assert app.layout.current_buffer.cursor_position == 4
      pipe.send_text('X\x1b[Z')
      assert await asyncio.wait_for(second, 5) == 'previous'
      assert pending.draft == actions.ComposerState('drafXt', 5)
      third = asyncio.create_task(terminal.activate(pending, app))
      await terminal.containing('drafXt')
      pipe.send_text('\r')
      assert await asyncio.wait_for(third, 5) == 'drop'
      assert pending.draft == actions.ComposerState('drafXt', 5)
    finally:
      second.cancel()
      await asyncio.gather(second, return_exceptions=True)


@pytest.mark.asyncio
async def test_pending_widgets_follow_registry_replacement() -> None:
  terminal = _Terminal(30, 120)
  await terminal.extensions.register(
    _Widget(extension.RenderRegion.WIDGET_ABOVE_PROMPT), None
  )
  async with _running(terminal) as (app, pipe):
    creating = asyncio.create_task(terminal.activate(_pending(), app))
    try:
      await terminal.containing('EXTENSION WIDGET')
      replacement = _Widget(extension.RenderRegion.WIDGET_BELOW_PROMPT)
      await terminal.extensions.register(replacement, None)
      pipe.send_text('after reload')
      frame = await terminal.containing('after reload')
      assert 'EXTENSION WIDGET' in frame[-3]
      assert 'loading repository…' in frame[-5]
      assert not replacement.configured
    finally:
      app.exit_for_handoff('next')
      assert await asyncio.wait_for(creating, 5) == 'next'


@pytest.mark.asyncio
@pytest.mark.parametrize('region', list(extension.RenderRegion))
async def test_pending_preserves_normal_widget_slots_without_running_hooks(
  region: extension.RenderRegion,
) -> None:
  terminal = _Terminal(30, 120)
  widget = _Widget(region)
  await terminal.extensions.register(widget, None)
  pending = _pending('draft')
  async with _running(terminal) as (app, pipe):
    creating = asyncio.create_task(terminal.activate(pending, app))
    before = await terminal.containing('EXTENSION WIDGET')
    assert 'loading repository…' in '\n'.join(before)
    assert not widget.configured
    row = next(i for i, line in enumerate(before) if 'EXTENSION WIDGET' in line)
    column = before[row].index('EXTENSION WIDGET') + 1
    pipe.send_text(f'\x1b[<0;{column};{row + 1}M\x1b[<0;{column};{row + 1}m\t')
    assert await asyncio.wait_for(creating, 5) == 'next'
    assert not widget.clicked
    extensions = terminal.extensions.create_view()
    snapshot = state.State(
      activity=state.ActivityState(busy=True, status='Working'),
      status_bar=state.StatusBarState('test-model', 'medium', 'auto'),
      session=state.SessionState(conversation_id='created'),
      lifecycle=state.LifecycleState(active=True),
      extensions=extensions.initial_states,
      layout=state.LayoutState(columns=120),
    )
    draft = buffer.Buffer()
    draft.text = 'draft'
    prompt = surface.input_window(draft)
    tabs = _tabs()
    workspace_frame = frame.Frame(
      extensions,
      lambda: snapshot,
      tabs,
      get_size=current.get_app_session().output.get_size,
    )
    composer = pane.Pane(
      workspace_frame.content_state,
      component.ComponentHost(workspace_frame.content_state),
      component.ComponentCollection(pane.components(extensions)),
      prompt,
      tabs,
      fullscreen=True,
      get_size=workspace_frame.content_size,
    )
    ready = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        workspace_frame.wrap(
          containers.HSplit([containers.Window(), composer.dock])
        ),
        focused_element=prompt,
      )
    )
    selected = asyncio.create_task(app.show_view(ready))
    try:
      after = await terminal.containing('Working')
      assert before[-8:] == after[-8:]
      assert 'EXTENSION WIDGET' in '\n'.join(after)
      snapshot = dataclasses.replace(
        snapshot,
        extensions=snapshot.extensions.updated(
          'core/status',
          repository_status.RepositoryStatus('git', 'feature', 12, 3, 2),
        ),
      )
      app.invalidate()
      updated = await terminal.containing('git feature')
      assert before[-1] == updated[-1]
      assert '2 changes' in '\n'.join(updated)
      assert app.layout.current_buffer is draft
    finally:
      app.exit_for_handoff('next')
      assert await asyncio.wait_for(selected, 5) == 'next'
