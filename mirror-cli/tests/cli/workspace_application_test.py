"""Workspace handoffs retain one terminal owner and release views on shutdown."""

import asyncio
import base64
import subprocess
from typing import override

import pytest
from prompt_toolkit import buffer, data_structures
from prompt_toolkit.application import application, current
from prompt_toolkit.input import defaults
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import base

from mirror.protocol import agent_pb2

from mirror.cli import (
  actions,
  screen_selection,
  transcript,
  transcript_view,
  workspace_application,
)


@pytest.fixture(autouse=True)
def _disable_system_clipboard_bridge(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Keep copy_text tests from touching a developer's real clipboard."""
  monkeypatch.setattr(workspace_application, '_clipboard_candidates', lambda: [])


class _Output(base.DummyOutput):
  def __init__(self, columns: int = 80) -> None:
    self.columns = columns
    self.entered = 0
    self.left = 0
    self.raw: list[str] = []
    self.text: list[str] = []

  @override
  def get_size(self) -> data_structures.Size:
    return data_structures.Size(rows=40, columns=self.columns)

  @override
  def write(self, data: str) -> None:
    self.text.append(data)

  @override
  def write_raw(self, data: str) -> None:
    self.raw.append(data)

  @override
  def enter_alternate_screen(self) -> None:
    self.entered += 1

  @override
  def quit_alternate_screen(self) -> None:
    self.left += 1


@pytest.mark.asyncio
async def test_run_views_waits_for_the_first_layout_before_rendering() -> None:
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    started, ready, rendered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    screen = workspace_application.WorkspaceApplication(
      full_screen=True,
      input=pipe,
      output=output,
      max_render_postpone_time=0,
      after_render=lambda _: rendered.set(),
    )
    view = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.Window(controls.FormattedTextControl('launch screen'))
      ),
      input=pipe,
      output=output,
    )

    async def workspaces() -> None:
      started.set()
      await ready.wait()
      await screen.show_view(view)

    running = asyncio.create_task(screen.run_views(workspaces))
    try:
      await asyncio.wait_for(started.wait(), 1)
      screen.invalidate()
      await asyncio.sleep(0)
      assert output.entered == 0
      assert output.text == []
      assert not rendered.is_set()

      ready.set()
      await asyncio.wait_for(rendered.wait(), 1)
      assert screen.layout is view.layout
      assert ''.join(output.text).strip() == 'launch screen'
      assert output.entered == 1
    finally:
      screen.exit()
      await asyncio.wait_for(running, 1)
    assert 'No layout specified' not in ''.join(output.text)
    assert output.left == 1


@pytest.mark.asyncio
async def test_workspace_handoff_keeps_one_application_and_terminal_screen() -> (
  None
):
  output = _Output()
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    screen = workspace_application.WorkspaceApplication(full_screen=True)
    screen.enable_views()
    rendered = asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    screen.after_render += after_render
    started = asyncio.Event()
    run = asyncio.create_task(screen.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 1)
    for name in ('alpha', 'beta'):
      view = workspace_application.WorkspaceApplication(
        layout=layout.Layout(
          containers.Window(controls.FormattedTextControl(name))
        )
      )
      view.ttimeoutlen = 0.05
      rendered.clear()
      selected = asyncio.create_task(screen.show_view(view))
      await asyncio.wait_for(rendered.wait(), 1)
      assert screen.layout is view.layout
      assert screen.vi_state is view.vi_state
      assert screen.emacs_state is view.emacs_state
      assert screen.clipboard is view.clipboard
      assert screen.ttimeoutlen == 0.05
      assert output.entered == 1
      assert output.left == 0
      if name == 'alpha':
        screen.exit_for_handoff('next')
        assert await asyncio.wait_for(selected, 1) == 'next'
        assert not run.done()
      else:
        screen.exit()
        await asyncio.wait_for(run, 1)
        screen.finish_views()
        assert await asyncio.wait_for(selected, 1) is None
    assert output.left == 1


@pytest.mark.asyncio
async def test_workspace_handoff_does_not_repaint_the_backgrounded_layout() -> (
  None
):
  output = _Output()
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    screen = workspace_application.WorkspaceApplication(
      full_screen=True, max_render_postpone_time=0
    )
    screen.enable_views()
    rendered = asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    screen.after_render += after_render
    started = asyncio.Event()
    run = asyncio.create_task(screen.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 1)
    body = containers.HSplit([
      containers.Window(controls.FormattedTextControl('alpha composer')),
      containers.Window(
        controls.FormattedTextControl('workspace tabs'), height=1
      ),
    ])
    first = workspace_application.WorkspaceApplication(
      layout=layout.Layout(body)
    )
    rendered.clear()
    selected = asyncio.create_task(screen.show_view(first))
    await asyncio.wait_for(rendered.wait(), 1)
    try:
      screen.exit_for_handoff('next')
      assert await asyncio.wait_for(selected, 1) == 'next'
      body.children.pop()
      previous_frame = output.raw.copy()
      rendered.clear()
      screen.invalidate()
      # Let the scheduled redraw run while the next workspace is not ready.
      await asyncio.sleep(0)
      assert not rendered.is_set()
      assert output.raw == previous_frame

      second = workspace_application.WorkspaceApplication(
        layout=layout.Layout(
          containers.Window(controls.FormattedTextControl('beta composer'))
        )
      )
      foreground_ready = asyncio.Event()
      selected = asyncio.create_task(
        screen.show_view(second, before_input=foreground_ready.wait)
      )
      await asyncio.wait_for(rendered.wait(), 1)
      foreground_ready.set()
      assert screen.layout is second.layout
      assert output.entered == 1
      assert output.left == 0
    finally:
      screen.exit()
      await asyncio.wait_for(run, 1)
      screen.finish_views()
      await asyncio.wait_for(selected, 1)
    assert output.left == 1


@pytest.mark.asyncio
async def test_drag_through_terminal_input_requests_the_selected_clipboard_text() -> (
  None
):
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    screen = workspace_application.WorkspaceApplication(
      input=pipe, output=output, full_screen=True, mouse_support=True
    )
    screen.enable_views()
    rendered = asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    screen.after_render += after_render
    started = asyncio.Event()
    run = asyncio.create_task(screen.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 1)
    view = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.Window(
          controls.FormattedTextControl('alpha\ncomposer\ngit main')
        )
      )
    )
    rendered.clear()
    selected = asyncio.create_task(screen.show_view(view))
    await asyncio.wait_for(rendered.wait(), 1)
    rendered.clear()
    pipe.send_text('\x1b[<0;1;1M\x1b[<32;8;3M\x1b[<0;8;3m')
    await asyncio.wait_for(rendered.wait(), 1)
    encoded = base64.b64encode(b'alpha\ncomposer\ngit main').decode('ascii')
    assert f'\x1b]52;c;{encoded}\x1b\\' in output.raw
    screen.exit()
    await asyncio.wait_for(run, 1)
    screen.finish_views()
    assert await asyncio.wait_for(selected, 1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('drag', [False, True])
@pytest.mark.parametrize('width', [10, 80])
async def test_code_hover_copies_source_and_preserves_drag_selection(
  drag: bool,
  width: int,
) -> None:
  output = _Output(width)
  body = '\t界🙂e\u0301 ┃  ' + 'long ' * 25 + '\n\n'
  history = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DETAIL,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  history.append(
    agent_pb2.AgentMessage(
      id='code',
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(content=f'```text\n{body}```'),
        success=True,
      ),
    )
  )
  viewport = transcript_view.TranscriptView(history.blocks, lambda: ())
  with defaults.create_pipe_input() as pipe:
    owner = workspace_application.WorkspaceApplication(
      input=pipe,
      output=output,
      full_screen=True,
      mouse_support=True,
    )
    owner.enable_views()
    rendered, started = asyncio.Event(), asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      if viewport.create_content(width, 39).line_count:
        rendered.set()

    owner.after_render += after_render
    running = asyncio.create_task(owner.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 1)
    view = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.HSplit([
          containers.Window(viewport, wrap_lines=False),
          containers.Window(
            controls.FormattedTextControl('composer'), height=1
          ),
        ])
      ),
      input=pipe,
      output=output,
    )
    selected = asyncio.create_task(owner.show_view(view))
    try:
      await asyncio.wait_for(rendered.wait(), 1)
      viewport.first()
      rendered.clear()
      owner.invalidate()
      await asyncio.wait_for(rendered.wait(), 1)
      content = viewport.create_content(width, 39)
      lines = [
        ''.join(part[1] for part in content.get_line(row))
        for row in range(content.line_count)
      ]
      row = next(index for index, line in enumerate(lines) if '┃' in line)
      rendered.clear()
      pipe.send_text(f'\x1b[<35;{lines[row].index("┃") + 1};{row + 1}M')
      await asyncio.wait_for(rendered.wait(), 1)
      row += 1
      line = ''.join(part[1] for part in content.get_line(row))
      column = line.index('⧉') + 1
      end = column - 1 if drag else column
      rendered.clear()
      pipe.send_text(f'\x1b[<0;{column};{row + 1}M')
      await asyncio.wait_for(rendered.wait(), 1)
      rendered.clear()
      pipe.send_text(f'\x1b[<32;{end};{row + 1}M\x1b[<0;{end};{row + 1}m')
      await asyncio.wait_for(rendered.wait(), 1)
      expected = ' ⧉' if drag else body
      encoded = base64.b64encode(expected.encode()).decode()
      assert [
        value for value in output.raw if value.startswith('\x1b]52;')
      ] == [f'\x1b]52;c;{encoded}\x1b\\']
      label = ''.join(
        part[1] for part in viewport.create_content(width, 39).get_line(row)
      )
      assert ('✓' in label) is not drag
      rendered.clear()
      pipe.send_text(f'\x1b[<35;{lines[row].index("┃") + 1};{row + 1}M')
      await asyncio.wait_for(rendered.wait(), 1)
      rendered.clear()
      pipe.send_text('\x1b[<35;1;40M')
      await asyncio.wait_for(rendered.wait(), 1)
      assert '⧉' not in ''.join(part[1] for part in content.get_line(row))
      assert '✓' not in ''.join(part[1] for part in content.get_line(row))
    finally:
      owner.exit()
      await asyncio.wait_for(running, 1)
      owner.finish_views()
      await asyncio.wait_for(selected, 1)


@pytest.mark.asyncio
async def test_code_copy_feedback_and_pointer_reset_without_disabling_clicks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('TERM_PROGRAM', 'ghostty')
  monkeypatch.delenv('TMUX', raising=False)
  monkeypatch.delenv('STY', raising=False)
  output = _Output(40)
  history = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DETAIL,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  history.append(
    agent_pb2.AgentMessage(
      id='code',
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(content='```\nrepeat\n```'),
        success=True,
      ),
    )
  )
  viewport = transcript_view.TranscriptView(history.blocks, lambda: ())
  selected = screen_selection.ScreenSelection(
    containers.HSplit([
      containers.Window(viewport, wrap_lines=False),
      containers.Window(controls.FormattedTextControl('composer'), height=1),
    ]),
    workspace_application.copy_text,
  )
  rendered, reset = asyncio.Event(), asyncio.Event()

  def after_render(_: application.Application[str | None]) -> None:
    content = viewport.create_content(40, 39)
    if content.line_count < 3:
      return
    rendered.set()
    if '⧉' in ''.join(part[1] for part in content.get_line(1)):
      reset.set()

  with defaults.create_pipe_input() as pipe:
    owner = workspace_application.WorkspaceApplication(
      input=pipe,
      output=output,
      layout=layout.Layout(selected),
      full_screen=True,
      mouse_support=True,
      after_render=after_render,
    )
    running = asyncio.create_task(owner.run_async())

    async def send(sequence: str) -> None:
      rendered.clear()
      pipe.send_text(sequence)
      await asyncio.wait_for(rendered.wait(), 1)

    try:
      await asyncio.wait_for(rendered.wait(), 1)
      await send('\x1b[<35;38;2M')
      assert '\x1b]22;pointer\x1b\\' in output.raw
      await send('\x1b[<0;38;2M\x1b[<0;38;2m')
      await send('\x1b[<0;38;2M\x1b[<0;38;2m')
      content = viewport.create_content(40, 39)
      assert 'Copied ✓' in ''.join(part[1] for part in content.get_line(1))
      assert (
        len([value for value in output.raw if value.startswith('\x1b]52;')])
        == 2
      )
      reset.clear()
      await send('\x1b[<0;38;2M')
      await asyncio.wait_for(reset.wait(), 3)
      assert 'Copied' not in ''.join(part[1] for part in content.get_line(1))
      await send('\x1b[<0;38;2m')
      assert (
        len([value for value in output.raw if value.startswith('\x1b]52;')])
        == 3
      )
      await send('\x1b[<35;1;40M')
      assert [value for value in output.raw if value.startswith('\x1b]22;')][
        -1
      ] == '\x1b]22;\x1b\\'
      await send('\x1b[<0;5;3M\x1b[<32;8;3M\x1b[<0;8;3m')
      await send('\x1b[<35;38;2M')
      assert [value for value in output.raw if value.startswith('\x1b]22;')][
        -1
      ] == '\x1b]22;pointer\x1b\\'
    finally:
      owner.exit()
      await asyncio.wait_for(running, 1)
  assert [value for value in output.raw if value.startswith('\x1b]22;')] == [
    '\x1b]22;pointer\x1b\\',
    '\x1b]22;\x1b\\',
    '\x1b]22;pointer\x1b\\',
    '\x1b]22;\x1b\\',
  ]


def test_copy_text_bounds_payload_without_touching_the_terminal_clipboard() -> (
  None
):
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    app = application.Application[str | None](input=pipe, output=output)
    with current.set_app(app):
      with pytest.raises(ValueError, match='75 KB'):
        workspace_application.copy_text('界' * 25_001)
      assert output.raw == []
      workspace_application.copy_text('界' * 25_000)
      assert len(output.raw) == 1
      assert len(output.raw[0]) == 100_009

def test_copy_text_bridges_to_system_clipboard(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    workspace_application, '_clipboard_candidates', lambda: [('xclip', '-selection', 'clipboard')]
  )
  calls: list[tuple[tuple[str, ...], bytes]] = []

  def fake_run(command, *, input=None, **kwargs):
    calls.append((tuple(command), input))
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(subprocess, 'run', fake_run)
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    app = application.Application[str | None](input=pipe, output=output)
    with current.set_app(app):
      workspace_application.copy_text('hello')
  encoded = base64.b64encode(b'hello').decode('ascii')
  assert output.raw == [f'\x1b]52;c;{encoded}\x1b\\']
  assert calls == [(('xclip', '-selection', 'clipboard'), b'hello')]


def test_copy_text_skips_system_bridge_when_unavailable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(workspace_application, '_clipboard_candidates', lambda: [])
  calls: list[object] = []

  def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    calls.append(args)
    return subprocess.CompletedProcess(args[0] if args else (), 0)

  monkeypatch.setattr(subprocess, 'run', fake_run)
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    app = application.Application[str | None](input=pipe, output=output)
    with current.set_app(app):
      workspace_application.copy_text('hello')
  assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('modal', [False, True])
async def test_keys_in_the_handoff_burst_reach_only_the_next_workspace(
  modal: bool,
) -> None:
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(
      input=pipe,
      output=_Output(),
    ),
  ):
    screen = workspace_application.WorkspaceApplication(full_screen=True)
    screen.enable_views()
    rendered = asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    screen.after_render += after_render
    started = asyncio.Event()
    run = asyncio.create_task(screen.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 1)
    alpha, beta = buffer.Buffer(), buffer.Buffer()
    bindings = key_bindings.KeyBindings()

    @bindings.add('tab', eager=True)
    def next_view(_: key_processor.KeyPressEvent) -> None:
      screen.exit_for_handoff('next')

    first = workspace_application.WorkspaceApplication(
      layout=layout.Layout(containers.Window(controls.BufferControl(alpha))),
      key_bindings=bindings,
    )
    second = workspace_application.WorkspaceApplication(
      layout=layout.Layout(containers.Window(controls.BufferControl(beta))),
    )
    selected = asyncio.create_task(screen.show_view(first))
    rendered.clear()
    await asyncio.wait_for(rendered.wait(), 1)
    pipe.send_text('\tqueued')
    assert await asyncio.wait_for(selected, 1) == 'next'
    assert alpha.text == ''
    next_selection = asyncio.create_task(
      screen.show_view(second, accept_buffered_input=lambda: not modal)
    )
    rendered.clear()
    await asyncio.wait_for(rendered.wait(), 1)
    assert beta.text == ('' if modal else 'queued')
    screen.exit()
    await asyncio.wait_for(run, 1)
    screen.finish_views()
    await asyncio.wait_for(next_selection, 1)


@pytest.mark.asyncio
async def test_startup_can_be_cancelled_before_a_view_is_ready() -> None:
  output = _Output()
  with (
    defaults.create_pipe_input() as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    screen = workspace_application.WorkspaceApplication(full_screen=True)
    screen.enable_views()
    started = asyncio.Event()
    run = asyncio.create_task(screen.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 1)
    pipe.send_text('\x03')
    assert await asyncio.wait_for(run, 1) is None
    assert output.text == []
    assert output.entered == output.left == 0


class _ResizeOutput(_Output):
  def __init__(self) -> None:
    super().__init__()
    self.synchronized = False
    self.unsynchronized_erases = 0

  @override
  def write_raw(self, data: str) -> None:
    if data == '\x1b[?2026h':
      self.synchronized = True
    elif data == '\x1b[?2026l':
      self.synchronized = False

  @override
  def erase_down(self) -> None:
    if not self.synchronized:
      self.unsynchronized_erases += 1


@pytest.mark.asyncio
async def test_fullscreen_resize_does_not_flush_an_erase_before_the_next_frame() -> (
  None
):
  output = _ResizeOutput()
  with defaults.create_pipe_input() as pipe:
    app = workspace_application.WorkspaceApplication(
      full_screen=True,
      input=pipe,
      output=output,
      layout=layout.Layout(
        containers.Window(controls.FormattedTextControl('visible frame'))
      ),
    )
    rendered = asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    app.after_render += after_render
    run = asyncio.create_task(app.run_async())
    try:
      await asyncio.wait_for(rendered.wait(), 1)
      rendered.clear()
      app._on_resize()  # pyright: ignore[reportPrivateUsage] - exercise the native SIGWINCH callback.
      assert output.unsynchronized_erases == 0
      await asyncio.wait_for(rendered.wait(), 1)
      assert output.entered == 1
      assert output.left == 0
      assert output.unsynchronized_erases == 0
    finally:
      app.exit()
      await run


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_run_views_propagates_completion_and_failure(
  failure: bool,
) -> None:
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    screen = workspace_application.WorkspaceApplication(
      full_screen=True,
      input=pipe,
      output=output,
    )

    async def workspaces() -> None:
      if failure:
        raise ValueError('workspace failure')

    if failure:
      with pytest.raises(ValueError, match='workspace failure'):
        await screen.run_views(workspaces)
    else:
      await screen.run_views(workspaces)
    assert output.text == []
    assert output.entered == output.left == 0


@pytest.mark.asyncio
async def test_run_views_can_quit_during_workspace_handoff() -> None:
  output = _Output()
  with defaults.create_pipe_input() as pipe:
    screen = workspace_application.WorkspaceApplication(
      full_screen=True, input=pipe, output=output
    )
    rendered, between_views = asyncio.Event(), asyncio.Event()
    view = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.Window(controls.FormattedTextControl('alpha'))
      ),
      input=pipe,
      output=output,
      after_render=lambda _: rendered.set(),
    )

    async def workspaces() -> None:
      assert await screen.show_view(view) == 'next'
      between_views.set()
      await asyncio.Event().wait()

    running = asyncio.create_task(screen.run_views(workspaces))
    await asyncio.wait_for(rendered.wait(), 1)
    screen.exit_for_handoff('next')
    await asyncio.wait_for(between_views.wait(), 1)
    pipe.send_text('\x03')
    await asyncio.wait_for(running, 1)
    assert output.entered == 1
    assert output.left == 1


@pytest.mark.asyncio
async def test_run_views_waits_for_workspace_cleanup_on_terminal_exit() -> None:
  with defaults.create_pipe_input() as pipe:
    screen = workspace_application.WorkspaceApplication(
      full_screen=True,
      input=pipe,
      output=_Output(),
    )
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def workspaces() -> None:
      started.set()
      try:
        await asyncio.Event().wait()
      finally:
        cleaned.set()

    running = asyncio.create_task(screen.run_views(workspaces))
    await asyncio.wait_for(started.wait(), 1)
    screen.exit()
    await asyncio.wait_for(running, 1)
    assert cleaned.is_set()
