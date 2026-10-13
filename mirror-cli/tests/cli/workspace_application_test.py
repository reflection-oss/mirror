"""Workspace handoffs retain one terminal owner and release views on shutdown."""

import asyncio
import base64
from typing import override

import pytest
from prompt_toolkit import buffer
from prompt_toolkit.application import application, current
from prompt_toolkit.input import defaults
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import base

from mirror.cli import workspace_application


class _Output(base.DummyOutput):
  def __init__(self) -> None:
    self.entered = 0
    self.left = 0
    self.raw: list[str] = []

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
    run = asyncio.create_task(screen.run_async())
    await asyncio.wait_for(rendered.wait(), 1)
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
    run = asyncio.create_task(screen.run_async())
    await asyncio.wait_for(rendered.wait(), 1)
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
    run = asyncio.create_task(screen.run_async())
    await asyncio.wait_for(rendered.wait(), 1)
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
    run = asyncio.create_task(screen.run_async())
    await asyncio.wait_for(rendered.wait(), 1)
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
    run = asyncio.create_task(screen.run_async())
    await asyncio.wait_for(rendered.wait(), 1)
    pipe.send_text('\x03')
    assert await asyncio.wait_for(run, 1) is None


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
  with defaults.create_pipe_input() as pipe:
    screen = workspace_application.WorkspaceApplication(
      full_screen=True,
      input=pipe,
      output=_Output(),
    )

    async def workspaces() -> None:
      if failure:
        raise ValueError('workspace failure')

    if failure:
      with pytest.raises(ValueError, match='workspace failure'):
        await screen.run_views(workspaces)
    else:
      await screen.run_views(workspaces)


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
