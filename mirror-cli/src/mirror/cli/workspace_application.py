"""Terminal handoff shared by workspace agents and creation views."""

from __future__ import annotations

import asyncio
import base64
import os
import signal
from collections.abc import Awaitable, Callable
from typing import override

from prompt_toolkit import keys
from prompt_toolkit.application import application
from prompt_toolkit.key_binding import key_bindings, key_processor

from mirror.cli import screen_selection, terminal


class WorkspaceApplication(application.Application[str | None]):
  """A prompt-toolkit application that can hand off without moving the frame."""

  _preserve_frame_on_exit = False
  _handoff_pending = False
  _between_views = False
  _selected_view: application.Application[str | None] | None = None
  _view_result: asyncio.Future[str | None] | None = None
  resize_handler: Callable[[], None] | None = None

  @override
  def _on_resize(self) -> None:
    view = self._selected_view
    if (
      isinstance(view, WorkspaceApplication) and view.resize_handler is not None
    ):
      view.resize_handler()
    if self.full_screen:
      # Renderer.render already discards its old screen when dimensions change.
      # Keep that repaint in the normal synchronized, frame-limited draw path.
      self.invalidate()
    else:
      super()._on_resize()

  async def run_views(
    self, run_workspaces: Callable[[], Awaitable[None]]
  ) -> None:
    """Own workspace supervision for exactly the terminal application's lifetime."""
    self.enable_views()

    async def supervise() -> None:
      try:
        await run_workspaces()
      except (
        Exception
      ) as error:  # Propagate the supervisor failure through run_async.
        if not self.is_done:
          self.exit(exception=error)
      else:
        if not self.is_done:
          self.exit()

    def start_supervisor() -> None:
      self.create_background_task(supervise())

    try:
      await self.run_async(pre_run=start_supervisor)
    finally:
      self.finish_views()

  def enable_views(self) -> None:
    """Keep one terminal application alive while selected workspace views change."""
    self._selected_view = None
    self._view_result = None
    self._between_views = False
    self._accepts_input = False
    self._handoff_keys: list[key_processor.KeyPress] = []
    self._screen_selection: screen_selection.ScreenSelection | None = None
    process_keys = self.key_processor.process_keys

    def process_selected_keys() -> None:
      if self._screen_selection is not None and any(
        press.key
        not in (keys.Keys.Vt100MouseEvent, keys.Keys.WindowsMouseEvent)
        for press in self.key_processor.input_queue
      ):
        self._screen_selection.clear()
      result = self._view_result
      if self.is_done or (
        self._accepts_input and result is not None and not result.done()
      ):
        process_keys()
      elif any(
        press.key in (keys.Keys.ControlC, keys.Keys.SIGINT)
        for press in self.key_processor.input_queue
      ):
        self.key_processor.empty_queue()
        if self.is_running and not self.is_done:
          self.exit()

    self.key_processor.process_keys = process_selected_keys
    self.before_render += self._before_view
    self.after_render += self._after_view

  def _before_view(self, _: application.Application[str | None]) -> None:
    if self._selected_view is not None:
      self._selected_view.before_render.fire()

  def _after_view(self, _: application.Application[str | None]) -> None:
    if self._selected_view is not None:
      self._selected_view.after_render.fire()

  async def show_view(
    self,
    view: application.Application[str | None],
    *,
    before_input: Callable[[], Awaitable[None]] | None = None,
    accept_buffered_input: Callable[[], bool] = lambda: True,
  ) -> str | None:
    """Select retained layout and editor state without replacing terminal ownership."""
    if self.full_screen:
      if not isinstance(
        view.layout.container, screen_selection.ScreenSelection
      ):
        view.layout.container = screen_selection.ScreenSelection(
          view.layout.container, self._copy_selection
        )
      self._screen_selection = view.layout.container
      self._screen_selection.clear()
    self._selected_view = view
    self._between_views = False
    self.layout = view.layout
    self.key_bindings = view.key_bindings
    self.style = view.style
    self.cursor = view.cursor
    self.clipboard = view.clipboard
    self.editing_mode = view.editing_mode
    self.vi_state = view.vi_state
    self.emacs_state = view.emacs_state
    self.ttimeoutlen = view.ttimeoutlen
    self.timeoutlen = view.timeoutlen
    result: asyncio.Future[str | None] = (
      asyncio.get_running_loop().create_future()
    )
    self._view_result = result
    self._accepts_input = False
    if (
      isinstance(view, WorkspaceApplication) and view.resize_handler is not None
    ):
      view.resize_handler()
    self.invalidate()
    try:
      if before_input is not None:
        await before_input()
      queued = self._handoff_keys + self.key_processor.empty_queue()
      self._handoff_keys = []
      self.key_processor.reset()
      if accept_buffered_input():
        self.key_processor.feed_multiple([
          press
          for press in queued
          if press.key
          not in (
            keys.Keys.Vt100MouseEvent,
            keys.Keys.WindowsMouseEvent,
          )
        ])
      self._accepts_input = True
      self.key_processor.process_keys()
      return await result
    finally:
      if self._screen_selection is not None:
        self._screen_selection.clear()
      self._accepts_input = False
      self._selected_view = None
      self._view_result = None
      self._between_views = True

  def _copy_selection(self, text: str) -> None:
    # OSC 52 requests a write to the attached terminal's clipboard, including SSH.
    encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
    self.output.write_raw(f'\x1b]52;c;{encoded}\x1b\\')
    self.output.flush()

  def finish_views(self) -> None:
    """Release a selected view when the owning application stops."""
    result = getattr(self, '_view_result', None)
    if result is not None and not result.done():
      result.set_result(None)

  def exit_for_handoff(self, result: str) -> None:
    """Exit to the supervisor without prompt-toolkit's final content render."""
    pending = getattr(self, '_view_result', None)
    if pending is not None:
      if not pending.done():
        self._accepts_input = False
        self._handoff_keys.extend(self.key_processor.empty_queue())
        pending.set_result(result)
    elif not self.is_done:
      self._preserve_frame_on_exit = True
      self.exit(result=result)

  def prepare_for_run(self) -> None:
    """Hold the replacement transcript until the initial prompt is drawn."""
    self._preserve_frame_on_exit = False
    self._handoff_pending = True
    self.output.write_raw(terminal.SYNC_BEGIN)
    self.output.flush()

  def finish_handoff(self) -> None:
    """Present the complete frame, also releasing it on a failed activation."""
    if self._handoff_pending:
      self._handoff_pending = False
      self.output.write_raw(terminal.SYNC_END)
      self.output.flush()

  @override
  def _start_auto_refresh_task(self) -> None:
    self.create_background_task(self._auto_refresh())

  async def _auto_refresh(self) -> None:
    """Use live refresh intervals instead of prompt-toolkit's startup snapshot."""
    while interval := self.refresh_interval:
      await asyncio.sleep(interval)
      if self.refresh_interval:
        self.invalidate()

  @override
  def _redraw(self, render_as_done: bool = False) -> None:
    # Background hooks can change the old layout before its replacement is ready.
    if self._between_views and not render_as_done:
      return
    # A non-full-screen final render moves the cursor one row past the layout,
    # scrolling a full-height frame. The next worktree replaces this frame.
    if render_as_done and self._preserve_frame_on_exit:
      return
    try:
      if self.full_screen:
        self.output.write_raw(terminal.SYNC_BEGIN)
      super()._redraw(render_as_done)
    finally:
      if self.full_screen:
        self.output.write_raw(terminal.SYNC_END)
        self.output.flush()
      self.finish_handoff()


def terminal_bindings() -> key_bindings.KeyBindings:
  """Suspend the foreground shell job without ending the application."""
  bindings = key_bindings.KeyBindings()

  @bindings.add('c-z', save_before=lambda _: False)
  def suspend(event: key_processor.KeyPressEvent) -> None:
    app = event.app
    try:
      supported = (
        hasattr(signal, 'SIGTSTP')
        and os.isatty(app.input.fileno())
        and os.isatty(app.output.fileno())
        and os.tcgetpgrp(app.input.fileno()) == os.getpgrp()
        # A non-job-control shell can share its own group with the child.
        and os.getpgrp() != os.getsid(0)
      )
    except (OSError, ValueError, NotImplementedError):
      supported = False
    if supported:
      # The framework serializes terminal writes, cooks input, and redraws on fg.
      app.suspend_to_background()
    else:
      app.output.bell()

  return bindings
