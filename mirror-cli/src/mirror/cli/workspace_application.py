"""Terminal handoff shared by workspace agents and creation views."""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import signal
import subprocess
from collections.abc import Awaitable, Callable
from typing import override

from prompt_toolkit import keys
from prompt_toolkit.application import application, current
from prompt_toolkit.key_binding import key_bindings, key_processor

from mirror.cli import screen_selection, terminal


_CLIPBOARD_COMMANDS: tuple[tuple[str, ...], ...] = (
  ('xclip', '-selection', 'clipboard'),
  ('xsel', '--clipboard', '--input'),
  ('wl-copy',),
  ('pbcopy',),
)


def _clipboard_candidates() -> list[tuple[str, ...]]:
  """Return clipboard helpers worth trying for this session.

  Availability alone is not enough: a helper can be installed but unusable in
  the current session (for example ``xclip`` with no ``DISPLAY``).  Order the
  helpers so the one matching the active display is tried first, while still
  allowing later helpers to be tried when an earlier one fails.
  """
  available: list[tuple[str, ...]] = []
  for command in _CLIPBOARD_COMMANDS:
    if shutil.which(command[0]) is None:
      continue
    # Skip helpers that cannot possibly work in this session, but keep them
    # as a last resort if every preferred helper is filtered out.
    if command[0] in ('xclip', 'xsel') and not os.environ.get('DISPLAY'):
      continue
    if command[0] == 'wl-copy' and not os.environ.get('WAYLAND_DISPLAY'):
      continue
    available.append(command)
  if not available:
    available = [
      command
      for command in _CLIPBOARD_COMMANDS
      if shutil.which(command[0]) is not None
    ]
  if os.environ.get('WAYLAND_DISPLAY'):
    available.sort(key=lambda command: command[0] != 'wl-copy')
  return available


def _system_clipboard_command() -> tuple[str, ...] | None:
  """Return the first clipboard helper worth trying, if any."""
  candidates = _clipboard_candidates()
  return candidates[0] if candidates else None


def _run_clipboard_candidates(
  candidates: list[tuple[str, ...]], data: bytes
) -> None:
  """Try clipboard helpers until one succeeds."""
  for command in candidates:
    try:
      result = subprocess.run(
        command,
        input=data,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=2,
      )
    except (OSError, subprocess.SubprocessError):
      continue
    if result.returncode == 0:
      return


def _running_event_loop() -> asyncio.AbstractEventLoop | None:
  """Return the running event loop, if any."""
  try:
    loop = asyncio.get_event_loop()
  except RuntimeError:
    loop = None
  if loop is not None and loop.is_running():
    return loop
  try:
    app = current.get_app()
  except LookupError:
    return None
  loop = getattr(app, 'loop', None)
  if loop is not None and getattr(loop, 'is_running', lambda: False)():
    return loop
  return None


def _copy_to_system_clipboard(text: str) -> None:
  """Best-effort copy through an OS clipboard helper.

  Terminals such as MATE Terminal (VTE) ignore the OSC 52 sequence above, so
  bridge the selection to the system clipboard when a helper is available.
  Helpers that are installed but unusable (for example ``xclip`` without a
  display) fail and the next helper is tried; all failures are ignored because
  OSC 52 remains the portable fallback.

  The helper runs off the event-loop thread when a loop is running, so a
  stalled helper cannot freeze input or rendering.
  """
  data = text.encode('utf-8')
  candidates = _clipboard_candidates()
  preferred = _system_clipboard_command()
  if preferred is not None and preferred in candidates:
    candidates = [preferred] + [
      command for command in candidates if command != preferred
    ]
  if not candidates:
    return
  loop = _running_event_loop()
  if loop is not None:
    loop.run_in_executor(None, _run_clipboard_candidates, candidates, data)
  else:
    _run_clipboard_candidates(candidates, data)


def copy_text(text: str) -> None:
  """Copy text to the terminal and system clipboards.

  The OSC 52 sequence is the portable terminal-clipboard protocol; the
  system-clipboard bridge covers terminals that ignore it (e.g. VTE/MATE).
  """
  data = text.encode('utf-8')
  if len(data) > 75_000:
    raise ValueError('Text exceeds the terminal clipboard limit of 75 KB.')
  encoded = base64.b64encode(data).decode('ascii')
  output = current.get_app().output
  output.write_raw(f'\x1b]52;c;{encoded}\x1b\\')
  output.flush()
  _copy_to_system_clipboard(text)


class WorkspaceApplication(application.Application[str | None]):
  """A prompt-toolkit application that can hand off without moving the frame."""

  _preserve_frame_on_exit = False
  _handoff_pending = False
  _between_views = False
  _selected_view: application.Application[str | None] | None = None
  _view_result: asyncio.Future[str | None] | None = None
  resize_handler: Callable[[], None] | None = None
  rendering: bool = False
  """Whether frame-local UI measurements can be shared by the active draw."""

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
    self._between_views = True
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
    try:
      copy_text(text)
    except ValueError:
      self.output.bell()

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
    if render_as_done and isinstance(
      self.layout.container, screen_selection.ScreenSelection
    ):
      self.layout.container.clear()
    # Startup and handoff have no selected layout, even for a final redraw.
    if self._between_views:
      return
    # A non-full-screen final render moves the cursor one row past the layout,
    # scrolling a full-height frame. The next worktree replaces this frame.
    if render_as_done and self._preserve_frame_on_exit:
      return
    try:
      if self.full_screen:
        self.output.write_raw(terminal.SYNC_BEGIN)
      self.rendering = True
      super()._redraw(render_as_done)
    finally:
      self.rendering = False
      if self.full_screen:
        self.output.write_raw(terminal.SYNC_END)
        self.output.flush()
      self.finish_handoff()


def terminal_bindings() -> key_bindings.KeyBindings:
  """Core terminal bindings shared by workspace views."""
  bindings = key_bindings.KeyBindings()

  @bindings.add('c-l', save_before=lambda _: False)
  def ignore_redraw(event: key_processor.KeyPressEvent) -> None:
    # Mask prompt_toolkit's clear-screen fallback, including in modal views.
    del event

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
