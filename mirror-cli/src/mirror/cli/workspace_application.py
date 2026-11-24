"""Terminal handoff shared by workspace agents and creation views."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import logging
import os
import re
import signal
import sys
import weakref
from collections.abc import Awaitable, Callable
from typing import override

from mirror_transcript import prompt_application, prompt_input, transcript_pb2
from prompt_toolkit import keys, mouse_events
from prompt_toolkit.application import application, current
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.key_binding.bindings import mouse as mouse_bindings
from prompt_toolkit.output import vt100
from rich import text

from mirror.cli import (
  actions,
  extended_keys,
  frame_pacing,
  rendering,
  row_renderer,
  screen_selection,
  scroll_burst,
  terminal,
  workspace_tags,
)
from mirror.cli.components import worktree_tabs


def copy_text(text: str) -> None:
  """Request a clipboard write, bridging direct local Apple Terminal sessions."""
  data = text.encode('utf-8')
  if len(data) > 75_000:
    raise ValueError('Text exceeds the terminal clipboard limit of 75 KB.')
  encoded = base64.b64encode(data).decode('ascii')
  app = current.get_app()
  output = app.output
  output.write_raw(f'\x1b]52;c;{encoded}\x1b\\')
  output.flush()
  if (
    isinstance(app, WorkspaceApplication)
    and app.is_running
    and output.stdout is not None
    and output.stdout.isatty()
    and terminal.needs_local_clipboard(sys.platform, os.environ)
  ):
    app.copy_to_system_clipboard(data)


_SGR_MOUSE = re.compile(r'\x1b\[<\d+;\d+;\d+[Mm]\Z')


def _is_sgr_mouse(press: key_processor.KeyPress) -> bool:
  """Whether a press is an SGR mouse report: wheel, motion, or button."""
  return (
    press.key == keys.Keys.Vt100MouseEvent
    and _SGR_MOUSE.match(press.data) is not None
  )


@functools.cache
def _mouse_binding() -> key_bindings.Binding:
  """prompt_toolkit's own VT100 mouse handler, which routes to mouse handlers."""
  return next(
    binding
    for binding in mouse_bindings.load_mouse_bindings().bindings
    if binding.keys == (keys.Keys.Vt100MouseEvent,)
  )


def _title_text(label: str, width: int = 80) -> str:
  label = ' '.join(rendering.strip_ansi(label).split())
  label = ''.join(character for character in label if character.isprintable())
  title = text.Text(label.strip())
  # Bound zero-width characters as well as cells.
  title.truncate(width, overflow='ellipsis')
  return title.plain[:width]


def invalidate_transcript() -> None:
  """Schedule transcript damage without rebuilding unrelated workspace chrome."""
  app = current.get_app()
  if isinstance(app, WorkspaceApplication):
    app.invalidate(transcript_only=True)
  else:
    app.invalidate()


class WorkspaceApplication(application.Application[str | None]):
  """A prompt-toolkit application that can hand off without moving the frame."""

  background_error_handler: Callable[[str], None] | None = None
  _background_error_reported = False
  _background_error_pending = False
  _render_failed = False

  refresh_key: Callable[[], object | None] | None = None
  """Inputs that can change without invalidation; None preserves periodic refresh."""
  _refresh_stamp: object | None = None
  _last_draw = 0.0
  _clipboard_task: asyncio.Task[None] | None = None
  _clipboard_pending: (
    tuple[Callable[[bytes], Awaitable[None]], bytes] | None
  ) = None

  _preserve_frame_on_exit = False
  _handoff_pending = False
  _between_views = False
  _selected_view: application.Application[str | None] | None = None
  _view_result: asyncio.Future[str | None] | None = None
  _scroll_burst: scroll_burst.ScrollBurst | None = None
  _frame_pacer: frame_pacing.FramePacer | None = None
  _terminal_title: str | None = None
  resize_handler: Callable[[], None] | None = None
  scroll_handler: (
    Callable[[mouse_events.MouseEvent], key_bindings.NotImplementedOrNone]
    | None
  ) = None
  native_event_handler: (
    Callable[[transcript_pb2.TranscriptEvent], None] | None
  ) = None
  rendering: bool = False
  """Whether frame-local UI measurements can be shared by the active draw."""

  def copy_to_system_clipboard(
    self,
    data: bytes,
    write: Callable[[bytes], Awaitable[None]] = terminal.write_local_clipboard,
  ) -> asyncio.Task[None]:
    """Serialize helper writes, retaining only the newest pending selection."""
    self._clipboard_pending = (write, data)
    if self._clipboard_task is None or self._clipboard_task.done():
      self._clipboard_task = self.create_background_task(
        self._flush_clipboard()
      )
    return self._clipboard_task

  async def _flush_clipboard(self) -> None:
    try:
      while self._clipboard_pending is not None:
        write, data = self._clipboard_pending
        self._clipboard_pending = None
        await write(data)
    finally:
      self._clipboard_pending = None

  ui_revision: object = object()
  """Identity of the latest invalidation affecting more than the transcript."""

  animation_revision: object = object()
  """Identity of the latest clock tick for time-dependent chrome."""

  def refresh_animations(self) -> None:
    """Advance timed widgets without invalidating static retained layers."""
    self.animation_revision = object()
    super().invalidate()

  @override
  def invalidate(self, *, transcript_only: bool = False) -> None:
    if not transcript_only:
      self.ui_revision = object()
    super().invalidate()

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

  @override
  async def run_async(
    self,
    pre_run: Callable[[], None] | None = None,
    set_exception_handler: bool = True,
    handle_sigint: bool = True,
    slow_callback_duration: float = 0.5,
  ) -> str | None:
    self._background_error_reported = False
    self._background_error_pending = False
    self._render_failed = False
    async with contextlib.AsyncExitStack() as stack:
      if self.full_screen and isinstance(self.output, vt100.Vt100_Output):
        await stack.enter_async_context(
          prompt_application.attach(self, on_event=self._native_event)
        )
      stack.enter_context(extended_keys.reporting(self.output, os.environ))
      try:
        return await super().run_async(
          pre_run, set_exception_handler, handle_sigint, slow_callback_duration
        )
      finally:
        self._clipboard_pending = None
        self.finish_scroll_burst()
        if self._terminal_title is not None:
          self.output.clear_title()
          self.output.flush()
          self._terminal_title = None

  def _native_event(self, event: transcript_pb2.TranscriptEvent) -> None:
    view = self._selected_view or self
    if (
      isinstance(view, WorkspaceApplication)
      and view.native_event_handler is not None
    ):
      view.native_event_handler(event)

  @override
  def _handle_exception(
    self, loop: asyncio.AbstractEventLoop, context: dict[str, object]
  ) -> None:
    del loop
    error = context.get('exception')
    error = (
      error
      if isinstance(error, BaseException)
      else RuntimeError('Background event loop failed; see the CLI log.')
    )
    # Never use prompt-toolkit's interactive exception handler: each failure
    # queues a terminal suspension and another blocking Enter prompt.
    logging.getLogger(__name__).error(
      'Background event loop failure', exc_info=error
    )
    self._background_error_pending = not self._background_error_reported
    self._show_background_error()

  def _show_background_error(self) -> None:
    if not self._background_error_pending:
      return
    view = self._selected_view or self
    handler = (
      view.background_error_handler
      if isinstance(view, WorkspaceApplication)
      else None
    )
    if handler is not None:
      self._background_error_pending = False
      self._background_error_reported = True
      handler('A background operation failed. See the CLI log for details.')

  def set_workspace_title(
    self,
    label: str,
    *,
    status: actions.WorktreeAgentState | None = None,
    tag: workspace_tags.WorkspaceTag | None = None,
  ) -> None:
    """Publish a bounded workspace label only when the terminal title changes."""
    if (
      isinstance(self.output, vt100.Vt100_Output) and self.output.term == 'dumb'
    ):
      return
    badge = worktree_tabs.status_indicator(status)[0] if status else ''
    # Names distinguish tags that share an icon and differ only by color.
    tag_label = _title_text(f'{tag.icon} {tag.name}', 24) if tag else ''
    label = ' · '.join(part for part in (tag_label, _title_text(label)) if part)
    value = _title_text(
      f'{badge} Mirror · {label}' if label else f'{badge} Mirror'
    )
    if value != self._terminal_title:
      self.output.set_title(value)
      self.output.flush()
      self._terminal_title = value

  def begin_scroll_burst(self) -> None:
    """Protect accepted wheel dispatch, including the first event's allocations."""
    if self._scroll_burst is not None and self._accepts_input:
      self._scroll_burst.touch()

  def finish_scroll_burst(self) -> None:
    """Restore scheduling before handing off terminal or application ownership."""
    if self._scroll_burst is not None:
      self._scroll_burst.close()
    if self._frame_pacer is not None:
      self._frame_pacer.reset()

  @override
  def suspend_to_background(self, suspend_group: bool = True) -> None:
    self.finish_scroll_burst()
    super().suspend_to_background(suspend_group)

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
    self.finish_scroll_burst()
    self._scroll_burst = scroll_burst.ScrollBurst(self)
    self._frame_pacer = frame_pacing.FramePacer(
      self._draw_deferred_frame, abandoned=self._forget_frame_queries
    )
    self._selected_view = None
    self._view_result = None
    self._between_views = True
    self._accepts_input = False
    self._handoff_keys: list[key_processor.KeyPress] = []
    self._screen_selection: screen_selection.ScreenSelection | None = None
    self._transcript_selection = ''
    process_keys = self.key_processor.process_keys

    def process_selected_keys() -> None:
      self._take_frame_acknowledgements()
      if isinstance(self.input, prompt_input.Input):
        self._transcript_selection = (
          self.input.take_selection() or self._transcript_selection
        )
      received = bool(self.key_processor.input_queue)
      pressed = any(
        press.key
        not in (keys.Keys.Vt100MouseEvent, keys.Keys.WindowsMouseEvent)
        for press in self.key_processor.input_queue
      )
      result = self._view_result
      if self.is_done or (
        self._accepts_input and result is not None and not result.done()
      ):
        self._dispatch_mouse_reports()
        if self.key_processor.input_queue:
          process_keys()
      elif any(
        press.key in (keys.Keys.ControlC, keys.Keys.SIGINT)
        for press in self.key_processor.input_queue
      ):
        self.key_processor.empty_queue()
        if self.is_running and not self.is_done:
          self.exit()
      # Handlers such as quote-selection read the selection a key dismisses,
      # so clear it afterwards, and not while a chord such as Alt+L is open.
      if received and not self.key_processor.key_buffer:
        self._transcript_selection = ''
        if pressed and self._screen_selection is not None:
          self._screen_selection.clear()

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
    self.finish_scroll_burst()
    if self.full_screen:
      if not isinstance(
        view.layout.container, screen_selection.ScreenSelection
      ):
        view.layout.container = screen_selection.ScreenSelection(
          view.layout.container, self._copy_selection
        )
      self._screen_selection = view.layout.container
      self._screen_selection.scroll_handler = (
        self._scroll_view
        if isinstance(view, WorkspaceApplication)
        and view.scroll_handler is not None
        else None
      )
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
    self._show_background_error()
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
      self.finish_scroll_burst()
      if self._screen_selection is not None:
        self._screen_selection.clear()
      self._accepts_input = False
      self._selected_view = None
      self._view_result = None
      self._between_views = True

  def _scroll_view(
    self, event: mouse_events.MouseEvent
  ) -> key_bindings.NotImplementedOrNone:
    """Route wheel input to the active workspace view."""
    view = self._selected_view
    if (
      not isinstance(view, WorkspaceApplication) or view.scroll_handler is None
    ):
      return NotImplemented
    return view.scroll_handler(event)

  def selected_text(self) -> str:
    """Return the text selected on screen, or an empty string."""
    if self._screen_selection is not None and (
      text := self._screen_selection.text()
    ):
      return text
    return self._transcript_selection

  def _copy_selection(self, text: str) -> None:
    try:
      copy_text(text)
    except ValueError:
      self.output.bell()

  def finish_views(self) -> None:
    """Release a selected view when the owning application stops."""
    self.finish_scroll_burst()
    result = getattr(self, '_view_result', None)
    if result is not None and not result.done():
      result.set_result(None)

  def exit_for_handoff(self, result: str) -> None:
    """Exit to the supervisor without prompt-toolkit's final content render."""
    self.finish_scroll_burst()
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
    # The held handoff frame must not wait behind an earlier acknowledgement.
    if self._frame_pacer is not None:
      self._frame_pacer.reset()
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
    """Poll legacy inputs without repainting unchanged or recent frames."""
    loop = asyncio.get_running_loop()
    delay = self.refresh_interval
    while delay:
      await asyncio.sleep(delay)
      if not self.refresh_interval:
        break
      # Retry when due instead of skipping an entire animation interval.
      delay = self.refresh_interval - (loop.time() - self._last_draw)
      if delay > 0:
        continue
      stamp = self._refresh_token()
      if stamp is None:
        self.refresh_animations()
      elif stamp != self._refresh_stamp:
        self.invalidate()
      delay = self.refresh_interval

  def _refresh_token(self) -> object | None:
    view = self._selected_view or self
    if isinstance(view, WorkspaceApplication) and view.refresh_key is not None:
      key = view.refresh_key()
      if key is not None:
        return view, key
    return None

  @override
  def _redraw(self, render_as_done: bool = False) -> None:
    if self._render_failed:
      return
    try:
      self._draw_frame(render_as_done)
    except Exception as error:  # Unwind a broken renderer through run_async.
      self._render_failed = True
      logging.getLogger(__name__).exception('Terminal rendering failed')
      if not self.is_running or self.is_done:
        raise
      self.exit(exception=error)

  def _draw_frame(self, render_as_done: bool) -> None:
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
    row_renderer.install(self)
    # The final frame hands the terminal back; a reply would reach the shell.
    pacer = None if render_as_done else self._paced_frames()
    if pacer is not None and pacer.defer():
      return
    drawn = self.render_counter
    try:
      if self.full_screen:
        self.output.write_raw(terminal.SYNC_BEGIN)
      self.rendering = True
      super()._redraw(render_as_done)
      if self.loop is not None:
        self._last_draw = self.loop.time()
      self._refresh_stamp = self._refresh_token()
    finally:
      self.rendering = False
      if self.full_screen:
        self.output.write_raw(terminal.SYNC_END)
      self.finish_handoff()
      if (
        pacer is not None
        and self.loop is not None
        and self.render_counter != drawn
        and isinstance(self.renderer, row_renderer.Renderer)
      ):
        self.output.write_raw(frame_pacing.FRAME_QUERY)
        self.renderer.expect_cursor_report()
        pacer.sent(self.loop)
      if self.full_screen:
        self.output.flush()

  def _dispatch_mouse_reports(self) -> None:
    """Route leading SGR mouse reports straight to the mouse binding.

    Matching a press against key bindings walks the whole layout twice, and a
    wheel or trackpad burst delivers dozens of reports per read. Mouse reports
    only ever reach the mouse binding, so skip the match.
    """
    processor, queue = self.key_processor, self.key_processor.input_queue
    # A buffered prefix or repeat count belongs to the following presses.
    if processor.key_buffer or processor.arg is not None:
      return
    binding = _mouse_binding()
    while queue and _is_sgr_mouse(queue[0]):
      press = queue.popleft()
      event = key_processor.KeyPressEvent(
        weakref.ref(processor), None, [press], [], False
      )
      # Mirror `Binding.call`, which repaints unless the handler declines.
      if binding.handler(event) is not NotImplemented:
        self.invalidate()

  def _take_frame_acknowledgements(self) -> None:
    """Consume replies to frame queries before key bindings can see them.

    A handled key invalidates the application, so a reply processed as a key
    would draw another frame, query again, and repaint forever.
    """
    pacer, queue = self._frame_pacer, self.key_processor.input_queue
    # Full-screen rendering never queries the cursor itself; replies are ours.
    if pacer is None or not self.full_screen:
      return
    replies = [press for press in queue if press.key == keys.Keys.CPRResponse]
    if not replies:
      return
    remaining = [press for press in queue if press.key != keys.Keys.CPRResponse]
    queue.clear()
    queue.extend(remaining)
    for press in replies:
      # The renderer tracks replies that exit and suspension wait for.
      row, _ = map(int, press.data[2:-1].split(';'))
      self.renderer.report_absolute_cursor_row(row)
      pacer.acknowledge()

  def _draw_deferred_frame(self) -> None:
    # Bypass `invalidate`'s chrome revision: deferred frames already hold it.
    super().invalidate()

  def _forget_frame_queries(self) -> None:
    if isinstance(self.renderer, row_renderer.Renderer):
      self.renderer.forget_cursor_reports()

  def _paced_frames(self) -> frame_pacing.FramePacer | None:
    """Return the pacer when the terminal can acknowledge full-screen frames."""
    pacer = self._frame_pacer
    if (
      pacer is None
      or not pacer.enabled
      or not self.full_screen
      or not isinstance(self.renderer, row_renderer.Renderer)
      or not self.output.responds_to_cpr
    ):
      return None
    return pacer


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
