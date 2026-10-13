"""Buffered terminal output, synchronized repaints, and scrollback writes."""

from __future__ import annotations

import asyncio
import enum
import os
import pathlib
import shlex
import sys
import webbrowser
from collections.abc import Callable, Coroutine
from typing import Any

# prompt_toolkit shadows its `run_in_terminal` submodule with the function of
# the same name, so only the function can be imported, not the module.
from prompt_toolkit.application import run_in_terminal
from rich import console as _console

from mirror.cli import rendering as _rendering

# Synchronized output: terminals that know these hold back the frames in
# between and present the whole write at once. Others ignore them.
SYNC_BEGIN = '\x1b[?2026h'
SYNC_END = '\x1b[?2026l'


def editor_command(path: pathlib.Path) -> tuple[str, ...]:
  """Resolve the user's editor and pass the file as a literal argument."""
  argv = shlex.split(
    os.environ.get('VISUAL') or os.environ.get('EDITOR') or 'vi'
  )
  if not argv:
    raise ValueError('editor command is empty')
  return (*argv, str(path))


async def open_url(url: str) -> None:
  """Open a browser without letting it interfere with terminal rendering."""
  if sys.platform == 'darwin':
    pid = os.posix_spawn('/usr/bin/open', ('open', url), os.environ)
    await asyncio.to_thread(os.waitpid, pid, 0)
  else:
    await run_in_terminal(lambda: webbrowser.open(url))


class TerminalClear(enum.Enum):
  """How a paint treats what is already on the terminal.

  `KEEP` prints under existing content (live turns). `SCREEN` is the usual
  Ctrl+L sense: erase the visible display and leave saved scrollback alone.
  Startup, session switches, and worktree tab changes use this too.
  `SCROLLBACK` does the same visible erase, then CSI 3J to drop saved lines.
  `/clear` is the path that asks for it. `--flush-scrollback` promotes every
  `SCREEN` paint to `SCROLLBACK`.
  """

  KEEP = 'keep'
  SCREEN = 'screen'
  SCROLLBACK = 'scrollback'


class TerminalOutput:
  """Buffered emit, synchronized repaints, and scrollback management."""

  def __init__(
    self,
    console: _console.Console,
    *,
    is_replaying: Callable[[], bool],
    is_active: Callable[[], bool] = lambda: True,
    create_task: Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]],
    flush_scrollback: bool = False,
    request_draw: Callable[[], None] | None = None,
    theme_name: Callable[[], str] = lambda: 'dark',
  ) -> None:
    self.console = console
    self._theme_name = theme_name
    self._is_replaying = is_replaying
    self._is_active = is_active
    self.flush_scrollback = flush_scrollback
    self._request_draw = request_draw
    self.create_task: Callable[
      [Coroutine[Any, Any, None]], asyncio.Task[None]
    ] = create_task
    self.emit_buffer: list[_console.RenderableType] = []
    self._flushing = False
    self._sync_depth = 0
    self.header: list[_console.RenderableType] = []
    self.trailer: list[_console.RenderableType] = []

  def pin_header(self, *rows: _console.RenderableType) -> None:
    """Pin the rows a repaint keeps above the history, however cropped."""
    self.header = list(rows)

  async def emit(self, renderables: list[_console.RenderableType]) -> None:
    """Draw retained content on an owned screen, or print into inline scrollback."""
    if not renderables:
      return
    if self._request_draw is not None:
      self._request_draw()
      return
    if not self._is_active() or self._is_replaying():
      self.emit_buffer.extend(renderables)
      return
    width = max(1, self.console.width)
    ansi = (
      _rendering.render_blocks(
        renderables, width, theme_name=self._theme_name()
      )
      + '\n\n'
    )
    self._begin_sync()
    try:
      await run_in_terminal(lambda: self.print_ansi(ansi))
    finally:
      self._end_sync()

  async def emit_after(
    self,
    prior: asyncio.Task[None],
    renderables: list[_console.RenderableType],
  ) -> None:
    """Emit renderables after an earlier scrollback write completes."""
    await prior
    await self.emit(renderables)

  def queue_emit(self, renderables: list[_console.RenderableType]) -> None:
    """Buffer renderables so event bursts print in one terminal suspension."""
    if self._request_draw is not None:
      self._request_draw()
      return
    self.emit_buffer.extend(renderables)
    if not self._is_replaying() and self._is_active():
      self.schedule_emit()

  def schedule_emit(self) -> None:
    """Start one task that drains all currently buffered terminal output."""
    if self.emit_buffer and not self._flushing and self._is_active():
      self._flushing = True
      task = self.create_task(self._flush_emits())
      # Hold redraws even before the task starts: live rows may already be gone.
      self._begin_sync()
      task.add_done_callback(lambda _: self._end_sync())

  async def _flush_emits(self) -> None:
    """Drain the emit buffer, batching whatever accumulates per pass."""
    try:
      while self.emit_buffer and not self._is_replaying() and self._is_active():
        batch = self.emit_buffer[:]
        self.emit_buffer.clear()
        await self.emit(batch)
    finally:
      self._flushing = False

  def _begin_sync(self) -> None:
    # DECSET 2026 is not nestable; overlapping emits must share one frame.
    if self._sync_depth == 0 and self.console.is_terminal:
      self.console.file.write(SYNC_BEGIN)
      self.console.file.flush()
    self._sync_depth += 1

  def _end_sync(self) -> None:
    self._sync_depth -= 1
    if self._sync_depth == 0 and self.console.is_terminal:
      self.console.file.write(SYNC_END)
      self.console.file.flush()

  def print_ansi(
    self, ansi: str, *, clear: TerminalClear = TerminalClear.KEEP
  ) -> None:
    """Write pre-rendered ANSI, clearing first when a repaint asks for it.

    Live turns (`KEEP`) use synchronized output so a burst presents as one
    frame. Replacing paints flush CSI 2J (and CSI 3J when dropping saved
    lines) before the history. 3J follows 2J because xterm.js copies the
    viewport into scrollback on 2J. Replacement paints do not manage 2026
    here: a workspace handoff holds it through the initial prompt render.
    """
    ansi = _rendering.sanitize_ansi(ansi)
    if self._request_draw is not None:
      self._request_draw()
      return
    clear = self.effective_clear(clear)
    out = self.console.file
    frame = ''
    if clear is not TerminalClear.KEEP:
      frame = '\x1b[2J\x1b[H'
      if clear is TerminalClear.SCROLLBACK:
        frame += '\x1b[3J'
    synchronized = (
      self.console.is_terminal
      and clear is TerminalClear.KEEP
      and self._sync_depth == 0
    )
    if self.console.record:
      # Diagnostic/test consoles need Rich to retain an exportable transcript.
      out.write(frame)
      if synchronized:
        out.write(SYNC_BEGIN)
      try:
        self.console.print(
          ansi,
          soft_wrap=True,
          markup=False,
          highlight=False,
          emoji=False,
          end='',
        )
      finally:
        if synchronized:
          out.write(SYNC_END)
        out.flush()
      return
    if synchronized:
      frame += SYNC_BEGIN
    frame += ansi
    if synchronized:
      frame += SYNC_END
    out.write(frame)
    out.flush()

  def effective_clear(self, clear: TerminalClear) -> TerminalClear:
    """`SCREEN` becomes `SCROLLBACK` when `--flush-scrollback` is on."""
    if self.flush_scrollback and clear is TerminalClear.SCREEN:
      return TerminalClear.SCROLLBACK
    return clear
