"""Real terminal transport for in-process CLI fixtures and native children."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import pathlib
import pty
import signal
import subprocess
import termios
from collections.abc import Callable, Generator
from typing import TextIO, override

from prompt_toolkit import data_structures
from prompt_toolkit.input import base, vt100
from prompt_toolkit.output import vt100 as vt100_output


class Input(vt100.Vt100Input, base.PipeInput):
  """Keep the fixture's send API while the application reads an actual PTY."""

  def __init__(
    self, stdin: TextIO, master: int, receive: Callable[[], None]
  ) -> None:
    super().__init__(stdin)
    self.master = master
    self.drain_output = receive

  @override
  def send_bytes(self, data: bytes) -> None:
    while data:
      data = data[os.write(self.master, data) :]

  @override
  def send_text(self, data: str) -> None:
    self.send_bytes(data.encode())

  def resize(self, size: data_structures.Size) -> None:
    if termios.tcgetwinsize(self.fileno()) == (size.rows, size.columns):
      return
    termios.tcsetwinsize(self.fileno(), (size.rows, size.columns))
    # These fixtures have no controlling-terminal process group. Notify only
    # native children owned by this test process, never the host terminal.
    processes = subprocess.check_output(
      ['ps', '-ax', '-o', 'pid=,ppid=,comm='], text=True
    )
    for line in processes.splitlines():
      pid, parent, executable = line.split(maxsplit=2)
      if int(parent) != os.getpid():
        continue
      # Linux truncates `comm`; /proc identifies the owned child's executable.
      process_executable = pathlib.Path(f'/proc/{pid}/exe')
      if process_executable.is_symlink():
        executable = os.readlink(process_executable)
      if pathlib.Path(executable).name == 'mirror-transcript':
        os.kill(int(pid), signal.SIGWINCH)


@contextlib.contextmanager
def input(
  output: vt100_output.Vt100_Output,
  rendered: Callable[[], None] = lambda: None,
) -> Generator[Input]:
  """Capture complete native frames through the fixture's existing VT sink."""
  master, slave = pty.openpty()
  termios.tcsetwinsize(
    slave, (output.get_size().rows, output.get_size().columns)
  )
  os.set_blocking(master, False)
  loop = asyncio.get_running_loop()
  decoder = codecs.getincrementaldecoder('utf-8')()
  pending = ''

  def receive() -> None:
    nonlocal pending
    while True:
      try:
        data = os.read(master, 65536)
      except BlockingIOError:
        return
      if not data:
        return
      pending += decoder.decode(data)
      while '\x1b[?2026l' in pending:
        frame, pending = pending.split('\x1b[?2026l', 1)
        output.write_raw(frame + '\x1b[?2026l')
        output.flush()
        rendered()

  try:
    with os.fdopen(slave, 'r', encoding='utf-8') as stdin:
      pipe = Input(stdin, master, receive)
      with pipe.raw_mode():
        loop.add_reader(master, receive)
        try:
          yield pipe
        finally:
          loop.remove_reader(master)
  finally:
    os.close(master)
