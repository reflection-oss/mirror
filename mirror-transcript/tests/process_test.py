import asyncio
import codecs
import contextlib
import os
import pathlib
import pty
import shlex
import signal
import sys
import termios
import tty
from collections.abc import AsyncGenerator, Callable

import pyte
import pytest


class Terminal:
  def __init__(self, descriptor: int) -> None:
    self.descriptor = descriptor
    self.screen = pyte.Screen(20, 6)
    self.stream = pyte.Stream(self.screen)
    self.decoder = codecs.getincrementaldecoder('utf-8')()

  async def until(self, condition: Callable[[], bool]) -> None:
    loop = asyncio.get_running_loop()
    ready: asyncio.Future[None] = loop.create_future()

    def read() -> None:
      try:
        data = os.read(self.descriptor, 65536)
        if not data:
          raise EOFError('Terminal closed before the requested frame')
        self.stream.feed(self.decoder.decode(data))
        if condition() and not ready.done():
          ready.set_result(None)
      except (OSError, EOFError) as error:
        if not ready.done():
          ready.set_exception(error)

    loop.add_reader(self.descriptor, read)
    try:
      if not condition():
        await asyncio.wait_for(ready, 10)
    finally:
      loop.remove_reader(self.descriptor)


@contextlib.asynccontextmanager
async def terminal_host(
  tmp_path: pathlib.Path,
  script: str,
  *arguments: str,
  binary: pathlib.Path | None = None,
) -> AsyncGenerator[tuple[Terminal, asyncio.subprocess.Process]]:
  master, slave = pty.openpty()
  termios.tcsetwinsize(slave, (6, 20))
  if script == 'transcript_host.py':
    tty.setraw(slave)
  environment = {
    'HOME': str(tmp_path),
    'TERM': 'xterm-256color',
    'PYTHONNOUSERSITE': '1',
    'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
  }
  host = await asyncio.create_subprocess_exec(
    sys.executable,
    str(pathlib.Path(__file__).with_name(script)),
    os.ttyname(slave),
    str(
      binary or pathlib.Path(os.environ['MIRROR_TRANSCRIPT_BINARY']).resolve()
    ),
    *arguments,
    env=environment,
    cwd=tmp_path,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  try:
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'ready\n'
    yield Terminal(master), host
  finally:
    if host.returncode is None:
      os.kill(host.pid, signal.SIGCONT)
      if script == 'transcript_host.py':
        assert host.stdin is not None
        host.stdin.write(b'q')
        await host.stdin.drain()
      else:
        os.write(master, b'\x11')
    try:
      _, errors = await asyncio.wait_for(host.communicate(), 10)
    finally:
      if host.returncode is None:
        host.kill()
        await host.wait()
      os.close(master)
      os.close(slave)
    assert host.returncode == 0, errors.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['', 'synchronized'])
async def test_prompt_toolkit_stall_and_acknowledged_foreground_handoff(
  tmp_path: pathlib.Path,
  mode: str,
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py', mode) as (
    terminal,
    host,
  ):
    await terminal.until(lambda: terminal.screen.display[1].strip() == '37')
    os.kill(host.pid, signal.SIGSTOP)
    _, status = await asyncio.to_thread(os.waitpid, host.pid, os.WUNTRACED)
    assert os.WIFSTOPPED(status)
    os.write(terminal.descriptor, b'\x1b[<64;4;3M')
    await terminal.until(lambda: terminal.screen.display[1].strip() == '36')
    assert terminal.screen.display[0].strip() == 'header'
    assert terminal.screen.display[5].strip() == 'status'
    assert (terminal.screen.cursor.x, terminal.screen.cursor.y) == (0, 4)
    os.kill(host.pid, signal.SIGCONT)
    os.write(terminal.descriptor, 'draft é'.encode())
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft é')
    )
    os.write(terminal.descriptor, b'\x1bOQ')
    await terminal.until(
      lambda: any('FOREGROUND_READY' in row for row in terminal.screen.display)
    )
    os.write(terminal.descriptor, b'handoff\n')
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'returned\n'
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft é')
    )
    assert terminal.screen.display[1].strip() == '36'
    os.write(terminal.descriptor, b'!')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft é!')
    )
    os.write(terminal.descriptor, b'\x11')
    assert await asyncio.wait_for(host.wait(), 10) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('key', 'error'), [(b'\x1bOR', 'RuntimeError'), (b'\x1bOS', 'ValueError')]
)
async def test_key_handler_exception_keeps_input_and_acknowledgements_alive(
  tmp_path: pathlib.Path, key: bytes, error: str
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py') as (terminal, host):
    os.write(terminal.descriptor, key)
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == (
      f'error {error}: key handler failed; active=True\n'.encode()
    )
    os.write(terminal.descriptor, b'still typing')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('still typing')
    )
    os.write(terminal.descriptor, b'\x11')
    assert await asyncio.wait_for(host.wait(), 10) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('error', ['RuntimeError', 'ValueError'])
async def test_event_callback_exception_keeps_input_and_acknowledgements_alive(
  tmp_path: pathlib.Path, error: str
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py', error) as (
    terminal,
    host,
  ):
    await terminal.until(lambda: terminal.screen.display[1].strip() == '37')
    os.write(terminal.descriptor, b'\x1b[<64;4;3M')
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == (
      f'error {error}: event handler failed; active=True\n'.encode()
    )
    os.write(terminal.descriptor, b'still typing')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('still typing')
    )
    os.write(terminal.descriptor, b'\x11')
    assert await asyncio.wait_for(host.wait(), 10) == 0


@pytest.mark.asyncio
async def test_replacement_window_reconnects_without_dropping_its_active_view(
  tmp_path: pathlib.Path,
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py') as (terminal, host):
    os.write(terminal.descriptor, b'\x1b[15~')
    assert host.stdout is not None
    assert (
      await asyncio.wait_for(host.stdout.readline(), 10) == b'reconnected\n'
    )
    await terminal.until(lambda: terminal.screen.display[1].strip() == 'new 37')
    os.write(terminal.descriptor, b'replacement')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('replacement')
    )
    os.write(terminal.descriptor, b'\x11')
    assert await asyncio.wait_for(host.wait(), 10) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('key', [b'\x1b[17~', b'\x1b[18~', b'\x1b[19~'])
async def test_renderer_failure_republishes_sources_and_preserves_draft(
  tmp_path: pathlib.Path, key: bytes
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py') as (terminal, host):
    os.write(terminal.descriptor, b'draft')
    await terminal.until(lambda: terminal.screen.display[4].startswith('draft'))
    os.write(terminal.descriptor, key)
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
    os.write(terminal.descriptor, b'!')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft!')
    )
    assert terminal.screen.display[1].strip() == '37'
    os.write(terminal.descriptor, b'\x1b[<64;4;3M')
    await terminal.until(lambda: terminal.screen.display[1].strip() == '36')
    os.write(terminal.descriptor, key)
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
    os.write(terminal.descriptor, b'?')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft!?')
    )
    os.write(terminal.descriptor, key)
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
    os.write(terminal.descriptor, b'.')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft!?.')
    )
    os.write(terminal.descriptor, key)
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
    os.write(terminal.descriptor, b'!')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft!?.!')
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('mode', 'reason'),
  [
    ('oversized', b'Transcript command exceeds 8 MiB'),
    ('rejected', b'transcript order contains duplicate or zero IDs'),
  ],
)
async def test_renderer_reports_persistent_publication_failure(
  tmp_path: pathlib.Path, mode: str, reason: bytes
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py', mode) as (
    terminal,
    host,
  ):
    loop = asyncio.get_running_loop()
    # Drain terminal output while waiting on the host's separate error channel.
    loop.add_reader(terminal.descriptor, os.read, terminal.descriptor, 65536)
    try:
      os.write(terminal.descriptor, b'\x1b[21~')
      assert host.stdout is not None
      assert (
        await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
      )
      assert (
        await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
      )
      assert (
        await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
      )
      failure = await asyncio.wait_for(host.stdout.readline(), 10)
      assert failure.startswith(b'failed: ')
      assert reason in failure
      assert await asyncio.wait_for(host.wait(), 10) == 0
    finally:
      loop.remove_reader(terminal.descriptor)


@pytest.mark.asyncio
async def test_renderer_sigkill_preserves_python_application(
  tmp_path: pathlib.Path,
) -> None:
  pid_file = tmp_path / 'renderer.pid'
  binary = tmp_path / 'renderer'
  native = pathlib.Path(os.environ['MIRROR_TRANSCRIPT_BINARY']).resolve()
  binary.write_text(
    f'#!/bin/sh\necho $$ > {shlex.quote(str(pid_file))}\n'
    f'exec {shlex.quote(str(native))} "$@"\n'
  )
  binary.chmod(0o700)
  async with terminal_host(tmp_path, 'prompt_host.py', binary=binary) as (
    terminal,
    host,
  ):
    os.write(terminal.descriptor, b'draft')
    await terminal.until(lambda: terminal.screen.display[4].startswith('draft'))
    os.kill(int(pid_file.read_text()), signal.SIGKILL)
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
    os.write(terminal.descriptor, b'!')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('draft!')
    )
    assert terminal.screen.display[1].strip() == '37'


@pytest.mark.asyncio
async def test_renderer_exit_during_handoff_does_not_steal_foreground_input(
  tmp_path: pathlib.Path,
) -> None:
  async with terminal_host(tmp_path, 'prompt_host.py') as (terminal, host):
    os.write(terminal.descriptor, b'\x1b[20~')
    await terminal.until(
      lambda: any('FOREGROUND_READY' in row for row in terminal.screen.display)
    )
    os.write(terminal.descriptor, b'handoff\n')
    assert host.stdout is not None
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'returned\n'
    assert await asyncio.wait_for(host.stdout.readline(), 10) == b'recovered\n'
    os.write(terminal.descriptor, b'after handoff')
    await terminal.until(
      lambda: terminal.screen.display[4].startswith('after handoff')
    )


@pytest.mark.asyncio
async def test_scrolling_while_python_is_stopped_preserves_chrome_and_cursor(
  tmp_path: pathlib.Path,
) -> None:
  async with terminal_host(tmp_path, 'transcript_host.py') as (terminal, host):
    os.kill(host.pid, signal.SIGSTOP)
    _, status = await asyncio.to_thread(os.waitpid, host.pid, os.WUNTRACED)
    assert os.WIFSTOPPED(status)
    await terminal.until(lambda: terminal.screen.display[1].startswith('seven'))
    os.write(terminal.descriptor, b'\x1b[<64;4;3M')
    await terminal.until(lambda: terminal.screen.display[1].startswith('six'))
    assert terminal.screen.display[0].rstrip() == 'header'
    assert terminal.screen.display[4].rstrip() == 'draft>'
    assert terminal.screen.display[5].rstrip() == 'status'
    assert (terminal.screen.cursor.x, terminal.screen.cursor.y) == (7, 4)
    assert terminal.screen.buffer[4][0].fg == 'blue'
    assert terminal.screen.buffer[1][0].fg == 'green'
    os.write(terminal.descriptor, b'\x1b[<65;4;3M')
    await terminal.until(lambda: terminal.screen.display[1].startswith('seven'))
