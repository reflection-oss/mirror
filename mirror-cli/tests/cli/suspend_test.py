"""Exercise shell job control in a private controlling terminal, never the host TTY."""

import os
import pathlib
import pty
import select
import shlex
import signal
import subprocess
import sys
import termios
import time

import pyte
import pytest
from prompt_toolkit import application, buffer, layout
from prompt_toolkit.input import base as input_base
from prompt_toolkit.input import vt100_parser
from prompt_toolkit.output import base as output_base

from mirror.cli import actions, keymap, state, workspace_application


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.asyncio
async def test_suspend_preserves_draft_and_approvals_without_tty(
  enabled: bool,
) -> None:
  current = state.State(
    activity=state.ActivityState(busy=True),
    status_bar=state.StatusBarState('test', 'medium', 'approve'),
    session=state.SessionState('test'),
    lifecycle=state.LifecycleState(active=True),
    approval=state.ApprovalState((
      actions.ApprovalRequest('first', 'bash', '{}'),
      actions.ApprovalRequest('second', 'bash', '{}'),
    )),
  )
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [
      keymap.defaults(
        current,
        completion_visible=False,
        modal_active=False,
        can_focus_tasks=False,
      )
    ],
    dispatched.append,
    dismiss=lambda: None,
    enabled=enabled,
  )
  composer = buffer.Buffer()
  composer.text = 'draft'
  app = workspace_application.WorkspaceApplication(
    layout=layout.Layout(layout.Window(layout.BufferControl(composer))),
    key_bindings=mapping.bindings,
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  with application.set_app(app):
    parser = vt100_parser.Vt100Parser(app.key_processor.feed)
    parser.feed('\x1a')
    app.key_processor.process_keys()
    assert composer.text == 'draft'
    assert dispatched == []
    await app.cancel_and_wait_for_background_tasks()


# Exec after acquiring the private controlling terminal; no preexec_fn in pytest.
_SHELL = """
import fcntl, os, termios
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
os.execv('/bin/bash', ['bash', '--noprofile', '--norc', '-i'])
"""

_APP = """
import pathlib, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
worker = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True
)
pathlib.Path('children').write_text(f'{child.pid} {worker.pid}')
from prompt_toolkit import buffer, layout
from mirror.cli import keymap, workspace_application
composer = buffer.Buffer()
keys = keymap.Keymap(
    lambda: [], lambda action: None, dismiss=lambda: None, enabled=False
).refresh()
@keys.add('c-d')
def done(event):
    event.app.exit(result=composer.text)
app = workspace_application.WorkspaceApplication(
    layout=layout.Layout(layout.Window(layout.BufferControl(composer))),
    key_bindings=keys,
)
# Mirror enters raw mode before run_async, too.
with app.input.raw_mode():
    app.prepare_for_run()
    try:
        print('RESULT:' + str(app.run()))
    finally:
        child.terminate()
        worker.terminate()
        child.wait()
        worker.wait()
        pathlib.Path('children').unlink()
"""


def _read_until(master: int, expected: bytes) -> bytes:
  output = b''
  deadline = time.monotonic() + 15
  while expected not in output:
    remaining = deadline - time.monotonic()
    assert remaining > 0, (expected, output)
    ready, _, _ = select.select([master], [], [], remaining)
    assert ready, (expected, output)
    output += os.read(master, 65536)
  return output


def test_suspend_restores_terminal_and_fg_preserves_prompt(
  tmp_path: pathlib.Path,
) -> None:
  master, slave = pty.openpty()
  shell = subprocess.Popen(
    [sys.executable, '-c', _SHELL],
    stdin=slave,
    stdout=slave,
    stderr=slave,
    start_new_session=True,
    env=os.environ | {'PS1': 'SHELL> ', 'PROMPT_TOOLKIT_NO_CPR': '1'},
    cwd=tmp_path,
  )
  job_pid = None
  script = tmp_path / 'app.py'
  script.write_text(_APP)
  try:
    _read_until(master, b'SHELL> ')
    original = termios.tcgetattr(slave)
    # Record the job leader so failure cleanup never targets an unrelated group.
    command = 'echo $$ > job.pid; exec ' + shlex.join([
      sys.executable,
      str(script),
    ])
    os.write(master, (shlex.join(['/bin/bash', '-c', command]) + '\n').encode())
    initial = _read_until(master, b'\x1b[?2004h')
    job_pid = int((tmp_path / 'job.pid').read_text())
    os.write(master, b'draft\x1a')
    stopped = _read_until(master, b'SHELL> ')
    assert b'Stopped' in stopped
    screen = pyte.Screen(80, 24)
    pyte.ByteStream(screen).feed(initial + stopped)
    assert not screen.cursor.hidden
    assert b'\x1b[?2004l' in stopped
    assert termios.tcgetattr(slave) == original
    child, worker = (tmp_path / 'children').read_text().split()
    assert 'T' in subprocess.check_output(
      ['ps', '-o', 'stat=', '-p', child], text=True
    )
    assert 'T' not in subprocess.check_output(
      ['ps', '-o', 'stat=', '-p', worker], text=True
    )
    os.write(master, b'fg\n')
    resumed = _read_until(master, b'\x1b[?2004h')
    assert b'fg' in resumed
    os.write(master, b'\x1a')
    _read_until(master, b'SHELL> ')
    assert termios.tcgetattr(slave) == original
    os.write(master, b'fg\n')
    _read_until(master, b'\x1b[?2004h')
    os.write(master, b'\x04')
    finished = _read_until(master, b'SHELL> ')
    assert b'RESULT:draft' in finished
    job_pid = None
  finally:
    if job_pid is not None:
      try:
        os.killpg(job_pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    if (tmp_path / 'children').exists():
      _, worker = (tmp_path / 'children').read_text().split()
      try:
        os.kill(int(worker), signal.SIGKILL)
      except ProcessLookupError:
        pass
    shell.kill()
    shell.wait(timeout=5)
    os.close(master)
    os.close(slave)
