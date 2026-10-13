import asyncio
import io
import pathlib
from collections.abc import Coroutine
from typing import Any

import pytest
from rich import console

from mirror.cli import terminal


@pytest.mark.asyncio
@pytest.mark.parametrize('record', [False, True])
async def test_emit_overlapping_writes_share_one_synchronized_frame(
  record: bool,
) -> None:
  buffer = io.StringIO()
  output = terminal.TerminalOutput(
    console.Console(file=buffer, force_terminal=True, record=record),
    is_replaying=lambda: False,
    create_task=asyncio.create_task,
  )

  await asyncio.gather(output.emit(['first']), output.emit(['second']))

  frame = buffer.getvalue()
  assert frame.startswith(terminal.SYNC_BEGIN)
  assert frame.endswith(terminal.SYNC_END)
  assert frame.count(terminal.SYNC_BEGIN) == 1
  assert frame.count(terminal.SYNC_END) == 1
  assert 'first' in frame
  assert 'second' in frame


@pytest.mark.asyncio
async def test_queue_emit_releases_frame_when_cancelled_before_start() -> None:
  tasks: list[asyncio.Task[None]] = []

  def create_task(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(coro)
    tasks.append(task)
    return task

  buffer = io.StringIO()
  output = terminal.TerminalOutput(
    console.Console(file=buffer, force_terminal=True),
    is_replaying=lambda: False,
    create_task=create_task,
  )

  output.queue_emit(['unprinted'])
  (task,) = tasks
  task.cancel()
  with pytest.raises(asyncio.CancelledError):
    await task

  assert buffer.getvalue() == terminal.SYNC_BEGIN + terminal.SYNC_END


@pytest.mark.parametrize(
  ('visual', 'editor', 'expected'),
  [
    (
      '"editor with spaces" --wait',
      'ignored',
      ('editor with spaces', '--wait'),
    ),
    ('', 'editor --wait', ('editor', '--wait')),
    ('', '', ('vi',)),
  ],
)
def test_editor_command(
  monkeypatch: pytest.MonkeyPatch,
  visual: str,
  editor: str,
  expected: tuple[str, ...],
) -> None:
  monkeypatch.setenv('VISUAL', visual)
  monkeypatch.setenv('EDITOR', editor)
  path = pathlib.Path('/tmp/literal; $(unsafe)')
  assert terminal.editor_command(path) == (*expected, str(path))


@pytest.mark.parametrize('command', ['   ', '"unclosed'])
def test_editor_command_rejects_invalid_argv(
  monkeypatch: pytest.MonkeyPatch,
  command: str,
) -> None:
  monkeypatch.setenv('VISUAL', command)
  with pytest.raises(ValueError):
    terminal.editor_command(pathlib.Path('/tmp/prompt.md'))
