import asyncio
import pathlib
import sys

import pytest

from mirror.cli import extension, shell


@pytest.mark.asyncio
async def test_run_returns_raw_streams_and_exit_status(
  tmp_path: pathlib.Path,
) -> None:
  result = await shell.Shell(tmp_path).run([
    sys.executable,
    '-c',
    "import os; os.write(1, b'out\\x00'); os.write(2, b'err\\xff'); raise SystemExit(7)",
  ])
  assert result == shell.Result(b'out\x00', b'err\xff', 7)


@pytest.mark.asyncio
async def test_open_supports_multiple_stdin_messages(
  tmp_path: pathlib.Path,
) -> None:
  async with shell.Shell(tmp_path).open(['/bin/cat']) as process:
    await process.write(b'first\n')
    assert await process.read_stdout() == b'first\n'
    await process.write(b'second\n')
    assert await process.read_stdout() == b'second\n'
    process.close_stdin()
    assert await process.read_stdout() == b''
    assert await process.wait() == 0


@pytest.mark.asyncio
async def test_workspaces_keep_directories_and_environments_separate(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  first = tmp_path / 'first'
  second = tmp_path / 'second'
  first.mkdir()
  second.mkdir()
  monkeypatch.setenv('HOST_CANARY', 'synthetic-host-secret')
  command = [
    sys.executable,
    '-c',
    "import os; print(os.getcwd()); print(os.environ.get('WORKSPACE')); print(os.environ.get('HOST_CANARY'))",
  ]
  left, right = await asyncio.gather(
    shell.Shell(first, environment={'WORKSPACE': 'left'}).run(command),
    shell.Shell(second, environment={'WORKSPACE': 'right'}).run(command),
  )
  assert left.stdout == f'{first}\nleft\nNone\n'.encode()
  assert right.stdout == f'{second}\nright\nNone\n'.encode()


@pytest.mark.asyncio
@pytest.mark.parametrize('descriptor', [1, 2])
async def test_run_bounds_both_output_streams(
  tmp_path: pathlib.Path, descriptor: int
) -> None:
  async with asyncio.timeout(5):
    with pytest.raises(RuntimeError, match='byte limit'):
      await shell.Shell(tmp_path).run(
        [
          sys.executable,
          '-c',
          f'import os\nwhile True: os.write({descriptor}, b"x" * 65536)',
        ],
        limit=100,
      )


@pytest.mark.asyncio
async def test_run_timeout_aborts_without_replay(
  tmp_path: pathlib.Path,
) -> None:
  aborted = asyncio.Event()

  async def abort() -> None:
    aborted.set()

  async with asyncio.timeout(5):
    with pytest.raises(TimeoutError):
      await shell.Shell(tmp_path, on_abort=abort).run(
        [sys.executable, '-c', 'import time; time.sleep(60)'],
        timeout=0.1,
      )
  assert aborted.is_set()


@pytest.mark.asyncio
async def test_cancel_closes_process_before_returning(
  tmp_path: pathlib.Path,
) -> None:
  opened = asyncio.Event()
  aborted = asyncio.Event()
  processes: list[shell.Process] = []

  async def abort() -> None:
    aborted.set()

  async def run() -> None:
    async with shell.Shell(tmp_path, on_abort=abort).open([
      '/bin/cat'
    ]) as process:
      processes.append(process)
      opened.set()
      await asyncio.Event().wait()

  task = asyncio.create_task(run())
  await opened.wait()
  task.cancel()
  async with asyncio.timeout(5):
    with pytest.raises(asyncio.CancelledError):
      await task
    assert await processes[0].wait() == 0
  assert not aborted.is_set()


@pytest.mark.asyncio
async def test_open_consumer_error_does_not_abort_completed_process(
  tmp_path: pathlib.Path,
) -> None:
  aborted = asyncio.Event()

  async def abort() -> None:
    aborted.set()

  with pytest.raises(ValueError, match='consumer failed'):
    async with shell.Shell(tmp_path, on_abort=abort).open(['/bin/cat']):
      raise ValueError('consumer failed')
  assert not aborted.is_set()


@pytest.mark.asyncio
async def test_close_drains_unconsumed_output(tmp_path: pathlib.Path) -> None:
  async with asyncio.timeout(5):
    async with shell.Shell(tmp_path).open([
      sys.executable,
      '-c',
      "import os; os.write(1, b'x' * 1048576); os.write(2, b'y' * 1048576)",
    ]) as process:
      pass
    assert await process.wait() == 0


@pytest.mark.asyncio
async def test_validation_failure_never_launches(
  tmp_path: pathlib.Path,
) -> None:
  marker = tmp_path / 'launched'

  def validate() -> None:
    raise RuntimeError('workspace disconnected')

  with pytest.raises(RuntimeError, match='disconnected'):
    await shell.Shell(tmp_path, validate=validate).run([
      '/usr/bin/touch',
      str(marker),
    ])
  assert not marker.exists()


@pytest.mark.asyncio
async def test_nonzero_launcher_exit_invalidates_bound_backend(
  tmp_path: pathlib.Path,
) -> None:
  aborted = asyncio.Event()

  async def abort() -> None:
    aborted.set()

  result = await shell.Shell(tmp_path, on_abort=abort).run([
    '/bin/sh',
    '-c',
    'exit 7',
  ])
  assert result.returncode == 7
  assert aborted.is_set()


@pytest.mark.asyncio
async def test_workspace_run_uses_bound_shell_without_agent(
  tmp_path: pathlib.Path,
) -> None:
  host = tmp_path / 'host'
  target = tmp_path / 'target'
  host.mkdir()
  target.mkdir()
  workspace = extension.Workspace(host, remote=False, shell=shell.Shell(target))
  result = await workspace.run(
    'printf workspace > marker; printf output; printf error >&2; exit 7'
  )
  assert result == shell.Result(b'output', b'error', 7)
  assert (target / 'marker').read_text() == 'workspace'
  assert not (host / 'marker').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', [False, True])
async def test_workspace_run_without_backend_never_executes_on_host(
  tmp_path: pathlib.Path, remote: bool
) -> None:
  workspace = extension.Workspace(tmp_path, remote=remote)
  with pytest.raises(RuntimeError, match='unavailable'):
    await workspace.run('touch marker')
  assert not (tmp_path / 'marker').exists()
