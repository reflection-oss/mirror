"""Raw workspace processes through the worker's native microsandbox transport."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import pathlib
import shlex
import signal
import tempfile
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import cast

from mirror.cli import shell
from mirror.cli.sandbox import setup as sandbox_setup


@dataclasses.dataclass(frozen=True)
class SandboxShell(shell.Shell):
  """A native process adapter; VM creation and policy stay in the Rust worker."""

  worker_command: tuple[str, ...] = ()
  configuration: Mapping[str, object] = dataclasses.field(
    default_factory=dict[str, object]
  )

  async def quarantine(self) -> None:
    """Persist uncertain completion outside the guest mount before stopping it."""
    result = await shell.Shell(pathlib.Path.home()).run(
      (
        *sandbox_setup.worker_command(self.worker_command),
        'sandbox',
        'quarantine',
        str(self.configuration['id']),
      ),
      timeout=90,
    )
    if result.returncode:
      raise RuntimeError(
        'Could not quarantine sandbox: '
        + result.stderr.decode(errors='replace').strip()
      )

  @contextlib.asynccontextmanager
  async def open(
    self, argv: Sequence[str], *, data: str = ''
  ) -> AsyncGenerator[shell.Process]:
    if not argv:
      raise ValueError('Workspace process requires an executable')
    # Completion evidence is host-owned, never in the mounted workspace.
    with tempfile.TemporaryDirectory(prefix='mirror-process-') as directory:
      outcome = pathlib.Path(directory) / 'outcome'
      launcher = shell.Shell(
        pathlib.Path.home(),
        termination_signal=signal.SIGTERM,
        # Allow admission plus sandboxd's process-group cancellation grace.
        termination_timeout=35,
      )
      process: _Process | None = None
      completed = False
      try:
        async with launcher.open(
          (
            *sandbox_setup.worker_command(self.worker_command),
            'sandbox',
            'exec',
            json.dumps(dict(self.configuration)),
            str(self.cwd),
            str(outcome),
            shlex.join(argv),
          ),
          data=data,
        ) as child:
          process = _Process(child, outcome)
          yield process
        completed = True
      finally:
        if process is not None:
          try:
            await process.wait()
          except _OutcomeUnknown:
            await self.quarantine()
            raise
          except shell.ProcessError:
            # Preserve the caller's timeout or cancellation after confirmed cleanup.
            if completed:
              raise


class _OutcomeUnknown(RuntimeError):
  pass


@dataclasses.dataclass(frozen=True)
class _Process:
  child: shell.Process
  outcome: pathlib.Path

  async def read_stdout(self) -> bytes:
    return await self.child.read_stdout()

  async def read_stderr(self) -> bytes:
    return await self.child.read_stderr()

  async def wait(self) -> int:
    await self.child.wait()
    try:
      value: object = json.loads(self.outcome.read_bytes())
    except (OSError, ValueError) as error:
      raise _OutcomeUnknown(
        'Sandbox command completion is unknown; it will not be replayed'
      ) from error
    if isinstance(value, dict):
      fields = cast(dict[str, object], value)
      status = fields.get('Ok')
      if isinstance(status, int) and not isinstance(status, bool):
        return status
      if isinstance(error := fields.get('Err'), str):
        raise shell.ProcessError(f'Sandbox process failed: {error}')
    raise _OutcomeUnknown(
      f'Sandbox command completion is unknown; it will not be replayed: {value}'
    )
