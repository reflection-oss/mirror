"""Workspace processes independent of agents, tools and terminal views."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import pathlib
import signal
from collections.abc import (
  AsyncGenerator,
  Awaitable,
  Callable,
  Mapping,
  Sequence,
)
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class Result:
  """Raw command output; a nonzero exit status is not a transport failure."""

  stdout: bytes
  stderr: bytes
  returncode: int


class ProcessError(RuntimeError):
  """A process error with no uncertain execution outcome."""


class Process(Protocol):
  """Output byte streams and exit status; drain concurrently before waiting."""

  async def read_stdout(self) -> bytes: ...

  async def read_stderr(self) -> bytes: ...

  async def wait(self) -> int: ...


class _LocalProcess:
  """Bounded pipes and lifetime for one host-launched process."""

  def __init__(self, process: asyncio.subprocess.Process) -> None:
    self._process = process

  async def read_stdout(self) -> bytes:
    assert self._process.stdout is not None
    return await self._process.stdout.read(65536)

  async def read_stderr(self) -> bytes:
    assert self._process.stderr is not None
    return await self._process.stderr.read(65536)

  async def wait(self) -> int:
    return await self._process.wait()

  async def close(
    self, termination_signal: signal.Signals, termination_timeout: float
  ) -> bool:
    """Reap the process; report whether forced termination was necessary."""

    async def drain(read_chunk: Callable[[], Awaitable[bytes]]) -> None:
      while await read_chunk():
        pass

    # Drain while waiting: a child can be blocked on a full output pipe.
    finishing = asyncio.gather(
      drain(self.read_stdout),
      drain(self.read_stderr),
      self._process.wait(),
    )
    try:
      async with asyncio.timeout(1):
        await asyncio.shield(finishing)
      return False
    except TimeoutError:
      try:
        os.killpg(self._process.pid, termination_signal)
      except ProcessLookupError:
        pass
      try:
        async with asyncio.timeout(termination_timeout):
          await asyncio.shield(finishing)
      except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
          os.killpg(self._process.pid, signal.SIGKILL)
        try:
          async with asyncio.timeout(1):
            await asyncio.shield(finishing)
        except TimeoutError:
          finishing.cancel()
          await asyncio.gather(finishing, return_exceptions=True)
          raise RuntimeError(
            'Workspace process did not close after termination'
          ) from None
      return True


@dataclasses.dataclass(frozen=True)
class Shell:
  """A trusted process launcher bound to one workspace and environment."""

  cwd: pathlib.Path
  environment: Mapping[str, str] | None = dataclasses.field(
    default=None, repr=False
  )
  prefix: tuple[str, ...] = ()
  termination_signal: signal.Signals = dataclasses.field(
    default=signal.SIGKILL, kw_only=True
  )
  termination_timeout: float = dataclasses.field(default=1, kw_only=True)
  validate: Callable[[], None] | None = dataclasses.field(
    default=None, repr=False
  )
  on_abort: Callable[[], Awaitable[None]] | None = dataclasses.field(
    default=None, repr=False
  )

  @contextlib.asynccontextmanager
  async def open(
    self, argv: Sequence[str], *, data: str = ''
  ) -> AsyncGenerator[Process]:
    """Supply UTF-8 stdin once and stream output from the bound workspace."""
    if not argv:
      raise ValueError('Workspace process requires an executable')
    if self.validate is not None:
      self.validate()
    input_bytes = data.encode('utf-8')
    child = await asyncio.create_subprocess_exec(
      *self.prefix,
      *argv,
      cwd=self.cwd,
      env=self.environment,
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      start_new_session=True,
      limit=65536,
    )
    process = _LocalProcess(child)
    try:
      assert child.stdin is not None
      child.stdin.write(input_bytes)
      child.stdin.close()
      yield process
    finally:

      async def cleanup() -> None:
        forced = True
        try:
          forced = await process.close(
            self.termination_signal, self.termination_timeout
          )
        finally:
          if (forced or child.returncode) and self.on_abort is not None:
            await self.on_abort()

      closing = asyncio.create_task(cleanup())
      try:
        await asyncio.shield(closing)
      except asyncio.CancelledError:
        await closing
        raise

  async def run(
    self,
    argv: Sequence[str],
    *,
    data: str = '',
    timeout: float = 30,
    limit: int = 2 * 1024 * 1024,
  ) -> Result:
    """Capture a bounded command without rendering, retrying or model execution."""
    if timeout <= 0 or limit <= 0:
      raise ValueError('Command timeout and byte limit must be positive')
    async with self.open(argv, data=data) as process:

      async def read(read_chunk: Callable[[], Awaitable[bytes]]) -> bytes:
        output = bytearray()
        while chunk := await read_chunk():
          if len(output) + len(chunk) > limit:
            raise RuntimeError(
              f'Workspace command output exceeds its byte limit ({limit} bytes)'
            )
          output.extend(chunk)
        return bytes(output)

      async with asyncio.timeout(timeout):
        stdout = asyncio.create_task(read(process.read_stdout))
        stderr = asyncio.create_task(read(process.read_stderr))
        tasks = (stdout, stderr)
        try:
          await asyncio.gather(*tasks)
          return Result(stdout.result(), stderr.result(), await process.wait())
        finally:
          for task in tasks:
            task.cancel()
          await asyncio.gather(*tasks, return_exceptions=True)
