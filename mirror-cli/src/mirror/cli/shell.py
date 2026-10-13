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


class Process(Protocol):
  """Byte streams and exit status; drain output concurrently before waiting."""

  async def read_stdout(self) -> bytes: ...

  async def read_stderr(self) -> bytes: ...

  async def write(self, data: bytes) -> None: ...

  def close_stdin(self) -> None: ...

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

  async def write(self, data: bytes) -> None:
    assert self._process.stdin is not None
    for offset in range(0, len(data), 65536):
      self._process.stdin.write(data[offset : offset + 65536])
      await self._process.stdin.drain()

  def close_stdin(self) -> None:
    assert self._process.stdin is not None
    self._process.stdin.close()

  async def wait(self) -> int:
    return await self._process.wait()

  async def close(self) -> bool:
    """Reap the process; report whether forced termination was necessary."""
    self.close_stdin()

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
        os.killpg(self._process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
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
  validate: Callable[[], None] | None = dataclasses.field(
    default=None, repr=False
  )
  on_abort: Callable[[], Awaitable[None]] | None = dataclasses.field(
    default=None, repr=False
  )

  @contextlib.asynccontextmanager
  async def open(self, argv: Sequence[str]) -> AsyncGenerator[Process]:
    """Open a process with byte-faithful pipes; never change execution targets."""
    if not argv:
      raise ValueError('Workspace process requires an executable')
    if self.validate is not None:
      self.validate()
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
      yield process
    finally:

      async def cleanup() -> None:
        forced = True
        try:
          forced = await process.close()
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
    data: bytes = b'',
    timeout: float = 30,
    limit: int = 2 * 1024 * 1024,
  ) -> Result:
    """Capture a bounded command without rendering, retrying or model execution."""
    if timeout <= 0 or limit <= 0:
      raise ValueError('Command timeout and byte limit must be positive')
    async with self.open(argv) as process:

      async def feed() -> None:
        try:
          await process.write(data)
        except (BrokenPipeError, ConnectionResetError):
          pass
        finally:
          process.close_stdin()

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
        feeding = asyncio.create_task(feed())
        stdout = asyncio.create_task(read(process.read_stdout))
        stderr = asyncio.create_task(read(process.read_stderr))
        tasks = (feeding, stdout, stderr)
        try:
          await asyncio.gather(*tasks)
          return Result(stdout.result(), stderr.result(), await process.wait())
        finally:
          for task in tasks:
            task.cancel()
          await asyncio.gather(*tasks, return_exceptions=True)
