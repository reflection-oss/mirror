"""Provider tokens read from command-line logins on this machine."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Sequence

_TIMEOUT_SECONDS = 60
_MAX_TOKEN_BYTES = 16 * 1024


class TokenCommandError(Exception):
  """A token command that did not print exactly one token."""


async def capture(command: Sequence[str]) -> str:
  """Run a token command and return its single line of output.

  Raises:
    TokenCommandError: The command could not start, failed, timed out, or did
      not print exactly one line. The message never contains its output.
  """
  try:
    process = await asyncio.create_subprocess_exec(
      *command,
      stdin=asyncio.subprocess.DEVNULL,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.DEVNULL,
      start_new_session=True,
    )
  except OSError as error:
    raise TokenCommandError(
      f'could not start {command[0]}: {error.strerror or error}'
    ) from None
  try:
    stdout, _ = await asyncio.wait_for(
      process.communicate(), timeout=_TIMEOUT_SECONDS
    )
  except TimeoutError:
    await _stop(process)
    raise TokenCommandError(
      f'timed out after {_TIMEOUT_SECONDS} seconds'
    ) from None
  except asyncio.CancelledError:
    await _stop(process)
    raise
  if process.returncode != 0:
    raise TokenCommandError(f'exited with status {process.returncode}')
  if len(stdout) > _MAX_TOKEN_BYTES:
    raise TokenCommandError(f'printed more than {_MAX_TOKEN_BYTES} bytes')
  lines = [
    line.strip() for line in stdout.decode(errors='replace').splitlines()
  ]
  lines = [line for line in lines if line]
  if len(lines) != 1:
    raise TokenCommandError('did not print exactly one line')
  return lines[0]


async def _stop(process: asyncio.subprocess.Process) -> None:
  with contextlib.suppress(ProcessLookupError):
    os.killpg(process.pid, signal.SIGKILL)
  await process.communicate()
