"""Streaming workspace file discovery for the editor picker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from mirror.cli import extension, selection

# Only this trusted source runs in the workspace; filenames and contents are data.
_SCAN = r"""
import json, os, pathlib, stat, subprocess, sys
root = pathlib.Path.cwd()
with subprocess.Popen([
    'rg', '--files', '--hidden', '--null', '--no-require-git',
    '--glob', '!.git', '--glob', '!.jj', '--', str(root),
], stdout=subprocess.PIPE) as process:
    pending = b''
    while chunk := process.stdout.read1(65536):
        *names, pending = (pending + chunk).split(b'\0')
        for name in names:
            path = pathlib.Path(os.fsdecode(name))
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, encoding='utf-8') as source:
                    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                        continue
                    if '\0' in source.read(8192):
                        continue
            except (OSError, UnicodeError):
                continue
            print(json.dumps(str(path.relative_to(root))), flush=True)
    code = process.wait()
    sys.exit(0 if code in (0, 1) else code)
"""


async def batches(
  workspace: extension.Workspace,
) -> AsyncGenerator[tuple[selection.Choice, ...], None]:
  """Stream text filenames without inspecting paths on the frontend host."""
  if workspace.shell is None:
    raise RuntimeError('Workspace process execution is unavailable')
  async with workspace.shell.open(('python3', '-I', '-c', _SCAN)) as process:
    process.close_stdin()
    pending = b''

    async def read_errors() -> bytes:
      detail = bytearray()
      while chunk := await process.read_stderr():
        detail.extend(chunk[: max(0, 65536 - len(detail))])
      return bytes(detail)

    stderr = asyncio.create_task(read_errors())
    try:
      while chunk := await process.read_stdout():
        *lines, pending = (pending + chunk).split(b'\n')
        if len(pending) > 65536:
          raise ValueError('Workspace filename exceeds its byte limit')
        choices: list[selection.Choice] = []
        for line in lines:
          value: object = json.loads(line)
          if not isinstance(value, str):
            raise ValueError('Invalid workspace file listing')
          choices.append(selection.Choice(value, repr(value)[1:-1]))
        yield tuple(choices)
      code = await process.wait()
      detail = await stderr
      if code or pending:
        raise ValueError(
          detail.decode(errors='replace').strip()
          or 'Incomplete workspace file listing'
        )
    finally:
      stderr.cancel()
      await asyncio.gather(stderr, return_exceptions=True)
