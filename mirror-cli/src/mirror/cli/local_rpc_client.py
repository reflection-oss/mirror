"""Local AgentRPC worker transport."""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import json
import os
import pathlib
import signal
import sys
from collections.abc import Mapping, Sequence

import grpc.aio
from google.protobuf import message
from mirror.protocol import agent_pb2

from mirror.cli import host_tools as _host_tools
from mirror.cli import rpc_client

_WORKER_LAUNCHER = (
  'import os,sys; '
  'os.chdir(sys.argv[1]); '
  'os.setsid(); '
  'os.execvp(sys.argv[2], sys.argv[2:])'
)


def _signal_group(pid: int, sig: signal.Signals) -> bool:
  """Signal `pid`'s process group; False if it is already gone."""
  try:
    os.killpg(pid, sig)
    return True
  except ProcessLookupError:
    return False


class Transport:
  """Start a local worker process and own its AgentRPC channel."""

  def __init__(
    self,
    worker_command: Sequence[str],
    worker_config: Mapping[str, object],
    *,
    workspace: pathlib.Path,
    log_dir: pathlib.Path,
    on_message: rpc_client.MessageHandler | None = None,
  ) -> None:
    self._worker_command = list(worker_command)
    self._worker_config = dict(worker_config)
    self._workspace = workspace
    self._log_dir = log_dir
    self._on_message = on_message
    self._process: asyncio.subprocess.Process | None = None
    self._channel: grpc.aio.Channel | None = None
    self._log_tasks: list[asyncio.Task[None]] = []

  @property
  def process(self) -> asyncio.subprocess.Process | None:
    """The active worker process."""
    return self._process

  async def connect(
    self, host_tools: Sequence[_host_tools.ToolDefinition] = ()
  ) -> grpc.aio.Channel:
    """Start the worker and return its ready AgentRPC channel."""
    address = '127.0.0.1:0'
    config = self._worker_config | {
      'rpc_address': address,
      'exit_on_stdin_close': True,
      'host_tools': [
        {
          key: value
          for key, value in dataclasses.asdict(tool).items()
          if key != 'global_'
        }
        for tool in host_tools
      ],
    }
    self._log_dir.mkdir(parents=True, exist_ok=True)
    try:
      self._process = await asyncio.create_subprocess_exec(
        str(pathlib.Path(sys.executable).resolve()),
        '-c',
        _WORKER_LAUNCHER,
        str(self._workspace),
        *self._worker_command,
        json.dumps(config),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=False,
      )
      process = self._process
      self._log_tasks = [
        asyncio.create_task(
          self._collect_logs(process.stderr, self._log_dir / 'stderr')
        ),
      ]
      try:
        address = await asyncio.wait_for(
          self._read_worker_address(process), timeout=30
        )
      except TimeoutError as error:
        raise RuntimeError(
          'agent worker did not report its address within 30 seconds; '
          f'see {self._log_dir / "stderr"}'
        ) from error
      self._log_tasks.append(
        asyncio.create_task(
          self._collect_logs(process.stdout, self._log_dir / 'stdout')
        )
      )
      self._channel = grpc.aio.insecure_channel(
        address, options=rpc_client.GRPC_CHANNEL_OPTIONS
      )
      ready = asyncio.create_task(self._channel.channel_ready())
      exited = asyncio.create_task(process.wait())
      done, _ = await asyncio.wait(
        (ready, exited), return_when=asyncio.FIRST_COMPLETED
      )
      if exited in done:
        ready.cancel()
        await asyncio.gather(ready, return_exceptions=True)
        raise RuntimeError(
          f'agent worker exited with status {process.returncode}; '
          f'see {self._log_dir / "stderr"}'
        )
      exited.cancel()
      await asyncio.gather(exited, return_exceptions=True)
      return self._channel
    except BaseException:
      await self.close()
      raise

  async def close(self) -> None:
    """Close the channel, stop the worker, and await its log readers."""
    if self._channel is not None:
      await self._channel.close(grace=None)
      self._channel = None
    process = self._process
    if process is not None and process.returncode is None:
      _signal_group(process.pid, signal.SIGTERM)
      try:
        await asyncio.wait_for(process.wait(), timeout=3)
      except TimeoutError:
        _signal_group(process.pid, signal.SIGKILL)
        await process.wait()
    if self._log_tasks:
      _, pending = await asyncio.wait(self._log_tasks, timeout=1)
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
    self._log_tasks.clear()
    self._process = None

  async def _read_worker_address(
    self, process: asyncio.subprocess.Process
  ) -> str:
    """Read the address after the worker has atomically bound its listener."""
    if process.stdout is None:
      raise RuntimeError('agent worker has no stdout pipe')
    address_prefix = b'MIRROR_RPC_ADDRESS='
    error_prefix = b'MIRROR_RPC_ERROR='
    while line := await process.stdout.readline():
      if self._consume_event(line):
        continue
      if line.startswith(address_prefix):
        return line.removeprefix(address_prefix).decode().strip()
      if line.startswith(error_prefix):
        raise RuntimeError(line.removeprefix(error_prefix).decode().strip())
      self._append_log(self._log_dir / 'stdout', line)
    status = await process.wait()
    raise RuntimeError(
      f'agent worker exited with status {status} before reporting its address; '
      f'see {self._log_dir / "stderr"}'
    )

  def _consume_event(self, line: bytes) -> bool:
    prefix = b'MIRROR_SESSION_MESSAGE='
    if not line.startswith(prefix):
      return False
    try:
      event = agent_pb2.SessionMessage.FromString(
        base64.b64decode(line.removeprefix(prefix).strip(), validate=True)
      )
    except (binascii.Error, message.DecodeError):
      raise RuntimeError('Invalid worker event') from None
    if self._on_message is not None:
      self._on_message(event)
    return True

  @staticmethod
  def _append_log(path: pathlib.Path, content: bytes) -> None:
    with path.open('ab') as sink:
      sink.write(content)
      sink.flush()

  async def _collect_logs(
    self, stream: asyncio.StreamReader | None, path: pathlib.Path
  ) -> None:
    if stream is None:
      return
    with path.open('ab') as sink:
      try:
        while line := await stream.readline():
          if path.name == 'stdout' and self._consume_event(line):
            continue
          sink.write(line)
          sink.flush()
      except Exception:  # noqa: BLE001 - never let worker logging block.
        while await stream.read(1 << 16):
          pass
