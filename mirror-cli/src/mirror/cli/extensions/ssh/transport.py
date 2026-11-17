"""Connect to an installed Mirror over a private, frontend-owned SSH tunnel."""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import re
import shlex
import tempfile
from collections.abc import Awaitable, Mapping, Sequence
from typing import Protocol, TextIO, cast

import grpc.aio
from mirror.protocol import (
  daemon_pb2,
  daemon_pb2_grpc,
)

from mirror.cli import frontend, rpc_client


class _Daemon(Protocol):
  def GetDaemonInfo(
    self, request: daemon_pb2.GetDaemonInfoRequest, *, timeout: float
  ) -> Awaitable[daemon_pb2.GetDaemonInfoResponse]: ...  # noqa: N802, generated RPC name

  def GetFrontendDefaults(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.GetFrontendDefaultsRequest,
    *,
    timeout: float,
    metadata: Sequence[tuple[str, str]],
  ) -> Awaitable[daemon_pb2.GetFrontendDefaultsResponse]: ...


def host_name(value: str) -> str:
  """Accept an OpenSSH alias or user@hostname, never shell syntax or options."""
  if not re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.@-]{0,252}', value):
    raise ValueError(
      'Enter user@host or an SSH config alias, without spaces or options.'
    )
  return value


def socket_path(value: str) -> str:
  """Validate remote discovery before using it as a forwarding target."""
  prefix = 'MIRROR_DAEMON_ADDRESS=unix:'
  if not value.startswith(prefix):
    raise ValueError('The devserver did not report a daemon socket.')
  path = value[len(prefix) :].removesuffix('\n')
  if not path.startswith('/') or any(
    character in ':\r\n\x00' or ord(character) < 32 for character in path
  ):
    raise ValueError('The devserver reported an invalid socket path.')
  return path


def ssh_environment(values: Mapping[str, str]) -> dict[str, str]:
  """Keep SSH authentication local without inheriting provider or loader settings."""
  return {
    name: values[name]
    for name in (
      'PATH',
      'HOME',
      'USER',
      'LOGNAME',
      'SSH_AUTH_SOCK',
      'TMPDIR',
      'LANG',
    )
    if name in values
  }


def _startup_command(terminal: TextIO) -> str:
  """Give build diagnostics a PTY while keeping the address on a separate pipe."""
  launch = (
    'if command -v mirror >/dev/null 2>&1; then '
    'exec mirror daemon address >&3 3>&-; fi; '
    'exec "${SHELL:-/bin/sh}" -ic '
    "'command -v mirror >/dev/null 2>&1 || exit 127; mirror daemon address >&3 3>&-'"
  )
  plain = f'exec 3>&1; {launch} >&2'
  if not terminal.isatty():
    return plain
  size = os.get_terminal_size(terminal.fileno())
  interactive = shlex.quote(
    f'stty cols {size.columns or 80} rows {size.lines or 24} -echo; {launch}'
  )
  term = shlex.quote(os.environ.get('TERM', 'xterm-256color'))
  return (
    f'exec 3>&1; export TERM={term}; '
    'if command -v script >/dev/null 2>&1 && command -v stty >/dev/null 2>&1; then '
    'case "$(uname -s)" in '
    f'Linux) exec script -qefc {interactive} /dev/null >&2;; '
    f'Darwin) exec script -q /dev/null "${{SHELL:-/bin/sh}}" -c {interactive} >&2;; '
    f'esac; fi; {launch} >&2'
  )


class Tunnel:
  """Own a private SSH master; no user's existing master is reused or stopped."""

  def __init__(
    self, host: str, context: frontend.Context, *, executable: str = 'ssh'
  ) -> None:
    self.host = host_name(host)
    self.context = context
    self._executable = executable
    # macOS's default temporary root can exceed Unix socket path limits.
    self._directory = tempfile.TemporaryDirectory(prefix='mr-ssh-', dir='/tmp')
    self.directory = pathlib.Path(self._directory.name)
    self._control = self.directory / 'control'
    self._closed = False

  def command(self, *arguments: str) -> tuple[str, ...]:
    return (
      self._executable,
      # Control operations use only this master, not forwards from SSH config.
      *(
        ('-F', '/dev/null')
        if '-O' in arguments
        else ('-o', 'ClearAllForwardings=yes')
      ),
      '-T',
      '-S',
      str(self._control),
      '-o',
      'ControlMaster=auto',
      '-o',
      'ControlPersist=60',
      '-o',
      'ForwardAgent=no',
      '-o',
      'ForwardX11=no',
      '-o',
      'BatchMode=no',
      '-o',
      'SendEnv=-*',
      '-o',
      'PermitLocalCommand=no',
      '-o',
      'StrictHostKeyChecking=ask',
      '-o',
      'ExitOnForwardFailure=yes',
      '-o',
      'ConnectTimeout=15',
      '-o',
      'ServerAliveInterval=15',
      '-o',
      'ServerAliveCountMax=3',
      *arguments,
      self.host,
    )

  async def run(
    self,
    command: str,
    *,
    check: bool = True,
    options: tuple[str, ...] = (),
    timeout: float | None = 120,
  ) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
      *self.command(*options),
      *([command] if command else []),
      env=ssh_environment(os.environ),
      stdin=asyncio.subprocess.DEVNULL,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.DEVNULL
      if self._closed
      else self.context.terminal,
    )
    try:
      async with asyncio.timeout(timeout):
        if process.stdout is None:
          raise RuntimeError('SSH stdout is unavailable')
        try:
          await process.stdout.readexactly(65537)
        except asyncio.IncompleteReadError as error:
          output = error.partial
        else:
          raise ValueError('SSH output exceeded the 64 KiB limit.')
        status = await process.wait()
      if check and status:
        raise RuntimeError(
          f'SSH command failed on {self.host} (exit {status}).'
        )
      return status, output.decode('utf-8')
    except TimeoutError:
      raise TimeoutError(f'SSH command on {self.host} timed out.') from None
    finally:
      if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
          process.kill()
      # Reaping also needs EOF on a paused stdout pipe, including after SIGKILL.
      await process.communicate()

  async def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if self._control.exists():
        with contextlib.suppress(OSError, TimeoutError):
          async with asyncio.timeout(5):
            await self.run('', check=False, options=('-O', 'exit'))
    finally:
      self._directory.cleanup()


async def connect(
  context: frontend.Context,
  identity: str,
  host: str,
  *,
  ssh: str = 'ssh',
) -> frontend.Connection:
  """Attach to the host's running daemon; never start, update or replace it."""
  tunnel = Tunnel(host, context, executable=ssh)
  handed_off = False
  try:
    print(
      f'Connecting to {tunnel.host} with OpenSSH…',
      file=context.terminal,
      flush=True,
    )
    # A development launcher may need an arbitrarily long cold build.
    status, output = await tunnel.run(
      _startup_command(context.terminal), check=False, timeout=None
    )
    if status == 255:
      raise RuntimeError(
        f'SSH connection to {tunnel.host} failed. Check the SSH host/alias and key; run `ssh {tunnel.host}` to see the SSH error.'
      )
    if status in (126, 127):
      raise RuntimeError(
        f'The remote shell could not run `mirror` on {tunnel.host}. Check its PATH or alias definition.'
      )
    if status:
      raise RuntimeError(
        f'`mirror daemon address` failed on {tunnel.host} (exit {status}). It requires a running daemon. Run that command on the devserver to see the error.'
      )
    remote_socket = socket_path(output)
    local_socket = tunnel.directory / 'rpc'
    await tunnel.run(
      '', options=('-O', 'forward', '-L', f'{local_socket}:{remote_socket}')
    )
    address = f'unix:{local_socket}'
    async with grpc.aio.insecure_channel(
      address,
      options=rpc_client.GRPC_CHANNEL_OPTIONS
      + (
        ('grpc.enable_http_proxy', 0),
        ('grpc.default_authority', 'localhost'),
      ),
    ) as channel:
      daemon = cast(_Daemon, daemon_pb2_grpc.DaemonRPCStub(channel))
      info = await daemon.GetDaemonInfo(
        daemon_pb2.GetDaemonInfoRequest(), timeout=10
      )
      if info.protocol_revision != daemon_pb2.PROTOCOL_REVISION_CURRENT:
        raise ValueError(
          f'Mirror on {tunnel.host} uses an incompatible protocol. Install compatible Mirror versions on both machines, then reconnect.'
        )
      if info.upgrading:
        raise ValueError(
          'The devserver daemon is upgrading. Wait for it to finish, then reconnect.'
        )
      try:
        await daemon.GetFrontendDefaults(
          daemon_pb2.GetFrontendDefaultsRequest(),
          timeout=10,
          metadata=(('mirror-daemon-id', info.instance_id),),
        )
      except grpc.aio.AioRpcError:
        raise RuntimeError(
          'Could not load devserver workspace defaults. Run `mirror daemon address` on the devserver, then reconnect with /ssh.'
        ) from None
    handed_off = True
    return frontend.Connection(
      identity, host, address, info.instance_id, tunnel.close
    )
  except grpc.aio.AioRpcError:
    raise RuntimeError(
      'Could not reach the devserver daemon. Reconnect with /ssh.'
    ) from None
  finally:
    if not handed_off:
      await tunnel.close()
