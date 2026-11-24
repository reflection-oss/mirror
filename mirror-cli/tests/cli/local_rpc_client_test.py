"""Daemon announcements are verified before startup events reach the client."""

import asyncio
import json
import os
import pathlib
import sys
import tempfile
from typing import override

import grpc.aio
import pytest
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  daemon_pb2_grpc,
)

from mirror.cli import local_rpc_client, shell


async def _check_transport(root: pathlib.Path, reported_address: str) -> None:
  event = agent_pb2.SessionMessage(
    replay_done=agent_pb2.ReplayDone(conversation_id='synthetic')
  )
  requests: list[daemon_pb2.AttachWorkspaceRequest] = []

  class Daemon(daemon_pb2_grpc.DaemonRPCServicer):
    @override
    def AttachWorkspace(  # noqa: N802, generated RPC name
      self,
      request: daemon_pb2.AttachWorkspaceRequest,
      context: grpc.ServicerContext,
    ) -> daemon_pb2.AttachWorkspaceResponse:
      del context
      requests.append(request)
      return daemon_pb2.AttachWorkspaceResponse(
        attachment_id='attachment',
        execution_host=True,
        worker_instance_id='incarnation',
        workspace=daemon_pb2.Workspace(
          id=request.workspace_id, session_id='synthetic'
        ),
        startup_messages=[event],
      )

  state = pathlib.Path(os.environ['MIRROR_DAEMON_STATE_ROOT'])
  assert state.stat().st_mode & 0o777 == 0o700
  address = f'unix:{state / "daemon.sock"}'
  announcement = root / 'announcement'
  announcement.write_text(
    'MIRROR_DAEMON_ADDRESS=' + (reported_address or address)
  )
  ensures = root / 'ensures'
  daemon = root / 'mirror-daemon'
  daemon.write_text(
    f'#!{sys.executable}\n'
    'import pathlib, sys\n'
    "assert sys.argv[1] == '--ensure' and '--replace' not in sys.argv\n"
    "assert '--state-root' in sys.argv and '--database' in sys.argv\n"
    "assert '--socket' not in sys.argv and '--worker' not in sys.argv\n"
    "sys.stderr.buffer.write(b'x' * (1 << 18) + b'\\n')\n"
    "print('MIRROR_DAEMON_WARNING=synthetic deferred update', file=sys.stderr)\n"
    f'with pathlib.Path({str(ensures)!r}).open("a") as output:\n'
    '  output.write("ensure\\n")\n'
    f'print(pathlib.Path({str(announcement)!r}).read_text())\n'
  )
  daemon.chmod(0o700)
  server = grpc.aio.server()
  daemon_pb2_grpc.add_DaemonRPCServicer_to_server(Daemon(), server)  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
  server.add_insecure_port(address)
  await server.start()
  received: list[agent_pb2.SessionMessage] = []
  warnings: list[str] = []

  async def receive(message: agent_pb2.SessionMessage) -> None:
    await asyncio.sleep(0)
    received.append(message)

  transport = local_rpc_client.Transport(
    [str(root / 'mirror-agent')],
    {
      'workspace_id': 'workspace',
      'root_search_guard': True,
      'conversation_id': 'synthetic',
      'api': 'anthropic-messages',
      'model_api': 'openai-codex-responses',
      'base_url': 'http://default.invalid',
      'model_base_url': 'http://selected.invalid',
      'api_key': 'synthetic-provider-key',
    },
    workspace=root,
    log_dir=root,
    on_message=receive,
    on_warning=warnings.append,
  )
  try:
    with pytest.raises(shell.ProcessError, match='Workspace is not attached'):
      transport.workspace_connection()
    if reported_address:
      with pytest.raises(RuntimeError, match='unexpected endpoint'):
        await asyncio.wait_for(transport.connect(), timeout=5)
      assert requests == []
      assert received == []
      assert warnings == ['synthetic deferred update']
    else:
      connection = await asyncio.wait_for(transport.connect(), timeout=5)
      assert received == [event]
      assert connection.execution_host
      assert connection.daemon_attachment_id == 'attachment'
      assert connection.attachment is not None
      assert connection.attachment.WhichOneof('payload') is None
      assert len(requests) == 1
      assert requests[0].workspace_id == 'workspace'
      assert requests[0].configuration.root_search_guard
      assert (
        requests[0].configuration.inference.api == 'openai-chat-completions'
      )
      assert (
        requests[0].configuration.inference.base_url
        == 'http://selected.invalid'
      )
      await transport.close()
      with pytest.raises(shell.ProcessError, match='Workspace is not attached'):
        transport.workspace_connection()
      assert connection.reconnect is not None
      reattached = await connection.reconnect(False)
      assert reattached.execution_host
      assert reattached.attachment is not None
      assert reattached.attachment.WhichOneof('payload') is None
      assert transport.workspace_connection() == (
        reattached.channel,
        reattached.daemon_attachment_id,
      )
      assert len(requests) == 2
      assert requests[1].existing_worker_id == 'incarnation'
      assert requests[1].workspace_id == 'workspace'
      assert not requests[1].require_new_worker
      assert not requests[1].configuration.HasField('inference')
      assert requests[0].configuration.api_key == 'synthetic-provider-key'
      assert not requests[1].configuration.HasField('api_key')
      assert received == [event]
      assert (state / 'daemon.sock').is_socket()
      await transport.close()
      await connection.reconnect(True)
      assert ensures.read_text() == 'ensure\n'
      await transport.close()
      await server.stop(None)
      with pytest.raises(grpc.aio.AioRpcError):
        await connection.reconnect(False)
      assert ensures.read_text() == 'ensure\n'
      address = f'unix:{state / "replacement.sock"}'
      announcement.write_text('MIRROR_DAEMON_ADDRESS=' + address)
      server = grpc.aio.server()
      daemon_pb2_grpc.add_DaemonRPCServicer_to_server(Daemon(), server)  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
      server.add_insecure_port(address)
      await server.start()
      await connection.reconnect(True)
      assert ensures.read_text() == 'ensure\nensure\n'
      assert requests[-1].existing_worker_id == 'incarnation'
      assert not requests[-1].require_new_worker
      assert not requests[-1].configuration.HasField('api_key')
      assert received == [event]
      assert warnings == [
        'synthetic deferred update',
        'synthetic deferred update',
      ]
    assert not (root / 'stderr').exists()
    assert not (root / 'stdout').exists()
  finally:
    await transport.close()
    await server.stop(None)


@pytest.mark.asyncio
async def test_manage_restart_explicitly_replaces_a_connect_only_attachment(
  tmp_path: pathlib.Path,
) -> None:
  launcher = tmp_path / 'mirror-daemon'
  launcher.write_text(
    f'#!{sys.executable}\n'
    'import sys\n'
    "assert sys.argv[1:3] == ['--ensure', '--replace']\n"
    "print('MIRROR_DAEMON_ADDRESS=unix:/synthetic/daemon.sock')\n"
  )
  launcher.chmod(0o700)
  transport = local_rpc_client.Transport(
    [str(tmp_path / 'mirror-agent')],
    {'database': str(tmp_path / 'state.db'), 'daemon_connect_only': True},
    workspace=tmp_path,
    log_dir=tmp_path,
  )
  assert await transport.manage('restart') == (
    'Mirror daemon restarted; interrupted local turns remain paused.'
  )


@pytest.mark.asyncio
async def test_manage_restart_rejects_an_ssh_attachment(
  tmp_path: pathlib.Path,
) -> None:
  transport = local_rpc_client.Transport(
    [str(tmp_path / 'mirror-agent')],
    {'daemon_address': 'unix:/synthetic/forwarded.sock'},
    workspace=tmp_path,
    log_dir=tmp_path,
  )
  with pytest.raises(local_rpc_client.DaemonConnectionError, match='/ssh'):
    await transport.manage('restart')


@pytest.mark.asyncio
@pytest.mark.parametrize('reported_address', ['', '127.0.0.1:1'])
async def test_transport_drains_diagnostics_and_awaits_startup_events(
  tmp_path: pathlib.Path,
  reported_address: str,
) -> None:
  # Isolate discovery environment without changing other tests' process globals.
  with tempfile.TemporaryDirectory(prefix='mr-') as state:
    script = (
      'import asyncio, pathlib, runpy; '
      f'namespace = runpy.run_path({str(pathlib.Path(__file__).resolve())!r}); '
      f'asyncio.run(namespace[{_check_transport.__name__!r}]('
      f'pathlib.Path({str(tmp_path)!r}), {reported_address!r}))'
    )
    process = await asyncio.create_subprocess_exec(
      sys.executable,
      '-c',
      script,
      env={
        'HOME': str(tmp_path),
        'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
        'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
        'MIRROR_DAEMON_STATE_ROOT': state,
        'MIRROR_RUST_WORKER': str(tmp_path / 'mirror-agent'),
        'GRPC_ENABLE_FORK_SUPPORT': '0',
      },
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.STDOUT,
    )
    try:
      output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
      assert process.returncode == 0, output.decode()
    finally:
      if process.returncode is None:
        process.kill()
        await process.wait()


@pytest.mark.asyncio
async def test_manage_address_configures_host_mcp_references_without_printing_them(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path / 'mcp'))
  monkeypatch.setenv('FIXTURE_TOKEN', 'synthetic-host-secret')
  declaration = {
    'url': 'https://fixture.test/mcp',
    'auth': {'type': 'bearer', 'tokenEnvVar': 'FIXTURE_TOKEN'},
  }
  (tmp_path / 'mcp').mkdir()
  (tmp_path / 'mcp/fixture.json').write_text(json.dumps(declaration))
  requests: list[daemon_pb2.ConfigureDaemonRequest] = []

  class Daemon(daemon_pb2_grpc.DaemonRPCServicer):
    @override
    def ConfigureDaemon(  # noqa: N802, generated RPC name
      self,
      request: daemon_pb2.ConfigureDaemonRequest,
      context: grpc.ServicerContext,
    ) -> daemon_pb2.ConfigureDaemonResponse:
      del context
      requests.append(request)
      return daemon_pb2.ConfigureDaemonResponse()

  with tempfile.TemporaryDirectory(prefix='mr-') as state:
    address = f'unix:{state}/daemon.sock'
    launcher = tmp_path / 'mirror-daemon'
    launcher.write_text(
      f'#!{sys.executable}\n'
      'import sys\n'
      "assert sys.argv[1] == '--connect-only'\n"
      f"print('MIRROR_DAEMON_ADDRESS={address}')\n"
    )
    launcher.chmod(0o700)
    server = grpc.aio.server()
    daemon_pb2_grpc.add_DaemonRPCServicer_to_server(Daemon(), server)  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
    server.add_insecure_port(address)
    await server.start()
    transport = local_rpc_client.Transport(
      [str(tmp_path / 'mirror-agent')], {}, workspace=tmp_path, log_dir=tmp_path
    )
    try:
      assert (
        await transport.manage('address') == f'MIRROR_DAEMON_ADDRESS={address}'
      )
    finally:
      await transport.close()
      await server.stop(None)
  assert len(requests) == 1
  assert requests[0].configurations[0].mcp_connections.connections[
    0
  ] == agent_pb2.McpConnection(
    name='fixture',
    config=agent_pb2.McpConnectionConfig(
      url='https://fixture.test/mcp', bearer_env_var='FIXTURE_TOKEN'
    ),
    credential_directory=str(tmp_path / 'mcp/credentials'),
  )
  assert 'synthetic-host-secret' not in str(requests[0])
