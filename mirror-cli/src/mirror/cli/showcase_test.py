import asyncio
import contextlib
import os
import pathlib
import sys

import pytest
from mirror.protocol import daemon_pb2

from mirror.cli import remote_rpc_client, showcase


@pytest.mark.asyncio
async def test_showcase_session_serves_synthetic_catalog_and_remote_identity(
  tmp_path: pathlib.Path,
) -> None:
  with contextlib.chdir(tmp_path):
    async with showcase.session(pathlib.Path('.')) as (
      metadata,
      address,
      identity,
    ):
      [record] = metadata.snapshot(
        daemon_pb2.WatchWorkspacesRequest()
      ).workspaces
      assert record.session_id == identity
      assert record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
      assert await remote_rpc_client.list_remote_sessions(
        address, secure=False
      ) == [identity]
      assert metadata.prompt_history(identity) == []
  assert not (tmp_path / '.mirror').exists()


@pytest.mark.asyncio
async def test_showcase_worker_reads_private_startup_frame(
  tmp_path: pathlib.Path,
) -> None:
  address = 'unix:showcase.sock'
  startup = daemon_pb2.WorkerStartup(rpc_address=address).SerializeToString()
  worker = await asyncio.create_subprocess_exec(
    sys.executable,
    '-m',
    'mirror.cli.showcase',
    '--config-stdin',
    cwd=tmp_path,
    env={'HOME': str(tmp_path), 'PYTHONPATH': os.environ.get('PYTHONPATH', '')},
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  try:
    assert worker.stdin is not None
    assert worker.stdout is not None
    assert worker.stderr is not None
    worker.stdin.write(len(startup).to_bytes(4, 'big') + startup)
    await worker.stdin.drain()
    announced = await worker.stdout.readline()
    assert announced, (await worker.stderr.read()).decode()
    assert announced == f'MIRROR_RPC_ADDRESS={address}\n'.encode()
    assert (tmp_path / 'showcase.sock').is_socket()
  finally:
    if worker.returncode is None:
      worker.terminate()
    await worker.communicate()
  assert worker.returncode == 0
