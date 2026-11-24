"""Tests for hosted ACP sessions over a fake local daemon."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import override

import grpc
import pytest
import pytest_asyncio
from mirror.protocol import (
  daemon_pb2,
  daemon_pb2_grpc,
  mirror_pb2,
)

from mirror.cli import acp_remote

_ENDPOINT = acp_remote.Endpoint('gateway.example:443')


class _Daemon(daemon_pb2_grpc.DaemonRPCServicer):
  """Records what a frontend registers and starts; knows one local workspace."""

  def __init__(self) -> None:
    self.workspaces = [
      daemon_pb2.Workspace(
        id='local-1', kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY
      ),
      daemon_pb2.Workspace(
        id='kept-1',
        session_id='kept-1',
        kind=daemon_pb2.WORKSPACE_KIND_REMOTE,
      ),
    ]
    self.created: list[daemon_pb2.CreateWorkspaceRequest] = []
    self.started: list[daemon_pb2.StartRemoteSessionRequest] = []
    self.attached: list[daemon_pb2.AttachWorkspaceRequest] = []

  @override
  def AttachWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.AttachWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.AttachWorkspaceResponse:
    del context
    self.attached.append(request)
    return daemon_pb2.AttachWorkspaceResponse(
      attachment_id='attachment',
      worker_instance_id='worker',
      workspace=daemon_pb2.Workspace(
        id=request.workspace_id, session_id=request.expected_session_id
      ),
    )

  @override
  def GetDaemonInfo(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.GetDaemonInfoRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.GetDaemonInfoResponse:
    del request, context
    return daemon_pb2.GetDaemonInfoResponse(supports_remote_sessions=True)

  @override
  def WatchWorkspaces(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.WatchWorkspacesRequest,
    context: grpc.ServicerContext,
  ) -> Iterator[daemon_pb2.WatchWorkspacesResponse]:
    del request, context
    yield daemon_pb2.WatchWorkspacesResponse(workspaces=self.workspaces)

  @override
  def CreateWorkspace(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.CreateWorkspaceRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.CreateWorkspaceResponse:
    del context
    self.created.append(request)
    return daemon_pb2.CreateWorkspaceResponse(
      workspace=daemon_pb2.Workspace(
        id=request.workspace_id,
        session_id=request.remote.session_id,
        kind=daemon_pb2.WORKSPACE_KIND_REMOTE,
      )
    )

  @override
  def StartRemoteSession(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.StartRemoteSessionRequest,
    context: grpc.ServicerContext,
  ) -> daemon_pb2.StartRemoteSessionResponse:
    del context
    self.started.append(request)
    return daemon_pb2.StartRemoteSessionResponse()


@pytest_asyncio.fixture
async def daemon(
  monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[_Daemon, str]]:
  for name in ('grpc_proxy', 'https_proxy', 'http_proxy'):
    monkeypatch.delenv(name, raising=False)
  servicer = _Daemon()
  with tempfile.TemporaryDirectory(prefix='mr-acp-') as root:
    address = f'unix:{root}/daemon.sock'
    server = grpc.aio.server()
    daemon_pb2_grpc.add_DaemonRPCServicer_to_server(servicer, server)  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
    server.add_insecure_port(address)
    await server.start()
    try:
      yield servicer, address
    finally:
      await server.stop(None)


@pytest.fixture
def clone(tmp_path: pathlib.Path) -> pathlib.Path:
  """An empty repository whose `origin` a server can clone."""
  repository = tmp_path / 'widgets'
  subprocess.run(
    ('git', 'init', '--initial-branch=main', str(repository)),
    check=True,
    capture_output=True,
  )
  subprocess.run(
    (
      'git',
      '-C',
      str(repository),
      'remote',
      'add',
      'origin',
      'git@github.com:acme/widgets.git',
    ),
    check=True,
    capture_output=True,
  )
  return repository


@pytest.mark.asyncio
async def test_transport_starts_a_session_in_the_directory_s_repository(
  daemon: tuple[_Daemon, str], clone: pathlib.Path
) -> None:
  servicer, address = daemon
  factory = acp_remote.transport([str(clone / 'mirror-agent')], _ENDPOINT)
  transport, session_id = await factory(
    {'daemon_address': address, 'create_session': True},
    clone,
    'unused',
    lambda message: None,
  )
  told_before_connecting = (list(servicer.created), list(servicer.started))

  await transport.connect()
  await transport.close()

  assert str(uuid.UUID(session_id)) == session_id
  assert told_before_connecting == ([], [])
  assert servicer.created == [
    daemon_pb2.CreateWorkspaceRequest(
      workspace_id=session_id,
      remote=daemon_pb2.RemoteSource(session_id=session_id),
    )
  ]
  assert servicer.started == [
    daemon_pb2.StartRemoteSessionRequest(
      remote_endpoint='https://gateway.example:443',
      request=mirror_pb2.StartSessionRequest(
        session_id=session_id,
        workspace=mirror_pb2.Workspace(
          name='widgets',
          repository_url='https://github.com/acme/widgets',
          revision='HEAD',
        ),
      ),
    )
  ]
  assert [
    (request.workspace_id, request.remote_endpoint)
    for request in servicer.attached
  ] == [(session_id, 'https://gateway.example:443')]


@pytest.mark.asyncio
async def test_transport_rejoins_a_kept_session_without_starting_one(
  daemon: tuple[_Daemon, str], tmp_path: pathlib.Path
) -> None:
  servicer, address = daemon
  factory = acp_remote.transport([str(tmp_path / 'mirror-agent')], _ENDPOINT)
  transport, session_id = await factory(
    {'daemon_address': address, 'create_session': False},
    tmp_path,
    'kept-1',
    lambda message: None,
  )

  await transport.connect()
  await transport.close()

  assert session_id == 'kept-1'
  assert (servicer.created, servicer.started) == ([], [])
  assert [request.workspace_id for request in servicer.attached] == ['kept-1']


@pytest.mark.asyncio
async def test_transport_refuses_a_session_id_bound_to_a_local_workspace(
  daemon: tuple[_Daemon, str], tmp_path: pathlib.Path
) -> None:
  servicer, address = daemon
  factory = acp_remote.transport([str(tmp_path / 'mirror-agent')], _ENDPOINT)
  transport, _ = await factory(
    {'daemon_address': address, 'create_session': False},
    tmp_path,
    'local-1',
    lambda message: None,
  )

  with pytest.raises(RuntimeError, match='local-1 is already registered'):
    await transport.connect()

  assert servicer.attached == []


@pytest.mark.asyncio
async def test_offered_models_are_empty_when_the_server_cannot_list_them(
  daemon: tuple[_Daemon, str],
) -> None:
  _, address = daemon
  daemon_without_model_list = acp_remote.Endpoint(address, secure=False)

  assert await acp_remote.offered_models(daemon_without_model_list) == ()
