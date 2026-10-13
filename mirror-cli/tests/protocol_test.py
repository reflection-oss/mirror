"""One installed wheel supplies both the CLI and its generated protocol modules."""

from __future__ import annotations

import pathlib
from unittest import mock

import pytest
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
  client_pb2,
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror import distribution, protocol
from mirror.cli import extension, state

_PROTOCOL_PACKAGE = 'mirror.protocol'
_PROTOCOL_PATH = 'mirror/protocol'


def test_wheel_bundles_generated_modules_and_type_information() -> None:
  files = {str(path) for path in distribution.installed().files or ()}
  required = {
    f'{_PROTOCOL_PATH}/agent_pb2.py',
    f'{_PROTOCOL_PATH}/agent_pb2.pyi',
    f'{_PROTOCOL_PATH}/agent_pb2_grpc.py',
    f'{_PROTOCOL_PATH}/client_pb2.py',
    f'{_PROTOCOL_PATH}/client_pb2.pyi',
    f'{_PROTOCOL_PATH}/mirror_pb2.py',
    f'{_PROTOCOL_PATH}/mirror_pb2_grpc.py',
  }
  assert required <= files
  assert {path for path in files if path.endswith('/agent_pb2.py')} == {
    f'{_PROTOCOL_PATH}/agent_pb2.py'
  }
  assert protocol.agent_pb2 is agent_pb2
  assert protocol.client_pb2 is client_pb2
  assert protocol.agent_pb2_grpc is agent_pb2_grpc
  assert protocol.mirror_pb2 is mirror_pb2
  assert protocol.mirror_pb2_grpc is mirror_pb2_grpc
  assert agent_pb2.DESCRIPTOR.name == f'{_PROTOCOL_PATH}/agent.proto'
  assert client_pb2.DESCRIPTOR.dependencies[0] is agent_pb2.DESCRIPTOR
  assert mirror_pb2.DESCRIPTOR.package == 'mirror.v1'
  assert (
    agent_pb2_grpc.AgentRPCStub.__module__
    == f'{_PROTOCOL_PACKAGE}.agent_pb2_grpc'
  )
  assert (
    mirror_pb2_grpc.MirrorRPCStub.__module__
    == f'{_PROTOCOL_PACKAGE}.mirror_pb2_grpc'
  )


@pytest.mark.asyncio
async def test_external_extension_imports_the_installed_protocol(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / '.mirror' / 'extensions' / 'example'
  directory.mkdir(parents=True)
  (directory / 'extension.py').write_text("""from mirror.cli import extension
from mirror.protocol import agent_pb2

class Example(extension.Extension[agent_pb2.ExtensionRecord, None, None]):
  @classmethod
  def id(cls):
    return 'example'
  @staticmethod
  def initial_state():
    return agent_pb2.ExtensionRecord(extension='example', kind='snapshot')
""")
  registry = extension.ExtensionRegistry()
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'low', 'auto'),
    session=state.SessionState('session'),
    lifecycle=state.LifecycleState(active=True),
  )
  try:
    assert await registry.load(directory) == 'example'
    host = registry.create_host(
      extension.Context(
        extension.Workspace(tmp_path / 'workspace', remote=False),
        lambda _: None,
        lambda: current,
        lambda _: None,
        registry.refresh_tools,
        lambda *_: pytest.fail('Unexpected tool call'),
        transcript=tuple,
        builtins=tuple,
        extension_tools=tuple,
        workspace_controller=mock.Mock(),
        create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
      )
    )
    try:
      assert host.initial_states['example'] == agent_pb2.ExtensionRecord(
        extension='example', kind='snapshot'
      )
    finally:
      await host.close()
  finally:
    await registry.close()
