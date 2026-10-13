"""Snapshot and subscription ordering through a real AgentRPC channel."""

import asyncio
from collections.abc import AsyncIterator
from typing import override

import grpc.aio
import pytest
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
)

from mirror.cli import actions, remote_rpc_client, rpc_client, transcript


class _Peer(agent_pb2_grpc.AgentRPCServicer):
  def __init__(self, *, incomplete: bool = False) -> None:
    self.attached = asyncio.Event()
    self.snapshotted = asyncio.Event()
    self.input_after_snapshot = asyncio.Event()
    self.incomplete = incomplete
    self.old = agent_pb2.AgentMessage(
      id='old',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(content='old output')
      ),
    )
    self.new = agent_pb2.AgentMessage(
      id='new',
      sequence=2,
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(content='new output')
      ),
    )

  @override
  async def GetMessages(
    self,
    request: agent_pb2.GetMessagesRequest,
    context: grpc.aio.ServicerContext[
      agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
    ],
  ) -> agent_pb2.GetMessagesResponse:
    del context
    assert self.attached.is_set()
    self.snapshotted.set()
    return agent_pb2.GetMessagesResponse(
      messages=[self.old] if request.offset == 0 else [],
      total=3 if self.incomplete else 1,
    )

  @override
  async def Session(
    self,
    request_iterator: AsyncIterator[agent_pb2.ClientMessage],
    context: grpc.aio.ServicerContext[
      agent_pb2.ClientMessage, agent_pb2.SessionMessage
    ],
  ) -> AsyncIterator[agent_pb2.SessionMessage]:
    del context
    async for request in request_iterator:
      if request.HasField('set_inference_config'):
        self.attached.set()
        yield agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id=request.message_id)
        )
        yield agent_pb2.SessionMessage(agent_message=self.old)
        yield agent_pb2.SessionMessage(agent_message=self.new)
      elif request.HasField('message'):
        assert self.snapshotted.is_set()
        self.input_after_snapshot.set()


@pytest.mark.asyncio
async def test_remote_replay_subscribes_before_snapshot_and_preserves_queued_input() -> (
  None
):
  peer = _Peer()
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(peer, server)  # pyright: ignore[reportUnknownMemberType] - protoc registration has no parameter annotations.
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  source = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.TRANSCRIPT,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  replaying = True

  def receive(response: agent_pb2.SessionMessage) -> None:
    nonlocal replaying
    if response.HasField('replay_done'):
      source.adopt_replayed()
      replaying = False
    elif response.HasField('agent_message'):
      if replaying:
        source.append_replayed(response.agent_message)
      else:
        source.append(response.agent_message)

  async def error(message: str) -> None:
    pytest.fail(message)

  client = rpc_client.Client(
    remote_rpc_client.Transport(f'127.0.0.1:{port}', secure=False),
    'conversation',
    on_message=receive,
    on_error=error,
    inference_config=agent_pb2.InferenceConfig(model='synthetic'),
  )
  client.enqueue(
    agent_pb2.AgentMessage(
      id='input',
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='queued input')
      ),
    )
  )
  try:
    await client.start(replay=True, wait_for_attachment=True)
    await asyncio.wait_for(peer.input_after_snapshot.wait(), timeout=2)
    assert source.entries() == (peer.old, peer.new)
  finally:
    await client.close()
    await server.stop(0)


@pytest.mark.asyncio
async def test_incomplete_remote_history_is_not_reported_as_complete() -> None:
  peer = _Peer(incomplete=True)
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(peer, server)  # pyright: ignore[reportUnknownMemberType] - protoc registration has no parameter annotations.
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()

  async def error(message: str) -> None:
    pytest.fail(message)

  client = rpc_client.Client(
    remote_rpc_client.Transport(f'127.0.0.1:{port}', secure=False),
    'conversation',
    on_message=lambda _: None,
    on_error=error,
    inference_config=agent_pb2.InferenceConfig(model='synthetic'),
  )
  try:
    with pytest.raises(RuntimeError, match='Incomplete conversation history'):
      await client.start(replay=True, wait_for_attachment=True)
  finally:
    await client.close()
    await server.stop(0)
