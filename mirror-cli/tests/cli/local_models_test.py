"""The typed RPC client: state, reconnection and cancellation."""

import asyncio

import pytest
from google.protobuf import empty_pb2
from mirror.protocol import (
  local_models_pb2,
)

from mirror.cli import local_models_client
from tests.cli import local_models_service


@pytest.mark.asyncio
async def test_execute_applies_the_state_the_daemon_replies_with(
  runtime: local_models_client.Client,
) -> None:
  await runtime.execute(local_models_pb2.ExecuteRequest(context=32768))

  assert runtime.state.context == 32768


@pytest.mark.asyncio
async def test_closing_does_not_stop_the_daemon_and_calls_reconnect(
  runtime: local_models_client.Client,
) -> None:
  await runtime.execute(
    local_models_pb2.ExecuteRequest(start=empty_pb2.Empty())
  )
  await runtime.close()
  await runtime.execute(
    local_models_pb2.ExecuteRequest(status=empty_pb2.Empty())
  )
  assert runtime.state.phase == local_models_pb2.LOCAL_MODEL_SERVER_PHASE_READY


@pytest.mark.asyncio
async def test_cancelling_a_call_cancels_the_operation_on_the_daemon(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.ready.clear()
  call = asyncio.create_task(
    runtime.execute(
      local_models_pb2.ExecuteRequest(
        load=local_models_pb2.LocalModelLoad(model='synthetic')
      )
    )
  )
  await asyncio.wait_for(local_service.requested.wait(), 2)
  call.cancel()
  with pytest.raises(asyncio.CancelledError):
    await call
  await asyncio.wait_for(local_service.cancelled.wait(), 2)
