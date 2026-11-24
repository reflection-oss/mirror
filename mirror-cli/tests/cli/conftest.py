"""Fixtures that run the CLI against an in-process local model service."""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator

import grpc.aio
import pytest
import pytest_asyncio
from mirror.protocol import (
  local_models_pb2_grpc,
)

from mirror.cli import local_models_client
from tests.cli import local_models_service


@pytest.fixture
def local_service(tmp_path: pathlib.Path) -> local_models_service.LocalService:
  return local_models_service.LocalService(tmp_path)


@pytest_asyncio.fixture
async def runtime(
  local_service: local_models_service.LocalService,
) -> AsyncIterator[local_models_client.Client]:
  server = grpc.aio.server()
  local_models_pb2_grpc.add_LocalModelsServiceServicer_to_server(
    local_service, server
  )
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  try:
    async with local_models_client.Client(f'127.0.0.1:{port}') as client:
      yield client
  finally:
    await server.stop(0)
