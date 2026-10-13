from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import pathlib
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from mirror.cli.statsig import config
from mirror.cli.statsig import constants as statsig_constants
from mirror.cli.statsig.events import client, constants, event

# Seconds to wait for the client to reach the fake endpoint.
_TIMEOUT = 5.0


@dataclasses.dataclass(frozen=True)
class _Request:
  headers: dict[str, str]
  body: Any


class _FakeStatsig:
  def __init__(self) -> None:
    self.requests: asyncio.Queue[_Request] = asyncio.Queue()
    # Statuses to answer with, in order; then 202, Statsig's success status.
    self.statuses: list[int] = []

  async def handle(
    self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
  ) -> None:
    head = await reader.readuntil(b'\r\n\r\n')
    headers: dict[str, str] = {}
    for line in head.decode().split('\r\n')[1:]:
      name, _, value = line.partition(':')
      headers[name.lower()] = value.strip()
    body = await reader.readexactly(int(headers['content-length']))
    status = self.statuses.pop(0) if self.statuses else 202
    await self.requests.put(_Request(headers, json.loads(body)))
    writer.write(
      f'HTTP/1.1 {status} X\r\nContent-Length: 0\r\n'
      'Connection: close\r\n\r\n'.encode()
    )
    await writer.drain()
    writer.close()

  async def next_request(self) -> _Request:
    return await asyncio.wait_for(self.requests.get(), _TIMEOUT)


@contextlib.asynccontextmanager
async def _fake_statsig(
  monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[_FakeStatsig]:
  fake = _FakeStatsig()
  server = await asyncio.start_server(fake.handle, '127.0.0.1', 0)
  port = server.sockets[0].getsockname()[1]
  monkeypatch.setattr(
    constants, 'STATSIG_ENDPOINT', f'http://127.0.0.1:{port}/v1/log_event'
  )
  monkeypatch.setattr(constants, 'RETRY_BACKOFF_PATTERN', (0.01,))
  async with server:
    yield fake


def _client(directory: pathlib.Path) -> client.EventClient:
  return client.EventClient(
    config.StatsigConfig(
      user_id='install-1',
      session_id='launch-1',
      unsent_events_directory=directory,
    )
  )


@pytest.mark.asyncio
async def test_log_event_sends_event_with_user_and_key(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  async with _fake_statsig(monkeypatch) as statsig, _client(tmp_path) as sender:
    await sender.enable()
    sender.log_event(
      event.Event('task_completed', value=3, metadata={'surface': 'cli'})
    )
    request = await statsig.next_request()

  assert (
    request.headers['statsig-api-key'] == statsig_constants.STATSIG_CLIENT_KEY
  )
  [sent] = request.body['events']
  assert sent['eventName'] == 'task_completed'
  assert sent['value'] == 3
  assert sent['metadata'] == {'surface': 'cli'}
  assert sent['user']['userID'] == 'install-1'
  assert sent['user']['custom']['sessionID'] == 'launch-1'
  assert (
    request.body['statsigMetadata']['sdkType'] == statsig_constants.SDK_TYPE
  )


@pytest.mark.asyncio
async def test_log_event_while_off_sends_nothing(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  async with _fake_statsig(monkeypatch) as statsig:
    async with _client(tmp_path) as sender:
      sender.log_event(event.Event('before_enable'))
      await sender.enable()
      await sender.disable()
      sender.log_event(event.Event('after_disable'))

    assert statsig.requests.empty()
  assert not list(tmp_path.glob('*.jsonl'))


@pytest.mark.asyncio
async def test_log_event_retries_retryable_status(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  async with _fake_statsig(monkeypatch) as statsig, _client(tmp_path) as sender:
    statsig.statuses = [503]
    await sender.enable()
    sender.log_event(event.Event('flaky'))
    first = await statsig.next_request()
    retry = await statsig.next_request()

  assert first.body == retry.body


@pytest.mark.asyncio
async def test_log_event_drops_rejected_event(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  async with _fake_statsig(monkeypatch) as statsig:
    async with _client(tmp_path) as sender:
      statsig.statuses = [400]
      await sender.enable()
      sender.log_event(event.Event('rejected'))
      await statsig.next_request()

    assert statsig.requests.empty()
  assert not list(tmp_path.glob('*.jsonl'))


@pytest.mark.asyncio
async def test_log_event_drops_invalid_event(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  async with _fake_statsig(monkeypatch) as statsig:
    async with _client(tmp_path) as sender:
      await sender.enable()
      with caplog.at_level(logging.WARNING):
        sender.log_event(event.Event('Not An Identifier'))

    assert statsig.requests.empty()
  assert 'Statsig event dropped: Invalid Statsig event name' in caplog.text


@pytest.mark.asyncio
async def test_enable_resends_events_saved_by_disable(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  async with _fake_statsig(monkeypatch) as statsig, _client(tmp_path) as sender:
    statsig.statuses = [503, 503]
    await sender.enable()
    sender.log_event(event.Event('offline'))
    await statsig.next_request()
    await statsig.next_request()
    await sender.disable()
    await sender.enable()
    resent = await statsig.next_request()

  assert resent.body['events'][0]['eventName'] == 'offline'
