from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pathlib
import time
import uuid
from collections.abc import Sequence
from typing import cast

from mirror.cli.statsig.events import constants


def save(directory: pathlib.Path, events: Sequence[bytes]) -> None:
  events = events[-constants.MAX_UNSENT_EVENTS :]
  if not events:
    return
  directory.mkdir(mode=0o700, parents=True, exist_ok=True)
  name = uuid.uuid4().hex
  partial = directory / f'{name}.tmp'
  descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  with os.fdopen(descriptor, 'wb') as file:
    file.write(b'\n'.join(events) + b'\n')
  os.replace(partial, directory / f'{name}.jsonl')


def load(directory: pathlib.Path) -> list[bytes]:
  if not directory.is_dir():
    return []
  cutoff_ms = time.time_ns() // 1_000_000 - constants.UNSENT_EVENT_MAX_AGE_MS
  events: list[tuple[int, bytes]] = []
  lock_path = directory / 'load.lock'
  with lock_path.open('a') as lock:
    # A concurrent load waits here, so each saved event is resent only once.
    # The OS releases the lock when the file closes or the process dies.
    fcntl.flock(lock, fcntl.LOCK_EX)
    for path in directory.iterdir():
      if path.suffix == '.jsonl':
        events.extend(_read_and_delete(path, cutoff_ms))
      elif path != lock_path:
        _delete_if_abandoned(path, cutoff_ms)
  events.sort(key=lambda timed: timed[0])
  return [event for _, event in events[-constants.MAX_UNSENT_EVENTS :]]


def _read_and_delete(
  path: pathlib.Path, cutoff_ms: int
) -> list[tuple[int, bytes]]:
  try:
    lines = path.read_bytes().splitlines()
  finally:
    path.unlink(missing_ok=True)
  events: list[tuple[int, bytes]] = []
  for line in lines:
    timestamp = _timestamp(line)
    if timestamp is not None and timestamp >= cutoff_ms:
      events.append((timestamp, line))
  return events


def _timestamp(line: bytes) -> int | None:
  try:
    request = cast(object, json.loads(line))
  except ValueError:
    return None
  match request:
    case {'events': [{'time': int() as timestamp}]}:
      return timestamp
    case _:
      return None


def _delete_if_abandoned(path: pathlib.Path, cutoff_ms: int) -> None:
  with contextlib.suppress(OSError):
    if path.stat().st_mtime_ns // 1_000_000 < cutoff_ms:
      path.unlink()
