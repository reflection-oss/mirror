from __future__ import annotations

import asyncio
import functools
from typing import Never

import pytest

from mirror.cli import extension_faults


def _broken() -> Never:
  raise ValueError('broken extension')


def test_callbacks_share_failure_state_and_report_once() -> None:
  errors: list[Exception] = []
  calls: list[str] = []
  boundary = extension_faults.Boundary(errors.append)
  deferred = functools.partial(
    boundary.call, lambda: calls.append('called'), None
  )
  assert boundary.call(_broken, 'fallback') == 'fallback'
  deferred()
  boundary.call(_broken, None)
  assert len(errors) == 1
  assert boundary.error == 'ValueError: broken extension'
  assert calls == []


def test_retired_callbacks_do_not_run_or_report_faults() -> None:
  errors: list[Exception] = []
  current = True
  boundary = extension_faults.Boundary(errors.append, lambda: current)
  deferred = functools.partial(boundary.call, _broken, 'unavailable')
  current = False
  assert deferred() == 'unavailable'
  assert errors == []


@pytest.mark.asyncio
async def test_fault_cancels_sibling_work_but_preserves_the_failure_result() -> (
  None
):
  errors: list[Exception] = []
  started = asyncio.Event()
  stopped = asyncio.Event()
  boundary = extension_faults.Boundary(errors.append)

  async def pending() -> None:
    started.set()
    try:
      await asyncio.Event().wait()
    finally:
      stopped.set()

  task = boundary.spawn(pending, None)
  try:
    await started.wait()
    assert await boundary.call_async(_broken, 'failed') == 'failed'
    await stopped.wait()
    assert task.cancelled()
    assert len(errors) == 1
  finally:
    await boundary.cancel()


@pytest.mark.asyncio
async def test_async_exceptions_after_suspension_disable_the_boundary() -> None:
  errors: list[Exception] = []
  boundary = extension_faults.Boundary(errors.append)

  async def broken() -> Never:
    await asyncio.sleep(0)
    raise ValueError('after await')

  assert await boundary.call_async(broken, 'failed') == 'failed'
  assert not boundary.active()
  assert len(errors) == 1


@pytest.mark.asyncio
async def test_caller_cancellation_is_not_an_extension_fault() -> None:
  errors: list[Exception] = []
  started = asyncio.Event()
  boundary = extension_faults.Boundary(errors.append)

  async def pending() -> None:
    started.set()
    await asyncio.Event().wait()

  task = asyncio.create_task(boundary.call_async(pending, None))
  await started.wait()
  task.cancel()
  with pytest.raises(asyncio.CancelledError):
    await task
  assert boundary.active()
  assert errors == []


@pytest.mark.asyncio
async def test_retiring_callback_does_not_cancel_its_waiting_caller() -> None:
  errors: list[Exception] = []
  current = True
  started = asyncio.Event()
  boundary = extension_faults.Boundary(errors.append, lambda: current)

  async def pending() -> str:
    started.set()
    await asyncio.Event().wait()
    return 'completed'

  task = asyncio.create_task(boundary.call_async(pending, 'unavailable'))
  await started.wait()
  current = False
  await boundary.cancel()

  assert await task == 'unavailable'
  assert errors == []


@pytest.mark.asyncio
async def test_retired_task_does_not_construct_its_coroutine() -> None:
  errors: list[Exception] = []
  current = True
  boundary = extension_faults.Boundary(errors.append, lambda: current)
  task = boundary.spawn(_broken, 'unavailable')
  current = False
  assert await task == 'unavailable'
  assert errors == []
