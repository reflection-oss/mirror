"""Failure state and task cancellation for one extension in one workspace."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class Boundary:
  """Disable a failed extension without disabling its replacement."""

  def __init__(
    self,
    on_error: Callable[[Exception], None],
    is_current: Callable[[], bool] = lambda: True,
  ) -> None:
    self.error: str | None = None
    self._on_error = on_error
    self._is_current = is_current
    self._tasks: set[asyncio.Task[object]] = set()

  def active(self) -> bool:
    return self.error is None and self._is_current()

  def fail(self, error: Exception) -> None:
    """Disable this instance and cancel its other callbacks."""
    if self.active():
      self.error = f'{type(error).__name__}: {error}'
      for task in self._tasks:
        if task is not asyncio.current_task():
          task.cancel()
      self._on_error(error)

  def call[**P, T](
    self,
    callback: Callable[P, T],
    fallback: T,
    *args: P.args,
    **kwargs: P.kwargs,
  ) -> T:
    try:
      if self.active():
        return callback(*args, **kwargs)
    except (
      Exception
    ) as error:  # Extension callbacks must not escape into the CLI.
      self.fail(error)
    return fallback

  def spawn[T](
    self, callback: Callable[[], Awaitable[T]], fallback: T
  ) -> asyncio.Task[T]:
    async def run() -> T:
      try:
        if self.active():
          return await callback()
      except Exception as error:  # Cancellation is not an extension fault.
        self.fail(error)
      return fallback

    task = asyncio.create_task(run())
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)
    return task

  async def call_async[T](
    self, callback: Callable[[], Awaitable[T]], fallback: T
  ) -> T:
    try:
      return await self.spawn(callback, fallback)
    except asyncio.CancelledError:
      caller = asyncio.current_task()
      if self.active() or (caller is not None and caller.cancelling()):
        raise
      return fallback

  async def cancel(self) -> None:
    tasks = tuple(self._tasks)
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
