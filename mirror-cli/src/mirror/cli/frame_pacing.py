"""Pace full-screen frames by terminal acknowledgement, not by write completion.

A write to the terminal returns once the pty, SSH, or multiplexer buffers
accept the bytes, long before the terminal has drawn them. Redrawing on every
input then queues one frame per wheel tick behind the terminal, and the screen
replays each intermediate scroll position long after input stops.

Each frame is followed by a cursor position request; terminals answer in
order, after drawing the output before it. Drawing continues while the oldest
unanswered frame is no older than the measured round trip plus some slack, so
link latency alone does not cap the frame rate, but a terminal that falls
behind receives no further frames until it catches up. The next frame then
paints the newest state.
"""

from __future__ import annotations

import asyncio
import collections
import time
from collections.abc import Callable

FRAME_QUERY = '\x1b[6n'
"""Cursor position request; its reply follows all preceding output."""


class FramePacer:
  """Bound how long drawn frames may wait in front of the terminal.

  Args:
    redraw: Schedules a redraw of the latest state after a deferred frame.
    abandoned: Called when outstanding queries are given up on.
    timeout: Seconds to wait for an acknowledgement before releasing frames,
      so a lost reply or a slow link can stall drawing only briefly.
    slack: Seconds of queueing, beyond the measured round trip, that the
      oldest unacknowledged frame may accumulate before drawing pauses.
    max_in_flight: Upper bound on unacknowledged frames.
    unanswered_limit: Consecutive timeouts, with no reply ever received,
      after which the terminal is treated as not answering requests and
      pacing is disabled.
    clock: Monotonic time source in seconds.
  """

  def __init__(
    self,
    redraw: Callable[[], None],
    *,
    abandoned: Callable[[], None] = lambda: None,
    timeout: float = 0.25,
    slack: float = 0.02,
    max_in_flight: int = 8,
    unanswered_limit: int = 3,
    clock: Callable[[], float] = time.monotonic,
  ) -> None:
    self._redraw = redraw
    self._abandoned = abandoned
    self._timeout = timeout
    self._slack = slack
    self._max_in_flight = max_in_flight
    self._unanswered_limit = unanswered_limit
    self._clock = clock
    self._loop: asyncio.AbstractEventLoop | None = None
    self._timer: asyncio.TimerHandle | None = None
    self._sent_at: collections.deque[float] = collections.deque()
    # Recent acknowledgement latencies; their minimum is the unqueued round trip.
    self._latencies: collections.deque[float] = collections.deque(maxlen=32)
    # Send times of queries abandoned by a timeout or reset; their replies may
    # still arrive, until they are older than the timeout and presumed lost.
    self._stale: collections.deque[float] = collections.deque()
    self._deferred = False
    self._answered = False
    self._unanswered = 0
    self.enabled = True

  def defer(self) -> bool:
    """Return whether to skip this redraw, coalescing it into a later frame."""
    if not self._sent_at:
      return False
    if len(self._sent_at) < self._max_in_flight and self._latencies:
      age = self._clock() - self._sent_at[0]
      if age <= min(self._latencies) + self._slack:
        return False
    self._deferred = True
    return True

  def sent(self, loop: asyncio.AbstractEventLoop) -> None:
    """Record a frame that was drawn and followed by `FRAME_QUERY`."""
    self._loop = loop
    self._sent_at.append(self._clock())
    if self._timer is None:
      self._timer = loop.call_later(self._timeout, self._expire)

  def acknowledge(self) -> None:
    """Accept the terminal's reply to the oldest outstanding frame query."""
    now = self._clock()
    while self._stale and now - self._stale[0] > self._timeout:
      self._stale.popleft()
    if self._stale:
      self._stale.popleft()
      return
    if not self._sent_at:
      return
    self._answered = True
    self._unanswered = 0
    self._latencies.append(now - self._sent_at.popleft())
    if self._timer is not None:
      self._timer.cancel()
      self._timer = None
    if self._sent_at and self._loop is not None:
      remaining = self._timeout - (now - self._sent_at[0])
      self._timer = self._loop.call_later(max(0.0, remaining), self._expire)
    self._release()

  def _expire(self) -> None:
    self._timer = None
    if not self._answered:
      self._unanswered += 1
      if self._unanswered >= self._unanswered_limit:
        self.enabled = False
    self._stale.extend(self._sent_at)
    self._sent_at.clear()
    self._abandoned()
    self._release()

  def _release(self) -> None:
    if self._deferred:
      self._deferred = False
      self._redraw()

  def reset(self) -> None:
    """Forget outstanding frames when terminal ownership or the screen changes."""
    if self._timer is not None:
      self._timer.cancel()
      self._timer = None
    self._stale.extend(self._sent_at)
    self._sent_at.clear()
    self._deferred = False
