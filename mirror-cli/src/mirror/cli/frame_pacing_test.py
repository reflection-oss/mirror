import asyncio

import pytest

from mirror.cli import frame_pacing


@pytest.mark.asyncio
async def test_frame_pacer_coalesces_redraws_until_acknowledged() -> None:
  redraws: list[None] = []
  pacer = frame_pacing.FramePacer(lambda: redraws.append(None), timeout=60)
  assert not pacer.defer()
  pacer.sent(asyncio.get_running_loop())
  assert [pacer.defer() for _ in range(300)] == [True] * 300
  assert redraws == []
  pacer.acknowledge()
  assert redraws == [None], 'One redraw paints the newest state'
  assert not pacer.defer()
  pacer.acknowledge()
  assert redraws == [None], 'Unsolicited replies do not redraw'


@pytest.mark.asyncio
async def test_frame_pacer_releases_and_disables_for_silent_terminals() -> None:
  released = asyncio.Event()
  abandoned: list[None] = []
  pacer = frame_pacing.FramePacer(
    released.set,
    abandoned=lambda: abandoned.append(None),
    timeout=0,
    unanswered_limit=2,
  )
  loop = asyncio.get_running_loop()
  pacer.sent(loop)
  assert pacer.defer()
  await asyncio.wait_for(released.wait(), 1)
  assert pacer.enabled
  released.clear()
  pacer.sent(loop)
  assert pacer.defer()
  await asyncio.wait_for(released.wait(), 1)
  assert not pacer.enabled
  assert abandoned == [None, None]


@pytest.mark.asyncio
async def test_frame_pacer_keeps_pacing_slow_links_that_have_answered() -> None:
  now = 0.0
  released = asyncio.Event()
  pacer = frame_pacing.FramePacer(
    released.set, timeout=0, unanswered_limit=1, clock=lambda: now
  )
  loop = asyncio.get_running_loop()
  pacer.sent(loop)
  pacer.acknowledge()
  pacer.sent(loop)
  now = 1.0
  assert pacer.defer()
  await asyncio.wait_for(released.wait(), 1)
  assert pacer.enabled


@pytest.mark.asyncio
async def test_frame_pacer_ignores_late_replies_to_abandoned_frames() -> None:
  now = 0.0
  pacer = frame_pacing.FramePacer(lambda: None, timeout=60, clock=lambda: now)
  loop = asyncio.get_running_loop()
  pacer.sent(loop)
  pacer.acknowledge()
  pacer.sent(loop)
  pacer.sent(loop)
  pacer.reset()  # A handoff abandons both queries; the terminal still answers.
  pacer.sent(loop)
  now = 1.0
  pacer.acknowledge()
  pacer.acknowledge()
  assert pacer.defer(), 'Late replies must not acknowledge the newer frame'
  pacer.acknowledge()
  assert not pacer.defer()


@pytest.mark.asyncio
async def test_frame_pacer_pipelines_frames_within_the_round_trip() -> None:
  now = 0.0
  redraws: list[None] = []
  pacer = frame_pacing.FramePacer(
    lambda: redraws.append(None), timeout=60, slack=0.01, clock=lambda: now
  )
  loop = asyncio.get_running_loop()
  pacer.sent(loop)
  now = 0.05
  pacer.acknowledge()  # A 50 ms link with no queueing.
  pacer.sent(loop)
  now = 0.08
  assert not pacer.defer(), 'Latency alone must not cap the frame rate'
  pacer.sent(loop)
  now = 0.12
  assert pacer.defer(), 'A frame queued beyond the round trip pauses drawing'
  now = 0.13
  pacer.acknowledge()
  assert redraws == [None], 'Each reply retries the newest deferred state'


@pytest.mark.asyncio
async def test_frame_pacer_recovers_when_an_abandoned_reply_never_arrives() -> (
  None
):
  now = 0.0
  pacer = frame_pacing.FramePacer(lambda: None, timeout=60, clock=lambda: now)
  loop = asyncio.get_running_loop()
  pacer.sent(loop)
  pacer.acknowledge()
  pacer.sent(loop)
  now = 60.0
  pacer.reset()  # The terminal never answers this query.
  pacer.sent(loop)
  now = 61.0
  pacer.acknowledge()
  assert not pacer.defer(), 'A lost reply must not stall the next frame'
