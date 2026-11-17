"""Deterministic work bounds for sticky headers, independent of machine speed."""

import cProfile

from rich import cells

from mirror.cli import rendering, transcript
from tests.cli import reference_transcript_view as transcript_view


def test_create_content_reuses_summary_during_redraw_and_navigation() -> None:
  message = 'word ' * 400_000
  live = ['answer\n' * 60]
  view = transcript_view.TranscriptView(
    lambda: (),
    lambda: live,
    pending=lambda: (
      transcript.Block('pending', (0, 0), rendering.user_text(message)),
    ),
  )
  first = view.create_content(80, 20).get_line(0)
  assert first[0][1].startswith('›  word ')

  with cProfile.Profile() as profile:
    view.create_content(80, 2)
    view.create_content(80, 20)
    view.scroll(-1)
    view.create_content(80, 20)
    view.page(1)
    last = view.create_content(80, 20).get_line(0)

  assert last == first
  rescans = [
    entry.code
    for entry in profile.getstats()
    if entry.code
    in (
      cells.split_graphemes.__code__,
      rendering.strip_ansi.__code__,
      "<method 'split' of 'str' objects>",
      "<method 'join' of 'str' objects>",
    )
  ]
  assert rescans == [], 'Redraws must not rescan or segment the user prompt'
