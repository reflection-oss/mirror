from __future__ import annotations

import pytest
from mirror.protocol import agent_pb2

from mirror.cli import quotes


@pytest.mark.parametrize(
  ('text', 'expected'),
  [
    ('short', '▌short\u2060'),
    ('first\nsecond\nthird', '▌first… +2 lines\u2060'),
    ('x' * 50 + '\nnext', f'▌{"x" * 39}… +1 line\u2060'),
  ],
)
def test_label_summarizes_the_first_line(text: str, expected: str) -> None:
  assert quotes.label(text) == expected


def test_split_chip_separates_rail_preview_and_detail() -> None:
  assert quotes.split_chip('▌first… +2 lines\u2060\u2060') == (
    '▌',
    'first…',
    ' +2 lines',
  )


def test_add_reuses_identical_text_and_hides_label_collisions() -> None:
  first, table = quotes.add('same\nA', ())
  again, table = quotes.add('same\nA', table)
  other, table = quotes.add('same\nB', table)

  assert again == first
  assert (first.label, other.label) == (
    '▌same… +1 line\u2060',
    '▌same… +1 line\u2060\u2060',
  )
  assert table == (first, other)


def test_message_sends_chips_as_quoted_parts_and_keeps_other_text() -> None:
  quote, table = quotes.add('line one\nline two', ())

  message = quotes.message(f'Intro {quote.label} why? ▌other', table)

  assert message == agent_pb2.UserMessage(
    parts=[
      agent_pb2.UserMessagePart(text='Intro '),
      agent_pb2.UserMessagePart(quoted_selection='line one\nline two'),
      agent_pb2.UserMessagePart(text=' why? ▌other'),
    ]
  )


def test_message_without_chips_is_plain_content() -> None:
  assert quotes.message('just text', ()) == agent_pb2.UserMessage(
    content='just text'
  )


def test_display_round_trips_a_sent_message() -> None:
  quote, table = quotes.add('line one\nline two', ())
  text = f'Intro\n{quote.label} why?'

  display, found = quotes.display(quotes.message(text, table))

  assert (display, found) == (text, table)


def test_chip_before_includes_the_space_typed_after_it() -> None:
  quote, table = quotes.add('a', ())
  text = f'x {quote.label} '

  assert quotes.chip_before(text, len(text), table) == f'{quote.label} '
  assert quotes.chip_before(text, 2, table) is None


def test_chip_before_prefers_a_chip_over_the_chip_it_quotes() -> None:
  inner, table = quotes.add('bar', ())
  outer, table = quotes.add(inner.label, table)

  assert quotes.chip_before(outer.label, len(outer.label), table) == (
    outer.label
  )


def test_message_keeps_a_chip_apart_from_text_that_extends_its_label() -> None:
  short, table = quotes.add('return', ())
  long, table = quotes.add('return result', table)

  message = quotes.message(f'{short.label} result and {long.label}', table)

  assert long.label == '▌return result\u2060'
  assert message == agent_pb2.UserMessage(
    parts=[
      agent_pb2.UserMessagePart(quoted_selection='return'),
      agent_pb2.UserMessagePart(text=' result and '),
      agent_pb2.UserMessagePart(quoted_selection='return result'),
    ]
  )


def test_message_drops_the_end_of_a_chip_that_no_longer_resolves() -> None:
  assert quotes.message('▌gone\u2060 now', ()) == agent_pb2.UserMessage(
    content='▌gone now'
  )


@pytest.mark.parametrize(
  ('previous', 'expected'), [(2, 5), (5, 2)], ids=['right', 'left']
)
def test_step_over_end_moves_the_cursor_past_a_chip_end(
  previous: int, expected: int
) -> None:
  assert quotes.step_over_end('▌ab\u2060\u2060 ', 4, previous) == expected
