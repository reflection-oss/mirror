"""Incremental rich streaming preserves text and settled Markdown blocks."""

import pytest

from mirror.cli import rendering


def _lines(value: str) -> list[str]:
  return [
    line.rstrip()
    for line in rendering.strip_ansi(value).splitlines()
    if line.strip()
  ]


@pytest.mark.parametrize(
  'source',
  [
    'First paragraph.\n\nSecond **paragraph**.',
    '# Heading\n\n- one\n- two\n\nFollowing paragraph.',
    '```python\nx = 1\n```\n\nFollowing paragraph.',
    '| Name | Value |\n| --- | --- |\n| a | 123 |\n\nFollowing paragraph.',
    '[link][reference]\n\n[reference]: https://example.com',
  ],
)
def test_streaming_answer_matches_canonical_rich_text(source: str) -> None:
  stream = rendering.StreamingAnswer(())
  stream.update(source[: len(source) // 2])
  actual = rendering.render_blocks(stream.update(source), 80)
  expected = rendering.renderable_to_ansi(
    rendering.assistant_render(source, hyperlinks=False), 80
  )
  # A continuation has the same gutter without repeating the turn's icon.
  assert [
    line.removeprefix('● ').removeprefix('  ') for line in _lines(actual)
  ] == [line.removeprefix('● ').removeprefix('  ') for line in _lines(expected)]


def test_streaming_answer_reuses_completed_blocks() -> None:
  stream = rendering.StreamingAnswer(())
  settled = stream.update('First paragraph.\n\nSecond paragraph.')[0]
  assert (
    stream.update('First paragraph.\n\nSecond paragraph grows.')[0] is settled
  )


def test_streaming_answer_can_restart_after_an_interrupted_turn() -> None:
  stream = rendering.StreamingAnswer(())
  stream.update('Old conversation.\n\nOld tail.')
  visible = rendering.render_blocks(stream.update('New conversation.'), 80)
  assert 'New conversation.' in visible
  assert 'Old' not in visible


def test_finishing_a_table_reuses_its_already_rendered_preview() -> None:
  stream = rendering.StreamingAnswer(())
  table = '| Name | Value |\n| --- | --- |\n| a | 123 |'
  preview = stream.update(table)[0]
  rendering.render_blocks((preview,), 80)
  completed = stream.update(table + '\n\nFollowing paragraph.')
  assert completed[0] is preview
