"""Conversation visibility and settled rendering through the real viewport."""

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit import formatted_text
from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import controls

from mirror.cli import (
  actions,
  rendering,
  tool_groups,
  transcript,
  transcript_view,
)


def _source() -> transcript.Transcript:
  return transcript.Transcript(
    view_mode=lambda: actions.ViewMode.TRANSCRIPT,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )


def _answer(identity: str, text: str, sequence: int) -> agent_pb2.AgentMessage:
  return agent_pb2.AgentMessage(
    id=identity,
    sequence=sequence,
    chat=agent_pb2.ChatRecord(
      assistant=agent_pb2.AssistantMessage(content=text), success=True
    ),
  )


def _text(content: controls.UIContent) -> str:
  return '\n'.join(
    ''.join(fragment[1] for fragment in content.get_line(index))
    for index in range(content.line_count)
  )


def test_viewport_can_reach_history_older_than_the_visible_tail() -> None:
  source = _source()
  source.extend(_answer(str(i), f'answer {i}', i + 1) for i in range(1500))
  view = transcript_view.TranscriptView(source, lambda: ())
  assert 'answer 1499' in _text(view.create_content(80, 8))
  view.scroll(-100000)
  assert 'answer 0' in _text(view.create_content(80, 8))


def test_new_output_preserves_reading_position() -> None:
  source = _source()
  source.extend(_answer(str(i), f'answer {i}', i + 1) for i in range(8))
  view = transcript_view.TranscriptView(source, lambda: ())
  view.create_content(80, 8)
  view.scroll(-100)
  before = _text(view.create_content(80, 8))
  source.append(_answer('new', 'new output', 9))
  assert _text(view.create_content(80, 8)) == before
  view.follow()
  assert 'new output' in _text(view.create_content(80, 8))


def test_switching_conversations_never_retains_other_conversation_rows() -> (
  None
):
  source = _source()
  source.append(_answer('alpha', 'ALPHA ONLY', 1))
  view = transcript_view.TranscriptView(source, lambda: ())
  assert 'ALPHA ONLY' in _text(view.create_content(80, 8))
  view.clear()
  source.clear_history()
  source.append(_answer('beta', 'BETA ONLY', 1))
  visible = _text(view.create_content(80, 8))
  assert 'BETA ONLY' in visible
  assert 'ALPHA' not in visible


def test_theme_change_repaints_cached_rows_without_moving_the_reading_position() -> (
  None
):
  source = _source()
  source.append(
    agent_pb2.AgentMessage(
      id='user',
      chat=agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content='Keep me')),
    )
  )
  selected_theme = 'dark'
  view = transcript_view.TranscriptView(
    source,
    lambda: (),
    theme_name=lambda: selected_theme,
  )
  view.create_content(80, 8)
  view.first()
  dark = view.create_content(80, 8)
  before = _text(dark)
  assert 'bg:#282828' in dark.get_line(0)[0][0]

  selected_theme = 'light'
  light = view.create_content(80, 8)
  assert _text(light) == before
  assert 'bg:#eeeeee' in light.get_line(0)[0][0]
  assert view.reading


def test_settled_blocks_are_reused_when_output_is_appended() -> None:
  source = _source()
  source.append(_answer('old', '**unchanged**', 1))
  settled = source.blocks()[0]
  source.append(_answer('new', 'new output', 2))
  assert source.blocks()[0] is settled


def test_replay_overlap_is_deduplicated_without_mutating_the_record() -> None:
  source = _source()
  entry = _answer('saved', 'saved output', 1)
  source.append_replayed(entry)
  source.adopt_replayed()
  assert source.append(entry) is False
  assert source.entries() == (entry,)


def test_empty_conversation_does_not_show_previous_history() -> None:
  source = _source()
  source.append(_answer('old', 'previous output', 1))
  view = transcript_view.TranscriptView(source, lambda: ())
  view.create_content(80, 8)
  source.clear_history()
  assert _text(view.create_content(80, 8)) == ''


def test_unanswered_calls_remain_visible_until_their_result_arrives() -> None:
  source = _source()
  call = agent_pb2.ToolCall(
    id='call',
    type='function',
    function=agent_pb2.FunctionCall(name='bash', arguments='{"command":"pwd"}'),
  )
  source.append(
    agent_pb2.AgentMessage(
      id='assistant',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(tool_calls=[call])
      ),
    )
  )
  view = transcript_view.TranscriptView(source, lambda: ())
  assert 'pwd' in _text(view.create_content(80, 20))
  source.append(
    agent_pb2.AgentMessage(
      id='result',
      sequence=2,
      chat=agent_pb2.ChatRecord(
        tool=agent_pb2.ToolMessage(tool_call_id='call', content='RESULT'),
        success=True,
      ),
    )
  )
  visible = _text(view.create_content(80, 20))
  assert 'RESULT' in visible
  assert not any(
    block.key.startswith('unanswered:') for block in source.blocks()
  )


def test_finalizing_a_stream_keeps_the_current_reading_line() -> None:
  source = _source()
  source.append(_answer('old', 'OLD HISTORY', 1))
  answer = '\n\n'.join(f'Paragraph {i}' for i in range(40))
  live = [rendering.assistant_render(answer, hyperlinks=False)]
  view = transcript_view.TranscriptView(source, lambda: live)
  view.create_content(80, 8)
  view.scroll(-10)
  before = _text(view.create_content(80, 8))
  source.append(_answer('final', answer, 2))
  view.settle_live('final')
  live.clear()
  assert _text(view.create_content(80, 8)) == before
  assert view.reading


def test_group_break_survives_an_intervening_invisible_record() -> None:
  source = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.COMPACT,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  for index in range(2):
    if index:
      source.break_group()
      source.append(
        agent_pb2.AgentMessage(context_window=agent_pb2.ContextWindowRecord())
      )
    source.append(
      agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(
            tool_calls=[
              agent_pb2.ToolCall(
                id=str(index),
                type='function',
                function=agent_pb2.FunctionCall(
                  name='read',
                  arguments='{"file_path":"example.py"}',
                ),
              )
            ]
          ),
        )
      )
    )
    source.append(
      agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(
          tool=agent_pb2.ToolMessage(
            tool_call_id=str(index), content='example'
          ),
          success=True,
        )
      )
    )
  groups = [
    block
    for block in source.blocks()
    if isinstance(block.renderable, tool_groups.ToolGroup)
  ]
  assert len(groups) == 2
  actual = rendering.render_blocks(
    [block.renderable for block in source.blocks()], 80
  )
  expected = source.render_history([], [], console_width=80, scrollback_limit=0)
  assert actual == expected.rstrip()


def test_viewport_keeps_character_styles_when_coalescing_ansi_fragments() -> (
  None
):
  source = _source()
  source.append(_answer('styled', 'plain **bold** and `code`', 1))
  view = transcript_view.TranscriptView(source, lambda: ())
  content = view.create_content(80, 20)
  expected = formatted_text.to_formatted_text(
    ansi.ANSI(
      rendering.renderable_to_ansi(
        rendering.assistant_render(
          'plain **bold** and `code`', hyperlinks=False
        ),
        80,
      ).rstrip()
    )
  )
  expected_rows = list(formatted_text.split_lines(expected))
  for index, row in enumerate(expected_rows):
    assert [
      (style, character)
      for style, text, *_ in content.get_line(index)
      for character in text
    ] == [(style, character) for style, text, *_ in row for character in text]


def test_user_header_sticks_after_crossing_the_top_and_changes_by_turn() -> (
  None
):
  source = _source()
  source.extend([
    agent_pb2.AgentMessage(
      id='first-user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='FIRST USER')
      ),
    ),
    _answer(
      'first-answer', '\n\n'.join(f'First paragraph {i}' for i in range(12)), 2
    ),
    agent_pb2.AgentMessage(
      id='second-user',
      sequence=3,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='SECOND USER')
      ),
    ),
    _answer(
      'second-answer',
      '\n\n'.join(f'Second paragraph {i}' for i in range(12)),
      4,
    ),
  ])
  view = transcript_view.TranscriptView(source, lambda: ())
  view.create_content(80, 12)
  view.first()
  top = _text(view.create_content(80, 12))
  assert top.count('FIRST USER') == 1
  assert 'SECOND USER' not in top
  view.scroll(10)
  first_section = _text(view.create_content(80, 12))
  assert 'FIRST USER' in '\n'.join(first_section.splitlines()[:3])
  assert first_section.count('FIRST USER') == 1
  view.scroll(35)
  second_section = _text(view.create_content(80, 12))
  assert 'SECOND USER' in '\n'.join(second_section.splitlines()[:3])
  assert 'FIRST USER' not in second_section
  view.scroll(-35)
  assert 'FIRST USER' in '\n'.join(
    _text(view.create_content(80, 12)).splitlines()[:3]
  )
  view.follow()
  second_section = _text(view.create_content(80, 12))
  assert 'SECOND USER' in '\n'.join(second_section.splitlines()[:3])
  assert 'FIRST USER' not in second_section
  narrow = _text(view.create_content(24, 6))
  assert 'SECOND USER' in '\n'.join(narrow.splitlines()[:2])
  view.first()
  assert 'FIRST USER' in '\n'.join(
    _text(view.create_content(24, 6)).splitlines()[:3]
  )


@pytest.mark.parametrize('final_width', [80, 24])
def test_finalization_retains_the_occurrence_of_repeated_prose_after_resize(
  final_width: int,
) -> None:
  source = _source()
  answer = '\n\n'.join(f'Repeated paragraph.\n\nContext {i}' for i in range(50))
  live = [rendering.assistant_render(answer, hyperlinks=False)]
  view = transcript_view.TranscriptView(source, lambda: live)
  view.create_content(80, 8)
  view.scroll(-13)
  before = _text(view.create_content(80, 8))
  assert 'Repeated paragraph.' in before.splitlines()[0]
  context = next(
    line.strip() for line in before.splitlines() if 'Context' in line
  )
  source.append(_answer('final', answer, 1))
  view.settle_live('final')
  live.clear()
  after = _text(view.create_content(final_width, 8))
  assert context in after
  assert 'Context 0\n' not in after


def test_sticky_header_keeps_the_middle_and_end_of_a_long_user_message_reachable() -> (
  None
):
  source = _source()
  source.append(
    agent_pb2.AgentMessage(
      id='long-user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(
          content='\n'.join(f'USER ROW {i}' for i in range(50))
        )
      ),
    )
  )
  source.append(_answer('answer', 'ASSISTANT ANSWER', 2))
  view = transcript_view.TranscriptView(source, lambda: ())
  view.create_content(40, 12)
  view.first()
  view.create_content(40, 12)
  view.scroll(1)
  assert 'USER ROW 8' in _text(view.create_content(40, 12))
  view.scroll(19)
  assert 'USER ROW 20' in _text(view.create_content(40, 12))
  view.scroll(20)
  assert 'USER ROW 40' in _text(view.create_content(40, 12))
  view.follow()
  assert 'ASSISTANT ANSWER' in _text(view.create_content(40, 12))


def test_paging_with_a_sticky_header_does_not_skip_conversation_rows() -> None:
  source = _source()
  source.append(
    agent_pb2.AgentMessage(
      id='user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='USER HEADER')
      ),
    )
  )
  source.append(
    _answer(
      'answer',
      '```text\n' + '\n'.join(f'ROW {i:03d}' for i in range(60)) + '\n```',
      2,
    )
  )
  view = transcript_view.TranscriptView(source, lambda: ())
  view.create_content(80, 12)
  view.first()
  seen = _text(view.create_content(80, 12))
  for _ in range(10):
    view.page(1)
    seen += '\n' + _text(view.create_content(80, 12))
  assert all(f'ROW {i:03d}' in seen for i in range(60)), seen


def test_resizing_keeps_the_reading_content_when_prose_rewraps() -> None:
  source = _source()
  source.append(
    _answer(
      'prose',
      '\n\n'.join(
        f'Paragraph {i:03d}: '
        + 'some text that wraps at different terminal widths ' * 3
        for i in range(80)
      ),
      1,
    )
  )
  view = transcript_view.TranscriptView(source, lambda: ())
  view.create_content(80, 12)
  view.scroll(-60)
  before = _text(view.create_content(80, 12))
  paragraph = next(
    line.strip().split(':')[0]
    for line in before.splitlines()
    if 'Paragraph' in line
  )
  after = _text(view.create_content(40, 12))
  assert paragraph in after
