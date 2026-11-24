"""Conversation visibility and settled rendering through the real viewport."""

import asyncio
import base64
import contextlib
import cProfile
import io
import unicodedata
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import override

import freezegun
import pyte
import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit import (
  application,
  data_structures,
  formatted_text,
  mouse_events,
  utils,
)
from prompt_toolkit.input import defaults
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import base, color_depth, vt100
from rich import cells, markdown, text

from mirror.cli import (
  actions,
  hyperlinks,
  rendering,
  themes,
  tool_groups,
  transcript,
  workspace_application,
)
from tests.cli import reference_transcript_view as transcript_view


def _source(
  *entries: agent_pb2.AgentMessage,
  mode: actions.ViewMode = actions.ViewMode.DETAIL,
) -> transcript.Transcript:
  source = transcript.Transcript(
    view_mode=lambda: mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  source.extend(entries)
  return source


def _answer(
  identity: str, text: str, sequence: int = 1
) -> agent_pb2.AgentMessage:
  return agent_pb2.AgentMessage(
    id=identity,
    sequence=sequence,
    chat=agent_pb2.ChatRecord(
      assistant=agent_pb2.AssistantMessage(content=text), success=True
    ),
  )


def _text(content: controls.UIContent) -> str:
  return '\n'.join(
    formatted_text.fragment_list_to_text(content.get_line(index))
    for index in range(content.line_count)
  )


def _paint(
  view: transcript_view.TranscriptView,
  width: int,
  height: int,
  theme_name: str = 'dark',
) -> pyte.Screen:
  output = io.StringIO()
  with defaults.create_pipe_input() as pipe:
    app = application.Application[None](
      layout=layout.Layout(containers.Window(view, wrap_lines=False)),
      style=themes.prompt_style(theme_name),
      full_screen=True,
      color_depth=color_depth.ColorDepth.DEPTH_24_BIT,
      input=pipe,
      output=vt100.Vt100_Output(
        output,
        lambda: data_structures.Size(rows=height, columns=width),
        enable_cpr=False,
      ),
    )
    with application.set_app(app):
      app.renderer.render(app, app.layout)
  terminal = pyte.Screen(width, height)
  pyte.Stream(terminal).feed(output.getvalue())
  return terminal


def _user(content: str) -> agent_pb2.AgentMessage:
  return agent_pb2.AgentMessage(
    id='user',
    sequence=1,
    chat=agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content=content)),
  )


class _Output(base.DummyOutput):
  def __init__(self) -> None:
    self.raw: list[str] = []

  @override
  def write_raw(self, data: str) -> None:
    self.raw.append(data)


@pytest.fixture
def output() -> Iterator[_Output]:
  captured = _Output()
  with defaults.create_pipe_input() as pipe:
    app = application.Application[str | None](input=pipe, output=captured)
    with application.set_app(app):
      yield captured


def _mouse(
  view: transcript_view.TranscriptView,
  point: data_structures.Point,
  kind: mouse_events.MouseEventType = mouse_events.MouseEventType.MOUSE_MOVE,
) -> None:
  view.mouse_handler(
    mouse_events.MouseEvent(
      point,
      kind,
      mouse_events.MouseButton.NONE
      if kind == mouse_events.MouseEventType.MOUSE_MOVE
      else mouse_events.MouseButton.LEFT,
      frozenset(),
    )
  )


def _hover_code(
  view: transcript_view.TranscriptView, content: controls.UIContent
) -> data_structures.Point:
  lines = _text(content).splitlines()
  row = next(index for index, line in enumerate(lines) if '┃' in line)
  _mouse(view, data_structures.Point(lines[row].index('┃'), row))
  line = ''.join(part[1] for part in content.get_line(row + 1))
  return data_structures.Point(line.index('⧉'), row + 1)


@pytest.mark.parametrize('width', [10, 24, 80])
@pytest.mark.parametrize(
  ('markdown', 'expected'),
  [
    ('```python\n\tvalue = 1  \n\n```', '\tvalue = 1  \n\n'),
    ('~~~python title\nprint("```")\n~~~', 'print("```")\n'),
    ('> ```text\n>   indented\n> ```', '  indented\n'),
    ('- Example\n\n  ```text\n  nested\n  ```', 'nested\n'),
    ('Example:\n\n    indented\n    code', 'indented\ncode\n'),
    ('```text\npartial', 'partial'),
    ('```text\r\none\r\ntwo\r\n```', 'one\ntwo\n'),
    ('```text\n界🙂e\u0301\u00a0┃\n```', '界🙂e\u0301\u00a0┃\n'),
    (
      '```text\nliteral \\x1b[31m\n\x1b[31mred\x1b[0m\n```',
      'literal \\x1b[31m\nred\n',
    ),
    ('```text\n' + 'long line ' * 20 + '\n```', 'long line ' * 20 + '\n'),
  ],
)
def test_code_copy_preserves_code_body_independently_of_width(
  width: int,
  markdown: str,
  expected: str,
  output: _Output,
) -> None:
  source = _source()
  source.append(_answer('code', markdown, 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  content = view.create_content(width, 1000)
  button = _hover_code(view, content)
  _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
  encoded = base64.b64encode(expected.encode()).decode()
  assert output.raw == [f'\x1b]52;c;{encoded}\x1b\\']


def test_code_copy_controls_only_appear_for_settled_nonempty_blocks() -> None:
  source = _source()
  source.append(_answer('settled', '```text\nsettled\n```\n\n```\n```', 1))
  live = rendering.assistant_render('```text\nstreaming\n```')
  view = transcript_view.TranscriptView(source.blocks, lambda: (live,))
  content = view.create_content(40, 30)
  assert '⧉' not in _text(content)
  button = _hover_code(view, content)
  assert _text(content).count('⧉') == 1
  _mouse(view, button)
  assert 'click to copy' in _text(content)
  _mouse(view, data_structures.Point(3, 5))
  assert '⧉' not in _text(content)
  _mouse(view, data_structures.Point(3, 9))
  assert '⧉' not in _text(content)
  assert '⧉' not in rendering.renderable_to_ansi(live, 40)


def test_code_copy_distinguishes_identical_blocks(output: _Output) -> None:
  source = _source()
  source.append(_answer('code', '```\nsame\n```\n\n```\nsame\n```', 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  content = view.create_content(40, 30)
  button = _hover_code(view, content)
  _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
  _mouse(view, data_structures.Point(3, 5))
  assert _text(content).count('⧉') == 1
  assert 'Copied' not in _text(content)


@pytest.mark.parametrize('change', ['scroll', 'resize', 'conversation'])
def test_code_copy_hover_clears_after_layout_changes(change: str) -> None:
  source = _source()
  source.append(_answer('old', '```\nprevious\n```\n\nMore prose', 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  content = view.create_content(40, 8)
  _hover_code(view, content)
  assert '⧉' in _text(content)
  if change == 'scroll':
    view.scroll(2)
  elif change == 'conversation':
    source.clear_history()
    view.clear()
    source.append(_answer('new', '```\nprevious\n```', 1))
  content = view.create_content(30 if change == 'resize' else 40, 8)
  assert '⧉' not in _text(content)


def test_code_copy_oversize_payload_is_not_sent(output: _Output) -> None:
  source = _source()
  source.append(_answer('large', '```\n' + '界' * 25_001 + '\n```', 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  view.create_content(80, 8)
  view.first()
  content = view.create_content(80, 8)
  button = _hover_code(view, content)
  with freezegun.freeze_time() as clock:
    _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
    assert 'Too large' in _text(content)
    assert not output.raw
    clock.tick(2)
    assert 'Too large' not in _text(content)
    assert '⧉' in _text(content)


def test_code_copy_feedback_expires_and_repeated_clicks_still_copy(
  output: _Output,
) -> None:
  source = _source()
  source.append(_answer('code', '```\nrepeat\n```', 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  content = view.create_content(40, 30)
  button = _hover_code(view, content)
  with freezegun.freeze_time() as clock:
    _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
    assert 'Copied ✓' in _text(content)
    _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
    assert len(output.raw) == 2
    clock.tick(2)
    assert 'Copied' not in _text(content)
    assert '⧉' in _text(content)
    _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
    assert len(output.raw) == 3
    assert 'Copied ✓' in _text(content)


@pytest.mark.parametrize('theme', themes.NAMES)
def test_code_copy_applies_semantic_colors_to_each_state(
  theme: str, output: _Output
) -> None:
  source = _source()
  source.append(_answer('code', '```\nbody\n```', 1))
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), theme_name=lambda: theme
  )
  content = view.create_content(40, 30)
  sheet = themes.prompt_style(theme)
  button = _hover_code(view, content)
  idle = next(
    style for style, text, *_ in content.get_line(button.y) if '⧉' in text
  )
  assert (
    sheet.get_attrs_for_style_str(idle).color
    == sheet.get_attrs_for_style_str('class:ui.muted').color
  )
  _mouse(view, button)
  hovered = next(
    style for style, text, *_ in content.get_line(button.y) if '⧉' in text
  )
  assert (
    sheet.get_attrs_for_style_str(hovered).color
    == sheet.get_attrs_for_style_str('class:ui.info').color
  )
  _mouse(view, button, mouse_events.MouseEventType.MOUSE_UP)
  copied = next(
    style for style, text, *_ in content.get_line(button.y) if '✓' in text
  )
  assert (
    sheet.get_attrs_for_style_str(copied).color
    == sheet.get_attrs_for_style_str('class:ui.success').color
  )
  assert (
    sheet.get_attrs_for_style_str(copied).bgcolor
    == sheet.get_attrs_for_style_str(idle).bgcolor
  )
  assert len(output.raw) == 1


def test_code_copy_control_has_top_and_right_insets() -> None:
  source = _source()
  source.append(_answer('code', '```\nbody\n```', 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  content = view.create_content(40, 30)
  button = _hover_code(view, content)
  lines = _text(content).splitlines()
  assert button == data_structures.Point(37, 1)
  assert not lines[0].removeprefix('●  ┃').strip()
  assert lines[1].endswith(' ⧉  ')
  assert 'body' in lines[2]


def test_interactive_diff_panel_has_no_code_copy_action() -> None:
  diff = rendering.tool_result_renderable(
    'edit',
    'example.txt',
    'Done',
    '--- example.txt\n+++ example.txt\n@@ -1 +1 @@\n-old\n+new',
    compact=True,
    activity_kind='tool',
    error=False,
  )
  assert diff is not None
  painter = rendering.ansi_console(io.StringIO(), 80, interactive=True)
  segments = list(painter.render(diff))
  assert any('┃' in segment.text for segment in segments)
  assert not any(segment.style and segment.style.meta for segment in segments)


@pytest.mark.parametrize('width', [1, 2, 3, 4, 5, 10])
def test_code_copy_does_not_change_narrow_code_layout(width: int) -> None:
  source = _source()
  source.append(_answer('code', '```text\nword\n```', 1))
  settled = transcript_view.TranscriptView(source.blocks, lambda: ())
  live = transcript_view.TranscriptView(
    _source().blocks,
    lambda: (rendering.assistant_render('```text\nword\n```'),),
  )
  assert _text(settled.create_content(width, 30)) == _text(
    live.create_content(width, 30)
  )


def test_viewport_can_reach_history_older_than_the_visible_tail() -> None:
  source = _source()
  source.extend(_answer(str(i), f'answer {i}', i + 1) for i in range(1500))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  assert 'answer 1499' in _text(view.create_content(80, 8))
  view.scroll(-100000)
  assert 'answer 0' in _text(view.create_content(80, 8))


def test_new_output_preserves_reading_position() -> None:
  source = _source()
  source.extend(_answer(str(i), f'answer {i}', i + 1) for i in range(8))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
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
    source.blocks,
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
  assert 'bg:#e8e8e8' in light.get_line(0)[0][0]
  assert view.reading


def test_streaming_code_uses_the_current_palette_and_matches_settled() -> None:
  source = _source()
  code = '```python\nvalue = "hello"\n```'
  live = [rendering.assistant_render(code)]
  selected_theme = 'dark'
  view = transcript_view.TranscriptView(
    source.blocks, lambda: live, theme_name=lambda: selected_theme
  )
  dark = _paint(view, 40, 8, selected_theme)
  selected_theme = 'light'
  light = _paint(view, 40, 8, selected_theme)
  assert light.display[2].strip() == '┃ value = "hello"'
  assert light.buffer[2][5].fg != dark.buffer[2][5].fg
  source.append(_answer('final', code))
  view.settle_live('final')
  live.clear()
  settled = _paint(view, 40, 8, selected_theme)
  assert light.buffer[2] == settled.buffer[2]


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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
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
  view = transcript_view.TranscriptView(source.blocks, lambda: live)
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
    view_mode=lambda: actions.ViewMode.DEFAULT,
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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  assert _text(view.create_content(80, 20)).rstrip() == (
    '·  Explored · 1 read\n\n·  Explored · 1 read'
  )
  actual = rendering.render_blocks(
    [block.renderable for block in source.blocks()], 80
  )
  expected = source.render_history([], [], console_width=80, scrollback_limit=0)
  assert actual == expected.rstrip()


@pytest.mark.parametrize('theme', ['dark', 'light', 'mono', 'terminal'])
@pytest.mark.parametrize(
  'markdown',
  [
    'plain **bold** and `code`',
    '[link](https://example.invalid)\n\n> quoted text\n\n- one\n- two',
    '| A | B |\n| - | - |\n| one | two |',
    '```python\n\tvalue = 1  \n\n```\n\n> ```text\n> nested\n> ```',
  ],
)
def test_viewport_keeps_character_styles_when_coalescing_ansi_fragments(
  theme: str,
  markdown: str,
) -> None:
  source = _source()
  source.append(_answer('styled', markdown, 1))
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), theme_name=lambda: theme
  )
  content = view.create_content(80, 20)
  expected = formatted_text.to_formatted_text(
    hyperlinks.ANSI(
      rendering.renderable_to_ansi(
        rendering.assistant_render(markdown),
        80,
        theme_name=theme,
      ).rstrip()
    )
  )
  expected_rows = list(formatted_text.split_lines(expected))
  for index, row in enumerate(expected_rows):
    actual_cells = [
      (
        ' '.join(
          token
          for token in style.split()
          if not token.startswith('class:code-') and token != 'ansidefault'
        ),
        character,
      )
      for style, text, *_ in content.get_line(index)
      if '[ZeroWidthEscape]' not in style
      for character in text
    ]
    expected_cells = [
      (
        ' '.join(token for token in style.split() if token != 'ansidefault'),
        character,
      )
      for style, text, *_ in row
      if '[ZeroWidthEscape]' not in style
      for character in text
    ]
    # The window fills unstyled space omitted at semantic part boundaries.
    assert actual_cells + [('', ' ')] * (80 - len(actual_cells)) == (
      expected_cells + [('', ' ')] * (80 - len(expected_cells))
    )
  assert content.line_count == len(expected_rows) + 1


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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
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


@pytest.mark.parametrize(
  ('rows', 'header_shown'), [(0, False), (1, False), (2, True), (3, True)]
)
def test_sticky_header_appears_once_the_user_text_scrolls_out(
  rows: int, header_shown: bool
) -> None:
  source = _source()
  source.extend([
    agent_pb2.AgentMessage(
      id='user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='ONLY USER')
      ),
    ),
    _answer('answer', '\n\n'.join(f'Paragraph {i}' for i in range(20)), 2),
  ])
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  view.create_content(80, 12)
  view.first()
  view.create_content(80, 12)
  view.scroll(rows)
  text = _text(view.create_content(80, 12))
  assert text.count('ONLY USER') == 1
  assert (view.scrolling_range().start == 1) == header_shown


@pytest.mark.parametrize('final_width', [80, 24])
def test_finalization_retains_the_occurrence_of_repeated_prose_after_resize(
  final_width: int,
) -> None:
  source = _source()
  answer = '\n\n'.join(f'Repeated paragraph.\n\nContext {i}' for i in range(50))
  live = [rendering.assistant_render(answer, hyperlinks=False)]
  view = transcript_view.TranscriptView(source.blocks, lambda: live)
  view.create_content(80, 8)
  view.scroll(-12)
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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  view.create_content(40, 12)
  view.first()
  view.create_content(40, 12)
  view.scroll(1)
  assert 'USER ROW 0' in _text(view.create_content(40, 12)).splitlines()[0]
  view.scroll(19)
  assert 'USER ROW 19' in _text(view.create_content(40, 12)).splitlines()[1]
  view.scroll(20)
  assert 'USER ROW 39' in _text(view.create_content(40, 12)).splitlines()[1]
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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  view.create_content(80, 12)
  view.first()
  seen = _text(view.create_content(80, 12))
  for _ in range(10):
    view.page(1)
    seen += '\n' + _text(view.create_content(80, 12))
  assert all(f'ROW {i:03d}' in seen for i in range(60)), seen


@pytest.mark.parametrize(
  ('message', 'width', 'expected'),
  [
    ('Short', 20, '›  Short'),
    ('123456', 8, '›  1234…'),
    ('abcdefghijklmnopqrstuvwxyz', 12, '›  abcdefgh…'),
    (
      ' first\tline\r\n second\vthird\ffourth\u2028last ',
      40,
      '›  first line second third fourth last',
    ),
    ('你好世界继续', 10, '›  你好世…'),
    ('e\u0301' * 12, 8, '›  ' + 'e\u0301' * 4 + '…'),
    ('🙂' * 8, 8, '›  🙂🙂…'),
    ('\x1b[31mFirst\x1b[0m\nSecond', 20, '›  First Second'),
    ('Narrow', 1, '…'),
    ('Narrow', 2, '›…'),
    ('Narrow', 3, '›…'),
  ],
)
def test_sticky_summary_uses_one_terminal_row(
  message: str, width: int, expected: str
) -> None:
  source = _source()
  original = _user(message)
  answers = [_answer(str(i), f'A{i}', i + 2) for i in range(20)]
  source.extend([original, *answers])
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  frame = _paint(view, width, 8)
  header = _text(view.create_content(width, 8)).splitlines()[0].rstrip()
  assert header == expected
  assert utils.get_cwidth(header) <= width
  assert cells.cell_len(header) <= width
  assert frame.display[0].rstrip() == unicodedata.normalize('NFC', expected)

  body = _source()
  body.extend(answers)
  without_header = _paint(
    transcript_view.TranscriptView(body.blocks, lambda: ()), width, 7
  )
  assert frame.display[1:] == without_header.display
  assert source.entries()[0].chat.user.content == message


@pytest.mark.parametrize(
  ('message', 'expected'),
  [('👩\u200d💻' * 8, '›  👩\u200d💻…'), ('✈️' * 8, '›  ✈️✈️…')],
)
def test_sticky_summary_keeps_emoji_sequences_whole(
  message: str, expected: str
) -> None:
  source = _source()
  source.extend([_user(message), _answer('answer', 'TEXT\n\n' * 20, 2)])
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  header = _text(view.create_content(8, 6)).splitlines()[0]
  assert header.rstrip() == expected
  assert utils.get_cwidth(header) <= 8
  assert cells.cell_len(header) <= 8
  assert 'TEXT' in _paint(view, 8, 6).display[2]


@pytest.mark.parametrize('height', [1, 2])
def test_sticky_summary_leaves_no_extra_padding_in_short_terminals(
  height: int,
) -> None:
  source = _source()
  source.extend([_user('Long message\n' * 20), _answer('answer', 'ANSWER', 2)])
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  frame = _paint(view, 12, height)
  assert frame.display[0] == '›  Long mes…'
  assert view.create_content(12, height).line_count == height


@pytest.mark.parametrize('name', themes.NAMES)
def test_sticky_summary_is_shaded_but_normal_user_message_is_unchanged(
  name: str,
) -> None:
  message = 'FIRST LINE\nSECOND LINE with original spacing'
  source = _source()
  source.extend([_user(message), _answer('answer', 'ANSWER\n\n' * 20, 2)])
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), theme_name=lambda: name
  )
  summary = _paint(view, 80, 12, name)
  assert (
    summary.display[0].rstrip()
    == '›  FIRST LINE SECOND LINE with original spacing'
  )
  assert not summary.buffer[0][2].italics
  assert (
    summary.buffer[0][2].bg
    == {
      'dark': '282828',
      'light': 'e8e8e8',
      'terminal': 'default',
      'mono': 'default',
    }[name]
  )
  assert not summary.buffer[0][2].reverse
  assert summary.buffer[0][79].bg == summary.buffer[0][2].bg
  assert summary.buffer[0][79].reverse == summary.buffer[0][2].reverse

  view.first()
  original = _paint(view, 40, 12, name)
  assert original.display[0].strip() == ''
  assert original.display[1].rstrip() == '›  FIRST LINE'
  assert original.display[2].rstrip() == '   SECOND LINE with original spacing'
  assert not original.buffer[1][2].italics
  assert source.entries()[0].chat.user.content == message
  view.scroll(10)
  _paint(view, 40, 12, name)
  view.first()
  assert _paint(view, 40, 12, name).display == original.display


def test_leaving_follow_mode_scrolls_one_body_row_with_a_sticky_summary() -> (
  None
):
  source = _source()
  source.extend([
    _user('Long request\n' * 10),
    _answer(
      'answer',
      '```text\n' + '\n'.join(f'ROW {i}' for i in range(30)) + '\n```',
      2,
    ),
  ])
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  following = _paint(view, 30, 10)
  view.scroll(-1)
  reading = _paint(view, 30, 10)
  assert view.reading
  assert reading.display[0] == following.display[0]
  assert reading.display[2:] == following.display[1:-1]
  view.scroll(1)
  assert not view.reading
  assert _paint(view, 30, 10).display == following.display


def test_sticky_summary_resizes_without_moving_streaming_content() -> None:
  source = _source()
  source.append(_user('A long request\nwith multiple lines ' * 3))
  answer = '\n\n'.join(f'Paragraph {i:02d}' for i in range(40))
  live = [rendering.assistant_render(answer, hyperlinks=False)]
  view = transcript_view.TranscriptView(source.blocks, lambda: live)
  _paint(view, 60, 8)
  view.scroll(-13)
  before = _paint(view, 60, 8)
  narrow = _paint(view, 22, 8)
  assert narrow.display[0] == '›  A long request wit…'
  assert [row.rstrip() for row in narrow.display[1:]] == [
    row.rstrip() for row in before.display[1:]
  ]
  live[0] = rendering.assistant_render(
    answer + '\n\nNew output', hyperlinks=False
  )
  assert _paint(view, 22, 8).display == narrow.display
  source.append(_answer('final', answer + '\n\nNew output', 2))
  view.settle_live('final')
  live.clear()
  assert _paint(view, 60, 8).display == before.display
  assert view.reading
  view.follow()
  assert 'New output' in '\n'.join(_paint(view, 60, 8).display[1:])


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
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
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


def _pinned_conversation() -> transcript.Transcript:
  source = _source()
  source.append(
    agent_pb2.AgentMessage(
      id='user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(
          content='\n'.join(f'USER LINE {i}' for i in range(5))
        )
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
  return source


def test_scrolling_down_while_following_leaves_the_viewport_unchanged() -> None:
  view = transcript_view.TranscriptView(
    _pinned_conversation().blocks, lambda: ()
  )
  following = _text(view.create_content(80, 20))
  view.scroll(3)
  assert _text(view.create_content(80, 20)) == following
  assert not view.reading
  view.page(1)
  assert _text(view.create_content(80, 20)) == following
  assert not view.reading


def test_scrolling_under_a_sticky_header_moves_by_the_requested_rows() -> None:
  view = transcript_view.TranscriptView(
    _pinned_conversation().blocks, lambda: ()
  )
  following = _text(view.create_content(80, 20))
  assert 'ROW 048' in following.splitlines()[6]
  view.scroll(-3)
  assert 'ROW 045' in _text(view.create_content(80, 20)).splitlines()[6]
  assert view.reading


def test_scrolling_back_to_the_bottom_resumes_following() -> None:
  view = transcript_view.TranscriptView(
    _pinned_conversation().blocks, lambda: ()
  )
  following = _text(view.create_content(80, 20))
  view.scroll(-3)
  view.create_content(80, 20)
  view.scroll(3)
  assert not view.reading
  assert _text(view.create_content(80, 20)) == following


def test_scrolling_a_conversation_that_fits_never_enters_reading_mode() -> None:
  source = _source()
  source.append(_answer('short', 'short answer', 1))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  view.create_content(80, 20)
  view.scroll(-3)
  assert not view.reading
  view.scroll(3)
  assert not view.reading


def test_scrolling_up_through_a_pinned_message_moves_every_tick() -> None:
  source = _source()
  source.append(_answer('older', '\n\n'.join(['OLDER ANSWER'] * 4), 1))
  source.append(
    agent_pb2.AgentMessage(
      id='tall-user',
      sequence=2,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(
          content='\n'.join(f'USER ROW {i:02d}' for i in range(20))
        )
      ),
    )
  )
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  frames = [_text(view.create_content(40, 20))]
  view.scroll(-3)
  frames.append(_text(view.create_content(40, 20)))
  view.scroll(-3)
  frames.append(_text(view.create_content(40, 20)))
  view.scroll(-3)
  frames.append(_text(view.create_content(40, 20)))
  assert len(set(frames)) == len(frames), frames
  assert 'OLDER ANSWER' in frames[-1]


def _markdown_render_calls(profile: cProfile.Profile) -> int:
  return sum(
    entry.callcount
    for entry in profile.getstats()
    if entry.code == markdown.Markdown.__rich_console__.__code__
  )


@pytest.mark.parametrize('height', [20, 180])
def test_viewport_only_formats_visible_parts_and_reuses_them(
  height: int,
) -> None:
  source = _source(
    _answer('long', '\n\n'.join(f'Paragraph {i}' for i in range(1000)))
  )
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  with cProfile.Profile() as cold:
    assert 'Paragraph 999' in _text(view.create_content(80, height))
  assert _markdown_render_calls(cold) < 3 * height, (
    'Offscreen parts must not be laid out'
  )
  view.scroll(-3)
  view.create_content(80, height)
  with cProfile.Profile() as warm:
    view.scroll(1)
    view.create_content(80, height)
  assert _markdown_render_calls(warm) == 0


def test_view_mode_change_replaces_cached_presentation() -> None:
  mode = actions.ViewMode.DETAIL
  source = transcript.Transcript(
    view_mode=lambda: mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  source.append(_user('Request'))
  entry = _answer('answer', 'Visible answer')
  entry.chat.assistant.reasoning_content = 'Detailed reasoning'
  source.append(entry)
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  before = _text(view.create_content(80, 20))
  assert 'Detailed reasoning' in before
  mode = actions.ViewMode.FOCUS
  changed = _text(view.create_content(80, 20))
  assert 'Detailed reasoning' not in changed
  assert [line.rstrip() for line in changed.splitlines()] == [
    '',
    '›  Request',
    '',
    '',
    '●  Visible answer',
  ]
  mode = actions.ViewMode.DETAIL
  assert _text(view.create_content(80, 20)) == before


def test_revised_tool_group_does_not_reuse_old_rows() -> None:
  result = agent_pb2.AgentMessage(
    chat=agent_pb2.ChatRecord(
      tool=agent_pb2.ToolMessage(name='read', content='result'), success=True
    )
  )
  source = _source(result, mode=actions.ViewMode.DEFAULT)
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  assert '1 read' in _text(view.create_content(80, 8))
  original = source.blocks()[0]
  source.append(result)
  assert '2 reads' in _text(view.create_content(80, 8))
  assert source.blocks()[0].key == original.key
  assert source.blocks()[0].revision != original.revision


def test_rewind_and_reused_message_id_do_not_reuse_discarded_rows() -> None:
  source = _source(_answer('first', 'Retained'), _answer('second', 'Discarded'))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  assert 'Discarded' in _text(view.create_content(80, 8))
  source.truncate(1)
  source.append(_answer('second', 'Replacement'))
  visible = _text(view.create_content(80, 8))
  assert 'Replacement' in visible
  assert 'Discarded' not in visible


def test_pending_input_can_change_and_settle_without_stale_rows() -> None:
  source = _source()
  pending = (
    transcript.Block(
      'pending',
      (0, 0),
      rendering.user_text(agent_pb2.UserMessage(content='First prompt')),
    ),
  )
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), pending=lambda: pending
  )
  original = view.create_content(80, 8)
  pending = (
    transcript.Block(
      'pending',
      (0, 1),
      rendering.user_text(agent_pb2.UserMessage(content='Revised prompt')),
    ),
  )
  revised = _text(view.create_content(80, 8))
  assert 'Revised prompt' in revised
  assert 'First prompt' not in revised
  assert 'First prompt' in _text(original)
  source.append(_answer('answer', 'Settled answer'))
  pending = ()
  settled = _text(view.create_content(80, 8))
  assert 'Settled answer' in settled
  assert 'prompt' not in settled


def test_live_revision_refreshes_the_tail_without_rescanning_history() -> None:
  source = _source(*(_answer(str(i), f'Answer {i}') for i in range(10000)))
  live = (rendering.assistant_render('Old tail'),)
  view = transcript_view.TranscriptView(source.blocks, lambda: live)
  assert 'Old tail' in _text(view.create_content(80, 8))
  live = (rendering.assistant_render('New tail'),)
  with cProfile.Profile() as profile:
    visible = _text(view.create_content(80, 8))
  assert 'New tail' in visible
  assert 'Old tail' not in visible
  assert (
    sum(
      entry.callcount
      for entry in profile.getstats()
      if entry.code == '<built-in method builtins.isinstance>'
    )
    < 1000
  ), 'Sticky-header lookup must not revisit 10,000 settled blocks'


def test_ui_content_cannot_mutate_cached_rows() -> None:
  source = _source(_answer('answer', 'Original answer'))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  content = view.create_content(80, 4)
  before = _text(content)
  content.get_line(0).clear()
  assert _text(content) == before
  assert _text(view.create_content(80, 4)) == before
  view.clear()
  assert _text(content) == before


def test_evicted_layouts_reconstruct_identical_rows() -> None:
  source = _source(*(_answer(str(i), f'Answer {i}') for i in range(8)))
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), max_cache_bytes=4096
  )
  view.create_content(80, 2)
  view.first()
  before = _text(view.create_content(80, 2))
  view.scroll(1000)
  view.create_content(80, 2)
  view.first()
  with cProfile.Profile() as profile:
    assert _text(view.create_content(80, 2)) == before
  assert _markdown_render_calls(profile) > 0, 'The old layout must be evicted'


@pytest.mark.parametrize(
  ('answer', 'budget', 'width', 'height'),
  [
    ('words ' * 100, 0, 20, 4),
    ('words ' * 100, 1024, 20, 4),
    ('words ' * 30 + '\n\n' + 'more ' * 30, 1024, 80, 20),
  ],
)
def test_oversized_layouts_are_reused_only_within_one_paint(
  answer: str, budget: int, width: int, height: int
) -> None:
  source = _source(_answer('answer', answer))
  cached = transcript_view.TranscriptView(source.blocks, lambda: ())
  with cProfile.Profile() as cold:
    expected = _text(cached.create_content(width, height))
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), max_cache_bytes=budget
  )
  assert _text(view.create_content(width, height)) == expected
  with cProfile.Profile() as profile:
    assert _text(view.create_content(width, height)) == expected
  assert _markdown_render_calls(profile) == _markdown_render_calls(cold)
  assert _markdown_render_calls(profile) > 0


def test_oversized_layout_does_not_evict_small_cached_layouts() -> None:
  source = _source(
    _answer('small', 'Small answer'), _answer('large', 'wrapping words ' * 500)
  )
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), max_cache_bytes=2048
  )
  view.create_content(80, 2)
  view.first()
  before = _text(view.create_content(80, 2))
  view.follow()
  view.create_content(80, 2)
  view.first()
  with cProfile.Profile() as profile:
    assert _text(view.create_content(80, 2)) == before
  assert _markdown_render_calls(profile) == 0


@pytest.mark.parametrize(
  ('prefix', 'committed'),
  [
    ('', ()),
    ('Prefix\n\n' * 900, (rendering.assistant_render('Prefix\n\n' * 900),)),
  ],
)
@pytest.mark.parametrize('separator', ['', '\n\n'])
def test_settlement_does_not_render_offscreen_parts(
  prefix: str, committed: tuple[rendering.Answer, ...], separator: str
) -> None:
  source = _source(_answer('old', 'OLD HISTORY'))
  tail = separator + '\n\n'.join(f'Paragraph {i}' for i in range(1000))
  stream = rendering.StreamingAnswer(())
  stream.update(prefix)
  live = [*committed, *stream.update(tail, source_offset=len(prefix))]
  view = transcript_view.TranscriptView(source.blocks, lambda: live)
  view.create_content(80, 8)
  view.scroll(-10)
  before = _text(view.create_content(80, 8))
  source.append(_answer('final', prefix + tail))
  with cProfile.Profile() as profile:
    view.settle_live('final')
    live.clear()
    assert _text(view.create_content(80, 8)) == before
  assert _markdown_render_calls(profile) < 100, (
    'Settlement must not lay out the prefix'
  )


def test_settlement_and_resize_keep_the_live_anchor_after_existing_history() -> (
  None
):
  source = _source(_answer('old', 'OLD HISTORY'))
  answer = '\n\n'.join(
    f'Paragraph {i}: ' + 'wrapping words ' * 8 for i in range(100)
  )
  live = list(rendering.StreamingAnswer(()).update(answer))
  view = transcript_view.TranscriptView(source.blocks, lambda: live)
  view.create_content(80, 8)
  view.scroll(-12)
  before = _text(view.create_content(80, 8))
  paragraph = next(
    line.strip().split(':')[0]
    for line in before.splitlines()
    if 'Paragraph' in line
  )
  source.append(_answer('final', answer))
  view.settle_live('final')
  live.clear()
  assert paragraph in _text(view.create_content(40, 8))


@pytest.mark.parametrize('width', [24, 80])
@pytest.mark.parametrize(
  'answer',
  [
    '# Heading\n\nFirst **paragraph**.\n\nSecond paragraph.',
    '- First\n- Second\n\nFollowing paragraph.',
    '```python\nvalue = "界é"\n```\n\nFollowing paragraph.',
    '| Name | Value |\n| --- | --- |\n| example | 123 |\n\nFollowing paragraph.',
    '[link][ref]\n\nFollowing paragraph.\n\n[ref]: https://example.com',
    '[ref]: https://example.com',
    '---\n\n- First\n- Second\n\nFollowing paragraph.',
    '<!-- hidden -->\n\nFirst paragraph.\n\n<!-- hidden -->\n\nLast paragraph.',
  ],
)
@pytest.mark.parametrize('reasoning', ['', 'Reasoning **with emphasis**.'])
def test_chunked_answer_preserves_canonical_wrapping(
  answer: str, width: int, reasoning: str
) -> None:
  entry = _answer('answer', answer)
  entry.chat.assistant.reasoning_content = reasoning
  source = _source(entry)
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  actual = _text(view.create_content(width, 100))
  rendered = rendering.renderable_to_ansi(source.blocks()[0].renderable, width)
  expected = formatted_text.fragment_list_to_text(
    formatted_text.to_formatted_text(hyperlinks.ANSI(rendered))
  )
  assert [line.rstrip() for line in actual.rstrip().splitlines()] == [
    line.rstrip() for line in expected.rstrip().splitlines()
  ]


class _Formatted:
  """Names of formatted `_Lines` blocks, recorded from the layout thread."""

  def __init__(self) -> None:
    self._loop = asyncio.get_running_loop()
    self._changed = asyncio.Event()
    self.names: list[str] = []

  def record(self, name: str) -> None:
    self.names.append(name)
    self._loop.call_soon_threadsafe(self._changed.set)

  async def wait_for(self, name: str) -> None:
    while name not in self.names:
      self._changed.clear()
      await self._changed.wait()


class _Lines:
  def __init__(self, name: str, formatted: _Formatted) -> None:
    self.name = name
    self.formatted = formatted

  def __rich__(self) -> text.Text:
    self.formatted.record(self.name)
    return text.Text('\n'.join(f'{self.name} row {row}' for row in range(4)))


def _line_blocks(
  count: int, formatted: _Formatted
) -> tuple[transcript.Block, ...]:
  return tuple(
    transcript.Block(f'b{i}', (0, 0), _Lines(f'b{i}', formatted))
    for i in range(count)
  )


class _SizedOutput(base.DummyOutput):
  @override
  def get_size(self) -> data_structures.Size:
    return data_structures.Size(rows=10, columns=40)


@contextlib.asynccontextmanager
async def _running(
  view: transcript_view.TranscriptView,
) -> AsyncGenerator[Callable[[Callable[[str], bool]], asyncio.Future[str]]]:
  """Run `view` so layouts format off-loop; yields a wait for a painted frame."""
  rendered = asyncio.Event()
  with defaults.create_pipe_input() as pipe:
    app = workspace_application.WorkspaceApplication(
      layout=layout.Layout(containers.Window(view)),
      input=pipe,
      output=_SizedOutput(),
      after_render=lambda _: rendered.set(),
    )
    started = asyncio.Event()
    running = asyncio.create_task(app.run_async(pre_run=started.set))

    async def painted(done: Callable[[str], bool]) -> str:
      frame = ''
      try:
        async with asyncio.timeout(5):
          while True:
            with application.current.set_app(app):
              frame = _text(view.create_content(40, 10))
            if done(frame):
              return frame
            rendered.clear()
            await rendered.wait()
      except TimeoutError:
        raise AssertionError(
          f'Expected frame never painted:\n{frame}'
        ) from None

    try:
      await started.wait()
      with application.current.set_app(app):
        yield lambda done: asyncio.ensure_future(painted(done))
    finally:
      app.exit()
      await running


@pytest.mark.asyncio
async def test_long_scroll_formats_only_the_destination() -> None:
  formatted = _Formatted()
  blocks = _line_blocks(300, formatted)
  view = transcript_view.TranscriptView(lambda: blocks, lambda: ())
  async with _running(view) as painted:
    await painted(lambda frame: 'b299 row 3' in frame)
    formatted.names.clear()
    view.scroll(-1000)
    await painted(
      lambda frame: 'b29' not in frame and frame.startswith('b0 row 0')
    )
  assert len(set(formatted.names)) <= 4, sorted(set(formatted.names))


@pytest.mark.asyncio
async def test_scrolls_into_unformatted_history_match_exact_layout() -> None:
  reference = transcript_view.TranscriptView(
    lambda: _line_blocks(30, _Formatted()), lambda: ()
  )
  expected = [_text(reference.create_content(40, 10))]
  for move in [-1] * 6 + [None]:
    if move is None:
      reference.page(-1)
    else:
      reference.scroll(move)
    expected.append(_text(reference.create_content(40, 10)))
  blocks = _line_blocks(30, _Formatted())
  view = transcript_view.TranscriptView(lambda: blocks, lambda: ())
  async with _running(view) as painted:
    shown = [await painted(lambda frame: frame == expected[0])]
    for move in [-1] * 6 + [None]:
      if move is None:
        view.page(-1)
      else:
        view.scroll(move)
      index = len(shown)
      shown.append(
        await painted(lambda frame, index=index: frame == expected[index])
      )
  assert shown == expected


@pytest.mark.asyncio
async def test_idle_view_formats_nearby_history_before_it_is_shown() -> None:
  formatted = _Formatted()
  blocks = _line_blocks(60, formatted)
  view = transcript_view.TranscriptView(lambda: blocks, lambda: ())
  async with _running(view) as painted:
    await painted(lambda frame: 'b59 row 3' in frame)
    await formatted.wait_for('b55')
    view.scroll(-10)
    await painted(lambda frame: frame.startswith('b56 row 0'))
  assert formatted.names.count('b56') == formatted.names.count('b57') == 1
