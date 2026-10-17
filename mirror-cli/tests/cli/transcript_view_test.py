import asyncio
import threading

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit import (
  application,
  data_structures,
  formatted_text,
  mouse_events,
)
from prompt_toolkit.input import defaults
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import base
from rich import console

from mirror.cli import actions, rendering, transcript, transcript_view
from mirror.cli import workspace_application


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


class _BlockedRendering:
  def __init__(self) -> None:
    self.loop = asyncio.get_running_loop()
    self.started = asyncio.Event()
    self.finished = asyncio.Event()
    self.release = threading.Event()
    self.timed_out = False

  def __rich_console__(
    self, output: console.Console, options: console.ConsoleOptions
  ) -> console.RenderResult:
    del output, options
    self.loop.call_soon_threadsafe(self.started.set)
    self.timed_out = not self.release.wait(10)
    yield 'OLD WORKSPACE'
    self.loop.call_soon_threadsafe(self.finished.set)



@pytest.fixture(autouse=True)
def _disable_system_clipboard_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(workspace_application, '_clipboard_candidates', lambda: [])


@pytest.mark.asyncio
@pytest.mark.parametrize('cache_bytes', [0, 8 * 1024 * 1024])
async def test_render_keeps_input_live_and_discards_switched_work(
  cache_bytes: int,
) -> None:
  source = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DETAIL,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  blocked = _BlockedRendering()
  source.add_decoration(blocked)
  view = transcript_view.TranscriptView(
    source.blocks, lambda: (), max_cache_bytes=cache_bytes
  )
  bindings = key_bindings.KeyBindings()
  interrupted, switched = asyncio.Event(), asyncio.Event()
  frames: list[str] = []

  @bindings.add('c-c')
  def interrupt(event: key_processor.KeyPressEvent) -> None:
    del event
    interrupted.set()

  def after_render(app: application.Application[None]) -> None:
    del app
    content = view.create_content(80, 10)
    frame = '\n'.join(
      ''.join(fragment[1] for fragment in content.get_line(index))
      for index in range(content.line_count)
    )
    frames.append(frame)
    if blocked.finished.is_set() and 'NEW WORKSPACE' in frame:
      switched.set()

  with defaults.create_pipe_input() as pipe:
    app: application.Application[None] = application.Application(
      layout=layout.Layout(containers.Window(view)),
      key_bindings=bindings,
      input=pipe,
      output=base.DummyOutput(),
      after_render=after_render,
    )
    running = asyncio.create_task(app.run_async())
    try:
      async with asyncio.timeout(15):
        await blocked.started.wait()
        pipe.send_text('\x03')
        await interrupted.wait()
        assert not blocked.timed_out, 'rendering blocked keyboard dispatch'
        view.clear()
        source.clear_history()
        source.add_decoration(rendering.assistant_render('NEW WORKSPACE'))
        blocked.release.set()
        app.invalidate()
        await switched.wait()
        assert not any('OLD WORKSPACE' in frame for frame in frames)
    finally:
      blocked.release.set()
      app.exit()
      await running


def _shell_result(
  content: str, *, background: bool = False, success: bool = True
) -> agent_pb2.AgentMessage:
  if background:
    return agent_pb2.AgentMessage(
      id='result',
      task_result=agent_pb2.TaskResultRecord(
        tool_name='bash', task_id='task', content=content, success=success
      ),
    )
  return agent_pb2.AgentMessage(
    id='result',
    chat=agent_pb2.ChatRecord(
      tool=agent_pb2.ToolMessage(name='bash', content=content),
      display='ls',
      success=success,
    ),
  )


def _click(view: transcript_view.TranscriptView, row: int = 0) -> None:
  _mouse(
    view, data_structures.Point(0, row), mouse_events.MouseEventType.MOUSE_UP
  )


@pytest.mark.parametrize('background', [False, True])
@pytest.mark.parametrize('success', [True, False])
def test_compact_shell_click_expands_and_collapses_output(
  background: bool, success: bool
) -> None:
  mode = actions.ViewMode.DEFAULT
  # Exceed the general preview character budget.
  output = '\n'.join(f'output line {number:03}' for number in range(200))
  source = transcript.Transcript(
    view_mode=lambda: mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  source.append(_shell_result(output, background=background, success=success))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  collapsed = [
    line.rstrip()
    for line in _text(view.create_content(80, 240)).splitlines()
    if line.strip()
  ]
  glyph = '·' if success else '✗'
  assert collapsed[0].startswith(glyph)
  assert collapsed[0].endswith(' ▸')
  assert collapsed[1:] == ['   └ output line 000 … (+199 lines)']
  _click(view)
  expanded = _text(view.create_content(80, 240))
  assert expanded.count('Ran') == 1
  assert ' ▾' in expanded
  for line in output.splitlines():
    assert line in expanded
  _click(view)
  assert 'output line 001' not in _text(view.create_content(80, 240))

  mode = actions.ViewMode.DETAIL
  assert 'output line 000' in _text(view.create_content(80, 240))
  mode = actions.ViewMode.FOCUS
  assert 'output line' not in _text(view.create_content(80, 240))


@pytest.mark.parametrize(
  ('name', 'display', 'success', 'detail_visible'),
  [
    ('read', None, True, False),
    ('read', None, False, False),
    ('edit', '--- f.txt\n+++ f.txt\n@@ -1 +1 @@\n-old\n+new\n', True, True),
  ],
)
def test_compact_rows_other_than_shell_are_not_expandable(
  name: str, display: str | None, success: bool, detail_visible: bool
) -> None:
  source = _source(
    agent_pb2.AgentMessage(
      id='result',
      chat=agent_pb2.ChatRecord(
        tool=agent_pb2.ToolMessage(name=name, content='hidden output'),
        display=display,
        success=success,
      ),
    ),
    mode=actions.ViewMode.DEFAULT,
  )
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  collapsed = _text(view.create_content(80, 20))
  assert '▸' not in collapsed
  assert ('+ new' in collapsed) is detail_visible
  assert 'hidden output' not in collapsed
  _click(view)
  assert _text(view.create_content(80, 20)) == collapsed


def test_compact_shell_header_keeps_the_gutter_and_trails_a_chevron() -> None:
  command = 'cd /workspace/project && bazel test //project/toolbox/mirror:all'
  source = _source(
    agent_pb2.AgentMessage(
      id='ran',
      chat=agent_pb2.ChatRecord(
        tool=agent_pb2.ToolMessage(name='bash', content='output'),
        display=command,
        success=True,
      ),
    ),
    mode=actions.ViewMode.DEFAULT,
  )
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  lines = [
    line.rstrip()
    for line in _text(view.create_content(40, 20)).splitlines()
    if line.strip()
  ]
  assert lines == [
    '·  Ran cd /workspace/project && bazel… ▸',
    '   └ output',
  ]
  assert all(len(line) <= 40 for line in lines)
  _click(view)
  expanded = _text(view.create_content(40, 20))
  assert '·  Ran cd /workspace/project && bazel ▾' in expanded
  assert '//project/toolbox/mirror:all' in expanded


def test_click_on_preview_or_expanded_output_toggles_the_row() -> None:
  source = _source(
    _shell_result('line one\nline two'), mode=actions.ViewMode.DEFAULT
  )
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  collapsed = _text(view.create_content(80, 20)).splitlines()
  row = next(index for index, line in enumerate(collapsed) if '└' in line)
  _mouse(
    view, data_structures.Point(4, row), mouse_events.MouseEventType.MOUSE_UP
  )
  expanded = _text(view.create_content(80, 20)).splitlines()
  row = next(index for index, line in enumerate(expanded) if 'line two' in line)
  _mouse(
    view, data_structures.Point(4, row), mouse_events.MouseEventType.MOUSE_UP
  )
  assert 'line two' not in _text(view.create_content(80, 20))


def test_view_mode_change_collapses_every_expanded_tool() -> None:
  mode = actions.ViewMode.DEFAULT
  source = transcript.Transcript(
    view_mode=lambda: mode,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  for identity in ('first', 'second'):
    source.append(
      agent_pb2.AgentMessage(
        id=identity,
        chat=agent_pb2.ChatRecord(
          tool=agent_pb2.ToolMessage(
            name='bash', content=f'{identity} output\nmore'
          ),
          success=True,
        ),
      )
    )
    source.append(_answer(f'{identity} answer', f'{identity} answer'))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  lines = _text(view.create_content(80, 20)).splitlines()
  for row in [index for index, line in enumerate(lines) if '▸' in line]:
    _click(view, row)
  assert 'more' in _text(view.create_content(80, 20))
  mode = actions.ViewMode.FOCUS
  view.create_content(80, 20)
  mode = actions.ViewMode.DEFAULT
  collapsed = _text(view.create_content(80, 20))
  assert '·  Ran ▸' in collapsed
  assert 'more' not in collapsed


def test_tool_click_keeps_reading_position_after_scrolling_and_resizing() -> (
  None
):
  source = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DEFAULT,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  source.append(_answer('earlier', 'Earlier paragraph.\n\n' * 10, 1))
  source.append(_shell_result('Result line.\n' * 8))
  source.append(_answer('later', 'Later paragraph.', 3))
  view = transcript_view.TranscriptView(source.blocks, lambda: ())
  view.create_content(80, 12)
  view.scroll(-1)
  before = _text(view.create_content(40, 12)).splitlines()
  row = next(index for index, line in enumerate(before) if '▸' in line)
  _click(view, row)
  expanded = _text(view.create_content(40, 12)).splitlines()
  assert expanded[0] == before[0]
  assert [line.rstrip() for line in expanded[-3:]] == [
    '   └ Result line.',
    '     Result line.',
    '     Result line.',
  ]
  _click(view, row)
  assert _text(view.create_content(40, 12)).splitlines() == before
