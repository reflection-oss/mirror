import io

import pytest
from mirror.protocol import agent_pb2
from rich import cells

from mirror.cli import actions, rendering, state, stream, transcript


@pytest.fixture
def compact_history() -> transcript.Transcript:
  return transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DEFAULT,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )


def _lines(
  history: transcript.Transcript, entry: agent_pb2.AgentMessage
) -> list[str]:
  rendered = history.project(entry, {})
  assert rendered is not None
  output = io.StringIO()
  rendering.ansi_console(output, 120, interactive=False).print(rendered)
  rendered = rendering.strip_ansi(output.getvalue()).strip()
  return [line.rstrip() for line in rendered.splitlines()]


def test_fork_record_renders_parent_without_workspace_paths(
  compact_history: transcript.Transcript,
) -> None:
  entry = agent_pb2.AgentMessage(
    fork=agent_pb2.ForkRecord(
      parent_session_id='parent',
      inherited_sequence=5,
      source_workspace='/old/[repo]',
      destination_workspace='/new/repo',
    )
  )
  assert _lines(compact_history, entry) == ['◆  Forked from parent']


def test_compaction_shows_context_tokens_before_and_after(
  compact_history: transcript.Transcript,
) -> None:
  entry = agent_pb2.AgentMessage(
    context_window=agent_pb2.ContextWindowRecord(
      compaction=agent_pb2.Compaction(
        summary='summary',
        context_tokens_before=120_000,
        context_tokens_after=18_000,
      )
    )
  )
  assert _lines(compact_history, entry)[0] == (
    '·  context compacted 120,000 → 18,000 tokens'
  )


@pytest.mark.parametrize('width', [11, 32, 80])
def test_code_panel_keeps_content_and_fits_terminal(width: int) -> None:
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(
      rendering.assistant_render('```python\nvalue = 1\n```'), width
    )
  )
  assert 'value' in rendered
  assert 'value=1' in ''.join(rendered.replace('┃', '').split())
  assert max(cells.cell_len(line) for line in rendered.splitlines()) <= width
  assert '┃' in rendered
  assert '╭' not in rendered


@pytest.mark.parametrize(
  'content',
  [
    'An answer.\n\n```python\nprint(1)\n```',
    '| A | B |\n| - | - |\n| one | two |',
    '> Quoted words.\n\n1. First\n2. Second',
    '- First\n- Second',
  ],
)
def test_streamed_and_settled_markdown_share_presentation(content: str) -> None:
  preview, _ = stream.render_stream_preview(
    state.StreamPreview(content=content), width=60, max_lines=40
  )
  live = ''.join(fragment[1] for fragment in preview.__pt_formatted_text__())
  settled = rendering.strip_ansi(
    rendering.renderable_to_ansi(rendering.assistant_render(content), 60)
  )
  assert live.rstrip() == settled.rstrip()


def test_heading_is_left_aligned_and_reasoning_stays_separate() -> None:
  rendered = rendering.assistant_turn_renderable('Thoughts', '# Answer\n\nBody')
  assert rendered is not None
  output = rendering.strip_ansi(rendering.renderable_to_ansi(rendered, 60))
  lines = output.splitlines()
  assert lines[0].rstrip() == 'Thoughts'
  assert not lines[1].strip()
  assert '\n   Answer' in output or '●  Answer' in output


@pytest.mark.parametrize('theme_name', ['dark', 'light', 'terminal', 'mono'])
def test_reasoning_is_italic_without_a_heading(theme_name: str) -> None:
  preview, _ = stream.render_stream_preview(
    state.StreamPreview(reasoning='Consider the options.'),
    width=60,
    max_lines=40,
    theme_name=theme_name,
  )
  settled = rendering.renderable_to_ansi(
    rendering.reasoning_text('Consider the options.'), 60, theme_name=theme_name
  )
  assert rendering.strip_ansi(preview.value) == 'Consider the options.'
  assert preview.value == settled.rstrip()
  assert 'italic' in preview.__pt_formatted_text__()[0][0]


def test_assistant_list_starts_beside_its_glyph() -> None:
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(
      rendering.assistant_render('- First\n- Second'), 40
    )
  )
  assert rendered.splitlines()[0].startswith('●')
  assert 'First' in rendered.splitlines()[0]


@pytest.mark.parametrize(
  ('content', 'success', 'expected'),
  [
    (
      'Exit code: 1\nstdout:\nfailed\n',
      True,
      ['✗  Ran pytest · exit 1', '   └ failed'],
    ),
    (
      'Timed out.\nExit code: -1\nstdout:\npartial\n',
      True,
      ['✗  Ran pytest · timed out', '   └ partial'],
    ),
    (
      'Command timed out after 60000ms\nPartial stdout:\npartial\n[Completed in 60.0s]',
      False,
      ['✗  Ran pytest · timed out', '   └ partial'],
    ),
    (
      'Command timed out after 5ms\n[Completed in 0.1s]',
      True,
      ['·  Ran pytest', '   └ Command timed out after 5ms'],
    ),
    (
      'Exit code: 0\nstdout:\npassed\n',
      True,
      ['·  Ran pytest', '   └ passed'],
    ),
  ],
)
def test_collapsed_shell_results_preserve_exit_and_timeout_status(
  compact_history: transcript.Transcript,
  content: str,
  success: bool,
  expected: list[str],
) -> None:
  foreground = agent_pb2.AgentMessage(
    chat=agent_pb2.ChatRecord(
      tool=agent_pb2.ToolMessage(
        name='bash', tool_call_id='tool', content=content
      ),
      display='pytest',
      success=success,
    )
  )
  background = agent_pb2.AgentMessage(
    task_result=agent_pb2.TaskResultRecord(
      tool_name='bash',
      task_id='task',
      content=content,
      display='pytest',
      success=success,
    )
  )
  assert _lines(compact_history, foreground) == expected
  assert _lines(compact_history, background) == expected


@pytest.mark.parametrize(
  ('entry', 'expected'),
  [
    (
      agent_pb2.AgentMessage(
        metadata=agent_pb2.MetadataRecord(exit_code=0),
        chat=agent_pb2.ChatRecord(
          tool=agent_pb2.ToolMessage(
            name='bash',
            tool_call_id='tool',
            content='Exit code: 1\n[Completed in 0.1s]',
          ),
          display='pytest',
          success=True,
        ),
      ),
      '·  Ran pytest',
    ),
    (
      agent_pb2.AgentMessage(
        metadata=agent_pb2.MetadataRecord(exit_code=0),
        task_result=agent_pb2.TaskResultRecord(
          tool_name='bash',
          task_id='task',
          content='Exit code: 1\n[Completed in 0.1s]',
          display='pytest',
          success=True,
        ),
      ),
      '·  Ran pytest',
    ),
    (
      agent_pb2.AgentMessage(
        metadata=agent_pb2.MetadataRecord(exit_code=1),
        chat=agent_pb2.ChatRecord(
          tool=agent_pb2.ToolMessage(
            name='bash',
            tool_call_id='tool',
            content='output spilled to a file',
          ),
          display='pytest',
          success=True,
        ),
      ),
      '✗  Ran pytest · exit 1',
    ),
    (
      agent_pb2.AgentMessage(
        metadata=agent_pb2.MetadataRecord(exit_code=1),
        task_result=agent_pb2.TaskResultRecord(
          tool_name='bash',
          task_id='task',
          content='output spilled to a file',
          display='pytest',
          success=True,
        ),
      ),
      '✗  Ran pytest · exit 1',
    ),
  ],
)
def test_collapsed_shell_results_use_recorded_exit_status(
  compact_history: transcript.Transcript,
  entry: agent_pb2.AgentMessage,
  expected: str,
) -> None:
  assert _lines(compact_history, entry)[0] == expected
