import pytest
from rich import cells

from mirror.cli import rendering, state, stream


@pytest.mark.parametrize('width', [10, 32, 80])
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
  assert (
    output.index('Reasoning')
    < output.index('Thoughts')
    < output.index('Answer')
  )
  assert '\n  Answer' in output or '● Answer' in output


def test_assistant_list_starts_beside_its_glyph() -> None:
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(
      rendering.assistant_render('- First\n- Second'), 40
    )
  )
  assert rendered.splitlines()[0].startswith('●')
  assert 'First' in rendered.splitlines()[0]
