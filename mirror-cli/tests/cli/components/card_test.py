import pathlib

import pytest
from rich import cells

from mirror.cli import actions, commands, rendering
from mirror.cli.components import card


@pytest.mark.parametrize('width', [12, 32, 80])
def test_card_uses_single_line_borders_and_fits_terminal(width: int) -> None:
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(
      card.sections(
        'Status', ((('Model', 'model-name'),), (('Directory', '/tmp/project'),))
      ),
      width,
    )
  )
  assert '╭' in rendered
  assert '╰' in rendered
  assert '═' not in rendered
  assert max(cells.cell_len(line) for line in rendered.splitlines()) <= width


@pytest.mark.parametrize('width', [32, 120])
def test_welcome_card_keeps_facts_and_setup_without_a_second_emblem(
  width: int,
) -> None:
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(
      rendering.render_banner(
        'example-model',
        pathlib.Path.home() / 'project',
        setup_url='https://example.test',
        setup_provider='Test',
      ),
      width,
    )
  )
  assert 'Reflection mirror' in rendered
  assert ' · ' not in rendered
  assert 'example-model' in rendered
  assert '~/project' in rendered
  assert 'Connect to Test' in rendered
  assert 'Paste your key below' in rendered
  assert '╱' not in rendered
  assert '╰┳╯' not in rendered
  assert max(cells.cell_len(line) for line in rendered.splitlines()) <= width


@pytest.mark.parametrize('width', [32, 100])
def test_welcome_facts_are_unframed_and_keep_identity_model_and_directory(
  width: int,
) -> None:
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(
      rendering.render_banner('example-model', '/workspace/mirror'), width
    )
  )
  assert 'Reflection mirror' in rendered
  assert 'example-model' in rendered
  assert '/workspace/mirror' in rendered
  assert '╭' not in rendered
  assert '╰' not in rendered
  assert max(cells.cell_len(line) for line in rendered.splitlines()) <= width


def test_status_card_retains_all_session_facts_without_markup() -> None:
  action = commands.dispatch(
    '/status',
    commands.CommandContext(
      '[red]model',
      'medium',
      'conversation-id',
      True,
      '/tmp/workspace',
      session_id='session-id',
    ),
  )
  assert isinstance(action, actions.ShowText)
  rendered = rendering.strip_ansi(
    rendering.renderable_to_ansi(action.text, 100)
  )
  assert '[red]model' in rendered
  assert 'medium' in rendered
  assert 'conversation-id' in rendered
  assert 'session-id' in rendered
  assert '/tmp/workspace' in rendered
  assert 'yes' in rendered
