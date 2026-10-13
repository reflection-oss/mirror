import pytest
from prompt_toolkit import formatted_text

from mirror import workspace
from mirror.cli import selection, state
from mirror.cli.components import resume, selector


@pytest.mark.parametrize('scope', tuple(workspace.SessionScope))
def test_resume_scope_tabs_match_keyboard_order(
  scope: workspace.SessionScope,
) -> None:
  tabs = formatted_text.fragment_list_to_text(resume.scope_tabs(scope, 80))
  assert tabs.index('cwd') < tabs.index('git') < tabs.index('all')
  assert f'[{scope.value}]' in tabs


def test_resume_scope_tabs_keep_active_scope_on_narrow_terminal() -> None:
  assert (
    formatted_text.fragment_list_to_text(
      resume.scope_tabs(workspace.SessionScope.GIT, 10)
    )
    == '[git] '
  )


def test_resume_rows_keep_title_and_session_details_separate() -> None:
  fragments = selector.render_choices(
    state.MenuState(active=True),
    (
      selection.Choice(
        'session', 'A conversation', '2026-10-12 · abcdef12 · /tmp/project'
      ),
    ),
    title='Resume',
    stacked=True,
    width=60,
  )
  lines = formatted_text.fragment_list_to_text(fragments).splitlines()
  assert 'A conversation' in lines[1]
  assert '2026-10-12 · abcdef12 · /tmp/project' in lines[2]
  assert lines[1].startswith(' › ')
  assert lines[2].startswith('   ')
  assert len(lines[1]) == len(lines[2]) == 59
