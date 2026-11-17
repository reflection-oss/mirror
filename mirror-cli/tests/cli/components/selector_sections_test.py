import pytest
from prompt_toolkit import formatted_text, utils

from mirror.cli import selection, state
from mirror.cli.components import selector

_CHOICES = (
  selection.Choice('a', 'Alpha', section='Connected'),
  selection.Choice('b', 'Beta', section='Connected'),
  selection.Choice('c', 'Charlie', section='Available'),
)


def test_section_navigation_selects_only_choices_and_wraps() -> None:
  picker = state.SelectorState(
    selector=selection.Selector('test', 'Test', _CHOICES, str),
    menu=state.MenuState(active=True, selected=1, max_rows=2),
  )
  assert picker.menu.move(1, 3).selected_item(_CHOICES) == _CHOICES[2]
  assert picker.menu.move(2, 3).selected_item(_CHOICES) == _CHOICES[0]
  visible = picker.menu.move(1, 3)
  fragments = selector.render_choices(visible, _CHOICES, title='Test')
  lines = formatted_text.fragment_list_to_text(fragments).splitlines()
  assert lines[1] == ' Connected'
  assert lines[3] == ''
  assert lines[4] == ' Available'
  assert len(lines) == selector.height(visible, _CHOICES)


def test_filtered_section_remains_visible_without_selecting_a_heading() -> None:
  picker = state.SelectorState(
    selector=selection.Selector('test', 'Test', _CHOICES, str), query='Charlie'
  )
  assert picker.filtered_choices() == (_CHOICES[2],)
  assert picker.selected_choice() == _CHOICES[2]
  output = formatted_text.fragment_list_to_text(
    selector.render_choices(
      state.MenuState(active=True), picker.filtered_choices(), title='Test'
    )
  )
  assert 'Available' in output
  assert 'Connected' not in output


@pytest.mark.parametrize('width', [1, 12, 24, 80])
def test_selector_clips_cells_and_keeps_selected_current_cues(
  width: int,
) -> None:
  fragments = selector.render_choices(
    state.MenuState(active=True),
    (selection.Choice('wide', '界' * 100, section='Long section heading'),),
    title='Long title',
    current='wide',
    width=width,
  )
  lines = list(formatted_text.split_lines(fragments))
  assert (
    max(formatted_text.fragment_list_width(line) for line in lines) <= width
  )
  if width >= 12:
    assert '› * ' in formatted_text.fragment_list_to_text(fragments)
  assert not any('█' in fragment[1] for fragment in fragments)


def test_grouped_choices_keep_aligned_status_and_detail_columns() -> None:
  fragments = selector.render_choices(
    state.MenuState(active=True),
    (
      selection.Choice(
        'log', 'Log', 'Save', status='< On >', section='Preferences'
      ),
      selection.Choice(
        'memory', 'Memory', 'Remember', status='< Off >', section='Preferences'
      ),
    ),
    title='Settings',
    width=60,
  )
  lines = formatted_text.fragment_list_to_text(fragments).splitlines()
  assert lines[1] == ' Preferences'
  assert lines[2].strip() == '› Log     < On >  Save'
  assert lines[3].strip() == 'Memory  < Off >  Remember'


def test_stacked_choice_keeps_badge_and_title_ellipsis_within_row() -> None:
  fragments = selector.render_choices(
    state.MenuState(active=True),
    (
      selection.Choice(
        'long',
        'A long conversation title that cannot fit',
        'Details',
        prefix='●',
      ),
    ),
    title='Workspaces',
    width=30,
    stacked=True,
    footer='',
  )
  lines = formatted_text.fragment_list_to_text(fragments).splitlines()
  assert lines[1].strip().startswith('› ●')
  assert lines[1].rstrip().endswith('…')
  assert len(lines[1]) < 30
  assert lines[2].strip() == 'Details'
  assert 'Enter' not in lines[0]


def test_inline_details_share_a_column_and_each_choice_uses_one_row() -> None:
  choices = (
    selection.Choice('cwd', 'New', '/projects/demo', prefix='+'),
    selection.Choice('tree', 'Worktree', 'Isolated checkout', prefix='+'),
  )
  menu = state.MenuState(active=True)
  fragments = selector.render_choices(menu, choices, title='New', width=80)
  lines = formatted_text.fragment_list_to_text(fragments).splitlines()
  assert 'New' in lines[1]
  assert 'Worktree' in lines[2]
  assert lines[1].index('/projects/demo') == lines[2].index('Isolated checkout')
  assert len(lines) == selector.height(menu, choices) == 4


@pytest.mark.parametrize('width', [32, 40, 80])
def test_type_column_survives_long_titles_in_narrow_rows(width: int) -> None:
  choices = (
    selection.Choice(
      'folder',
      'A long conversation title that needs truncation',
      '2026-11-03',
      prefix='○',
      status='folder  ',
    ),
    selection.Choice(
      'worktree', 'Short', '2026-11-02', prefix='↻', status='worktree'
    ),
  )
  fragments = selector.render_choices(
    state.MenuState(active=True),
    choices,
    title='Workspaces',
    current='folder',
    width=width,
    footer='',
  )
  lines = formatted_text.fragment_list_to_text(fragments).splitlines()
  assert 'folder' in lines[1]
  assert 'worktree' in lines[2]
  assert lines[1].index('folder') == lines[2].index('worktree')
  assert max(utils.get_cwidth(line) for line in lines) <= width


def test_section_labels_do_not_merge_distinct_directories_or_hide_search_keys() -> (
  None
):
  choices = (
    selection.Choice('a', 'First', section='/one/demo', section_label='demo'),
    selection.Choice(
      'b',
      'Second',
      section='/two/demo',
      section_label='demo',
      search_terms=('model-alias',),
    ),
  )
  menu = state.MenuState(active=True)
  output = formatted_text.fragment_list_to_text(
    selector.render_choices(menu, choices, title='Workspaces')
  )
  assert output.count('\n demo\n') == 2
  picker = state.SelectorState(
    selector=selection.Selector('test', 'Test', choices, str), query='/two/demo'
  )
  assert picker.filtered_choices() == (choices[1],)
  assert state.SelectorState(
    selector=picker.selector, query='model-alias'
  ).filtered_choices() == (choices[1],)


def test_selected_row_fill_leaves_one_cell_inset() -> None:
  fragments = selector.render_choices(
    state.MenuState(active=True),
    (selection.Choice('a', 'A'),),
    title='Test',
    width=40,
  )
  line = list(formatted_text.split_lines(fragments))[1]
  assert utils.get_cwidth(formatted_text.fragment_list_to_text(line)) == 39
  assert (
    sum(
      utils.get_cwidth(value)
      for style, value, *_ in line
      if style == 'class:menu-selected'
    )
    == 38
  )
