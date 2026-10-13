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
  assert lines[3] == ' Available'
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
