"""Tests for the reusable scrollable-menu primitive."""

from __future__ import annotations

import pytest
from prompt_toolkit.utils import get_cwidth

from mirror.cli import state
from mirror.cli.components import menu


def test_selection_wraps_and_view_tracks_selected_item() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(title='Things', max_rows=2)
  )
  items = ('one', 'two', 'three', 'four')
  menu_state = state.MenuState(max_rows=2)

  menu_state = menu_state.move(-1, len(items))
  assert menu_state.selected_item(items) == 'four'
  assert component.view(menu_state, items).visible == ('three', 'four')

  menu_state = menu_state.move(1, len(items))
  assert menu_state.selected_item(items) == 'one'
  menu_state = state.MenuState(selected=2, start=1, max_rows=2)
  view = component.view(menu_state, items)
  assert view.visible == ('two', 'three')
  assert view.has_above and view.has_below

  menu_state = menu_state.move(1, 2)
  assert menu_state.selected_item(items[:2]) == 'one'


def test_viewport_only_scrolls_when_selection_crosses_an_edge() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(title='Things', max_rows=3)
  )
  items = ('one', 'two', 'three', 'four', 'five')
  menu_state = state.MenuState(selected=4, start=2, max_rows=3)

  assert component.view(menu_state, items).visible == ('three', 'four', 'five')

  menu_state = menu_state.move(-1, len(items))
  assert component.view(menu_state, items).visible == ('three', 'four', 'five')
  menu_state = menu_state.move(-1, len(items))
  assert component.view(menu_state, items).visible == ('three', 'four', 'five')
  menu_state = menu_state.move(-1, len(items))
  assert component.view(menu_state, items).visible == ('two', 'three', 'four')


def test_render_uses_custom_rows_and_aligns_scroll_triangles() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(
      title='Things',
      max_rows=2,
      scroll_column=24,
      border_style='class:custom-border',
      scroll_style='class:custom-scroll',
    )
  )
  items = ('one', 'two', 'three', 'four')
  menu_state = state.MenuState(selected=2, start=1, max_rows=2)

  fragments = component.render(
    menu_state,
    items,
    lambda item, _, selected: [
      ('class:selected' if selected else 'class:item', item)
    ],
  )
  text = ''.join(fragment[1] for fragment in fragments)
  lines = text.splitlines()

  assert ('class:custom-border', ' ') in fragments
  assert ('class:selected', 'three') in fragments
  assert lines[0].index('▴') == lines[-1].index('▾') == 24
  assert component.height(menu_state, items) == 4


def test_scroll_column_expands_for_wide_header() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(title='界 menu', max_rows=1, scroll_column=10)
  )
  items = ('one', 'two', 'three')
  menu_state = state.MenuState(selected=1, start=1, max_rows=1)
  text = ''.join(
    fragment[1]
    for fragment in component.render(
      menu_state, items, lambda item, *_: [('', item)]
    )
  )
  lines = text.splitlines()
  top_prefix = lines[0].partition('▴')[0]
  bottom_prefix = lines[-1].partition('▾')[0]

  assert get_cwidth(top_prefix) == get_cwidth(bottom_prefix)
  assert get_cwidth(top_prefix) >= get_cwidth(' 界 menu  (2/3)') + 3


def test_confirm_panel_frames_its_body_and_reports_its_height() -> None:
  panel = menu.ConfirmPanel(
    title='Discard the rest?',
    lines=('2 turns will go.', 'Files are not restored.'),
    hint='y discard · n keep',
  )

  text = ''.join(fragment[1] for fragment in panel.render())
  lines = text.splitlines()

  assert lines == [
    '╭─ Discard the rest?',
    '│ 2 turns will go.',
    '│ Files are not restored.',
    '╰─ y discard · n keep',
  ]
  assert panel.height == len(lines)


def test_max_rows_must_be_positive() -> None:
  with pytest.raises(ValueError, match='max_rows'):
    menu.MenuConfig(title='Broken', max_rows=0)
  with pytest.raises(ValueError, match='scroll_column'):
    menu.MenuConfig(title='Broken', scroll_column=-1)


def test_render_exposes_absolute_row_index_and_custom_chrome() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(
      title='Choose',
      max_rows=2,
      header_style='class:header',
      footer='Enter choose · Esc cancel',
      footer_style='class:hint',
    )
  )
  items = ('one', 'two', 'three')
  menu_state = state.MenuState(selected=2, start=1, max_rows=2)

  fragments = component.render(
    menu_state,
    items,
    lambda item, index, selected: [
      ('class:selected' if selected else 'class:item', f'{index + 1}: {item}')
    ],
  )

  assert ('class:header', ' Choose') in fragments
  assert ('class:menu-detail', '  (3/3)') in fragments
  assert ('class:selected', '3: three') in fragments
  assert ('class:hint', 'Enter choose · Esc cancel') in fragments
  assert component.height(menu_state, items) == 4


def test_render_supports_dynamic_chrome_and_empty_picker() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(
      title='Choose',
      footer='Enter choose',
      footer_style='class:hint',
    )
  )
  menu_state = state.MenuState()

  fragments = component.render(
    menu_state,
    (),
    lambda item, *_: [('', item)],
    title='Choose [all]',
    footer='Tab scope · Esc cancel',
    empty_row=[('class:empty', 'Nothing in this scope.')],
  )

  assert ('class:menu-border', ' Choose [all]') in fragments
  assert ('class:empty', 'Nothing in this scope.') in fragments
  assert ('class:hint', 'Tab scope · Esc cancel') in fragments
  assert component.height(menu_state, (), show_empty=True) == 3


def test_render_header_suffix_sits_before_scroll_arrow() -> None:
  component = menu.ScrollableMenu[str](
    menu.MenuConfig(title='Choose', max_rows=1, scroll_column=10)
  )
  items = ('one', 'two')
  menu_state = state.MenuState(selected=1, start=1, max_rows=1)

  text = ''.join(
    fragment[1]
    for fragment in component.render(
      menu_state,
      items,
      lambda item, *_: [('', item)],
      header_suffix=[
        ('class:scope', 'cwd | '),
        ('class:active', 'git'),
        ('class:scope', ' | all ⇄ tab'),
      ],
      footer='',
    )
  )
  lines = text.splitlines()

  assert 'cwd | git | all ⇄ tab' in lines[0]
  assert lines[0].endswith('▴')
  assert lines[-1].strip() == '▾'
