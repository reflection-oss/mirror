"""Small, reusable scrollable-menu primitive for prompt-toolkit views."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

from prompt_toolkit import formatted_text
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension
from prompt_toolkit.utils import get_cwidth

from mirror.cli import state, themes

type RowRenderer[Item] = Callable[[Item, int, bool], StyleAndTextTuples]


def fit_line(fragments: StyleAndTextTuples, width: int) -> StyleAndTextTuples:
  """Clip one display line by terminal cells, preserving its fragment styles."""
  result: StyleAndTextTuples = []
  remaining = max(0, width)
  for fragment in fragments:
    shown = ''
    for character in fragment[1]:
      size = get_cwidth(character)
      if size > remaining:
        break
      shown += character
      remaining -= size
    result.append(
      (fragment[0], shown, fragment[2])
      if len(fragment) == 3
      else (fragment[0], shown)
    )
    if shown != fragment[1]:
      break
  return result


def window(
  fragments: StyleAndTextTuples, width: int, *, theme_name: str = 'dark'
) -> containers.Window:
  """Give selectors a shared inset surface without changing their controllers."""
  edge = '▄' if themes.rich_theme(theme_name).styles['ui.menu'].bgcolor else ' '
  content = (
    [('class:menu-top-padding', edge * width + '\n'), *fragments]
    if fragments
    else []
  )
  return containers.Window(
    controls.FormattedTextControl(content),
    height=dimension.Dimension.exact(
      1 + sum(fragment[1].count('\n') for fragment in content) if content else 0
    ),
    style='class:menu-surface',
    wrap_lines=False,
  )


def _normalize_header_suffix(
  header_suffix: str | StyleAndTextTuples | None,
) -> StyleAndTextTuples:
  """Turn a plain or styled header suffix into spaced fragments."""
  if header_suffix is None:
    return []
  if isinstance(header_suffix, str):
    return [('', f'  {header_suffix}')] if header_suffix else []
  if not header_suffix:
    return []
  return [('', '  '), *header_suffix]


@dataclasses.dataclass(frozen=True)
class MenuConfig:
  """Menu chrome and viewport configuration."""

  title: str
  max_rows: int = 6
  scroll_column: int = 24
  border_style: str = 'class:menu-border'
  scroll_style: str = 'class:menu-scroll'
  header_style: str | None = None
  footer: str = ''
  footer_style: str = 'class:menu-footer'

  def __post_init__(self) -> None:
    if self.max_rows < 1:
      raise ValueError('max_rows must be positive')
    if self.scroll_column < 0:
      raise ValueError('scroll_column must not be negative')


@dataclasses.dataclass(frozen=True)
class ConfirmPanel:
  """A yes/no question framed in the same chrome as the menus."""

  title: str
  lines: tuple[str, ...]
  hint: str

  @property
  def height(self) -> int:
    """Rows the panel occupies: the body plus its title and hint."""
    return len(self.lines) + 2

  def render(self) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = [
      ('class:menu-border', f'╭─ {self.title}\n')
    ]
    for line in self.lines:
      fragments.extend((
        ('class:menu-border', '│ '),
        ('class:menu-item', f'{line}\n'),
      ))
    fragments.extend((
      ('class:menu-border', '╰─ '),
      ('class:menu-hint', self.hint),
    ))
    return fragments


@dataclasses.dataclass(frozen=True)
class MenuView[Item]:
  """The visible projection of a menu's current selection."""

  items: tuple[Item, ...]
  visible: tuple[Item, ...]
  start: int
  selected: int
  has_above: bool
  has_below: bool


class ScrollableMenu[Item]:
  """Configurable viewport and chrome rendering for immutable menu state."""

  def __init__(self, config: MenuConfig) -> None:
    self.config = config

  def view(
    self, menu_state: state.MenuState, items: Sequence[Item]
  ) -> MenuView[Item]:
    all_items = tuple(items)
    if not all_items:
      return MenuView((), (), 0, 0, False, False)
    selected = min(menu_state.selected, len(all_items) - 1)
    max_start = max(0, len(all_items) - menu_state.max_rows)
    start = min(menu_state.start, max_start)
    if selected < start:
      start = selected
    elif selected >= start + menu_state.max_rows:
      start = selected - menu_state.max_rows + 1
    visible = all_items[start : start + menu_state.max_rows]
    return MenuView(
      items=all_items,
      visible=visible,
      start=start,
      selected=selected,
      has_above=start > 0,
      has_below=start + len(visible) < len(all_items),
    )

  def height(
    self,
    menu_state: state.MenuState,
    items: Sequence[Item],
    *,
    show_empty: bool = False,
  ) -> int:
    view = self.view(menu_state, items)
    if view.items:
      return len(view.visible) + 2
    return 3 if show_empty else 0

  def render(
    self,
    menu_state: state.MenuState,
    items: Sequence[Item],
    render_row: RowRenderer[Item],
    *,
    title: str | None = None,
    header_suffix: str | StyleAndTextTuples | None = None,
    footer: str | None = None,
    empty_row: StyleAndTextTuples | None = None,
    min_rows: int = 0,
    heading: Callable[[Item, int], str] | None = None,
    width: int | None = None,
  ) -> StyleAndTextTuples:
    """Render an unframed header, aligned rows and edge scroll indicators."""
    view = self.view(menu_state, items)
    if not view.items and empty_row is None:
      return []

    active_title = title if title is not None else self.config.title
    position = (
      f'  ({view.selected + 1}/{len(view.items)})' if view.items else ''
    )
    header = f' {active_title}'
    suffix_fragments = _normalize_header_suffix(header_suffix)
    header_width = get_cwidth(header + position) + sum(
      get_cwidth(fragment[1]) for fragment in suffix_fragments
    )
    active_footer = footer if footer is not None else self.config.footer
    arrow_column = (
      max(0, width - 1)
      if width is not None
      else max(
        self.config.scroll_column,
        header_width + 3,
        get_cwidth(active_footer) + 2,
      )
    )
    fragments: StyleAndTextTuples = [
      (
        self.config.header_style or self.config.border_style,
        header,
      ),
      ('class:menu-detail', position),
      *suffix_fragments,
    ]
    hint = ''
    if width is not None:
      available = arrow_column - header_width - 2
      hint = next(
        (
          hint
          for hint in (
            active_footer,
            'Enter select · Esc cancel',
            'Enter · Esc',
          )
          if hint and get_cwidth(hint) <= available
        ),
        '',
      )
      if hint:
        fragments.extend((
          ('', ' ' * (arrow_column - header_width - get_cwidth(hint) - 1)),
          (self.config.footer_style, hint),
          ('', ' '),
        ))
    fragments = fit_line(fragments, arrow_column)
    used = sum(get_cwidth(fragment[1]) for fragment in fragments)
    fragments.extend((
      ('', ' ' * max(0, arrow_column - used)),
      (
        self.config.scroll_style
        if view.has_above
        else 'class:menu-scroll-disabled',
        '▴',
      ),
      ('', '\n'),
    ))

    if view.items:
      rows = (
        render_row(item, index, index == view.selected)
        for index, item in enumerate(view.visible, start=view.start)
      )
    else:
      rows = iter((empty_row or [],))
    for offset, row in enumerate(rows):
      if heading is not None and view.items:
        title = heading(view.visible[offset], view.start + offset)
        if title:
          fragments.extend((
            (self.config.border_style, ' '),
            ('class:menu-section', title + '\n'),
          ))
      for line in formatted_text.split_lines(row):
        fragments.append((self.config.border_style, ' '))
        fragments.extend(line)
        fragments.append(('', '\n'))

    padding = max(0, min_rows - max(1, len(view.visible)))
    fragments.extend([('', '\n')] * padding)
    footer_hint = next(
      (
        value
        for value in (active_footer, 'Enter · Esc', 'Esc')
        if value and get_cwidth(value) < arrow_column
      ),
      '',
    )
    footer_fragments: StyleAndTextTuples = (
      [('', ' '), (self.config.footer_style, footer_hint)]
      if not hint and active_footer
      else []
    )
    footer_fragments = fit_line(footer_fragments, arrow_column)
    used = sum(get_cwidth(fragment[1]) for fragment in footer_fragments)
    fragments.extend((
      *footer_fragments,
      ('', ' ' * max(0, arrow_column - used)),
      (
        self.config.scroll_style
        if view.has_below
        else 'class:menu-scroll-disabled',
        '▾',
      ),
    ))
    return fragments
