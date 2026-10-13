"""Shared, stateless rendering and terminal input for declarative selectors."""

import dataclasses
from collections.abc import Callable, Sequence
from typing import override

from prompt_toolkit import buffer, data_structures, formatted_text
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension, processors
from prompt_toolkit.utils import get_cwidth

from rich import text

from mirror.cli import rendering, selection, state, themes
from mirror.cli.components import component, fuzzy_score, menu, resume

_MENU = menu.ScrollableMenu[selection.Choice](
  menu.MenuConfig(
    title='',
    max_rows=8,
    footer='Enter to select · Esc to cancel',
  )
)


def highlight(text: str, query: str, style: str) -> StyleAndTextTuples:
  positions = fuzzy_score.positions(text, query) or ()
  fragments: StyleAndTextTuples = []
  start = 0
  for index in positions:
    if start < index:
      fragments.append((style, text[start:index]))
    fragments.append((f'{style} bold underline', text[index : index + 1]))
    start = index + 1
  if start < len(text):
    fragments.append((style, text[start:]))
  return fragments


def height(
  menu_state: state.MenuState, choices: Sequence[selection.Choice]
) -> int:
  view = _MENU.view(menu_state, choices)
  headings = sum(
    bool(choice.section)
    and (index == view.start or choice.section != choices[index - 1].section)
    for index, choice in enumerate(view.visible, view.start)
  )
  return (
    _MENU.height(menu_state, choices, show_empty=menu_state.active) + headings
  )


def render_choices(
  menu_state: state.MenuState,
  choices: Sequence[selection.Choice],
  *,
  title: str,
  query: str = '',
  current: str | None = None,
  empty: str = 'No matching choices.',
  header_suffix: str | None = None,
  min_rows: int = 0,
  numbered: bool = False,
  width: int | None = None,
  stacked: bool = False,
  footer: str | None = None,
) -> StyleAndTextTuples:
  view = _MENU.view(menu_state, choices)
  label_width = max((get_cwidth(choice.label) for choice in choices), default=0)
  if width is not None:
    label_width = min(
      label_width, max(0, width - 4) if stacked else max(8, (width - 4) // 3)
    )

  def heading(choice: selection.Choice, index: int) -> str:
    return (
      choice.section
      if index == view.start or choice.section != choices[index - 1].section
      else ''
    )

  def row(
    choice: selection.Choice, index: int, selected: bool
  ) -> StyleAndTextTuples:
    style = 'class:menu-selected' if selected else 'class:menu-item'
    label = choice.label
    if get_cwidth(label) > label_width:
      label = (
        formatted_text.fragment_list_to_text(
          menu.fit_line([('', label)], max(0, label_width - 1))
        )
        + '…'
      )
    label_query = (
      query if fuzzy_score.positions(label, query) is not None else ''
    )
    padding = ' ' * max(0, label_width - get_cwidth(label))
    prefix = (
      [
        *formatted_text.to_formatted_text(
          choice.prefix, style=f'{style} {choice.prefix_style}'
        ),
        (style, '  '),
      ]
      if choice.prefix
      else []
    )
    detail_style = (
      f'{style} class:menu-detail-selected' if selected else 'class:menu-detail'
    )
    status: StyleAndTextTuples = []
    if choice.status is not None:
      status = [(style, f'{padding}  {choice.status}')]
      padding = ''
    fragments = [
      (style, '› ' if selected else '  '),
      (style, ('* ' if choice.value == current else '  ') if current else ''),
      (style, f'{index + 1}. ' if numbered and index < 9 else ''),
      *prefix,
      *highlight(label, label_query, style),
      *status,
      *highlight(
        f'\n  {choice.detail}'
        if stacked
        else (f'{padding}  {choice.detail}' if choice.detail else ''),
        '' if label_query else query,
        detail_style,
      ),
      (style, '  (current)' if choice.value == current else ''),
    ]
    if width is None:
      return fragments
    fitted: StyleAndTextTuples = []
    for index, line in enumerate(formatted_text.split_lines(fragments)):
      if index:
        fitted.append(('', '\n'))
      line = menu.fit_line(line, max(0, width - 2))
      used = sum(get_cwidth(fragment[1]) for fragment in line)
      fitted.extend(line)
      fitted.append((style, ' ' * max(0, width - 2 - used)))
    return fitted

  fragments = _MENU.render(
    menu_state,
    choices,
    row,
    title=title,
    header_suffix=header_suffix,
    empty_row=[('class:menu-item', empty)],
    min_rows=min_rows,
    footer=footer
    or (
      '1–9 to select · ↑↓/Enter · Esc to cancel'
      if numbered
      else ('Enter · Esc' if width is not None and width < 42 else None)
    ),
    heading=heading,
    width=width,
  )
  if width is None:
    return fragments
  fitted: StyleAndTextTuples = []
  for index, line in enumerate(formatted_text.split_lines(fragments)):
    if index:
      fitted.append(('', '\n'))
    fitted.extend(menu.fit_line(line, width))
  return fitted


class Selector(component.Component):
  """The single selector component used by core features and extensions."""

  def __init__(self, get_max_height: Callable[[], int] | None = None) -> None:
    self._get_max_height = get_max_height
    # Retain only native scroll state; request content remains in State.
    self._approval_header: tuple[str, containers.Window] | None = None

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    request = state.selector.selector
    if request is None:
      self._approval_header = None
      return containers.Window(height=0)
    choices = state.selector.filtered_choices()
    empty = 'No matching choices.' if state.selector.query else request.empty
    is_resume = request.id == 'resume'
    # An async producer that has not populated yet reserves the full menu box so
    # the overlay does not resize when its results arrive.
    awaiting = (
      request.load is not None and not choices and not state.selector.query
    )
    min_rows = (
      state.selector.menu.max_rows if request.prefiltered or awaiting else 0
    )
    approval = state.approval.request
    is_approval = (
      approval is not None and request.id == f'approval:{approval.call_id}'
    )
    menu_state = state.selector.menu
    limit = self._get_max_height() if self._get_max_height is not None else None
    if limit is not None:
      menu_state = dataclasses.replace(
        menu_state,
        max_rows=max(
          1, min(menu_state.max_rows, limit - (4 if is_approval else 3))
        ),
      )
      # Section headings share the bounded viewport with selectable rows.
      while menu_state.max_rows > 1 and height(
        menu_state, choices
      ) + 1 > limit - int(is_approval):
        menu_state = dataclasses.replace(
          menu_state, max_rows=menu_state.max_rows - 1
        )
      min_rows = min(min_rows, menu_state.max_rows)
    fragments = render_choices(
      menu_state,
      choices,
      title='Tool approval' if is_approval else request.title,
      query=state.selector.query,
      current=request.current,
      empty='' if request.loading else empty,
      stacked=is_resume,
      footer='Tab scope · Enter · Esc' if is_resume else None,
      min_rows=min_rows,
      numbered=request.numbered,
      width=state.layout.columns,
    )
    if is_resume:
      fragments = [
        *resume.scope_tabs(state.resume.scope, state.layout.columns),
        ('', '\n'),
        *fragments,
      ]
    if is_approval:
      header = rendering.strip_ansi(
        rendering.renderable_to_ansi(
          rendering.tool_block(text.Text(request.title), wrap=True),
          state.layout.columns,
          theme_name=themes.current(state.extensions),
        )
      ).rstrip()
      header_height = len(header.splitlines())
      if limit is not None:
        menu_window = menu.window(fragments, state.layout.columns)
        menu_height = menu_window.preferred_height(
          state.layout.columns, limit
        ).preferred
        cached = self._approval_header
        if cached is None or cached[0] != request.id:
          window = containers.Window(wrap_lines=False)
          window.content = controls.FormattedTextControl(
            show_cursor=False,
            get_cursor_position=lambda: data_structures.Point(
              0, window.vertical_scroll
            ),
          )
          self._approval_header = (request.id, window)
        else:
          window = cached[1]
        control = window.content
        assert isinstance(control, controls.FormattedTextControl)
        header_fragments: StyleAndTextTuples = [
          ('class:footer-approval-manual', header)
        ]
        control.text = header_fragments
        window.height = dimension.Dimension.exact(
          max(0, min(header_height, limit - menu_height))
        )
        return containers.HSplit([
          window,
          menu_window,
        ])
      fragments = [('class:footer-approval-manual', header + '\n'), *fragments]
    return menu.window(
      fragments,
      state.layout.columns,
      theme_name=themes.current(state.extensions),
    )


class SearchInput:
  """Retained input bridge that publishes search actions, never selection state."""

  def __init__(self, changed: Callable[[str], object]) -> None:
    def on_change(value: buffer.Buffer) -> None:
      changed(value.text)

    self.buffer = buffer.Buffer(multiline=False, on_text_changed=on_change)
    self.control = controls.BufferControl(
      buffer=self.buffer, input_processors=[processors.BeforeInput('› ')]
    )
