"""Shared, stateless rendering and terminal input for declarative selectors."""

import dataclasses
from collections.abc import Callable, Sequence
from typing import override

from prompt_toolkit import buffer, data_structures, formatted_text
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls, dimension, processors
from prompt_toolkit.utils import get_cwidth
from rich import cells, console, text

from mirror.cli import rendering, selection, state, themes
from mirror.cli.composer import processors as composer_processors
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


def _label_suffix(label: str, width: int) -> str:
  """Fit a label suffix using the terminal renderer's cell measurements."""
  offset = len(label)
  remaining = max(0, width)
  # Rich supplies grapheme boundaries, but its cell widths differ from the UI's.
  for start, stop, _ in reversed(cells.split_graphemes(label)[0]):
    size = get_cwidth(label[start:stop])
    if size > remaining:
      break
    offset = start
    remaining -= size
  return label[offset:]


def height(
  menu_state: state.MenuState,
  choices: Sequence[selection.Choice],
  *,
  pinned_count: int = 0,
  stacked: bool = False,
) -> int:
  scrolling = choices[:-pinned_count] if pinned_count else choices
  view = _MENU.view(menu_state, scrolling)
  explicit_sections = {
    choice.section for choice in choices if choice.is_heading
  }
  headings = sum(
    1 + int(index > view.start)
    for index, choice in enumerate(view.visible, view.start)
    if choice.section
    and choice.section not in explicit_sections
    and (index == view.start or choice.section != choices[index - 1].section)
  )
  return (
    _MENU.height(menu_state, scrolling, show_empty=menu_state.active)
    + headings
    + pinned_count
    + sum(
      choice.is_heading + (bool(choice.detail) and stacked)
      for choice in view.visible
    )
    + sum(
      choice.is_heading + (bool(choice.detail) and stacked)
      for choice in choices[len(scrolling) :]
    )
  )


def render_choices(
  menu_state: state.MenuState,
  choices: Sequence[selection.Choice],
  *,
  title: formatted_text.AnyFormattedText,
  query: str = '',
  current: str | None = None,
  empty: str = 'No matching choices.',
  header_suffix: str | None = None,
  min_rows: int = 0,
  numbered: bool = False,
  pinned_count: int = 0,
  width: int | None = None,
  stacked: bool = False,
  wrap_details: bool = False,
  footer: str | None = None,
  truncate_labels_from_start: bool = False,
  minimal_chrome: bool = False,
) -> StyleAndTextTuples:
  row_budget = menu_state.max_rows
  wrapping_console = console.Console() if wrap_details else None
  scrolling = choices[:-pinned_count] if pinned_count else choices
  numbered_from = len(scrolling) if pinned_count else 0
  view = _MENU.view(menu_state, scrolling)
  explicit_sections = {
    choice.section for choice in choices if choice.is_heading
  }
  label_widths: dict[str, int] = {}
  for choice in choices:
    if choice.is_heading:
      continue
    label_width = max(
      label_widths.get(choice.section, 0), get_cwidth(choice.label)
    )
    if width is not None:
      available = width - 4
      if choice.status is not None:
        available -= get_cwidth(choice.status) + 2
        available -= (
          formatted_text.fragment_list_width(
            formatted_text.to_formatted_text(choice.prefix)
          )
          + (2 if choice.prefix else 0)
          + (2 if current else 0)
        )
      label_width = min(
        label_width,
        max(0, available)
        if stacked or width < 60
        else max(0, (width - 4) // 2),
      )
    label_widths[choice.section] = label_width
  label_width = max(label_widths.values(), default=0)

  shortcut_width = max(
    (get_cwidth(choice.shortcut) for choice in choices), default=0
  )
  # Keep the command and description usable before reserving shortcut space.
  if width is None or stacked or width < label_width + shortcut_width + 8:
    shortcut_width = 0

  def heading(choice: selection.Choice, index: int) -> str:
    if choice.section in explicit_sections:
      return ''
    if index != view.start and choice.section == choices[index - 1].section:
      return ''
    label = choice.section_label or choice.section
    if width is not None and get_cwidth(label) > width - 2:
      if choice.section_label:
        return (
          formatted_text.fragment_list_to_text(
            menu.fit_line([('', label)], max(0, width - 3))
          )
          + '…'
        )
      return '…' + _label_suffix(label, max(0, width - 3))
    return label

  def row(
    choice: selection.Choice, index: int, selected: bool
  ) -> StyleAndTextTuples:
    if pinned_count:
      selected = index == min(menu_state.selected, len(choices) - 1)
    style = 'class:menu-selected' if selected else 'class:menu-item'
    label = choice.label
    label_width = label_widths.get(choice.section, 0)
    if choice.is_heading:
      style = 'class:menu-selected' if selected else 'class:menu-section'
      label_width = get_cwidth(label)
      if width is not None:
        label_width = min(
          label_width,
          max(0, width - 9 - get_cwidth(choice.detail) - (2 if current else 0)),
        )
    row_label_width = label_width
    if (
      numbered
      and numbered_from <= index < numbered_from + 9
      and width is not None
    ):
      row_label_width = max(
        label_width, min(get_cwidth(label), max(0, width - 7))
      )
    if width is not None and not choice.is_heading and (stacked or width < 60):
      prefix_width = formatted_text.fragment_list_width(
        formatted_text.to_formatted_text(choice.prefix)
      )
      row_label_width = min(
        get_cwidth(label),
        max(
          0,
          width
          - 4
          - (prefix_width + 2 if prefix_width else 0)
          - (2 if current else 0)
          - (get_cwidth(choice.status) + 2 if choice.status is not None else 0),
        ),
      )
    if get_cwidth(label) > row_label_width:
      if truncate_labels_from_start or choice.is_heading:
        label = '…' + _label_suffix(label, row_label_width - 1)
      else:
        label = (
          formatted_text.fragment_list_to_text(
            menu.fit_line([('', label)], max(0, row_label_width - 1))
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
      (
        style,
        f'{index - numbered_from + 1}. '
        if numbered and numbered_from <= index < numbered_from + 9
        else '',
      ),
      *prefix,
    ]
    label_column = formatted_text.fragment_list_width(fragments)
    fragments.extend((*highlight(label, label_query, style), *status))
    detail = choice.detail
    content_width = (
      (width or 0) - 2 - (shortcut_width + 2 if shortcut_width else 0)
    )
    detail_column = (
      sum(get_cwidth(fragment[1]) for fragment in fragments) + len(padding) + 2
    )
    detail_width = content_width - (label_column if stacked else detail_column)
    if width is not None and wrapping_console is None:
      if detail_width < (1 if stacked or choice.is_heading else 8):
        detail = ''
      elif get_cwidth(detail) > detail_width:
        detail = (
          formatted_text.fragment_list_to_text(
            menu.fit_line([('', detail)], detail_width - 1)
          )
          + '…'
        )
    if (
      wrapping_console is not None
      and not stacked
      and detail_width >= 2
      and index < len(scrolling)
    ):
      lines = text.Text(detail).wrap(wrapping_console, detail_width)
      # A long detail must not push fixed actions outside the viewport.
      if pinned_count and len(lines) > row_budget:
        lines = lines[:row_budget]
        lines[-1] = text.Text('…' + _label_suffix(detail, detail_width - 1))
      detail = ('\n' + ' ' * detail_column).join(line.plain for line in lines)
    fragments.extend((
      *highlight(
        (f'\n{" " * label_column}{detail}' if detail else '')
        if stacked
        else (f'{padding}  {detail}' if detail else ''),
        '' if label_query else query,
        detail_style,
      ),
      (style, '  (current)' if choice.value == current else ''),
    ))
    spacing: StyleAndTextTuples = [('', '\n')] if choice.is_heading else []
    if width is None:
      return spacing + fragments
    fitted = spacing
    for index, line in enumerate(formatted_text.split_lines(fragments)):
      if index:
        fitted.append(('', '\n'))
      line = menu.fit_line(line, max(0, content_width))
      used = sum(get_cwidth(fragment[1]) for fragment in line)
      fitted.extend(line)
      fitted.append((style, ' ' * max(0, width - 2 - shortcut_width - used)))
      if shortcut_width:
        shortcut = choice.shortcut if index == 0 else ''
        shortcut_style = f'{style} bold' if selected else 'class:ui.accent bold'
        fitted.append((shortcut_style, shortcut))
        fitted.append((
          style,
          ' ' * (shortcut_width - get_cwidth(shortcut)),
        ))
    return fitted

  # Wrapped descriptions share a fixed line budget; unused lines stay padded.
  while wrap_details:
    used_rows = sum(
      1
      + formatted_text.fragment_list_to_text(
        row(item, index, index == view.selected)
      ).count('\n')
      for index, item in enumerate(view.visible, start=view.start)
    )
    if used_rows <= row_budget or menu_state.max_rows == 1:
      # `min_rows` counts items, so add only the unused physical lines.
      min_rows = max(min_rows, len(view.visible) + row_budget - used_rows)
      break
    menu_state = dataclasses.replace(
      menu_state, max_rows=menu_state.max_rows - 1
    )
    view = _MENU.view(menu_state, scrolling)

  fragments = _MENU.render(
    menu_state,
    scrolling,
    row,
    title=title,
    header_suffix=header_suffix,
    empty_row=[('class:menu-item', empty)],
    min_rows=min_rows,
    pinned_rows=tuple(
      row(choice, index, index == menu_state.selected)
      for index, choice in enumerate(
        choices[len(scrolling) :], start=len(scrolling)
      )
    ),
    footer=footer
    if footer is not None
    else (
      '1–9 to select · ↑↓/Enter · Esc to cancel'
      if numbered
      else ('Enter · Esc' if width is not None and width < 42 else None)
    ),
    heading=heading,
    width=width,
    minimal_chrome=minimal_chrome,
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

  @override
  def cache_key(self, state: state.State) -> object | None:
    return () if state.selector.selector is None else None

  def __init__(
    self,
    get_max_height: Callable[[], int] | None = None,
    *,
    inset: bool = True,
    minimal_chrome: bool = False,
  ) -> None:
    self._get_max_height = get_max_height
    self._inset = inset
    self._minimal_chrome = minimal_chrome
    # Cache only the current header's wrapping and native scroll position.
    self._header: tuple[str, containers.Window] | None = None
    self._wrapped_header: (
      tuple[
        console.RenderableType,
        int,
        str | themes.Theme,
        formatted_text.ANSI,
        int,
      ]
      | None
    ) = None

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    request = state.selector.selector
    if request is None:
      self._header = None
      self._wrapped_header = None
      return containers.Window(height=0)
    choices = state.selector.filtered_choices()
    empty = 'No matching choices.' if state.selector.query else request.empty
    is_resume = request.id == 'resume'
    stacked = request.stacked or is_resume
    # An async producer that has not populated yet reserves the full menu box so
    # the overlay does not resize when its results arrive.
    awaiting = (
      (request.load is not None or request.loading)
      and not choices
      and not state.selector.query
    )
    min_rows = (
      state.selector.menu.max_rows if request.prefiltered or awaiting else 0
    )
    has_header = bool(request.header)
    menu_state = state.selector.menu
    limit = self._get_max_height() if self._get_max_height is not None else None
    if limit is not None:
      menu_state = dataclasses.replace(
        menu_state,
        max_rows=max(
          1,
          min(
            menu_state.max_rows,
            limit - (4 if has_header else 3) - request.pinned_count,
          ),
        ),
      )
      # Section headings share the bounded viewport with selectable rows.
      while menu_state.max_rows > 1 and height(
        menu_state, choices, pinned_count=request.pinned_count, stacked=stacked
      ) + 1 > limit - int(has_header):
        menu_state = dataclasses.replace(
          menu_state, max_rows=menu_state.max_rows - 1
        )
      min_rows = min(min_rows, menu_state.max_rows)
    fragments = render_choices(
      menu_state,
      choices,
      title=request.title,
      query=state.selector.query,
      current=request.current,
      empty='' if request.loading else empty,
      stacked=stacked,
      wrap_details=request.wrap_details,
      footer=request.footer
      if request.footer is not None
      else ('Tab scope · Enter · Esc' if is_resume else None),
      min_rows=min_rows,
      numbered=request.numbered,
      pinned_count=request.pinned_count,
      width=state.layout.columns,
      truncate_labels_from_start=request.truncate_labels_from_start,
      minimal_chrome=self._minimal_chrome,
    )
    if is_resume:
      fragments = [
        *resume.scope_tabs(state.resume.scope, state.layout.columns),
        ('', '\n'),
        *fragments,
      ]
    width = state.layout.columns
    theme = themes.current(state.extensions, state.extension_themes)
    menu_window = menu.window(
      fragments, width, theme_name=theme, inset=self._inset
    )
    if not has_header:
      return menu_window
    cached_header = self._wrapped_header
    if cached_header is None or cached_header[:3] != (
      request.header,
      width,
      theme,
    ):
      # Wrapping large payloads must not run again on every animation frame.
      body = (
        rendering.tool_block(
          text.Text(request.header, style='ui.warning'), wrap=True
        )
        if isinstance(request.header, str)
        else request.header
      )
      rendered = rendering.renderable_to_ansi(
        body, width, theme_name=theme
      ).rstrip()
      cached_header = self._wrapped_header = (
        request.header,
        width,
        theme,
        formatted_text.ANSI(rendered),
        len(rendered.splitlines()),
      )
    header, header_height = cached_header[3:]
    available = (
      max(0, limit - menu_window.preferred_height(width, limit).preferred)
      if limit is not None
      else header_height
    )
    overflow = header_height > available
    page_size = max(0, min(header_height, available - int(overflow)))
    cached = self._header
    if cached is None or cached[0] != request.id:
      window = containers.Window(wrap_lines=False)
      window.content = controls.FormattedTextControl(
        show_cursor=False,
        get_cursor_position=lambda: data_structures.Point(
          0, window.vertical_scroll
        ),
      )
      self._header = (request.id, window)
    else:
      window = cached[1]
    control = window.content
    assert isinstance(control, controls.FormattedTextControl)
    control.text = header
    window.height = dimension.Dimension.exact(page_size)
    max_scroll = max(0, header_height - max(1, page_size))
    window.vertical_scroll = min(window.vertical_scroll, max_scroll)
    bindings = key_bindings.KeyBindings()

    @bindings.add('pageup', is_global=True)
    @bindings.add('pagedown', is_global=True)
    def page_header(event: key_processor.KeyPressEvent) -> None:
      direction = -1 if event.key_sequence[-1].key == 'pageup' else 1
      window.vertical_scroll = max(
        0,
        min(
          max_scroll, window.vertical_scroll + direction * max(1, page_size - 1)
        ),
      )

    hint = (
      f'PgUp/Dn · {window.vertical_scroll + 1}–{window.vertical_scroll + page_size}/{header_height}'
      if page_size
      else 'Enlarge terminal to inspect details'
    )
    return containers.HSplit(
      [
        window,
        containers.Window(
          controls.FormattedTextControl([('class:ui.muted', f'  {hint}')]),
          height=int(overflow),
        ),
        menu_window,
      ],
      key_bindings=bindings,
    )


def search_hint(state: state.State) -> StyleAndTextTuples | None:
  """The open selector's placeholder, while nothing is typed in its search."""
  selector = state.selector.selector
  if selector is None or not selector.placeholder or state.selector.query:
    return None
  return [('class:slash-argument-placeholder', selector.placeholder)]


class SearchInput:
  """Retained input bridge that publishes search actions, never selection state."""

  def __init__(
    self,
    changed: Callable[[str], object],
    preview: Callable[[], StyleAndTextTuples | None],
  ) -> None:
    def on_change(value: buffer.Buffer) -> None:
      changed(value.text)

    self.buffer = buffer.Buffer(multiline=False, on_text_changed=on_change)
    # The hint goes in before the prompt arrow so it lands at the cursor.
    self.control = controls.BufferControl(
      buffer=self.buffer,
      input_processors=[
        composer_processors.CompletionPreviewProcessor(preview),
        processors.BeforeInput('› '),
      ],
    )
