"""Stateless running-task component."""

from __future__ import annotations

from typing import override

from prompt_toolkit import utils
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import state
from mirror.cli.components import component

_ELLIPSIS = '…'


class Tasks(component.Component):
  """Running background tasks shown below the prompt."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    """Render task rows from immutable application state."""
    return containers.Window(
      controls.FormattedTextControl(
        _content(state),
        focusable=True,
        show_cursor=False,
      ),
      height=dimension.Dimension.exact(_height(state)),
    )


def _height(state: state.State) -> int:
  if state.status_bar.view_mode.value == 'extra-compact':
    return 0
  return len(state.tasks.tasks) + 1 if state.tasks.tasks else 0


def _content(state: state.State) -> StyleAndTextTuples:
  if not state.tasks.tasks:
    return []
  fragments: StyleAndTextTuples = [('class:task-header', '  Tasks\n')]
  for index, task in enumerate(state.tasks.tasks):
    if index:
      fragments.append(('', '\n'))
    connector = '└─' if index == len(state.tasks.tasks) - 1 else '├─'
    tree = f'  {connector} '
    selected = task.id == state.tasks.selected_id
    hint = ' · press x to cancel' if selected else ''
    label = f'{task.tool_name} · {task.id} · {task.display or "running…"}{hint}'
    label_width = state.layout.columns - utils.get_cwidth(tree)
    style = 'class:task-selected' if selected else 'class:task'
    fragments.extend([
      ('class:task-tree', tree),
      (style, _single_line_cells(label, label_width)),
    ])
  return fragments


def _single_line_cells(text: str, max_width: int) -> str:
  """Collapse text to one terminal row bounded by max_width cells."""
  if max_width <= 0:
    return ''
  compact = ' '.join(text.split())
  if utils.get_cwidth(compact) <= max_width:
    return compact
  if max_width <= 1:
    return _ELLIPSIS
  budget = max_width - 1
  kept: list[str] = []
  width = 0
  for char in compact:
    char_width = utils.get_cwidth(char)
    if width + char_width > budget:
      break
    kept.append(char)
    width += char_width
  return ''.join(kept).rstrip() + _ELLIPSIS
