"""Compact worktree tabs for the Mirror footer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from prompt_toolkit import formatted_text, utils
from prompt_toolkit.formatted_text import StyleAndTextTuples

from mirror.cli import (
  actions,
  selection,
  state,
  workspace_tags,
  workspaces,
)
from mirror.cli.components import breathing_dots

_HINT = 'Tab / Shift+Tab'


def render_tabs(
  tabs: Sequence[workspaces.Workspace],
  selected: str,
  *,
  width: int = 100,
  tags: Mapping[str, workspace_tags.WorkspaceTag] | None = None,
) -> StyleAndTextTuples:
  """Render one tab row with a colored overline above the selected tab."""
  tabs = tuple(tab for tab in tabs if tab.available)
  if not tabs or (len(tabs) < 2 and not tags):
    return []
  selected_index = next(
    (index for index, tab in enumerate(tabs) if tab.identity == selected), 0
  )
  minimum = (
    _overflow_width(selected_index, compact=True)
    + _overflow_width(len(tabs) - selected_index - 1, compact=True)
    + 5
  )
  # Reserve navigation before fitting tabs, unless even a minimal tab cannot fit.
  show_hint = len(tabs) > 1 and width >= minimum + len(_HINT) + 2
  if show_hint:
    width -= len(_HINT) + 2
  rows = [
    _tab_fragments(
      tab, selected=index == selected_index, tag=(tags or {}).get(tab.identity)
    )
    for index, tab in enumerate(tabs)
  ]
  widths = [
    sum(utils.get_cwidth(fragment[1]) for fragment in row) for row in rows
  ]
  compact = (
    _overflow_width(selected_index)
    + _overflow_width(len(tabs) - selected_index - 1)
    + 5
    > width
  )
  start, end = _visible_range(widths, selected_index, width, compact=compact)
  before = _overflow_tabs(tabs[:start], before=True, compact=compact)
  after = _overflow_tabs(tabs[end:], before=False, compact=compact)
  if end - start == 1:
    reserved = (
      widths[start]
      - utils.get_cwidth(rows[start][0][1])
      + 2
      + _overflow_width(start, compact=compact)
      + _overflow_width(len(tabs) - end, compact=compact)
    )
    rows[start] = _tab_fragments(
      tabs[start],
      selected=True,
      label_width=max(0, width - reserved),
      tag=(tags or {}).get(tabs[start].identity),
    )
  fragments = list(before)
  for index in range(start, end):
    row = rows[index]
    if fragments:
      fragments.append(('', ' '))
    fragments.extend(row)
  if after:
    fragments.extend((('', ' '), *after))
  if show_hint:
    fragments.extend((('', '  '), ('class:worktree-hint', _HINT)))
  return fragments


def _tab_fragments(
  tab: workspaces.Workspace,
  *,
  selected: bool,
  label_width: int = 28,
  tag: workspace_tags.WorkspaceTag | None = None,
) -> StyleAndTextTuples:
  style = 'class:worktree-tab-selected' if selected else 'class:worktree-tab'
  label = _short_label(
    tab.label,
    max(0, label_width - utils.get_cwidth(('@' if tab.remote else ''))),
  )
  return [
    (style, f' {("@" if tab.remote else "")}{label} '),
    *(
      [(f'{style} fg:{tag.color}' if tag.color else style, f'{tag.icon} ')]
      if tag
      else []
    ),
    *_status(tab.state, style),
  ]


def _visible_range(
  widths: Sequence[int], selected: int, width: int, *, compact: bool
) -> tuple[int, int]:
  """Find the longest contiguous row containing the selected tab and its overflow count."""
  if sum(widths) + len(widths) - 1 <= width:
    return 0, len(widths)
  prefix = [0]
  for size in widths:
    prefix.append(prefix[-1] + size)
  best = (selected, selected + 1)
  for start in range(selected + 1):
    minimum_end = max(selected, start + best[1] - best[0])
    for end in range(len(widths), minimum_end, -1):
      needed = (
        prefix[end]
        - prefix[start]
        + end
        - start
        - 1
        + _overflow_width(start, compact=compact)
        + _overflow_width(len(widths) - end, compact=compact)
      )
      if needed <= width:
        best = (start, end)
        break
  return best


def _overflow_width(count: int, *, compact: bool = False) -> int:
  if not count:
    return 0
  label = str(count) if compact else f'{count} more'
  return utils.get_cwidth(f' {label} ○ ›') + 1


def _overflow_tabs(
  tabs: Sequence[workspaces.Workspace], *, before: bool, compact: bool
) -> StyleAndTextTuples:
  """Summarize hidden destinations with their highest-priority status."""
  if not tabs:
    return []
  priority = (
    actions.WorktreeAgentState.NOT_STARTED,
    actions.WorktreeAgentState.SAVED,
    actions.WorktreeAgentState.IDLE,
    actions.WorktreeAgentState.CREATING,
    actions.WorktreeAgentState.RUNNING,
    actions.WorktreeAgentState.UNREAD,
    actions.WorktreeAgentState.ATTENTION,
    actions.WorktreeAgentState.FAILED,
  )
  agent = max((tab.state for tab in tabs), key=priority.index)
  if agent is actions.WorktreeAgentState.SAVED:
    agent = actions.WorktreeAgentState.NOT_STARTED
  elif agent is actions.WorktreeAgentState.CREATING:
    agent = actions.WorktreeAgentState.RUNNING
  icon, status_style = _indicator(agent)
  label = str(len(tabs)) if compact else f'{len(tabs)} more'
  return [
    ('class:worktree-tab', f'‹ {label} ' if before else f' {label} '),
    *formatted_text.to_formatted_text(
      icon, style=f'class:worktree-tab {status_style}'
    ),
    ('class:worktree-tab', ' ' if before else ' ›'),
  ]


def _short_label(label: str, width: int) -> str:
  if width <= 0:
    return ''
  if utils.get_cwidth(label) <= width:
    return label
  while utils.get_cwidth(label) > width - 1:
    label = label[1:]
  return f'…{label}'


def workspace_choices(
  tabs: Sequence[workspaces.Workspace],
) -> tuple[selection.Choice, ...]:
  """Offer each switchable worktree and remote to the shared selector."""
  return tuple(_workspace_choice(tab) for tab in tabs if tab.available)


def _workspace_choice(tab: workspaces.Workspace) -> selection.Choice:
  icon, status_style = _indicator(tab.state)
  status = tab.state.name.lower().replace('_', ' ')
  if isinstance(tab, workspaces.AgentWorkspace) and not tab.remote:
    kind = 'worktree workspace' if tab.managed else 'folder workspace'
    status = f'{kind} · {status}'
  return selection.Choice(
    tab.identity,
    f'{("@" if tab.remote else "")}{tab.label}',
    f'{status} · {tab.path if not tab.remote and tab.path else tab.identity}',
    prefix=icon,
    prefix_style=status_style,
  )


def _status(
  agent: actions.WorktreeAgentState, tab_style: str
) -> StyleAndTextTuples:
  icon, status_style = _indicator(agent)
  return [
    *formatted_text.to_formatted_text(
      icon, style=f'{tab_style} {status_style}'
    ),
    (tab_style, ' '),
  ]


def _indicator(
  agent: actions.WorktreeAgentState,
) -> tuple[formatted_text.AnyFormattedText, str]:
  return {
    actions.WorktreeAgentState.CREATING: (
      breathing_dots.render,
      'class:worktree-running',
    ),
    actions.WorktreeAgentState.NOT_STARTED: ('·', 'class:worktree'),
    actions.WorktreeAgentState.SAVED: ('↻', 'class:worktree'),
    actions.WorktreeAgentState.IDLE: ('○', 'class:worktree'),
    actions.WorktreeAgentState.RUNNING: ('●', 'class:worktree-running'),
    actions.WorktreeAgentState.ATTENTION: ('!', 'class:worktree-attention'),
    actions.WorktreeAgentState.UNREAD: ('✓', 'class:worktree-running'),
    actions.WorktreeAgentState.FAILED: ('×', 'class:worktree-failed'),
  }[agent]


def workspace_selector(
  roster: Sequence[workspaces.Workspace],
  current: str,
  previous: state.SelectorState,
) -> selection.Selector[actions.Action]:
  """Render choices directly, preserving the highlighted row when it disappears."""
  choices = workspace_choices(roster)
  opened = previous.selector
  highlighted = (
    previous.selected_choice()
    if opened is not None and opened.id == 'workspaces'
    else None
  )
  if (
    opened is not None
    and highlighted is not None
    and not any(choice.value == highlighted.value for choice in choices)
  ):
    removed = selection.Choice(
      highlighted.value,
      highlighted.label,
      'removed',
      prefix='×',
      prefix_style='class:worktree',
    )
    position = min(opened.choices.index(highlighted), len(choices))
    choices = (*choices[:position], removed, *choices[position:])
  return selection.Selector(
    id='workspaces',
    title='Workspaces',
    choices=choices,
    on_select=actions.SelectWorkspaceRequested,
    current=current,
    selected=highlighted.value if highlighted is not None else current,
    empty='No workspaces available.',
    keep_open=True,
  )
