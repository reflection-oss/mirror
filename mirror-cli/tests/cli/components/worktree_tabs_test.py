import dataclasses
import pathlib

import pytest
from freezegun import api as freezegun_api
from prompt_toolkit import formatted_text, utils

from mirror.cli import (
  actions,
  reducer,
  selection,
  state,
  workspace_tags,
)
from mirror.cli.components import breathing_dots, selector, worktree_tabs


@dataclasses.dataclass
class _Workspace:
  label: str
  state: actions.WorktreeAgentState
  identity: str
  remote: bool = False
  path: pathlib.Path | None = None
  available: bool = True

  def show_error(self, message: str) -> None:
    raise AssertionError(message)

  async def close(self) -> None:
    pass


def _state() -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
    supervised=True,
  )


def test_render_worktree_tabs() -> None:
  tabs = (
    _Workspace(
      'default',
      actions.WorktreeAgentState.IDLE,
      remote=False,
      identity='git:default',
    ),
    _Workspace(
      'mirror-1234567890123456789012345678',
      actions.WorktreeAgentState.ATTENTION,
      remote=True,
      identity='remote:other',
    ),
  )

  rendered = worktree_tabs.render_tabs(tabs, tabs[0].identity)
  row = ''.join(text for _, text in rendered).split('\n')[-1]

  assert row.startswith(' default ○   @…')
  assert row.endswith('Tab / Shift+Tab')
  assert ('class:worktree-tab-selected', ' default ') in rendered
  assert ('class:worktree-tab class:worktree-attention ', '!') in rendered
  assert 'more' not in row


def test_footer_fits_the_allocated_width() -> None:
  tabs = (
    _Workspace(
      'private-repo',
      actions.WorktreeAgentState.IDLE,
      identity='git:private',
    ),
    _Workspace('other', actions.WorktreeAgentState.IDLE, identity='git:other'),
  )
  wide = worktree_tabs.render_tabs(tabs, tabs[0].identity, width=44)
  narrow = worktree_tabs.render_tabs(tabs, tabs[0].identity, width=20)

  assert (
    ''.join(text for _, text in wide).split('\n')[-1]
    == ' private-repo ○   other ○   Tab / Shift+Tab'
  )
  assert (
    ''.join(text for _, text in narrow).split('\n')[-1]
    == ' …epo ○   1 more ○ ›'
  )


def test_worktree_component_hides_single_tab() -> None:
  workspace = _Workspace(
    'default', actions.WorktreeAgentState.IDLE, 'git:default'
  )
  assert worktree_tabs.render_tabs((workspace,), workspace.identity) == []
  other = _Workspace('other', actions.WorktreeAgentState.IDLE, 'git:other')
  assert worktree_tabs.render_tabs((workspace, other), workspace.identity)


@pytest.mark.parametrize('elapsed', [0, 0.65])
def test_render_tabs_creating_uses_breathing_dots(
  elapsed: float, frozen_ui_time: freezegun_api.FrozenDateTimeFactory
) -> None:
  tabs = (
    _Workspace(
      'main',
      actions.WorktreeAgentState.CREATING,
      identity='git:main',
    ),
    _Workspace(
      'other',
      actions.WorktreeAgentState.IDLE,
      identity='git:other',
    ),
  )
  frozen_ui_time.tick(elapsed)
  rendered = worktree_tabs.render_tabs(tabs, tabs[0].identity)
  assert rendered[1:3] == breathing_dots.render(
    style='class:worktree-tab-selected class:worktree-running'
  )
  assert '\n' not in formatted_text.fragment_list_to_text(rendered)


def test_workspace_choices_creating_animates_when_rendered(
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  workspace = _Workspace(
    'feature', actions.WorktreeAgentState.CREATING, 'pending:feature'
  )
  choices = worktree_tabs.workspace_choices((workspace,))
  initial = selector.render_choices(
    state.MenuState(), choices, title='Workspaces'
  )
  frozen_ui_time.tick(0.65)
  advanced = selector.render_choices(
    state.MenuState(), choices, title='Workspaces'
  )

  assert '·●  feature' in formatted_text.fragment_list_to_text(initial)
  assert '●·  feature' in formatted_text.fragment_list_to_text(advanced)
  assert (
    'class:menu-selected class:worktree-running class:activity-spinner-highlight',
    '·',
  ) in advanced


@pytest.mark.parametrize(
  ('count', 'selected_index', 'visible', 'before', 'after'),
  [(2, 1, 2, 0, 0), (5, 3, 3, 2, 0), (100, 49, 2, 48, 50)],
)
def test_footer_keeps_one_tab_row_with_many_worktrees(
  count: int, selected_index: int, visible: int, before: int, after: int
) -> None:
  tabs = tuple(
    _Workspace(
      f'codex/feature-{i}',
      actions.WorktreeAgentState.IDLE,
      identity=f'git:{i}',
    )
    for i in range(count)
  )
  rendered = worktree_tabs.render_tabs(
    tabs, tabs[selected_index].identity, width=96
  )
  text = ''.join(text for _, text in rendered).split('\n')[-1]
  assert f'codex/feature-{selected_index}' in text
  assert text.count('codex/feature-') == visible
  assert text.count('more') == bool(before) + bool(after)
  assert f'‹ {before} more ' in text or before == 0
  assert f' {after} more ○ ›' in text or after == 0
  assert '\n' not in text
  assert utils.get_cwidth(text) <= 96


@pytest.mark.parametrize(
  ('selected', 'expected'),
  [
    ('a', ' a ○   b ○   c ○   d ○   6 more ○ ›'),
    ('f', '‹ 4 more ○   e ○   f ○   4 more ○ ›'),
    ('j', '‹ 6 more ○   g ○   h ○   i ○   j ○ '),
  ],
)
def test_overflow_counts_track_each_side_of_the_visible_tabs(
  selected: str, expected: str
) -> None:
  tabs = tuple(
    _Workspace(
      name,
      actions.WorktreeAgentState.IDLE,
      identity=f'git:{name}',
    )
    for name in 'abcdefghij'
  )

  rendered = worktree_tabs.render_tabs(tabs, f'git:{selected}', width=54)

  assert (
    ''.join(text for _, text in rendered).split('\n')[-1]
    == expected + '  Tab / Shift+Tab'
  )
  assert ('class:worktree-tab-selected', f' {selected} ') in rendered
  row = ''.join(text for _, text in rendered)
  assert '\n' not in row
  assert ('class:worktree-tab-selected', f' {selected} ') in rendered


@pytest.mark.parametrize(
  ('first', 'second', 'icon', 'style'),
  [
    (
      actions.WorktreeAgentState.NOT_STARTED,
      actions.WorktreeAgentState.SAVED,
      '·',
      'worktree',
    ),
    (
      actions.WorktreeAgentState.NOT_STARTED,
      actions.WorktreeAgentState.IDLE,
      '○',
      'worktree',
    ),
    (
      actions.WorktreeAgentState.IDLE,
      actions.WorktreeAgentState.CREATING,
      '●',
      'worktree-running',
    ),
    (
      actions.WorktreeAgentState.RUNNING,
      actions.WorktreeAgentState.IDLE,
      '●',
      'worktree-running',
    ),
    (
      actions.WorktreeAgentState.CREATING,
      actions.WorktreeAgentState.RUNNING,
      '●',
      'worktree-running',
    ),
    (
      actions.WorktreeAgentState.RUNNING,
      actions.WorktreeAgentState.UNREAD,
      '✓',
      'worktree-running',
    ),
    (
      actions.WorktreeAgentState.ATTENTION,
      actions.WorktreeAgentState.UNREAD,
      '!',
      'worktree-attention',
    ),
    (
      actions.WorktreeAgentState.ATTENTION,
      actions.WorktreeAgentState.FAILED,
      '×',
      'worktree-failed',
    ),
    (
      actions.WorktreeAgentState.FAILED,
      actions.WorktreeAgentState.ATTENTION,
      '×',
      'worktree-failed',
    ),
  ],
)
def test_overflow_status_uses_priority_and_only_its_hidden_destinations(
  first: actions.WorktreeAgentState,
  second: actions.WorktreeAgentState,
  icon: str,
  style: str,
) -> None:
  tabs = (
    _Workspace('before-a', first, identity='git:a'),
    _Workspace('before-b', second, identity='git:b'),
    _Workspace(
      'current',
      actions.WorktreeAgentState.FAILED,
      identity='git:current',
    ),
    _Workspace('after-a', actions.WorktreeAgentState.IDLE, identity='git:d'),
    _Workspace('after-b', actions.WorktreeAgentState.RUNNING, identity='git:e'),
  )

  rendered = worktree_tabs.render_tabs(tabs, tabs[2].identity, width=35)

  assert (
    ''.join(text for _, text in rendered).split('\n')[-1]
    == f'‹ 2 more {icon}   current ×   2 more ● ›'
  )
  assert rendered[0] == ('class:worktree-tab', '‹ 2 more ')
  assert rendered[1] == (f'class:worktree-tab class:{style} ', icon)
  assert rendered[-3] == ('class:worktree-tab', ' 2 more ')
  assert rendered[-2] == ('class:worktree-tab class:worktree-running ', '●')


@pytest.mark.parametrize(
  ('width', 'hidden'), [(14, 1), (20, 1), (40, 1), (80, 0)]
)
@pytest.mark.parametrize(
  'agent_state',
  [actions.WorktreeAgentState.IDLE, actions.WorktreeAgentState.CREATING],
)
def test_footer_reserves_the_count_when_truncating_wide_labels(
  width: int,
  hidden: int,
  agent_state: actions.WorktreeAgentState,
) -> None:
  tabs = (
    _Workspace(
      '界' * 40,
      agent_state,
      identity='git:main',
    ),
    _Workspace('other', actions.WorktreeAgentState.IDLE, identity='git:other'),
  )
  rendered = worktree_tabs.render_tabs(tabs, tabs[0].identity, width=width)
  text = ''.join(text for _, text in rendered).split('\n')[-1]
  assert text.count('›') == hidden
  assert '\n' not in text
  assert utils.get_cwidth(text) <= width


@pytest.mark.parametrize('selected', [0, 5, 10])
def test_navigation_hint_remains_visible_with_overflow(selected: int) -> None:
  tabs = tuple(
    _Workspace(
      f'feature-{index}',
      actions.WorktreeAgentState.IDLE,
      identity=f'git:{index}',
    )
    for index in range(11)
  )

  rendered = worktree_tabs.render_tabs(tabs, tabs[selected].identity, width=80)
  text = ''.join(text for _, text in rendered).split('\n')[-1]

  assert text.endswith('  Tab / Shift+Tab')
  assert '/workspaces' not in text
  assert 'more' in text
  assert f'feature-{selected}' in text
  assert utils.get_cwidth(text) <= 80


def test_workspaces_open_shared_selector_at_current_workspace() -> None:
  tabs = (
    _Workspace(
      'main',
      actions.WorktreeAgentState.IDLE,
      identity='git:main',
      path=pathlib.Path('/repo/main'),
    ),
    _Workspace('creating…', actions.WorktreeAgentState.CREATING, 'pending:new'),
    _Workspace(
      'feature',
      actions.WorktreeAgentState.RUNNING,
      identity='git:feature',
      path=pathlib.Path('/repo/feature'),
    ),
    _Workspace(
      'spu',
      actions.WorktreeAgentState.SAVED,
      remote=True,
      identity='remote:spu',
    ),
  )
  opened, _ = reducer.reduce(
    _state(),
    actions.OpenSelector(
      worktree_tabs.workspace_selector(
        tabs, 'git:feature', state.SelectorState()
      )
    ),
  )
  searched, _ = reducer.reduce(opened, actions.SelectorSearchChanged('remote'))

  assert opened.selector.id == 'workspaces'
  assert [choice.value for choice in opened.selector.filtered_choices()] == [
    'git:main',
    'pending:new',
    'git:feature',
    'remote:spu',
  ]
  assert opened.selector.selected_choice() == selection.Choice(
    'git:feature',
    'feature',
    'running · /repo/feature',
    prefix='●',
    prefix_style='class:worktree-running',
  )
  assert [choice.label for choice in searched.selector.filtered_choices()] == [
    '@spu'
  ]


def test_open_workspace_selector_follows_roster_changes() -> None:
  main = _Workspace(
    'main',
    actions.WorktreeAgentState.IDLE,
    identity='git:main',
    path=pathlib.Path('/repo/main'),
  )
  feature = _Workspace(
    'feature',
    actions.WorktreeAgentState.IDLE,
    identity='git:feature',
    path=pathlib.Path('/repo/feature'),
  )
  added = _Workspace(
    'fix',
    actions.WorktreeAgentState.IDLE,
    identity='git:fix',
    path=pathlib.Path('/repo/fix'),
  )
  current, _ = reducer.reduce(
    _state(),
    actions.OpenSelector(
      worktree_tabs.workspace_selector(
        (main, feature), 'git:main', state.SelectorState()
      )
    ),
  )
  current, _ = reducer.reduce(current, actions.MoveSelectorSelection(1))

  running = dataclasses.replace(
    feature, state=actions.WorktreeAgentState.RUNNING
  )
  current, _ = reducer.reduce(
    current,
    actions.OpenSelector(
      worktree_tabs.workspace_selector(
        (main, added, running), 'git:main', current.selector
      )
    ),
  )

  assert [choice.value for choice in current.selector.filtered_choices()] == [
    'git:main',
    'git:fix',
    'git:feature',
  ]
  assert current.selector.selected_choice() == selection.Choice(
    'git:feature',
    'feature',
    'running · /repo/feature',
    prefix='●',
    prefix_style='class:worktree-running',
  )


@pytest.mark.parametrize('query', ['', 'fea'])
def test_removed_highlight_is_preserved_when_roster_changes(query: str) -> None:
  main = _Workspace(
    'main',
    actions.WorktreeAgentState.IDLE,
    identity='git:main',
    path=pathlib.Path('/repo/main'),
  )
  feature = _Workspace(
    'feature',
    actions.WorktreeAgentState.IDLE,
    identity='git:feature',
    path=pathlib.Path('/repo/feature'),
  )
  other = _Workspace(
    'other',
    actions.WorktreeAgentState.IDLE,
    identity='git:other',
    path=pathlib.Path('/repo/other'),
  )
  current, _ = reducer.reduce(
    _state(),
    actions.OpenSelector(
      worktree_tabs.workspace_selector(
        (main, feature, other), 'git:main', state.SelectorState()
      )
    ),
  )
  current, _ = reducer.reduce(current, actions.SelectorSearchChanged(query))
  current, _ = reducer.reduce(current, actions.MoveSelectorSelection(1))

  current, _ = reducer.reduce(
    current,
    actions.OpenSelector(
      worktree_tabs.workspace_selector(
        (main, other), 'git:main', current.selector
      )
    ),
  )
  assert current.selector.query == query
  assert current.selector.selector is not None
  assert [choice.value for choice in current.selector.selector.choices] == [
    'git:main',
    'git:feature',
    'git:other',
  ]
  assert current.selector.selected_choice() == selection.Choice(
    'git:feature',
    'feature',
    'removed',
    prefix='×',
    prefix_style='class:worktree',
  )


@pytest.mark.parametrize('color', [None, '#ffffff'])
def test_selected_tag_keeps_its_highlight_and_icon_only(
  color: str | None,
) -> None:
  tabs = (
    _Workspace('default', actions.WorktreeAgentState.IDLE, 'default'),
    _Workspace('feature', actions.WorktreeAgentState.IDLE, 'feature'),
  )
  tag = workspace_tags.WorkspaceTag('blocked', 'Blocked', color)
  rendered = worktree_tabs.render_tabs(
    tabs, 'feature', tags={'default': tag, 'feature': tag}
  )
  row = ''.join(text for _, text in rendered)
  assert '\n' not in row
  assert '▶' not in row
  assert ('class:worktree-tab-selected', ' feature ') in rendered
  assert ('class:worktree-tab', ' default ') in rendered
  assert 'Blocked' not in row
  assert row.count('■') == 2


def test_single_workspace_shows_its_tag() -> None:
  tab = _Workspace('folder', actions.WorktreeAgentState.IDLE, 'folder-id')
  rendered = worktree_tabs.render_tabs(
    [tab],
    tab.identity,
    tags={'folder-id': workspace_tags.WorkspaceTag('feature', 'Feature')},
  )
  assert (
    ''
    .join(text for _, text in rendered)
    .split('\n')[-1]
    .startswith(' folder ■ ')
  )


def test_tagged_workspace_fits_narrow_view() -> None:
  tab = _Workspace('default', actions.WorktreeAgentState.IDLE, 'default')
  rendered = worktree_tabs.render_tabs(
    [tab],
    tab.identity,
    width=16,
    tags={
      'default': workspace_tags.WorkspaceTag(
        'feature', 'A long feature label', '#ffffff'
      )
    },
  )
  assert (
    utils.get_cwidth(''.join(text for _, text in rendered).split('\n')[-1])
    <= 16
  )
  assert all(style != 'class:worktree-tab-marker' for style, _ in rendered)


def test_wide_selected_label_keeps_its_highlight() -> None:
  tabs = (
    _Workspace('界', actions.WorktreeAgentState.IDLE, 'first'),
    _Workspace('選択', actions.WorktreeAgentState.IDLE, 'selected'),
  )
  rendered = worktree_tabs.render_tabs(tabs, 'selected')
  row = ''.join(text for _, text in rendered)
  assert '\n' not in row
  assert ('class:worktree-tab-selected', ' 選択 ') in rendered
  assert row.startswith(' 界 ○   選択 ○ ')


@pytest.mark.parametrize('icon', ['★', '🚀'])
def test_tab_shows_custom_icon_without_name(icon: str) -> None:
  tab = _Workspace('workspace', actions.WorktreeAgentState.IDLE, 'stable')
  rendered = worktree_tabs.render_tabs(
    [tab],
    'stable',
    width=24,
    tags={
      'stable': workspace_tags.WorkspaceTag(
        'tag', 'Never shown on the tab', icon=icon
      )
    },
  )
  row = ''.join(text for _, text in rendered)
  assert '\n' not in row
  assert row == f' workspace {icon} ○ '
  assert utils.get_cwidth(row) <= 24
