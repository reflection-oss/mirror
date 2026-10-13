from __future__ import annotations

import asyncio
import dataclasses
import pathlib
from collections.abc import Mapping

import pytest

from mirror import config
from mirror.cli import (
  actions,
  commands,
  extension,
  extension_state,
  models,
  reducer,
  selection,
  state,
  workspace_controller,
)
from mirror.cli.extensions.memory import extension as memory_extension
from mirror.cli.extensions.settings import extension as settings_extension
from mirror.cli.extensions.settings import setting_template
from mirror.cli.extensions.settings.definitions import history_detail

_EXAMPLE_MENU = setting_template.Menu(
  'settings',
  'Settings',
  '',
  (
    setting_template.Menu(
      'advanced',
      'Advanced',
      'Advanced settings',
      (
        history_detail.SETTING,
        setting_template.Menu(
          'tools',
          'Tools',
          'Tool actions',
          (
            setting_template.Action(
              'notice', 'Show notice', '', actions.ShowText('Notice')
            ),
          ),
        ),
      ),
    ),
  ),
)


@dataclasses.dataclass
class _IdleAgent:
  """An initial workspace agent that settings must never start or call."""

  workspace: pathlib.Path
  reasoning_effort: str = 'medium'
  approval_mode: str = 'auto'
  model: str = 'model'
  model_selection: models.ModelOption = models.ModelOption('model', 'model')
  conversation_id: str = 'conversation'
  activity_status: str = ''
  queued_message_count: int = 0
  worktree_state: actions.WorktreeAgentState = (
    actions.WorktreeAgentState.NOT_STARTED
  )

  def transcript_entries(self) -> tuple[()]:
    pytest.fail('Unexpected transcript access')

  def submit_instruction(self, instruction: str) -> None:
    del instruction  # Unused.
    pytest.fail('Unexpected instruction')

  def show_error(self, message: str) -> None:
    pytest.fail(f'Unexpected agent error: {message}')

  async def start(self) -> None:
    pytest.fail('Unexpected agent start')

  async def close(self) -> None:
    pytest.fail('Unexpected agent close')


@dataclasses.dataclass
class _Lifecycle:
  """Keep settings in a scratch home without changing extension lifetimes."""

  root: pathlib.Path
  registry: extension.ExtensionRegistry

  async def set_enabled(
    self, directory: pathlib.Path, enabled: bool
  ) -> str | None:
    del directory, enabled  # Unused.
    pytest.fail('Unexpected extension enablement change')

  async def reload(self, extension_id: str) -> str | None:
    del extension_id  # Unused.
    pytest.fail('Unexpected extension reload')


@dataclasses.dataclass
class _Harness:
  """Settings effects with real selector reduction and scratch persistence."""

  plugin: settings_extension.SettingsExtension
  context: extension.Context
  dispatched: list[actions.Action]
  restarts: list[Mapping[str, object]]
  settings: config.Config


def _harness(
  tmp_path: pathlib.Path,
  *,
  memory_available: bool = True,
  remote: bool = False,
  busy: bool = False,
  approval_mode: str = 'auto',
) -> _Harness:
  """Build an isolated context; only worker restarts are recorded, not run."""
  dispatched: list[actions.Action] = []
  restarts: list[Mapping[str, object]] = []
  settings = config.Config(tmp_path / '.mirror' / 'config.json')
  current = state.State(
    activity=state.ActivityState(busy=busy),
    status_bar=state.StatusBarState('model', 'medium', approval_mode),
    session=state.SessionState('conversation', remote=remote),
    lifecycle=state.LifecycleState(active=True),
    extensions=extension_state.ExtensionStates(
      {memory_extension.MemoryExtension.id(): None} if memory_available else {}
    ),
  )

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    dispatched.append(action)
    current, _ = reducer.reduce(current, action)

  async def restart(updates: Mapping[str, object]) -> bool:
    restarts.append(dict(updates))
    return True

  controller = workspace_controller.WorkspaceController(
    {},
    workspace=tmp_path,
    reasoning_effort='medium',
    create_agent=lambda spec, _controller: _IdleAgent(spec.workspace),
    database=tmp_path / 'cli.db',
    settings=settings,
  )
  context = extension.Context(
    extension.Workspace(tmp_path, remote=remote),
    dispatch=dispatch,
    get_state=lambda: current,
    enqueue=lambda _: pytest.fail('Unexpected agent message'),
    refresh_tools=lambda _: pytest.fail('Unexpected tool refresh'),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=controller,
    restart_worker=None if remote else restart,
    extensions=_Lifecycle(
      settings.path.parent / 'extensions', extension.ExtensionRegistry()
    ),
  )
  return _Harness(
    settings_extension.SettingsExtension(),
    context,
    dispatched,
    restarts,
    settings,
  )


def _selector(harness: _Harness) -> selection.Selector[object]:
  current = harness.context.get_state().selector
  assert current.active
  assert current.selector is not None
  return current.selector


async def _run(harness: _Harness, action: object) -> None:
  """Run an action from a command or selector through the extension reducer."""
  assert isinstance(action, actions.RoutedExtensionAction)
  _, (effect,) = harness.plugin.reduce(harness.context.get_state(), action)
  assert isinstance(effect, extension.PrivateEffect)
  await harness.plugin.handle_effect(harness.context, effect.payload)


async def _open_settings(harness: _Harness) -> selection.Selector[object]:
  (command,) = harness.plugin.slash_commands()
  assert command.name == 'settings'
  action = commands.dispatch(
    '/settings',
    commands.CommandContext(
      'model',
      'medium',
      'conversation',
      harness.context.get_state().activity.busy,
      str(harness.context.workspace.cwd),
      remote=harness.context.workspace.remote,
    ),
    (command,),
  )
  assert action == actions.RoutedExtensionAction(
    'core/settings', settings_extension.Show()
  )
  await _run(harness, action)
  return _selector(harness)


async def _select(
  harness: _Harness, key: str, delta: int | None = None
) -> None:
  """Activate the visible selector's callback, preserving Enter vs. arrows."""
  request = _selector(harness)
  if delta is None:
    action = request.on_select(key)
  else:
    assert request.on_adjust is not None
    action = request.on_adjust(key, delta)
  await _run(harness, action)


@pytest.mark.asyncio
@pytest.mark.parametrize('delta', [None, 1], ids=['enter', 'right'])
async def test_handle_effect_root_memory_opens_submenu_without_toggling(
  tmp_path: pathlib.Path, delta: int | None
) -> None:
  harness = _harness(tmp_path)
  root = await _open_settings(harness)
  assert root.id == 'settings'
  assert root.keep_open
  assert [choice.value for choice in root.choices] == [
    'analytics',
    'memory',
    'history-detail',
    'permissions',
  ]
  assert root.choices[1].prefix == ''
  assert root.choices[1].status == '          >'
  assert root.choices[2].label.strip() == 'Conversation detail'
  assert root.choices[2].status == '< Compact >'
  assert root.choices[3].label.strip() == 'CLI mode'
  assert root.choices[3].status == '< Auto    >'

  await _select(harness, 'memory', delta)

  submenu = _selector(harness)
  assert submenu.id == 'settings/memory'
  assert submenu.extension_id == 'core/settings'
  assert submenu.keep_open
  assert [choice.value for choice in submenu.choices] == [
    'memory',
    'memory-import',
  ]
  assert submenu.choices[0].prefix == ''
  assert submenu.choices[0].status == '< Off >'
  assert submenu.choices[1].status == '      >'
  assert not harness.settings.path.exists()
  assert harness.restarts == []


@pytest.mark.asyncio
async def test_handle_effect_root_memory_left_is_noop(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  root = await _open_settings(harness)
  harness.dispatched.clear()

  await _select(harness, 'memory', -1)

  assert _selector(harness) is root
  assert harness.dispatched == []
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('target', 'label', 'memory_dir'),
  [(True, 'On', str(memory_extension.STORE)), (False, 'Off', None)],
)
async def test_handle_effect_submenu_memory_enter_persists_and_restarts(
  tmp_path: pathlib.Path, target: bool, label: str, memory_dir: str | None
) -> None:
  harness = _harness(tmp_path)
  harness.settings.set_extension_config(
    memory_extension.KEY, {'enabled': not target}
  )
  await _open_settings(harness)
  await _select(harness, 'memory')
  harness.dispatched.clear()

  await _select(harness, 'memory')

  assert config.Config(harness.settings.path).read() == {
    'extensions': {'builtin/memory': {'config': {'enabled': target}}}
  }
  assert harness.restarts == [{'memory_dir': memory_dir}]
  assert harness.dispatched[0] == actions.ExtensionConfigChanged(
    'core/memory', {'enabled': target}
  )
  submenu = _selector(harness)
  assert submenu.id == 'settings/memory'
  assert submenu.choices[0].status == f'< {label:3} >'


@pytest.mark.asyncio
async def test_handle_effect_memory_import_closes_and_routes_to_memory(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  await _open_settings(harness)
  await _select(harness, 'memory')
  harness.dispatched.clear()

  await _select(harness, 'memory-import')

  assert harness.dispatched == [
    actions.ClearSelector(),
    actions.RoutedExtensionAction(
      'core/memory', memory_extension.Run(('import',))
    ),
  ]
  assert harness.context.get_state().selector.selector is None
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('delta', [-1, 1], ids=['left', 'right'])
async def test_handle_effect_memory_import_arrows_are_noop(
  tmp_path: pathlib.Path, delta: int
) -> None:
  harness = _harness(tmp_path)
  await _open_settings(harness)
  await _select(harness, 'memory')
  submenu = _selector(harness)
  harness.dispatched.clear()

  await _select(harness, 'memory-import', delta)

  assert _selector(harness) is submenu
  assert harness.dispatched == []
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
async def test_handle_effect_memory_cancel_returns_to_highlighted_root(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  await _open_settings(harness)
  await _select(harness, 'memory')
  cancel = _selector(harness).cancel_action
  assert cancel == actions.RoutedExtensionAction(
    'core/settings', settings_extension.Show(selected='memory')
  )

  harness.context.dispatch(actions.ClearSelector())
  await _run(harness, cancel)

  root = _selector(harness)
  assert root.id == 'settings'
  selected = harness.context.get_state().selector.selected_choice()
  assert selected is not None
  assert selected.value == 'memory'
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
async def test_handle_effect_memory_change_rejected_when_turn_becomes_busy(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  await _open_settings(harness)
  await _select(harness, 'memory')
  submenu = _selector(harness)
  harness.context.dispatch(actions.TurnStarted(observed_at=0))
  harness.dispatched.clear()

  await _select(harness, 'memory')

  assert harness.dispatched == [
    actions.ShowText(
      'The memory setting is not available in this session.',
      style='ui.error',
    ),
  ]
  assert _selector(harness) is submenu
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'replacement',
  [
    actions.ClearSelector(),
    actions.OpenSelector(
      selection.Selector(
        'settings/memory',
        'Replacement memory menu',
        (),
        lambda _: actions.ClearSelector(),
        extension_id='core/settings',
      )
    ),
  ],
  ids=['dismissed', 'replaced-with-same-menu-id'],
)
async def test_handle_effect_slow_restart_preserves_current_selector(
  tmp_path: pathlib.Path,
  replacement: actions.ClearSelector | actions.OpenSelector,
) -> None:
  harness = _harness(tmp_path)
  started = asyncio.Event()
  finish = asyncio.Event()

  async def restart(updates: Mapping[str, object]) -> bool:
    harness.restarts.append(dict(updates))
    started.set()
    await finish.wait()
    return True

  harness.context = dataclasses.replace(harness.context, restart_worker=restart)
  await _open_settings(harness)
  await _select(harness, 'memory')
  async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
    tasks.create_task(_select(harness, 'memory'))
    await started.wait()
    harness.context.dispatch(replacement)
    current = harness.context.get_state().selector.selector
    harness.dispatched.clear()
    finish.set()

  assert harness.context.get_state().selector.selector is current
  assert harness.dispatched == []
  assert harness.restarts == [{'memory_dir': str(memory_extension.STORE)}]
  assert memory_extension.enabled(harness.settings.read())


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('memory_available', 'remote', 'busy', 'expected'),
  [
    (False, False, False, ['analytics', 'history-detail', 'permissions']),
    (True, True, False, ['permissions']),
    (True, False, True, ['permissions']),
    (True, True, True, ['permissions']),
  ],
  ids=['extension-absent', 'remote', 'busy-local', 'busy-remote'],
)
async def test_handle_effect_cli_mode_preserves_other_settings_availability(
  tmp_path: pathlib.Path,
  memory_available: bool,
  remote: bool,
  busy: bool,
  expected: list[str],
) -> None:
  harness = _harness(
    tmp_path, memory_available=memory_available, remote=remote, busy=busy
  )
  root = await _open_settings(harness)
  assert [choice.value for choice in root.choices] == expected
  assert root.choices[-1].label.strip() == 'CLI mode'
  assert root.choices[-1].detail == 'Choose automatic or manual tool approval'
  harness.dispatched.clear()

  await _select(harness, 'permissions')

  assert harness.context.get_state().status_bar.approval_mode == 'approve'
  root = _selector(harness)
  assert root.id == 'settings'
  assert root.keep_open
  assert root.selected == 'permissions'
  assert harness.dispatched == [
    actions.SetApprovalMode('approve'),
    actions.OpenSelector(root),
  ]
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('delta', 'initial_mode', 'expected_mode', 'expected_status'),
  [
    (None, 'auto', 'approve', '< Manual  >'),
    (1, 'approve', 'auto', '< Auto    >'),
    (-1, 'auto', 'approve', '< Manual  >'),
  ],
  ids=['enter', 'right', 'left'],
)
async def test_handle_effect_cli_mode_toggles_inline(
  tmp_path: pathlib.Path,
  delta: int | None,
  initial_mode: str,
  expected_mode: str,
  expected_status: str,
) -> None:
  harness = _harness(tmp_path, approval_mode=initial_mode)
  await _open_settings(harness)
  harness.dispatched.clear()

  await _select(harness, 'permissions', delta)

  assert harness.context.get_state().status_bar.approval_mode == expected_mode
  root = _selector(harness)
  assert root.id == 'settings'
  assert root.keep_open
  assert harness.dispatched == [
    actions.SetApprovalMode(expected_mode),
    actions.OpenSelector(root),
  ]
  choice = harness.context.get_state().selector.selected_choice()
  assert choice is not None
  assert choice.value == 'permissions'
  assert choice.label.strip() == 'CLI mode'
  assert choice.status == expected_status

  await _select(harness, 'permissions', delta)

  assert harness.context.get_state().status_bar.approval_mode == initial_mode
  assert _selector(harness).selected == 'permissions'
  assert harness.restarts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('delta', 'expected'),
  [
    (None, actions.ViewMode.EXTRA_COMPACT),
    (1, actions.ViewMode.EXTRA_COMPACT),
    (-1, actions.ViewMode.TRANSCRIPT),
  ],
  ids=['enter', 'right', 'left'],
)
async def test_handle_effect_root_conversation_detail_still_changes_inline(
  tmp_path: pathlib.Path, delta: int | None, expected: actions.ViewMode
) -> None:
  harness = _harness(tmp_path)
  await _open_settings(harness)
  harness.dispatched.clear()

  await _select(harness, 'history-detail', delta)

  assert harness.context.get_state().status_bar.view_mode is expected
  assert harness.dispatched[0] == actions.SetViewMode(expected)
  root = _selector(harness)
  assert root.id == 'settings'
  assert root.selected == 'history-detail'
  assert harness.restarts == []
  assert not harness.settings.path.exists()


@pytest.mark.asyncio
async def test_handle_effect_template_nested_menus_navigate_back(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  harness.plugin = settings_extension.SettingsExtension(_EXAMPLE_MENU)
  await _open_settings(harness)

  await _select(harness, 'advanced')
  await _select(harness, 'tools')

  submenu = _selector(harness)
  assert submenu.id == 'settings/advanced/tools'
  assert submenu.title == 'Settings › Advanced › Tools'
  assert [choice.value for choice in submenu.choices] == ['notice']
  assert submenu.choices[0].status == '   >'

  harness.context.dispatch(actions.ClearSelector())
  await _run(harness, submenu.cancel_action)

  parent = _selector(harness)
  assert parent.id == 'settings/advanced'
  assert parent.selected == 'tools'

  harness.context.dispatch(actions.ClearSelector())
  await _run(harness, parent.cancel_action)

  assert _selector(harness).id == 'settings'
  assert _selector(harness).selected == 'advanced'


@pytest.mark.asyncio
async def test_handle_effect_template_action_dispatches_and_closes_menu(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  harness.plugin = settings_extension.SettingsExtension(_EXAMPLE_MENU)
  await _open_settings(harness)
  await _select(harness, 'advanced')
  await _select(harness, 'tools')
  harness.dispatched.clear()

  await _select(harness, 'notice')

  assert harness.dispatched == [
    actions.ClearSelector(),
    actions.ShowText('Notice'),
  ]
  assert harness.context.get_state().selector.selector is None


@pytest.mark.asyncio
async def test_handle_effect_template_setting_refreshes_its_own_menu(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  harness.plugin = settings_extension.SettingsExtension(_EXAMPLE_MENU)
  await _open_settings(harness)
  await _select(harness, 'advanced')

  await _select(harness, 'history-detail', -1)

  assert (
    harness.context.get_state().status_bar.view_mode
    is actions.ViewMode.TRANSCRIPT
  )
  submenu = _selector(harness)
  assert submenu.id == 'settings/advanced'
  assert submenu.selected == 'history-detail'
  assert submenu.choices[0].status == '< Full    >'


@pytest.mark.asyncio
async def test_handle_effect_template_ignores_selection_from_parent_menu(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  harness.plugin = settings_extension.SettingsExtension(_EXAMPLE_MENU)
  parent = await _open_settings(harness)
  pending = parent.on_select('advanced')
  await _run(harness, pending)
  submenu = _selector(harness)
  harness.dispatched.clear()

  await _run(harness, pending)

  assert _selector(harness) is submenu
  assert harness.dispatched == []


@pytest.mark.asyncio
async def test_handle_effect_template_menu_with_no_available_items(
  tmp_path: pathlib.Path,
) -> None:
  harness = _harness(tmp_path)
  harness.plugin = settings_extension.SettingsExtension(
    setting_template.Menu(
      'settings',
      'Settings',
      '',
      (
        setting_template.Menu(
          'hidden', 'Hidden', '', (), available=lambda _context: False
        ),
        setting_template.Action(
          'hidden-action',
          'Hidden action',
          '',
          actions.ShowText('Hidden'),
          available=lambda _context: False,
        ),
      ),
    )
  )

  menu = await _open_settings(harness)

  assert menu.choices == ()
  assert menu.title == 'Settings'
