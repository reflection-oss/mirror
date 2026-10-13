from __future__ import annotations

import asyncio
import dataclasses
import pathlib
import sys
from collections.abc import Sequence
from typing import ClassVar, Never, override
from unittest import mock

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers, controls, mouse_handlers, screen

from mirror import config
from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  extension_state,
  host_tools,
  keymap,
  reducer,
  selection,
  state,
  user_extensions,
)


@pytest.mark.asyncio
async def test_startup_continues_after_builtin_import_failure(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
  tmp_path: pathlib.Path,
) -> None:
  monkeypatch.setitem(
    sys.modules, 'mirror.cli.extensions.context_usage.extension', None
  )
  registry = extension.ExtensionRegistry()
  try:
    manager = user_extensions.UserExtensions(
      registry,
      tmp_path / 'extensions',
      bundled=pathlib.Path(extension.__file__).with_name('extensions'),
    )
    await manager.start()
    assert 'core/context-usage' not in registry.extension_ids()
    assert 'core/todo' not in registry.extension_ids()
    assert 'core/inference-controls' in registry.extension_ids()
    assert 'Could not load extension' in caplog.text
    assert 'context_usage' in caplog.text
  finally:
    await registry.close()


@dataclasses.dataclass(frozen=True)
class _CounterState:
  count: int = 0


@dataclasses.dataclass(frozen=True)
class _Increment(actions.ExtensionAction):
  amount: int


@dataclasses.dataclass(frozen=True)
class _RecordCount:
  count: int


class _CounterExtension(extension.Extension[_CounterState, _RecordCount, None]):
  extension_id = ''
  region = extension.RenderRegion.STATUS_LEFT
  instance_count: ClassVar[int] = 0
  active_count: ClassVar[int] = 0

  def __init__(self) -> None:
    type(self).instance_count += 1
    self.hydrated_records: tuple[agent_pb2.AgentMessage, ...] = ()
    self.tip_text = f'{self.id()} tip'
    self.rendered_tips: tuple[str, ...] = ()

  @override
  def tips(self) -> tuple[str, ...]:
    return (self.tip_text,)

  async def on_load(self, context: None) -> None:
    type(self).active_count += 1

  async def on_unload(self, context: None) -> None:
    type(self).active_count -= 1

  @classmethod
  def id(cls) -> str:
    return cls.extension_id

  def render_region(self) -> extension.RenderRegion:
    return self.region

  @staticmethod
  def initial_state() -> _CounterState:
    return _CounterState()

  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[
    _CounterState,
    tuple[effects.CoreEffect | extension.PrivateEffect[_RecordCount], ...],
  ]:
    current = self.require_state(global_state)
    if isinstance(action, actions.RoutedExtensionAction):
      if not isinstance(action.action, _Increment):
        return current, ()
      amount = action.action.amount
    elif isinstance(action, actions.SetActive) and action.value:
      amount = 1
    else:
      return current, ()
    next_state = _CounterState(current.count + amount)
    return next_state, (
      effects.EmitText(f'{self.id()}: {next_state.count}'),
      self.effect(_RecordCount(next_state.count)),
    )

  def render(
    self,
    global_state: state.State,
    extension_state: _CounterState,
  ) -> containers.AnyContainer:
    del extension_state
    self.rendered_tips = global_state.extension_tips
    return containers.Window()

  def hydrate(
    self,
    snapshot: agent_pb2.ExtensionRecord | None,
    records: Sequence[agent_pb2.AgentMessage],
    global_state: state.State,
  ) -> _CounterState:
    self.hydrated_records = tuple(records)
    if snapshot is None:
      return self.initial_state()
    return _CounterState(int(snapshot.data.fields['count'].number_value))

  async def handle_effect(
    self, context: extension.Context, effect: _RecordCount
  ) -> None:
    context.dispatch(actions.ShowText(f'{self.id()}: {effect.count}'))


class _BrokenExtension(_CounterExtension):
  extension_id = 'test/broken'

  def __init__(self, phase: str) -> None:
    super().__init__()
    self.phase = phase

  def _check(self, phase: str) -> None:
    if self.phase == phase:
      raise ValueError(f'broken {phase}')

  def handle_message(
    self, context: extension.Context, message: agent_pb2.SessionMessage
  ) -> None:
    self._check('message')

  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    _CounterState,
    tuple[effects.CoreEffect | extension.PrivateEffect[_RecordCount], ...],
  ]:
    self._check('reduce')
    return super().reduce(global_state, action)

  def hydrate(
    self,
    snapshot: agent_pb2.ExtensionRecord | None,
    records: Sequence[agent_pb2.AgentMessage],
    global_state: state.State,
  ) -> _CounterState:
    self._check('hydrate')
    return super().hydrate(snapshot, records, global_state)

  def render(
    self, global_state: state.State, extension_state: _CounterState
  ) -> containers.AnyContainer:
    self._check('render')
    return super().render(global_state, extension_state)

  def tools(self, context: extension.Context) -> list[host_tools.HostTool]:
    self._check('tools')
    return []

  async def on_foreground(self, context: extension.Context) -> None:
    self._check('foreground')

  async def on_background(self, context: extension.Context) -> None:
    self._check('background')

  async def handle_effect(
    self, context: extension.Context, effect: _RecordCount
  ) -> None:
    self._check('effect')

  async def on_unload(self, context: None) -> None:
    await super().on_unload(context)
    self._check('unload')


class _EchoTool:
  def __init__(self, name: str = 'echo') -> None:
    self.definition = host_tools.ToolDefinition(name, 'Echo arguments', {})

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    return host_tools.ToolResult(arguments, True)


class _FailingTool(_EchoTool):
  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    raise ValueError('tool failed')


class _ConstructionFailureTool(_EchoTool):
  def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> Never:
    raise ValueError('tool failed')


@pytest.mark.asyncio
@pytest.mark.parametrize('tool', [_FailingTool(), _ConstructionFailureTool()])
async def test_registry_isolates_tool_errors_without_an_extension_boundary(
  tool: host_tools.HostTool,
) -> None:
  registry = host_tools.HostToolRegistry()
  registry.register('tools', [tool, _EchoTool('healthy')])
  try:
    assert await registry.execute('tools__echo', '{}') == host_tools.ToolResult(
      'tool failed', False
    )
    assert await registry.execute(
      'tools__healthy', 'ok'
    ) == host_tools.ToolResult('ok', True)
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('selected', 'expected'),
  [
    (None, ('extension__echo',)),
    ((), ()),
    (('extension__echo',), ('extension__echo',)),
    (('extension__private',), ('extension__private',)),
    (
      ('extension__echo', 'extension__private'),
      ('extension__echo', 'extension__private'),
    ),
  ],
)
async def test_workspace_tool_selection_is_exact(
  selected: tuple[str, ...] | None,
  expected: tuple[str, ...],
) -> None:
  registry = host_tools.HostToolRegistry(selected=selected)
  private = _EchoTool('private')
  private.definition = dataclasses.replace(private.definition, global_=False)
  registry.register('extension', [_EchoTool(), private])
  try:
    registry.validate_selection()
    assert tuple(tool.name for tool in registry.tools()) == expected
    assert (await registry.execute('extension__echo', '{}')).success == (
      'extension__echo' in expected
    )
    assert (await registry.execute('extension__private', '{}')).success == (
      'extension__private' in expected
    )
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('selected', 'expected'),
  [
    (None, ('extension__echo', 'extension__new_tool')),
    (('extension__echo',), ('extension__echo',)),
    ((), ()),
  ],
)
async def test_new_tools_follow_defaults_without_widening_allowlists(
  selected: tuple[str, ...] | None, expected: tuple[str, ...]
) -> None:
  registry = host_tools.HostToolRegistry(selected=selected)
  registry.register('extension', [_EchoTool()])
  try:
    await registry.replace('extension', [_EchoTool(), _EchoTool('new_tool')])
    assert tuple(tool.name for tool in registry.tools()) == expected
    assert (await registry.execute('extension__new_tool', 'hello')).success == (
      'extension__new_tool' in expected
    )
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_extensions_with_the_same_tool_name_remain_independent() -> None:
  registry = host_tools.HostToolRegistry()
  registry.register('workspace', [_EchoTool()])
  registry.register('global', [_EchoTool()])
  try:
    assert tuple(tool.name for tool in registry.tools()) == (
      'workspace__echo',
      'global__echo',
    )
    assert (
      await registry.execute('workspace__echo', 'first')
    ).content == 'first'
    assert (
      await registry.execute('global__echo', 'second')
    ).content == 'second'
    await registry.replace('global', [_EchoTool()])
    await registry.unregister('workspace')
    assert not (await registry.execute('workspace__echo', '{}')).success
    assert (
      await registry.execute('global__echo', 'remaining')
    ).content == 'remaining'
  finally:
    await registry.close()


@pytest.mark.parametrize(
  ('owner', 'name'),
  [('a', 'b__c'), ('a__b', 'c'), ('a/b', 'c'), ('a_b', 'c')],
)
def test_tool_names_do_not_require_parsing(owner: str, name: str) -> None:
  assert (
    host_tools.tool_name(owner, name) == f'{owner.replace("/", "_")}__{name}'
  )


@pytest.mark.asyncio
async def test_duplicate_names_are_rejected_without_replacing_existing_tools() -> (
  None
):
  registry = host_tools.HostToolRegistry()
  registry.register('a', [_EchoTool('b__c')])
  try:
    with pytest.raises(ValueError, match='Duplicate host tool'):
      registry.register('a__b', [_EchoTool('c')])
    with pytest.raises(ValueError, match='Duplicate host tool'):
      await registry.replace('a', [_EchoTool('b__c'), _EchoTool('b__c')])
    assert (
      await registry.execute('a__b__c', 'still registered')
    ).content == 'still registered'
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_registry_snapshots_definitions_until_replacement() -> None:
  registry = host_tools.HostToolRegistry()
  tool = _EchoTool()
  original = tool.definition
  registry.register('tools', [tool])
  try:
    tool.definition = dataclasses.replace(original, name='renamed')
    assert registry.tools() == (
      dataclasses.replace(original, name='tools__echo'),
    )
    assert await registry.execute('tools__echo', 'ok') == host_tools.ToolResult(
      'ok', True
    )
    await registry.replace('tools', [tool])
    assert registry.tools() == (
      dataclasses.replace(tool.definition, name='tools__renamed'),
    )
    assert not (await registry.execute('tools__echo', '{}')).success
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_host_tool_registry_routes_and_notifies_without_owning_providers() -> (
  None
):
  changed: list[bool] = []
  registry = host_tools.HostToolRegistry()

  async def restart() -> bool:
    changed.append(True)
    registry.worker_starting()
    return True

  first_plugin = _EchoTool('first_plugin_echo')
  plugin = _EchoTool('plugin_echo')
  registry.register('first_plugin', [first_plugin])
  registry.register('plugin', [plugin])
  assert registry.tools() == (
    dataclasses.replace(
      first_plugin.definition, name='first_plugin__first_plugin_echo'
    ),
    dataclasses.replace(plugin.definition, name='plugin__plugin_echo'),
  )
  assert await registry.execute(
    'first_plugin__first_plugin_echo', 'first'
  ) == host_tools.ToolResult('first', True)
  assert await registry.execute(
    'plugin__plugin_echo', 'second'
  ) == host_tools.ToolResult('second', True)
  replacement = _EchoTool('plugin_echo')
  replacement.definition = dataclasses.replace(
    replacement.definition, description='Replacement'
  )
  registry.register('conflict', [replacement])
  assert registry.tools() == (
    dataclasses.replace(
      first_plugin.definition, name='first_plugin__first_plugin_echo'
    ),
    dataclasses.replace(plugin.definition, name='plugin__plugin_echo'),
    dataclasses.replace(replacement.definition, name='conflict__plugin_echo'),
  )
  assert changed == []
  registry.worker_starting()
  registry.worker_started(restart)
  await registry.unregister('conflict')
  await registry.wait_for_update()
  assert registry.tools() == (
    dataclasses.replace(
      first_plugin.definition, name='first_plugin__first_plugin_echo'
    ),
    dataclasses.replace(plugin.definition, name='plugin__plugin_echo'),
  )
  assert changed == [True]
  await registry.unregister('plugin')
  await registry.wait_for_update()
  assert registry.tools() == (
    dataclasses.replace(
      first_plugin.definition, name='first_plugin__first_plugin_echo'
    ),
  )
  assert not (await registry.execute('plugin__plugin_echo', '{}')).success
  assert changed == [True, True]
  await registry.close()


@pytest.mark.asyncio
async def test_registry_batches_startup_changes_and_stops_updates_on_disconnect() -> (
  None
):
  registry = host_tools.HostToolRegistry()
  catalogs: list[tuple[str, ...]] = []

  async def restart() -> bool:
    catalogs.append(tuple(tool.name for tool in registry.tools()))
    registry.worker_starting()
    return True

  registry.register('first', [_EchoTool('first')])
  registry.worker_starting()
  registry.register('second', [_EchoTool('second')])
  registry.register('third', [_EchoTool('third')])
  registry.worker_started(restart)
  await registry.wait_for_update()
  assert catalogs == [('first__first', 'second__second', 'third__third')]
  await registry.worker_stopped()
  await registry.replace('second', [_EchoTool('replacement')])
  assert catalogs == [('first__first', 'second__second', 'third__third')]
  await registry.close()


@pytest.mark.asyncio
async def test_catalog_refresh_waits_for_an_idle_turn_and_coalesces_changes() -> (
  None
):
  registry = host_tools.HostToolRegistry()
  catalogs: list[tuple[str, ...]] = []

  async def restart() -> bool:
    catalogs.append(tuple(tool.name for tool in registry.tools()))
    registry.worker_starting()
    return True

  registry.worker_starting()
  registry.worker_started(restart)
  try:
    registry.activity_changed(True)
    registry.register('first', [_EchoTool('first')])
    await asyncio.sleep(0)
    assert catalogs == []
    registry.activity_changed(False)
    registry.activity_changed(True)
    await asyncio.sleep(0)
    assert catalogs == []
    registry.register('second', [_EchoTool('second')])
    registry.activity_changed(False)
    await registry.wait_for_update()
    assert catalogs == [('first__first', 'second__second')]
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_registered_tools_are_advertised_and_stop_on_unload() -> None:
  registry = host_tools.HostToolRegistry()
  tool = _EchoTool()
  registry.register('extension', [tool])
  try:
    assert registry.tools() == (
      dataclasses.replace(tool.definition, name='extension__echo'),
    )
    assert await registry.execute(
      'extension__echo', 'hello'
    ) == host_tools.ToolResult('hello', True)
    await registry.unregister('extension')
    assert not (await registry.execute('extension__echo', 'hello')).success
  finally:
    await registry.close()


@pytest.mark.asyncio
async def test_unregister_removes_tools_without_owning_running_calls() -> None:
  requested = asyncio.Event()
  finish = asyncio.Event()

  class WaitingTool(_EchoTool):
    async def execute(
      self,
      arguments: str,
      context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
    ) -> host_tools.ToolResult:
      requested.set()
      await finish.wait()
      return host_tools.ToolResult(arguments, True)

  registry = host_tools.HostToolRegistry()
  registry.register('extension', [WaitingTool()])
  call = asyncio.create_task(registry.execute('extension__echo', '{}'))
  try:
    await asyncio.wait_for(requested.wait(), timeout=2)
    await registry.unregister('extension')
    assert registry.tools() == ()
    assert not (await registry.execute('extension__echo', '{}')).success
    finish.set()
    assert await call == host_tools.ToolResult('{}', True)
  finally:
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)
    await registry.close()


@pytest.mark.asyncio
async def test_registry_executes_below_model_approval_boundary() -> None:
  registry = host_tools.HostToolRegistry()
  tool = _EchoTool()
  assert tool.definition.needs_approval
  registry.register('extension', [tool])
  try:
    result = await registry.execute('extension__echo', '{}')
    assert result == host_tools.ToolResult('{}', True)
  finally:
    await registry.close()


class _ToolExtension(_CounterExtension):
  extension_id = 'test/tools'

  def __init__(self) -> None:
    super().__init__()
    self.opened: list[_EchoTool] = []

  def tools(self, context: extension.Context) -> list[host_tools.HostTool]:
    tool = _EchoTool()
    self.opened.append(tool)
    return [tool]


@pytest.mark.asyncio
async def test_host_tools_are_workspace_owned() -> None:
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=lambda _: None,
    get_state=lambda: _state(extension_state.ExtensionStates()),
    enqueue=lambda _message: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  plugin = _ToolExtension()
  registry = extension.ExtensionRegistry()
  await registry.register(plugin, None)
  first = registry.create_host(context)
  second = registry.create_host(context)
  await first.start()
  await second.start()
  worker_tools = first.host_tool_registry
  assert worker_tools.tools() == (
    dataclasses.replace(plugin.opened[0].definition, name='test_tools__echo'),
  )
  assert await worker_tools.execute(
    'test_tools__echo', '{"value": 1}'
  ) == host_tools.ToolResult('{"value": 1}', True)
  await first.close()
  assert first.host_tool_registry.tools() == ()
  assert second.host_tool_registry.tools() == (
    dataclasses.replace(plugin.opened[1].definition, name='test_tools__echo'),
  )
  await registry.unload(plugin.id())
  assert second.host_tool_registry.tools() == ()
  await second.close()
  await registry.close()


@pytest.mark.asyncio
async def test_selected_extension_isolates_tools_and_tracks_shared_reload() -> (
  None
):
  shared = extension.ExtensionRegistry()
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=lambda _: None,
    get_state=lambda: _state(extension_state.ExtensionStates()),
    enqueue=lambda _: None,
    refresh_tools=shared.refresh_tools,
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
  )
  sibling = shared.create_host(context)
  reviewer = shared.create_host(context, extensions=(_ToolExtension.id(),))
  plugin = _ToolExtension()
  plugin.global_ = False
  active_before = _ToolExtension.active_count
  try:
    await shared.register(plugin, None)
    assert _ToolExtension.active_count == active_before + 1
    await reviewer.start()
    await sibling.start()
    assert reviewer.host_tool_registry.tools() == (
      dataclasses.replace(plugin.opened[0].definition, name='test_tools__echo'),
    )
    assert sibling.host_tool_registry.tools() == ()
    assert reviewer.tips == ('test/tools tip',)
    current = _state(reviewer.initial_states)
    reviewer.render(current, extension.RenderRegion.STATUS_LEFT)
    assert plugin.rendered_tips == ('test/tools tip',)
    assert current.extension_tips == ()
    assert sibling.tips == ()
    assert shared.extension_ids() == (plugin.id(),)
    await shared.register(_AlphaExtension(), None)
    assert reviewer.status('test/alpha') is not None
    assert sibling.status('test/alpha') is not None
    await shared.unload('test/alpha')
    assert reviewer.status('test/alpha') is None
    assert reviewer.status(plugin.id()) is not None
    replacement = _ToolExtension()
    replacement.global_ = False
    replacement.tip_text = 'Updated tool tip'
    await shared.register(replacement, None)
    assert reviewer.tips == ('Updated tool tip',)
    reviewer.render(current, extension.RenderRegion.STATUS_LEFT)
    assert replacement.rendered_tips == ('Updated tool tip',)
    assert current.extension_tips == ()
    assert _ToolExtension.active_count == active_before + 1
    assert reviewer.host_tool_registry.tools()
    assert sibling.host_tool_registry.tools() == ()
    await reviewer.close()
    await shared.unload(plugin.id())
    assert reviewer.host_tool_registry.tools() == ()
    assert shared.extension_ids() == ()
    assert reviewer.tips == ()
    assert _ToolExtension.active_count == active_before
    await shared.register(_ToolExtension(), None)
    assert sibling.host_tool_registry.tools()
    assert reviewer.host_tool_registry.tools() == ()
  finally:
    await reviewer.close()
    await sibling.close()
    await shared.close()


class _AlphaExtension(_CounterExtension):
  extension_id = 'test/alpha'
  region = extension.RenderRegion.WIDGET_BELOW_PROMPT


class _BetaExtension(_CounterExtension):
  extension_id = 'test/beta'
  region = extension.RenderRegion.WIDGET_ABOVE_PROMPT


@pytest.mark.asyncio
async def test_extension_ids_remain_unique_during_concurrent_registration() -> (
  None
):
  started = asyncio.Event()
  release = asyncio.Event()

  class SlowExtension(_BetaExtension):
    async def on_load(self, context: None) -> None:
      started.set()
      await release.wait()
      await super().on_load(context)

  registry = extension.ExtensionRegistry()
  pending = asyncio.create_task(registry.register(SlowExtension(), None))
  try:
    await started.wait()
    await registry.register(_BetaExtension(), None)
    release.set()
    await pending
    assert registry.extension_ids() == ('test/beta',)
  finally:
    release.set()
    await asyncio.gather(pending, return_exceptions=True)
    await registry.close()


class _ToolFaultExtension(_BetaExtension):
  definition = host_tools.ToolDefinition('broken_tool', 'A faulty tool', {})

  def __init__(self) -> None:
    super().__init__()
    self.calls: list[str] = []

  def tools(self, context: extension.Context) -> list[host_tools.HostTool]:
    return [self]

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    self.calls.append('tool')
    raise ValueError('broken tool')

  def slash_commands(self) -> tuple[commands.CommandDef, ...]:
    return (commands.CommandDef('example', 'Example', self._command),)

  def _command(
    self, context: commands.CommandContext, arguments: str
  ) -> actions.Action:
    self.calls.append('command')
    if arguments == 'fail':
      raise ValueError('broken command')
    return actions.ShowText('healthy')

  def _text(self) -> str:
    self.calls.append('paint')
    return 'healthy'

  def render(
    self, global_state: state.State, extension_state: _CounterState
  ) -> containers.AnyContainer:
    return containers.Window(
      controls.FormattedTextControl(self._text), height=1
    )


@pytest.mark.asyncio
async def test_tool_failure_disables_captured_ui_callbacks_without_poisoning_reload() -> (
  None
):
  registry = extension.ExtensionRegistry()
  broken = _ToolFaultExtension()
  await registry.register(broken, None)
  host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/workspace'), remote=False),
      lambda _: None,
      lambda: _state(host.initial_states),
      lambda _: None,
      registry.refresh_tools,
      lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  context = commands.CommandContext(
    'model', 'low', 'session', False, '/workspace'
  )
  try:
    await host.start()
    command = host.slash_commands[0]
    widget = containers.to_container(
      host.render(
        _state(host.initial_states), extension.RenderRegion.STATUS_LEFT
      )
    )
    assert host.tips == ('test/beta tip',)
    result = await host.host_tool_registry.execute(
      'test_beta__broken_tool', '{}'
    )
    assert not result.success
    assert host.tips == ()
    assert command.handler(context, '') == actions.ShowText(
      'Extension test/beta is unavailable. Fix and reload.'
    )
    widget.write_to_screen(
      screen.Screen(),
      mouse_handlers.MouseHandlers(),
      screen.WritePosition(0, 0, 20, 1),
      '',
      True,
      None,
    )
    assert broken.calls == ['tool']
    assert not (
      await host.host_tool_registry.execute('test_beta__broken_tool', '{}')
    ).success
    assert broken.calls == ['tool']

    await registry.unload(broken.id())
    replacement = _ToolFaultExtension()
    await registry.register(replacement, None)
    assert host.tips == ('test/beta tip',)
    command.handler(context, '')
    assert broken.calls == ['tool']
    assert host.slash_commands[0].handler(context, '') == actions.ShowText(
      'healthy'
    )
    assert replacement.calls == ['command']
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_command_failure_disables_tools() -> None:
  registry = extension.ExtensionRegistry()
  broken = _ToolFaultExtension()
  await registry.register(broken, None)
  host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/workspace'), remote=False),
      lambda _: None,
      lambda: _state(host.initial_states),
      lambda _: None,
      registry.refresh_tools,
      lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  context = commands.CommandContext(
    'model', 'low', 'session', False, '/workspace'
  )
  try:
    await host.start()
    assert host.slash_commands[0].handler(context, 'fail') == actions.ShowText(
      'Extension test/beta disabled: ValueError: broken command. Fix and reload.',
      style='ui.error',
    )
    assert not (
      await host.host_tool_registry.execute('test_beta__broken_tool', '{}')
    ).success
    assert broken.calls == ['command']
    widget = containers.to_container(
      host.render(
        _state(host.initial_states), extension.RenderRegion.WIDGET_ABOVE_PROMPT
      )
    )
    assert widget.preferred_height(30, 20).preferred == 3
    assert widget.preferred_height(30, 20).max == 3
    assert widget.preferred_height(120, 40).max == 1
    rendered = screen.Screen()
    widget.write_to_screen(
      rendered,
      mouse_handlers.MouseHandlers(),
      screen.WritePosition(0, 0, 30, 3),
      '',
      True,
      None,
    )
    assert (
      ''.join(cell.char for cell in rendered.data_buffer[2].values()).rstrip()
      == 'x and reload.'
    )
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_configuration_callback_and_action_preserve_state_and_snapshots() -> (
  None
):
  observed: list[tuple[object, int]] = []
  contexts: list[extension.Context] = []
  changes: list[actions.ExtensionConfigChanged] = []

  class Configured(_AlphaExtension):
    @override
    def default_config(self) -> config.JsonObject:
      return {'value': 1}

    @override
    async def on_config(self, context: extension.Context) -> None:
      contexts.append(context)
      observed.append((
        context.config['value'],
        self.current_state(context).count,
      ))

    @override
    def slash_commands(self) -> Sequence[commands.CommandDef]:
      return (
        commands.CommandDef(
          'configured',
          'Configured command',
          lambda context, _: actions.ShowText(str(context.config['value'])),
        ),
      )

  registry = extension.ExtensionRegistry()
  plugin = Configured()
  await registry.register(plugin, None)
  current = _state(extension_state.ExtensionStates())

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    updated, _ = host.reduce(current, action)
    current = dataclasses.replace(current, extensions=updated)
    if isinstance(action, actions.ExtensionConfigChanged):
      changes.append(action)

  host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/workspace'), remote=False),
      dispatch,
      lambda: current,
      lambda _: None,
      registry.refresh_tools,
      lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
    )
  )
  try:
    await host.start()
    assert observed == [(1, 0)]
    assert commands.menu_commands('/', host.slash_commands)
    command_context = commands.CommandContext(
      'model', 'low', 'session', False, '/workspace'
    )
    assert commands.dispatch(
      '/configured', command_context, host.slash_commands
    ) == actions.ShowText('1')
    dispatch(actions.RoutedExtensionAction(plugin.id(), _Increment(3)))
    await registry.configure(plugin.id(), {'config': {'value': 2}})
    assert observed == [(1, 0), (2, 3)]
    assert changes == [
      actions.ExtensionConfigChanged(plugin.id(), {'value': 2})
    ]
    assert plugin.configuration(current) == {'value': 2}
    assert contexts[0].config == {'value': 1}
    contexts[1].config['value'] = 'mutated'
    assert plugin.configuration(current) == {'value': 2}
    await registry.configure(plugin.id(), {'config': {'value': 2}})
    assert observed == [(1, 0), (2, 3)]
    assert commands.menu_commands('/', host.slash_commands)
    command_context = commands.CommandContext(
      'model', 'low', 'session', False, '/workspace'
    )
    assert commands.dispatch(
      '/configured', command_context, host.slash_commands
    ) == actions.ShowText('2')
    with pytest.raises(ValueError, match='config must be a JSON object'):
      await registry.configure(plugin.id(), {'config': None})
    assert plugin.configuration(current) == {'value': 2}
    await registry.configure(plugin.id(), {})
    assert plugin.configuration(current) == {'value': 1}
    assert observed == [(1, 0), (2, 3), (1, 3)]
  finally:
    await host.close()
    await registry.close()


def _state(extension_states: extension_state.ExtensionStates) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    extensions=extension_states,
  )


def _snapshot(extension_id: str, count: int) -> agent_pb2.AgentMessage:
  message = agent_pb2.AgentMessage(
    extension=agent_pb2.ExtensionRecord(
      extension=extension_id,
      kind='state',
    )
  )
  message.extension.data.fields['count'].number_value = count
  return message


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'phase',
  [
    'tools',
    'message',
    'reduce',
    'hydrate',
    'render',
    'foreground',
    'background',
    'effect',
    'unload',
  ],
)
async def test_plugin_exceptions_do_not_escape_or_disable_healthy_plugins(
  phase: str, caplog: pytest.LogCaptureFixture
) -> None:
  registry = extension.ExtensionRegistry()
  broken = _BrokenExtension(phase)
  healthy = _BetaExtension()
  await registry.register(broken, None)
  await registry.register(healthy, None)
  current: state.State
  host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/workspace'), remote=False),
      lambda _: None,
      lambda: current,
      lambda _: None,
      registry.refresh_tools,
      lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  current = _state(host.initial_states)
  try:
    await host.start()
    host.handle_message(agent_pb2.SessionMessage())
    host.reduce(current, actions.SetActive(True))
    host.reduce(current, actions.HydrateExtensions(()))
    host.render(current, extension.RenderRegion.STATUS_LEFT)
    await host.foreground()
    await host.background()
    await host.handle_effect(
      effects.RunExtensionEffect(broken.id(), _RecordCount(1))
    )
    updated, emitted = host.reduce(
      current, actions.RoutedExtensionAction(healthy.id(), _Increment(3))
    )
    assert updated[healthy.id()] == _CounterState(3)
    assert emitted
  finally:
    await host.close()
    await registry.close()
  assert f'broken {phase}' in caplog.text


@pytest.mark.asyncio
async def test_host_hydrates_records_during_reduction() -> None:
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=lambda _: None,
    get_state=lambda: _state(extension_state.ExtensionStates()),
    enqueue=lambda _message: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(_BetaExtension(), None)
  host = registry.create_host(context)
  current = _state(host.initial_states)
  extension_states, emitted = host.reduce(
    current, actions.HydrateExtensions((_snapshot(_BetaExtension.id(), 3),))
  )

  assert extension_states[_BetaExtension.id()] == _CounterState(3)
  assert emitted == ()


@pytest.mark.asyncio
async def test_host_hydrates_latest_snapshot_since_compaction() -> None:
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=lambda _: None,
    get_state=lambda: _state(extension_state.ExtensionStates()),
    enqueue=lambda _message: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  plugin = _BetaExtension()
  await registry.register(plugin, None)
  host = registry.create_host(context)
  current = _state(host.initial_states)
  compacted = _snapshot(_BetaExtension.id(), 2)
  marker = agent_pb2.AgentMessage(
    context_window=agent_pb2.ContextWindowRecord(
      compaction=agent_pb2.Compaction(recent=[compacted])
    )
  )

  hydrated, _ = host.reduce(
    current,
    actions.HydrateExtensions((_snapshot(_BetaExtension.id(), 1), marker)),
  )
  assert plugin.hydrated_records == (marker,)
  update = _snapshot(_BetaExtension.id(), 3)
  chat = agent_pb2.AgentMessage(
    chat=agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content='hello'))
  )
  latest, _ = host.reduce(
    current,
    actions.HydrateExtensions((
      _snapshot(_BetaExtension.id(), 1),
      marker,
      update,
      chat,
    )),
  )
  assert plugin.hydrated_records == (marker, update, chat)

  assert hydrated[_BetaExtension.id()] == _CounterState(2)
  assert latest[_BetaExtension.id()] == _CounterState(3)


@pytest.mark.asyncio
async def test_fork_hydration_uses_the_child_prefix_not_parent_state() -> None:
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=lambda _: None,
    get_state=lambda: _state(extension_state.ExtensionStates()),
    enqueue=lambda _message: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(_BetaExtension(), None)
  host = registry.create_host(context)
  prefix = _snapshot(_BetaExtension.id(), 2)
  suffix = _snapshot(_BetaExtension.id(), 3)
  initial = _state(host.initial_states)
  parent_states, _ = host.reduce(
    initial, actions.HydrateExtensions((prefix, suffix))
  )
  parent = _state(parent_states)

  child, _ = host.reduce(parent, actions.HydrateExtensions((prefix,)))
  empty_child, _ = host.reduce(parent, actions.HydrateExtensions(()))

  assert parent.extensions[_BetaExtension.id()] == _CounterState(3)
  assert child[_BetaExtension.id()] == _CounterState(2)
  assert empty_child[_BetaExtension.id()] == _CounterState()


@pytest.mark.asyncio
async def test_host_routes_private_actions_and_broadcasts_host_actions() -> (
  None
):
  current: state.State
  dispatched: list[actions.Action] = []
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=dispatched.append,
    get_state=lambda: current,
    enqueue=lambda _message: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(_BetaExtension(), None)
  await registry.register(_AlphaExtension(), None)
  host = registry.create_host(context)
  current = _state(host.initial_states)
  await host.start()
  dispatched.clear()

  assert isinstance(
    host.render(current, extension.RenderRegion.WIDGET_ABOVE_PROMPT),
    containers.HSplit,
  )
  assert isinstance(
    host.render(current, extension.RenderRegion.WIDGET_BELOW_PROMPT),
    containers.HSplit,
  )
  assert isinstance(
    host.render(current, extension.RenderRegion.STATUS_RIGHT),
    containers.Window,
  )
  assert (
    host
    .render(current, extension.RenderRegion.WIDGET_BELOW_PROMPT)
    .preferred_height(80, 20)
    .preferred
    == 0
  )

  missing = dataclasses.replace(
    current, extensions=extension_state.ExtensionStates()
  )
  unavailable = dataclasses.replace(
    current,
    extensions=extension_state.ExtensionStates({
      'test/alpha': None,
      'test/beta': None,
    }),
  )
  assert isinstance(
    host.render(missing, extension.RenderRegion.WIDGET_ABOVE_PROMPT),
    containers.Window,
  )
  assert isinstance(
    host.render(unavailable, extension.RenderRegion.WIDGET_BELOW_PROMPT),
    containers.Window,
  )
  dispatched.append(actions.RoutedExtensionAction('test/beta', _Increment(2)))
  extension_states, emitted = host.reduce(current, dispatched[0])

  assert current.extensions['test/alpha'] == _CounterState()
  assert extension_states['test/alpha'] == _CounterState()
  assert extension_states['test/beta'] == _CounterState(2)
  assert emitted == (
    effects.EmitText('test/beta: 2'),
    effects.RunExtensionEffect('test/beta', _RecordCount(2)),
  )

  current = dataclasses.replace(current, extensions=extension_states)
  selector = selection.Selector(
    'amount',
    'Choose amount',
    (selection.Choice('4', 'Four'),),
    lambda value: actions.RoutedExtensionAction(
      _BetaExtension.id(), _Increment(int(value))
    ),
    extension_id=_BetaExtension.id(),
  )
  opened, _ = reducer.reduce(current, actions.OpenSelector(selector))
  choice = opened.selector.selected_choice()
  assert choice is not None
  selection_action = actions.SelectorChoiceSelected(selector.id, choice.value)
  closed, selection_effects = reducer.reduce(opened, selection_action)
  selected_states, selected_effects = host.reduce(
    closed, selector.on_select(choice.value)
  )
  assert selected_states['test/beta'] == _CounterState(6)
  assert selected_effects == (
    effects.EmitText('test/beta: 6'),
    effects.RunExtensionEffect('test/beta', _RecordCount(6)),
  )
  assert not closed.selector.active
  assert selection_effects == (effects.SyncSelectorInput(False, ''),)

  active_states, active_effects = host.reduce(
    current, actions.SetActive(value=True)
  )
  assert active_states['test/alpha'] == _CounterState(1)
  assert active_states['test/beta'] == _CounterState(3)
  assert active_effects == (
    effects.EmitText('test/beta: 3'),
    effects.RunExtensionEffect('test/beta', _RecordCount(3)),
    effects.EmitText('test/alpha: 1'),
    effects.RunExtensionEffect('test/alpha', _RecordCount(1)),
  )

  private_effect = emitted[1]
  assert isinstance(private_effect, effects.RunExtensionEffect)
  await host.handle_effect(private_effect)

  assert dispatched[-1] == actions.ShowText('test/beta: 2')

  reset, reset_effects = host.reduce(
    dataclasses.replace(current, extensions=extension_states),
    actions.SessionOpened('next'),
  )
  assert reset['test/alpha'] == _CounterState()
  assert reset['test/beta'] == _CounterState()
  assert reset_effects == ()

  await host.close()
  inactive_states, inactive_effects = host.reduce(
    dataclasses.replace(current, extensions=active_states),
    actions.RoutedExtensionAction('test/beta', _Increment(1)),
  )
  assert inactive_states['test/beta'] == _CounterState(3)
  assert inactive_effects == ()
  assert isinstance(
    host.render(current, extension.RenderRegion.WIDGET_ABOVE_PROMPT),
    containers.Window,
  )
  assert isinstance(
    host.render(current, extension.RenderRegion.WIDGET_BELOW_PROMPT),
    containers.Window,
  )
  await host.close()


@pytest.mark.asyncio
async def test_registry_shares_instances_but_not_workspace_state() -> None:
  _BetaExtension.instance_count = 0
  _BetaExtension.active_count = 0
  first: state.State
  second: state.State
  registry = extension.ExtensionRegistry()
  await registry.register(_BetaExtension(), None)
  first_host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/first'), remote=False),
      dispatch=lambda _: None,
      get_state=lambda: first,
      enqueue=lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  second_host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/second'), remote=False),
      dispatch=lambda _: None,
      get_state=lambda: second,
      enqueue=lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  first = _state(first_host.initial_states)
  second = _state(second_host.initial_states)
  await first_host.start()
  await second_host.start()

  first_states, _ = first_host.reduce(
    first, actions.RoutedExtensionAction('test/beta', _Increment(2))
  )

  assert _BetaExtension.instance_count == 1
  assert _BetaExtension.active_count == 1
  assert first_states['test/beta'] == _CounterState(2)
  assert second.extensions['test/beta'] == _CounterState()
  await first_host.close()
  assert _BetaExtension.active_count == 1
  await second_host.close()
  assert _BetaExtension.active_count == 1
  await registry.close()
  assert _BetaExtension.active_count == 0


@pytest.mark.asyncio
async def test_command_collision_overrides_builtin_and_reloads(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'commands'
  directory.mkdir()
  entrypoint = directory / 'extension.py'
  source = """from mirror.cli import actions, commands, extension
class Plugin(extension.Extension[None, None, None]):
  @classmethod
  def id(cls): return 'commands'
  def slash_commands(self):
    return (commands.CommandDef('help', 'Help', lambda context, arguments: actions.ShowText('help')),)
"""
  entrypoint.write_text(source)
  registry = extension.ExtensionRegistry()
  current = _state(extension_state.ExtensionStates())
  host = registry.create_host(
    extension.Context(
      extension.Workspace(tmp_path, remote=False),
      dispatch=lambda _: None,
      get_state=lambda: current,
      enqueue=lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  assert await registry.load(directory) == 'commands'
  assert host.slash_commands[0].name == 'help'
  assert host.slash_commands[0].extension_id == 'commands'

  entrypoint.write_text(source.replace("'help'", "'custom'"))
  assert await registry.load(directory) == 'commands'
  assert host.slash_commands[0].name == 'custom'
  await registry.unload('commands')
  assert host.slash_commands == ()
  await host.close()


@pytest.mark.asyncio
async def test_keybinding_override_restores_previous_extension_on_unload(
  tmp_path: pathlib.Path,
) -> None:
  source = """from mirror.cli import actions, commands, extension, keymap
class Plugin(extension.Extension[None, None, None]):
  @classmethod
  def id(cls): return 'first'
  def slash_commands(self):
    return (commands.CommandDef('first', 'First', lambda context, arguments: actions.ShowText('first')),)
  def keybindings(self):
    return (keymap.Binding(('tab',), actions.QuitRequested()),)
"""
  first = tmp_path / 'first'
  second = tmp_path / 'second'
  first.mkdir()
  second.mkdir()
  entrypoint = first / 'extension.py'
  entrypoint.write_text(source)
  registry = extension.ExtensionRegistry()
  current = _state(extension_state.ExtensionStates())
  host = registry.create_host(
    extension.Context(
      extension.Workspace(tmp_path, remote=False),
      lambda _: None,
      lambda: current,
      lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda *_: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
    )
  )
  try:
    assert await registry.load(first) == 'first'
    (second / 'extension.py').write_text(
      source
      .replace('first', 'second')
      .replace("'tab'", "'c-i'")
      .replace('QuitRequested', 'RerenderRequested')
    )
    assert await registry.load(second) == 'second'
    resolved = keymap.resolve((host.keybindings,))
    assert resolved[0].action == actions.RerenderRequested()
    await registry.unload('second')
    resolved = keymap.resolve((host.keybindings,))
    assert resolved[0].action == actions.QuitRequested()
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', ['.git', '.jj'])
async def test_registry_loads_global_and_ancestor_extensions(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  root = tmp_path / 'repo'
  cwd = root / 'nested'
  cwd.mkdir(parents=True)
  (root / marker).mkdir()
  global_dir = tmp_path / 'global'
  for directory, name in (
    (global_dir, 'global'),
    (root / '.mirror' / 'extensions', 'root'),
    (cwd / '.mirror' / 'extensions', 'nested'),
  ):
    plugin = directory / name
    plugin.mkdir(parents=True)
    (plugin / 'extension.py').write_text(
      'from mirror.cli import extension\n'
      'class Plugin(extension.Extension[None, None, None]):\n'
      '  @classmethod\n'
      f'  def id(cls): return "{name}"\n'
    )
  registry = extension.ExtensionRegistry()
  try:
    manager = user_extensions.UserExtensions(registry, global_dir, cwd=cwd)
    await manager.start(trust_extensions=True)
    assert registry.extension_ids() == (
      'global',
      'root',
      'nested',
    )
  finally:
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('trust', 'expected'),
  [(False, False), (True, True)],
)
@pytest.mark.parametrize('symlink', [False, True])
async def test_extensions_require_trust_before_import(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
  trust: bool,
  expected: bool,
  symlink: bool,
) -> None:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  directory = tmp_path / '.mirror' / 'extensions' / 'untrusted'
  directory.mkdir(parents=True)
  marker = tmp_path / 'executed'
  (directory / 'extension.py').write_text(
    'from pathlib import Path\n'
    f'Path({str(marker)!r}).touch()\n'
    'from mirror.cli import extension\n'
    'class Plugin(extension.Extension[None, None, None]):\n'
    '  @classmethod\n'
    '  def id(cls): return "untrusted"\n'
  )
  if symlink:
    (directory / 'extension.py').rename(tmp_path / 'outside.py')
    (directory / 'extension.py').symlink_to(tmp_path / 'outside.py')
  registry = extension.ExtensionRegistry()
  try:
    manager = user_extensions.UserExtensions(registry, directory.parent)
    await manager.start(trust_extensions=trust)
    assert bool(registry.extension_ids()) == expected
    assert marker.exists() == expected
    assert capsys.readouterr().err == ''
  finally:
    await registry.close()
