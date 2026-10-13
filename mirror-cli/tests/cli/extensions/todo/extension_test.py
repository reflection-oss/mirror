from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable
from unittest import mock

import pytest
from google.protobuf import json_format, struct_pb2
from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers

from mirror.cli import (
  actions,
  effects,
  extension,
  extension_state,
  reducer,
  state,
)
from mirror.cli.extensions.todo import extension as todo


def _application(
  extension_states: extension_state.ExtensionStates,
) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    extensions=extension_states,
  )


async def _bound_extension() -> tuple[
  extension.ExtensionHost,
  Callable[[], state.State],
  list[effects.RunExtensionEffect],
  list[agent_pb2.AgentMessage],
  extension.ExtensionRegistry,
]:
  current: state.State
  pending: list[effects.RunExtensionEffect] = []
  records: list[agent_pb2.AgentMessage] = []

  def dispatch(action: actions.Action) -> None:
    nonlocal current
    extension_states, emitted = host.reduce(current, action)
    current, core_effects = reducer.reduce(current, action)
    assert core_effects == ()
    current = dataclasses.replace(current, extensions=extension_states)
    pending.extend(emitted)

  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=dispatch,
    get_state=lambda: current,
    enqueue=records.append,
    refresh_tools=lambda name: registry.refresh_tools(name),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(
    todo.TodoExtension(), None, overrides={'enabled': True}
  )
  host = registry.create_host(context)
  current = _application(host.initial_states)
  await host.start()
  return host, lambda: current, pending, records, registry


@pytest.mark.asyncio
async def test_extension_available_in_remote_sessions() -> None:
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=True),
    dispatch=lambda _: None,
    get_state=lambda: _application(extension_state.ExtensionStates()),
    enqueue=lambda _message: None,
    refresh_tools=lambda name: registry.refresh_tools(name),
    call_tool=lambda *_: pytest.fail('Unexpected tool call'),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=lambda *_: pytest.fail('Unexpected agent creation'),
  )
  registry = extension.ExtensionRegistry()
  await registry.register(
    todo.TodoExtension(), None, overrides={'enabled': True}
  )
  host = registry.create_host(context)

  assert host.initial_states['core/todo'] == todo.TodoState()
  await host.close()
  await registry.close()


@pytest.mark.asyncio
async def test_tool_updates_renders_and_persists_todos() -> None:
  host, get_application, pending, records, registry = await _bound_extension()
  initial = get_application()

  added = await host.host_tool_registry.execute(
    'todo__todo', '{"action":"add","text":"write tests"}'
  )
  toggled = await host.host_tool_registry.execute(
    'todo__todo', '{"action":"toggle","id":1}'
  )
  listed = await host.host_tool_registry.execute(
    'todo__todo', '{"action":"list"}'
  )

  current = get_application()
  assert initial.extensions['core/todo'] == todo.TodoState()
  assert current.extensions['core/todo'] == todo.TodoState(
    (todo.Todo(1, 'write tests', done=True),), next_id=2
  )
  assert added.content == 'Added todo #1: write tests'
  assert toggled.content == 'Todo #1 completed'
  assert listed.content == '[x] #1: write tests'
  rendered = containers.to_container(
    host.render(current, extension.RenderRegion.WIDGET_ABOVE_PROMPT)
  )
  assert rendered.preferred_height(80, 100).preferred == 2

  for effect in pending:
    await host.handle_effect(effect)

  snapshot = records[-1]
  assert snapshot.id
  assert snapshot.extension.extension == 'core/todo'
  assert snapshot.extension.kind == 'state'
  assert snapshot.extension.survives_compaction
  assert json_format.MessageToDict(snapshot.extension.data) == {
    'todos': [{'id': 1, 'text': 'write tests', 'done': True}],
    'next_id': 2,
  }

  await registry.unload('core/todo')
  assert 'core/todo' not in get_application().extensions
  assert host.host_tool_registry.tools() == ()
  assert not (await host.host_tool_registry.execute('todo__todo', '{}')).success


@pytest.mark.asyncio
async def test_host_rejects_invalid_tool_arguments() -> None:
  host, _, pending, _, _ = await _bound_extension()

  result = await host.host_tool_registry.execute('todo__todo', '{')

  assert not result.success
  assert result.content.startswith('Invalid tool arguments:')
  assert pending == []


@pytest.mark.asyncio
async def test_invalid_tool_input_does_not_change_state() -> None:
  host, get_application, pending, _, _ = await _bound_extension()

  result = await host.host_tool_registry.execute(
    'todo__todo', '{"action":"toggle","id":9}'
  )

  assert not result.success
  assert result.content == 'Error: todo #9 not found'
  assert get_application().extensions['core/todo'] == (todo.TodoState())
  assert pending == []


@pytest.mark.asyncio
async def test_hydration_uses_only_the_latest_todo_snapshot() -> None:
  host, get_application, _, _, _ = await _bound_extension()
  first = _record({
    'todos': [{'id': 1, 'text': 'first', 'done': False}],
    'next_id': 2,
  })
  invalid = _record({'todos': 'invalid', 'next_id': 3})

  hydrated, _ = host.reduce(
    get_application(), actions.HydrateExtensions((first,))
  )
  invalid_latest, _ = host.reduce(
    get_application(), actions.HydrateExtensions((first, invalid))
  )

  assert hydrated['core/todo'] == todo.TodoState(
    (todo.Todo(1, 'first'),), next_id=2
  )
  assert invalid_latest['core/todo'] == todo.TodoState()


def _record(data: dict[str, object]) -> agent_pb2.AgentMessage:
  structured = struct_pb2.Struct()
  json_format.ParseDict(data, structured)
  return agent_pb2.AgentMessage(
    extension=agent_pb2.ExtensionRecord(
      extension='core/todo', kind='state', data=structured
    )
  )
