from __future__ import annotations

import dataclasses
import pathlib
from typing import cast
from unittest import mock

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers, mouse_handlers, screen
from rich import text as rich_text

from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  extension_state,
  state,
)
from mirror.cli.extensions.context_usage import extension as context_usage
from mirror.cli.extensions.tokens_per_second import (
  extension as tokens_per_second,
)
from mirror.cli.extensions.view_mode import extension as view_mode


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


def _context_message(tokens: int, window: int) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      context_usage=agent_pb2.ContextUsage(tokens=tokens, window=window)
    )
  )


def test_measurement_renders_compact_percent_and_window() -> None:
  assert context_usage.ContextUsage().render() == '0% ctx'
  assert context_usage.ContextUsage(65_536, 128_000).render() == (
    '51% ctx · 128K'
  )
  assert context_usage.ContextUsage(600_000, 1_000_000).render() == (
    '60% ctx · 1M'
  )
  assert context_usage.ContextUsage(600_000, 1_500_000).render() == (
    '40% ctx · 1.5M'
  )


@pytest.mark.asyncio
async def test_status_extensions_observe_and_render_in_explicit_order() -> None:
  current: state.State
  dispatched: list[actions.Action] = []
  context = extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=dispatched.append,
    get_state=lambda: current,
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
  await registry.register(view_mode.ViewModeExtension(), None)
  context_entrypoint = pathlib.Path(cast(str, context_usage.__file__)).parent
  tokens_entrypoint = pathlib.Path(cast(str, tokens_per_second.__file__)).parent
  assert await registry.load(context_entrypoint) == 'core/context-usage'
  assert await registry.load(tokens_entrypoint) == 'core/tokens-per-second'
  host = registry.create_host(context)
  current = _application(host.initial_states)
  await host.start()
  dispatched.clear()

  host.handle_message(agent_pb2.SessionMessage(tokens_per_second=42.0))
  host.handle_message(_context_message(65_536, 128_000))
  extension_states, token_effects = host.reduce(current, dispatched[0])
  current = dataclasses.replace(current, extensions=extension_states)
  extension_states, context_effects = host.reduce(current, dispatched[1])
  current = dataclasses.replace(current, extensions=extension_states)
  rendered = containers.to_container(
    host.render(current, extension.RenderRegion.STATUS_RIGHT)
  )

  assert token_effects == context_effects == ()
  assert isinstance(rendered, containers.VSplit)
  width = rendered.preferred_width(80).preferred
  painted = screen.Screen()
  rendered.write_to_screen(
    painted,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, width, 1),
    '',
    True,
    None,
  )
  text = ''.join(painted.data_buffer[0][x].char for x in range(width))
  assert text.endswith(' 42 tok/s · 51% ctx · 128K  COMPACT  ')
  assert 'class:ui.view-compact' in painted.data_buffer[0][width - 2].style
  assert 'class:footer' in painted.data_buffer[0][width - 1].style

  switching = dataclasses.replace(
    current,
    session=dataclasses.replace(
      current.session,
      pending_switch=actions.PendingSwitch('next', create=False),
    ),
  )
  assert isinstance(
    host.render(switching, extension.RenderRegion.STATUS_RIGHT),
    containers.Window,
  )

  await host.close()
  await registry.close()


def test_context_retains_report_after_failure_and_replay() -> None:
  plugin = context_usage.ContextUsageExtension()
  current = _application(
    extension_state.ExtensionStates({plugin.id(): plugin.initial_state()})
  )
  report = agent_pb2.SystemPromptReport(
    estimated_tokens=123,
    tool_tokens=77,
    tools=[
      agent_pb2.SystemPromptReport.Part(source='read', estimated_tokens=7),
      agent_pb2.SystemPromptReport.Part(
        source='mcp__slack__search', estimated_tokens=70
      ),
    ],
    parts=[
      agent_pb2.SystemPromptReport.Part(
        source='/repo/AGENTS.md', estimated_tokens=123
      )
    ],
  )
  updated, _ = plugin.reduce(
    current,
    actions.RoutedExtensionAction(
      plugin.id(), context_usage.PromptObserved(report, 'window')
    ),
  )
  current = dataclasses.replace(
    current, extensions=current.extensions.updated(plugin.id(), updated)
  )
  retained, emitted = plugin.reduce(
    current, actions.WorkerFailed('Invalid prompt budget: too large')
  )
  assert retained.prompt_report == report
  assert len(emitted) == 1
  assert isinstance(emitted[0], effects.EmitText)
  assert isinstance(emitted[0].text, rich_text.Text)
  assert '/repo/AGENTS.md' in emitted[0].text.plain
  assert emitted[0].text.spans
  lines = emitted[0].text.plain.splitlines()
  assert lines[0].split() == ['◆', 'Context', '~200']
  assert any(line.split() == ['MCP', '·', 'slack', '~70'] for line in lines)
  assert any(line.split() == ['Native', '~7'] for line in lines)
  assert any(line.split() == ['search', '~70'] for line in lines)
  assert "'" not in emitted[0].text.plain
  stale = agent_pb2.AgentMessage(
    id='window',
    context_window=agent_pb2.ContextWindowRecord(
      system_prompt='saved',
      system_prompt_report=agent_pb2.SystemPromptReport(tool_tokens=999),
    ),
  )
  assert plugin.hydrate(None, [stale], current).prompt_report == report
  context = mock.Mock(
    spec=extension.Context,
    get_state=mock.Mock(return_value=current),
    dispatch=mock.Mock(),
  )
  plugin.handle_message(context, agent_pb2.SessionMessage(agent_message=stale))
  context.dispatch.assert_not_called()
  assert plugin.hydrate(None, [], current).prompt_report == report
  command = plugin.slash_commands()[0]
  assert command.name == 'context' and command.available_when_busy
  request = command.handler(
    commands.CommandContext(
      'model', 'medium', 'conversation', False, '/workspace'
    ),
    '',
  )
  assert plugin.reduce(current, request)[1] == emitted
  empty = _application(
    extension_state.ExtensionStates({plugin.id(): plugin.initial_state()})
  )
  assert plugin.reduce(empty, request)[1] == (
    effects.EmitText('System prompt: not measured yet.'),
  )
