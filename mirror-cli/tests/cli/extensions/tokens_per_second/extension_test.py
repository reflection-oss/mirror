from __future__ import annotations

import dataclasses
import pathlib
from typing import cast

from unittest import mock

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers, mouse_handlers, screen

from mirror.cli import actions, extension, extension_state, state
from mirror.cli.extensions.tokens_per_second import (
  extension as tokens_per_second,
)


def _render_text(container: containers.AnyContainer) -> str:
  content = containers.to_container(container)
  width = content.preferred_width(80).preferred
  painted = screen.Screen()
  content.write_to_screen(
    painted,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, width, 1),
    '',
    True,
    None,
  )
  return ''.join(painted.data_buffer[0][x].char for x in range(width))


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


@pytest.mark.asyncio
async def test_extension_observes_reduces_and_renders_agent_measurement() -> (
  None
):
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
  entrypoint = pathlib.Path(cast(str, tokens_per_second.__file__)).parent
  assert await registry.load(entrypoint) == 'core/tokens-per-second'
  host = registry.create_host(context)
  current = _application(host.initial_states)
  await host.start()

  host.handle_message(agent_pb2.SessionMessage(tokens_per_second=10.0))
  extension_states, effects = host.reduce(current, dispatched.pop())
  current = dataclasses.replace(current, extensions=extension_states)
  rendered = containers.to_container(
    host.render(current, extension.RenderRegion.STATUS_RIGHT)
  )

  assert effects == ()
  assert isinstance(rendered, containers.VSplit)
  first = _render_text(rendered.children[0])
  assert first.endswith(' 10 tok/s')
  assert len(first) == 17

  host.handle_message(agent_pb2.SessionMessage(tokens_per_second=20.0))
  extension_states, _ = host.reduce(current, dispatched.pop())
  current = dataclasses.replace(current, extensions=extension_states)
  host.handle_message(agent_pb2.SessionMessage(tokens_per_second=30.0))
  extension_states, _ = host.reduce(current, dispatched.pop())
  current = dataclasses.replace(current, extensions=extension_states)
  host.handle_message(agent_pb2.SessionMessage(tokens_per_second=40.0))
  extension_states, _ = host.reduce(current, dispatched.pop())
  current = dataclasses.replace(current, extensions=extension_states)
  rendered = containers.to_container(
    host.render(current, extension.RenderRegion.STATUS_RIGHT)
  )
  assert isinstance(rendered, containers.VSplit)
  final = _render_text(rendered.children[0])
  assert len(final) == len(first)
  glyphs = [glyph for glyph in final if glyph in '▁▂▃▄▅▆▇█']
  assert glyphs == sorted(glyphs) and len(set(glyphs)) > 1

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


@pytest.mark.asyncio
async def test_extension_uses_worker_usage_despite_batched_delivery() -> None:
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
  await registry.register(tokens_per_second.TokensPerSecondExtension(), None)
  host = registry.create_host(context)
  current = _application(host.initial_states)
  await host.start()

  def observe(message: agent_pb2.SessionMessage) -> str:
    nonlocal current
    host.handle_message(message)
    extension_states, effects = host.reduce(current, dispatched.pop())
    assert effects == ()
    current = dataclasses.replace(current, extensions=extension_states)
    rendered = containers.to_container(
      host.render(current, extension.RenderRegion.STATUS_RIGHT)
    )
    assert isinstance(rendered, containers.VSplit)
    return _render_text(rendered.children[0])

  dispatched.clear()
  host.handle_message(
    agent_pb2.SessionMessage(
      runtime_delta=agent_pb2.RuntimeDelta(
        stream_content=agent_pb2.StreamContent(content='x' * 100_000),
      )
    )
  )
  assert dispatched == []

  initial = observe(_usage(40, 5.0))
  assert '- tok/s' in initial
  assert '- tok/s' in observe(_usage(44, 5.001))
  # Deliver instantly: throughput depends on worker time, not arrival time.
  live = observe(_usage(120, 7.0))
  assert ' 40 tok/s' in live
  assert observe(_usage(120, 7.0)) == live
  assert observe(_usage(200, 9.0)) == live

  finished = observe(
    agent_pb2.SessionMessage(
      agent_message=agent_pb2.AgentMessage(
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(),
          usage=agent_pb2.Usage(completion_tokens=120),
        )
      ),
      stream_final=True,
      tokens_per_second=1.0,
    )
  )
  assert ' 40 tok/s' in finished
  assert ' 40 tok/s' in observe(_usage(10, 1.0))
  assert ' 40 tok/s' in observe(
    agent_pb2.SessionMessage(
      runtime_delta=agent_pb2.RuntimeDelta(stream_aborted={})
    )
  )
  assert ' 40 tok/s' in observe(_usage(900, 20.0))
  assert ' 43 tok/s' in observe(_usage(1000, 22.0))
  assert ' 43 tok/s' in observe(
    agent_pb2.SessionMessage(stream_final=True, tokens_per_second=1.0)
  )

  await host.close()
  await registry.close()


def _usage(tokens: int, elapsed: float) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      stream_usage=agent_pb2.StreamUsage(
        completion_tokens=tokens, elapsed_seconds=elapsed
      ),
    )
  )
