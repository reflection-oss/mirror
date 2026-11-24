"""Model and speed selection in a remote tab, against a server's list."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import cast, override
from unittest import mock

import grpc.aio
import pytest
from mirror.protocol import (
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import actions, commands, extension, models, selection, state
from mirror.cli.extensions.inference_controls import (
  extension as inference_controls,
)

_MODELS = (
  mirror_pb2.Model(
    id='gpt-6-astra',
    label='GPT-6 Astra',
    api='openai-responses',
    context_window=272_000,
    default=True,
    fast_mode=True,
  ),
  mirror_pb2.Model(
    id='claude-opus-5-5',
    label='Claude Opus 5.5',
    api='anthropic-messages',
    context_window=1_000_000,
  ),
  mirror_pb2.Model(
    id='helper',
    label='Helper',
    api='openai-responses',
    context_window=128_000,
    subagent_only=True,
  ),
)
_STATE = state.State(
  lifecycle=state.LifecycleState(active=True),
  activity=state.ActivityState(),
  status_bar=state.StatusBarState('gpt-6-astra', 'medium', 'auto'),
  session=state.SessionState('conversation-1', remote=True),
)


class _Runtime(mirror_pb2_grpc.MirrorRuntimeServicer):
  @override
  async def ListModels(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ListModelsRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.ListModelsRequest,
      mirror_pb2.ListModelsResponse,
    ],
  ) -> mirror_pb2.ListModelsResponse:
    del request, context
    return mirror_pb2.ListModelsResponse(models=_MODELS)


@contextlib.asynccontextmanager
async def _server() -> AsyncGenerator[str]:
  server = grpc.aio.server()
  mirror_pb2_grpc.add_MirrorRuntimeServicer_to_server(  # pyright: ignore[reportUnknownMemberType] - protoc registration is untyped.
    _Runtime(), server
  )
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  try:
    yield f'127.0.0.1:{port}'
  finally:
    await server.stop(0)


def _context(
  endpoint: str, dispatched: list[actions.Action]
) -> extension.Context:
  return extension.Context(
    workspace=mock.Mock(),
    dispatch=dispatched.append,
    get_state=lambda: _STATE,
    enqueue=lambda _: None,
    refresh_tools=lambda _: asyncio.sleep(0),
    call_tool=mock.AsyncMock(),
    transcript=tuple,
    builtins=tuple,
    extension_tools=tuple,
    workspace_controller=mock.Mock(),
    create_agent=mock.AsyncMock(),
    remote_endpoint=endpoint,
    remote_secure=False,
  )


def _command_context(model: str) -> commands.CommandContext:
  return commands.CommandContext(
    model, 'medium', 'conversation-1', False, '/workspace', remote=True
  )


async def _run(
  command: str, endpoint: str, model: str = 'gpt-6-astra'
) -> list[actions.Action]:
  routed = commands.dispatch(
    command,
    _command_context(model),
    inference_controls.InferenceControlsExtension().slash_commands(),
  )
  return await _handle(routed, endpoint)


async def _handle(
  routed: actions.Action | None, endpoint: str
) -> list[actions.Action]:
  server_models = cast(
    extension.Extension[None, object, None],
    inference_controls.InferenceControlsExtension(),
  )
  assert isinstance(routed, actions.RoutedExtensionAction)
  assert routed.extension_id == server_models.id()
  _, (effect,) = server_models.reduce(_STATE, routed)
  assert isinstance(effect, extension.PrivateEffect)
  dispatched: list[actions.Action] = []
  await server_models.handle_effect(
    _context(endpoint, dispatched), effect.payload
  )
  return dispatched


@pytest.mark.asyncio
async def test_model_picker_lists_the_servers_session_models() -> None:
  async with _server() as endpoint:
    (opened,) = await _run('/model', endpoint)
    assert isinstance(opened, actions.OpenSelector)
    assert opened.selector.load is not None
    choices = await opened.selector.load()

  assert choices == (
    selection.Choice(
      'gpt-6-astra', 'GPT-6 Astra', 'default · gpt-6-astra · 272,000 tokens'
    ),
    selection.Choice(
      'claude-opus-5-5',
      'Claude Opus 5.5',
      'claude-opus-5-5 · 1,000,000 tokens',
    ),
  )
  assert opened.selector.current == 'gpt-6-astra'
  assert opened.selector.on_select('claude-opus-5-5') == (
    actions.SetRemoteModel('claude-opus-5-5', 'anthropic-messages')
  )


@pytest.mark.asyncio
async def test_typed_model_switches_with_the_servers_protocol() -> None:
  async with _server() as endpoint:
    dispatched = await _run('/model claude-opus-5-5', endpoint)

  assert dispatched == [
    actions.SetRemoteModel('claude-opus-5-5', 'anthropic-messages')
  ]


@pytest.mark.parametrize('model_id', ['gpt-5.6-sol', 'helper'])
@pytest.mark.asyncio
async def test_typed_model_is_refused_unless_listed_for_sessions(
  model_id: str,
) -> None:
  async with _server() as endpoint:
    dispatched = await _run(f'/model {model_id}', endpoint)

  assert dispatched == [
    actions.ShowText(
      f"{model_id} is not on the server's model list.", style='ui.warning'
    )
  ]


@pytest.mark.asyncio
async def test_fast_speed_pins_the_server_default_before_going_fast() -> None:
  async with _server() as endpoint:
    dispatched = await _run('/speed fast', endpoint, model='server default')

  assert dispatched == [
    actions.SetRemoteModel('gpt-6-astra', 'openai-responses'),
    actions.SetSpeed('fast'),
  ]


@pytest.mark.asyncio
async def test_fast_speed_is_refused_for_a_model_the_server_runs_normally() -> (
  None
):
  async with _server() as endpoint:
    dispatched = await _run('/speed fast', endpoint, model='claude-opus-5-5')

  assert dispatched == [
    actions.ShowText(
      'Claude Opus 5.5 has no fast mode on this server.', style='ui.warning'
    )
  ]


@pytest.mark.asyncio
async def test_speed_picker_checks_fast_against_the_server_list() -> None:
  server_models = inference_controls.InferenceControlsExtension()
  opened = commands.dispatch(
    '/speed',
    _command_context('server default'),
    server_models.slash_commands(),
  )
  assert isinstance(opened, actions.OpenSelector)

  async with _server() as endpoint:
    dispatched = await _handle(opened.selector.on_select('fast'), endpoint)

  assert opened.selector.on_select('normal') == actions.SetSpeed('normal')
  assert dispatched == [
    actions.SetRemoteModel('gpt-6-astra', 'openai-responses'),
    actions.SetSpeed('fast'),
  ]


def test_speed_in_a_local_tab_ignores_the_server() -> None:
  server_models = inference_controls.InferenceControlsExtension()

  action = commands.dispatch(
    '/speed fast',
    commands.CommandContext(
      'claude-opus-5-5',
      'medium',
      'conversation-1',
      False,
      '/workspace',
      model_selection=models.ModelOption(
        'claude-opus-5-5', 'claude-opus-5-5', api='anthropic-messages'
      ),
    ),
    server_models.slash_commands(),
  )

  assert action == actions.SetSpeed('fast')
