"""The models and speeds a remote tab may pick, read from its server."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Sequence
from typing import Protocol, cast

import grpc
from mirror.protocol import (
  mirror_pb2,
  mirror_pb2_grpc,
)

from mirror.cli import (
  actions,
  commands,
  extension,
  remote_rpc_client,
  selection,
)

EXTENSION_ID = 'core/inference-controls'
_SERVER_DEFAULT = 'server default'
_RESOLUTION_ERRORS = (
  ValueError,
  TypeError,
  OSError,
  RuntimeError,
  grpc.RpcError,
  TimeoutError,
)


class _ModelStub(Protocol):
  def ListModels(  # noqa: N802, generated RPC name
    self, request: mirror_pb2.ListModelsRequest, *, timeout: float
  ) -> Awaitable[mirror_pb2.ListModelsResponse]: ...


async def list_models(
  endpoint: str, *, secure: bool = True
) -> list[mirror_pb2.Model]:
  """Read the models the server runs, in the order it lists them."""
  transport = remote_rpc_client.Transport(endpoint, secure=secure)
  try:
    channel = (await transport.connect()).channel
    stub = cast(_ModelStub, mirror_pb2_grpc.MirrorRuntimeStub(channel))
    response = await stub.ListModels(mirror_pb2.ListModelsRequest(), timeout=30)
    return list(response.models)
  finally:
    await transport.close()


@dataclasses.dataclass(frozen=True)
class ServerModel(actions.ExtensionAction):
  model_id: str | None
  current: str | None = None


@dataclasses.dataclass(frozen=True)
class ServerFastMode(actions.ExtensionAction):
  model_id: str


def _session_models(
  listed: Sequence[mirror_pb2.Model],
) -> list[mirror_pb2.Model]:
  return [model for model in listed if not model.subagent_only]


def session_model(
  listed: Sequence[mirror_pb2.Model], model_id: str
) -> mirror_pb2.Model | None:
  """The listed session model a remote tab runs, resolving the server default."""
  models = _session_models(listed)
  if model_id in ('', _SERVER_DEFAULT):
    return next((model for model in models if model.default), None)
  return next((model for model in models if model.id == model_id), None)


def server_model_choices(
  listed: Sequence[mirror_pb2.Model],
) -> tuple[selection.Choice, ...]:
  """The server's session models as picker rows, in the server's order."""
  return tuple(
    selection.Choice(
      model.id,
      model.label,
      ' · '.join((
        *(('default',) if model.default else ()),
        model.id,
        f'{model.context_window:,} tokens',
      )),
    )
    for model in _session_models(listed)
  )


def server_model_action(
  listed: Sequence[mirror_pb2.Model], model_id: str
) -> actions.Action:
  """Switch to `model_id` when the server lists it for sessions."""
  model = next(
    (model for model in _session_models(listed) if model.id == model_id),
    None,
  )
  if model is None:
    return actions.ShowText(
      f"{model_id} is not on the server's model list.", style='ui.warning'
    )
  return actions.SetRemoteModel(model.id, model.api)


def model_command(
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  """Pick the model a remote tab runs from the server's list."""
  if len(arguments.split()) > 1:
    return actions.ShowText(
      'Model id must not contain spaces.', style='ui.warning'
    )
  return actions.RoutedExtensionAction(
    EXTENSION_ID, ServerModel(arguments or None, context.model)
  )


def _choose_model(
  endpoint: str, request: ServerModel, *, secure: bool = True
) -> actions.OpenSelector:
  """Open immediately and list the server's models as they load."""
  available: list[mirror_pb2.Model] = []

  async def load() -> tuple[selection.Choice, ...]:
    try:
      listed = await list_models(endpoint, secure=secure)
    except _RESOLUTION_ERRORS as error:
      return (selection.Choice('retry', 'Retry server models', str(error)),)
    available.extend(listed)
    return server_model_choices(listed)

  def on_select(value: str) -> actions.Action:
    if value == 'retry':
      return actions.RoutedExtensionAction(EXTENSION_ID, request)
    return server_model_action(available, value)

  return actions.OpenSelector(
    selection.Selector(
      'model',
      'Switch model on the server',
      (),
      on_select,
      extension_id=EXTENSION_ID,
      current=request.current,
      load=load,
    )
  )


async def server_model(
  context: extension.Context, request: ServerModel
) -> actions.Action:
  """Pick from, or check a typed id against, the server's model list."""
  endpoint = context.remote_endpoint
  if endpoint is None:
    return actions.ShowText(
      'Configure --remote-endpoint before choosing a server model.',
      style='ui.error',
    )
  if request.model_id is None:
    return _choose_model(endpoint, request, secure=context.remote_secure)
  try:
    listed = await list_models(endpoint, secure=context.remote_secure)
  except _RESOLUTION_ERRORS as error:
    return actions.ShowText(
      f'Could not read the server models: {error}', style='ui.error'
    )
  return server_model_action(listed, request.model_id)


def fast_mode_actions(
  listed: Sequence[mirror_pb2.Model], model_id: str
) -> tuple[actions.Action, ...]:
  """Turn on fast mode when the server lists it for the tab's model.

  The tab is first switched to that model explicitly, so a session on the
  server default learns the protocol that decides how fast mode is requested.
  """
  model = session_model(listed, model_id)
  if model is None:
    return (
      actions.ShowText(
        f"{model_id} is not on the server's model list.", style='ui.warning'
      ),
    )
  if not model.fast_mode:
    return (
      actions.ShowText(
        f'{model.label} has no fast mode on this server.', style='ui.warning'
      ),
    )
  return (actions.SetRemoteModel(model.id, model.api), actions.SetSpeed('fast'))


def speed_command(
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  """Set a remote tab's speed, checking fast mode against the server's list."""
  speed = arguments.strip().lower()
  if speed == 'fast':
    return actions.RoutedExtensionAction(
      EXTENSION_ID, ServerFastMode(context.model)
    )
  if speed == 'normal':
    return actions.SetSpeed(speed)
  if speed:
    return actions.ShowText(
      f'Unknown speed: {arguments}. Choices: normal, fast', style='ui.warning'
    )
  return actions.OpenSelector(
    selection.Selector(
      'speed',
      'Model speed',
      (
        selection.Choice('normal', 'Normal', 'Provider default processing'),
        selection.Choice(
          'fast',
          'Fast',
          'Faster processing; may cost more or use limits faster',
        ),
      ),
      lambda value: speed_command(context, value),
      current=context.speed,
      selected=context.speed,
      extension_id=EXTENSION_ID,
    )
  )


async def server_fast_mode(
  context: extension.Context, request: ServerFastMode
) -> tuple[actions.Action, ...]:
  """Check the tab's model against the server's list before going fast."""
  endpoint = context.remote_endpoint
  if endpoint is None:
    return (
      actions.ShowText(
        'Configure --remote-endpoint before changing the model speed.',
        style='ui.error',
      ),
    )
  try:
    listed = await list_models(endpoint, secure=context.remote_secure)
  except _RESOLUTION_ERRORS as error:
    return (
      actions.ShowText(
        f'Could not read the server models: {error}', style='ui.error'
      ),
    )
  return fast_mode_actions(listed, request.model_id)
