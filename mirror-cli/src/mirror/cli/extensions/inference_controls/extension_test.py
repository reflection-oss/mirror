"""Live provider catalogs in the model picker."""

from __future__ import annotations

import asyncio
import pathlib
from unittest import mock

import pytest

from mirror.cli import (
  actions,
  commands,
  effects,
  inference_choices,
  models,
  reducer,
  reflection,
  selection,
  state,
)
from mirror.cli.extensions.inference_controls import (
  extension as inference_controls,
)


def _state() -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
  )


def _dispatch(command: str) -> actions.Action | None:
  available = (
    *commands.COMMANDS,
    *inference_controls.InferenceControlsExtension().slash_commands(),
  )
  context = commands.CommandContext(
    'model', 'medium', 'conversation', False, '/workspace'
  )
  return commands.dispatch(command, context, available)


@pytest.mark.asyncio
@pytest.mark.parametrize('environment_key', ['', 'synthetic-environment-key'])
async def test_reflection_picker_loads_all_account_models(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
  environment_key: str,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('REFLECTION_API_KEY', environment_key)
  reflection.save_key('synthetic-saved-key')
  fetch = mock.AsyncMock(
    return_value={
      'data': [{'id': 'Beam-1.0-501B-A23B'}, {'id': 'example/other'}],
    }
  )

  action = _dispatch('/model reflection')

  assert isinstance(action, actions.OpenSelector)
  assert action.selector.load is not None
  with mock.patch.object(inference_choices, '_fetch_json', fetch):
    choices = await action.selector.load()
  assert choices == (
    selection.Choice('Beam-1.0-501B-A23B', 'Beam-1.0-501B-A23B'),
    selection.Choice('example/other', 'example/other'),
  )
  fetch.assert_awaited_once_with(
    'https://api.reflection.ai/openai/v1/models',
    headers={
      'Authorization': f'Bearer {environment_key or "synthetic-saved-key"}'
    },
  )
  assert action.selector.on_select('example/other') == actions.ModelResolved(
    models.ModelOption('example/other', 'example/other', api='reflection')
  )


def test_reflection_without_a_key_routes_to_sign_in(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)

  assert _dispatch('/model reflection') == actions.ModelResolved(
    models.ModelOption('Beam-1.0-501B-A23B', 'reflection', api='reflection')
  )


def test_vercel_without_a_key_routes_to_its_default_model(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('AI_GATEWAY_API_KEY', raising=False)

  action = _dispatch('/model vercel')

  assert action == actions.ModelResolved(
    models.ModelOption('reflection/beam', 'vercel', api='vercel')
  )


def test_vercel_with_a_key_opens_its_gateway_catalog(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('AI_GATEWAY_API_KEY', 'synthetic-gateway-key')

  action = _dispatch('/model vercel')

  assert isinstance(action, actions.OpenSelector)
  assert action.selector.id == 'vercel-models'
  assert action.selector.load is inference_choices.fetch_vercel_models
  assert action.selector.on_select('reflection/beam') == actions.ModelResolved(
    models.ModelOption('reflection/beam', 'reflection/beam', api='vercel')
  )
  _, load_effects = reducer.reduce(_state(), action)
  assert effects.LoadSelectorChoices('vercel-models') in load_effects


def test_vercel_catalog_offers_only_language_models(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('AI_GATEWAY_API_KEY', 'synthetic-gateway-key')
  payload = {
    'object': 'list',
    'data': [
      {
        'id': 'reflection/beam',
        'name': 'Beam',
        'type': 'language',
        'context_window': 262144,
        'pricing': {'input': '0.000001', 'output': '0.000004'},
      },
      {'id': 'openai/text-embedding-3-large', 'type': 'embedding'},
      {'id': 'openai/gpt-image-2', 'type': 'image'},
      {'id': 'example/untyped'},
      {'not': 'a model'},
    ],
  }
  fetch = mock.AsyncMock(return_value=payload)

  with mock.patch.object(inference_choices, '_fetch_json', fetch):
    choices = asyncio.run(inference_choices.fetch_vercel_models())

  fetch.assert_awaited_once_with(
    'https://ai-gateway.vercel.sh/v1/models',
    headers={'Authorization': 'Bearer synthetic-gateway-key'},
  )
  assert choices == (
    selection.Choice(
      'reflection/beam',
      'Beam',
      'reflection/beam · 262,144 ctx · $1.00/M in',
    ),
    selection.Choice('example/untyped', 'example/untyped', 'example/untyped'),
  )
