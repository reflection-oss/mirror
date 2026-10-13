from __future__ import annotations

import asyncio
import json
import pathlib
from unittest import mock

import pytest

from mirror.cli import (
  actions,
  api_keys,
  commands,
  effects,
  extension,
  models,
  reducer,
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


def _context(current: state.State) -> extension.Context:
  return extension.Context(
    extension.Workspace(pathlib.Path('/workspace'), remote=False),
    dispatch=lambda _: None,
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


def _commands(
  current: state.State,
) -> tuple[commands.CommandDef, ...]:
  del current
  return (
    *commands.COMMANDS,
    *inference_controls.InferenceControlsExtension().slash_commands(),
  )


def _command_context() -> commands.CommandContext:
  return commands.CommandContext(
    'model', 'medium', 'conversation', False, '/workspace'
  )


@pytest.mark.parametrize('effort', ['high', 'max'])
def test_extension_commands_return_application_actions(effort: str) -> None:
  current = _state()
  available = _commands(current)

  model = commands.dispatch('/model next', _command_context(), available)
  reasoning = commands.dispatch(
    f'/reasoning {effort}', _command_context(), available
  )

  assert model == actions.SetModel('next')
  assert reasoning == actions.SetReasoning(effort)
  assert model is not None
  assert reducer.reduce(current, model) == (
    current,
    (effects.ResolveModel('next'),),
  )
  help_text = commands.dispatch('/help', _command_context(), available)
  assert isinstance(help_text, actions.ShowText)
  assert '/model' in help_text.text
  assert '/reasoning' in help_text.text


def test_extension_commands_validate_arguments() -> None:
  available = _commands(_state())

  model = commands.dispatch('/model two models', _command_context(), available)
  reasoning = commands.dispatch(
    '/reasoning extreme', _command_context(), available
  )

  assert model == actions.ShowText(
    'Model id must not contain spaces.', style='ui.warning'
  )
  assert reasoning == actions.ShowText(
    'Unknown reasoning effort: extreme. Choices: '
    'none, minimal, low, medium, high, xhigh, max',
    style='ui.warning',
  )


def _clear_credentials(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Make the protocol readiness hints deterministic (no keys, no login)."""
  monkeypatch.setenv('HOME', str(tmp_path))
  for env in (
    'OPENAI_API_KEY',
    'OPENROUTER_API_KEY',
    'ANTHROPIC_API_KEY',
    'REFLECTION_API_KEY',
  ):
    monkeypatch.delenv(env, raising=False)


def test_extension_commands_open_selectors(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = tmp_path / 'models.json'
  config.write_text(json.dumps([{'id': 'next', 'label': 'Next'}]))
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  _clear_credentials(tmp_path, monkeypatch)
  available = _commands(_state())

  model = commands.dispatch('/model', _command_context(), available)
  reasoning = commands.dispatch('/reasoning', _command_context(), available)

  assert model == actions.OpenSelector(
    selection.Selector(
      'model',
      'Switch model or protocol',
      (
        selection.Choice(
          'reflection',
          'Reflection API',
          'reflection · select to sign in',
        ),
        selection.Choice(
          'openai-codex-responses',
          'ChatGPT plan billing (Codex login)',
          'openai-codex-responses · run codex login',
        ),
        selection.Choice(
          'openai-responses',
          'OpenAI Responses API',
          'openai-responses · set $OPENAI_API_KEY',
        ),
        selection.Choice(
          'openai-chat-completions',
          'OpenRouter',
          'openai-chat-completions · set $OPENROUTER_API_KEY',
        ),
        selection.Choice(
          'anthropic-messages',
          'Anthropic Messages API',
          'anthropic-messages · select to sign in',
        ),
        selection.Choice('next', 'Next', 'next'),
      ),
      inference_controls._select_inference,  # pyright: ignore[reportPrivateUsage]
      current='model',
      extension_id=inference_controls.InferenceControlsExtension.id(),
      empty='No models available.',
    )
  )
  assert reasoning == actions.OpenSelector(
    selection.Selector(
      'reasoning',
      'Reasoning effort',
      tuple(
        selection.Choice(
          value, 'provider default' if value == 'none' else value
        )
        for value in (
          'none',
          'minimal',
          'low',
          'medium',
          'high',
          'xhigh',
          'max',
        )
      ),
      actions.SetReasoning,
      current='medium',
      extension_id=inference_controls.InferenceControlsExtension.id(),
      selected='medium',
    )
  )


def test_model_command_offers_protocols_without_a_config(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # The complaint that motivated this: an empty config used to show nothing.
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(tmp_path / 'absent.json'))
  _clear_credentials(tmp_path, monkeypatch)
  available = _commands(_state())

  model = commands.dispatch('/model', _command_context(), available)

  assert isinstance(model, actions.OpenSelector)
  assert tuple(choice.value for choice in model.selector.choices) == (
    'reflection',
    'openai-codex-responses',
    'openai-responses',
    'openai-chat-completions',
    'anthropic-messages',
  )


def test_protocol_without_credentials_falls_back_to_default_model(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _clear_credentials(tmp_path, monkeypatch)
  current = _state()

  action = commands.dispatch(
    '/model anthropic-messages', _command_context(), _commands(current)
  )

  # No key, so skip the picker and reuse the generic model-switch path.
  assert action == actions.ModelResolved(
    models.ModelOption(
      'claude-opus-4-8', 'anthropic-messages', api='anthropic-messages'
    )
  )
  assert reducer.reduce(current, action) == (
    current,
    (
      effects.SwitchModel(
        models.ModelOption(
          'claude-opus-4-8',
          'anthropic-messages',
          api='anthropic-messages',
        )
      ),
    ),
  )


def test_keyed_protocols_open_a_live_fetched_picker(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _clear_credentials(tmp_path, monkeypatch)
  monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
  monkeypatch.setenv('OPENAI_API_KEY', 'test')

  anthropic = commands.dispatch(
    '/model anthropic-messages', _command_context(), _commands(_state())
  )
  openai = commands.dispatch(
    '/model openai-responses', _command_context(), _commands(_state())
  )

  assert isinstance(anthropic, actions.OpenSelector)
  assert anthropic.selector.id == 'anthropic-models'
  assert anthropic.selector.load is inference_controls._fetch_anthropic_models  # pyright: ignore[reportPrivateUsage]
  assert isinstance(openai, actions.OpenSelector)
  assert openai.selector.id == 'openai-models'
  assert openai.selector.load is inference_controls._fetch_openai_models  # pyright: ignore[reportPrivateUsage]


def test_codex_offers_a_curated_static_list() -> None:
  action = commands.dispatch(
    '/model openai-codex-responses', _command_context(), _commands(_state())
  )

  assert isinstance(action, actions.OpenSelector)
  assert action.selector.id == 'codex-models'
  assert action.selector.load is None
  assert action.selector.choices[0].value == 'gpt-6-astra'


def test_anthropic_catalog_parses_display_names() -> None:
  payload = {
    'data': [
      {
        'type': 'model',
        'id': 'claude-opus-4-8',
        'display_name': 'Claude Opus 4.8',
      },
      {'type': 'model', 'id': 'claude-haiku-4-5'},
    ]
  }

  choices = inference_controls._anthropic_choices(payload)  # pyright: ignore[reportPrivateUsage]

  assert choices == (
    selection.Choice('claude-opus-4-8', 'Claude Opus 4.8', 'claude-opus-4-8'),
    selection.Choice(
      'claude-haiku-4-5', 'claude-haiku-4-5', 'claude-haiku-4-5'
    ),
  )


def test_openai_catalog_filters_non_chat_models_on_openai_host() -> None:
  payload = {
    'data': [
      {'id': 'gpt-6-astra', 'owned_by': 'openai'},
      {'id': 'text-embedding-3-large', 'owned_by': 'openai'},
      {'id': 'whisper-1', 'owned_by': 'openai'},
      {'id': 'davinci-002', 'owned_by': 'openai'},
      {'id': 'gpt-3.5-turbo-instruct', 'owned_by': 'openai'},
      {'id': 'sora-2', 'owned_by': 'openai'},
    ]
  }

  filtered = inference_controls._openai_choices(payload, filter_noise=True)  # pyright: ignore[reportPrivateUsage]
  unfiltered = inference_controls._openai_choices(payload, filter_noise=False)  # pyright: ignore[reportPrivateUsage]

  assert tuple(choice.value for choice in filtered) == ('gpt-6-astra',)
  assert len(unfiltered) == 6
  # No context/pricing to show and `owned_by` is noise, so the row is id-only.
  assert filtered[0] == selection.Choice('gpt-6-astra', 'gpt-6-astra')


def test_openrouter_opens_a_live_fetched_model_picker() -> None:
  current = _state()

  action = commands.dispatch(
    '/model openai-chat-completions', _command_context(), _commands(current)
  )

  assert isinstance(action, actions.OpenSelector)
  selector = action.selector
  assert selector.id == 'openrouter-models'
  assert selector.choices == ()
  assert selector.load is inference_controls._fetch_openrouter_models  # pyright: ignore[reportPrivateUsage]
  assert selector.on_select is inference_controls._openrouter_model_selected  # pyright: ignore[reportPrivateUsage]
  # A selector with an async producer asks the CLI to fetch its choices.
  _, load_effects = reducer.reduce(current, action)
  assert effects.LoadSelectorChoices('openrouter-models') in load_effects


def test_openrouter_model_choice_switches_over_chat_completions() -> None:
  action = inference_controls._openrouter_model_selected(  # pyright: ignore[reportPrivateUsage]
    'anthropic/claude-opus-4-8'
  )

  assert action == actions.ModelResolved(
    models.ModelOption(
      'anthropic/claude-opus-4-8',
      'anthropic/claude-opus-4-8',
      api='openai-chat-completions',
    )
  )


def test_openrouter_catalog_parses_into_choices() -> None:
  payload = {
    'data': [
      {
        'id': 'openai/gpt-6-astra',
        'name': 'OpenAI: GPT-6 Astra',
        'context_length': 400000,
        'pricing': {'prompt': '0.000005'},
      },
      {'id': 'z-ai/glm-5', 'context_length': 200000},
      {'not': 'a model'},
    ]
  }

  choices = inference_controls._openrouter_choices(payload)  # pyright: ignore[reportPrivateUsage]

  assert choices == (
    selection.Choice(
      'openai/gpt-6-astra',
      'OpenAI: GPT-6 Astra',
      'openai/gpt-6-astra · 400,000 ctx · $5.00/M in',
    ),
    selection.Choice('z-ai/glm-5', 'z-ai/glm-5', 'z-ai/glm-5 · 200,000 ctx'),
  )


def test_unload_only_closes_the_owning_extensions_selector() -> None:
  selector = selection.Selector(
    'model', 'Model', (), actions.SetModel, extension_id='user/model'
  )
  current, _ = reducer.reduce(_state(), actions.OpenSelector(selector))
  unrelated, _ = reducer.reduce(
    current, actions.ExtensionUnloaded('user/clock')
  )
  assert unrelated.selector == current.selector
  unloaded, _ = reducer.reduce(current, actions.ExtensionUnloaded('user/model'))
  assert not unloaded.selector.active


def test_selector_choice_factory_returns_application_action() -> None:
  selector = selection.Selector(
    'model',
    'Switch model',
    (selection.Choice('next', 'Next', 'next'),),
    actions.SetModel,
    extension_id=inference_controls.InferenceControlsExtension.id(),
  )
  current, _ = reducer.reduce(_state(), actions.OpenSelector(selector))

  choice = current.selector.selected_choice()

  assert choice is not None
  assert selector.on_select(choice.value) == actions.SetModel('next')
  closed, emitted = reducer.reduce(
    current, actions.SelectorChoiceSelected(selector.id, choice.value)
  )
  assert not closed.selector.active
  assert emitted == (effects.SyncSelectorInput(False, ''),)


def test_logout_returns_credential_removal_action() -> None:
  assert (
    commands.dispatch('/logout', _command_context(), _commands(_state()))
    == actions.ForgetReflectionKey()
  )


def test_login_uses_the_shared_model_sign_in_flow(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _clear_credentials(tmp_path, monkeypatch)
  assert commands.dispatch(
    '/login', _command_context(), _commands(_state())
  ) == actions.ModelResolved(
    models.ModelOption(
      models.default_model('reflection'), 'reflection', api='reflection'
    )
  )


@pytest.mark.parametrize('environment', [False, True])
def test_login_reports_existing_credentials_without_switching_models(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  environment: bool,
) -> None:
  _clear_credentials(tmp_path, monkeypatch)
  if environment:
    monkeypatch.setenv('REFLECTION_API_KEY', 'synthetic-env-key')
  else:
    inference_controls.reflection.save_key('synthetic-saved-key')

  action = commands.dispatch('/login', _command_context(), _commands(_state()))

  assert isinstance(action, actions.ShowText)
  assert 'already configured' in action.text
  assert (
    'REFLECTION_API_KEY' if environment else 'saved API key'
  ) in action.text
  assert 'synthetic-env-key' not in action.text
  assert 'synthetic-saved-key' not in action.text


def test_saved_anthropic_key_enables_catalog(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _clear_credentials(tmp_path, monkeypatch)
  assert not api_keys.has_anthropic_key()
  api_keys.save_key(api_keys.anthropic_key_path(), 'synthetic-saved-key')
  selector = inference_controls.model_selector('current')
  action = selector.on_select('anthropic-messages')
  assert isinstance(action, actions.OpenSelector)
  assert action.selector.id == 'anthropic-models'
