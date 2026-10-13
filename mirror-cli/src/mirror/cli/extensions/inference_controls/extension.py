"""Model and reasoning commands for the Mirror CLI."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from typing import cast, override

import httpx2

from mirror.cli import (
  actions,
  api_keys,
  commands,
  extension,
  models,
  reflection,
  selection,
)

_OPENROUTER_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
_ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com'
_OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1'

# api.openai.com serves a large non-chat catalog; these substrings mark models
# no chat/responses session would pick, and are hidden only for that host.
_OPENAI_NON_CHAT = (
  'embed',
  'whisper',
  'tts',
  'audio',
  'transcribe',
  'dall-e',
  'image',
  'moderation',
  'realtime',
  'search',
  'instruct',
  'babbage',
  'davinci',
  'sora',
)

# Offline subset of Codex rust-v0.162.1's bundled catalog, not account discovery.
# https://github.com/openai/codex/blob/rust-v0.162.1/codex-rs/models-manager/models.json
_CODEX_MODELS = (
  ('gpt-6-astra', 'GPT-6-Astra'),
  ('gpt-6.1-sol', 'GPT-6.1-Sol'),
  ('gpt-5.6-sol', 'GPT-5.6-Sol'),
)

_REASONING_EFFORTS = (
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
)

_PROTOCOL_IDS = frozenset(api for api, _ in models.WIRE_PROTOCOLS)

# Where each protocol reads its credentials, for the picker's readiness hint.
_PROTOCOL_KEY_ENV = {
  'openai-responses': 'OPENAI_API_KEY',
  'openai-chat-completions': 'OPENROUTER_API_KEY',
  'anthropic-messages': 'ANTHROPIC_API_KEY',
}


def _credential_hint(api: str) -> str:
  """A short readiness note for a protocol, shown beside its picker row."""
  if api == 'anthropic-messages':
    return 'key set' if api_keys.has_anthropic_key() else 'select to sign in'
  if api == 'reflection':
    return 'key set' if reflection.has_key() else 'select to sign in'
  if api == 'openai-codex-responses':
    path = os.path.expanduser('~/.codex/auth.json')
    return 'logged in' if os.path.exists(path) else 'run codex login'
  env = _PROTOCOL_KEY_ENV.get(api)
  if env is None:
    return ''
  return 'key set' if os.environ.get(env) else f'set ${env}'


def _protocol_choices() -> tuple[selection.Choice, ...]:
  """The wire protocols, always offered even without a models config."""
  choices: list[selection.Choice] = []
  for api, label in models.WIRE_PROTOCOLS:
    hint = _credential_hint(api)
    detail = f'{api} · {hint}' if hint else api
    choices.append(selection.Choice(api, label, detail))
  return tuple(choices)


def _catalog_entries(payload: object) -> list[dict[str, object]]:
  """The object entries under a `{"data": [...]}` catalog response."""
  if not isinstance(payload, dict):
    return []
  data = cast(dict[str, object], payload).get('data')
  if not isinstance(data, list):
    return []
  entries: list[dict[str, object]] = []
  for entry in cast(list[object], data):
    if isinstance(entry, dict):
      entries.append(cast(dict[str, object], entry))
  return entries


def _openrouter_detail(model_id: str, entry: dict[str, object]) -> str:
  """A one-line summary (id, context window, input price) for a picker row."""
  parts = [model_id]
  context = entry.get('context_length')
  if isinstance(context, int) and context > 0:
    parts.append(f'{context:,} ctx')
  pricing = entry.get('pricing')
  if isinstance(pricing, dict):
    prompt = cast(dict[str, object], pricing).get('prompt')
    if isinstance(prompt, str):
      try:
        per_million = float(prompt) * 1_000_000
      except ValueError:
        per_million = None
      if per_million is not None:
        parts.append(f'${per_million:.2f}/M in')
  return ' · '.join(parts)


def _openrouter_choices(payload: object) -> tuple[selection.Choice, ...]:
  """Turn an OpenRouter `/models` response into picker choices."""
  choices: list[selection.Choice] = []
  for entry in _catalog_entries(payload):
    model_id = entry.get('id')
    if not isinstance(model_id, str) or not model_id:
      continue
    name = entry.get('name')
    label = name if isinstance(name, str) and name else model_id
    choices.append(
      selection.Choice(model_id, label, _openrouter_detail(model_id, entry))
    )
  return tuple(choices)


def _anthropic_choices(payload: object) -> tuple[selection.Choice, ...]:
  """Turn an Anthropic `/v1/models` response into picker choices."""
  choices: list[selection.Choice] = []
  for entry in _catalog_entries(payload):
    model_id = entry.get('id')
    if not isinstance(model_id, str) or not model_id:
      continue
    name = entry.get('display_name')
    label = name if isinstance(name, str) and name else model_id
    choices.append(selection.Choice(model_id, label, model_id))
  return tuple(choices)


def _openai_choices(
  payload: object, *, filter_noise: bool
) -> tuple[selection.Choice, ...]:
  """Turn an OpenAI-compatible `/models` response into picker choices."""
  choices: list[selection.Choice] = []
  for entry in _catalog_entries(payload):
    model_id = entry.get('id')
    if not isinstance(model_id, str) or not model_id:
      continue
    if filter_noise and any(token in model_id for token in _OPENAI_NON_CHAT):
      continue
    # The catalog carries no context/pricing, and `owned_by` ("openai",
    # "system", …) is noise, so the id alone is the whole row.
    choices.append(selection.Choice(model_id, model_id))
  return tuple(choices)


def _codex_choices() -> tuple[selection.Choice, ...]:
  """Offline choices; the subscription backend still decides account access."""
  return tuple(
    selection.Choice(model_id, label, model_id)
    for model_id, label in _CODEX_MODELS
  )


async def _fetch_json(
  url: str,
  *,
  headers: dict[str, str] | None = None,
  params: dict[str, str | int] | None = None,
) -> object:
  async with httpx2.AsyncClient(timeout=15.0) as client:
    response = await client.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


async def _fetch_openrouter_models() -> tuple[selection.Choice, ...]:
  """Fetch the live OpenRouter model catalog for the secondary selector."""
  base = os.environ.get('OPENROUTER_BASE_URL') or _OPENROUTER_DEFAULT_BASE_URL
  payload = await _fetch_json(f'{base.rstrip("/")}/models')
  return _openrouter_choices(payload)


async def _fetch_anthropic_models() -> tuple[selection.Choice, ...]:
  """Fetch the live Anthropic model catalog for the secondary selector."""
  base = os.environ.get('ANTHROPIC_BASE_URL') or _ANTHROPIC_DEFAULT_BASE_URL
  payload = await _fetch_json(
    f'{base.rstrip("/")}/v1/models',
    headers={
      'x-api-key': api_keys.anthropic_key(),
      'anthropic-version': '2023-06-01',
    },
    params={'limit': 1000},
  )
  return _anthropic_choices(payload)


async def _fetch_openai_models() -> tuple[selection.Choice, ...]:
  """Fetch the OpenAI-compatible model catalog for the secondary selector."""
  base = os.environ.get('OPENAI_BASE_URL') or _OPENAI_DEFAULT_BASE_URL
  key = os.environ.get('OPENAI_API_KEY') or 'EMPTY'
  payload = await _fetch_json(
    f'{base.rstrip("/")}/models', headers={'Authorization': f'Bearer {key}'}
  )
  # Only api.openai.com serves the noisy non-chat catalog worth filtering.
  return _openai_choices(payload, filter_noise='openai.com' in base)


def _model_selected(model_id: str, api: str) -> actions.Action:
  """Switch to one specific model over the given wire protocol."""
  return actions.ModelResolved(models.ModelOption(model_id, model_id, api=api))


def _openrouter_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'openai-chat-completions')


def _anthropic_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'anthropic-messages')


def _openai_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'openai-responses')


def _codex_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'openai-codex-responses')


def _default_model(api: str) -> actions.Action:
  """Switch to a protocol's default model without a secondary picker."""
  return actions.ModelResolved(
    models.ModelOption(models.default_model(api), api, api=api)
  )


def _model_picker(
  selector_id: str,
  title: str,
  on_select: Callable[[str], actions.Action],
  *,
  empty: str,
  load: Callable[[], Awaitable[tuple[selection.Choice, ...]]] | None = None,
  choices: tuple[selection.Choice, ...] = (),
) -> actions.Action:
  return actions.OpenSelector(
    selection.Selector(
      selector_id,
      title,
      choices,
      on_select,
      extension_id=InferenceControlsExtension.id(),
      empty=empty,
      load=load,
    )
  )


def _select_inference(value: str) -> actions.Action:
  """Route a picker choice to a model picker, protocol default, or model id."""
  # Each protocol offers many models, so drill into a secondary picker. When a
  # protocol's credentials are absent the fetch cannot succeed, so fall back to
  # its default model rather than opening a picker that would only error.
  if value in ('openai-chat-completions', 'openrouter'):
    return _model_picker(
      'openrouter-models',
      'OpenRouter model',
      _openrouter_model_selected,
      empty='Fetching OpenRouter models…',
      load=_fetch_openrouter_models,
    )
  if value == 'anthropic-messages':
    if not api_keys.has_anthropic_key():
      return _default_model(value)
    return _model_picker(
      'anthropic-models',
      'Anthropic model',
      _anthropic_model_selected,
      empty='Fetching Anthropic models…',
      load=_fetch_anthropic_models,
    )
  if value == 'openai-responses':
    if not (
      os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_BASE_URL')
    ):
      return _default_model(value)
    return _model_picker(
      'openai-models',
      'OpenAI model',
      _openai_model_selected,
      empty='Fetching OpenAI models…',
      load=_fetch_openai_models,
    )
  if value == 'openai-codex-responses':
    return _model_picker(
      'codex-models',
      'ChatGPT model',
      _codex_model_selected,
      empty='No models available.',
      choices=_codex_choices(),
    )
  if value in _PROTOCOL_IDS:
    return _default_model(value)
  return actions.SetModel(value)


def model_command(
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  """Select a host-owned model or open its picker."""
  if arguments:
    if len(arguments.split()) > 1:
      return actions.ShowText(
        'Model id must not contain spaces.', style='ui.warning'
      )
    return _select_inference(arguments)
  return actions.OpenSelector(model_selector(context.model))


def model_selector(current_model: str) -> selection.Selector[actions.Action]:
  """Build the shared model picker for commands and sign-in."""
  configured = tuple(
    selection.Choice(option.model_id, option.label, option.model_id)
    for option in models.load_models()
  )
  return selection.Selector(
    'model',
    'Switch model or protocol',
    _protocol_choices() + configured,
    _select_inference,
    current=current_model,
    extension_id=InferenceControlsExtension.id(),
    empty='No models available.',
  )


def _login(_: commands.CommandContext, arguments: str) -> actions.Action:
  del arguments
  if reflection.has_key():
    source = (
      'REFLECTION_API_KEY'
      if reflection.has_environment_key()
      else 'a saved API key'
    )
    return actions.ShowText(
      f'Reflection is already configured with {source}. '
      'Use /model reflection to switch to it, or /logout to remove the saved key.'
    )
  return _default_model('reflection')


def _logout(_: commands.CommandContext, arguments: str) -> actions.Action:
  del arguments
  return actions.ForgetReflectionKey()


def _set_reasoning(
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  effort = arguments.lower()
  if not effort:
    return actions.OpenSelector(
      selection.Selector(
        'reasoning',
        'Reasoning effort',
        tuple(
          selection.Choice(
            value,
            'provider default' if value == 'none' else value,
          )
          for value in _REASONING_EFFORTS
        ),
        actions.SetReasoning,
        current=context.reasoning_effort,
        extension_id=InferenceControlsExtension.id(),
        selected=context.reasoning_effort,
      )
    )
  if effort not in _REASONING_EFFORTS:
    return actions.ShowText(
      f'Unknown reasoning effort: {arguments}. Choices: '
      f'{", ".join(_REASONING_EFFORTS)}',
      style='ui.warning',
    )
  return actions.SetReasoning(effort)


class InferenceControlsExtension(extension.Extension[None, None, None]):
  """Contribute the core inference-control commands."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/inference-controls'

  @override
  def tips(self) -> tuple[str, ...]:
    return (
      '/model picks a different model for the next request',
      '/reasoning adjusts how much the model reasons',
    )

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'login',
        'Check Reflection credentials or sign in',
        _login,
      ),
      commands.CommandDef(
        'logout',
        'Remove the saved Reflection API key (not environment credentials)',
        _logout,
      ),
      commands.CommandDef(
        'model',
        'Switch the model or wire protocol',
        model_command,
        arguments=commands.OptionalArguments('[model|protocol]'),
        available_when_busy=True,
      ),
      commands.CommandDef(
        'reasoning',
        'Set reasoning effort',
        _set_reasoning,
        arguments=commands.OptionalArguments('[effort]'),
        available_when_busy=True,
      ),
    )
