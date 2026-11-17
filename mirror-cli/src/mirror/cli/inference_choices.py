"""The models and reasoning efforts Mirror offers, with their catalogs."""

from __future__ import annotations

import os
from typing import cast

import httpx2

from mirror.cli import api_keys, models, reflection, selection

_OPENROUTER_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
_ANTHROPIC_DEFAULT_BASE_URL = 'https://api.anthropic.com'
_OPENAI_DEFAULT_BASE_URL = 'https://api.openai.com/v1'
_PROVIDER_BASE_URLS = {
  'openai-responses': _OPENAI_DEFAULT_BASE_URL,
  'openai-chat-completions': _OPENROUTER_DEFAULT_BASE_URL,
  'anthropic-messages': _ANTHROPIC_DEFAULT_BASE_URL,
}

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

REASONING_EFFORTS = (
  'none',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
)

# Where each protocol reads its credentials, for the picker's readiness hint.
PROTOCOL_KEY_ENV = {
  'openai-responses': 'OPENAI_API_KEY',
  'openai-chat-completions': 'OPENROUTER_API_KEY',
  'anthropic-messages': 'ANTHROPIC_API_KEY',
  'gemini': 'GEMINI_API_KEY',
  'vercel': 'AI_GATEWAY_API_KEY',
}

GEMINI_CHOICES = (
  selection.Choice(
    models.GEMINI_MODEL,
    'Gemini 4 Argon',
    'gemini-4-argon · 1,048,576 shared ctx · 65,536 max output · paid API',
  ),
)


def effort_label(effort: str) -> str:
  """How a reasoning effort is shown; `none` leaves it to the provider."""
  return 'provider default' if effort == 'none' else effort


def codex_logged_in() -> bool:
  """Whether a Codex login exists for ChatGPT plan billing."""
  return os.path.exists(os.path.expanduser('~/.codex/auth.json'))


def ready(api: str) -> bool:
  """Whether a wire protocol's credentials are in place, without signing in."""
  if api == 'reflection':
    return reflection.has_key()
  if api == 'openai-codex-responses':
    return codex_logged_in()
  # A self-hosted endpoint may need no key, as `models.validate_connection`
  # allows; the provider's own endpoint still does.
  endpoint = models.base_env(api)
  url = os.environ.get(endpoint, '').rstrip('/') if endpoint else ''
  if url and url != _PROVIDER_BASE_URLS.get(api):
    return True
  if api == 'anthropic-messages':
    return api_keys.has_anthropic_key()
  env = PROTOCOL_KEY_ENV.get(api)
  return env is not None and bool(os.environ.get(env, '').strip())


async def protocol_models(api: str) -> tuple[selection.Choice, ...]:
  """The models a ready protocol's picker offers, or its default model."""
  if api == 'reflection':
    return await fetch_reflection_models()
  if api == 'openai-chat-completions':
    return await fetch_openrouter_models()
  if api == 'anthropic-messages':
    return await fetch_anthropic_models()
  if api == 'openai-responses':
    return await fetch_openai_models()
  if api == 'vercel':
    return await fetch_vercel_models()
  if api == 'openai-codex-responses':
    return codex_choices()
  if api == 'gemini':
    return GEMINI_CHOICES
  return default_choices(api)


def default_choices(api: str) -> tuple[selection.Choice, ...]:
  """A protocol's default model, offered where it has no list."""
  model_id = models.default_model(api)
  return (selection.Choice(model_id, model_id, model_id),)


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


def _vercel_choices(payload: object) -> tuple[selection.Choice, ...]:
  """Turn a Vercel AI Gateway `/models` response into chat picker choices."""
  choices: list[selection.Choice] = []
  for entry in _catalog_entries(payload):
    model_id = entry.get('id')
    if not isinstance(model_id, str) or not model_id:
      continue
    # The catalog also lists embedding, image, and other non-chat models.
    if entry.get('type', 'language') != 'language':
      continue
    parts = [model_id]
    context = entry.get('context_window')
    if isinstance(context, int) and context > 0:
      parts.append(f'{context:,} ctx')
    pricing = entry.get('pricing')
    if isinstance(pricing, dict):
      prompt = cast(dict[str, object], pricing).get('input')
      if isinstance(prompt, str):
        try:
          parts.append(f'${float(prompt) * 1_000_000:.2f}/M in')
        except ValueError:
          pass
    name = entry.get('name')
    label = name if isinstance(name, str) and name else model_id
    choices.append(selection.Choice(model_id, label, ' · '.join(parts)))
  return tuple(choices)


def codex_choices() -> tuple[selection.Choice, ...]:
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


async def fetch_reflection_models() -> tuple[selection.Choice, ...]:
  """Fetch the models available to the signed-in Reflection account."""
  try:
    key = os.environ.get('REFLECTION_API_KEY', '').strip()
    if not key:
      key = reflection.key_path().read_text().strip()
    if not key or any(char.isspace() for char in key):
      raise ValueError('Missing or invalid API key')
    payload = await _fetch_json(
      f'{reflection.BASE_URL}/models',
      headers={'Authorization': f'Bearer {key}'},
    )
  except (OSError, ValueError, httpx2.HTTPError):
    # Transport errors and malformed provider responses can contain credentials.
    raise ValueError('Could not load Reflection models') from None
  return _openai_choices(payload, filter_noise=False)


async def fetch_openrouter_models() -> tuple[selection.Choice, ...]:
  """Fetch the live OpenRouter model catalog for the secondary selector."""
  base = os.environ.get('OPENROUTER_BASE_URL') or _OPENROUTER_DEFAULT_BASE_URL
  payload = await _fetch_json(f'{base.rstrip("/")}/models')
  return _openrouter_choices(payload)


async def fetch_anthropic_models() -> tuple[selection.Choice, ...]:
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


async def fetch_openai_models() -> tuple[selection.Choice, ...]:
  """Fetch the OpenAI-compatible model catalog for the secondary selector."""
  base = os.environ.get('OPENAI_BASE_URL') or _OPENAI_DEFAULT_BASE_URL
  key = os.environ.get('OPENAI_API_KEY') or 'EMPTY'
  payload = await _fetch_json(
    f'{base.rstrip("/")}/models', headers={'Authorization': f'Bearer {key}'}
  )
  # Only api.openai.com serves the noisy non-chat catalog worth filtering.
  return _openai_choices(payload, filter_noise='openai.com' in base)


async def fetch_vercel_models() -> tuple[selection.Choice, ...]:
  """Fetch the Vercel AI Gateway model catalog for the secondary selector."""
  key = os.environ.get('AI_GATEWAY_API_KEY', '').strip()
  payload = await _fetch_json(
    f'{models.VERCEL_BASE_URL}/models',
    headers={'Authorization': f'Bearer {key}'},
  )
  return _vercel_choices(payload)
