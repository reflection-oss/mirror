"""Config-driven model definitions for the interactive model picker."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
from typing import Any, cast

import httpx2

from mirror.cli import reflection


@dataclasses.dataclass(frozen=True)
class ModelOption:
  """One selectable model in the picker."""

  model_id: str
  label: str
  description: str = ''
  # Wire protocol; empty means "inherit the session's api".
  api: str = ''
  # Endpoint or `managed`; empty with an explicit api uses its default.
  base_url: str = ''


# The shared llama.cpp server the daemon manages, as a saved selection.
MANAGED = 'managed'


# The wire protocols the harness can speak, paired with picker labels. The ids
# match the Rust `ModelApi` names and the `--api` flag choices.
WIRE_PROTOCOLS: tuple[tuple[str, str], ...] = (
  ('reflection', 'Reflection API'),
  ('openai-codex-responses', 'ChatGPT plan billing (Codex login)'),
  ('openai-responses', 'OpenAI Responses API'),
  ('openai-chat-completions', 'OpenRouter'),
  ('anthropic-messages', 'Anthropic Messages API'),
  ('gemini', 'Google Gemini Developer API'),
  ('vercel', 'Vercel AI Gateway'),
  ('llama_cpp', 'llama.cpp (on this machine)'),
)

GEMINI_MODEL = 'gemini-4-argon'
GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai'
GEMINI_REASONING_EFFORTS = ('none', 'low', 'medium', 'high')

VERCEL_MODEL = 'reflection/beam'
VERCEL_BASE_URL = 'https://ai-gateway.vercel.sh/v1'


# A Reflection API alias of the default Beam model. The API serves it under this
# name too, and a saved last-model selection may name it.
_UNVERSIONED_BEAM = 'Beam-501B-A23B'


def default_model(api: str) -> str:
  """The model id a wire protocol starts from when none is named."""
  if api == 'reflection':
    return 'Beam-1.0-501B-A23B'
  if api == 'gemini':
    return GEMINI_MODEL
  if api == 'vercel':
    return VERCEL_MODEL
  if api == 'anthropic-messages':
    return 'claude-opus-5-5'
  if api in ('openai-responses', 'openai-codex-responses'):
    return 'gpt-6-astra'
  return 'openai/gpt-6-astra'


def supports_fast_mode(api: str) -> bool:
  """Whether a protocol has a speed control, subject to model/account access."""
  return api in (
    'openai-responses',
    'openai-codex-responses',
    'anthropic-messages',
  )


def supports_images(option: ModelOption) -> bool:
  """Whether a selection reads images.

  Reflection's models read only text, on the Reflection API and under their
  `reflection/` name on a gateway. Any other model is assumed to read images,
  and its provider rejects them when it cannot. The agent applies the same
  rule when it replays images to a model.
  """
  return option.api != 'reflection' and not option.model_id.startswith(
    'reflection/'
  )


def inherited_reasoning_effort(api: str, effort: str) -> str:
  """Keep inherited effort when supported, otherwise use the provider default."""
  if api == 'gemini' and effort not in GEMINI_REASONING_EFFORTS:
    return 'none'
  return effort


def key_env(api: str) -> str:
  """The environment variable holding the API key for a wire protocol."""
  if api == 'reflection':
    return 'REFLECTION_API_KEY'
  if api == 'openai-responses':
    return 'OPENAI_API_KEY'
  if api == 'gemini':
    return 'GEMINI_API_KEY'
  if api == 'vercel':
    return 'AI_GATEWAY_API_KEY'
  if api == 'anthropic-messages':
    return 'ANTHROPIC_API_KEY'
  return 'OPENROUTER_API_KEY'


def base_env(api: str) -> str | None:
  """The environment variable holding the base URL for a wire protocol."""
  if api in ('reflection', 'gemini', 'vercel'):
    return None
  if api == 'llama_cpp':
    return 'MIRROR_LLAMA_CPP_BASE_URL'
  if api == 'openai-responses':
    return 'OPENAI_BASE_URL'
  if api == 'anthropic-messages':
    return 'ANTHROPIC_BASE_URL'
  return 'OPENROUTER_BASE_URL'


def local_base_url(value: str = '') -> str:
  """Validate explicit CLI endpoints; Rust also validates at every dispatch."""
  try:
    url = httpx2.URL(
      value
      or os.environ.get('MIRROR_LLAMA_CPP_BASE_URL')
      or 'http://127.0.0.1:8080/v1'
    )
  except httpx2.InvalidURL:
    raise ValueError('Invalid llama.cpp URL.') from None
  if (
    url.scheme != 'http'
    or url.host not in ('127.0.0.1', '::1', 'localhost')
    or url.userinfo
    or url.query
    or url.fragment
    or url.path.rstrip('/') != '/v1'
  ):
    raise ValueError(
      'llama.cpp requires a loopback HTTP /v1 endpoint without credentials, query or fragment.'
    )
  return str(
    url.copy_with(host='127.0.0.1' if url.host == 'localhost' else url.host)
  ).rstrip('/')


def validate_connection(api: str, base_url: str) -> None:
  """Check local provider prerequisites without probing model availability."""
  if api == 'llama_cpp':
    if base_url != MANAGED:
      local_base_url(base_url)
    return
  if api == 'reflection':
    if base_url and base_url.rstrip('/') != reflection.BASE_URL:
      raise ValueError(
        '--api reflection uses the Reflection endpoint; use '
        '--api openai-chat-completions for custom endpoints'
      )
    return
  if api == 'gemini':
    if base_url and base_url.rstrip('/') != GEMINI_BASE_URL:
      raise ValueError(
        'gemini uses the Gemini Developer API, not Vertex AI or custom '
        'endpoints. Use openai-chat-completions for a custom endpoint.'
      )
    if not os.environ.get('GEMINI_API_KEY', '').strip():
      raise ValueError(
        'Set GEMINI_API_KEY from Google AI Studio and restart Mirror. '
        'Gemini 4 Argon requires paid API access; a Google AI plan is not an API key.'
      )
    return
  if api == 'vercel':
    if base_url and base_url.rstrip('/') != VERCEL_BASE_URL:
      raise ValueError(
        'vercel uses the Vercel AI Gateway endpoint, not custom endpoints. '
        'Use openai-chat-completions for a custom endpoint.'
      )
    if not os.environ.get('AI_GATEWAY_API_KEY', '').strip():
      raise ValueError(
        'Set AI_GATEWAY_API_KEY to a Vercel AI Gateway API key and restart Mirror.'
      )
    return
  if api in ('', 'openai-codex-responses', 'anthropic-messages'):
    return
  endpoint_env = base_env(api)
  if base_url or (endpoint_env and os.environ.get(endpoint_env)):
    return
  if not os.environ.get(key_env(api), '').strip():
    raise ValueError(
      f'No API key found. Export {key_env(api)} or use --base-url '
      'for a keyless self-hosted endpoint.'
    )


def validate_selection(
  option: ModelOption,
  reasoning_effort: str,
  *,
  max_tokens: int | None = None,
  compaction_api: str | None = None,
) -> None:
  """Check provider-specific constraints before committing a selection.

  `compaction_api` is the API a separate compaction model runs on, if there is one.
  """
  if option.api == 'vercel':
    validate_connection(option.api, option.base_url)
    return
  if option.api == 'llama_cpp':
    validate_connection(option.api, option.base_url)
    if compaction_api == 'llama_cpp':
      raise ValueError(
        'A separate compaction model cannot run on llama.cpp; '
        'choose one on another provider.'
      )
    return
  if option.api != 'gemini':
    return
  if option.model_id != GEMINI_MODEL:
    raise ValueError(
      'gemini supports gemini-4-argon only. Select /model gemini.'
    )
  if reasoning_effort not in GEMINI_REASONING_EFFORTS:
    raise ValueError(
      'Gemini 4 Argon supports provider default, low, medium, or high reasoning. '
      'Restart with --reasoning-effort medium, or use /effort medium before switching models.'
    )
  if max_tokens is not None and not 0 < max_tokens <= 65_536:
    raise ValueError(
      'Gemini requires --max-tokens between 1 and 65536. '
      'Long decode is unavailable on the OpenAI-compatible API.'
    )
  validate_connection(option.api, option.base_url)


def _config_path() -> pathlib.Path:
  """Where model definitions are loaded from."""
  env = os.environ.get('MIRROR_MODELS_CONFIG')
  if env:
    return pathlib.Path(env)
  return pathlib.Path.home() / '.mirror' / 'models.json'


def _last_model_path() -> pathlib.Path:
  return _config_path().parent / 'last_model'


def save_last_model(model_id: str) -> None:
  """Persist the model id so the next session defaults to it."""
  path = _last_model_path()
  # A read-only or shared config location must not break the session.
  with contextlib.suppress(OSError):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model_id)


def load_last_model() -> str | None:
  """Return the most recently used model id, if any."""
  # The record is a convenience, so an unreadable one must not end the session.
  try:
    model_id = _last_model_path().read_text().strip()
  except OSError:
    return None
  return model_id or None


def save_last_selection(option: ModelOption) -> None:
  """Remember a complete selection as the default for fresh workspaces."""
  path = _config_path().parent / 'last_model.json'
  with contextlib.suppress(OSError):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
      json.dumps({
        'model': option.model_id,
        'api': option.api,
        'base_url': option.base_url,
      })
    )


def load_last_selection() -> ModelOption | None:
  """Load the last complete selection, ignoring absent or invalid records."""
  try:
    raw: object = json.loads(
      (_config_path().parent / 'last_model.json').read_text()
    )
  except (OSError, ValueError):
    return None
  if not isinstance(raw, dict):
    return None
  value = cast(dict[str, object], raw)
  model, api, base_url = (
    value.get('model'),
    value.get('api'),
    value.get('base_url'),
  )
  if (
    not isinstance(model, str)
    or not model
    or not isinstance(api, str)
    or not isinstance(base_url, str)
  ):
    return None
  return ModelOption(model, model, api=api, base_url=base_url)


def load_models() -> tuple[ModelOption, ...]:
  """Load model definitions from the config file.

  The config file is a JSON list of objects with `id`, `label`, `description`
  (optional), `api` (optional), and `base_url` (optional). If the file does not
  exist, an empty tuple is returned.
  """
  path = _config_path()
  if not path.exists():
    return ()
  try:
    raw: Any = json.loads(path.read_text())
  except json.JSONDecodeError as error:
    raise ValueError(f'{path}: invalid JSON: {error}') from error
  if not isinstance(raw, list):
    raise ValueError(f'{path}: expected a JSON list, got {type(raw).__name__}')
  entries = cast(list[Any], raw)
  options: list[ModelOption] = []
  for i, entry in enumerate(entries):
    if not isinstance(entry, dict):
      raise ValueError(
        f'{path}: entry {i} is a {type(entry).__name__}, expected an object'
      )
    entry = cast(dict[str, Any], entry)
    model_id = entry.get('id')
    label = entry.get('label')
    if not isinstance(model_id, str) or not model_id:
      raise ValueError(f'{path}: entry {i} missing string "id"')
    if not isinstance(label, str) or not label:
      raise ValueError(f'{path}: entry {i} missing string "label"')
    options.append(
      ModelOption(
        model_id=model_id,
        label=label,
        description=str(entry.get('description', '')),
        api=str(entry.get('api', '')),
        base_url=str(entry.get('base_url', '')),
      )
    )
  return tuple(options)


def find_model(model_id: str) -> ModelOption:
  """Return the matching option, or a bare option if not configured."""
  for option in load_models():
    if option.model_id == model_id:
      return option
  if model_id == GEMINI_MODEL:
    return ModelOption(model_id, 'Gemini 4 Argon', api='gemini')
  if is_reflection_default(model_id):
    return ModelOption(model_id, 'Reflection', api='reflection')
  return ModelOption(model_id, model_id)


def is_reflection_default(model_id: str) -> bool:
  """Whether `model_id` names the Reflection API's default model."""
  return model_id in (default_model('reflection'), _UNVERSIONED_BEAM)
