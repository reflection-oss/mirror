"""Config-driven model definitions for the interactive model picker."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
from typing import Any, cast


@dataclasses.dataclass(frozen=True)
class ModelOption:
  """One selectable model in the picker."""

  model_id: str
  label: str
  description: str = ''
  # Wire protocol; empty means "inherit the session's api".
  api: str = ''
  # Endpoint URL; empty means "use the adapter's default".
  base_url: str = ''


# The wire protocols the harness can speak, paired with picker labels. The ids
# match the Rust `ModelApi` names and the `--api` flag choices.
WIRE_PROTOCOLS: tuple[tuple[str, str], ...] = (
  ('reflection', 'Reflection API'),
  ('openai-codex-responses', 'ChatGPT plan billing (Codex login)'),
  ('openai-responses', 'OpenAI Responses API'),
  ('openai-chat-completions', 'OpenRouter'),
  ('anthropic-messages', 'Anthropic Messages API'),
)


def default_model(api: str) -> str:
  """The model id a wire protocol starts from when none is named."""
  if api == 'reflection':
    return 'Beam-501B-A23B'
  if api == 'anthropic-messages':
    return 'claude-opus-4-8'
  if api in ('openai-responses', 'openai-codex-responses'):
    return 'gpt-6-astra'
  return 'openai/gpt-6-astra'


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
  raw: Any = json.loads(path.read_text())
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
  if model_id == default_model('reflection'):
    return ModelOption(model_id, 'Reflection', api='reflection')
  return ModelOption(model_id, model_id)
