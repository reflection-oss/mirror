"""Reflection API key onboarding for the local CLI."""

from __future__ import annotations

import enum
import os
import pathlib

from mirror.cli import api_keys

KEY_URL = 'https://platform.reflection.ai/api-keys'
BASE_URL = 'https://api.reflection.ai/openai/v1'


class SignInResult(enum.Enum):
  """The user's next step after a masked credential prompt."""

  CONNECTED = enum.auto()
  CHOOSE_MODEL = enum.auto()
  CANCELLED = enum.auto()


def key_path() -> pathlib.Path:
  """Return the local Reflection credential file path."""
  return pathlib.Path.home() / '.mirror' / 'reflection_api_key'


def has_key() -> bool:
  """Whether an environment override or saved credential is available."""
  return has_environment_key() or key_path().is_file()


def save_key(key: str) -> None:
  """Save a Reflection key using the shared private file writer."""
  api_keys.save_key(key_path(), key)


def has_environment_key() -> bool:
  """Whether the environment overrides saved Reflection credentials."""
  return bool(os.environ.get('REFLECTION_API_KEY', '').strip())


def remove_key() -> bool:
  """Remove the saved credential without reading it or changing the environment."""
  try:
    key_path().unlink()
  except FileNotFoundError:
    return False
  except OSError:
    raise OSError('Could not remove saved Reflection API key') from None
  return True
