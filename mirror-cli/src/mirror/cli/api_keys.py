"""Private local API-key files shared by provider onboarding."""

from __future__ import annotations

import os
import pathlib
import tempfile

ANTHROPIC_KEY_URL = 'https://platform.claude.com/settings/keys'


def anthropic_key_path() -> pathlib.Path:
  home = pathlib.Path(os.environ.get('HOME', ''))
  if not home.is_absolute():
    raise OSError('Set HOME to an absolute path to save an Anthropic API key')
  return home / '.mirror' / 'anthropic_api_key'


def has_anthropic_key() -> bool:
  try:
    return bool(anthropic_key())
  except OSError:
    return False


def anthropic_key() -> str:
  """Use the same environment-first precedence as the worker."""
  if key := os.environ.get('ANTHROPIC_API_KEY', '').strip():
    return key
  base_url = os.environ.get('ANTHROPIC_BASE_URL') or 'https://api.anthropic.com'
  if base_url.rstrip('/') != 'https://api.anthropic.com':
    return ''
  try:
    return anthropic_key_path().read_text().strip()
  except FileNotFoundError:
    return ''
  except (OSError, UnicodeError):
    raise OSError('Could not read saved Anthropic API key') from None


def save_key(path: pathlib.Path, key: str) -> None:
  """Atomically save a user-only key file without changing the environment."""
  key = key.strip()
  if not key or any(char.isspace() for char in key):
    raise ValueError('API key must be nonempty and contain no whitespace')
  temporary: pathlib.Path | None = None
  try:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
      mode='w', dir=path.parent, delete=False
    ) as stream:
      temporary = pathlib.Path(stream.name)
      stream.write(key)
    temporary.replace(path)
  except OSError:
    raise OSError('Could not save API key') from None
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)
