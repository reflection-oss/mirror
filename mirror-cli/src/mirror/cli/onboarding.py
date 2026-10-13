"""First-use notices and startup-directory trust for the interactive CLI."""

from __future__ import annotations

import pathlib
import sys
import uuid
from collections.abc import Callable
from typing import cast

from rich import console, text

from mirror import config

USAGE_WARNING = (
  'You control how much autonomy you give Mirror. You may allow Mirror access '
  'to your files or to access tools on your behalf. Mirror acts on your '
  'instructions and on your behalf. You are responsible for the actions Mirror '
  'takes, for the access you provide, and for complying with the terms of any '
  'third-party service you connect. Mirror can make mistakes, including taking '
  'actions you did not intend or that cannot be undone. Review Mirror’s actions, '
  'the files and tools it accesses and the code and commands that it runs.'
)

FOLDER_WARNING = (
  'You are providing Mirror access to the contents of this directory. By '
  'granting access you are allowing Mirror to take actions, access files and '
  'utilize tools including accessing code and commands. Files in this directory '
  'may contain content or instructions that could influence what Mirror does. '
  'Only grant access if you know where these files came from and trust them.'
)


def _confirm(message: str) -> bool:
  console.Console().print(message, markup=False, highlight=False)
  if not sys.stdin.isatty():
    raise SystemExit(
      'mirror: open Mirror in a terminal to review and accept this notice.'
    )
  try:
    return input(
      '\nContinue and trust this folder? [y/N] '
    ).strip().lower() in {'y', 'yes'}
  except (EOFError, KeyboardInterrupt):
    return False


def is_trusted_directory(
  directory: pathlib.Path, *, path: pathlib.Path | None = None
) -> bool:
  """Check existing consent without granting it or writing configuration."""
  settings = config.Config(
    path or pathlib.Path.home() / '.mirror' / 'config.json'
  )
  return str(directory.resolve()) in _trusted_directories(settings.read())


def ensure_consent(
  directory: pathlib.Path,
  *,
  path: pathlib.Path | None = None,
  confirm: Callable[[str], bool] = _confirm,
) -> None:
  """Require explicit acknowledgment once per user and once per resolved directory."""
  settings = config.Config(
    path or pathlib.Path.home() / '.mirror' / 'config.json'
  )
  directory = directory.resolve()
  values = settings.read()
  notices: list[str] = []
  if values.get('usage_warning_acknowledged') is not True:
    notices.append(f'Using Mirror\n\n{USAGE_WARNING}')
  if str(directory) not in _trusted_directories(values):
    notices.append(f'Trust directory {str(directory)!r}?\n\n{FOLDER_WARNING}')
  if not notices:
    return
  if not confirm('\n\n'.join(notices)):
    raise SystemExit('mirror: canceled; no agent was started.')

  def trust(values: dict[str, object]) -> None:
    values['usage_warning_acknowledged'] = True
    directories = _trusted_directories(values)
    if str(directory) not in directories:
      directories.append(str(directory))
    values['trusted_directories'] = directories

  settings.update(trust)


def ensure_install_id(*, path: pathlib.Path | None = None) -> None:
  """Create the random install ID that identifies this user's analytics."""
  settings = config.Config(
    path or pathlib.Path.home() / '.mirror' / 'config.json'
  )
  if settings.read().get('install_id'):
    return

  def assign(values: dict[str, object]) -> None:
    # for now, we create a new install ID for each user
    values.setdefault('install_id', uuid.uuid4().hex)

  settings.update(assign)


def take_analytics_notice(
  *, path: pathlib.Path | None = None
) -> text.Text | None:
  settings = config.Config(
    path or pathlib.Path.home() / '.mirror' / 'config.json'
  )
  if settings.read().get('analytics_notice_shown') is True:
    return None

  def remember(values: dict[str, object]) -> None:
    values['analytics_notice_shown'] = True

  settings.update(remember)
  notice = text.Text()
  notice.append(
    'Mirror collects anonymous feature usage to improve the product. '
  )
  notice.append('It never collects your prompts, code or file contents. ')
  notice.append('To turn this off, see ', style='dim')
  notice.append('/settings', style='bold')
  notice.append('.', style='dim')
  return notice


def _trusted_directories(values: dict[str, object]) -> list[str]:
  directories = values.get('trusted_directories')
  if not isinstance(directories, list):
    return []
  return [
    value for value in cast(list[object], directories) if isinstance(value, str)
  ]
