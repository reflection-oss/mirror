from __future__ import annotations

import dataclasses
import importlib.metadata
import locale
import os
import platform

from mirror.cli.statsig import constants


@dataclasses.dataclass(frozen=True)
class Environment:
  # The Mirror CLI version, e.g. `0.4.2`.
  app_version: str = ''
  # The OS name, e.g. `Darwin` or `Linux`.
  system_name: str = ''
  # The OS release, e.g. `25.6.0`.
  system_version: str = ''
  # The CPU architecture, e.g. `arm64` or `x86_64`.
  architecture: str = ''
  # The user's locale, e.g. `en_US`.
  locale: str = ''
  # The IANA timezone name, e.g. `America/New_York`.
  timezone: str = ''
  # The terminal program from `$TERM_PROGRAM`, e.g. `iTerm.app`.
  terminal: str = ''


def detect() -> Environment:
  """Detect the current install's environment; unknown facts are left empty."""
  try:
    app_version = importlib.metadata.version('mirror')
  except importlib.metadata.PackageNotFoundError:
    app_version = ''
  try:
    locale_name = locale.getlocale()[0] or ''
  except ValueError:
    locale_name = ''
  # Locale and terminal come from user-controlled environment variables.
  limit = constants.MAX_ENVIRONMENT_FIELD_LENGTH
  return Environment(
    app_version=app_version,
    system_name=platform.system(),
    system_version=platform.release(),
    architecture=platform.machine(),
    # `C` and `POSIX` mean no locale is configured.
    locale='' if locale_name in ('C', 'POSIX') else locale_name[:limit],
    timezone=os.path.realpath('/etc/localtime').partition('zoneinfo/')[2],
    terminal=os.environ.get('TERM_PROGRAM', '')[:limit],
  )
