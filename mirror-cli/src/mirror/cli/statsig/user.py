from __future__ import annotations

from mirror.cli.statsig import constants
from mirror.cli.statsig import environment as environment_lib


def payload(
  user_id: str, session_id: str, environment: environment_lib.Environment
) -> dict[str, object]:
  custom = {
    'sessionID': session_id,
    'systemName': environment.system_name,
    'systemVersion': environment.system_version,
    'architecture': environment.architecture,
    'timezone': environment.timezone,
    'terminal': environment.terminal,
  }
  user: dict[str, object] = {
    'userID': user_id,
    'appVersion': environment.app_version,
    'locale': environment.locale,
    'custom': {key: value for key, value in custom.items() if value},
  }
  return {key: value for key, value in user.items() if value}


def statsig_metadata(
  environment: environment_lib.Environment,
) -> dict[str, str]:
  """Build the request-level `statsigMetadata` identifying this client."""
  metadata = {'sdkType': constants.SDK_TYPE}
  if environment.app_version:
    metadata['sdkVersion'] = environment.app_version
  return metadata
