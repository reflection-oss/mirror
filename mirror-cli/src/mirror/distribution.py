"""Installed metadata for release and development distributions."""

import json
import pathlib
from importlib import metadata

REMOTE_ENDPOINT = 'remote_endpoint.json'


def installed() -> metadata.Distribution:
  """Return the release distribution, or the development distribution."""
  try:
    return metadata.distribution('reflection-mirror')
  except metadata.PackageNotFoundError:
    return metadata.distribution('mirror')


def remote_endpoint(
  package: pathlib.Path = pathlib.Path(__file__).parent,
) -> str | None:
  """Return the Mirror server this build was packaged with, if any."""
  try:
    content = (package / REMOTE_ENDPOINT).read_text()
  except FileNotFoundError:
    return None
  endpoint = json.loads(content)['endpoint']
  if not isinstance(endpoint, str) or not endpoint:
    raise ValueError(f'Invalid packaged remote endpoint: {endpoint!r}')
  return endpoint
