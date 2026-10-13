"""Installed metadata for release and development distributions."""

from importlib import metadata


def installed() -> metadata.Distribution:
  """Return the release distribution, or the development distribution."""
  try:
    return metadata.distribution('mirror-agent')
  except metadata.PackageNotFoundError:
    return metadata.distribution('mirror')
