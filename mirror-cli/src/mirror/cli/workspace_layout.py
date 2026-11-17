"""Client-local tab snapshots; the last explicit edit wins, not daemon activity."""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import tempfile

from google.protobuf import message
from mirror.protocol import daemon_pb2


def path(
  directory: pathlib.Path, database: str = '', *, remote_host: str = ''
) -> pathlib.Path:
  """Scope local layouts to launch context and remote layouts to connection identity."""
  if remote_host:
    return settings_path(
      pathlib.Path.home() / '.mirror', remote_host
    ).with_suffix('.pb')
  state = pathlib.Path(
    database
    or os.environ.get('MIRROR_STATE_DB')
    or pathlib.Path.home() / '.mirror' / 'state.db'
  ).resolve()
  location = str(directory.resolve())
  key = hashlib.sha256(location.encode()).hexdigest()
  return state.with_name(f'{state.name}.tabs') / f'{key}.pb'


def settings_path(root: pathlib.Path, scope: str) -> pathlib.Path:
  """Keep daemon-specific UI preferences and directory consent out of global settings."""
  key = hashlib.sha256(scope.encode()).hexdigest()
  return root / 'connections' / f'{key}.json'


def load(location: pathlib.Path) -> daemon_pb2.WorkspaceTabs:
  """Read a startup snapshot without watching other clients' tab changes."""
  try:
    return daemon_pb2.WorkspaceTabs.FromString(location.read_bytes())
  except FileNotFoundError:
    return daemon_pb2.WorkspaceTabs()
  except (OSError, message.DecodeError) as error:
    logging.getLogger(__name__).warning('Cannot restore tabs: %s', error)
    return daemon_pb2.WorkspaceTabs()


def save(location: pathlib.Path, tabs: daemon_pb2.WorkspaceTabs) -> None:
  """Atomically replace the snapshot without merging independent CLI layouts."""
  location.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=location.parent, delete=False) as output:
    temporary = pathlib.Path(output.name)
    try:
      output.write(tabs.SerializeToString())
      output.close()
      temporary.replace(location)
    finally:
      temporary.unlink(missing_ok=True)
