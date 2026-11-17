"""Client-side sandbox settings; the daemon owns clones and VM lifecycle."""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import uuid
from collections.abc import Mapping
from typing import cast

DEFAULT_IMAGE = 'python:3.12-bookworm'
# Hosts the VM may reach over HTTPS; all other traffic is denied.
DEFAULT_EGRESS = (
  'github.com',
  'api.github.com',
  'codeload.github.com',
  'objects.githubusercontent.com',
  'pypi.org',
  'files.pythonhosted.org',
  'registry.npmjs.org',
)
DEFAULT_CPUS = 2
DEFAULT_MEMORY_MIB = 2048
# The worker reads `cpus` as a `u8` and `memory_mib` as a `u32`.
_MAX_CPUS = 255
_MAX_MEMORY_MIB = 2**32 - 1


class SandboxError(RuntimeError):
  """A sandbox configuration is invalid."""


@dataclasses.dataclass(frozen=True)
class Sandbox:
  """The microVM configuration sent to the daemon."""

  id: str
  image: str = DEFAULT_IMAGE
  egress: tuple[str, ...] = DEFAULT_EGRESS
  cpus: int = DEFAULT_CPUS
  memory_mib: int = DEFAULT_MEMORY_MIB

  def worker_config(self, sandboxd: pathlib.Path) -> dict[str, object]:
    """The worker's `sandbox` configuration, mounting `sandboxd` in the VM."""
    return {
      'id': self.id,
      'image': self.image,
      'sandboxd': str(sandboxd),
      'egress': list(self.egress),
      'cpus': self.cpus,
      'memory_mib': self.memory_mib,
    }

  def to_json(self) -> str:
    """The form `from_json` reads back."""
    return json.dumps(dataclasses.asdict(self))

  @classmethod
  def from_json(cls, text: str) -> Sandbox:
    """Read a stored sandbox, rejecting anything this version cannot run."""
    value = cast(object, json.loads(text))
    if not isinstance(value, dict):
      raise SandboxError(f'stored sandbox is not an object: {text!r}')
    values = cast(dict[str, object], value)
    return _sandbox(str(values.get('id', '')), values)

  @classmethod
  def from_settings(cls, settings: Mapping[str, object]) -> Sandbox:
    """A new sandbox configured by the `sandbox` settings object."""
    configured = settings.get('sandbox', {})
    if not isinstance(configured, dict):
      raise SandboxError('the "sandbox" setting must be an object')
    return _sandbox(str(uuid.uuid4()), cast(dict[str, object], configured))


def _sandbox(sandbox_id: str, values: Mapping[str, object]) -> Sandbox:
  image = values.get('image', DEFAULT_IMAGE)
  egress = values.get('egress', list(DEFAULT_EGRESS))
  cpus = values.get('cpus', DEFAULT_CPUS)
  memory_mib = values.get('memory_mib', DEFAULT_MEMORY_MIB)
  try:
    uuid.UUID(sandbox_id)
  except ValueError as error:
    raise SandboxError(f'sandbox id {sandbox_id!r} is not a UUID') from error
  if not isinstance(image, str) or not image:
    raise SandboxError('sandbox "image" must be a non-empty string')
  hosts = cast(list[object], egress) if isinstance(egress, list) else None
  if hosts is None or not all(isinstance(host, str) for host in hosts):
    raise SandboxError('sandbox "egress" must be a list of host names')
  if (
    not isinstance(cpus, int)
    or isinstance(cpus, bool)
    or not 1 <= cpus <= _MAX_CPUS
  ):
    raise SandboxError(f'sandbox "cpus" must be from 1 to {_MAX_CPUS}')
  if (
    not isinstance(memory_mib, int)
    or isinstance(memory_mib, bool)
    or not 256 <= memory_mib <= _MAX_MEMORY_MIB
  ):
    raise SandboxError(
      f'sandbox "memory_mib" must be from 256 to {_MAX_MEMORY_MIB}'
    )
  return Sandbox(
    sandbox_id, image, tuple(str(host) for host in hosts), cpus, memory_mib
  )


def sandboxd_path() -> pathlib.Path:
  """The Linux `sandboxd` to mount, from this install or `$MIRROR_SANDBOXD`."""
  override = os.environ.get('MIRROR_SANDBOXD')
  if override:
    return pathlib.Path(override).absolute()
  return pathlib.Path(__file__).parents[1] / 'bin' / 'linux' / 'mirror-sandboxd'
