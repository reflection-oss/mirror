from __future__ import annotations

import pytest

from mirror.cli import sandboxes


def test_sandbox_settings_survive_being_stored() -> None:
  sandbox = sandboxes.Sandbox.from_settings({
    'sandbox': {
      'image': 'python:3.12-bookworm',
      'egress': ['github.com'],
      'cpus': 4,
      'memory_mib': 4096,
    }
  })

  assert sandboxes.Sandbox.from_json(sandbox.to_json()) == sandbox
  assert (sandbox.image, sandbox.egress, sandbox.cpus, sandbox.memory_mib) == (
    'python:3.12-bookworm',
    ('github.com',),
    4,
    4096,
  )


def test_sandbox_without_settings_uses_the_defaults() -> None:
  sandbox = sandboxes.Sandbox.from_settings({})

  assert (sandbox.image, sandbox.egress) == (
    sandboxes.DEFAULT_IMAGE,
    sandboxes.DEFAULT_EGRESS,
  )


@pytest.mark.parametrize(
  ('setting', 'value'), [('cpus', 256), ('memory_mib', 2**32)]
)
def test_sandbox_settings_reject_a_size_the_worker_cannot_read(
  setting: str, value: int
) -> None:
  with pytest.raises(sandboxes.SandboxError, match=setting):
    sandboxes.Sandbox.from_settings({'sandbox': {setting: value}})


def test_sandbox_settings_reject_egress_that_is_not_a_list() -> None:
  with pytest.raises(sandboxes.SandboxError, match='egress'):
    sandboxes.Sandbox.from_settings({'sandbox': {'egress': 'github.com'}})
