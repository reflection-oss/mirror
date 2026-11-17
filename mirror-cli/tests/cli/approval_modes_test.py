import pytest
from mirror.protocol import agent_pb2

from mirror.cli import approval_modes


@pytest.mark.parametrize(
  ('name', 'mode'),
  [
    ('approve', agent_pb2.APPROVAL_MODE_HITL),
    ('auto', agent_pb2.APPROVAL_MODE_AUTO),
    ('bypass', agent_pb2.APPROVAL_MODE_BYPASS),
  ],
)
def test_to_protocol_round_trips_each_mode(name: str, mode: int) -> None:
  assert approval_modes.to_protocol(name) == mode
  assert approval_modes.from_protocol(mode) == name


def test_to_protocol_asks_for_an_unknown_name() -> None:
  assert approval_modes.to_protocol('yolo') == agent_pb2.APPROVAL_MODE_HITL


def test_toggled_switches_manual_and_auto_and_leaves_bypass() -> None:
  assert [
    approval_modes.toggled('approve'),
    approval_modes.toggled('auto'),
    approval_modes.toggled('bypass'),
  ] == ['auto', 'approve', 'approve']
