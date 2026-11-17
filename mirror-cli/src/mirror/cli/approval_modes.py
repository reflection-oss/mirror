"""Tool approval modes as the CLI names them and as the agent protocol sends them."""

from mirror.protocol import agent_pb2

NAMES = ('approve', 'auto', 'bypass')

_PROTOCOL: dict[str, agent_pb2.ApprovalMode] = {
  'approve': agent_pb2.APPROVAL_MODE_HITL,
  'auto': agent_pb2.APPROVAL_MODE_AUTO,
  'bypass': agent_pb2.APPROVAL_MODE_BYPASS,
}


def to_protocol(name: object) -> agent_pb2.ApprovalMode:
  """Encodes a CLI mode name, asking for approval when the name is unknown."""
  if not isinstance(name, str):
    return agent_pb2.APPROVAL_MODE_HITL
  return _PROTOCOL.get(name, agent_pb2.APPROVAL_MODE_HITL)


def from_protocol(mode: int) -> str:
  """Names a protocol mode, asking for approval when the mode is unknown."""
  return next(
    (name for name, value in _PROTOCOL.items() if value == mode), 'approve'
  )


def toggled(name: str) -> str:
  """Switches between manual and automatic approval; bypass is never toggled on."""
  return 'auto' if name == 'approve' else 'approve'
