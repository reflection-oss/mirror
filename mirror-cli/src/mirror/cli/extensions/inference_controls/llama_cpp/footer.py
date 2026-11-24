"""How the llama.cpp server and its downloads look in the footer."""

from __future__ import annotations

from mirror.protocol import (
  local_models_pb2,
)

from mirror.cli import local_models_client

# Phases in which the server is coming up, and in which it answers requests.
STARTING = (
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_STARTING,
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_RESTARTING,
)
RUNNING = (
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_READY,
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING,
  local_models_pb2.LOCAL_MODEL_SERVER_PHASE_SERVING,
)
_WORKING = 'class:footer-approval-manual'
_IDLE = 'class:footer'
_METER_CELLS = 7


def tokens(count: int) -> str:
  """A token count in binary thousands, as models advertise it: 262144 is 256K."""
  return f'{round(count / 1024)}K' if count >= 1024 else str(count)


def short_name(name: str) -> str:
  """`owner/Repo-GGUF:Q4_K_M` as `Repo`."""
  base = name.partition(':')[0].rpartition('/')[2] or name
  return base.removesuffix('-GGUF').removesuffix('-gguf') or base


def status(runtime: local_models_client.Client, model: str) -> tuple[str, str]:
  """The footer style and text for the local server, as seen by `model`."""
  state = runtime.state
  serving_model = (
    state.phase == local_models_pb2.LOCAL_MODEL_SERVER_PHASE_SERVING
    and state.active_model == model
  )
  match state.phase:
    case local_models_pb2.LOCAL_MODEL_SERVER_PHASE_STARTING:
      return _WORKING, '◐ starting…'
    case local_models_pb2.LOCAL_MODEL_SERVER_PHASE_RESTARTING:
      return _WORKING, '◐ restarting…'
    case local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING:
      return _WORKING, f'◐ loading {short_name(state.active_model)}…'
    case _ if serving_model and state.asleep:
      return _IDLE, '○ asleep'
    case _ if serving_model:
      return 'class:footer-approval-auto', '● serving'
    case local_models_pb2.LOCAL_MODEL_SERVER_PHASE_SERVING:
      return _IDLE, '○ ready'
    case local_models_pb2.LOCAL_MODEL_SERVER_PHASE_READY:
      return _IDLE, '○ ready'
    case _:
      return _IDLE, '○ stopped'


def downloads(
  runtime: local_models_client.Client,
) -> tuple[tuple[str, str], ...]:
  """Footer fragments for background downloads; never part of the transcript.

  A download that ended shows how it ended for as long as the daemon lists it.
  """
  active = [
    item
    for item in runtime.state.transfers
    if item.phase == local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_DOWNLOADING
  ]
  fragments: list[tuple[str, str]] = []
  if active:
    first = active[0]
    filled, remaining = _meter(first.done_bytes, first.total_bytes)
    percent = (
      f' {first.done_bytes * 100 // first.total_bytes}%'
      if first.total_bytes
      else ''
    )
    more = f' +{len(active) - 1}' if len(active) > 1 else ''
    fragments += [
      (_IDLE, f'{short_name(first.name)} '),
      ('class:footer-approval-auto', filled),
      (_IDLE, f'{remaining}{percent}{more}'),
    ]
  for item in runtime.state.transfers:
    if item.phase != local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_DOWNLOADING:
      if fragments:
        fragments.append((_IDLE, ' · '))
      fragments.append((_IDLE, _ended(short_name(item.name), item.phase)))
  return tuple(fragments)


def _ended(label: str, phase: int) -> str:
  if phase == local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_FINISHED:
    return f'✓ {label} downloaded'
  if phase == local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_FAILED:
    return f'✗ {label} download failed'
  return f'⏸ {label} paused'


def _meter(done: int, total: int) -> tuple[str, str]:
  """A thin line meter as (filled, remaining), ending in a rounded head."""
  if not total:
    return '', '─' * _METER_CELLS
  full = int(min(done / total, 1) * _METER_CELLS)
  if full >= _METER_CELLS:
    return '━' * _METER_CELLS, ''
  return '━' * full + '╸', '─' * (_METER_CELLS - full - 1)
