from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Mapping

from mirror.cli.statsig import environment as environment_lib
from mirror.cli.statsig import user as user_lib
from mirror.cli.statsig.events import constants

type MetadataValue = str | int | float | bool


@dataclasses.dataclass(frozen=True)
class Event:
  name: str
  value: str | int | float | None = None
  metadata: Mapping[str, MetadataValue] = dataclasses.field(
    default_factory=dict[str, MetadataValue]
  )


def encode(
  event: Event,
  *,
  user_id: str,
  session_id: str,
  environment: environment_lib.Environment,
) -> bytes:
  """Validate an event and encode the complete request that delivers it.

  Raises:
    ValueError: The event breaks the naming, type, or size rules.
  """
  _validate(event)
  payload: dict[str, object] = {
    'eventName': event.name,
    'user': user_lib.payload(user_id, session_id, environment),
    'time': time.time_ns() // 1_000_000,
    'metadata': dict(event.metadata),
  }
  if event.value is not None:
    payload['value'] = event.value
  body = {
    'events': [payload],
    'statsigMetadata': user_lib.statsig_metadata(environment),
  }
  return _to_json(body).encode('utf-8')


def _validate(event: Event) -> None:
  if constants.IDENTIFIER_FORMAT.fullmatch(event.name) is None:
    raise ValueError('Invalid Statsig event name')
  if len(event.name) > constants.MAX_EVENT_NAME_LENGTH:
    raise ValueError(
      'Statsig event names must not exceed '
      f'{constants.MAX_EVENT_NAME_LENGTH} characters'
    )

  if event.value is not None and type(event.value) not in (str, int, float):
    raise ValueError('Statsig event values must be strings or numbers')
  if (
    isinstance(event.value, str)
    and len(event.value) > constants.MAX_EVENT_VALUE_LENGTH
  ):
    raise ValueError(
      'Statsig event value strings must not exceed '
      f'{constants.MAX_EVENT_VALUE_LENGTH} characters'
    )

  for key, value in event.metadata.items():
    if constants.IDENTIFIER_FORMAT.fullmatch(key) is None:
      raise ValueError('Invalid Statsig metadata key')
    if type(value) not in (str, int, float, bool):
      raise ValueError('Statsig metadata values must be scalars')
  if len(_to_json(event.metadata)) > constants.MAX_METADATA_LENGTH:
    raise ValueError(
      'Statsig event metadata must not exceed '
      f'{constants.MAX_METADATA_LENGTH} characters as JSON'
    )


def _to_json(payload: Mapping[str, object]) -> str:
  return json.dumps(payload, allow_nan=False, separators=(',', ':'))
