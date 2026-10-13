"""Immutable state shared by the extension host and UI reducer."""

from __future__ import annotations

import copy
import dataclasses
import types
from collections.abc import Mapping

from mirror import config


@dataclasses.dataclass(frozen=True, init=False)
class ExtensionStates:
  """Immutable state keyed by stable extension IDs."""

  _values: Mapping[str, object]
  _configs: Mapping[str, config.JsonObject]

  def __init__(
    self,
    values: Mapping[str, object] | None = None,
    configs: Mapping[str, config.JsonObject] | None = None,
  ) -> None:
    object.__setattr__(
      self,
      '_configs',
      types.MappingProxyType(copy.deepcopy(dict(configs or {}))),
    )
    object.__setattr__(
      self,
      '_values',
      types.MappingProxyType(dict(values) if values is not None else {}),
    )

  def __contains__(self, extension_id: str) -> bool:
    return extension_id in self._values

  def __getitem__(self, extension_id: str) -> object:
    return self._values[extension_id]

  def updated(self, extension_id: str, value: object) -> ExtensionStates:
    """Return a snapshot with one extension state replaced."""
    return ExtensionStates(
      dict(self._values) | {extension_id: value}, self._configs
    )

  def configuration(self, extension_id: str) -> config.JsonObject:
    """Return an independent configuration snapshot for one extension."""
    return copy.deepcopy(self._configs.get(extension_id, {}))

  def configured(
    self, extension_id: str, value: config.JsonObject
  ) -> ExtensionStates:
    """Return the same state with a new host-owned configuration snapshot."""
    configs = dict(self._configs)
    if value:
      configs[extension_id] = value
    else:
      configs.pop(extension_id, None)
    return ExtensionStates(self._values, configs)

  def removed(self, extension_id: str) -> ExtensionStates:
    """Return a snapshot without one extension's state."""
    values = dict(self._values)
    values.pop(extension_id, None)
    configs = dict(self._configs)
    configs.pop(extension_id, None)
    return ExtensionStates(values, configs)
