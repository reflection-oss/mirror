"""User settings stored in ~/.mirror/config.json."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import fcntl
import json
import logging
import pathlib
import tempfile
from collections.abc import Awaitable, Callable
from typing import Literal, cast

type JsonValue = (
  None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]


@dataclasses.dataclass(frozen=True)
class ExtensionSettings:
  """Host controls and effective plugin settings, resolved without importing code."""

  enabled: bool | Literal['hidden'] = True
  config: JsonObject = dataclasses.field(default_factory=dict[str, JsonValue])


def default_path() -> pathlib.Path:
  """Where the CLI keeps its settings."""
  return pathlib.Path.home() / '.mirror' / 'config.json'


def save_approval_mode(mode: str) -> None:
  """Remember the user's approval mode for future CLI launches."""
  with contextlib.suppress(OSError, ValueError):
    Config(default_path()).update(
      lambda settings: settings.update({'approval_mode': mode})
    )


def analytics_enabled(settings: dict[str, object]) -> bool:
  """Whether the user permits analytics; only an explicit false opts out."""
  return settings.get('analytics') is not False


def checkpoints_enabled(settings: dict[str, object]) -> bool:
  """Whether the user explicitly permits repository checkpoints."""
  return settings.get('checkpoints_enabled') is True


def same_config(left: JsonObject, right: JsonObject) -> bool:
  """Compare JSON settings without treating booleans as equal to numbers."""
  return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def merge_config(defaults: JsonObject, overrides: JsonObject) -> JsonObject:
  """Merge objects recursively; all other explicit values replace defaults."""
  result = copy.deepcopy(defaults)
  for key, value in overrides.items():
    previous = result.get(key)
    result[key] = (
      merge_config(previous, value)
      if isinstance(previous, dict) and isinstance(value, dict)
      else copy.deepcopy(value)
    )
  return result


def extension_overrides(settings: object, key: str) -> JsonObject:
  """Read one user entry, preserving the existing boolean enablement shorthand."""
  if settings is None:
    return {}
  if not isinstance(settings, dict):
    raise ValueError('extensions must be a JSON object')
  value = cast(dict[str, object], settings).get(key, {})
  if isinstance(value, bool):
    return {'enabled': value}
  if not isinstance(value, dict):
    raise ValueError(f'Extension {key!r} settings must be a JSON object')
  return cast(JsonObject, value)


def _validate_extension_settings(value: JsonObject) -> None:
  if 'hidden' in value:
    raise ValueError('Use enabled: "hidden" instead of a separate hidden flag')
  enabled = value.get('enabled', True)
  if not isinstance(enabled, bool) and enabled != 'hidden':
    raise ValueError('enabled must be true, false, or "hidden"')
  if 'config' in value and not isinstance(value['config'], dict):
    raise ValueError('config must be a JSON object')


def resolve_extension(
  overrides: JsonObject, defaults: JsonObject | None = None
) -> ExtensionSettings:
  """Overlay user config on extension defaults, keeping host controls separate."""
  _validate_extension_settings(overrides)
  return ExtensionSettings(
    enabled=cast(bool | Literal['hidden'], overrides.get('enabled', True)),
    config=merge_config(
      defaults or {}, cast(JsonObject, overrides.get('config', {}))
    ),
  )


def extension_choices(settings: object) -> dict[str, bool]:
  """Read recognized enabled flags without interpreting invalid values as approval."""
  if not isinstance(settings, dict):
    return {}
  choices: dict[str, bool] = {}
  for key, entry in cast(dict[str, object], settings).items():
    enabled = (
      cast(dict[str, object], entry).get('enabled')
      if isinstance(entry, dict)
      else entry
    )
    if isinstance(enabled, bool):
      choices[key] = enabled
  return choices


class Config:
  """Read settings on demand so running CLIs share extension choices."""

  def __init__(self, path: pathlib.Path) -> None:
    self.path = path
    self._signature: tuple[int, int, int] | None = None
    self._cached: dict[str, object] | None = None

  def read(self) -> dict[str, object]:
    """Return a copy of the cached config, reparsing only after a file change."""
    try:
      stat = self.path.stat()
      signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except FileNotFoundError:
      signature = None
    if self._cached is None or signature != self._signature:
      data: object = (
        json.loads(self.path.read_text()) if signature is not None else {}
      )
      if not isinstance(data, dict):
        raise ValueError(f'Config must be a JSON object: {self.path}')
      self._cached = cast(dict[str, object], data)
      self._signature = signature
    return copy.deepcopy(self._cached)

  async def watch(
    self,
    apply_config: Callable[[dict[str, object]], Awaitable[None]],
    *,
    interval: float = 5.0,
  ) -> None:
    """Apply changed valid configurations; leave running settings alone on errors."""
    previous: JsonObject | None = None
    last_error: str | None = None
    while True:
      try:
        settings = self.read()
      except (OSError, ValueError) as error:
        if str(error) != last_error:
          logging.getLogger(__name__).warning(
            'Ignoring invalid config %s: %s', self.path, error
          )
        last_error = str(error)
      else:
        last_error = None
        current = cast(JsonObject, settings)
        if previous is None or not same_config(current, previous):
          await apply_config(settings)
          previous = current
      await asyncio.sleep(interval)

  def set_extension(self, key: str, enabled: bool) -> None:
    """Update one choice without losing other settings or another CLI's changes."""

    def change(settings: dict[str, object]) -> None:
      entries = settings.get('extensions')
      extensions = (
        cast(dict[str, object], entries) if isinstance(entries, dict) else {}
      )
      entry = extensions.get(key)
      value = cast(dict[str, object], entry) if isinstance(entry, dict) else {}
      extensions[key] = {**value, 'enabled': enabled}
      settings['extensions'] = extensions

    self.update(change)

  def set_extension_config(self, key: str, updates: JsonObject) -> None:
    """Merge `updates` into one extension's config, keeping its enablement."""

    def change(settings: dict[str, object]) -> None:
      entries = settings.get('extensions')
      extensions = (
        cast(dict[str, object], entries) if isinstance(entries, dict) else {}
      )
      entry = extensions.get(key)
      if isinstance(entry, dict):
        value = cast(dict[str, object], entry)
      elif isinstance(entry, bool):
        value = {'enabled': entry}
      else:
        value = {}
      configured = value.get('config')
      current = (
        cast(dict[str, object], configured)
        if isinstance(configured, dict)
        else {}
      )
      extensions[key] = {**value, 'config': {**current, **updates}}
      settings['extensions'] = extensions

    self.update(change)

  def update(self, change: Callable[[dict[str, object]], None]) -> None:
    """Apply `change` to the settings under the file lock and write atomically."""
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self.path.with_suffix('.lock').open('a') as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      settings = self.read()
      change(settings)
      with tempfile.NamedTemporaryFile(
        mode='w', dir=self.path.parent, delete=False
      ) as output:
        temporary = pathlib.Path(output.name)
        try:
          output.write(json.dumps(settings, indent=2) + '\n')
          output.close()
          temporary.replace(self.path)
        finally:
          temporary.unlink(missing_ok=True)
