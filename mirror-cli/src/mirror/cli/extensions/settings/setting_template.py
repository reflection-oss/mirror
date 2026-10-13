"""Declarative settings rows: preferences, actions, and nested menus."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable

from mirror import config
from mirror.cli import actions, extension

ON = 'on'
OFF = 'off'


@dataclasses.dataclass(frozen=True)
class Option:
  """One value a setting can take."""

  value: str
  label: str


ON_OFF = (Option(ON, 'On'), Option(OFF, 'Off'))


def _always_available(context: extension.Context) -> bool:
  del context  # Unused.
  return True


@dataclasses.dataclass(frozen=True)
class Setting:
  """A preference row with discrete values, changed by Enter or arrow keys."""

  key: str
  # title of the setting
  title: str
  # description of the setting
  description: str
  # options the user can choose between, for ex: (Option(ON, 'On'), Option(OFF, 'Off'))
  # or (Option(1, 'Compact'), Option(2, 'Detailed'), Option(3, 'Minimal'))
  options: tuple[Option, ...]
  # function to get the current value of the setting
  current: Callable[[extension.Context], str]
  # function to apply the new value of the setting
  apply: Callable[[extension.Context, str], Awaitable[None]]
  # function to check if the setting is available for the current workspace
  available: Callable[[extension.Context], bool] = _always_available

  def label(self, value: str) -> str:
    """Return the display label for `value`, or `value` itself if unknown."""
    return next(
      (option.label for option in self.options if option.value == value), value
    )


@dataclasses.dataclass(frozen=True)
class Action:
  """A row that closes the menu and dispatches an application action on Enter."""

  key: str
  title: str
  description: str
  action: actions.Action
  available: Callable[[extension.Context], bool] = _always_available


@dataclasses.dataclass(frozen=True)
class Menu:
  """A root menu or a submenu opened with Enter or Right.

  Item keys must be unique within this menu and must not contain `/`.
  Unavailable menus hide all their descendants.
  """

  key: str
  title: str
  description: str
  items: tuple[Item, ...]
  available: Callable[[extension.Context], bool] = _always_available


type Item = Setting | Action | Menu


def local_idle(context: extension.Context) -> bool:
  """Whether settings restricted to idle local sessions are available."""
  return not context.workspace.remote and not context.get_state().activity.busy


def user_config(context: extension.Context) -> config.Config:
  """Return the user's settings file for the extensions in `context`."""
  path = (
    context.extensions.root.parent / 'config.json'
    if context.extensions is not None
    else config.default_path()
  )
  return config.Config(path)
