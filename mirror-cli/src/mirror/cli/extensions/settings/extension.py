"""The `/settings` menu for preferences and tool permissions."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence
from typing import override

from mirror.cli import actions, commands, effects, extension, selection, state
from mirror.cli.extensions.settings import setting_template
from mirror.cli.extensions.settings.definitions import (
  analytics,
  history_detail,
  memory,
  permissions,
)

_MENU = setting_template.Menu(
  'settings',
  'Settings',
  '',
  (
    analytics.SETTING,
    memory.MENU,
    history_detail.SETTING,
    permissions.SETTING,
  ),
)


@dataclasses.dataclass(frozen=True)
class Show(actions.ExtensionAction):
  """Open a menu at `path`, optionally with one row highlighted."""

  selected: str | None = None
  path: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Select(actions.ExtensionAction):
  """Activate a menu row, or adjust it with an arrow key when `delta` is set."""

  key: str
  delta: int | None = None
  path: tuple[str, ...] = ()


def _find_item(
  context: extension.Context, menu: setting_template.Menu, key: str
) -> setting_template.Item:
  for item in menu.items:
    if item.key == key and item.available(context):
      return item
  raise ValueError(f'The {key} setting is not available in this session.')


def _menus(
  context: extension.Context, root: setting_template.Menu, path: tuple[str, ...]
) -> tuple[setting_template.Menu, ...]:
  """Resolve a visible menu path and its ancestors for navigation."""
  menus = [root]
  for key in path:
    item = _find_item(context, menus[-1], key)
    if not isinstance(item, setting_template.Menu):
      raise ValueError(f'The {key} setting is not a submenu.')
    menus.append(item)
  return tuple(menus)


def _menu_id(root: setting_template.Menu, path: tuple[str, ...]) -> str:
  return '/'.join((root.key, *path))


def _route(action: actions.ExtensionAction) -> actions.RoutedExtensionAction:
  return actions.RoutedExtensionAction(SettingsExtension.id(), action)


def _choices(
  context: extension.Context, items: tuple[setting_template.Item, ...]
) -> tuple[selection.Choice, ...]:
  # Size columns by every option, not just current values, so toggling a
  # setting never shifts the titles and descriptions.
  value_width = 0
  for item in items:
    if isinstance(item, setting_template.Setting):
      value_width = max(
        value_width, max(len(option.label) for option in item.options)
      )
  title_width = max((len(item.title) for item in items), default=0)
  rows: list[selection.Choice] = []
  for item in items:
    if isinstance(item, setting_template.Setting):
      value = item.label(item.current(context)).ljust(value_width)
      status = f'< {value} >'
    else:
      status = '>'.rjust(value_width + 4)
    rows.append(
      selection.Choice(
        item.key,
        item.title.ljust(title_width),
        item.description,
        status=status,
      )
    )
  return tuple(rows)


def _open_menu(
  context: extension.Context, root: setting_template.Menu, show: Show
) -> None:
  menus = _menus(context, root, show.path)
  items = tuple(item for item in menus[-1].items if item.available(context))
  title = ' › '.join(menu.title for menu in menus)
  if any(isinstance(item, setting_template.Setting) for item in items):
    title += ' · ←/→ to change'
  cancel = (
    _route(Show(selected=show.path[-1], path=show.path[:-1]))
    if show.path
    else None
  )
  context.dispatch(
    actions.OpenSelector(
      selection.Selector(
        _menu_id(root, show.path),
        title,
        _choices(context, items),
        lambda key: _route(Select(key, path=show.path)),
        extension_id=SettingsExtension.id(),
        selected=show.selected,
        keep_open=True,
        on_adjust=lambda key, delta: _route(Select(key, delta, show.path)),
        cancel_action=cancel,
      )
    )
  )


def _stepped(item: setting_template.Setting, current: str, delta: int) -> str:
  values = [option.value for option in item.options]
  index = values.index(current) if current in values else -1
  return values[(index + delta) % len(values)]


class SettingsExtension(extension.Extension[None, Show | Select, None]):
  """Configure preferences without owning analytics or worker lifetimes."""

  def __init__(self, menu: setting_template.Menu = _MENU) -> None:
    self._menu = menu
    # One shared instance serves multiple workspaces; never overlap restarts.
    self._change_lock = asyncio.Lock()

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/settings'

  @override
  def tips(self) -> tuple[str, ...]:
    return ('Use /settings to adjust preferences',)

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'settings',
        'Change preferences and tool approval',
        lambda _context, _args: _route(Show()),
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    None,
    tuple[effects.CoreEffect | extension.PrivateEffect[Show | Select], ...],
  ]:
    del global_state
    if not isinstance(action, actions.RoutedExtensionAction):
      return None, ()
    if action.extension_id != self.id():
      return None, ()
    if not isinstance(action.action, (Show, Select)):
      return None, ()
    return None, (self.effect(action.action),)

  @override
  async def handle_effect(
    self, context: extension.Context, effect: Show | Select
  ) -> None:
    try:
      if isinstance(effect, Show):
        _open_menu(context, self._menu, effect)
      else:
        await self._select(context, effect)
    except (OSError, ValueError) as error:
      context.dispatch(actions.ShowText(str(error), style='ui.error'))

  async def _select(self, context: extension.Context, selected: Select) -> None:
    async with self._change_lock:
      request = context.get_state().selector.selector
      if (
        request is None
        or request.extension_id != self.id()
        or request.id != _menu_id(self._menu, selected.path)
      ):
        return
      menu = _menus(context, self._menu, selected.path)[-1]
      item = _find_item(context, menu, selected.key)
      if isinstance(item, setting_template.Menu):
        if selected.delta is None or selected.delta > 0:
          _open_menu(context, self._menu, Show(path=(*selected.path, item.key)))
        return
      if isinstance(item, setting_template.Action):
        if selected.delta is None:
          context.dispatch(actions.ClearSelector())
          context.dispatch(item.action)
        return
      current = item.current(context)
      delta = selected.delta if selected.delta is not None else 1
      target = _stepped(item, current, delta)
      if target != current:
        await item.apply(context, target)
      # A slow worker restart must not reopen a dismissed or replaced menu.
      if context.get_state().selector.selector is request:
        _open_menu(
          context, self._menu, Show(selected=item.key, path=selected.path)
        )
