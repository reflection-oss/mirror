"""Choose terminal colors with /themes; preferences apply across workspaces."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import override

from mirror import config
from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  selection,
  state,
  themes,
)


@dataclasses.dataclass(frozen=True)
class Choose(actions.ExtensionAction):
  name: str = ''


def _route(name: str) -> actions.RoutedExtensionAction:
  return actions.RoutedExtensionAction(themes.EXTENSION_ID, Choose(name))


class ThemesExtension(extension.Extension[None, Choose, None]):
  @classmethod
  @override
  def id(cls) -> str:
    return themes.EXTENSION_ID

  @override
  def default_config(self) -> config.JsonObject:
    return {'name': 'dark'}

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'themes',
        'Choose colors for this terminal',
        lambda _context, args: _route(args.strip()),
        arguments=commands.OptionalArguments('[dark|light|terminal|mono]'),
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    None, tuple[effects.CoreEffect | extension.PrivateEffect[Choose], ...]
  ]:
    del global_state
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, Choose)
    ):
      return None, (self.effect(action.action),)
    return None, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: Choose
  ) -> None:
    if not effect.name:
      context.dispatch(
        actions.OpenSelector(
          selection.Selector(
            'themes',
            'Themes',
            (
              selection.Choice(
                'dark', 'Dark', 'Mirror colors for dark backgrounds'
              ),
              selection.Choice(
                'light', 'Light', 'Readable on light backgrounds'
              ),
              selection.Choice(
                'terminal', 'Terminal', 'ANSI colors; terminal background'
              ),
              selection.Choice(
                'mono',
                'Monochrome',
                'No color; reverse selection and text cues',
              ),
            ),
            _route,
            extension_id=self.id(),
            current=themes.configured(
              context.get_state().extensions.configuration(self.id())
            ),
          )
        )
      )
      return
    if effect.name not in themes.NAMES:
      context.dispatch(
        actions.ShowText(
          'Usage: /themes [dark|light|terminal|mono]', style='ui.warning'
        )
      )
      return
    path = (
      context.extensions.root.parent / 'config.json'
      if context.extensions
      else config.default_path()
    )
    try:
      config.Config(path).set_extension_config(
        themes.CONFIG_KEY, {'name': effect.name}
      )
    except (OSError, ValueError) as error:
      context.dispatch(
        actions.ShowText(f'Could not save theme: {error}', style='ui.error')
      )
      return
    context.dispatch(
      actions.ExtensionConfigChanged(
        self.id(), context.config | {'name': effect.name}
      )
    )
