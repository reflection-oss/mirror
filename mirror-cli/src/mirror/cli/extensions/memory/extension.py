"""Personal memory: show its status and import notes for this machine."""

from __future__ import annotations

import dataclasses
import pathlib
import shlex
from collections.abc import Mapping, Sequence
from typing import override

from mirror import config
from mirror.cli import actions, commands, effects, extension, shell, state

# The entry under `extensions` in the settings file. Memory has one switch,
# `config.enabled` in that entry; the worker and the command both read it.
KEY = 'builtin/memory'
STORE = pathlib.Path.home() / '.mirror' / 'memory'
_USAGE = 'Usage: /memory [status | import [sources…]]'


def enabled(settings: Mapping[str, object]) -> bool:
  """Whether sessions on this machine get the user's personal memory.

  Only the JSON boolean `true` turns it on, and a disabled extension counts
  as off whatever its config says.
  """
  try:
    overrides = config.extension_overrides(settings.get('extensions'), KEY)
  except ValueError:
    return False
  if overrides.get('enabled') is False:
    return False
  configured = overrides.get('config')
  return isinstance(configured, dict) and configured.get('enabled') is True


def worker_config(settings: Mapping[str, object]) -> dict[str, object]:
  """What memory adds to the worker's launch values: its store, when on."""
  return {'memory_dir': str(STORE) if enabled(settings) else None}


@dataclasses.dataclass(frozen=True)
class Run(actions.ExtensionAction):
  """One `/memory` invocation, already validated."""

  arguments: tuple[str, ...]


def _handle_memory(
  context: commands.CommandContext, args: str
) -> actions.Action:
  del context
  try:
    arguments = tuple(shlex.split(args)) or ('status',)
  except ValueError as error:
    return actions.ShowText(f'{_USAGE} ({error})', style='ui.warning')
  if arguments[0] not in ('status', 'import'):
    return actions.ShowText(_USAGE, style='ui.warning')
  if arguments[0] != 'import' and len(arguments) > 1:
    return actions.ShowText(_USAGE, style='ui.warning')
  return actions.RoutedExtensionAction(MemoryExtension.id(), Run(arguments))


def _settings(context: extension.Context) -> config.Config:
  if context.extensions is None:
    return config.Config(config.default_path())
  return config.Config(context.extensions.root.parent / 'config.json')


def _show(
  context: extension.Context, headline: str, details: Sequence[str] = ()
) -> None:
  context.dispatch(actions.ShowText(f'◆ {headline}', style='ui.info'))
  for line in details:
    skipped = line.lstrip().startswith('skipped ')
    context.dispatch(
      actions.ShowText(line, style='ui.warning' if skipped else 'ui.muted')
    )


class MemoryExtension(extension.Extension[None, Run, None]):
  """Show personal memory status for this machine and import notes."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/memory'

  @override
  def default_config(self) -> config.JsonObject:
    return {'enabled': False}

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'memory',
        'Show memory status or import memories',
        _handle_memory,
        arguments=commands.OptionalArguments('[status|import]'),
      ),
    )

  @override
  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[
    None, tuple[effects.CoreEffect | extension.PrivateEffect[Run], ...]
  ]:
    del global_state
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, Run)
    ):
      return None, (self.effect(action.action),)
    return None, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: Run
  ) -> None:
    if not context.worker_command or context.worker_cwd is None:
      context.dispatch(
        actions.ShowText(
          'Memory is unavailable in this workspace.', style='ui.warning'
        )
      )
      return
    action = effect.arguments[0]
    if action == 'import':
      await _import(context, effect.arguments[1:])
    else:
      try:
        on = enabled(_settings(context).read())
      except (OSError, ValueError) as error:
        context.dispatch(
          actions.ShowText(
            f'could not read memory settings: {error}', style='ui.error'
          )
        )
        return
      _show(context, f'memory {"on" if on else "off"}')


async def set_enabled(context: extension.Context, on: bool) -> bool:
  """Persist the switch, publish it, and relaunch this session's worker.

  Returns:
    Whether this session's worker was restarted.

  Raises:
    OSError: The settings file could not be written.
    ValueError: The settings file is malformed.
  """
  _settings(context).set_extension_config(KEY, {'enabled': on})
  context.dispatch(
    actions.ExtensionConfigChanged(MemoryExtension.id(), {'enabled': on})
  )
  return context.restart_worker is not None and await context.restart_worker({
    'memory_dir': str(STORE) if on else None
  })


async def _import(
  context: extension.Context, arguments: tuple[str, ...]
) -> None:
  """Run `mirror-agent memory import` beside the session and show its report."""
  if context.worker_cwd is None:
    raise RuntimeError('Memory worker directory is unavailable')
  # Personal memory belongs to the host, not the workspace execution backend.
  result = await shell.Shell(context.worker_cwd).run(
    (
      *context.worker_command,
      'memory',
      'import',
      '--store',
      str(STORE),
      *arguments,
    ),
    timeout=300,
  )
  text = (result.stdout + result.stderr).decode(errors='replace').rstrip()
  if result.returncode != 0:
    context.dispatch(
      actions.ShowText(
        text or f'mirror-agent memory import exited with {result.returncode}',
        style='ui.error',
      )
    )
    return
  headline, *details = text.splitlines() or ['']
  _show(context, headline, details)
