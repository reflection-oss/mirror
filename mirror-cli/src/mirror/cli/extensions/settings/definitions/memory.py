"""Whether local sessions on this machine use the user's personal memory."""

from __future__ import annotations

from mirror.cli import actions, extension
from mirror.cli.extensions.memory import extension as memory_extension
from mirror.cli.extensions.settings import setting_template


def _available(context: extension.Context) -> bool:
  return (
    context.restart_worker is not None
    and not context.get_state().activity.busy
    and memory_extension.MemoryExtension.id() in context.get_state().extensions
  )


def _current(context: extension.Context) -> str:
  enabled = memory_extension.enabled(
    setting_template.user_config(context).read()
  )
  return setting_template.ON if enabled else setting_template.OFF


async def _apply(context: extension.Context, value: str) -> None:
  if context.get_state().activity.busy:
    raise ValueError(
      'Finish or interrupt the current turn before changing memory.'
    )
  await memory_extension.set_enabled(context, value == setting_template.ON)


MENU = setting_template.Menu(
  'memory',
  'Memory',
  'Configure memory and import from other harnesses',
  (
    setting_template.Setting(
      'memory',
      'Memory',
      'Remember context across sessions; changing it restarts the session',
      setting_template.ON_OFF,
      _current,
      _apply,
    ),
    setting_template.Action(
      'memory-import',
      'Import memories',
      'Import from Claude Code and Codex on this machine',
      actions.RoutedExtensionAction(
        memory_extension.MemoryExtension.id(),
        memory_extension.Run(('import',)),
      ),
    ),
  ),
  available=_available,
)
