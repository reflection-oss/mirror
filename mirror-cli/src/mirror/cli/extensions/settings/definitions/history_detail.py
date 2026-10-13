"""How much of each turn the conversation shows in one workspace."""

from __future__ import annotations

from mirror.cli import actions, extension
from mirror.cli.extensions.settings import setting_template


def _current(context: extension.Context) -> str:
  return context.get_state().status_bar.view_mode.value


async def _apply(context: extension.Context, value: str) -> None:
  context.dispatch(actions.SetViewMode(actions.ViewMode(value)))


SETTING = setting_template.Setting(
  'history-detail',
  'Conversation detail',
  'How much of each turn to show in this workspace.',
  (
    setting_template.Option(actions.ViewMode.TRANSCRIPT.value, 'Full'),
    setting_template.Option(actions.ViewMode.COMPACT.value, 'Compact'),
    setting_template.Option(actions.ViewMode.EXTRA_COMPACT.value, 'Minimal'),
  ),
  _current,
  _apply,
  available=setting_template.local_idle,
)
