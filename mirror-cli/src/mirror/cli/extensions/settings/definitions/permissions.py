"""Tool approval mode toggled inline in Settings."""

from __future__ import annotations

from mirror.cli import actions, extension
from mirror.cli.extensions.settings import setting_template


def _current(context: extension.Context) -> str:
  return context.get_state().status_bar.approval_mode


async def _apply(context: extension.Context, value: str) -> None:
  context.dispatch(actions.SetApprovalMode(value))


SETTING = setting_template.Setting(
  'permissions',
  'CLI mode',
  'Choose automatic, manual or bypass tool approval',
  (
    setting_template.Option('auto', 'Auto'),
    setting_template.Option('approve', 'Manual'),
    setting_template.Option('bypass', 'Bypass'),
  ),
  _current,
  _apply,
)
