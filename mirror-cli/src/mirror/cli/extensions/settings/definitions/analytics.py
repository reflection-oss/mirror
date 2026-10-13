"""Whether Mirror shares anonymous usage analytics with Reflection."""

from __future__ import annotations

from mirror import config
from mirror.cli import extension
from mirror.cli.extensions.settings import setting_template


def _current(context: extension.Context) -> str:
  enabled = config.analytics_enabled(
    setting_template.user_config(context).read()
  )
  return setting_template.ON if enabled else setting_template.OFF


async def _apply(context: extension.Context, value: str) -> None:
  def change(values: dict[str, object]) -> None:
    values['analytics'] = value == setting_template.ON

  setting_template.user_config(context).update(change)


SETTING = setting_template.Setting(
  'analytics',
  'Usage analytics',
  'Share anonymous product usage with Reflection.',
  setting_template.ON_OFF,
  _current,
  _apply,
  available=setting_template.local_idle,
)
