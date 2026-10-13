"""One delayed tip per user turn, below the working indicator."""

import dataclasses
import random
import time
from typing import override

from prompt_toolkit.layout import containers, controls

from mirror import config
from mirror.cli import actions, commands, extension, state
from mirror.cli.extensions.settings import setting_template

CORE_TIPS = (
  'Ctrl+L redraws the screen',
  '/worktree opens a new workspace on a fresh Git worktree',
  '/folder <path> opens an existing folder as a workspace',
  'Use /extensions to turn extensions on and off',
  'Use Tab and Shift+Tab to switch between workspaces',
  'Ctrl+R searches your past prompts',
  'Ctrl+J adds a new line without sending',
  '/workspaces finds and switches between open workspaces',
  '/resume reopens an earlier conversation when idle',
  'Use /new to start a fresh conversation',
  'When idle, /rewind branches from a past turn',
  '/close closes an idle workspace without deleting it',
  '/tag gives a workspace an icon',
  'Use /rename to rename a workspace',
  'Type / to browse commands',
  'Type $ to browse skills and add one to your prompt',
  'Press Enter while Mirror works to queue a follow-up',
  '/compact summarizes older context when Mirror is idle',
  '/extensions-add adds a GitHub extension',
  'Use /tips to disable this widget',
)


@dataclasses.dataclass(frozen=True)
class _State:
  tip: str | None = None
  started_at: float | None = None
  user_message_seen: bool = False


@dataclasses.dataclass(frozen=True)
class _SetVisibility(actions.ExtensionAction):
  visible: bool | None


def _tips(context: commands.CommandContext, args: str) -> actions.Action:
  del context  # Unused.
  token = args.strip().lower()
  if token not in ('', 'enable', 'disable'):
    return actions.ShowText('Usage: /tips [enable|disable]', style='ui.warning')
  return actions.RoutedExtensionAction(
    TipsExtension.id(), _SetVisibility(token == 'enable' if token else None)
  )


def _turn_started(
  global_state: state.State, current: _State, action: actions.Action
) -> float | None:
  """Recognize local, remote, and queued user turns without counting tool steps."""
  if global_state.session.replaying:
    return None
  if isinstance(action, actions.TurnStarted):
    return action.observed_at
  if (
    isinstance(action, actions.AgentTurnStarted) and current.started_at is None
  ):
    return action.observed_at
  if isinstance(action, actions.UserMessageObserved) and action.content.strip():
    # The first user message can acknowledge a turn already started locally.
    if current.started_at is None or current.user_message_seen:
      return time.monotonic()
  return None


class TipsExtension(extension.Extension[_State, _SetVisibility, None]):
  """Keep a random tip stable once a user turn has been working for 15 seconds."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/tips'

  @staticmethod
  @override
  def initial_state() -> _State:
    return _State()

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_BELOW_ACTIVITY

  @override
  def tips(self) -> tuple[str, ...]:
    return CORE_TIPS

  @override
  def default_config(self) -> config.JsonObject:
    # Keep the extension loaded so /tips enable can restore a hidden bar.
    return {'visible': True}

  @override
  def slash_commands(self) -> tuple[commands.CommandDef, ...]:
    return (
      commands.CommandDef(
        'tips',
        'Show or hide tips',
        _tips,
        arguments=commands.OptionalArguments('[enable|disable]'),
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[_State, tuple[extension.PrivateEffect[_SetVisibility], ...]]:
    current = self.require_state(global_state)
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, _SetVisibility)
    ):
      return current, (self.effect(action.action),)
    if isinstance(action, actions.TurnFinished):
      return dataclasses.replace(current, started_at=None), ()
    started_at = _turn_started(global_state, current, action)
    if started_at is not None:
      candidates = (
        tuple(tip for tip in global_state.extension_tips if tip != current.tip)
        or global_state.extension_tips
      )
      tip = random.choice(candidates) if candidates else None
      return _State(
        tip, started_at, isinstance(action, actions.UserMessageObserved)
      ), ()
    if (
      isinstance(action, actions.UserMessageObserved)
      and action.content.strip()
      and not global_state.session.replaying
    ):
      return dataclasses.replace(current, user_message_seen=True), ()
    return current, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _SetVisibility
  ) -> None:
    visible = effect.visible
    if visible is None:
      visible = self.configuration(context.get_state()).get('visible') is False
    settings: config.JsonObject = {'visible': visible}
    try:
      setting_template.user_config(context).set_extension_config(
        'builtin/tips', settings
      )
    except (OSError, ValueError) as error:
      context.dispatch(
        actions.ShowText(
          f'Could not save tips preference: {error}', style='ui.error'
        )
      )
      return
    context.dispatch(actions.ExtensionConfigChanged(self.id(), settings))
    context.dispatch(
      actions.ShowText(
        f'Tips {"enabled" if visible else "disabled"}', style='ui.muted'
      )
    )

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    if (
      self.configuration(global_state).get('visible') is False
      or not global_state.activity.busy
      or global_state.session.replaying
      or extension_state.started_at is None
      or time.monotonic() - extension_state.started_at < 15
      or extension_state.tip not in global_state.extension_tips
    ):
      return None
    return containers.Window(
      controls.FormattedTextControl(f'  └ Tip: {extension_state.tip}'),
      height=1,
      wrap_lines=False,
      style='class:ui.muted',
    )
