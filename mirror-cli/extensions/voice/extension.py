"""Optional local dictation: hold Space; first use downloads Whisper base."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import time
from collections.abc import Sequence
from typing import Literal, override

from mirror.cli.extensions.voice import process as voice_process
from mirror.cli.extensions.voice import space_hold
from prompt_toolkit import filters, formatted_text
from prompt_toolkit.key_binding.bindings import basic
from prompt_toolkit.layout import containers, controls

from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  keymap,
  selection,
  state,
)

type _Phase = Literal['idle', 'preparing', 'recording', 'transcribing']
_WAVEFORM = '▁▂▃▄▅▆▇█'
_HISTORY_SIZE = 12


@dataclasses.dataclass(frozen=True)
class Toggle(actions.ExtensionAction):
  """Start dictation, stop recording, or cancel preparation/transcription."""


@dataclasses.dataclass(frozen=True)
class _Setup(actions.ExtensionAction):
  pass


@dataclasses.dataclass(frozen=True)
class SpacePressed(actions.ExtensionAction):
  """Type a space or refresh the push-to-talk gesture."""


@dataclasses.dataclass(frozen=True)
class _FlushSpace(actions.ExtensionAction):
  pass


@dataclasses.dataclass(frozen=True)
class _WaitForSpace:
  started: float


@dataclasses.dataclass(frozen=True)
class _Released(actions.ExtensionAction):
  pass


type _Effect = (
  Literal['toggle', 'start', 'cancel', 'hold', 'prepare'] | _WaitForSpace
)


@dataclasses.dataclass(frozen=True)
class _State:
  phase: _Phase = 'idle'
  levels: tuple[float, ...] = ()
  elapsed: float = 0
  progress: int = 0
  held: bool = False
  space: space_hold.SpaceHold | None = None


@dataclasses.dataclass(frozen=True)
class _Status(actions.ExtensionAction):
  phase: _Phase


@dataclasses.dataclass(frozen=True)
class _Recording(actions.ExtensionAction):
  level: float
  elapsed: float


@dataclasses.dataclass(frozen=True)
class _Progress(actions.ExtensionAction):
  percent: int


@dataclasses.dataclass(frozen=True)
class _Transcript(actions.ExtensionAction):
  text: str


def _message(line: bytes) -> _Status | _Recording | _Progress | _Transcript:
  """Validate telemetry from the isolated audio helper."""
  value: object = json.loads(line)
  match value:
    case {'status': 'recording'}:
      return _Status('recording')
    case {'status': 'transcribing'}:
      return _Status('transcribing')
    case {
      'level': (int() | float()) as level,
      'elapsed': (int() | float()) as elapsed,
    } if 0 <= level <= 1 and elapsed >= 0 and math.isfinite(elapsed):
      return _Recording(level, elapsed)
    case {'progress': int() as percent} if 0 <= percent <= 100:
      return _Progress(percent)
    case {'text': str() as text}:
      return _Transcript(text)
    case _:
      raise ValueError('Invalid voice helper response')


def _toggle(context: commands.CommandContext, arguments: str) -> actions.Action:
  del context, arguments
  return actions.RoutedExtensionAction(Voice.id(), Toggle())


def _consent(choice: str) -> actions.Action:
  if choice == 'prepare':
    return actions.RoutedExtensionAction(Voice.id(), _Setup())
  return actions.RerenderRequested()


class Voice(extension.Extension[_State, _Effect, None]):
  """Keep audio and transcription on the CLI host, even for remote agents."""

  def __init__(self) -> None:
    self._approved = False
    self._task: asyncio.Task[None] | None = None
    self._process: asyncio.subprocess.Process | None = None
    self._context: extension.Context | None = None
    self._recording = False
    self._foreground: extension.Context | None = None

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/voice'

  @override
  def default_enabled(self) -> bool:
    return False

  @staticmethod
  @override
  def initial_state() -> _State:
    return _State()

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'voice',
        'Set up or toggle local voice dictation',
        _toggle,
        available_when_busy=True,
      ),
    )

  @override
  def keybindings(self) -> Sequence[keymap.Binding]:
    return (
      keymap.Binding(
        ('c-b',), actions.RoutedExtensionAction(self.id(), Toggle())
      ),
      keymap.Binding(
        (' ',),
        actions.RoutedExtensionAction(self.id(), SpacePressed()),
        filter=(
          filters.has_focus('composer')
          & ~filters.is_read_only
          & ~basic.in_quoted_insert
        ),
        eager=True,
      ),
      keymap.Binding(
        ('<any>',),
        actions.RoutedExtensionAction(self.id(), _FlushSpace()),
        filter=filters.Condition(self._space_pending),
        replay=True,
      ),
    )

  def _space_pending(self) -> bool:
    if self._foreground is None:
      return False
    current = self.current_state(self._foreground)
    return current.space is not None and not current.held

  @override
  async def on_foreground(self, context: extension.Context) -> None:
    self._foreground = context

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_LEFT

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    if global_state.session.pending_switch is not None:
      return None
    fragments: formatted_text.StyleAndTextTuples = [('class:footer', '🎙')]
    match extension_state.phase:
      case 'recording':
        # Map -60..0 dBFS to glyph heights so ordinary speech is visible.
        wave = ''.join(
          _WAVEFORM[int((20 * math.log10(max(level, 0.001)) + 60) / 60 * 7)]
          for level in extension_state.levels
        ).rjust(_HISTORY_SIZE, _WAVEFORM[0])
        minutes, seconds = divmod(int(extension_state.elapsed), 60)
        fragments = [
          ('class:ui.error', '🎙'),
          ('class:ui.accent', f' {wave}'),
          ('class:footer', f' {minutes:02}:{seconds:02}'),
        ]
      case 'transcribing':
        fragments.append(('class:footer', f' {extension_state.progress}%'))
      case 'preparing':
        fragments.append(('class:footer', ' …'))
      case 'idle':
        pass
    return containers.Window(
      controls.FormattedTextControl(fragments),
      style='class:footer',
      dont_extend_height=True,
      dont_extend_width=True,
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    _State,
    tuple[
      effects.CoreEffect | extension.PrivateEffect[_Effect],
      ...,
    ],
  ]:
    current = self.require_state(global_state)
    if isinstance(action, (actions.InputSubmitted, actions.SetPendingSwitch)):
      return _State(), (self.effect('cancel'),)
    if not isinstance(action, actions.RoutedExtensionAction):
      return current, ()
    blocked = (
      global_state.selector.active
      or global_state.secret.request is not None
      or global_state.resume.active
      or global_state.rewind.active
      or global_state.approval.request is not None
      or global_state.drop_confirmation is not None
      or global_state.session.pending_switch is not None
    )
    if (
      isinstance(action.action, (SpacePressed, Toggle, _Setup))
      and current.phase == 'idle'
      and blocked
    ):
      return current, ()
    match action.action:
      case SpacePressed():
        now = time.monotonic()
        space = current.space
        if current.held and space is not None:
          return dataclasses.replace(current, space=space.press(now)), ()
        if current.phase != 'idle':
          return current, (effects.InsertComposerText(' '),)
        if space is None or space.original != global_state.composer:
          space = space_hold.SpaceHold(global_state.composer, now, now)
          return dataclasses.replace(current, space=space), (
            self.effect(_WaitForSpace(now)),
          )
        space = space.press(now)
        if space.confirmed:
          return _State('preparing', held=True, space=space), (
            self.effect('hold'),
          )
        return dataclasses.replace(current, space=space), (
          self.effect(_WaitForSpace(space.started)),
        )
      case _FlushSpace():
        space = current.space
        if space is None or current.held:
          return current, ()
        return dataclasses.replace(current, space=None), (
          (effects.InsertComposerText(' ' * space.count),)
          if not blocked and space.original == global_state.composer
          else ()
        )
      case _Released():
        return dataclasses.replace(current, held=False, space=None), ()
      case _Status(phase):
        held = current.held and phase != 'idle'
        return _State(
          phase, held=held, space=current.space if held else None
        ), ()
      case _Recording(level, elapsed) if current.phase == 'recording':
        return dataclasses.replace(
          current,
          levels=(*current.levels, level)[-_HISTORY_SIZE:],
          elapsed=elapsed,
        ), ()
      case _Progress(percent) if current.phase == 'transcribing':
        return dataclasses.replace(current, progress=percent), ()
      case _Transcript(text):
        return _State(), (effects.InsertComposerText(text),)
      case Toggle() | _Setup():
        return current, (
          self.effect(
            'prepare' if isinstance(action.action, _Setup) else 'toggle'
          ),
        )
      case _:
        return current, ()

  @override
  async def on_background(self, context: extension.Context) -> None:
    if (
      self._foreground is not None
      and self._foreground.dispatch == context.dispatch
    ):
      self._foreground = None
    if self._context is None or self._context.dispatch == context.dispatch:
      self.dispatch(context, _Status('idle'))
      await self._cancel()

  @override
  async def on_unload(self, context: None) -> None:
    await self._cancel()

  async def _cancel(self) -> None:
    if self._task is not None:
      self._task.cancel()
      await asyncio.gather(self._task, return_exceptions=True)

  async def _wait_for_space(
    self, context: extension.Context, started: float
  ) -> None:
    while True:
      current = self.current_state(context)
      space = current.space
      if space is None or space.started != started or current.held:
        return
      delay = space.deadline - time.monotonic()
      if delay <= 0:
        self.dispatch(context, _FlushSpace())
        return
      await asyncio.sleep(delay)

  def _stop_recording(self, context: extension.Context) -> None:
    assert self._process is not None and self._process.stdin is not None
    self._recording = False
    self._process.stdin.write(b'\n')
    self.dispatch(context, _Status('transcribing'))

  async def _hold(self, context: extension.Context) -> None:
    gesture = self.current_state(context).space
    if gesture is None:
      return
    recording = asyncio.create_task(self.handle_effect(context, 'start'))
    try:
      while not recording.done():
        space = self.current_state(context).space
        if space is None or space.started != gesture.started:
          recording.cancel()
          break
        delay = space.last + space_hold.RELEASE_SECONDS - time.monotonic()
        if delay <= 0:
          self.dispatch(context, _Released())
          if self._recording:
            self._stop_recording(context)
          else:
            recording.cancel()
          break
        await asyncio.sleep(delay)
      await recording
    finally:
      if not recording.done():
        recording.cancel()
      await asyncio.gather(recording, return_exceptions=True)
      space = self.current_state(context).space
      if space is not None and space.started == gesture.started:
        self.dispatch(context, _Released())

  def _request_setup(self, context: extension.Context) -> None:
    self.dispatch(context, _Status('idle'))
    context.dispatch(
      actions.OpenSelector(
        selection.Selector(
          'voice-consent',
          'Local voice: install and run locked audio dependencies and download Whisper base (~150 MB)? Requires uv; audio stays on this machine.',
          (
            selection.Choice('cancel', 'Not now'),
            selection.Choice(
              'prepare', 'Install dependencies and enable voice'
            ),
          ),
          _consent,
          extension_id=self.id(),
          selected='cancel',
          numbered=True,
        )
      )
    )

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _Effect
  ) -> None:
    if effect == 'cancel':
      self.dispatch(context, _Status('idle'))
      await self._cancel()
      return
    if not context.get_state().lifecycle.active:
      return
    if isinstance(effect, _WaitForSpace):
      await self._wait_for_space(context, effect.started)
      return
    if self._task is not None:
      if self._recording:
        self._stop_recording(context)
      else:
        await self._cancel()
      return
    if effect in ('toggle', 'hold') and not self._approved:
      self._request_setup(context)
      return
    if effect == 'hold':
      await self._hold(context)
      return
    self._context = context
    self.dispatch(context, _Status('preparing'))
    self._task = asyncio.create_task(
      self._prepare() if effect == 'prepare' else self._capture(context)
    )
    try:
      await self._task
    except TimeoutError:
      self._approved = False
      context.dispatch(
        actions.ShowText(
          'Voice setup timed out; retry /voice.', style='ui.error'
        )
      )
    except (OSError, RuntimeError, ValueError) as error:
      self._approved = False
      context.dispatch(
        actions.ShowText(f'Voice dictation failed: {error}', style='ui.error')
      )
    finally:
      self._task = None
      self._context = None
      self._process = None
      self._recording = False
      self.dispatch(context, _Status('idle'))

  async def _prepare(self) -> None:
    async with asyncio.timeout(600):
      async with voice_process.worker('--prepare') as process:
        assert process.stdout is not None
        await process.stdout.read()
    self._approved = True

  async def _capture(self, context: extension.Context) -> None:
    conversation = context.get_state().session.conversation_id
    text: str | None = None
    async with voice_process.worker() as process:
      self._process = process
      assert process.stdout is not None
      async for line in process.stdout:
        message = _message(line)
        if isinstance(message, _Transcript):
          text = message.text
        else:
          if isinstance(message, _Status):
            self._recording = message.phase == 'recording'
          self.dispatch(context, message)
    current = context.get_state()
    if (
      current.session.conversation_id != conversation
      or not current.lifecycle.active
      or current.session.pending_switch is not None
    ):
      return
    if text is None:
      raise ValueError('Voice helper exited without a transcript')
    if text:
      self.dispatch(context, _Transcript(text))
    else:
      context.dispatch(
        actions.ShowText('No speech detected.', style='ui.warning')
      )
