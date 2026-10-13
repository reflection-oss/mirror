"""Provider-reported model throughput for the Mirror footer."""

from __future__ import annotations

import dataclasses
from typing import override

from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers, controls

from mirror.cli import actions, effects, extension, state

_ALPHA = 0.3
_HISTORY_SIZE = 7
_MIN_SAMPLE_SECONDS = 1.0
_SPARKLINE_GLYPHS = '▁▂▃▄▅▆▇█'


@dataclasses.dataclass(frozen=True)
class _State:
  history: tuple[float, ...] = ()
  first_tokens: int = 0
  first_elapsed: float | None = None
  live_rate: float | None = None


@dataclasses.dataclass(frozen=True)
class _UsageObserved(actions.ExtensionAction):
  tokens: int
  elapsed: float


@dataclasses.dataclass(frozen=True)
class _StreamFinished(actions.ExtensionAction):
  tokens_per_second: float | None


@dataclasses.dataclass(frozen=True)
class _StreamAborted(actions.ExtensionAction):
  pass


def _sparkline(history: tuple[float, ...]) -> str:
  if not history:
    return ''
  high = max(history)
  if high <= 0:
    return _SPARKLINE_GLYPHS[0] * len(history)
  scale = (len(_SPARKLINE_GLYPHS) - 1) / high
  return ''.join(_SPARKLINE_GLYPHS[int(value * scale)] for value in history)


def _average(current: _State, sample: float) -> float:
  return (
    sample
    if not current.history
    else _ALPHA * sample + (1 - _ALPHA) * current.history[-1]
  )


class TokensPerSecondExtension(extension.Extension[_State, None, None]):
  """Render throughput using provider tokens and worker-side timing only."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/tokens-per-second'

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_RIGHT

  @staticmethod
  @override
  def initial_state() -> _State:
    return _State()

  @override
  def handle_message(
    self, context: extension.Context, message: agent_pb2.SessionMessage
  ) -> None:
    if message.stream_final or message.HasField('tokens_per_second'):
      self.dispatch(
        context,
        _StreamFinished(
          message.tokens_per_second
          if message.HasField('tokens_per_second')
          else None,
        ),
      )
    elif message.WhichOneof('payload') == 'runtime_delta':
      delta = message.runtime_delta
      if delta.WhichOneof('payload') == 'stream_usage':
        usage = delta.stream_usage
        self.dispatch(
          context,
          _UsageObserved(usage.completion_tokens, usage.elapsed_seconds),
        )
      elif delta.WhichOneof('payload') == 'stream_aborted':
        self.dispatch(context, _StreamAborted())

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[_State, tuple[effects.CoreEffect, ...]]:
    current = self.require_state(global_state)
    if isinstance(action, actions.CloseStream):
      return _State(history=current.history), ()
    if not isinstance(action, actions.RoutedExtensionAction):
      return current, ()
    match action.action:
      case _UsageObserved(tokens, elapsed):
        if tokens == 0:
          return current, ()
        if current.first_elapsed is None:
          return dataclasses.replace(
            current, first_tokens=tokens, first_elapsed=elapsed
          ), ()
        duration = elapsed - current.first_elapsed
        if duration < _MIN_SAMPLE_SECONDS:
          return current, ()
        # Exclude both the first batch and its unknown generation time.
        rate = (tokens - current.first_tokens) / duration
        return dataclasses.replace(current, live_rate=rate), ()
      case _StreamFinished(fallback):
        sample = (
          current.live_rate if current.live_rate is not None else fallback
        )
        if sample is None:
          return _State(history=current.history), ()
        history = (*current.history, _average(current, sample))[-_HISTORY_SIZE:]
        return _State(history=history), ()
      case _StreamAborted():
        return _State(history=current.history), ()
      case _:
        return current, ()

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    if global_state.session.pending_switch is not None:
      return None
    history = extension_state.history
    if extension_state.live_rate is not None:
      history = (
        *history,
        _average(extension_state, extension_state.live_rate),
      )[-_HISTORY_SIZE:]
    average = history[-1] if history else None
    value = '  -' if average is None else f'{min(average, 999):>3.0f}'
    sparkline = _sparkline(history).rjust(_HISTORY_SIZE)
    return containers.Window(
      controls.FormattedTextControl([
        ('class:footer', f'{sparkline} {value} tok/s')
      ]),
      style='class:footer',
      dont_extend_height=True,
      dont_extend_width=True,
    )
