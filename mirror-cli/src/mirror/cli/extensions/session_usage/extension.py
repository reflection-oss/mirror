"""On-demand provider usage from the session's recorded history, not context size."""

from __future__ import annotations

import dataclasses
import datetime
import itertools
from collections.abc import Sequence
from typing import override

from mirror.protocol import agent_pb2
from rich import text as rich_text

from mirror.cli import actions, commands, effects, extension, state


@dataclasses.dataclass(frozen=True)
class _ShowUsage(actions.ExtensionAction):
  pass


def report(records: Sequence[agent_pb2.AgentMessage]) -> rich_text.Text:
  """Summarize the canonical transcript without recounting compaction copies."""
  assistants = [
    record for record in records if record.chat.HasField('assistant')
  ]
  compactions = [
    record.context_window.compaction
    for record in records
    if record.context_window.HasField('compaction')
  ]
  reported: dict[str, list[tuple[agent_pb2.Usage, str]]] = {}
  for record in assistants:
    if record.chat.HasField('usage'):
      model = record.inference.model or 'Unknown model'
      reported.setdefault(model, []).append((
        record.chat.usage,
        record.inference.api,
      ))
  for compaction in compactions:
    if compaction.HasField('usage'):
      model = compaction.model or compaction.inference.model or 'Unknown model'
      reported.setdefault(model, []).append((
        compaction.usage,
        compaction.inference.api,
      ))
  rows = [('Recorded session usage', 'this agent')]
  timestamps = [
    record.timestamp.seconds
    for record in records
    if record.HasField('timestamp')
  ]
  if len(timestamps) >= 2:
    span = datetime.timedelta(seconds=max(timestamps) - min(timestamps))
    rows.append(('  Recorded span (wall)', str(span)))
  rows.append((
    '  Responses with usage',
    f'{sum(record.chat.HasField("usage") for record in assistants):,} / {len(assistants):,}',
  ))
  if compactions:
    rows.append((
      '  Compactions with usage',
      f'{sum(entry.HasField("usage") for entry in compactions):,} / {len(compactions):,}',
    ))
  for model, usages in sorted(reported.items()):
    rows.extend(_model_rows(model, usages))
  if not reported:
    rows.append(('  Tokens', 'not recorded'))
  return _render(rows)


def _render(rows: Sequence[tuple[str, str]]) -> rich_text.Text:
  """Keep the report's presentation inside the extension boundary."""
  labels = [
    ''.join(filter(str.isprintable, rich_text.Text.from_ansi(label).plain))
    for label, _ in rows
  ]
  width = max(36, max(map(len, labels), default=0))
  text = rich_text.Text()
  for index, (label, (_, value)) in enumerate(zip(labels, rows, strict=True)):
    if not label:
      text.append('\n')
      continue
    heading = not label.startswith(' ')
    marker = '◆ ' if index == 0 else '▸ ' if heading else '  '
    text.append(
      f'{marker}{label:<{width}}  ', style='ui.info.bold' if heading else 'dim'
    )
    text.append(f'{value:>12}\n', style='ui.info.bold' if heading else '')
  return text


def _model_rows(
  model: str, requests: Sequence[tuple[agent_pb2.Usage, str]]
) -> list[tuple[str, str]]:
  usages = [usage for usage, _ in requests]
  rows = [
    ('', ''),
    (model, f'{len(usages):,} request{"s" if len(usages) != 1 else ""}'),
    ('  Prompt', f'{sum(usage.prompt_tokens for usage in usages):,}'),
    ('  Output', f'{sum(usage.completion_tokens for usage in usages):,}'),
    ('  Cache read', f'{sum(usage.cached_tokens for usage in usages):,}'),
    ('  Cache write', f'{sum(usage.cache_write_tokens for usage in usages):,}'),
    (
      '  Requests with cache reads',
      f'{sum(usage.cached_tokens > 0 for usage in usages):,} / {len(usages):,}',
    ),
  ]
  inputs = list(itertools.starmap(_input_tokens, requests))
  total_input = sum(tokens for tokens in inputs if tokens is not None)
  if None not in inputs and total_input:
    cached = sum(usage.cached_tokens for usage in usages)
    rows.append((
      '  Reported cache share',
      f'{cached / total_input:.1%} of input',
    ))
  return rows


def _input_tokens(usage: agent_pb2.Usage, api: str) -> int | None:
  """Normalize only known APIs; missing metadata cannot identify cache semantics."""
  if api in {'anthropic-messages', 'anthropic'}:
    # Anthropic's prompt count excludes both cache reads and cache writes.
    return usage.prompt_tokens + usage.cached_tokens + usage.cache_write_tokens
  if (
    api
    in {
      'reflection',
      'gemini',
      'vercel',
      'openai-chat-completions',
      'openai-chat-completions-non-streaming',
      'openrouter',
      'openai-responses',
      'openai-codex-responses',
    }
    and usage.cached_tokens <= usage.prompt_tokens
  ):
    return usage.prompt_tokens
  return None


class SessionUsageExtension(extension.Extension[None, _ShowUsage, None]):
  """Project saved usage only when requested; keep no accounting state."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/session-usage'

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'usage',
        'Show recorded session tokens, cache reads, and elapsed span',
        lambda _context, _args: actions.RoutedExtensionAction(
          self.id(), _ShowUsage()
        ),
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    None, tuple[effects.CoreEffect | extension.PrivateEffect[_ShowUsage], ...]
  ]:
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, _ShowUsage)
    ):
      if (
        global_state.session.replaying
        or global_state.session.pending_switch is not None
      ):
        return None, (
          effects.EmitText('Session history is still loading; retry /usage.'),
        )
      return None, (self.effect(action.action),)
    return None, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _ShowUsage
  ) -> None:
    del effect  # Unused.
    context.dispatch(actions.ShowText(report(context.transcript())))
