"""Model context-window usage for the Mirror footer."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import override

from mirror.protocol import agent_pb2
from prompt_toolkit.layout import containers, controls
from rich import text as rich_text

from mirror.cli import actions, commands, effects, extension, rendering, state


@dataclasses.dataclass(frozen=True)
class ContextUsage:
  """An immutable model context-window measurement."""

  tokens: int = 0
  window: int | None = None
  prompt_report: agent_pb2.SystemPromptReport | None = None
  prompt_record_id: str = ''

  def render(self) -> str:
    """Render compact context usage and the known window size."""
    percent = 100 * self.tokens / self.window if self.window else 0
    label = f'{min(percent, 999):.0f}% ctx'
    if self.window is not None:
      label = f'{label} · {rendering.format_window(self.window)}'
    return label


@dataclasses.dataclass(frozen=True)
class UsageObserved(actions.ExtensionAction):
  """One model prompt's context-window usage."""

  tokens: int
  window: int


@dataclasses.dataclass(frozen=True)
class PromptObserved(actions.ExtensionAction):
  report: agent_pb2.SystemPromptReport
  record_id: str = ''


@dataclasses.dataclass(frozen=True)
class _ShowContext(actions.ExtensionAction):
  pass


def _report_text(
  report: agent_pb2.SystemPromptReport | None,
) -> str | rich_text.Text:
  if report is None:
    return 'System prompt: not measured yet.'
  tool_tokens = report.tool_tokens if report.HasField('tool_tokens') else None
  total = (
    report.estimated_tokens + tool_tokens if tool_tokens is not None else None
  )
  rows: list[tuple[str, int | None]] = [
    ('Context', total),
    ('  System prompt', report.estimated_tokens),
    ('  Tools', tool_tokens),
  ]
  sections: dict[str, dict[str, int]] = {}
  home = str(pathlib.Path.home()) + '/'
  for part in report.parts:
    source = part.source
    section = (
      'Skills'
      if source.endswith('/SKILL.md') or source == 'Skills framing'
      else 'Instructions'
      if source.endswith(('/AGENTS.md', '/CLAUDE.md'))
      or source
      in {'Project instructions framing', 'Supplied/resolved instructions'}
      else 'Other prompt parts'
    )
    label = '~/' + source[len(home) :] if source.startswith(home) else source
    if source.endswith(' framing'):
      label = 'Framing'
    entries = sections.setdefault(section, {})
    entries[label] = entries.get(label, 0) + part.estimated_tokens
  for section, entries in sections.items():
    rows.extend([('', None), (section, sum(entries.values()))])
    rows.extend((f'  {source}', tokens) for source, tokens in entries.items())
  if tool_tokens is not None:
    rows.extend([('', None), ('Tools', tool_tokens)])
    groups: dict[str, list[tuple[str, int]]] = {}
    for tool in report.tools:
      parts = tool.source.split('__', 2)
      if len(parts) == 3 and parts[0] == 'mcp':
        group, name = f'MCP · {parts[1]}', parts[2]
      elif len(parts) > 1:
        group, name = f'Extension · {parts[0]}', '__'.join(parts[1:])
      else:
        group, name = 'Native', tool.source
      groups.setdefault(group, []).append((name, tool.estimated_tokens))
    for group, tools in sorted(groups.items()):
      rows.append((f'  {group}', sum(tokens for _, tokens in tools)))
      rows.extend(
        (f'    {name}', tokens)
        for name, tokens in sorted(
          tools, key=lambda item: item[1], reverse=True
        )
      )
  labels = [
    rendering.strip_ansi(label).replace('\n', ' ').replace('\r', ' ')
    for label, _ in rows
  ]
  width = max(36, max(len(label) for label in labels))
  text = rich_text.Text()
  for label, (_, tokens) in zip(labels, rows, strict=True):
    if not label:
      text.append('\n')
      continue
    heading = not label.startswith(' ')
    marker = '◆ ' if label == 'Context' else '▸ ' if heading else '  '
    text.append(
      f'{marker}{label:<{width}}  ', style='ui.info.bold' if heading else 'dim'
    )
    count = f'~{tokens:,}' if tokens is not None else 'not measured'
    text.append(f'{count:>12}\n', style='ui.info.bold' if heading else '')
  text.append(
    '\nApproximate tokens · saved prompt + current tools · excludes conversation history',
    style='dim',
  )
  return text


class ContextUsageExtension(extension.Extension[ContextUsage, None, None]):
  """Observe and render model context-window usage."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/context-usage'

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_RIGHT

  @override
  def render_order(self) -> int:
    return 1

  @staticmethod
  @override
  def initial_state() -> ContextUsage:
    return ContextUsage()

  @override
  def handle_message(
    self, context: extension.Context, message: agent_pb2.SessionMessage
  ) -> None:
    if message.HasField('agent_message') and message.agent_message.HasField(
      'context_window'
    ):
      record = message.agent_message
      window = record.context_window
      current = self.require_state(context.get_state())
      if (
        window.system_prompt
        and current.prompt_report is not None
        and current.prompt_record_id == record.id
      ):
        return  # A replay must not replace this window's refreshed tool catalog.
      if window.HasField('system_prompt_report'):
        self.dispatch(
          context, PromptObserved(window.system_prompt_report, record.id)
        )
      return
    if message.WhichOneof('payload') != 'runtime_delta':
      return
    delta = message.runtime_delta
    if delta.WhichOneof('payload') != 'context_usage':
      return
    usage = delta.context_usage
    if usage.window > 0:
      self.dispatch(context, UsageObserved(usage.tokens, usage.window))

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'context',
        'Show the latest system prompt construction and its sources',
        lambda _context, _args: actions.RoutedExtensionAction(
          self.id(), _ShowContext()
        ),
        available_when_busy=True,
      ),
    )

  @override
  def hydrate(
    self,
    snapshot: agent_pb2.ExtensionRecord | None,
    records: Sequence[agent_pb2.AgentMessage],
    global_state: state.State,
  ) -> ContextUsage:
    current = self.require_state(global_state)
    for record in reversed(records):
      if record.HasField('context_window'):
        if (
          current.prompt_report is not None
          and current.prompt_record_id == record.id
        ):
          return current
        report = (
          record.context_window.system_prompt_report
          if record.context_window.HasField('system_prompt_report')
          else None
        )
        return dataclasses.replace(
          current, prompt_report=report, prompt_record_id=record.id
        )
    return current

  @override
  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[ContextUsage, tuple[effects.CoreEffect, ...]]:
    current = self.require_state(global_state)
    if isinstance(action, actions.RoutedExtensionAction) and isinstance(
      action.action, UsageObserved
    ):
      return dataclasses.replace(
        current, tokens=action.action.tokens, window=action.action.window
      ), ()
    if isinstance(action, actions.RoutedExtensionAction):
      if isinstance(action.action, PromptObserved):
        return dataclasses.replace(
          current,
          prompt_report=action.action.report,
          prompt_record_id=action.action.record_id,
        ), ()
      if isinstance(action.action, _ShowContext):
        return current, (effects.EmitText(_report_text(current.prompt_report)),)
    if (
      isinstance(action, actions.WorkerFailed)
      and 'Invalid prompt budget:' in action.message
    ):
      return current, (effects.EmitText(_report_text(current.prompt_report)),)
    return current, ()

  @override
  def render(
    self,
    global_state: state.State,
    extension_state: ContextUsage,
  ) -> containers.AnyContainer | None:
    """Render the current model context usage."""
    if global_state.session.pending_switch is not None:
      return None
    return containers.Window(
      controls.FormattedTextControl([
        ('class:footer', extension_state.render())
      ]),
      style='class:footer',
      dont_extend_height=True,
      dont_extend_width=True,
    )
