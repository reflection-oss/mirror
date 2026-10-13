"""Stateless rows for streamed and running tool calls."""

from __future__ import annotations

import functools
from typing import override

from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import containers, controls, dimension
from rich import console, table, text

from mirror.cli import actions as _actions
from mirror.cli import rendering, themes, tool_groups
from mirror.cli import state as _state
from mirror.cli.components import component


class Pending(component.Component):
  """Tool calls that are assembling or awaiting results."""

  @override
  def render(self, state: _state.State) -> containers.AnyContainer:
    return containers.Window(
      controls.FormattedTextControl(render(state)),
      height=dimension.Dimension.exact(rows(state)),
      wrap_lines=False,
    )


def render(state: _state.State) -> ansi.ANSI:
  """Render in-flight tool calls exactly like their settled form."""
  approval = state.approval.request
  running = tuple(
    call
    for call in state.stream.running
    if approval is None or call.call_id != approval.call_id
  )
  return _render(
    state.status_bar.view_mode,
    running,
    state.stream.pending,
    state.tool_group,
    state.layout.columns,
    themes.current(state.extensions),
  )


@functools.lru_cache(maxsize=16)
def _render(
  view_mode: _actions.ViewMode,
  running: tuple[_actions.RunningToolCall, ...],
  pending: tuple[_actions.PendingToolCall, ...],
  group: tool_groups.ToolGroup | None,
  columns: int,
  theme_name: str,
) -> ansi.ANSI:
  """Share tool-row rendering between measurement and painting."""
  body = 'running…'
  if view_mode is _actions.ViewMode.COMPACT:
    parts: list[console.RenderableType] = []
    calls = (*running, *pending)
    key = tool_groups.group_key(calls[-1].name) if calls else None
    summary = group
    if summary is None and key is not None:
      summary = tool_groups.ToolGroup(key=key)
    if summary is not None:
      parts.append(summary.summary(active=bool(calls)))
    label = group.latest if group is not None else ''
    wrap = False
    if calls:
      current = calls[-1]
      wrap = current.name == 'bash'
      args = rendering.command_preview(current.args) if wrap else current.args
      verb = {
        'bash': 'Running',
        'read': 'Reading',
        'grep': 'Searching',
        'find': 'Searching',
        'glob': 'Searching',
        'ls': 'Scanning',
        'edit': 'Editing',
        'write': 'Writing',
        'web_search': 'Web searching',
        'web_browse': 'Web browsing',
        'subagent': 'Subagent',
      }.get(current.name, tool_groups.display_name(current.name).capitalize())
      label = f'{verb} {args}'
      if len(calls) > 1:
        label += f' · {len(calls)} running'
    if label:
      prefix = '  └ ' if summary is not None else '· '
      row = table.Table.grid()
      row.add_column(width=len(prefix))
      row.add_column()
      row.add_row(
        text.Text(prefix, style='dim'),
        text.Text(
          rendering.strip_ansi(label),
          style='dim',
          overflow='fold' if wrap else 'ellipsis',
          no_wrap=not wrap,
        ),
      )
      parts.append(row)
    return ansi.ANSI(
      rendering.renderable_to_ansi(
        console.Group(*parts), columns, theme_name=theme_name
      ).rstrip()
    )
  blocks = [
    *(
      rendering.running_tool_renderable(*call.render_hint, body)
      for call in running
    ),
    *(
      rendering.running_tool_renderable(*call.render_hint, body)
      for call in pending
    ),
  ]
  return ansi.ANSI(
    rendering.render_blocks(blocks, columns, theme_name=theme_name)
  )


def rows(state: _state.State) -> int:
  """Return rows needed by running calls not gated on approval."""
  if state.status_bar.view_mode is _actions.ViewMode.EXTRA_COMPACT:
    return 0
  approval = state.approval.request
  has_running = any(
    approval is None or call.call_id != approval.call_id
    for call in state.stream.running
  )
  has_summary = (
    state.status_bar.view_mode is _actions.ViewMode.COMPACT
    and state.tool_group is not None
  )
  if not has_running and not state.stream.pending and not has_summary:
    return 0
  text = ''.join(
    fragment[1]
    for fragment in render(state).__pt_formatted_text__()
    if fragment[0] != '[ZeroWidthEscape]'
  )
  return text.count('\n') + 1


def count(state: _state.State) -> int:
  """Return the number of durable and still-streaming tool calls."""
  if state.status_bar.view_mode is _actions.ViewMode.EXTRA_COMPACT:
    return 0
  return len(state.stream.running) + len(state.stream.pending)
