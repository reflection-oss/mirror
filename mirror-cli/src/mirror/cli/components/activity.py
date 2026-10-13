"""Transient agent-activity component for the terminal UI."""

from __future__ import annotations

from typing import override

from prompt_toolkit.formatted_text import (
  StyleAndTextTuples as _StyleAndTextTuples,
)
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import rendering, state, tool_groups
from mirror.cli.components import component, spinner, text_shimmer


class Activity(component.Component):
  """The transient agent-activity row above the status bar."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    """Render the activity row from immutable application state."""
    fragments = _content(state)
    return containers.Window(
      controls.FormattedTextControl(fragments),
      height=dimension.Dimension.exact(1 if fragments else 0),
      style='class:activity',
    )


def _content(state: state.State) -> _StyleAndTextTuples:
  pending_switch = state.session.pending_switch is not None
  busy = state.activity.busy
  status = state.activity.status
  if (
    pending_switch
    or state.approval.request is not None
    or not busy
    or not status
  ):
    return []
  if state.turn.compaction_started is not None:
    status = 'Compacting'
  elif status in {'thinking…', 'running tool…'}:
    calls = (*state.stream.running, *state.stream.pending)
    key = state.tool_group.key if state.tool_group is not None else None
    if key is None and calls:
      keys = {tool_groups.group_key(call.name) for call in calls}
      if len(keys) == 1:
        key = keys.pop()
    status = tool_groups.status_label(key) if key is not None else 'Working'
  tokens = (
    f' ({state.turn.output_tokens:,} tokens)'
    if state.turn.output_tokens
    else ''
  )
  timer = ''
  if state.activity.work_started is not None:
    timer = f' · {rendering.Stopwatch(state.activity.work_started).render()}'
  detach_hint = (
    '  ·  Ctrl+B to detach' if state.activity.foreground_tasks else ''
  )
  return [
    ('class:activity', ' '),
    *spinner.render(state.activity.spinner_color, style='class:activity'),
    ('class:activity', ' '),
    *text_shimmer.render(status, style='class:activity'),
    ('class:activity', tokens + timer + detach_hint),
  ]
