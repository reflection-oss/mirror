"""Immutable application state for the Mirror TUI."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from mirror import skills, workspace
from mirror.cli import (
  actions,
  cli_store,
  extension_state,
  selection,
  tool_groups,
)
from mirror.cli.components import spinner


@dataclasses.dataclass(frozen=True)
class ActivityState:
  """State observed by transient activity and task controls."""

  busy: bool = False
  status: str = ''
  work_started: float | None = None
  work_paused: float | None = None
  foreground_tasks: frozenset[str] = frozenset()
  spinner_color: spinner.Color = spinner.Color.PALE_PINK


@dataclasses.dataclass(frozen=True)
class StatusBarState:
  """State observed by the persistent status bar."""

  model: str
  reasoning_effort: str
  approval_mode: str
  view_mode: actions.ViewMode = actions.ViewMode.COMPACT
  resize_dirty: bool = False


@dataclasses.dataclass(frozen=True)
class ApprovalState:
  """Manual tool-approval state."""

  requests: tuple[actions.ApprovalRequest, ...] = ()

  @property
  def request(self) -> actions.ApprovalRequest | None:
    return self.requests[0] if self.requests else None


@dataclasses.dataclass(frozen=True)
class TurnState:
  """Queued inputs, compaction, and output usage for the active user turn."""

  queued: tuple[str, ...] = ()
  submitted_inputs: tuple[tuple[str, str], ...] = ()
  awaiting_inputs: tuple[tuple[str, str], ...] = ()
  input_order: tuple[str, ...] = ()
  interrupt_id: str | None = None
  agent_turn: bool = False
  compaction_started: float | None = None
  output_tokens: int = 0

  def is_queued_input(self, message_id: str) -> bool:
    """The busy-turn queue is the suffix of the ordered uncommitted inputs."""
    return (
      bool(self.queued) and message_id in self.input_order[-len(self.queued) :]
    )

  @property
  def pending_instructions(self) -> tuple[str, ...]:
    """Uncommitted submissions in local order, unaffected by ack arrival order."""
    tracked = dict((*self.submitted_inputs, *self.awaiting_inputs))
    return tuple(tracked[key] for key in self.input_order if key in tracked)


@dataclasses.dataclass(frozen=True)
class LifecycleState:
  """Agent and worktree lifecycle state."""

  active: bool
  unread: bool = False
  failure: str | None = None
  started: bool = False
  quitting: bool = False


@dataclasses.dataclass(frozen=True)
class SessionState:
  """Conversation and replay state."""

  conversation_id: str
  remote: bool = False
  pending_switch: actions.PendingSwitch | None = None
  has_user_message: bool = False
  replaying: bool = True
  rerendering: bool = False
  history_width: int | None = None
  resize_generation: int = 0


@dataclasses.dataclass(frozen=True)
class MenuState:
  """Visibility, selection, and viewport for one menu."""

  active: bool = False
  selected: int = 0
  start: int = 0
  max_rows: int = 6

  def reset(self) -> MenuState:
    return MenuState(active=self.active, max_rows=self.max_rows)

  def open(self, *, selected: int = 0) -> MenuState:
    return MenuState(active=True, selected=selected, max_rows=self.max_rows)

  def move(self, delta: int, item_count: int) -> MenuState:
    if not item_count:
      return dataclasses.replace(self, selected=0, start=0)
    selected = (min(self.selected, item_count - 1) + delta) % item_count
    start = min(self.start, max(0, item_count - self.max_rows))
    if selected < start:
      start = selected
    elif selected >= start + self.max_rows:
      start = selected - self.max_rows + 1
    return dataclasses.replace(self, selected=selected, start=start)

  def selected_item[Item](self, items: Sequence[Item]) -> Item | None:
    """Return the selected item after clamping dynamic choices."""
    return items[min(self.selected, len(items) - 1)] if items else None


@dataclasses.dataclass(frozen=True)
class ResumeState:
  """Resume-picker state."""

  confirmation: cli_store.ConversationInfo | None = None
  scope: workspace.SessionScope = workspace.SessionScope.CWD

  @property
  def active(self) -> bool:
    return self.confirmation is not None


@dataclasses.dataclass(frozen=True)
class SelectorState:
  """The choice selector currently shown by the CLI."""

  menu: MenuState = MenuState(max_rows=8)
  selector: selection.Selector[object] | None = None
  query: str = ''
  search_pending: bool = False

  def filtered_choices(self) -> tuple[selection.Choice, ...]:
    choices = self.selector.choices if self.selector is not None else ()
    if self.selector is not None and self.selector.prefiltered:
      return choices
    return selection.fuzzy_matches(
      choices, self.query, lambda choice: (choice.label, choice.detail)
    )

  @property
  def id(self) -> str | None:
    return self.selector.id if self.selector is not None else None

  @property
  def active(self) -> bool:
    return self.menu.active and self.selector is not None

  @property
  def numbered(self) -> bool:
    return self.active and self.selector is not None and self.selector.numbered

  def selected_choice(self) -> selection.Choice | None:
    """Return the selected choice only when results match the current query."""
    if self.search_pending:
      return None
    return self.menu.selected_item(self.filtered_choices())


@dataclasses.dataclass(frozen=True)
class SecretState:
  """The masked prompt currently shown by the CLI, without its input."""

  request: actions.SecretRequest | None = None
  length: int = 0

  @property
  def active(self) -> bool:
    return self.request is not None


PERMISSION_MODES = ('auto', 'approve')


@dataclasses.dataclass(frozen=True)
class StagedRewind:
  """A reversible preview of continuing from an earlier turn."""

  target: actions.UserTurn
  discards: int
  confirming: bool = False


@dataclasses.dataclass(frozen=True)
class RewindState:
  """Rewind picker and staged-preview state."""

  staged: StagedRewind | None = None

  @property
  def confirming(self) -> bool:
    return self.staged is not None and self.staged.confirming

  @property
  def active(self) -> bool:
    return self.confirming


@dataclasses.dataclass(frozen=True)
class CompletionState:
  """Slash-command and skill-completion menu state."""

  slash: MenuState = MenuState()
  skill: MenuState = MenuState()
  commands: tuple[actions.CommandSelection, ...] = ()
  skills: tuple[skills.Skill, ...] = ()
  argument_hint: str | None = None
  argument_optional: bool = False

  @property
  def active(self) -> bool:
    return bool(self.skills or self.commands)


@dataclasses.dataclass(frozen=True)
class TasksState:
  """Running task rows and selection."""

  tasks: tuple[actions.Task, ...] = ()
  selected_id: str | None = None

  def move(self, delta: int) -> TasksState:
    """Move selection, clearing it above the first row."""
    task_ids = tuple(task.id for task in self.tasks)
    if not task_ids:
      return dataclasses.replace(self, selected_id=None)
    if self.selected_id not in task_ids:
      return dataclasses.replace(self, selected_id=task_ids[0])
    index = task_ids.index(self.selected_id) + delta
    selected_id = None if index < 0 else task_ids[min(index, len(task_ids) - 1)]
    return dataclasses.replace(self, selected_id=selected_id)


@dataclasses.dataclass(frozen=True)
class LayoutState:
  """Terminal dimensions observed by layout-sensitive components."""

  columns: int = 80


@dataclasses.dataclass(frozen=True)
class StreamPreview:
  """The text accumulated for the active model stream."""

  content: str = ''
  reasoning: str = ''
  refusal: str = ''
  committed_reasoning: str | None = None
  committed_content: str | None = None


@dataclasses.dataclass(frozen=True)
class StreamState:
  """Immutable logical state for live text and tool-call rows."""

  preview: StreamPreview | None = None
  tool_calls: tuple[actions.StreamToolCall, ...] = ()
  pending: tuple[actions.PendingToolCall, ...] = ()
  running: tuple[actions.RunningToolCall, ...] = ()


@dataclasses.dataclass(frozen=True)
class State:
  """The immutable application state shared by terminal components."""

  activity: ActivityState
  status_bar: StatusBarState
  session: SessionState
  lifecycle: LifecycleState
  workspace: str = ''
  approval: ApprovalState = ApprovalState()
  turn: TurnState = TurnState()
  resume: ResumeState = ResumeState()
  selector: SelectorState = SelectorState()
  secret: SecretState = SecretState()
  rewind: RewindState = RewindState()
  completion: CompletionState = CompletionState()
  composer: actions.ComposerState = actions.ComposerState()
  tasks: TasksState = TasksState()
  supervised: bool = False
  drop_confirmation: actions.DropConfirmation | None = None
  layout: LayoutState = LayoutState()
  stream: StreamState = StreamState()
  tool_group: tool_groups.ToolGroup | None = None
  extensions: extension_state.ExtensionStates = (
    extension_state.ExtensionStates()
  )
  # The host supplies this catalog in extension reducer and render snapshots.
  extension_tips: tuple[str, ...] = ()
