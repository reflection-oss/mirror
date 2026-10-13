"""Immutable actions accepted by the Mirror UI reducer."""

from __future__ import annotations

import dataclasses
import enum
import random
import time
from collections.abc import Callable
from typing import Literal

from mirror.protocol import (
  agent_pb2,
  mirror_pb2,
)
from rich import console as rich_console

from mirror import config, skills, workspace
from mirror.cli import cli_store, models, selection, tool_approvals, tool_groups
from mirror.cli.components import spinner


@dataclasses.dataclass(frozen=True)
class ShowExtensions:
  """Open the core extension menu."""


@dataclasses.dataclass(frozen=True)
class AddExtension:
  """Declare and download a GitHub extension without granting execution consent."""

  url: str


@dataclasses.dataclass(frozen=True)
class UpdateExtensions:
  """Choose sources to update, or download and reload a selection."""

  name: str | None = None


@dataclasses.dataclass(frozen=True)
class RemoveExtension:
  """Choose a GitHub source to remove, then confirm removal of its declaration."""

  key: str | None = None
  confirmed: bool = False


@dataclasses.dataclass(frozen=True)
class ToggleExtension:
  """Toggle one extension's persistent enablement."""

  key: str


@dataclasses.dataclass(frozen=True)
class ExtensionConsent:
  """Answer a request to run newly discovered extension code."""

  key: str
  enabled: bool | None


class ViewMode(enum.Enum):
  """How much detail the history shows when rendered to the terminal."""

  TRANSCRIPT = 'transcript'
  COMPACT = 'compact'
  EXTRA_COMPACT = 'extra-compact'


@dataclasses.dataclass(frozen=True)
class PendingSwitch:
  """A conversation switch awaiting its replacement process's replay."""

  conversation_id: str
  create: bool


@dataclasses.dataclass(frozen=True)
class ApprovalRequest:
  """A tool call awaiting a human approval decision."""

  call_id: str
  name: str
  args: str
  remember: tool_approvals.ApprovalRule | None = None


@dataclasses.dataclass(frozen=True)
class SecretRequest:
  """A masked prompt whose input only the requesting extension receives."""

  id: str
  title: str
  extension_id: str
  on_submit: Callable[[str], object]
  hint: str = 'Enter to submit · Esc to cancel'
  cancel_action: Action | None = None


@dataclasses.dataclass(frozen=True)
class UserTurn:
  """A user turn offered as a rewind point."""

  entry_id: str
  text: str

  def preview(self, limit: int = 56) -> str:
    """Render the turn as one bounded line."""
    compact = ' '.join(self.text.split())
    if len(compact) <= limit:
      return compact
    return f'{compact[: limit - 1].rstrip()}…'


@dataclasses.dataclass(frozen=True)
class ComposerState:
  """Input snapshot observed by declarative completion components."""

  text: str = ''
  cursor: int = 0


@dataclasses.dataclass(frozen=True)
class Task:
  """One running background task."""

  id: str
  tool_name: str
  display: str | None


class WorktreeAgentState(enum.Enum):
  """The status of one agent in the worktree tabs."""

  NOT_STARTED = enum.auto()
  SAVED = enum.auto()
  IDLE = enum.auto()
  CREATING = enum.auto()
  RUNNING = enum.auto()
  ATTENTION = enum.auto()
  UNREAD = enum.auto()
  FAILED = enum.auto()


@dataclasses.dataclass(frozen=True)
class DropConfirmation:
  """An unmanaged worktree awaiting explicit deletion approval."""

  label: str
  path: str


@dataclasses.dataclass(frozen=True)
class StreamToolCall:
  """The bounded argument prefix of an assembling streamed tool call."""

  index: int
  name: str = ''
  arguments: str = ''


@dataclasses.dataclass(frozen=True)
class PendingToolCall:
  """A streamed tool call with enough information to render."""

  index: int
  name: str
  args: str
  activity_kind: str

  @property
  def render_hint(self) -> tuple[str, str, str]:
    """The fields consumed by tool-row renderers."""
    return self.name, self.args, self.activity_kind


@dataclasses.dataclass(frozen=True)
class RunningToolCall:
  """A durable tool call awaiting its result."""

  call_id: str
  name: str
  args: str
  activity_kind: str
  arguments_json: str

  @property
  def render_hint(self) -> tuple[str, str, str]:
    """The fields consumed by tool-row renderers."""
    return self.name, self.args, self.activity_kind


@dataclasses.dataclass(frozen=True)
class ComposerSubmit:
  pass


@dataclasses.dataclass(frozen=True)
class ComposerNewline:
  pass


@dataclasses.dataclass(frozen=True)
class ComposerEditExternally:
  """Edit the draft in the user's external editor without submitting."""


@dataclasses.dataclass(frozen=True)
class ComposerUndo:
  pass


@dataclasses.dataclass(frozen=True)
class ComposerRedo:
  pass


@dataclasses.dataclass(frozen=True)
class MoveSelection:
  delta: int


@dataclasses.dataclass(frozen=True)
class AcceptSelection:
  index: int | None = None


@dataclasses.dataclass(frozen=True)
class AdjustSelection:
  """Step the highlighted selector choice's value by `delta`."""

  delta: int


@dataclasses.dataclass(frozen=True)
class CloseSelector:
  pass


@dataclasses.dataclass(frozen=True)
class CycleResumeScope:
  pass


@dataclasses.dataclass(frozen=True)
class ShowText:
  """Print `text` into the transcript (optionally styled)."""

  text: rich_console.RenderableType
  style: str = ''


@dataclasses.dataclass(frozen=True)
class QuitApp:
  """Exit the REPL."""


@dataclasses.dataclass(frozen=True)
class ClearScreen:
  """Flush terminal scrollback and redraw the conversation."""


@dataclasses.dataclass(frozen=True)
class ResumeSession:
  """Restart the worker for the conversation matching `reference`."""

  reference: str | None = None


@dataclasses.dataclass(frozen=True)
class NewSession:
  """Restart the worker for a new conversation."""


@dataclasses.dataclass(frozen=True)
class WorkspaceTagCommand:
  """Run a workspace tag command."""

  args: str = 'cycle'


@dataclasses.dataclass(frozen=True)
class RenameWorktree:
  """Set the selected worktree's user-facing alias."""

  alias: str


@dataclasses.dataclass(frozen=True)
class CompactConversation:
  """Request manual context compaction."""


@dataclasses.dataclass(frozen=True)
class RewindSession:
  """Pick an earlier user turn to continue the conversation from."""


@dataclasses.dataclass(frozen=True)
class OpenFolder:
  """Open an existing directory as a folder workspace."""

  path: str


@dataclasses.dataclass(frozen=True)
class CloseWorkspace:
  """Close the selected workspace without deleting its directory."""


@dataclasses.dataclass(frozen=True)
class CreateWorktree:
  """Fork the current checkout and select its optionally named worktree."""

  name: str | None = None


@dataclasses.dataclass(frozen=True)
class LoadRemoteSessions:
  """Discover remote sessions for this invocation."""


@dataclasses.dataclass(frozen=True)
class CreateRemote:
  """Start or resume a named hosted agent; a new session starts in `workspace`."""

  title: str | None = None
  workspace: mirror_pb2.Workspace | None = None


@dataclasses.dataclass(frozen=True)
class Drop:
  """Delete the current managed execution environment and close its agent."""


@dataclasses.dataclass(frozen=True)
class CycleViewMode:
  """Cycle to the next history view mode."""


@dataclasses.dataclass(frozen=True)
class SetViewMode:
  """Set the history view mode."""

  mode: ViewMode


@dataclasses.dataclass(frozen=True)
class SetToolGroup:
  """Replace the live tool-group summary, or clear it at a breakpoint."""

  summary: tool_groups.ToolGroup | None


@dataclasses.dataclass(frozen=True)
class SelectPermissions:
  """Open the permission-mode picker."""


@dataclasses.dataclass(frozen=True)
class SetStatus:
  value: str


@dataclasses.dataclass(frozen=True)
class RecordResponseUsage:
  """Add one completed response's reported output tokens to the current turn."""

  completion_tokens: int


@dataclasses.dataclass(frozen=True)
class ObserveStreamText:
  content: str = ''
  reasoning: str = ''
  refusal: str = ''


@dataclasses.dataclass(frozen=True)
class CommitStreamReasoning:
  pass


@dataclasses.dataclass(frozen=True)
class CommitStreamContent:
  pass


@dataclasses.dataclass(frozen=True)
class ObserveStreamToolCall:
  call: StreamToolCall
  pending: PendingToolCall | None


@dataclasses.dataclass(frozen=True)
class ClearStreamPending:
  pass


@dataclasses.dataclass(frozen=True)
class CloseStream:
  pass


@dataclasses.dataclass(frozen=True)
class StartRunningToolCalls:
  calls: tuple[RunningToolCall, ...]


@dataclasses.dataclass(frozen=True)
class FinishRunningToolCall:
  call_id: str


@dataclasses.dataclass(frozen=True)
class ClearRunningToolCalls:
  pass


@dataclasses.dataclass(frozen=True)
class WorktreeDropConfirmationRequested:
  confirmation: DropConfirmation


@dataclasses.dataclass(frozen=True)
class ClearWorktreeDrop:
  pass


@dataclasses.dataclass(frozen=True)
class ConfirmWorktreeDrop:
  pass


@dataclasses.dataclass(frozen=True)
class TurnStarted:
  """Start user work with a star color chosen once when the action is created."""

  observed_at: float = dataclasses.field(default_factory=time.monotonic)
  spinner_color: spinner.Color = dataclasses.field(
    default_factory=lambda: random.choice(tuple(spinner.Color))
  )


@dataclasses.dataclass(frozen=True)
class AgentTurnStarted:
  observed_at: float = dataclasses.field(default_factory=time.monotonic)


@dataclasses.dataclass(frozen=True)
class WorkerLoadingStarted:
  pass


@dataclasses.dataclass(frozen=True)
class WorkerLoadingFinished:
  pass


@dataclasses.dataclass(frozen=True)
class WorkerRetryStarted:
  pass


@dataclasses.dataclass(frozen=True)
class WorkerRetryFinished:
  pass


@dataclasses.dataclass(frozen=True)
class WorkerRecoveryFinished:
  replaying: bool


@dataclasses.dataclass(frozen=True)
class InputSubmitted:
  message_id: str
  instruction: str


@dataclasses.dataclass(frozen=True)
class InputAcknowledged:
  message_id: str


@dataclasses.dataclass(frozen=True)
class UnsentInputsDiscarded:
  """Forget locally cancelled submissions that never reached a worker."""


@dataclasses.dataclass(frozen=True)
class InputRejected:
  message_id: str


@dataclasses.dataclass(frozen=True)
class RestoreDraft:
  instruction: str


@dataclasses.dataclass(frozen=True)
class CommandFailed:
  message: str


@dataclasses.dataclass(frozen=True)
class SupervisorError:
  message: str


@dataclasses.dataclass(frozen=True)
class SetPendingSwitch:
  value: PendingSwitch | None


@dataclasses.dataclass(frozen=True)
class ToggleApprovalMode:
  pass


@dataclasses.dataclass(frozen=True)
class ModelSwitchFinished:
  model_id: str


@dataclasses.dataclass(frozen=True)
class CancelOrQuitRequested:
  """Cancel the current interaction or work, or quit when idle and empty."""


@dataclasses.dataclass(frozen=True)
class QuitRequested:
  pass


@dataclasses.dataclass(frozen=True)
class OpenWorkspaces:
  """Browse worktrees and remotes without activating them."""


@dataclasses.dataclass(frozen=True)
class CycleWorktreeRequested:
  offset: int


@dataclasses.dataclass(frozen=True)
class SelectWorktreeRequested:
  index: int


@dataclasses.dataclass(frozen=True)
class SelectWorkspaceRequested:
  """Switch to the workspace with this stable identity."""

  identity: str


@dataclasses.dataclass(frozen=True)
class DetachRequested:
  pass


@dataclasses.dataclass(frozen=True)
class NavigateHistory:
  """Navigate the owned conversation viewport without changing the editor."""

  direction: Literal['page_up', 'page_down', 'oldest', 'latest', 'up', 'down']


@dataclasses.dataclass(frozen=True)
class InterruptRequested:
  pass


@dataclasses.dataclass(frozen=True)
class InterruptSent:
  message_id: str


@dataclasses.dataclass(frozen=True)
class InterruptDeclined:
  message_id: str


@dataclasses.dataclass(frozen=True)
class SetActive:
  value: bool


@dataclasses.dataclass(frozen=True)
class CliStarted:
  pass


@dataclasses.dataclass(frozen=True)
class SessionCreated:
  conversation_id: str


@dataclasses.dataclass(frozen=True)
class SessionAttached:
  conversation_id: str


@dataclasses.dataclass(frozen=True)
class WorkerStarted:
  pass


@dataclasses.dataclass(frozen=True)
class WorkerStopped:
  pass


@dataclasses.dataclass(frozen=True)
class WorkerFailed:
  message: str


@dataclasses.dataclass(frozen=True)
class ExitRequested:
  pass


@dataclasses.dataclass(frozen=True)
class SetQuitting:
  pass


@dataclasses.dataclass(frozen=True)
class ApprovalRequested:
  request: ApprovalRequest


@dataclasses.dataclass(frozen=True)
class MoveApproval:
  delta: int


@dataclasses.dataclass(frozen=True)
class ClearApproval:
  call_id: str


@dataclasses.dataclass(frozen=True)
class DecideApproval:
  approved: bool | None = None
  remember: bool = False
  call_id: str | None = None


@dataclasses.dataclass(frozen=True)
class MoveCompletionRequested:
  """Move the active composer-completion selection."""

  delta: int
  skill_count: int
  command_count: int


@dataclasses.dataclass(frozen=True)
class CommandSelection:
  """Command metadata captured by the completion UI."""

  name: str
  completion: str
  edit_arguments: bool
  available_when_busy: bool
  description: str = ''
  prompt_hint: str | None = None
  arguments_are_optional: bool = False


@dataclasses.dataclass(frozen=True)
class CompleteSelectionRequested:
  """Complete the selected composer suggestion."""

  selected_skill: skills.Skill | None
  selected_command: CommandSelection | None


@dataclasses.dataclass(frozen=True)
class DismissCompletionRequested:
  """Dismiss completion with an optional skill-token edit."""

  edit: ComposerState | None


@dataclasses.dataclass(frozen=True)
class SubmitRequested:
  """Submit the current composer interaction."""

  selected_skill: skills.Skill | None
  selected_command: CommandSelection | None


@dataclasses.dataclass(frozen=True)
class SubmitInstruction:
  instruction: str


@dataclasses.dataclass(frozen=True)
class UserMessageObserved:
  content: str
  message_id: str = ''
  spinner_color: spinner.Color = dataclasses.field(
    default_factory=lambda: random.choice(tuple(spinner.Color))
  )


@dataclasses.dataclass(frozen=True)
class CompactionStarted:
  started: float


@dataclasses.dataclass(frozen=True)
class CompactionFinished:
  pass


@dataclasses.dataclass(frozen=True)
class TurnFinished:
  pass


@dataclasses.dataclass(frozen=True)
class SessionSwitchStarted:
  pending: PendingSwitch


@dataclasses.dataclass(frozen=True)
class SessionOpened:
  conversation_id: str
  composer: ComposerState | None = None


@dataclasses.dataclass(frozen=True)
class SetReplaying:
  value: bool


@dataclasses.dataclass(frozen=True)
class ReplayFinished:
  has_user_message: bool


@dataclasses.dataclass(frozen=True)
class RerenderRequested:
  note: str | None = None
  drop_scrollback: bool = False


@dataclasses.dataclass(frozen=True)
class RerenderStarted:
  pass


@dataclasses.dataclass(frozen=True)
class RepaintFinished:
  reset_busy: bool


@dataclasses.dataclass(frozen=True)
class HistoryRendered:
  width: int


@dataclasses.dataclass(frozen=True)
class TerminalResized:
  columns: int


@dataclasses.dataclass(frozen=True)
class ResumeChoicesLoaded:
  scope: workspace.SessionScope
  choices: tuple[cli_store.ConversationInfo, ...]


@dataclasses.dataclass(frozen=True)
class ResumeReferenceResolved:
  reference: str
  matches: tuple[cli_store.ConversationInfo, ...]


@dataclasses.dataclass(frozen=True)
class SetResumeScope:
  scope: workspace.SessionScope


@dataclasses.dataclass(frozen=True)
class ClearResume:
  pass


@dataclasses.dataclass(frozen=True)
class AcceptResumeConfirmation:
  pass


@dataclasses.dataclass(frozen=True)
class CancelModelSelection:
  """Cancel the model picker opened from sign-in."""


@dataclasses.dataclass(frozen=True)
class ForgetReflectionKey:
  """Remove the local saved Reflection credential."""


@dataclasses.dataclass(frozen=True)
class SetModel:
  model_id: str


@dataclasses.dataclass(frozen=True)
class SetRemoteModel:
  """Choose the remote session's model; an empty `api` keeps its protocol."""

  model_id: str
  api: str


@dataclasses.dataclass(frozen=True)
class SetReasoning:
  effort: str


@dataclasses.dataclass(frozen=True)
class ModelResolved:
  option: models.ModelOption


@dataclasses.dataclass(frozen=True)
class OpenSelector:
  selector: selection.Selector[Action]


@dataclasses.dataclass(frozen=True)
class SelectorSearchChanged:
  query: str


@dataclasses.dataclass(frozen=True)
class SelectorChoicesLoaded:
  selector_id: str
  choices: tuple[selection.Choice, ...]


@dataclasses.dataclass(frozen=True)
class SelectorResultsLoaded:
  """Already-ranked results for one asynchronous selector query."""

  selector_id: str
  query: str
  choices: tuple[selection.Choice, ...]
  loading: bool


@dataclasses.dataclass(frozen=True)
class RequestPromptHistory:
  pass


@dataclasses.dataclass(frozen=True)
class RestorePrompt:
  text: str


@dataclasses.dataclass(frozen=True)
class MoveSelectorSelection:
  delta: int


@dataclasses.dataclass(frozen=True)
class ClearSelector:
  pass


@dataclasses.dataclass(frozen=True)
class SelectorChoiceSelected:
  """A value chosen from the visible selector."""

  selector_id: str
  value: str


@dataclasses.dataclass(frozen=True)
class RequestSecret:
  request: SecretRequest


@dataclasses.dataclass(frozen=True)
class SecretLengthChanged:
  length: int


@dataclasses.dataclass(frozen=True)
class SubmitSecret:
  pass


@dataclasses.dataclass(frozen=True)
class CancelSecret:
  pass


@dataclasses.dataclass(frozen=True)
class ClearSecret:
  pass


@dataclasses.dataclass(frozen=True)
class SetApprovalMode:
  mode: str


@dataclasses.dataclass(frozen=True)
class RewindTurnsLoaded:
  conversation_id: str
  turns: tuple[UserTurn, ...]


@dataclasses.dataclass(frozen=True)
class StageRewind:
  turn: UserTurn
  discards: int


@dataclasses.dataclass(frozen=True)
class SetRewindConfirming:
  value: bool


@dataclasses.dataclass(frozen=True)
class CloseRewindOverlay:
  pass


@dataclasses.dataclass(frozen=True)
class CancelRewind:
  pass


@dataclasses.dataclass(frozen=True)
class ConfirmRewind:
  pass


@dataclasses.dataclass(frozen=True)
class ComposerChanged:
  text: str
  cursor: int


@dataclasses.dataclass(frozen=True)
class CompletionChanged:
  skills: tuple[skills.Skill, ...] = ()
  commands: tuple[CommandSelection, ...] = ()
  argument_hint: str | None = None
  argument_optional: bool = False


@dataclasses.dataclass(frozen=True)
class TaskChanged:
  task_id: str
  task: Task | None
  foreground: bool


@dataclasses.dataclass(frozen=True)
class SelectFirstTask:
  pass


@dataclasses.dataclass(frozen=True)
class MoveTaskSelection:
  delta: int


@dataclasses.dataclass(frozen=True)
class ClearTaskSelection:
  pass


@dataclasses.dataclass(frozen=True)
class CancelSelectedTask:
  pass


@dataclasses.dataclass(frozen=True)
class ExtensionAction:
  """Base class for immutable actions owned by an extension."""


@dataclasses.dataclass(frozen=True)
class RoutedExtensionAction:
  extension_id: str
  action: ExtensionAction


@dataclasses.dataclass(frozen=True)
class ExtensionLoaded:
  extension_id: str


@dataclasses.dataclass(frozen=True)
class ExtensionConfigChanged:
  """Notify one extension without resetting state or cancelling its effects."""

  extension_id: str
  config: config.JsonObject


@dataclasses.dataclass(frozen=True)
class ExtensionUnloaded:
  extension_id: str


@dataclasses.dataclass(frozen=True)
class HydrateExtensions:
  """Restore extensions from replayed conversation records."""

  records: tuple[agent_pb2.AgentMessage, ...]


type Action = (
  ComposerSubmit
  | ComposerNewline
  | ComposerEditExternally
  | ComposerUndo
  | ComposerRedo
  | MoveSelection
  | AcceptSelection
  | AdjustSelection
  | CloseSelector
  | CycleResumeScope
  | RoutedExtensionAction
  | ExtensionUnloaded
  | ExtensionLoaded
  | ExtensionConfigChanged
  | HydrateExtensions
  | ShowText
  | ShowExtensions
  | AddExtension
  | UpdateExtensions
  | RemoveExtension
  | ToggleExtension
  | ExtensionConsent
  | ClearScreen
  | QuitApp
  | CycleViewMode
  | SetViewMode
  | SetToolGroup
  | SelectPermissions
  | ResumeSession
  | NewSession
  | RenameWorktree
  | WorkspaceTagCommand
  | CompactConversation
  | RewindSession
  | LoadRemoteSessions
  | CreateRemote
  | OpenFolder
  | CloseWorkspace
  | CreateWorktree
  | Drop
  | SetStatus
  | RecordResponseUsage
  | ObserveStreamText
  | CommitStreamReasoning
  | CommitStreamContent
  | ObserveStreamToolCall
  | ClearStreamPending
  | CloseStream
  | StartRunningToolCalls
  | FinishRunningToolCall
  | ClearRunningToolCalls
  | WorktreeDropConfirmationRequested
  | ClearWorktreeDrop
  | ConfirmWorktreeDrop
  | TurnStarted
  | AgentTurnStarted
  | WorkerLoadingStarted
  | WorkerLoadingFinished
  | WorkerRetryStarted
  | WorkerRetryFinished
  | WorkerRecoveryFinished
  | InputSubmitted
  | InputAcknowledged
  | UnsentInputsDiscarded
  | InputRejected
  | RestoreDraft
  | CommandFailed
  | SupervisorError
  | SetPendingSwitch
  | ToggleApprovalMode
  | ModelSwitchFinished
  | CancelOrQuitRequested
  | QuitRequested
  | OpenWorkspaces
  | CycleWorktreeRequested
  | SelectWorktreeRequested
  | SelectWorkspaceRequested
  | DetachRequested
  | InterruptRequested
  | NavigateHistory
  | InterruptSent
  | InterruptDeclined
  | SetActive
  | CliStarted
  | SessionCreated
  | SessionAttached
  | WorkerStarted
  | WorkerStopped
  | WorkerFailed
  | ExitRequested
  | SetQuitting
  | ApprovalRequested
  | MoveApproval
  | ClearApproval
  | DecideApproval
  | MoveCompletionRequested
  | CompleteSelectionRequested
  | DismissCompletionRequested
  | SubmitRequested
  | SubmitInstruction
  | UserMessageObserved
  | CompactionStarted
  | CompactionFinished
  | TurnFinished
  | SessionSwitchStarted
  | SessionOpened
  | SetReplaying
  | ReplayFinished
  | RerenderRequested
  | RerenderStarted
  | RepaintFinished
  | HistoryRendered
  | TerminalResized
  | ResumeChoicesLoaded
  | ResumeReferenceResolved
  | SetResumeScope
  | ClearResume
  | AcceptResumeConfirmation
  | CancelModelSelection
  | ForgetReflectionKey
  | SetModel
  | SetRemoteModel
  | SetReasoning
  | ModelResolved
  | OpenSelector
  | SelectorSearchChanged
  | SelectorChoicesLoaded
  | SelectorResultsLoaded
  | RequestPromptHistory
  | RestorePrompt
  | MoveSelectorSelection
  | ClearSelector
  | SelectorChoiceSelected
  | RequestSecret
  | SecretLengthChanged
  | SubmitSecret
  | CancelSecret
  | ClearSecret
  | SetApprovalMode
  | RewindTurnsLoaded
  | StageRewind
  | SetRewindConfirming
  | CloseRewindOverlay
  | CancelRewind
  | ConfirmRewind
  | ComposerChanged
  | CompletionChanged
  | TaskChanged
  | SelectFirstTask
  | MoveTaskSelection
  | ClearTaskSelection
  | CancelSelectedTask
)
