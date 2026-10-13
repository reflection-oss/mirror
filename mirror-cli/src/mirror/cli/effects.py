"""Immutable effects emitted by Mirror UI reducers."""

from __future__ import annotations

import dataclasses

from mirror.protocol import (
  agent_pb2,
  mirror_pb2,
)
from rich import console as rich_console

from mirror import workspace
from mirror.cli import actions, cli_store, models, tool_approvals


@dataclasses.dataclass(frozen=True)
class ShowExtensions:
  """Display the core extension menu."""


@dataclasses.dataclass(frozen=True)
class ChangeExtension:
  """Apply an extension toggle, trust decision, or source operation."""

  request: (
    actions.ToggleExtension
    | actions.ExtensionConsent
    | actions.AddExtension
    | actions.UpdateExtensions
    | actions.RemoveExtension
  )


@dataclasses.dataclass(frozen=True)
class RefreshCompletion:
  """Refresh completion choices after the extension command catalog changes."""


@dataclasses.dataclass(frozen=True)
class InsertComposerNewline:
  """Insert a newline in the live input buffer when it accepts one."""


@dataclasses.dataclass(frozen=True)
class InsertComposerText:
  """Insert text at the live composer's cursor without submitting it."""

  text: str


@dataclasses.dataclass(frozen=True)
class EditComposerExternally:
  """Hand the draft to the external editor and restore it on success."""


@dataclasses.dataclass(frozen=True)
class UndoComposer:
  """Undo an edit in the prompt-toolkit buffer."""


@dataclasses.dataclass(frozen=True)
class RedoComposer:
  """Redo an edit in the prompt-toolkit buffer."""


@dataclasses.dataclass(frozen=True)
class AcceptSelection:
  """Invoke the selected generic picker's callback."""


@dataclasses.dataclass(frozen=True)
class AdjustSelection:
  """Invoke the selected generic picker's adjust callback."""

  delta: int


@dataclasses.dataclass(frozen=True)
class CancelSelection:
  """Close the picker and dispatch its optional cancellation action."""


@dataclasses.dataclass(frozen=True)
class EmitText:
  """Print styled text into terminal scrollback."""

  text: rich_console.RenderableType
  style: str = ''


@dataclasses.dataclass(frozen=True)
class EmitError:
  """Print an error into terminal scrollback."""

  message: str


@dataclasses.dataclass(frozen=True)
class RecordError:
  """Retain and display an error reported by the worktree supervisor."""

  message: str


@dataclasses.dataclass(frozen=True)
class ExitApp:
  """Exit the prompt-toolkit application."""

  result: str | None = None
  erase_when_done: bool = False
  defer_until_running: bool = False
  handoff: bool = False


@dataclasses.dataclass(frozen=True)
class OpenFolder:
  """Open an existing directory as a folder workspace."""

  path: str


@dataclasses.dataclass(frozen=True)
class CloseWorkspace:
  """Close the selected workspace without deleting its directory."""


@dataclasses.dataclass(frozen=True)
class CreateManagedWorktree:
  """Create and select a supervised worktree."""

  name: str | None


@dataclasses.dataclass(frozen=True)
class DropManagedWorktree:
  """Drop the selected supervised worktree."""

  confirmed: bool = False


@dataclasses.dataclass(frozen=True)
class WorkspaceTagCommand:
  """Manage the selected workspace tag or tag definitions."""

  args: str = 'cycle'


@dataclasses.dataclass(frozen=True)
class RenameWorktree:
  """Persist an alias for the selected worktree."""

  alias: str


@dataclasses.dataclass(frozen=True)
class LoadRemoteSessions:
  """Start remote workspace discovery."""


@dataclasses.dataclass(frozen=True)
class CreateRemoteAgent:
  """Create or select a remote agent tab."""

  title: str | None
  workspace: mirror_pb2.Workspace | None = None


@dataclasses.dataclass(frozen=True)
class RetargetRemote:
  session_id: str


@dataclasses.dataclass(frozen=True)
class RefreshWorkspaces:
  """Discover worktrees added or removed outside this Mirror session."""


@dataclasses.dataclass(frozen=True)
class CycleWorktree:
  """Hand off to the next or previous available worktree."""

  offset: int


@dataclasses.dataclass(frozen=True)
class SelectWorktree:
  """Hand off to one available worktree tab."""

  index: int


@dataclasses.dataclass(frozen=True)
class SelectWorkspace:
  """Hand off to the workspace with this stable identity."""

  identity: str


@dataclasses.dataclass(frozen=True)
class SaveWorktreePreferences:
  """Persist the selected worktree's agent preferences."""

  reasoning_effort: str
  approval_mode: str


@dataclasses.dataclass(frozen=True)
class SendInferenceConfiguration:
  """Apply current inference settings to subsequent requests."""


@dataclasses.dataclass(frozen=True)
class DetachTools:
  """Tell the worker to stop awaiting task-backed tools."""


@dataclasses.dataclass(frozen=True)
class SetWorkerApprovalMode:
  """Set the approval mode used by the worker."""

  mode: str


@dataclasses.dataclass(frozen=True)
class SendApprovalDecision:
  """Answer one worker approval request."""

  call_id: str
  approved: bool
  remember: tool_approvals.ApprovalRule | None = None
  source: agent_pb2.ApprovalSource = agent_pb2.APPROVAL_SOURCE_MANUAL


@dataclasses.dataclass(frozen=True)
class RunSlashCommand:
  """Parse and run one slash command."""

  text: str


@dataclasses.dataclass(frozen=True)
class SendQueuedInstruction:
  """Send an instruction while a turn is already active."""

  instruction: str


@dataclasses.dataclass(frozen=True)
class RunTurn:
  """Start an instruction as a new turn."""

  instruction: str
  rewind_to: str | None = None


@dataclasses.dataclass(frozen=True)
class UserPromptSubmitted:
  instruction: str


@dataclasses.dataclass(frozen=True)
class FocusComposer:
  """Move prompt-toolkit focus to the composer."""


@dataclasses.dataclass(frozen=True)
class FocusTasks:
  """Move prompt-toolkit focus to the task list."""


@dataclasses.dataclass(frozen=True)
class CancelTask:
  """Ask the worker to cancel one task."""

  task_id: str


@dataclasses.dataclass(frozen=True)
class StartNewConversation:
  """Switch to a newly generated conversation id."""


@dataclasses.dataclass(frozen=True)
class CompactConversation:
  """Send a compaction control request."""


@dataclasses.dataclass(frozen=True)
class LoadRewindTurns:
  """Load eligible rewind points for one conversation."""

  conversation_id: str


@dataclasses.dataclass(frozen=True)
class ResolveModel:
  """Resolve one model id to its configured option."""

  model_id: str


@dataclasses.dataclass(frozen=True)
class ReloadResumeChoices:
  """Load conversations for one resume scope."""

  scope: workspace.SessionScope


@dataclasses.dataclass(frozen=True)
class ResolveResumeReference:
  """Find conversations matching one user-provided reference."""

  reference: str


@dataclasses.dataclass(frozen=True)
class ResumeConversation:
  """Resume one selected conversation."""

  info: cli_store.ConversationInfo
  allow_workspace_mismatch: bool = False


@dataclasses.dataclass(frozen=True)
class CancelModelSelection:
  """Cancel the model picker opened from sign-in."""


@dataclasses.dataclass(frozen=True)
class ForgetReflectionKey:
  """Remove the local saved Reflection credential."""


@dataclasses.dataclass(frozen=True)
class SwitchModel:
  """Restart the worker with the selected model."""

  option: models.ModelOption


@dataclasses.dataclass(frozen=True)
class SwitchRemoteModel:
  """Point the remote session at a model without local sign-in."""

  model_id: str
  api: str


@dataclasses.dataclass(frozen=True)
class SaveLastModel:
  """Persist the selected model as the next CLI default."""

  model_id: str


@dataclasses.dataclass(frozen=True)
class SyncSelectorInput:
  """Synchronize the composer control with the declarative selector state."""

  active: bool
  query: str


@dataclasses.dataclass(frozen=True)
class SyncSecretInput:
  """Clear the masked input and show it or the composer."""


@dataclasses.dataclass(frozen=True)
class SubmitSecret:
  """Hand the masked input to the extension that requested it."""


@dataclasses.dataclass(frozen=True)
class CancelSecret:
  """Close the masked prompt and dispatch its cancel action."""


@dataclasses.dataclass(frozen=True)
class LoadPromptHistory:
  """Load prompt recall choices outside the event loop."""


@dataclasses.dataclass(frozen=True)
class LoadSelectorChoices:
  """Run the open selector's async producer and feed back its choices."""

  selector_id: str


@dataclasses.dataclass(frozen=True)
class SetComposerText:
  """Replace text and cursor position in the prompt-toolkit buffer."""

  text: str
  cursor: int | None = None


@dataclasses.dataclass(frozen=True)
class ResetComposer:
  """Reset the prompt-toolkit buffer."""

  append_to_history: bool = False


@dataclasses.dataclass(frozen=True)
class RerenderHistory:
  """Repaint conversation history under an optional divider."""

  note: str | None = None
  drop_scrollback: bool = False
  preserve_header: bool = False


@dataclasses.dataclass(frozen=True)
class RetryWorker:
  """Retry the failed worker."""


@dataclasses.dataclass(frozen=True)
class NavigateHistory:
  """Apply navigation to the view that owns conversation reading positions."""

  action: actions.NavigateHistory


@dataclasses.dataclass(frozen=True)
class InterruptTurn:
  """Ask the worker to interrupt the active turn."""


@dataclasses.dataclass(frozen=True)
class RunExtensionEffect:
  """An effect routed to one extension."""

  extension: str
  payload: object


type CoreEffect = (
  ShowExtensions
  | ChangeExtension
  | RefreshCompletion
  | InsertComposerNewline
  | InsertComposerText
  | EditComposerExternally
  | UndoComposer
  | RedoComposer
  | AcceptSelection
  | AdjustSelection
  | CancelSelection
  | EmitText
  | EmitError
  | RecordError
  | ExitApp
  | OpenFolder
  | CloseWorkspace
  | CreateManagedWorktree
  | DropManagedWorktree
  | RenameWorktree
  | WorkspaceTagCommand
  | LoadRemoteSessions
  | CreateRemoteAgent
  | RefreshWorkspaces
  | RetargetRemote
  | CycleWorktree
  | SelectWorktree
  | SelectWorkspace
  | SaveWorktreePreferences
  | SendInferenceConfiguration
  | DetachTools
  | SetWorkerApprovalMode
  | SendApprovalDecision
  | RunSlashCommand
  | SendQueuedInstruction
  | RunTurn
  | UserPromptSubmitted
  | FocusComposer
  | FocusTasks
  | CancelTask
  | StartNewConversation
  | CompactConversation
  | LoadRewindTurns
  | ResolveModel
  | ReloadResumeChoices
  | ResolveResumeReference
  | ResumeConversation
  | CancelModelSelection
  | ForgetReflectionKey
  | SwitchModel
  | SwitchRemoteModel
  | SaveLastModel
  | SyncSelectorInput
  | SyncSecretInput
  | SubmitSecret
  | CancelSecret
  | LoadPromptHistory
  | LoadSelectorChoices
  | SetComposerText
  | ResetComposer
  | RerenderHistory
  | RetryWorker
  | InterruptTurn
  | NavigateHistory
)

type Effect = CoreEffect | RunExtensionEffect
