"""Workspace implementations presented by the CLI supervisor."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Literal, Protocol, override

from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
)

from mirror.cli import actions, models
from mirror.cli.sandbox import setup as sandbox_setup


class Workspace(Protocol):
  """One selectable workspace, regardless of whether its agent exists yet."""

  identity: str

  @property
  def available(self) -> bool: ...

  @property
  def remote(self) -> bool: ...

  @property
  def label(self) -> str: ...

  @property
  def state(self) -> actions.WorktreeAgentState: ...

  @property
  def path(self) -> pathlib.Path | None: ...

  def show_error(self, message: str) -> None: ...

  async def close(self) -> None: ...


@dataclasses.dataclass(eq=False)
class PendingWorkspace:
  """One creation request, retained on failure and replaced on success."""

  identity: str
  label: str
  error: str | None = None
  notice: str | None = None
  finished: bool = False
  create_session: bool = True
  path: pathlib.Path | None = None
  remote: bool = False
  available: bool = True
  draft: actions.ComposerState = dataclasses.field(
    default_factory=actions.ComposerState
  )
  preferences: daemon_pb2.WorkspacePreferences = dataclasses.field(
    default_factory=daemon_pb2.WorkspacePreferences
  )

  @property
  def state(self) -> actions.WorktreeAgentState:
    return (
      actions.WorktreeAgentState.FAILED
      if self.error is not None
      else actions.WorktreeAgentState.CREATING
    )

  def show_error(self, message: str) -> None:
    self.notice = message

  async def close(self) -> None:
    self.finished = True


@dataclasses.dataclass(eq=False)
class EmptyWorkspace:
  """The landing view, retaining unsent drafts for explicit workspace reopen."""

  identity: str = 'empty'
  label: str = 'Workspaces'
  notice: str = ''
  session_id: str | None = None
  draft: actions.ComposerState = dataclasses.field(
    default_factory=actions.ComposerState
  )
  preferences: daemon_pb2.WorkspacePreferences = dataclasses.field(
    default_factory=daemon_pb2.WorkspacePreferences
  )
  error: str | None = None
  finished: bool = False
  path: pathlib.Path | None = None
  remote: bool = False
  available: bool = False
  state: actions.WorktreeAgentState = actions.WorktreeAgentState.NOT_STARTED

  def show_error(self, message: str) -> None:
    self.notice = message

  async def close(self) -> None:
    self.finished = True


@dataclasses.dataclass(eq=False, kw_only=True)
class SandboxWorkspace(PendingWorkspace):
  """Consent and provisioning state for a sandbox's pending tab."""

  source: AgentWorkspace | None
  name: str | None = None
  options: sandbox_setup.Options = dataclasses.field(
    default_factory=sandbox_setup.Options
  )
  task: asyncio.Task[Workspace] | None = dataclasses.field(
    default=None, repr=False
  )
  phase: str = ''
  setup_action: Literal['install', 'download', 'start', 'copy'] = 'start'

  @property
  def launching(self) -> bool:
    return self.task is not None and not self.task.done()

  @property
  @override
  def state(self) -> actions.WorktreeAgentState:
    return (
      super().state
      if self.launching or self.error is not None
      else actions.WorktreeAgentState.NOT_STARTED
    )


class Agent(Protocol):
  """One independently running workspace agent and its terminal UI."""

  restarting: bool

  def composer_state(self) -> actions.ComposerState: ...

  async def switch_workspace_session(
    self, session_id: str | None = None
  ) -> bool: ...

  @property
  def workspace(self) -> pathlib.Path: ...

  @property
  def reasoning_effort(self) -> str: ...

  @property
  def approval_mode(self) -> str: ...

  @property
  def model(self) -> str: ...

  @property
  def model_selection(self) -> models.ModelOption: ...

  @property
  def conversation_id(self) -> str: ...

  @property
  def activity_status(self) -> str: ...

  @property
  def queued_message_count(self) -> int: ...

  @property
  def worktree_state(self) -> actions.WorktreeAgentState: ...

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]: ...

  def submit_instruction(self, instruction: str) -> None: ...

  def submit_message(self, message: agent_pb2.UserMessage) -> None: ...

  def escalate(self) -> None: ...

  def show_error(self, message: str) -> None: ...

  async def fork_workspace(
    self, session_id: str, directory: pathlib.Path | None, name: str
  ) -> pathlib.Path: ...

  async def start(self) -> None: ...

  async def close(self) -> None: ...


@dataclasses.dataclass(frozen=True)
class AgentSpec:
  """Inputs for constructing one supervised agent."""

  config: Mapping[str, object]
  workspace: pathlib.Path
  reasoning_effort: str
  active: bool
  remote: bool
  extensions: tuple[str, ...] = ()
  tools: tuple[str, ...] | None = None
  workspace_id: str | None = None
  speed: str = 'normal'


@dataclasses.dataclass(frozen=True)
class RemoteSession:
  """A session hosted by a remote Mirror server."""

  id: str


@dataclasses.dataclass(eq=False)
class AgentWorkspace:
  """A folder, worktree or remote workspace with its agent and terminal view."""

  identity: str
  location: daemon_pb2.Workspace | RemoteSession
  agent: Agent
  resume_pending: bool
  alias: str | None = None
  # VCS roster entries, unlike independently opened tabs.
  discovered: bool = False
  dropping: bool = False
  draft: actions.ComposerState | None = None
  _start_task: asyncio.Task[None] | None = dataclasses.field(
    default=None, init=False
  )

  @property
  def id(self) -> str:
    return self.location.id

  @property
  def name(self) -> str:
    return (
      f'remote-{self.location.id[:8]}'
      if isinstance(self.location, RemoteSession)
      else self.location.name
    )

  @property
  def label(self) -> str:
    return self.alias or self.name

  @property
  def path(self) -> pathlib.Path:
    return (
      self.agent.workspace
      if isinstance(self.location, RemoteSession)
      else pathlib.Path(self.location.directory)
    )

  @property
  def remote(self) -> bool:
    return isinstance(self.location, RemoteSession)

  @property
  def available(self) -> bool:
    return not self.dropping and (
      isinstance(self.location, RemoteSession)
      or self.location.state
      in (
        daemon_pb2.WORKSPACE_STATE_AVAILABLE,
        daemon_pb2.WORKSPACE_STATE_CLOSED,
      )
    )

  @property
  def state(self) -> actions.WorktreeAgentState:
    return (
      actions.WorktreeAgentState.SAVED
      if self.resume_pending
      else self.agent.worktree_state
    )

  @property
  def is_sandbox(self) -> bool:
    """Whether tools run in a daemon-owned sandbox environment."""
    return (
      isinstance(self.location, daemon_pb2.Workspace)
      and self.location.kind == daemon_pb2.WORKSPACE_KIND_SANDBOX
    )

  def show_error(self, message: str) -> None:
    self.agent.show_error(message)

  async def start(self) -> asyncio.Task[None]:
    """Start once, letting loading state paint without waiting for readiness."""
    if self._start_task is None:
      self.resume_pending = False
      self._start_task = asyncio.create_task(self.agent.start())
    await asyncio.sleep(0)
    return self._start_task

  async def close(self) -> None:
    if self._start_task is not None:
      self._start_task.cancel()
      await asyncio.gather(self._start_task, return_exceptions=True)
    await self.agent.close()
