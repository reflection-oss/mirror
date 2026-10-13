"""Workspace implementations presented by the CLI supervisor."""

from __future__ import annotations

import asyncio
import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Protocol

from mirror.protocol import agent_pb2

from mirror.cli import actions, models


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
  path: pathlib.Path | None = None
  remote: bool = False
  available: bool = True

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


class Agent(Protocol):
  """One independently running workspace agent and its terminal UI."""

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

  def show_error(self, message: str) -> None: ...

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


@dataclasses.dataclass(frozen=True)
class Directory:
  """An existing directory, with identity independent of its path or VCS."""

  id: str
  path: pathlib.Path
  label: str
  managed: bool = False


@dataclasses.dataclass(frozen=True)
class RemoteSession:
  """A session hosted by a remote Mirror server."""

  id: str


@dataclasses.dataclass(eq=False)
class AgentWorkspace:
  """A folder, worktree or remote workspace with its agent and terminal view."""

  identity: str
  location: Directory | RemoteSession
  agent: Agent
  resume_pending: bool
  alias: str | None = None
  # VCS roster entries, unlike independently opened tabs.
  discovered: bool = False
  dropping: bool = False
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
      else self.location.label
    )

  @property
  def label(self) -> str:
    return self.alias or self.name

  @property
  def path(self) -> pathlib.Path:
    return (
      self.agent.workspace
      if isinstance(self.location, RemoteSession)
      else self.location.path
    )

  @property
  def remote(self) -> bool:
    return isinstance(self.location, RemoteSession)

  @property
  def available(self) -> bool:
    return not self.dropping and (self.remote or self.path.is_dir())

  @property
  def state(self) -> actions.WorktreeAgentState:
    return (
      actions.WorktreeAgentState.SAVED
      if self.resume_pending
      else self.agent.worktree_state
    )

  @property
  def managed(self) -> bool:
    """Whether Mirror owns this local directory and may delete its checkout."""
    return isinstance(self.location, Directory) and self.location.managed

  def drop_confirmation(self) -> actions.DropConfirmation | None:
    return None

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
