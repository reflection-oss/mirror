"""Workspace roster, selection, creation, and persistence."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import pathlib
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal

from mirror.protocol import (
  agent_pb2,
  client_pb2,
  mirror_pb2,
)

from mirror import config
from mirror import workspace as workspace_identity
from mirror.cli import (
  actions,
  agents,
  cli_store,
  models,
  onboarding,
  workspace_tags,
  workspaces,
  worktrees,
)
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event


@dataclasses.dataclass(frozen=True)
class AgentSnapshot:
  state: actions.WorktreeAgentState
  model: str
  reasoning_effort: str
  conversation_id: str
  status: str | None
  queued_messages: int


@dataclasses.dataclass(frozen=True)
class WorkspaceSnapshot:
  id: str
  name: str
  path: pathlib.Path
  current: bool
  agent: AgentSnapshot
  tag: workspace_tags.WorkspaceTag | None = None
  tag_id: str | None = None
  kind: Literal['folder', 'worktree'] = 'worktree'
  cwd: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class WorkspaceList:
  workspaces: tuple[WorkspaceSnapshot, ...]
  creating: str | None


@dataclasses.dataclass(frozen=True)
class MessageSnapshot:
  timestamp: float
  role: Literal['user', 'assistant']
  content: str


@dataclasses.dataclass(frozen=True)
class WorkspaceInspection:
  workspace: WorkspaceSnapshot
  messages: tuple[MessageSnapshot, ...]


@dataclasses.dataclass(frozen=True)
class WorkspaceSubmission:
  workspace: WorkspaceSnapshot
  status: Literal['created', 'accepted']
  cursor: float | None = None


def _drop_worktree(worktree: worktrees.Worktree, confirmed: bool) -> None:
  worktrees.drop_worktree(worktree, confirmed=confirmed)


def _transcript_messages(
  workspace: workspaces.AgentWorkspace,
) -> tuple[MessageSnapshot, ...]:
  messages: list[MessageSnapshot] = []
  for entry in workspace.agent.transcript_entries():
    if entry.WhichOneof('payload') != 'chat':
      continue
    chat = entry.chat
    role = chat.WhichOneof('message')
    if role == 'user' and not chat.HasField('display'):
      content = chat.user.content
    elif role == 'assistant':
      assistant = chat.assistant
      content = (
        assistant.content
        if assistant.HasField('content') and assistant.content
        else (
          assistant.refusal
          if assistant.HasField('refusal') and assistant.refusal
          else ''
        )
      )
    else:
      continue
    if content:
      messages.append(
        MessageSnapshot(
          timestamp=entry.timestamp.seconds
          + entry.timestamp.nanos / 1_000_000_000,
          role=role,
          content=content,
        )
      )
  return tuple(messages)


class WorkspaceController:
  """Own the workspace roster and its backend operations, independently of views."""

  def __init__(
    self,
    worker_config: Mapping[str, object],
    *,
    workspace: pathlib.Path,
    reasoning_effort: str,
    create_agent: Callable[
      [workspaces.AgentSpec, WorkspaceController], workspaces.Agent
    ],
    database: str | pathlib.Path | None = None,
    settings: config.Config | None = None,
    continue_recent: bool = False,
    display_unmanaged_worktrees: bool = False,
    remote_enabled: bool = False,
    list_remote_sessions: Callable[[], Awaitable[Sequence[str]]] | None = None,
    delete_remote_session: Callable[[str], Awaitable[None]] | None = None,
    discover: Callable[[pathlib.Path], Sequence[worktrees.Worktree]] = (
      worktrees.list_worktrees
    ),
    create: Callable[[pathlib.Path, str | None], worktrees.Worktree] = (
      worktrees.create_worktree
    ),
    drop: Callable[[worktrees.Worktree, bool], None] = _drop_worktree,
    is_trusted_directory: Callable[
      [pathlib.Path], bool
    ] = onboarding.is_trusted_directory,
  ) -> None:
    self._base_config = dict(worker_config)
    self._reasoning_effort = reasoning_effort
    self._create_agent = create_agent
    self._database = database
    self._settings = settings or config.Config(config.default_path())
    self._tags = workspace_tags.definitions(self._settings.read())
    self._continue_recent = continue_recent
    self._display_unmanaged_worktrees = display_unmanaged_worktrees
    self._remote_enabled = remote_enabled
    self._list_remote_sessions = list_remote_sessions
    self._discover = discover
    self._create = create
    self._drop = drop
    self._last_running_count: int | None = None
    self._is_trusted_directory = is_trusted_directory
    self._delete_remote_session = delete_remote_session
    self._roster_order_established = False
    self._reported_workspace_count: tuple[int, int] | None = None
    self._reconcile_task: asyncio.Task[None] | None = None
    self._remote_session_task: asyncio.Task[None] | None = None
    self._message_delivery_tasks: set[asyncio.Task[None]] = set()
    self._worktree_creation_tasks: set[
      asyncio.Task[workspaces.AgentWorkspace]
    ] = set()
    self._workspace_deletion_tasks: dict[
      asyncio.Task[None], workspaces.AgentWorkspace
    ] = {}
    self._shutting_down = False
    launch_directory = workspace.resolve()
    self.launch_workspace = launch_directory
    current = worktrees.current_worktree(launch_directory)
    self._launch_worktree_path = current.path
    self._repository_key = workspace_identity.git_workspace_key(
      str(current.path)
    ) or str(current.path)
    with cli_store.Store(self._database) as store:
      self._overlays = store.worktree_overlays(self._repository_key)
      self._closed_workspaces = store.closed_workspaces(self._repository_key)
      saved = store.saved_workspaces(self._repository_key)
    self._migrate_overlay(current)
    self._relative_workspace = launch_directory.relative_to(current.path)
    overlay = self._overlays.get(current.id, cli_store.WorktreeOverlay())
    self.workspaces: list[workspaces.Workspace] = [
      self._new_workspace(
        current,
        worker_config,
        agent_workspace=launch_directory,
        active=True,
        alias=overlay.alias,
      )
    ]
    self.selected = self.workspaces[0]
    for workspace_id, path, name in saved:
      directory = pathlib.Path(path)
      if not self._is_trusted_directory(directory):
        continue
      self.workspaces.append(
        self._new_workspace(
          workspaces.Directory(
            workspace_id,
            directory,
            name,
            worktrees.is_managed_directory(directory),
          ),
          self._new_worker_config(),
          agent_workspace=directory,
          alias=self._overlays.get(
            workspace_id, cli_store.WorktreeOverlay()
          ).alias,
        )
      )
    self.report_workspace_count()

  def running_sessions_changed(self) -> None:
    """Log the number of running workspace tabs if it differs from the last."""
    if not statsig_client.get().enabled:
      self._last_running_count = None
      return
    running = sum(
      workspace.state is actions.WorktreeAgentState.RUNNING
      for workspace in self.visible_workspaces()
    )
    if running == self._last_running_count:
      return
    self._last_running_count = running
    statsig_client.get().log_event(
      statsig_event.Event('mirror_running_sessions', value=running)
    )

  def _migrate_overlay(self, worktree: worktrees.Worktree) -> None:
    legacy_id = workspace_identity.resolve_cwd(worktree.path)
    if worktree.id not in self._overlays and legacy_id in self._overlays:
      with cli_store.Store(self._database) as store:
        store.move_worktree_overlay(
          self._repository_key, legacy_id, worktree.id
        )
      self._overlays[worktree.id] = self._overlays.pop(legacy_id)

  def _new_worker_config(
    self,
    source: workspaces.Agent | None = None,
    *,
    workspace: pathlib.Path | None = None,
  ) -> dict[str, object]:
    approval_mode = (
      source.approval_mode
      if source is not None
      else self._base_config.get('approval_mode', 'approve')
    )
    config = self._base_config | {
      'conversation_id': uuid.uuid4().hex,
      'create_session': True,
      'allow_workspace_mismatch': False,
      'approval_mode': approval_mode,
    }
    if source is None and self._continue_recent and workspace is not None:
      with cli_store.Store(self._database) as store:
        recent = store.latest(cwd=str(workspace))
      if recent is not None:
        config.update({'conversation_id': recent.id, 'create_session': False})
    selection = (
      source.model_selection
      if source is not None
      else models.load_last_selection()
    )
    if selection is not None:
      config.update({
        'model': selection.model_id,
        'api': selection.api,
        'model_api': selection.api,
        'base_url': None,
        'model_base_url': selection.base_url or None,
      })
    return config

  def _new_workspace(
    self,
    location: worktrees.Worktree
    | workspaces.Directory
    | workspaces.RemoteSession,
    config: Mapping[str, object],
    *,
    agent_workspace: pathlib.Path,
    reasoning_effort: str | None = None,
    active: bool = False,
    alias: str | None = None,
  ) -> workspaces.AgentWorkspace:
    remote = isinstance(location, workspaces.RemoteSession)
    discovered = isinstance(location, worktrees.Worktree)
    provider = (
      location.provider
      if isinstance(location, worktrees.Worktree)
      else 'remote'
      if remote
      else 'workspace'
    )
    if isinstance(location, worktrees.Worktree):
      location = workspaces.Directory(
        location.id,
        location.path,
        location.label,
        worktrees.is_managed(location),
      )
    workspace_id = location.id
    overlay = self._overlays.get(workspace_id, cli_store.WorktreeOverlay())
    preferences = overlay.preferences
    config = dict(config)
    if preferences.HasField('model') and not (
      active and config.get('model_selection_override')
    ):
      selection = models.find_model(preferences.model)
      config.update({
        'model': preferences.model,
        'model_api': preferences.model_api
        if preferences.HasField('model_api')
        else selection.api or None,
        'model_base_url': preferences.model_base_url or None
        if preferences.HasField('model_base_url')
        else selection.base_url or None,
        'base_url': None,
      })
      if remote:
        config['remote_model_api'] = preferences.model_api
    if preferences.HasField('system_prompt'):
      config['system_prompt'] = preferences.system_prompt
    extension_tools = None
    if preferences.HasField('tools_allowlist'):
      builtins = agents.builtins()
      config['tools'] = tuple(
        name for name in preferences.tools_allowlist.names if name in builtins
      )
      extension_tools = tuple(
        name
        for name in preferences.tools_allowlist.names
        if name not in builtins
      )
    if preferences.HasField('approval_mode') and not (
      active and config.get('approval_mode_override')
    ):
      config['approval_mode'] = (
        'auto'
        if preferences.approval_mode == agent_pb2.APPROVAL_MODE_AUTO
        else 'approve'
      )
    if preferences.HasField('reasoning_effort'):
      reasoning_effort = preferences.reasoning_effort
    agent = self._create_agent(
      workspaces.AgentSpec(
        config=config,
        workspace=agent_workspace,
        reasoning_effort=reasoning_effort or self._reasoning_effort,
        active=active,
        remote=remote,
        workspace_id=workspace_id if not remote else None,
        extensions=tuple(preferences.extensions),
        tools=extension_tools,
      ),
      self,
    )
    if (
      not remote
      and not preferences.HasField('model')
      and not (active and config.get('model_selection_override'))
    ):
      self._save_model(workspace_id, agent, agent.model_selection)
    return workspaces.AgentWorkspace(
      identity=f'{provider}:{workspace_id}',
      location=location,
      discovered=discovered,
      agent=agent,
      resume_pending=not bool(config.get('create_session', False)),
      alias=alias,
    )

  def _new_discovered_workspace(
    self, worktree: worktrees.Worktree
  ) -> workspaces.AgentWorkspace:
    agent_workspace = worktree.path / self._relative_workspace
    overlay = self._overlays.get(worktree.id, cli_store.WorktreeOverlay())
    config = self._new_worker_config(workspace=agent_workspace)
    return self._new_workspace(
      worktree,
      config,
      agent_workspace=agent_workspace,
      alias=overlay.alias,
    )

  def _source_workspace(
    self, source: pathlib.Path | str
  ) -> workspaces.AgentWorkspace:
    if isinstance(source, str):
      return self._target_workspace(source)
    matches = [
      workspace
      for workspace in self._agent_workspaces()
      if not workspace.remote
      and workspace.available
      and workspace.agent.workspace == source
    ]
    if len(matches) > 1:
      raise worktrees.WorktreeError(
        'source directory is ambiguous; use its workspace id'
      )
    if not matches:
      raise worktrees.WorktreeError(
        f'source workspace is unavailable for directory {str(source)!r}'
      )
    return matches[0]

  def _target_workspace(self, workspace_id: str) -> workspaces.AgentWorkspace:
    available = [
      workspace
      for workspace in self._agent_workspaces()
      if not workspace.remote and workspace.available
    ]
    for workspace in available:
      if workspace.id == workspace_id:
        return workspace
    matches = [
      workspace
      for workspace in available
      if workspace.label.casefold() == workspace_id.casefold()
    ]
    if len(matches) > 1:
      raise worktrees.WorktreeError(
        f'workspace {workspace_id!r} is ambiguous; use its id'
      )
    if not matches:
      raise worktrees.WorktreeError(f'workspace {workspace_id!r} not found')
    return matches[0]

  def _workspace_snapshot(
    self, workspace: workspaces.AgentWorkspace, source: pathlib.Path | str
  ) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
      id=workspace.id,
      name=workspace.label,
      tag=self.workspace_tag(workspace.id),
      tag_id=self._overlays.get(
        workspace.id, cli_store.WorktreeOverlay()
      ).tag_id,
      path=workspace.path,
      current=workspace is self._source_workspace(source),
      kind='worktree' if workspace.managed else 'folder',
      cwd=workspace.agent.workspace,
      agent=AgentSnapshot(
        state=workspace.state,
        model=workspace.agent.model,
        reasoning_effort=workspace.agent.reasoning_effort,
        conversation_id=workspace.agent.conversation_id,
        status=workspace.agent.activity_status or None,
        queued_messages=workspace.agent.queued_message_count,
      ),
    )

  def visible_workspaces(self) -> tuple[workspaces.Workspace, ...]:
    """Return the visible workspace objects, not a second representation."""
    return tuple(
      workspace for workspace in self.workspaces if workspace.available
    )

  def report_workspace_count(self) -> None:
    """Report changed visible counts; forget the observation while disabled."""
    client = statsig_client.get()
    if not client.enabled:
      self._reported_workspace_count = None
      return
    workspaces = self._agent_workspaces()
    local_count = sum(1 for w in workspaces if w.available and not w.remote)
    remote_count = sum(1 for w in workspaces if w.available and w.remote)
    counts = (local_count, remote_count)
    if counts == self._reported_workspace_count:
      return
    self._reported_workspace_count = counts

    client.log_event(
      statsig_event.Event(
        'mirror_workspace_count',
        value=local_count,
        metadata={'location': 'local'},
      )
    )
    client.log_event(
      statsig_event.Event(
        'mirror_workspace_count',
        value=remote_count,
        metadata={'location': 'remote'},
      )
    )

  def _agent_workspaces(self) -> list[workspaces.AgentWorkspace]:
    return [
      workspace
      for workspace in self.workspaces
      if isinstance(workspace, workspaces.AgentWorkspace)
    ]

  def _selected_agent_workspace(self) -> workspaces.AgentWorkspace:
    workspace = self.selected
    if not isinstance(workspace, workspaces.AgentWorkspace):
      raise worktrees.WorktreeError('this workspace is not ready')
    return workspace

  async def _reconcile(self) -> None:
    """Refresh the cached roster without replacing the active agent."""
    # Directory availability may change without a worker state transition.
    self.report_workspace_count()
    self.running_sessions_changed()
    discovery_path = next(
      (
        workspace.path
        for workspace in self._agent_workspaces()
        if workspace.discovered and workspace.available
      ),
      None,
    )
    if discovery_path is None:
      return
    try:
      discovered = await asyncio.to_thread(self._discover, discovery_path)
    except (OSError, worktrees.WorktreeError):
      return
    # Git discovery puts the primary checkout first; jj has a stable default ID.
    primary = next(
      (
        item.path
        for item in discovered
        if item.provider == 'jj' and item.id == 'default'
      ),
      None,
    )
    if discovered and discovered[0].provider == 'git':
      primary = discovered[0].path
    old_workspaces = tuple(self.workspaces)
    selected = self.selected
    existing = {
      workspace.path: workspace
      for workspace in self._agent_workspaces()
      if workspace.discovered
    }
    refreshed: list[workspaces.Workspace] = []
    for item in discovered:
      if item.id in self._closed_workspaces:
        continue
      if (
        not self._display_unmanaged_worktrees
        and item.path != self._launch_worktree_path
        and item.path != primary
        and not worktrees.is_managed(item)
      ):
        continue
      self._migrate_overlay(item)
      workspace = existing.get(item.path)
      if workspace is None:
        workspace = self._new_discovered_workspace(item)
      else:
        workspace.location = workspaces.Directory(
          item.id, item.path, item.label, worktrees.is_managed(item)
        )
      refreshed.append(workspace)
    refreshed.extend(
      workspace
      for workspace in old_workspaces
      if isinstance(workspace, workspaces.PendingWorkspace)
      or workspace.remote
      or (
        isinstance(workspace, workspaces.AgentWorkspace)
        and not workspace.discovered
      )
    )
    if selected not in refreshed:
      refreshed.append(selected)
    if self._roster_order_established:
      positions = {
        workspace.identity: index
        for index, workspace in enumerate(old_workspaces)
      }
      refreshed.sort(
        key=lambda workspace: (
          workspace.remote,
          positions.get(workspace.identity, len(old_workspaces)),
        )
      )
    self._roster_order_established = True
    self.workspaces[:] = refreshed
    self.report_workspace_count()
    self.running_sessions_changed()
    removed = [
      workspace for workspace in old_workspaces if workspace not in refreshed
    ]
    if removed:
      await asyncio.gather(*(workspace.close() for workspace in removed))

  def _reconcile_done(self, task: asyncio.Task[None]) -> None:
    """Consume background discovery failures."""
    if self._reconcile_task is task:
      self._reconcile_task = None
    if task.cancelled():
      return
    if error := task.exception():
      self.selected.show_error(f'could not reconcile worktrees: {error}')

  def _schedule_reconcile(self) -> None:
    if (
      self._shutting_down
      or self._reconcile_task is not None
      or self._worktree_creation_tasks
      or self._workspace_deletion_tasks
    ):
      return
    task = asyncio.create_task(self._reconcile())
    self._reconcile_task = task
    task.add_done_callback(self._reconcile_done)

  def _workspace_at(self, tab_index: int) -> workspaces.Workspace | None:
    available = self.visible_workspaces()
    return available[tab_index] if 0 <= tab_index < len(available) else None

  def can_select(self, tab_index: int) -> bool:
    """Whether a visible workspace exists and is not already selected."""
    target = self._workspace_at(tab_index)
    return target is not None and target is not self.selected

  def can_cycle(self) -> bool:
    """Whether cycling can select a different visible workspace."""
    return len(self.visible_workspaces()) > 1

  def set_display_unmanaged_worktrees(self, enabled: bool) -> None:
    """Refresh the roster when the display preference changes."""
    if self._display_unmanaged_worktrees != enabled:
      self._display_unmanaged_worktrees = enabled
      self.refresh()

  async def initialize(self) -> None:
    """Finish initial discovery before presenting the workspace roster."""
    self._schedule_reconcile()
    if self._reconcile_task is not None:
      # `_reconcile_done` reports failures while retaining the launch workspace.
      await asyncio.gather(self._reconcile_task, return_exceptions=True)

  def refresh(self) -> None:
    """Schedule discovery of externally added or removed worktrees."""
    self._schedule_reconcile()

  def list_workspaces(self, source: pathlib.Path | str) -> WorkspaceList:
    self._source_workspace(source)
    return WorkspaceList(
      workspaces=tuple(
        self._workspace_snapshot(workspace, source)
        for workspace in self._agent_workspaces()
        if not workspace.remote and workspace.available
      ),
      creating=', '.join(
        tab.label
        for tab in self.workspaces
        if tab.state is actions.WorktreeAgentState.CREATING
      )
      or None,
    )

  async def send_message(
    self, source: pathlib.Path | str, workspace_id: str, message: str
  ) -> WorkspaceSubmission:
    sender = self._source_workspace(source)
    target = self._target_workspace(workspace_id)
    if target is sender:
      raise worktrees.WorktreeError('cannot send a message to this workspace')
    messages = _transcript_messages(target)
    cursor = messages[-1].timestamp if messages else None
    instruction = (
      f'Message from the agent in workspace {sender.label!r}. '
      'Complete this request in your workspace and leave the response here '
      'for the sending agent to inspect.\n\n'
      f'{message}'
    )
    task = asyncio.create_task(
      self._deliver_message(sender, target, instruction)
    )
    self._message_delivery_tasks.add(task)
    task.add_done_callback(self._message_delivery_tasks.discard)
    return WorkspaceSubmission(
      workspace=self._workspace_snapshot(target, source),
      cursor=cursor,
      status='accepted',
    )

  async def _deliver_message(
    self,
    source: workspaces.AgentWorkspace,
    target: workspaces.AgentWorkspace,
    instruction: str,
  ) -> None:
    try:
      startup = await target.start()
      await startup
      target.agent.submit_instruction(instruction)
    except (
      Exception
    ) as error:  # Report failures across the background delivery boundary.
      source.agent.submit_instruction(
        f'Message delivery to workspace {target.id!r} failed: {error}'
      )

  def inspect_workspace(
    self, source: pathlib.Path | str, workspace_id: str
  ) -> WorkspaceInspection:
    self._source_workspace(source)
    target = self._target_workspace(workspace_id)
    return WorkspaceInspection(
      workspace=self._workspace_snapshot(target, source),
      messages=_transcript_messages(target),
    )

  def list_workspace_tags(self) -> tuple[workspace_tags.WorkspaceTag, ...]:
    """Expose ordered definitions to extensions and workspace API consumers."""
    self._tags = workspace_tags.definitions(self._settings.read())
    return self._tags

  def save_workspace_tag(self, tag: workspace_tags.WorkspaceTag) -> None:
    """Create or update a label by its stable ID."""
    workspace_tags.save(self._settings, tag)
    self.list_workspace_tags()

  def delete_workspace_tag(self, tag_id: str) -> None:
    """Delete a label and clear its assignments, including closed workspaces."""
    workspace_tags.delete(self._settings, tag_id)
    self.list_workspace_tags()
    with cli_store.Store(self._database) as store:
      store.delete_workspace_tag(tag_id)
      self._overlays = store.worktree_overlays(self._repository_key)

  def workspace_tag(
    self, workspace_id: str
  ) -> workspace_tags.WorkspaceTag | None:
    """Resolve the label; an unknown persisted ID remains unmodified."""
    tag_id = self._overlays.get(
      workspace_id, cli_store.WorktreeOverlay()
    ).tag_id
    if tag_id is None:
      return None
    return next((tag for tag in self._tags if tag.id == tag_id), None)

  def set_workspace_tag(self, workspace_id: str, tag_id: str | None) -> None:
    """Assign/clear by exact workspace ID, never by alias or roster position."""
    if not any(
      workspace.id == workspace_id and workspace.available
      for workspace in self._agent_workspaces()
    ):
      raise ValueError(f'workspace {workspace_id!r} is unavailable')
    if tag_id is not None and not any(
      tag.id == tag_id for tag in self.list_workspace_tags()
    ):
      raise ValueError(f'unknown workspace tag {tag_id!r}')
    with cli_store.Store(self._database) as store:
      store.set_workspace_tag(self._repository_key, workspace_id, tag_id)
      self._overlays = store.worktree_overlays(self._repository_key)

  def cycle_workspace_tag(
    self, *, workspace_id: str | None = None
  ) -> workspace_tags.WorkspaceTag | None:
    """Cycle an exact workspace ID, or the active workspace when omitted."""
    if workspace_id is None:
      workspace_id = self._selected_agent_workspace().id
    if not any(
      workspace.id == workspace_id and workspace.available
      for workspace in self._agent_workspaces()
    ):
      raise ValueError(f'workspace {workspace_id!r} is unavailable')
    current = self._overlays.get(
      workspace_id, cli_store.WorktreeOverlay()
    ).tag_id
    tags = self.list_workspace_tags()
    if current is None and not tags:
      return None
    choices = [None, *(tag.id for tag in tags)]
    # Unknown persisted IDs clear on an explicit cycle, never on load.
    next_id = (
      choices[(choices.index(current) + 1) % len(choices)]
      if current in choices
      else None
    )
    self.set_workspace_tag(workspace_id, next_id)
    return self.workspace_tag(workspace_id)

  def tab_tags(self) -> dict[str, workspace_tags.WorkspaceTag]:
    """Project metadata onto UI identities without changing the roster."""
    return {
      workspace.identity: tag
      for workspace in self._agent_workspaces()
      if (tag := self.workspace_tag(workspace.id)) is not None
    }

  def rename(self, alias: str) -> str:
    """Alias the selected worktree without changing its VCS identifier."""
    source = self._selected_agent_workspace()
    normalized = ' '.join(alias.split())
    for workspace in self._agent_workspaces():
      if workspace is source:
        continue
      names = (workspace.id, workspace.name, workspace.alias)
      if any(
        name is not None and name.casefold() == normalized.casefold()
        for name in names
      ):
        raise ValueError(f'worktree alias {normalized!r} is already in use')
    overlay = self._overlays.get(source.id, cli_store.WorktreeOverlay())
    with cli_store.Store(self._database) as store:
      normalized = store.set_worktree_alias(
        self._repository_key,
        source.id,
        normalized,
      )
    source.alias = normalized
    self._overlays[source.id] = dataclasses.replace(overlay, alias=normalized)
    return normalized

  def _save_preferences(
    self,
    workspace_id: str,
    reasoning_effort: str,
    approval_mode: str,
  ) -> None:
    try:
      with cli_store.Store(self._database) as store:
        overlay = store.worktree_overlays(self._repository_key).get(
          workspace_id,
          self._overlays.get(workspace_id, cli_store.WorktreeOverlay()),
        )
        overlay.preferences.reasoning_effort = reasoning_effort
        overlay.preferences.approval_mode = (
          agent_pb2.APPROVAL_MODE_AUTO
          if approval_mode == 'auto'
          else agent_pb2.APPROVAL_MODE_HITL
        )
        store.set_worktree_preferences(
          self._repository_key, workspace_id, overlay.preferences
        )
        self._overlays[workspace_id] = overlay
    except (OSError, sqlite3.Error):
      return

  def save_preferences(
    self,
    reasoning_effort: str,
    approval_mode: str,
    *,
    agent: workspaces.Agent | None = None,
  ) -> None:
    """Persist preferences for the emitting agent or selected workspace."""
    source = (
      self._selected_agent_workspace()
      if agent is None
      else next(
        workspace
        for workspace in self._agent_workspaces()
        if workspace.agent is agent
      )
    )
    self._save_preferences(source.id, reasoning_effort, approval_mode)

  def save_model(
    self, agent: workspaces.Agent, selection: models.ModelOption
  ) -> None:
    """Persist a selection on its owning workspace, not the currently active tab."""
    source = next(
      workspace
      for workspace in self._agent_workspaces()
      if workspace.agent is agent
    )
    self._save_model(source.id, agent, selection)

  def _save_model(
    self,
    workspace_id: str,
    agent: workspaces.Agent,
    selection: models.ModelOption,
  ) -> None:
    try:
      with cli_store.Store(self._database) as store:
        overlay = store.worktree_overlays(self._repository_key).get(
          workspace_id, cli_store.WorktreeOverlay()
        )
        overlay.preferences.model = selection.model_id
        overlay.preferences.model_api = selection.api
        overlay.preferences.model_base_url = selection.base_url
        store.set_worktree_preferences(
          self._repository_key, workspace_id, overlay.preferences
        )
      self._overlays[workspace_id] = overlay
    except (OSError, sqlite3.Error) as error:
      agent.show_error(f'could not save model preference: {error}')

  def retarget_remote(self, agent: workspaces.Agent, session_id: str) -> None:
    """Keep a remote agent's tab and preferences on its selected session."""
    source = next(
      (
        workspace
        for workspace in self._agent_workspaces()
        if workspace.agent is agent
      ),
      None,
    )
    if source is None or not source.remote or source.id == session_id:
      return
    previous_id = source.id
    source.location = workspaces.RemoteSession(session_id)
    overlay = self._overlays.pop(previous_id, None)
    if overlay is not None:
      self._overlays[session_id] = overlay
    try:
      with cli_store.Store(self._database) as store:
        store.move_worktree_overlay(
          self._repository_key, previous_id, session_id
        )
    except (OSError, sqlite3.Error) as error:
      source.agent.show_error(f'could not save session preferences: {error}')

  async def select(self, tab_index: int) -> bool:
    """Select a visible worktree by its zero-based tab position."""
    target = self._workspace_at(tab_index)
    if target is None or target is self.selected:
      return False
    self.selected = target
    self._schedule_reconcile()
    return True

  async def select_workspace(self, identity: str) -> bool:
    """Resolve a picker result against the current roster at activation time."""
    target = next(
      (
        workspace
        for workspace in self.visible_workspaces()
        if identity == workspace.identity
        or (
          isinstance(workspace, workspaces.AgentWorkspace)
          and identity == workspace.id
        )
      ),
      None,
    )
    if target is None:
      self.selected.show_error(
        'That workspace is no longer available. Reopen /workspaces.'
      )
      return False
    if target is self.selected:
      return False
    self.selected = target
    self._schedule_reconcile()
    return True

  async def cycle(self, offset: int) -> None:
    """Select a cached workspace and reconcile the roster in the background."""
    available = self.visible_workspaces()
    if self.selected in available:
      if len(available) < 2:
        return
      self.selected = available[
        (available.index(self.selected) + offset) % len(available)
      ]
    elif available:
      index = self.workspaces.index(self.selected)
      candidates = (
        self.workspaces[index + 1 :]
        if offset > 0
        else reversed(self.workspaces[:index])
      )
      self.selected = next(
        (workspace for workspace in candidates if workspace.available),
        available[0] if offset > 0 else available[-1],
      )
    self._schedule_reconcile()

  def drop_confirmation(self) -> actions.DropConfirmation | None:
    """Describe the selected workspace if confirmation is required."""
    workspace = self.selected
    if isinstance(workspace, workspaces.AgentWorkspace):
      return workspace.drop_confirmation()
    return None

  async def drop(self, *, confirmed: bool = False) -> bool:
    """Start deleting the selected workspace without blocking its terminal UI."""
    source = self.selected
    if (
      isinstance(source, workspaces.AgentWorkspace)
      and not source.remote
      and not source.managed
    ):
      source.show_error('Folder workspaces cannot be deleted; use /close.')
      return False
    await self._cancel_reconcile()
    if source is not self.selected:
      return False
    available = tuple(
      workspace
      for workspace in self.visible_workspaces()
      if workspace is not source
    )
    if not available or (
      isinstance(source, workspaces.AgentWorkspace)
      and not any(
        isinstance(other, workspaces.AgentWorkspace) for other in available
      )
    ):
      source.show_error('cannot drop the only available worktree')
      return False
    if isinstance(source, workspaces.PendingWorkspace):
      if source.error is None:
        return False
    elif not isinstance(source, workspaces.AgentWorkspace):
      return False
    self.selected = next(
      (
        workspace
        for workspace in self.workspaces[self.workspaces.index(source) + 1 :]
        if workspace.available
      ),
      available[-1],
    )
    if isinstance(source, workspaces.PendingWorkspace):
      self.workspaces.remove(source)
      await source.close()
      return True
    source.dropping = True
    self.report_workspace_count()
    self.running_sessions_changed()
    task = asyncio.create_task(self._delete_workspace(source, confirmed))
    self._workspace_deletion_tasks[task] = source
    task.add_done_callback(self._workspace_deletion_done)
    await asyncio.sleep(0)
    return True

  async def _delete_workspace(
    self, source: workspaces.AgentWorkspace, confirmed: bool
  ) -> None:
    """Finish deletion and remove its workspace from the roster."""
    try:
      if isinstance(source.location, workspaces.RemoteSession):
        if self._delete_remote_session is None:
          raise RuntimeError('remote workspace provider is not configured')
        await self._delete_remote_session(source.location.id)
      elif source.managed:
        checkout = await asyncio.to_thread(
          worktrees.current_worktree, source.path
        )
        if checkout.path != source.path or not worktrees.is_managed(checkout):
          raise RuntimeError('directory is no longer a managed checkout')
        await asyncio.to_thread(self._drop, checkout, confirmed)
        statsig_client.get().log_event(
          statsig_event.Event('mirror_worktree_dropped', value=1)
        )
      else:
        raise RuntimeError('folder directories cannot be deleted')
    except (OSError, RuntimeError) as error:
      source.agent.show_error(str(error))
      return
    self.workspaces.remove(source)
    self.report_workspace_count()
    try:
      with cli_store.Store(self._database) as store:
        store.delete_worktree_overlay(self._repository_key, source.id)
      self._overlays.pop(source.id, None)
    finally:
      await source.close()

  def _workspace_deletion_done(self, task: asyncio.Task[None]) -> None:
    """Consume deletion failures and resume roster reconciliation."""
    workspace = self._workspace_deletion_tasks.pop(task)
    workspace.dropping = False
    self.report_workspace_count()
    self.running_sessions_changed()
    if not task.cancelled() and (error := task.exception()) is not None:
      target = workspace if workspace in self.workspaces else self.selected
      target.show_error(f'could not drop worktree: {error}')
    self._schedule_reconcile()

  def _local_source(self) -> workspaces.AgentWorkspace:
    selected = self.selected
    local = [
      workspace
      for workspace in self._agent_workspaces()
      if not workspace.remote and workspace.available
    ]
    if not local:
      raise worktrees.WorktreeError('no local workspace is available')
    return (
      selected
      if selected in local and isinstance(selected, workspaces.AgentWorkspace)
      else local[0]
    )

  def _insert_local(self, workspace: workspaces.Workspace) -> None:
    self._roster_order_established = True
    index = next(
      (
        index
        for index, existing in enumerate(self.workspaces)
        if existing.remote
      ),
      len(self.workspaces),
    )
    self.workspaces.insert(index, workspace)
    self.report_workspace_count()
    self.running_sessions_changed()

  async def _create_configured_workspace(
    self,
    pending: workspaces.PendingWorkspace,
    source: workspaces.AgentWorkspace,
    name: str | None,
    preferences: client_pb2.WorkspacePreferences | None = None,
  ) -> workspaces.AgentWorkspace:
    source_root = (
      worktrees.current_worktree(source.path).path
      if not source.discovered
      else source.path
    )
    relative_workspace = source.agent.workspace.relative_to(source_root)
    if preferences is None:
      preferences = client_pb2.WorkspacePreferences()
      source_preferences = self._overlays.get(
        source.id, cli_store.WorktreeOverlay()
      ).preferences
      if source_preferences.HasField('tools_allowlist'):
        preferences.tools_allowlist.CopyFrom(source_preferences.tools_allowlist)
    else:
      requested = client_pb2.WorkspacePreferences()
      requested.CopyFrom(preferences)
      preferences = requested
    if not preferences.HasField('model'):
      selection = source.agent.model_selection
      preferences.model = selection.model_id
      preferences.model_api = selection.api
      preferences.model_base_url = selection.base_url
    if not preferences.HasField('reasoning_effort'):
      preferences.reasoning_effort = source.agent.reasoning_effort
    if not preferences.HasField('approval_mode'):
      preferences.approval_mode = (
        agent_pb2.APPROVAL_MODE_AUTO
        if source.agent.approval_mode == 'auto'
        else agent_pb2.APPROVAL_MODE_HITL
      )
    created = await asyncio.to_thread(self._create, source.path, name)
    self._overlays[created.id] = cli_store.WorktreeOverlay(
      preferences=preferences
    )
    workspace = self._new_workspace(
      created,
      self._new_worker_config(source.agent),
      reasoning_effort=source.agent.reasoning_effort,
      agent_workspace=created.path / relative_workspace,
    )
    workspace.identity = pending.identity
    self.workspaces[self.workspaces.index(pending)] = workspace
    self.report_workspace_count()
    self.running_sessions_changed()
    if self.selected is pending:
      self.selected = workspace
    pending.finished = True
    try:
      with cli_store.Store(self._database) as store:
        store.set_worktree_preferences(
          self._repository_key, created.id, preferences
        )
    except (OSError, sqlite3.Error) as error:
      raise worktrees.WorktreeError(
        f'Created workspace {created.id!r} at {created.path}, but could not save '
        f'its configuration: {error}. The tab retains the requested configuration '
        'for this CLI session only; save it successfully before restarting.'
      ) from error
    return workspace

  def _start_worktree_creation(
    self,
    source: workspaces.AgentWorkspace,
    name: str | None,
    preferences: client_pb2.WorkspacePreferences | None = None,
  ) -> asyncio.Task[workspaces.AgentWorkspace]:
    if self._shutting_down:
      raise worktrees.WorktreeError('cannot create a worktree during shutdown')
    if not source.discovered:
      repository_key = workspace_identity.git_workspace_key(str(source.path))
      if repository_key != self._repository_key:
        raise worktrees.WorktreeError(
          'Worktree provisioning is scoped to the launch repository; '
          'use a folder workspace for other directories.'
        )
    name = name.strip() or None if name is not None else None
    pending = workspaces.PendingWorkspace(
      f'tab:{uuid.uuid4().hex}', name or 'creating…'
    )
    self._insert_local(pending)
    task = asyncio.create_task(
      self._create_worktree(pending, source, name, preferences)
    )
    self._worktree_creation_tasks.add(task)

    def completed(task: asyncio.Task[workspaces.AgentWorkspace]) -> None:
      self._worktree_creation_tasks.discard(task)
      if not task.cancelled():
        task.exception()

    task.add_done_callback(completed)
    return task

  async def _create_worktree(
    self,
    pending: workspaces.PendingWorkspace,
    source: workspaces.AgentWorkspace,
    name: str | None,
    preferences: client_pb2.WorkspacePreferences | None,
  ) -> workspaces.AgentWorkspace:
    try:
      await self._cancel_reconcile()
      workspace = await self._create_configured_workspace(
        pending, source, name, preferences
      )
    except Exception as error:  # Keep creation failures on their own tab.
      current = next(
        workspace
        for workspace in self.workspaces
        if workspace.identity == pending.identity
      )
      if isinstance(current, workspaces.PendingWorkspace):
        current.error = str(error)
      else:
        current.show_error(str(error))
      raise
    await workspace.start()
    return workspace

  async def open_folder(
    self,
    source: pathlib.Path | str,
    directory: pathlib.Path,
    *,
    name: str | None = None,
  ) -> WorkspaceSnapshot:
    """Open an already trusted directory; each call creates an independent handle."""
    sender = self._source_workspace(source)
    if self._shutting_down:
      raise worktrees.WorktreeError('cannot open a folder during shutdown')
    path = (sender.agent.workspace / directory.expanduser()).resolve(
      strict=True
    )
    if not path.is_dir():
      raise ValueError(f'not a directory: {path}')
    if not self._is_trusted_directory(path):
      raise ValueError(
        f'Folder {str(path)!r} is not trusted. Open it in Mirror interactively first.'
      )
    label = name.strip() if name is not None else path.name or str(path)
    if not label:
      raise ValueError('workspace name must not be empty')
    await self._cancel_reconcile()
    location = workspaces.Directory(
      uuid.uuid4().hex, path, label, worktrees.is_managed_directory(path)
    )
    workspace = self._new_workspace(
      location,
      self._new_worker_config(sender.agent),
      agent_workspace=path,
    )
    try:
      with cli_store.Store(self._database) as store:
        store.save_workspace(
          self._repository_key, location.id, str(path), label
        )
    except (OSError, sqlite3.Error):
      await workspace.close()
      raise
    self._insert_local(workspace)
    return self._workspace_snapshot(workspace, sender.id)

  async def start_task(
    self, workspace_id: str, instruction: str
  ) -> WorkspaceSnapshot:
    """Start or submit to a workspace's agent using the same handle as selection."""
    workspace = self._target_workspace(workspace_id)
    startup = await workspace.start()
    await startup
    workspace.agent.submit_instruction(instruction)
    return self._workspace_snapshot(workspace, workspace.id)

  async def close_workspace(self, workspace_id: str) -> bool:
    """Close a local workspace without deleting its directory or agent history."""
    source = next(
      (
        item
        for item in self._agent_workspaces()
        if item.id == workspace_id and not item.remote
      ),
      None,
    )
    if source is None:
      raise ValueError(f'workspace {workspace_id!r} not found')
    await self._cancel_reconcile()
    others = [item for item in self.visible_workspaces() if item is not source]
    if not any(isinstance(item, workspaces.AgentWorkspace) for item in others):
      raise ValueError('cannot close the only available workspace')
    with cli_store.Store(self._database) as store:
      store.close_workspace(self._repository_key, source.id)
    self._closed_workspaces.add(source.id)
    if self.selected is source:
      self.selected = next(
        (
          item
          for item in self.workspaces[self.workspaces.index(source) + 1 :]
          if item.available
        ),
        others[-1],
      )
    self.workspaces.remove(source)
    self.report_workspace_count()
    self.running_sessions_changed()
    await source.close()
    return True

  async def create_worktree_workspace(
    self,
    source: pathlib.Path | str,
    name: str | None = None,
    *,
    preferences: client_pb2.WorkspacePreferences | None = None,
  ) -> WorkspaceSnapshot:
    """Create a managed worktree with the existing provisioning and startup flow."""
    result = await self.spawn_workspace(
      source, name, None, preferences=preferences
    )
    return result.workspace

  async def spawn_workspace(
    self,
    source: pathlib.Path | str,
    name: str | None,
    message: str | None,
    *,
    preferences: client_pb2.WorkspacePreferences | None = None,
  ) -> WorkspaceSubmission:
    """Create a workspace and save its configuration and tool selections."""
    task = self._start_worktree_creation(
      self._source_workspace(source), name, preferences
    )
    # Tab handoff cancels UI effects, not controller-owned workspace creation.
    workspace = await asyncio.shield(task)
    if message is not None:
      return await self.send_message(source, workspace.id, message)
    return WorkspaceSubmission(
      workspace=self._workspace_snapshot(workspace, source),
      status='created',
    )

  async def create(self, name: str | None = None) -> bool:
    """Start creating a worktree without blocking the selected agent."""
    try:
      self._start_worktree_creation(self._local_source(), name)
    except worktrees.WorktreeError as error:
      self.selected.show_error(str(error))
      return False
    await asyncio.sleep(0)
    return True

  def _remote_tab_index(self, session_id: str) -> int | None:
    for tab_index, workspace in enumerate(self.visible_workspaces()):
      if (
        isinstance(workspace, workspaces.AgentWorkspace)
        and workspace.remote
        and workspace.id == session_id
      ):
        return tab_index
    return None

  def _append_remote(
    self,
    source: workspaces.AgentWorkspace,
    session_id: str,
    *,
    create_session: bool,
    title: str | None = None,
    remote_workspace: mirror_pb2.Workspace | None = None,
  ) -> workspaces.AgentWorkspace:
    config = self._new_worker_config(source.agent) | {
      'conversation_id': session_id,
      'create_session': create_session,
      'model': 'server default',
      'remote_workspace': remote_workspace,
    }
    if self._delete_remote_session is None:
      raise RuntimeError('remote workspace provider is not configured')
    workspace = self._new_workspace(
      workspaces.RemoteSession(session_id),
      config,
      agent_workspace=source.agent.workspace,
      reasoning_effort=source.agent.reasoning_effort,
      alias=title,
    )
    self.workspaces.append(workspace)
    self.report_workspace_count()
    self.running_sessions_changed()
    return workspace

  def load_remote_sessions(self) -> None:
    """Start remote discovery at most once per invocation."""
    if (
      self._list_remote_sessions is not None
      and self._remote_session_task is None
    ):
      self._remote_session_task = asyncio.create_task(
        self._load_remote_sessions()
      )

  async def _load_remote_sessions(self) -> None:
    if self._list_remote_sessions is None:
      return
    source = self._local_source()
    try:
      session_ids = await self._list_remote_sessions()
    except Exception as error:  # noqa: BLE001 - isolate remote discovery.
      logging.getLogger(__name__).exception('could not list remote sessions')
      source.agent.show_error(
        f'could not list remote sessions: {type(error).__name__}: {error}'
      )
      return
    for session_id in session_ids:
      if self._remote_tab_index(session_id) is None:
        overlay = self._overlays.get(session_id, cli_store.WorktreeOverlay())
        self._append_remote(
          source,
          session_id,
          create_session=False,
          title=overlay.alias,
        )

  async def create_remote(
    self,
    title: str | None = None,
    remote_workspace: mirror_pb2.Workspace | None = None,
  ) -> int | None:
    """Start or select a remote agent and return its visible tab index.

    A new session starts in `remote_workspace`; resuming one ignores it.
    """
    selected = self.selected
    source = (
      selected
      if isinstance(selected, workspaces.AgentWorkspace)
      else self._local_source()
    )
    if not self._remote_enabled:
      source.agent.show_error('remote AgentRPC endpoint is not configured')
      return None
    create_session = title is None
    if title is None:
      session_id = str(uuid.uuid4())
    else:
      try:
        session_id = str(uuid.UUID(title))
        title = self._overlays.get(
          session_id, cli_store.WorktreeOverlay()
        ).alias
      except ValueError:
        session_id = str(uuid.uuid4())
        create_session = True
        with cli_store.Store(self._database) as store:
          title = store.set_worktree_alias(
            self._repository_key, session_id, title
          )
        self._overlays[session_id] = cli_store.WorktreeOverlay(alias=title)
    existing = self._remote_tab_index(session_id)
    if existing is not None:
      return existing
    workspace = self._append_remote(
      source,
      session_id,
      create_session=create_session,
      title=title,
      remote_workspace=remote_workspace if create_session else None,
    )
    workspace.resume_pending = False
    await workspace.start()
    return self._remote_tab_index(session_id)

  async def _cancel_reconcile(self) -> None:
    task, self._reconcile_task = self._reconcile_task, None
    if task is None:
      return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

  async def _cancel_background_tasks(self) -> None:
    self._shutting_down = True
    await asyncio.gather(
      *tuple(self._worktree_creation_tasks), return_exceptions=True
    )
    self._worktree_creation_tasks.clear()
    await asyncio.gather(
      *tuple(self._workspace_deletion_tasks), return_exceptions=True
    )
    tasks = list(self._message_delivery_tasks)
    if self._reconcile_task is not None:
      tasks.append(self._reconcile_task)
      self._reconcile_task = None
    if self._remote_session_task is not None:
      tasks.append(self._remote_session_task)
      self._remote_session_task = None
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    self._message_delivery_tasks.clear()

  async def close(self) -> None:
    """Drain owned operations, then close all workspace backends."""
    await self._cancel_background_tasks()
    await asyncio.gather(*(workspace.close() for workspace in self.workspaces))
