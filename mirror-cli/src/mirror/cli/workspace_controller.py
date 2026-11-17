"""Shared workspace views with client-local drafts, consent, and extension hosts."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import pathlib
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from typing import Literal, cast

import grpc
import grpc.aio
from mirror.protocol import (
  daemon_pb2,
  mirror_pb2,
)

from mirror import config, project
from mirror.cli import approval_modes
from mirror.cli import (
  actions,
  agents,
  daemon_client,
  models,
  onboarding,
  rpc_client,
  sandboxes,
  selection,
  workspace_layout,
  workspace_tags,
  workspaces,
  worktrees,
)
from mirror.cli.sandbox import setup as sandbox_setup
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
  kind: Literal['folder', 'worktree', 'sandbox'] = 'worktree'
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


def _normalize_alias(alias: str) -> str:
  normalized = ' '.join(alias.split())
  if not normalized:
    raise ValueError('worktree alias must not be empty')
  return normalized[:79] + '…' if len(normalized) > 80 else normalized


def workspace_kind(
  record: daemon_pb2.Workspace,
) -> Literal['folder', 'worktree', 'sandbox', 'remote']:
  """Describe a catalog entry without constructing its agent."""
  if record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE:
    return 'remote'
  if record.kind == daemon_pb2.WORKSPACE_KIND_SANDBOX:
    return 'sandbox'
  if record.is_worktree:
    return 'worktree'
  return 'folder'


def can_delete_workspace_files(record: daemon_pb2.Workspace) -> bool:
  """Offer file deletion only for managed worktrees and sandboxes."""
  if record.HasField('can_delete_files'):
    return record.can_delete_files
  return record.kind == daemon_pb2.WORKSPACE_KIND_SANDBOX or (
    record.kind == daemon_pb2.WORKSPACE_KIND_DIRECTORY
    and record.is_worktree
    and worktrees.is_managed_directory(pathlib.Path(record.directory))
  )


@dataclasses.dataclass
class ViewState:
  """Drafts and navigation retained only within one daemon connection scope."""

  drafts: dict[str, actions.ComposerState] = dataclasses.field(
    default_factory=dict[str, actions.ComposerState]
  )
  collapsed_sections: set[str] = dataclasses.field(default_factory=set[str])
  landing_draft: actions.ComposerState = dataclasses.field(
    default_factory=actions.ComposerState
  )
  landing_session: str | None = None


class WorkspaceController:
  """Present daemon sessions without owning their discovery or lifecycle."""

  def __init__(
    self,
    worker_config: Mapping[str, object],
    *,
    workspace: pathlib.Path,
    store: daemon_client.Client,
    reasoning_effort: str,
    create_agent: Callable[
      [workspaces.AgentSpec, WorkspaceController], workspaces.Agent
    ],
    settings: config.Config | None = None,
    session_id: str | None = None,
    display_unmanaged_worktrees: bool = False,
    remote_enabled: bool = False,
    list_remote_sessions: Callable[[], Awaitable[Sequence[str]]] | None = None,
    delete_remote_session: Callable[[str], Awaitable[None]] | None = None,
    is_trusted_directory: Callable[
      [pathlib.Path], bool
    ] = onboarding.is_trusted_directory,
    sandboxd: pathlib.Path | None = None,
    view_state: ViewState | None = None,
  ) -> None:
    # Startup already ensured the daemon; tab attachments must not replace it.
    self._base_config = dict(worker_config) | {'daemon_connect_only': True}
    self._reasoning_effort = reasoning_effort
    self._create_agent = create_agent
    self._client = store
    self.remote_host = store.remote_host
    self.daemon_defaults = (
      store.frontend_defaults() if self.remote_host else None
    )
    self._settings = settings or config.Config(config.default_path())
    if self.remote_host:
      self._settings = config.Config(
        workspace_layout.settings_path(
          self._settings.path.parent,
          str(worker_config.get('daemon_scope') or self.remote_host),
        )
      )
    self._tags = workspace_tags.definitions(self._settings.read())
    self._is_trusted_directory = is_trusted_directory
    self._sandboxd = sandboxd or sandboxes.sandboxd_path()
    self._remote_enabled = remote_enabled
    self._list_remote_sessions = list_remote_sessions
    self._delete_remote_session = delete_remote_session
    self._remote_session_task: asyncio.Task[None] | None = None
    self._reconcile_task: asyncio.Task[None] | None = None
    self._watch_task: asyncio.Task[None] | None = None
    self._tasks: set[asyncio.Task[None]] = set()
    self._detach_tasks: set[asyncio.Task[None]] = set()
    self._creation_tasks: set[asyncio.Task[workspaces.AgentWorkspace]] = set()
    self._last_running_count: int | None = None
    self._reported_workspace_count: tuple[int, int] | None = None
    self._shutting_down = False
    self._daemon_stopped = False
    self._disconnected_workspaces: set[str] = set()
    self._view_state = view_state or ViewState()
    self._closed_drafts = self._view_state.drafts
    self.collapsed_workspace_sections = self._view_state.collapsed_sections
    self._empty = workspaces.EmptyWorkspace(
      draft=self._view_state.landing_draft,
      session_id=self._view_state.landing_session,
    )
    self.launch_workspace = (
      pathlib.Path(self.daemon_defaults.directory)
      if self.daemon_defaults is not None
      else workspace.resolve()
    )
    self._layout_path = workspace_layout.path(
      self.launch_workspace,
      str(self._base_config.get('database') or ''),
      remote_host=str(
        worker_config.get('daemon_scope') or store.remote_host or ''
      ),
    )
    self._query = daemon_pb2.WatchWorkspacesRequest(
      include_unmanaged=display_unmanaged_worktrees,
      include_closed=True,
    )
    snapshot = store.snapshot(self._query)
    self._records = {record.id: record for record in snapshot.workspaces}
    current = (
      next(
        (
          record
          for record in snapshot.workspaces
          if session_id in (record.session_id, *record.previous_session_ids)
          and (
            self._remote_enabled
            or record.kind != daemon_pb2.WORKSPACE_KIND_REMOTE
          )
        ),
        None,
      )
      if session_id
      else None
    )
    if session_id and current is None:
      raise ValueError('The requested conversation has no daemon workspace')
    if current is not None and session_id != current.session_id:
      if (
        current.kind == daemon_pb2.WORKSPACE_KIND_DIRECTORY
        and not self.is_trusted_directory(pathlib.Path(current.directory))
      ):
        raise ValueError('Trust the workspace directory before opening it')
      current.CopyFrom(
        store.switch_session(
          current.id, current.session_id, session_id=session_id
        )
      )
    if (
      current is not None and current.state == daemon_pb2.WORKSPACE_STATE_CLOSED
    ):
      current.CopyFrom(store.open_workspace(current.id))
    self.selected: workspaces.Workspace = self._empty
    self.workspaces: list[workspaces.Workspace] = []
    if current is not None:
      self.selected = self._new_view(current, active=True)
      self.workspaces.append(self.selected)
    else:
      self._restore_layout()
    self._apply_snapshot(snapshot)
    if current is not None:
      self._save_layout()

  def _restore_layout(self) -> None:
    tabs = workspace_layout.load(self._layout_path)
    for identity in dict.fromkeys(tabs.workspace_ids):
      record = self._records.get(identity)
      if record is None or (
        record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
        and not self._remote_enabled
      ):
        continue
      if (
        record.kind == daemon_pb2.WORKSPACE_KIND_DIRECTORY
        and not self.is_trusted_directory(pathlib.Path(record.directory))
      ):
        continue
      if record.state != daemon_pb2.WORKSPACE_STATE_AVAILABLE:
        continue
      view = self._new_view(
        record, active=identity == tabs.selected_workspace_id
      )
      self.workspaces.append(view)
      if identity == tabs.selected_workspace_id:
        self.selected = view

  def _save_layout(self) -> None:
    # Only explicit tab edits write; refresh, disconnect and exit must not win.
    tabs = daemon_pb2.WorkspaceTabs(
      workspace_ids=[
        view.id
        if isinstance(view, workspaces.AgentWorkspace)
        else view.identity.removeprefix('tab:')
        for view in self.workspaces
      ],
      selected_workspace_id=self.selected.id
      if isinstance(self.selected, workspaces.AgentWorkspace)
      else self.selected.identity.removeprefix('tab:'),
    )
    try:
      workspace_layout.save(self._layout_path, tabs)
    except OSError as error:
      self.selected.show_error(f'Could not save open tabs: {error}')

  def daemon_models(self) -> tuple[models.ModelOption, ...] | None:
    if self.daemon_defaults is None:
      return None
    return tuple(
      models.ModelOption(
        item.model, item.model, api=item.api, base_url=item.base_url
      )
      for item in self.daemon_defaults.models
    )

  def resolve_directory(
    self, entered: str, *, base: pathlib.Path | None = None
  ) -> daemon_pb2.ResolveDirectoryResponse:
    """Resolve on the selected host; a missing leaf still needs explicit creation."""
    base = base or self.launch_workspace
    if self.remote_host:
      return self._client.resolve_directory(entered, base=str(base))
    path = base / pathlib.Path(entered).expanduser()
    if not path.exists():
      path = path.parent.resolve(strict=True) / path.name
      return daemon_pb2.ResolveDirectoryResponse(
        directory=str(path), exists=False
      )
    path = path.resolve(strict=True)
    if not path.is_dir():
      raise ValueError(f'not a directory: {path}')
    return daemon_pb2.ResolveDirectoryResponse(directory=str(path), exists=True)

  def is_trusted_directory(self, directory: pathlib.Path) -> bool:
    if not self.remote_host:
      return self._is_trusted_directory(directory)
    trusted = self._settings.read().get('trusted_directories')
    return isinstance(trusted, list) and str(directory) in trusted

  def trust_directory(self, directory: pathlib.Path) -> None:
    """Remember explicit consent only in this connection's settings."""
    if not self.remote_host:
      onboarding.ensure_consent(
        directory, path=self._settings.path, confirm=lambda _: True
      )
      return

    def trust(values: dict[str, object]) -> None:
      previous = values.get('trusted_directories')
      directories = (
        [
          value
          for value in cast(list[object], previous)
          if isinstance(value, str)
        ]
        if isinstance(previous, list)
        else []
      )
      values['trusted_directories'] = list(
        dict.fromkeys([*directories, str(directory)])
      )

    self._settings.update(trust)

  def _defaults(
    self, source: workspaces.Agent | None = None
  ) -> daemon_pb2.WorkspacePreferences:
    if source is None and self.daemon_defaults is not None:
      preferences = daemon_pb2.WorkspacePreferences()
      preferences.CopyFrom(self.daemon_defaults.preferences)
      return preferences
    option = (
      source.model_selection
      if source is not None
      else models.ModelOption(
        str(self._base_config.get('model', '')),
        '',
        api=str(
          self._base_config.get('model_api')
          or self._base_config.get('api')
          or ''
        ),
        base_url=str(
          self._base_config.get('model_base_url')
          or self._base_config.get('base_url')
          or ''
        ),
      )
    )
    approval = (
      source.approval_mode
      if source is not None
      else self._base_config.get('approval_mode')
    )
    return daemon_pb2.WorkspacePreferences(
      approval_mode=approval_modes.to_protocol(approval),
      definition=mirror_pb2.AgentDefinition(
        model=option.model_id,
        model_api=option.api,
        model_base_url=option.base_url,
        reasoning_effort=source.reasoning_effort
        if source is not None
        else self._reasoning_effort,
      ),
    )

  def _new_view(
    self,
    opened: daemon_pb2.Workspace,
    *,
    active: bool = False,
    create_session: bool = False,
    remote_workspace: mirror_pb2.Workspace | None = None,
    escalated: bool = False,
  ) -> workspaces.AgentWorkspace:
    self._records[opened.id] = opened
    if (
      self._empty.session_id == opened.session_id
      and self.selected is self._empty
    ):
      self._closed_drafts[opened.session_id] = self._empty.draft
    remote = opened.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
    source = None
    if remote:
      source = (
        self.selected.agent
        if isinstance(self.selected, workspaces.AgentWorkspace)
        else next(
          (view.agent for view in self._agent_workspaces() if not view.remote),
          None,
        )
      )
    preferences = self._defaults(source)
    if remote:
      preferences.definition.model = 'server default'
      preferences.definition.ClearField('model_api')
      preferences.definition.ClearField('model_base_url')
    preferences.MergeFrom(opened.preferences)
    if active:
      defaults = self._defaults()
      if self._base_config.get('approval_mode_override'):
        preferences.approval_mode = defaults.approval_mode
      if self._base_config.get('reasoning_effort_override'):
        preferences.definition.reasoning_effort = self._reasoning_effort
      if self._base_config.get('model_selection_override'):
        preferences.definition.model = defaults.definition.model
        preferences.definition.model_api = defaults.definition.model_api
        preferences.definition.model_base_url = (
          defaults.definition.model_base_url
        )
    definition = preferences.definition
    if opened.HasField('inference'):
      definition.model = opened.inference.model
      definition.model_api = opened.inference.api
      definition.model_base_url = opened.inference.base_url
      definition.reasoning_effort = opened.inference.reasoning_effort
      preferences.approval_mode = opened.approval_mode
    option = (
      models.ModelOption(definition.model, definition.model)
      if self.remote_host
      else models.find_model(definition.model)
    )
    values = self._base_config | {
      'conversation_id': opened.session_id,
      'create_session': create_session,
      'escalate_session': remote and escalated,
      'attach_running': opened.HasField('inference'),
      'remote_workspace': remote_workspace,
      'remote_model_api': definition.model_api if remote else None,
      'model': definition.model,
      'model_api': definition.model_api
      if definition.HasField('model_api')
      else option.api,
      'model_base_url': definition.model_base_url
      if definition.HasField('model_base_url')
      else option.base_url,
      'base_url': None,
      'approval_mode': approval_modes.from_protocol(preferences.approval_mode),
      'workspace_id': opened.id,
      'sandboxed': opened.kind == daemon_pb2.WORKSPACE_KIND_SANDBOX,
    }
    if definition.HasField('system_prompt'):
      values['system_prompt'] = definition.system_prompt
    extension_tools = None
    if definition.HasField('tools_allowlist'):
      builtins = agents.builtins()
      values['tools'] = tuple(
        name for name in definition.tools_allowlist.names if name in builtins
      )
      extension_tools = tuple(
        name
        for name in definition.tools_allowlist.names
        if name not in builtins
      )
    reasoning_effort = definition.reasoning_effort or self._reasoning_effort
    if (
      not remote
      and not opened.HasField('inference')
      and not (active and values.get('reasoning_effort_override'))
    ):
      reasoning_effort = models.inherited_reasoning_effort(
        str(values.get('model_api') or values.get('api') or ''),
        reasoning_effort,
      )
    directory = pathlib.Path(opened.directory)
    if remote:
      directory = (
        source.workspace if source is not None else self.launch_workspace
      )
    speed = 'normal'
    if opened.HasField('inference'):
      speed = rpc_client.inference_speed(opened.inference)
    elif definition.speed == 'fast' and not (
      active and values.get('model_selection_override')
    ):
      speed = 'fast'
    agent = self._create_agent(
      workspaces.AgentSpec(
        config=values,
        workspace=directory,
        reasoning_effort=reasoning_effort,
        active=active,
        remote=remote,
        speed=speed,
        workspace_id=None if remote else opened.id,
        extensions=tuple(opened.preferences.extensions),
        tools=extension_tools,
      ),
      self,
    )
    return workspaces.AgentWorkspace(
      identity=f'tab:{opened.id}',
      location=workspaces.RemoteSession(opened.session_id)
      if remote
      else opened,
      agent=agent,
      alias=opened.name if remote else None,
      resume_pending=not create_session,
      draft=self._closed_drafts.pop(opened.session_id, None),
    )

  def _apply_snapshot(
    self, snapshot: daemon_pb2.WatchWorkspacesResponse
  ) -> None:
    """Update views by identity; keep selection and drafts local."""
    if self._daemon_stopped:
      return
    positions = {
      view.identity: index for index, view in enumerate(self.workspaces)
    }
    self._records = {record.id: record for record in snapshot.workspaces}
    current = {
      view.id: view for view in self._agent_workspaces() if not view.remote
    }
    pending = {
      view.identity: view
      for view in self.workspaces
      if isinstance(view, workspaces.PendingWorkspace)
    }
    views: list[workspaces.Workspace] = []
    for record in snapshot.workspaces:
      if record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE:
        continue
      if record.id not in current and record.state in (
        daemon_pb2.WORKSPACE_STATE_UNSPECIFIED,
        daemon_pb2.WORKSPACE_STATE_UNAVAILABLE,
      ):
        continue
      identity = f'tab:{record.id}'
      if record.id not in current and identity not in pending:
        continue
      if record.state in (
        daemon_pb2.WORKSPACE_STATE_PROVISIONING,
        daemon_pb2.WORKSPACE_STATE_FAILED,
      ):
        placeholder = pending.get(identity) or workspaces.PendingWorkspace(
          identity, record.name
        )
        placeholder.error = record.error if record.HasField('error') else None
        views.append(placeholder)
        continue
      view = current.get(record.id)
      if record.state == daemon_pb2.WORKSPACE_STATE_CLOSED:
        # Keep the initiating view through its explicit stop/start transaction.
        if view is None or (
          not view.agent.restarting
          and not view.dropping
          and record.id not in self._disconnected_workspaces
        ):
          continue
      if view is None:
        view = self._new_view(
          record,
          active=self.selected is pending.get(identity),
          create_session=identity in pending
          and pending[identity].create_session,
        )
      elif (
        record.id in self._disconnected_workspaces
        and record.state == daemon_pb2.WORKSPACE_STATE_AVAILABLE
      ) or (
        view.agent.conversation_id != record.session_id
        and not view.agent.restarting
        and view.agent.worktree_state
        in (
          actions.WorktreeAgentState.NOT_STARTED,
          actions.WorktreeAgentState.FAILED,
        )
      ):
        self._preserve_draft(view)
        replacement = self._new_view(record)
        if self.selected is view:
          self.selected = replacement
        view = replacement
      else:
        view.location = record
      if placeholder := pending.get(identity):
        view.identity = placeholder.identity
        view.draft = placeholder.draft
        view.resume_pending = not placeholder.create_session
        placeholder.finished = True
        if self.selected is placeholder:
          self.selected = view
      if record.state == daemon_pb2.WORKSPACE_STATE_AVAILABLE:
        self._disconnected_workspaces.discard(record.id)
      views.append(view)
    views.extend(
      view
      for view in current.values()
      if view.id in self._disconnected_workspaces and view not in views
    )
    for view in self._agent_workspaces():
      if view.remote:
        if record := self._records.get(view.id):
          view.alias = record.name or None
        views.append(view)
    views.extend(
      view
      for view in pending.values()
      if not view.finished and view not in views
    )
    views.sort(
      key=lambda view: (
        view.remote,
        positions.get(view.identity, len(positions)),
      )
    )
    removed = [view for view in self.workspaces if view not in views]
    for view in removed:
      self._preserve_draft(view)
      self._detach(view)
    if self.selected not in views:
      self.selected = views[0] if views else self._empty
    self.workspaces[:] = views
    self.report_workspace_count()
    self.running_sessions_changed()

  def _preserve_draft(self, view: workspaces.Workspace) -> None:
    if not isinstance(view, workspaces.AgentWorkspace):
      if self.selected is view and isinstance(
        view, workspaces.PendingWorkspace
      ):
        self._empty.draft = view.draft
        self._empty.session_id = None
      return
    draft = view.draft or view.agent.composer_state()
    self._closed_drafts[view.agent.conversation_id] = draft
    if self.selected is view:
      self._empty.draft = draft
      self._empty.session_id = view.agent.conversation_id

  def _daemon_closed(self) -> None:
    if self._daemon_stopped:
      return
    self._daemon_stopped = True
    self._disconnected_workspaces.update(
      view.id for view in self._agent_workspaces()
    )
    self.selected.show_error(
      'Daemon disconnected. Tabs are kept; waiting to reconnect.'
    )
    self.running_sessions_changed()

  async def restart_daemon(self, restart: Callable[[], Awaitable[str]]) -> None:
    """Suspend metadata reconciliation until explicit replacement finishes."""
    if self._watch_task is not None:
      self._watch_task.cancel()
      await asyncio.gather(self._watch_task, return_exceptions=True)
    self._daemon_closed()
    try:
      await restart()
    finally:
      if not self._shutting_down:
        self._watch_task = asyncio.create_task(self._watch())

  async def _reconnect(
    self, snapshot: daemon_pb2.WatchWorkspacesResponse
  ) -> None:
    records = {record.id: record for record in snapshot.workspaces}
    attached = self._agent_workspaces()
    for view in attached:
      self._preserve_draft(view)
    await asyncio.gather(
      *(self._detach(view) for view in attached), return_exceptions=True
    )
    if self._shutting_down:
      return
    for view in attached:
      if view not in self.workspaces:
        continue
      record = records.get(view.id)
      if record is None:
        view.show_error('Workspace is missing from the replacement daemon.')
        continue
      replacement = self._new_view(record)
      if record.state == daemon_pb2.WORKSPACE_STATE_AVAILABLE:
        self._disconnected_workspaces.discard(record.id)
      if self.selected is view:
        self.selected = replacement
      self.workspaces[self.workspaces.index(view)] = replacement
    self._daemon_stopped = False
    self._apply_snapshot(snapshot)

  def _background(
    self, operation: Coroutine[object, object, None]
  ) -> asyncio.Task[None]:
    def completed(task: asyncio.Task[None]) -> None:
      self._tasks.discard(task)
      if not task.cancelled() and (error := task.exception()) is not None:
        self.selected.show_error(str(error))

    task = asyncio.create_task(operation)
    self._tasks.add(task)
    task.add_done_callback(completed)
    return task

  def _detach(self, view: workspaces.Workspace) -> asyncio.Task[None]:
    task = self._background(view.close())
    self._detach_tasks.add(task)
    task.add_done_callback(self._detach_tasks.discard)
    return task

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

  def _source_workspace(
    self, source: pathlib.Path | str
  ) -> workspaces.AgentWorkspace:
    if isinstance(source, str):
      return self._target_workspace(source)
    matches = [
      workspace
      for workspace in self._agent_workspaces()
      if not workspace.remote and workspace.agent.workspace == source
    ]
    if len(matches) != 1:
      raise worktrees.WorktreeError(
        f'source directory {str(source)!r} is unavailable or ambiguous; use its session id'
      )
    return matches[0]

  def _target_workspace(self, identity: str) -> workspaces.AgentWorkspace:
    local = [
      workspace
      for workspace in self._agent_workspaces()
      if not workspace.remote and workspace.available
    ]
    exact = next(
      (workspace for workspace in local if workspace.id == identity), None
    )
    if exact is not None:
      return exact
    matches = [
      workspace
      for workspace in local
      if workspace.label.casefold() == identity.casefold()
    ]
    if len(matches) != 1:
      raise worktrees.WorktreeError(
        f'session {identity!r} is unavailable or ambiguous; use its id'
      )
    return matches[0]

  def _workspace_snapshot(
    self, workspace: workspaces.AgentWorkspace, source: pathlib.Path | str
  ) -> WorkspaceSnapshot:
    kind = workspace_kind(self._records[workspace.id])
    if kind == 'remote':
      raise ValueError('Workspace snapshots require a local workspace')
    return WorkspaceSnapshot(
      id=workspace.id,
      name=workspace.label,
      path=workspace.path,
      current=workspace is self._source_workspace(source),
      tag=self.workspace_tag(workspace.id),
      tag_id=self._records.get(
        workspace.id, daemon_pb2.Workspace()
      ).preferences.tag_id
      or None,
      kind=kind,
      cwd=workspace.agent.workspace,
      agent=AgentSnapshot(
        workspace.state,
        workspace.agent.model,
        workspace.agent.reasoning_effort,
        workspace.agent.conversation_id,
        workspace.agent.activity_status or None,
        workspace.agent.queued_message_count,
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

  def workspace_catalog(self) -> tuple[daemon_pb2.Workspace, ...]:
    """Return catalog metadata without opening views or attaching agents."""
    return tuple(self._records.values())

  async def conversation_summaries(
    self,
  ) -> list[daemon_client.ConversationInfo]:
    """Read summaries only for conversations still in the workspace catalog."""
    return await asyncio.to_thread(
      self._client.conversation_details,
      tuple(record.session_id for record in self._records.values()),
    )

  def _selected_agent_workspace(self) -> workspaces.AgentWorkspace:
    if not isinstance(self.selected, workspaces.AgentWorkspace):
      raise worktrees.WorktreeError('this session is not ready')
    return self.selected

  async def _reconcile(self) -> None:
    self._apply_snapshot(
      await asyncio.to_thread(self._client.snapshot, self._query)
    )

  async def _watch(self) -> None:
    try:
      async for snapshot in self._client.watch(self._query):
        if snapshot is None:
          self._daemon_closed()
        elif self._daemon_stopped:
          await self._reconnect(snapshot)
        else:
          self._apply_snapshot(snapshot)
    except grpc.aio.AioRpcError as error:
      self.selected.show_error(f'Session subscription failed: {error}')

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
    self._query.include_unmanaged = enabled
    if self._watch_task is not None:
      self._watch_task.cancel()
      self._watch_task = asyncio.create_task(self._watch())
    self.refresh()

  async def initialize(self) -> None:
    self.load_remote_sessions()
    for workspace in self._agent_workspaces():
      if (
        not workspace.remote
        and isinstance(workspace.location, daemon_pb2.Workspace)
        and workspace.location.HasField('inference')
        and (workspace.is_sandbox or self.is_trusted_directory(workspace.path))
      ):
        await workspace.start()
    if self._watch_task is None:
      self._watch_task = asyncio.create_task(self._watch())

  def refresh(self) -> None:
    """Request a fresh server snapshot, without doing discovery in this client."""
    if (
      self._shutting_down
      or self._daemon_stopped
      or (self._reconcile_task is not None and not self._reconcile_task.done())
    ):
      return
    self._reconcile_task = asyncio.create_task(self._reconcile())

    def completed(task: asyncio.Task[None]) -> None:
      if not task.cancelled() and (error := task.exception()) is not None:
        self.selected.show_error(str(error))

    self._reconcile_task.add_done_callback(completed)

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
    if sender is target:
      raise worktrees.WorktreeError('cannot send a message to this session')
    messages = _transcript_messages(target)
    instruction = (
      f'Message from the agent in workspace {sender.label!r}. '
      'Complete this request in your workspace and leave the response here '
      f'for the sending agent to inspect.\n\n{message}'
    )

    async def deliver() -> None:
      try:
        await self.start_task(target.id, instruction)
      except Exception as error:  # Report failures to the initiating agent.
        sender.agent.submit_instruction(
          f'Message delivery to workspace {target.id!r} failed: {error}'
        )

    self._background(deliver())
    return WorkspaceSubmission(
      self._workspace_snapshot(target, source),
      'accepted',
      messages[-1].timestamp if messages else None,
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

  async def open_session(
    self, session_id: str, instruction: str | None = None
  ) -> None:
    """Select a workspace or explicitly resume one of its retained conversations."""
    if self._daemon_stopped:
      raise ValueError('Daemon disconnected; waiting to reconnect.')
    snapshot = await asyncio.to_thread(self._client.snapshot, self._query)
    record = next(
      (
        record
        for record in snapshot.workspaces
        if session_id in (record.session_id, *record.previous_session_ids)
      ),
      None,
    )
    if record is None:
      raise ValueError('The requested conversation has no daemon workspace')
    if (
      record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE
      and not self._remote_enabled
    ):
      raise ValueError('Remote AgentRPC endpoint is not configured')
    if record.state not in (
      daemon_pb2.WORKSPACE_STATE_AVAILABLE,
      daemon_pb2.WORKSPACE_STATE_CLOSED,
    ):
      raise ValueError(record.error or 'This workspace is not available')
    if (
      record.kind == daemon_pb2.WORKSPACE_KIND_DIRECTORY
      and not self.is_trusted_directory(pathlib.Path(record.directory))
    ):
      raise ValueError('Trust the workspace directory before opening it')
    if record.session_id != session_id:
      view = next(
        (view for view in self._agent_workspaces() if view.id == record.id),
        None,
      )
      if view is not None and record.HasField('inference'):
        switched = await view.agent.switch_workspace_session(session_id)
        if not switched:
          return
        updated = await asyncio.to_thread(self._client.snapshot, self._query)
        record.CopyFrom(
          next(saved for saved in updated.workspaces if saved.id == record.id)
        )
      else:
        record.CopyFrom(
          await asyncio.to_thread(
            self._client.switch_session,
            record.id,
            record.session_id,
            session_id=session_id,
          )
        )
        if view is not None:
          self._preserve_draft(view)
          await view.close()
          if view in self.workspaces:
            self.workspaces.remove(view)
    if record.state == daemon_pb2.WORKSPACE_STATE_CLOSED:
      record.CopyFrom(
        await asyncio.to_thread(self._client.open_workspace, record.id)
      )
    if self._daemon_stopped:
      raise ValueError('Daemon disconnected; waiting to reconnect.')
    view = next(
      (view for view in self._agent_workspaces() if view.id == record.id), None
    )
    if view is None:
      view = self._new_view(record)
      self.workspaces.append(view)
    self.selected = view
    self._save_layout()
    self.report_workspace_count()
    self.running_sessions_changed()
    if instruction is not None:
      await (await view.start())
      view.agent.submit_instruction(instruction)

  def list_workspace_tags(self) -> tuple[workspace_tags.WorkspaceTag, ...]:
    """Expose ordered definitions to extensions and workspace API consumers."""
    self._tags = workspace_tags.definitions(self._settings.read())
    return self._tags

  def save_workspace_tag(self, tag: workspace_tags.WorkspaceTag) -> None:
    """Create or update a label by its stable ID."""
    workspace_tags.save(self._settings, tag)
    self.list_workspace_tags()

  def delete_workspace_tag(self, tag_id: str) -> None:
    workspace_tags.delete(self._settings, tag_id)
    self.list_workspace_tags()
    for identity, record in tuple(self._records.items()):
      if record.preferences.tag_id == tag_id:
        self.set_workspace_tag(identity, None)

  def workspace_tag(
    self, workspace_id: str
  ) -> workspace_tags.WorkspaceTag | None:
    tag_id = self._records.get(
      workspace_id, daemon_pb2.Workspace()
    ).preferences.tag_id
    return next((tag for tag in self._tags if tag.id == tag_id), None)

  def set_workspace_tag(self, workspace_id: str, tag_id: str | None) -> None:
    if workspace_id not in self._records:
      raise ValueError(f'workspace {workspace_id!r} is unavailable')
    if tag_id is not None and not any(
      tag.id == tag_id for tag in self.list_workspace_tags()
    ):
      raise ValueError(f'unknown workspace tag {tag_id!r}')
    self._update(
      workspace_id,
      daemon_pb2.Workspace(
        preferences=daemon_pb2.WorkspacePreferences(tag_id=tag_id)
      ),
      ('preferences.tag_id',),
    )

  def cycle_workspace_tag(
    self, *, workspace_id: str | None = None
  ) -> workspace_tags.WorkspaceTag | None:
    identity = workspace_id or self._selected_agent_workspace().id
    current = (
      self._records.get(identity, daemon_pb2.Workspace()).preferences.tag_id
      or None
    )
    choices = [None, *(tag.id for tag in self.list_workspace_tags())]
    self.set_workspace_tag(
      identity,
      choices[(choices.index(current) + 1) % len(choices)]
      if current in choices
      else None,
    )
    return self.workspace_tag(identity)

  def tab_tags(self) -> dict[str, workspace_tags.WorkspaceTag]:
    """Project metadata onto UI identities without changing the roster."""
    return {
      workspace.identity: tag
      for workspace in self._agent_workspaces()
      if (tag := self.workspace_tag(workspace.id)) is not None
    }

  def rename(self, alias: str) -> str:
    workspace = self._selected_agent_workspace()
    normalized = _normalize_alias(alias)
    for other in self._agent_workspaces():
      if other is not workspace and any(
        name.casefold() == normalized.casefold()
        for name in (other.id, other.name, other.label)
      ):
        raise ValueError(f'worktree alias {normalized!r} is already in use')
    record = self._update(
      workspace.id, daemon_pb2.Workspace(name=normalized), ('name',)
    )
    if workspace.remote:
      workspace.alias = record.name
    else:
      workspace.location = record
    return record.name

  def save_preferences(
    self,
    reasoning_effort: str,
    approval_mode: str,
    speed: str | None = None,
    *,
    agent: workspaces.Agent | None = None,
  ) -> None:
    workspace = (
      self._selected_agent_workspace()
      if agent is None
      else next(
        workspace
        for workspace in self._agent_workspaces()
        if workspace.agent is agent
      )
    )
    fields = [
      'preferences.approval_mode',
      'preferences.definition.reasoning_effort',
    ]
    if speed is not None:
      fields.append('preferences.definition.speed')
    self._update(
      workspace.id,
      daemon_pb2.Workspace(
        preferences=daemon_pb2.WorkspacePreferences(
          approval_mode=approval_modes.to_protocol(approval_mode),
          definition=mirror_pb2.AgentDefinition(
            reasoning_effort=reasoning_effort,
            speed='fast' if speed == 'fast' else None,
          ),
        )
      ),
      fields,
    )

  def save_model(
    self, agent: workspaces.Agent, selection: models.ModelOption
  ) -> None:
    workspace = next(
      workspace
      for workspace in self._agent_workspaces()
      if workspace.agent is agent
    )
    self._update(
      workspace.id,
      daemon_pb2.Workspace(
        preferences=daemon_pb2.WorkspacePreferences(
          definition=mirror_pb2.AgentDefinition(
            model=selection.model_id,
            model_api=selection.api,
            model_base_url=selection.base_url,
          ),
        )
      ),
      (
        'preferences.definition.model',
        'preferences.definition.model_api',
        'preferences.definition.model_base_url',
      ),
    )

  def retarget_remote(self, agent: workspaces.Agent, session_id: str) -> None:
    workspace = next(
      (
        workspace
        for workspace in self._agent_workspaces()
        if workspace.agent is agent
      ),
      None,
    )
    if workspace is None or not workspace.remote or workspace.id == session_id:
      return
    previous_id = workspace.id
    previous = self._records.get(previous_id)
    try:
      record = self._client.create(
        daemon_pb2.CreateWorkspaceRequest(
          workspace_id=session_id,
          name=previous.name if previous is not None else workspace.alias or '',
          preferences=previous.preferences
          if previous is not None
          else self._defaults(agent),
          remote=daemon_pb2.RemoteSource(session_id=session_id),
        )
      )
      self._records[session_id] = record
      workspace.location = workspaces.RemoteSession(session_id)
      self._save_layout()
      if previous is not None:
        self._client.close_workspace(previous_id)
        self._client.drop(
          daemon_pb2.DeleteWorkspaceRequest(workspace_id=previous_id)
        )
      self._records.pop(previous_id, None)
    except grpc.RpcError as error:
      agent.show_error(f'could not save session preferences: {error}')

  async def select(self, tab_index: int) -> bool:
    target = self._workspace_at(tab_index)
    if target is None or target is self.selected:
      return False
    self.selected = target
    self._save_layout()
    return True

  async def select_workspace(self, identity: str) -> bool:
    for index, workspace in enumerate(self.visible_workspaces()):
      if workspace.identity == identity or (
        isinstance(workspace, workspaces.AgentWorkspace)
        and workspace.id == identity
      ):
        return await self.select(index)
    self.selected.show_error('That session is no longer available.')
    return False

  async def cycle(self, offset: int) -> None:
    available = self.visible_workspaces()
    if available:
      index = (
        available.index(self.selected) if self.selected in available else 0
      )
      await self.select((index + offset) % len(available))

  def _remove_view(self, view: workspaces.Workspace) -> None:
    self._preserve_draft(view)
    self.workspaces.remove(view)
    if self.selected is view:
      available = self.visible_workspaces()
      self.selected = available[0] if available else self._empty
    self._save_layout()
    self.report_workspace_count()
    self.running_sessions_changed()

  def drop_confirmation(self) -> actions.DropConfirmation | None:
    source = self.selected
    if (
      isinstance(source, workspaces.AgentWorkspace)
      and isinstance(source.location, daemon_pb2.Workspace)
      and not can_delete_workspace_files(source.location)
    ):
      raise ValueError(
        'Only managed worktrees and sandboxes can be dropped. '
        'Use /close or remove the folder from /open; its files will be kept.'
      )
    if self._settings.read().get('confirm_drop') is False:
      return None
    return actions.DropConfirmation(
      self.selected.label,
      self.selected.id
      if isinstance(self.selected, workspaces.AgentWorkspace)
      and self.selected.remote
      else str(self.selected.path or ''),
    )

  async def drop(self, *, skip_confirmation: bool = False) -> bool:
    source = self.selected
    # Bind overlapping requests to their original selection, not its replacement.
    await asyncio.sleep(0)
    if source is not self.selected:
      return False
    if isinstance(source, workspaces.PendingWorkspace):
      if source.error is None:
        return False
      if record := self._records.get(source.identity.removeprefix('tab:')):
        await asyncio.to_thread(
          self._client.drop,
          daemon_pb2.DeleteWorkspaceRequest(workspace_id=record.id),
        )
      self._remove_view(source)
      return True
    try:
      self.drop_confirmation()
    except ValueError as error:
      source.show_error(str(error))
      return False
    if not any(
      workspace is not source and workspace.available
      for workspace in self._agent_workspaces()
    ):
      source.show_error('cannot drop the only available session')
      return False
    if not isinstance(source, workspaces.AgentWorkspace) or source.dropping:
      return False
    if skip_confirmation:
      self._settings.update(
        lambda settings: settings.update({'confirm_drop': False})
      )
    source.dropping = True
    available = self.visible_workspaces()
    self.selected = next(
      (
        workspace
        for workspace in self.workspaces[self.workspaces.index(source) + 1 :]
        if workspace.available
      ),
      available[-1],
    )
    self.report_workspace_count()
    self.running_sessions_changed()

    async def remove() -> None:
      try:
        if source.remote:
          if self._delete_remote_session is None:
            raise RuntimeError('remote endpoint is not configured')
          await asyncio.to_thread(self._client.close_workspace, source.id)
          await self._delete_remote_session(source.id)
          await asyncio.to_thread(
            self._client.drop,
            daemon_pb2.DeleteWorkspaceRequest(workspace_id=source.id),
          )
        else:
          await asyncio.to_thread(self._client.close_workspace, source.id)
          await asyncio.to_thread(
            self._client.drop,
            daemon_pb2.DeleteWorkspaceRequest(
              workspace_id=source.id,
              delete_files=True,
            ),
          )
        if source in self.workspaces:
          self._remove_view(source)
          await asyncio.shield(self._detach(source))
        self.refresh()
      except (OSError, RuntimeError, grpc.RpcError) as error:
        message = f'Could not delete {source.label!r}: {error}'
        if (
          not self.remote_host
          and not source.remote
          and source in self.workspaces
        ):
          try:
            record = await asyncio.to_thread(
              self._client.open_workspace, source.id
            )
          except grpc.RpcError as reopen_error:
            self._disconnected_workspaces.add(source.id)
            message += f'\nCould not reopen its tab: {reopen_error}'
          else:
            self._preserve_draft(source)
            await asyncio.shield(self._detach(source))
            if not self._shutting_down and source in self.workspaces:
              self._disconnected_workspaces.add(source.id)
              restored = self._new_view(record)
              restored.show_error(message)
              self.workspaces[self.workspaces.index(source)] = restored
              if self.selected is source:
                self.selected = restored
        self.selected.show_error(message)
        self.refresh()

    def completed(task: asyncio.Task[None]) -> None:
      del task
      source.dropping = False
      self.report_workspace_count()
      self.running_sessions_changed()

    self._background(remove()).add_done_callback(completed)
    return True

  def _local_source(self) -> workspaces.AgentWorkspace:
    if (
      isinstance(self.selected, workspaces.AgentWorkspace)
      and not self.selected.remote
    ):
      return self.selected
    for workspace in self._agent_workspaces():
      if not workspace.remote and workspace.available:
        return workspace
    raise worktrees.WorktreeError('no local session is available')

  def _creation_source(self) -> workspaces.AgentWorkspace | None:
    try:
      return self._local_source()
    except worktrees.WorktreeError:
      return None

  def new_workspace_selector(self) -> selection.Selector[actions.Action]:
    """Offer supported workspace kinds, remembering the last accepted choice."""
    previous = self._settings.read().get('new_workspace_kind')
    source = self._creation_source()
    source_path = (
      source.agent.workspace if source is not None else self.launch_workspace
    )
    if source is not None and source.is_sandbox:
      worktree_detail = 'Requires a host repository, not a sandbox'
    elif self.remote_host:
      worktree_detail = f'From {source_path}'
    elif (repository := project.repository_root(source_path)) is not None:
      worktree_detail = f'From {repository}'
    else:
      worktree_detail = f'Requires Git or Jujutsu · {source_path}'
    workspace_actions: dict[str, actions.Action] = {
      'normal': actions.CreateWorkspace('normal'),
      'folder': actions.RestorePrompt('/folder '),
      'worktree': actions.CreateWorkspace('worktree'),
      'sandbox': actions.CreateSandbox(),
      'remote': actions.CreateRemote(),
    }
    return selection.Selector(
      id='new-workspace',
      title='New workspace',
      choices=(
        selection.Choice('normal', 'New', str(self.launch_workspace)),
        selection.Choice(
          'folder', 'New working directory', 'Enter a path to another directory'
        ),
        selection.Choice('worktree', 'Worktree', worktree_detail),
        selection.Choice(
          'sandbox', 'Sandbox', 'Run in an isolated VM on the daemon host'
        ),
      )
      + (
        (selection.Choice('remote', 'Remote', 'Run on the configured server'),)
        if self._remote_enabled
        else ()
      ),
      numbered=True,
      on_select=lambda kind: workspace_actions[kind],
      selected=previous
      if isinstance(previous, str) and previous in ('normal', 'worktree')
      else 'normal',
    )

  async def create_workspace(
    self,
    kind: str,
    *,
    directory: pathlib.Path | None = None,
    create_directory: bool = False,
  ) -> str:
    kinds: dict[str, Literal['directory', 'worktree', 'sandbox']] = {
      'normal': 'directory',
      'folder': 'directory',
      'worktree': 'worktree',
    }
    if kind not in kinds:
      raise ValueError(f'unknown session kind: {kind!r}')
    if kind == 'normal' and directory is None:
      directory = self.launch_workspace
    if directory is not None:
      resolved = await asyncio.to_thread(self.resolve_directory, str(directory))
      directory = pathlib.Path(resolved.directory)
      if not resolved.exists and not create_directory:
        raise FileNotFoundError(f'directory is unavailable: {directory}')
      if not self.is_trusted_directory(directory):
        raise ValueError('Trust the workspace directory before opening it')
    pending, _ = self._start_creation(
      self._creation_source(),
      kinds[kind],
      directory=directory,
      create_directory=create_directory,
    )
    self._settings.update(
      lambda settings: settings.update({'new_workspace_kind': kind})
    )
    return pending.identity

  async def open_folder(
    self,
    source: pathlib.Path | str,
    directory: pathlib.Path,
    *,
    name: str | None = None,
    create_directory: bool = False,
  ) -> WorkspaceSnapshot:
    sender = self._source_workspace(source)
    resolved = await asyncio.to_thread(
      self.resolve_directory, str(directory), base=sender.agent.workspace
    )
    path = pathlib.Path(resolved.directory)
    if not resolved.exists and not create_directory:
      raise FileNotFoundError(f'directory is unavailable: {path}')
    if not self.is_trusted_directory(path):
      raise ValueError(
        f'Folder {str(path)!r} is not trusted. Open it in Mirror interactively first.'
      )
    _, task = self._start_creation(
      sender,
      'directory',
      name,
      directory=path,
      create_directory=create_directory,
    )
    return self._workspace_snapshot(await asyncio.shield(task), sender.id)

  def _update(
    self,
    identity: str,
    record: daemon_pb2.Workspace,
    fields: Sequence[str],
  ) -> daemon_pb2.Workspace:
    record.id = identity
    result = self._client.update(record, fields)
    self._records[identity] = result
    return result

  async def fork_workspace(
    self, source: pathlib.Path | str, name: str | None = None
  ) -> workspaces.PendingWorkspace:
    """Show a pending tab while the daemon forks files and conversation history."""
    sender = self._source_workspace(source)
    if self._shutting_down or sender.remote or sender.is_sandbox:
      raise worktrees.WorktreeError(
        'forking requires an available host workspace'
      )
    if (
      sender.agent.queued_message_count
      or sender.state is actions.WorktreeAgentState.RUNNING
    ):
      raise worktrees.WorktreeError(
        'wait for the conversation and queued messages before forking'
      )
    identity = str(uuid.uuid4())
    pending = workspaces.PendingWorkspace(
      f'tab:{identity}', name or f'fork-{identity[:8]}', create_session=False
    )
    self.workspaces.append(pending)
    self.workspaces.sort(key=lambda workspace: workspace.remote)
    task = asyncio.create_task(self._fork_workspace(sender, pending, identity))
    self._creation_tasks.add(task)

    def completed(task: asyncio.Task[workspaces.AgentWorkspace]) -> None:
      self._creation_tasks.discard(task)
      if not task.cancelled():
        task.exception()

    task.add_done_callback(completed)
    return pending

  async def _fork_workspace(
    self,
    source: workspaces.AgentWorkspace,
    pending: workspaces.PendingWorkspace,
    identity: str,
  ) -> workspaces.AgentWorkspace:
    try:
      pending.path = await source.agent.fork_workspace(
        identity, None, pending.label
      )
      snapshot = await asyncio.to_thread(self._client.snapshot, self._query)
      self._apply_snapshot(snapshot)
      return self._target_workspace(identity)
    except Exception as error:  # Retain the identity for inspection after partial provisioning.
      pending.error = (
        f'Fork {identity} ({pending.label}) did not finish: {error}. '
        'Check this outcome before retrying; no files were removed.'
      )
      raise worktrees.WorktreeError(pending.error) from error

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
    source = next(
      (view for view in self._agent_workspaces() if view.id == workspace_id),
      None,
    )
    await asyncio.to_thread(self._client.close_workspace, workspace_id)
    if source is not None and source in self.workspaces:
      self._remove_view(source)
      await asyncio.shield(self._detach(source))
    return True

  async def delete_workspace(
    self, workspace_id: str, *, delete_files: bool = False
  ) -> None:
    """Remove a confirmed entry, optionally deleting managed workspace files."""
    if self._daemon_stopped or self._shutting_down:
      raise ValueError('Daemon disconnected; waiting to reconnect.')
    record = self._records.get(workspace_id)
    if record is None:
      raise ValueError('That workspace is no longer available.')
    if delete_files and not can_delete_workspace_files(record):
      raise ValueError('Only managed worktrees and sandboxes can delete files.')
    if record.kind == daemon_pb2.WORKSPACE_KIND_REMOTE:
      if self._delete_remote_session is None:
        raise ValueError('Remote AgentRPC endpoint is not configured')
      await asyncio.to_thread(self._client.close_workspace, workspace_id)
      await self._delete_remote_session(record.session_id)
    else:
      await asyncio.to_thread(self._client.close_workspace, workspace_id)
    view = next(
      (view for view in self._agent_workspaces() if view.id == workspace_id),
      None,
    )
    if view is not None:
      self._remove_view(view)
      await asyncio.shield(self._detach(view))
    await asyncio.to_thread(
      self._client.drop,
      daemon_pb2.DeleteWorkspaceRequest(
        workspace_id=workspace_id, delete_files=delete_files
      ),
    )
    await self._reconcile()

  def _start_creation(
    self,
    source: workspaces.AgentWorkspace | None,
    kind: Literal['directory', 'worktree', 'sandbox'],
    name: str | None = None,
    *,
    directory: pathlib.Path | None = None,
    preferences: daemon_pb2.WorkspacePreferences | None = None,
    sandbox: sandboxes.Sandbox | None = None,
    copy_repository: bool = False,
    create_directory: bool = False,
  ) -> tuple[
    workspaces.PendingWorkspace, asyncio.Task[workspaces.AgentWorkspace]
  ]:
    if self._shutting_down or self._daemon_stopped:
      raise worktrees.WorktreeError('client is disconnected')
    if copy_repository and source is None:
      raise ValueError('Copying a repository requires a source workspace')
    source_path = (
      source.agent.workspace if source is not None else self.launch_workspace
    )
    if kind == 'worktree':
      if source is not None and source.is_sandbox:
        raise ValueError('Worktrees require a host repository, not a sandbox')
      if not self.remote_host and project.repository_root(source_path) is None:
        raise ValueError(f'Worktrees require Git or Jujutsu: {source_path}')
    if (
      (source is None and kind != 'sandbox') or kind == 'worktree'
    ) and not self.is_trusted_directory(directory or source_path):
      raise ValueError(
        'Trust the workspace directory before creating a workspace'
      )
    request_id = str(uuid.uuid4())
    pending = workspaces.PendingWorkspace(
      f'tab:{request_id}', name or 'Creating…'
    )
    self.workspaces.append(pending)
    self.workspaces.sort(key=lambda view: view.remote)
    resolved = self._defaults(source.agent if source is not None else None)
    if preferences is not None:
      resolved.MergeFrom(preferences)
    request = daemon_pb2.CreateWorkspaceRequest(
      workspace_id=request_id,
      name=name or (directory or source_path).name,
      preferences=resolved,
      create_directory=create_directory,
    )
    if kind == 'directory':
      request.directory = str(directory or source_path)
    elif kind == 'worktree':
      request.worktree_from = str(source_path)
    elif sandbox is not None:
      request.sandbox.CopyFrom(
        daemon_pb2.SandboxSource(
          source_workspace_id=source.id
          if copy_repository and source is not None
          else None,
          image=sandbox.image,
          egress=sandbox.egress,
          cpus=sandbox.cpus,
          memory_mib=sandbox.memory_mib,
        )
      )
    else:
      raise ValueError('Sandbox creation requires explicit resource choices')

    async def create() -> workspaces.AgentWorkspace:
      try:
        record = await asyncio.to_thread(self._client.create, request)
        if self._shutting_down or self._daemon_stopped:
          raise worktrees.WorktreeError('client is disconnected')
        # Pre-creation snapshots must not remove the newly created tab.
        if self._reconcile_task is not None:
          self._reconcile_task.cancel()
        if self._watch_task is not None:
          self._watch_task.cancel()
          self._watch_task = asyncio.create_task(self._watch())
        if kind == 'worktree':
          # A new checkout inherits trust only from its trusted source.
          self.trust_directory(pathlib.Path(record.directory))
        existing = next(
          (view for view in self._agent_workspaces() if view.id == record.id),
          None,
        )
        view = existing or self._new_view(
          record, active=self.selected is pending, create_session=True
        )
        if existing is None:
          view.identity = pending.identity
          view.draft = pending.draft
          view.resume_pending = False
        if pending in self.workspaces:
          self.workspaces[self.workspaces.index(pending)] = view
        elif view not in self.workspaces:
          self.workspaces.append(view)
        if self.selected is pending:
          self.selected = view
        pending.finished = True
        self._save_layout()
        self.report_workspace_count()
        self.running_sessions_changed()
        if kind != 'directory':
          await view.start()
        return view
      except Exception as error:  # Keep request failures on the initiating tab.
        pending.error = str(error)
        raise

    task = asyncio.create_task(create())
    self._creation_tasks.add(task)

    def completed(task: asyncio.Task[workspaces.AgentWorkspace]) -> None:
      self._creation_tasks.discard(task)
      if not task.cancelled():
        task.exception()

    task.add_done_callback(completed)
    return pending, task

  async def create_worktree_workspace(
    self,
    source: pathlib.Path | str,
    name: str | None = None,
    *,
    preferences: daemon_pb2.WorkspacePreferences | None = None,
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
    preferences: daemon_pb2.WorkspacePreferences | None = None,
  ) -> WorkspaceSubmission:
    _, task = self._start_creation(
      self._source_workspace(source),
      'worktree',
      name,
      preferences=preferences,
    )
    workspace = await asyncio.shield(task)
    if message is not None:
      return await self.send_message(source, workspace.id, message)
    return WorkspaceSubmission(
      self._workspace_snapshot(workspace, source), 'created'
    )

  async def create(self, name: str | None = None) -> bool:
    self._start_creation(
      self._local_source(),
      'worktree',
      name,
    )
    return True

  async def create_sandbox(
    self, name: str | None = None
  ) -> workspaces.SandboxWorkspace | workspaces.PendingWorkspace | None:
    if self.daemon_defaults is not None:
      options = self.daemon_defaults.sandbox
      pending, _ = self._start_creation(
        self._creation_source(),
        'sandbox',
        name,
        sandbox=sandboxes.Sandbox(
          str(uuid.uuid4()),
          options.image,
          tuple(options.egress),
          options.cpus,
          options.memory_mib,
        ),
      )
      return pending
    pending = workspaces.SandboxWorkspace(
      f'tab:{uuid.uuid4()}',
      '◇ ' + (name or 'sandbox'),
      source=self._local_source()
      if any(not workspace.remote for workspace in self._agent_workspaces())
      else None,
      name=name,
      preferences=self._defaults(),
    )
    try:
      pending.options = sandbox_setup.Options.load(self._settings.read())
      missing = [
        label for label, ready in sandbox_setup.host_checks() if not ready
      ]
      if missing:
        raise sandboxes.SandboxError('\n'.join(missing))
      if not self._sandboxd.is_file():
        raise sandboxes.SandboxError(
          f'This install is missing its sandbox helper: {self._sandboxd}'
        )
    except (OSError, ValueError, RuntimeError) as error:
      pending.error = str(error)
    self.workspaces.append(pending)
    return pending

  def start_sandbox(
    self,
    pending: workspaces.SandboxWorkspace,
    action: Literal['install', 'download', 'start', 'copy'] = 'start',
  ) -> None:
    if pending.launching or self._shutting_down:
      return
    pending.error = None
    pending.setup_action = action

    async def approved_setup() -> workspaces.Workspace:
      try:
        if action == 'install':
          pending.phase = 'Downloading Microsandbox…'
          pending.options.executable = await sandbox_setup.install_runtime()
        elif action == 'download':
          pending.phase = 'Downloading sandbox environment…'
          await sandbox_setup.prepare_image(pending.options)
        else:
          if (
            not pending.options.runtime_ready()
            or not pending.options.image_ready()
          ):
            raise sandboxes.SandboxError(
              'Approve runtime and environment setup first.'
            )
          pending.options.save(self._settings)
          created, task = self._start_creation(
            pending.source,
            'sandbox',
            pending.name,
            sandbox=sandboxes.Sandbox.from_settings(self._settings.read()),
            copy_repository=action == 'copy',
          )
          created.draft = pending.draft
          if self.selected is pending:
            self.selected = created
          self.workspaces.remove(pending)
          return await asyncio.shield(task)
        pending.options.save(self._settings)
        return pending
      except BaseException as error:
        pending.error = str(error) or type(error).__name__
        raise

    pending.task = asyncio.create_task(approved_setup())

  async def cancel_sandbox(self, pending: workspaces.SandboxWorkspace) -> None:
    """Reap provisioning before the tab or its private clone can be removed."""
    if pending.task is not None:
      pending.task.cancel()
      await asyncio.gather(pending.task, return_exceptions=True)

  def _open_remote(self, session_id: str) -> workspaces.AgentWorkspace | None:
    for workspace in self.visible_workspaces():
      if (
        isinstance(workspace, workspaces.AgentWorkspace)
        and workspace.remote
        and workspace.id == session_id
      ):
        return workspace
    return None

  def _remote_tab_index(self, session_id: str) -> int | None:
    workspace = self._open_remote(session_id)
    if workspace is None:
      return None
    return self.visible_workspaces().index(workspace)

  def _remote_record(
    self, session_id: str, title: str | None = None
  ) -> daemon_pb2.Workspace:
    snapshot = self._client.snapshot(self._query)
    record = next(
      (record for record in snapshot.workspaces if record.id == session_id),
      None,
    )
    if record is None:
      record = self._client.create(
        daemon_pb2.CreateWorkspaceRequest(
          workspace_id=session_id,
          name=_normalize_alias(title) if title is not None else '',
          remote=daemon_pb2.RemoteSource(session_id=session_id),
        )
      )
    if record.kind != daemon_pb2.WORKSPACE_KIND_REMOTE:
      raise ValueError('the session is already registered as a local workspace')
    return record

  async def _append_remote(
    self,
    session_id: str,
    *,
    create_session: bool,
    title: str | None = None,
    remote_workspace: mirror_pb2.Workspace | None = None,
    escalated: bool = False,
  ) -> workspaces.AgentWorkspace:
    record = await asyncio.to_thread(self._remote_record, session_id, title)
    existing = self._open_remote(session_id)
    if existing is not None:
      if escalated:
        existing.agent.escalate()
      return existing
    workspace = self._new_view(
      record,
      create_session=create_session,
      remote_workspace=remote_workspace,
      escalated=escalated,
    )
    self.workspaces.append(workspace)
    self._save_layout()
    self.report_workspace_count()
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
    try:
      sessions = await self._list_remote_sessions()
      for session in sessions:
        if session not in self._records:
          self._records[session] = await asyncio.to_thread(
            self._remote_record, session
          )
    except (
      Exception
    ) as error:  # Isolate failure of an optional remote connection.
      logging.getLogger(__name__).warning(
        'Remote session discovery failed: %s', type(error).__name__
      )
      self.selected.show_error(str(error))

  async def create_remote(
    self,
    title: str | None = None,
    remote_workspace: mirror_pb2.Workspace | None = None,
    *,
    escalated: bool = False,
  ) -> int | None:
    """Create or select a remote tab; escalate only when explicitly requested."""
    if not self._remote_enabled:
      self.selected.show_error('remote AgentRPC endpoint is not configured')
      return None
    create = title is None
    try:
      identity = (
        str(uuid.UUID(title)) if title is not None else str(uuid.uuid4())
      )
    except ValueError:
      identity, create = str(uuid.uuid4()), True
    existing = self._open_remote(identity)
    if existing is not None:
      if escalated:
        existing.agent.escalate()
      return self._remote_tab_index(identity)
    workspace = await self._append_remote(
      identity,
      create_session=create,
      title=title if create else None,
      remote_workspace=remote_workspace if create else None,
      escalated=escalated,
    )
    await workspace.start()
    return self._remote_tab_index(identity)

  async def close(self) -> None:
    """Detach this UI; accepted backend operations belong to the daemon."""
    self._shutting_down = True
    for workspace in self.workspaces:
      self._preserve_draft(workspace)
    self._view_state.landing_draft = self._empty.draft
    self._view_state.landing_session = self._empty.session_id
    for workspace in self.workspaces:
      if isinstance(workspace, workspaces.SandboxWorkspace):
        await self.cancel_sandbox(workspace)
    tasks = [
      task
      for task in (
        self._watch_task,
        self._reconcile_task,
        self._remote_session_task,
        *self._tasks,
      )
      if task is not None
    ]
    creations = tuple(self._creation_tasks)
    for task in (*tasks, *creations):
      if task not in self._detach_tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.gather(*creations, return_exceptions=True)
    await asyncio.gather(*(workspace.close() for workspace in self.workspaces))
    self._client.close()
