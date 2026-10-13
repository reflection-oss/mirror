"""The interactive Rich + prompt_toolkit REPL."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import functools
import json
import os
import pathlib
import subprocess
import tempfile
import time
import uuid
from collections.abc import (
  Awaitable,
  Callable,
  Coroutine,
  Iterable,
  Mapping,
  Sequence,
)
from typing import Any, cast, override

import grpc.aio
from mirror.protocol import agent_pb2
from prompt_toolkit import buffer as _buffer
from prompt_toolkit import cursor_shapes as _cursor_shapes
from prompt_toolkit import document as _document
from prompt_toolkit import enums as _enums
from prompt_toolkit import filters as _filters
from prompt_toolkit import formatted_text as _formatted_text
from prompt_toolkit import history as _history
from prompt_toolkit import mouse_events as _mouse_events
from prompt_toolkit import key_binding as _key_binding
from prompt_toolkit import styles as _style
from prompt_toolkit.application import application as _application
from prompt_toolkit.application import current as _current

# prompt_toolkit exposes only the shadowing `run_in_terminal` function.
from prompt_toolkit.application import run_in_terminal as _run_in_terminal
from prompt_toolkit.formatted_text import ansi as _ansi
from prompt_toolkit.layout import containers as _containers
from prompt_toolkit.layout import controls as _controls
from prompt_toolkit.layout import dimension as _dimension
from prompt_toolkit.layout import layout as _layout
from prompt_toolkit.layout import processors as _processors
from prompt_toolkit.layout import scrollable_pane as _scrollable_pane
from rich import console as _console
from rich import rule as _rule
from rich import text as _text

from mirror import config as _config
from mirror import skills as _skills
from mirror import workspace as _workspace
from mirror.cli import actions as _actions
from mirror.cli import agents as _agents
from mirror.cli import api_keys as _api_keys
from mirror.cli import bash_approvals
from mirror.cli import cli_store as _cli_store
from mirror.cli import commands as _commands
from mirror.cli import conversation_metrics as _conversation_metrics
from mirror.cli import effects as _effects
from mirror.cli import extended_keys as _extended_keys
from mirror.cli import extension as _extension
from mirror.cli import host_tools as _host_tools
from mirror.cli import keymap as _keymap
from mirror.cli import local_rpc_client as _local_rpc_client
from mirror.cli import logging as _logging
from mirror.cli import models as _models
from mirror.cli import onboarding as _onboarding
from mirror.cli import reducer as _reducer
from mirror.cli import reflection as _reflection
from mirror.cli import remote_rpc_client as _remote_rpc_client
from mirror.cli import rendering as _rendering
from mirror.cli import themes as _themes
from mirror.cli import rpc_client as _rpc_client
from mirror.cli import selection as _selection
from mirror.cli import shell as _shell
from mirror.cli import state as _state
from mirror.cli import stream as _stream_mod
from mirror.cli import terminal as _terminal_mod
from mirror.cli import tool_approvals as _tool_approvals
from mirror.cli import tool_groups as _tool_groups
from mirror.cli import transcript as _transcript_mod
from mirror.cli import transcript_view as _transcript_view
from mirror.cli import user_extensions as _user_extensions
from mirror.cli import workspace_application as _workspace_application
from mirror.cli import workspace_controller as _workspace_controller
from mirror.cli import workspace_tags as _workspace_tags
from mirror.cli import workspaces as _workspaces
from mirror.cli.components import activity as _activity
from mirror.cli.components import banner as _banner
from mirror.cli.components import breathing_dots as _breathing_dots
from mirror.cli.components import completion as _completion
from mirror.cli.components import component as _component
from mirror.cli.components import drop as _drop
from mirror.cli.components import failure as _failure
from mirror.cli.components import pending as _pending
from mirror.cli.components import queued as _queued
from mirror.cli.components import resume as _resume
from mirror.cli.components import rewind as _rewind
from mirror.cli.components import secret_prompt as _secret_prompt
from mirror.cli.components import selector as _selector
from mirror.cli.components import status_bar as _status_bar
from mirror.cli.components import tasks as _tasks
from mirror.cli.components import worktree_tabs as _worktree_tabs
from mirror.cli.composer import editing as _editing
from mirror.cli.composer import processors as _composer_processors
from mirror.cli.composer import skill_completion as _skill_completion
from mirror.cli.composer import surface as _composer_surface
from mirror.cli.extensions.inference_controls import (
  extension as _inference_controls,
)
from mirror.cli.statsig import config as _statsig_config
from mirror.cli.statsig.events import client as _statsig_client
from mirror.cli.statsig.events import event as _statsig_event

# This limit also gates composer redraws, not just streaming token updates.
_MAX_REDRAW_FPS = 120
# Skip prompt_toolkit's per-SIGWINCH erase until the drag settles.
_RESIZE_SETTLE_SECONDS = 0.2


class _SQLiteHistory(_history.History):
  """Read-only prompt history backed by SQLite."""

  def __init__(
    self,
    database: str | pathlib.Path | None,
    conversation_id: str,
  ) -> None:
    super().__init__()
    self._database = database
    self._conversation_id = conversation_id

  @override
  def load_history_strings(self) -> Iterable[str]:
    with _cli_store.Store(self._database) as store:
      return store.recent_user_messages(self._conversation_id)

  def reload(self, conversation_id: str) -> None:
    """Reset the cache so the next load uses the new conversation_id."""
    self._conversation_id = conversation_id
    self._loaded = False
    self._loaded_strings = []

  @override
  def store_string(self, string: str) -> None:
    del string  # The worker writes user messages to SQLite; the TUI only reads.


class _Repl:
  """A prompt_toolkit REPL with a persistent bottom pane over Rich output."""

  def __init__(
    self,
    worker_command: Sequence[str],
    worker_config: Mapping[str, object],
    *,
    session_id: str,
    workspace: pathlib.Path | None = None,
    workspace_id: str | None = None,
    console: _console.Console | None = None,
    reasoning_effort: str = 'max',
    database: str | pathlib.Path | None = None,
    scrollback_limit: int = 1000,
    flush_scrollback: bool = False,
    workspace_effects: asyncio.Queue[tuple[_Repl, _effects.Effect]]
    | None = None,
    extension_registry: _extension.ExtensionRegistry,
    workspace_controller: _workspace_controller.WorkspaceController,
    user_extensions: _user_extensions.UserExtensions | None = None,
    tool_approvals: _tool_approvals.ToolApprovals | None = None,
    workspace_extensions: Sequence[str] = (),
    workspace_tools: Sequence[str] | None = None,
    active: bool = True,
    exit_on_worker_failure: bool = True,
    remote_endpoint: str | None = None,
    remote_secure: bool = True,
    configured_remote_endpoint: str | None = None,
    first_render: asyncio.Event | None = None,
    extensions_loaded: asyncio.Task[None] | None = None,
    fullscreen: bool = False,
    initial_theme: str = 'dark',
  ) -> None:
    self._fullscreen = fullscreen
    self._theme_repaint_pending = False
    self._tool_approvals = tool_approvals or _tool_approvals.ToolApprovals()
    self._worker_command = tuple(worker_command)
    self._session_id = session_id
    self._extensions_loaded = extensions_loaded
    self._workspace_controller = workspace_controller
    self._secret_input: _buffer.Buffer | None = None
    self._onboarding_model: (
      asyncio.Future[_models.ModelOption | None] | None
    ) = None
    self._host_tools_registry = _host_tools.HostToolRegistry(
      selected=workspace_tools
    )
    self._workspace = (workspace or pathlib.Path.cwd()).resolve()
    self._worker_configuration = _rpc_client.WorkerConfiguration(
      worker_config, cwd=self._workspace
    )
    self._remote_endpoint = remote_endpoint
    self._conversation_metrics = _conversation_metrics.ConversationMetrics(
      remote=remote_endpoint is not None
    )
    # Version the built-in execution contract, including /bin/bash -c.
    self._local_approval_scope = json.dumps((
      'local-v1',
      str(self._workspace),
      os.environ.get('MIRROR_WORKSPACE'),
    ))
    self._remote_secure = remote_secure
    self._workspace_effects = workspace_effects
    self._navigation: _containers.AnyContainer = _containers.Window(height=0)
    self._exit_on_worker_failure = exit_on_worker_failure
    self._user_extensions = user_extensions or _user_extensions.UserExtensions(
      extension_registry,
      pathlib.Path.home() / '.mirror' / 'extensions',
      cwd=self._workspace,
      bundled=pathlib.Path(__file__).with_name('extensions'),
    )
    self._extension_context = _extension.Context(
      workspace=_extension.Workspace(
        cwd=self._workspace,
        remote=remote_endpoint is not None,
        id=workspace_id,
        shell=_shell.Shell(self._workspace)
        if remote_endpoint is None
        else None,
      ),
      dispatch=lambda action: self.dispatch(action),
      get_state=lambda: self._state,
      enqueue=self._enqueue,
      refresh_tools=extension_registry.refresh_tools,
      workspace_controller=workspace_controller,
      call_tool=self.call_tool,
      transcript=self._extension_transcript,
      create_agent=self.create_agent,
      builtins=_agents.builtins,
      extension_tools=lambda: tuple(
        tool.name for tool in self._host_tools_registry.tools()
      ),
      restart_worker=(
        (lambda updates: self._restart_worker(updates, replay=False))
        if remote_endpoint is None
        else None
      ),
      worker_command=self._worker_command,
      worker_cwd=self._workspace,
      run_terminal=self._run_terminal_command,
      remote_endpoint=configured_remote_endpoint or remote_endpoint,
      remote_secure=remote_secure,
      extensions=self._user_extensions,
    )
    self._extension_host = extension_registry.create_host(
      self._extension_context,
      self._host_tools_registry,
      extensions=workspace_extensions,
    )
    extension_states = self._extension_host.initial_states
    if _themes.EXTENSION_ID not in extension_states:
      # Paint the saved palette before the deferred extension load completes.
      extension_states = extension_states.configured(
        _themes.EXTENSION_ID, {'name': initial_theme}
      )
    conversation_id = cast(str, worker_config['conversation_id'])
    self._state: _state.State = _state.State(
      activity=_state.ActivityState(),
      status_bar=_state.StatusBarState(
        model=cast(str, worker_config['model']),
        reasoning_effort=reasoning_effort,
        approval_mode=cast(str, worker_config.get('approval_mode', 'approve')),
      ),
      session=_state.SessionState(
        conversation_id=conversation_id,
        remote=remote_endpoint is not None,
      ),
      lifecycle=_state.LifecycleState(active=active),
      workspace=str(self._workspace),
      supervised=workspace_effects is not None,
      extensions=extension_states,
    )
    self._database = database
    self._scrollback_limit = scrollback_limit
    self._activation_preview_pending = False
    self._client: _rpc_client.Client | None = None
    self._worker_ready = asyncio.Event()
    keymap_error: _actions.ShowText | None = None
    try:
      remappings = _keymap.load_config()
    except (ValueError, OSError) as error:
      remappings = ()
      keymap_error = _actions.ShowText(
        f'{error}. Using default keybindings.', style='ui.error'
      )
    self._keymap = _keymap.Keymap(
      self._keybinding_layers,
      self.dispatch,
      dismiss=self._dismissal_action,
      remappings=remappings,
      enabled=_filters.Condition(lambda: self._secret_input is None),
    )
    self._project_skills = tuple(_skills.load_skills(self._workspace))
    self._extension_skills = self._extension_host.skills
    self._skill_mention_processor = _skill_completion.SkillMentionProcessor(())
    self._update_available_skills()
    extension_left = self._extension_host.component(
      _extension.RenderRegion.STATUS_LEFT
    )
    extension_right = self._extension_host.component(
      _extension.RenderRegion.STATUS_RIGHT
    )
    self._components = _component.ComponentCollection({
      'activity': _activity.Activity(),
      'banner': _banner.Banner(
        self._session_directory(),
        lambda columns, rows: (
          self._bottom_pane.preferred_height(columns, rows).preferred
          + self._welcome_notices_window.preferred_height(
            columns, rows
          ).preferred
          + self._welcome_header_window.preferred_height(
            columns, rows
          ).preferred
        ),
      ),
      'completion': _completion.Completion(),
      'drop': _drop.DropConfirmation(),
      'extensions_below_activity': self._extension_host.component(
        _extension.RenderRegion.WIDGET_BELOW_ACTIVITY
      ),
      'extensions_above_prompt': self._extension_host.component(
        _extension.RenderRegion.WIDGET_ABOVE_PROMPT
      ),
      'extensions_below_prompt': self._extension_host.component(
        _extension.RenderRegion.WIDGET_BELOW_PROMPT
      ),
      'failure': _failure.Failure(),
      'pending': _pending.Pending(),
      'queued': _queued.Queued(),
      'resume': _resume.Resume(),
      'selector': _selector.Selector(
        get_max_height=(lambda: self._selector_max_height())
        if fullscreen
        else None
      ),
      'secret': _secret_prompt.SecretPrompt(),
      'rewind_banner': _rewind.RewindBanner(),
      'rewind_picker': _rewind.RewindPicker(),
      'status_bar': _status_bar.StatusBar(extension_left, extension_right),
      'tasks': _tasks.Tasks(),
    })
    self._selector_search = _selector.SearchInput(
      lambda query: self.dispatch(_actions.SelectorSearchChanged(query))
    )
    self._selector_keys = _controls.FormattedTextControl('', focusable=True)
    self._extension_secret_input = _secret_prompt.SecretInput(
      lambda length: self.dispatch(_actions.SecretLengthChanged(length))
    )
    self._history = _SQLiteHistory(
      self._database, self._state.session.conversation_id
    )

    def create_terminal_task(
      coroutine: Coroutine[Any, Any, None],
    ) -> asyncio.Task[None]:
      return self._app.create_background_task(coroutine)

    self._terminal = _terminal_mod.TerminalOutput(
      console or _console.Console(theme=_themes.rich_theme()),
      theme_name=lambda: _themes.current(self._state.extensions),
      is_replaying=self._replacing_history,
      is_active=lambda: self._state.lifecycle.active,
      create_task=create_terminal_task,
      flush_scrollback=flush_scrollback,
      request_draw=(lambda: self._app.invalidate()) if fullscreen else None,
    )
    self._transcript = _transcript_mod.Transcript(
      view_mode=lambda: self._state.status_bar.view_mode,
      skill_names=self._available_skill_names,
      get_rewind_entry_id=lambda: (
        self._state.rewind.staged.target.entry_id
        if self._state.rewind.staged is not None
        else None
      ),
    )
    self._stream_pane = _stream_mod.StreamPane(
      self._terminal,
      get_state=lambda: self._state,
      dispatch=lambda actions: self.dispatch(*actions),
      skill_names=self._available_skill_names,
      get_columns=lambda: self._app.output.get_size().columns,
      get_max_lines=self._streaming_max_visible_lines,
    )
    self._live_stamp: (
      tuple[_state.StreamPreview | None, _actions.ViewMode] | None
    ) = None
    self._live_blocks_cache: tuple[_console.RenderableType, ...] = ()
    self._streaming_answer = _rendering.StreamingAnswer(
      self._available_skill_names
    )
    self._committed_stamp: (
      tuple[str | None, str | None, _actions.ViewMode] | None
    ) = None
    self._committed_blocks: tuple[_console.RenderableType, ...] = ()
    self._viewport = _transcript_view.TranscriptView(
      self._transcript,
      self._live_blocks,
      navigate=self.dispatch,
      theme_name=lambda: _themes.current(self._state.extensions),
    )
    self._drafts: dict[str, _document.Document] = {}
    self._editing_prompt = False
    self._input = _buffer.Buffer(
      name='composer',
      history=self._history,
      multiline=True,
      on_text_changed=self._on_composer_changed,
      on_cursor_position_changed=self._on_composer_changed,
      read_only=_filters.Condition(self._modal_ui_active),
    )
    self._prompt_editor: _editing.Editor | None = None
    self._app = self._build_app()
    if first_render is not None:

      def on_render(_: _application.Application[str | None]) -> None:
        first_render.set()

      self._app.after_render += on_render
    if keymap_error is not None:

      def show_keymap_error() -> None:
        self.dispatch(keymap_error)

      self._app.pre_run_callables.append(show_keymap_error)
    columns = max(1, self._app.output.get_size().columns)
    self.dispatch(_actions.TerminalResized(columns))

  def _update_available_skills(self) -> None:
    by_name = {
      skill.name: skill
      for skill in (*self._extension_skills, *self._project_skills)
    }
    self._available_skills = tuple(by_name.values())
    self._available_skill_names = frozenset(by_name)
    self._skill_completion = _skill_completion.SkillCompletion(
      self._available_skills
    )
    self._skill_mention_processor.update(self._available_skills)

  def dispatch(
    self, *actions: _actions.Action
  ) -> tuple[asyncio.Task[None], ...]:
    """Reduce actions, repaint once, and return the started effect tasks."""
    if not actions:
      return ()
    if any(
      isinstance(action, (_actions.SetViewMode, _actions.CycleViewMode))
      for action in actions
    ):
      self._flush_tool_group()
      self._stream_pane.reset_height()
    was_running = self._running
    next_state = self._state
    effects: list[_effects.Effect] = []
    for action in actions:
      extension_states, extension_effects = self._extension_host.reduce(
        next_state, action
      )
      next_state, action_effects = _reducer.reduce(
        next_state, action, observed_at=time.monotonic()
      )
      next_state = dataclasses.replace(next_state, extensions=extension_states)
      effects.extend((*action_effects, *extension_effects))
    self._theme_repaint_pending |= _themes.current(
      self._state.extensions
    ) != _themes.current(next_state.extensions)
    self._state = next_state
    if self._theme_repaint_pending and self._can_repaint_theme():
      self._theme_repaint_pending = False
      if self._transcript.has_conversation:
        effects.append(_effects.RerenderHistory(preserve_header=True))
    if self._running != was_running:
      self._workspace_controller.running_sessions_changed()
    self._host_tools_registry.activity_changed(next_state.activity.busy)
    extension_skills = self._extension_host.skills
    if self._extension_skills != extension_skills:
      self._extension_skills = extension_skills
      self._update_available_skills()
      self._host_tools_registry.context_changed()
    if hasattr(self, '_app'):
      self._app.refresh_interval = (
        1 / _MAX_REDRAW_FPS if next_state.activity.busy else 0.1
      )
      self._app.invalidate()
    tasks: list[asyncio.Task[None]] = []
    for effect in effects:
      result = self._run_effect(effect)
      if isinstance(result, tuple):
        tasks.extend(result)
      elif result is not None:
        tasks.append(result)
    if hasattr(self, '_app'):
      self._sync_prompt_editing()
    return tuple(tasks)

  def _run_effect(
    self, effect: _effects.Effect
  ) -> asyncio.Task[None] | tuple[asyncio.Task[None], ...] | None:
    """Execute an imperative effect emitted by the pure reducer."""
    match effect:
      case _effects.NavigateHistory(action):
        if self._fullscreen and not self._modal_ui_active():
          self._viewport.navigate(action)
        return None
      case _effects.ShowExtensions():
        return self.dispatch(self._user_extensions.selector())
      case _effects.ChangeExtension(request):
        return self._app.create_background_task(
          self._user_extensions.change(self._extension_context, request)
        )
      case _effects.RefreshCompletion():
        return self.dispatch(
          _completion.choices_changed(
            self._state.composer.text,
            self._state.composer.cursor,
            self._skill_completion,
            self.get_available_commands(),
          )
        )
      case _effects.InsertComposerNewline():
        self._insert_newline()
      case _effects.InsertComposerText(text):
        self._input.save_to_undo_stack()
        self._input.insert_text(text)
      case _effects.EditComposerExternally():
        return self._app.create_background_task(self._edit_composer())
      case _effects.UndoComposer():
        self._app.current_buffer.undo()
      case _effects.RedoComposer():
        self._app.current_buffer.redo()
      case _effects.AcceptSelection():
        return self.dispatch(*self._selector_choice())
      case _effects.AdjustSelection(delta):
        return self.dispatch(*self._selector_adjust(delta))
      case _effects.CancelSelection():
        return self.dispatch(*self._selector_cancel())
      case _effects.EmitText(text, style):
        renderable = (
          _text.Text(text, style=style) if isinstance(text, str) else text
        )
        if self._live_notices():
          self._transcript.add_decoration(renderable, welcome=True)
          return None
        if self._fullscreen:
          return self._record_decoration(renderable)
        if self._state.session.replaying:
          self._terminal.trailer.append(renderable)
        return self._app.create_background_task(
          self._terminal.emit([renderable])
        )
      case _effects.EmitError(message):
        if self._fullscreen:
          return self._record_decoration(_rendering.error_text(message))
        return self._app.create_background_task(
          self._terminal.emit([_rendering.error_text(message)])
        )
      case _effects.RecordError(message):
        return self._record_supervisor_error(message)
      case _effects.ExitApp(
        result, erase_when_done, defer_until_running, handoff
      ):
        self._app.erase_when_done = erase_when_done
        if defer_until_running and not self._app.is_running:
          self._app.pre_run_callables.append(
            functools.partial(self._app.exit, result=result)
          )
        elif handoff and result is not None:
          self._app.exit_for_handoff(result)
        else:
          self._app.exit(result=result)
      case (
        _effects.OpenFolder()
        | _effects.CloseWorkspace()
        | _effects.CreateManagedWorktree()
        | _effects.DropManagedWorktree()
        | _effects.RenameWorktree()
        | _effects.WorkspaceTagCommand()
        | _effects.LoadRemoteSessions()
        | _effects.RefreshWorkspaces()
        | _effects.CycleWorktree()
        | _effects.SelectWorktree()
        | _effects.SelectWorkspace()
        | _effects.SaveWorktreePreferences()
        | _effects.RetargetRemote()
      ):
        if self._workspace_effects is not None:
          self._workspace_effects.put_nowait((self, effect))
      case _effects.CreateRemoteAgent():
        return self._app.create_background_task(self._prepare_remote(effect))
      case _effects.SendInferenceConfiguration():
        self._send_inference_config()
      case _effects.CompactConversation():
        if self._client is None or self._state.lifecycle.quitting:
          raise RuntimeError('Agent worker is not running.')
        request = self._new_request()
        request.compact.SetInParent()
        self._send_request(request)
      case _effects.DetachTools():
        request = self._new_request()
        request.detach_tools.SetInParent()
        self._send_request(request)
      case _effects.SetWorkerApprovalMode(mode):
        _config.save_approval_mode(mode)
        self._worker_configuration = self._worker_configuration.updated({
          'approval_mode': mode
        })
        request = self._new_request()
        request.set_approval_mode.mode = (
          agent_pb2.APPROVAL_MODE_AUTO
          if mode == 'auto'
          else agent_pb2.APPROVAL_MODE_HITL
        )
        self._send_request(request)
      case _effects.SendApprovalDecision(call_id, approved, remember, source):
        if approved and remember is not None:
          try:
            self._tool_approvals.remember(remember)
          except (OSError, ValueError, RecursionError):
            self.dispatch(
              _actions.ShowText(
                'Could not save tool approval; approving this call only. '
                'Check ~/.mirror/tool_approvals.json and restart Mirror.'
              )
            )
        request = self._new_request()
        request.hitl_response.call_id = call_id
        request.hitl_response.approved = approved
        request.hitl_response.source = source
        self._send_request(request)
      case _effects.RunSlashCommand(text):
        return self._app.create_background_task(self._run_slash(text))
      case _effects.SendQueuedInstruction(instruction):
        self._send_instruction(instruction)
        _statsig_client.get().log_event(
          _statsig_event.Event(
            'mirror_message_queued',
            value=1,
            metadata={
              'conversation_id': self._state.session.conversation_id,
              'workspace_kind': 'remote'
              if self._remote_endpoint is not None
              else 'local',
            },
          )
        )
      case _effects.RunTurn(instruction, rewind_to):
        message_id = None
        if rewind_to is None:
          message_id = uuid.uuid4().hex
          self.dispatch(
            _actions.TurnStarted(),
            _actions.InputSubmitted(message_id, instruction),
          )
          if self._state.lifecycle.started:
            self._send_instruction(instruction, message_id=message_id)
            return None
        return self._app.create_background_task(
          self._run_turn(
            instruction, rewind_to=rewind_to, message_id=message_id
          )
        )
      case _effects.UserPromptSubmitted(instruction):
        self._log_user_prompt_submitted(instruction)
      case _effects.FocusComposer():
        self._app.layout.focus(self._input_window)
      case _effects.FocusTasks():
        self._app.layout.focus(self._task_window)
      case _effects.CancelTask(task_id):
        request = self._new_request()
        request.cancel_task.task_id = task_id
        self._send_request(request)
      case _effects.StartNewConversation():
        return self._app.create_background_task(
          self._switch_session(uuid.uuid4().hex, create=True)
        )
      case _effects.LoadRewindTurns(conversation_id):
        return self._app.create_background_task(
          self._load_rewind_turns(conversation_id)
        )
      case _effects.ResolveModel(model_id):
        return self._app.create_background_task(self._resolve_model(model_id))
      case _effects.ReloadResumeChoices(scope):
        return self._app.create_background_task(
          self._reload_resume_choices(scope)
        )
      case _effects.ResolveResumeReference(reference):
        return self._app.create_background_task(
          self._resolve_resume_reference(reference)
        )
      case _effects.ResumeConversation(info, allow_workspace_mismatch):
        return self._app.create_background_task(
          self._switch_session(
            info.id,
            create=False,
            allow_workspace_mismatch=allow_workspace_mismatch,
          )
        )
      case _effects.CancelModelSelection():
        if (
          self._onboarding_model is not None
          and not self._onboarding_model.done()
        ):
          self._onboarding_model.set_result(None)
      case _effects.ForgetReflectionKey():
        return self._app.create_background_task(self._forget_reflection_key())
      case _effects.SwitchModel(option):
        return self._app.create_background_task(self._switch_model(option))
      case _effects.SwitchRemoteModel(model_id, api):
        self._switch_remote_model(model_id, api)
      case _effects.SaveLastModel():
        selection = self.model_selection
        if self._workspace_effects is not None:
          self._workspace_effects.put_nowait((self, effect))
        if self._remote_endpoint is None:
          _models.save_last_selection(selection)
      case _effects.SyncSelectorInput(active, query):
        if self._selector_search.buffer.text != query:
          self._selector_search.buffer.text = query
          self._selector_search.buffer.cursor_position = len(query)
        if self._state.selector.numbered:
          self._input_window.content = self._selector_keys
        else:
          self._input_window.content = (
            self._selector_search.control if active else self._prompt_control()
          )
        self._app.layout.focus(self._input_window)
      case _effects.SyncSecretInput():
        self._extension_secret_input.clear()
        if not self._state.selector.active:
          self._input_window.content = self._prompt_control()
          self._app.layout.focus(self._input_window)
      case _effects.SubmitSecret():
        return self._extension_host.submit_secret(
          self._extension_secret_input.take()
        )
      case _effects.CancelSecret():
        request = self._state.secret.request
        if request is not None and request.cancel_action is not None:
          return self.dispatch(_actions.ClearSecret(), request.cancel_action)
      case _effects.LoadPromptHistory():
        return self._app.create_background_task(self._load_prompt_history())
      case _effects.LoadSelectorChoices(selector_id):
        return self._app.create_background_task(
          self._load_selector_choices(selector_id)
        )
      case _effects.SetComposerText(text, cursor):
        # Programmatic draft restoration must work while a modal owns input.
        self._input.set_document(
          _document.Document(text, len(text) if cursor is None else cursor),
          bypass_readonly=True,
        )
      case _effects.ResetComposer(append_to_history):
        self._reset_input(append_to_history=append_to_history)
      case _effects.RerenderHistory(note, drop_scrollback, preserve_header):
        return self._app.create_background_task(
          self._rerender(
            note=note,
            drop_scrollback=drop_scrollback,
            preserve_header=preserve_header,
          )
        )
      case _effects.RetryWorker():
        return self._app.create_background_task(self._retry_worker())
      case _effects.InterruptTurn():
        if self._client is None:
          self.dispatch(_actions.UnsentInputsDiscarded())
          return None
        request = self._new_request()
        request.interrupt.SetInParent()
        self.dispatch(_actions.InterruptSent(request.message_id))
        self._send_request(request)
      case _effects.RunExtensionEffect():
        return self._extension_host.handle_effect(effect)
    return None

  async def _run_terminal_command(
    self,
    argv: tuple[str, ...],
    cwd: str,
    *,
    on_success: Callable[[], None] | None = None,
  ) -> bool:
    """Suspend input and report success after a local process releases the terminal."""
    if not self._state.lifecycle.active:
      return False
    running = asyncio.ensure_future(
      _run_in_terminal(
        # Change directory in the spawned child, not via fork in the gRPC host.
        functools.partial(
          subprocess.run,
          (
            '/bin/sh',
            '-c',
            'cd -- "$1" && shift && exec "$@"',
            'mirror-editor',
            cwd,
            *argv,
          ),
          # The process stdout/stderr point at logs, not the UI terminal.
          stdout=self.console.file,
          stderr=self.console.file,
          close_fds=False,
          check=False,
        ),
        in_executor=True,
      )
    )
    try:
      try:
        result = await asyncio.shield(running)
      finally:
        # Cancellation must not restore the UI while the child owns the tty.
        if not running.done():
          await running
    except OSError as error:
      self.dispatch(
        _actions.ShowText(
          f'Could not run terminal command: {error}', style='ui.error'
        )
      )
      return False
    if result.returncode:
      self.dispatch(
        _actions.ShowText(
          f'Command exited with status {result.returncode}.', style='ui.error'
        )
      )
      return False
    if on_success is not None:
      on_success()
    return True

  async def _edit_composer(self) -> None:
    """Round-trip the draft through the existing terminal handoff."""
    if (
      self._editing_prompt
      or not self._state.lifecycle.active
      or self._modal_ui_active()
      or self._state.secret.active
      or self._secret_input is not None
      or self._onboarding_model is not None
      or self._state.approval.request is not None
      or self._state.tasks.selected_id is not None
      or self._state.session.pending_switch is not None
    ):
      return
    self._editing_prompt = True
    original = self._input.document
    conversation = self._state.session.conversation_id
    try:
      with tempfile.TemporaryDirectory(prefix='mirror-prompt-') as directory:
        path = pathlib.Path(directory) / 'prompt.md'
        # Adapt the prompt to a text file's terminating newline and back.
        path.write_text(original.text + '\n', encoding='utf-8')

        def apply_edit() -> None:
          text = path.read_text(encoding='utf-8').removesuffix('\n')
          if (
            self._input.document != original
            or self._state.session.conversation_id != conversation
            or not self._state.lifecycle.active
            or self._state.secret.active
            or self._secret_input is not None
            or self._state.session.pending_switch is not None
          ):
            self.dispatch(
              _actions.ShowText('Draft changed; editor result not applied.')
            )
            return
          if text != original.text:
            self._input.save_to_undo_stack()
            self._input.exit_selection()
            self._app.vi_state.reset()
            self._input.set_document(
              _document.Document(text, len(text)),
              bypass_readonly=True,
            )

        await self._run_terminal_command(
          _terminal_mod.editor_command(path),
          self._state.workspace,
          on_success=apply_edit,
        )
    except (OSError, UnicodeError, ValueError) as error:
      self.dispatch(
        _actions.ShowText(f'Could not edit prompt: {error}', style='ui.error')
      )
    finally:
      self._editing_prompt = False

  @property
  def console(self) -> _console.Console:
    """The terminal console used for final supervisor messages."""
    return self._terminal.console

  @property
  def workspace(self) -> pathlib.Path:
    """The worktree this agent operates in."""
    return self._workspace

  @property
  def reasoning_effort(self) -> str:
    """The reasoning effort inherited by a newly created worktree."""
    return self._state.status_bar.reasoning_effort

  @property
  def approval_mode(self) -> str:
    """The approval mode inherited by a newly created worktree."""
    return self._state.status_bar.approval_mode

  def create_agent(
    self,
    options: _agents.Options,
    on_message: _rpc_client.MessageHandler,
    on_error: _rpc_client.ErrorHandler,
  ) -> _rpc_client.Client:
    """Prepare an independent local agent using this workspace's settings."""
    if self._remote_endpoint is not None:
      raise NotImplementedError(
        'Extension agent creation is currently local-only'
      )
    configuration = self._worker_configuration.updated({
      'model': self._state.status_bar.model,
      'approval_mode': self._state.status_bar.approval_mode,
      'conversation_id': str(uuid.uuid4()),
      'create_session': True,
    })
    if options.system_prompt is not None:
      configuration = configuration.updated({
        'system_prompt': options.system_prompt
      })
    return _rpc_client.Client(
      self._transport(configuration, on_message),
      str(configuration.values['conversation_id']),
      on_message=on_message,
      on_error=on_error,
      host_tools_registry=self._host_tools_registry,
      inference_config=configuration.inference_config(
        self._state.status_bar.model,
        self._state.status_bar.reasoning_effort,
        remote=False,
      ),
    )

  async def call_tool(
    self, name: str, arguments: Mapping[str, object]
  ) -> _host_tools.ToolResult:
    """Execute a client-owned or workspace tool without running the model."""
    if self._client is None:
      raise RuntimeError('agent is not connected')
    return await self._client.call_tool(name, arguments)

  @property
  def model(self) -> str:
    """The model currently configured for this agent."""
    return self._state.status_bar.model

  @property
  def model_selection(self) -> _models.ModelOption:
    """The complete effective selection, without provider credentials."""
    inference = self._worker_configuration.inference_config(
      self.model,
      self.reasoning_effort,
      remote=self._remote_endpoint is not None,
    )
    return _models.ModelOption(
      self.model, self.model, api=inference.api, base_url=inference.base_url
    )

  @property
  def conversation_id(self) -> str:
    """The agent's current durable conversation."""
    return self._state.session.conversation_id

  @property
  def activity_status(self) -> str:
    """The agent's current short activity label."""
    return self._state.activity.status

  @property
  def queued_message_count(self) -> int:
    """The number of instructions waiting behind the active turn."""
    return len(self._state.turn.queued)

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """Return the agent's current durable trace."""
    return self._transcript.entries()

  def _extension_transcript(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """Copy the durable trace once the open session has finished replaying."""
    if self._state.session.replaying:
      return ()
    copies: list[agent_pb2.AgentMessage] = []
    for entry in self._transcript.entries():
      copies.append(agent_pb2.AgentMessage())
      copies[-1].CopyFrom(entry)
    return tuple(copies)

  def submit_instruction(self, instruction: str) -> None:
    """Queue an instruction for this agent."""
    self.dispatch(_actions.SubmitInstruction(instruction))

  @property
  def _running(self) -> bool:
    return self.worktree_state is _actions.WorktreeAgentState.RUNNING

  @property
  def worktree_state(self) -> _actions.WorktreeAgentState:
    """The current agent state shown by the worktree controller."""
    if self._state.lifecycle.failure is not None:
      return _actions.WorktreeAgentState.FAILED
    if not self._state.lifecycle.started:
      return _actions.WorktreeAgentState.NOT_STARTED
    if self._state.approval.request is not None:
      return _actions.WorktreeAgentState.ATTENTION
    # History replacement blocks input, but is not itself agent activity.
    # A live turn may still arrive before the replacement has been painted.
    if self._state.activity.busy and (
      not self._replacing_history()
      or self._state.activity.work_started is not None
    ):
      return _actions.WorktreeAgentState.RUNNING
    if self._state.lifecycle.unread:
      return _actions.WorktreeAgentState.UNREAD
    return _actions.WorktreeAgentState.IDLE

  @property
  def selector(self) -> _state.SelectorState:
    return self._state.selector

  def handoff(self, result: str) -> None:
    """Return navigation to the CLI loop without moving the terminal frame."""
    if self._state.lifecycle.active:
      self._app.exit_for_handoff(result)

  def show_error(self, message: str) -> None:
    """Publish an error reported by the worktree supervisor."""
    self.dispatch(_actions.SupervisorError(message))

  def _record_supervisor_error(self, message: str) -> asyncio.Task[None] | None:
    """Retain and display a supervisor error as an imperative effect."""
    renderable = _rendering.error_text(message)
    self._flush_tool_group()
    self._transcript.add_decoration(renderable)
    if not self._state.lifecycle.active:
      return None
    if self._state.session.replaying:
      return self._app.create_background_task(self._terminal.emit([renderable]))
    if not self._replacing_history():
      self._terminal.queue_emit([renderable])
    return None

  def _record_decoration(self, renderable: _console.RenderableType) -> None:
    """Keep a UI-only event reconstructible without adding it to the trace."""
    self._flush_tool_group()
    self._transcript.add_decoration(renderable)
    if self._state.lifecycle.active and not self._replacing_history():
      self._terminal.queue_emit([renderable])

  def _abort_stream(self) -> None:
    """Retain unfinalized output before closing its only live representation."""
    if self._fullscreen:
      identities = [
        self._transcript.add_decoration(block) for block in self._live_blocks()
      ]
      self._viewport.settle_live(identities)
    self._stream_pane.abort()

  def _live_notices(self) -> bool:
    return self._secret_input is None and not self._transcript.has_conversation

  def _append_entry(self, entry: agent_pb2.AgentMessage) -> None:
    """Hand welcome notices to scrollback before the first conversation entry."""
    was_empty = not self._transcript.has_conversation
    self._transcript.append(entry)
    if was_empty and self._transcript.has_conversation:
      self._terminal.queue_emit(list(self._transcript.welcome_notices()))

  def _can_focus_tasks(self) -> bool:
    """Whether Down should move from the prompt into the task rows."""
    if (
      self._extra_compact()
      or not bool(self._state.tasks.tasks)
      or self._state.tasks.selected_id is not None
      or self._state.approval.request is not None
      or _completion.visible(self._state)
      or self._state.drop_confirmation is not None
      or self._state.resume.active
      or self._state.selector.active
      or self._state.secret.active
      or self._state.rewind.active
    ):
      return False
    if self._app.layout.current_window is not self._input_window:
      return False
    document = self._input.document
    return (
      not document.text
      or document.cursor_position_row == len(document.lines) - 1
    )

  def _streaming_max_visible_lines(self) -> int:
    """Fit the live stream, and the blank row under it, above the bottom pane."""
    if not hasattr(self, '_app'):
      return 1
    size = self._app.output.get_size()
    columns = max(1, size.columns)
    bottom_height = self._bottom_pane.preferred_height(
      columns, size.rows
    ).preferred
    return max(1, size.rows - bottom_height - _stream_mod.STREAM_PANE_MARGIN)

  def _on_composer_changed(self, buffer: _buffer.Buffer) -> None:
    """Publish the current composer snapshot and reset menu selection."""
    self.dispatch(
      _actions.ComposerChanged(buffer.text, buffer.cursor_position),
      _completion.choices_changed(
        buffer.text,
        buffer.cursor_position,
        self._skill_completion,
        self.get_available_commands(),
      ),
    )

  def _reset_input(self, *, append_to_history: bool = False) -> None:
    """Reset prompt_toolkit and publish the resulting composer snapshot."""
    self._app.vi_state.reset()
    self._input.reset(append_to_history=append_to_history)
    composer = _actions.ComposerState(
      self._input.text,
      self._input.cursor_position,
    )
    if composer != self._state.composer:
      self._on_composer_changed(self._input)

  def _session_directory(self) -> str:
    return (
      'remote' if self._remote_endpoint is not None else str(self._workspace)
    )

  def _same_session_directory(self, directory: str) -> bool:
    return _workspace.resolve_cwd(directory) == str(self._workspace)

  def get_available_commands(self) -> tuple[_commands.CommandDef, ...]:
    return _commands.resolve_commands((
      *_commands.COMMANDS,
      *self._extension_host.slash_commands,
    ))

  def _command_context(self) -> _commands.CommandContext:
    return _commands.CommandContext(
      model=self._state.status_bar.model,
      reasoning_effort=self._state.status_bar.reasoning_effort,
      conversation_id=self._state.session.conversation_id,
      busy=self._state.activity.busy,
      cwd=self._session_directory(),
      remote=self._remote_endpoint is not None,
      session_id=self._session_id,
    )

  def _extra_compact(self) -> bool:
    """Whether the current view mode hides thinking and tool activity."""
    return self._state.status_bar.view_mode is _actions.ViewMode.EXTRA_COMPACT

  def _store(self) -> _cli_store.Store:
    return _cli_store.Store(self._database)

  def _session_infos(
    self, scope: _workspace.SessionScope
  ) -> list[_cli_store.ConversationInfo]:
    """Read conversations through a short-lived SQLite connection."""
    with self._store() as store:
      return store.list_infos(cwd=str(self._workspace), scope=scope)

  async def _reload_resume_choices(
    self, scope: _workspace.SessionScope
  ) -> None:
    """Refresh scoped picker rows without blocking the event loop."""
    choices = await asyncio.to_thread(self._session_infos, scope)
    self.dispatch(_actions.ResumeChoicesLoaded(scope, tuple(choices)))

  async def _prepare_remote(self, request: _effects.CreateRemoteAgent) -> None:
    if (
      await self._extension_host.prepare_remote(
        _actions.CreateRemote(request.title, request.workspace)
      )
      and self._workspace_effects is not None
    ):
      self._workspace_effects.put_nowait((self, request))

  async def _resolve_resume_reference(self, reference: str) -> None:
    """Resolve one conversation reference and dispatch the immutable result."""
    with self._store() as store:
      matches = tuple(store.find_by_reference(reference))
    await asyncio.gather(
      *self.dispatch(_actions.ResumeReferenceResolved(reference, matches))
    )

  async def _resolve_model(self, model_id: str) -> None:
    """Resolve a model id and dispatch its configured option."""
    option = _models.find_model(model_id)
    await asyncio.gather(*self.dispatch(_actions.ModelResolved(option)))

  async def _forget_reflection_key(self) -> None:
    """Remove saved credentials and reconnect if the active model needs them."""
    try:
      removed = _reflection.remove_key()
    except OSError as error:
      await asyncio.gather(
        *self.dispatch(_actions.ShowText(str(error), style='ui.error'))
      )
      return
    message = (
      'Removed the saved Reflection API key.'
      if removed
      else 'No saved Reflection API key was found.'
    )
    if _reflection.has_environment_key():
      message += (
        ' REFLECTION_API_KEY is still set; unset it in your shell and restart'
        ' Mirror to stop using it.'
      )
    message += ' This does not revoke the key or cancel in-flight requests.'
    await asyncio.gather(*self.dispatch(_actions.ShowText(message)))
    if self.model_selection.api == 'reflection' and not _reflection.has_key():
      await self.ensure_api_key()

  async def _switch_model(self, option: _models.ModelOption) -> None:
    """Select the inference configuration for subsequent model requests."""
    if self._onboarding_model is not None:
      if not self._onboarding_model.done():
        self._onboarding_model.set_result(option)
      return
    selected = await self._authenticate_model(option, initial=False)
    if selected is not None:
      self._apply_model(selected)

  def _switch_remote_model(self, model_id: str, api: str) -> None:
    """Commit a server model selection, which needs no local credentials."""
    current = self._worker_configuration.values.get('remote_model_api')
    self._worker_configuration = self._worker_configuration.updated({
      'model': model_id,
      'remote_model_api': api or current,
    })
    self.dispatch(_actions.ModelSwitchFinished(model_id))

  def _apply_model(self, option: _models.ModelOption) -> None:
    """Commit a model selection after its sign-in flow completes."""
    updates = {
      'model': option.model_id,
      'model_api': option.api or self.model_selection.api or None,
      'model_base_url': option.base_url or None,
      'base_url': None,
    }
    self._worker_configuration = self._worker_configuration.updated(updates)
    self.dispatch(_actions.ModelSwitchFinished(option.model_id))

  async def _load_prompt_history(self) -> None:
    request = self._state.selector.selector
    texts = await asyncio.to_thread(
      lambda: tuple(self._history.load_history_strings())
    )
    if self._state.selector.selector is not request:
      return
    self.dispatch(
      _actions.SelectorChoicesLoaded(
        'history',
        tuple(
          _selection.Choice(text, ' '.join(text.split())) for text in texts
        ),
      )
    )

  async def _load_selector_choices(self, selector_id: str) -> None:
    """Run the open selector's async producer and feed back its choices."""
    request = self._state.selector.selector
    if request is None or request.id != selector_id or request.load is None:
      return
    loader = request.load
    try:
      choices = await loader()
    except Exception as error:  # noqa: BLE001, isolation boundary for producers
      # The producer is arbitrary extension code, so any failure stays here and
      # surfaces to the user rather than tearing down the event loop.
      if self._state.selector.selector is request:
        self.dispatch(*self._selector_cancel())
      self.dispatch(
        _actions.ShowText(
          f'Could not load {selector_id}: {error}', style='ui.warning'
        )
      )
      return
    if self._state.selector.selector is request:
      self.dispatch(_actions.SelectorChoicesLoaded(selector_id, choices))

  def _selector_choice(self) -> tuple[_actions.Action, ...]:
    """Return the selection event and its configured action."""
    if self._state.selector.search_pending:
      return ()
    selector = self._state.selector.selector
    choice = self._state.selector.selected_choice()
    if selector is None or choice is None:
      return (_actions.ClearSelector(),)
    return (
      _actions.SelectorChoiceSelected(selector.id, choice.value),
      cast(_actions.Action, selector.on_select(choice.value)),
    )

  def _selector_adjust(self, delta: int) -> tuple[_actions.Action, ...]:
    selector = self._state.selector.selector
    choice = self._state.selector.selected_choice()
    if selector is None or selector.on_adjust is None or choice is None:
      return ()
    return (cast(_actions.Action, selector.on_adjust(choice.value, delta)),)

  def _selector_cancel(self) -> tuple[_actions.Action, ...]:
    selector = self._state.selector.selector
    cancel = selector.cancel_action if selector is not None else None
    if self._onboarding_model is not None:
      cancel = _actions.CancelModelSelection()
    return (
      (_actions.ClearSelector(), cast(_actions.Action, cancel))
      if cancel is not None
      else (_actions.ClearSelector(),)
    )

  def _prompt_control(self) -> _controls.UIControl:
    return (
      self._extension_secret_input.control
      if self._state.secret.active
      else self._composer_control
    )

  def _modal_ui_active(self) -> bool:
    """Whether an overlay owns the keyboard and the input is read-only."""
    return (
      self._state.drop_confirmation is not None
      or self._state.resume.active
      or self._state.selector.active
      or self._state.secret.active
      or self._state.rewind.active
    )

  def _rewind_escape_active(self) -> bool:
    """Whether Esc should unstage a rewind rather than close an overlay."""
    if (
      self._state.resume.active
      or self._state.selector.active
      or self._state.secret.active
      or _completion.visible(self._state)
    ):
      return False
    return self._state.rewind.active or self._state.rewind.staged is not None

  async def _user_turns(self) -> tuple[_actions.UserTurn, ...]:
    """Return eligible user rewind points from oldest to newest."""
    client = self._client
    if client is None:
      return ()
    entries = await client.messages()
    return tuple(
      _actions.UserTurn(entry.id, entry.chat.user.content)
      for entry in entries
      if entry.WhichOneof('payload') == 'chat'
      and entry.chat.WhichOneof('message') == 'user'
      and not entry.chat.HasField('display')
      and entry.chat.user.content.strip()
    )

  async def _load_rewind_turns(self, conversation_id: str) -> None:
    """Load rewind points and dispatch them against their conversation."""
    turns = await self._user_turns()
    await asyncio.gather(
      *self.dispatch(_actions.RewindTurnsLoaded(conversation_id, turns))
    )

  async def _switch_session(
    self,
    conversation_id: str,
    *,
    create: bool,
    allow_workspace_mismatch: bool = False,
  ) -> None:
    """Replace the worker process with one for another conversation."""
    self.dispatch(
      _actions.ClearResume(),
      _actions.SessionSwitchStarted(
        _actions.PendingSwitch(conversation_id, create)
      ),
    )
    # Keep the current screen until the replacement process's replay lands.
    self._transcript.clear_replayed()
    self._start_switch_screen()
    self._app.invalidate()
    await self._restart_worker(
      {
        'conversation_id': conversation_id,
        'create_session': create,
        'allow_workspace_mismatch': allow_workspace_mismatch,
      },
      replay=True,
      opening_conversation=conversation_id,
    )

  def _start_switch_screen(self) -> None:
    """Buffer the header the switched-to history will print under."""
    switch = self._state.session.pending_switch
    if switch is None:
      return
    self.dispatch(_actions.SetToolGroup(None))
    self._terminal.emit_buffer.clear()
    self._terminal.pin_header(
      _rule.Rule(
        'New conversation' if switch.create else 'Resumed conversation',
        style='ui.info',
      ),
    )

  def _replacing_history(self) -> bool:
    """Whether a replayed history is on its way to the screen."""
    return (
      self._state.session.replaying
      or self._state.session.rerendering
      or self._state.session.pending_switch is not None
    )

  def _sync_console_size(self) -> None:
    size = self._app.output.get_size()
    # Rich only honors an explicit width when height is set too.
    self._terminal.console.size = (max(1, size.columns), max(1, size.rows))

  async def _settle_resize(self, generation: int) -> None:
    await asyncio.sleep(_RESIZE_SETTLE_SECONDS)
    if generation != self._state.session.resize_generation:
      return
    self._sync_console_size()
    if not self._replacing_history():
      self._relayout_prompt()
    if self._state.status_bar.resize_dirty:
      self._app.invalidate()

  def _can_repaint_theme(self) -> bool:
    return (
      self._state.lifecycle.active
      and not self._state.activity.busy
      and not self._state.session.replaying
      and not self._state.session.rerendering
      and self._state.session.pending_switch is None
    )

  async def _rerender(
    self,
    *,
    note: str | None = None,
    drop_scrollback: bool = False,
    preserve_header: bool = False,
  ) -> None:
    """Redraw visible history at the current size and rewind point."""
    if self._fullscreen:
      if drop_scrollback:
        self._viewport.follow()
      self.dispatch(
        _actions.HistoryRendered(self._app.output.get_size().columns)
      )
      return
    if preserve_header:
      await self._stream_pane.wait_for_emits()
      if not self._can_repaint_theme():
        self._theme_repaint_pending = True
        return
    self.dispatch(_actions.RerenderStarted())
    await self._stream_pane.wait_for_emits()
    if drop_scrollback:
      divider = note or 'Screen cleared'
    else:
      divider = note or 'History re-rendered'
    if not preserve_header:
      self._terminal.pin_header(_rule.Rule(divider, style='ui.info'))
    self._app.invalidate()
    await self._repaint(replacing=True, drop_scrollback=drop_scrollback)

  def _sync_prompt_editing(self) -> None:
    suspended = (
      self._modal_ui_active()
      or self._secret_input is not None
      or self._onboarding_model is not None
      or self._state.approval.request is not None
      or self._state.tasks.selected_id is not None
    )
    editor = None if suspended else self._extension_host.prompt_editor
    if self._prompt_editor is not None and editor is None:
      self._app.output.reset_cursor_shape()
    self._prompt_editor = editor
    _editing.sync(
      self._app,
      self._input,
      self._prompt_editor.mode
      if self._prompt_editor
      else _enums.EditingMode.EMACS,
      suspended=suspended,
    )

  def _dismissal_action(self) -> _actions.Action | None:
    if self._onboarding_model is not None:
      return _actions.CancelModelSelection()
    dismissal = _keymap.dismissal_action(
      self._state,
      completion_visible=_completion.visible(self._state),
      rewind_escape_active=self._rewind_escape_active(),
    )
    if dismissal is not None:
      return dismissal
    if self._fullscreen and self._viewport.reading and not _filters.vi_mode():
      return _actions.NavigateHistory('latest')
    return None

  def _keybinding_layers(self) -> tuple[tuple[_keymap.Binding, ...], ...]:
    """Supply current UI availability and extension bindings."""
    if self._secret_input is not None:
      return ()
    if self._onboarding_model is not None:
      return (
        (
          _keymap.Binding(('up',), _actions.MoveSelection(-1)),
          _keymap.Binding(('down',), _actions.MoveSelection(1)),
          _keymap.Binding(('c-k',), _actions.MoveSelection(-1)),
          _keymap.Binding(('c-j',), _actions.MoveSelection(1)),
          _keymap.Binding(('enter',), _actions.AcceptSelection()),
          _keymap.Binding(('c-c',), _actions.QuitRequested()),
          _keymap.Binding(('c-d',), _actions.QuitRequested()),
        ),
      )
    return (
      _keymap.defaults(
        self._state,
        completion_visible=_completion.visible(self._state),
        modal_active=self._modal_ui_active(),
        can_focus_tasks=self._can_focus_tasks(),
        owned_history=self._fullscreen,
      ),
      self._extension_host.keybindings,
    )

  def _welcome_visible(self) -> bool:
    return (
      self._secret_input is None
      and not self._transcript.has_conversation
      and self._stream_pane.preview is None
    )

  def _component_state(self) -> _state.State:
    if not self._welcome_visible():
      return self._state
    return dataclasses.replace(
      self._state,
      layout=dataclasses.replace(
        self._state.layout,
        columns=_banner.input_width(self._state.layout.columns),
      ),
    )

  def _build_app(self) -> _workspace_application.WorkspaceApplication:
    """Assemble activity, menus, composer, and workspace widgets."""
    bindings = self._keymap.bindings
    component_host = _component.ComponentHost(self._component_state)

    pending_window = _containers.ConditionalContainer(
      component_host.mount(self._components['pending']),
      filter=_filters.Condition(
        lambda: not self._fullscreen or self._state.approval.request is not None
      ),
    )
    # The stream pane owns its trailing blank row and collapses when empty.
    streaming_window = _containers.Window(
      _controls.FormattedTextControl(
        self._stream_pane.ansi,
        show_cursor=False,
        get_cursor_position=self._stream_pane.cursor_position,
      ),
      height=lambda: _dimension.Dimension.exact(self._stream_pane.pane_height),
      wrap_lines=False,
    )
    queued_window = component_host.mount(self._components['queued'])
    completion_menu_window = component_host.mount(
      self._components['completion']
    )
    drop_confirm_window = component_host.mount(self._components['drop'])
    resume_menu_window = component_host.mount(self._components['resume'])
    selector_window = component_host.mount(self._components['selector'])
    secret_window = component_host.mount(self._components['secret'])
    rewind_banner_window = component_host.mount(
      self._components['rewind_banner']
    )
    rewind_menu_window = component_host.mount(self._components['rewind_picker'])
    worktree_window = _containers.DynamicContainer(lambda: self._navigation)
    footer_window = _containers.ConditionalContainer(
      component_host.mount(self._components['status_bar']),
      filter=_filters.Condition(lambda: self._secret_input is None),
    )
    setup_footer = _containers.ConditionalContainer(
      _containers.Window(
        _controls.FormattedTextControl([
          ('class:ui.accent bold', ' Enter'),
          ('class:ui.muted', ' to connect  ·  '),
          ('bold', 'Esc'),
          ('class:ui.muted', ' to choose another model  ·  '),
          ('bold', 'Ctrl+C'),
          ('class:ui.muted', ' to cancel'),
        ]),
        height=1,
      ),
      filter=_filters.Condition(lambda: self._secret_input is not None),
    )
    input_window = self._input_window = _containers.Window(
      _controls.BufferControl(
        buffer=self._input,
        include_default_input_processors=False,
        input_processors=[
          _processors.HighlightSelectionProcessor(),
          # Color typed `/…` first so BeforeInput's › prompt stays unstyled.
          _composer_processors.SlashInputProcessor(),
          self._skill_mention_processor,
          _composer_processors.CompletionPreviewProcessor(
            lambda: _completion.prompt_preview(self._state)
          ),
          _processors.BeforeInput(f'{_rendering.PROMPT} '),
        ],
      ),
      height=lambda: (
        _dimension.Dimension.exact(0)
        if self._state.selector.numbered
        else _dimension.Dimension(
          min=1,
          max=max(1, self._app.output.get_size().rows // 3)
          if self._fullscreen and hasattr(self, '_app')
          else None,
        )
      ),
      dont_extend_height=True,  # ...but never stretches past them.
      wrap_lines=True,
    )
    self._composer_control = input_window.content
    # Approval replaces the composer; failures leave commands usable.
    composer_visible = _filters.Condition(
      lambda: (
        self._state.approval.request is None
        or self._state.selector.active
        or self._state.secret.active
      )
    )
    composer_window = _containers.ConditionalContainer(
      _containers.HSplit([
        completion_menu_window,
        _composer_surface.wrap_input(
          input_window,
          lambda: (
            self._state.selector.active or _completion.visible(self._state)
          ),
          _containers.HSplit([footer_window, setup_footer]),
          theme_name=lambda: _themes.current(self._state.extensions),
        ),
      ]),
      filter=composer_visible,
    )
    modal_footer_window = _containers.ConditionalContainer(
      footer_window, filter=~composer_visible
    )
    composer_top_spacer = _containers.ConditionalContainer(
      _containers.Window(
        height=lambda: (
          1
          if self._secret_input is None
          and (
            activity_window.preferred_height(
              self._state.layout.columns, 1
            ).preferred
            or self._state.lifecycle.failure
            or extensions_above_prompt.preferred_height(
              self._state.layout.columns, 1
            ).preferred
          )
          else 0
        )
      ),
      filter=composer_visible,
    )
    failure_window = component_host.mount(self._components['failure'])
    activity_window = _containers.ConditionalContainer(
      component_host.mount(self._components['activity']),
      filter=_filters.Condition(
        lambda: (
          self._secret_input is None
          and not (
            self._state.activity.status == 'loading…'
            and self._worker_configuration.values.get('create_session', False)
          )
        )
      ),
    )
    extensions_below_activity = _containers.ConditionalContainer(
      component_host.mount(self._components['extensions_below_activity']),
      filter=_filters.Condition(
        lambda: bool(
          activity_window.preferred_height(
            self._state.layout.columns, 1
          ).preferred
        )
      ),
    )
    task_window = self._task_window = component_host.mount(
      self._components['tasks']
    )
    task_footer_spacer = _containers.Window(
      height=lambda: _dimension.Dimension.exact(
        1 if self._state.tasks.tasks and not self._extra_compact() else 0
      )
    )
    extensions_above_prompt = _containers.to_container(
      component_host.mount(self._components['extensions_above_prompt'])
    )
    extensions_below_prompt = _containers.to_container(
      component_host.mount(self._components['extensions_below_prompt'])
    )
    if self._fullscreen:
      # Widgets retain their placement, but cannot consume the composer budget.
      def bounded_widget(child: _containers.Container) -> _containers.Container:
        def height() -> _dimension.Dimension:
          size = self._app.output.get_size()
          limit = max(1, size.rows // 6)
          return _dimension.Dimension(
            preferred=min(
              child.preferred_height(size.columns, size.rows).preferred, limit
            ),
            max=limit,
          )

        return _containers.HSplit(
          [_scrollable_pane.ScrollablePane(child, show_scrollbar=False)],
          height=height,
        )

      extensions_above_prompt = bounded_widget(extensions_above_prompt)
      extensions_below_prompt = bounded_widget(extensions_below_prompt)
    activity_prefix = _containers.HSplit([
      pending_window,
      _containers.Window(
        height=lambda: _dimension.Dimension.exact(
          1
          if (
            pending_window.filter()
            and _pending.rows(self._state)
            and self._state.turn.pending_instructions
          )
          else 0
        )
      ),  # Blank row between running tool calls and queued messages.
      queued_window,
      _containers.Window(
        height=lambda: _dimension.Dimension.exact(
          1
          if (pending_window.filter() and _pending.rows(self._state))
          or self._state.turn.pending_instructions
          else 0
        )
      ),  # Blank row between activity panes and the input, when any.
      activity_window,
      extensions_below_activity,
      failure_window,
      _containers.Window(
        height=lambda: int(
          bool(
            activity_window.preferred_height(
              self._state.layout.columns, 1
            ).preferred
          )
          and bool(
            extensions_above_prompt.preferred_height(
              self._state.layout.columns, 1
            ).preferred
          )
        )
      ),
      extensions_above_prompt,
      composer_top_spacer,
      drop_confirm_window,
      resume_menu_window,
    ])
    activity_suffix = [
      secret_window,
      rewind_menu_window,
      rewind_banner_window,  # Closest to the composer it aims at an old turn.
    ]
    activity_region = _containers.HSplit([
      activity_prefix,
      selector_window,
      _containers.ConditionalContainer(
        _containers.HSplit(activity_suffix),
        filter=_filters.Condition(
          lambda: not self._fullscreen or not self._state.selector.active
        ),
      ),
    ])
    navigation_region = _containers.HSplit([
      task_window,
      task_footer_spacer,
      modal_footer_window,
      extensions_below_prompt,
    ])
    prompt_pane = _containers.HSplit([
      activity_region,
      composer_window,
      navigation_region,
    ])
    self._bottom_pane = _containers.HSplit([prompt_pane, worktree_window])
    banner_window = _containers.ConditionalContainer(
      _component.ComponentHost(lambda: self._state).mount(
        self._components['banner']
      ),
      filter=_filters.Condition(self._welcome_visible),
    )
    notices_window = self._welcome_notices_window = (
      _containers.ConditionalContainer(
        _containers.Window(
          _controls.FormattedTextControl(
            lambda: [
              *_ansi.ANSI(
                _rendering.render_blocks(
                  list(self._transcript.welcome_notices()),
                  self._component_state().layout.columns,
                  theme_name=_themes.current(self._state.extensions),
                )
                + '\n'
              ).__pt_formatted_text__(),
              ('[SetCursorPosition]', ''),
            ],
            show_cursor=False,
          ),
          height=lambda: _dimension.Dimension(
            max=max(1, self._app.output.get_size().rows // 3)
          ),
          dont_extend_height=True,
          wrap_lines=False,
        ),
        filter=_filters.Condition(
          lambda: (
            self._live_notices() and bool(self._transcript.welcome_notices())
          )
        ),
      )
    )
    pinned_header: _containers.Container = _containers.Window(height=0)
    root = _containers.HSplit(
      [banner_window, notices_window, streaming_window, self._bottom_pane],
      align=_containers.VerticalAlign.TOP,
    )
    if self._fullscreen:
      fixed = _containers.HSplit([
        composer_window,
        navigation_region,
        worktree_window,
      ])
      body = activity_region
      body_pane = _scrollable_pane.ScrollablePane(
        body,
        show_scrollbar=False,
        keep_cursor_visible=False,
        keep_focused_window_visible=False,
      )

      def body_height() -> int:
        size = self._app.output.get_size()
        return max(
          0,
          size.rows * 2 // 3
          - fixed.preferred_height(size.columns, size.rows).preferred,
        )

      def selector_max_height() -> int:
        size = self._app.output.get_size()
        limit = body_height()
        prefix = activity_prefix.preferred_height(
          size.columns, size.rows
        ).preferred
        # Prioritize the active selector; concurrent activity uses spare rows.
        minimum = (
          len(self._state.selector.filtered_choices()) + 4
          if self._state.approval.request is not None
          else 4
        )
        reserved = min(prefix, max(0, limit - minimum))
        return max(4, limit - reserved)

      self._selector_max_height = selector_max_height

      def activity_height() -> _dimension.Dimension:
        size = self._app.output.get_size()
        preferred = body.preferred_height(
          size.columns, body_pane.max_available_height
        ).preferred
        visible = min(preferred, body_height())
        body_pane.vertical_scroll = max(0, preferred - visible)
        return _dimension.Dimension.exact(visible)

      body_pane.height = activity_height
      prompt_pane = _containers.HSplit([
        body_pane,
        composer_window,
        navigation_region,
      ])
      self._bottom_pane = _containers.HSplit([prompt_pane, worktree_window])
      dock = _containers.HSplit(
        [self._bottom_pane],
        height=lambda: _dimension.Dimension.exact(self._dock_height()),
      )
      pinned_header = _containers.ConditionalContainer(
        _containers.Window(
          _controls.FormattedTextControl(
            lambda: _ansi.ANSI(
              _rendering.without_hyperlinks(
                _rendering.render_blocks(
                  self._terminal.header,
                  self._state.layout.columns,
                  theme_name=_themes.current(self._state.extensions),
                )
              )
            ),
            show_cursor=False,
          ),
          height=lambda: _dimension.Dimension(
            max=max(0, self._app.output.get_size().rows - self._dock_height())
          ),
          dont_extend_height=True,
          wrap_lines=False,
        ),
        filter=_filters.Condition(
          lambda: (
            bool(self._terminal.header)
            and not self._transcript.has_conversation
          )
        ),
      )

      def jump_to_latest(event: _mouse_events.MouseEvent) -> None:
        if (
          event.button == _mouse_events.MouseButton.LEFT
          and event.event_type == _mouse_events.MouseEventType.MOUSE_UP
        ):
          self.dispatch(_actions.NavigateHistory('latest'))

      root = _containers.HSplit([
        pinned_header,
        banner_window,
        _containers.HSplit([
          _containers.Window(
            self._viewport, wrap_lines=False, always_hide_cursor=True
          ),
          _containers.ConditionalContainer(
            _containers.Window(
              _controls.FormattedTextControl([
                ('', '↓ Back to bottom · Esc', jump_to_latest)
              ]),
              height=1,
              style='class:footer',
              align=_containers.WindowAlign.CENTER,
            ),
            filter=_filters.Condition(lambda: self._viewport.reading),
          ),
        ]),
        dock,
      ])
    self._welcome_header_window = pinned_header

    def presentation() -> _containers.Container:
      if not self._welcome_visible():
        return root
      output = _current.get_app().output
      rows = output.get_size().rows
      column = _banner.centered(
        _containers.HSplit([notices_window, streaming_window, prompt_pane]),
        self._state.layout.columns,
      )
      welcome = _containers.HSplit([pinned_header, banner_window, column])

      def height() -> _dimension.Dimension:
        natural = sum(
          child.preferred_height(self._state.layout.columns, rows).preferred
          for child in welcome.children
        )
        return _dimension.Dimension(min=0, preferred=natural, max=natural)

      welcome.height = height
      # Decorative Unicode must not degrade legacy/ASCII terminals.
      try:
        '⠂'.encode(output.encoding() or 'ascii')
      except UnicodeEncodeError:
        return _containers.HSplit([welcome, worktree_window])
      # Tabs span the terminal, not the narrower composer column.
      navigation_rows = worktree_window.preferred_height(
        self._state.layout.columns, rows
      ).preferred
      return _containers.HSplit([
        _banner.Background(welcome, max(0, rows - navigation_rows)),
        worktree_window,
      ])

    app = _workspace_application.WorkspaceApplication(
      layout=_layout.Layout(
        _containers.DynamicContainer(presentation),
        focused_element=input_window,
      ),
      key_bindings=bindings,
      style=_style.merge_styles([
        _style.DynamicStyle(
          lambda: _themes.prompt_style(_themes.current(self._state.extensions))
        ),
        _style.DynamicStyle(
          lambda: self._prompt_editor.style if self._prompt_editor else None
        ),
      ]),
      color_depth=_themes.prompt_color_depth(),
      cursor=_cursor_shapes.DynamicCursorShapeConfig(
        lambda: self._prompt_editor.cursor if self._prompt_editor else None
      ),
      full_screen=self._fullscreen,
      min_redraw_interval=1 / _MAX_REDRAW_FPS,
      max_render_postpone_time=0 if self._fullscreen else 0.01,
      refresh_interval=0.1,
    )

    def sync_before_render(app: _application.Application[str | None]) -> None:
      # Inactive workspaces do not receive the foreground app's resize events.
      columns = max(1, app.output.get_size().columns)
      if columns != self._state.layout.columns:
        self.dispatch(_actions.TerminalResized(columns))
      self._sync_prompt_editing()

    app.before_render += sync_before_render
    _extended_keys.install()
    # Without this prompt_toolkit takes 1s to disambiguate ESC
    app.ttimeoutlen = 0.05
    app.timeoutlen = 1.0
    relayout = app._on_resize  # pyright: ignore[reportPrivateUsage]

    def _on_resize() -> None:
      columns = max(1, app.output.get_size().columns)
      self.dispatch(_actions.TerminalResized(columns))
      app.create_background_task(
        self._settle_resize(self._state.session.resize_generation)
      )

    self._relayout_prompt = relayout
    app._on_resize = _on_resize  # pyright: ignore[reportPrivateUsage]
    if self._fullscreen:

      def resize_view() -> None:
        columns = max(1, app.output.get_size().columns)
        self.dispatch(
          _actions.TerminalResized(columns), _actions.HistoryRendered(columns)
        )

      app.resize_handler = resize_view
    return app

  def _dock_height(self) -> int:
    size = self._app.output.get_size()
    preferred = self._bottom_pane.preferred_height(
      size.columns, size.rows
    ).preferred
    return max(1, min(preferred, max(1, size.rows * 2 // 3)))

  def _live_blocks(self) -> tuple[_console.RenderableType, ...]:
    preview = self._stream_pane.preview
    stamp = (preview, self._state.status_bar.view_mode)
    if stamp == self._live_stamp:
      return self._live_blocks_cache
    self._live_stamp = stamp
    committed_stamp = (
      preview.committed_reasoning if preview else None,
      preview.committed_content if preview else None,
      self._state.status_bar.view_mode,
    )
    if committed_stamp != self._committed_stamp:
      self._committed_stamp = committed_stamp
      self._committed_blocks = tuple(self._stream_pane.committed_blocks())
    blocks = list(self._committed_blocks)
    if preview is not None:
      tail = _rendering.assistant_turn_renderable(
        None if self._extra_compact() else preview.reasoning,
        None,
        None,
        hyperlinks=False,
        skill_names=self._available_skill_names,
        compact_reasoning=self._state.status_bar.view_mode
        is _actions.ViewMode.COMPACT,
      )
      if tail is not None:
        blocks.append(tail)
      blocks.extend(self._streaming_answer.update(preview.content))
      refusal = _rendering.assistant_turn_renderable(
        None, None, preview.refusal, hyperlinks=False
      )
      if refusal is not None:
        blocks.append(refusal)
    else:
      self._streaming_answer.update('')
    self._live_blocks_cache = tuple(blocks)
    return self._live_blocks_cache

  async def _run_slash(self, text: str) -> None:
    """Parse slash text and dispatch its semantic action."""
    try:
      action = _commands.dispatch(
        text, self._command_context(), self.get_available_commands()
      )
      if action is None:
        return
      await asyncio.gather(*self.dispatch(action))
    except (
      _cli_store.SessionOpenError,
      RuntimeError,
      ValueError,
      OSError,
    ) as error:
      # OSError covers unreadable configuration files.
      await asyncio.gather(*self.dispatch(_actions.CommandFailed(str(error))))

  def _insert_newline(self) -> None:
    """Insert a newline only when the active input accepts multiple lines."""
    if (
      _completion.visible(self._state)
      or self._state.resume.active
      or self._state.selector.active
      or self._state.rewind.active
    ):
      return  # Keep completion entry single-line while the dropdown is open.
    self._app.current_buffer.newline()

  def _live_entry(
    self,
    entry: agent_pb2.AgentMessage,
    *,
    include_reasoning: bool = True,
    include_content: bool = True,
  ) -> _console.RenderableType | None:
    """Project an entry from the running turn, and let it move the panes."""
    hints = {
      call.call_id: (call.name, call.arguments_json, call.activity_kind)
      for call in self._state.stream.running
    }
    elapsed = self._observe_live(entry)
    rendered = self._transcript.project(
      entry,
      hints,
      elapsed=elapsed,
      include_reasoning=include_reasoning,
      include_content=include_content,
    )
    if isinstance(rendered, _tool_groups.ToolGroup):
      previous = self._state.tool_group
      if previous is not None and previous.key != rendered.key:
        self._flush_tool_group(mark_boundary=False)
        previous = None
      self.dispatch(
        _actions.SetToolGroup(
          rendered if previous is None else previous.merge(rendered)
        )
      )
      return None
    if rendered is not None:
      self._flush_tool_group(mark_boundary=False)
    return rendered

  def _flush_tool_group(self, *, mark_boundary: bool = True) -> None:
    """Move a completed tool group from the live pane to scrollback."""
    summary = self._state.tool_group
    if summary is None:
      return
    if mark_boundary:
      self._transcript.break_group()
    self.dispatch(_actions.SetToolGroup(None))
    self._terminal.queue_emit([summary])

  def _observe_live(self, entry: agent_pb2.AgentMessage) -> str | None:
    """Update live panes and return any completed compaction duration."""
    kind = entry.WhichOneof('payload')
    if kind == 'context_window' and entry.context_window.HasField('compaction'):
      started = self._state.turn.compaction_started
      self.dispatch(_actions.CompactionFinished())
      return (
        _rendering.Stopwatch(started).render() if started is not None else None
      )
    if kind != 'chat':
      return None
    message = entry.chat
    actions: list[_actions.Action] = [
      _actions.SetStatus(_rendering.next_status_label(message))
    ]
    role = message.WhichOneof('message')
    if role == 'user':
      content = message.user.content
      if content.strip():
        actions.append(_actions.UserMessageObserved(content, entry.id))
    elif role == 'assistant':
      if message.HasField('usage'):
        actions.append(
          _actions.RecordResponseUsage(message.usage.completion_tokens)
        )
      calls = tuple(
        _actions.RunningToolCall(
          call.id,
          call.function.name,
          _rendering.compact_args(
            call.function.arguments, tool_name=call.function.name
          ),
          _rendering.activity_kind(call.function.name),
          call.function.arguments,
        )
        for call in message.assistant.tool_calls
        if call.type == 'function'
      )
      if calls:
        actions.append(_actions.StartRunningToolCalls(calls))
    elif role == 'tool':
      if any(
        call.call_id != message.tool.tool_call_id
        for call in self._state.stream.running
      ):
        actions[0] = _actions.SetStatus('running tool…')
      actions.append(_actions.FinishRunningToolCall(message.tool.tool_call_id))
    self.dispatch(*actions)
    return None

  def _on_worker_message(self, response: agent_pb2.SessionMessage) -> None:
    if response.HasField('agent_message'):
      if self._transcript.contains(
        response.agent_message, replayed=self._state.session.replaying
      ):
        self._handle_message(response)
        return
      self._record_cli_message(response.agent_message)
    self._handle_message(response)

  def _on_worker_startup_message(
    self, response: agent_pb2.SessionMessage
  ) -> None:
    """Observe redacted startup metadata without treating it as a stored record."""
    if response.HasField('agent_message'):
      self._extension_host.handle_message(response)
    else:
      self._on_worker_message(response)

  def _record_cli_message(self, message: agent_pb2.AgentMessage) -> None:
    if self._remote_endpoint is not None:
      return  # PostgreSQL, not local CLI metadata, indexes remote sessions.
    timestamp = (
      message.timestamp.ToDatetime(tzinfo=datetime.UTC)
      if message.HasField('timestamp')
      else datetime.datetime.now(datetime.UTC)
    )
    with self._store() as store:
      if (
        message.WhichOneof('payload') == 'chat'
        and message.chat.WhichOneof('message') == 'user'
        and not message.chat.HasField('display')
        and message.chat.user.content.strip()
      ):
        store.record_user_message(
          self._state.session.conversation_id,
          entry_id=message.id,
          sequence=message.sequence,
          timestamp=timestamp,
          content=message.chat.user.content,
          cwd=str(self._workspace),
          model=self._state.status_bar.model,
        )

  def _handle_message(self, response: agent_pb2.SessionMessage) -> None:
    """Handle one AgentRPC message."""
    if response.HasField('agent_message') and self._transcript.contains(
      response.agent_message, replayed=self._state.session.replaying
    ):
      if response.stream_final:
        self._stream_pane.close()
      return
    self._conversation_metrics.observe(
      response,
      conversation_id=self._state.session.conversation_id,
      replaying=self._state.session.replaying,
    )
    self._extension_host.handle_message(response)
    payload = response.WhichOneof('payload')
    if payload is None:
      self._app.invalidate()
      return
    if payload == 'agent_message' and response.stream_final:
      preview = self._stream_pane.preview
      # The transcript keeps the whole turn, not only what remains to print.
      entry = response.agent_message
      self._append_entry(entry)
      if self._fullscreen:
        self._viewport.settle_live(entry.id)
      rendered_entry = agent_pb2.AgentMessage()
      rendered_entry.CopyFrom(entry)
      message = rendered_entry.chat.assistant
      reasoning_committed = (
        preview is not None
        and preview.committed_reasoning is not None
        and message.reasoning_content == preview.committed_reasoning
      )
      committed_content = (
        preview.committed_content if preview is not None else None
      )
      content_committed = False
      if (
        committed_content is not None
        and message.HasField('content')
        and message.content.startswith(committed_content)
      ):
        # Strip the already-committed prefix so only new content is rendered.
        message.content = message.content[len(committed_content) :]
        content_committed = True
      # The authoritative message below carries the calls' durable ids.
      self._stream_pane.handoff(
        self._live_entry(
          rendered_entry,
          include_reasoning=not reasoning_committed,
          include_content=not content_committed or bool(message.content),
        )
      )
      return
    if payload == 'agent_message':
      entry = response.agent_message
      if self._state.session.replaying:
        self._transcript.append_replayed(entry)
        return
      self._append_entry(entry)
      if (rendered := self._live_entry(entry)) is not None:
        self._terminal.queue_emit([rendered])
      if not self._state.session.replaying:
        self._app.invalidate()
      return
    if payload == 'hitl_request':
      self._flush_tool_group()
      call = response.hitl_request.call
      name = call.function.name
      scope = (
        self._local_approval_scope
        if self._remote_endpoint is None
        else json.dumps((
          'remote-v1',
          self._remote_endpoint,
          self._remote_secure,
          self._state.session.conversation_id,
        ))
      )
      remember: _tool_approvals.ApprovalRule | None = _tool_approvals.from_call(
        name, call.function.arguments, scope
      )
      if remember is not None:
        source = agent_pb2.APPROVAL_SOURCE_REMEMBERED_EXACT
        try:
          allowed = self._tool_approvals.contains(remember)
          if (
            not allowed
            and self._tool_approvals.available
            and self._remote_endpoint is None
            and name == 'bash'
          ):
            source = agent_pb2.APPROVAL_SOURCE_REMEMBERED_PREFIX
            prefixes = bash_approvals.prefixes(call.function.arguments, scope)
            allowed = any(
              self._tool_approvals.contains(rule) for rule in prefixes
            )
            if prefixes:
              remember = bash_approvals.suggested_rule(prefixes[0])
        except (OSError, ValueError, RecursionError):
          allowed = False
          self.dispatch(
            _actions.ShowText(
              'Remembered tool approvals are unavailable; asking for approval. '
              'Check ~/.mirror/tool_approvals.json and restart Mirror.'
            )
          )
        if allowed:
          self._run_effect(
            _effects.SendApprovalDecision(call.id, True, source=source)
          )
          return
      request = _actions.ApprovalRequest(
        call_id=call.id,
        name=name,
        args=_rendering.approval_args(call.function.arguments, tool_name=name),
        remember=remember if self._tool_approvals.available else None,
      )
      self.dispatch(_actions.ApprovalRequested(request))
      return
    if payload == 'hitl_resolved':
      self.dispatch(_actions.ClearApproval(response.hitl_resolved.call_id))
      return
    if payload == 'input_ack':
      self.dispatch(_actions.InputAcknowledged(response.input_ack.message_id))
      return
    if payload == 'input_nack':
      message_id = response.input_nack.message_id
      if message_id == self._state.turn.interrupt_id:
        self.dispatch(_actions.InterruptDeclined(message_id))
        return
      if self._transcript.transcript_index(message_id) is not None:
        return  # Committed history wins over a late rejection.
      if not any(
        submitted_id == message_id
        for submitted_id, _ in (
          *self._state.turn.submitted_inputs,
          *self._state.turn.awaiting_inputs,
        )
      ):
        self._record_decoration(
          _rendering.error_text(response.input_nack.error)
        )
        return
      self._stream_pane.clear_pending()
      self.dispatch(_actions.InputRejected(message_id))
      self._record_decoration(_rendering.error_text(response.input_nack.error))
      return
    if payload == 'replay_done':
      self._on_replay_done(response.replay_done)
      return
    if payload != 'runtime_delta':
      raise ValueError('session message has no payload')
    delta = response.runtime_delta
    kind = delta.WhichOneof('payload')
    if kind == 'stream_usage':
      return
    if kind in {
      'stream_content',
      'stream_reasoning',
      'stream_refusal',
      'stream_tool_call',
    }:
      visible_text = (
        (kind == 'stream_content' and bool(delta.stream_content.content))
        or (kind == 'stream_refusal' and bool(delta.stream_refusal.refusal))
        or (
          kind == 'stream_reasoning'
          and bool(delta.stream_reasoning.reasoning)
          and self._state.status_bar.view_mode is _actions.ViewMode.TRANSCRIPT
        )
      )
      if visible_text:
        self._flush_tool_group()
      self._stream_pane.on_partial(delta)
      return
    if kind == 'stream_aborted':
      self._abort_stream()
      status = 'thinking…' if self._state.activity.busy else ''
      self.dispatch(_actions.SetStatus(status))
      return
    if kind == 'rewound':
      with self._store() as store:
        store.rewind(
          self._state.session.conversation_id,
          delta.rewound.entry_id,
        )
      # Drop discarded turns so a re-render cannot restore them.
      entry_id = delta.rewound.entry_id
      cut = self._transcript.transcript_index(entry_id)
      if cut is not None:
        self._transcript.truncate(cut)
      else:
        self._transcript.clear_history()
      self._record_decoration(
        _text.Text(
          f'Discarded {delta.rewound.removed_entries} entries: output above the '
          'rewind marker is no longer part of this conversation.'
        )
      )
      return
    if kind == 'turn_started':
      self.dispatch(_actions.AgentTurnStarted())
      return
    if kind == 'compaction_started':
      self._flush_tool_group()
      self.dispatch(_actions.CompactionStarted(_rendering.Stopwatch().started))
      return
    if kind == 'task_update':
      update = delta.task_update
      statuses = {
        agent_pb2.TASK_STATUS_RUNNING: 'running',
        agent_pb2.TASK_STATUS_SUCCEEDED: 'succeeded',
        agent_pb2.TASK_STATUS_FAILED: 'failed',
        agent_pb2.TASK_STATUS_CANCELLED: 'cancelled',
      }
      status = statuses.get(update.status, 'failed')
      task_id = update.id
      task = None
      if update.detached and status == 'running':
        task = _actions.Task(
          id=task_id,
          tool_name=update.tool_name,
          display=update.display if update.HasField('display') else None,
        )
      foreground = not update.detached and status == 'running'
      self.dispatch(_actions.TaskChanged(task_id, task, foreground))
      return
    if kind == 'turn_error':
      self.dispatch(_actions.CompactionFinished())
      self._stream_pane.clear_pending()
      self._record_decoration(_rendering.error_text(delta.turn_error.message))
    elif kind == 'turn_interrupted':
      self._stream_pane.clear_pending()
      self._record_decoration(_rendering.interrupt_text())
    elif kind == 'turn_completed':
      self._flush_tool_group()
      leftovers = (
        []
        if self._extra_compact()
        else [
          _rendering.unanswered_call(
            call.render_hint,
            compact=self._state.status_bar.view_mode
            is _actions.ViewMode.COMPACT,
          )
          for call in self._state.stream.running
        ]
      )
      stream_action: _actions.Action = (
        _actions.ClearStreamPending()
        if self._stream_pane.preview is not None
        else _actions.CloseStream()
      )
      self.dispatch(
        _actions.TurnFinished(),
        _actions.ClearRunningToolCalls(),
        stream_action,
      )
      self._stream_pane.reset_height()
      if leftovers:
        self._terminal.queue_emit(leftovers)
      # A replay ending must not end an in-flight turn.
      # Queued lines clear as their user messages land in the transcript.
      if self._stream_pane.preview is not None:
        self._abort_stream()  # Fallback for a truncated stream lifecycle.
        return
    if not self._state.session.replaying:
      self._app.invalidate()  # Refresh the pending pane and footer.

  def _on_replay_done(self, result: agent_pb2.ReplayDone) -> None:
    """Print a replayed history, once it is the one a pending switch awaits."""
    switch = self._state.session.pending_switch
    if switch is not None and result.conversation_id != switch.conversation_id:
      self._transcript.clear_replayed()
      return  # The conversation being left; its replay stays off the screen.
    if self._fullscreen:
      self.dispatch(_actions.CloseStream())
    self.dispatch(
      _actions.HydrateExtensions(self._transcript.replayed()),
      _actions.ReplayFinished(result.has_user_message),
      _actions.ClearRunningToolCalls(),
    )
    self._stream_pane.reset_height()
    # The replay is the conversation now: it replaces what was on screen.
    self._transcript.adopt_replayed()
    if self._state.lifecycle.active:
      self._app.create_background_task(
        # Startup, session switches, and re-renders replace the visible screen.
        self._repaint(
          replacing=True, viewport_only=self._activation_preview_pending
        )
      )
    self._activation_preview_pending = False
    if not self._state.lifecycle.active:
      self._terminal.header = []
      self._terminal.emit_buffer.clear()

  async def _exit_with_error(self, message: str) -> None:
    """Record a worker failure and report it when its worktree is active."""
    self._worker_ready.clear()
    if self._state.turn.pending_instructions:
      message += (
        ' Unconfirmed messages restored to the composer; '
        'check conversation history before sending again.'
      )
    self.dispatch(_actions.WorkerFailed(message))
    error = _rendering.error_text(message)
    self._transcript.add_decoration(error)
    if not self._state.lifecycle.active:
      return
    # Replace rather than appending the failure under the previous worktree.
    self._terminal.trailer.extend(self._terminal.emit_buffer)
    self._terminal.emit_buffer.clear()
    await self._repaint(replacing=True)
    if self._exit_on_worker_failure:
      self.dispatch(_actions.ExitRequested())

  async def _repaint(
    self,
    *,
    replacing: bool,
    drop_scrollback: bool = False,
    viewport_only: bool = False,
  ) -> None:
    """Draw pinned rows and the history that fits the active clear mode."""
    was_switch = self._state.session.pending_switch is not None
    if self._fullscreen:
      self.dispatch(
        _actions.HistoryRendered(self._app.output.get_size().columns),
        _actions.RepaintFinished(was_switch),
      )
      return
    was_rerender = self._state.session.rerendering
    if drop_scrollback:
      clear = _terminal_mod.TerminalClear.SCROLLBACK
    elif replacing:
      clear = _terminal_mod.TerminalClear.SCREEN
    else:
      clear = _terminal_mod.TerminalClear.KEEP
    clear = self._terminal.effective_clear(clear)

    def paint() -> None:
      # Snapshot inside the terminal suspension, after any in-flight writes.
      if self._state.tool_group is not None:
        self._transcript.break_group()
        self.dispatch(_actions.SetToolGroup(None))
      if was_rerender:
        self._terminal.emit_buffer.clear()
      self._theme_repaint_pending = False
      ansi = self._render_history(
        trailer=[
          *self._terminal.trailer,
          *self._stream_pane.committed_blocks(),
        ],
        viewport_only=viewport_only,
      )
      self._terminal.header = []
      self._terminal.trailer = []
      if ansi or replacing:
        self._terminal.print_ansi(ansi, clear=clear)

    try:
      await _run_in_terminal(paint)
    finally:
      self.dispatch(_actions.RepaintFinished(was_switch))
      # Live output that arrived while the history was being collected.
      self._terminal.schedule_emit()

  def _render_history(
    self,
    *,
    header: Sequence[_console.RenderableType] | None = None,
    trailer: Sequence[_console.RenderableType] | None = None,
    viewport_only: bool = False,
  ) -> str:
    """The pinned rows, then the newest history the line budget allows."""
    self._sync_console_size()
    width = max(1, self._terminal.console.width)
    limit = self._scrollback_limit
    if viewport_only:
      rows = max(1, self._terminal.console.height)
      limit = min(limit, rows) if limit else rows
    ansi = self._transcript.render_history(
      list(self._terminal.header if header is None else header),
      list(self._terminal.trailer if trailer is None else trailer),
      console_width=width,
      scrollback_limit=limit,
      include_welcome_notices=not self._live_notices(),
      theme_name=_themes.current(self._state.extensions),
      cropped_history_hint=(
        'Ctrl+L to restore history up to --scrollback-limit'
        if viewport_only
        else None
      ),
    )
    self.dispatch(_actions.HistoryRendered(width))
    return ansi

  async def _retry_worker(self) -> None:
    """Retry a failed worker, replaying only when no history was loaded."""
    self.dispatch(_actions.WorkerRetryStarted())
    try:
      await self._shutdown_worker()
      replay = self._fullscreen or not self._state.session.has_user_message
      if replay:
        self.dispatch(_actions.SetReplaying(True))
      await self._start_worker(replay=replay)
    except Exception as error:  # noqa: BLE001 - remain retryable in the UI.
      await self._exit_with_error(
        f'could not start agent worker: {type(error).__name__}: {error}'
      )
    else:
      self.dispatch(_actions.WorkerRetryFinished())

  async def _commit_rewind(self, entry_id: str, instruction: str) -> None:
    """Fork append-only history and send the revision only to the child."""
    self.dispatch(_actions.TurnStarted(), _actions.SetStatus('rewinding…'))
    try:
      if self._remote_endpoint is not None:
        await self._shutdown_worker()
        forked = await _remote_rpc_client.fork_remote_session(
          self._remote_endpoint,
          self._state.session.conversation_id,
          entry_id,
          secure=self._remote_secure,
        )
      else:
        if self._client is None:
          raise RuntimeError('the agent worker is not running')
        forked = await self._client.fork_session(entry_id)
    except (grpc.aio.AioRpcError, RuntimeError) as error:
      self.dispatch(_actions.RestoreDraft(instruction))
      await self._exit_with_error(f'could not rewind: {error}')
      return
    self.dispatch(_actions.SessionCreated(forked))
    cut = self._transcript.transcript_index(entry_id)
    if cut is not None:
      self._transcript.truncate(cut)
    await self._shutdown_worker()
    self._worker_configuration = self._worker_configuration.updated({
      'conversation_id': forked,
      'create_session': False,
      'allow_workspace_mismatch': False,
    })
    self.dispatch(
      _actions.SessionSwitchStarted(
        _actions.PendingSwitch(forked, create=False)
      )
    )
    self._transcript.clear_replayed()
    self._on_session_opened(forked)
    try:
      await self._start_worker(replay=True)
    except Exception as error:  # noqa: BLE001 - retry the committed child, never the parent.
      self.dispatch(_actions.RestoreDraft(instruction))
      await self._exit_with_error(
        f'could not start rewound conversation: {error}'
      )
      return
    self.dispatch(_actions.TurnStarted())
    self._send_instruction(instruction)

  async def _run_turn(
    self,
    instruction: str,
    *,
    rewind_to: str | None = None,
    message_id: str | None = None,
  ) -> None:
    """Send one instruction to the worker; events render as messages arrive."""
    if rewind_to is not None:
      await self._commit_rewind(rewind_to, instruction)
      return
    if message_id is not None and not any(
      item[0] == message_id for item in self._state.turn.submitted_inputs
    ):
      return  # Cancelled before the effect ran; never send the restored draft.
    if not self._state.lifecycle.started:
      self.dispatch(_actions.SetReplaying(False))
      try:
        await self._shutdown_worker()
        await self._start_worker(replay=False)
      except Exception as error:  # noqa: BLE001 - keep other worktrees usable.
        self.dispatch(_actions.RestoreDraft(instruction))
        await self._exit_with_error(
          f'could not start agent worker: {type(error).__name__}: {error}'
        )
        return
      if message_id is not None:
        return  # Startup drained the tracked inputs, using their original IDs.
    self.dispatch(_actions.TurnStarted())
    self._send_instruction(instruction, message_id=message_id)

  def _log_user_prompt_submitted(self, instruction: str) -> None:
    """Report the size of a prompt the user sent, never its content."""
    try:
      utf8_byte_count = len(instruction.encode('utf-8'))
    except UnicodeEncodeError:
      return  # No UTF-8 length exists for an unpaired surrogate.
    _statsig_client.get().log_event(
      _statsig_event.Event(
        'mirror_user_prompt_submitted',
        metadata={
          'approval_mode': self.approval_mode,
          'character_count': len(instruction),
          'utf8_byte_count': utf8_byte_count,
          'workspace_kind': (
            'remote' if self._remote_endpoint is not None else 'local'
          ),
        },
      )
    )

  def _send_instruction(
    self, instruction: str, *, message_id: str | None = None
  ) -> None:
    """Append an instruction to the selected conversation."""
    request = self._new_request()
    if message_id is not None:
      request.message_id = message_id
    request.message.id = request.message_id
    request.message.chat.user.content = instruction
    request.message.chat.success = True
    if message_id is None:
      self.dispatch(_actions.InputSubmitted(request.message_id, instruction))
    try:
      if self._state.lifecycle.quitting:
        raise RuntimeError('Agent worker is closing; message not sent.')
      self._send_request(request)
    except RuntimeError as error:
      self.dispatch(_actions.InputRejected(request.message_id))
      self._record_decoration(_rendering.error_text(str(error)))

  def _enqueue(self, message: agent_pb2.AgentMessage) -> None:
    """Append one durable extension message through the active worker."""
    if self._client is not None and not self._state.lifecycle.quitting:
      self._client.enqueue(message)

  def _send_inference_config(
    self, configuration: _rpc_client.WorkerConfiguration | None = None
  ) -> None:
    """Apply the complete inference settings to subsequent model requests."""
    if self._state.lifecycle.quitting:
      return
    request = self._new_request()
    request.set_inference_config.CopyFrom(
      (configuration or self._worker_configuration).inference_config(
        self._state.status_bar.model,
        self._state.status_bar.reasoning_effort,
        remote=self._remote_endpoint is not None,
      )
    )
    self._send_request(request)

  def _new_request(self) -> agent_pb2.ClientMessage:
    """Create an addressed request with a fresh idempotency key."""
    return agent_pb2.ClientMessage(
      session_id=self._state.session.conversation_id,
      message_id=uuid.uuid4().hex,
    )

  def _send_request(self, request: agent_pb2.ClientMessage) -> None:
    """Queue one typed request through the active worker client."""
    request.session_id = self._state.session.conversation_id
    if self._client is not None and not self._state.lifecycle.quitting:
      self._client.send(request)

  def _on_session_opened(self, conversation_id: str) -> None:
    """Reset terminal state before replaying another conversation."""
    composer: _actions.ComposerState | None = None
    if (
      self._fullscreen
      and conversation_id != self._state.session.conversation_id
    ):
      pending = self._state.turn.pending_instructions
      draft = self._input.document
      if pending:
        text = '\n'.join((*pending, draft.text) if draft.text else pending)
        draft = _document.Document(text, len(text))
      self._drafts[self._state.session.conversation_id] = draft
      self._viewport.clear()
      self._stream_pane.close()
      self._transcript.clear_history()
      restored = self._drafts.get(conversation_id, _document.Document())
      composer = _actions.ComposerState(restored.text, restored.cursor_position)
    if self._remote_endpoint is not None:
      if self._workspace_effects is not None:
        self._workspace_effects.put_nowait((
          self,
          _effects.RetargetRemote(conversation_id),
        ))
    self._transcript.clear_decorations()
    self.dispatch(_actions.SessionOpened(conversation_id, composer))
    self._start_switch_screen()
    self._history.reload(conversation_id)
    # Resetting makes prompt_toolkit reload history; preserve the current draft.
    self._input.reset(self._input.document)

  def _transport(
    self,
    configuration: _rpc_client.WorkerConfiguration,
    on_message: _rpc_client.MessageHandler | None = None,
  ) -> _rpc_client.Transport:
    if self._remote_endpoint is not None:
      return _remote_rpc_client.Transport(
        self._remote_endpoint,
        secure=self._remote_secure,
      )
    log_dir = (
      pathlib.Path.home()
      / '.mirror'
      / 'logs'
      / str(configuration.values['conversation_id'])
    )
    return _local_rpc_client.Transport(
      self._worker_command,
      dict(configuration.values)
      | {
        'mcp_connections': [
          dataclasses.asdict(connection)
          for connection in self._extension_host.mcp_connections
        ],
        'additional_skills': [
          dataclasses.asdict(skill) | {'path': str(skill.path)}
          for skill in self._extension_host.skills
        ],
      },
      workspace=self._workspace,
      log_dir=log_dir,
      on_message=on_message or self._on_worker_startup_message,
    )

  async def _start_worker(
    self,
    *,
    replay: bool,
    configuration: _rpc_client.WorkerConfiguration | None = None,
  ) -> None:
    """Start a fresh worker behind the existing command/event interface."""
    conversation_id = self._state.session.conversation_id
    if self._extensions_loaded is not None:
      await asyncio.shield(self._extensions_loaded)
    launch_configuration = configuration or self._worker_configuration
    await self._extension_host.start()
    if conversation_id != self._state.session.conversation_id:
      return  # Session switching owns the replacement worker and its inputs.
    self._extension_host.validate_selection()
    self._host_tools_registry.validate_selection()
    self._host_tools_registry.worker_starting()
    client = _rpc_client.Client(
      self._transport(launch_configuration),
      conversation_id,
      on_message=self._on_worker_message,
      on_error=self._exit_with_error,
      host_tools_registry=self._host_tools_registry
      if self._remote_endpoint is None
      else None,
      inference_config=launch_configuration.inference_config(
        self._state.status_bar.model,
        self._state.status_bar.reasoning_effort,
        remote=self._remote_endpoint is not None,
      ),
    )
    self._client = client
    # Input entered during extension loading must retain its idempotency key.
    for message_id, instruction in self._state.turn.submitted_inputs:
      self._send_instruction(instruction, message_id=message_id)
    try:
      if self._remote_endpoint is not None and bool(
        launch_configuration.values.get('create_session', False)
      ):
        await _remote_rpc_client.start_remote_session(
          self._remote_endpoint,
          self._state.session.conversation_id,
          secure=self._remote_secure,
          workspace=launch_configuration.remote_workspace,
        )
        self.dispatch(
          _actions.SessionCreated(self._state.session.conversation_id)
        )
      if self._fullscreen:
        await client.start(replay=replay, wait_for_attachment=True)
      else:
        await client.start(replay=replay)
      if self._remote_endpoint is not None:
        request = self._new_request()
        request.set_approval_mode.mode = (
          agent_pb2.APPROVAL_MODE_AUTO
          if self._state.status_bar.approval_mode == 'auto'
          else agent_pb2.APPROVAL_MODE_HITL
        )
        client.send(request)
    except BaseException:
      if self._client is client:
        self._client = None
      raise
    self._worker_configuration = launch_configuration.after_start()
    if self._remote_endpoint is None and bool(
      launch_configuration.values.get('create_session', False)
    ):
      self.dispatch(
        _actions.SessionCreated(self._state.session.conversation_id)
      )
    self.dispatch(
      _actions.WorkerStarted(),
      _actions.SessionAttached(self._state.session.conversation_id),
    )
    self._worker_ready.set()
    if self._remote_endpoint is None:
      self._host_tools_registry.worker_started(
        lambda: self._restart_worker({}, replay=False)
      )

  async def _shutdown_worker(self) -> None:
    """Close and discard the active worker client."""
    await self._host_tools_registry.worker_stopped()
    self.dispatch(_actions.WorkerStopped())
    self._worker_ready.clear()
    client, self._client = self._client, None
    if client is not None:
      await client.close()

  async def _restart_worker(
    self,
    updates: Mapping[str, object],
    *,
    replay: bool,
    opening_conversation: str | None = None,
  ) -> bool:
    """Replace the worker, restoring the previous one on failure."""
    old_conversation = self._state.session.conversation_id
    old_configuration = self._worker_configuration
    launch_configuration = old_configuration.updated(updates)
    await self._shutdown_worker()
    if self._fullscreen:
      replay = True
      self._transcript.clear_replayed()
      self.dispatch(_actions.SetReplaying(True))
    if opening_conversation is not None:
      self._on_session_opened(opening_conversation)
    try:
      await self._start_worker(
        replay=replay,
        configuration=launch_configuration,
      )
      return True
    except Exception as error:  # noqa: BLE001 - recover the previous worker.
      if opening_conversation is not None:
        self.dispatch(
          _actions.SetPendingSwitch(
            _actions.PendingSwitch(old_conversation, create=False)
          ),
          _actions.SetReplaying(True),
        )
        self._on_session_opened(old_conversation)
      if self._fullscreen:
        self._transcript.clear_replayed()
      try:
        await self._start_worker(
          replay=self._fullscreen or opening_conversation is not None,
          configuration=old_configuration,
        )
      except Exception as recovery_error:  # noqa: BLE001 - no worker remains.
        await self._exit_with_error(
          'could not restore agent worker: '
          f'{type(recovery_error).__name__}: {recovery_error}'
        )
        return False
      replaying = (
        False if opening_conversation is None else self._state.session.replaying
      )
      self.dispatch(_actions.WorkerRecoveryFinished(replaying))
      self._record_decoration(
        _rendering.error_text(
          f'could not start agent worker: {type(error).__name__}: {error}'
        )
      )
      return False

  async def ensure_api_key(self) -> bool:
    """Authenticate the current model, allowing selection of another provider."""
    selected = await self._authenticate_model(
      self.model_selection, initial=True
    )
    if selected is None:
      return False
    if selected != self.model_selection:
      self._apply_model(selected)
    return True

  async def _authenticate_model(
    self, option: _models.ModelOption, *, initial: bool
  ) -> _models.ModelOption | None:
    """Resolve sign-in or model selection before committing a configuration."""
    if self._secret_input is not None:
      return None
    while True:
      api = option.api or self.model_selection.api
      if api == 'reflection':
        needs_key = not _reflection.has_key()
      elif api == 'anthropic-messages':
        needs_key = not _api_keys.has_anthropic_key() and not (
          option.base_url
          or os.environ.get('ANTHROPIC_BASE_URL')
          or (initial and self._worker_configuration.values.get('base_url'))
        )
      else:
        needs_key = False
      if not needs_key:
        return option
      provider = 'Anthropic' if api == 'anthropic-messages' else 'Reflection'
      if self._state.activity.busy and not initial:
        self.dispatch(
          _actions.ShowText(
            f'Finish or interrupt the current turn before signing in to {provider}.',
            style='ui.warning',
          )
        )
        return None
      result = await self._prompt_api_key(exit_on_cancel=initial, api=api)
      if result is _reflection.SignInResult.CONNECTED:
        return option
      if result is _reflection.SignInResult.CANCELLED:
        return None
      selected = await self._choose_onboarding_model()
      if selected is not None:
        option = selected
      elif not initial:
        return None

  async def _choose_onboarding_model(self) -> _models.ModelOption | None:
    """Await the shared model picker without starting an unauthenticated worker."""
    self._onboarding_model = asyncio.get_running_loop().create_future()
    try:
      self.dispatch(
        _actions.OpenSelector(_inference_controls.model_selector(self.model))
      )
      return await self._onboarding_model
    finally:
      self._onboarding_model = None
      self.dispatch(_actions.ClearSelector())

  async def _prompt_api_key(
    self, *, exit_on_cancel: bool, api: str = 'reflection'
  ) -> _reflection.SignInResult:
    """Collect credentials in an isolated buffer, never in chat or selector state."""
    result = _reflection.SignInResult.CANCELLED
    ready = asyncio.Event()
    buffer = self._secret_input = _buffer.Buffer(multiline=False)
    bindings = _key_binding.KeyBindings()

    @bindings.add('enter')
    def accept(_: object) -> None:
      nonlocal result
      try:
        if api == 'anthropic-messages':
          _api_keys.save_key(_api_keys.anthropic_key_path(), buffer.text)
        else:
          _reflection.save_key(buffer.text)
      except (ValueError, OSError) as error:
        self.dispatch(_actions.ShowText(str(error), style='ui.error'))
      else:
        result = _reflection.SignInResult.CONNECTED
        buffer.reset()
        ready.set()

    @bindings.add('escape', eager=True)
    def choose_model(_: object) -> None:
      nonlocal result
      result = _reflection.SignInResult.CHOOSE_MODEL
      buffer.reset()
      ready.set()

    @bindings.add('c-c')
    @bindings.add('c-d')
    def cancel(_: object) -> None:
      buffer.reset()
      if exit_on_cancel:
        self._app.exit()
      ready.set()

    previous = self._input_window.content
    self._input_window.content = _controls.BufferControl(
      buffer=buffer,
      input_processors=[
        _processors.PasswordProcessor(),
        _processors.BeforeInput([('class:ui.accent bold', ' API key › ')]),
      ],
      key_bindings=bindings,
    )
    self._app.layout.focus(self._input_window)
    self._app.invalidate()
    try:
      welcome = _rendering.render_banner(
        self._state.status_bar.model,
        self._session_directory(),
        setup_url=_api_keys.ANTHROPIC_KEY_URL
        if api == 'anthropic-messages'
        else _reflection.KEY_URL,
        setup_provider='Anthropic'
        if api == 'anthropic-messages'
        else 'Reflection',
      )
      self._terminal.pin_header(welcome)
      await self._repaint(replacing=True)
      await ready.wait()
    finally:
      buffer.reset()
      self._secret_input = None
      self._input_window.content = previous
      self._terminal.pin_header()
      self._app.invalidate()
    await self._repaint(replacing=True)

    return result

  async def start(self) -> None:
    """Start this worktree's agent while its application remains responsive."""
    await self._extension_host.start()
    if self._state.lifecycle.active:
      self.dispatch(_actions.CliStarted())
    self.dispatch(_actions.WorkerLoadingStarted())
    try:
      await self._start_worker(replay=True)
    except Exception as error:  # noqa: BLE001 - report startup in the TUI.
      await self._exit_with_error(
        f'could not start agent worker: {type(error).__name__}: {error}'
      )
    else:
      self.dispatch(_actions.WorkerLoadingFinished())

  def _clear_visible_screen(self) -> None:
    """Erase the visible screen, leaving a later paint to draw history."""
    self._terminal.print_ansi(
      '',
      clear=self._terminal.effective_clear(_terminal_mod.TerminalClear.SCREEN),
    )

  def _paint_activation(
    self, *, show_context: bool, label: str | None = None
  ) -> None:
    """Replace the transcript before the application's initial prompt render."""
    if self._fullscreen:
      self._app.invalidate()
      return
    if show_context:
      self._terminal.header = []
      self._terminal.emit_buffer.clear()
      divider = _rule.Rule(
        f'{(label or self._workspace.name)} · {self._session_directory()}',
        style='ui.info',
      )
      header: list[_console.RenderableType] = [divider]
      clear = self._terminal.effective_clear(_terminal_mod.TerminalClear.SCREEN)
      if self._state.session.replaying and (
        self._state.lifecycle.started or self._state.activity.busy
      ):
        self._activation_preview_pending = True
        # Clear now so a failed replay cannot leave the previous worktree.
        self._terminal.pin_header(*header)
        self._clear_visible_screen()
      else:
        if self._state.tool_group is not None:
          self._transcript.break_group()
          self.dispatch(_actions.SetToolGroup(None))
        ansi = self._render_history(
          header=header,
          trailer=self._stream_pane.committed_blocks(),
          viewport_only=True,
        )
        self._terminal.print_ansi(ansi, clear=clear)
    else:
      # The live welcome card belongs to the prompt, not terminal history.
      self._terminal.print_ansi(
        self._render_history(header=[]),
        clear=_terminal_mod.TerminalClear.SCREEN,
      )

  async def activate(
    self,
    *,
    show_context: bool,
    label: str | None = None,
    navigation: _containers.AnyContainer | None = None,
    before_render: Callable[[_application.Application[str | None]], None]
    | None = None,
    screen: _workspace_application.WorkspaceApplication | None = None,
  ) -> str | None:
    """Run this view with navigation supplied by the CLI composition root."""
    self._navigation = (
      navigation if navigation is not None else _containers.Window(height=0)
    )
    if before_render is not None:
      self._app.before_render += before_render
    await self._extension_host.start()
    await self._stream_pane.wait_for_emits()
    view = self._app
    if screen is not None:
      self._app = screen
    self._app.erase_when_done = False
    self.dispatch(_actions.SetActive(True))
    try:
      # Accept raw input before the first visible frame, not just run_async's redraw.
      with self._app.input.raw_mode():
        self._app.prepare_for_run()
        self._paint_activation(show_context=show_context, label=label)

        async def foreground() -> None:
          await self._extension_host.foreground()
          await self._user_extensions.on_foreground(self._extension_context)

        if screen is not None:
          return await screen.show_view(
            view,
            before_input=foreground,
            accept_buffered_input=lambda: (
              self._secret_input is None and not self._modal_ui_active()
            ),
          )
        await foreground()
        return await self._app.run_async()
    finally:
      self._app.finish_handoff()
      self._app = view
      self._navigation = _containers.Window(height=0)
      if before_render is not None:
        self._app.before_render -= before_render
      self._activation_preview_pending = False
      await self._user_extensions.on_background(self._extension_context)
      await self._extension_host.background()
      self.dispatch(_actions.SetActive(False))

  async def close(self) -> None:
    """Close this agent's worker connection and extensions."""
    self.dispatch(_actions.SetQuitting())
    async with contextlib.AsyncExitStack() as cleanup:
      cleanup.push_async_callback(self._extension_host.close)
      await self._shutdown_worker()

  def print_resume_hint(self) -> None:
    """Print how to continue after shutdown."""
    with self._terminal.console.use_theme(
      _themes.rich_theme(_themes.current(self._state.extensions))
    ):
      self._terminal.console.print(
        'Bye. Continue with mirror --continue', style='ui.muted'
      )

  async def run(self) -> None:
    """Run one worktree without the multi-worktree supervisor."""
    try:
      self._clear_visible_screen()
      await self.start()
      await self.activate(show_context=False)
    finally:
      try:
        await self.close()
      finally:
        self.print_resume_hint()


async def run_workspaces(
  controller: _workspace_controller.WorkspaceController,
  activate: Callable[[_workspaces.Workspace, bool], Awaitable[str | None]],
  queue: asyncio.Queue[tuple[_Repl, _effects.Effect]],
) -> None:
  """Wire workspace lifecycle and view results without a back-reference."""
  effects_task = asyncio.create_task(
    _consume_workspace_effects(controller, queue)
  )
  show_context = False
  try:
    await controller.initialize()
    while True:
      action = await activate(controller.selected, show_context)
      if action is None:
        break
      if action == 'drop':
        await controller.drop()
      elif action in {'next', 'previous'}:
        await controller.cycle(1 if action == 'next' else -1)
      elif action.startswith('select:'):
        await controller.select(int(action.removeprefix('select:')))
      show_context = True
  finally:
    effects_task.cancel()
    await asyncio.gather(effects_task, return_exceptions=True)
    await controller.close()


async def _activate_pending(
  workspace: _workspaces.PendingWorkspace,
  navigation: _containers.AnyContainer,
  screen: _workspace_application.WorkspaceApplication | None = None,
  theme_name: Callable[[], str] = lambda: _themes.effective('dark'),
) -> str | None:
  """Render pending content; the model contains no prompt-toolkit state."""
  bindings = _workspace_application.terminal_bindings()

  def handoff(result: str) -> Callable[[_key_binding.KeyPressEvent], None]:
    return lambda event: (screen or app).exit_for_handoff(result)

  bindings.add('tab')(handoff('next'))
  bindings.add('s-tab')(handoff('previous'))

  def quit_view(_: _key_binding.KeyPressEvent) -> None:
    (screen or app).exit()

  bindings.add('c-c')(quit_view)
  bindings.add('c-d')(quit_view)
  bindings.add(
    'enter', filter=_filters.Condition(lambda: workspace.error is not None)
  )(handoff('drop'))
  for index in range(9):
    bindings.add(f'c-{index + 1}')(handoff(f'select:{index}'))
  _extended_keys.install()

  def content() -> _formatted_text.StyleAndTextTuples:
    if workspace.error is not None:
      return [
        (
          'class:worktree-failed',
          f'{workspace.label}: {workspace.error}\n{workspace.notice or "Enter to dismiss · Ctrl+C to quit"}',
        )
      ]
    label = 'worktree' if workspace.label == 'creating…' else workspace.label
    return [
      *_breathing_dots.render(style='class:worktree-running'),
      (
        'class:worktree-running',
        f' Creating {label}…\n{workspace.notice or "Ctrl+C to quit"}',
      ),
    ]

  def replaced(_: _application.Application[str | None]) -> None:
    if workspace.finished and not app.is_done:
      (screen or app).exit_for_handoff('replaced')

  app = _workspace_application.WorkspaceApplication(
    layout=_layout.Layout(
      _containers.HSplit([
        _containers.Window(
          _controls.FormattedTextControl(content), dont_extend_height=True
        ),
        navigation,
      ])
    ),
    key_bindings=bindings,
    style=_style.DynamicStyle(lambda: _themes.prompt_style(theme_name())),
    color_depth=_themes.prompt_color_depth(),
    refresh_interval=0.1,
    before_render=replaced,
  )
  with app.input.raw_mode():
    if screen is None:
      app.prepare_for_run()
      app.output.erase_screen()
      app.output.cursor_goto(0, 0)
    try:
      if screen is not None:
        return await screen.show_view(app, accept_buffered_input=lambda: False)
      return await app.run_async()
    finally:
      app.finish_handoff()


async def _consume_workspace_effects(
  controller: _workspace_controller.WorkspaceController,
  queue: asyncio.Queue[tuple[_Repl, _effects.Effect]],
) -> None:
  while True:
    agent, effect = await queue.get()
    try:
      selected = controller.selected
      if not isinstance(
        effect,
        (
          _effects.SaveWorktreePreferences,
          _effects.SaveLastModel,
          _effects.RetargetRemote,
        ),
      ) and (
        not isinstance(selected, _workspaces.AgentWorkspace)
        or selected.agent is not agent
      ):
        continue
      await handle_workspace_effect(controller, agent, effect)
    except Exception as error:  # Isolate failed commands.
      agent.show_error(str(error))
    finally:
      queue.task_done()


def _workspace_tag_command(
  controller: _workspace_controller.WorkspaceController, args: str
) -> _actions.Action | None:
  """Route tag commands to silent mutations or visual selectors."""
  parts = args.split(maxsplit=3)
  selected = controller.selected
  if not isinstance(selected, _workspaces.AgentWorkspace):
    raise ValueError('workspace is not ready for tagging')
  tags = controller.list_workspace_tags()
  match parts:
    case ['menu'] | ['list']:
      return _workspace_tags.menu(tags, controller.workspace_tag(selected.id))
    case ['manage']:
      return _workspace_tags.manage(tags)
    case ['new']:
      return _workspace_tags.icons()
    case ['new', icon] | ['new-color', icon]:
      _workspace_tags.WorkspaceTag('preview', 'Preview', icon=icon)
      return _workspace_tags.colors(icon)
    case ['create', color, icon] | ['create-color', icon, color]:
      tag = _workspace_tags.WorkspaceTag(
        uuid.uuid4().hex,
        _workspace_tags.icon_name(icon),
        None if color == '-' else color,
        icon,
      )
      controller.save_workspace_tag(tag)
      controller.set_workspace_tag(selected.id, tag.id)
      return None
    case ['cycle']:
      controller.cycle_workspace_tag()
      return None
    case ['clear'] | ['assign', _]:
      controller.set_workspace_tag(
        selected.id, parts[1] if parts[0] == 'assign' else None
      )
      return None
    case ['define', tag_id, color, name]:
      existing = next((tag for tag in tags if tag.id == tag_id), None)
      controller.save_workspace_tag(
        _workspace_tags.WorkspaceTag(
          tag_id,
          name,
          None if color == '-' else color,
          existing.icon if existing else '■',
        )
      )
      return _workspace_tags.menu(
        controller.list_workspace_tags(), controller.workspace_tag(selected.id)
      )
    case ['delete', tag_id]:
      controller.delete_workspace_tag(tag_id)
      return _workspace_tags.manage(controller.list_workspace_tags())
    case [operation, tag_id, *value]:
      tag = next((tag for tag in tags if tag.id == tag_id), None)
      if tag is None:
        raise ValueError(f'unknown workspace tag {tag_id!r}')
      return _edit_workspace_tag(controller, tag, operation, ' '.join(value))
    case _:
      raise ValueError('Use /tag to choose or customize a workspace icon.')


def _edit_workspace_tag(
  controller: _workspace_controller.WorkspaceController,
  tag: _workspace_tags.WorkspaceTag,
  operation: str,
  value: str,
) -> _actions.Action:
  match operation:
    case 'edit':
      return _workspace_tags.edit(tag)
    case 'icons':
      return _workspace_tags.icons(tag.id)
    case 'colors':
      return _workspace_tags.colors(tag.icon, tag)
    case 'symbol':
      tag = dataclasses.replace(tag, icon=value)
    case 'color':
      tag = dataclasses.replace(tag, color=None if value == '-' else value)
    case 'rename':
      tag = dataclasses.replace(tag, name=value)
    case _:
      raise ValueError('Use /tag to choose or customize a workspace icon.')
  controller.save_workspace_tag(tag)
  return _workspace_tags.edit(tag)


async def handle_workspace_effect(
  controller: _workspace_controller.WorkspaceController,
  agent: _Repl,
  effect: _effects.Effect,
) -> None:
  """Apply an emitted workspace command at the composition root."""
  match effect:
    case _effects.OpenFolder(path):
      source = controller.selected
      if not isinstance(source, _workspaces.AgentWorkspace):
        return
      if source.remote:
        raise ValueError('Open folders from a local workspace.')
      directory = (agent.workspace / pathlib.Path(path).expanduser()).resolve(
        strict=True
      )
      if not directory.is_dir():
        raise ValueError(f'not a directory: {directory}')
      try:
        await _run_in_terminal(
          functools.partial(_onboarding.ensure_consent, directory),
          in_executor=True,
        )
      except SystemExit:
        return
      workspace = await controller.open_folder(source.id, directory)
      await controller.select_workspace(workspace.id)
      agent.handoff('folder')
    case _effects.CloseWorkspace():
      selected = controller.selected
      if isinstance(selected, _workspaces.AgentWorkspace):
        if await controller.close_workspace(selected.id):
          agent.handoff('closed')
    case _effects.CreateManagedWorktree(name):
      await controller.create(name)
    case _effects.DropManagedWorktree(confirmed):
      confirmation = None if confirmed else controller.drop_confirmation()
      if confirmation is not None:
        agent.dispatch(_actions.WorktreeDropConfirmationRequested(confirmation))
      elif await controller.drop(confirmed=confirmed):
        agent.handoff('dropped')
    case _effects.WorkspaceTagCommand(args):
      if (action := _workspace_tag_command(controller, args)) is not None:
        agent.dispatch(action)
    case _effects.RenameWorktree(alias):
      normalized = controller.rename(alias)
      agent.dispatch(
        _actions.ShowText(f'Worktree alias set to {normalized!r}.')
      )
    case _effects.LoadRemoteSessions():
      controller.load_remote_sessions()
    case _effects.CreateRemoteAgent(title, workspace):
      index = await controller.create_remote(title, workspace)
      if index is not None and controller.can_select(index):
        agent.handoff(f'select:{index}')
    case _effects.RefreshWorkspaces():
      controller.refresh()
      agent.dispatch(
        _actions.OpenSelector(
          _worktree_tabs.workspace_selector(
            controller.workspaces,
            controller.selected.identity,
            agent.selector,
          )
        )
      )
    case _effects.CycleWorktree(offset):
      controller.refresh()
      if controller.can_cycle():
        agent.handoff('next' if offset > 0 else 'previous')
    case _effects.SelectWorktree(index):
      if controller.can_select(index):
        agent.handoff(f'select:{index}')
    case _effects.SelectWorkspace(identity):
      if identity == controller.selected.identity:
        agent.dispatch(_actions.CloseSelector())
      elif await controller.select_workspace(identity):
        agent.dispatch(_actions.CloseSelector())
        agent.handoff('selected')
    case _effects.SaveWorktreePreferences(reasoning_effort, approval_mode):
      controller.save_preferences(reasoning_effort, approval_mode, agent=agent)
    case _effects.SaveLastModel():
      controller.save_model(agent, agent.model_selection)
    case _effects.RetargetRemote(session_id):
      controller.retarget_remote(agent, session_id)
    case _:
      raise ValueError(f'not a workspace effect: {effect!r}')


async def _activate_workspace(
  controller: _workspace_controller.WorkspaceController,
  workspace: _workspaces.Workspace,
  show_context: bool,
  screen: _workspace_application.WorkspaceApplication | None = None,
  *,
  theme_name: Callable[[], str] = lambda: _themes.effective('dark'),
) -> str | None:
  navigation = _containers.Window(
    _controls.FormattedTextControl(
      lambda: _worktree_tabs.render_tabs(
        controller.workspaces,
        workspace.identity,
        tags=controller.tab_tags(),
        width=_current.get_app().output.get_size().columns,
      )
    ),
    height=lambda: int(
      len(controller.visible_workspaces()) > 1 or bool(controller.tab_tags())
    ),
    wrap_lines=False,
  )
  if isinstance(workspace, _workspaces.PendingWorkspace):
    return await _activate_pending(
      workspace, navigation, screen, theme_name=theme_name
    )
  if not isinstance(workspace, _workspaces.AgentWorkspace):
    raise TypeError('unknown workspace implementation')
  agent = workspace.agent
  if not isinstance(agent, _Repl):
    raise TypeError('workspace has no terminal view')
  await workspace.start()

  def refresh_picker(_: _application.Application[str | None]) -> None:
    if (
      agent.selector.active
      and agent.selector.selector is not None
      and agent.selector.selector.id == 'workspaces'
    ):
      request = _worktree_tabs.workspace_selector(
        controller.workspaces, workspace.identity, agent.selector
      )
      if request != agent.selector.selector:
        agent.dispatch(_actions.OpenSelector(request))

  activate = functools.partial(
    agent.activate,
    show_context=show_context,
    label=workspace.label,
    navigation=navigation,
    before_render=refresh_picker,
  )
  if screen is not None:
    return await activate(screen=screen)
  return await activate()


async def apply_analytics_config(
  updated: dict[str, object],
  workspaces: _workspace_controller.WorkspaceController,
) -> None:
  """Apply consent and synchronize the current workspace and running counts."""
  enabled = (
    _config.analytics_enabled(updated)
    and updated.get('analytics_notice_shown') is True
  )
  if enabled:
    await _statsig_client.get().enable()
  else:
    await _statsig_client.get().disable()
  workspaces.report_workspace_count()
  workspaces.running_sessions_changed()


async def run_tui(
  worker_command: Sequence[str],
  worker_config: Mapping[str, object],
  *,
  reasoning_effort: str = 'max',
  database: str | pathlib.Path | None = None,
  scrollback_limit: int = 1000,
  flush_scrollback: bool = False,
  continue_recent: bool = False,
  remote_endpoint: str | None = None,
  remote_secure: bool = True,
  trust_extensions: bool = False,
) -> None:
  """Run independently warm agents across all local and remote workspaces."""
  session_id = uuid.uuid4().hex
  conversation_id = cast(str, worker_config['conversation_id'])
  with _logging.capture_output(conversation_id) as terminal:
    extension_registry = _extension.ExtensionRegistry()
    settings = _config.Config(pathlib.Path.home() / '.mirror' / 'config.json')
    analytics_directory = pathlib.Path.home() / '.mirror' / 'analytics'
    # Without a valid install ID, this launch sends no analytics.
    analytics_config: _statsig_config.StatsigConfig | None = None
    initial_settings: dict[str, object] = {}
    try:
      initial_settings = settings.read()
      install_id = initial_settings.get('install_id')
      if isinstance(install_id, str) and install_id:
        analytics_config = _statsig_config.StatsigConfig(
          user_id=f'install-{install_id}',
          session_id=session_id,
          unsent_events_directory=analytics_directory,
        )
    except (OSError, ValueError):
      pass
    selected_theme = _themes.saved(initial_settings)
    _statsig_client.initialize(analytics_config)
    tool_approvals = _tool_approvals.ToolApprovals()
    checkpoints_enabled = _config.checkpoints_enabled(settings.read())
    user_extensions = _user_extensions.UserExtensions(
      extension_registry,
      pathlib.Path.home() / '.mirror' / 'extensions',
      cwd=pathlib.Path.cwd(),
      bundled=pathlib.Path(__file__).with_name('extensions'),
      settings=settings,
    )
    first_render = asyncio.Event()
    initial_agent: _Repl | None = None

    async def load_extensions() -> None:
      await first_render.wait()
      if initial_agent is not None and not await initial_agent.ensure_api_key():
        return
      if initial_agent is not None:
        notice = _onboarding.take_analytics_notice()
        if notice is not None:
          initial_agent.dispatch(_actions.ShowText(notice))
      await user_extensions.start(trust_extensions=trust_extensions)

    extensions_loaded = asyncio.create_task(load_extensions())

    workspace_effects: asyncio.Queue[tuple[_Repl, _effects.Effect]] = (
      asyncio.Queue()
    )

    def create_agent(
      spec: _workspaces.AgentSpec,
      controller: _workspace_controller.WorkspaceController,
    ) -> _Repl:
      nonlocal initial_agent
      launch_args = dict(spec.config)
      launch_args['checkpoints_enabled'] = checkpoints_enabled
      agent = _Repl(
        worker_command,
        launch_args,
        session_id=session_id,
        workspace=spec.workspace,
        workspace_id=spec.workspace_id,
        console=_console.Console(file=terminal),
        reasoning_effort=spec.reasoning_effort,
        workspace_extensions=spec.extensions,
        workspace_tools=spec.tools,
        database=database,
        scrollback_limit=scrollback_limit,
        flush_scrollback=flush_scrollback,
        workspace_effects=workspace_effects,
        workspace_controller=controller,
        extension_registry=extension_registry,
        user_extensions=user_extensions,
        tool_approvals=tool_approvals,
        active=spec.active,
        exit_on_worker_failure=False,
        remote_endpoint=remote_endpoint if spec.remote else None,
        configured_remote_endpoint=remote_endpoint,
        remote_secure=remote_secure,
        first_render=first_render,
        extensions_loaded=extensions_loaded,
        fullscreen=True,
        initial_theme=selected_theme,
      )
      if initial_agent is None:
        initial_agent = agent
      return agent

    workspaces = _workspace_controller.WorkspaceController(
      worker_config,
      workspace=pathlib.Path.cwd(),
      reasoning_effort=reasoning_effort,
      create_agent=create_agent,
      database=database,
      settings=settings,
      continue_recent=continue_recent,
      display_unmanaged_worktrees=(
        initial_settings.get('display_unmanaged_worktrees') is True
      ),
      remote_enabled=remote_endpoint is not None,
      list_remote_sessions=(
        functools.partial(
          _remote_rpc_client.list_remote_sessions,
          remote_endpoint,
          secure=remote_secure,
        )
        if remote_endpoint is not None
        else None
      ),
      delete_remote_session=(
        functools.partial(
          _remote_rpc_client.delete_remote_session,
          remote_endpoint,
          secure=remote_secure,
        )
        if remote_endpoint is not None
        else None
      ),
    )

    async def apply_config(updated: dict[str, object]) -> None:
      nonlocal selected_theme
      selected_theme = _themes.saved(updated)
      await apply_analytics_config(updated, workspaces)
      workspaces.set_display_unmanaged_worktrees(
        updated.get('display_unmanaged_worktrees') is True
      )
      await asyncio.shield(extensions_loaded)
      await user_extensions.apply_settings(updated.get('extensions'))

    config_task = asyncio.create_task(settings.watch(apply_config))

    screen = _workspace_application.WorkspaceApplication(
      full_screen=True,
      min_redraw_interval=1 / _MAX_REDRAW_FPS,
      max_render_postpone_time=0,
      refresh_interval=0.1,
      mouse_support=True,
    )
    run_selected = functools.partial(
      run_workspaces,
      workspaces,
      functools.partial(
        _activate_workspace,
        workspaces,
        screen=screen,
        theme_name=lambda: selected_theme,
      ),
      workspace_effects,
    )
    try:
      await screen.run_views(run_selected)
    finally:
      config_task.cancel()
      extensions_loaded.cancel()
      await asyncio.gather(
        config_task, extensions_loaded, return_exceptions=True
      )
      await _statsig_client.get().disable()
      try:
        await extension_registry.close()
      finally:
        _console.Console(
          file=terminal,
          theme=_themes.rich_theme(selected_theme),
        ).print('Bye. Continue with mirror --continue', style='ui.muted')
