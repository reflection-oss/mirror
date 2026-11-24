"""The interactive Rich + prompt_toolkit REPL."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import subprocess
import tempfile
import time
import uuid
from collections.abc import (
  AsyncGenerator,
  Awaitable,
  Callable,
  Coroutine,
  Iterable,
  Mapping,
  Sequence,
)
from typing import Any, Literal, cast, override

import grpc.aio
from mirror_transcript import transcript_pb2
from mirror.protocol import (
  agent_pb2,
  daemon_pb2,
  local_models_pb2,
  mirror_pb2,
)
from prompt_toolkit import buffer as _buffer
from prompt_toolkit import cursor_shapes as _cursor_shapes
from prompt_toolkit import data_structures as _data_structures
from prompt_toolkit import document as _document
from prompt_toolkit import enums as _enums
from prompt_toolkit import filters as _filters
from prompt_toolkit import formatted_text as _formatted_text
from prompt_toolkit import history as _history
from prompt_toolkit import key_binding as _key_binding
from prompt_toolkit import keys as _keys
from prompt_toolkit import mouse_events as _mouse_events
from prompt_toolkit import styles as _style
from prompt_toolkit.application import application as _application
from prompt_toolkit.application import current as _current

# prompt_toolkit exposes only the shadowing `run_in_terminal` function.
from prompt_toolkit.application import run_in_terminal as _run_in_terminal
from prompt_toolkit.key_binding import key_bindings as _key_bindings
from prompt_toolkit.layout import containers as _containers
from prompt_toolkit.layout import controls as _controls
from prompt_toolkit.layout import dimension as _dimension
from prompt_toolkit.layout import layout as _layout
from prompt_toolkit.layout import processors as _processors
from rich import console as _console
from rich import rule as _rule
from rich import text as _text

from mirror import config as _config
from mirror import skills as _skills
from mirror import workspace as _workspace
from mirror.cli import approval_modes
from mirror.cli import actions as _actions
from mirror.cli import agents as _agents
from mirror.cli import api_keys as _api_keys
from mirror.cli import bash_approvals, row_renderer
from mirror.cli import commands as _commands
from mirror.cli import conversation_metrics as _conversation_metrics
from mirror.cli import daemon_client as _daemon_client
from mirror.cli import effects as _effects
from mirror.cli import extended_keys as _extended_keys
from mirror.cli import extension as _extension
from mirror.cli import frontend as _frontend
from mirror.cli import host_tools as _host_tools
from mirror.cli import hyperlinks as _hyperlinks
from mirror.cli import images as _images
from mirror.cli import inference_choices as _inference_choices
from mirror.cli import keymap as _keymap
from mirror.cli import local_models_client as _local_models_client
from mirror.cli import local_rpc_client as _local_rpc_client
from mirror.cli import logging as _logging
from mirror.cli import models as _models
from mirror.cli import onboarding as _onboarding
from mirror.cli import pending_view as _pending_view
from mirror.cli import reducer as _reducer
from mirror.cli import reflection as _reflection
from mirror.cli import remote_rpc_client as _remote_rpc_client
from mirror.cli import rendering as _rendering
from mirror.cli import retained as _retained
from mirror.cli import mcp_configuration
from mirror.cli import rpc_client as _rpc_client
from mirror.cli import screen_selection as _screen_selection
from mirror.cli import quotes as _quotes
from mirror.cli import selection as _selection
from mirror.cli import shell as _shell
from mirror.cli import state as _state
from mirror.cli import stream as _stream_mod
from mirror.cli import terminal as _terminal_mod
from mirror.cli import themes as _themes
from mirror.cli import tool_approvals as _tool_approvals
from mirror.cli import tool_groups as _tool_groups
from mirror.cli import transcript as _transcript_mod
from mirror.cli import native_transcript as _native_transcript
from mirror.cli import user_extensions as _user_extensions
from mirror.cli import workspace_application as _workspace_application
from mirror.cli import workspace_browser as _workspace_browser
from mirror.cli import workspace_controller as _workspace_controller
from mirror.cli import workspace_tags as _workspace_tags
from mirror.cli import workspaces as _workspaces
from mirror.cli import worktrees as _worktrees
from mirror.cli.components import completion as _completion
from mirror.cli.components import component as _component
from mirror.cli.components import layout as _component_layout
from mirror.cli.components import pending as _pending
from mirror.cli.components import secret_prompt as _secret_prompt
from mirror.cli.components import selector as _selector
from mirror.cli.components import worktree_tabs as _worktree_tabs
from mirror.cli.composer import editing as _editing
from mirror.cli.composer import frame as _composer_frame
from mirror.cli.composer import pane as _composer_pane
from mirror.cli.composer import processors as _composer_processors
from mirror.cli.composer import skill_completion as _skill_completion
from mirror.cli.composer import surface as _composer_surface
from mirror.cli.extensions.inference_controls import (
  extension as _inference_controls,
)
from mirror.cli.sandbox import view as _sandbox_view
from mirror.cli.statsig import config as _statsig_config
from mirror.cli.statsig.events import client as _statsig_client
from mirror.cli.statsig.events import event as _statsig_event

# This limit also gates composer redraws, not just streaming token updates.
_MAX_REDRAW_FPS = 120
# Activity animations advance on a periodic redraw while a turn runs. Each
# redraw renders the whole layout, so this rate bounds CPU use between events.
_ACTIVITY_ANIMATION_FPS = 30
# Skip prompt_toolkit's per-SIGWINCH erase until the drag settles.
_RESIZE_SETTLE_SECONDS = 0.2


class _PromptHistory(_history.History):
  """Canonical prompt history fetched through the daemon."""

  def __init__(
    self,
    metadata: _daemon_client.Client,
    conversation_id: str,
    recall: Callable[[Sequence[agent_pb2.UserMessage]], list[str]],
  ) -> None:
    super().__init__()
    self._metadata = metadata
    self._conversation_id = conversation_id
    # Turns prompts into composer text, registering their quotes.
    self._recall = recall

  @override
  def load_history_strings(self) -> Iterable[str]:
    return self._recall(self._metadata.prompt_history(self._conversation_id))

  async def prompts(self) -> list[str]:
    """Fetch prompts off the event loop and recall them on it."""
    messages = await asyncio.to_thread(
      self._metadata.prompt_history, self._conversation_id
    )
    return self._recall(messages)

  @override
  async def load(self) -> AsyncGenerator[str, None]:
    loaded_strings = self._loaded_strings
    if not self._loaded:
      strings = await self.prompts()
      if loaded_strings is not self._loaded_strings:
        return
      if not self._loaded:
        # Keep drafts appended while loading; concurrent readers install once.
        self._loaded_strings.extend(strings)
        self._loaded = True
    for text in self._loaded_strings:
      yield text

  def reload(self, conversation_id: str) -> None:
    """Reset the cache so the next load uses the new conversation_id."""
    self._conversation_id = conversation_id
    self._loaded = False
    self._loaded_strings = []

  @override
  def store_string(self, string: str) -> None:
    del string  # Only canonical inputs are durable; drafts remain client-local.


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
    speed: str = 'normal',
    metadata: _daemon_client.Client,
    scrollback_limit: int = 1000,
    flush_scrollback: bool = False,
    workspace_effects: asyncio.Queue[tuple[_Repl, _effects.Effect]]
    | None = None,
    extension_registry: _extension.ExtensionRegistry,
    workspace_controller: _workspace_controller.WorkspaceController,
    user_extensions: _user_extensions.UserExtensions | None = None,
    tool_approvals: _tool_approvals.ToolApprovals | None = None,
    images: _images.Attachments | None = None,
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
    initial_view_mode: _actions.ViewMode = _actions.ViewMode.DEFAULT,
    on_view_mode_changed: Callable[[_actions.ViewMode], None] | None = None,
    read_only: bool = False,
    frontend: _frontend.Context | None = None,
    local_runtime: _local_models_client.Client | None = None,
  ) -> None:
    self._frontend = frontend
    self._fullscreen = fullscreen
    self._on_view_mode_changed = on_view_mode_changed
    self._local_runtime = local_runtime
    self._theme_repaint_pending = False
    self._tool_approvals = tool_approvals or _tool_approvals.ToolApprovals()
    self._images = images or _images.Attachments()
    self._worker_command = tuple(worker_command)
    self._session_id = session_id
    self._extensions_loaded = extensions_loaded
    self._workspace_controller = workspace_controller
    self._secret_input: _buffer.Buffer | None = None
    self._model_switch_generation = 0
    self._reasoning_choices = 0
    self._onboarding_model: (
      asyncio.Future[_models.ModelOption | None] | None
    ) = None
    self._host_tools_registry = _host_tools.HostToolRegistry(
      selected=workspace_tools
    )
    self._daemon_host = str(worker_config.get('daemon_host') or '')
    self._workspace = workspace or pathlib.Path.cwd()
    if not self._daemon_host:
      self._workspace = self._workspace.resolve()
    self._worker_configuration = _rpc_client.WorkerConfiguration(
      worker_config, cwd=self._workspace
    )
    self._remote_endpoint = remote_endpoint
    self._conversation_metrics = _conversation_metrics.ConversationMetrics(
      remote=remote_endpoint is not None
    )
    # Version the built-in execution contract, including /bin/bash -c.
    self._local_approval_scope = json.dumps((
      f'ssh:{self._daemon_host}' if self._daemon_host else 'local-v1',
      str(self._workspace),
      os.environ.get('MIRROR_WORKSPACE'),
    ))
    self._remote_secure = remote_secure
    self._workspace_effects = workspace_effects
    self._navigation: _containers.AnyContainer = _containers.Window(height=0)
    self._navigation_refresh_key: Callable[[], object | None] = lambda: ()
    self._exit_on_worker_failure = exit_on_worker_failure
    self._user_extensions = user_extensions or _user_extensions.UserExtensions(
      extension_registry,
      pathlib.Path.home() / '.mirror' / 'extensions',
      cwd=self._workspace,
      bundled=pathlib.Path(__file__).with_name('extensions'),
      bundled_only=bool(self._daemon_host),
    )
    self._workspace_shell = (
      None
      if read_only
      else _daemon_client.Shell(
        self._workspace, connection=self._workspace_connection
      )
    )
    self._extension_context = _extension.Context(
      workspace=_extension.Workspace(
        cwd=self._workspace,
        remote=remote_endpoint is not None or bool(self._daemon_host),
        id=workspace_id,
        shell=self._workspace_shell,
      ),
      dispatch=lambda action: self.dispatch(action),
      get_state=lambda: self._state,
      enqueue=self._enqueue,
      refresh_tools=extension_registry.refresh_tools,
      set_mcp_connections=self._set_mcp_connections,
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
        if remote_endpoint is None and not self._daemon_host
        else None
      ),
      worker_command=() if self._daemon_host else self._worker_command,
      worker_cwd=self._workspace
      if not self._daemon_host and not worker_config.get('sandboxed')
      else None,
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
    inference = self._worker_configuration.inference_config(
      cast(str, worker_config['model']),
      reasoning_effort,
      remote=remote_endpoint is not None,
    )
    if speed != 'fast' or not _models.supports_fast_mode(inference.api):
      speed = 'normal'
    self._state: _state.State = _state.State(
      activity=_state.ActivityState(),
      status_bar=_state.StatusBarState(
        model=cast(str, worker_config['model']),
        reasoning_effort=reasoning_effort,
        speed=speed,
        approval_mode=cast(str, worker_config.get('approval_mode', 'approve')),
        view_mode=initial_view_mode,
      ),
      session=_state.SessionState(
        read_only=read_only,
        conversation_id=conversation_id,
        remote=remote_endpoint is not None,
        replaying=not read_only,
        restore_pending=not read_only
        and not bool(worker_config.get('create_session', True)),
      ),
      lifecycle=_state.LifecycleState(active=active),
      workspace=str(self._workspace),
      supervised=workspace_effects is not None,
      extensions=extension_states,
      extension_themes=self._extension_host.themes,
    )
    self._metadata = metadata
    self._scrollback_limit = scrollback_limit
    self._activation_preview_pending = False
    self._client: _rpc_client.Client | None = None
    self.restarting = False
    self._inspection: _Repl | None = None
    self._inspection_parent: _Repl | None = None
    self._inspection_status = 'running'
    self._children: dict[str, _Repl] = {}
    self._worker_ready = asyncio.Event()
    keymap_error: _actions.ShowText | None = None
    try:
      remappings = () if read_only else _keymap.load_config()
    except (ValueError, OSError) as error:
      remappings = ()
      keymap_error = _actions.ShowText(
        f'{error}. Using default keybindings.', style='ui.error'
      )
    self._keymap = _keymap.Keymap(
      self._keybinding_layers,
      self.dispatch,
      dismiss=self._dismissal_action,
      on_escape_interrupt=self._conversation_metrics.interrupt_requested,
      remappings=remappings,
      enabled=_filters.Condition(lambda: self._secret_input is None),
    )
    self._project_skills = (
      ()
      if read_only
      or self._daemon_host
      or worker_config.get('sandbox') is not None
      else tuple(_skills.load_skills(self._workspace))
    )
    self._extension_skills = self._extension_host.skills
    self._skill_mention_processor = _skill_completion.SkillMentionProcessor(())
    self._update_available_skills()
    self._components = _component.ComponentCollection(
      _composer_pane.components(
        self._extension_host,
        (lambda: self._composer_pane.selector_max_height())
        if fullscreen
        else None,
        skip_first_queued=self._initial_input_pending,
        shortcut_hint=self._keymap.shortcut_hint,
      )
    )
    self._selector_search = _selector.SearchInput(
      lambda query: self.dispatch(_actions.SelectorSearchChanged(query)),
      lambda: _selector.search_hint(self._state),
    )
    self._selector_keys = _controls.FormattedTextControl('', focusable=True)
    self._extension_secret_input = _secret_prompt.SecretInput(
      lambda length: self.dispatch(_actions.SecretLengthChanged(length))
    )
    self._history = _PromptHistory(
      metadata, self._state.session.conversation_id, self._recall_prompts
    )

    def create_terminal_task(
      coroutine: Coroutine[Any, Any, None],
    ) -> asyncio.Task[None]:
      return self._app.create_background_task(coroutine)

    self._terminal = _terminal_mod.TerminalOutput(
      console or _console.Console(theme=_themes.rich_theme()),
      theme_name=lambda: _themes.current(
        self._state.extensions, self._state.extension_themes
      ),
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
      tuple[
        _state.StreamPreview | None,
        _actions.ViewMode,
        tuple[_actions.PendingToolCall, ...],
        tuple[_actions.RunningToolCall, ...],
        _tool_groups.ToolGroup | None,
        str | None,
        str,
        bool,
      ]
      | None
    ) = None
    self._live_blocks_cache: tuple[_console.RenderableType, ...] = ()
    self._streaming_answer = _rendering.StreamingAnswer(
      self._available_skill_names
    )
    self._committed_stamp: (
      tuple[str | None, str | None, _actions.ViewMode] | None
    ) = None
    self._committed_blocks: tuple[_console.RenderableType, ...] = ()
    self._committed_source_offset = 0
    self._viewport = _native_transcript.View(
      self._viewport_blocks,
      self._live_blocks,
      navigate=self.dispatch,
      pending=self._initial_input,
      theme_name=lambda: _themes.current(
        self._state.extensions, self._state.extension_themes
      ),
    )
    self._drafts: dict[str, _document.Document] = {}
    self._editing_prompt = False
    self._last_composer: _actions.ComposerState | None = None
    self._input = _buffer.Buffer(
      name='composer',
      history=self._history,
      multiline=True,
      on_text_changed=self._on_composer_changed,
      on_cursor_position_changed=self._on_composer_changed,
      read_only=_filters.Condition(
        lambda: self._state.session.read_only or self._modal_ui_active()
      ),
    )
    self._prompt_editor: _editing.Editor | None = None
    self._app = self._build_app()
    self._view: _workspace_application.WorkspaceApplication = self._app
    if first_render is not None:

      def on_render(_: _application.Application[str | None]) -> None:
        first_render.set()

      self._app.after_render += on_render
    if keymap_error is not None:
      # Workspace views never run pre_run_callables, so show the error when the
      # view first draws, then stop listening. Use the app passed in: by then
      # self._app may already be the shared screen.
      def show_keymap_error(app: _application.Application[str | None]) -> None:
        app.after_render -= show_keymap_error
        self.dispatch(keymap_error)

      self._app.after_render += show_keymap_error
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
    themes = (
      self._inspection_parent._state.extension_themes
      if self._inspection_parent is not None
      else self._extension_host.themes
    )
    if not actions and self._state.extension_themes == themes:
      return ()
    if any(
      isinstance(action, (_actions.SetViewMode, _actions.CycleViewMode))
      for action in actions
    ):
      self._flush_tool_group()
      self._stream_pane.reset_height()
    was_running = self._running
    transcript_only = bool(actions) and all(
      isinstance(action, _actions.NavigateHistory) for action in actions
    )
    next_state = self._state
    effects: list[_effects.Effect] = []
    for action in actions:
      if isinstance(action, _actions.SetReasoning):
        # A model switch still resolving keeps this choice over its own.
        self._reasoning_choices += 1
      if isinstance(
        action, (_actions.WorkerFailed, _actions.WorkerLoadingStarted)
      ):
        self._conversation_metrics.suspend()
      extension_states, extension_effects = self._extension_host.reduce(
        next_state, action
      )
      next_state, action_effects = _reducer.reduce(
        next_state, action, observed_at=time.monotonic()
      )
      if extension_states is not next_state.extensions:
        transcript_only = False
        next_state = dataclasses.replace(
          next_state, extensions=extension_states
        )
      transcript_only &= not extension_effects
      effects.extend((*action_effects, *extension_effects))
    if next_state.extension_themes != themes:
      transcript_only = False
      next_state = dataclasses.replace(next_state, extension_themes=themes)
    self._theme_repaint_pending |= _themes.current(
      self._state.extensions, self._state.extension_themes
    ) != _themes.current(next_state.extensions, next_state.extension_themes)
    previous_state = self._state
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
      visible = self._inspection or self
      self._app.refresh_interval = (
        1 / _ACTIVITY_ANIMATION_FPS if visible._state.activity.busy else 0.1
      )
      if (
        self._fullscreen
        and next_state.lifecycle.active
        and next_state.activity == previous_state.activity
        and next_state.extensions is previous_state.extensions
        and not self._modal_ui_active()
        and any(
          isinstance(action, _actions.ObserveStreamText) for action in actions
        )
        and all(
          isinstance(action, (_actions.ObserveStreamText, _actions.SetStatus))
          for action in actions
        )
      ):
        with _current.set_app(self._app):
          if not self._viewport.refresh():
            self._app.invalidate()
      else:
        self._app.invalidate(transcript_only=transcript_only)
    tasks: list[asyncio.Task[None]] = []
    for effect in effects:
      result = self._run_effect(effect)
      if isinstance(result, tuple):
        tasks.extend(result)
      elif result is not None:
        tasks.append(result)
    if hasattr(self, '_app'):
      self._sync_prompt_editing()
    mode = next_state.status_bar.view_mode
    if mode is not previous_state.status_bar.view_mode:
      if self._on_view_mode_changed is not None:
        try:
          self._on_view_mode_changed(mode)
        except (OSError, ValueError) as error:
          tasks.extend(
            self.dispatch(
              _actions.ShowText(
                f'Could not save view mode: {error}', style='ui.error'
              )
            )
          )
      for child in self._children.values():
        child.dispatch(_actions.SetViewMode(mode))
    return tuple(tasks)

  def _run_effect(
    self, effect: _effects.Effect
  ) -> asyncio.Task[None] | tuple[asyncio.Task[None], ...] | None:
    """Execute an imperative effect emitted by the pure reducer."""
    if self._state.session.read_only and not isinstance(
      effect,
      (
        _effects.NavigateHistory,
        _effects.CloseInspection,
        _effects.RerenderHistory,
      ),
    ):
      return None
    if self._daemon_host and _effects.requires_local_configuration(effect):
      if self._client is not None and self._client.shared_settings is not None:
        self._receive_shared_settings(self._client.shared_settings)
      return self.dispatch(
        _actions.ShowText(
          'This command configures the CLI host. Configure it on the daemon host instead.'
        )
      )
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
      case _effects.QuoteSelection():
        return self.dispatch(_actions.QuoteAdded(self._app.selected_text()))
      case _effects.InsertComposerText(text):
        self._input.save_to_undo_stack()
        self._input.insert_text(text)
      case _effects.EditComposerExternally():
        return self._app.create_background_task(self._edit_composer())
      case _effects.UndoComposer():
        self._app.current_buffer.undo()
      case _effects.RedoComposer():
        self._app.current_buffer.redo()
      case _effects.PasteClipboardImage():
        return self._app.create_background_task(self._paste_clipboard_image())
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
        if (
          result is None
          and not defer_until_running
          and not handoff
          and (prompt := self._extension_host.quit_prompt()) is not None
        ):
          self.dispatch(prompt)
          return
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
        _effects.CreateWorkspace()
        | _effects.OpenFolder()
        | _effects.OpenSessionWorkspace()
        | _effects.CreateManagedWorktree()
        | _effects.ForkWorkspace()
        | _effects.CreateSandboxWorkspace()
        | _effects.OpenNewWorkspace()
        | _effects.CloseWorkspace()
        | _effects.RestartDaemon()
        | _effects.DropWorkspace()
        | _effects.RenameWorktree()
        | _effects.WorkspaceTagCommand()
        | _effects.LoadRemoteSessions()
        | _effects.RefreshWorkspaces()
        | _effects.OpenWorkspaceBrowser()
        | _effects.CycleWorktree()
        | _effects.SelectWorktree()
        | _effects.SelectWorkspace()
        | _effects.SaveWorktreePreferences()
        | _effects.RetargetSession()
      ):
        if self._workspace_effects is not None:
          self._workspace_effects.put_nowait((self, effect))
      case _effects.CreateRemoteAgent():
        return self._app.create_background_task(self._prepare_remote(effect))
      case _effects.SetWebProvider(provider):
        self._worker_configuration = self._worker_configuration.updated({
          'web_provider': provider
        })
        if self._client is not None:
          self._client.set_web_provider(provider)
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
        if not self._daemon_host:
          _config.save_approval_mode(mode)
        self._worker_configuration = self._worker_configuration.updated({
          'approval_mode': mode
        })
        request = self._new_request()
        request.set_approval_mode.mode = approval_modes.to_protocol(mode)
        self._send_request(request)
      case _effects.SendApprovalDecision(
        call_id, approved, remember, source, reason
      ):
        if approved and remember is not None and not self._daemon_host:
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
        if reason is not None:
          request.hitl_response.reason = reason
        self._send_request(request)
      case _effects.RunFrontendCommand(name, arguments):
        if self._frontend is not None:
          return self._frontend.start(name, arguments, dispatch=self.dispatch)
      case _effects.RunSlashCommand(text):
        return self._app.create_background_task(self._run_slash(text))
      case _effects.RunShellCommand(command):
        return self._app.create_background_task(
          self._run_shell_command(command)
        )
      case _effects.SendQueuedInstruction(instruction):
        self._send_instruction(instruction)
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
      case _effects.InspectTask(task):
        self._inspect(task)
      case _effects.CloseInspection():
        if self._inspection_parent is not None:
          return self._inspection_parent._run_effect(effect)
        child, self._inspection = self._inspection, None
        if child is not None:
          child._app = child._view
          self._clear_screen_selection()
          self._app.layout.focus(
            self._task_window
            if self._state.tasks.selected_id
            else self._input_window
          )
          child.dispatch(_actions.SetActive(False))
          if child._session_id not in self._children:
            return self._app.create_background_task(child.close())
      case _effects.FocusComposer():
        if self._inspection is None:
          self._app.layout.focus(self._input_window)
      case _effects.FocusTasks():
        if self._inspection is None:
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
        if self._daemon_host:
          return None
        selection = self.model_selection
        if self._workspace_effects is not None:
          self._workspace_effects.put_nowait((self, effect))
        if self._remote_endpoint is None and not self._daemon_host:
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
        if self._client is None or not self._worker_ready.is_set():
          self.dispatch(_actions.UnsentInputsDiscarded())
          return None
        request = self._new_request()
        request.interrupt.SetInParent()
        self.dispatch(_actions.InterruptSent(request.message_id))
        self._send_request(request)
      case _effects.RunExtensionEffect():
        return self._extension_host.handle_effect(effect)
    return None

  def _create_child(self, task_id: str) -> _Repl:
    child = _Repl(
      (),
      {
        'conversation_id': task_id,
        'model': self.model,
        'create_session': False,
      },
      session_id=task_id,
      workspace=self._workspace,
      console=self.console,
      metadata=self._metadata,
      extension_registry=_extension.ExtensionRegistry(),
      user_extensions=self._user_extensions,
      workspace_controller=self._workspace_controller,
      exit_on_worker_failure=False,
      fullscreen=True,
      on_view_mode_changed=self._on_view_mode_changed,
      read_only=True,
      active=False,
    )
    child._inspection_parent = self
    child._state = dataclasses.replace(
      child._state,
      status_bar=self._state.status_bar,
      extensions=self._state.extensions,
      extension_themes=self._state.extension_themes,
    )
    child.dispatch(_actions.TurnStarted())
    return child

  def _inspect(self, task: _actions.Task) -> None:
    child = self._children.get(task.id)
    if child is None or self._inspection is not None:
      return
    self._inspection = child
    child._app = self._app
    child.dispatch(
      _actions.SetActive(True),
      _actions.SetViewMode(self._state.status_bar.view_mode),
    )
    self._clear_screen_selection()
    self._app.layout.focus(child._input_window)

  def _clear_screen_selection(self) -> None:
    surface = self._app.layout.container
    if isinstance(surface, _screen_selection.ScreenSelection):
      surface.clear()

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
    app = _current.get_app()
    if isinstance(app, _workspace_application.WorkspaceApplication):
      app.finish_scroll_burst()
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

  async def _paste_clipboard_image(self) -> None:
    """Attach the clipboard's image to the draft, or say why it cannot be."""
    if not _models.supports_images(self.model_selection):
      self.dispatch(
        _actions.ShowText(_images.unsupported(self.model), style='ui.warning')
      )
      return
    try:
      data = await _images.read_clipboard()
    except _images.ClipboardError as error:
      self.dispatch(_actions.ShowText(str(error), style='ui.warning'))
      return
    self._attach_image(data)

  def _attach_pasted_image(self, text: str) -> bool:
    """Attach the image file that pasted text names; False leaves it text."""
    path = _images.pasted_path(text)
    if (
      path is None
      or self._input.read_only()
      or not _models.supports_images(self.model_selection)
    ):
      return False
    data = _images.read_file(path)
    if data is None or not _images.is_image(data):
      return False
    if problem := _images.too_large(data):
      self.dispatch(_actions.ShowText(problem, style='ui.warning'))
      return False
    self._attach_image(data)
    return True

  def _attach_image(self, data: bytes) -> None:
    """Put an image's label at the composer cursor, spaced from its neighbors."""
    label = self._images.add(data)
    document = self._input.document
    before, after = document.text_before_cursor, document.text_after_cursor
    lead = ' ' if before and not before[-1].isspace() else ''
    trail = '' if after[:1].isspace() else ' '
    inserted = f'{before}{lead}{label}{trail}'
    self._input.save_to_undo_stack()
    # A clipboard read can finish after a modal took the composer.
    self._input.set_document(
      _document.Document(inserted + after, len(inserted)),
      bypass_readonly=True,
    )

  def _composer_bindings(self) -> _key_bindings.KeyBindings:
    """Editing keys for image labels, below the remappable keymap."""
    bindings = _key_bindings.KeyBindings()
    focused = _filters.has_focus(self._input)

    def label_before_cursor() -> str | None:
      return self._images.label_before(self._input.document.text_before_cursor)

    @bindings.add(_keys.Keys.BracketedPaste, filter=focused)
    def paste(event: _key_binding.KeyPressEvent) -> None:
      # prompt_toolkit's own paste normalizes line endings the same way.
      text = event.data.replace('\r\n', '\n').replace('\r', '\n')
      if not self._attach_pasted_image(text):
        self._input.insert_text(text)

    @bindings.add(
      'backspace',
      filter=focused
      & ~_filters.vi_navigation_mode
      & ~_filters.has_selection
      & _filters.Condition(lambda: label_before_cursor() is not None),
    )
    def delete_label(event: _key_binding.KeyPressEvent) -> None:
      del event  # Unused.
      self._input.delete_before_cursor(len(label_before_cursor() or ''))

    return bindings

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

  async def create_agent(
    self,
    options: _agents.Options,
    on_message: _rpc_client.MessageHandler,
    on_error: _rpc_client.ErrorHandler,
  ) -> _rpc_client.Client:
    """Prepare an independent local agent using this workspace's settings."""
    if (
      self._remote_endpoint is not None
      or self._daemon_host
      or self._worker_configuration.values.get('sandboxed')
    ):
      raise NotImplementedError(
        'Extension agent creation requires a local directory workspace'
      )
    if self._client is None or not self._client.execution_host:
      raise RuntimeError('Only the execution host may create extension agents')
    record = await _local_rpc_client.create_workspace(
      self._worker_command, self._worker_configuration.values, self._workspace
    )
    configuration = self._worker_configuration.updated({
      'model': self._state.status_bar.model,
      'approval_mode': self._state.status_bar.approval_mode,
      'conversation_id': record.session_id,
      'workspace_id': record.id,
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
      on_settings_error=self.show_error,
      host_tools_registry=self._host_tools_registry,
      web_provider=configuration.web_provider,
      inference_config=configuration.inference_config(
        self._state.status_bar.model,
        self._state.status_bar.reasoning_effort,
        remote=False,
        speed=self._state.status_bar.speed,
      ),
    )

  async def _run_shell_command(self, command: str) -> None:
    """Run Bash through the current workspace's tool executor, not the model."""
    if not command:
      return
    conversation_id = self.conversation_id
    client = self._client
    try:
      result = await self.call_tool('bash', {'command': command})
    except (grpc.aio.AioRpcError, RuntimeError) as error:
      renderable = _rendering.error_text(str(error))
    else:
      renderable = _rendering.tool_block(
        f'! {command}',
        _text.Text(_rendering.strip_ansi(result.content)),
        error=not result.success,
        wrap=True,
      )
    if self.conversation_id == conversation_id and self._client is client:
      self.dispatch(_actions.ShowText(renderable))

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

  async def fork_workspace(
    self, session_id: str, directory: pathlib.Path | None, name: str
  ) -> pathlib.Path:
    """Capture a host-local fork without switching or interrupting the parent."""
    if self._client is None or self._remote_endpoint is not None:
      raise RuntimeError('forking requires a connected host-local worker')
    return await self._client.fork_workspace(session_id, directory, name)

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

  def composer_state(self) -> _actions.ComposerState:
    """Snapshot the draft and unconfirmed inputs without submitting them again."""
    disconnected, _ = _reducer.reduce(self._state, _actions.WorkerClosed())
    return disconnected.composer

  def transcript_entries(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """Return the agent's current durable trace."""
    return self._transcript.entries()

  def _extension_transcript(self) -> tuple[agent_pb2.AgentMessage, ...]:
    """Copy the durable trace once the open session has finished replaying."""
    if self._state.session.replaying:
      return ()
    return self._transcript.entries()

  def submit_instruction(self, instruction: str) -> None:
    """Queue an instruction for this agent."""
    self.dispatch(_actions.SubmitInstruction(instruction))

  def submit_message(self, message: agent_pb2.UserMessage) -> None:
    """Queue a composed message, with chips for its quotes."""
    text, table = _quotes.display(message, self._state.quotes)
    self.dispatch(
      _actions.QuotesRecalled(table), _actions.SubmitInstruction(text)
    )

  def escalate(self) -> None:
    """Escalate this remote session as its owner and report it here."""
    if self._remote_endpoint is not None:
      self._app.create_background_task(
        self._escalate_remote_session(self._remote_endpoint)
      )

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
    self._transcript.add_decoration(
      renderable, welcome=self._fullscreen and self._live_notices()
    )
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
    self._transcript.add_decoration(
      renderable, welcome=self._fullscreen and self._live_notices()
    )
    if self._state.lifecycle.active and not self._replacing_history():
      self._terminal.queue_emit([renderable])

  def _abort_stream(self) -> None:
    """Retain unfinalized output before closing its only live representation."""
    if self._fullscreen:
      if self._state.lifecycle.active or self._viewport.reading:
        # Retain block identities for reading positions, including hidden tabs.
        blocks = self._live_blocks(show_shortcuts=False)
      else:
        blocks = self._live_tool_blocks('')
        if (preview := self._stream_pane.preview) is not None:
          rendered = _rendering.assistant_turn_renderable(
            (preview.committed_reasoning or '') + preview.reasoning
            if self._shows_reasoning()
            else None,
            (preview.committed_content or '') + preview.content,
            preview.refusal,
            hyperlinks=False,
            skill_names=self._available_skill_names,
          )
          if rendered is not None:
            blocks = (rendered, *blocks)
      identities = [self._transcript.add_decoration(block) for block in blocks]
      self._viewport.settle_live(identities)
    self._stream_pane.abort()

  def _live_notices(self) -> bool:
    return (
      self._secret_input is None or self._fullscreen
    ) and not self._transcript.has_conversation

  def _append_entry(self, entry: agent_pb2.AgentMessage) -> None:
    """Hand welcome notices to scrollback before the first conversation entry."""
    was_empty = not self._transcript.has_conversation
    self._transcript.append(entry)
    if was_empty and self._transcript.has_conversation:
      self._terminal.queue_emit(list(self._transcript.welcome_notices()))

  def _can_focus_tasks(self) -> bool:
    """Whether Down should move from the prompt into the task rows."""
    if (
      self._focus()
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
    if self._browsing_prompt_history():
      return False
    document = self._input.document
    return (
      not document.text
      or document.cursor_position_row == len(document.lines) - 1
    )

  def _browsing_prompt_history(self) -> bool:
    """Whether Down still has newer history entries to step through."""
    return self._input.working_index < len(self._input.history.get_strings())

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
    previous = self._last_composer or self._state.composer
    cursor = _quotes.step_over_end(
      buffer.text, buffer.cursor_position, previous.cursor
    )
    if cursor != buffer.cursor_position:
      # Setting the cursor calls this handler again with the moved cursor.
      buffer.cursor_position = cursor
      return
    composer = _actions.ComposerState(buffer.text, buffer.cursor_position)
    if self._last_composer == composer == self._state.composer:
      return
    self._last_composer = composer
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
    if self._daemon_host:
      return directory == str(self._workspace)
    return _workspace.resolve_cwd(directory) == str(self._workspace)

  def get_available_commands(self) -> tuple[_commands.CommandDef, ...]:
    return _commands.resolve_commands((
      *_commands.COMMANDS,
      *self._extension_host.slash_commands,
      *(
        (command.definition() for command in self._frontend.commands())
        if self._frontend
        else ()
      ),
    ))

  def _command_context(self) -> _commands.CommandContext:
    return _commands.CommandContext(
      model=self._state.status_bar.model,
      reasoning_effort=self._state.status_bar.reasoning_effort,
      conversation_id=self._state.session.conversation_id,
      busy=self._state.activity.busy,
      cwd=self._session_directory(),
      daemon_models=self._workspace_controller.daemon_models()
      if self._daemon_host
      else None,
      remote=self._remote_endpoint is not None,
      session_id=self._session_id,
      speed=self._state.status_bar.speed,
      model_selection=self.model_selection,
    )

  def _focus(self) -> bool:
    """Whether the current view mode hides tool activity."""
    return self._state.status_bar.view_mode is _actions.ViewMode.FOCUS

  def _shows_reasoning(self) -> bool:
    return self._state.status_bar.view_mode.shows_reasoning

  def _session_infos(
    self, scope: _workspace.SessionScope
  ) -> list[_daemon_client.ConversationInfo]:
    """Filter daemon history for the current picker scope."""
    infos = self._metadata.conversations()
    if scope is _workspace.SessionScope.ALL:
      return infos
    directory = str(self._workspace)
    if scope is _workspace.SessionScope.GIT and not self._daemon_host:
      key = _workspace.git_workspace_key(directory)
      if key is not None:
        return [
          info
          for info in infos
          if _workspace.git_workspace_key(info.cwd) == key
        ]
    return [info for info in infos if info.cwd == directory]

  async def _reload_resume_choices(
    self, scope: _workspace.SessionScope
  ) -> None:
    """Refresh scoped picker rows without blocking the event loop."""
    choices = await asyncio.to_thread(self._session_infos, scope)
    self.dispatch(_actions.ResumeChoicesLoaded(scope, tuple(choices)))

  async def _escalate_remote_session(self, endpoint: str) -> None:
    try:
      await _remote_rpc_client.escalate_remote_session(
        endpoint,
        self._state.session.conversation_id,
        secure=self._remote_secure,
      )
    except grpc.aio.AioRpcError as error:
      self.dispatch(
        _actions.ShowText(
          f'Session not escalated: {error.details() or error.code().name}',
          style='ui.error',
        )
      )
      return
    self.dispatch(_actions.ShowText(_remote_rpc_client.ESCALATION_NOTICE))

  async def _prepare_remote(self, request: _effects.CreateRemoteAgent) -> None:
    if (
      await self._extension_host.prepare_remote(
        _actions.CreateRemote(
          request.title, request.workspace, request.escalated
        )
      )
      and self._workspace_effects is not None
    ):
      self._workspace_effects.put_nowait((self, request))

  async def _resolve_resume_reference(self, reference: str) -> None:
    """Resolve one conversation reference and dispatch the immutable result."""
    infos = await asyncio.to_thread(self._metadata.conversations)
    exact = tuple(info for info in infos if info.id == reference)
    matches = exact or tuple(
      info for info in infos if info.id.startswith(reference)
    )
    await asyncio.gather(
      *self.dispatch(_actions.ResumeReferenceResolved(reference, matches))
    )

  async def _resolve_model(self, model_id: str) -> None:
    """Resolve a model id and dispatch its configured option."""
    configured = (
      self._workspace_controller.daemon_models() if self._daemon_host else None
    )
    option = (
      next(
        (
          option
          for option in configured
          if option.model_id == model_id or option.api == model_id
        ),
        dataclasses.replace(
          self.model_selection, model_id=model_id, label=model_id
        ),
      )
      if configured is not None
      else _models.find_model(model_id)
    )
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
    if not option.api and not option.base_url:
      # A bare model id keeps the current protocol and server.
      current = self.model_selection
      option = dataclasses.replace(
        option, api=current.api, base_url=current.base_url
      )
    if self._onboarding_model is not None:
      if not self._onboarding_model.done():
        self._onboarding_model.set_result(option)
      return
    self._model_switch_generation += 1
    generation = self._model_switch_generation
    choices = self._reasoning_choices
    selected = (
      (option, self.reasoning_effort)
      if self._daemon_host
      else await self._authenticate_model(option, initial=False)
    )
    if selected is not None and generation == self._model_switch_generation:
      option, effort = selected
      if choices != self._reasoning_choices:
        effort = self.reasoning_effort
      if self._daemon_host:
        try:
          await asyncio.to_thread(
            self._workspace_controller.save_model, self, option
          )
        except grpc.RpcError as error:
          self.show_error(f'Could not save model: {error}')
          return
      current = self.model_selection
      if (
        self._client is not None
        and self._remote_endpoint is None
        and (option.api != current.api or option.base_url != current.base_url)
      ):
        # Provider credentials enter through startup stdin, never the shared
        # event stream; the daemon restores a managed selection at startup.
        if not await self._restart_worker(
          {
            'model': option.model_id,
            'model_api': option.api,
            'model_base_url': option.base_url or None,
            'base_url': None,
            'reasoning_effort': effort,
          },
          replay=True,
        ):
          return
      self._apply_model(option, effort)
      if self._state.lifecycle.failure is not None:
        await self._retry_worker()

  def _switch_remote_model(self, model_id: str, api: str) -> None:
    """Commit a server model selection, which needs no local credentials."""
    current = self._worker_configuration.values.get('remote_model_api')
    self._worker_configuration = self._worker_configuration.updated({
      'model': model_id,
      'remote_model_api': api or current,
    })
    self.dispatch(_actions.ModelSwitchFinished(model_id))

  def _apply_model(
    self, option: _models.ModelOption, reasoning_effort: str
  ) -> None:
    """Commit a model selection after its sign-in flow completes."""
    updates = {
      'model': option.model_id,
      'model_api': option.api or None,
      'model_base_url': option.base_url or None,
      'base_url': None,
    }
    self._worker_configuration = self._worker_configuration.updated(updates)
    self.dispatch(
      _actions.ModelSwitchFinished(option.model_id, reasoning_effort)
    )

  async def _load_prompt_history(self) -> None:
    request = self._state.selector.selector
    texts = await self._history.prompts()
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

  def _recall_prompts(
    self, messages: Sequence[agent_pb2.UserMessage]
  ) -> list[str]:
    """Composer text for recalled prompts, keeping their quotes resolvable."""
    texts: list[str] = []
    table = self._state.quotes
    for message in messages:
      text, table = _quotes.display(message, table)
      texts.append(text)
    if table != self._state.quotes:
      self.dispatch(_actions.QuotesRecalled(table))
    return texts

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
      self._inspection is not None
      or self._state.drop_confirmation is not None
      or self._state.resume.active
      or self._state.selector.active
      or self._state.secret.active
      or self._state.rewind.active
    )

  def _scroll_history(
    self, event: _mouse_events.MouseEvent
  ) -> _key_bindings.NotImplementedOrNone:
    """Leave modal wheel input to its own scrollable controls."""
    if self._inspection is not None:
      return self._inspection._scroll_history(event)
    if self._modal_ui_active():
      return NotImplemented
    self._app.begin_scroll_burst()
    return self._viewport.mouse_handler(event)

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
      _actions.UserTurn(entry.id, entry.chat.user)
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
    """Select a bound local workspace, or switch the hosted conversation."""
    if self._remote_endpoint is None:
      if create:
        await self.switch_workspace_session()
        return
      if self._workspace_effects is None:
        raise RuntimeError(
          'Local session selection requires the workspace controller'
        )
      self.dispatch(_actions.ClearResume())
      self._workspace_effects.put_nowait((
        self,
        _effects.OpenSessionWorkspace(conversation_id),
      ))
      return
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
        'workspace_id': conversation_id,
        'create_session': create,
        'allow_workspace_mismatch': allow_workspace_mismatch,
      },
      replay=True,
      opening_conversation=conversation_id,
    )

  async def switch_workspace_session(
    self, session_id: str | None = None
  ) -> bool:
    """Use the normal restart path with a different conversation binding."""
    workspace_id = self._extension_context.workspace.id
    if (
      self._remote_endpoint is not None
      or self._state.session.read_only
      or self._state.activity.busy
      or self._state.tasks.tasks
      or self._state.turn.pending_instructions
      or self._client is None
      or not self._client.execution_host
      or workspace_id is None
    ):
      self.show_error(
        'Only the idle execution host may switch this workspace conversation.'
      )
      return False
    return await self._restart_worker(
      {'create_session': session_id is None},
      replay=True,
      opening_conversation=session_id,
      rebind_workspace=workspace_id,
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

  def _initial_input_pending(self) -> bool:
    return (
      self._fullscreen
      and not self._transcript.has_conversation
      and bool(self._state.turn.pending_instructions)
    )

  def _initial_input(self) -> tuple[_transcript_mod.Block, ...]:
    """Place the first optimistic input in the conversation, not the dock."""
    if not self._initial_input_pending():
      return ()
    return (
      _transcript_mod.Block(
        self._state.turn.input_order[0],
        (0, 0),
        _rendering.user_text(
          _quotes.message(
            self._state.turn.pending_instructions[0], self._state.quotes
          ),
          self._available_skill_names,
        ),
      ),
    )

  def _queued_input_pending(self) -> bool:
    return len(self._state.turn.pending_instructions) > (
      1 if self._initial_input_pending() else 0
    )

  def _refresh_key(self) -> object | None:
    if self._state.activity.busy or self._modal_ui_active():
      return None
    navigation = self._navigation_refresh_key()
    if navigation is None:
      return None
    regions = tuple(
      self._extension_host.cache_key(self._state, region)
      for region in _extension.RenderRegion
    )
    if any(key is None for key in regions):
      return None
    return navigation, regions, self._extension_host.themes

  def _component_state(self) -> _state.State:
    return (
      self._frame.content_state() if self._frame is not None else self._state
    )

  def _build_app(self) -> _workspace_application.WorkspaceApplication:
    """Assemble activity, menus, composer, and workspace widgets."""
    bindings = _key_bindings.DynamicKeyBindings(
      lambda: (self._inspection or self)._keymap.bindings
    )
    worktree_window = _containers.DynamicContainer(lambda: self._navigation)
    get_size = _current.get_app_session().output.get_size
    self._frame = (
      _composer_frame.Frame(
        self._extension_host,
        lambda: self._state,
        worktree_window,
        get_size=get_size,
      )
      if self._fullscreen
      else None
    )
    content_size = (
      self._frame.content_size if self._frame is not None else get_size
    )
    component_host = _component.ComponentHost(self._component_state)

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
    input_window = self._input_window = _composer_surface.input_window(
      self._input,
      hidden=lambda: self._state.selector.numbered,
      max_height=lambda: (
        max(1, content_size().rows // 3) if self._fullscreen else None
      ),
      input_processors=[
        self._skill_mention_processor,
        _composer_processors.QuoteChipProcessor(lambda: self._state.quotes),
        _composer_processors.ImageLabelProcessor(self._images.spans),
        _composer_processors.CompletionPreviewProcessor(
          lambda: _completion.prompt_preview(self._state)
        ),
      ],
    )
    self._composer_control = input_window.content

    def jump_to_latest(event: _mouse_events.MouseEvent) -> None:
      if (
        event.button == _mouse_events.MouseButton.LEFT
        and event.event_type == _mouse_events.MouseEventType.MOUSE_UP
      ):
        self.dispatch(_actions.NavigateHistory('latest'))

    history_hint = _containers.ConditionalContainer(
      _containers.Window(
        _controls.FormattedTextControl(
          lambda: [
            (
              '',
              '↓ Back to bottom · Esc',
              jump_to_latest,
            )
          ]
        ),
        height=1,
        style='class:footer',
        align=_containers.WindowAlign.CENTER,
      ),
      filter=_filters.Condition(
        lambda: self._fullscreen and self._viewport.reading
      ),
    )
    self._composer_pane = _composer_pane.Pane(
      self._component_state,
      component_host,
      self._components,
      input_window,
      worktree_window,
      fullscreen=self._fullscreen,
      above_input=history_hint,
      get_size=lambda: _data_structures.Size(
        rows=content_size().rows,
        columns=self._component_state().layout.columns,
      ),
      secret_active=lambda: self._secret_input is not None,
      queued_input_pending=self._queued_input_pending,
    )
    self._bottom_pane = self._composer_pane.bottom
    self._task_window = self._composer_pane.tasks
    notices_window = _containers.ConditionalContainer(
      _containers.Window(
        _controls.FormattedTextControl(
          lambda: [
            *_hyperlinks.ANSI(
              _rendering.render_blocks(
                list(self._transcript.welcome_notices()),
                self._component_state().layout.columns,
                theme_name=_themes.current(
                  self._state.extensions, self._state.extension_themes
                ),
              )
              + '\n'
            ).__pt_formatted_text__(),
            ('[SetCursorPosition]', ''),
          ],
          show_cursor=False,
        ),
        height=lambda: _dimension.Dimension(
          max=max(1, content_size().rows // 3)
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
    root = _containers.HSplit(
      [notices_window, streaming_window, self._bottom_pane],
      align=_containers.VerticalAlign.TOP,
    )
    if self._fullscreen:
      dock = _retained.Layer(
        self._composer_pane.dock,
        lambda: (self._viewport.reading, self._composer_pane.paint_key()),
      )
      pinned_header = _containers.ConditionalContainer(
        _containers.Window(
          _controls.FormattedTextControl(
            lambda: _hyperlinks.ANSI(
              _rendering.render_blocks(
                self._terminal.header,
                self._component_state().layout.columns,
                theme_name=_themes.current(
                  self._state.extensions, self._state.extension_themes
                ),
              )
              + '\n'
            ),
            show_cursor=False,
          ),
          height=lambda: _dimension.Dimension(
            max=max(0, content_size().rows - self._dock_height())
          ),
          dont_extend_height=True,
          wrap_lines=False,
        ),
        filter=_filters.Condition(
          lambda: (
            bool(self._terminal.header)
            and not self._transcript.has_conversation
            and not self._initial_input_pending()
          )
        ),
      )

      root = _component_layout.Rows([
        pinned_header,
        self._viewport.window(
          lambda: [
            transcript_pb2.TranscriptRoute(
              input=data, move=_native_transcript.movement(action)
            )
            for data, action in self._keymap.history_routes()
          ],
          lambda: not self._modal_ui_active(),
        ),
        dock,
      ])
    # A root marker makes empty cells differ from the renderer's blank screen.
    root.style = 'class:theme-root'

    def presentation() -> _containers.Container:
      child = self._inspection
      if child is not None:
        status = child._inspection_status
        role = {
          'running': 'accent',
          'succeeded': 'success',
          'failed': 'error',
        }.get(status, 'muted')
        return _containers.HSplit([
          _containers.Window(
            _controls.FormattedTextControl([
              ('class:ui.muted', '  Parent › '),
              (
                'class:ui.text bold',
                f'Sub-agent {child._session_id}    ',
              ),
              (f'class:ui.{role}', f'● {status.capitalize()}'),
            ]),
            height=1,
          ),
          _containers.Window(height=1, char='─', style='class:ui.faint'),
          child._view.layout.container,
        ])
      return root

    view: _containers.Container = _containers.DynamicContainer(presentation)
    if self._frame is not None:
      view = self._frame.wrap(view)
    app = _workspace_application.WorkspaceApplication(
      layout=_layout.Layout(view, focused_element=input_window),
      key_bindings=_key_bindings.merge_key_bindings([
        self._composer_bindings(),
        bindings,
      ]),
      style=_style.merge_styles([
        _style.DynamicStyle(
          lambda: _themes.prompt_style(
            _themes.current(
              self._state.extensions, self._state.extension_themes
            )
          )
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
      refresh_interval=0.1,
    )

    row_renderer.install(app)
    app.refresh_key = self._refresh_key
    app.background_error_handler = self.show_error
    app.native_event_handler = self._native_event

    app.before_render += self._sync_before_render
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
      app.scroll_handler = self._scroll_history
    return app

  def _native_event(self, event: transcript_pb2.TranscriptEvent) -> None:
    self._viewport.observe(event)
    if self._inspection is not None:
      self._inspection._native_event(event)

  def _sync_before_render(
    self, app: _application.Application[str | None]
  ) -> None:
    # Faults can remove contributions without an application action.
    self.dispatch()
    # Inactive workspaces do not receive the foreground app's resize events.
    columns = max(1, app.output.get_size().columns)
    if columns != self._state.layout.columns:
      self.dispatch(_actions.TerminalResized(columns))
    self._sync_prompt_editing()
    if self._inspection is not None:
      self._inspection._app = self._app
      self._inspection._state = dataclasses.replace(
        self._inspection._state,
        extensions=self._state.extensions,
        extension_themes=self._state.extension_themes,
      )
      self._inspection._sync_before_render(app)

  def _dock_height(self) -> int:
    return self._composer_pane.height()

  def _viewport_blocks(self) -> tuple[_transcript_mod.Block, ...]:
    """Keep active tools in their live renderer until they finish."""
    return self._transcript.blocks(
      active_call_ids=tuple(
        call.call_id for call in self._state.stream.running
      ),
      active_group=(
        self._state.tool_group
        if self._state.status_bar.view_mode is _actions.ViewMode.DEFAULT
        else None
      ),
    )

  def _live_blocks(
    self, *, show_shortcuts: bool = True
  ) -> tuple[_console.RenderableType, ...]:
    preview = self._stream_pane.preview
    approval = self._state.approval.request
    background_shortcut = (
      self._keymap.shortcut_hint(_actions.DetachRequested())
      if show_shortcuts and self._state.can_detach_tools
      else ''
    )
    stamp = (
      preview,
      self._state.status_bar.view_mode,
      self._state.stream.pending,
      self._state.stream.running,
      self._state.tool_group,
      approval.call_id if approval is not None else None,
      background_shortcut,
      self._viewport.reading,
    )
    if stamp == self._live_stamp:
      return self._live_blocks_cache
    self._live_stamp = stamp
    if self._fullscreen:
      source = (
        _rendering.assistant_turn_renderable(
          (preview.committed_reasoning or '') + preview.reasoning
          if self._shows_reasoning()
          else None,
          (preview.committed_content or '') + preview.content,
          preview.refusal,
          skill_names=self._available_skill_names,
        )
        if preview is not None
        else None
      )
      self._live_blocks_cache = (
        *((source,) if source is not None else ()),
        *self._live_tool_blocks(background_shortcut),
      )
      return self._live_blocks_cache
    committed_stamp = (
      preview.committed_reasoning if preview else None,
      preview.committed_content if preview else None,
      self._state.status_bar.view_mode,
    )
    if committed_stamp != self._committed_stamp:
      self._committed_stamp = committed_stamp
      self._committed_blocks = tuple(self._stream_pane.committed_blocks())
      self._committed_source_offset = len(
        _rendering.strip_ansi(committed_stamp[1] or '').lstrip()
      )
    blocks = list(self._committed_blocks)
    if preview is not None:
      tail = _rendering.assistant_turn_renderable(
        preview.reasoning if self._shows_reasoning() else None,
        None,
        None,
        skill_names=self._available_skill_names,
      )
      if tail is not None:
        blocks.append(tail)
      blocks.extend(
        self._streaming_answer.update(
          preview.content,
          source_offset=self._committed_source_offset,
          preview=not self._viewport.reading,
        )
      )
      refusal = _rendering.assistant_turn_renderable(
        None, None, preview.refusal
      )
      if refusal is not None:
        blocks.append(refusal)
    else:
      self._streaming_answer.update('')
    blocks.extend(self._live_tool_blocks(background_shortcut))
    self._live_blocks_cache = tuple(blocks)
    return self._live_blocks_cache

  def _live_tool_blocks(
    self, background_shortcut: str
  ) -> tuple[_console.RenderableType, ...]:
    """Keep live and retained tool output consistent with the scrollback view."""
    if self._state.approval.request is not None:
      return ()
    return _pending.renderables(
      self._state.status_bar.view_mode,
      self._state.stream.running,
      self._state.stream.pending,
      self._state.tool_group,
      background_shortcut,
    )

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
      # Queued inputs hold composer text, so compare in the same form.
      content = _quotes.display(message.user, self._state.quotes)[0]
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

  async def _on_worker_startup_message(
    self, response: agent_pb2.SessionMessage
  ) -> None:
    """Observe redacted startup metadata without treating it as a stored record."""
    if response.HasField('agent_message'):
      self._extension_host.handle_message(response)
    else:
      self._handle_message(response)

  def _handle_message(self, response: agent_pb2.SessionMessage) -> None:
    """Handle one AgentRPC message."""
    if response.HasField('agent_message') and self._transcript.contains(
      response.agent_message, replayed=self._state.session.replaying
    ):
      if response.stream_final:
        self._stream_pane.close()
      return
    if not self._state.session.read_only:
      self._conversation_metrics.observe(
        response,
        conversation_id=self._state.session.conversation_id,
        replaying=self._state.session.replaying,
        turn_duration_seconds=self._state.activity.work_duration(
          time.monotonic()
        ),
      )
    self._extension_host.handle_message(response)
    payload = response.WhichOneof('payload')
    if payload is None:
      self._app.invalidate()
      return
    if payload == 'replay_start':
      self._worker_reconnecting(
        'Synchronizing session state…', synchronizing=True
      )
      return
    if payload == 'state_restored':
      # In-band recovery keeps command replies and host-tool execution alive.
      self._reconcile_inputs(restore_unconfirmed=False)
      self.dispatch(_actions.WorkerStarted(), _actions.WorkerLoadingFinished())
      self._worker_ready.set()
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
      remember: _tool_approvals.ApprovalRule | None = (
        None
        if self._daemon_host or response.hitl_request.always_ask
        else _tool_approvals.from_call(name, call.function.arguments, scope)
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
        args=call.function.arguments,
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
      if self._client is not None and self._client.shared_settings is not None:
        self._receive_shared_settings(self._client.shared_settings)
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
          and self._shows_reasoning()
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
      if (
        self._fullscreen
        and not self._state.session.remote
        and not self._state.session.read_only
        and update.tool_name == 'subagent'
      ):
        child = self._children.get(task_id)
        if child is None and status == 'running':
          child = self._children[task_id] = self._create_child(task_id)
        if child is not None:
          for message in update.messages:
            child._handle_message(message)
          child._inspection_status = status
          if status != 'running':
            completed = agent_pb2.SessionMessage()
            completed.runtime_delta.turn_completed.SetInParent()
            child._handle_message(completed)
            del self._children[task_id]
            if child is not self._inspection:
              self._app.create_background_task(child.close())
      if update.messages:
        return
      task = None
      if update.detached and status == 'running':
        task = _actions.Task(
          id=task_id,
          tool_name=update.tool_name,
          display=update.display if update.HasField('display') else None,
          inspectable=task_id in self._children,
        )
      foreground = not update.detached and status == 'running'
      self.dispatch(_actions.TaskChanged(task_id, task, foreground))
      return
    if kind == 'turn_error':
      self.dispatch(_actions.CompactionFinished())
      self._stream_pane.clear_pending()
      self._record_decoration(_rendering.error_text(delta.turn_error.message))
    elif kind == 'turn_interrupted':
      if self._stream_pane.preview is not None:
        self._abort_stream()
      self._stream_pane.clear_pending()
      self._record_decoration(_rendering.interrupt_text())
    elif kind == 'turn_completed':
      duration = (
        None
        if self._state.session.replaying
        else self._state.activity.work_duration(time.monotonic())
      )
      self._flush_tool_group()
      leftovers = (
        []
        if self._focus()
        else [
          _rendering.unanswered_call(
            call.render_hint,
            compact=self._state.status_bar.view_mode
            is _actions.ViewMode.DEFAULT,
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
        _actions.TurnFinished(duration=duration),
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

  async def _exit_with_error(
    self, message: str, *, disconnected: bool = False
  ) -> None:
    """Record a worker failure and report it when its worktree is active."""
    self._worker_ready.clear()
    if self._state.turn.pending_instructions:
      message += (
        ' Unconfirmed messages restored to the composer; '
        'check conversation history before sending again.'
      )
    self.dispatch(_actions.WorkerFailed(message, disconnected=disconnected))
    for child in self._children.values():
      child._abort_stream()
      child._inspection_status = 'disconnected'
      child.dispatch(
        _actions.WorkerFailed('Parent disconnected.'),
        _actions.ClearRunningToolCalls(),
      )
    error = _rendering.error_text(message)
    self._transcript.add_decoration(error, welcome=self._fullscreen)
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
      theme_name=_themes.current(
        self._state.extensions, self._state.extension_themes
      ),
    )
    self.dispatch(_actions.HistoryRendered(width))
    return ansi

  async def _retry_worker(self) -> None:
    """Retry a failed worker, replaying only when no history was loaded."""
    try:
      if (
        self._state.lifecycle.disconnect_reason is not None
        and self._client is not None
        and await self._client.reconnect()
      ):
        return
      self.dispatch(_actions.WorkerRetryStarted())
      await self._shutdown_worker()
      replay = self._fullscreen or not self._state.session.has_user_message
      if replay:
        # A bounded replay must not shift retained command-output positions.
        self._transcript.clear_replayed()
        for entry in self._transcript.entries():
          self._transcript.append_replayed(entry)
        self.dispatch(_actions.SetReplaying(True))
      await self._start_worker(replay=replay)
    except _rpc_client.WorkspaceClosed:
      await self._worker_closed()
    except _rpc_client.AttachmentLost as error:
      await self._worker_disconnected(str(error))
    except Exception as error:  # noqa: BLE001 - remain retryable in the UI.
      await self._exit_with_error(
        f'could not start agent worker: {type(error).__name__}: {error}',
        disconnected=self._state.lifecycle.disconnect_reason is not None,
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
        if self._workspace_effects is None:
          raise RuntimeError('Local rewind requires the workspace controller')
        forked = await self._client.fork_session(entry_id)
        self._workspace_effects.put_nowait((
          self,
          _effects.OpenSessionWorkspace(
            forked, _quotes.message(instruction, self._state.quotes)
          ),
        ))
        self.dispatch(_actions.TurnFinished())
        return
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
      'workspace_id': forked,
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
    self._conversation_metrics.steering_submitted()
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
    request.message.chat.user.CopyFrom(
      _quotes.message(instruction, self._state.quotes)
    )
    request.message.chat.success = True
    if message_id is None:
      self.dispatch(_actions.InputSubmitted(request.message_id, instruction))
    images = self._images.attached(instruction)
    problem = (
      _images.unsupported(self.model)
      if images and not _models.supports_images(self.model_selection)
      else _images.too_many(images)
    )
    if problem is not None:
      # Rejecting the input returns the draft, images included, to the composer.
      self.dispatch(
        _actions.InputRejected(request.message_id),
        _actions.ShowText(problem, style='ui.warning'),
      )
      return
    request.message.chat.user.images.extend(images)
    try:
      if self._state.lifecycle.quitting:
        raise RuntimeError('Agent worker is closing; message not sent.')
      if self._worker_ready.is_set():
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
        speed=self._state.status_bar.speed,
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
      self._extension_host.prepare_request(request)
      self._client.send(request)
      if request.HasField('detach_tools'):
        self._conversation_metrics.tools_background_requested()

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
      self._workspace_controller.retarget_remote(self, conversation_id)
      self._worker_configuration = self._worker_configuration.updated({
        'workspace_id': conversation_id,
      })
    elif self._workspace_effects is not None:
      self._workspace_effects.put_nowait((
        self,
        _effects.RetargetSession(conversation_id),
      ))
    self._transcript.clear_decorations()
    self.dispatch(_actions.SessionOpened(conversation_id, composer))
    self._start_switch_screen()
    self._history.reload(conversation_id)
    # Resetting makes prompt_toolkit reload history; preserve the current draft.
    self._input.reset(self._input.document)

  async def restart_daemon(self) -> str:
    """Run explicit replacement independently of the current worker attachment."""
    transport = _local_rpc_client.Transport(
      self._worker_command,
      self._worker_configuration.values,
      workspace=self._workspace,
      log_dir=self._workspace,
      on_warning=self.show_error,
    )
    return await transport.manage('restart')

  def _workspace_connection(self) -> tuple[grpc.aio.Channel, str]:
    if self._client is None:
      raise _shell.ProcessError('Workspace is not attached')
    if not isinstance(self._client.transport, _local_rpc_client.Transport):
      raise RuntimeError('Workspace is not attached to the selected daemon')
    return self._client.transport.workspace_connection()

  def _set_mcp_connections(
    self, connections: Sequence[mcp_configuration.McpConnection]
  ) -> None:
    if self._client is None or not self._client.execution_host:
      return
    request = self._new_request()
    request.set_mcp_connections.CopyFrom(
      mcp_configuration.runtime_connections(connections, reconnect=True)
    )
    self._send_request(request)

  def _transport(
    self,
    configuration: _rpc_client.WorkerConfiguration,
    on_message: _rpc_client.MessageHandler | None = None,
  ) -> _rpc_client.Transport:
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
        'mcp_connections': self._extension_host.mcp_connections,
        'additional_skills': [
          dataclasses.asdict(skill) | {'path': str(skill.path)}
          for skill in self._extension_host.skills
        ],
      },
      workspace=self._workspace,
      log_dir=log_dir,
      on_message=on_message or self._on_worker_startup_message,
      remote_endpoint=self._remote_endpoint,
      remote_secure=self._remote_secure,
    )

  def _show_reattaching(self, reattaching: bool) -> None:
    resumed = 'thinking…' if self._state.activity.busy else ''
    self.dispatch(
      _actions.SetStatus('reconnecting…' if reattaching else resumed)
    )

  def _receive_shared_settings(self, settings: daemon_pb2.Workspace) -> None:
    inference = settings.inference
    model = inference.model or (
      'server default' if self._remote_endpoint is not None else ''
    )
    mode = approval_modes.from_protocol(settings.approval_mode)
    self._worker_configuration = self._worker_configuration.updated({
      'model': model,
      'model_api': inference.api,
      'remote_model_api': inference.api
      if self._remote_endpoint is not None
      else None,
      'model_base_url': inference.base_url or None,
      'base_url': None,
      'approval_mode': mode,
    })
    self.dispatch(
      _actions.SharedSettingsReceived(
        model,
        inference.reasoning_effort,
        mode,
        _rpc_client.inference_speed(inference),
      )
    )

  async def _start_worker(
    self,
    *,
    replay: bool,
    configuration: _rpc_client.WorkerConfiguration | None = None,
  ) -> None:
    """Start a fresh worker behind the existing command/event interface."""
    conversation_id = self._state.session.conversation_id
    launch_configuration = configuration or self._worker_configuration
    if (
      self._remote_endpoint is None
      and not self._daemon_host
      and not launch_configuration.values.get('sandboxed')
      and not _onboarding.is_trusted_directory(self.workspace)
    ):
      raise PermissionError(
        f'Open Mirror in {str(self.workspace)!r} to review folder trust before attaching'
      )
    if self._extensions_loaded is not None:
      await asyncio.shield(self._extensions_loaded)
    # Onboarding can change the selected provider while startup is waiting.
    launch_configuration = configuration or self._worker_configuration
    self._extension_host.prepare()
    if conversation_id != self._state.session.conversation_id:
      return  # Session switching owns the replacement worker and its inputs.
    launch_configuration = launch_configuration.updated({
      'approval_mode': self._state.status_bar.approval_mode,
    })
    self._extension_host.validate_selection()
    self._host_tools_registry.validate_selection()
    self._host_tools_registry.worker_starting()
    client = _rpc_client.Client(
      self._transport(launch_configuration),
      conversation_id,
      on_message=self._handle_message,
      on_error=self._exit_with_error,
      on_settings=self._receive_shared_settings,
      on_settings_error=self.show_error,
      on_closed=self._worker_closed,
      on_reconnect=self._worker_reconnecting,
      on_disconnect=self._worker_disconnected,
      resynchronize=True,
      on_reattaching=self._show_reattaching,
      host_tools_registry=self._host_tools_registry
      if self._remote_endpoint is None and not self._daemon_host
      else None,
      web_provider=launch_configuration.web_provider
      if self._remote_endpoint is None and not self._daemon_host
      else None,
      inference_config=None
      if self._daemon_host
      else launch_configuration.inference_config(
        self._state.status_bar.model,
        self._state.status_bar.reasoning_effort,
        remote=self._remote_endpoint is not None,
        speed=self._state.status_bar.speed,
      ),
    )
    self._client = client
    try:
      if self._remote_endpoint is not None and bool(
        launch_configuration.values.get('create_session', False)
      ):
        workspace = launch_configuration.remote_workspace
        self.dispatch(
          _actions.SetStatus(
            f'preparing remote workspace {workspace.name}…'
            if workspace is not None
            else 'starting remote session…'
          )
        )
        await _remote_rpc_client.start_remote_session(
          self._remote_endpoint,
          self._state.session.conversation_id,
          secure=self._remote_secure,
          workspace=launch_configuration.remote_workspace,
          daemon_address=self._metadata.local_address,
        )
        self.dispatch(
          _actions.SessionCreated(self._state.session.conversation_id)
        )
      if self._remote_endpoint is not None and bool(
        launch_configuration.values.get('escalate_session', False)
      ):
        await self._escalate_remote_session(self._remote_endpoint)
      if self._remote_endpoint is not None:
        self.dispatch(_actions.SetStatus('connecting to remote agent…'))
      await client.start(
        replay=replay,
        wait_for_attachment=True,
        on_attached=self._extension_host.start,
      )
    except BaseException:
      if self._client is client:
        self._client = None
      raise
    self._worker_configuration = launch_configuration.after_start()
    if client.shared_settings is not None:
      self._receive_shared_settings(client.shared_settings)
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
    # Input entered during startup keeps its original idempotency key.
    for message_id, instruction in self._state.turn.submitted_inputs:
      self._send_instruction(instruction, message_id=message_id)
    if self._state.lifecycle.active:
      await self._foreground_extensions()
    if (
      self._remote_endpoint is None
      and not self._daemon_host
      and client.execution_host
    ):
      self._host_tools_registry.worker_started(
        lambda: self._restart_worker({}, replay=False)
      )

  async def _foreground_extensions(self) -> None:
    if (
      self._state.lifecycle.active
      and self._worker_ready.is_set()
      and self._client is not None
      and await self._extension_host.foreground()
    ):
      await self._user_extensions.on_foreground(self._extension_context)

  def _worker_reconnecting(
    self, reason: str | None, *, synchronizing: bool = False
  ) -> None:
    if reason is not None:
      self._worker_ready.clear()
      self.dispatch(
        _actions.WorkerReconnecting(reason, synchronizing=synchronizing),
        _actions.SetReplaying(True),
        _actions.CloseStream(),
        _actions.ClearRunningToolCalls(),
      )
      self._transcript.clear_replayed()
      # Legacy reattachments only replay a tail; in-band snapshots include the prefix.
      if not synchronizing:
        for entry in self._transcript.entries():
          self._transcript.append_replayed(entry)
      return
    self._reconcile_inputs(restore_unconfirmed=True)
    self.dispatch(_actions.WorkerStarted(), _actions.WorkerLoadingFinished())
    self._worker_ready.set()
    # Selection during reconnect deferred this tab's foreground hooks.
    self._app.create_background_task(self._foreground_extensions())

  def _reconcile_inputs(self, *, restore_unconfirmed: bool) -> None:
    pending = dict((
      *self._state.turn.submitted_inputs,
      *self._state.turn.awaiting_inputs,
    ))
    for message_id in reversed(self._state.turn.input_order):
      if message_id not in pending:
        continue
      instruction = pending[message_id]
      if self._transcript.transcript_index(message_id) is not None:
        self.dispatch(_actions.UserMessageObserved(instruction, message_id))
      elif restore_unconfirmed:
        self.dispatch(_actions.InputRejected(message_id))
        self.show_error(
          'Message delivery unconfirmed; draft restored. Check history before sending again.'
        )

  async def _worker_disconnected(self, message: str) -> None:
    self.dispatch(_actions.SetReplaying(False))
    await self._exit_with_error(message, disconnected=True)

  async def _worker_closed(self) -> None:
    """Treat a deliberate shared close as a lifecycle change, not a transport failure."""
    self._worker_ready.clear()
    self.dispatch(_actions.WorkerClosed())
    self._workspace_controller.refresh()

  async def _shutdown_worker(self) -> None:
    """Close and discard the active worker client."""
    await self._user_extensions.on_background(self._extension_context)
    await self._extension_host.background()
    await self._host_tools_registry.worker_stopped()
    self.dispatch(_actions.WorkerStopped())
    self._worker_ready.clear()
    client, self._client = self._client, None
    if client is not None:
      await client.close()
    if self._inspection is not None:
      self._inspection._app = self._inspection._view
      self._children[self._inspection._session_id] = self._inspection
      self._inspection = None
      self._clear_screen_selection()
      self._app.layout.focus(self._input_window)
    for child in self._children.values():
      await child.close()
    self._children.clear()

  async def _restart_worker(
    self,
    updates: Mapping[str, object],
    *,
    replay: bool,
    opening_conversation: str | None = None,
    rebind_workspace: str | None = None,
  ) -> bool:
    """Explicitly close an idle owned workspace before attaching new configuration."""
    transport: _local_rpc_client.Transport | None = None
    if self._remote_endpoint is None and self._client is not None:
      if self._state.activity.busy or not self._client.execution_host:
        self.show_error(
          'Only the idle execution host may restart this workspace.'
        )
        return False
      if not isinstance(self._client.transport, _local_rpc_client.Transport):
        raise RuntimeError('Local workspace is not connected to the daemon')
      transport = self._client.transport
    self.restarting = True
    try:
      old_conversation = self._state.session.conversation_id
      old_configuration = self._worker_configuration
      launch_configuration = old_configuration.updated(
        dict(updates)
        | {
          'require_new_worker': opening_conversation is None,
          'attach_running': False,
        }
      )
      # Detach before stopping the worker so its expected EOF cannot report a failure.
      await self._shutdown_worker()
      if transport is not None:
        try:
          await transport.close_workspace()
        except grpc.RpcError:
          await self._exit_with_error(
            'Could not close the workspace; no replacement was started.'
          )
          return False
      if rebind_workspace is not None:
        try:
          opened = await asyncio.to_thread(
            self._metadata.switch_session,
            rebind_workspace,
            old_conversation,
            session_id=opening_conversation,
          )
        except grpc.RpcError as error:
          await self._exit_with_error(
            f'Could not switch conversation; inspect before retrying: {error}'
          )
          return False
        self.dispatch(
          _actions.ClearResume(),
          _actions.SessionSwitchStarted(
            _actions.PendingSwitch(
              opened.session_id, create=opening_conversation is None
            )
          ),
        )
        opening_conversation = opened.session_id
        launch_configuration = launch_configuration.updated({
          'conversation_id': opened.session_id,
          'require_new_worker': True,
        })
        self._worker_configuration = launch_configuration
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
        if self._daemon_host:
          await self._exit_with_error(f'Could not start workspace: {error}')
          return False
        if rebind_workspace is not None:
          await self._exit_with_error(
            f'Conversation switched, but its worker did not start: {error}'
          )
          return False
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
          False
          if opening_conversation is None
          else self._state.session.replaying
        )
        self.dispatch(_actions.WorkerRecoveryFinished(replaying))
        self._record_decoration(
          _rendering.error_text(
            f'could not start agent worker: {type(error).__name__}: {error}'
          )
        )
        return False
    finally:
      self.restarting = False
      self._workspace_controller.refresh()

  async def ensure_api_key(self) -> bool:
    """Authenticate a new worker; an existing daemon worker owns its credentials."""
    if self._daemon_host or self._worker_configuration.values.get(
      'attach_running'
    ):
      return True
    option = self.model_selection
    if option.api == 'llama_cpp':
      # A saved server that no longer answers must not block /model and /effort.
      if option.base_url != _models.MANAGED:
        selected = await self._resolve_local_selection(option)
        if selected is not None and selected != (option, self.reasoning_effort):
          self._apply_model(*selected)
      return True
    selected = await self._authenticate_model(option, initial=True)
    if selected is None:
      return False
    if selected != (option, self.reasoning_effort):
      self._apply_model(*selected)
    return True

  def _compaction_api(self, option: _models.ModelOption) -> str | None:
    """The API a separate compaction model runs on with `option`, if one is set."""
    values = self._worker_configuration.values
    if not values.get('compaction_model'):
      return None
    return str(values.get('compaction_model_api') or option.api)

  async def _resolve_local_selection(
    self, option: _models.ModelOption
  ) -> tuple[_models.ModelOption, str] | None:
    """Check a llama.cpp selection against its server and pick its effort.

    The current effort carries over when the model offers it; otherwise the
    selection starts at the daemon's default for that model.
    """
    try:
      if self._local_runtime is None:
        raise ValueError('Local models are unavailable in this session.')
      if option.base_url != _models.MANAGED:
        option = dataclasses.replace(
          option, base_url=_models.local_base_url(option.base_url)
        )
      _models.validate_selection(
        option,
        self.reasoning_effort,
        compaction_api=self._compaction_api(option),
      )
      info = (
        await self._local_runtime.execute(
          local_models_pb2.ExecuteRequest(
            base_url=option.base_url, info=option.model_id
          )
        )
      ).info
    except ValueError as error:
      self.dispatch(_actions.ShowText(str(error), style='ui.error'))
      return None
    effort = self.reasoning_effort
    return option, effort if effort in info.efforts else info.default_effort

  async def _authenticate_model(
    self, option: _models.ModelOption, *, initial: bool
  ) -> tuple[_models.ModelOption, str] | None:
    """Resolve sign-in or local readiness, and the effort that goes with them."""
    if self._secret_input is not None:
      return None
    while True:
      api = option.api or self.model_selection.api
      if api == 'llama_cpp':
        return await self._resolve_local_selection(option)
      effort = self.reasoning_effort
      if self.model_selection.api == 'llama_cpp':
        # The on/off thinking controls exist only on llama.cpp.
        effort = _models.inherited_reasoning_effort(
          api,
          effort if effort in _inference_choices.REASONING_EFFORTS else 'none',
        )
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
        return option, effort
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
        return option, effort
      if result is _reflection.SignInResult.CANCELLED:
        return None
      selected = await self._choose_onboarding_model()
      if selected is not None:
        option = selected
      elif not initial:
        return None

  async def _choose_onboarding_model(self) -> _models.ModelOption | None:
    """Await the shared model picker without starting an unauthenticated worker."""
    controls = self._extension_host.status(
      _inference_controls.InferenceControlsExtension.id()
    )
    # The llama.cpp menus run in that extension, which acts only once the
    # workspace starts, so initial sign-in cannot offer them.
    runtime = (
      self._local_runtime
      if controls is not None and controls.status == 'active'
      else None
    )
    self._onboarding_model = asyncio.get_running_loop().create_future()
    try:
      self.dispatch(
        _actions.OpenSelector(
          _inference_controls.model_selector(self.model, runtime)
        )
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
      welcome = _rendering.render_key_setup(
        'Anthropic' if api == 'anthropic-messages' else 'Reflection',
        _api_keys.ANTHROPIC_KEY_URL
        if api == 'anthropic-messages'
        else _reflection.KEY_URL,
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
    self.dispatch(_actions.WorkerLoadingStarted())
    try:
      await self._start_worker(replay=True)
      if self._state.lifecycle.active:
        self.dispatch(_actions.CliStarted())
    except _rpc_client.WorkspaceClosed:
      await self._worker_closed()
    except _rpc_client.AttachmentLost as error:
      await self._worker_disconnected(str(error))
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
      # The inline prompt already renders first-use notices.
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
    navigation_refresh_key: Callable[[], object | None] | None = None,
  ) -> str | None:
    """Run this view with navigation supplied by the CLI composition root."""
    self._navigation = (
      navigation if navigation is not None else _containers.Window(height=0)
    )
    self._navigation_refresh_key = navigation_refresh_key or (
      lambda: (
        None if navigation is not None or before_render is not None else ()
      )
    )
    if before_render is not None:
      self._app.before_render += before_render
    await self._stream_pane.wait_for_emits()
    view = self._app
    if screen is not None:
      self._app = screen
    self._app.erase_when_done = False
    self.dispatch(_actions.SetActive(True))
    try:
      # Supervised views must not restore raw mode after their owner has exited.
      with (
        contextlib.nullcontext()
        if screen is not None
        else self._app.input.raw_mode()
      ):
        self._app.prepare_for_run()
        self._paint_activation(show_context=show_context, label=label)

        async def foreground() -> None:
          if self._state.lifecycle.failure is not None:
            return
          await self._foreground_extensions()

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
      if self._inspection is not None:
        self._inspection._app = self._inspection._view
      self._navigation = _containers.Window(height=0)
      self._navigation_refresh_key = lambda: ()
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
      await self._viewport.close()
      await self._stream_pane.wait_for_emits()

  def print_resume_hint(self) -> None:
    """Print how to continue after shutdown."""
    with self._terminal.console.use_theme(
      _themes.rich_theme(
        _themes.current(self._state.extensions, self._state.extension_themes)
      )
    ):
      self._terminal.console.print(
        'Bye. Resume with mirror --continue.', style='ui.muted'
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
  *,
  workspace_changed: Callable[[], None] = lambda: None,
) -> None:
  """Wire workspace lifecycle and view results without a back-reference."""
  effects_task = asyncio.create_task(
    _consume_workspace_effects(controller, queue, workspace_changed)
  )
  show_context = False
  try:
    await controller.initialize()
    while True:
      workspace_changed()
      action = await activate(controller.selected, show_context)
      if action is None:
        break
      if action == 'drop':
        await controller.drop()
      elif action in {'next', 'previous'}:
        await controller.cycle(1 if action == 'next' else -1)
      elif action in {
        'install-sandbox',
        'download-sandbox',
        'start-sandbox',
        'copy-sandbox',
        'cancel-sandbox',
      } and isinstance(controller.selected, _workspaces.SandboxWorkspace):
        pending = controller.selected
        if action == 'cancel-sandbox':
          await controller.cancel_sandbox(pending)
          await controller.drop()
        elif action == 'install-sandbox':
          controller.start_sandbox(pending, 'install')
        elif action == 'download-sandbox':
          controller.start_sandbox(pending, 'download')
        elif action == 'copy-sandbox':
          controller.start_sandbox(pending, 'copy')
        else:
          controller.start_sandbox(pending)
      elif action.startswith('select:'):
        await controller.select(int(action.removeprefix('select:')))
      elif action.startswith('create:'):
        try:
          kind = action.removeprefix('create:')
          if kind == 'sandbox':
            pending = await controller.create_sandbox()
            if pending is not None:
              await controller.select_workspace(pending.identity)
          elif kind == 'remote':
            index = await controller.create_remote()
            if index is not None:
              await controller.select(index)
          else:
            identity = await controller.create_workspace(kind)
            await controller.select_workspace(identity)
        except (ValueError, _worktrees.WorktreeError, grpc.RpcError) as error:
          controller.selected.show_error(str(error))
      elif action.startswith(('folder:', 'create-folder:')):
        operation, _, directory = action.partition(':')
        try:
          identity = await controller.create_workspace(
            'folder',
            directory=pathlib.Path(directory),
            create_directory=operation == 'create-folder',
          )
          await controller.select_workspace(identity)
        except (OSError, ValueError, grpc.RpcError) as error:
          controller.selected.show_error(str(error))
      elif action.startswith('resume:'):
        try:
          await controller.open_session(action.removeprefix('resume:'))
        except (OSError, ValueError, grpc.RpcError) as error:
          controller.selected.show_error(str(error))
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
  extension_registry: _extension.ExtensionRegistry | None = None,
  is_current: Callable[[], bool] = lambda: True,
) -> str | None:
  """Render an unattached workspace through the ordinary draft composer."""
  bindings = _workspace_application.terminal_bindings()
  sandbox = (
    workspace if isinstance(workspace, _workspaces.SandboxWorkspace) else None
  )
  menu = _state.MenuState(active=True, max_rows=3)

  def choices() -> tuple[_selection.Choice, ...]:
    return _sandbox_view.choices(sandbox) if sandbox is not None else ()

  def handoff(result: str) -> Callable[[_key_binding.KeyPressEvent], None]:
    return lambda event: (screen or app).exit_for_handoff(result)

  @bindings.add('up', filter=_filters.Condition(lambda: bool(choices())))
  @bindings.add('down', filter=_filters.Condition(lambda: bool(choices())))
  def move(event: _key_binding.KeyPressEvent) -> None:
    nonlocal menu
    menu = menu.move(
      -1 if event.key_sequence[0].key == 'up' else 1, len(choices())
    )

  @bindings.add('enter', filter=_filters.Condition(lambda: bool(choices())))
  def accept(_: _key_binding.KeyPressEvent) -> None:
    choice = menu.selected_item(choices())
    if choice is not None:
      (screen or app).exit_for_handoff(choice.value)

  @bindings.add('<any>', filter=_filters.Condition(lambda: bool(choices())))
  def numbered(event: _key_binding.KeyPressEvent) -> None:
    if len(event.data) == 1 and event.data in '123456789':
      index = int(event.data) - 1
      if index < len(choices()):
        (screen or app).exit_for_handoff(choices()[index].value)

  draft = _buffer.Buffer(
    multiline=True,
    document=_document.Document(workspace.draft.text, workspace.draft.cursor),
    read_only=_filters.Condition(lambda: bool(choices())),
  )

  def draft_changed(value: _buffer.Buffer) -> None:
    workspace.draft = _actions.ComposerState(value.text, value.cursor_position)

  draft.on_text_changed += draft_changed
  draft.on_cursor_position_changed += draft_changed

  @bindings.add(
    'enter',
    filter=_filters.Condition(
      lambda: not choices() and workspace.error is None
    ),
  )
  def unavailable(event: _key_binding.KeyPressEvent) -> None:
    event.app.output.bell()

  @bindings.add('c-j', filter=_filters.Condition(lambda: not choices()))
  def newline(_: _key_binding.KeyPressEvent) -> None:
    draft.insert_text('\n')

  if sandbox is not None:
    bindings.add('escape', eager=_keymap.standalone_escape)(
      handoff('cancel-sandbox')
    )
  bindings.add('tab')(handoff('next'))
  bindings.add('s-tab')(handoff('previous'))

  def quit_view(_: _key_binding.KeyPressEvent) -> None:
    (screen or app).exit()

  bindings.add('c-c')(quit_view)
  bindings.add('c-d')(quit_view)
  bindings.add(
    'enter',
    filter=_filters.Condition(
      lambda: sandbox is None and workspace.error is not None
    ),
  )(handoff('drop'))
  for index in range(9):
    bindings.add(f'c-{index + 1}')(handoff(f'select:{index}'))
  _extended_keys.install()

  def replaced(_: _application.Application[str | None]) -> None:
    if (workspace.finished or not is_current()) and not app.is_done:
      (screen or app).exit_for_handoff('replaced')

  app = _workspace_application.WorkspaceApplication(
    layout=_pending_view.layout_for(
      workspace,
      lambda: menu,
      navigation,
      draft,
      theme_name,
      (extension_registry or _extension.ExtensionRegistry()).create_view(
        extensions=tuple(workspace.preferences.extensions)
      ),
    ),
    full_screen=True,
    key_bindings=bindings,
    style=_style.DynamicStyle(lambda: _themes.prompt_style(theme_name())),
    color_depth=_themes.prompt_color_depth(),
    refresh_interval=0.1,
    before_render=replaced,
  )
  app.ttimeoutlen = 0.05
  # Only standalone views own terminal modes; the supervisor already owns them.
  with contextlib.nullcontext() if screen is not None else app.input.raw_mode():
    if screen is None:
      app.prepare_for_run()
      app.output.erase_screen()
      app.output.cursor_goto(0, 0)
    try:
      if screen is not None:
        return await screen.show_view(
          app,
          accept_buffered_input=lambda: (
            not choices() and workspace.error is None
          ),
        )
      return await app.run_async()
    finally:
      app.finish_handoff()


async def _consume_workspace_effects(
  controller: _workspace_controller.WorkspaceController,
  queue: asyncio.Queue[tuple[_Repl, _effects.Effect]],
  workspace_changed: Callable[[], None],
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
          _effects.RetargetSession,
        ),
      ) and (
        not isinstance(selected, _workspaces.AgentWorkspace)
        or selected.agent is not agent
      ):
        continue
      await handle_workspace_effect(controller, agent, effect)
      workspace_changed()
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
    case _effects.RestartDaemon():
      await controller.restart_daemon(agent.restart_daemon)
    case _effects.OpenNewWorkspace():
      agent.dispatch(_actions.OpenSelector(controller.new_workspace_selector()))
    case _effects.CreateWorkspace(kind):
      identity = await controller.create_workspace(kind)
      if await controller.select_workspace(identity):
        agent.handoff('workspace')
    case _effects.OpenFolder(path, confirmed, create_directory):
      source = controller.selected
      if not isinstance(source, _workspaces.AgentWorkspace):
        return
      if source.remote:
        raise ValueError('Open folders from a local workspace.')
      resolved = await asyncio.to_thread(
        controller.resolve_directory, path, base=agent.workspace
      )
      directory = pathlib.Path(resolved.directory)
      if not resolved.exists or not controller.is_trusted_directory(directory):
        # Consent applies only to the canonical path shown, not a retargeted link.
        if (
          not confirmed
          or directory != pathlib.Path(path)
          or (not resolved.exists and not create_directory)
        ):
          agent.dispatch(
            _actions.OpenSelector(
              _selection.Selector(
                id='folder-trust',
                title='Trust directory?'
                if resolved.exists
                else 'Create directory?',
                header=_text.Text.assemble(
                  (str(directory), 'ui.text.bold'),
                  '\n\nMirror can read/edit files and run commands here.\n'
                  'Only open directories whose files and instructions you trust.',
                  style='ui.muted',
                ),
                choices=(
                  _selection.Choice('cancel', 'Cancel'),
                  _selection.Choice(
                    'trust',
                    'Trust and open' if resolved.exists else 'Create and open',
                  ),
                ),
                selected='cancel',
                numbered=True,
                on_select=lambda value: (
                  _actions.OpenFolder(
                    str(directory),
                    confirmed=True,
                    create_directory=not resolved.exists,
                  )
                  if value == 'trust'
                  else _actions.CloseSelector()
                ),
              )
            )
          )
          return
        controller.trust_directory(directory)
      workspace = await controller.open_folder(
        source.id, directory, create_directory=confirmed and create_directory
      )
      await controller.select_workspace(workspace.id)
      agent.handoff('folder')
    case _effects.OpenSessionWorkspace(session_id, message):
      await controller.open_session(session_id, message)
      agent.handoff('selected')
    case _effects.CloseWorkspace():
      selected = controller.selected
      if isinstance(selected, _workspaces.AgentWorkspace):
        if await controller.close_workspace(selected.id):
          agent.handoff('closed')
    case _effects.ForkWorkspace(name):
      source = controller.selected
      if not isinstance(source, _workspaces.AgentWorkspace) or source.remote:
        raise ValueError('Fork from a host-local workspace.')
      pending = await controller.fork_workspace(source.id, name)
      if await controller.select_workspace(pending.identity):
        agent.handoff('fork')
    case _effects.CreateManagedWorktree(name):
      await controller.create(name)
    case _effects.CreateSandboxWorkspace(name):
      pending = await controller.create_sandbox(name)
      if pending is not None and await controller.select_workspace(
        pending.identity
      ):
        agent.handoff('sandbox')
    case _effects.DropWorkspace(confirmed, skip_confirmation):
      confirmation = None if confirmed else controller.drop_confirmation()
      if confirmation is not None:
        agent.dispatch(_actions.WorktreeDropConfirmationRequested(confirmation))
      elif await controller.drop(skip_confirmation=skip_confirmation):
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
    case _effects.CreateRemoteAgent(title, workspace, escalated):
      index = await controller.create_remote(
        title, workspace, escalated=escalated
      )
      if index is not None and controller.can_select(index):
        agent.handoff(f'select:{index}')
    case _effects.OpenWorkspaceBrowser():
      controller.refresh()
      agent.handoff('open')
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
    case _effects.SaveWorktreePreferences(
      reasoning_effort, approval_mode, speed
    ):
      controller.save_preferences(
        reasoning_effort, approval_mode, speed, agent=agent
      )
    case _effects.SaveLastModel():
      controller.save_model(agent, agent.model_selection)
    case _effects.RetargetSession(session_id):
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
  extension_registry: _extension.ExtensionRegistry | None = None,
  frontend_context: _frontend.Context | None = None,
  extension_lifecycle: _extension.ExtensionLifecycle | None = None,
) -> str | None:
  def navigation_text() -> _formatted_text.StyleAndTextTuples:
    return _worktree_tabs.render_tabs(
      controller.workspaces,
      workspace.identity,
      tags=controller.tab_tags(),
      width=_current.get_app().output.get_size().columns,
    )

  navigation = _containers.Window(
    _controls.FormattedTextControl(navigation_text),
    height=lambda: (
      2
      if len(controller.visible_workspaces()) > 1 or controller.tab_tags()
      else 0
    ),
    wrap_lines=False,
  )
  if isinstance(workspace, _workspaces.EmptyWorkspace):
    return await _workspace_browser.activate(
      controller,
      screen,
      theme_name=theme_name,
      frontend_context=frontend_context,
      extension_registry=extension_registry,
      extension_lifecycle=extension_lifecycle,
    )
  if isinstance(workspace, _workspaces.PendingWorkspace):
    result = await _activate_pending(
      workspace,
      navigation,
      screen,
      theme_name=theme_name,
      extension_registry=extension_registry,
      is_current=lambda: controller.selected is workspace,
    )
    if workspace.finished:
      replacement = next(
        (
          item
          for item in controller.workspaces
          if item.identity == workspace.identity
        ),
        None,
      )
      if isinstance(replacement, _workspaces.AgentWorkspace):
        replacement.draft = workspace.draft
    return result
  if not isinstance(workspace, _workspaces.AgentWorkspace):
    raise TypeError('unknown workspace implementation')
  agent = workspace.agent
  if not isinstance(agent, _Repl):
    raise TypeError('workspace has no terminal view')
  starting = await workspace.start()
  if workspace.is_sandbox and not starting.done():
    pending = _workspaces.PendingWorkspace(
      workspace.identity,
      workspace.label,
      notice='Preparing sandbox…',
      draft=workspace.draft or _actions.ComposerState(),
      preferences=daemon_pb2.WorkspacePreferences(
        definition=mirror_pb2.AgentDefinition(
          model=agent.model,
          reasoning_effort=agent.reasoning_effort,
        ),
        approval_mode=approval_modes.to_protocol(agent.approval_mode),
      ),
    )

    def ready(_: asyncio.Task[None]) -> None:
      pending.finished = True

    starting.add_done_callback(ready)
    result = await _activate_pending(
      pending,
      navigation,
      screen,
      theme_name=theme_name,
      extension_registry=extension_registry,
    )
    workspace.draft = pending.draft
    return result
  if workspace.draft is not None:
    agent.dispatch(
      _actions.RestorePrompt(workspace.draft.text, workspace.draft.cursor)
    )
    workspace.draft = None

  def refresh_picker(_: _application.Application[str | None]) -> None:
    if controller.selected is not workspace:
      agent.handoff('workspace')
      return
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
    navigation_refresh_key=navigation_text,
  )
  result = (
    await activate(screen=screen) if screen is not None else await activate()
  )
  if result == 'open':
    return await _workspace_browser.activate(
      controller,
      screen,
      theme_name=theme_name,
      frontend_context=frontend_context,
      extension_registry=extension_registry,
      extension_lifecycle=extension_lifecycle,
    )
  return result


def _selected_llama_server(
  controller: _workspace_controller.WorkspaceController,
) -> str:
  """The visible workspace's llama.cpp server, or empty for any other model."""
  selected = controller.selected
  if not isinstance(selected, _workspaces.AgentWorkspace):
    return ''
  option = selected.agent.model_selection
  return option.base_url if option.api == 'llama_cpp' else ''


def _saved_view_mode(settings: Mapping[str, object]) -> _actions.ViewMode:
  """Read the shared display preference, falling back for unknown values."""
  try:
    return _actions.ViewMode(settings.get('view_mode', 'default'))
  except ValueError:
    return _actions.ViewMode.DEFAULT


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
  scrollback_limit: int = 1000,
  flush_scrollback: bool = False,
  selected_session: str | None = None,
  startup: Literal['new', 'latest', 'all'] = 'new',
  remote_endpoint: str | None = None,
  remote_secure: bool = True,
  show_remote_discovery_errors: bool = True,
  trust_extensions: bool = False,
  collect_analytics: bool = True,
  metadata: _daemon_client.Client | None = None,
) -> None:
  """Run one daemon frontend at a time; switching never stops shared workers."""
  connection: _frontend.Connection | None = None
  views: dict[str, _workspace_controller.ViewState] = {}
  try:
    while True:
      scope = connection.id if connection else 'local'
      values = (
        worker_config
        if connection is None
        else {
          'daemon_address': connection.address,
          'daemon_host': connection.host,
          'daemon_scope': connection.id,
          'daemon_instance_id': connection.daemon_id,
        }
      )
      change = await _run_tui(
        worker_command,
        values,
        connection=connection,
        view_state=views.setdefault(scope, _workspace_controller.ViewState()),
        reasoning_effort=reasoning_effort,
        scrollback_limit=scrollback_limit,
        flush_scrollback=flush_scrollback,
        selected_session=selected_session,
        startup=startup,
        remote_endpoint=remote_endpoint if connection is None else None,
        remote_secure=remote_secure,
        show_remote_discovery_errors=show_remote_discovery_errors,
        trust_extensions=trust_extensions,
        collect_analytics=collect_analytics and connection is None,
        metadata=_daemon_client.Client(
          connection.address,
          remote_host=connection.host,
          daemon_id=connection.daemon_id,
        )
        if connection
        else metadata,
      )
      if change is None:
        return
      previous, connection = connection, change.connection
      if previous:
        await previous.close()
      metadata = None
      selected_session = None
      startup = 'all'
  finally:
    if connection:
      await connection.close()


async def _run_tui(
  worker_command: Sequence[str],
  worker_config: Mapping[str, object],
  *,
  reasoning_effort: str = 'max',
  scrollback_limit: int = 1000,
  flush_scrollback: bool = False,
  selected_session: str | None = None,
  startup: Literal['new', 'latest', 'all'] = 'new',
  remote_endpoint: str | None = None,
  remote_secure: bool = True,
  show_remote_discovery_errors: bool = True,
  trust_extensions: bool = False,
  collect_analytics: bool = True,
  metadata: _daemon_client.Client | None = None,
  connection: _frontend.Connection | None = None,
  view_state: _workspace_controller.ViewState | None = None,
) -> _frontend.Change | None:
  """Open a new session, resume the latest, or restore the saved tabs."""
  change: _frontend.Change | None = None
  session_id = uuid.uuid4().hex
  with _logging.capture_output(selected_session or session_id) as terminal:
    extension_registry = _extension.ExtensionRegistry()
    await extension_registry.load(
      pathlib.Path(__file__).with_name('extensions') / 'ssh'
    )
    settings = _config.Config(pathlib.Path.home() / '.mirror' / 'config.json')
    analytics_directory = pathlib.Path.home() / '.mirror' / 'analytics'
    # Without a valid install ID, this launch sends no analytics.
    analytics_config: _statsig_config.StatsigConfig | None = None
    initial_settings: dict[str, object] = {}
    try:
      initial_settings = settings.read()
      install_id = initial_settings.get('install_id')
      if collect_analytics and isinstance(install_id, str) and install_id:
        analytics_config = _statsig_config.StatsigConfig(
          user_id=f'install-{install_id}',
          session_id=session_id,
          unsent_events_directory=analytics_directory,
        )
    except (OSError, ValueError):
      pass
    selected_theme = _themes.saved(initial_settings)
    selected_view_mode = _saved_view_mode(initial_settings)
    _statsig_client.initialize(analytics_config)
    tool_approvals = _tool_approvals.ToolApprovals()
    # One table for every workspace: a draft sent from another one keeps its images.
    images = _images.Attachments()
    checkpoints_enabled = _config.checkpoints_enabled(settings.read())
    user_extensions = _user_extensions.UserExtensions(
      extension_registry,
      pathlib.Path.home() / '.mirror' / 'extensions',
      cwd=pathlib.Path.cwd(),
      bundled=pathlib.Path(__file__).with_name('extensions'),
      settings=settings,
      bundled_only=bool(worker_config.get('daemon_host')),
    )
    first_agent_render = asyncio.Event()
    allow_extensions = asyncio.Event()
    initial_agent: _Repl | None = None

    async def load_extensions() -> None:
      await allow_extensions.wait()
      await user_extensions.start(trust_extensions=trust_extensions)

    extensions_loaded = asyncio.create_task(load_extensions())

    async def prepare_agent() -> None:
      await first_agent_render.wait()
      if initial_agent is None or not await initial_agent.ensure_api_key():
        return
      notice = _onboarding.take_analytics_notice()
      if notice is not None:
        initial_agent.dispatch(_actions.ShowText(notice))
      allow_extensions.set()
      await extensions_loaded

    agent_ready = asyncio.create_task(prepare_agent())

    workspace_effects: asyncio.Queue[tuple[_Repl, _effects.Effect]] = (
      asyncio.Queue()
    )

    def change_view_mode(mode: _actions.ViewMode) -> None:
      nonlocal selected_view_mode
      if mode is selected_view_mode:
        return
      selected_view_mode = mode
      extension_registry.dispatch(_actions.SetViewMode(mode))
      settings.update(lambda values: values.update(view_mode=mode.value))

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
        speed=spec.speed,
        workspace_extensions=()
        if worker_config.get('daemon_host')
        else spec.extensions,
        workspace_tools=None
        if worker_config.get('daemon_host')
        else spec.tools,
        metadata=metadata_client,
        scrollback_limit=scrollback_limit,
        flush_scrollback=flush_scrollback,
        workspace_effects=workspace_effects,
        workspace_controller=controller,
        extension_registry=extension_registry,
        user_extensions=user_extensions,
        tool_approvals=tool_approvals,
        images=images,
        local_runtime=local_runtime,
        active=spec.active,
        exit_on_worker_failure=False,
        remote_endpoint=remote_endpoint if spec.remote else None,
        configured_remote_endpoint=remote_endpoint,
        remote_secure=remote_secure,
        first_render=first_agent_render,
        extensions_loaded=agent_ready,
        fullscreen=True,
        initial_theme=selected_theme,
        initial_view_mode=selected_view_mode,
        on_view_mode_changed=change_view_mode,
        frontend=frontend_context,
      )
      if initial_agent is None:
        initial_agent = agent
      return agent

    daemon_warnings: list[str] = []
    metadata_client: _daemon_client.Client
    if metadata is not None:
      metadata_client = metadata
    else:
      metadata_client = await _local_rpc_client.Transport(
        worker_command,
        worker_config,
        workspace=pathlib.Path.cwd(),
        log_dir=pathlib.Path.home() / '.mirror' / 'logs',
        on_warning=daemon_warnings.append,
      ).daemon_client()

    pending_connection = contextlib.AsyncExitStack()

    def switch(request: _frontend.Change) -> None:
      nonlocal change
      if change is not None or screen.is_done:
        raise RuntimeError('Frontend is already disconnecting.')
      screen.exit()
      change = request
      if request.connection is not None:
        pending_connection.push_async_callback(request.connection.close)

    local_stack = contextlib.AsyncExitStack()
    local_runtime = (
      None
      if worker_config.get('daemon_host')
      else await local_stack.enter_async_context(metadata_client.local_models())
    )
    frontend_context = _frontend.Context(
      connection=connection,
      metadata=metadata_client,
      terminal=terminal,
      switch=switch,
      commands=extension_registry.frontend_commands,
      statuses=extension_registry.frontend_statuses,
      local_runtime=local_runtime,
      local_selection=lambda: _selected_llama_server(workspaces),
    )
    extension_registry.bind_frontend(frontend_context)
    workspaces = _workspace_controller.WorkspaceController(
      worker_config,
      store=metadata_client,
      view_state=view_state,
      workspace=pathlib.Path('/')
      if worker_config.get('daemon_host')
      else pathlib.Path.cwd(),
      reasoning_effort=reasoning_effort,
      create_agent=create_agent,
      is_trusted_directory=_onboarding.is_trusted_directory,
      settings=settings,
      session_id=selected_session,
      startup=startup,
      display_unmanaged_worktrees=(
        initial_settings.get('display_unmanaged_worktrees') is True
      ),
      remote_enabled=remote_endpoint is not None,
      show_remote_discovery_errors=show_remote_discovery_errors,
      list_remote_sessions=(
        functools.partial(
          _remote_rpc_client.list_remote_sessions,
          remote_endpoint,
          daemon_address=metadata_client.local_address,
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

    if startup != 'all' and isinstance(
      workspaces.selected, _workspaces.EmptyWorkspace
    ):
      identity = await workspaces.create_workspace('normal')
      await workspaces.select_workspace(identity)

    for warning in daemon_warnings:
      workspaces.selected.show_error(warning)

    async def apply_config(updated: dict[str, object]) -> None:
      nonlocal selected_theme, selected_view_mode
      selected_theme = _themes.saved(updated)
      mode = _saved_view_mode(updated)
      if mode is not selected_view_mode:
        selected_view_mode = mode
        extension_registry.dispatch(_actions.SetViewMode(mode))
      if collect_analytics:
        await apply_analytics_config(updated, workspaces)
      workspaces.set_display_unmanaged_worktrees(
        updated.get('display_unmanaged_worktrees') is True
      )
      await asyncio.shield(extensions_loaded)
      await user_extensions.apply_settings()

    config_task = asyncio.create_task(settings.watch(apply_config))

    screen = _workspace_application.WorkspaceApplication(
      full_screen=True,
      color_depth=_themes.prompt_color_depth(),
      min_redraw_interval=1 / _MAX_REDRAW_FPS,
      refresh_interval=0.1,
      mouse_support=True,
    )

    def update_title(
      _: _application.Application[str | None] | None = None,
    ) -> None:
      if initial_settings.get('terminal_title') is not False:
        selected = workspaces.selected
        screen.set_workspace_title(
          selected.label,
          status=selected.state,
          tag=workspaces.tab_tags().get(selected.identity),
        )

    def on_render(_: _application.Application[str | None]) -> None:
      # Pending views must wait for the agent's composer before onboarding.
      if initial_agent is None and isinstance(
        workspaces.selected, _workspaces.EmptyWorkspace
      ):
        allow_extensions.set()

    screen.after_render += on_render
    screen.before_render += update_title
    row_renderer.install(screen)
    run_selected = functools.partial(
      run_workspaces,
      workspaces,
      functools.partial(
        _activate_workspace,
        workspaces,
        screen=screen,
        theme_name=lambda: selected_theme,
        extension_registry=extension_registry,
        frontend_context=frontend_context,
        extension_lifecycle=user_extensions,
      ),
      workspace_effects,
      workspace_changed=update_title,
    )
    async with pending_connection, local_stack:
      try:
        await screen.run_views(run_selected)
      finally:
        config_task.cancel()
        agent_ready.cancel()
        extensions_loaded.cancel()
        await asyncio.gather(
          config_task, agent_ready, extensions_loaded, return_exceptions=True
        )
        await _statsig_client.get().disable()
        try:
          await extension_registry.close()
        finally:
          _console.Console(
            file=terminal,
            theme=_themes.rich_theme(selected_theme),
          ).print(
            'Disconnected.'
            if change
            else 'Bye. Resume with mirror --continue.',
            style='ui.muted',
          )
      pending_connection.pop_all()
      return change
