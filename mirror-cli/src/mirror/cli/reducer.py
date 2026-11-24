"""Pure state transitions for the Mirror TUI."""

from __future__ import annotations

import dataclasses
import shlex

from mirror import workspace
from mirror.cli import approval_modes
from mirror.cli import actions as _actions
from mirror.cli import (
  bash_approvals,
  commands,
  effects,
  quotes,
  rendering,
  selection,
  tool_approvals,
)
from mirror.cli import state as _state
from mirror.cli.composer import skill_completion


def _with_activity(
  state: _state.State, activity: _state.ActivityState
) -> _state.State:
  return dataclasses.replace(state, activity=activity)


def _with_status_bar(
  state: _state.State, status_bar: _state.StatusBarState
) -> _state.State:
  return dataclasses.replace(state, status_bar=status_bar)


def _clear_composer(state: _state.State) -> _state.State:
  """Clear the composer and its completion state."""
  return dataclasses.replace(
    state,
    composer=_actions.ComposerState(),
    completion=_state.CompletionState(),
  )


def _add_quote(
  state: _state.State, text: str
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  text = quotes.normalize(text)
  if not text:
    return state, ()
  quote, table = quotes.add(text, state.quotes)
  before = state.composer.text[: state.composer.cursor]
  after = state.composer.text[state.composer.cursor :]
  lead = ' ' if before and not before[-1].isspace() else ''
  trail = '' if after[:1].isspace() else ' '
  return dataclasses.replace(state, quotes=table), (
    effects.InsertComposerText(f'{lead}{quote.label}{trail}'),
  )


def _delete_quote_chip(
  state: _state.State,
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  chip = quotes.chip_before(
    state.composer.text, state.composer.cursor, state.quotes
  )
  if chip is None:
    return state, ()
  cursor = state.composer.cursor - len(chip)
  text = (
    state.composer.text[:cursor] + state.composer.text[cursor + len(chip) :]
  )
  return dataclasses.replace(
    state, composer=_actions.ComposerState(text, cursor)
  ), (effects.SetComposerText(text, cursor),)


def _dismiss_completion(state: _state.State) -> _state.State:
  """Dismiss completion when a modal overlay takes focus."""
  return dataclasses.replace(
    state,
    completion=dataclasses.replace(
      state.completion,
      slash=_state.MenuState(),
      skill=_state.MenuState(),
    ),
  )


def _restore_draft(
  state: _state.State, instruction: str, *, remove_queued: bool = True
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  queued = list(state.turn.queued)
  if remove_queued and instruction in queued:
    queued.remove(instruction)
  current = state.composer.text
  text = (
    current
    if current == instruction or current.startswith(f'{instruction}\n')
    else f'{instruction}\n{current}'
    if current
    else instruction
  )
  turn = dataclasses.replace(state.turn, queued=tuple(queued))
  composer = _actions.ComposerState(text, len(text))
  next_state = dataclasses.replace(state, turn=turn, composer=composer)
  if composer == state.composer:
    return next_state, ()
  return next_state, (effects.SetComposerText(text),)


def _settled_busy(turn: _state.TurnState) -> bool:
  return bool(turn.queued) or turn.agent_turn


def _unsaved_instructions(turn: _state.TurnState) -> tuple[str, ...]:
  tracked = turn.pending_instructions
  remaining = list(tracked)
  untracked: list[str] = []
  for instruction in turn.queued:
    if instruction in remaining:
      remaining.remove(instruction)
    else:
      untracked.append(instruction)
  return (*tracked, *untracked)


def _submit_instruction(
  state: _state.State, instruction: str
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  if state.lifecycle.disconnect_reason is not None:
    return _restore_draft(state, instruction)
  if state.activity.busy:
    turn = dataclasses.replace(
      state.turn,
      queued=(*state.turn.queued, instruction),
    )
    return dataclasses.replace(state, turn=turn), (
      effects.SendQueuedInstruction(instruction),
    )
  return state, (effects.RunTurn(instruction),)


def _stage_rewind(
  state: _state.State,
  target: _actions.UserTurn,
  discards: int,
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  rewind = _state.RewindState(staged=_state.StagedRewind(target, discards))
  text, table = quotes.display(target.message, state.quotes)
  composer = _actions.ComposerState(text, len(text))
  note = (
    f'↩ Rewound past {discards} turn(s), still saved above · '
    'send to fork from here, Esc to undo'
  )
  return dataclasses.replace(
    state,
    rewind=rewind,
    composer=composer,
    completion=_state.CompletionState(),
    quotes=table,
  ), (effects.SetComposerText(text), effects.RerenderHistory(note))


def _sync_approval_selector(
  state: _state.State,
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  """Present the queue head when no other modal owns the shared selector."""
  request = state.approval.request
  selector_id = f'approval:{request.call_id}' if request is not None else None
  if state.selector.id is not None and state.selector.id.startswith(
    'approval:'
  ):
    if state.selector.id == selector_id:
      return state, ()
    state = dataclasses.replace(state, selector=_state.SelectorState())
  if (
    request is None
    or state.selector.active
    or state.secret.active
    or state.resume.active
    or state.rewind.active
    or state.drop_confirmation is not None
    or state.lifecycle.quitting
  ):
    return state, ()
  choices = (selection.Choice('approve', 'Approve'),)
  if request.remember is not None:
    label, detail = 'Remember', 'This exact call'
    if isinstance(request.remember, tool_approvals.ArgvPrefixApproval):
      label = 'Always allow'
      detail = f'Prefix: {shlex.join(request.remember.argv)} · this workspace'
      if (
        len(detail) > 120
        and request.remember
        in bash_approvals.prefixes(request.args, request.remember.scope)[:1]
      ):
        detail = 'This command + literal args · this workspace'
    choices += (selection.Choice('remember', label, detail=detail),)
  choices += (selection.Choice('reject', 'Reject'),)
  name = (
    rendering
    .strip_ansi(request.name)
    .removeprefix('mcp__')
    .replace('__', ' · ')
  )
  return _reduce(
    state,
    _actions.OpenSelector(
      selection.Selector(
        id=f'approval:{request.call_id}',
        title=f'Approve {name}?',
        header=rendering.approval_body(request.args, tool_name=request.name),
        choices=choices,
        on_select=lambda value: _actions.DecideApproval(
          value in ('approve', 'remember'),
          remember=value == 'remember',
          call_id=request.call_id,
        ),
        numbered=True,
        keep_open=True,
        cancel_action=_actions.InterruptRequested(),
      )
    ),
  )


def _sync_work_timer(
  previous: _state.State, updated: _state.State, observed_at: float
) -> _state.State:
  """Exclude approval waits from elapsed work without ending the active turn."""
  started, paused = updated.activity.work_started, updated.activity.work_paused
  if started is None:
    paused = None
  elif updated.approval.request is not None:
    if paused is None or started != previous.activity.work_started:
      paused = max(started, observed_at)
  elif paused is not None:
    started += max(0, observed_at - paused)
    paused = None
  if (started, paused) == (
    updated.activity.work_started,
    updated.activity.work_paused,
  ):
    return updated
  return _with_activity(
    updated,
    dataclasses.replace(
      updated.activity, work_started=started, work_paused=paused
    ),
  )


def reduce(
  state: _state.State, action: _actions.Action, *, observed_at: float = 0.0
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  """Reduce actions and declare synchronization of the generic selector input."""
  updated, emitted = _reduce(state, action)
  updated = _sync_work_timer(state, updated, observed_at)
  updated, approval_effects = _sync_approval_selector(updated)
  emitted = (*emitted, *approval_effects)
  if (state.selector.id, state.selector.query, state.selector.numbered) != (
    updated.selector.id,
    updated.selector.query,
    updated.selector.numbered,
  ):
    emitted = (
      *emitted,
      effects.SyncSelectorInput(
        updated.selector.active, updated.selector.query
      ),
    )
  return updated, emitted


def _reduce(
  state: _state.State, action: _actions.Action
) -> tuple[_state.State, tuple[effects.Effect, ...]]:
  match action:
    case _actions.RunFrontendCommand(name, arguments):
      return state, (effects.RunFrontendCommand(name, arguments),)
    case _actions.ComposerSubmit():
      return _reduce(
        state,
        _actions.SubmitRequested(
          state.completion.skill.selected_item(state.completion.skills),
          state.completion.slash.selected_item(state.completion.commands),
        ),
      )
    case _actions.ComposerNewline():
      return state, (effects.InsertComposerNewline(),)
    case _actions.ComposerEditExternally():
      return state, (effects.EditComposerExternally(),)
    case _actions.QuoteSelectionRequested():
      return state, (effects.QuoteSelection(),)
    case _actions.QuoteAdded(text):
      return _add_quote(state, text)
    case _actions.DeleteQuoteChip():
      return _delete_quote_chip(state)
    case _actions.QuotesRecalled(table):
      return dataclasses.replace(state, quotes=table), ()
    case _actions.ComposerUndo():
      return state, (effects.UndoComposer(),)
    case _actions.ComposerRedo():
      return state, (effects.RedoComposer(),)
    case _actions.ComposerPasteImage():
      return state, (effects.PasteClipboardImage(),)
    case _actions.MoveSelection(delta):
      if state.selector.active:
        return _reduce(state, _actions.MoveSelectorSelection(delta))
      return _reduce(
        state,
        _actions.MoveCompletionRequested(
          delta,
          len(state.completion.skills),
          len(state.completion.commands),
        ),
      )
    case _actions.AdjustSelection(delta):
      selector = state.selector.selector
      if (
        state.selector.active
        and selector is not None
        and selector.on_adjust is not None
      ):
        return state, (effects.AdjustSelection(delta),)
      return state, ()
    case _actions.AcceptSelection(index):
      if index is not None:
        selector = state.selector.selector
        if (
          not state.selector.numbered
          or selector is None
          or not selector.numbered_from
          <= index
          < min(
            selector.numbered_from + 9, len(state.selector.filtered_choices())
          )
        ):
          return state, ()
        state = dataclasses.replace(
          state,
          selector=dataclasses.replace(
            state.selector,
            menu=dataclasses.replace(state.selector.menu, selected=index),
          ),
        )
      if state.selector.active:
        return state, (effects.AcceptSelection(),)
      return _reduce(
        state,
        _actions.CompleteSelectionRequested(
          state.completion.skill.selected_item(state.completion.skills),
          state.completion.slash.selected_item(state.completion.commands),
        ),
      )
    case _actions.CloseSelector():
      if state.selector.active:
        if (
          state.selector.selector is not None
          and state.selector.selector.cancel_action is not None
        ):
          return state, (effects.CancelSelection(),)
        return _reduce(state, _actions.ClearSelector())
      if state.completion.active:
        edit = skill_completion.SkillCompletion(state.completion.skills).cancel(
          state.composer.text,
          state.composer.cursor,
        )
        return _reduce(
          state,
          _actions.DismissCompletionRequested(
            None
            if edit is None
            else _actions.ComposerState(edit.text, edit.cursor),
          ),
        )
      if not state.resume.active:
        if state.rewind.active:
          return _reduce(state, _actions.CloseRewindOverlay())
        if state.rewind.staged is not None:
          return _reduce(state, _actions.CancelRewind())
      return state, ()
    case _actions.CycleResumeScope():
      return _reduce(
        state,
        _actions.SetResumeScope(
          workspace.next_session_scope(state.resume.scope)
        ),
      )
    case (
      _actions.RoutedExtensionAction()
      | _actions.ExtensionConfigChanged()
      | _actions.HydrateExtensions()
    ):
      return state, ()
    case _actions.ExtensionLoaded():
      return state, (
        effects.RefreshCompletion(),
      ) if state.composer.text else ()
    case _actions.ExtensionUnloaded(extension_id):
      selector = state.selector.selector
      if selector is not None and selector.extension_id == extension_id:
        state = dataclasses.replace(state, selector=_state.SelectorState())
      return state, (
        effects.RefreshCompletion(),
      ) if state.composer.text else ()
    case _actions.ExtensionConsentResolved(key, revision):
      selector = state.selector.selector
      if selector is not None and isinstance(
        selector.cancel_action, _actions.ExtensionConsentBatch
      ):
        remaining = tuple(
          request
          for request in selector.cancel_action.requests
          if (request.key, request.revision) != (key, revision)
        )
        if not remaining:
          return _reduce(state, _actions.ClearSelector())
        return dataclasses.replace(
          state,
          selector=dataclasses.replace(
            state.selector,
            selector=dataclasses.replace(
              selector, cancel_action=_actions.ExtensionConsentBatch(remaining)
            ),
          ),
        ), ()
      return state, ()
    case _actions.ShowExtensions():
      return state, (effects.ShowExtensions(),)
    case (
      _actions.ToggleExtension()
      | _actions.ExtensionConsentBatch()
      | _actions.AddExtension()
      | _actions.UpdateExtensions()
      | _actions.RemoveExtension()
    ):
      return state, (effects.ChangeExtension(action),)
    case _actions.ShowText(text, style):
      return state, (effects.EmitText(text, style),)
    case _actions.QuitApp():
      return state, (effects.ExitApp(),)
    case _actions.RestartDaemon(confirmed):
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      if confirmed:
        return state, (effects.RestartDaemon(),)
      return _reduce(
        state,
        _actions.OpenSelector(
          selection.Selector(
            id='daemon-restart',
            title='Restart the shared daemon?',
            header=(
              'Interrupt local workers in every CLI and Electron window.\n'
              'Local turns stay paused; pending approvals are canceled.\n'
              'Unfinished tool outcomes may be unknown. Tabs and drafts are kept.\n'
              'Remote agents keep running.'
            ),
            choices=(
              selection.Choice('cancel', 'Cancel'),
              selection.Choice('restart', 'Interrupt workers and restart'),
            ),
            selected='cancel',
            numbered=True,
            on_select=lambda value: (
              _actions.RestartDaemon(confirmed=True)
              if value == 'restart'
              else _actions.CloseSelector()
            ),
          )
        ),
      )
    case _actions.NewSession():
      if state.session.read_only or state.activity.busy or state.tasks.tasks:
        return state, (
          effects.EmitError(
            'Finish the current turn and tasks before clearing this session.'
          ),
        )
      return state, (effects.StartNewConversation(),)
    case _actions.WorkspaceTagCommand(args):
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.WorkspaceTagCommand(args),)
    case _actions.RenameWorktree(alias):
      if not state.supervised:
        return state, (
          effects.EmitError('worktree supervision is unavailable'),
        )
      return state, (effects.RenameWorktree(alias),)
    case _actions.CompactConversation():
      return state, (effects.CompactConversation(),)
    case _actions.RewindSession():
      return state, (effects.LoadRewindTurns(state.session.conversation_id),)
    case _actions.CancelModelSelection():
      return state, (effects.CancelModelSelection(),)
    case _actions.ForgetReflectionKey():
      return state, (effects.ForgetReflectionKey(),)
    case _actions.SetWebProvider(provider):
      if state.session.remote:
        return state, (
          effects.EmitError('web provider is configured by the server'),
        )
      return state, (effects.SetWebProvider(provider),)
    case _actions.SetModel(model_id):
      if state.session.remote:
        return state, (
          effects.EmitError('remote model is configured by the server'),
        )
      return state, (effects.ResolveModel(model_id),)
    case _actions.SetRemoteModel(model_id, api):
      if not state.session.remote:
        return state, (
          effects.EmitError('server models are picked from a remote tab'),
        )
      return state, (effects.SwitchRemoteModel(model_id, api),)
    case _actions.SetSpeed(speed):
      status_bar = dataclasses.replace(state.status_bar, speed=speed)
      if speed == 'fast':
        message = (
          'Fast mode requests faster processing; it may cost more or use '
          'limits faster. Availability depends on your model and account.'
        )
      else:
        message = 'Model speed set to normal (provider default processing).'
      return _with_status_bar(state, status_bar), (
        effects.SendInferenceConfiguration(),
        effects.EmitText(
          message, style='ui.warning' if speed == 'fast' else ''
        ),
        effects.SaveWorktreePreferences(
          state.status_bar.reasoning_effort,
          state.status_bar.approval_mode,
          speed,
        ),
      )
    case _actions.SetReasoning(effort):
      status_bar = dataclasses.replace(
        state.status_bar,
        reasoning_effort=effort,
      )
      return _with_status_bar(state, status_bar), (
        effects.SendInferenceConfiguration(),
        effects.SaveWorktreePreferences(
          effort, state.status_bar.approval_mode, state.status_bar.speed
        ),
      )
    case _actions.SelectPermissions():
      return _reduce(
        state,
        _actions.OpenSelector(
          selection.Selector(
            id='permissions',
            title='Tool permissions',
            choices=(
              selection.Choice('auto', 'Automatic approval'),
              selection.Choice('approve', 'Manual approval'),
              selection.Choice('bypass', 'Bypass approval'),
            ),
            on_select=_actions.SetApprovalMode,
            current=state.status_bar.approval_mode,
            selected=state.status_bar.approval_mode,
          )
        ),
      )
    case _actions.RequestPromptHistory():
      opened, _ = _reduce(
        state,
        _actions.OpenSelector(
          selection.Selector(
            id='history',
            title='Prompt history',
            choices=(),
            on_select=_actions.RestorePrompt,
          )
        ),
      )
      return opened, (effects.LoadPromptHistory(),)
    case _actions.RestorePrompt(text, cursor):
      return dataclasses.replace(
        state,
        composer=_actions.ComposerState(
          text, len(text) if cursor is None else cursor
        ),
      ), (effects.SetComposerText(text, cursor),)
    case _actions.StageRewind(turn, discards):
      return _stage_rewind(state, turn, discards)
    case _actions.OpenFolder(path, confirmed, create_directory):
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.OpenFolder(path, confirmed, create_directory),)
    case _actions.OpenNewWorkspace() | _actions.CreateWorkspace():
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      if isinstance(action, _actions.CreateWorkspace):
        return state, (effects.CreateWorkspace(action.kind),)
      return state, (effects.OpenNewWorkspace(),)
    case _actions.CloseWorkspace():
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.CloseWorkspace(),)
    case _actions.ForkWorkspace(name):
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.ForkWorkspace(name),)
    case _actions.CreateWorktree(name):
      if not state.supervised:
        return state, (
          effects.EmitError('worktree supervision is unavailable'),
        )
      return state, (effects.CreateManagedWorktree(name),)
    case _actions.CreateSandbox(name):
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.CreateSandboxWorkspace(name),)
    case _actions.LoadRemoteSessions():
      return state, (effects.LoadRemoteSessions(),)
    case _actions.CreateRemote(title, remote_workspace, escalated):
      if not state.supervised:
        return state, (effects.EmitError('agent supervision is unavailable'),)
      return state, (
        effects.CreateRemoteAgent(title, remote_workspace, escalated),
      )
    case _actions.Drop():
      if not state.supervised:
        return state, (
          effects.EmitError('worktree supervision is unavailable'),
        )
      return state, (effects.DropWorkspace(),)
    case _actions.SetStatus(value):
      activity = dataclasses.replace(state.activity, status=value)
      return _with_activity(state, activity), ()
    case _actions.RecordResponseUsage(completion_tokens):
      turn = dataclasses.replace(
        state.turn, output_tokens=state.turn.output_tokens + completion_tokens
      )
      return dataclasses.replace(state, turn=turn), ()
    case _actions.ObserveStreamText(content, reasoning, refusal):
      preview = state.stream.preview or _state.StreamPreview()
      preview = dataclasses.replace(
        preview,
        content=preview.content + content,
        reasoning=preview.reasoning + reasoning,
        refusal=preview.refusal + refusal,
      )
      stream = dataclasses.replace(state.stream, preview=preview)
      return dataclasses.replace(state, stream=stream), ()
    case _actions.CommitStreamReasoning():
      preview = state.stream.preview
      if (
        preview is None
        or not preview.reasoning
        or preview.committed_reasoning is not None
      ):
        return state, ()
      preview = dataclasses.replace(
        preview,
        committed_reasoning=preview.reasoning,
        reasoning='',
      )
      stream = dataclasses.replace(state.stream, preview=preview)
      return dataclasses.replace(state, stream=stream), ()
    case _actions.CommitStreamContent():
      preview = state.stream.preview
      if (
        preview is None
        or not preview.content
        or preview.committed_content is not None
      ):
        return state, ()
      preview = dataclasses.replace(
        preview,
        committed_content=preview.content,
        content='',
      )
      stream = dataclasses.replace(state.stream, preview=preview)
      return dataclasses.replace(state, stream=stream), ()
    case _actions.ObserveStreamToolCall(call, pending):
      tool_calls = tuple(
        call if item.index == call.index else item
        for item in state.stream.tool_calls
      )
      if not any(item.index == call.index for item in tool_calls):
        tool_calls = (*tool_calls, call)
      pending_calls = state.stream.pending
      if pending is not None:
        pending_calls = tuple(
          pending if item.index == pending.index else item
          for item in pending_calls
        )
        if not any(item.index == pending.index for item in pending_calls):
          pending_calls = (*pending_calls, pending)
      stream = dataclasses.replace(
        state.stream,
        tool_calls=tool_calls,
        pending=pending_calls,
      )
      return dataclasses.replace(state, stream=stream), ()
    case _actions.ClearStreamPending():
      stream = dataclasses.replace(state.stream, pending=())
      return dataclasses.replace(state, stream=stream), ()
    case _actions.CloseStream():
      stream = dataclasses.replace(
        state.stream,
        preview=None,
        tool_calls=(),
        pending=(),
      )
      return dataclasses.replace(state, stream=stream), ()
    case _actions.StartRunningToolCalls(calls):
      stream = dataclasses.replace(
        state.stream, running=(*state.stream.running, *calls)
      )
      return dataclasses.replace(state, stream=stream), ()
    case _actions.FinishRunningToolCall(call_id):
      stream = dataclasses.replace(
        state.stream,
        running=tuple(
          call for call in state.stream.running if call.call_id != call_id
        ),
      )
      return dataclasses.replace(state, stream=stream), ()
    case _actions.ClearRunningToolCalls():
      stream = dataclasses.replace(state.stream, running=())
      return dataclasses.replace(state, stream=stream), ()
    case _actions.WorktreeDropConfirmationRequested(confirmation):
      return _reduce(
        dataclasses.replace(state, drop_confirmation=confirmation),
        _actions.OpenSelector(
          selection.Selector(
            id='worktree-drop',
            title='Delete workspace?',
            header=(
              f'{confirmation.label}\n{confirmation.path}\n'
              'The workspace and its contents will be deleted, including uncommitted changes.'
            ),
            choices=(
              selection.Choice('cancel', 'Cancel'),
              selection.Choice('delete', 'Delete'),
              selection.Choice('delete-always', "Delete and don't ask again"),
            ),
            on_select=lambda value: (
              _actions.ConfirmWorktreeDrop(
                skip_confirmation=value == 'delete-always'
              )
              if value in ('delete', 'delete-always')
              else _actions.ClearWorktreeDrop()
            ),
            selected='cancel',
            numbered=True,
            cancel_action=_actions.ClearWorktreeDrop(),
          )
        ),
      )
    case _actions.ClearWorktreeDrop():
      return dataclasses.replace(
        state,
        drop_confirmation=None,
        selector=_state.SelectorState()
        if state.selector.id == 'worktree-drop'
        else state.selector,
      ), ()
    case _actions.ConfirmWorktreeDrop(skip_confirmation):
      if state.drop_confirmation is None:
        return state, ()
      cleared, _ = _reduce(state, _actions.ClearWorktreeDrop())
      return cleared, (
        effects.DropWorkspace(
          confirmed=True, skip_confirmation=skip_confirmation
        ),
      )
    case _actions.TurnStarted(observed_at, spinner_color):
      state = dataclasses.replace(
        state, turn=dataclasses.replace(state.turn, output_tokens=0)
      )
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status='thinking…',
        work_started=observed_at,
        completed_duration=None,
        spinner_color=spinner_color,
      )
      return _with_activity(state, activity), ()
    case _actions.AgentTurnStarted(observed_at):
      turn = dataclasses.replace(state.turn, agent_turn=True)
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        completed_duration=None,
        status=state.activity.status
        if state.activity.busy and state.activity.status
        else 'thinking…',
        work_started=(
          state.activity.work_started
          if state.activity.work_started is not None
          else observed_at
        ),
      )
      return dataclasses.replace(state, turn=turn, activity=activity), ()
    case _actions.WorkerLoadingStarted():
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status='restoring conversation…'
        if state.session.restore_pending
        else 'loading…',
        completed_duration=None,
      )
      return _with_activity(state, activity), ()
    case _actions.WorkerLoadingFinished():
      busy = _settled_busy(state.turn)
      activity = dataclasses.replace(
        state.activity,
        busy=busy,
        status='thinking…' if busy else '',
      )
      return _with_activity(state, activity), ()
    case _actions.WorkerRetryStarted():
      activity = dataclasses.replace(state.activity, busy=True)
      return _with_activity(state, activity), ()
    case _actions.WorkerRetryFinished():
      activity = dataclasses.replace(
        state.activity,
        busy=_settled_busy(state.turn),
      )
      return _with_activity(state, activity), ()
    case _actions.WorkerRecoveryFinished(replaying):
      session = dataclasses.replace(state.session, replaying=replaying)
      activity = dataclasses.replace(
        state.activity,
        busy=_settled_busy(state.turn) or replaying,
      )
      return dataclasses.replace(
        state,
        session=session,
        activity=activity,
      ), ()
    case _actions.InputSubmitted(message_id, instruction):
      if message_id in state.turn.input_order:
        return state, ()
      submitted = (*state.turn.submitted_inputs, (message_id, instruction))
      turn = dataclasses.replace(
        state.turn,
        submitted_inputs=submitted,
        input_order=(*state.turn.input_order, message_id),
      )
      return dataclasses.replace(state, turn=turn), ()
    case _actions.InputAcknowledged(message_id):
      acknowledged = tuple(
        item for item in state.turn.submitted_inputs if item[0] == message_id
      )
      submitted = tuple(
        item for item in state.turn.submitted_inputs if item[0] != message_id
      )
      turn = dataclasses.replace(
        state.turn,
        submitted_inputs=submitted,
        awaiting_inputs=(*state.turn.awaiting_inputs, *acknowledged),
      )
      return dataclasses.replace(state, turn=turn), ()
    case _actions.UnsentInputsDiscarded():
      turn = dataclasses.replace(
        state.turn,
        submitted_inputs=(),
        input_order=tuple(key for key, _ in state.turn.awaiting_inputs),
      )
      updated = dataclasses.replace(state, turn=turn)
      restore_effects: list[effects.Effect] = []
      for _, instruction in reversed(state.turn.submitted_inputs):
        updated, emitted = _restore_draft(updated, instruction)
        restore_effects.extend(emitted)
      return updated, (*restore_effects,)
    case _actions.InputRejected(message_id):
      instruction = next(
        (
          instruction
          for submitted_id, instruction in (
            *state.turn.submitted_inputs,
            *state.turn.awaiting_inputs,
          )
          if submitted_id == message_id
        ),
        None,
      )
      if instruction is None:
        return state, ()
      turn = dataclasses.replace(
        state.turn,
        submitted_inputs=tuple(
          item for item in state.turn.submitted_inputs if item[0] != message_id
        ),
        awaiting_inputs=tuple(
          item for item in state.turn.awaiting_inputs if item[0] != message_id
        ),
        compaction_started=None,
        input_order=tuple(
          key for key in state.turn.input_order if key != message_id
        ),
      )
      restored, draft_effects = _restore_draft(
        dataclasses.replace(state, turn=turn),
        instruction,
        remove_queued=state.turn.is_queued_input(message_id),
      )
      activity = dataclasses.replace(
        restored.activity,
        busy=_settled_busy(restored.turn),
      )
      return dataclasses.replace(restored, activity=activity), (*draft_effects,)
    case _actions.RestoreDraft(instruction):
      return _restore_draft(state, instruction)
    case _actions.CommandFailed(message):
      return state, (effects.EmitError(message),)
    case _actions.SupervisorError(message):
      return state, (effects.RecordError(message),)
    case _actions.SetPendingSwitch(value):
      session = dataclasses.replace(state.session, pending_switch=value)
      return dataclasses.replace(state, session=session), ()
    case _actions.ToggleApprovalMode():
      mode = approval_modes.toggled(state.status_bar.approval_mode)
      status_bar = dataclasses.replace(state.status_bar, approval_mode=mode)
      return _with_status_bar(state, status_bar), (
        effects.SetWorkerApprovalMode(mode),
        effects.SaveWorktreePreferences(
          state.status_bar.reasoning_effort, mode, state.status_bar.speed
        ),
      )
    case _actions.SharedSettingsReceived(
      model, reasoning_effort, approval_mode, speed
    ):
      return _with_status_bar(
        state,
        dataclasses.replace(
          state.status_bar,
          model=model,
          reasoning_effort=reasoning_effort,
          approval_mode=approval_mode,
          speed=speed,
        ),
      ), ()
    case _actions.ModelSwitchFinished(model_id, reasoning_effort):
      # Fast processing is opt-in for each model selection.
      status_bar = dataclasses.replace(
        state.status_bar,
        model=model_id,
        speed='normal',
        reasoning_effort=reasoning_effort
        if reasoning_effort is not None
        else state.status_bar.reasoning_effort,
      )
      return _with_status_bar(state, status_bar), (
        effects.SendInferenceConfiguration(),
        effects.SaveLastModel(model_id),
        effects.SaveWorktreePreferences(
          status_bar.reasoning_effort, status_bar.approval_mode
        ),
      )
    case _actions.SetToolGroup(summary):
      return dataclasses.replace(state, tool_group=summary), ()
    case _actions.CycleViewMode() | _actions.SetViewMode():
      if isinstance(action, _actions.CycleViewMode):
        modes = tuple(_actions.ViewMode)
        index = modes.index(state.status_bar.view_mode)
        mode = modes[(index + 1) % len(modes)]
      else:
        mode = action.mode
      status_bar = dataclasses.replace(state.status_bar, view_mode=mode)
      updated = _with_status_bar(state, status_bar)
      if (
        state.session.replaying
        or state.session.rerendering
        or state.session.pending_switch is not None
      ):
        return updated, ()
      session = dataclasses.replace(state.session, rerendering=True)
      status_bar = dataclasses.replace(status_bar, resize_dirty=False)
      return dataclasses.replace(
        updated,
        session=session,
        status_bar=status_bar,
      ), (effects.RerenderHistory(),)
    case _actions.SetActive(value):
      lifecycle = dataclasses.replace(
        state.lifecycle,
        active=value,
        unread=False if value else state.lifecycle.unread,
      )
      return dataclasses.replace(state, lifecycle=lifecycle), ()
    case (
      _actions.CliStarted()
      | _actions.SessionCreated()
      | _actions.SessionAttached()
    ):
      return state, ()
    case _actions.WorkerReconnecting():
      lifecycle = dataclasses.replace(
        state.lifecycle,
        started=False,
        disconnect_reason=action.message,
        synchronizing=action.synchronizing,
        failure=None,
      )
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status='synchronizing…' if action.synchronizing else 'reconnecting…',
        foreground_tasks=frozenset(),
      )
      turn = dataclasses.replace(
        state.turn, agent_turn=False, interrupt_id=None, compaction_started=None
      )
      return dataclasses.replace(
        state,
        lifecycle=lifecycle,
        activity=activity,
        turn=turn,
        tasks=_state.TasksState(),
        approval=_state.ApprovalState(),
      ), ()
    case _actions.WorkerStarted():
      lifecycle = dataclasses.replace(
        state.lifecycle,
        started=True,
        failure=None,
        disconnect_reason=None,
        synchronizing=False,
      )
      return dataclasses.replace(state, lifecycle=lifecycle), ()
    case _actions.WorkerStopped():
      lifecycle = dataclasses.replace(state.lifecycle, started=False)
      turn = dataclasses.replace(state.turn, agent_turn=False)
      return dataclasses.replace(state, lifecycle=lifecycle, turn=turn), ()
    case _actions.WorkerClosed() | _actions.WorkerFailed():
      message = (
        action.message if isinstance(action, _actions.WorkerFailed) else None
      )
      lifecycle = dataclasses.replace(
        state.lifecycle,
        started=False,
        failure=message,
        synchronizing=False,
        disconnect_reason=message
        if isinstance(action, _actions.WorkerFailed) and action.disconnected
        else None,
      )
      activity = dataclasses.replace(state.activity, busy=False, status='')
      turn = dataclasses.replace(
        state.turn,
        submitted_inputs=(),
        awaiting_inputs=(),
        input_order=(),
        interrupt_id=None,
        agent_turn=False,
      )
      updated = dataclasses.replace(
        state,
        lifecycle=lifecycle,
        activity=activity,
        turn=turn,
        session=dataclasses.replace(state.session, replaying=False),
        selector=_state.SelectorState()
        if state.selector.id == 'history'
        else state.selector,
      )
      restore_effects: list[effects.Effect] = []
      for instruction in reversed(_unsaved_instructions(state.turn)):
        updated, emitted = _restore_draft(updated, instruction)
        restore_effects.extend(emitted)
      return updated, (*restore_effects,)
    case _actions.ExitRequested():
      return state, (effects.ExitApp(defer_until_running=True),)
    case _actions.SetQuitting():
      lifecycle = dataclasses.replace(state.lifecycle, quitting=True)
      return dataclasses.replace(state, lifecycle=lifecycle), ()
    case _actions.ApprovalRequested(request):
      if any(
        queued.call_id == request.call_id for queued in state.approval.requests
      ):
        return state, ()  # Announced again when the stream reattached.
      approval = dataclasses.replace(
        state.approval, requests=(*state.approval.requests, request)
      )
      return dataclasses.replace(state, approval=approval), ()
    case _actions.MoveApproval(delta):
      # Keep configured approval shortcuts on the shared selector.
      if state.selector.id is None or not state.selector.id.startswith(
        'approval:'
      ):
        return state, ()
      return _reduce(state, _actions.MoveSelectorSelection(delta))
    case _actions.ClearApproval(call_id):
      approval = _state.ApprovalState(
        requests=tuple(
          request
          for request in state.approval.requests
          if request.call_id != call_id
        ),
      )
      return dataclasses.replace(state, approval=approval), ()
    case _actions.DecideApproval(approved, remember, call_id):
      if state.lifecycle.disconnect_reason is not None:
        return state, ()
      request = state.approval.request
      if request is None or (
        call_id is not None and request.call_id != call_id
      ):
        return state, ()
      if approved is None:
        choice = state.selector.selected_choice()
        if state.selector.id != f'approval:{request.call_id}' or choice is None:
          return state, ()
        approved = choice.value in ('approve', 'remember')
        remember = remember or choice.value == 'remember'
      if remember and (not approved or request.remember is None):
        return state, ()
      approval = _state.ApprovalState(requests=state.approval.requests[1:])
      return dataclasses.replace(state, approval=approval), (
        effects.SendApprovalDecision(
          request.call_id,
          approved,
          request.remember if remember else None,
          reason=None if approved else 'Tool call rejected by the user.',
        ),
      )
    case _actions.MoveCompletionRequested(delta, skill_count, command_count):
      if skill_count:
        skill = state.completion.skill.move(delta, skill_count)
        completion = dataclasses.replace(state.completion, skill=skill)
        return dataclasses.replace(state, completion=completion), ()
      if command_count:
        slash = state.completion.slash.move(delta, command_count)
        completion = dataclasses.replace(state.completion, slash=slash)
        return dataclasses.replace(state, completion=completion), ()
      return state, ()
    case _actions.CompleteSelectionRequested(selected_skill, selected_command):
      if selected_skill is not None:
        edit = skill_completion.SkillCompletion(
          state.completion.skills
        ).complete(
          state.composer.text,
          state.composer.cursor,
          selected_skill,
        )
        if edit is None:
          return state, ()
        return dataclasses.replace(
          state,
          composer=_actions.ComposerState(edit.text, edit.cursor),
          completion=_state.CompletionState(),
        ), (effects.SetComposerText(edit.text, edit.cursor),)
      if selected_command is None:
        return state, ()
      text = f'/{selected_command.completion} '
      composer = _actions.ComposerState(text, len(text))
      return dataclasses.replace(
        state,
        composer=composer,
        completion=_state.CompletionState(),
      ), (effects.SetComposerText(text),)
    case _actions.DismissCompletionRequested(edit):
      if edit is None:
        return _clear_composer(state), (effects.ResetComposer(),)
      return dataclasses.replace(
        state,
        composer=edit,
        completion=_state.CompletionState(),
      ), (effects.SetComposerText(edit.text, edit.cursor),)
    case _actions.SubmitRequested(selected_skill, selected_command):
      if (
        state.lifecycle.disconnect_reason is not None
        and not state.composer.text.startswith('/')
      ):
        if state.lifecycle.failure is not None:
          return state, (effects.RetryWorker(),)
        return state, ()
      if state.resume.active:
        return _reduce(state, _actions.AcceptResumeConfirmation())
      if state.rewind.confirming:
        return _reduce(state, _actions.ConfirmRewind())
      if selected_skill is not None:
        return _reduce(
          state,
          _actions.CompleteSelectionRequested(selected_skill, None),
        )
      if selected_command is not None:
        submission = commands.selected_submission(
          selected_command, busy=state.activity.busy
        )
        if isinstance(submission, _actions.RestorePrompt):
          return _reduce(
            state,
            _actions.CompleteSelectionRequested(None, selected_command),
          )
        return _clear_composer(state), (
          effects.ResetComposer(),
          effects.RunSlashCommand(submission),
        )
      text, cursor = state.composer.text, state.composer.cursor
      if text[:cursor].endswith('\\'):
        text = text[: cursor - 1] + '\n' + text[cursor:]
        return dataclasses.replace(
          state, composer=_actions.ComposerState(text, cursor)
        ), (effects.SetComposerText(text, cursor),)
      instruction = text.strip()
      if not instruction:
        if state.lifecycle.failure is not None and not state.activity.busy:
          return state, (effects.RetryWorker(),)
        return state, ()
      if instruction.startswith('!'):
        if not state.lifecycle.started:
          return state, (
            effects.EmitText(
              'Wait for the workspace to finish loading before running Bash.',
              style='ui.warning',
            ),
          )
        return _clear_composer(state), (
          effects.ResetComposer(append_to_history=True),
          effects.RunShellCommand(instruction[1:].lstrip()),
        )
      if commands.looks_like_slash_command(state.composer.text):
        return _clear_composer(state), (
          effects.ResetComposer(),
          effects.RunSlashCommand(instruction),
        )
      if state.rewind.staged is not None:
        return _reduce(state, _actions.SetRewindConfirming(True))
      next_state, instruction_effects = _submit_instruction(
        _clear_composer(state), instruction
      )
      return next_state, (
        effects.ResetComposer(append_to_history=True),
        *instruction_effects,
        effects.UserPromptSubmitted(instruction),
      )
    case _actions.SubmitInstruction(instruction):
      return _submit_instruction(state, instruction)
    case _actions.UserMessageObserved(content, message_id, spinner_color):
      if not content.strip():
        return state, ()
      session = dataclasses.replace(state.session, has_user_message=True)
      queued = state.turn.queued
      activity = state.activity
      if content in queued and (
        not message_id or state.turn.is_queued_input(message_id)
      ):
        remaining = list(queued)
        remaining.remove(content)
        queued = tuple(remaining)
        activity = dataclasses.replace(activity, spinner_color=spinner_color)
      turn = dataclasses.replace(
        state.turn,
        queued=queued,
        output_tokens=0,
        input_order=tuple(
          key for key in state.turn.input_order if key != message_id
        ),
        submitted_inputs=tuple(
          item for item in state.turn.submitted_inputs if item[0] != message_id
        ),
        awaiting_inputs=tuple(
          item for item in state.turn.awaiting_inputs if item[0] != message_id
        ),
      )
      return dataclasses.replace(
        state, session=session, turn=turn, activity=activity
      ), ()
    case _actions.CompactionStarted(started):
      turn = dataclasses.replace(state.turn, compaction_started=started)
      return dataclasses.replace(state, turn=turn), ()
    case _actions.CompactionFinished():
      turn = dataclasses.replace(state.turn, compaction_started=None)
      return dataclasses.replace(state, turn=turn), ()
    case _actions.TurnFinished(duration):
      turn = dataclasses.replace(
        state.turn,
        compaction_started=None,
        interrupt_id=None,
        agent_turn=False,
        output_tokens=0,
      )
      lifecycle = dataclasses.replace(
        state.lifecycle,
        unread=state.lifecycle.unread or not state.lifecycle.active,
      )
      busy = bool(turn.queued)
      activity = dataclasses.replace(
        state.activity,
        busy=busy,
        status=('thinking…' if busy else '')
        if state.activity.status == 'interrupting…'
        else state.activity.status,
        work_started=None,
        completed_duration=(
          duration
          if duration is not None
          else state.activity.completed_duration
        ),
      )
      return dataclasses.replace(
        state,
        turn=turn,
        lifecycle=lifecycle,
        activity=activity,
      ), ()
    case _actions.SessionSwitchStarted(pending):
      session = dataclasses.replace(
        state.session,
        replaying=True,
        restore_pending=not pending.create,
        pending_switch=pending,
      )
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status='loading conversation…',
        completed_duration=None,
      )
      return dataclasses.replace(
        state,
        session=session,
        activity=activity,
        turn=dataclasses.replace(state.turn, agent_turn=False),
        rewind=_state.RewindState(),
      ), ()
    case _actions.SessionOpened(conversation_id, composer):
      session = dataclasses.replace(
        state.session,
        conversation_id=conversation_id,
        has_user_message=False,
        restore_pending=not state.session.pending_switch.create
        if state.session.pending_switch is not None
        else state.session.restore_pending,
      )
      activity = dataclasses.replace(
        state.activity,
        foreground_tasks=frozenset(),
        completed_duration=None,
      )
      updated = dataclasses.replace(
        state,
        session=session,
        turn=_state.TurnState(),
        activity=activity,
        tasks=_state.TasksState(),
      )
      if composer is not None:
        return dataclasses.replace(updated, composer=composer), (
          effects.SetComposerText(composer.text, composer.cursor),
        )
      if pending := _unsaved_instructions(state.turn):
        return _restore_draft(updated, '\n'.join(pending))
      return updated, ()
    case _actions.SetReplaying(value):
      session = dataclasses.replace(state.session, replaying=value)
      return dataclasses.replace(state, session=session), ()
    case _actions.ReplayFinished(has_user_message):
      session = dataclasses.replace(
        state.session,
        replaying=False,
        restore_pending=False,
        # Background replay has no repaint to finish the switch.
        pending_switch=(
          state.session.pending_switch if state.lifecycle.active else None
        ),
        has_user_message=(state.session.has_user_message or has_user_message),
      )
      turn = dataclasses.replace(state.turn, compaction_started=None)
      activity = state.activity
      if not state.lifecycle.active:
        activity = dataclasses.replace(activity, busy=_settled_busy(turn))
      return dataclasses.replace(
        state,
        session=session,
        turn=turn,
        activity=activity,
      ), ()
    case _actions.ClearScreen():
      if (
        state.activity.busy
        or state.session.replaying
        or state.session.rerendering
        or state.session.pending_switch is not None
      ):
        return state, ()
      session = dataclasses.replace(state.session, rerendering=True)
      status_bar = dataclasses.replace(state.status_bar, resize_dirty=False)
      return dataclasses.replace(
        state,
        session=session,
        status_bar=status_bar,
      ), (effects.RerenderHistory('Screen cleared', drop_scrollback=True),)
    case _actions.RerenderRequested(note, drop_scrollback):
      if (
        state.activity.busy
        or state.session.replaying
        or state.session.rerendering
        or state.session.pending_switch is not None
      ):
        return state, ()
      session = dataclasses.replace(state.session, rerendering=True)
      status_bar = dataclasses.replace(state.status_bar, resize_dirty=False)
      return dataclasses.replace(
        state,
        session=session,
        status_bar=status_bar,
      ), (effects.RerenderHistory(note, drop_scrollback),)
    case _actions.RerenderStarted():
      session = dataclasses.replace(state.session, rerendering=True)
      return dataclasses.replace(state, session=session), ()
    case _actions.RepaintFinished(reset_busy):
      session = dataclasses.replace(
        state.session,
        pending_switch=None,
        rerendering=False,
      )
      activity = state.activity
      if reset_busy:
        activity = dataclasses.replace(activity, busy=_settled_busy(state.turn))
      return dataclasses.replace(
        state,
        session=session,
        activity=activity,
      ), ()
    case _actions.HistoryRendered(width):
      session = dataclasses.replace(state.session, history_width=width)
      status_bar = dataclasses.replace(state.status_bar, resize_dirty=False)
      return dataclasses.replace(
        state,
        session=session,
        status_bar=status_bar,
      ), ()
    case _actions.TerminalResized(columns):
      session = dataclasses.replace(
        state.session,
        resize_generation=state.session.resize_generation + 1,
      )
      layout = _state.LayoutState(columns=columns)
      resize_dirty = state.status_bar.resize_dirty or (
        state.session.history_width is not None
        and columns != state.session.history_width
      )
      status_bar = dataclasses.replace(
        state.status_bar,
        resize_dirty=resize_dirty,
      )
      return dataclasses.replace(
        state,
        session=session,
        layout=layout,
        status_bar=status_bar,
      ), ()
    case _actions.ResumeSession(reference):
      if state.session.remote:
        return state, (
          effects.EmitError(
            'use /remote <title-or-uuid> to resume a remote session'
          ),
        )
      if reference is not None:
        return state, (effects.ResolveResumeReference(reference),)
      resume = dataclasses.replace(state.resume, confirmation=None)
      opened, _ = _reduce(
        dataclasses.replace(state, resume=resume),
        _actions.OpenSelector(
          selection.Selector(
            id='resume',
            title='Resume conversation',
            choices=(),
            on_select=_actions.ResumeSession,
            empty='No other conversations in this scope.',
          )
        ),
      )
      return opened, (effects.ReloadResumeChoices(resume.scope),)
    case _actions.ResumeChoicesLoaded(scope, choices):
      if state.selector.id != 'resume' or scope is not state.resume.scope:
        return state, ()
      return _reduce(
        state,
        _actions.SelectorChoicesLoaded(
          'resume',
          tuple(
            selection.Choice(
              info.id,
              info.title or '(untitled)',
              (
                f'{info.updated_at.astimezone():%Y-%m-%d %H:%M} · '
                if info.updated_at is not None
                else ''
              )
              + f'{info.id[:8]} · {info.cwd}',
            )
            for info in choices
            if info.id != state.session.conversation_id
          ),
        ),
      )
    case _actions.ResumeReferenceResolved(reference, matches):
      if not matches:
        return state, (
          effects.EmitError(f'No conversation matches {reference!r}.'),
        )
      if len(matches) > 1:
        ids = ', '.join(info.id[:8] for info in matches[:5])
        return state, (
          effects.EmitError(f'Conversation id prefix is ambiguous: {ids}'),
        )
      info = next(iter(matches))
      if info.id == state.session.conversation_id:
        return state, (effects.EmitText('That conversation is already open.'),)
      if workspace.resolve_cwd(info.cwd) != state.workspace:
        return dataclasses.replace(
          state,
          resume=_state.ResumeState(confirmation=info),
        ), ()
      return state, (effects.ResumeConversation(info),)
    case _actions.SetResumeScope(scope):
      resume = dataclasses.replace(state.resume, scope=scope)
      updated, _ = _reduce(
        dataclasses.replace(state, resume=resume),
        _actions.SelectorChoicesLoaded('resume', ()),
      )
      return updated, (effects.ReloadResumeChoices(scope),)
    case _actions.ClearResume():
      return dataclasses.replace(state, resume=_state.ResumeState()), ()
    case _actions.AcceptResumeConfirmation():
      info = state.resume.confirmation
      if info is None:
        return state, ()
      return dataclasses.replace(state, resume=_state.ResumeState()), (
        effects.ResumeConversation(info, allow_workspace_mismatch=True),
      )
    case _actions.ModelResolved(option):
      if state.session.remote:
        return state, (
          effects.EmitError('remote model is configured by the server'),
        )
      return state, (effects.SwitchModel(option),)
    case _actions.OpenSelector(selector):
      if state.drop_confirmation is not None and selector.id != 'worktree-drop':
        return state, ()
      previous = state.selector.selector
      keep_open = (
        selector.keep_open
        and previous is not None
        and previous.id == selector.id
      )
      if (
        keep_open
        and previous is not None
        and isinstance(previous.cancel_action, _actions.ExtensionConsentBatch)
        and isinstance(selector.cancel_action, _actions.ExtensionConsentBatch)
      ):
        # Checkbox refreshes must not resurrect already resolved revisions.
        selector = dataclasses.replace(
          selector, cancel_action=previous.cancel_action
        )
      selector_state = _state.SelectorState(
        menu=_state.MenuState(max_rows=4 if selector.id == 'resume' else 8),
        selector=selector,
        query=state.selector.query
        if keep_open and not selector.numbered
        else '',
      )
      selected = next(
        (
          index
          for index, choice in enumerate(selector_state.filtered_choices())
          if choice.value == selector.selected
        ),
        0,
      )
      menu = (
        dataclasses.replace(
          state.selector.menu,
          selected=selected,
          start=min(state.selector.menu.start, selected),
        )
        if keep_open
        else selector_state.menu.open(selected=selected)
      )
      selector_state = dataclasses.replace(selector_state, menu=menu)
      opened = _dismiss_completion(
        dataclasses.replace(
          state, selector=selector_state, secret=_state.SecretState()
        )
      )
      emitted: tuple[effects.Effect, ...] = (
        (effects.SyncSecretInput(),) if state.secret.active else ()
      )
      if selector.load is not None:
        return opened, (*emitted, effects.LoadSelectorChoices(selector.id))
      return opened, emitted
    case _actions.SelectorSearchChanged(query):
      if not state.selector.active or state.selector.numbered:
        return state, ()
      request = state.selector.selector
      if request is not None and request.prefiltered:
        request = dataclasses.replace(request, loading=True)
      return dataclasses.replace(
        state,
        selector=dataclasses.replace(
          state.selector,
          query=query,
          search_pending=request is not None and request.prefiltered,
          selector=request,
          menu=state.selector.menu.reset(),
        ),
      ), ()
    case _actions.SelectorChoicesLoaded(selector_id, choices, header):
      request = state.selector.selector
      if request is None or request.id != selector_id:
        return state, ()
      current = state.selector.selected_choice()
      updated = dataclasses.replace(
        state.selector,
        selector=dataclasses.replace(
          request,
          choices=choices,
          header=request.header if header is None else header,
          loading=False,
        ),
      )
      filtered = updated.filtered_choices()
      selected = next(
        (
          index
          for index, choice in enumerate(filtered)
          if current is not None and choice.value == current.value
        ),
        0,
      )
      return dataclasses.replace(
        state,
        selector=dataclasses.replace(
          updated, menu=updated.menu.reset().move(selected, len(filtered))
        ),
      ), ()
    case _actions.SelectorResultsLoaded(selector_id, query, choices, loading):
      request = state.selector.selector
      if (
        request is None
        or not request.prefiltered
        or request.id != selector_id
        or state.selector.query != query
      ):
        return state, ()
      current = state.selector.selected_choice()
      selected = next(
        (index for index, choice in enumerate(choices) if choice == current),
        state.selector.menu.selected,
      )
      menu = dataclasses.replace(state.selector.menu, selected=selected)
      return dataclasses.replace(
        state,
        selector=dataclasses.replace(
          state.selector,
          selector=dataclasses.replace(
            request, choices=choices, loading=loading
          ),
          menu=menu.move(0, len(choices)),
          search_pending=False,
        ),
      ), ()
    case _actions.MoveSelectorSelection(delta):
      selector = state.selector.selector
      if selector is None:
        return state, ()
      menu = state.selector.menu.move(
        delta, len(state.selector.filtered_choices())
      )
      return dataclasses.replace(
        state,
        selector=dataclasses.replace(state.selector, menu=menu),
      ), ()
    case _actions.ClearSelector():
      return dataclasses.replace(state, selector=_state.SelectorState()), ()
    case _actions.SelectorChoiceSelected(selector_id, value):
      selector = state.selector.selector
      choice = state.selector.selected_choice()
      if (
        selector is None
        or selector.id != selector_id
        or choice is None
        or choice.value != value
      ):
        return state, ()
      if selector.keep_open:
        return state, ()
      return dataclasses.replace(state, selector=_state.SelectorState()), ()
    case _actions.RequestSecret(request):
      return _dismiss_completion(
        dataclasses.replace(
          state,
          selector=_state.SelectorState(),
          secret=_state.SecretState(request),
        )
      ), (effects.SyncSecretInput(),)
    case _actions.SecretLengthChanged(length):
      if not state.secret.active:
        return state, ()
      return dataclasses.replace(
        state, secret=dataclasses.replace(state.secret, length=length)
      ), ()
    case _actions.SubmitSecret():
      return state, (effects.SubmitSecret(),) if state.secret.active else ()
    case _actions.CancelSecret():
      request = state.secret.request
      if request is not None and request.cancel_action is not None:
        return state, (effects.CancelSecret(),)
      return _reduce(state, _actions.ClearSecret())
    case _actions.ClearSecret():
      if not state.secret.active:
        return state, ()
      return dataclasses.replace(state, secret=_state.SecretState()), (
        effects.SyncSecretInput(),
      )
    case _actions.SetApprovalMode(mode):
      return dataclasses.replace(
        state,
        status_bar=dataclasses.replace(state.status_bar, approval_mode=mode),
      ), (
        effects.SetWorkerApprovalMode(mode),
        effects.SaveWorktreePreferences(
          state.status_bar.reasoning_effort, mode, state.status_bar.speed
        ),
      )
    case _actions.RewindTurnsLoaded(conversation_id, turns):
      if (
        conversation_id != state.session.conversation_id or state.activity.busy
      ):
        return state, ()
      if not turns:
        return state, (
          effects.EmitText(
            'Nothing to rewind to: this conversation has no user turns.'
          ),
        )
      newest = tuple(reversed(turns))
      by_id = {
        turn.entry_id: (turn, index + 1) for index, turn in enumerate(newest)
      }
      return _reduce(
        state,
        _actions.OpenSelector(
          selection.Selector(
            id='rewind',
            title='Rewind to a user turn',
            choices=tuple(
              selection.Choice(turn.entry_id, f'↩{index + 1}  {turn.preview()}')
              for index, turn in enumerate(newest)
            ),
            on_select=lambda entry_id: _actions.StageRewind(*by_id[entry_id]),
          )
        ),
      )
    case _actions.SetRewindConfirming(value):
      if state.rewind.staged is None:
        return state, ()
      staged = dataclasses.replace(state.rewind.staged, confirming=value)
      rewind = dataclasses.replace(state.rewind, staged=staged)
      return dataclasses.replace(state, rewind=rewind), ()
    case _actions.CloseRewindOverlay():
      staged = state.rewind.staged
      if staged is not None:
        staged = dataclasses.replace(staged, confirming=False)
      return dataclasses.replace(
        state,
        rewind=_state.RewindState(staged=staged),
      ), ()
    case _actions.CancelRewind():
      staged = state.rewind.staged
      if staged is None:
        return state, ()
      next_state = dataclasses.replace(state, rewind=_state.RewindState())
      if (
        state.composer.text
        != quotes.display(staged.target.message, state.quotes)[0]
      ):
        return next_state, (effects.RerenderHistory('Rewind cancelled'),)
      return dataclasses.replace(
        next_state, composer=_actions.ComposerState()
      ), (
        effects.ResetComposer(),
        effects.RerenderHistory('Rewind cancelled'),
      )
    case _actions.ConfirmRewind():
      staged = state.rewind.staged
      instruction = state.composer.text.strip()
      if staged is None or not staged.confirming or not instruction:
        return state, ()
      return dataclasses.replace(
        state,
        rewind=_state.RewindState(),
        composer=_actions.ComposerState(),
        completion=_state.CompletionState(),
      ), (
        effects.ResetComposer(append_to_history=True),
        effects.RunTurn(instruction, rewind_to=staged.target.entry_id),
        effects.UserPromptSubmitted(instruction),
      )
    case _actions.ComposerChanged(text, cursor):
      composer = _actions.ComposerState(text=text, cursor=cursor)
      if composer == state.composer:
        return state, ()
      return dataclasses.replace(
        state,
        composer=composer,
        completion=_state.CompletionState(),
      ), ()
    case _actions.CompletionChanged(
      skill_choices, command_choices, hint, optional, empty_skills
    ):
      return dataclasses.replace(
        state,
        completion=dataclasses.replace(
          state.completion,
          skills=skill_choices,
          commands=command_choices,
          argument_hint=hint,
          argument_optional=optional,
          empty_skills=empty_skills,
        ),
      ), ()
    case _actions.TaskChanged(task_id, task, foreground):
      tasks = list(state.tasks.tasks)
      index = next(
        (index for index, item in enumerate(tasks) if item.id == task_id),
        None,
      )
      if index is not None:
        if task is None:
          tasks.pop(index)
        else:
          tasks[index] = task
      elif task is not None:
        tasks.append(task)
      selected_id = state.tasks.selected_id
      selection_removed = task is None and selected_id == task_id
      if selection_removed:
        selected_id = None
      task_state = _state.TasksState(
        tasks=tuple(tasks), selected_id=selected_id
      )
      foreground_tasks = state.activity.foreground_tasks
      if foreground:
        foreground_tasks |= {task_id}
      else:
        foreground_tasks -= {task_id}
      activity = dataclasses.replace(
        state.activity,
        foreground_tasks=foreground_tasks,
      )
      next_state = dataclasses.replace(
        state,
        tasks=task_state,
        activity=activity,
      )
      if selection_removed:
        return next_state, (effects.FocusComposer(),)
      return next_state, ()
    case _actions.SelectFirstTask():
      tasks = state.tasks.move(0)
      next_state = dataclasses.replace(state, tasks=tasks)
      if tasks.selected_id is None:
        return next_state, ()
      return next_state, (effects.FocusTasks(),)
    case _actions.MoveTaskSelection(delta):
      tasks = state.tasks.move(delta)
      next_state = dataclasses.replace(state, tasks=tasks)
      if state.tasks.selected_id is not None and tasks.selected_id is None:
        return next_state, (effects.FocusComposer(),)
      return next_state, ()
    case _actions.ClearTaskSelection():
      if state.tasks.selected_id is None:
        return state, ()
      tasks = dataclasses.replace(state.tasks, selected_id=None)
      return dataclasses.replace(state, tasks=tasks), (effects.FocusComposer(),)
    case _actions.InspectSelectedTask():
      task = next(
        (
          task
          for task in state.tasks.tasks
          if task.id == state.tasks.selected_id
        ),
        None,
      )
      if task is None or not task.inspectable:
        return state, ()
      return state, (effects.InspectTask(task),)
    case _actions.CloseInspection():
      return state, (effects.CloseInspection(),)
    case _actions.CancelSelectedTask():
      task_id = state.tasks.selected_id
      if task_id is None:
        return state, ()
      tasks = dataclasses.replace(state.tasks, selected_id=None)
      return dataclasses.replace(state, tasks=tasks), (
        effects.FocusComposer(),
        effects.CancelTask(task_id),
      )
    case _actions.NavigateHistory():
      return state, (effects.NavigateHistory(action),)
    case _actions.InterruptRequested():
      if state.lifecycle.disconnect_reason is not None:
        return state, ()
      if not state.activity.busy and state.approval.request is None:
        return state, ()
      activity = dataclasses.replace(state.activity, status='interrupting…')
      return dataclasses.replace(state, activity=activity), (
        effects.InterruptTurn(),
      )
    case _actions.CancelOrQuitRequested():
      if state.resume.confirmation is not None:
        return _reduce(state, _actions.ClearResume())
      if state.secret.active:
        return _reduce(state, _actions.CancelSecret())
      if state.selector.active or state.rewind.active:
        return _reduce(state, _actions.CloseSelector())
      if state.drop_confirmation is not None:
        return _reduce(state, _actions.ClearWorktreeDrop())
      if state.composer.text:
        return _clear_composer(state), (
          effects.ResetComposer(append_to_history=True),
        )
      if state.activity.busy:
        return _reduce(state, _actions.InterruptRequested())
      return _reduce(state, _actions.QuitRequested())
    case _actions.InterruptSent(message_id):
      turn = dataclasses.replace(state.turn, interrupt_id=message_id)
      return dataclasses.replace(state, turn=turn), ()
    case _actions.InterruptDeclined(message_id):
      if message_id != state.turn.interrupt_id:
        return state, ()
      busy = _settled_busy(state.turn)
      activity = dataclasses.replace(
        state.activity,
        busy=busy,
        status='thinking…' if busy else '',
        work_started=state.activity.work_started if busy else None,
      )
      turn = dataclasses.replace(state.turn, interrupt_id=None)
      return dataclasses.replace(state, activity=activity, turn=turn), ()
    case _actions.QuitRequested():
      return state, (effects.ExitApp(),)
    case _actions.OpenWorkspaceBrowser():
      if state.drop_confirmation is not None:
        return state, ()
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.OpenWorkspaceBrowser(),)
    case _actions.OpenWorkspaces():
      if state.drop_confirmation is not None:
        return state, ()
      if not state.supervised:
        return state, (
          effects.EmitError('worktree supervision is unavailable'),
        )
      return state, (effects.RefreshWorkspaces(),)
    case _actions.SelectWorkspaceRequested(identity):
      if state.drop_confirmation is not None:
        return state, ()
      return state, (effects.SelectWorkspace(identity),)
    case _actions.CycleWorktreeRequested(offset):
      if not offset or state.drop_confirmation is not None:
        return state, ()
      return state, (effects.CycleWorktree(offset),)
    case _actions.SelectWorktreeRequested(index):
      if state.drop_confirmation is not None:
        return state, ()
      return state, (effects.SelectWorktree(index),)
    case _actions.DetachRequested():
      if state.can_detach_tools:
        return state, (effects.DetachTools(),)
      return state, ()
