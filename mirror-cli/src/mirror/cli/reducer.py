"""Pure state transitions for the Mirror TUI."""

from __future__ import annotations

import dataclasses
import shlex

from mirror import workspace
from mirror.cli import actions as _actions
from mirror.cli import commands, effects, rendering, selection, tool_approvals
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
  composer = _actions.ComposerState(target.text, len(target.text))
  note = (
    f'↩ Rewound past {discards} turn(s), still saved above · '
    'send to fork from here, Esc to undo'
  )
  return dataclasses.replace(
    state,
    rewind=rewind,
    composer=composer,
    completion=_state.CompletionState(),
  ), (effects.SetComposerText(target.text), effects.RerenderHistory(note))


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
    label = (
      f'Approve and always allow commands starting with `{shlex.join(request.remember.argv)}`'
      if isinstance(request.remember, tool_approvals.ArgvPrefixApproval)
      else 'Approve and remember this exact call'
    )
    choices += (
      selection.Choice(
        'remember',
        label,
        detail='Local Bash in this workspace; literal arguments only, no shell operators'
        if isinstance(request.remember, tool_approvals.ArgvPrefixApproval)
        else 'Exact arguments only; no command-family rule',
      ),
    )
  choices += (selection.Choice('reject', 'Reject'),)
  return _reduce(
    state,
    _actions.OpenSelector(
      selection.Selector(
        id=f'approval:{request.call_id}',
        title=f'{rendering.strip_ansi(request.name)}({rendering.strip_ansi(request.args)})',
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
    case _actions.ComposerUndo():
      return state, (effects.UndoComposer(),)
    case _actions.ComposerRedo():
      return state, (effects.RedoComposer(),)
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
        if not state.selector.numbered or not 0 <= index < min(
          9, len(state.selector.filtered_choices())
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
    case _actions.ShowExtensions():
      return state, (effects.ShowExtensions(),)
    case (
      _actions.ToggleExtension()
      | _actions.ExtensionConsent()
      | _actions.AddExtension()
      | _actions.UpdateExtensions()
      | _actions.RemoveExtension()
    ):
      return state, (effects.ChangeExtension(action),)
    case _actions.ShowText(text, style):
      return state, (effects.EmitText(text, style),)
    case _actions.QuitApp():
      return state, (effects.ExitApp(),)
    case _actions.NewSession():
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
    case _actions.SetReasoning(effort):
      status_bar = dataclasses.replace(
        state.status_bar,
        reasoning_effort=effort,
      )
      return _with_status_bar(state, status_bar), (
        effects.SendInferenceConfiguration(),
        effects.SaveWorktreePreferences(effort, state.status_bar.approval_mode),
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
    case _actions.RestorePrompt(text):
      return dataclasses.replace(
        state, composer=_actions.ComposerState(text, len(text))
      ), (effects.SetComposerText(text),)
    case _actions.StageRewind(turn, discards):
      return _stage_rewind(state, turn, discards)
    case _actions.OpenFolder(path):
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.OpenFolder(path),)
    case _actions.CloseWorkspace():
      if not state.supervised:
        return state, (
          effects.EmitError('workspace supervision is unavailable'),
        )
      return state, (effects.CloseWorkspace(),)
    case _actions.CreateWorktree(name):
      if not state.supervised:
        return state, (
          effects.EmitError('worktree supervision is unavailable'),
        )
      return state, (effects.CreateManagedWorktree(name),)
    case _actions.LoadRemoteSessions():
      return state, (effects.LoadRemoteSessions(),)
    case _actions.CreateRemote(title, remote_workspace):
      if not state.supervised:
        return state, (effects.EmitError('agent supervision is unavailable'),)
      return state, (effects.CreateRemoteAgent(title, remote_workspace),)
    case _actions.Drop():
      if not state.supervised:
        return state, (
          effects.EmitError('worktree supervision is unavailable'),
        )
      return state, (effects.DropManagedWorktree(),)
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
      return _dismiss_completion(
        dataclasses.replace(state, drop_confirmation=confirmation)
      ), ()
    case _actions.ClearWorktreeDrop():
      return dataclasses.replace(state, drop_confirmation=None), ()
    case _actions.ConfirmWorktreeDrop():
      if state.drop_confirmation is None:
        return state, ()
      return dataclasses.replace(state, drop_confirmation=None), (
        effects.DropManagedWorktree(confirmed=True),
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
        spinner_color=spinner_color,
      )
      return _with_activity(state, activity), ()
    case _actions.AgentTurnStarted(observed_at):
      turn = dataclasses.replace(state.turn, agent_turn=True)
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status=state.activity.status
        if state.activity.busy and state.activity.status
        else 'thinking…',
        work_started=state.activity.work_started or observed_at,
      )
      return dataclasses.replace(state, turn=turn, activity=activity), ()
    case _actions.WorkerLoadingStarted():
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status='loading…',
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
      return dataclasses.replace(state, turn=turn), ()
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
      mode = 'approve' if state.status_bar.approval_mode == 'auto' else 'auto'
      status_bar = dataclasses.replace(state.status_bar, approval_mode=mode)
      return _with_status_bar(state, status_bar), (
        effects.SetWorkerApprovalMode(mode),
        effects.SaveWorktreePreferences(
          state.status_bar.reasoning_effort, mode
        ),
      )
    case _actions.ModelSwitchFinished(model_id):
      status_bar = dataclasses.replace(state.status_bar, model=model_id)
      return _with_status_bar(state, status_bar), (
        effects.SendInferenceConfiguration(),
        effects.SaveLastModel(model_id),
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
    case _actions.WorkerStarted():
      lifecycle = dataclasses.replace(
        state.lifecycle,
        started=True,
        failure=None,
      )
      return dataclasses.replace(state, lifecycle=lifecycle), ()
    case _actions.WorkerStopped():
      lifecycle = dataclasses.replace(state.lifecycle, started=False)
      turn = dataclasses.replace(state.turn, agent_turn=False)
      return dataclasses.replace(state, lifecycle=lifecycle, turn=turn), ()
    case _actions.WorkerFailed(message):
      lifecycle = dataclasses.replace(
        state.lifecycle,
        started=False,
        failure=message,
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
          request.call_id, approved, request.remember if remember else None
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
        if selected_command.edit_arguments and (
          not state.activity.busy or selected_command.available_when_busy
        ):
          return _reduce(
            state,
            _actions.CompleteSelectionRequested(None, selected_command),
          )
        return _clear_composer(state), (
          effects.ResetComposer(),
          effects.RunSlashCommand(f'/{selected_command.name}'),
        )
      instruction = state.composer.text.strip()
      if not instruction:
        if state.lifecycle.failure is not None and not state.activity.busy:
          return state, (effects.RetryWorker(),)
        return state, ()
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
    case _actions.TurnFinished():
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
        work_started=state.activity.work_started if busy else None,
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
        pending_switch=pending,
      )
      activity = dataclasses.replace(
        state.activity,
        busy=True,
        status='loading conversation…',
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
      )
      activity = dataclasses.replace(
        state.activity,
        foreground_tasks=frozenset(),
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
              f'{info.updated_at.astimezone():%Y-%m-%d %H:%M} · {info.id[:8]} · {info.cwd}',
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
      previous = state.selector.selector
      keep_open = (
        selector.keep_open
        and previous is not None
        and previous.id == selector.id
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
    case _actions.SelectorChoicesLoaded(selector_id, choices):
      request = state.selector.selector
      if request is None or request.id != selector_id:
        return state, ()
      return dataclasses.replace(
        state,
        selector=dataclasses.replace(
          state.selector,
          selector=dataclasses.replace(request, choices=choices),
          menu=state.selector.menu.reset(),
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
          state.status_bar.reasoning_effort, mode
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
      if state.composer.text != staged.target.text:
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
      skill_choices, command_choices, hint, optional
    ):
      return dataclasses.replace(
        state,
        completion=dataclasses.replace(
          state.completion,
          skills=skill_choices,
          commands=command_choices,
          argument_hint=hint,
          argument_optional=optional,
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
      if not state.activity.busy and state.approval.request is None:
        return state, ()
      drafts = (
        *_unsaved_instructions(state.turn),
        *((state.composer.text,) if state.composer.text else ()),
      )
      text = '\n'.join(drafts)
      activity = dataclasses.replace(state.activity, status='interrupting…')
      turn = dataclasses.replace(
        state.turn,
        queued=(),
        submitted_inputs=(),
        awaiting_inputs=(),
        input_order=(),
      )
      composer = _actions.ComposerState(text, len(text))
      return dataclasses.replace(
        state,
        activity=activity,
        turn=turn,
        composer=composer,
        completion=_state.CompletionState(),
      ), (effects.SetComposerText(text), effects.InterruptTurn())
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
    case _actions.OpenWorkspaces():
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
      if state.activity.busy:
        return state, (effects.DetachTools(),)
      return state, ()
