"""Tests for immutable Mirror UI state transitions."""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from mirror import workspace
from mirror.cli import (
  actions,
  cli_store,
  models,
  reducer,
  selection,
  state,
)
from mirror.cli import (
  effects as _effects,
)


def _state(*, worktree_supervised: bool = False) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState(
      model='test-model',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    session=state.SessionState(conversation_id='conversation'),
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    supervised=worktree_supervised,
  )


def test_async_selector_uses_ranked_results_without_filtering_again() -> None:
  request = selection.Selector(
    'files',
    'Files',
    (),
    actions.SetModel,
    prefiltered=True,
    loading=True,
  )
  current, _ = reducer.reduce(_state(), actions.OpenSelector(request))
  current, _ = reducer.reduce(current, actions.SelectorSearchChanged('query'))
  ranked = (selection.Choice('value', 'externally ranked result'),)

  current, _ = reducer.reduce(
    current,
    actions.SelectorResultsLoaded('files', 'query', ranked, False),
  )

  assert current.selector.filtered_choices() == ranked
  assert current.selector.selector is not None
  assert not current.selector.selector.loading


def test_async_selector_discards_old_queries_and_closed_picker_results() -> (
  None
):
  request = selection.Selector(
    'files',
    'Files',
    (),
    actions.SetModel,
    prefiltered=True,
  )
  current, _ = reducer.reduce(_state(), actions.OpenSelector(request))
  current, _ = reducer.reduce(current, actions.SelectorSearchChanged('new'))
  stale = actions.SelectorResultsLoaded(
    'files',
    'old',
    (selection.Choice('old', 'old'),),
    False,
  )

  unchanged, _ = reducer.reduce(current, stale)
  assert unchanged is current
  closed, _ = reducer.reduce(current, actions.ClearSelector())
  unchanged, _ = reducer.reduce(closed, stale)
  assert unchanged is closed
  reopened, _ = reducer.reduce(
    closed,
    actions.OpenSelector(dataclasses.replace(request, id='new-files')),
  )
  unchanged, _ = reducer.reduce(reopened, stale)
  assert unchanged is reopened


def _conversation(conversation_id: str) -> cli_store.ConversationInfo:
  now = datetime.datetime.now(datetime.UTC)
  return cli_store.ConversationInfo(
    conversation_id,
    None,
    '/workspace',
    'test-model',
    now,
    now,
  )


def test_command_failure_declares_error_effect() -> None:
  current = _state()

  unchanged, effects = reducer.reduce(
    current, actions.CommandFailed('bad config')
  )

  assert unchanged is current
  assert effects == (_effects.EmitError('bad config'),)


def test_supervisor_error_declares_recording_effect() -> None:
  current = _state()

  unchanged, effects = reducer.reduce(
    current, actions.SupervisorError('drop failed')
  )

  assert unchanged is current
  assert effects == (_effects.RecordError('drop failed'),)


def test_show_text_command_declares_terminal_effect() -> None:
  current = _state()

  unchanged, effects = reducer.reduce(
    current, actions.ShowText('hello', style='yellow')
  )

  assert unchanged is current
  assert effects == (_effects.EmitText('hello', style='yellow'),)


def test_quit_command_declares_exit_effect() -> None:
  current = _state()

  unchanged, effects = reducer.reduce(current, actions.QuitApp())
  requested, requested_effects = reducer.reduce(
    current, actions.ExitRequested()
  )

  assert unchanged is current
  assert effects == (_effects.ExitApp(),)
  assert requested is current
  assert requested_effects == (_effects.ExitApp(defer_until_running=True),)


def test_clear_screen_command_declares_scrollback_rerender() -> None:
  current, _ = reducer.reduce(_state(), actions.SetReplaying(False))

  updated, effects = reducer.reduce(current, actions.ClearScreen())

  assert updated.session.rerendering
  assert effects == (
    _effects.RerenderHistory('Screen cleared', drop_scrollback=True),
  )


def test_view_commands_rerender_idle_and_busy() -> None:
  current, _ = reducer.reduce(_state(), actions.SetReplaying(False))

  cycled, effects = reducer.reduce(current, actions.CycleViewMode())
  busy, _ = reducer.reduce(current, actions.TurnStarted())
  changed_busy, busy_effects = reducer.reduce(
    busy, actions.SetViewMode(actions.ViewMode.EXTRA_COMPACT)
  )

  assert current.status_bar.view_mode is actions.ViewMode.COMPACT
  assert cycled.status_bar.view_mode is actions.ViewMode.EXTRA_COMPACT
  assert cycled.session.rerendering
  assert effects == (_effects.RerenderHistory(),)
  assert changed_busy.status_bar.view_mode is actions.ViewMode.EXTRA_COMPACT
  assert busy_effects == (_effects.RerenderHistory(),)
  assert changed_busy.activity == busy.activity
  assert changed_busy.turn == busy.turn


def test_worktree_commands_declare_background_controller_effects() -> None:
  current = _state(worktree_supervised=True)

  unchanged, create_effects = reducer.reduce(
    current, actions.CreateWorktree('feature')
  )
  dropping, drop_effects = reducer.reduce(current, actions.Drop())
  assert unchanged is current
  assert create_effects == (_effects.CreateManagedWorktree('feature'),)
  assert dropping is current
  assert drop_effects == (_effects.DropManagedWorktree(),)


@pytest.mark.parametrize('text', ['draft', 'first\nsecond', ' \n', '/he'])
@pytest.mark.parametrize('busy', [False, True])
def test_cancel_or_quit_clears_only_the_draft(text: str, busy: bool) -> None:
  current = dataclasses.replace(
    _state(),
    activity=state.ActivityState(busy=busy),
    composer=actions.ComposerState(text, 1),
    completion=state.CompletionState(slash=state.MenuState(active=True)),
    turn=state.TurnState(queued=('queued',)),
  )

  cleared, effects = reducer.reduce(current, actions.CancelOrQuitRequested())

  assert cleared == dataclasses.replace(
    current,
    composer=actions.ComposerState(),
    completion=state.CompletionState(),
  )
  assert effects == (_effects.ResetComposer(append_to_history=True),)
  _, effects = reducer.reduce(current, actions.QuitRequested())
  assert effects == (_effects.ExitApp(),)


def test_cancel_or_quit_interrupts_busy_turn_then_quits_when_idle() -> None:
  current, _ = reducer.reduce(_state(), actions.TurnStarted())
  interrupted, effects = reducer.reduce(
    current, actions.CancelOrQuitRequested()
  )
  assert effects == (_effects.SetComposerText(''), _effects.InterruptTurn())
  assert interrupted.activity.busy
  repeated, effects = reducer.reduce(
    interrupted, actions.CancelOrQuitRequested()
  )
  assert effects == (_effects.SetComposerText(''), _effects.InterruptTurn())
  idle, _ = reducer.reduce(repeated, actions.TurnFinished())
  _, effects = reducer.reduce(idle, actions.CancelOrQuitRequested())
  assert effects == (_effects.ExitApp(),)


@pytest.mark.parametrize('busy', [False, True])
def test_approval_cancel_uses_interrupt_even_before_turn_started(
  busy: bool,
) -> None:
  current, _ = reducer.reduce(
    dataclasses.replace(_state(), activity=state.ActivityState(busy=busy)),
    actions.ApprovalRequested(actions.ApprovalRequest('call', 'bash', 'ls')),
  )
  assert current.selector.selector is not None
  assert current.selector.selector.cancel_action == actions.InterruptRequested()
  interrupted, effects = reducer.reduce(current, actions.InterruptRequested())
  assert interrupted.approval.request is not None
  assert effects == (_effects.SetComposerText(''), _effects.InterruptTurn())


def test_quit_and_worktree_navigation_declare_controller_effects() -> None:
  current = _state(worktree_supervised=True)
  _, quit_effects = reducer.reduce(current, actions.QuitRequested())
  _, cycle_effects = reducer.reduce(current, actions.CycleWorktreeRequested(-1))
  _, selected_effects = reducer.reduce(
    current, actions.SelectWorktreeRequested(0)
  )
  _, select_effects = reducer.reduce(
    current, actions.SelectWorktreeRequested(1)
  )

  assert quit_effects == (_effects.ExitApp(),)
  assert cycle_effects == (_effects.CycleWorktree(-1),)
  assert selected_effects == (_effects.SelectWorktree(0),)
  assert select_effects == (_effects.SelectWorktree(1),)


def test_worktree_commands_report_unavailable_supervision() -> None:
  current = _state()

  _, create_effects = reducer.reduce(current, actions.CreateWorktree())
  _, drop_effects = reducer.reduce(current, actions.Drop())

  unavailable = (_effects.EmitError('worktree supervision is unavailable'),)
  assert create_effects == unavailable
  assert drop_effects == unavailable


def test_resolved_model_switches_and_updates_status() -> None:
  current = _state()
  option = models.ModelOption('local', 'Local')

  resolved, switch_effects = reducer.reduce(
    current, actions.ModelResolved(option)
  )
  switched, finish_effects = reducer.reduce(
    current, actions.ModelSwitchFinished('local')
  )

  assert resolved is current
  assert switch_effects == (_effects.SwitchModel(option),)
  assert switched.status_bar.model == 'local'
  assert finish_effects == (
    _effects.SendInferenceConfiguration(),
    _effects.SaveLastModel('local'),
  )


def test_rewind_command_loads_then_opens_available_turns() -> None:
  current = _state()
  first = actions.UserTurn('first', 'first try')
  second = actions.UserTurn('second', 'second try')

  unchanged, load_effects = reducer.reduce(current, actions.RewindSession())
  opened, loaded_effects = reducer.reduce(
    current,
    actions.RewindTurnsLoaded('conversation', (first, second)),
  )

  assert unchanged is current
  assert load_effects == (_effects.LoadRewindTurns('conversation'),)
  assert opened.selector.active
  assert [choice.value for choice in opened.selector.filtered_choices()] == [
    'second',
    'first',
  ]
  assert loaded_effects == (_effects.SyncSelectorInput(True, ''),)


def test_empty_rewind_result_declares_message_effect() -> None:
  current = _state()

  unchanged, effects = reducer.reduce(
    current, actions.RewindTurnsLoaded('conversation', ())
  )

  assert unchanged is current
  assert effects == (
    _effects.EmitText(
      'Nothing to rewind to: this conversation has no user turns.'
    ),
  )


def test_rename_command_declares_worktree_alias_effect() -> None:
  current = _state(worktree_supervised=True)

  unchanged, rename_effects = reducer.reduce(
    current, actions.RenameWorktree('New title')
  )

  assert unchanged is current
  assert rename_effects == (_effects.RenameWorktree('New title'),)


def test_new_session_command_declares_switch_effect() -> None:
  current = _state()

  unchanged, effects = reducer.reduce(current, actions.NewSession())

  assert unchanged is current
  assert effects == (_effects.StartNewConversation(),)


def test_resume_command_declares_loading_and_resolution_effects() -> None:
  current = _state()
  info = _conversation('other')

  opened, open_effects = reducer.reduce(current, actions.ResumeSession())
  unchanged, lookup_effects = reducer.reduce(
    current, actions.ResumeSession('other')
  )
  resolved, resolved_effects = reducer.reduce(
    current,
    actions.ResumeReferenceResolved('other', (info,)),
  )
  loaded, loaded_effects = reducer.reduce(
    opened,
    actions.ResumeChoicesLoaded(
      workspace.SessionScope.CWD,
      (_conversation('conversation'), info),
    ),
  )

  assert opened.selector.active
  assert open_effects == (
    _effects.ReloadResumeChoices(workspace.SessionScope.CWD),
    _effects.SyncSelectorInput(True, ''),
  )
  assert unchanged is current
  assert lookup_effects == (_effects.ResolveResumeReference('other'),)
  assert [choice.value for choice in loaded.selector.filtered_choices()] == [
    info.id
  ]
  assert loaded_effects == ()
  assert resolved is current
  assert resolved_effects == (_effects.ResumeConversation(info),)


def test_resume_choices_ignore_closed_or_superseded_loads() -> None:
  current = _state()
  opened, _ = reducer.reduce(current, actions.ResumeSession())
  switched, _ = reducer.reduce(
    opened, actions.SetResumeScope(workspace.SessionScope.GIT)
  )
  info = _conversation('other')

  stale, stale_effects = reducer.reduce(
    switched,
    actions.ResumeChoicesLoaded(workspace.SessionScope.CWD, (info,)),
  )
  closed, _ = reducer.reduce(switched, actions.ClearSelector())
  closed_result, closed_effects = reducer.reduce(
    closed,
    actions.ResumeChoicesLoaded(workspace.SessionScope.GIT, (info,)),
  )

  assert stale is switched
  assert stale_effects == ()
  assert closed_result is closed
  assert closed_effects == ()


def test_resume_resolution_reports_missing_and_current_conversations() -> None:
  current = _state()

  _, missing_effects = reducer.reduce(
    current, actions.ResumeReferenceResolved('missing', ())
  )
  _, current_effects = reducer.reduce(
    current,
    actions.ResumeReferenceResolved(
      'conversation',
      (_conversation('conversation'),),
    ),
  )

  assert missing_effects == (
    _effects.EmitError("No conversation matches 'missing'."),
  )
  assert current_effects == (
    _effects.EmitText('That conversation is already open.'),
  )


def test_reasoning_action_updates_inference_configuration() -> None:
  current = _state()

  changed, changed_effects = reducer.reduce(
    current, actions.SetReasoning('high')
  )

  assert changed.status_bar.reasoning_effort == 'high'
  assert changed_effects == (
    _effects.SendInferenceConfiguration(),
    _effects.SaveWorktreePreferences('high', 'auto'),
  )


def test_extension_unload_preserves_builtin_selector() -> None:
  opened, _ = reducer.reduce(_state(), actions.SelectPermissions())
  assert opened.selector.selector is not None
  assert opened.selector.selector.extension_id is None

  unchanged, effects = reducer.reduce(
    opened, actions.ExtensionUnloaded('inference-controls')
  )

  assert unchanged is opened
  assert effects == ()


def test_permission_picker_updates_worker_mode() -> None:
  opened, _ = reducer.reduce(_state(), actions.SelectPermissions())
  moved, _ = reducer.reduce(opened, actions.MoveSelectorSelection(1))
  closed, _ = reducer.reduce(
    moved, actions.SelectorChoiceSelected('permissions', 'approve')
  )
  request = moved.selector.selector
  assert request is not None
  changed, effects = reducer.reduce(closed, request.on_select('approve'))

  assert opened.selector.active
  assert opened.selector.menu.selected == 0
  assert changed.status_bar.approval_mode == 'approve'
  assert not changed.selector.active
  assert effects == (
    _effects.SetWorkerApprovalMode('approve'),
    _effects.SaveWorktreePreferences('medium', 'approve'),
  )


def test_reduce_replaces_state_without_mutating_input() -> None:
  current = _state()

  updated, effects = reducer.reduce(current, actions.TurnStarted())

  assert not current.activity.busy
  assert updated.activity.busy
  assert effects == ()


def test_detach_request_declares_effect_only_while_turn_is_active() -> None:
  current, _ = reducer.reduce(
    _state(worktree_supervised=True), actions.TurnStarted()
  )

  unchanged, effects = reducer.reduce(current, actions.DetachRequested())

  assert unchanged is current
  assert effects == (_effects.DetachTools(),)
  finished, _ = reducer.reduce(current, actions.TurnFinished())
  assert reducer.reduce(finished, actions.DetachRequested()) == (finished, ())


def test_completion_move_prefers_skill_matches() -> None:
  current = _state()

  skill, skill_effects = reducer.reduce(
    current,
    actions.MoveCompletionRequested(1, skill_count=2, command_count=3),
  )
  command, command_effects = reducer.reduce(
    current,
    actions.MoveCompletionRequested(1, skill_count=0, command_count=3),
  )

  assert skill.completion.skill.selected == 1
  assert skill.completion.slash.selected == 0
  assert skill_effects == ()
  assert command.completion.slash.selected == 1
  assert command_effects == ()


def test_completion_request_updates_composer_through_an_effect() -> None:
  current, _ = reducer.reduce(_state(), actions.ComposerChanged('/r', 2))
  selected = actions.CommandSelection(
    name='rename',
    completion='rename',
    edit_arguments=True,
    available_when_busy=True,
  )
  updated, effects = reducer.reduce(
    current, actions.CompleteSelectionRequested(None, selected)
  )

  assert updated.composer == actions.ComposerState('/rename ', 8)
  assert effects == (_effects.SetComposerText('/rename '),)


def test_dismiss_completion_applies_or_resets_the_composer() -> None:
  current, _ = reducer.reduce(
    _state(), actions.ComposerChanged('Use $p later', 6)
  )
  edit = actions.ComposerState('Use  later', 4)

  updated, effects = reducer.reduce(
    current, actions.DismissCompletionRequested(edit)
  )
  reset, reset_effects = reducer.reduce(
    current, actions.DismissCompletionRequested(None)
  )

  assert updated.composer == edit
  assert effects == (_effects.SetComposerText('Use  later', 4),)
  assert reset.composer == actions.ComposerState()
  assert reset_effects == (_effects.ResetComposer(),)


def test_submit_request_clears_composer_and_declares_turn_effects() -> None:
  current, _ = reducer.reduce(_state(), actions.ComposerChanged('  first  ', 9))

  updated, effects = reducer.reduce(
    current, actions.SubmitRequested(None, None)
  )

  assert updated.composer == actions.ComposerState()
  assert effects == (
    _effects.ResetComposer(append_to_history=True),
    _effects.RunTurn('first'),
    _effects.UserPromptSubmitted('first'),
  )


def test_submit_request_retries_a_failed_worker() -> None:
  failed, _ = reducer.reduce(_state(), actions.WorkerFailed('failed'))

  unchanged, effects = reducer.reduce(
    failed, actions.SubmitRequested(None, None)
  )

  assert unchanged is failed
  assert effects == (_effects.RetryWorker(),)


def test_instruction_submission_declares_queue_or_turn_effect() -> None:
  idle = _state()

  unchanged, idle_effects = reducer.reduce(
    idle, actions.SubmitInstruction('first')
  )
  busy, _ = reducer.reduce(idle, actions.TurnStarted())
  queued, busy_effects = reducer.reduce(
    busy, actions.SubmitInstruction('follow-up')
  )

  assert unchanged is idle
  assert idle_effects == (_effects.RunTurn('first'),)
  assert queued.turn.queued == ('follow-up',)
  assert busy_effects == (_effects.SendQueuedInstruction('follow-up'),)


def test_approval_decision_clears_request_and_declares_effects() -> None:
  request = actions.ApprovalRequest('call', 'bash', 'ls')
  current, _ = reducer.reduce(_state(), actions.ApprovalRequested(request))
  current, _ = reducer.reduce(current, actions.MoveApproval(1))

  updated, effects = reducer.reduce(current, actions.DecideApproval())

  assert updated.approval == state.ApprovalState()
  assert effects == (
    _effects.SendApprovalDecision('call', False),
    _effects.SyncSelectorInput(False, ''),
  )


def test_explicit_rejection_keeps_next_approval_without_interrupting() -> None:
  current, _ = reducer.reduce(_state(), actions.TurnStarted())
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('first', 'bash', 'ls')),
  )
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('second', 'bash', 'pwd')),
  )
  updated, effects = reducer.reduce(current, actions.DecideApproval(False))
  assert updated.selector.id == 'approval:second'
  assert updated.activity.busy
  assert effects == (
    _effects.SendApprovalDecision('first', False),
    _effects.SyncSelectorInput(True, ''),
  )


def test_interrupt_restores_queued_and_draft_text_before_requesting_stop() -> (
  None
):
  current, _ = reducer.reduce(_state(), actions.TurnStarted())
  current, _ = reducer.reduce(current, actions.SubmitInstruction('queued'))
  current, _ = reducer.reduce(current, actions.ComposerChanged('draft', 5))

  updated, effects = reducer.reduce(current, actions.InterruptRequested())

  assert updated.activity.status == 'interrupting…'
  assert updated.turn.queued == ()
  assert updated.composer == actions.ComposerState('queued\ndraft', 12)
  assert effects == (
    _effects.SetComposerText('queued\ndraft'),
    _effects.InterruptTurn(),
  )


def test_domain_actions_own_nested_state_transitions() -> None:
  current = _state()
  switched, _ = reducer.reduce(
    current,
    actions.SessionSwitchStarted(actions.PendingSwitch('next', create=False)),
  )

  assert switched.session.replaying
  assert switched.session.pending_switch is not None
  assert switched.activity.busy
  assert switched.activity.status == 'loading conversation…'
  assert current.session.pending_switch is None


def test_composer_action_resets_completion_selection() -> None:
  current, _ = reducer.reduce(
    _state(),
    actions.MoveCompletionRequested(3, skill_count=0, command_count=10),
  )

  updated, _ = reducer.reduce(current, actions.ComposerChanged('/he', 3))

  assert updated.composer == actions.ComposerState('/he', 3)
  assert updated.completion == state.CompletionState()


def test_modal_open_and_close_preserves_composer_and_clears_completion() -> (
  None
):
  current, _ = reducer.reduce(_state(), actions.ComposerChanged('/', 1))
  current, _ = reducer.reduce(
    current, actions.MoveCompletionRequested(2, skill_count=0, command_count=10)
  )
  turn = actions.UserTurn('entry', 'try again')

  resume, _ = reducer.reduce(current, actions.ResumeSession())
  selector_picker, _ = reducer.reduce(
    current,
    actions.OpenSelector(
      selection.Selector(
        'choice',
        'Choose',
        (selection.Choice('value', 'Value'),),
        lambda _: actions.ClearSelector(),
        extension_id='test/selector',
      )
    ),
  )
  permissions, _ = reducer.reduce(current, actions.SelectPermissions())
  rewind, _ = reducer.reduce(
    current, actions.RewindTurnsLoaded('conversation', (turn,))
  )
  worktree_drop, _ = reducer.reduce(
    current,
    actions.WorktreeDropConfirmationRequested(
      actions.DropConfirmation('feature', '/workspace/feature')
    ),
  )

  preserved = (current.composer, state.CompletionState())
  assert (resume.composer, resume.completion) == preserved
  assert (selector_picker.composer, selector_picker.completion) == preserved
  assert (permissions.composer, permissions.completion) == preserved
  assert (rewind.composer, rewind.completion) == preserved
  assert (worktree_drop.composer, worktree_drop.completion) == preserved

  resume, _ = reducer.reduce(resume, actions.ClearSelector())
  selector_picker, _ = reducer.reduce(selector_picker, actions.ClearSelector())
  permissions, _ = reducer.reduce(permissions, actions.ClearSelector())
  rewind, _ = reducer.reduce(rewind, actions.CloseRewindOverlay())
  worktree_drop, _ = reducer.reduce(worktree_drop, actions.ClearWorktreeDrop())

  assert resume.composer == current.composer
  assert selector_picker.composer == current.composer
  assert permissions.composer == current.composer
  assert rewind.composer == current.composer
  assert worktree_drop.composer == current.composer


def test_rewind_actions_own_picker_visibility() -> None:
  turn = actions.UserTurn('entry', 'try again')

  opened, _ = reducer.reduce(
    _state(), actions.RewindTurnsLoaded('conversation', (turn,))
  )
  closed, _ = reducer.reduce(
    opened, actions.SelectorChoiceSelected('rewind', turn.entry_id)
  )
  request = opened.selector.selector
  assert request is not None
  staged, effects = reducer.reduce(closed, request.on_select(turn.entry_id))

  assert opened.selector.active
  assert not staged.selector.active
  assert staged.rewind.staged == state.StagedRewind(turn, discards=1)
  assert effects == (
    _effects.SetComposerText('try again'),
    _effects.RerenderHistory(
      '↩ Rewound past 1 turn(s), still saved above · '
      'send to fork from here, Esc to undo'
    ),
  )


def test_stream_text_commits_without_mutating_observed_snapshot() -> None:
  current = _state()

  observed, _ = reducer.reduce(
    current,
    actions.ObserveStreamText(content='answer', reasoning='thought'),
  )
  committed, _ = reducer.reduce(observed, actions.CommitStreamReasoning())

  assert current.stream.preview is None
  assert observed.stream.preview == state.StreamPreview(
    content='answer', reasoning='thought'
  )
  assert committed.stream.preview == state.StreamPreview(
    content='answer', committed_reasoning='thought'
  )


def test_stream_tool_calls_transition_between_pending_and_running() -> None:
  current = _state()
  assembling = actions.StreamToolCall(0, 'bash', '{"command":"ls"}')
  pending = actions.PendingToolCall(0, 'bash', 'ls', 'tool')
  running = actions.RunningToolCall(
    'call', 'bash', 'ls', 'tool', '{"command":"ls"}'
  )

  streamed, _ = reducer.reduce(
    current, actions.ObserveStreamToolCall(assembling, pending)
  )
  started, _ = reducer.reduce(
    streamed, actions.StartRunningToolCalls((running,))
  )
  settled, _ = reducer.reduce(started, actions.FinishRunningToolCall('call'))

  assert streamed.stream.tool_calls == (assembling,)
  assert streamed.stream.pending == (pending,)
  assert started.stream.running == (running,)
  assert settled.stream.running == ()


def test_task_updates_preserve_row_order() -> None:
  first = actions.Task('one', 'bash', None)
  second = actions.Task('two', 'bash', None)
  current, _ = reducer.reduce(
    _state(), actions.TaskChanged('one', first, False)
  )
  current, _ = reducer.reduce(
    current, actions.TaskChanged('two', second, False)
  )
  updated_first = actions.Task('one', 'bash', 'updated')

  updated, _ = reducer.reduce(
    current,
    actions.TaskChanged('one', updated_first, False),
  )

  assert updated.tasks.tasks == (updated_first, second)


def test_task_selection_declares_focus_and_cancel_effects() -> None:
  task = actions.Task('one', 'bash', None)
  current, _ = reducer.reduce(_state(), actions.TaskChanged('one', task, False))

  selected, focus_effects = reducer.reduce(current, actions.SelectFirstTask())
  cancelled, cancel_effects = reducer.reduce(
    selected, actions.CancelSelectedTask()
  )

  assert selected.tasks.selected_id == 'one'
  assert focus_effects == (_effects.FocusTasks(),)
  assert cancelled.tasks.selected_id is None
  assert cancel_effects == (
    _effects.FocusComposer(),
    _effects.CancelTask('one'),
  )


def test_selector_search_preserves_draft_and_cancellation_declares_focus() -> (
  None
):
  current = dataclasses.replace(
    _state(), composer=actions.ComposerState('draft text', 3)
  )
  opened, effects = reducer.reduce(current, actions.RequestPromptHistory())
  loaded, _ = reducer.reduce(
    opened,
    actions.SelectorChoicesLoaded(
      'history',
      (
        selection.Choice('fix the bug', 'fix the bug'),
        selection.Choice('add a feature', 'add a feature'),
      ),
    ),
  )
  filtered, search_effects = reducer.reduce(
    loaded, actions.SelectorSearchChanged('fxbg')
  )
  assert filtered.selector.selected_choice() == selection.Choice(
    'fix the bug', 'fix the bug'
  )
  assert filtered.composer == current.composer
  assert effects == (
    _effects.LoadPromptHistory(),
    _effects.SyncSelectorInput(True, ''),
  )
  assert search_effects == (_effects.SyncSelectorInput(True, 'fxbg'),)
  closed, close_effects = reducer.reduce(filtered, actions.ClearSelector())
  assert closed.composer == current.composer
  assert close_effects == (_effects.SyncSelectorInput(False, ''),)
  stale, stale_effects = reducer.reduce(
    closed,
    actions.SelectorChoicesLoaded(
      'history', (selection.Choice('late', 'late'),)
    ),
  )
  assert stale is closed
  assert stale_effects == ()


def test_filtered_rewind_retains_original_discard_distance() -> None:
  turns = (
    actions.UserTurn('first', 'oldest prompt'),
    actions.UserTurn('second', 'latest prompt'),
  )
  opened, _ = reducer.reduce(
    _state(), actions.RewindTurnsLoaded('conversation', turns)
  )
  filtered, _ = reducer.reduce(opened, actions.SelectorSearchChanged('old'))
  choice = filtered.selector.selected_choice()
  request = filtered.selector.selector
  assert choice is not None and request is not None
  assert request.on_select(choice.value) == actions.StageRewind(turns[0], 2)


def test_approval_queue_preserves_selection_and_advances_after_decision() -> (
  None
):
  first = actions.ApprovalRequest('first', 'bash', 'one')
  second = actions.ApprovalRequest('second', 'bash', 'two')
  current, _ = reducer.reduce(_state(), actions.ApprovalRequested(first))
  current, _ = reducer.reduce(current, actions.MoveSelection(1))
  current, _ = reducer.reduce(current, actions.ApprovalRequested(second))

  assert current.approval.request == first
  assert current.selector.menu.selected == 1
  selector = current.selector.selector
  choice = current.selector.selected_choice()
  assert selector is not None and choice is not None
  current, effects = reducer.reduce(current, selector.on_select(choice.value))
  assert current.approval.request == second
  assert current.selector.menu.selected == 0
  assert effects == (
    _effects.SendApprovalDecision('first', False),
    _effects.SyncSelectorInput(True, ''),
  )

  current, _ = reducer.reduce(current, actions.MoveSelection(1))
  current, _ = reducer.reduce(current, actions.ClearApproval('first'))
  assert current.approval.request == second
  assert current.selector.menu.selected == 1

  current, effects = reducer.reduce(
    current, actions.DecideApproval(True, call_id='second')
  )
  assert current.approval == state.ApprovalState()
  assert effects == (
    _effects.SendApprovalDecision('second', True),
    _effects.SyncSelectorInput(False, ''),
  )


def test_approval_waits_for_worktree_drop_confirmation() -> None:
  confirmation = actions.DropConfirmation('feature', '/workspace/feature')
  current, _ = reducer.reduce(
    _state(), actions.WorktreeDropConfirmationRequested(confirmation)
  )
  request = actions.ApprovalRequest('call', 'bash', 'ls')

  current, _ = reducer.reduce(current, actions.ApprovalRequested(request))

  assert current.drop_confirmation == confirmation
  assert not current.selector.active
  current, effects = reducer.reduce(current, actions.ClearWorktreeDrop())
  assert current.approval.request == request
  assert current.selector.id == 'approval:call'
  assert effects == (_effects.SyncSelectorInput(True, ''),)


def test_approval_resolution_removes_only_the_matching_request() -> None:
  first = actions.ApprovalRequest('first', 'bash', 'one')
  second = actions.ApprovalRequest('second', 'bash', 'two')
  third = actions.ApprovalRequest('third', 'bash', 'three')
  current, _ = reducer.reduce(_state(), actions.ApprovalRequested(first))
  current, _ = reducer.reduce(current, actions.ApprovalRequested(second))
  current, _ = reducer.reduce(current, actions.ApprovalRequested(third))
  current, _ = reducer.reduce(current, actions.MoveSelection(1))

  current, _ = reducer.reduce(current, actions.ClearApproval('second'))
  assert current.approval.requests == (first, third)
  assert current.selector.menu.selected == 1

  current, _ = reducer.reduce(current, actions.ClearApproval('first'))
  assert current.approval.request == third
  assert current.selector.menu.selected == 0

  current, _ = reducer.reduce(current, actions.ClearApproval('third'))
  assert current.approval == state.ApprovalState()


def test_secret_prompt_replaces_the_selector_and_keeps_only_its_length() -> (
  None
):
  picker = selection.Selector('token', 'Token', (), actions.SetModel)
  request = actions.SecretRequest('token', 'Paste a token', 'test/secret', str)
  opened, _ = reducer.reduce(_state(), actions.OpenSelector(picker))

  prompted, prompt_effects = reducer.reduce(
    opened, actions.RequestSecret(request)
  )
  typed, _ = reducer.reduce(prompted, actions.SecretLengthChanged(6))
  _, submit_effects = reducer.reduce(typed, actions.SubmitSecret())
  cleared, clear_effects = reducer.reduce(typed, actions.ClearSecret())

  assert not prompted.selector.active
  assert _effects.SyncSecretInput() in prompt_effects
  assert typed.secret == state.SecretState(request, 6)
  assert submit_effects == (_effects.SubmitSecret(),)
  assert not cleared.secret.active
  assert clear_effects == (_effects.SyncSecretInput(),)


@pytest.mark.parametrize(
  'action', [actions.CancelSecret(), actions.CancelOrQuitRequested()]
)
def test_cancelling_a_secret_prompt_runs_its_cancel_action(
  action: actions.Action,
) -> None:
  request = actions.SecretRequest(
    'token',
    'Paste a token',
    'test/secret',
    str,
    cancel_action=actions.ShowText('Cancelled.'),
  )
  prompted, _ = reducer.reduce(_state(), actions.RequestSecret(request))

  _, cancel_effects = reducer.reduce(prompted, action)

  assert cancel_effects == (_effects.CancelSecret(),)


def test_approval_selector_does_not_steal_another_selector() -> None:
  current, _ = reducer.reduce(_state(), actions.SelectPermissions())
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(
      actions.ApprovalRequest('call', 'write', 'test.txt')
    ),
  )
  assert current.selector.id == 'permissions'

  current, _ = reducer.reduce(current, actions.ClearSelector())
  assert current.selector.id == 'approval:call'
  assert current.selector.numbered


def test_stale_approval_selector_cannot_approve_the_next_call() -> None:
  current, _ = reducer.reduce(
    _state(),
    actions.ApprovalRequested(
      actions.ApprovalRequest('first', 'write', 'one.txt')
    ),
  )
  selector = current.selector.selector
  assert selector is not None
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(
      actions.ApprovalRequest('second', 'write', 'two.txt')
    ),
  )
  current, _ = reducer.reduce(current, actions.ClearApproval('first'))

  unchanged, effects = reducer.reduce(current, selector.on_select('approve'))
  assert unchanged == current
  assert effects == ()
  assert current.selector.id == 'approval:second'


def test_server_models_are_picked_only_in_remote_tabs() -> None:
  remote = dataclasses.replace(
    _state(), session=state.SessionState('conversation', remote=True)
  )
  pick = actions.SetRemoteModel('claude-opus-4-8', 'anthropic-messages')

  assert reducer.reduce(remote, pick)[1] == (
    _effects.SwitchRemoteModel('claude-opus-4-8', 'anthropic-messages'),
  )
  assert reducer.reduce(_state(), pick)[1] == (
    _effects.EmitError('server models are picked from a remote tab'),
  )
