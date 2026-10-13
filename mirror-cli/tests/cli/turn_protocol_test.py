"""TUI turn state against the agent's input, interrupt, and closure protocol."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from mirror.cli import actions, effects, reducer, state


def _state() -> state.State:
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
  )


def _reduce(current: state.State, *steps: actions.Action) -> state.State:
  for step in steps:
    current, _ = reducer.reduce(current, step)
  return current


def _busy_with_acknowledged_queue() -> state.State:
  return _reduce(
    _state(),
    actions.TurnStarted(),
    actions.SubmitInstruction('status?'),
    actions.InputSubmitted('queued-1', 'status?'),
    actions.InputAcknowledged('queued-1'),
  )


def test_authoritative_input_before_ack_clears_optimistic_submission() -> None:
  current = _reduce(
    _state(),
    actions.InputSubmitted('first', 'hello'),
    actions.UserMessageObserved('hello', 'first'),
    actions.InputAcknowledged('first'),
    actions.InputRejected('first'),
  )
  assert current.turn.pending_instructions == ()
  assert current.turn.submitted_inputs == ()
  assert current.turn.awaiting_inputs == ()
  assert current.composer.text == ''


def test_ack_order_does_not_reorder_concurrent_submissions() -> None:
  current = _reduce(
    _state(),
    actions.InputSubmitted('first', 'one'),
    actions.InputSubmitted('second', 'two'),
    actions.InputAcknowledged('second'),
    actions.InputAcknowledged('second'),
    actions.InputSubmitted('first', 'one'),
  )
  assert current.turn.pending_instructions == ('one', 'two')
  current = _reduce(current, actions.WorkerFailed('connection lost'))
  assert current.turn.pending_instructions == ()
  assert current.composer.text == 'one\ntwo'


def test_unrelated_authoritative_text_does_not_consume_local_submission() -> (
  None
):
  current = _reduce(
    _state(),
    actions.TurnStarted(),
    actions.SubmitInstruction('same'),
    actions.InputSubmitted('local', 'same'),
    actions.UserMessageObserved('same', 'another-client'),
  )
  assert current.turn.pending_instructions == ('same',)
  assert current.turn.queued == ('same',)


def test_rejection_removes_only_its_optimistic_input_and_preserves_draft() -> (
  None
):
  current = _reduce(
    _state(),
    actions.InputSubmitted('first', 'one'),
    actions.InputSubmitted('second', 'two'),
    actions.RestoreDraft('new draft'),
    actions.InputRejected('first'),
    actions.InputRejected('first'),
    actions.InputAcknowledged('first'),
  )
  assert current.turn.pending_instructions == ('two',)
  assert current.composer.text == 'one\nnew draft'


def test_session_opened_restores_pending_inputs_without_carrying_them_over() -> (
  None
):
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.SubmitInstruction('follow up'),
    actions.InputSubmitted('queued-2', 'follow up'),
    actions.RestoreDraft('newer draft'),
  )

  opened, emitted = reducer.reduce(current, actions.SessionOpened('other'))

  assert opened.session.conversation_id == 'other'
  assert opened.turn == state.TurnState()
  assert opened.composer.text == 'status?\nfollow up\nnewer draft'
  assert emitted == (effects.SetComposerText(opened.composer.text),)
  assert (
    _reduce(
      opened,
      actions.InputAcknowledged('queued-2'),
      actions.InputRejected('queued-1'),
    )
    == opened
  )


def test_rejection_after_acknowledgment_releases_the_queued_instruction() -> (
  None
):
  current = _reduce(_busy_with_acknowledged_queue(), actions.TurnFinished())
  assert current.activity.busy

  current = _reduce(current, actions.InputRejected('queued-1'))

  assert current.turn.queued == ()
  assert current.turn.awaiting_inputs == ()
  assert not current.activity.busy
  assert current.composer.text == 'status?'


def test_saved_message_releases_its_acknowledged_input() -> None:
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.UserMessageObserved('status?', 'queued-1'),
    actions.TurnFinished(),
  )

  assert current.turn.queued == ()
  assert current.turn.awaiting_inputs == ()
  assert not current.activity.busy


def test_saved_message_resets_usage_and_releases_acknowledged_input() -> None:
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.RecordResponseUsage(534),
    actions.UserMessageObserved('status?', 'queued-1'),
  )

  assert current.turn.output_tokens == 0
  assert current.turn.queued == ()
  assert current.turn.awaiting_inputs == ()


def test_finished_turn_clears_usage_without_dropping_queue() -> None:
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.AgentTurnStarted(),
    actions.RecordResponseUsage(534),
    actions.TurnFinished(),
  )

  assert current.turn.output_tokens == 0
  assert current.turn.interrupt_id is None
  assert not current.turn.agent_turn
  assert current.turn.queued == ('status?',)
  assert current.turn.awaiting_inputs == (('queued-1', 'status?'),)
  assert current.activity.busy


def test_worker_failure_restores_acknowledged_queued_instruction() -> None:
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.WorkerFailed('agent service closed; reconnect'),
  )

  assert current.turn.queued == ()
  assert current.composer.text == 'status?'

  current = _reduce(current, actions.WorkerRetryFinished())

  assert not current.activity.busy


def test_declined_interrupt_ends_interrupting() -> None:
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.TurnFinished(),
    actions.InterruptRequested(),
    actions.InterruptSent('interrupt-1'),
  )
  assert current.activity.status == 'interrupting…'

  current = _reduce(current, actions.InterruptDeclined('interrupt-1'))

  assert not current.activity.busy
  assert current.activity.status == ''
  assert current.turn.interrupt_id is None


def test_declined_interrupt_for_an_earlier_request_is_ignored() -> None:
  current = _reduce(
    _busy_with_acknowledged_queue(),
    actions.InterruptRequested(),
    actions.InterruptSent('interrupt-2'),
  )

  updated = _reduce(current, actions.InterruptDeclined('interrupt-1'))

  assert updated == current


def test_finished_interrupt_does_not_label_the_next_agent_turn() -> None:
  current = _reduce(
    _state(),
    actions.AgentTurnStarted(),
    actions.InterruptRequested(),
    actions.InterruptSent('interrupt-1'),
    actions.TurnFinished(),
    actions.AgentTurnStarted(),
  )

  assert current.activity.status == 'thinking…'
  assert current.turn.interrupt_id is None


def test_finished_interrupt_clears_usage_and_agent_turn() -> None:
  current = _reduce(
    _state(),
    actions.AgentTurnStarted(),
    actions.RecordResponseUsage(534),
    actions.InterruptRequested(),
    actions.InterruptSent('interrupt-1'),
    actions.TurnFinished(),
  )

  assert current.turn.output_tokens == 0
  assert current.turn.interrupt_id is None
  assert not current.turn.agent_turn
  assert not current.activity.busy


def test_repaint_keeps_an_agent_started_turn_busy() -> None:
  current = _reduce(
    _state(),
    actions.RerenderStarted(),
    actions.AgentTurnStarted(),
    actions.RepaintFinished(True),
  )

  assert current.activity.busy

  current = _reduce(current, actions.TurnFinished())

  assert not current.activity.busy


def test_agent_started_turn_marks_an_idle_client_busy() -> None:
  current = _reduce(_state(), actions.AgentTurnStarted(observed_at=5.0))

  assert current.activity.busy
  assert current.activity.status == 'thinking…'
  assert current.activity.work_started == 5.0

  current = _reduce(current, actions.TurnFinished())

  assert not current.activity.busy


@pytest.mark.parametrize(
  'decision',
  [
    actions.DecideApproval(True),
    actions.DecideApproval(False),
    actions.ClearApproval('call'),
  ],
)
def test_approval_wait_is_excluded_from_work_time(
  decision: actions.Action,
) -> None:
  current, _ = reducer.reduce(_state(), actions.TurnStarted(10), observed_at=10)
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('call', 'bash', 'ls')),
    observed_at=12,
  )
  current, _ = reducer.reduce(
    current, actions.SetStatus('running tool…'), observed_at=42
  )
  assert current.activity.work_paused == 12
  assert current.activity.busy

  current, _ = reducer.reduce(current, decision, observed_at=72)
  assert current.activity.work_started == 70
  assert current.activity.work_paused is None
  assert current.approval.request is None
  assert current.activity.busy


def test_queued_approvals_pause_until_the_last_decision() -> None:
  current, _ = reducer.reduce(_state(), actions.TurnStarted(10), observed_at=10)
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('first', 'bash', 'ls')),
    observed_at=12,
  )
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('second', 'bash', 'pwd')),
    observed_at=22,
  )
  current, _ = reducer.reduce(
    current, actions.DecideApproval(True), observed_at=32
  )
  assert current.activity.work_paused == 12
  assert current.approval.request is not None
  assert current.approval.request.call_id == 'second'

  current, _ = reducer.reduce(
    current, actions.ClearApproval('second'), observed_at=42
  )
  assert current.activity.work_started == 40
  assert current.activity.work_paused is None

  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('third', 'bash', 'ls')),
    observed_at=52,
  )
  current, _ = reducer.reduce(
    current, actions.DecideApproval(False), observed_at=82
  )
  assert current.activity.work_started == 70
  assert current.activity.work_paused is None


def test_finished_turn_does_not_carry_approval_wait_into_next_turn() -> None:
  current, _ = reducer.reduce(_state(), actions.TurnStarted(10), observed_at=10)
  current, _ = reducer.reduce(
    current,
    actions.ApprovalRequested(actions.ApprovalRequest('call', 'bash', 'ls')),
    observed_at=12,
  )
  current, _ = reducer.reduce(current, actions.TurnFinished(), observed_at=90)
  assert current.activity.work_started is None
  assert current.activity.work_paused is None

  current, _ = reducer.reduce(
    current, actions.ClearApproval('call'), observed_at=92
  )
  current, _ = reducer.reduce(
    current, actions.TurnStarted(100), observed_at=100
  )
  assert current.activity.work_started == 100
  assert current.activity.work_paused is None


def test_work_started_during_approval_only_counts_time_after_decision() -> None:
  current, _ = reducer.reduce(
    _state(),
    actions.ApprovalRequested(actions.ApprovalRequest('call', 'bash', 'ls')),
    observed_at=12,
  )
  current, _ = reducer.reduce(
    current, actions.AgentTurnStarted(20), observed_at=20
  )
  current, _ = reducer.reduce(
    current, actions.DecideApproval(True), observed_at=40
  )
  assert current.activity.work_started == 40
  assert current.activity.work_paused is None


_Event = tuple[str, str, str]

_TEXTS = ('first', 'second')


@dataclasses.dataclass(frozen=True)
class _Agent:
  """A worker that follows the agent's input, interrupt, and closure rules."""

  requests: tuple[_Event, ...] = ()
  events: tuple[_Event, ...] = ()
  pending: tuple[tuple[str, str], ...] = ()
  in_turn: bool = False
  stale: bool = False
  closed: bool = False

  def serve(self) -> _Agent:
    (kind, message_id, text), requests = self.requests[0], self.requests[1:]
    if kind == 'input':
      return dataclasses.replace(
        self,
        requests=requests,
        events=(*self.events, ('ack', message_id, '')),
        pending=(*self.pending, (message_id, text)),
      )
    if self.in_turn:
      return dataclasses.replace(
        self,
        requests=requests,
        events=(
          *self.events,
          ('turn_interrupted', '', ''),
          ('turn_completed', '', ''),
        ),
        pending=(),
        in_turn=False,
      )
    return dataclasses.replace(
      self, requests=requests, events=(*self.events, ('nack', message_id, ''))
    )

  def persist_pending(self) -> _Agent:
    if self.stale:
      return self.fail(starting=not self.in_turn)
    durable = tuple(('durable', item, text) for item, text in self.pending)
    started = () if self.in_turn else (('turn_started', '', ''),)
    return dataclasses.replace(
      self,
      events=(*self.events, *durable, *started),
      pending=(),
      in_turn=True,
    )

  def finish(self) -> _Agent:
    if self.stale:
      return self.fail(starting=False)
    return dataclasses.replace(
      self,
      events=(*self.events, ('turn_completed', '', '')),
      in_turn=False,
    )

  def start_task_turn(self) -> _Agent:
    return dataclasses.replace(
      self, events=(*self.events, ('turn_started', '', '')), in_turn=True
    )

  def fail(self, *, starting: bool) -> _Agent:
    completed = (
      (('turn_completed', '', ''),) if starting or self.in_turn else ()
    )
    rejected = tuple(('nack', item, '') for item, _ in self.pending)
    unserved = tuple(
      ('nack', item, '') for kind, item, _ in self.requests if kind == 'input'
    )
    return _Agent(
      events=(
        *self.events,
        *completed,
        *rejected,
        *unserved,
        ('closed', '', ''),
      ),
      stale=True,
      closed=True,
    )

  @property
  def quiescent(self) -> bool:
    return (
      not self.requests
      and not self.events
      and (self.closed or (not self.in_turn and not self.pending))
    )


@dataclasses.dataclass(frozen=True)
class _World:
  """One reachable combination of TUI state, agent state, and used budgets."""

  client: state.State
  agent: _Agent = _Agent()
  failed: bool = False
  repainting: bool = False
  repaints: int = 0
  next_id: int = 0
  submissions: int = 0
  escapes: int = 0
  foreign_writes: int = 0
  task_results: int = 0
  retries: int = 0

  def key(self) -> tuple[object, ...]:
    activity = self.client.activity
    turn = self.client.turn
    return (
      activity.busy,
      activity.status,
      activity.work_started is None,
      turn.queued,
      turn.submitted_inputs,
      turn.awaiting_inputs,
      turn.interrupt_id,
      turn.agent_turn,
      self.client.composer.text,
      self.agent,
      self.failed,
      self.repainting,
      self.repaints,
      self.next_id,
      self.submissions,
      self.escapes,
      self.foreign_writes,
      self.task_results,
      self.retries,
    )


def _apply(
  world: _World, action: actions.Action
) -> tuple[_World, tuple[effects.Effect, ...]]:
  client, emitted = reducer.reduce(world.client, action)
  return dataclasses.replace(world, client=client), emitted


def _submit(world: _World) -> _World:
  text = _TEXTS[world.submissions]
  world = dataclasses.replace(world, submissions=world.submissions + 1)
  world, emitted = _apply(world, actions.SubmitInstruction(text))
  for effect in emitted:
    if isinstance(effect, effects.RunTurn):
      world, _ = _apply(world, actions.TurnStarted())
    if isinstance(effect, (effects.RunTurn, effects.SendQueuedInstruction)):
      message_id = f'input-{world.next_id}'
      world, _ = _apply(world, actions.InputSubmitted(message_id, text))
      agent = dataclasses.replace(
        world.agent,
        requests=(*world.agent.requests, ('input', message_id, text)),
      )
      world = dataclasses.replace(world, agent=agent, next_id=world.next_id + 1)
  return world


def _escape(world: _World) -> _World | None:
  pressed, emitted = _apply(world, actions.InterruptRequested())
  if effects.InterruptTurn() not in emitted:
    return None
  message_id = f'interrupt-{world.next_id}'
  pressed, _ = _apply(pressed, actions.InterruptSent(message_id))
  agent = dataclasses.replace(
    pressed.agent,
    requests=(*pressed.agent.requests, ('interrupt', message_id, '')),
  )
  return dataclasses.replace(
    pressed, agent=agent, next_id=world.next_id + 1, escapes=world.escapes + 1
  )


def _deliver(world: _World) -> _World:
  (kind, message_id, text), events = (
    world.agent.events[0],
    world.agent.events[1:],
  )
  world = dataclasses.replace(
    world, agent=dataclasses.replace(world.agent, events=events)
  )
  turn = world.client.turn
  tracked = (*turn.submitted_inputs, *turn.awaiting_inputs)
  match kind:
    case 'ack':
      return _apply(world, actions.InputAcknowledged(message_id))[0]
    case 'nack' if message_id == turn.interrupt_id:
      return _apply(world, actions.InterruptDeclined(message_id))[0]
    case 'nack' if any(item == message_id for item, _ in tracked):
      return _apply(world, actions.InputRejected(message_id))[0]
    case 'durable':
      return _apply(world, actions.UserMessageObserved(text, message_id))[0]
    case 'turn_started':
      return _apply(world, actions.AgentTurnStarted())[0]
    case 'turn_completed':
      return _apply(world, actions.TurnFinished())[0]
    case 'closed':
      world = dataclasses.replace(world, failed=True)
      return _apply(world, actions.WorkerFailed('agent service closed'))[0]
    case _:
      return world


def _successors(world: _World) -> Iterator[tuple[str, _World]]:
  agent = world.agent
  if not world.failed and world.submissions < len(_TEXTS):
    yield 'submit', _submit(world)
  if not world.failed and not world.escapes:
    pressed = _escape(world)
    if pressed is not None:
      yield 'escape', pressed
  if not agent.stale and not world.foreign_writes:
    yield (
      'foreign_write',
      dataclasses.replace(
        world,
        agent=dataclasses.replace(agent, stale=True),
        foreign_writes=1,
      ),
    )
  if (
    agent.requests and not agent.closed and (agent.in_turn or not agent.pending)
  ):
    yield 'serve', dataclasses.replace(world, agent=agent.serve())
  if agent.pending and not agent.closed:
    yield 'persist', dataclasses.replace(world, agent=agent.persist_pending())
  if agent.in_turn and not agent.pending and not agent.closed:
    yield 'finish', dataclasses.replace(world, agent=agent.finish())
  if (
    not agent.in_turn
    and not agent.pending
    and not agent.closed
    and not agent.stale
    and not world.task_results
  ):
    yield (
      'task_result',
      dataclasses.replace(world, agent=agent.start_task_turn(), task_results=1),
    )
  if not world.failed and not world.repaints and not world.client.activity.busy:
    repainted, _ = _apply(world, actions.RerenderStarted())
    yield 'repaint', dataclasses.replace(repainted, repainting=True, repaints=1)
  if world.repainting:
    painted, _ = _apply(world, actions.RepaintFinished(True))
    yield 'repaint done', dataclasses.replace(painted, repainting=False)
  if agent.events:
    yield f'deliver {agent.events[0][0]}', _deliver(world)
  if world.failed and not agent.events and not world.retries:
    retried, _ = _apply(world, actions.WorkerRetryFinished())
    yield (
      'retry',
      dataclasses.replace(retried, agent=_Agent(), failed=False, retries=1),
    )


def _violation(world: _World) -> str | None:
  activity = world.client.activity
  turn = world.client.turn
  if activity.busy and activity.status == 'interrupting…':
    if turn.interrupt_id is None:
      return 'shows interrupting with no interrupt outstanding'
  agent = world.agent
  if agent.in_turn and not agent.closed and not agent.events:
    if not activity.busy:
      return 'idle while the agent runs a turn'
  if not agent.quiescent or world.repainting:
    return None
  if activity.busy:
    return 'busy after the agent went quiet'
  if turn.queued or turn.awaiting_inputs:
    return 'inputs left unresolved after the agent went quiet'
  return None


def test_every_bounded_schedule_settles_once_the_agent_is_quiet() -> None:
  seen: set[tuple[object, ...]] = set()
  stack: list[tuple[_World, tuple[str, ...]]] = [(_World(_state()), ())]
  while stack:
    world, trace = stack.pop()
    key = world.key()
    if key in seen:
      continue
    seen.add(key)
    violation = _violation(world)
    assert violation is None, f'{violation}: {" -> ".join(trace)}'
    stack.extend(
      (successor, (*trace, label)) for label, successor in _successors(world)
    )
  assert len(seen) > 5_000
