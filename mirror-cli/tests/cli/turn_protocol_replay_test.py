"""TUI reducer against the client schedules TLC generates from LoopClient."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from mirror.cli import actions, effects, reducer, state

_TRACES = pathlib.Path(__file__).parent / 'testdata' / 'loop_client'
_TEXTS = {1: 'first', 2: 'second'}
_INTERRUPT_ID = 'interrupt'
_SERVICE_ACTIONS = frozenset({
  'Acknowledge',
  'Boundary',
  'Commit',
  'ForeignAppend',
  'Init',
  'ModelResponse0',
  'RefuseAppend',
  'SaveAssistant',
  'ServeInput',
  'ServeInterrupt',
  'StartInput',
  'TaskResult',
})


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


def _message_id(model_id: int) -> str:
  return f'input-{model_id}'


def _model_ids(inputs: tuple[tuple[str, str], ...]) -> list[int]:
  return sorted(int(item.removeprefix('input-')) for item, _ in inputs)


def _projection(current: state.State) -> dict[str, Any]:
  turn = current.turn
  model_ids = {text: model_id for model_id, text in _TEXTS.items()}
  return {
    'busy': current.activity.busy,
    'interrupting': turn.interrupt_id is not None,
    'failed': current.lifecycle.failure is not None,
    'queue': sorted(model_ids[text] for text in turn.queued),
    'unacked': _model_ids(turn.submitted_inputs),
    'awaiting': _model_ids(turn.awaiting_inputs),
  }


def _submit(current: state.State, model_id: int) -> state.State:
  text = _TEXTS[model_id]
  current, emitted = reducer.reduce(current, actions.SubmitInstruction(text))
  if effects.RunTurn(text) in emitted:
    current = _reduce(current, actions.TurnStarted())
  else:
    assert effects.SendQueuedInstruction(text) in emitted
  return _reduce(current, actions.InputSubmitted(_message_id(model_id), text))


def _interrupt(current: state.State) -> state.State:
  current, emitted = reducer.reduce(current, actions.InterruptRequested())
  assert effects.InterruptTurn() in emitted
  return _reduce(current, actions.InterruptSent(_INTERRUPT_ID))


def _nack(current: state.State, message_id: str) -> state.State:
  turn = current.turn
  if message_id == turn.interrupt_id:
    return _reduce(current, actions.InterruptDeclined(message_id))
  tracked = (*turn.submitted_inputs, *turn.awaiting_inputs)
  if any(item == message_id for item, _ in tracked):
    return _reduce(current, actions.InputRejected(message_id))
  return current


def _deliver(current: state.State, event: dict[str, Any]) -> state.State:
  match event['type']:
    case 'ack':
      return _reduce(
        current, actions.InputAcknowledged(_message_id(event['id']))
      )
    case 'durable':
      return _reduce(
        current,
        actions.UserMessageObserved(
          _TEXTS[event['id']], _message_id(event['id'])
        ),
      )
    case 'nack':
      return _nack(current, _message_id(event['id']))
    case 'no_turn':
      return _nack(current, _INTERRUPT_ID)
    case 'turn_started':
      return _reduce(current, actions.AgentTurnStarted())
    case 'turn_completed':
      return _reduce(current, actions.TurnFinished())
    case 'closed':
      return _reduce(
        current, actions.WorkerFailed('agent service closed; reconnect')
      )
    case unknown:
      raise ValueError(f'unknown replay event {unknown!r}')


def _step(current: state.State, step: dict[str, Any]) -> state.State:
  match step['action']:
    case action if action in _SERVICE_ACTIONS:
      return current
    case 'Submit1':
      return _submit(current, 1)
    case 'Submit2':
      return _submit(current, 2)
    case 'Interrupt':
      return _interrupt(current)
    case 'Retry':
      return _reduce(
        current,
        actions.WorkerRetryStarted(),
        actions.WorkerStarted(),
        actions.WorkerRetryFinished(),
      )
    case 'Deliver':
      return _deliver(current, step['event'])
    case unknown:
      raise ValueError(f'unknown replay action {unknown!r}')


@pytest.mark.parametrize(
  'scenario', ['incident_fixed', 'idle_interrupt', 'task_turn']
)
def test_reducer_follows_the_modeled_client(scenario: str) -> None:
  trace = json.loads((_TRACES / f'{scenario}.json').read_text())
  assert trace['scenario'] == scenario

  current = _state()
  for index, step in enumerate(trace['steps']):
    current = _step(current, step)
    assert _projection(current) == step['client'], (index, step['action'])
