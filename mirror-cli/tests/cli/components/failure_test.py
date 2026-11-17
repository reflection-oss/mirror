import pytest
from prompt_toolkit.layout import containers, mouse_handlers, screen

from mirror.cli import actions, reducer, state
from mirror.cli.components import failure


@pytest.fixture
def snapshot() -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('test'),
    lifecycle=state.LifecycleState(active=True),
  )


def _render(snapshot: state.State, width: int) -> str:
  container = containers.to_container(failure.Failure().render(snapshot))
  height = container.preferred_height(width, 20).preferred
  target = screen.Screen()
  container.write_to_screen(
    target,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, width, height),
    '',
    True,
    None,
  )
  rows: list[str] = []
  for row in range(height):
    rows.append(
      ''.join(target.data_buffer[row][column].char for column in range(width))
    )
  return ' '.join(''.join(rows).split())


@pytest.mark.parametrize('width', [40, 100])
@pytest.mark.parametrize(
  ('action', 'expected'),
  [
    (
      actions.WorkerReconnecting(
        'Daemon event buffer overflowed; the frontend missed updates.'
      ),
      'Connection lost. Reconnecting…',
    ),
    (
      actions.WorkerReconnecting('Session lagged', synchronizing=True),
      'Synchronizing session state…',
    ),
    (
      actions.WorkerFailed(
        'AgentRPC UNAVAILABLE: daemon unreachable', disconnected=True
      ),
      "Couldn't reconnect. Press Enter to retry.",
    ),
    (
      actions.WorkerFailed('UNAUTHENTICATED: sign in to the provider'),
      'Agent error: UNAUTHENTICATED: sign in to the provider Press Enter to retry.',
    ),
  ],
)
def test_render_explains_failure_and_recovery(
  snapshot: state.State, action: actions.Action, expected: str, width: int
) -> None:
  snapshot, _ = reducer.reduce(snapshot, action)
  assert _render(snapshot, width) == expected


def test_render_healthy_agent_has_no_notice(snapshot: state.State) -> None:
  container = containers.to_container(failure.Failure().render(snapshot))
  assert container.preferred_height(80, 20).preferred == 0


def test_cache_key_tracks_failure_details(snapshot: state.State) -> None:
  component = failure.Failure()
  first, _ = reducer.reduce(snapshot, actions.WorkerFailed('worker exited'))
  second, _ = reducer.reduce(first, actions.WorkerFailed('invalid model'))
  assert component.cache_key(first) != component.cache_key(second)
