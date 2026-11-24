"""The hint in a selector's search line while it filters or searches by typing."""

import dataclasses

import pytest

from mirror.cli import actions, selection, state
from mirror.cli.components import selector


def _state(query: str = '') -> state.State:
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
  )
  picker = selection.Selector[actions.Action](
    'test',
    'Test',
    (),
    lambda _: actions.ClearSelector(),
    placeholder='Type to filter',
  )
  return dataclasses.replace(
    current, selector=state.SelectorState(selector=picker, query=query)
  )


def test_search_hint_goes_away_once_the_user_types() -> None:
  assert selector.search_hint(_state(query='ab')) is None


@pytest.mark.asyncio
async def test_search_input_shows_the_hint_while_empty() -> None:
  current = _state()
  search = selector.SearchInput(
    lambda _: None, lambda: selector.search_hint(current)
  )

  line = search.control.create_content(width=40, height=1).get_line(0)

  assert ''.join(text for _, text, *_ in line).strip() == '› Type to filter'
