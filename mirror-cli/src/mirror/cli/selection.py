"""Declarative choice selectors for CLI extensions."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Sequence

from prompt_toolkit import formatted_text

from mirror.cli.components import fuzzy_score


def fuzzy_matches[Item](
  items: Sequence[Item], query: str, terms: Callable[[Item], Sequence[str]]
) -> tuple[Item, ...]:
  """Rank each item by its best matching term with stable ties."""
  if not query:
    return tuple(items)
  candidates = tuple(
    (index, term) for index, item in enumerate(items) for term in terms(item)
  )
  ranked = fuzzy_score.ranked_indices(
    tuple(term for _, term in candidates), query
  )
  indices = dict.fromkeys(candidates[index][0] for index in ranked)
  return tuple(items[index] for index in indices)


@dataclasses.dataclass(frozen=True)
class Choice:
  """One value offered by a selector."""

  value: str
  label: str
  detail: str = ''
  # Resolve callable prefixes during rendering so animations stay live.
  prefix: formatted_text.AnyFormattedText = ''
  prefix_style: str = ''
  # None keeps the label/detail layout; a string adds a middle status column.
  status: str | None = None
  # Presentation only: headings never become selectable values.
  section: str = ''


@dataclasses.dataclass(frozen=True)
class Selector[ActionT]:
  """One selector an extension wants the CLI to show."""

  id: str
  title: str
  choices: tuple[Choice, ...]
  on_select: Callable[[str], ActionT]
  extension_id: str | None = None
  current: str | None = None
  selected: str | None = None
  empty: str = 'No choices available.'
  keep_open: bool = False
  # Replace search with 1–9 shortcuts for the first nine choices.
  numbered: bool = False
  # Async producers supply ranked choices via SelectorResultsLoaded.
  prefiltered: bool = False
  loading: bool = False
  # Optional one-shot async producer. When set, the CLI fetches the choices as
  # the selector opens and feeds them back through SelectorChoicesLoaded, so a
  # live-populated selector needs no bespoke action or effect of its own.
  load: Callable[[], Awaitable[tuple[Choice, ...]]] | None = None
  cancel_action: ActionT | None = None
  # When set, Left and Right call this with the highlighted choice's value and
  # -1 or 1, so a choice can step through values without another selector.
  on_adjust: Callable[[str, int], ActionT] | None = None
