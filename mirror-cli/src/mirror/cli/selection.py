"""Declarative choice selectors for CLI extensions."""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Sequence

from prompt_toolkit import formatted_text
from rich import console

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
  # A trailing column when the fixed-width row has room.
  shortcut: str = ''
  section_label: str = ''
  """Optional display label; `section` remains the grouping and search key."""
  is_heading: bool = False
  """A selectable section heading, replacing the automatic group heading."""
  search_terms: tuple[str, ...] = ()
  """Searchable metadata that need not be repeated in the visible row."""


@dataclasses.dataclass(frozen=True)
class Selector[ActionT]:
  """One selector an extension wants the CLI to show."""

  id: str
  title: formatted_text.AnyFormattedText
  choices: tuple[Choice, ...]
  on_select: Callable[[str], ActionT]
  extension_id: str | None = None
  current: str | None = None
  selected: str | None = None
  empty: str = 'No choices available.'
  keep_open: bool = False
  # Replace search with 1–9 shortcuts, using pinned actions when present.
  numbered: bool = False
  # Keep trailing actions visible outside the scrolling choices' row limit.
  pinned_count: int = 0
  # Wrap row details within the scrolling viewport's physical line budget.
  wrap_details: bool = False
  # Put each description below its label, sharing the bounded viewport.
  stacked: bool = False
  footer: str | None = None
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
  # Context above the choices, scrollable in a bounded viewport.
  header: console.RenderableType = ''
  # Preserve filenames or other meaningful suffixes when labels are too wide.
  truncate_labels_from_start: bool = False

  @property
  def numbered_from(self) -> int:
    """Number pinned actions when present, otherwise number the whole list."""
    return len(self.choices) - self.pinned_count if self.pinned_count else 0
