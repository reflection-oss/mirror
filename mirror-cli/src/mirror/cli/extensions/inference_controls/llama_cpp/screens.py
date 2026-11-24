"""Screens for managing llama.cpp, kept current by the extension that shows them."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncGenerator, Awaitable, Callable

from rich import console, text

from mirror.cli import actions, extension, selection
from mirror.cli.components import card

EXTENSION_ID = 'core/inference-controls'
# A row any screen may show after a failure; choosing it runs the screen again.
REFRESH_ROW = ':refresh'
DEBOUNCE_SECONDS = 0.25


@dataclasses.dataclass(frozen=True)
class Update:
  """Rows to show, and the context above them when it changes."""

  choices: tuple[selection.Choice, ...]
  header: str | None = None


@dataclasses.dataclass(frozen=True)
class Screen:
  """A selector and the producer that keeps its rows current.

  The producer receives the typed query. It runs again when a `prefiltered`
  selector's query changes, so it ranks and searches for itself; otherwise it
  runs once and keeps yielding while the screen is open.
  """

  selector: selection.Selector[actions.Action]
  stream: Callable[[str], AsyncGenerator[Update]] | None = None


class Step(actions.ExtensionAction):
  """What a row asks the extension to do."""


@dataclasses.dataclass(frozen=True)
class Open(Step):
  """Show another screen."""

  screen: Screen


@dataclasses.dataclass(frozen=True)
class Do(Step):
  """Run an operation, then show `then`; without it the screen stays as is."""

  work: Callable[[], Awaitable[object]]
  then: Screen | None = None


@dataclasses.dataclass(frozen=True)
class Load(Step):
  """Load a model on its server, then switch the session to it."""

  name: str
  base: str


@dataclasses.dataclass(frozen=True)
class Edit(Step):
  """Open the per-model settings file in the user's editor."""

  path: str


@dataclasses.dataclass(frozen=True)
class Quit(Step):
  """Quit, stopping the llama.cpp server first when asked."""

  stop: bool


@dataclasses.dataclass(frozen=True)
class Refresh(Step):
  """Run the open screen's producer again, or stop it once the screen closed."""


def step(action: Step) -> actions.Action:
  """Route a step to the extension that handles it."""
  return actions.RoutedExtensionAction(EXTENSION_ID, action)


def goto(screen: Screen) -> actions.Action:
  """The action that shows `screen`."""
  return step(Open(screen))


def perform(
  work: Callable[[], Awaitable[object]], then: Screen | None = None
) -> actions.Action:
  """The action that runs `work`, then shows `then` if there is one."""
  return step(Do(work, then))


def _framed(header: str) -> console.RenderableType:
  return card.frame(text.Text(header, style='ui.menu-text')) if header else ''


class Runner:
  """Shows one screen at a time and keeps its rows current."""

  def __init__(self, debounce_seconds: float = DEBOUNCE_SECONDS) -> None:
    self._debounce_seconds = debounce_seconds
    self._task: asyncio.Task[None] | None = None
    self._refresh = asyncio.Event()

  def refresh(self) -> None:
    """Wake the producer: it runs again while its screen is open, else stops."""
    self._refresh.set()

  async def show(self, context: extension.Context, screen: Screen) -> None:
    """Replace the open screen, and keep feeding it until it closes."""
    this = asyncio.current_task()
    previous, self._task = self._task, this
    if previous is not None and previous is not this:
      previous.cancel()
      await asyncio.gather(previous, return_exceptions=True)
    selector = screen.selector

    def on_select(value: str) -> actions.Action:
      if value == REFRESH_ROW:
        return step(Refresh())
      return selector.on_select(value)

    context.dispatch(
      actions.OpenSelector(
        dataclasses.replace(
          selector,
          extension_id=EXTENSION_ID,
          wrap_details=True,
          loading=screen.stream is not None,
          header=_framed(selector.header)
          if isinstance(selector.header, str)
          else selector.header,
          on_select=on_select,
        )
      )
    )
    try:
      if screen.stream is not None:
        await self._keep_current(context, screen)
    finally:
      if self._task is this:
        self._task = None

  async def _keep_current(
    self, context: extension.Context, screen: Screen
  ) -> None:
    assert screen.stream is not None
    selector = screen.selector
    while context.get_state().selector.id == selector.id:
      self._refresh.clear()
      query = context.get_state().selector.query
      if query and selector.prefiltered:
        try:
          async with asyncio.timeout(self._debounce_seconds):
            await self._refresh.wait()
        except TimeoutError:
          pass
        else:
          continue
      producing = asyncio.create_task(self._publish(context, screen, query))
      refreshing = asyncio.create_task(self._refresh.wait())
      try:
        await asyncio.wait(
          {producing, refreshing}, return_when=asyncio.FIRST_COMPLETED
        )
        if producing.done():
          producing.result()
          if context.get_state().selector.id != selector.id:
            return
        await refreshing
      finally:
        producing.cancel()
        refreshing.cancel()
        await asyncio.gather(producing, refreshing, return_exceptions=True)

  async def _publish(
    self, context: extension.Context, screen: Screen, query: str
  ) -> None:
    assert screen.stream is not None
    try:
      async with contextlib.aclosing(screen.stream(query)) as updates:
        async for update in updates:
          if not self._show_rows(context, screen, query, update):
            return
    except (ValueError, OSError) as error:
      self._show_rows(
        context,
        screen,
        query,
        Update((selection.Choice(REFRESH_ROW, 'Retry', str(error)),)),
      )

  def _show_rows(
    self,
    context: extension.Context,
    screen: Screen,
    query: str,
    update: Update,
  ) -> bool:
    """Show `update` if the screen is still open, and say whether it is."""
    selector = screen.selector
    if context.get_state().selector.id != selector.id:
      return False
    if selector.prefiltered:
      context.dispatch(
        actions.SelectorResultsLoaded(selector.id, query, update.choices, False)
      )
    else:
      context.dispatch(
        actions.SelectorChoicesLoaded(
          selector.id,
          update.choices,
          None if update.header is None else _framed(update.header),
        )
      )
    return True
