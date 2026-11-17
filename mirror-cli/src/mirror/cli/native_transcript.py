"""Python owns transcript membership and order; native code owns its viewport."""

from __future__ import annotations

import itertools
import uuid
from collections.abc import Callable, Iterable, Sequence

from mirror_transcript import (
  client,
  model,
  prompt_renderer,
  transcript_pb2,
)
from prompt_toolkit import mouse_events
from prompt_toolkit.application import current
from prompt_toolkit.key_binding import key_bindings
from rich import console

from mirror.cli import (
  actions,
  rendering,
  terminal,
  themes,
  transcript,
  transcript_content,
  workspace_application,
)


def movement(action: actions.NavigateHistory) -> transcript_pb2.TranscriptMove:
  """Translate the navigation action already selected by Python's keymap."""
  kind, amount = {
    'page_up': (transcript_pb2.TranscriptMove.PAGE, -1),
    'page_down': (transcript_pb2.TranscriptMove.PAGE, 1),
    'oldest': (transcript_pb2.TranscriptMove.FIRST, 0),
    'latest': (transcript_pb2.TranscriptMove.FOLLOW, 0),
    'up': (transcript_pb2.TranscriptMove.ROWS, -action.rows),
    'down': (transcript_pb2.TranscriptMove.ROWS, action.rows),
  }[action.direction]
  return transcript_pb2.TranscriptMove(kind=kind, amount=amount)


class View:
  """Incremental source projection with stable identities and explicit settlement."""

  def __init__(
    self,
    source: Callable[[], tuple[transcript.Block, ...]],
    live: Callable[[], Sequence[console.RenderableType]],
    *,
    navigate: Callable[[actions.NavigateHistory], object],
    pending: Callable[[], Sequence[transcript.Block]],
    theme_name: Callable[[], str | themes.Theme],
  ) -> None:
    self.id = uuid.uuid4().hex
    self.reading = False
    self._source, self._live, self._pending = source, live, pending
    self._navigate = navigate
    self._theme_name = theme_name
    self._ids: dict[str, int] = {}
    self._next_id = itertools.count(1)
    self._sent: dict[str, tuple[object, object]] = {}
    self._order: tuple[int, ...] | None = None
    self._styles: dict[str, str] = {}
    self._settlements: list[transcript_pb2.TranscriptSettlement] = []
    self._moves: list[transcript_pb2.TranscriptCommand] = []
    self._live_count = 0
    self._revision = 0
    self._reset = False
    self._window: prompt_renderer.Window | None = None

  def _identity(self, key: str) -> int:
    if key not in self._ids:
      self._ids[key] = next(self._next_id)
    return self._ids[key]

  def window(
    self,
    routes: Callable[[], list[transcript_pb2.TranscriptRoute]],
    interactive: Callable[[], bool],
  ) -> prompt_renderer.Window:
    """Reserve the same Python layout slot formerly occupied by transcript rows."""
    self._window = prompt_renderer.Window(
      self.id,
      self.publish,
      routes,
      interactive,
      canvas=lambda: self._styles.get('canvas', ''),
      reconnect=self._reconnect,
    )
    return self._window

  def publish(self) -> Iterable[transcript_pb2.TranscriptCommand]:
    """Preserve exactly settled → pending → live order, never infer it from IDs."""
    depth = current.get_app().color_depth
    styles = transcript_content.palette(self._theme_name(), depth)
    projection = transcript_content.Projection(styles, depth)
    settled = self._source()
    pending = self._pending()
    live = self._live()
    sources = [*settled, *pending]
    sources.extend(
      transcript.Block(f'live:{index}', (0, 0), value)
      for index, value in enumerate(live)
    )
    changed: list[transcript_pb2.Element] = []
    sent: dict[str, tuple[object, object]] = {}
    ids: list[int] = []
    for block in sources:
      identity = self._identity(block.key)
      ids.append(identity)
      stamp = (block.revision, block.renderable)
      sent[block.key] = stamp
      if self._sent.get(block.key) == stamp and styles == self._styles:
        continue
      source = block.source
      blocks = projection.blocks(source) if source is not None else []
      blocks.append(model.Block(lines=[[]]))
      sticky = (
        model.Span(
          rendering.icon_prefix(rendering.PROMPT)
          + ' '.join(source.content.split()),
          styles['message'],
        )
        if isinstance(source, rendering.UserText)
        else None
      )
      changed.append(
        client.element(model.Element(identity, blocks=blocks, sticky=sticky))
      )
    order = tuple(ids)
    if (
      changed
      or order != self._order
      or styles != self._styles
      or self._settlements
      or self._reset
    ):
      self._revision += 1
      yield transcript_pb2.TranscriptCommand(
        update=transcript_pb2.TranscriptUpdate(
          view=self.id,
          revision=self._revision,
          set=changed,
          order=transcript_pb2.TranscriptOrder(ids=ids)
          if order != self._order
          else None,
          theme=transcript_pb2.Theme(
            text=styles['text'],
            muted=styles['muted'],
            accent=styles['accent'],
            roles=styles,
          )
          if styles != self._styles
          else None,
          settlements=self._settlements,
          reset=self._reset,
        )
      )
    self._ids = {key: self._ids[key] for key in sent}
    self._sent, self._order, self._styles = sent, order, styles
    self._live_count = len(live)
    self._settlements = []
    self._reset = False
    yield from self._moves
    self._moves.clear()

  def observe(self, event: transcript_pb2.TranscriptEvent) -> None:
    """Reading state is a notification, never permission for native input or paint."""
    if (
      event.WhichOneof('payload') == 'position'
      and event.position.view == self.id
      and self.reading != event.position.reading
    ):
      self.reading = event.position.reading
      current.get_app().invalidate()
    if (
      event.WhichOneof('payload') == 'action'
      and event.action.view == self.id
      and event.action.revision == self._revision
    ):
      match event.action.WhichOneof('action'):
        case 'copy':
          try:
            workspace_application.copy_text(event.action.copy)
          except ValueError:
            current.get_app().output.bell()
        case 'link':
          current.get_app().create_background_task(
            terminal.open_url(event.action.link)
          )
        case _:
          pass

  def settle_live(self, identities: str | Sequence[str]) -> None:
    """Remap live identities in the same native commit that publishes final sources."""
    targets = (
      [identities] * self._live_count
      if isinstance(identities, str)
      else identities
    )
    self._settlements.extend(
      transcript_pb2.TranscriptSettlement(
        view=self.id,
        before=self._identity(f'live:{index}'),
        after=self._identity(key),
      )
      for index, key in enumerate(targets)
    )

  def navigate(self, action: actions.NavigateHistory) -> None:
    self._moves.append(
      transcript_pb2.TranscriptCommand(
        navigate=transcript_pb2.TranscriptNavigate(
          view=self.id, move=movement(action)
        )
      )
    )
    if action.direction == 'latest':
      self.reading = False
    current.get_app().invalidate()

  def follow(self) -> None:
    self.navigate(actions.NavigateHistory('latest'))

  def refresh(self) -> bool:
    current.get_app().invalidate()
    return True

  def _reconnect(self) -> None:
    self._revision += 1
    self.reading = False
    self._sent.clear()
    self._order = None
    self._styles = {}
    self._reset = True

  def clear(self) -> None:
    self._reconnect()
    self._ids.clear()
    self._moves.clear()
    self._settlements.clear()

  async def close(self) -> None:
    self.clear()
    if self._window is not None:
      self._window.closed = True
    current.get_app().invalidate()

  def mouse_handler(
    self, event: mouse_events.MouseEvent
  ) -> key_bindings.NotImplementedOrNone:
    if event.event_type == mouse_events.MouseEventType.SCROLL_UP:
      self._navigate(actions.NavigateHistory('up', 1))
    elif event.event_type == mouse_events.MouseEventType.SCROLL_DOWN:
      self._navigate(actions.NavigateHistory('down', 1))
    else:
      return NotImplemented
    return None
