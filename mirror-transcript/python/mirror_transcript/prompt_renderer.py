"""Retain prompt-toolkit layout and terminal policy; publish chrome, not diffs."""

from __future__ import annotations

import copy
import io
from collections.abc import Awaitable, Callable, Iterable
from typing import override

from mirror_transcript import transcript_pb2
from prompt_toolkit import application, cursor_shapes, renderer
from prompt_toolkit.layout import containers, layout, mouse_handlers, screen
from prompt_toolkit.output import vt100
from prompt_toolkit.styles import base

MARKER = 'class:mirror-transcript'


class Window(containers.Window):
  """A Python-owned allocation whose text and viewport live in the native child."""

  def __init__(
    self,
    view: str,
    publish: Callable[[], Iterable[transcript_pb2.TranscriptCommand]],
    routes: Callable[[], list[transcript_pb2.TranscriptRoute]],
    interactive: Callable[[], bool],
    *,
    canvas: Callable[[], str] = lambda: '',
    reconnect: Callable[[], None] = lambda: None,
  ) -> None:
    super().__init__(char=' ', style=MARKER, always_hide_cursor=True)
    self.view = view
    self.publish = publish
    self.routes = routes
    self.interactive = interactive
    self.canvas = canvas
    self.reconnect = reconnect
    self.closed = False


class Stream(io.TextIOBase):
  """Every active output write goes through the native writer, including OSC."""

  def __init__(
    self,
    original: vt100.Vt100_Output,
    send: Callable[[transcript_pb2.TranscriptCommand], int],
  ) -> None:
    self.original = original
    self.send = send
    self.discard = False
    self.direct = False
    self.pending: list[str] | None = None

  encoding = 'utf-8'

  @override
  def fileno(self) -> int:
    return self.original.fileno()

  @override
  def isatty(self) -> bool:
    return True

  @override
  def write(self, value: str) -> int:
    if self.discard:
      return len(value)
    if self.pending is not None:
      self.pending.append(value)
    elif self.direct:
      self.original.write_raw(value)
    elif value:
      self.send(
        transcript_pb2.TranscriptCommand(terminal=value.encode('utf-8'))
      )
    return len(value)

  @override
  def flush(self) -> None:
    if self.direct:
      self.original.flush()


class Renderer(renderer.Renderer):
  """Use the framework's screen, mouse handlers, styles, and lifecycle unchanged."""

  def __init__(
    self,
    previous: renderer.Renderer,
    output: vt100.Vt100_Output,
    stream: Stream,
    pause: Callable[[], Awaitable[None]],
    input_sequence: Callable[[], int],
  ) -> None:
    self.stream = stream
    self.pause = pause
    self.input_sequence = input_sequence
    self._views: dict[str, Window] = {}
    super().__init__(
      previous.style,
      output,
      full_screen=True,
      mouse_support=previous.mouse_support,
    )

  @override
  def request_absolute_cursor_position(self) -> None:
    # Fullscreen needs no CPR. The async framework barrier is used for handoff.
    return

  @override
  async def wait_for_cpr_responses(self, timeout: int = 1) -> None:
    del timeout
    await self.pause()

  @override
  def render[T](
    self,
    app: application.Application[T],
    layout: layout.Layout,
    is_done: bool = False,
  ) -> None:
    if is_done:
      self.reset()
      return
    if self.stream.direct:
      self.stream.direct = False
      self.stream.send(transcript_pb2.TranscriptCommand(pause=False))
    frame = copy.copy(layout)
    frame.container = _Chrome(layout.container, self._begin_layout)
    try:
      super().render(app, frame)
    finally:
      self.stream.discard = False
    rendered = self.last_rendered_screen
    if rendered is not None:
      self._publish(app, rendered)

  def _begin_layout(self) -> None:
    # Flush framework mode changes; its subsequent screen diff is not terminal I/O.
    self.output.flush()
    self.stream.discard = True

  def _publish[T](
    self, app: application.Application[T], rendered: screen.Screen
  ) -> None:
    size = self.output.get_size()
    buffer = io.StringIO()
    encoder = vt100.Vt100_Output(buffer, self.output.get_size)
    rows: list[bytes] = []
    overlays: list[bytes] = []
    occlusions: list[transcript_pb2.TranscriptRect] = []
    native = next(
      (
        (window, position)
        for window, position in rendered.visible_windows_to_write_positions.items()
        if isinstance(window, Window) and not window.closed
      ),
      None,
    )
    position = native[1] if native else None
    attrs: dict[str, base.Attrs] = {}
    for y in range(size.rows):
      buffer.seek(0)
      buffer.truncate()
      encoder.reset_attributes()
      last = None
      overlay = ''
      x = 0
      while x < size.columns:
        cell = rendered.data_buffer[y][x]
        if cell.style not in attrs:
          attrs[cell.style] = app.style_transformation.transform_attrs(
            self.style.get_attrs_for_style_str(cell.style)
          )
        value = attrs[cell.style]
        if value != last:
          encoder.set_attributes(value, app.color_depth)
          last = value
        if escape := rendered.zero_width_escapes[y].get(x):
          encoder.write_raw(escape)
        encoder.write(cell.char)
        if (
          position is not None
          and position.ypos <= y < position.ypos + position.height
          and position.xpos <= x < position.xpos + position.width
          and MARKER not in cell.style
        ):
          extra = io.StringIO()
          overlay_encoder = vt100.Vt100_Output(extra, self.output.get_size)
          overlay_encoder.cursor_goto(y + 1, x + 1)
          overlay_encoder.set_attributes(value, app.color_depth)
          overlay_encoder.write_raw(rendered.zero_width_escapes[y].get(x, ''))
          overlay_encoder.write(cell.char)
          overlay_encoder.flush()
          overlay += extra.getvalue()
          occlusions.append(
            transcript_pb2.TranscriptRect(
              x=x, y=y, width=max(1, cell.width), height=1
            )
          )
        x += max(1, cell.width)
      encoder.reset_attributes()
      encoder.flush()
      rows.append(buffer.getvalue().encode())
      overlays.append(overlay.encode())
    cursor = rendered.get_cursor_position(app.layout.current_window)
    shape = app.cursor.get_cursor_shape(app)
    frame = transcript_pb2.ChromeFrame(
      size=transcript_pb2.Size(columns=size.columns, rows=size.rows),
      rows=rows,
      cursor_x=min(cursor.x, size.columns - 1),
      cursor_y=min(cursor.y, size.rows - 1),
      cursor_visible=rendered.show_cursor,
      cursor_shape={
        cursor_shapes.CursorShape.BLOCK: 2,
        cursor_shapes.CursorShape.BEAM: 6,
        cursor_shapes.CursorShape.UNDERLINE: 4,
        cursor_shapes.CursorShape.BLINKING_BLOCK: 1,
        cursor_shapes.CursorShape.BLINKING_BEAM: 5,
        cursor_shapes.CursorShape.BLINKING_UNDERLINE: 3,
      }.get(shape, 0),
      input_sequence=self.input_sequence(),
      overlay_rows=overlays,
      occlusions=occlusions,
    )
    if native is not None:
      window, position = native
      if self._views.get(window.view) is not window:
        window.reconnect()
        self._views[window.view] = window
      for command in window.publish():
        match command.WhichOneof('payload'):
          case 'update' if not frame.HasField('update'):
            if command.update.view != window.view:
              raise ValueError('Transcript update targets another window')
            frame.update.CopyFrom(command.update)
          case 'navigate' if command.navigate.view == window.view:
            frame.moves.append(command.navigate.move)
          case _:
            raise ValueError('Expected one transcript update and navigation')
      frame.view = window.view
      frame.transcript.CopyFrom(
        transcript_pb2.TranscriptRect(
          x=position.xpos,
          y=position.ypos,
          width=position.width,
          height=position.height,
        )
      )
      frame.canvas = window.canvas()
      frame.wheel = window.interactive()
      frame.routes.extend(window.routes() if frame.wheel else ())
    self.stream.send(transcript_pb2.TranscriptCommand(frame=frame))
    for identity, window in tuple(self._views.items()):
      if window.closed:
        self.stream.send(transcript_pb2.TranscriptCommand(drop_view=identity))
        del self._views[identity]


class _Chrome(containers.DynamicContainer):
  def __init__(
    self, child: containers.Container, begin: Callable[[], None]
  ) -> None:
    super().__init__(lambda: child)
    self._begin = begin

  @override
  def write_to_screen(
    self,
    screen: screen.Screen,
    mouse_handlers: mouse_handlers.MouseHandlers,
    write_position: screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    self._begin()
    super().write_to_screen(
      screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
    )
