import io
from typing import TextIO, cast

import pytest
from mirror_transcript import prompt_renderer, transcript_pb2
from prompt_toolkit import application, data_structures
from prompt_toolkit.application import current
from prompt_toolkit.input import base
from prompt_toolkit.layout import containers, layout, scrollable_pane
from prompt_toolkit.output import vt100


@pytest.mark.parametrize('rows, columns', [(6, 10), (3, 20), (12, 40)])
def test_render_resize_during_layout_keeps_frame_geometry_consistent(
  rows: int, columns: int
) -> None:
  size = data_structures.Size(rows=6, columns=20)
  commands: list[transcript_pb2.TranscriptCommand] = []

  def send(command: transcript_pb2.TranscriptCommand) -> int:
    commands.append(command)
    return len(commands)

  async def pause() -> None:
    pass

  def resize() -> str:
    nonlocal size
    size = data_structures.Size(rows=rows, columns=columns)
    return prompt_renderer.MARKER

  window = prompt_renderer.Window('test', list, list, lambda: True)
  window.style = resize
  original = vt100.Vt100_Output(io.StringIO(), lambda: size)
  stream = prompt_renderer.Stream(original, send)
  output = vt100.Vt100_Output(cast(TextIO, stream), lambda: size)
  app = application.Application[None](
    layout=layout.Layout(window),
    input=base.DummyInput(),
    output=output,
    full_screen=True,
  )
  app.renderer = prompt_renderer.Renderer(
    app.renderer, output, stream, pause, lambda: 0
  )

  with current.set_app(app):
    app.renderer.render(app, app.layout)
    frame = commands[-1].frame
    assert frame.size == transcript_pb2.Size(rows=6, columns=20)
    assert len(frame.rows) == 6
    assert frame.transcript == transcript_pb2.TranscriptRect(width=20, height=6)

    app.renderer.render(app, app.layout)
    frame = commands[-1].frame
    assert frame.size == transcript_pb2.Size(rows=rows, columns=columns)
    assert len(frame.rows) == rows
    assert frame.transcript == transcript_pb2.TranscriptRect(
      width=columns, height=rows
    )


@pytest.mark.parametrize(
  'scroll, width, height', [(0, 20, 6), (5, 20, 5), (10, 0, 0)]
)
def test_render_clips_virtual_canvas_to_chrome(
  scroll: int, width: int, height: int
) -> None:
  size = data_structures.Size(rows=6, columns=20)
  commands: list[transcript_pb2.TranscriptCommand] = []

  def send(command: transcript_pb2.TranscriptCommand) -> int:
    commands.append(command)
    return len(commands)

  async def pause() -> None:
    pass

  window = prompt_renderer.Window('test', list, list, lambda: True)
  window.height = 10
  pane = scrollable_pane.ScrollablePane(
    containers.HSplit([window, containers.Window(height=10)]),
    show_scrollbar=False,
    keep_cursor_visible=False,
    keep_focused_window_visible=False,
  )
  pane.vertical_scroll = scroll
  original = vt100.Vt100_Output(io.StringIO(), lambda: size)
  stream = prompt_renderer.Stream(original, send)
  output = vt100.Vt100_Output(cast(TextIO, stream), lambda: size)
  app = application.Application[None](
    layout=layout.Layout(pane),
    input=base.DummyInput(),
    output=output,
    full_screen=True,
  )
  app.renderer = prompt_renderer.Renderer(
    app.renderer, output, stream, pause, lambda: 0
  )

  with current.set_app(app):
    app.renderer.render(app, app.layout)
  frame = commands[-1].frame
  assert frame.HasField('transcript') == (height > 0)
  assert frame.transcript == transcript_pb2.TranscriptRect(
    width=width, height=height
  )
