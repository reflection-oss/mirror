"""Real prompt-toolkit lifecycle exercised in an isolated terminal host."""

import asyncio
import os
import pathlib
import sys
from typing import override

from mirror_transcript import (
  prompt_application,
  prompt_renderer,
  transcript_pb2,
)
from prompt_toolkit import application, buffer
from prompt_toolkit.application import current, run_in_terminal
from prompt_toolkit.input import vt100 as vt100_input
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import vt100


def document(prefix: str = '') -> list[transcript_pb2.TranscriptCommand]:
  return [
    transcript_pb2.TranscriptCommand(
      update=transcript_pb2.TranscriptUpdate(
        view='test',
        set=[
          transcript_pb2.Element(
            id=1,
            lines=[
              transcript_pb2.Line(
                spans=[transcript_pb2.Span(text=f'{prefix}{index}')]
              )
              for index in range(40)
            ],
          )
        ],
        order=transcript_pb2.TranscriptOrder(ids=[1]),
      )
    )
  ]


class SynchronizedApplication(application.Application[None]):
  """Mirror's redraw wrapper also writes outside the renderer's render call."""

  @override
  def _redraw(self, render_as_done: bool = False) -> None:
    self.output.write_raw('\x1b[?2026h')
    try:
      super()._redraw(render_as_done)
    finally:
      self.output.write_raw('\x1b[?2026l')
      self.output.flush()


async def main() -> None:
  with (
    open(sys.argv[1], 'r') as stdin,  # noqa: ASYNC230, preallocated local PTY.
    open(sys.argv[1], 'w') as stdout,  # noqa: ASYNC230, preallocated local PTY.
  ):
    input = vt100_input.Vt100Input(stdin)
    output = vt100.Vt100_Output.from_pty(stdout)
    bindings = key_bindings.KeyBindings()
    draft = buffer.Buffer(multiline=True)
    editor = containers.Window(controls.BufferControl(draft), height=1)
    window = prompt_renderer.Window('test', document, list, lambda: True)
    application_type = (
      SynchronizedApplication
      if 'synchronized' in sys.argv[3:]
      else application.Application[None]
    )
    app = application_type(
      layout=layout.Layout(
        containers.HSplit([
          containers.Window(controls.FormattedTextControl('header'), height=1),
          containers.DynamicContainer(lambda: window),
          editor,
          containers.Window(controls.FormattedTextControl('status'), height=1),
        ]),
        focused_element=editor,
      ),
      input=input,
      output=output,
      key_bindings=bindings,
      full_screen=True,
      mouse_support=True,
    )
    painted = asyncio.Event()

    def report_error(
      loop: asyncio.AbstractEventLoop, context: dict[str, object]
    ) -> None:
      del loop
      error = context['exception']
      print(
        f'error {type(error).__name__}: {error}; active={current.get_app() is app}',
        flush=True,
      )

    def after_render(_: application.Application[None]) -> None:
      painted.set()

    app.after_render.add_handler(after_render)

    @bindings.add('c-q')
    def quit(event: key_processor.KeyPressEvent) -> None:
      event.app.exit()

    @bindings.add('f3')
    def runtime_error(event: key_processor.KeyPressEvent) -> None:
      del event
      raise RuntimeError('key handler failed')

    @bindings.add('f4')
    def value_error(event: key_processor.KeyPressEvent) -> None:
      del event
      raise ValueError('key handler failed')

    @bindings.add('f5')
    def replace_window(event: key_processor.KeyPressEvent) -> None:
      del event
      nonlocal window
      window.closed = True
      window = prompt_renderer.Window(
        'test',
        lambda: document('new '),
        list,
        lambda: True,
        reconnect=lambda: print('reconnected', flush=True),
      )

    callback_error = {
      'RuntimeError': RuntimeError,
      'ValueError': ValueError,
    }.get(sys.argv[3] if len(sys.argv) > 3 else '')

    def on_event(event: transcript_pb2.TranscriptEvent) -> None:
      if (
        callback_error is not None
        and event.WhichOneof('payload') == 'position'
        and event.position.reading
      ):
        raise callback_error('event handler failed')

    def foreground() -> None:
      os.write(stdout.fileno(), b'FOREGROUND_READY\r\n')
      value = os.read(stdin.fileno(), 4096)
      if value != b'handoff\n':
        raise ValueError(f'Unexpected foreground input: {value!r}')

    @bindings.add('f2')
    async def handoff(event: key_processor.KeyPressEvent) -> None:
      del event
      await run_in_terminal(foreground, in_executor=True)
      print('returned', flush=True)

    async with prompt_application.attach(
      app, on_event=on_event, binary=pathlib.Path(sys.argv[2])
    ) as bridge:

      async def ready() -> None:
        await painted.wait()
        await bridge.wait(
          bridge.connection.send(transcript_pb2.TranscriptCommand(terminal=b''))
        )
        print('ready', flush=True)

      def start() -> None:
        app.create_background_task(ready())

      asyncio.get_running_loop().set_exception_handler(report_error)
      await app.run_async(pre_run=start, set_exception_handler=False)


if __name__ == '__main__':
  asyncio.run(main())
