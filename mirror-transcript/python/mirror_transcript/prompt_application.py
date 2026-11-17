"""A single native writer around an otherwise ordinary Python application."""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
from collections.abc import AsyncGenerator, Callable
from typing import TextIO, cast

from mirror_transcript import (
  client,
  prompt_input,
  prompt_renderer,
  transcript_pb2,
)
from prompt_toolkit import application
from prompt_toolkit.application import current
from prompt_toolkit.output import vt100


class Bridge:
  """Drain native events independently from redraws and acknowledge terminal release."""

  def __init__(
    self,
    connection: client.Client,
    input: prompt_input.Input,
    stream: prompt_renderer.Stream,
  ) -> None:
    self.connection = connection
    self.input = input
    self.stream = stream
    self.presented = 0
    self.changed = asyncio.Event()
    self.failure: Exception | None = None

  async def wait(self, sequence: int) -> None:
    """Wait for a specific native commit, never for a guessed flush interval."""
    async with asyncio.timeout(5):
      while self.presented < sequence:
        self.changed.clear()
        if self.failure is not None:
          raise self.failure
        await self.changed.wait()

  async def pause(self) -> None:
    """Transfer terminal ownership only after the native reader/writer has stopped."""
    if not self.stream.direct:
      # Redraws can still write while the framework waits for terminal release.
      pending: list[str] = []
      self.stream.pending = pending
      try:
        await self.wait(
          self.connection.send(transcript_pb2.TranscriptCommand(pause=True))
        )
        self.stream.direct = True
      finally:
        self.stream.pending = None
      self.stream.write(''.join(pending))
      self.stream.flush()

  async def receive[T](
    self,
    app: application.Application[T],
    on_event: Callable[[transcript_pb2.TranscriptEvent], None],
  ) -> None:
    try:
      while True:
        event = await self.connection.receive()
        match event.WhichOneof('payload'):
          case 'presented':
            self.presented = event.presented
            self.changed.set()
          case 'input':
            self.input.feed(event.input.sequence, event.input.data)
            app.invalidate()
          case 'rejected':
            raise ValueError(
              f'Native transcript rejected command {event.rejected.command}: {event.rejected.reason}'
            )
          case 'resize':
            app.invalidate()
          case _:
            with current.set_app(app):
              try:
                on_event(event)
              except Exception as error:  # noqa: BLE001, event-handler isolation.
                asyncio.get_running_loop().call_exception_handler({
                  'message': 'Exception in transcript event callback',
                  'exception': error,
                })
    except (
      EOFError,
      OSError,
      ValueError,
      asyncio.IncompleteReadError,
    ) as error:
      self.failure = error
      self.changed.set()
      if app.is_running and not app.is_done:
        app.exit(exception=error)


@contextlib.asynccontextmanager
async def attach[T](
  app: application.Application[T],
  *,
  on_event: Callable[[transcript_pb2.TranscriptEvent], None] = lambda _: None,
  binary: pathlib.Path | None = None,
) -> AsyncGenerator[Bridge]:
  """Attach before run_async; leave signals, cooked/raw modes, and jobs to Python."""
  original_input, original_output, original_renderer = (
    app.input,
    app.output,
    app.renderer,
  )
  if not app.full_screen or not isinstance(original_output, vt100.Vt100_Output):
    raise ValueError(
      'Native transcript requires a fullscreen VT100 application'
    )
  path = os.ttyname(original_input.fileno())
  # The child may read as soon as it starts, before Application enters raw mode.
  with original_input.raw_mode():
    async with client.open_client(path, binary) as connection:
      input = prompt_input.Input(original_input)
      stream = prompt_renderer.Stream(original_output, connection.send)
      output = vt100.Vt100_Output(
        cast(
          TextIO, stream
        ),  # TextIOBase implements the framework's stream contract.
        original_output.get_size,
        term=original_output.term,
        default_color_depth=original_output.get_default_color_depth(),
        enable_cpr=True,
      )
      bridge = Bridge(connection, input, stream)
      app.input, app.output = input, output
      app.renderer = prompt_renderer.Renderer(
        original_renderer, output, stream, bridge.pause, lambda: input.sequence
      )
      receiver = asyncio.create_task(bridge.receive(app, on_event))
      try:
        yield bridge
      finally:
        try:
          if bridge.failure is None:
            await bridge.pause()
        finally:
          receiver.cancel()
          await asyncio.gather(receiver, return_exceptions=True)
          app.input, app.output, app.renderer = (
            original_input,
            original_output,
            original_renderer,
          )
