"""A single native writer around an otherwise ordinary Python application."""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
    self.frame_sequence = 0
    self.restarts = 0
    self.changed = asyncio.Event()
    self.failure: Exception | None = None
    self.failed = asyncio.Event()
    self.stopped = asyncio.Event()
    self.active = asyncio.Event()
    self.active.set()

  def fail(self, error: Exception) -> None:
    """Wake the supervisor without propagating renderer failures into the app."""
    if self.failure is None:
      logging.getLogger(__name__).warning(
        'Native transcript renderer failed: %s', error
      )
      self.frame_sequence = 0
      self.failure = error
      self.failed.set()
      self.changed.set()

  def send(self, command: transcript_pb2.TranscriptCommand) -> int:
    """Drop writes during recovery; the next renderer gets a full publication."""
    if command.WhichOneof('payload') == 'pause' and not command.pause:
      self.active.set()
    if self.failure is None:
      try:
        sequence = self.connection.send(command)
        if command.HasField('frame') and not self.frame_sequence:
          self.frame_sequence = sequence
        return sequence
      except Exception as error:  # noqa: BLE001, native transport isolation.
        self.fail(error)
    return 0

  async def wait(self, sequence: int) -> bool:
    """Wait for a native commit, or return False once its failed child is gone."""
    connection, stopped = self.connection, self.stopped
    try:
      async with asyncio.timeout(5):
        while (
          self.connection is connection
          and self.failure is None
          and self.presented < sequence
        ):
          self.changed.clear()
          await self.changed.wait()
    except TimeoutError:
      self.fail(
        TimeoutError(
          f'Native transcript did not acknowledge command {sequence}'
        )
      )
    if self.failure is not None or self.connection is not connection:
      await stopped.wait()
      return False
    return True

  async def pause(self) -> None:
    """Transfer terminal ownership only after the native reader/writer has stopped."""
    self.active.clear()
    if not self.stream.direct:
      # Redraws can still write while the framework waits for terminal release.
      pending: list[str] = []
      self.stream.pending = pending
      try:
        await self.wait(self.send(transcript_pb2.TranscriptCommand(pause=True)))
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
            # Startup and terminal-only acknowledgements do not prove recovery.
            if self.frame_sequence and self.presented >= self.frame_sequence:
              self.restarts = 0
            self.changed.set()
          case 'input':
            self.input.feed(
              event.input.sequence, event.input.data, event.input.selection
            )
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
    except Exception as error:  # noqa: BLE001, native protocol isolation.
      self.fail(error)


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
    async with contextlib.AsyncExitStack() as stack:
      connection = await stack.enter_async_context(
        client.open_client(path, binary)
      )
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
      stream.send = bridge.send
      app.input, app.output = input, output
      renderer = prompt_renderer.Renderer(
        original_renderer, output, stream, bridge.pause, lambda: input.sequence
      )
      app.renderer = renderer

      async def supervise() -> None:
        try:
          while True:
            receiver = asyncio.create_task(bridge.receive(app, on_event))
            try:
              await bridge.failed.wait()
            finally:
              receiver.cancel()
              await asyncio.gather(receiver, return_exceptions=True)
              await stack.aclose()
              if bridge.failure is not None and not stream.direct:
                stream.restore_terminal()
              bridge.stopped.set()
            while True:
              await bridge.active.wait()
              if bridge.restarts >= 3:
                if (
                  app.is_running
                  and not app.is_done
                  and bridge.failure is not None
                ):
                  app.exit(exception=bridge.failure)
                return
              await asyncio.sleep(1)
              if not bridge.active.is_set():
                continue
              bridge.restarts += 1
              bridge.stopped = asyncio.Event()
              try:
                connection = await stack.enter_async_context(
                  client.open_client(path, binary)
                )
              except Exception as error:  # noqa: BLE001, child startup isolation.
                logging.getLogger(__name__).warning(
                  'Cannot restart native transcript renderer: %s', error
                )
                bridge.failure = error
                bridge.stopped.set()
                continue
              if not bridge.active.is_set():
                await stack.aclose()
                bridge.stopped.set()
                bridge.restarts -= 1
                continue
              bridge.connection = connection
              bridge.presented = 0
              bridge.failure = None
              bridge.failed.clear()
              input.reconnect()
              renderer.reconnect()
              app.invalidate()
              break
        finally:
          await stack.aclose()
          bridge.stopped.set()

      supervisor = asyncio.create_task(supervise())
      try:
        yield bridge
      finally:
        try:
          await bridge.pause()
        finally:
          supervisor.cancel()
          await asyncio.gather(supervisor, return_exceptions=True)
          app.input, app.output, app.renderer = (
            original_input,
            original_output,
            original_renderer,
          )
