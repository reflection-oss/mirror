"""Length-prefixed transcript protocol, independent of Python terminal layout."""

from __future__ import annotations

import asyncio
import contextlib
import os
import pathlib
import struct
from collections.abc import AsyncGenerator, Sequence

from mirror_transcript import content, model, transcript_pb2

_MAX_FRAME = 8 * 1024 * 1024
_TERMINAL_ENV = ('TERM', 'COLORTERM', 'LANG', 'LC_ALL', 'LC_CTYPE')


class Client:
  """One event-loop sender and one receiver; native scrolling needs neither."""

  def __init__(
    self, writer: asyncio.StreamWriter, reader: asyncio.StreamReader
  ) -> None:
    self._writer = writer
    self._reader = reader
    self._sequence = 0

  def send(self, command: transcript_pb2.TranscriptCommand) -> int:
    """Queue one bounded command; reject rather than grow an unbounded backlog."""
    command.sequence = self._sequence + 1
    payload = command.SerializeToString()
    if len(payload) > _MAX_FRAME:
      raise ValueError('Transcript command exceeds 8 MiB')
    if (
      self._writer.transport.get_write_buffer_size() + len(payload)
      > 2 * _MAX_FRAME
    ):
      raise BufferError('Transcript command backlog exceeds 16 MiB')
    self._writer.write(struct.pack('<I', len(payload)) + payload)
    self._sequence = command.sequence
    return self._sequence

  async def drain(self) -> None:
    """Wait for queued commands to reach the native process."""
    await self._writer.drain()

  async def receive(self) -> transcript_pb2.TranscriptEvent:
    """Read the next acknowledgement, forwarded input, or viewport notification."""
    (length,) = struct.unpack('<I', await self._reader.readexactly(4))
    if not 0 < length <= _MAX_FRAME:
      raise ValueError('Invalid transcript event length')
    event = transcript_pb2.TranscriptEvent.FromString(
      await self._reader.readexactly(length)
    )
    if event.WhichOneof('payload') is None:
      raise ValueError('Transcript event has no payload')
    return event


def element(value: model.Element) -> transcript_pb2.Element:
  """Encode a display copy without mutating the application's sources."""
  return transcript_pb2.Element(
    id=value.id,
    lines=lines(value.lines),
    blocks=blocks(content.bounded(value.blocks)),
    sticky=transcript_pb2.Span(text=value.sticky.text, sgr=value.sticky.sgr)
    if value.sticky
    else None,
  )


def lines(values: Sequence[model.Line]) -> list[transcript_pb2.Line]:
  return [
    transcript_pb2.Line(
      spans=[transcript_pb2.Span(text=span.text, sgr=span.sgr) for span in line]
    )
    for line in values
  ]


def blocks(values: Sequence[model.Block]) -> list[transcript_pb2.Block]:
  return [
    transcript_pb2.Block(
      text=value.text,
      kind=value.kind.name,
      lines=lines(value.lines),
      children=blocks(value.children),
      sgr=value.sgr,
      title=value.title,
      padding=value.padding,
      collapse=value.collapse,
      language=value.language,
      title_sgr=value.title_sgr,
      spread=value.spread,
      insets=value.insets,
      marker=transcript_pb2.Span(text=value.marker.text, sgr=value.marker.sgr)
      if value.marker
      else None,
      clip=value.clip,
      trim_end=value.trim_end,
      ratio=value.ratio,
      border_sgr=value.border_sgr,
      overflow=value.overflow.name,
      show_file_headers=value.show_file_headers,
      collapsed=blocks(value.collapsed),
      no_wrap=value.no_wrap,
      gap=value.gap,
      height=value.height,
      syntax=value.syntax,
      alignment=value.alignment.name,
      word_wrap=value.word_wrap,
    )
    for value in values
  ]


@contextlib.asynccontextmanager
async def open_client(
  terminal: str, binary: pathlib.Path | None = None
) -> AsyncGenerator[Client]:
  """Start only the transcript child; the application retains terminal modes."""
  binary = binary or pathlib.Path(
    os.environ.get('MIRROR_TRANSCRIPT_BIN')
    or pathlib.Path(__file__).parent / 'bin' / 'mirror-transcript'
  )
  process = await asyncio.create_subprocess_exec(
    str(binary),
    '--terminal',
    terminal,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.DEVNULL,
    env={key: os.environ[key] for key in _TERMINAL_ENV if key in os.environ},
  )
  if process.stdin is None or process.stdout is None:
    raise RuntimeError('Transcript process did not open its protocol pipes')
  try:
    bridge = Client(process.stdin, process.stdout)
    async with asyncio.timeout(5):
      ready = await bridge.receive()
    if (
      ready.WhichOneof('payload') != 'ready'
      or ready.ready != 'mirror-transcript/1'
    ):
      raise ValueError(
        'Native process does not implement transcript protocol 1'
      )
    yield bridge
  finally:
    process.stdin.close()
    try:
      async with asyncio.timeout(2):
        await process.wait()
    except TimeoutError:
      process.kill()
      await process.wait()
