"""Isolated Python owner used to stop the host without stopping its renderer."""

import asyncio
import pathlib
import sys

from mirror_transcript import client, transcript_pb2


async def main() -> None:
  async with client.open_client(
    sys.argv[1], pathlib.Path(sys.argv[2])
  ) as bridge:
    bridge.send(
      transcript_pb2.TranscriptCommand(
        update=transcript_pb2.TranscriptUpdate(
          view='workspace',
          set=[
            transcript_pb2.Element(
              id=1,
              lines=[
                transcript_pb2.Line(
                  spans=[transcript_pb2.Span(text=value, sgr='32')]
                )
                for value in (
                  'zero',
                  'one',
                  'two',
                  'three',
                  'four',
                  'five',
                  'six',
                  'seven',
                  'eight',
                  'nine',
                )
              ],
            )
          ],
          order=transcript_pb2.TranscriptOrder(ids=[1]),
        )
      )
    )
    frame = bridge.send(
      transcript_pb2.TranscriptCommand(
        frame=transcript_pb2.ChromeFrame(
          size=transcript_pb2.Size(columns=20, rows=6),
          rows=[
            f'\x1b[34m{value:20}\x1b[0m'.encode()
            for value in ('header', '', '', '', 'draft> ', 'status')
          ],
          cursor_x=7,
          cursor_y=4,
          cursor_visible=True,
          cursor_shape=6,
          view='workspace',
          wheel=True,
          transcript=transcript_pb2.TranscriptRect(
            x=0, y=1, width=20, height=3
          ),
        )
      )
    )
    await bridge.drain()
    while (event := await bridge.receive()).WhichOneof(
      'payload'
    ) != 'presented' or event.presented != frame:
      if event.WhichOneof('payload') == 'rejected':
        raise RuntimeError(event.rejected.reason)
    print('ready', flush=True)
    await asyncio.to_thread(sys.stdin.buffer.read, 1)


if __name__ == '__main__':
  asyncio.run(main())
