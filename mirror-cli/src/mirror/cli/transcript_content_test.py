from __future__ import annotations

from mirror.protocol import agent_pb2
from prompt_toolkit.output import color_depth

from mirror.cli import rendering, transcript_content


def test_user_text_keeps_quote_chip_styles() -> None:
  depth = color_depth.ColorDepth.DEPTH_24_BIT
  styles = transcript_content.palette('dark', depth)
  message = agent_pb2.UserMessage(
    parts=[
      agent_pb2.UserMessagePart(text='Why '),
      agent_pb2.UserMessagePart(quoted_selection='first\nsecond'),
    ]
  )

  [frame] = transcript_content.Projection(styles, depth).blocks(
    rendering.user_text(message)
  )

  [line] = frame.children[0].lines
  assert [(span.text, span.sgr) for span in line][1:] == [
    ('▌', styles['quote-chip-rail']),
    ('first…', styles['quote.italic']),
    (' +1 line', styles['quote-chip-detail']),
  ]
