import base64
import io
from collections.abc import Sequence

import pytest
from mirror_transcript import transcript_pb2
from prompt_toolkit import application, data_structures
from prompt_toolkit.application import current
from prompt_toolkit.input import base
from prompt_toolkit.output import vt100
from rich import console, text

from mirror.cli import native_transcript, transcript


def _view(
  settled: list[transcript.Block],
  pending: list[transcript.Block],
  live: list[console.RenderableType],
) -> native_transcript.View:
  return native_transcript.View(
    lambda: tuple(settled),
    lambda: live,
    pending=lambda: pending,
    navigate=lambda _: None,
    theme_name=lambda: 'dark',
  )


def _texts(
  commands: Sequence[transcript_pb2.TranscriptCommand],
) -> dict[int, str]:
  return {
    element.id: element.blocks[0].lines[0].spans[0].text
    for element in commands[0].update.set
  }


def test_publish_settlement_keeps_python_order_and_stable_pending_identity() -> (
  None
):
  settled: list[transcript.Block] = []
  pending = [transcript.Block('queued', (0, 0), text.Text('queued user'))]
  live: list[console.RenderableType] = [text.Text('streaming answer')]
  view = _view(settled, pending, live)
  first = list(view.publish())
  queued, streaming = first[0].update.order.ids
  assert _texts(first) == {queued: 'queued user', streaming: 'streaming answer'}
  settled.append(
    transcript.Block('answer', (1, 0), text.Text('completed answer'))
  )
  view.settle_live('answer')
  live[:] = [text.Text('echo')]
  final = list(view.publish())
  answer, retained, echo = final[0].update.order.ids
  assert retained == queued
  assert _texts(final) == {answer: 'completed answer', echo: 'echo'}
  assert final[0].update.settlements[0].before == streaming
  assert final[0].update.settlements[0].after == answer
  assert list(view.publish()) == []


@pytest.mark.parametrize('stale', [False, True])
def test_action_after_source_replacement_is_ignored(stale: bool) -> None:
  terminal = io.StringIO()
  output = vt100.Vt100_Output(terminal, lambda: data_structures.Size(5, 40))
  app = application.Application[None](input=base.DummyInput(), output=output)
  view = _view([], [], [text.Text('answer')])
  with current.set_app(app):
    commands = list(view.publish())
    revision = commands[0].update.revision
    if stale:
      view.clear()
    view.observe(
      transcript_pb2.TranscriptEvent(
        action=transcript_pb2.TranscriptAction(
          view=view.id,
          revision=revision,
          copy='selected text',
        )
      )
    )
  encoded = base64.b64encode(b'selected text').decode()
  assert (f'\x1b]52;c;{encoded}\x1b\\' in terminal.getvalue()) is not stale


def test_oversized_copy_does_not_abort_the_terminal() -> None:
  terminal = io.StringIO()
  output = vt100.Vt100_Output(terminal, lambda: data_structures.Size(5, 40))
  app = application.Application[None](input=base.DummyInput(), output=output)
  view = _view([], [], [text.Text('answer')])
  with current.set_app(app):
    commands = list(view.publish())
    revision = commands[0].update.revision
    view.observe(
      transcript_pb2.TranscriptEvent(
        action=transcript_pb2.TranscriptAction(
          view=view.id,
          revision=revision,
          copy='x' * 75001,
        )
      )
    )
    output.flush()
  assert '\x07' in terminal.getvalue()
  assert '\x1b]52;' not in terminal.getvalue()
