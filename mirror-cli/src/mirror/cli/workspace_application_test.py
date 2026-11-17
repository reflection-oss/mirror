import asyncio
from typing import override

import pytest
from mirror.protocol import agent_pb2
from prompt_toolkit import data_structures, keys
from prompt_toolkit.application import application
from prompt_toolkit.input import defaults
from prompt_toolkit.key_binding import key_bindings, key_processor
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import base

from mirror.cli import (
  actions,
  frame_pacing,
  transcript,
  workspace_application,
)
from tests.cli import reference_transcript_view as transcript_view


class _Output(base.DummyOutput):
  def __init__(self, *, answers_cpr: bool = False) -> None:
    self.raw: list[str] = []
    self._answers_cpr = answers_cpr

  @property
  @override
  def responds_to_cpr(self) -> bool:
    return self._answers_cpr

  @override
  def get_size(self) -> data_structures.Size:
    return data_structures.Size(rows=40, columns=80)

  @override
  def write_raw(self, data: str) -> None:
    self.raw.append(data)


@pytest.mark.asyncio
async def test_fullscreen_frames_query_the_terminal_and_consume_replies() -> (
  None
):
  output = _Output(answers_cpr=True)
  bound_replies: list[key_processor.KeyPressEvent] = []
  bindings = key_bindings.KeyBindings()
  bindings.add(keys.Keys.CPRResponse)(bound_replies.append)
  with defaults.create_pipe_input() as pipe:
    owner = workspace_application.WorkspaceApplication(
      input=pipe, output=output, full_screen=True
    )
    owner.enable_views()
    rendered, started = asyncio.Event(), asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    owner.after_render += after_render
    running = asyncio.create_task(owner.run_async(pre_run=started.set))
    await started.wait()
    view = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.Window(controls.FormattedTextControl('frame'))
      ),
      key_bindings=bindings,
      input=pipe,
      output=output,
    )
    selected = asyncio.create_task(owner.show_view(view))
    try:
      await rendered.wait()
      queries = output.raw.count(frame_pacing.FRAME_QUERY)
      assert queries
      assert owner.renderer.waiting_for_cpr
      owner.key_processor.feed_multiple(
        [key_processor.KeyPress(keys.Keys.CPRResponse, '\x1b[1;1R')] * queries
      )
      owner.key_processor.process_keys()
      assert not owner.renderer.waiting_for_cpr
      assert not bound_replies, 'A reply handled as a key would redraw'
    finally:
      owner.exit()
      await running
      owner.finish_views()
      await selected


@pytest.mark.asyncio
async def test_trackpad_burst_scrolls_without_matching_each_report() -> None:
  history = transcript.Transcript(
    view_mode=lambda: actions.ViewMode.DETAIL,
    skill_names=frozenset(),
    get_rewind_entry_id=lambda: None,
  )
  for index in range(40):
    history.append(
      agent_pb2.AgentMessage(
        id=f'answer-{index}',
        sequence=index + 1,
        chat=agent_pb2.ChatRecord(
          assistant=agent_pb2.AssistantMessage(content=f'answer {index}'),
          success=True,
        ),
      )
    )
  scrolled: list[actions.NavigateHistory] = []
  viewport = transcript_view.TranscriptView(
    history.blocks, lambda: (), navigate=scrolled.append
  )
  walks = 0

  def chrome() -> containers.AnyContainer:
    nonlocal walks
    walks += 1
    return containers.Window(controls.FormattedTextControl('chrome'), height=1)

  output = _Output()
  with defaults.create_pipe_input() as pipe:
    owner = workspace_application.WorkspaceApplication(
      input=pipe, output=output, full_screen=True, mouse_support=True
    )
    owner.enable_views()
    rendered, started = asyncio.Event(), asyncio.Event()

    def after_render(_: application.Application[str | None]) -> None:
      rendered.set()

    owner.after_render += after_render
    running = asyncio.create_task(owner.run_async(pre_run=started.set))
    await started.wait()
    view = workspace_application.WorkspaceApplication(
      layout=layout.Layout(
        containers.HSplit([
          containers.Window(viewport, wrap_lines=False),
          containers.DynamicContainer(chrome),
        ])
      ),
      input=pipe,
      output=output,
    )
    view.scroll_handler = viewport.mouse_handler
    selected = asyncio.create_task(owner.show_view(view))
    try:
      await rendered.wait()
      walks = 0
      rendered.clear()
      # Trackpads interleave motion and horizontal reports with vertical ones.
      pipe.send_text(('\x1b[<64;1;1M' * 5 + '\x1b[<35;2;1M\x1b[<66;2;1M') * 12)
      await rendered.wait()
      assert scrolled == [actions.NavigateHistory('up', rows=60)]
      assert walks < 10, f'{walks} layout walks for one burst'
    finally:
      owner.exit()
      await running
      owner.finish_views()
      await selected
