from collections.abc import Mapping
from typing import override

import pytest
from mirror_transcript import model
from prompt_toolkit.layout import containers, controls, mouse_handlers, screen
from rich import text

from mirror.cli import extension, extension_content, rendering, state


class _Source(extension.Extension[int, None, None]):
  @classmethod
  @override
  def id(cls) -> str:
    return 'source'

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_ABOVE_PROMPT

  @staticmethod
  @override
  def initial_state() -> int:
    return 1

  @override
  def render_content(
    self, global_state: state.State, extension_state: int
  ) -> text.Text:
    return text.Text('source widget')


class _Legacy(_Source):
  @override
  def render(
    self, global_state: state.State, extension_state: int
  ) -> containers.Window:
    return containers.Window(controls.FormattedTextControl('legacy widget'))


class _Native(_Legacy):
  def __init__(self, visible: bool) -> None:
    self.visible = visible

  @override
  @override
  def render_native(
    self,
    global_state: state.State,
    extension_state: int,
    styles: Mapping[str, str],
  ) -> model.Block | None:
    return model.Block('native widget') if self.visible else None


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('widget', 'expected'),
  [
    (_Source(), 'source widget'),
    (_Legacy(), 'legacy widget'),
    (_Native(True), 'native widget'),
    (_Native(False), ''),
  ],
)
async def test_widget_hooks_preserve_authoritative_native_none(
  widget: _Source, expected: str
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(widget, None)
  view = registry.create_view()
  snapshot = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    lifecycle=state.LifecycleState(active=True),
    workspace='/workspace',
    extensions=view.initial_states,
  )
  output = screen.Screen()
  containers.to_container(
    view.render(snapshot, extension.RenderRegion.WIDGET_ABOVE_PROMPT)
  ).write_to_screen(
    output,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, 40, 2),
    '',
    True,
    None,
  )
  assert (
    ''.join(output.data_buffer[0][column].char for column in range(40)).rstrip()
    == expected
  )
  await registry.close()


@pytest.mark.parametrize('width', [10, 14])
def test_native_widget_uses_its_allocation_not_terminal_width(
  width: int,
) -> None:
  widget = extension_content.window(
    extension_content.Native(
      model.Block(kind=model.Kind.PANEL, children=[model.Block('short')])
    ),
    'dark',
  )
  widget.preferred_width(40)
  assert widget.preferred_height(width, 3).preferred == 3
  output = screen.Screen()
  widget.write_to_screen(
    output,
    mouse_handlers.MouseHandlers(),
    screen.WritePosition(0, 0, width, 3),
    '',
    True,
    None,
  )
  assert ''.join(
    output.data_buffer[0][column].char for column in range(width)
  ) == ('╭' + '─' * (width - 2) + '╮')
  assert ''.join(
    output.data_buffer[2][column].char for column in range(width)
  ) == ('╰' + '─' * (width - 2) + '╯')


def test_native_widget_preserves_nested_content_and_styles() -> None:
  source = model.Block(
    kind=model.Kind.ROW,
    children=[
      model.Block(lines=[[model.Span('界', '31')]]),
      model.Block('right'),
    ],
  )
  rendered = rendering.render_blocks([extension_content.Native(source)], 30)
  assert '界right' in text.Text.from_ansi(rendered).plain
  assert '\x1b[31m' in rendered
