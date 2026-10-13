"""Exercise native Vi bindings through Mirror's real shortcut precedence."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest
from prompt_toolkit import application, buffer, enums, layout
from prompt_toolkit.input import base as input_base
from prompt_toolkit.input import vt100_parser
from prompt_toolkit.output import base as output_base

from mirror.cli import actions, keymap, state
from mirror.cli.composer import editing
from mirror.cli.extensions.vim import extension as vim_extension


@contextlib.asynccontextmanager
async def editor(
  dispatched: list[actions.Action],
  *,
  completion: bool = False,
) -> AsyncIterator[tuple[application.Application[str | None], buffer.Buffer]]:
  composer = buffer.Buffer(multiline=True)
  mapping = keymap.Keymap(
    lambda: [
      keymap.defaults(
        state.State(
          activity=state.ActivityState(),
          status_bar=state.StatusBarState('test', 'medium', 'auto'),
          session=state.SessionState('test'),
          lifecycle=state.LifecycleState(active=True),
        ),
        completion_visible=completion,
        modal_active=False,
        can_focus_tasks=False,
      )
    ],
    dispatched.append,
    dismiss=lambda: actions.CloseSelector() if completion else None,
  )
  app = application.Application[str | None](
    layout=layout.Layout(layout.Window(layout.BufferControl(composer))),
    key_bindings=mapping.bindings,
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  editing.sync(app, composer, enums.EditingMode.VI, suspended=False)
  with application.set_app(app):
    try:
      yield app, composer
    finally:
      await app.cancel_and_wait_for_background_tasks()


def press(app: application.Application[str | None], text: str) -> None:
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  parser.feed(text)
  parser.flush()
  app.key_processor.process_keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('command', 'text', 'cursor', 'mode'),
  [
    ('0w', 'one two', 4, 'NORMAL'),
    ('0dw', 'two', 0, 'NORMAL'),
    ('0cwX', 'Xtwo', 1, 'INSERT'),
    ('0ciwX', 'X two', 1, 'INSERT'),
    ('0xxu', 'ne two', 0, 'NORMAL'),
    ('0yyp', 'one two\none two', 8, 'NORMAL'),
    ('0vld', 'e two', 0, 'NORMAL'),
    ('0rX', 'Xne two', 0, 'NORMAL'),
    ('0aX', 'oXne two', 2, 'INSERT'),
    ('0l', 'one two', 1, 'NORMAL'),
    ('0d', 'one two', 0, 'NORMAL'),
  ],
)
async def test_native_motions_and_edits(
  command: str,
  text: str,
  cursor: int,
  mode: str,
) -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, composer):
    press(app, 'one two')
    assert vim_extension.indicator(app) == 'INSERT'
    press(app, '\x1b')
    press(app, command)
    assert (composer.text, composer.cursor_position) == (text, cursor)
    assert vim_extension.indicator(app) == mode
    assert dispatched == []


@pytest.mark.asyncio
async def test_escape_transitions_without_interrupting() -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, composer):
    press(app, 'hello')
    press(app, '\x1b')
    assert vim_extension.indicator(app) == 'NORMAL'
    assert composer.cursor_position == 4
    app.timeoutlen = 0.01
    press(app, 'd')
    await asyncio.sleep(0.03)
    assert vim_extension.indicator(app) == 'OPERATOR'
    press(app, '\x1b')
    assert dispatched == []
    assert vim_extension.indicator(app) == 'NORMAL'
    press(app, 'v')
    assert vim_extension.indicator(app) == 'VISUAL'
    press(app, '\x1b')
    assert composer.selection_state is None
    assert dispatched == []
    press(app, '\x1b')
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize('normal', [False, True])
async def test_mirror_shortcuts_keep_their_meaning(normal: bool) -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, _):
    if normal:
      press(app, '\x1b')
    press(app, '\r\n\x03\x04\x12\x1f\x19\t\x0f')
    assert dispatched == [
      actions.ComposerSubmit(),
      actions.ComposerNewline(),
      actions.CancelOrQuitRequested(),
      actions.QuitRequested(),
      actions.RequestPromptHistory(),
      actions.ComposerUndo(),
      actions.ComposerRedo(),
      actions.CycleWorktreeRequested(1),
      actions.OpenWorkspaces(),
    ]


@pytest.mark.asyncio
async def test_completion_escape_and_navigation_win() -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched, completion=True) as (app, _):
    press(app, '\x1b')
    assert vim_extension.indicator(app) == 'INSERT'
    press(app, '\n\t')
    assert dispatched == [
      actions.CloseSelector(),
      actions.MoveSelection(1),
      actions.AcceptSelection(),
    ]


@pytest.mark.asyncio
async def test_paste_multiline_and_suspend_resume() -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, composer):
    press(app, '\x1b[200~one\ntwo\x1b[201~')
    assert composer.text == 'one\ntwo'
    press(app, '\x1b')
    press(app, 'gg0j')
    assert composer.cursor_position == 4
    press(app, 'v')
    editing.sync(app, composer, enums.EditingMode.VI, suspended=True)
    assert vim_extension.indicator(app) == ''
    assert composer.selection_state is None
    press(app, 'x')
    assert composer.text == 'one\nxtwo'
    editing.sync(app, composer, enums.EditingMode.VI, suspended=False)
    assert vim_extension.indicator(app) == 'INSERT'
    editing.sync(app, composer, enums.EditingMode.EMACS, suspended=False)
    press(app, '\x1b')
    assert dispatched == [actions.InterruptRequested()]


@pytest.mark.asyncio
async def test_normal_mode_paste_is_literal_unicode() -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, composer):
    press(app, '\x1b')
    press(app, '\x1b[200~dw你好\nhello\x1b[201~')
    assert composer.text == 'dw你好\nhello'
    assert dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('sequence', 'mode'),
  [
    ('\x1bh', 'NORMAL'),
    ('\x1b[1;3D', 'NORMAL'),
    ('\x1b[200~text\x1b[201~', 'INSERT'),
  ],
)
async def test_terminal_escape_sequences_do_not_interrupt(
  sequence: str,
  mode: str,
) -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, _):
    press(app, sequence)
    assert dispatched == []
    assert vim_extension.indicator(app) == mode
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'command', ['i', 'a', 'I', 'A', 'o', 'O', 's', 'S', 'C']
)
async def test_insert_commands_take_effect_without_waiting(
  command: str,
) -> None:
  dispatched: list[actions.Action] = []
  async with editor(dispatched) as (app, _):
    press(app, 'hello world')
    press(app, '\x1b')
    press(app, command)
    assert vim_extension.indicator(app) == 'INSERT'
    assert app.key_processor.key_buffer == []
    assert dispatched == []
