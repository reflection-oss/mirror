from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import pathlib
import sys
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import cast
from unittest import mock

import pytest

from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  reducer,
  selection,
  shell,
  state,
)
from mirror.cli.extensions.editor import extension as editor


def _state(cwd: pathlib.Path) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('model', 'medium', 'auto'),
    session=state.SessionState('conversation'),
    workspace=str(cwd),
    lifecycle=state.LifecycleState(active=True),
  )


class _Picker:
  def __init__(
    self,
    cwd: pathlib.Path,
    backend: shell.Shell | None = None,
    run_terminal: Callable[[tuple[str, ...], str], Awaitable[bool]]
    | None = None,
  ) -> None:
    self.plugin = editor.EditorExtension()
    self.current = _state(cwd)
    self.tasks: list[asyncio.Task[None]] = []
    self.context = extension.Context(
      extension.Workspace(cwd, remote=False, shell=backend or shell.Shell(cwd)),
      self.dispatch,
      lambda: self.current,
      lambda _: None,
      refresh_tools=lambda _: asyncio.sleep(0),
      call_tool=lambda name, arguments: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda options, on_message, on_error: pytest.fail(
        'Unexpected agent creation'
      ),
      run_terminal=run_terminal,
    )

  def dispatch(self, action: actions.Action) -> None:
    _, emitted = self.plugin.reduce(self.current, action)
    self.current, _ = reducer.reduce(self.current, action)
    for effect in emitted:
      assert isinstance(effect, extension.PrivateEffect)
      self.tasks.append(
        asyncio.create_task(
          self.plugin.handle_effect(self.context, effect.payload)
        )
      )

  def open(self) -> None:
    action = commands.dispatch(
      '/editor',
      commands.CommandContext(
        'model', 'medium', 'conversation', False, self.current.workspace
      ),
      self.plugin.slash_commands(),
    )
    assert action is not None
    self.dispatch(action)

  @property
  def request(self) -> selection.Selector[actions.Action]:
    request = self.current.selector.selector
    assert request is not None
    return cast(selection.Selector[actions.Action], request)


async def _wait_for(predicate: Callable[[], bool]) -> None:
  async with asyncio.timeout(5):
    while not predicate():
      await asyncio.sleep(0.005)


@contextlib.asynccontextmanager
async def _running_picker(
  cwd: pathlib.Path,
  backend: shell.Shell | None = None,
) -> AsyncGenerator[_Picker, None]:
  picker = _Picker(cwd, backend)
  picker.open()
  await asyncio.sleep(0)
  try:
    yield picker
  finally:
    picker.dispatch(actions.ClearSelector())
    async with asyncio.timeout(5):
      await asyncio.gather(*picker.tasks)


async def _picker(cwd: pathlib.Path) -> selection.Selector[actions.Action]:
  async with _running_picker(cwd) as picker:
    await _wait_for(lambda: not picker.request.loading)
    return picker.request


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('remote', 'expected'),
  [
    (False, (effects.RunExtensionEffect('core/editor', editor.OpenEditor()),)),
    (True, (effects.RunExtensionEffect('core/editor', editor.OpenEditor()),)),
  ],
)
async def test_editor_action_is_routed_without_location_declarations(
  tmp_path: pathlib.Path,
  remote: bool,
  expected: tuple[effects.RunExtensionEffect, ...],
) -> None:
  registry = extension.ExtensionRegistry()
  current = _state(tmp_path)
  host = registry.create_host(
    extension.Context(
      extension.Workspace(tmp_path, remote=remote),
      lambda _: None,
      lambda: current,
      lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda name, arguments: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda options, on_message, on_error: pytest.fail(
        'Unexpected agent creation'
      ),
    )
  )
  try:
    await registry.register(editor.EditorExtension(), None)
    await host.start()
    action = actions.RoutedExtensionAction('core/editor', editor.OpenEditor())
    _, emitted = host.reduce(current, action)
    assert emitted == expected
    assert host.keybindings
    await registry.unload('core/editor')
    _, emitted = host.reduce(current, action)
    assert emitted == ()
  finally:
    await host.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('repetitions', [1, 20000])
async def test_file_batches_yield_before_eof_and_reap_cancelled_child(
  tmp_path: pathlib.Path,
  repetitions: int,
) -> None:
  path = tmp_path / 'file.py'
  path.write_text('text')
  output = json.dumps('file.py') + '\n'
  pid = tmp_path / 'pid'
  backend = shell.Shell(
    tmp_path,
    prefix=(
      sys.executable,
      '-c',
      f'import os, pathlib, sys, time; pathlib.Path({str(pid)!r}).write_text(str(os.getpid())); '
      f'sys.stdout.write({output!r} * {repetitions}); sys.stdout.flush(); time.sleep(60)',
    ),
  )
  async with asyncio.timeout(5), _running_picker(tmp_path, backend) as picker:
    await _wait_for(lambda: bool(picker.request.choices))
    assert set(picker.request.choices) == {
      selection.Choice('file.py', 'file.py')
    }
    os.kill(int(pid.read_text()), 0)
  with pytest.raises(ProcessLookupError):
    os.kill(int(pid.read_text()), 0)


@pytest.mark.asyncio
async def test_picker_streams_results_and_preserves_query_and_selection(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  more = asyncio.Event()
  stopped = asyncio.Event()
  first = selection.Choice('one', 'one.py')
  second = selection.Choice('two', 'two.py')
  third = selection.Choice('three', 'three.py')

  async def batches(
    workspace: extension.Workspace,
  ) -> AsyncIterator[tuple[selection.Choice, ...]]:
    assert workspace.cwd == tmp_path
    try:
      yield (first, second)
      await more.wait()
      yield (third,)
      await asyncio.Event().wait()
    finally:
      stopped.set()

  monkeypatch.setattr(editor.files, 'batches', batches)
  async with _running_picker(tmp_path) as picker:
    assert picker.request.loading
    assert picker.request.choices == ()
    await _wait_for(lambda: len(picker.request.choices) == 2)
    picker.dispatch(actions.SelectorSearchChanged('py'))
    assert len(picker.request.choices) == 2
    await _wait_for(lambda: not picker.current.selector.search_pending)
    picker.dispatch(actions.MoveSelectorSelection(1))
    selected = picker.current.selector.selected_choice()
    more.set()
    await _wait_for(lambda: len(picker.request.choices) == 3)
    assert picker.current.selector.query == 'py'
    assert picker.current.selector.selected_choice() == selected
    assert picker.request.loading
    picker.dispatch(actions.ClearSelector())
    await asyncio.wait_for(stopped.wait(), 1)
    assert not picker.current.selector.active


@pytest.mark.asyncio
async def test_picker_search_does_not_block_input_or_cancellation(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  searching = threading.Event()
  release = threading.Event()
  stopped = asyncio.Event()
  ui_thread = threading.get_ident()

  async def batches(
    workspace: extension.Workspace,
  ) -> AsyncIterator[tuple[selection.Choice, ...]]:
    del workspace
    try:
      yield (selection.Choice('file', 'file.py'),)
      await asyncio.Event().wait()
    finally:
      stopped.set()

  def matches(
    choices: tuple[selection.Choice, ...], query: str
  ) -> tuple[selection.Choice, ...]:
    assert threading.get_ident() != ui_thread
    if query:
      searching.set()
      assert release.wait(5)
    return selection.fuzzy_matches(
      choices, query, lambda choice: (choice.label,)
    )

  monkeypatch.setattr(editor.files, 'batches', batches)
  monkeypatch.setattr(editor, '_matches', matches)
  async with _running_picker(tmp_path) as picker:
    try:
      await _wait_for(lambda: bool(picker.request.choices))
      picker.dispatch(actions.SelectorSearchChanged('py'))
      assert await asyncio.to_thread(searching.wait, 1)
      picker.dispatch(actions.CloseSelector())
      await asyncio.wait_for(stopped.wait(), 1)
      assert not picker.current.selector.active
    finally:
      release.set()


@pytest.mark.asyncio
async def test_picker_lists_repository_text_files(
  tmp_path: pathlib.Path,
) -> None:
  (tmp_path / '.git').mkdir()
  (tmp_path / '.git' / 'config').write_text('metadata')
  (tmp_path / '.jj').mkdir()
  (tmp_path / '.jj' / 'state').write_text('metadata')
  (tmp_path / '.gitignore').write_text('ignored/\n*.log\n')
  (tmp_path / 'ignored').mkdir()
  (tmp_path / 'ignored' / 'secret').write_text('secret')
  (tmp_path / 'debug.log').write_text('ignored')
  (tmp_path / 'binary').write_bytes(b'abc\x00def')
  (tmp_path / 'invalid').write_bytes(b'\xff\xfe')
  (tmp_path / 'empty').touch()
  (tmp_path / '.hidden').write_text('visible')
  (tmp_path / 'file with spaces').write_text('text')
  (tmp_path / 'link').symlink_to(tmp_path / 'file with spaces')
  nested = tmp_path / 'nested'
  nested.mkdir()
  (nested / 'unicode.py').write_text('é' * 10000)

  opened = await _picker(tmp_path)

  assert tuple(sorted(choice.label for choice in opened.choices)) == (
    '.gitignore',
    '.hidden',
    'empty',
    'file with spaces',
    'nested/unicode.py',
  )
  assert opened.extension_id == editor.EditorExtension.id()


@pytest.mark.asyncio
async def test_picker_scopes_to_cwd_and_respects_ancestor_ignores(
  tmp_path: pathlib.Path,
) -> None:
  (tmp_path / '.git').mkdir()
  (tmp_path / '.gitignore').write_text('*.log\n')
  (tmp_path / 'outside.py').write_text('outside')
  cwd = tmp_path / 'nested'
  cwd.mkdir()
  (cwd / 'inside.py').write_text('inside')
  (cwd / 'ignored.log').write_text('ignored')

  opened = await _picker(cwd)

  assert opened.choices == (selection.Choice('inside.py', 'inside.py'),)


@pytest.mark.asyncio
async def test_picker_empty_directory(tmp_path: pathlib.Path) -> None:
  opened = await _picker(tmp_path)
  assert opened.choices == ()
  assert opened.empty == 'No text files available.'


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('visual', 'configured', 'expected'),
  [
    ('nvim -f', 'vi', ('nvim', '-f')),
    ('', 'code --wait', ('code', '--wait')),
    ('', '', ('vi',)),
  ],
)
async def test_selection_runs_editor_with_literal_absolute_path(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  visual: str,
  configured: str,
  expected: tuple[str, ...],
) -> None:
  monkeypatch.setenv('VISUAL', visual)
  monkeypatch.setenv('EDITOR', configured)
  host = tmp_path / 'host'
  host.mkdir()
  backend = tmp_path / 'backend'
  backend.mkdir()
  (backend / 'nested').mkdir()
  path = backend / 'nested' / '-file; echo unsafe'
  path.write_text('text')
  (host / path.name).write_text('host')
  opened = await _picker(backend)
  (choice,) = opened.choices
  assert choice.value == f'nested/{path.name}'

  async def run_terminal(argv: tuple[str, ...], cwd: str) -> bool:
    local = pathlib.Path(argv[-1])
    assert argv[:-1] == expected
    assert local.name == path.name
    assert local.parent == pathlib.Path(cwd)
    assert local != path
    assert local.read_text() == 'text'
    local.write_text('edited')
    return True

  picker = _Picker(host, shell.Shell(backend), run_terminal=run_terminal)
  _, emitted = picker.plugin.reduce(
    picker.current, opened.on_select(choice.value)
  )
  (effect,) = emitted
  assert isinstance(effect, extension.PrivateEffect)
  await picker.plugin.handle_effect(picker.context, effect.payload)
  assert path.read_text() == 'edited'
  assert (host / path.name).read_text() == 'host'


@pytest.mark.asyncio
async def test_invalid_editor_reports_error(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('VISUAL', '"unterminated')
  (tmp_path / 'file').touch()
  opened = await _picker(tmp_path)
  (choice,) = opened.choices

  async def run_terminal(argv: tuple[str, ...], cwd: str) -> bool:
    del argv, cwd
    pytest.fail('Invalid editor must not run')

  picker = _Picker(tmp_path, run_terminal=run_terminal)
  messages: list[actions.Action] = []
  context = dataclasses.replace(picker.context, dispatch=messages.append)
  _, emitted = picker.plugin.reduce(
    picker.current, opened.on_select(choice.value)
  )
  (effect,) = emitted
  assert isinstance(effect, extension.PrivateEffect)
  await picker.plugin.handle_effect(context, effect.payload)
  (error,) = messages
  assert isinstance(error, actions.ShowText)
  assert isinstance(error.text, str)
  assert 'Could not edit workspace file' in error.text


@pytest.mark.asyncio
async def test_editor_loads_without_a_location_declaration() -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(editor.EditorExtension(), None)
  current = _state(pathlib.Path('/workspace'))
  host = registry.create_host(
    extension.Context(
      extension.Workspace(pathlib.Path('/workspace'), remote=True),
      lambda _: None,
      lambda: current,
      lambda _: None,
      refresh_tools=registry.refresh_tools,
      call_tool=lambda name, arguments: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=mock.Mock(),
      create_agent=lambda options, on_message, on_error: pytest.fail(
        'Unexpected agent creation'
      ),
    )
  )
  try:
    assert [command.name for command in host.slash_commands] == ['editor']
  finally:
    await host.close()
    await registry.close()
