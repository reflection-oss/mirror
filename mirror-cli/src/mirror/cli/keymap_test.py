from __future__ import annotations

import json
import pathlib

import pytest
from prompt_toolkit import application, buffer, keys, layout
from prompt_toolkit.input import ansi_escape_sequences, vt100_parser
from prompt_toolkit.input import base as input_base
from prompt_toolkit.output import base as output_base

from mirror.cli import actions, keymap, selection, state


@pytest.mark.parametrize('split', [0, 1])
@pytest.mark.parametrize(
  ('sequence', 'text', 'cursor', 'dispatched_actions'),
  [
    ('\x1bb', 'one two', 0, []),
    ('\x1bf', 'one two', 7, []),
    ('\x1b[1;3D', 'one two', 0, []),
    ('\x1b[1;3C', 'one two', 7, []),
    ('\x1b\x7f', 'two', 0, []),
    ('\x1b\x08', 'two', 0, []),
    ('\x1bd', 'one ', 4, []),
    ('\x1bu', 'one TWO', 7, []),
    ('\x1b!', 'one two', 4, []),
    ('\x1b[1;3A', 'one two', 4, []),
    ('\x1b', 'one two', 4, [actions.InterruptRequested()]),
  ],
)
@pytest.mark.asyncio
async def test_escape_preserves_editor_alt_bindings_without_delay(
  split: int,
  sequence: str,
  text: str,
  cursor: int,
  dispatched_actions: list[actions.Action],
) -> None:
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [(keymap.Binding(('escape',), actions.InterruptRequested()),)],
    dispatched.append,
    dismiss=lambda: None,
  )
  composer = buffer.Buffer()
  composer.text = 'one two'
  composer.cursor_position = 4
  app = application.Application[None](
    layout=layout.Layout(layout.Window(layout.BufferControl(composer))),
    key_bindings=mapping.bindings,
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  with application.set_app(app):
    parser.feed(sequence[:split])
    app.key_processor.process_keys()
    assert dispatched == []
    assert composer.text == 'one two'
    parser.feed(sequence[split:])
    parser.flush()
    app.key_processor.process_keys()
    assert composer.text == text
    assert composer.cursor_position == cursor
    assert dispatched == dispatched_actions
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.parametrize(
  'sequence',
  [
    sequence
    for sequence, key in ansi_escape_sequences.ANSI_SEQUENCES.items()
    if sequence.startswith('\x1b')
    and key not in (keys.Keys.Escape, keys.Keys.BracketedPaste)
  ]
  + ['\x1b' + chr(code) for code in range(32, 127)]
  + ['\x1bé', '\x1b[200~paste\x1b[201~'],
)
@pytest.mark.asyncio
async def test_supported_escape_sequences_never_interrupt(
  sequence: str,
) -> None:
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [(keymap.Binding(('escape',), actions.InterruptRequested()),)],
    dispatched.append,
    dismiss=lambda: None,
  )
  app = application.Application[None](
    key_bindings=mapping.bindings,
    layout=layout.Layout(layout.Window(layout.BufferControl(buffer.Buffer()))),
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  with application.set_app(app):
    parser.feed(sequence)
    parser.flush()
    app.key_processor.process_keys()
    assert dispatched == []
    assert app.key_processor.key_buffer == []
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.asyncio
async def test_single_key_binding_preserves_builtin_editor_chord() -> None:
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [(keymap.Binding(('c-x',), actions.InterruptRequested()),)],
    dispatched.append,
    dismiss=lambda: None,
  )
  composer = buffer.Buffer()
  composer.text = 'original'
  composer.save_to_undo_stack()
  composer.text = 'edited'
  app = application.Application[None](
    key_bindings=mapping.bindings,
    layout=layout.Layout(layout.Window(layout.BufferControl(composer))),
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  with application.set_app(app):
    parser.feed('\x18')
    app.key_processor.process_keys()
    assert dispatched == []
    parser.feed('\x15')
    app.key_processor.process_keys()
    assert composer.text == 'original'
    assert dispatched == []
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.parametrize(('prefix', 'encoded'), [('c-x', '\x18'), ('\\', '\\')])
@pytest.mark.asyncio
async def test_configured_chord_wins_over_prefix_and_alt_fallback(
  prefix: str, encoded: str
) -> None:
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [
      (
        keymap.Binding((prefix,), actions.InterruptRequested()),
        keymap.Binding((prefix, '!'), actions.CloseSelector()),
      )
    ],
    dispatched.append,
    dismiss=lambda: None,
  )
  app = application.Application[None](
    key_bindings=mapping.bindings,
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  with application.set_app(app):
    parser.feed(encoded)
    parser.flush()
    app.key_processor.process_keys()
    assert dispatched == []
    parser.feed('!')
    app.key_processor.process_keys()
    assert dispatched == [actions.CloseSelector()]
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.parametrize(
  ('remappings', 'after_escape', 'after_ctrl_c'),
  [
    (
      (),
      [actions.InterruptRequested()],
      [actions.InterruptRequested(), actions.QuitRequested()],
    ),
    (
      (keymap.Remapping(actions.InterruptRequested(), (('c-c',),)),),
      [],
      [actions.InterruptRequested()],
    ),
    (
      (keymap.Remapping(actions.InterruptRequested(), ()),),
      [],
      [actions.QuitRequested()],
    ),
    (
      (
        keymap.Remapping(actions.InterruptRequested(), (('escape',), ('c-c',))),
      ),
      [actions.InterruptRequested()],
      [actions.InterruptRequested(), actions.InterruptRequested()],
    ),
  ],
)
@pytest.mark.asyncio
async def test_interrupt_remapping_replaces_defaults(
  remappings: tuple[keymap.Remapping, ...],
  after_escape: list[actions.Action],
  after_ctrl_c: list[actions.Action],
) -> None:
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [
      (
        keymap.Binding(('escape',), actions.InterruptRequested()),
        keymap.Binding(('c-c',), actions.QuitRequested()),
      )
    ],
    dispatched.append,
    dismiss=lambda: None,
    remappings=remappings,
  )
  app = application.Application[None](
    key_bindings=mapping.bindings,
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  with application.set_app(app):
    parser.feed_and_flush('\x1b')
    app.key_processor.process_keys()
    assert dispatched == after_escape
    parser.feed('\x03')
    app.key_processor.process_keys()
    assert dispatched == after_ctrl_c
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.asyncio
async def test_disabling_interrupt_does_not_disable_escape_dismissal() -> None:
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: [(keymap.Binding(('escape',), actions.InterruptRequested()),)],
    dispatched.append,
    dismiss=actions.CloseSelector,
    remappings=(keymap.Remapping(actions.InterruptRequested(), ()),),
  )
  app = application.Application[None](
    key_bindings=mapping.bindings,
    input=input_base.DummyInput(),
    output=output_base.DummyOutput(),
  )
  parser = vt100_parser.Vt100Parser(app.key_processor.feed)
  with application.set_app(app):
    parser.feed_and_flush('\x1b')
    app.key_processor.process_keys()
    assert dispatched == [actions.CloseSelector()]
    await app.cancel_and_wait_for_background_tasks()


@pytest.mark.parametrize('action', [None, actions.QuitRequested()])
def test_binding_rejects_overriding_reserved_escape(
  action: actions.Action | None,
) -> None:
  with pytest.raises(ValueError, match='standalone Escape is reserved'):
    keymap.Binding(('escape',), action)


def test_remapping_replaces_all_inherited_shortcuts_for_matching_action() -> (
  None
):
  decrement = keymap.Binding(('up',), actions.MoveSelection(-1))
  layers = [
    (
      decrement,
      keymap.Binding(('down',), actions.MoveSelection(1)),
      keymap.Binding(('c-j',), actions.MoveSelection(1)),
    )
  ]
  remappings = (keymap.Remapping(actions.MoveSelection(1), (('f6',), ('f7',))),)
  assert keymap.resolve(layers, remappings) == (
    decrement,
    keymap.Binding(('f6',), actions.MoveSelection(1)),
    keymap.Binding(('f7',), actions.MoveSelection(1)),
  )


@pytest.mark.parametrize(
  ('selector_active', 'selected_task', 'completion', 'rewind', 'expected'),
  [
    (False, None, False, False, None),
    (False, 'task', False, False, actions.ClearTaskSelection()),
    (False, 'task', True, False, actions.CloseSelector()),
    (True, None, True, False, actions.CloseSelector()),
    (True, None, True, True, actions.CloseSelector()),
  ],
)
def test_dismissal_action_preserves_context_priority(
  selector_active: bool,
  selected_task: str | None,
  completion: bool,
  rewind: bool,
  expected: actions.Action | None,
) -> None:
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('test', 'medium', 'auto'),
    session=state.SessionState('test'),
    lifecycle=state.LifecycleState(active=True),
    selector=state.SelectorState(
      menu=state.MenuState(active=selector_active),
      selector=selection.Selector(
        'mcp-auth', 'Authenticate', (), lambda _: actions.ClearSelector()
      ),
    ),
    tasks=state.TasksState(selected_id=selected_task),
  )
  assert (
    keymap.dismissal_action(
      current, completion_visible=completion, rewind_escape_active=rewind
    )
    == expected
  )


@pytest.mark.parametrize('busy', [False, True])
def test_default_workspace_and_background_shortcuts(busy: bool) -> None:
  current = state.State(
    activity=state.ActivityState(busy=busy),
    status_bar=state.StatusBarState('test', 'medium', 'auto'),
    session=state.SessionState('test'),
    lifecycle=state.LifecycleState(active=True),
  )
  bindings = keymap.resolve([
    keymap.defaults(
      current,
      completion_visible=False,
      modal_active=False,
      can_focus_tasks=False,
    )
  ])

  assert (
    keymap.dismissal_action(
      current, completion_visible=False, rewind_escape_active=False
    )
    is None
  )
  assert keymap.Binding(('escape',), actions.InterruptRequested()) in bindings
  assert keymap.Binding(('c-o',), actions.OpenWorkspaces()) in bindings
  assert keymap.Binding(('c-b',), actions.DetachRequested()) in bindings
  assert keymap.Binding(('c-o',), actions.DetachRequested()) not in bindings


def test_workspace_selector_tab_browses_instead_of_switching() -> None:
  current = state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('test', 'medium', 'auto'),
    session=state.SessionState('test'),
    lifecycle=state.LifecycleState(active=True),
    selector=state.SelectorState(
      menu=state.MenuState(active=True),
      selector=selection.Selector(
        'workspaces', 'Workspaces', (), actions.SelectWorkspaceRequested
      ),
    ),
  )

  bindings = keymap.resolve([
    keymap.defaults(
      current,
      completion_visible=False,
      modal_active=True,
      can_focus_tasks=False,
    )
  ])

  assert keymap.Binding(('tab',), actions.MoveSelection(1)) in bindings
  assert keymap.Binding(('s-tab',), actions.MoveSelection(-1)) in bindings


def test_keymap_refresh_caches_and_rebuilds_without_dispatching() -> None:
  layers = [(keymap.Binding(('f6',), actions.CloseSelector()),)]
  dispatched: list[actions.Action] = []
  mapping = keymap.Keymap(
    lambda: layers, dispatched.append, dismiss=lambda: None
  )

  first = mapping.refresh()
  assert mapping.refresh() is first
  layers.append((keymap.Binding(('f6',), None),))
  overridden = mapping.refresh()
  assert overridden is not first
  assert mapping.refresh() is overridden
  assert dispatched == []


def test_resolve_prefers_config_over_extensions_over_defaults() -> None:
  default = keymap.Binding(('c-p',), actions.MoveSelection(-1))
  plugin = keymap.Binding(('c-p',), actions.AcceptSelection())
  configured = keymap.Binding(('c-p',), actions.CloseSelector())
  assert keymap.resolve(((default,), (plugin,), (configured,))) == (configured,)
  assert keymap.resolve(((default,), (plugin,))) == (plugin,)
  assert keymap.resolve(((default,),)) == (default,)


def test_resolve_last_registration_wins_for_equivalent_keys() -> None:
  first = keymap.Binding(('tab',), actions.AcceptSelection())
  later = keymap.Binding(('c-i',), actions.CloseSelector())
  assert keymap.resolve(((first, later),)) == (later,)


def test_resolve_unbinding_masks_inherited_actions() -> None:
  default = keymap.Binding(('f6',), actions.CloseSelector())
  unbind = keymap.Binding(('f6',), None)
  assert keymap.resolve(((default,), (unbind,))) == (unbind,)


def test_load_config_reads_action_shortcuts_and_disabling(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'keys.json'
  path.write_text(
    json.dumps([
      {
        'keys': [['c-x', 'j'], ['f6']],
        'action': {'type': 'MoveSelection', 'delta': 1},
      },
      {'keys': [], 'action': {'type': 'QuitRequested'}},
    ])
  )
  assert keymap.load_config(path) == (
    keymap.Remapping(actions.MoveSelection(1), (('c-x', 'j'), ('f6',))),
    keymap.Remapping(actions.QuitRequested(), ()),
  )


@pytest.mark.parametrize(
  'value',
  [
    {},
    [{'keys': [['f6']], 'action': {'type': 'MissingAction'}}],
    [{'keys': [['f6']], 'action': {'type': 'MoveSelection'}}],
    [{'keys': [['f6']], 'action': {'type': 'MoveSelection', 'unknown': 1}}],
    [{'keys': [list[str]()], 'action': {'type': 'QuitRequested'}}],
    [{'keys': [['invalid-key']], 'action': {'type': 'QuitRequested'}}],
    [{'keys': ['c-c'], 'action': {'type': 'InterruptRequested'}}],
    [{'keys': [[42]], 'action': {'type': 'InterruptRequested'}}],
    [{'keys': [['escape']], 'action': {'type': 'QuitRequested'}}],
    [{'keys': [['c-p']]}],
    [{'keys': [['c-p']], 'action': None}],
    [{'context': 'unknown', 'keys': [['c-p']], 'action': None}],
    [{'keys': [['c-p']], 'action': None, 'args': {'delta': 1}}],
    [{'keys': [['c-p']], 'action': 'x', 'args': {'nested': []}}],
    [{'keys': [['c-p']], 'action': 'x', 'typo': True}],
  ],
)
def test_load_config_rejects_malformed_bindings(
  tmp_path: pathlib.Path, value: object
) -> None:
  path = tmp_path / 'keys.json'
  path.write_text(json.dumps(value))
  with pytest.raises(ValueError, match='Invalid keybindings config'):
    keymap.load_config(path)


@pytest.mark.parametrize(
  ('value', 'expected'),
  [
    ({'type': 'DecideApproval'}, actions.DecideApproval()),
    ({'type': 'WorkerFailed', 'message': 'test'}, actions.WorkerFailed('test')),
    (
      {'type': 'SetViewMode', 'mode': {'type': 'ViewMode', 'value': 'compact'}},
      actions.SetViewMode(actions.ViewMode.COMPACT),
    ),
  ],
)
def test_load_config_constructs_existing_types_without_registration(
  tmp_path: pathlib.Path,
  value: dict[str, object],
  expected: actions.Action,
) -> None:
  path = tmp_path / 'keys.json'
  path.write_text(json.dumps([{'keys': [['f6']], 'action': value}]))
  assert keymap.load_config(path) == (keymap.Remapping(expected, (('f6',),)),)


def test_load_config_absent_file_has_no_overrides(
  tmp_path: pathlib.Path,
) -> None:
  assert keymap.load_config(tmp_path / 'missing.json') == ()


def test_history_navigation_is_remappable_through_the_existing_config(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'keys.json'
  path.write_text(
    json.dumps([
      {
        'keys': [['c-p']],
        'action': {'type': 'NavigateHistory', 'direction': 'page_up'},
      }
    ])
  )
  remappings = keymap.load_config(path)
  inherited = keymap.Binding(('pageup',), actions.NavigateHistory('page_up'))
  assert keymap.resolve(((inherited,),), remappings) == (
    keymap.Binding(('c-p',), actions.NavigateHistory('page_up')),
  )
