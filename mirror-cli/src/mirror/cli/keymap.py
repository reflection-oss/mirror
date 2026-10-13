"""Global keybindings, overridden by extensions and user configuration."""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib
import json
import pathlib
from collections.abc import Callable, Sequence
from typing import cast

from prompt_toolkit import application, filters
from prompt_toolkit.key_binding import key_bindings, key_processor

from mirror.cli import actions, state, workspace_application


@filters.Condition
def _input_queue_empty() -> bool:
  return not application.get_app().key_processor.input_queue


def _ignore_key(event: key_processor.KeyPressEvent) -> None:
  del event


@dataclasses.dataclass(frozen=True)
class Binding:
  """An action binding; None explicitly masks an inherited binding."""

  keys: tuple[str, ...]
  action: actions.Action | None
  filter: filters.FilterOrBool = True
  eager: bool = False
  # Run the action before passing the same keys to the normal editor bindings.
  replay: bool = False

  def __post_init__(self) -> None:
    if not self.keys:
      raise ValueError('a keybinding must contain at least one key')
    parsed = key_bindings.KeyBindings()
    parsed.add(*self.keys)(_ignore_key)
    normalized = parsed.bindings[0].keys
    if normalized == ('escape',) and not isinstance(
      self.action, actions.InterruptRequested
    ):
      raise ValueError(
        'standalone Escape is reserved for interrupt and UI dismissal'
      )
    object.__setattr__(self, 'keys', normalized)


@dataclasses.dataclass(frozen=True)
class Remapping:
  """Replace an action's inherited shortcuts; an empty list disables them."""

  action: actions.Action
  keys: tuple[tuple[str, ...], ...]

  def __post_init__(self) -> None:
    object.__setattr__(
      self, 'keys', tuple(Binding(keys, self.action).keys for keys in self.keys)
    )


class Keymap:
  """Resolve UI bindings; core terminal bindings stay available when disabled."""

  def __init__(
    self,
    layers: Callable[[], Sequence[Sequence[Binding]]],
    dispatch: Callable[[actions.Action], object],
    *,
    dismiss: Callable[[], actions.Action | None],
    remappings: Sequence[Remapping] = (),
    enabled: filters.FilterOrBool = True,
  ) -> None:
    self._layers = layers
    self._dispatch = dispatch
    self._dismiss = dismiss
    self._remappings = tuple(remappings)
    self._enabled = filters.to_filter(enabled)
    self._snapshot: tuple[Binding, ...] | None = None
    self._interrupt_on_escape = False
    self._replayed: list[key_processor.KeyPress] = []
    self._compiled = key_bindings.KeyBindings()
    self.bindings = key_bindings.DynamicKeyBindings(self.refresh)

  def refresh(self) -> key_bindings.KeyBindings:
    """Recompile and update the cache when the resolved bindings change."""
    resolved = resolve(self._layers(), self._remappings)
    if resolved == self._snapshot:
      return self._compiled
    self._interrupt_on_escape = any(
      binding.keys == ('escape',) for binding in resolved
    )
    compiled = workspace_application.terminal_bindings()
    # The parser queues Alt's Escape and following key together.
    escape_filter = self._enabled & (
      ~filters.vi_mode | filters.Condition(lambda: self._dismiss() is not None)
    )
    compiled.add('escape', filter=escape_filter, eager=_input_queue_empty)(
      self._escape
    )
    # Exact editor bindings win; unbound Alt keys must not fall back to Escape.
    compiled.add(
      'escape', '<any>', filter=escape_filter, save_before=lambda _: False
    )(_ignore_key)
    for binding in resolved:
      if binding.keys == ('escape',):
        continue
      record_undo = binding.action is not None and not isinstance(
        binding.action,
        (
          actions.ComposerUndo,
          actions.ComposerRedo,
          actions.RerenderRequested,
          actions.NavigateHistory,
          actions.CycleViewMode,
        ),
      )
      condition = self._enabled & filters.to_filter(binding.filter)
      if binding.replay:
        condition &= filters.Condition(self._not_replayed)
      handler = functools.partial(self._invoke_binding, binding)
      compiled.add(
        *binding.keys,
        filter=condition,
        eager=binding.eager or binding.replay,
        record_in_macro=not binding.replay,
        save_before=lambda _, save=record_undo and not binding.replay: save,
      )(handler)
      if binding.replay and binding.keys == ('<any>',):
        # Exact eager Escape otherwise beats an eager wildcard.
        compiled.add(
          'escape',
          filter=condition,
          eager=True,
          record_in_macro=False,
          save_before=lambda _: False,
        )(handler)
    self._snapshot = resolved
    self._compiled = compiled
    return compiled

  def _not_replayed(self) -> bool:
    return not any(
      any(key is replayed for replayed in self._replayed)
      for key in application.get_app().key_processor.key_buffer
    )

  def _invoke_binding(
    self, binding: Binding, event: key_processor.KeyPressEvent
  ) -> None:
    self._invoke(binding.action, event)
    if binding.replay:
      self._replayed = event.key_sequence
      event.key_processor.feed_multiple(event.key_sequence, first=True)

  def _escape(self, event: key_processor.KeyPressEvent) -> None:
    action = self._dismiss()
    if action is None and self._interrupt_on_escape:
      action = actions.InterruptRequested()
    self._invoke(action, event)

  def _invoke(
    self, action: actions.Action | None, event: key_processor.KeyPressEvent
  ) -> None:
    del event
    if action is not None:
      self._dispatch(action)


def dismissal_action(
  current: state.State,
  *,
  completion_visible: bool,
  rewind_escape_active: bool,
) -> actions.Action | None:
  """Return the internal Escape action for the current UI, if any."""
  if rewind_escape_active:
    return actions.CloseSelector()
  if current.resume.confirmation is not None:
    return actions.ClearResume()
  if current.secret.active:
    return actions.CancelSecret()
  if current.selector.active:
    return actions.CloseSelector()
  if current.drop_confirmation is not None:
    return actions.ClearWorktreeDrop()
  if completion_visible:
    return actions.CloseSelector()
  if current.tasks.selected_id is not None:
    return actions.ClearTaskSelection()
  return None


def defaults(
  current: state.State,
  *,
  completion_visible: bool,
  modal_active: bool,
  can_focus_tasks: bool,
  owned_history: bool = False,
) -> tuple[Binding, ...]:
  """Return defaults for the current state and live input availability."""
  bindings = [
    Binding(('enter',), actions.ComposerSubmit()),
    Binding(('c-j',), actions.ComposerNewline()),
    Binding(('c-_',), actions.ComposerUndo()),
    Binding(('c-y',), actions.ComposerRedo()),
    Binding(('escape',), actions.InterruptRequested()),
    Binding(('tab',), actions.CycleWorktreeRequested(1)),
    Binding(('s-tab',), actions.CycleWorktreeRequested(-1)),
    Binding(('f2',), actions.ToggleApprovalMode()),
    Binding(('c-s',), actions.WorkspaceTagCommand()),
    Binding(('c-c',), actions.CancelOrQuitRequested()),
    Binding(('c-d',), actions.QuitRequested()),
    Binding(('c-l',), actions.RerenderRequested()),
    Binding(('c-o',), actions.OpenWorkspaces()),
    Binding(('c-b',), actions.DetachRequested()),
    *(
      Binding((f'c-{index + 1}',), actions.SelectWorktreeRequested(index))
      for index in range(9)
    ),
  ]
  if (
    not modal_active
    and current.approval.request is None
    and current.tasks.selected_id is None
  ):
    bindings.append(Binding(('c-r',), actions.RequestPromptHistory()))
  if owned_history and not modal_active:
    bindings.extend((
      Binding(('pageup',), actions.NavigateHistory('page_up')),
      Binding(('pagedown',), actions.NavigateHistory('page_down')),
      Binding(('escape', 'home'), actions.NavigateHistory('oldest')),
      Binding(('escape', 'end'), actions.NavigateHistory('latest')),
    ))
  if can_focus_tasks:
    bindings.append(Binding(('down',), actions.SelectFirstTask()))
  if current.tasks.selected_id is not None:
    bindings.extend((
      Binding(('up',), actions.MoveTaskSelection(-1)),
      Binding(('down',), actions.MoveTaskSelection(1)),
      Binding(('x',), actions.CancelSelectedTask()),
    ))
  if completion_visible:
    bindings.extend(
      Binding((key,), actions.MoveSelection(-1)) for key in ('up', 'c-k')
    )
    bindings.extend(
      Binding((key,), actions.MoveSelection(1)) for key in ('down', 'c-j')
    )
    bindings.extend((
      Binding(('tab',), actions.AcceptSelection()),
      Binding(('c-r',), actions.RequestPromptHistory()),
    ))
  return (*bindings, *_picker_bindings(current))


def _picker_bindings(current: state.State) -> tuple[Binding, ...]:
  bindings: list[Binding] = []
  if current.drop_confirmation is not None:
    bindings.extend(
      Binding((key,), actions.ConfirmWorktreeDrop()) for key in ('y', 'Y')
    )
    bindings.extend(
      Binding((key,), actions.ClearWorktreeDrop()) for key in ('n', 'N')
    )
  if current.selector.active:
    bindings.extend(
      Binding((key,), actions.MoveSelection(-1)) for key in ('up', 'c-k')
    )
    bindings.extend(
      Binding((key,), actions.MoveSelection(1)) for key in ('down', 'c-j')
    )
    bindings.append(Binding(('enter',), actions.AcceptSelection()))
    selector = current.selector.selector
    if selector is not None and selector.on_adjust is not None:
      bindings.extend((
        Binding(('left',), actions.AdjustSelection(-1)),
        Binding(('right',), actions.AdjustSelection(1)),
      ))
  if current.secret.active:
    bindings.append(Binding(('enter',), actions.SubmitSecret()))
  if current.selector.numbered:
    bindings.extend(
      Binding((str(index + 1),), actions.AcceptSelection(index))
      for index in range(min(9, len(current.selector.filtered_choices())))
    )
  if current.selector.id == 'resume':
    bindings.append(Binding(('tab',), actions.CycleResumeScope()))
  if current.selector.id == 'workspaces':
    bindings.extend((
      Binding(('tab',), actions.MoveSelection(1)),
      Binding(('s-tab',), actions.MoveSelection(-1)),
    ))
  if current.selector.id == 'history':
    bindings.append(Binding(('c-r',), actions.MoveSelection(1)))
  if current.resume.confirmation is not None:
    bindings.extend(
      Binding((key,), actions.AcceptResumeConfirmation()) for key in ('y', 'Y')
    )
    bindings.extend(
      Binding((key,), actions.ClearResume()) for key in ('n', 'N')
    )
  if current.rewind.confirming:
    bindings.extend(
      Binding((key,), actions.ConfirmRewind()) for key in ('y', 'Y')
    )
    bindings.extend(
      Binding((key,), actions.SetRewindConfirming(False)) for key in ('n', 'N')
    )
  return tuple(bindings)


def resolve(
  layers: Sequence[Sequence[Binding]], remappings: Sequence[Remapping] = ()
) -> tuple[Binding, ...]:
  """Resolve inherited keys, then replace each configured action's shortcuts."""
  resolved: dict[tuple[str, ...], Binding] = {}
  for layer in layers:
    for binding in layer:
      resolved[binding.keys] = binding
  for remapping in remappings:
    inherited = next(
      (
        binding
        for binding in resolved.values()
        if binding.action == remapping.action
      ),
      None,
    )
    resolved = {
      keys: binding
      for keys, binding in resolved.items()
      if binding.action != remapping.action
    }
    for keys in remapping.keys:
      resolved[keys] = (
        dataclasses.replace(inherited, keys=keys)
        if inherited is not None
        else Binding(keys, remapping.action)
      )
  return tuple(resolved.values())


def _decode(value: object) -> object:
  """Construct dataclasses and enums from their Python type and field values."""
  if isinstance(value, list):
    return tuple(_decode(item) for item in cast(list[object], value))
  if not isinstance(value, dict):
    return value
  fields = dict(cast(dict[str, object], value))
  name = fields.pop('type', None)
  decoded = {key: _decode(item) for key, item in fields.items()}
  if name is None:
    return decoded
  if not isinstance(name, str):
    raise ValueError('type must be a Python class name')
  module_name, _, class_name = name.rpartition('.')
  module = importlib.import_module(module_name) if module_name else actions
  cls = cast(object, getattr(module, class_name))
  if isinstance(cls, type) and issubclass(cls, enum.Enum):
    return cls(**decoded)
  if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
    raise ValueError(f'{name!r} is not a dataclass or enum')
  return cls(**decoded)


def _config_remapping(value: object) -> Remapping:
  if not isinstance(value, dict):
    raise TypeError('each keybinding must be an object')
  fields = cast(dict[str, object], value)
  if fields.keys() - {'keys', 'action'}:
    raise ValueError(f'unknown keybinding fields: {fields.keys()}')
  keys = fields.get('keys')
  if not isinstance(keys, list):
    raise TypeError('keys must be a list of key sequences')
  sequences: list[tuple[str, ...]] = []
  for sequence in cast(list[object], keys):
    if not isinstance(sequence, list):
      raise TypeError('each key sequence must be a list of strings')
    raw_keys = cast(list[object], sequence)
    if not all(isinstance(key, str) for key in raw_keys):
      raise TypeError('key names must be strings')
    sequences.append(tuple(cast(list[str], raw_keys)))
  decoded = _decode(fields.get('action'))
  if isinstance(decoded, type) or not dataclasses.is_dataclass(decoded):
    raise ValueError('action must describe a dataclass instance')
  return Remapping(cast(actions.Action, decoded), tuple(sequences))


def load_config(path: pathlib.Path | None = None) -> tuple[Remapping, ...]:
  """Read ~/.mirror/keybindings.json; project directories cannot supply overrides."""
  if path is None:
    path = pathlib.Path.home() / '.mirror' / 'keybindings.json'
  try:
    contents = path.read_text(encoding='utf-8')
  except FileNotFoundError:
    return ()
  try:
    values = cast(object, json.loads(contents))
    if not isinstance(values, list):
      raise ValueError('keybindings config must be a list')
    return tuple(
      _config_remapping(value) for value in cast(list[object], values)
    )
  except (ValueError, TypeError, AttributeError, ImportError) as error:
    raise ValueError(
      f'Invalid keybindings config {str(path)!r}: {error}'
    ) from error
