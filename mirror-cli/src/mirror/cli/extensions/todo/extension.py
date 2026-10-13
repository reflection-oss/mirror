"""A durable todo extension for the Mirror CLI."""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import cast, override

from google.protobuf import json_format
from mirror.protocol import agent_pb2
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import actions, extension, host_tools
from mirror.cli import state as _state

_STATE_KIND = 'state'
_MAX_VISIBLE_ITEMS = 5


@dataclasses.dataclass(frozen=True)
class Todo:
  """One immutable todo item."""

  id: int
  text: str
  done: bool = False


@dataclasses.dataclass(frozen=True)
class TodoState:
  """The current todo snapshot."""

  todos: tuple[Todo, ...] = ()
  next_id: int = 1


@dataclasses.dataclass(frozen=True)
class _Add(actions.ExtensionAction):
  text: str


@dataclasses.dataclass(frozen=True)
class _Toggle(actions.ExtensionAction):
  id: int


@dataclasses.dataclass(frozen=True)
class _Clear(actions.ExtensionAction):
  pass


@dataclasses.dataclass(frozen=True)
class _Persist:
  state: TodoState


class TodoExtension(extension.Extension[TodoState, _Persist, None]):
  """Manage model-facing todos as immutable extension state."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/todo'

  @override
  def default_enabled(self) -> bool:
    return False

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_ABOVE_PROMPT

  @staticmethod
  @override
  def initial_state() -> TodoState:
    return TodoState()

  @override
  def reduce(
    self,
    global_state: _state.State,
    action: actions.Action,
  ) -> tuple[TodoState, tuple[extension.PrivateEffect[_Persist], ...]]:
    current = self.require_state(global_state)
    if not isinstance(action, actions.RoutedExtensionAction):
      return current, ()
    match action.action:
      case _Add(text):
        todo = Todo(current.next_id, text)
        next_state = TodoState((*current.todos, todo), current.next_id + 1)
      case _Toggle(todo_id):
        todos = tuple(
          dataclasses.replace(todo, done=not todo.done)
          if todo.id == todo_id
          else todo
          for todo in current.todos
        )
        next_state = dataclasses.replace(current, todos=todos)
      case _Clear():
        next_state = TodoState()
      case _:
        return current, ()
    return next_state, (self.effect(_Persist(next_state)),)

  @override
  def hydrate(
    self,
    snapshot: agent_pb2.ExtensionRecord | None,
    records: Sequence[agent_pb2.AgentMessage],
    global_state: _state.State,
  ) -> TodoState:
    del records, global_state
    if snapshot is None or snapshot.kind != _STATE_KIND:
      return self.initial_state()
    return (
      _state_from_data(json_format.MessageToDict(snapshot.data))
      or self.initial_state()
    )

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _Persist
  ) -> None:
    message = agent_pb2.AgentMessage(
      id=uuid.uuid4().hex,
      extension=agent_pb2.ExtensionRecord(
        extension=self.id(),
        kind=_STATE_KIND,
        survives_compaction=True,
      ),
    )
    json_format.ParseDict(dict(_data(effect.state)), message.extension.data)
    context.enqueue(message)

  @override
  def render(
    self,
    global_state: _state.State,
    extension_state: TodoState,
  ) -> containers.AnyContainer:
    """Render the current todo snapshot below the composer."""
    current = extension_state
    if not current.todos:
      return containers.Window(height=0)
    completed = sum(todo.done for todo in current.todos)
    fragments: StyleAndTextTuples = [
      ('class:ui.accent bold', f'Todos {completed}/{len(current.todos)}\n')
    ]
    text_limit = max(8, global_state.layout.columns - 10)
    for todo in current.todos[:_MAX_VISIBLE_ITEMS]:
      marker = '✓' if todo.done else '○'
      style = 'class:ui.faint' if todo.done else 'class:ui.text'
      text = ' '.join(todo.text.split())
      if len(text) > text_limit:
        text = f'{text[: text_limit - 1].rstrip()}…'
      fragments.append((style, f'  {marker} #{todo.id} {text}\n'))
    remaining = len(current.todos) - _MAX_VISIBLE_ITEMS
    if remaining > 0:
      fragments.append(('class:ui.faint', f'  … {remaining} more\n'))
    return containers.Window(
      controls.FormattedTextControl(fragments),
      height=dimension.Dimension.exact(len(fragments)),
    )

  @override
  def tools(self, context: extension.Context) -> list[host_tools.HostTool]:
    return [_TodoTool(self, context)]


class _TodoTool:
  """Workspace-bound todo tool execution."""

  def __init__(self, plugin: TodoExtension, context: extension.Context) -> None:
    self._plugin = plugin
    self._context = context

    self.definition = host_tools.ToolDefinition(
      name='todo',
      description=(
        'Manage a todo list. Actions: list, add (text), toggle (id), clear.'
      ),
      parameters={
        'type': 'object',
        'properties': {
          'action': {
            'type': 'string',
            'enum': ['list', 'add', 'toggle', 'clear'],
          },
          'text': {'type': 'string'},
          'id': {'type': 'integer'},
        },
        'required': ['action'],
        'additionalProperties': False,
      },
      needs_approval=False,
    )

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    try:
      value: object = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as error:
      return host_tools.ToolResult(f'Invalid tool arguments: {error}', False)
    if not isinstance(value, dict):
      return host_tools.ToolResult(
        'Tool arguments must be a JSON object', False
      )
    parsed = cast(dict[str, object], value)
    extension_context = self._context
    action = parsed.get('action')
    if action == 'list':
      return host_tools.ToolResult(self._list(extension_context), True)
    if action == 'add':
      text = parsed.get('text')
      if not isinstance(text, str) or not text.strip():
        return self._error('text is required')
      self._plugin.dispatch(extension_context, _Add(text.strip()))
      added = self._plugin.current_state(extension_context).todos[-1]
      return host_tools.ToolResult(
        f'Added todo #{added.id}: {added.text}', True
      )
    if action == 'toggle':
      todo_id = parsed.get('id')
      if type(todo_id) is not int:
        return self._error('id is required')
      todo = next(
        (
          item
          for item in self._plugin.current_state(extension_context).todos
          if item.id == todo_id
        ),
        None,
      )
      if todo is None:
        return self._error(f'todo #{todo_id} not found')
      self._plugin.dispatch(extension_context, _Toggle(todo_id))
      status = 'completed' if not todo.done else 'uncompleted'
      return host_tools.ToolResult(f'Todo #{todo_id} {status}', True)
    if action == 'clear':
      count = len(self._plugin.current_state(extension_context).todos)
      self._plugin.dispatch(extension_context, _Clear())
      return host_tools.ToolResult(f'Cleared {count} todos', True)
    return self._error(f'unknown action: {action!r}')

  def _list(self, context: extension.Context) -> str:
    return (
      '\n'.join(
        f'[{"x" if todo.done else " "}] #{todo.id}: {todo.text}'
        for todo in self._plugin.current_state(context).todos
      )
      or 'No todos'
    )

  def _error(self, message: str) -> host_tools.ToolResult:
    return host_tools.ToolResult(f'Error: {message}', False)


def _data(current: TodoState) -> Mapping[str, object]:
  return {
    'todos': [
      {'id': todo.id, 'text': todo.text, 'done': todo.done}
      for todo in current.todos
    ],
    'next_id': current.next_id,
  }


def _state_from_data(data: Mapping[str, object]) -> TodoState | None:
  raw_todos = data.get('todos')
  next_id = _integer(data.get('next_id'))
  if not isinstance(raw_todos, list) or next_id is None or next_id < 1:
    return None
  todos: list[Todo] = []
  for raw_value in cast(list[object], raw_todos):
    if not isinstance(raw_value, dict):
      return None
    raw_todo = cast(dict[str, object], raw_value)
    todo_id = _integer(raw_todo.get('id'))
    text = raw_todo.get('text')
    done = raw_todo.get('done')
    if todo_id is None or not isinstance(text, str) or type(done) is not bool:
      return None
    todos.append(Todo(todo_id, text, done))
  if todos and next_id <= max(todo.id for todo in todos):
    return None
  return TodoState(tuple(todos), next_id)


def _integer(value: object) -> int | None:
  if type(value) is int:
    return value
  if isinstance(value, float) and value.is_integer():
    return int(value)
  return None
