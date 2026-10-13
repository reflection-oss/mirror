"""Let the agent create, load, reload, and unload trusted Python extensions."""

from __future__ import annotations

import json
import pathlib
from typing import cast, override

from mirror.cli import extension, host_tools
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event

_WIDGET_TEMPLATE = '''"""A live user-created Mirror widget."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import override

from mirror.cli import actions, extension, state
from prompt_toolkit.layout import containers, controls


def _content() -> str:
  return 'Widget'


@dataclasses.dataclass(frozen=True)
class _Updated(actions.ExtensionAction):
  text: str


class Widget(extension.Extension[str, None, None]):

  @classmethod
  @override
  def id(cls) -> str:
    return __EXTENSION_ID__

  @staticmethod
  @override
  def initial_state() -> str:
    return 'Loading widget…'

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_BELOW_PROMPT

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[str, tuple[extension.PrivateEffect[None], ...]]:
    current = self.require_state(global_state)
    if isinstance(action, actions.ExtensionLoaded):
      return current, (self.effect(None),)
    if isinstance(action, actions.RoutedExtensionAction) and isinstance(
      action.action, _Updated
    ):
      return action.action.text, ()
    return current, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: None
  ) -> None:
    while True:
      self.dispatch(context, _Updated(_content()))
      await asyncio.sleep(1)

  @override
  def render(
    self, global_state: state.State, extension_state: str
  ) -> containers.AnyContainer:
    return containers.Window(
      controls.FormattedTextControl([('class:footer', extension_state)]),
      height=1,
    )
'''


class UserExtensions(extension.Extension[None, None, None]):
  """Opt in to agent-driven extension authoring without owning widget lifetimes."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/user-extensions'

  @override
  def default_enabled(self) -> bool:
    return False

  @override
  def tips(self) -> tuple[str, ...]:
    return (
      'Mirror can build extensions: try '
      '"Build a CPU and memory monitor extension"',
      'Mirror can build extensions: try "build me a pet extension"',
    )

  @override
  def skills(self) -> tuple[pathlib.Path, ...]:
    return (pathlib.Path(__file__).with_name('SKILL.md'),)

  @override
  def tools(self, context: extension.Context) -> list[host_tools.HostTool]:
    return (
      [UserExtensionsTool(context.extensions)]
      if context.extensions is not None
      else []
    )


class UserExtensionsTool:
  """Author user extensions through the core's lifecycle services."""

  def __init__(self, manager: extension.ExtensionLifecycle) -> None:
    self._manager = manager

  @property
  def definition(self) -> host_tools.ToolDefinition:
    return host_tools.ToolDefinition(
      name='extensions',
      description=(
        f'Manage trusted Python UI extensions in {self._manager.root}/<name>/extension.py. '
        'For a new widget: create a scaffold, edit it, then load. '
        'Use name for create/load and extension_id for reload/unload; the folder name is the extension ID. '
        'list shows loaded extensions and per-workspace faults. '
        'Code runs unsandboxed in every local workspace. '
        'load/unload persist on/off choices; unload keeps files. /extensions toggles them interactively.'
      ),
      parameters={
        'type': 'object',
        'properties': {
          'action': {
            'type': 'string',
            'enum': ['list', 'create', 'load', 'reload', 'unload'],
          },
          'name': {
            'type': 'string',
            'description': 'Single directory name, e.g. clock; required for create and load.',
          },
          'extension_id': {
            'type': 'string',
            'description': 'Unique ID returned by load or list, e.g. clock; required for reload and unload.',
          },
        },
        'required': ['action'],
        'additionalProperties': False,
      },
      needs_approval=True,
    )

  def _directory(self, name: str) -> pathlib.Path:
    if not name or pathlib.Path(name).name != name or name in {'.', '..'}:
      raise ValueError('name must be a single extension directory name')
    directory = (self._manager.root / name).resolve()
    if (
      directory.parent != self._manager.root
      or (directory / 'extension.py').resolve().parent != directory
    ):
      raise ValueError('extension source must remain inside its user directory')
    return directory

  def _create(self, directory: pathlib.Path) -> host_tools.ToolResult:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'extension.py'
    extension_id = directory.name
    source = _WIDGET_TEMPLATE.replace('__EXTENSION_ID__', repr(extension_id))
    with path.open('x', encoding='utf-8') as output:
      output.write(source)
    statsig_client.get().log_event(
      statsig_event.Event('mirror_extension_created', value=1)
    )
    return host_tools.ToolResult(
      f'Created {path}. Edit, then load with name={directory.name!r}; '
      f'after further edits, reload with extension_id={extension_id!r}. Keep the ID equal to the folder name.\n'
      'Describe the widget in its module docstring (shown in /extensions). '
      'Customize _content(); put async I/O in handle_effect, not render/reduce. '
      'Preserve routed updates and cancellation. '
      'Protocol imports: from mirror.protocol import agent_pb2.\n\n'
      f'```python\n{source}```',
      True,
    )

  async def execute(
    self,
    arguments: str,
    context: host_tools.ExecutionContext = host_tools.MODEL_CONTEXT,
  ) -> host_tools.ToolResult:
    registry = self._manager.registry
    try:
      value: object = json.loads(arguments)
      if not isinstance(value, dict):
        raise ValueError('arguments must be a JSON object')
      parsed = cast(dict[str, object], value)
      action = parsed.get('action')
      if action == 'list':
        loaded: dict[str, object] = {}
        for extension_id in registry.extension_ids():
          directory = registry.source_directory(extension_id)
          if directory is not None and directory.parent == self._manager.root:
            loaded[extension_id] = {
              'directory': str(directory),
              'workspaces': [
                {
                  'cwd': str(status.workspace.cwd),
                  'remote': status.workspace.remote,
                  'status': status.status,
                  'error': status.error,
                }
                for status in registry.statuses(extension_id)
              ],
            }
        return host_tools.ToolResult(json.dumps(loaded), True)
      if action in ('create', 'load'):
        name = parsed.get('name')
        if not isinstance(name, str):
          raise ValueError('name is required for create and load')
        directory = self._directory(name)
        if action == 'create':
          return self._create(directory)
        extension_id = await self._manager.set_enabled(directory, True)
      elif action in ('reload', 'unload'):
        extension_id = parsed.get('extension_id')
        if not isinstance(extension_id, str):
          raise ValueError('extension_id is required for reload and unload')
        directory = registry.source_directory(extension_id)
        if directory is None or directory.parent != self._manager.root:
          raise ValueError(f'not a user extension: {extension_id!r}')
        if action == 'unload':
          await self._manager.set_enabled(directory, False)
          return host_tools.ToolResult(f'Unloaded {extension_id}', True)
        self._directory(directory.name)
        loaded_id = await self._manager.reload(extension_id)
        if loaded_id is None:
          return host_tools.ToolResult(
            f'Disabled {extension_id} by configuration', True
          )
        extension_id = loaded_id
      else:
        raise ValueError(f'unknown action: {action!r}')
      return host_tools.ToolResult(f'Loaded {extension_id}', True)
    except (ValueError, OSError) as error:
      return host_tools.ToolResult(str(error), False)
