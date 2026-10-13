"""Typed state reducers and effect handlers for CLI extensions."""

from __future__ import annotations

import abc
import asyncio
import contextlib
import copy
import dataclasses
import enum
import functools
import importlib.util
import inspect
import logging
import pathlib
import sys
import types
import uuid
import weakref
from collections.abc import (
  Awaitable,
  Callable,
  Generator,
  Mapping,
  Sequence,
)
from typing import Any, Literal, Protocol, cast, override

from mirror.protocol import agent_pb2
from prompt_toolkit import application
from prompt_toolkit.layout import containers, controls

from mirror import config, skills
from mirror.cli import (
  actions,
  agents,
  commands,
  effects,
  extension_faults,
  extension_rendering,
  extension_state,
  host_tools,
  keymap,
  mcp_configuration,
  rpc_client,
  shell,
  state,
  workspace_controller,
  workspace_files,
)
from mirror.cli.components import component
from mirror.cli.composer import editing
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event


@dataclasses.dataclass(frozen=True)
class Workspace:
  """The workspace environment available to an extension."""

  cwd: pathlib.Path
  remote: bool
  id: str | None = None
  shell: shell.Shell | None = dataclasses.field(
    default=None, compare=False, repr=False
  )

  async def run(
    self,
    command: str,
    *,
    data: bytes = b'',
    timeout: float = 30,
    limit: int = 2 * 1024 * 1024,
  ) -> shell.Result:
    """Run Bash in this workspace, independently of agent/tool lifecycle."""
    if self.shell is None:
      raise RuntimeError('Workspace process execution is unavailable')
    return await self.shell.run(
      ('/bin/bash', '-c', command), data=data, timeout=timeout, limit=limit
    )

  async def read(
    self, path: str | pathlib.PurePath, *, max_bytes: int = 2 * 1024 * 1024
  ) -> str:
    """Read strict UTF-8 with a byte limit, preserving newlines; no symlinks."""
    if self.shell is None:
      raise RuntimeError('Workspace file access is unavailable')
    return await workspace_files.read(self.shell, path, max_bytes=max_bytes)

  async def write(
    self,
    path: str | pathlib.PurePath,
    content: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    expected: str | None = None,
  ) -> None:
    """Atomically replace UTF-8 text without following symlinks.

    Missing parent directories are created. Supplying `expected` requires an
    existing file with those contents and checks for changes before replacing it.
    """
    if self.shell is None:
      raise RuntimeError('Workspace file access is unavailable')
    await workspace_files.write(
      self.shell, path, content, max_bytes=max_bytes, expected=expected
    )


@dataclasses.dataclass(frozen=True)
class ExtensionStatus:
  """An extension's lifecycle and last fault in one applicable workspace."""

  workspace: Workspace
  status: Literal['not started', 'active', 'disabled']
  error: str | None


class ExtensionLifecycle(Protocol):
  """Core lifecycle operations shared by the menu and extension tools."""

  root: pathlib.Path
  registry: ExtensionRegistry

  async def set_enabled(
    self, directory: pathlib.Path, enabled: bool
  ) -> str | None: ...

  async def reload(self, extension_id: str) -> str | None: ...


@dataclasses.dataclass(frozen=True)
class Context:
  """Restricted services and environment available to one extension."""

  workspace: Workspace
  dispatch: Callable[[actions.Action], object]
  get_state: Callable[[], state.State]
  enqueue: Callable[[agent_pb2.AgentMessage], None]
  refresh_tools: Callable[[str], Awaitable[None]]
  call_tool: Callable[
    [str, Mapping[str, object]], Awaitable[host_tools.ToolResult]
  ]
  transcript: Callable[[], Sequence[agent_pb2.AgentMessage]]
  create_agent: Callable[
    [agents.Options, rpc_client.MessageHandler, rpc_client.ErrorHandler],
    rpc_client.Client,
  ]
  builtins: Callable[[], tuple[str, ...]]
  extension_tools: Callable[[], tuple[str, ...]]
  # Extensions may orchestrate the shared roster.
  workspace_controller: workspace_controller.WorkspaceController
  config: config.JsonObject = dataclasses.field(
    default_factory=dict[str, config.JsonValue]
  )
  # Relaunch this workspace's local worker with `updates` overlaid on its
  # launch values; None where the workspace runs remotely.
  restart_worker: Callable[[Mapping[str, object]], Awaitable[bool]] | None = (
    None
  )
  # The local agent worker's command line and host directory, for its subcommands.
  worker_command: tuple[str, ...] = ()
  worker_cwd: pathlib.Path | None = None
  run_terminal: Callable[[tuple[str, ...], str], Awaitable[bool]] | None = None
  # Configured server, including while the current workspace is local.
  remote_endpoint: str | None = None
  remote_secure: bool = True
  extensions: ExtensionLifecycle | None = None


@dataclasses.dataclass(frozen=True)
class PrivateEffect[EffectT]:
  """A payload handled by the extension that emitted it."""

  payload: EffectT


class RenderRegion(enum.Enum):
  """A region where an extension component can render."""

  STATUS_LEFT = 'status_left'
  STATUS_RIGHT = 'status_right'
  WIDGET_BELOW_ACTIVITY = 'widget_below_activity'
  WIDGET_ABOVE_PROMPT = 'widget_above_prompt'
  WIDGET_BELOW_PROMPT = 'widget_below_prompt'


_WIDGET_REGIONS = frozenset({
  RenderRegion.WIDGET_BELOW_ACTIVITY,
  RenderRegion.WIDGET_ABOVE_PROMPT,
  RenderRegion.WIDGET_BELOW_PROMPT,
})


class Extension[StateT, EffectT, ContextT](abc.ABC):
  """Base class for one isolated extension state machine."""

  reset_on_session_open = True
  global_ = True

  @classmethod
  @abc.abstractmethod
  def id(cls) -> str:
    """Return the stable extension ID."""
    ...

  def extensions(self) -> Sequence[Extension[Any, Any, None]]:
    """Register additional extensions; global_=False makes them opt-in per workspace."""
    return ()

  def description(self) -> str:
    """Return a short menu description, or use the source docstring by default."""
    return ''

  def render_region(self) -> RenderRegion | None:
    """Return the region where this extension renders, if any."""
    return None

  def render_order(self) -> int:
    """Return this extension's order within its render region."""
    return 0

  def status_separator(self) -> str:
    """Return a status widget's leading separator; use an empty string to omit."""
    return ' · '

  @staticmethod
  def initial_state() -> StateT | None:
    """Return fresh initial state for this extension."""
    return None

  def require_state(self, global_state: state.State) -> StateT:
    """Return this extension's state, failing when it is unavailable."""
    current = global_state.extensions[self.id()]
    if current is None:
      raise RuntimeError(f'extension state unavailable: {self.id()!r}')
    return cast(StateT, current)

  def default_enabled(self) -> bool | Literal['hidden']:
    """Return default enablement, or 'hidden' to also omit it from discovery."""
    return True

  def default_config(self) -> config.JsonObject:
    """Return default config values, recursively merged with user overrides."""
    return {}

  def configuration(self, global_state: state.State) -> config.JsonObject:
    """Return this invocation's configuration for reducers and renderers."""
    return global_state.extensions.configuration(self.id())

  def current_state(self, context: Context) -> StateT:
    """Return this workspace's current state."""
    return self.require_state(context.get_state())

  def dispatch(
    self, context: Context, action: actions.ExtensionAction
  ) -> object:
    """Dispatch an action routed to this extension in the calling workspace."""
    return context.dispatch(actions.RoutedExtensionAction(self.id(), action))

  async def on_load(self, context: ContextT) -> None:
    """Initialize shared resources once when this extension is loaded."""

  async def on_unload(self, context: ContextT) -> None:
    """Release shared resources after workspace calls have stopped."""

  async def on_config(self, context: Context) -> None:
    """Apply initial or changed configuration without resetting workspace state."""

  async def on_foreground(self, context: Context) -> None:
    """Observe a workspace becoming visible without changing extension lifetime."""

  async def on_background(self, context: Context) -> None:
    """Observe a workspace becoming hidden without cancelling its work."""

  async def on_remote_setup(
    self, context: Context, request: actions.CreateRemote
  ) -> bool:
    """Prepare a remote session before creation or selection.

    Return False to defer the request, for example while showing setup UI.
    After setup succeeds, dispatch the original request to check readiness again.
    """
    return True

  def handle_message(
    self, context: Context, message: agent_pb2.SessionMessage
  ) -> None:
    """Observe one message from the calling workspace's agent session."""
    del message

  def slash_commands(self) -> Sequence[commands.CommandDef]:
    """Return slash commands contributed by this extension."""
    return ()

  def tips(self) -> Sequence[str]:
    """Return short, plain single-line tips without the bar's `Tip:` prefix.

    Only override when tips are explicitly requested. Most extensions should
    use the empty default; tips are intended for broadly shared extensions,
    not personal or one-off widgets.

    Read once on load or reload. Tips appear only while the extension is
    enabled, healthy, and supported in the current workspace.
    """
    return ()

  def prompt_editor(self) -> editing.Editor | None:
    """Request a native prompt editor; None leaves the default unchanged."""
    return None

  def keybindings(self) -> Sequence[keymap.Binding]:
    """Return action bindings above defaults and below user configuration."""
    return ()

  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[
    StateT | None, tuple[effects.CoreEffect | PrivateEffect[EffectT], ...]
  ]:
    """Reduce one application action against global state."""
    del action
    return cast(StateT | None, global_state.extensions[self.id()]), ()

  def render(
    self,
    global_state: state.State,
    extension_state: StateT,
  ) -> containers.AnyContainer | None:
    """Render global and extension state, or return no content."""
    del global_state, extension_state
    return None

  def effect(self, payload: EffectT) -> PrivateEffect[EffectT]:
    """Emit a private effect for this extension's handler."""
    return PrivateEffect(payload)

  def tools(self, context: Context) -> list[host_tools.HostTool]:
    """Return tools bound to the calling workspace."""
    return []

  def mcp_connections(
    self, context: Context
  ) -> Sequence[mcp_configuration.McpConnection]:
    """Return declarative connections owned by the agent runtime."""
    return ()

  def skills(self) -> Sequence[pathlib.Path]:
    """Return SKILL.md paths contributed to applicable local workspaces."""
    return ()

  def tool_namespace(self) -> str:
    """Return the prefix of this extension's model-facing tool names."""
    return self.id()

  def hydrate(
    self,
    snapshot: agent_pb2.ExtensionRecord | None,
    records: Sequence[agent_pb2.AgentMessage],
    global_state: state.State,
  ) -> StateT | None:
    """Preserve live state unless overridden to restore from the bounded log."""
    return cast(StateT | None, global_state.extensions[self.id()])

  async def handle_effect(self, context: Context, effect: EffectT) -> None:
    """Reject effects unless the extension implements an effect handler."""
    raise TypeError(f'{self.id()} does not handle effect: {effect!r}')


def _extension_class(
  module: types.ModuleType,
) -> type[Extension[object, object, object]]:
  classes: list[type[Extension[object, object, object]]] = []
  values = cast(dict[str, object], vars(module))
  for value in values.values():
    if (
      not isinstance(value, type)
      or value is Extension
      or value.__module__ != module.__name__
      or inspect.isabstract(value)
      or not issubclass(value, Extension)
    ):
      continue
    classes.append(cast(type[Extension[object, object, object]], value))
  if len(classes) > 1:
    classes = [candidate for candidate in classes if candidate.global_]
  if len(classes) != 1:
    raise ImportError(
      f'expected one Extension entrypoint in {module.__file__}, '
      f'found {len(classes)}'
    )
  return classes[0]


def _load_skills(paths: Sequence[pathlib.Path]) -> tuple[skills.Skill, ...]:
  loaded: list[skills.Skill] = []
  for path in paths:
    skill = skills.Skill.from_file(path.resolve())
    if skill is None:
      raise ValueError(f'Invalid extension skill: {path}')
    loaded.append(skill)
  return tuple(loaded)


def _load_tips(tips: Sequence[object]) -> tuple[str, ...]:
  """Validate contributed copy before it enters the terminal layout."""
  if isinstance(tips, str):
    raise ValueError('Extension tips must be a sequence, not a single string')
  loaded: list[str] = []
  for tip in tips:
    if not isinstance(tip, str) or not tip.strip() or not tip.isprintable():
      raise ValueError(
        'Extension tips must be non-empty, printable single lines'
      )
    loaded.append(tip)
  return tuple(loaded)


@dataclasses.dataclass
class _LoadedExtension:
  extension: Extension[object, object, object]
  context: object
  commands: tuple[commands.CommandDef, ...]
  bindings: tuple[keymap.Binding, ...]
  skills: tuple[skills.Skill, ...]
  tips: tuple[str, ...]
  directory: pathlib.Path | None = None
  owner: str | None = None
  overrides: config.JsonObject = dataclasses.field(
    default_factory=dict[str, config.JsonValue]
  )
  settings: config.ExtensionSettings = dataclasses.field(
    default_factory=config.ExtensionSettings
  )


class ExtensionLoadError(ValueError):
  """An extension could not be imported or registered."""


class ExtensionRegistry:
  """Own registered extensions and their shared resources."""

  def __init__(self) -> None:
    self._extensions: dict[str, _LoadedExtension] = {}
    self._module_names: dict[str, str] = {}
    self._default_enablement: dict[pathlib.Path, bool | Literal['hidden']] = {}
    self._hosts: weakref.WeakSet[ExtensionHost] = weakref.WeakSet()
    self._namespace = uuid.uuid4().hex
    self._next_generation = 0

  async def register[StateT, EffectT, ContextT](
    self,
    extension: Extension[StateT, EffectT, ContextT],
    context: ContextT,
    *,
    directory: pathlib.Path | None = None,
    overrides: config.JsonObject | None = None,
    owner: str | None = None,
  ) -> bool:
    """Register an enabled extension, returning whether it was registered."""
    overrides = copy.deepcopy(overrides or {})
    settings = config.resolve_extension(overrides, extension.default_config())
    default_enabled = extension.default_enabled()
    if directory is not None:
      self._default_enablement[directory] = default_enabled
    if overrides.get('enabled', default_enabled) is not True:
      return False
    extension_id = extension.id()
    if not extension_id:
      raise ValueError('extension ID must not be empty')
    ancestor = owner
    while ancestor is not None:
      if extension_id == ancestor:
        raise ValueError(
          f'Extension cannot register its ancestor: {extension_id!r}'
        )
      ancestor = self._extensions[ancestor].owner
    try:
      await extension.on_load(context)
      loaded = _LoadedExtension(
        cast(Extension[object, object, object], extension),
        context,
        tuple(extension.slash_commands()),
        tuple(extension.keybindings()),
        _load_skills(extension.skills()),
        _load_tips(extension.tips()),
        directory=directory,
        owner=owner,
        overrides=overrides,
        settings=settings,
      )
    except BaseException:
      await extension.on_unload(context)
      raise
    await self.unload(extension_id)
    self._extensions[extension_id] = loaded
    try:
      for host in tuple(self._hosts):
        await host.loaded(loaded)
      for child in extension.extensions():
        await self.register(
          child,
          None,
          owner=extension_id,
          overrides={'config': settings.config},
        )
    except BaseException:
      await self.unload(extension_id)
      raise
    return True

  def extension_ids(self) -> tuple[str, ...]:
    """Return the IDs of all loaded extensions."""
    return tuple(self._extensions)

  def default_enabled(
    self, directory: pathlib.Path
  ) -> bool | Literal['hidden']:
    """Return defaults from the last imported definition, including disabled ones."""
    return self._default_enablement.get(directory, False)

  def source_directory(self, extension_id: str) -> pathlib.Path | None:
    """Return an extension's source directory, or None for in-memory extensions."""
    try:
      return self._extensions[extension_id].directory
    except KeyError:
      raise ValueError(f'extension is not loaded: {extension_id!r}') from None

  def description(self, directory: pathlib.Path) -> str:
    """Read a loaded extension's description without importing an unloaded one."""
    for loaded in self._extensions.values():
      if loaded.directory == directory:
        try:
          return loaded.extension.description()
        except Exception:
          logging.getLogger(__name__).exception(
            'Could not describe extension %s', directory
          )
    return ''

  def statuses(self, extension_id: str) -> tuple[ExtensionStatus, ...]:
    """Read lifecycle status from each applicable workspace without caching it."""
    statuses: list[ExtensionStatus] = []
    for host in self._hosts:
      status = host.status(extension_id)
      if status is not None:
        statuses.append(status)
    return tuple(
      sorted(
        statuses,
        key=lambda item: (str(item.workspace.cwd), item.workspace.remote),
      )
    )

  @contextlib.contextmanager
  def _import_extension(
    self, directory: pathlib.Path
  ) -> Generator[
    tuple[Extension[object, object, object], str | None], None, None
  ]:
    """Keep a module generation only if importing and registering it succeed."""
    directory = directory.resolve()
    entrypoint = (directory / 'extension.py').resolve()
    if (
      directory.parent
      == pathlib.Path(__file__).with_name('extensions').resolve()
      and entrypoint.parent == directory
    ):
      # Configured actions and built-ins must share their Python class identity.
      module = importlib.import_module(
        f'mirror.cli.extensions.{entrypoint.parent.name}.extension'
      )
      yield _extension_class(module)(), None
      return
    generation = self._next_generation
    self._next_generation += 1
    module_name = f'_mirror_extension_{self._namespace}_{generation}'
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
      raise ImportError(f'cannot load extension: {entrypoint}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
      exec(
        compile(entrypoint.read_bytes(), str(entrypoint), 'exec'),
        module.__dict__,
      )
      instance = _extension_class(module)()
      if instance.id() != directory.name:
        raise ValueError(
          f'Extension ID must match folder name {directory.name!r}; got {instance.id()!r}'
        )
      yield instance, module_name
    except BaseException:
      sys.modules.pop(module_name, None)
      raise

  async def load(
    self, directory: pathlib.Path, *, overrides: config.JsonObject | None = None
  ) -> str | None:
    """Load a definition, replacing an earlier registration with the same ID."""
    try:
      with self._import_extension(directory) as (extension, module_name):
        registered = await self.register(
          extension, None, directory=directory.resolve(), overrides=overrides
        )
        if not registered:
          if module_name is not None:
            sys.modules.pop(module_name, None)
          return None
        extension_id = extension.id()
        if module_name is not None:
          self._module_names[extension_id] = module_name
        return extension_id
    except Exception as error:  # Extension loading is an isolation boundary.
      logging.getLogger(__name__).exception(
        'Could not load extension %s', directory
      )
      raise ExtensionLoadError(
        f'Could not load extension {directory}:\n{type(error).__name__}: {error}'
      ) from error

  async def reload(
    self,
    extension_id: str,
    *,
    directory: pathlib.Path | None = None,
    overrides: config.JsonObject | None = None,
  ) -> str | None:
    """Reload a source, rejecting ID changes before unloading anything."""
    stored = self.source_directory(extension_id)
    previous = self._extensions[extension_id]
    module_name = self._module_names.get(extension_id)
    previous_module = sys.modules.get(module_name) if module_name else None
    directory = stored if directory is None else directory
    if directory is None:
      raise ValueError(f'extension has no source to reload: {extension_id!r}')
    try:
      with self._import_extension(directory) as (extension, module_name):
        if extension.id() != extension_id:
          raise ValueError(
            f'reloaded extension ID must remain {extension_id!r}'
          )
        if overrides is None:
          overrides = self._extensions[extension_id].overrides
        config.resolve_extension(overrides, extension.default_config())
        await self.unload(extension_id)
        try:
          registered = await self.register(
            extension, None, directory=directory.resolve(), overrides=overrides
          )
          family = {extension_id}
          for name, loaded in self._extensions.items():
            if name in family or loaded.owner in family:
              family.add(name)
              for status in self.statuses(name):
                if status.error is not None:
                  raise ExtensionLoadError(f'{name}: {status.error}')
        except BaseException:
          await self.unload(extension_id)
          if previous_module is not None:
            sys.modules[previous_module.__name__] = previous_module
          try:
            await self.register(
              previous.extension,
              previous.context,
              directory=previous.directory,
              overrides=previous.overrides,
              owner=previous.owner,
            )
          except BaseException:
            if previous_module is not None:
              sys.modules.pop(previous_module.__name__, None)
            raise
          if previous_module is not None:
            self._module_names[extension_id] = previous_module.__name__
          raise
        if not registered:
          if module_name is not None:
            sys.modules.pop(module_name, None)
          return None
        if module_name is not None:
          self._module_names[extension_id] = module_name
        return extension_id
    except Exception as error:  # Extension reloading is an isolation boundary.
      logging.getLogger(__name__).exception(
        'Could not reload extension %s', extension_id
      )
      raise ExtensionLoadError(
        f'Could not reload extension {extension_id}:\n{type(error).__name__}: {error}'
      ) from error

  async def configure(
    self, extension_id: str, overrides: config.JsonObject
  ) -> bool:
    """Apply settings, returning whether the extension remains enabled."""
    loaded = self._extensions[extension_id]
    settings = config.resolve_extension(
      overrides, loaded.extension.default_config()
    )
    default_enabled = loaded.extension.default_enabled()
    if loaded.directory is not None:
      self._default_enablement[loaded.directory] = default_enabled
    if overrides.get('enabled', default_enabled) is not True:
      await self.unload(extension_id)
      return False
    previous = loaded.settings
    loaded.overrides = copy.deepcopy(overrides)
    loaded.settings = settings
    for child_id, child in tuple(self._extensions.items()):
      if child.owner == extension_id:
        await self.configure(child_id, {'config': settings.config})
    if not config.same_config(previous.config, settings.config):
      for host in tuple(self._hosts):
        await host.configuration_changed(extension_id)
    return True

  def create_host(
    self,
    context: Context,
    host_tool_registry: host_tools.HostToolRegistry | None = None,
    *,
    extensions: Sequence[str] = (),
  ) -> ExtensionHost:
    """Bind shared extensions to one workspace without constructing copies."""
    host = ExtensionHost(
      context,
      self._extensions,
      self._hosts.discard,
      host_tool_registry,
      extensions,
    )
    self._hosts.add(host)
    return host

  async def refresh_tools(self, extension_id: str) -> None:
    """Refresh one extension's tools in all of its workspaces."""
    for host in tuple(self._hosts):
      await host.refresh_tools(extension_id)

  async def unload(self, extension_id: str) -> None:
    """Detach an extension from every workspace and unload its definition."""
    loaded = self._extensions.pop(extension_id, None)
    if loaded is None:
      return
    try:
      for child_id, child in tuple(self._extensions.items()):
        if child.owner == extension_id:
          await self.unload(child_id)
      for host in tuple(self._hosts):
        await host.unload(extension_id)
      try:
        await loaded.extension.on_unload(loaded.context)
      except (
        Exception
      ):  # An extension cleanup failure must not abort registry shutdown.
        logging.getLogger(__name__).exception(
          'Extension %s cleanup failed', extension_id
        )
    finally:
      module_name = self._module_names.pop(extension_id, None)
      if module_name is not None:
        sys.modules.pop(module_name, None)

  async def close(self) -> None:
    """Unload every extension module owned by this invocation."""
    async with contextlib.AsyncExitStack() as cleanup:
      for extension_id in self._extensions:
        cleanup.push_async_callback(self.unload, extension_id)


class ExtensionHost:
  """Own workspace extension instances, state, and effects."""

  def __init__(
    self,
    context: Context,
    extensions: Mapping[str, _LoadedExtension],
    detach: Callable[[ExtensionHost], None],
    host_tool_registry: host_tools.HostToolRegistry | None,
    selected: Sequence[str] = (),
  ) -> None:
    self.host_tool_registry = (
      host_tool_registry
      if host_tool_registry is not None
      else host_tools.HostToolRegistry()
    )
    self._context = context
    self._extensions = extensions
    self._selected = tuple(selected)
    self._detach = detach
    self._started = False
    self._foreground = False
    self._closed = False
    self._boundaries: dict[str, extension_faults.Boundary] = {}
    self._mcp_connections: dict[
      str, tuple[mcp_configuration.McpConnection, ...]
    ] = {}

  def validate_selection(self) -> None:
    """Prevent worker startup when an explicitly selected extension is unavailable."""
    for name in self._selected:
      if name not in self._items:
        raise ValueError(f'Required extension is unavailable: {name!r}')

  def status(self, extension_id: str) -> ExtensionStatus | None:
    """Report lifecycle state for an extension selected in this workspace."""
    loaded = self._extensions.get(extension_id)
    if self._closed or loaded is None:
      return None
    if not loaded.extension.global_ and extension_id not in self._selected:
      return None
    workspace = self._context.workspace
    boundary = self._boundaries.get(extension_id)
    error = boundary.error if boundary is not None else None
    status = 'active' if self._started else 'not started'
    return ExtensionStatus(
      workspace, 'disabled' if error is not None else status, error
    )

  def _boundary(self, name: str) -> extension_faults.Boundary:
    if name not in self._boundaries:
      loaded = self._extensions[name]
      self._boundaries[name] = extension_faults.Boundary(
        functools.partial(self._fail, name),
        lambda: not self._closed and self._extensions.get(name) is loaded,
      )
    return self._boundaries[name]

  def _fail(self, name: str, error: Exception) -> None:
    logging.getLogger(__name__).error(
      'Extension %s disabled', name, exc_info=error
    )
    application.get_app().invalidate()

  @property
  def _items(self) -> dict[str, Extension[object, object, object]]:
    if self._closed:
      return {}
    items = {
      name: loaded.extension
      for name, loaded in self._extensions.items()
      if (loaded.extension.global_ or name in self._selected)
      and self._boundary(name).active()
    }
    return dict(
      sorted(items.items(), key=lambda item: item[0] in self._selected)
    )

  async def start(self) -> None:
    """Start workspace effects without changing shared extension lifetime."""
    if self._started:
      return
    self._started = True
    for name in tuple(self._items):
      await self.loaded(self._extensions[name])

  async def loaded(self, loaded: _LoadedExtension) -> None:
    """Initialize this workspace after an extension is loaded."""
    if loaded.extension.id() not in self._items:
      return
    name = loaded.extension.id()
    boundary = self._boundary(name)
    try:
      if self._started:
        self.host_tool_registry.register(
          name,
          loaded.extension.tools(self._extension_context(name)),
          boundary=boundary,
          namespace=loaded.extension.tool_namespace(),
        )
      self._context.dispatch(actions.ExtensionLoaded(name))
    except Exception as error:  # Isolate extension initialization.
      boundary.fail(error)
    if self._started:
      await self._apply_config(name)
      self._set_mcp_connections(name, loaded.extension)
    if self._foreground:
      await boundary.call_async(
        lambda: loaded.extension.on_foreground(self._extension_context(name)),
        None,
      )

  async def foreground(self) -> None:
    """Notify extensions that this workspace is visible."""
    if self._foreground:
      return
    self._foreground = True
    for name in self._items:
      loaded = self._extensions[name]
      await self._boundary(name).call_async(
        lambda: loaded.extension.on_foreground(self._extension_context(name)),
        None,
      )

  async def prepare_remote(self, request: actions.CreateRemote) -> bool:
    """Require applicable setup hooks to succeed before opening a remote."""
    if self._closed:
      return False
    for name, loaded in tuple(self._extensions.items()):
      if self.status(name) is None:
        continue
      if (
        type(loaded.extension).on_remote_setup
        is Extension[object, object, object].on_remote_setup
      ):
        continue
      boundary = self._boundary(name)
      ready = await boundary.call_async(
        lambda: loaded.extension.on_remote_setup(
          self._extension_context(name), request
        ),
        False,
      )
      if not ready:
        if boundary.error is not None:
          self._context.dispatch(
            actions.ShowText(
              f'Remote setup failed in {name}. Fix and reload the extension.',
              style='ui.error',
            )
          )
        return False
    return True

  async def background(self) -> None:
    """Notify extensions without cancelling background workspace effects."""
    if not self._foreground:
      return
    self._foreground = False
    for name in self._items:
      loaded = self._extensions[name]
      await self._boundary(name).call_async(
        lambda: loaded.extension.on_background(self._extension_context(name)),
        None,
      )

  @property
  def mcp_connections(self) -> tuple[mcp_configuration.McpConnection, ...]:
    """Return configured connections without starting a client-side transport."""
    connections: list[mcp_configuration.McpConnection] = []
    names: set[str] = set()
    for owner in self._items:
      for connection in self._mcp_connections.get(owner, ()):
        if connection.name in names:
          raise ValueError(f'Duplicate MCP connection: {connection.name!r}')
        names.add(connection.name)
        connections.append(connection)
    return tuple(connections)

  async def refresh_tools(self, extension_id: str) -> None:
    """Replace callback tools and runtime-owned connections."""
    extension = self._items.get(extension_id)
    if self._started and extension is not None:
      self._set_mcp_connections(extension_id, extension)
      await self.host_tool_registry.replace(
        extension_id,
        extension.tools(self._extension_context(extension_id)),
        namespace=extension.tool_namespace(),
      )

  def _set_mcp_connections(
    self, extension_id: str, extension: Extension[object, object, object]
  ) -> None:
    connections = tuple(
      extension.mcp_connections(self._extension_context(extension_id))
    )
    if connections != self._mcp_connections.get(extension_id, ()):
      self._mcp_connections[extension_id] = connections
      self.host_tool_registry.context_changed()

  async def unload(self, extension_id: str) -> None:
    """Stop workspace calls before the shared extension releases its resources."""
    await self.cancel(extension_id)
    self._boundaries.pop(extension_id, None)
    if self._mcp_connections.pop(extension_id, ()):
      self.host_tool_registry.context_changed()
    await self.host_tool_registry.unregister(extension_id)
    self._context.dispatch(actions.ExtensionUnloaded(extension_id))

  async def cancel(self, extension_id: str) -> None:
    """Cancel and await this workspace's calls to an extension."""
    boundary = self._boundaries.get(extension_id)
    if boundary is not None:
      await boundary.cancel()

  async def close(self) -> None:
    """Stop workspace tasks without unloading shared extensions."""
    try:
      await self.background()
    finally:
      self._closed = True
      self._detach(self)
      for name in tuple(self._boundaries):
        await self.cancel(name)
      async with contextlib.AsyncExitStack() as cleanup:
        for name in self._extensions:
          cleanup.push_async_callback(self.host_tool_registry.unregister, name)

  @property
  def skills(self) -> tuple[skills.Skill, ...]:
    """Skills from enabled, healthy extensions on this local workspace."""
    if self._context.workspace.remote:
      return ()
    by_name: dict[str, skills.Skill] = {}
    for name in self._items:
      for skill in self._extensions[name].skills:
        by_name[skill.name] = skill
    return tuple(by_name.values())

  @property
  def tips(self) -> tuple[str, ...]:
    """Return tips from enabled, healthy extensions supported in this workspace."""
    contributed: list[str] = []
    for name in self._items:
      contributed.extend(self._extensions[name].tips)
    return tuple(contributed)

  @property
  def slash_commands(self) -> tuple[commands.CommandDef, ...]:
    """Return slash commands contributed by loaded extensions."""
    contributed: list[commands.CommandDef] = []
    for name in self._items:
      loaded = self._extensions[name]
      for command in loaded.commands:
        handler = functools.partial(
          self._command, name, self._boundary(name), command.handler
        )
        contributed.append(
          dataclasses.replace(
            command,
            handler=handler,
            extension_id=name,
          )
        )
    return tuple(contributed)

  def _command(
    self,
    name: str,
    boundary: extension_faults.Boundary,
    handler: commands.CommandHandler,
    context: commands.CommandContext,
    arguments: str,
  ) -> actions.Action | None:
    unavailable = actions.ShowText(
      f'Extension {name} is unavailable. Fix and reload.'
    )
    if not boundary.active():
      return unavailable
    try:
      result = handler(
        dataclasses.replace(
          context, config=copy.deepcopy(self._extensions[name].settings.config)
        ),
        arguments,
      )
      # Count only the handler's normal return, never the failure UI below.
      statsig_client.get().log_event(
        statsig_event.Event(
          'mirror_slash_command_dispatched',
          value=1,
          metadata={'command': 'custom'},
        )
      )
      if isinstance(result, actions.OpenSelector):
        return dataclasses.replace(
          result,
          selector=dataclasses.replace(
            result.selector,
            extension_id=name,
            on_select=functools.partial(
              boundary.call, result.selector.on_select, unavailable
            ),
          ),
        )
      return result
    except Exception as error:  # Includes deferred selector setup.
      boundary.fail(error)
    return actions.ShowText(
      f'Extension {name} disabled: {boundary.error}. Fix and reload.',
      style='ui.error',
    )

  @property
  def prompt_editor(self) -> editing.Editor | None:
    """Resolve the last enabled native editor preference."""
    for name, extension in reversed(tuple(self._items.items())):
      editor = self._boundary(name).call(extension.prompt_editor, None)
      if editor is not None:
        return editor
    return None

  @property
  def keybindings(self) -> tuple[keymap.Binding, ...]:
    """Supported extension bindings in registration order."""
    bindings: list[keymap.Binding] = []
    for name in self._items:
      bindings.extend(self._extensions[name].bindings)
    return tuple(bindings)

  @property
  def initial_states(self) -> extension_state.ExtensionStates:
    """The initial state of every loaded extension."""
    states = extension_state.ExtensionStates()
    for name, extension in self._items.items():
      try:
        states = states.updated(name, extension.initial_state()).configured(
          name, self._extensions[name].settings.config
        )
      except Exception as error:
        self._boundary(name).fail(error)
    return states

  def _extension_context(self, name: str) -> Context:
    return dataclasses.replace(
      self._context,
      config=copy.deepcopy(self._extensions[name].settings.config),
    )

  async def configuration_changed(self, name: str) -> None:
    """Publish the new snapshot before applying asynchronous configuration."""
    if not self._started or name not in self._items:
      return
    context = self._extension_context(name)
    self._context.dispatch(
      actions.ExtensionConfigChanged(name, copy.deepcopy(context.config))
    )
    await self._apply_config(name)

  async def _apply_config(self, name: str) -> None:
    loaded = self._extensions[name]
    if (
      type(loaded.extension).on_config
      is Extension[object, object, object].on_config
    ):
      return
    context = self._extension_context(name)
    await self._boundary(name).call_async(
      lambda: loaded.extension.on_config(context), None
    )

  def handle_message(self, message: agent_pb2.SessionMessage) -> None:
    """Forward an agent-session message to every extension."""
    for name, extension in self._items.items():
      self._boundary(name).call(
        lambda: extension.handle_message(
          self._extension_context(name), message
        ),
        None,
      )

  def _hydrate(
    self,
    trace: Sequence[agent_pb2.AgentMessage],
    global_state: state.State,
  ) -> extension_state.ExtensionStates:
    """Hydrate each extension from its latest bounded snapshot."""
    snapshots: dict[str, agent_pb2.ExtensionRecord] = {}
    records = trace
    for index in range(len(trace) - 1, -1, -1):
      entry = trace[index]
      if entry.WhichOneof('payload') == 'extension':
        snapshots.setdefault(entry.extension.extension, entry.extension)
        continue
      if entry.WhichOneof('payload') != 'context_window':
        continue
      records = trace[index:]
      for recent in reversed(entry.context_window.compaction.recent):
        if recent.WhichOneof('payload') == 'extension':
          snapshots.setdefault(recent.extension.extension, recent.extension)
      break
    states = self.initial_states
    for name, extension in self._items.items():
      try:
        states = states.updated(
          name, extension.hydrate(snapshots.get(name), records, global_state)
        )
      except Exception as error:  # Isolate extension hydration.
        self._boundary(name).fail(error)
    return states

  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[extension_state.ExtensionStates, tuple[effects.Effect, ...]]:
    """Route extension actions and broadcast host actions."""
    if isinstance(action, actions.SessionOpened):
      return extension_state.ExtensionStates(
        {
          extension_id: (
            plugin.initial_state()
            if plugin.reset_on_session_open
            else global_state.extensions[extension_id]
          )
          for extension_id, plugin in self._items.items()
        },
        {
          name: global_state.extensions.configuration(name)
          for name in self._items
        },
      ), ()
    if isinstance(action, actions.HydrateExtensions):
      return self._hydrate(action.records, global_state), ()
    if isinstance(action, actions.ExtensionUnloaded):
      return global_state.extensions.removed(action.extension_id), ()
    if isinstance(action, actions.ExtensionLoaded):
      extension = self._items.get(action.extension_id)
      if extension is None:
        return global_state.extensions, ()
      try:
        global_state = dataclasses.replace(
          global_state,
          extensions=global_state.extensions.updated(
            action.extension_id, extension.initial_state()
          ).configured(
            action.extension_id,
            self._extensions[action.extension_id].settings.config,
          ),
        )
      except Exception as error:
        self._boundary(action.extension_id).fail(error)
    if isinstance(action, actions.ExtensionConfigChanged):
      global_state = dataclasses.replace(
        global_state,
        extensions=global_state.extensions.configured(
          action.extension_id, action.config
        ),
      )
    if isinstance(
      action,
      (
        actions.RoutedExtensionAction,
        actions.ExtensionLoaded,
        actions.ExtensionConfigChanged,
      ),
    ):
      extension = self._items.get(action.extension_id)
      extensions = (
        () if extension is None else ((action.extension_id, extension),)
      )
    else:
      extensions = tuple(self._items.items())
    global_state = dataclasses.replace(global_state, extension_tips=self.tips)
    extension_states = global_state.extensions
    emitted: list[effects.Effect] = []
    for extension_id, extension in extensions:
      if not self._boundary(extension_id).active():
        continue
      try:
        next_state, extension_effects = extension.reduce(global_state, action)
        emitted_effects = [
          effects.RunExtensionEffect(extension_id, effect.payload)
          if isinstance(effect, PrivateEffect)
          else effect
          for effect in extension_effects
        ]
        extension_states = extension_states.updated(extension_id, next_state)
        if self._started:
          emitted.extend(emitted_effects)
      except (
        Exception
      ) as error:  # Isolate extension reducers and effect conversion.
        self._boundary(extension_id).fail(error)
    return extension_states, tuple(emitted)

  def component(self, region: RenderRegion) -> component.Component:
    """Return a component for one extension render region."""
    return _ExtensionComponent(self, region)

  def render(
    self, state: state.State, region: RenderRegion
  ) -> containers.AnyContainer:
    """Render active extensions assigned to one region."""
    state = dataclasses.replace(state, extension_tips=self.tips)
    ordered: list[tuple[int, str, containers.Container]] = []
    if self._started:
      for name, extension in self._items.items():
        boundary = self._boundary(name)
        if not boundary.active():
          continue
        try:
          if (
            extension.render_region() is not region
            or name not in state.extensions
          ):
            continue
          extension_state = state.extensions[name]
          if extension_state is None:
            continue
          order = int(extension.render_order())
          rendered = extension.render(state, extension_state)
          if rendered is not None:
            ordered.append((
              order,
              extension.status_separator(),
              extension_rendering.ExtensionContainer(rendered, boundary),
            ))
        except (
          Exception
        ) as error:  # Deferred painting is guarded by ExtensionContainer.
          boundary.fail(error)
    ordered.sort(key=lambda item: item[0])
    children: list[containers.Container] = [child for _, _, child in ordered]
    if region is RenderRegion.WIDGET_ABOVE_PROMPT:
      for name, boundary in self._boundaries.items():
        if boundary.error is None:
          continue
        children.append(
          containers.Window(
            controls.FormattedTextControl([
              (
                'class:worktree-failed',
                f'Extension {name} disabled: {boundary.error}. Fix and reload.',
              )
            ]),
            wrap_lines=True,
            dont_extend_height=True,
          )
        )
    if not children:
      return (
        containers.Window(height=0)
        if region in _WIDGET_REGIONS
        else containers.Window(width=0)
      )
    if region is RenderRegion.WIDGET_BELOW_PROMPT:
      return containers.HSplit(
        [
          child
          for child in children
          if child.preferred_height(state.layout.columns, 1).preferred
        ],
        padding=1,
      )
    if region in _WIDGET_REGIONS:
      return containers.HSplit(children)
    separated: list[containers.AnyContainer] = []
    for _, separator, child in ordered:
      if separator and (separated or region is RenderRegion.STATUS_LEFT):
        separated.append(
          self._status_separator(separator if separated else separator.lstrip())
        )
      separated.append(child)
    if region is RenderRegion.STATUS_RIGHT:
      separated.append(self._status_separator(' '))
    return containers.VSplit(separated)

  def _status_separator(self, text: str) -> containers.Window:
    return containers.Window(
      controls.FormattedTextControl([('class:footer', text)]),
      style='class:footer',
      dont_extend_height=True,
      dont_extend_width=True,
    )

  def handle_effect(
    self, effect: effects.RunExtensionEffect
  ) -> asyncio.Task[None]:
    """Start an effect owned by this workspace, independent of tab visibility."""
    loaded = self._extensions.get(effect.extension)
    if loaded is None:
      raise ValueError(f'unknown extension: {effect.extension!r}')
    return self._boundary(effect.extension).spawn(
      lambda: loaded.extension.handle_effect(
        self._extension_context(effect.extension), effect.payload
      ),
      None,
    )

  def submit_secret(self, secret: str) -> asyncio.Task[None] | None:
    """Close the masked prompt and hand its input to the requesting extension."""
    request = self._context.get_state().secret.request
    self._context.dispatch(actions.ClearSecret())
    if request is None:
      return None
    name = request.extension_id
    loaded = self._extensions.get(name)
    if loaded is None:
      return None
    return self._boundary(name).spawn(
      lambda: loaded.extension.handle_effect(
        self._extension_context(name), request.on_submit(secret)
      ),
      None,
    )


class _ExtensionComponent(component.Component):
  def __init__(self, host: ExtensionHost, region: RenderRegion) -> None:
    self._host = host
    self._region = region

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    return self._host.render(state, self._region)
