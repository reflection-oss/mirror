"""Provider-neutral tools hosted by the Mirror client process."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from mirror.cli import extension_faults

DEFAULT_TOOL_TIMEOUT_SECONDS = 300


class CallOrigin(enum.Enum):
  """Trusted source of a tool invocation."""

  MODEL = 'model'
  EXTENSION = 'extension'


@dataclasses.dataclass(frozen=True)
class ExecutionContext:
  """Client-local execution metadata, separate from tool arguments."""

  origin: CallOrigin = CallOrigin.MODEL


MODEL_CONTEXT = ExecutionContext()


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
  """One model-facing tool exposed by an extension session."""

  name: str
  description: str
  parameters: Mapping[str, object]
  needs_approval: bool = True
  timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
  global_: bool = True


@dataclasses.dataclass(frozen=True)
class ToolResult:
  """The normalized result of one extension tool call."""

  content: str
  success: bool
  display: str | None = None


class HostTool(Protocol):
  """One model-facing definition and its argument handler."""

  @property
  def definition(self) -> ToolDefinition: ...

  async def execute(
    self, arguments: str, context: ExecutionContext = MODEL_CONTEXT
  ) -> ToolResult: ...


type _BoundTool = tuple[
  extension_faults.Boundary | None, HostTool, ToolDefinition
]


def tool_name(extension_id: str, name: str) -> str:
  """Return the provider-safe, extension-qualified name used in calls and allowlists."""
  return ''.join(
    character
    if character.isascii() and (character.isalnum() or character in '_-')
    else '_'
    for character in f'{extension_id.removeprefix("core/")}__{name}'
  )


class HostToolRegistry:
  """One workspace's host-tool catalog and name-based call routing."""

  def __init__(
    self,
    *,
    selected: Sequence[str] | None = None,
  ) -> None:
    self._selected = frozenset(selected) if selected is not None else None
    self._restart_worker: Callable[[], Awaitable[bool]] | None = None
    self._restart_task: asyncio.Task[None] | None = None
    self._worker_revision = 0
    self._idle = asyncio.Event()
    self._idle.set()
    self._owners: dict[
      str,
      tuple[
        extension_faults.Boundary | None,
        dict[str, tuple[HostTool, ToolDefinition]],
      ],
    ] = {}
    self._revision = 0

  def register(
    self,
    owner: str,
    tools: Sequence[HostTool],
    *,
    boundary: extension_faults.Boundary | None = None,
    namespace: str | None = None,
  ) -> None:
    """Register namespaced tools, rejecting ambiguous model-facing names."""
    self._set_tools(owner, tools, boundary, namespace)

  def _set_tools(
    self,
    owner: str,
    tools: Sequence[HostTool],
    boundary: extension_faults.Boundary | None,
    namespace: str | None,
  ) -> None:
    registered_tools: dict[str, tuple[HostTool, ToolDefinition]] = {}
    for tool in tools:
      definition = tool.definition
      definition = dataclasses.replace(
        definition, name=tool_name(namespace or owner, definition.name)
      )
      if definition.name in registered_tools:
        raise ValueError(f'Duplicate host tool name: {definition.name!r}')
      registered_tools[definition.name] = (tool, definition)
    for existing_owner, (_, existing_tools) in self._owners.items():
      if existing_owner != owner:
        duplicates = existing_tools.keys() & registered_tools.keys()
        if duplicates:
          raise ValueError(f'Duplicate host tool names: {sorted(duplicates)!r}')
    previous = self.tools()
    self._owners.pop(owner, None)
    self._owners[owner] = (boundary, registered_tools)
    if self.tools() != previous:
      self.context_changed()

  async def replace(
    self,
    owner: str,
    tools: Sequence[HostTool],
    *,
    namespace: str | None = None,
  ) -> None:
    """Validate and replace a registration, then synchronize the worker catalog."""
    self._set_tools(
      owner, tools, self._owners.get(owner, (None, {}))[0], namespace
    )
    await self.wait_for_update()

  async def unregister(self, owner: str) -> None:
    """Remove an owner's tools from subsequent lookups."""
    previous = self.tools()
    self._owners.pop(owner, None)
    if self.tools() != previous:
      self.context_changed()

  def activity_changed(self, busy: bool) -> None:
    """Defer catalog refreshes until the active turn has finished."""
    if busy:
      self._idle.clear()
    else:
      self._idle.set()

  def worker_starting(self) -> None:
    """Capture the catalog revision about to be sent to a worker."""
    self._worker_revision = self._revision

  def worker_started(self, restart: Callable[[], Awaitable[bool]]) -> None:
    """Enable updates, including changes received while the worker was starting."""
    self._restart_worker = restart
    self._schedule_update()

  async def worker_stopped(self) -> None:
    """Disable updates and stop pending restarts when the worker disconnects."""
    self._restart_worker = None
    task = self._restart_task
    if task is not None and task is not asyncio.current_task():
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)

  def context_changed(self) -> None:
    """Refresh the idle worker after client-owned context changes."""
    self._revision += 1
    self._schedule_update()

  def _schedule_update(self) -> None:
    if self._restart_worker is None or self._worker_revision == self._revision:
      return
    if self._restart_task is None or self._restart_task.done():
      self._restart_task = asyncio.create_task(self._update_worker())

  async def _update_worker(self) -> None:
    while (
      self._restart_worker is not None
      and self._worker_revision != self._revision
    ):
      while not self._idle.is_set():
        await self._idle.wait()
      if not await self._restart_worker():
        return

  async def wait_for_update(self) -> None:
    """Wait for a catalog update without waiting on the restart callback itself."""
    task = self._restart_task
    if task is not None and task is not asyncio.current_task():
      await task

  def _tools(self) -> dict[str, _BoundTool]:
    selected_tools: dict[str, _BoundTool] = {}
    for boundary, tools in self._owners.values():
      for name, (tool, definition) in tools.items():
        selected = (
          definition.global_
          if self._selected is None
          else name in self._selected
        )
        if not selected:
          continue
        selected_tools[name] = (boundary, tool, definition)
    return selected_tools

  def validate_selection(self) -> None:
    """Reject unavailable explicitly selected tools before starting a worker."""
    for name in self._selected or ():
      tool = self._tools().get(name)
      if tool is None or (tool[0] is not None and not tool[0].active()):
        raise ValueError(f'Required tool is unavailable: {name!r}')

  def tools(self) -> Sequence[ToolDefinition]:
    """Return definitions in registration order for the next worker launch."""
    return tuple(definition for _, _, definition in self._tools().values())

  def has(self, name: str) -> bool:
    """Whether the name belongs to a selected registered tool."""
    return name in self._tools()

  async def execute(
    self,
    name: str,
    arguments: str,
    context: ExecutionContext = MODEL_CONTEXT,
  ) -> ToolResult:
    """Route a call directly to the registered tool."""
    registered = self._tools().get(name)
    if registered is None:
      return ToolResult(f'Unknown host tool: {name}', False)
    boundary, tool, definition = registered

    try:
      async with asyncio.timeout(definition.timeout_seconds):
        return await self._execute(tool, arguments, boundary, context)
    except TimeoutError:
      return ToolResult(f'Tool timed out: {name}', False)

  async def _execute(
    self,
    tool: HostTool,
    arguments: str,
    boundary: extension_faults.Boundary | None,
    context: ExecutionContext,
  ) -> ToolResult:
    if boundary is not None:
      return await boundary.call_async(
        lambda: tool.execute(arguments, context),
        ToolResult('Extension is unavailable. Fix and reload.', False),
      )
    try:
      return await tool.execute(arguments, context)
    except Exception as error:  # Every host tool is an exception boundary.
      return ToolResult(str(error), False)

  async def close(self) -> None:
    """Remove all registrations and cancel outstanding calls."""
    await self.worker_stopped()
    for owner in tuple(self._owners):
      await self.unregister(owner)
