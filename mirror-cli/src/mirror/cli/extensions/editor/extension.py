"""Stream text files under CWD into a searchable external-editor picker."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import pathlib
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from typing import override

import frizbee

from mirror.cli import (
  actions,
  commands,
  effects,
  extension,
  keymap,
  selection,
  state,
  terminal,
)
from mirror.cli.extensions.editor import files


@dataclasses.dataclass(frozen=True)
class OpenEditor(actions.ExtensionAction):
  """Open a streaming file picker for the external editor."""


@dataclasses.dataclass(frozen=True)
class _Scan:
  selector_id: str


@dataclasses.dataclass(frozen=True)
class _Wake:
  selector_id: str


@dataclasses.dataclass(frozen=True)
class _EditFile(actions.ExtensionAction):
  path: str


def _pick_file(
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  del context, arguments
  return actions.RoutedExtensionAction(EditorExtension.id(), OpenEditor())


def _selector_request() -> actions.OpenSelector:
  return actions.OpenSelector(
    selection.Selector(
      f'editor-{uuid.uuid4().hex}',
      'Open file in editor',
      (),
      _edit_file,
      extension_id=EditorExtension.id(),
      empty='No text files available.',
      prefiltered=True,
      loading=True,
    )
  )


def _edit_file(path: str) -> actions.Action:
  return actions.RoutedExtensionAction(EditorExtension.id(), _EditFile(path))


async def _collect(
  workspace: extension.Workspace,
  choices: list[selection.Choice],
  updated: asyncio.Event,
) -> None:
  try:
    async with contextlib.aclosing(files.batches(workspace)) as batches:
      async for batch in batches:
        choices.extend(batch)
        if len(choices) > 100_000:
          raise ValueError('Editor file listing exceeds 100,000 entries')
        updated.set()
  finally:
    updated.set()


async def _edit(context: extension.Context, path: str) -> None:
  temporary: pathlib.Path | None = None
  keep = False
  try:
    if context.run_terminal is None:
      raise RuntimeError('The frontend terminal is unavailable')
    relative = pathlib.PurePosixPath(path)
    if relative.is_absolute() or '..' in relative.parts:
      raise ValueError('Editor path escapes the workspace')
    original = await context.workspace.read(relative)
    temporary = pathlib.Path(tempfile.mkdtemp(prefix='mirror-editor-'))
    local = temporary / relative.name
    local.write_text(original, encoding='utf-8', newline='')
    command = terminal.editor_command(local)
    keep = True
    if not await context.run_terminal(command, str(temporary)):
      return
    with local.open('rb') as source:
      data = source.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
      raise ValueError('Editor file exceeds 2 MiB')
    content = data.decode('utf-8')
    if content != original:
      await context.workspace.write(relative, content, expected=original)
    keep = False
  except (OSError, ValueError, RuntimeError, TimeoutError) as error:
    context.dispatch(
      actions.ShowText(
        f'Could not edit workspace file: {error}', style='ui.error'
      )
    )
  finally:
    if temporary is not None:
      if keep:
        context.dispatch(
          actions.ShowText(
            f'Editor copy retained in {str(temporary)!r}.', style='ui.warning'
          )
        )
      else:
        shutil.rmtree(temporary)


def _matches(
  choices: tuple[selection.Choice, ...], query: str
) -> tuple[selection.Choice, ...]:
  """Rank a changing batch without retaining snapshots in the UI's caches."""
  if not query:
    return choices
  haystacks = frizbee.Haystacks(tuple(choice.label for choice in choices))
  matcher = frizbee.Matcher(
    query, max_typos=0, casing='smart', sort='score_then_index_asc'
  )
  return tuple(choices[match.index] for match in matcher.match_list(haystacks))


class EditorExtension(
  extension.Extension[None, OpenEditor | _Scan | _Wake | _EditFile, None]
):
  """Contribute a streaming workspace file picker and external editor command."""

  def __init__(self) -> None:
    self._scans: dict[str, tuple[asyncio.Event, asyncio.Task[None]]] = {}

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/editor'

  @override
  def tips(self) -> tuple[str, ...]:
    return ('Ctrl+P searches local files and opens one in your editor',)

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'editor',
        'Open a file under CWD in your editor',
        _pick_file,
        available_when_busy=True,
      ),
    )

  @override
  def keybindings(self) -> Sequence[keymap.Binding]:
    return (
      keymap.Binding(
        ('c-p',), actions.RoutedExtensionAction(self.id(), OpenEditor())
      ),
    )

  @override
  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[
    None,
    tuple[
      effects.CoreEffect
      | extension.PrivateEffect[OpenEditor | _Scan | _Wake | _EditFile],
      ...,
    ],
  ]:
    if isinstance(action, actions.RoutedExtensionAction) and isinstance(
      action.action, OpenEditor
    ):
      if (
        global_state.selector.active
        or global_state.resume.active
        or global_state.rewind.active
        or global_state.drop_confirmation is not None
      ):
        return None, ()
      return None, (self.effect(action.action),)
    emitted: list[
      effects.CoreEffect
      | extension.PrivateEffect[OpenEditor | _Scan | _Wake | _EditFile]
    ] = []
    request = global_state.selector.selector
    if (
      request is not None
      and request.extension_id == self.id()
      and isinstance(
        action,
        (
          actions.OpenSelector,
          actions.SelectorSearchChanged,
          actions.ClearSelector,
          actions.CloseSelector,
          actions.SelectorChoiceSelected,
          actions.SessionOpened,
        ),
      )
    ):
      emitted.append(self.effect(_Wake(request.id)))
    if (
      isinstance(action, actions.OpenSelector)
      and action.selector.extension_id == self.id()
    ):
      emitted.append(self.effect(_Scan(action.selector.id)))
    if isinstance(action, actions.RoutedExtensionAction) and isinstance(
      action.action, _EditFile
    ):
      emitted.append(self.effect(action.action))
    return None, tuple(emitted)

  @override
  async def handle_effect(
    self,
    context: extension.Context,
    effect: OpenEditor | _Scan | _Wake | _EditFile,
  ) -> None:
    if isinstance(effect, _EditFile):
      await _edit(context, effect.path)
      return
    if isinstance(effect, OpenEditor):
      context.dispatch(_selector_request())
      return
    if isinstance(effect, _Wake):
      if running := self._scans.get(effect.selector_id):
        updated, collector = running
        if context.get_state().selector.id != effect.selector_id:
          collector.cancel()
        updated.set()
      return
    if context.get_state().selector.id != effect.selector_id:
      return
    updated = asyncio.Event()
    choices: list[selection.Choice] = []
    collector = asyncio.create_task(
      _collect(context.workspace, choices, updated)
    )
    self._scans[effect.selector_id] = updated, collector
    try:
      await self._publish(
        context, effect.selector_id, choices, updated, collector
      )
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
      if context.get_state().selector.id == effect.selector_id:
        context.dispatch(actions.ClearSelector())
        context.dispatch(
          actions.ShowText(
            f'Could not list editor files: {error}', style='ui.error'
          )
        )
    finally:
      self._scans.pop(effect.selector_id, None)
      if not collector.cancelling():
        collector.cancel()
      await asyncio.gather(collector, return_exceptions=True)

  async def _publish(
    self,
    context: extension.Context,
    selector_id: str,
    choices: list[selection.Choice],
    updated: asyncio.Event,
    collector: asyncio.Task[None],
  ) -> None:
    while True:
      await updated.wait()
      # Coalesce file batches and keystrokes instead of ranking every arrival.
      await asyncio.sleep(0.03)
      updated.clear()
      current = context.get_state().selector
      if current.id != selector_id:
        return
      loading = not collector.done()
      if not loading:
        collector.result()
      matches = await asyncio.to_thread(_matches, tuple(choices), current.query)
      context.dispatch(
        actions.SelectorResultsLoaded(
          selector_id, current.query, matches, loading
        )
      )
