"""Git and Jujutsu status for the Mirror prompt."""

from __future__ import annotations

import asyncio
import dataclasses
import re
import shlex
from collections.abc import Callable, Sequence
from typing import cast, override

from prompt_toolkit.layout import containers, controls

from mirror.cli import actions, effects, extension, shell, state
from mirror.cli.components import layout

_REFRESH_DEBOUNCE = 0.2
_GIT_REFRESH_INTERVAL = 10
_JJ_REFRESH_INTERVAL = 300
_MUTATING_TOOLS = frozenset({'bash', 'edit', 'subagent', 'task', 'write'})


@dataclasses.dataclass(frozen=True)
class RepositoryStatus:
  """The current revision and working-copy change summary."""

  system: str
  revision: str
  additions: int
  deletions: int
  changes: int
  bookmarks: tuple[str, ...] = ()
  cwd: str = ''


@dataclasses.dataclass(frozen=True)
class _Loading:
  cwd: str = ''


@dataclasses.dataclass(frozen=True)
class _NotRepository:
  cwd: str = ''


@dataclasses.dataclass(frozen=True)
class _Unavailable:
  cwd: str = ''


_LOADING = _Loading()
_NOT_REPOSITORY = _NotRepository()
type _RepositoryStatusState = (
  RepositoryStatus | _Loading | _NotRepository | _Unavailable
)


def _line_counts(stat: str) -> tuple[int, int]:
  summary = stat.splitlines()[-1] if stat else ''
  additions = re.search(r'([\d,]+) insertion', summary)
  deletions = re.search(r'([\d,]+) deletion', summary)
  return (
    int(additions.group(1).replace(',', '')) if additions is not None else 0,
    int(deletions.group(1).replace(',', '')) if deletions is not None else 0,
  )


@dataclasses.dataclass(frozen=True)
class StatusChanged(actions.ExtensionAction):
  """A newly observed repository status."""

  value: _RepositoryStatusState | None


@dataclasses.dataclass(frozen=True)
class _Refresh:
  pass


_REFRESH = _Refresh()


@dataclasses.dataclass(frozen=True)
class _Watch:
  pass


async def _repository_system(workspace: extension.Workspace) -> str:
  result = await workspace.run("""
directory="$PWD"
while :; do
  if [ -d "$directory/.jj" ]; then printf jj; exit; fi
  [ -e "$directory/.git" ] && break
  [ "$directory" = / ] && break
  directory=$(dirname -- "$directory")
done
if git rev-parse --show-toplevel >/dev/null 2>&1; then printf git; fi
""")
  if result.returncode:
    raise ValueError('Could not identify the workspace repository')
  return result.stdout.decode().strip()


async def _command(
  workspace: extension.Workspace, arguments: Sequence[str]
) -> str:
  result = await workspace.run(shlex.join(arguments))
  if result.returncode:
    raise ValueError(f'{arguments[0]} exited with status {result.returncode}')
  return result.stdout.decode().strip()


async def _repository_status(
  workspace: extension.Workspace, system: str
) -> _RepositoryStatusState:
  if system == 'jj':
    try:
      return await _jj_status(workspace)
    except ValueError:
      try:
        return await _jj_status(workspace, ignore_working_copy=True)
      except ValueError:
        return _LOADING
  if system != 'git':
    return _NOT_REPOSITORY
  try:
    revision = await _command(workspace, ('git', 'branch', '--show-current'))
    if not revision:
      revision = await _command(
        workspace, ('git', 'rev-parse', '--short', 'HEAD')
      )
    entries = await _command(
      workspace, ('git', 'status', '--short', '--untracked-files=no')
    )
    try:
      base = await _command(workspace, ('git', 'rev-parse', '--verify', 'HEAD'))
    except ValueError:
      base = await _command(
        workspace, ('git', 'hash-object', '-t', 'tree', '/dev/null')
      )
    additions, deletions = _line_counts(
      await _command(workspace, ('git', 'diff', '--stat', '--no-color', base))
    )
    return RepositoryStatus(
      'git', revision, additions, deletions, len(entries.splitlines())
    )
  except ValueError:
    return _LOADING


async def _jj_status(
  workspace: extension.Workspace, *, ignore_working_copy: bool = False
) -> RepositoryStatus:
  options = ('--ignore-working-copy',) if ignore_working_copy else ()
  output = await _command(
    workspace,
    (
      'jj',
      *options,
      'log',
      '--no-graph',
      '--revisions',
      '@',
      '--template',
      'change_id.shortest(8) ++ "\\0" ++ '
      'local_bookmarks.map(|bookmark| bookmark.name()).join("\\0") ++ '
      '"\\0" ++ self.diff().files().len() ++ '
      '"\\0" ++ self.diff().stat().total_added() ++ '
      '"\\0" ++ self.diff().stat().total_removed()',
    ),
  )
  revision, *bookmarks, changes, additions, deletions = output.split('\0')
  return RepositoryStatus(
    'jj',
    revision,
    int(additions),
    int(deletions),
    int(changes),
    tuple(filter(None, bookmarks)),
  )


class RepositoryStatusExtension(
  extension.Extension[_RepositoryStatusState, _Refresh | _Watch, None]
):
  """Poll and render the active workspace's repository status."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/status'

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_BELOW_PROMPT

  def __init__(self) -> None:
    # Context copies retain their dispatch callback; separate hosts get distinct
    # callbacks even when their workspace locations and IDs are identical.
    self._refresh_requested: dict[
      Callable[[actions.Action], object], asyncio.Event
    ] = {}
    self._visible: set[Callable[[actions.Action], object]] = set()

  @staticmethod
  @override
  def initial_state() -> _RepositoryStatusState:
    return _Loading()

  @override
  def reduce(
    self,
    global_state: state.State,
    action: actions.Action,
  ) -> tuple[
    _RepositoryStatusState | None,
    tuple[effects.CoreEffect | extension.PrivateEffect[_Refresh | _Watch], ...],
  ]:
    if isinstance(action, actions.RoutedExtensionAction) and isinstance(
      action.action, StatusChanged
    ):
      return action.action.value, ()
    current = cast(
      _RepositoryStatusState | None, global_state.extensions[self.id()]
    )
    if (
      isinstance(action, actions.ExtensionLoaded)
      and action.extension_id == self.id()
    ):
      return current, (self.effect(_Watch()),)
    if isinstance(action, actions.WorkerStarted):
      return _Loading(), (self.effect(_REFRESH),)
    if isinstance(
      action,
      (
        actions.WorkerReconnecting,
        actions.WorkerStopped,
        actions.WorkerClosed,
        actions.WorkerFailed,
      ),
    ):
      return _Unavailable(), ()
    if isinstance(action, actions.FinishRunningToolCall) and any(
      call.call_id == action.call_id and call.name in _MUTATING_TOOLS
      for call in global_state.stream.running
    ):
      return current, (self.effect(_REFRESH),)
    return current, ()

  @override
  async def on_foreground(self, context: extension.Context) -> None:
    self._visible.add(context.dispatch)
    self._refresh_requested.setdefault(context.dispatch, asyncio.Event()).set()

  @override
  async def on_background(self, context: extension.Context) -> None:
    self._visible.discard(context.dispatch)

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _Refresh | _Watch
  ) -> None:
    """Run one poller per host, including tabs that share a workspace."""
    host = context.dispatch
    if isinstance(effect, _Refresh):
      if host in self._visible and (
        requested := self._refresh_requested.get(host)
      ):
        requested.set()
      return
    requested = self._refresh_requested.setdefault(host, asyncio.Event())
    try:
      await self._watch(context, requested)
    finally:
      self._refresh_requested.pop(host, None)
      self._visible.discard(host)

  async def _watch(
    self, context: extension.Context, requested: asyncio.Event
  ) -> None:
    """Read only while visible, waking immediately when a tab is shown."""
    visible = self._visible
    interval = _GIT_REFRESH_INTERVAL
    while True:
      try:
        await asyncio.wait_for(
          requested.wait(),
          timeout=interval if context.dispatch in visible else None,
        )
      except TimeoutError:
        pass
      else:
        await asyncio.sleep(_REFRESH_DEBOUNCE)
      requested.clear()
      if context.dispatch not in visible:
        continue
      current = context.get_state().extensions[self.id()]
      try:
        cwd = await _command(context.workspace, ('pwd', '-P'))
        system = await _repository_system(context.workspace)
        if context.dispatch not in visible:
          continue
        interval = (
          _JJ_REFRESH_INTERVAL if system == 'jj' else _GIT_REFRESH_INTERVAL
        )
        observed = dataclasses.replace(
          await _repository_status(context.workspace, system), cwd=cwd
        )
      except shell.ProcessError:
        interval = _GIT_REFRESH_INTERVAL
        observed = _Unavailable()
      except (TimeoutError, OSError, ValueError):
        # A recoverable command failure must not disable the extension.
        continue
      # Attachment changes invalidate a poll even if it finishes after recovery.
      if (
        context.dispatch not in visible
        or context.get_state().extensions[self.id()] is not current
      ):
        continue
      if (
        isinstance(observed, _Loading)
        and isinstance(current, RepositoryStatus)
        and current.cwd == observed.cwd
      ):
        continue
      if observed != current:
        self.dispatch(context, StatusChanged(observed))

  @override
  def cache_key(
    self, global_state: state.State, extension_state: _RepositoryStatusState
  ) -> object:
    return global_state.workspace, global_state.session.remote, extension_state

  @override
  def render(
    self,
    global_state: state.State,
    extension_state: _RepositoryStatusState,
  ) -> containers.AnyContainer:
    """Show the execution directory, never a remote tab's frontend path."""
    cwd = extension_state.cwd or (
      'remote' if global_state.session.remote else global_state.workspace
    )
    columns = [
      containers.Window(
        controls.FormattedTextControl([('class:footer', f'   {cwd} ')]),
        height=1,
        dont_extend_width=True,
      )
    ]
    if isinstance(extension_state, RepositoryStatus):
      changes = 'change' if extension_state.changes == 1 else 'changes'
      bookmarks = (
        f' {", ".join(extension_state.bookmarks)}'
        if extension_state.bookmarks
        else ''
      )
      system = (
        '' if extension_state.system == 'jj' else f'{extension_state.system} '
      )
      columns.append(
        containers.Window(
          controls.FormattedTextControl([
            (
              'class:footer',
              f' {system}{extension_state.revision}{bookmarks} ',
            )
          ]),
          height=1,
          dont_extend_width=True,
        )
      )
      details = controls.FormattedTextControl([
        ('class:footer', f'{extension_state.changes} {changes}  '),
        ('class:ui.success', f'+{extension_state.additions}'),
        ('class:footer', '/'),
        ('class:ui.error', f'-{extension_state.deletions} '),
      ])
    else:
      if isinstance(extension_state, _Unavailable):
        status = 'repository unavailable'
      elif isinstance(extension_state, _Loading):
        status = 'loading repository…'
      else:
        status = 'not a Git/Jujutsu repository'
      details = controls.FormattedTextControl([('class:footer', f'{status} ')])
    columns.extend([
      containers.Window(),
      containers.Window(
        details,
        height=1,
        align=containers.WindowAlign.RIGHT,
        dont_extend_width=True,
      ),
    ])
    return layout.Columns(columns, retain=True)
