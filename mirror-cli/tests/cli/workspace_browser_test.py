"""The landing screen browses metadata without opening or starting agents."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import datetime
import io
import pathlib
import queue
import signal
from collections.abc import AsyncGenerator
from typing import override

import pyte
import pytest
from mirror.protocol import daemon_pb2
from prompt_toolkit import data_structures
from prompt_toolkit.application import current
from prompt_toolkit.output import vt100

from mirror.cli import (
  actions,
  cli,
  daemon_client,
  onboarding,
  workspace_application,
  workspace_browser,
  workspace_controller,
  workspaces,
)
from tests import terminal_io


@pytest.fixture(autouse=True)
def isolated_home(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path / 'home'))


class _Output(vt100.Vt100_Output):
  def __init__(self, rows: int = 24, columns: int = 80) -> None:
    self.stream = io.StringIO()
    self.screen = pyte.Screen(columns, rows)
    self.terminal = pyte.Stream(self.screen)
    self.changed = asyncio.Event()
    super().__init__(
      self.stream,
      lambda: data_structures.Size(
        rows=self.screen.lines, columns=self.screen.columns
      ),
      term='xterm-256color',
      enable_cpr=False,
    )

  @override
  def encoding(self) -> str:
    return 'utf-8'

  @property
  def text(self) -> str:
    return '\n'.join(self.screen.display)

  @override
  def flush(self) -> None:
    super().flush()
    self.terminal.feed(self.stream.getvalue())
    self.stream.seek(0)
    self.stream.truncate()
    self.changed.set()

  async def wait_for_text(self, text: str) -> None:
    try:
      async with asyncio.timeout(5):
        while text not in self.text:
          self.changed.clear()
          await self.changed.wait()
    except TimeoutError:
      raise AssertionError(f'Missing {text!r}:\n{self.text}') from None


@contextlib.asynccontextmanager
async def _running_browser(
  controller: workspace_controller.WorkspaceController,
  *,
  rows: int = 24,
  columns: int = 80,
  theme: str = 'dark',
) -> AsyncGenerator[tuple[terminal_io.Input, _Output]]:
  output = _Output(rows, columns)
  with (
    terminal_io.input(output) as pipe,
    current.create_app_session(input=pipe, output=output),
  ):
    screen = workspace_application.WorkspaceApplication(
      full_screen=True, refresh_interval=0.1
    )
    screen.enable_views()
    started = asyncio.Event()
    terminal = asyncio.create_task(screen.run_async(pre_run=started.set))
    await asyncio.wait_for(started.wait(), 5)
    view = asyncio.create_task(
      cli._activate_workspace(  # pyright: ignore[reportPrivateUsage] - exercise the persistent terminal handoff.
        controller,
        controller.selected,
        False,
        screen=screen,
        theme_name=lambda: theme,
      )
    )
    try:
      await output.wait_for_text('Workspaces')
      yield pipe, output
    finally:
      screen.exit_for_handoff('back')
      await asyncio.wait_for(view, 5)
      screen.exit()
      await asyncio.wait_for(terminal, 5)
      screen.finish_views()
      await controller.close()


def _unexpected_agent(
  spec: workspaces.AgentSpec,
  controller: workspace_controller.WorkspaceController,
) -> workspaces.Agent:
  del spec, controller
  pytest.fail('Browsing must not construct an agent')


@pytest.mark.asyncio
@pytest.mark.parametrize('open_count', [0, 1, 2], ids=['none', 'one', 'many'])
@pytest.mark.parametrize(
  ('keys', 'expected'),
  [
    ('\x04', None),
    ('no-matching-workspace\r\x04', None),
    ('/open\rclosed-work\r', 'resume:closed'),
    ('\x0e\x04', None),
    ('\r', 'create:normal'),
    ('\n\n\n\n\n\n\x0b\r', 'resume:closed'),
    ('jk-no-matching\r\x04', None),
    ('other-work\r', 'resume:other'),
  ],
)
async def test_landing_browses_closed_and_other_folder_workspaces_without_mutation(
  tmp_path: pathlib.Path,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_client: daemon_client.Client,
  open_count: int,
  keys: str,
  expected: str | None,
) -> None:
  (tmp_path / 'other').mkdir()
  onboarding.ensure_consent(tmp_path, confirm=lambda _: True)
  onboarding.ensure_consent(tmp_path / 'other', confirm=lambda _: True)
  workspace_snapshot.workspaces.add(
    id='closed-id',
    session_id='closed',
    name='closed-work',
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_CLOSED,
  )
  workspace_snapshot.workspaces.add(
    id='other-id',
    session_id='other',
    name='other-work',
    directory=str(tmp_path / 'other'),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  workspace_snapshot.workspaces.extend(
    [
      daemon_pb2.Workspace(
        id='first-id',
        session_id='first',
        name='first-work',
        directory=str(tmp_path),
        kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
        state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
      ),
      daemon_pb2.Workspace(
        id='second-id',
        session_id='second',
        name='second-work',
        directory=str(tmp_path),
        kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
        state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
      ),
    ][:open_count]
  )
  before = workspace_snapshot.SerializeToString()
  controller = workspace_controller.WorkspaceController(
    {'model': 'test-model'},
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_unexpected_agent,
  )
  assert isinstance(controller.selected, workspaces.EmptyWorkspace)
  controller.selected.draft = actions.ComposerState('unsent draft', 4)
  output = _Output()
  try:
    with (
      terminal_io.input(output) as pipe,
      current.create_app_session(input=pipe, output=output),
    ):
      running = asyncio.create_task(
        cli._activate_workspace(controller, controller.selected, False)  # pyright: ignore[reportPrivateUsage] - exercise the real empty-view handoff.
      )
      try:
        await output.wait_for_text('Workspaces')
        assert 'Search:' not in output.text
        assert not output.screen.cursor.hidden
        assert (
          output.screen.display[output.screen.cursor.y].strip()
          == '› Search workspaces…'
        )
        await output.wait_for_text('↻  closed-work')
        pipe.send_text(keys)
        assert await asyncio.wait_for(running, 5) == expected
      finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    assert controller.visible_workspaces() == ()
    assert controller.selected.draft == actions.ComposerState('unsent draft', 4)
    assert workspace_snapshot.SerializeToString() == before
  finally:
    await controller.close()


@pytest.fixture
def described_workspaces(
  tmp_path: pathlib.Path,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_client: daemon_client.Client,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> workspace_controller.WorkspaceController:
  for identity, title, folder, status in (
    (
      'closed',
      'Fix reconnect races',
      'alpha',
      daemon_pb2.WORKSPACE_STATE_CLOSED,
    ),
    (
      'live',
      'Audit approval replay',
      'alpha',
      daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
    (
      'other',
      'Trim startup imports',
      'beta',
      daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    ),
  ):
    directory = str(tmp_path / 'projects' / folder)
    record = workspace_snapshot.workspaces.add(
      id=identity,
      session_id=identity,
      name=folder,
      directory=directory,
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=status,
    )
    if identity == 'live':
      record.inference.model = 'test-model'
    summary = conversation_history.conversations.add(
      id=identity,
      title=title,
      directory=directory,
      model='test-model',
    )
    summary.updated_at.FromDatetime(
      datetime.datetime(2026, 10, 23, 17, 0, tzinfo=datetime.UTC)
    )
  return workspace_controller.WorkspaceController(
    {'model': 'test-model'},
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_unexpected_agent,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('rows', 'columns', 'theme'),
  [
    (32, 80, 'dark'),
    (42, 120, 'light'),
    (48, 180, 'mono'),
    (32, 80, 'terminal'),
  ],
)
async def test_landing_uses_summary_rows_section_gaps_and_shared_prompt(
  described_workspaces: workspace_controller.WorkspaceController,
  rows: int,
  columns: int,
  theme: str,
) -> None:
  async with _running_browser(
    described_workspaces, rows=rows, columns=columns, theme=theme
  ) as (_, output):
    await output.wait_for_text('Trim startup imports')
    frame = output.screen.display
    alpha = next(
      index for index, line in enumerate(frame) if '/alpha  2' in line
    )
    beta = next(index for index, line in enumerate(frame) if '/beta  1' in line)
    assert 'Fix reconnect races' in frame[alpha + 1]
    assert '↻' in frame[alpha + 1]
    assert '2026-10-23' in frame[alpha + 1]
    assert 'test-model' not in output.text
    assert '↻  Audit approval replay' in frame[alpha + 2]
    assert 'activity unknown' not in output.text
    assert '▾' in frame[beta]
    header = next(line for line in frame if 'Workspaces' in line)
    assert header.index('Workspaces') == 1
    assert '▴' not in output.text
    assert '▾  Recent' in output.text
    assert '(1/' not in header
    assert output.screen.cursor.y > beta
    assert frame[output.screen.cursor.y].strip() == '› Search workspaces…'
    assert not output.screen.cursor.hidden
    assert 'Search:' not in output.text
    assert 'Ctrl-F' not in output.text
    assert 'Enter to select' not in output.text


@pytest.mark.asyncio
@pytest.mark.parametrize('columns', [40, 79, 180])
async def test_landing_creation_choices_stay_on_single_rows(
  described_workspaces: workspace_controller.WorkspaceController,
  columns: int,
) -> None:
  async with _running_browser(
    described_workspaces, rows=30, columns=columns
  ) as (_, output):
    frame = output.screen.display
    first = next(
      index for index, line in enumerate(frame) if '› +  New' in line
    )
    assert 'New working directory' in frame[first + 1]
    assert 'Worktree' in frame[first + 2]
    assert 'Sandbox' in frame[first + 3]
    assert 'Working directory' not in frame[first]


@pytest.mark.asyncio
async def test_landing_new_shows_full_launch_path_inline_when_it_fits(
  described_workspaces: workspace_controller.WorkspaceController,
) -> None:
  directory = str(described_workspaces.launch_workspace)
  async with _running_browser(
    described_workspaces, columns=len(directory) + 40
  ) as (_, output):
    row = next(line for line in output.screen.display if '› +  New' in line)
    assert directory in row


@pytest.mark.asyncio
@pytest.mark.parametrize('rows', [8, 12, 17])
async def test_landing_keeps_search_visible_in_small_terminal(
  described_workspaces: workspace_controller.WorkspaceController,
  rows: int,
) -> None:
  async with _running_browser(described_workspaces, rows=rows, columns=40) as (
    pipe,
    output,
  ):
    assert not output.screen.cursor.hidden
    assert (
      output.screen.display[output.screen.cursor.y].strip()
      == '› Search workspaces…'
    )
    pipe.send_text('reconnect')
    await output.wait_for_text('Fix reconnect races')
    assert 'Window too small' not in output.text
    assert (
      output.screen.display[output.screen.cursor.y].strip() == '› reconnect'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('keys', ['\r', '\x1b'])
@pytest.mark.parametrize(
  'open_delete',
  ['Fix reconnect\n\x1b[3~', '\n\n\n\n\n\n\n\n\n\x7f'],
  ids=['forward-delete-search', 'backspace-empty-search'],
)
async def test_landing_delete_defaults_to_cancel(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  keys: str,
  open_delete: str,
) -> None:
  before = workspace_snapshot.SerializeToString()
  async with _running_browser(described_workspaces) as (pipe, output):
    await output.wait_for_text('Fix reconnect races')
    pipe.send_text(open_delete)
    await output.wait_for_text('Remove from list?')
    assert workspace_snapshot.SerializeToString() == before
    pipe.send_text(keys)
    await output.wait_for_text('Workspaces')
    assert workspace_snapshot.SerializeToString() == before


@pytest.mark.asyncio
async def test_landing_confirmed_delete_removes_only_highlighted_workspace(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> None:
  history = conversation_history.SerializeToString()
  async with _running_browser(described_workspaces) as (pipe, output):
    await output.wait_for_text('Fix reconnect races')
    pipe.send_text('Fix reconnect\n\x1b[3~')
    await output.wait_for_text('Remove from list?')
    assert len(workspace_snapshot.workspaces) == 3
    pipe.send_text('\n\r')
    await output.wait_for_text('Workspace removed from list.')
    assert [record.id for record in workspace_snapshot.workspaces] == [
      'live',
      'other',
    ]
    assert conversation_history.SerializeToString() == history
    assert described_workspaces.visible_workspaces() == ()


@pytest.mark.asyncio
async def test_landing_deletes_managed_worktree_files_only_when_explicitly_selected(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  workspace_deletions: queue.Queue[
    tuple[daemon_pb2.DeleteWorkspaceRequest, concurrent.futures.Future[None]]
  ],
) -> None:
  directory = pathlib.Path.home() / '.mirror/worktrees/project/fix'
  directory.mkdir(parents=True)
  (directory / '.git').touch()
  record = workspace_snapshot.workspaces[0]
  record.directory = str(directory)
  record.name = directory.name
  record.is_worktree = True
  async with _running_browser(described_workspaces) as (pipe, output):
    await output.wait_for_text('Fix reconnect races')
    pipe.send_text('Fix reconnect races\x1b[3~')
    await output.wait_for_text('Delete worktree and files')
    assert workspace_deletions.empty()
    pipe.send_text('\n\n\r')
    request, completed = await asyncio.to_thread(
      workspace_deletions.get, True, 5
    )
    assert request.workspace_id == 'closed'
    assert request.delete_files
    completed.set_result(None)
    await output.wait_for_text('Workspace and files deleted.')
    assert [record.id for record in workspace_snapshot.workspaces] == [
      'live',
      'other',
    ]


@pytest.mark.asyncio
async def test_landing_small_terminal_keeps_startup_notice_and_search_visible(
  described_workspaces: workspace_controller.WorkspaceController,
) -> None:
  selected = described_workspaces.selected
  assert isinstance(selected, workspaces.EmptyWorkspace)
  selected.notice = 'Could not refresh workspaces.'
  async with _running_browser(described_workspaces, rows=8, columns=40) as (
    _,
    output,
  ):
    await output.wait_for_text('Could not refresh workspaces.')
    assert '› +  New' in output.text
    assert 'Window too small' not in output.text
    assert (
      output.screen.display[output.screen.cursor.y].strip()
      == '› Search workspaces…'
    )


@pytest.mark.asyncio
async def test_landing_resize_preserves_query_selection_and_input(
  described_workspaces: workspace_controller.WorkspaceController,
) -> None:
  async with _running_browser(described_workspaces, rows=44, columns=140) as (
    pipe,
    output,
  ):
    pipe.send_text('reconnect')
    await output.wait_for_text('Fix reconnect races')
    output.screen.resize(lines=8, columns=40)
    output.screen.reset()
    pipe.resize(output.get_size())
    signal.raise_signal(signal.SIGWINCH)
    await output.wait_for_text('Fix reconnect races')
    assert '› ↻  Fix reconnect races' in output.text
    assert (
      output.screen.display[output.screen.cursor.y].strip() == '› reconnect'
    )
    output.screen.resize(lines=30, columns=100)
    output.screen.reset()
    pipe.resize(output.get_size())
    signal.raise_signal(signal.SIGWINCH)
    await output.wait_for_text('Fix reconnect races')
    assert '› ↻  Fix reconnect races' in output.text
    assert (
      output.screen.display[output.screen.cursor.y].strip() == '› reconnect'
    )
    assert not output.screen.cursor.hidden


@pytest.mark.asyncio
async def test_landing_hints_follow_the_selected_action(
  described_workspaces: workspace_controller.WorkspaceController,
) -> None:
  async with _running_browser(described_workspaces) as (pipe, output):
    await output.wait_for_text('Enter create')
    assert 'delete' not in output.text
    pipe.send_text('\n')
    await output.wait_for_text('Enter choose')
    pipe.send_text('Fix reconnect')
    await output.wait_for_text('› Fix reconnect')
    await output.wait_for_text('↻  Fix reconnect races')
    pipe.send_text('\n')
    await output.wait_for_text('Del remove')
    assert 'Enter open' in output.text
    assert 'Esc clear' in output.text
    pipe.send_text('zzzz')
    await output.wait_for_text('No matching choices.')
    assert 'Enter' not in output.text
    assert 'delete' not in output.text
    assert 'Esc clear' in output.text


@pytest.mark.asyncio
async def test_landing_searches_metadata_without_repeating_it_in_rows(
  described_workspaces: workspace_controller.WorkspaceController,
) -> None:
  async with _running_browser(described_workspaces) as (pipe, output):
    pipe.send_text('test-model')
    await output.wait_for_text('› test-model')
    await output.wait_for_text('Fix reconnect races')
    assert '+  New' not in output.text
    assert output.text.count('test-model') == 1  # The query, not every row.


@pytest.mark.asyncio
async def test_landing_worktree_heading_is_readable_and_full_path_is_searchable(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  identity = '5ef7c010-2ccb-4ad0-8892-3f88ed038768'
  directory = pathlib.Path.home() / '.mirror/worktrees/demo-b75b519e' / identity
  workspace_snapshot.workspaces.add(
    id='worktree',
    session_id='worktree',
    name='Verify isolated checkout',
    directory=str(directory),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
  )
  async with _running_browser(described_workspaces) as (pipe, output):
    pipe.send_text(str(directory))
    await output.wait_for_text('Verify isolated checkout')
    assert f'~/.mirror/worktrees/demo-b75b519e/{identity}' in output.text
    assert '+  New' not in output.text


@pytest.mark.asyncio
@pytest.mark.parametrize('rows', [8, 12])
async def test_landing_small_confirmation_keeps_target_and_cancel_visible(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  rows: int,
) -> None:
  before = workspace_snapshot.SerializeToString()
  async with _running_browser(described_workspaces, rows=rows, columns=40) as (
    pipe,
    output,
  ):
    pipe.send_text('Fix reconnect')
    await output.wait_for_text('› Fix reconnect')
    await output.wait_for_text('↻  Fix reconnect races')
    pipe.send_text('\n\x1b[3~')
    await output.wait_for_text('Remove from list?')
    assert 'Fix reconnect races' in output.text
    assert 'Cancel' in output.text
    assert 'Window too small' not in output.text
    pipe.send_text('\r')
    await output.wait_for_text('Workspaces')
    assert workspace_snapshot.SerializeToString() == before


@pytest.mark.asyncio
async def test_browser_first_load_does_not_show_a_placeholder_order(
  described_workspaces: workspace_controller.WorkspaceController,
  conversation_history: daemon_pb2.ListConversationsResponse,
  conversation_responses: queue.Queue[
    concurrent.futures.Future[daemon_pb2.ListConversationsResponse]
  ],
) -> None:
  response: concurrent.futures.Future[daemon_pb2.ListConversationsResponse] = (
    concurrent.futures.Future()
  )
  conversation_responses.put(response)
  try:
    async with _running_browser(described_workspaces, rows=32) as (_, output):
      await output.wait_for_text('Loading conversations…')
      assert 'Conversation closed' not in output.text
      assert '/alpha  2' not in output.text
      assert '+  New' in output.text
      response.set_result(conversation_history)
      await output.wait_for_text('Fix reconnect races')
      catalog = output.text[output.text.index('/alpha  2') :]
      assert catalog.index('Fix reconnect races') < catalog.index(
        'Audit approval replay'
      )
  finally:
    if not response.done():
      response.set_result(conversation_history)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('kind', 'is_worktree', 'label'),
  [
    (daemon_pb2.WORKSPACE_KIND_DIRECTORY, False, 'folder'),
    (daemon_pb2.WORKSPACE_KIND_DIRECTORY, True, 'worktree'),
    (daemon_pb2.WORKSPACE_KIND_SANDBOX, False, 'sandbox'),
    (daemon_pb2.WORKSPACE_KIND_REMOTE, False, 'remote'),
  ],
)
async def test_browser_shows_catalog_type_without_needing_the_directory(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  kind: int,
  is_worktree: bool,
  label: str,
) -> None:
  workspace_snapshot.workspaces[0].kind = kind
  workspace_snapshot.workspaces[0].is_worktree = is_worktree
  async with _running_browser(described_workspaces, rows=32, columns=100) as (
    _,
    output,
  ):
    await output.wait_for_text(f'Fix reconnect races    {label}')
    row = next(
      line for line in output.screen.display if 'Fix reconnect races' in line
    )
    other = next(
      line for line in output.screen.display if 'Audit approval replay' in line
    )
    assert label in row
    assert row.index(label) == other.index('folder')
    assert row.index('2026-10-23') == other.index('2026-10-23')


@pytest.mark.asyncio
async def test_recent_excludes_history_without_a_workspace(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> None:
  removed = conversation_history.conversations.add(
    id='removed', title='Removed conversation', directory='/removed'
  )
  removed.updated_at.seconds = 1_800_000_000
  before = workspace_snapshot.SerializeToString()
  summaries = await described_workspaces.conversation_summaries()
  details = {info.id: info for info in summaries}
  choices = workspace_browser.recent_choices(
    details, workspace_browser.workspace_choices(described_workspaces, details)
  )
  assert [choice.value for choice in choices] == [
    'recent:other',
    'recent:live',
    'recent:closed',
  ]
  assert '/projects/beta' in choices[0].detail
  assert choices[0].section == 'Recent'
  assert workspace_snapshot.SerializeToString() == before
  await described_workspaces.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(('rows', 'columns'), [(30, 120), (8, 40)])
async def test_inline_recent_deletion_requires_confirmation(
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  rows: int,
  columns: int,
) -> None:
  before = workspace_snapshot.SerializeToString()
  async with _running_browser(
    described_workspaces, rows=rows, columns=columns
  ) as (pipe, output):
    pipe.send_text('Fix reconnect races')
    await output.wait_for_text('› Fix reconnect races')
    await output.wait_for_text('› ↻  Fix reconnect races')
    assert 'New working directory' not in output.text
    pipe.send_text('\x1b[3~')
    await output.wait_for_text('Remove from list?')
    assert workspace_snapshot.SerializeToString() == before
    pipe.send_text('\r')
    await output.wait_for_text('› Fix reconnect races')
    assert workspace_snapshot.SerializeToString() == before


@pytest.mark.asyncio
async def test_recent_requires_explicit_directory_trust(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> None:
  workspace_snapshot.workspaces.add(
    id='directory',
    session_id='current',
    previous_session_ids=['previous'],
    directory=str(tmp_path),
    kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
    state=daemon_pb2.WORKSPACE_STATE_CLOSED,
  )
  conversation_history.conversations.add(
    id='current', title='Current2 conversation', directory=str(tmp_path)
  )
  controller = workspace_controller.WorkspaceController(
    {'model': 'test'},
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_unexpected_agent,
  )
  before = workspace_snapshot.SerializeToString()
  output = _Output()
  try:
    with (
      terminal_io.input(output) as pipe,
      current.create_app_session(input=pipe, output=output),
    ):
      running = asyncio.create_task(workspace_browser.activate(controller))
      try:
        await output.wait_for_text('Workspaces')
        pipe.send_text('Current2')
        await output.wait_for_text('Current2 conversation')
        assert workspace_snapshot.SerializeToString() == before
        assert not running.done()
        pipe.send_text('\r')
        await output.wait_for_text('2. Trust and open')
        assert not onboarding.is_trusted_directory(tmp_path)
        pipe.send_text('2')
        assert await asyncio.wait_for(running, 5) == 'resume:current'
        assert onboarding.is_trusted_directory(tmp_path)
      finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    assert workspace_snapshot.SerializeToString() == before
  finally:
    await controller.close()


@pytest.mark.asyncio
async def test_recent_empty_state(
  described_workspaces: workspace_controller.WorkspaceController,
  conversation_history: daemon_pb2.ListConversationsResponse,
) -> None:
  conversation_history.Clear()
  async with _running_browser(described_workspaces) as (_, output):
    await output.wait_for_text('/alpha  2')
    assert 'Recent' not in output.text


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', ['\r', '1'])
async def test_recent_untrusted_directory_defaults_to_cancel(
  tmp_path: pathlib.Path,
  cancel: str,
  described_workspaces: workspace_controller.WorkspaceController,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  (tmp_path / 'projects' / 'alpha').mkdir(parents=True)
  before = workspace_snapshot.SerializeToString()
  async with _running_browser(described_workspaces) as (pipe, output):
    pipe.send_text('Fix reconnect')
    await output.wait_for_text('Fix reconnect races')
    pipe.send_text('\r')
    await output.wait_for_text('Trust directory?')
    pipe.send_text(cancel)
    await output.wait_for_text('› Fix reconnect')
    assert workspace_snapshot.SerializeToString() == before


@pytest.fixture
def many_workspaces(
  tmp_path: pathlib.Path,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  workspace_snapshot.workspaces.extend(
    daemon_pb2.Workspace(
      id=f'job-k-{index:02}',
      session_id=f'job-k-{index:02}',
      name=f'job-k-{index:02}',
      directory=str(tmp_path),
      kind=daemon_pb2.WORKSPACE_KIND_DIRECTORY,
      state=daemon_pb2.WORKSPACE_STATE_AVAILABLE,
    )
    for index in range(40)
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('many_workspaces')
async def test_landing_scrolls_bounded_rows_and_typing_searches(
  tmp_path: pathlib.Path,
  workspace_client: daemon_client.Client,
  workspace_snapshot: daemon_pb2.WatchWorkspacesResponse,
) -> None:
  onboarding.ensure_consent(tmp_path, confirm=lambda _: True)
  before = workspace_snapshot.SerializeToString()
  controller = workspace_controller.WorkspaceController(
    {'model': 'test-model'},
    workspace=tmp_path,
    store=workspace_client,
    reasoning_effort='medium',
    create_agent=_unexpected_agent,
  )
  output = _Output()
  try:
    with (
      terminal_io.input(output) as pipe,
      current.create_app_session(input=pipe, output=output),
    ):
      running = asyncio.create_task(
        cli._activate_workspace(controller, controller.selected, False)  # pyright: ignore[reportPrivateUsage] - exercise the real landing viewport.
      )
      try:
        await output.wait_for_text('job-k-00')
        assert '▾' in output.text
        assert '▴' not in output.text
        assert 'job-k-00' in output.text
        assert 'job-k-39' not in output.text
        pipe.send_text('\x0b\x0b')
        await output.wait_for_text('job-k-39')
        assert '▴' in output.text
        assert '▾' not in output.text
        assert 'job-k-00' not in output.text
        pipe.send_text('job-k-39\r')
        assert await asyncio.wait_for(running, 5) == 'resume:job-k-39'
      finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    assert controller.visible_workspaces() == ()
    assert workspace_snapshot.SerializeToString() == before
  finally:
    await controller.close()
