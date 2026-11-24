"""The llama.cpp screens, footer and extension against a synthetic service."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import pathlib
import stat
from collections.abc import AsyncGenerator, Callable
from typing import cast

import pytest
from mirror.protocol import (
  local_models_pb2,
)

from mirror.cli import (
  actions,
  daemon_client,
  extension,
  frontend,
  local_models_client,
  models,
  reducer,
  selection,
  state,
  workspace_controller,
)
from mirror.cli.extensions.inference_controls import (
  extension as inference_controls,
)
from mirror.cli.extensions.inference_controls.llama_cpp import (
  downloads,
  footer,
  host,
  menus,
  screens,
)
from tests.cli import local_models_service

_MODEL = 'owner/model-GGUF:Q4_K_M'
_SERVER = 'http://127.0.0.1:9000/v1'
_DOWNLOADING = local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_DOWNLOADING
_READY = local_models_pb2.LOCAL_MODEL_SERVER_PHASE_READY
_STOPPED = local_models_pb2.LOCAL_MODEL_SERVER_PHASE_STOPPED
_SERVING = local_models_pb2.LOCAL_MODEL_SERVER_PHASE_SERVING
_MANAGED_OPTION = models.ModelOption(
  _MODEL, _MODEL, api='llama_cpp', base_url=models.MANAGED
)


def _entry(
  name: str, status: str, context: int = 0, fit_context: int = 0
) -> local_models_pb2.LocalModelEntry:
  return local_models_pb2.LocalModelEntry(
    name=name,
    status=status,
    source='cache',
    context=context,
    fit_context=fit_context,
  )


def _transfer(
  name: str,
  phase: local_models_pb2.LocalModelTransferPhase.ValueType,
) -> local_models_pb2.LocalModelTransfer:
  return local_models_pb2.LocalModelTransfer(
    name=name, phase=phase, done_bytes=1, total_bytes=4, server=models.MANAGED
  )


async def _first(screen: screens.Screen, query: str = '') -> screens.Update:
  assert screen.stream is not None
  async with asyncio.timeout(5):
    async with contextlib.aclosing(screen.stream(query)) as updates:
      return await anext(updates)


def _values(update: screens.Update) -> list[str]:
  return [choice.value for choice in update.choices]


def _step(action: actions.Action) -> screens.Step:
  assert isinstance(action, actions.RoutedExtensionAction)
  assert isinstance(action.action, screens.Step)
  return action.action


async def _publish(
  runtime: local_models_client.Client,
  service: local_models_service.LocalService,
) -> None:
  """Publish the changes made to the daemon's state; wait until seen."""
  seen = runtime.state.revision
  service.publish()
  async with asyncio.timeout(5):
    await runtime.changed(seen)


async def _running(
  runtime: local_models_client.Client,
  service: local_models_service.LocalService,
) -> None:
  service.state.phase = _READY
  await _publish(runtime, service)


@pytest.mark.asyncio
async def test_models_list_replaces_an_inventory_row_with_its_download(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.models = [
    _entry(_MODEL, 'downloading'),
    _entry('owner/other-GGUF:IQ2_M', 'incomplete'),
  ]
  local_service.state.transfers.append(_transfer(_MODEL, _DOWNLOADING))
  await _running(runtime, local_service)

  update = await _first(menus.model_list(runtime))

  assert _values(update) == [
    f'pause:{_MODEL}',
    'resume:owner/other-GGUF:IQ2_M',
    'download',
    'servers',
    'settings',
  ]
  assert 'Downloading 25%' in update.choices[0].detail


@pytest.mark.asyncio
async def test_a_download_replaces_rows_only_in_its_own_servers_list(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.models = [_entry(_MODEL, 'unloaded')]
  await _running(runtime, local_service)
  await runtime.execute(
    local_models_pb2.ExecuteRequest(
      base_url=_SERVER,
      download=local_models_pb2.LocalModelDownload(name=_MODEL),
    )
  )

  managed = await _first(menus.model_list(runtime))
  external = await _first(menus.model_list(runtime, _SERVER))

  assert _values(managed)[0] == f'load:{_MODEL}'
  assert _values(external)[0] == f'pause:{_MODEL}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('entry', 'detail'),
  [
    (_entry(_MODEL, 'loaded', context=262144), '● In memory · 256K'),
    (
      _entry(_MODEL, 'unloaded', context=8192, fit_context=65536),
      'Downloaded · up to 64K',
    ),
  ],
)
async def test_models_list_describes_each_model_by_its_status(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
  entry: local_models_pb2.LocalModelEntry,
  detail: str,
) -> None:
  local_service.models = [entry]
  await _running(runtime, local_service)

  update = await _first(menus.model_list(runtime))

  assert update.choices[0].detail == detail


@pytest.mark.asyncio
async def test_models_list_shows_the_active_model_loading(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.models = [_entry(_MODEL, 'unloaded')]
  local_service.state.active_model = _MODEL
  local_service.state.phase = local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING
  await _publish(runtime, local_service)

  update = await _first(menus.model_list(runtime))

  assert update.choices[0].detail == '◐ Loading…'


@pytest.mark.asyncio
async def test_models_list_rereads_the_inventory_once_the_model_is_served(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.models = [_entry(_MODEL, 'loading')]
  local_service.state.active_model = _MODEL
  local_service.state.phase = local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING
  await _publish(runtime, local_service)
  screen = menus.model_list(runtime)
  assert screen.stream is not None

  async with asyncio.timeout(5), contextlib.aclosing(screen.stream('')) as rows:
    await anext(rows)
    local_service.models = [_entry(_MODEL, 'loaded')]
    local_service.state.phase = _SERVING
    local_service.publish()
    served = await anext(rows)

  assert served.choices[0].detail == '● In memory'


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('executable', 'phase', 'values', 'header'),
  [
    ('', _STOPPED, ['servers', 'settings'], 'llama-server not found'),
    (
      '/bin/llama-server',
      _STOPPED,
      ['start', 'servers', 'settings'],
      '○ stopped',
    ),
    (
      '/bin/llama-server',
      local_models_pb2.LOCAL_MODEL_SERVER_PHASE_STARTING,
      ['servers', 'settings'],
      '◐ starting…',
    ),
  ],
)
async def test_models_list_of_a_stopped_server_offers_to_start_it_once(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
  executable: str,
  phase: local_models_pb2.LocalModelServerPhase.ValueType,
  values: list[str],
  header: str,
) -> None:
  local_service.state.executable = executable
  local_service.state.phase = phase
  await _publish(runtime, local_service)

  update = await _first(menus.model_list(runtime))

  assert _values(update) == values
  assert update.header is not None and header in update.header


@pytest.mark.asyncio
async def test_an_external_servers_list_leads_back_to_the_managed_server(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.models = [_entry(_MODEL, 'unloaded')]

  external = menus.model_list(runtime, _SERVER)
  rows = await _first(external)
  opened = _step(external.selector.on_select('servers'))
  assert isinstance(opened, screens.Open)
  servers = await _first(opened.screen)
  cancel = opened.screen.selector.cancel_action
  assert cancel is not None
  back = _step(cancel)
  assert isinstance(back, screens.Open)

  assert _values(rows) == [f'load:{_MODEL}', 'download', 'servers']
  assert servers.choices[0].value == models.MANAGED
  assert back.screen.selector.title == external.selector.title


@pytest.mark.asyncio
async def test_choosing_a_downloading_model_pauses_its_download(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  pause = _step(menus.model_list(runtime).selector.on_select(f'pause:{_MODEL}'))
  assert isinstance(pause, screens.Do)

  await pause.work()

  assert local_service.requests[-1].cancel_download == _MODEL


@pytest.mark.asyncio
async def test_settings_offer_restart_and_stop_only_to_the_owner_of_a_running_server(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  stopped = await _first(menus.settings(runtime))
  await _running(runtime, local_service)
  running = await _first(menus.settings(runtime))
  local_service.state.owned = False
  await _publish(runtime, local_service)
  shared = await _first(menus.settings(runtime))

  assert _values(stopped) == ['context', 'sleep', 'presets']
  assert _values(running) == ['context', 'sleep', 'restart', 'stop', 'presets']
  assert _values(shared) == ['context', 'sleep', 'presets']


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('row', 'delta', 'field', 'value'),
  [('context', -1, 'context', 32768), ('sleep', 1, 'sleep_idle', 600)],
)
async def test_left_and_right_step_a_server_option_through_its_choices(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
  row: str,
  delta: int,
  field: str,
  value: int,
) -> None:
  local_service.state.context_choices.extend([32768, 65536])
  local_service.state.sleep_idle_choices.extend([0, 600])
  await _publish(runtime, local_service)
  screen = menus.settings(runtime)
  assert screen.selector.on_adjust is not None
  adjust = _step(screen.selector.on_adjust(row, delta))
  assert isinstance(adjust, screens.Do) and adjust.then is None

  await adjust.work()

  assert getattr(local_service.requests[-1], field) == value


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['9000', 'localhost:9000'])
async def test_typed_servers_are_probed_at_their_loopback_url(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
  query: str,
) -> None:
  local_service.models = [_entry(_MODEL, 'unloaded')]

  update = await _first(
    menus.servers(runtime, lambda: menus.model_list(runtime)), query
  )

  assert update.choices == (
    selection.Choice(_SERVER, _SERVER, 'llama.cpp · 1 model'),
  )
  assert local_service.requests[-1].base_url == _SERVER


@pytest.mark.asyncio
async def test_a_typed_server_that_is_not_loopback_is_rejected(
  runtime: local_models_client.Client,
) -> None:
  with pytest.raises(ValueError, match='loopback'):
    await _first(
      menus.servers(runtime, lambda: menus.model_list(runtime)),
      'example.com:8080',
    )


@pytest.mark.asyncio
async def test_effort_screen_offers_what_the_model_verified(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  await _running(runtime, local_service)
  opened = _step(menus.effort(runtime, _MANAGED_OPTION, 'none'))
  assert isinstance(opened, screens.Open)

  update = await _first(opened.screen)

  assert update.choices == (
    selection.Choice('none', 'Provider default'),
    selection.Choice('on', 'Thinking on'),
    selection.Choice('off', 'Thinking off'),
  )
  assert opened.screen.selector.on_select('on') == actions.SetReasoning('on')


def _candidate(
  name: str,
  *,
  trusted: bool = False,
  max_context: int = 0,
  size: int = 0,
  downloaded: int = 0,
  partial: bool = False,
) -> local_models_pb2.LocalModelCandidate:
  return local_models_pb2.LocalModelCandidate(
    name=name,
    trusted=trusted,
    pipeline='text-generation',
    parameters=7_000_000_000,
    max_context=max_context,
    download_bytes=size,
    downloaded=downloaded,
    partial=partial,
  )


@pytest.mark.asyncio
async def test_catalog_keeps_the_daemons_order_and_marks_trusted_publishers(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.candidates = [
    _candidate('someone/featured-GGUF'),
    _candidate('owner/a-GGUF', trusted=True, max_context=262144),
  ]
  home = menus.model_list(runtime)

  update = await _first(
    downloads.catalog(runtime, models.MANAGED, lambda: home), 'a'
  )

  assert update.choices == (
    selection.Choice(
      'someone/featured-GGUF', 'someone/featured-GGUF', 'text · 7.0B'
    ),
    selection.Choice(
      'owner/a-GGUF', 'owner/a-GGUF', 'Trusted · text · 7.0B · 256K ctx'
    ),
  )
  assert local_service.requests[-1].catalog == 'a'


@pytest.mark.asyncio
async def test_search_and_variants_mark_what_the_servers_disk_holds(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.candidates = [
    _candidate('o/r', downloaded=2),
    _candidate('o/r:Q4', size=2**30, downloaded=1),
    _candidate('o/r:Q8', size=2**30, partial=True),
  ]
  home = menus.model_list(runtime, _SERVER)

  found = await _first(downloads.catalog(runtime, _SERVER, lambda: home), 'r')
  quants = await _first(
    downloads.variants(runtime, _SERVER, 'o/r', lambda: home)
  )

  assert found.choices[0].detail.startswith('2 downloaded · ')
  assert quants.choices[1].detail == 'Downloaded · 1.0 GiB'
  assert quants.choices[2].detail == 'Partial download · 1.0 GiB'
  assert quants.header == 'Context size unknown'
  assert local_service.requests[-1].base_url == _SERVER


@pytest.mark.asyncio
async def test_accepting_a_review_sends_the_reviewed_manifest_and_lands_on_the_list(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local_service.manifest = local_models_pb2.LocalModelDownload(
    name=_MODEL,
    quantization='Q4_K_M',
    license='Apache\nsecond line',
    blobs=[local_models_pb2.LocalModelBlob(path='a.gguf', size=2**31)],
  )
  home = menus.model_list(runtime)
  screen = downloads.review(runtime, models.MANAGED, _MODEL, lambda: home)

  update = await _first(screen)
  accept = _step(screen.selector.on_select('download'))
  assert isinstance(accept, screens.Do)
  await accept.work()

  assert update.header == (
    'Q4_K_M · 2.0 GiB · https://huggingface.co/owner/model-GGUF\n'
    'Apache\nsecond line'
  )
  assert local_service.requests[-1].download == local_service.manifest
  assert accept.then is not None and accept.then.selector.id == 'llama.models'


def _runtime_state(
  server: local_models_pb2.LocalModelState,
) -> local_models_client.Client:
  client = local_models_client.Client('unused')
  client.state = server
  return client


@pytest.mark.parametrize(
  ('server', 'text'),
  [
    (
      local_models_pb2.LocalModelState(
        phase=local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING,
        active_model=_MODEL,
      ),
      '◐ loading model…',
    ),
    (
      local_models_pb2.LocalModelState(phase=_SERVING, active_model=_MODEL),
      '● serving',
    ),
    (
      local_models_pb2.LocalModelState(
        phase=_SERVING, active_model=_MODEL, asleep=True
      ),
      '○ asleep',
    ),
    (
      local_models_pb2.LocalModelState(phase=_SERVING, active_model='other'),
      '○ ready',
    ),
  ],
)
def test_footer_reports_the_server_as_the_selected_model_sees_it(
  server: local_models_pb2.LocalModelState, text: str
) -> None:
  assert footer.status(_runtime_state(server), _MODEL)[1] == text


def test_footer_shows_download_progress_and_how_downloads_ended() -> None:
  runtime = _runtime_state(
    local_models_pb2.LocalModelState(
      transfers=[
        _transfer(_MODEL, _DOWNLOADING),
        _transfer(
          'a/fresh-GGUF', local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_FINISHED
        ),
      ]
    )
  )

  assert footer.downloads(runtime) == (
    ('class:footer', 'model '),
    ('class:footer-approval-auto', '━╸'),
    ('class:footer', '───── 25%'),
    ('class:footer', ' · '),
    ('class:footer', '✓ fresh downloaded'),
  )


def test_presets_file_is_created_once_and_never_overwritten(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'presets.ini'

  host.ensure_presets_file(path)
  path.write_text('[*]\nctx-size = 1\n')
  host.ensure_presets_file(path)

  assert path.read_text() == '[*]\nctx-size = 1\n'
  assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_only_servers_the_user_started_are_detected() -> None:
  listing = """\
  10     1 /usr/bin/mirror local-models-guard
  11    10 llama-server --host 127.0.0.1 --port 18433
  12    11 /opt/llama-server --port 18434
  20     1 /opt/bin/llama-server --port=9000 -m model.gguf
  21     1 /opt/bin/llama-server --host 192.168.1.5 --port 9001
  22     1 vim llama-server
"""

  assert host.running_servers(listing) == (_SERVER,)


def _frontend(
  runtime: local_models_client.Client | None, selected: str = models.MANAGED
) -> frontend.Context:
  return frontend.Context(
    connection=None,
    metadata=cast(daemon_client.Client, None),
    terminal=io.StringIO(),
    switch=lambda _: None,
    commands=lambda: (),
    statuses=lambda _: (),
    local_runtime=runtime,
    local_selection=lambda: selected,
  )


class _Screens:
  """Opens screens as the extension does, against the real reducer."""

  def __init__(self, runtime: local_models_client.Client) -> None:
    self.plugin = inference_controls.InferenceControlsExtension(
      debounce_seconds=0
    )
    self.plugin.frontend_context = _frontend(runtime)
    self.current = state.State(
      activity=state.ActivityState(),
      status_bar=state.StatusBarState('model', 'medium', 'auto'),
      session=state.SessionState('conversation'),
      lifecycle=state.LifecycleState(active=True),
    )
    self.dispatched: list[actions.Action] = []
    self.tasks: list[asyncio.Task[None]] = []
    self._changed = asyncio.Event()
    self.context = extension.Context(
      extension.Workspace(pathlib.Path('/'), remote=False),
      self.dispatch,
      lambda: self.current,
      lambda _: None,
      refresh_tools=lambda _: asyncio.sleep(0),
      call_tool=lambda name, arguments: pytest.fail('Unexpected tool call'),
      transcript=tuple,
      builtins=tuple,
      extension_tools=tuple,
      workspace_controller=cast(workspace_controller.WorkspaceController, None),
      create_agent=lambda options, on_message, on_error: pytest.fail(
        'Unexpected agent creation'
      ),
    )

  def dispatch(self, action: actions.Action) -> None:
    self.dispatched.append(action)
    _, emitted = self.plugin.reduce(self.current, action)
    self.current, _ = reducer.reduce(self.current, action)
    self._changed.set()
    for effect in emitted:
      assert isinstance(effect, extension.PrivateEffect)
      self.tasks.append(
        asyncio.create_task(
          self.plugin.handle_effect(self.context, effect.payload)
        )
      )

  @property
  def values(self) -> list[str]:
    selector = self.current.selector.selector
    return [] if selector is None else [c.value for c in selector.choices]

  @property
  def selector(self) -> selection.Selector[object]:
    selector = self.current.selector.selector
    assert selector is not None
    return selector

  async def settled(self, condition: Callable[[], bool]) -> None:
    """Wait until a dispatched action makes `condition` true."""
    async with asyncio.timeout(5):
      while not condition():
        self._changed.clear()
        await self._changed.wait()

  async def close(self) -> None:
    self.dispatch(actions.ClearSelector())
    for task in self.tasks:
      task.cancel()
    await asyncio.gather(*self.tasks, return_exceptions=True)


@contextlib.asynccontextmanager
async def _shown(
  runtime: local_models_client.Client,
) -> AsyncGenerator[_Screens]:
  harness = _Screens(runtime)
  try:
    yield harness
  finally:
    await harness.close()


@pytest.mark.asyncio
async def test_a_failed_screen_offers_a_retry_that_runs_it_again(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  async with _shown(runtime) as shown:
    shown.dispatch(menus.effort(runtime, _MANAGED_OPTION, 'none'))
    await shown.settled(lambda: shown.values == [screens.REFRESH_ROW])
    assert shown.selector.choices[0].detail == 'llama.cpp is not running'

    await _running(runtime, local_service)
    shown.dispatch(
      cast(actions.Action, shown.selector.on_select(screens.REFRESH_ROW))
    )
    await shown.settled(lambda: 'on' in shown.values)


@pytest.mark.asyncio
async def test_typing_in_a_search_screen_runs_the_producer_for_the_new_query(
  runtime: local_models_client.Client,
) -> None:
  queries: list[str] = []

  async def stream(query: str) -> AsyncGenerator[screens.Update]:
    queries.append(query)
    yield screens.Update((selection.Choice(query, f'Result for {query!r}'),))

  screen = screens.Screen(
    selection.Selector(
      'llama.test',
      'Test',
      (),
      lambda _: actions.ClearSelector(),
      prefiltered=True,
    ),
    stream,
  )
  async with _shown(runtime) as shown:
    shown.dispatch(screens.step(screens.Open(screen)))
    await shown.settled(lambda: queries == [''])

    shown.dispatch(actions.SelectorSearchChanged('flint'))
    await shown.settled(lambda: shown.values == ['flint'])

    assert queries == ['', 'flint']
    assert not shown.current.selector.search_pending


@pytest.mark.asyncio
async def test_opening_another_screen_stops_the_previous_producer(
  runtime: local_models_client.Client,
) -> None:
  closed = asyncio.Event()

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    try:
      yield screens.Update(())
      await asyncio.Future[None]()
    finally:
      closed.set()

  first = screens.Screen(
    selection.Selector(
      'llama.first', 'First', (), lambda _: actions.ClearSelector()
    ),
    stream,
  )
  second = screens.Screen(
    selection.Selector(
      'llama.second', 'Second', (), lambda _: actions.ClearSelector()
    )
  )
  async with _shown(runtime) as shown:
    shown.dispatch(screens.step(screens.Open(first)))
    await shown.settled(lambda: shown.current.selector.id == 'llama.first')
    await shown.settled(lambda: not shown.selector.loading)

    shown.dispatch(screens.step(screens.Open(second)))

    async with asyncio.timeout(5):
      await closed.wait()
    await shown.settled(lambda: shown.current.selector.id == 'llama.second')


@pytest.mark.asyncio
async def test_closing_a_screen_stops_its_producer_and_task(
  runtime: local_models_client.Client,
) -> None:
  closed = asyncio.Event()

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    try:
      yield screens.Update(())
      await asyncio.Future[None]()
    finally:
      closed.set()

  screen = screens.Screen(
    selection.Selector(
      'llama.test', 'Test', (), lambda _: actions.ClearSelector()
    ),
    stream,
  )
  async with _shown(runtime) as shown:
    shown.dispatch(screens.step(screens.Open(screen)))
    await shown.settled(lambda: shown.current.selector.id == 'llama.test')
    await shown.settled(lambda: not shown.selector.loading)

    shown.dispatch(actions.ClearSelector())

    async with asyncio.timeout(5):
      await closed.wait()
      await shown.tasks[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('selected', 'owned', 'phase', 'prompts'),
  [
    (models.MANAGED, True, _READY, True),
    (_SERVER, True, _READY, False),
    (models.MANAGED, False, _READY, False),
    (models.MANAGED, True, _STOPPED, False),
  ],
)
async def test_quit_asks_only_about_a_server_this_session_uses_and_owns(
  runtime: local_models_client.Client,
  selected: str,
  owned: bool,
  phase: local_models_pb2.LocalModelServerPhase.ValueType,
  prompts: bool,
) -> None:
  runtime.state = local_models_pb2.LocalModelState(owned=owned, phase=phase)
  plugin = inference_controls.InferenceControlsExtension()
  plugin.frontend_context = _frontend(runtime, selected)

  assert (plugin.quit_prompt() is not None) is prompts


@pytest.mark.asyncio
async def test_answering_the_quit_prompt_stops_the_server_then_quits_without_asking_again(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  await _running(runtime, local_service)
  async with _shown(runtime) as shown:
    assert shown.plugin.quit_prompt() is not None
    shown.dispatch(screens.step(screens.Quit(stop=True)))
    await shown.settled(lambda: actions.QuitApp() in shown.dispatched)

    assert local_service.requests[-1].WhichOneof('operation') == 'stop'
    assert shown.plugin.quit_prompt() is None


@pytest.mark.asyncio
async def test_a_failed_stop_does_not_quit_and_asks_again(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  failure = actions.ShowText(
    'Could not stop the llama.cpp server: '
    'This server belongs to another Mirror daemon',
    style='ui.error',
  )
  await _running(runtime, local_service)
  async with _shown(runtime) as shown:
    assert shown.plugin.quit_prompt() is not None
    local_service.state.owned = False
    shown.dispatch(screens.step(screens.Quit(stop=True)))
    await shown.settled(lambda: failure in shown.dispatched)
    local_service.state.owned = True
    await _publish(runtime, local_service)

    assert actions.QuitApp() not in shown.dispatched
    assert shown.plugin.quit_prompt() is not None


@pytest.mark.asyncio
async def test_only_the_latest_model_choice_is_loaded_and_switched_to(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  slow = actions.ModelResolved(
    models.ModelOption('slow', 'slow', api='llama_cpp', base_url=models.MANAGED)
  )
  fast = actions.ModelResolved(
    models.ModelOption('fast', 'fast', api='llama_cpp', base_url=models.MANAGED)
  )
  await _running(runtime, local_service)
  local_service.ready.clear()
  async with _shown(runtime) as shown:
    shown.dispatch(screens.step(screens.Load('slow', models.MANAGED)))
    await asyncio.wait_for(local_service.requested.wait(), 2)
    local_service.ready.set()
    shown.dispatch(screens.step(screens.Load('fast', models.MANAGED)))
    await shown.settled(lambda: fast in shown.dispatched)
    await asyncio.gather(shown.tasks[0], return_exceptions=True)

    assert slow not in shown.dispatched


@pytest.mark.asyncio
async def test_a_pending_load_never_replaces_a_newer_cloud_choice(
  runtime: local_models_client.Client,
  local_service: local_models_service.LocalService,
) -> None:
  local = actions.ModelResolved(
    models.ModelOption('slow', 'slow', api='llama_cpp', base_url=models.MANAGED)
  )
  await _running(runtime, local_service)
  local_service.ready.clear()
  async with _shown(runtime) as shown:
    shown.dispatch(screens.step(screens.Load('slow', models.MANAGED)))
    await asyncio.wait_for(local_service.requested.wait(), 2)
    shown.dispatch(
      actions.ModelResolved(
        models.ModelOption('cloud', 'cloud', api='openai-chat-completions')
      )
    )
    local_service.ready.set()
    await asyncio.gather(shown.tasks[0], return_exceptions=True)

    assert local not in shown.dispatched


def test_llama_cpp_is_offered_only_with_a_local_runtime(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = tmp_path / 'models.json'
  config.write_text('[]')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))

  with_runtime = inference_controls.model_selector(
    'model', local_models_client.Client('unused')
  )
  without = inference_controls.model_selector('model')

  local = selection.Choice(
    'llama_cpp', 'llama.cpp (on this machine)', 'llama_cpp · on this machine'
  )
  assert local in with_runtime.choices
  assert local not in without.choices
