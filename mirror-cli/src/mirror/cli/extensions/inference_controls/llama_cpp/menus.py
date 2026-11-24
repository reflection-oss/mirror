"""The llama.cpp menus: models, server settings, other servers, and thinking."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import AsyncGenerator, Callable, Sequence

from google.protobuf import empty_pb2
from mirror.protocol import (
  local_models_pb2,
)

from mirror.cli import actions, local_models_client, models, selection
from mirror.cli.extensions.inference_controls.llama_cpp import (
  downloads,
  footer,
  host,
  screens,
)

_EMPTY = empty_pb2.Empty()
_STATUS = local_models_pb2.ExecuteRequest(status=_EMPTY)
_START = local_models_pb2.ExecuteRequest(start=_EMPTY)
_STOP = local_models_pb2.ExecuteRequest(stop=_EMPTY)
_RESTART = local_models_pb2.ExecuteRequest(restart=_EMPTY)
_TRANSFER_SHOWN = (
  local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_DOWNLOADING,
  local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_FAILED,
)
_EFFORT_LABELS = {
  'none': 'Provider default',
  'on': 'Thinking on',
  'off': 'Thinking off',
}


def _context_name(count: int) -> str:
  return footer.tokens(count) if count else 'Auto'


def _idle_name(seconds: int) -> str:
  if not seconds:
    return 'Off'
  if seconds % 3600 == 0:
    return f'{seconds // 3600} h'
  return f'{seconds // 60} min'


def _with_tilde(path: str) -> str:
  try:
    return f'~/{pathlib.Path(path).relative_to(pathlib.Path.home())}'
  except ValueError:
    return path


def _server_line(runtime: local_models_client.Client) -> str:
  """The server's live status, its model and anything it wants the user to know."""
  state = runtime.state
  _, status = footer.status(runtime, state.active_model)
  line = f'{status} · {state.active_model or "no model loaded"}'
  return f'{line} · {state.notice}' if state.notice else line


def model_list(
  runtime: local_models_client.Client, base: str = models.MANAGED
) -> screens.Screen:
  """The models of the shared server, or of the user-run server at `base`."""
  managed = base == models.MANAGED

  def home() -> screens.Screen:
    return model_list(runtime, base)

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    await runtime.execute(_STATUS)
    listed: object = None
    inventory: Sequence[local_models_pb2.LocalModelEntry] = ()
    while True:
      seen = runtime.state.revision
      if managed and runtime.state.phase not in footer.RUNNING:
        yield _stopped(runtime)
      else:
        # Cached models change when a download starts or ends, and what is in
        # memory when the server's phase or active model changes.
        key = (
          runtime.state.phase,
          runtime.state.active_model,
          tuple((item.name, item.phase) for item in runtime.state.transfers),
        )
        if key != listed:
          reply = await runtime.execute(
            local_models_pb2.ExecuteRequest(base_url=base, inventory=_EMPTY)
          )
          listed, inventory = key, reply.models
        yield screens.Update(
          _rows(runtime, inventory, base),
          _server_line(runtime) if managed else None,
        )
      await runtime.changed(seen)

  def choose(value: str) -> actions.Action:
    verb, _, name = value.partition(':')
    match verb:
      case 'download':
        return screens.goto(downloads.catalog(runtime, base, home))
      case 'servers':
        return screens.goto(servers(runtime, home))
      case 'settings':
        return screens.goto(settings(runtime))
      case 'start':
        return screens.perform(lambda: runtime.execute(_START))
      case 'pause':
        # Partial data is kept, so choosing the model again resumes it.
        return screens.perform(
          lambda: runtime.execute(
            local_models_pb2.ExecuteRequest(cancel_download=name)
          )
        )
      case 'resume':
        return screens.goto(downloads.review(runtime, base, name, home))
      case _:
        return screens.step(screens.Load(name, base))

  return screens.Screen(
    selection.Selector(
      'llama.models',
      'llama.cpp · models' if managed else 'llama.cpp · models on your server',
      (),
      choose,
      keep_open=True,
      # The actions after the models stay in view however many are listed.
      pinned_count=3 if managed else 2,
      placeholder='Type to filter models',
    ),
    stream,
  )


def _stopped(runtime: local_models_client.Client) -> screens.Update:
  state = runtime.state
  servers_row = selection.Choice('servers', 'Use another server…')
  settings_row = selection.Choice('settings', 'Server settings…')
  if not state.executable:
    return screens.Update(
      (servers_row, settings_row),
      'llama-server not found. Install it on your PATH.',
    )
  start = (
    ()
    if state.phase in footer.STARTING
    else (selection.Choice('start', 'Start server', 'Loads no model'),)
  )
  return screens.Update(
    (*start, servers_row, settings_row), _server_line(runtime)
  )


def _rows(
  runtime: local_models_client.Client,
  inventory: Sequence[local_models_pb2.LocalModelEntry],
  base: str,
) -> tuple[selection.Choice, ...]:
  """Downloads to `base`, then its cached models, then the actions in view."""
  transfers = [
    item
    for item in runtime.state.transfers
    if item.phase in _TRANSFER_SHOWN and item.server == base
  ]
  # A transfer row replaces the router's own inventory row for the same model.
  shown = {item.name for item in transfers}
  rows = [_transfer_row(item) for item in transfers]
  for item in inventory:
    if item.name not in shown:
      verb = 'resume' if item.status == 'incomplete' else 'load'
      detail = _model_detail(item, runtime.state)
      rows.append(selection.Choice(f'{verb}:{item.name}', item.name, detail))
  rows += [
    selection.Choice('download', 'Download a model…'),
    selection.Choice('servers', 'Use another server…'),
  ]
  if base == models.MANAGED:
    rows.append(selection.Choice('settings', 'Server settings…'))
  return tuple(rows)


def _transfer_row(
  item: local_models_pb2.LocalModelTransfer,
) -> selection.Choice:
  if item.phase == local_models_pb2.LOCAL_MODEL_TRANSFER_PHASE_FAILED:
    detail = f'Failed: {item.error or "unknown error"} · Enter resumes'
    return selection.Choice(f'resume:{item.name}', item.name, detail)
  percent = (
    f'{item.done_bytes * 100 // item.total_bytes}%' if item.total_bytes else '…'
  )
  done, total = (
    downloads.size(item.done_bytes),
    downloads.size(item.total_bytes),
  )
  return selection.Choice(
    f'pause:{item.name}',
    item.name,
    f'Downloading {percent} · {done} of {total} · Enter pauses',
  )


def _model_detail(
  item: local_models_pb2.LocalModelEntry,
  state: local_models_pb2.LocalModelState,
) -> str:
  context = f' · {footer.tokens(item.context)}' if item.context else ''
  asleep = f'Asleep · wakes on use{context}'
  # The live state is newer than the inventory for the active model.
  if item.name == state.active_model:
    if state.phase == local_models_pb2.LOCAL_MODEL_SERVER_PHASE_LOADING:
      return '◐ Loading…'
    if state.asleep:
      return asleep
  match item.status:
    case 'loaded':
      return f'● In memory{context}'
    case 'sleeping':
      return asleep
    case 'unloaded':
      # llama.cpp's own measurement of what fits, when this release reports one.
      fit = item.fit_context and f' · up to {footer.tokens(item.fit_context)}'
      return f'Downloaded{fit or context}'
    case 'incomplete':
      return 'Partial download'
    case _:
      return f'{item.status.capitalize()} · {item.source}{context}'


def _stepped(current: int, choices: tuple[int, ...], delta: int) -> int:
  """The choice `delta` steps from `current`, wrapping around."""
  if not choices:
    return current
  index = choices.index(current) if current in choices else -1
  return choices[(index + delta) % len(choices)]


def _option_rows(
  state: local_models_pb2.LocalModelState,
) -> tuple[selection.Choice, ...]:
  """The options ←/→ change, each value padded so changing it shifts nothing."""
  contexts = [_context_name(count) for count in (0, *state.context_choices)]
  idles = [_idle_name(seconds) for seconds in state.sleep_idle_choices]
  width = max(len(name) for name in (*contexts, *idles))
  context = _context_name(state.context).ljust(width)
  idle = _idle_name(state.sleep_idle).ljust(width)
  return (
    selection.Choice(
      'context', 'Context', 'Applies at next start', status=f'< {context} >'
    ),
    selection.Choice(
      'sleep', 'Unload when idle', 'Applies at next start', status=f'< {idle} >'
    ),
  )


async def _adjust(
  runtime: local_models_client.Client, value: str, delta: int
) -> None:
  """Step the server option on settings row `value`, if it has one."""
  state = runtime.state
  match value:
    case 'context':
      context = _stepped(state.context, (0, *state.context_choices), delta)
      request = local_models_pb2.ExecuteRequest(context=context)
    case 'sleep':
      choices = tuple(state.sleep_idle_choices)
      seconds = _stepped(state.sleep_idle, choices, delta)
      request = local_models_pb2.ExecuteRequest(sleep_idle=seconds)
    case _:
      return
  await runtime.execute(request)


def settings(runtime: local_models_client.Client) -> screens.Screen:
  """Server options, with the server's own state live above them."""

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    await runtime.execute(_STATUS)
    while True:
      seen = runtime.state.revision
      state = runtime.state
      rows = list(_option_rows(state))
      if state.phase in footer.RUNNING and state.owned:
        rows += [
          selection.Choice(
            'restart', 'Restart server', 'Reload with the current settings'
          ),
          selection.Choice(
            'stop', 'Stop server', 'Unloads the model and stops llama.cpp'
          ),
        ]
      if state.models_preset:
        presets = _with_tilde(state.models_preset)
        rows.append(
          selection.Choice('presets', 'Edit per-model settings…', presets)
        )
      yield screens.Update(
        tuple(rows),
        f'{_server_line(runtime)}\n'
        f'Server: {state.executable or "llama-server not found"}\n'
        f'Models: {_with_tilde(state.models_dir)}',
      )
      await runtime.changed(seen)

  # Each step waits for the one before, so it starts from the daemon's reply.
  stepping = asyncio.Lock()

  async def adjust(value: str, delta: int) -> None:
    async with stepping:
      await _adjust(runtime, value, delta)

  def choose(value: str) -> actions.Action:
    match value:
      case 'context' | 'sleep':
        return screens.perform(lambda: adjust(value, 1))
      case 'restart':
        return screens.perform(lambda: runtime.execute(_RESTART))
      case 'stop':
        return screens.perform(lambda: runtime.execute(_STOP))
      case _:
        return screens.step(screens.Edit(runtime.state.models_preset))

  return screens.Screen(
    selection.Selector(
      'llama.settings',
      'llama.cpp · server settings · ←/→ to change',
      (),
      choose,
      cancel_action=screens.goto(model_list(runtime)),
      keep_open=True,
      on_adjust=lambda value, delta: screens.perform(
        lambda: adjust(value, delta)
      ),
    ),
    stream,
  )


def servers(
  runtime: local_models_client.Client, home: Callable[[], screens.Screen]
) -> screens.Screen:
  """Offer the managed server, others running here, or one the user types."""

  async def found(url: str) -> selection.Choice:
    try:
      reply = await runtime.execute(
        local_models_pb2.ExecuteRequest(base_url=url, inventory=_EMPTY)
      )
    except ValueError:
      # The daemon's own wording describes an HTTP failure, not what to do next.
      raise ValueError(f'Nothing answering at {url}') from None
    count = len(reply.models)
    models_text = '1 model' if count == 1 else f'{count} models'
    return selection.Choice(url, url, f'llama.cpp · {models_text}')

  async def stream(query: str) -> AsyncGenerator[screens.Update]:
    if query.strip():
      yield screens.Update((await found(host.server_url(query)),))
      return
    results = await asyncio.gather(
      *(found(url) for url in await host.detect_servers()),
      return_exceptions=True,
    )
    yield screens.Update((
      selection.Choice(models.MANAGED, 'Managed server', 'Mirror runs it'),
      *(row for row in results if isinstance(row, selection.Choice)),
    ))

  return screens.Screen(
    selection.Selector(
      'llama.servers',
      'llama.cpp · servers',
      (),
      lambda url: screens.goto(model_list(runtime, url)),
      cancel_action=screens.goto(home()),
      prefiltered=True,
      header='Running here, or type a port',
      placeholder='Port or host:port, e.g. 8080',
    ),
    stream,
  )


def effort(
  runtime: local_models_client.Client,
  option: models.ModelOption,
  current: str,
) -> actions.Action:
  """Offer only the thinking controls this model's server verified."""

  async def stream(_: str) -> AsyncGenerator[screens.Update]:
    reply = await runtime.execute(
      local_models_pb2.ExecuteRequest(
        base_url=option.base_url, info=option.model_id
      )
    )
    yield screens.Update(
      tuple(
        selection.Choice(value, _EFFORT_LABELS.get(value, value))
        for value in reply.info.efforts
      )
    )

  return screens.goto(
    screens.Screen(
      selection.Selector(
        'llama.effort',
        f'Thinking · {option.model_id}',
        (),
        actions.SetReasoning,
        current=current,
        selected=current,
      ),
      stream,
    )
  )


def quit_prompt() -> screens.Screen:
  """Ask whether to stop the server on the way out; Esc cancels the quit."""
  return screens.Screen(
    selection.Selector(
      'llama.quit',
      'A llama.cpp server is still running',
      (
        selection.Choice('keep', 'Keep it running'),
        selection.Choice('stop', 'Stop it'),
      ),
      lambda value: screens.step(screens.Quit(stop=value == 'stop')),
    )
  )
