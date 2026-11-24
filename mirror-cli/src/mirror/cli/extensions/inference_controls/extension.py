"""Model, reasoning, and speed commands for the Mirror CLI."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import pathlib
import urllib.parse
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeAlias, override

from google.protobuf import empty_pb2
from mirror.protocol import (
  local_models_pb2,
)
from prompt_toolkit.layout import containers, controls

from mirror.cli import (
  actions,
  api_keys,
  commands,
  effects,
  extension,
  inference_choices,
  local_models_client,
  models,
  reflection,
  selection,
  state,
  terminal,
)
from mirror.cli.extensions.inference_controls import server_models
from mirror.cli.extensions.inference_controls.llama_cpp import (
  footer,
  host,
  menus,
  screens,
)

_PROTOCOL_IDS = frozenset(api for api, _ in models.WIRE_PROTOCOLS)
_NO_RUNTIME = 'Local model management is unavailable in this session.'


def _credential_hint(api: str) -> str:
  """A short readiness note for a protocol, shown beside its picker row."""
  if api == 'anthropic-messages':
    return 'key set' if api_keys.has_anthropic_key() else 'select to sign in'
  if api == 'reflection':
    return 'key set' if reflection.has_key() else 'select to sign in'
  if api == 'openai-codex-responses':
    return (
      'logged in' if inference_choices.codex_logged_in() else 'run codex login'
    )
  env = inference_choices.PROTOCOL_KEY_ENV.get(api)
  if env is None:
    return ''
  return api_keys.environment_key_hint(env)


def _protocol_choices(local: bool) -> tuple[selection.Choice, ...]:
  """The wire protocols, always offered even without a models config."""
  choices: list[selection.Choice] = []
  for api, label in models.WIRE_PROTOCOLS:
    if api == 'llama_cpp' and not local:
      continue
    hint = 'on this machine' if api == 'llama_cpp' else _credential_hint(api)
    detail = f'{api} · {hint}' if hint else api
    choices.append(selection.Choice(api, label, detail))
  return tuple(choices)


def _model_selected(model_id: str, api: str) -> actions.Action:
  """Switch to one specific model over the given wire protocol."""
  return actions.ModelResolved(models.ModelOption(model_id, model_id, api=api))


def _reflection_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'reflection')


def _openrouter_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'openai-chat-completions')


def _anthropic_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'anthropic-messages')


def _openai_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'openai-responses')


def _codex_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'openai-codex-responses')


def _gemini_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'gemini')


def _vercel_model_selected(model_id: str) -> actions.Action:
  return _model_selected(model_id, 'vercel')


def _default_model(api: str) -> actions.Action:
  """Switch to a protocol's default model without a secondary picker."""
  return actions.ModelResolved(
    models.ModelOption(models.default_model(api), api, api=api)
  )


def _model_picker(
  selector_id: str,
  title: str,
  on_select: Callable[[str], actions.Action],
  *,
  empty: str,
  load: Callable[[], Awaitable[tuple[selection.Choice, ...]]] | None = None,
  choices: tuple[selection.Choice, ...] = (),
) -> actions.Action:
  return actions.OpenSelector(
    selection.Selector(
      selector_id,
      title,
      choices,
      on_select,
      extension_id=InferenceControlsExtension.id(),
      empty=empty,
      load=load,
    )
  )


def _select_inference(
  value: str,
  runtime: local_models_client.Client | None = None,
  base: str = '',
) -> actions.Action:
  """Route a picker choice to a model picker, protocol default, or model id."""
  # Each protocol offers many models, so drill into a secondary picker. When a
  # protocol's credentials are absent the fetch cannot succeed, so fall back to
  # its default model rather than opening a picker that would only error.
  if value == 'reflection':
    if not reflection.has_key():
      return _default_model(value)
    return _model_picker(
      'reflection-models',
      'Reflection model',
      _reflection_model_selected,
      empty='Fetching Reflection models…',
      load=inference_choices.fetch_reflection_models,
    )
  if value == 'llama_cpp':
    if runtime is None:
      return actions.ShowText(_NO_RUNTIME, style='ui.warning')
    base = base or os.environ.get('MIRROR_LLAMA_CPP_BASE_URL', '')
    return screens.goto(menus.model_list(runtime, base or models.MANAGED))
  if value in ('openai-chat-completions', 'openrouter'):
    return _model_picker(
      'openrouter-models',
      'OpenRouter model',
      _openrouter_model_selected,
      empty='Fetching OpenRouter models…',
      load=inference_choices.fetch_openrouter_models,
    )
  if value == 'anthropic-messages':
    if not api_keys.has_anthropic_key():
      return _default_model(value)
    return _model_picker(
      'anthropic-models',
      'Anthropic model',
      _anthropic_model_selected,
      empty='Fetching Anthropic models…',
      load=inference_choices.fetch_anthropic_models,
    )
  if value == 'openai-responses':
    if not (
      os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_BASE_URL')
    ):
      return _default_model(value)
    return _model_picker(
      'openai-models',
      'OpenAI model',
      _openai_model_selected,
      empty='Fetching OpenAI models…',
      load=inference_choices.fetch_openai_models,
    )
  if value == 'openai-codex-responses':
    return _model_picker(
      'codex-models',
      'ChatGPT model',
      _codex_model_selected,
      empty='No models available.',
      choices=inference_choices.codex_choices(),
    )
  if value == 'gemini':
    return _model_picker(
      'gemini-models',
      'Google Gemini Developer API',
      _gemini_model_selected,
      empty='No supported models.',
      choices=inference_choices.GEMINI_CHOICES,
    )
  if value == 'vercel':
    if not os.environ.get('AI_GATEWAY_API_KEY', '').strip():
      return _default_model(value)
    return _model_picker(
      'vercel-models',
      'Vercel AI Gateway model',
      _vercel_model_selected,
      empty='Fetching Vercel AI Gateway models…',
      load=inference_choices.fetch_vercel_models,
    )
  if value in _PROTOCOL_IDS:
    return _default_model(value)
  return actions.SetModel(value)


def model_command(
  context: commands.CommandContext,
  arguments: str,
  runtime: local_models_client.Client | None = None,
) -> actions.Action:
  """Select a model or open its picker: the server's list in a remote tab."""
  if context.remote:
    return server_models.model_command(context, arguments)
  selected = context.model_selection
  base = (
    selected.base_url
    if selected.api == 'llama_cpp' and selected.base_url != models.MANAGED
    else ''
  )
  if arguments:
    if len(arguments.split()) > 1:
      return actions.ShowText(
        'Model id must not contain spaces.', style='ui.warning'
      )
    return (
      actions.SetModel(arguments)
      if context.daemon_models is not None
      else _select_inference(arguments, runtime, base)
    )
  return actions.OpenSelector(
    model_selector(
      context.model,
      runtime,
      base,
      options=context.daemon_models,
    )
  )


def model_selector(
  current_model: str,
  runtime: local_models_client.Client | None = None,
  base: str = '',
  *,
  options: tuple[models.ModelOption, ...] | None = None,
) -> selection.Selector[actions.Action]:
  """Build the shared model picker for commands and sign-in."""
  configured = tuple(
    selection.Choice(option.model_id, option.label, option.model_id)
    for option in (models.load_models() if options is None else options)
  )
  return selection.Selector(
    'model',
    'Switch model or protocol',
    (_protocol_choices(runtime is not None) if options is None else ())
    + configured,
    (lambda value: _select_inference(value, runtime, base))
    if options is None
    else actions.SetModel,
    current=current_model,
    extension_id=InferenceControlsExtension.id(),
    empty='No models available.',
  )


def _login(context: commands.CommandContext, arguments: str) -> actions.Action:
  del arguments
  if context.daemon_models is not None:
    return actions.ShowText(
      'Sign in on the daemon host; laptop credentials are not transferred.'
    )
  if reflection.has_key():
    source = (
      'REFLECTION_API_KEY'
      if reflection.has_environment_key()
      else 'a saved API key'
    )
    return actions.ShowText(
      f'Reflection is already configured with {source}. '
      'Use /model reflection to switch to it, or /logout to remove the saved key.'
    )
  return _default_model('reflection')


def _logout(_: commands.CommandContext, arguments: str) -> actions.Action:
  del arguments
  return actions.ForgetReflectionKey()


def _set_reasoning(
  context: commands.CommandContext,
  arguments: str,
  runtime: local_models_client.Client | None = None,
) -> actions.Action:
  effort = arguments.lower()
  efforts = inference_choices.REASONING_EFFORTS
  if not context.remote and context.model_selection.api == 'llama_cpp':
    if not effort:
      if runtime is None:
        return actions.ShowText(_NO_RUNTIME, style='ui.warning')
      return menus.effort(
        runtime, context.model_selection, context.reasoning_effort
      )
    # A typed llama.cpp setting may also toggle thinking; the model judges
    # whether its template supports it.
    efforts = (*efforts, 'on', 'off')
  elif not effort:
    return actions.OpenSelector(
      selection.Selector(
        'reasoning',
        'Reasoning effort',
        tuple(
          selection.Choice(value, inference_choices.effort_label(value))
          for value in efforts
        ),
        actions.SetReasoning,
        current=context.reasoning_effort,
        extension_id=InferenceControlsExtension.id(),
        selected=context.reasoning_effort,
      )
    )
  if effort not in efforts:
    return actions.ShowText(
      f'Unknown reasoning effort: {arguments}. Choices: {", ".join(efforts)}',
      style='ui.warning',
    )
  return actions.SetReasoning(effort)


def speed_command(
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  """Set the processing speed, or open its picker, for the current protocol."""
  if context.remote:
    return server_models.speed_command(context, arguments)
  speed = arguments.strip().lower()
  if speed and speed not in ('normal', 'fast'):
    return actions.ShowText(
      f'Unknown speed: {arguments}. Choices: normal, fast',
      style='ui.warning',
    )
  if speed == 'normal':
    return actions.SetSpeed(speed)
  if not models.supports_fast_mode(context.model_selection.api):
    return actions.ShowText(
      'Fast mode requires Anthropic Messages, OpenAI Responses, '
      'or a ChatGPT plan (Codex). '
      'Use /effort to adjust reasoning with other providers.',
      style='ui.warning',
    )
  if speed:
    return actions.SetSpeed(speed)
  return actions.OpenSelector(
    selection.Selector(
      'speed',
      'Model speed',
      (
        selection.Choice('normal', 'Normal', 'Provider default processing'),
        selection.Choice(
          'fast',
          'Fast',
          'Faster processing; may cost more or use limits faster',
        ),
      ),
      actions.SetSpeed,
      current=context.speed,
      selected=context.speed,
      extension_id=InferenceControlsExtension.id(),
    )
  )


@dataclasses.dataclass(frozen=True)
class _State:
  """Present only because the host skips extensions whose state is `None`."""


_STATE = _State()

_Effect: TypeAlias = (
  screens.Step | server_models.ServerModel | server_models.ServerFastMode
)


class InferenceControlsExtension(extension.Extension[_State, _Effect, None]):
  """Contribute the core inference-control commands."""

  def __init__(
    self, debounce_seconds: float = screens.DEBOUNCE_SECONDS
  ) -> None:
    self._screens = screens.Runner(debounce_seconds)
    self._loading: asyncio.Task[None] | None = None
    self._quit_decided = False

  @classmethod
  @override
  def id(cls) -> str:
    return screens.EXTENSION_ID

  @staticmethod
  @override
  def initial_state() -> _State:
    return _STATE

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_LEFT

  def _runtime(self) -> local_models_client.Client | None:
    context = self.frontend_context
    return context.local_runtime if context is not None else None

  def _footer(self, model: str) -> tuple[tuple[str, str], ...]:
    """Fragments for the local server's status and any background download."""
    context = self.frontend_context
    runtime = self._runtime()
    if context is None or runtime is None:
      return ()
    fragments: list[tuple[str, str]] = []
    selected = context.local_selection()
    if selected == models.MANAGED:
      style, text = footer.status(runtime, model)
      fragments += [('class:footer', 'llama.cpp '), (style, text)]
    elif selected:
      # A server the user runs: Mirror can't see its lifecycle, only where it is.
      host_name = urllib.parse.urlsplit(selected).netloc
      fragments += [('class:footer', f'llama.cpp · {host_name}')]
    if downloads := footer.downloads(runtime):
      if fragments:
        fragments.append(('class:footer', ' · '))
      fragments += downloads
    return tuple(fragments)

  @override
  def quit_prompt(self) -> actions.Action | None:
    context = self.frontend_context
    runtime = self._runtime()
    if (
      context is None
      or runtime is None
      or self._quit_decided
      or context.local_selection() != models.MANAGED
      or not runtime.state.owned
      or runtime.state.phase not in (*footer.STARTING, *footer.RUNNING)
    ):
      return None
    return screens.goto(menus.quit_prompt())

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    _State | None,
    tuple[
      effects.CoreEffect | extension.PrivateEffect[_Effect],
      ...,
    ],
  ]:
    selector = global_state.selector.selector
    if isinstance(action, actions.RoutedExtensionAction) and isinstance(
      action.action,
      screens.Step | server_models.ServerModel | server_models.ServerFastMode,
    ):
      return _STATE, (self.effect(action.action),)
    if (
      isinstance(action, (actions.ModelResolved, actions.SetModel))
      and self._loading is not None
    ):
      # Cancelled here rather than in an effect, so the load cannot finish and
      # switch back to its model before the newer choice applies.
      self._loading.cancel()
    if (
      isinstance(action, actions.ClearSelector)
      and selector is not None
      and selector.extension_id == self.id()
    ):
      # Wake the screen's producer, which stops once its selector is gone.
      return _STATE, (self.effect(screens.Refresh()),)
    if (
      isinstance(action, actions.SelectorSearchChanged)
      and selector is not None
      and selector.extension_id == self.id()
      and selector.prefiltered
    ):
      return _STATE, (self.effect(screens.Refresh()),)
    return _STATE, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: _Effect
  ) -> None:
    match effect:
      case server_models.ServerModel():
        context.dispatch(await server_models.server_model(context, effect))
      case server_models.ServerFastMode():
        for action in await server_models.server_fast_mode(context, effect):
          context.dispatch(action)
      case screens.Open(screen):
        await self._screens.show(context, screen)
      case screens.Refresh():
        self._screens.refresh()
      case screens.Do():
        await self._do(context, effect)
      case screens.Load():
        await self._load(context, effect)
      case screens.Edit(path):
        await self._edit(context, pathlib.Path(path))
      case screens.Quit(stop):
        await self._quit(context, stop)
      case _:
        raise TypeError(f'unhandled llama.cpp step: {effect!r}')

  async def _do(self, context: extension.Context, step: screens.Do) -> None:
    try:
      await step.work()
    except ValueError as error:
      context.dispatch(actions.ShowText(str(error), style='ui.error'))
    if step.then is not None:
      await self._screens.show(context, step.then)

  async def _edit(self, context: extension.Context, path: pathlib.Path) -> None:
    try:
      if context.run_terminal is None:
        raise RuntimeError('The frontend terminal is unavailable')
      host.ensure_presets_file(path)
      await context.run_terminal(
        terminal.editor_command(path), str(path.parent)
      )
    except (OSError, ValueError, RuntimeError) as error:
      context.dispatch(
        actions.ShowText(f'Could not edit {path}: {error}', style='ui.error')
      )

  async def _quit(self, context: extension.Context, stop: bool) -> None:
    runtime = self._runtime()
    self._quit_decided = True
    if stop and runtime is not None:
      # Stopping waits for the server to exit, which can take a few seconds, and
      # the prompt is already gone by then.
      context.dispatch(actions.ShowText('Stopping llama.cpp…', style='ui.info'))
      try:
        await runtime.execute(
          local_models_pb2.ExecuteRequest(stop=empty_pb2.Empty())
        )
      except ValueError as error:
        self._quit_decided = False
        context.dispatch(
          actions.ShowText(
            f'Could not stop the llama.cpp server: {error}', style='ui.error'
          )
        )
        return
    context.dispatch(actions.QuitApp())

  async def _load(self, context: extension.Context, step: screens.Load) -> None:
    """Make `step.name` the session's model once the server has it loaded.

    Only the latest choice proceeds: a load still in flight when another model
    is chosen is cancelled, which cancels it on the daemon too.
    """
    runtime = self._runtime()
    if runtime is None:
      context.dispatch(actions.ShowText(_NO_RUNTIME, style='ui.error'))
      return
    this = asyncio.current_task()
    previous, self._loading = self._loading, this
    if previous is not None and previous is not this:
      previous.cancel()
      await asyncio.gather(previous, return_exceptions=True)
    context.dispatch(actions.ClearSelector())
    try:
      await runtime.execute(
        local_models_pb2.ExecuteRequest(
          base_url=step.base,
          load=local_models_pb2.LocalModelLoad(
            model=step.name, allow_eviction=True
          ),
        )
      )
    except ValueError as error:
      context.dispatch(actions.ShowText(str(error), style='ui.error'))
      return
    finally:
      if self._loading is this:
        self._loading = None
    context.dispatch(
      actions.ModelResolved(
        models.ModelOption(
          step.name, step.name, api='llama_cpp', base_url=step.base
        )
      )
    )

  @override
  def cache_key(
    self, global_state: state.State, extension_state: _State
  ) -> object:
    del extension_state
    return self._footer(global_state.status_bar.model)

  @override
  def render(
    self, global_state: state.State, extension_state: _State
  ) -> containers.AnyContainer | None:
    del extension_state
    fragments = self._footer(global_state.status_bar.model)
    if not fragments:
      return None
    return containers.Window(
      controls.FormattedTextControl(list(fragments)),
      dont_extend_height=True,
      dont_extend_width=True,
    )

  @override
  def tips(self) -> tuple[str, ...]:
    return (
      '/model picks a different model for the next request',
      '/effort adjusts how much the model reasons',
    )

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'login',
        'Check Reflection credentials or sign in',
        _login,
      ),
      commands.CommandDef(
        'logout',
        'Remove the saved Reflection API key',
        _logout,
      ),
      commands.CommandDef(
        'model',
        'Switch the model or wire protocol',
        lambda context, arguments: model_command(
          context, arguments, self._runtime()
        ),
        arguments=commands.OptionalArguments('[model|protocol]'),
        available_when_busy=True,
      ),
      commands.CommandDef(
        'speed',
        'Set processing speed (fast may cost more or use limits faster)',
        speed_command,
        arguments=commands.OptionalArguments('[normal|fast]'),
        available_when_busy=True,
      ),
      commands.CommandDef(
        'effort',
        'Set reasoning effort',
        lambda context, arguments: _set_reasoning(
          context, arguments, self._runtime()
        ),
        arguments=commands.OptionalArguments('[effort]'),
        available_when_busy=True,
      ),
    )
