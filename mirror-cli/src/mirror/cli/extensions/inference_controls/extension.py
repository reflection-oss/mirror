"""Model, reasoning, and speed commands for the Mirror CLI."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from typing import override

from mirror.cli import (
  actions,
  api_keys,
  commands,
  extension,
  inference_choices,
  models,
  reflection,
  selection,
)

_PROTOCOL_IDS = frozenset(api for api, _ in models.WIRE_PROTOCOLS)


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


def _protocol_choices() -> tuple[selection.Choice, ...]:
  """The wire protocols, always offered even without a models config."""
  choices: list[selection.Choice] = []
  for api, label in models.WIRE_PROTOCOLS:
    hint = _credential_hint(api)
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


def _select_inference(value: str) -> actions.Action:
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
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  """Select a host-owned model or open its picker."""
  if arguments:
    if len(arguments.split()) > 1:
      return actions.ShowText(
        'Model id must not contain spaces.', style='ui.warning'
      )
    return (
      actions.SetModel(arguments)
      if context.daemon_models is not None
      else _select_inference(arguments)
    )
  return actions.OpenSelector(
    model_selector(context.model, options=context.daemon_models)
  )


def model_selector(
  current_model: str, *, options: tuple[models.ModelOption, ...] | None = None
) -> selection.Selector[actions.Action]:
  """Build the shared model picker for commands and sign-in."""
  configured = tuple(
    selection.Choice(option.model_id, option.label, option.model_id)
    for option in (models.load_models() if options is None else options)
  )
  return selection.Selector(
    'model',
    'Switch model or protocol',
    (_protocol_choices() if options is None else ()) + configured,
    _select_inference if options is None else actions.SetModel,
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
  context: commands.CommandContext, arguments: str
) -> actions.Action:
  effort = arguments.lower()
  efforts = inference_choices.REASONING_EFFORTS
  if not effort:
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
  speed = arguments.strip().lower()
  if speed and speed not in ('normal', 'fast'):
    return actions.ShowText(
      f'Unknown speed: {arguments}. Choices: normal, fast',
      style='ui.warning',
    )
  if speed == 'normal':
    return actions.SetSpeed(speed)
  if not models.supports_fast_mode(context.model_api):
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


class InferenceControlsExtension(extension.Extension[None, None, None]):
  """Contribute the core inference-control commands."""

  @classmethod
  @override
  def id(cls) -> str:
    return 'core/inference-controls'

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
        model_command,
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
        _set_reasoning,
        arguments=commands.OptionalArguments('[effort]'),
        available_when_busy=True,
      ),
    )
