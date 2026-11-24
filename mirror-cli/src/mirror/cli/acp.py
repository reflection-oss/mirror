"""Serve the Agent Client Protocol on stdio over local agent workers.

ACP (https://agentclientprotocol.com) is JSON-RPC 2.0 as newline-delimited
JSON on stdin/stdout. Clients such as Omnigent and Zed use it to drive any
coding agent. `mirror acp` maps one ACP session onto one local agent worker:
`session/new` starts a worker in the client's working directory,
`session/prompt` runs one turn and streams its deltas back, tool approvals
become `session/request_permission` requests, and `session/cancel` interrupts
the turn. Stdout carries the protocol; diagnostics go to stderr.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import pathlib
import sys
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from typing import Protocol, cast

import grpc
import httpx2
from mirror.protocol import (
  agent_pb2,
  mirror_pb2,
)

from mirror.cli import (
  api_keys,
  inference_choices,
  local_rpc_client,
  models,
  noninteractive,
  onboarding,
  reflection,
  rpc_client,
  selection,
)

PROTOCOL_VERSION = 1

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

# One inbound line; prompts embed file context, so the bound is generous.
_MAX_LINE_BYTES = 64 * 1024 * 1024
# Tool output forwarded in a tool card; a full file read would otherwise be
# copied into every client transcript.
_MAX_TOOL_OUTPUT_CHARS = 16 * 1024
_PERMISSION_OPTIONS = (
  {'optionId': 'allow', 'name': 'Allow', 'kind': 'allow_once'},
  {'optionId': 'reject', 'name': 'Reject', 'kind': 'reject_once'},
)
_MODEL_OPTION = 'model'
_EFFORT_OPTION = 'effort'
SERVER_DEFAULT = 'server default'
_SESSION_PAGE = 100
_LOCAL_ATTACH_SECONDS = 30
_HOSTED_ATTACH_SECONDS = 960
# A slow provider catalog must not hold up session creation for long.
_CATALOG_TIMEOUT_SECONDS = 5

type JSON = dict[str, object]
type WorkerConfigFactory = Callable[[str], Mapping[str, object]]
# Starts a worker for a new conversation and returns the conversation's id.
# Given a conversation it started, it replaces that conversation's worker with
# one started from the new configuration.
type TransportFactory = Callable[
  [Mapping[str, object], pathlib.Path, str, rpc_client.MessageHandler],
  Awaitable[tuple[rpc_client.Transport, str]],
]
type ModelSource = Callable[[], Awaitable[Sequence[models.ModelOption]]]
type SessionPages = Callable[
  [int, int], Awaitable[tuple[Sequence[mirror_pb2.Session], int]]
]


class RPCError(Exception):
  """A JSON-RPC error answered to the client."""

  def __init__(self, code: int, message: str) -> None:
    super().__init__(message)
    self.code = code
    self.message = message


class Output(Protocol):
  """Where protocol lines go; `sys.stdout.buffer` in production."""

  def write(self, data: bytes, /) -> object: ...
  def flush(self) -> object: ...


def _local_transport(
  worker_command: Sequence[str],
) -> TransportFactory:
  workspaces: dict[str, str] = {}

  async def factory(
    config: Mapping[str, object],
    workspace: pathlib.Path,
    conversation_id: str,
    on_message: rpc_client.MessageHandler,
  ) -> tuple[rpc_client.Transport, str]:
    workspace_id = workspaces.get(conversation_id)
    restart = workspace_id is not None
    if workspace_id is None:
      # The daemon allocates the immutable conversation binding.
      record = await local_rpc_client.create_workspace(
        worker_command, config, workspace
      )
      conversation_id, workspace_id = record.session_id, record.id
      workspaces[conversation_id] = workspace_id
    transport = local_rpc_client.Transport(
      worker_command,
      dict(config)
      | {
        'conversation_id': conversation_id,
        'workspace_id': workspace_id,
        'require_new_worker': restart,
      },
      workspace=workspace,
      log_dir=pathlib.Path.home() / '.mirror' / 'logs' / conversation_id,
      on_message=on_message,
    )
    if restart:
      # Provider credentials reach a worker only when it starts.
      await transport.close_workspace()
    return transport, conversation_id

  return factory


async def offered_models() -> tuple[models.ModelOption, ...]:
  """What `/model` offers: `models.json`, then each ready protocol's models.

  Configured entries come first, so their endpoints and labels win over a
  catalog's entry for the same model.
  """
  # llama.cpp has no catalog; the daemon's local model service lists its models.
  apis = [
    api
    for api, _ in models.WIRE_PROTOCOLS
    if api != 'llama_cpp' and inference_choices.ready(api)
  ]
  catalogs = await asyncio.gather(*(_catalog(api) for api in apis))
  offered: list[models.ModelOption] = []
  for api, choices in zip(apis, catalogs, strict=True):
    offered.extend(
      models.ModelOption(choice.value, choice.label, choice.detail, api=api)
      for choice in choices
    )
  try:
    configured = models.load_models()
  except (ValueError, OSError) as error:
    print(f'mirror acp: ignoring model config: {error}', file=sys.stderr)
    configured = ()
  return (*configured, *offered)


async def _catalog(api: str) -> tuple[selection.Choice, ...]:
  """A protocol's models, or its default model when they cannot be listed."""
  try:
    return await asyncio.wait_for(
      inference_choices.protocol_models(api), _CATALOG_TIMEOUT_SECONDS
    )
  except (httpx2.HTTPError, TimeoutError, ValueError, OSError) as error:
    print(
      f'mirror acp: could not list {api} models ({type(error).__name__})',
      file=sys.stderr,
    )
    return inference_choices.default_choices(api)


def _value_id(option: models.ModelOption) -> str:
  # One model id can be served by two protocols.
  return f'{option.api}/{option.model_id}'


def _model_choices(
  offered: Sequence[models.ModelOption], start: models.ModelOption
) -> dict[str, models.ModelOption]:
  """Offered models by value id, each on a protocol, and the starting model.

  The first entry for a value id wins. An entry that names no protocol runs on
  the starting one, as `/model` inherits the session's protocol and endpoint.
  The starting model keeps the endpoint the session runs on.
  """
  choices: dict[str, models.ModelOption] = {}
  for option in offered:
    if not option.api:
      option = dataclasses.replace(
        option, api=start.api, base_url=option.base_url or start.base_url
      )
    choices.setdefault(_value_id(option), option)
  start_id = _value_id(start)
  if start_id in choices:
    choices[start_id] = dataclasses.replace(
      choices[start_id], base_url=start.base_url
    )
  else:
    choices = {start_id: start} | choices
  return choices


def _local_start(
  configuration: rpc_client.WorkerConfiguration, effort: str
) -> models.ModelOption:
  """The model a local worker starts on, with the endpoint it runs on."""
  model = str(configuration.values['model'])
  inference = configuration.inference_config(model, effort, remote=False)
  return models.ModelOption(
    model, model, api=inference.api, base_url=inference.base_url
  )


def _hosted_start(
  offered: Sequence[models.ModelOption],
  configuration: rpc_client.WorkerConfiguration,
) -> models.ModelOption:
  """The model a hosted session runs, as its server lists it when it does."""
  model = str(configuration.values['model'])
  api = str(configuration.values.get('remote_model_api') or '')
  return next(
    (
      option
      for option in offered
      if option.model_id == model and api in ('', option.api)
    ),
    models.ModelOption(
      model, 'Server default' if model == SERVER_DEFAULT else model, api=api
    ),
  )


def _hosted_choices(
  offered: Sequence[models.ModelOption],
  start: models.ModelOption,
  *,
  new: bool,
) -> dict[str, models.ModelOption]:
  """A server's models by value id, the running one first.

  A session this process creates runs the server's default until it is told
  otherwise, so it also offers that default by name. A kept session may have
  been given another base model, which naming no model would keep, so it
  offers only what the server lists.
  """
  default = (models.ModelOption(SERVER_DEFAULT, 'Server default'),) * new
  return {_value_id(option): option for option in (start, *default, *offered)}


def _kept_selection(
  client: rpc_client.Client,
  configuration: rpc_client.WorkerConfiguration,
  effort: str,
) -> tuple[rpc_client.WorkerConfiguration, str]:
  """Adopt the model and effort an attached session reports it runs with."""
  settings = client.shared_settings
  if settings is None or not settings.HasField('inference'):
    return configuration, effort
  inference = settings.inference
  return (
    configuration.updated({
      'model': inference.model or SERVER_DEFAULT,
      'remote_model_api': inference.api or None,
    }),
    inference.reasoning_effort
    if inference.reasoning_effort in inference_choices.REASONING_EFFORTS
    else effort,
  )


def _check_selection(
  option: models.ModelOption, effort: str, max_tokens: object
) -> None:
  """Refuse what Mirror's own checks refuse; `mirror acp` cannot sign in."""
  try:
    models.validate_selection(
      option,
      effort,
      max_tokens=max_tokens if isinstance(max_tokens, int) else None,
    )
    models.validate_connection(option.api, option.base_url)
  except ValueError as error:
    raise RPCError(
      _INVALID_PARAMS, f'cannot use {option.model_id}: {error}'
    ) from None
  if option.api == 'reflection' and not reflection.has_key():
    missing = (
      'set REFLECTION_API_KEY or run `mirror --api reflection` to save a key '
      f'({reflection.KEY_URL})'
    )
  elif option.api == 'anthropic-messages' and not (
    api_keys.has_anthropic_key()
    or option.base_url
    or os.environ.get('ANTHROPIC_BASE_URL')
  ):
    missing = (
      'set ANTHROPIC_API_KEY or run `mirror --api anthropic-messages` to save '
      f'a key ({api_keys.ANTHROPIC_KEY_URL})'
    )
  else:
    return
  raise RPCError(_INVALID_PARAMS, f'cannot use {option.model_id}: {missing}')


def tool_kind(name: str) -> str:
  """Map a Mirror tool name onto ACP's tool-call kinds (icons only)."""
  lowered = name.strip().lower()
  if lowered == 'read':
    return 'read'
  if lowered in ('edit', 'write', 'multi_edit'):
    return 'edit'
  if lowered in ('bash', 'shell', 'exec'):
    return 'execute'
  if lowered in ('grep', 'find', 'glob', 'ls'):
    return 'search'
  if lowered in ('web_search', 'web_browse', 'fetch', 'web_fetch'):
    return 'fetch'
  return 'other'


def prompt_text(blocks: Sequence[object]) -> str:
  """Flatten ACP prompt blocks into the one message a turn accepts."""
  parts: list[str] = []
  for value in blocks:
    block = _object(value)
    if block is None:
      continue
    kind = block.get('type')
    if kind == 'text':
      text = block.get('text')
      if isinstance(text, str) and text:
        parts.append(text)
    elif kind == 'resource':
      resource = _object(block.get('resource'))
      if resource is not None:
        text = resource.get('text')
        uri = resource.get('uri', '')
        if isinstance(text, str) and text:
          parts.append(f'<file uri="{uri}">\n{text}\n</file>')
    elif kind == 'resource_link':
      uri = str(block.get('uri', ''))
      name = block.get('name')
      reference = f'{name} ({uri})' if isinstance(name, str) and name else uri
      if reference:
        parts.append(f'Referenced resource: {reference}')
    elif kind in ('image', 'audio'):
      parts.append(f'[{kind} attachment omitted: not supported by mirror acp]')
  return '\n\n'.join(parts)


def _object(value: object) -> JSON | None:
  """Narrow a decoded JSON value to an object."""
  return cast(JSON, value) if isinstance(value, dict) else None


def _array(value: object) -> list[object]:
  """Narrow a decoded JSON value to an array; anything else is empty."""
  return cast(list[object], value) if isinstance(value, list) else []


def _tool_arguments(arguments: str) -> JSON:
  try:
    parsed: object = json.loads(arguments) if arguments.strip() else {}
  except ValueError:
    return {'arguments': arguments}
  parsed_object = _object(parsed)
  # An empty object is a valid, empty argument list.
  return parsed_object if parsed_object is not None else {'arguments': parsed}


def _truncate(text: str) -> str:
  if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
    return text
  return text[:_MAX_TOOL_OUTPUT_CHARS] + '\n… [truncated]'


def _tool_call_update(
  call_id: str, name: str, arguments: str, status: str
) -> JSON:
  return {
    'sessionUpdate': 'tool_call',
    'toolCallId': call_id,
    'title': name,
    'kind': tool_kind(name),
    'status': status,
    'rawInput': _tool_arguments(arguments),
  }


def _tool_result_update(call_id: str, content: str, *, success: bool) -> JSON:
  update: JSON = {
    'sessionUpdate': 'tool_call_update',
    'toolCallId': call_id,
    'status': 'completed' if success else 'failed',
  }
  text = _truncate(content)
  if text:
    update['content'] = [
      {'type': 'content', 'content': {'type': 'text', 'text': text}}
    ]
  return update


def _chunk(kind: str, text: str) -> JSON:
  return {'sessionUpdate': kind, 'content': {'type': 'text', 'text': text}}


@dataclasses.dataclass
class TurnOutcome:
  """How one prompt ended."""

  stop_reason: str
  usage: JSON | None = None
  error: str | None = None


class Translator:
  """Translate one turn's session messages into ACP updates.

  Streamed deltas are forwarded as they arrive; committed records supply the
  tool calls and results, whose ids tie `tool_call` and `tool_call_update`
  together. A committed answer also supplies the end of its text when the
  stream stopped short of it: the daemon drops the deltas it has not yet
  delivered once the answer is committed. Usage from the turn's model calls is
  summed for the final result.
  """

  def __init__(self) -> None:
    self._usage: dict[str, int] = {}
    self._task_titles: dict[str, str] = {}
    self._streamed = ''
    self._interrupted = False
    self._notice: str | None = None

  def translate(
    self, event: agent_pb2.SessionMessage
  ) -> tuple[list[JSON], TurnOutcome | None]:
    payload = event.WhichOneof('payload')
    if payload == 'runtime_delta':
      return self._runtime_delta(event.runtime_delta)
    if payload == 'agent_message':
      return self._record(event.agent_message), None
    if payload == 'input_nack':
      return [], TurnOutcome(
        'end_turn',
        error=f'agent rejected the request: {event.input_nack.error}',
      )
    return [], None

  def usage(self) -> JSON | None:
    if not self._usage:
      return None
    result: JSON = {
      'inputTokens': self._usage.get('prompt_tokens', 0),
      'outputTokens': self._usage.get('completion_tokens', 0),
      'totalTokens': self._usage.get('prompt_tokens', 0)
      + self._usage.get('completion_tokens', 0),
    }
    if cached := self._usage.get('cached_tokens'):
      result['cachedReadTokens'] = cached
    if written := self._usage.get('cache_write_tokens'):
      result['cachedWriteTokens'] = written
    return result

  def _runtime_delta(
    self, delta: agent_pb2.RuntimeDelta
  ) -> tuple[list[JSON], TurnOutcome | None]:
    kind = delta.WhichOneof('payload')
    if kind == 'stream_content':
      text = delta.stream_content.content
      if not text:
        return [], None
      self._streamed += text
      return [_chunk('agent_message_chunk', text)], None
    if kind == 'stream_reasoning':
      text = delta.stream_reasoning.reasoning
      return ([_chunk('agent_thought_chunk', text)] if text else []), None
    if kind == 'stream_refusal':
      text = delta.stream_refusal.refusal
      if not text:
        return [], None
      self._streamed += text
      return [_chunk('agent_message_chunk', text)], None
    if kind == 'stream_aborted':
      self._streamed = ''
      return [], None
    if kind == 'context_usage':
      usage = delta.context_usage
      return [
        {
          'sessionUpdate': 'usage_update',
          'used': usage.tokens,
          'size': usage.window,
        }
      ], None
    if kind == 'task_update':
      return self._task_update(delta.task_update), None
    if kind == 'turn_completed':
      if self._interrupted:
        return [], TurnOutcome('cancelled', usage=self.usage())
      if self._notice is not None:
        # The worker gave up (for example on provider errors) and committed
        # a notice instead of a model answer.
        return [], TurnOutcome('end_turn', error=self._notice)
      return [], TurnOutcome('end_turn', usage=self.usage())
    if kind == 'turn_interrupted':
      # The worker follows this with `turn_completed`; ending the turn there
      # keeps that completion out of the next prompt's queue.
      self._interrupted = True
      return [], None
    if kind == 'turn_error':
      # Provider errors may contain request bodies or credentials.
      return [], TurnOutcome(
        'end_turn', error='agent turn failed; see the worker log for details'
      )
    return [], None

  def _task_update(self, task: agent_pb2.TaskUpdate) -> list[JSON]:
    call_id = f'task:{task.id}'
    title = task.display or task.tool_name or 'task'
    if task.status == agent_pb2.TASK_STATUS_RUNNING:
      if call_id in self._task_titles:
        return []
      self._task_titles[call_id] = title
      return [
        {
          'sessionUpdate': 'tool_call',
          'toolCallId': call_id,
          'title': title,
          'kind': tool_kind(task.tool_name),
          'status': 'in_progress',
        }
      ]
    if task.status in (
      agent_pb2.TASK_STATUS_FAILED,
      agent_pb2.TASK_STATUS_CANCELLED,
    ):
      self._task_titles.pop(call_id, None)
      return [
        {
          'sessionUpdate': 'tool_call_update',
          'toolCallId': call_id,
          'status': 'failed',
        }
      ]
    # Success carries its content in the task_result record that follows.
    return []

  def _record(self, record: agent_pb2.AgentMessage) -> list[JSON]:
    payload = record.WhichOneof('payload')
    if payload == 'task_result':
      result = record.task_result
      self._task_titles.pop(f'task:{result.task_id}', None)
      return [
        _tool_result_update(
          f'task:{result.task_id}', result.content, success=result.success
        )
      ]
    if payload != 'chat':
      return []
    chat = record.chat
    if chat.HasField('usage'):
      for field in (
        'prompt_tokens',
        'completion_tokens',
        'cached_tokens',
        'cache_write_tokens',
      ):
        self._usage[field] = self._usage.get(field, 0) + getattr(
          chat.usage, field
        )
    message = chat.WhichOneof('message')
    if message == 'assistant':
      return self._assistant(record)
    if message == 'tool':
      tool = chat.tool
      if not tool.tool_call_id:
        return []
      return [
        _tool_result_update(
          tool.tool_call_id, tool.content, success=chat.success
        )
      ]
    return []

  def _assistant(self, record: agent_pb2.AgentMessage) -> list[JSON]:
    chat = record.chat
    assistant = chat.assistant
    text = assistant.content or assistant.refusal
    if not record.HasField('inference'):
      # A synthetic notice (limits, provider failures), not a model answer.
      if text.strip() and not assistant.tool_calls:
        self._notice = text.strip()
      return []
    if chat.success:
      # A model answer after a notice means the worker recovered from it;
      # only a notice that ends the turn fails the prompt.
      self._notice = None
    updates: list[JSON] = []
    streamed, self._streamed = self._streamed, ''
    if text != streamed and text.startswith(streamed):
      updates.append(_chunk('agent_message_chunk', text[len(streamed) :]))
    updates.extend(
      _tool_call_update(
        call.id,
        call.function.name,
        call.function.arguments,
        'in_progress',
      )
      for call in assistant.tool_calls
      if call.id
    )
    return updates


def history_updates(records: Sequence[agent_pb2.AgentMessage]) -> list[JSON]:
  """Translate a conversation's committed records into the updates of a load.

  User messages, model answers, tool calls and tool results are replayed in
  order. Background task results are left out: the calls they answer were
  reported from runtime state, which the record of a conversation does not
  keep.
  """
  translator = Translator()
  updates: list[JSON] = []
  for record in records:
    if record.WhichOneof('payload') != 'chat':
      continue
    if record.chat.WhichOneof('message') == 'user':
      if record.chat.user.content:
        updates.append(_chunk('user_message_chunk', record.chat.user.content))
      continue
    replayed, _ = translator.translate(
      agent_pb2.SessionMessage(agent_message=record)
    )
    updates.extend(replayed)
  return updates


def _session_info(session: mirror_pb2.Session, directory: str) -> JSON:
  """One `session/list` entry; a hosted session has no folder of its own."""
  info: JSON = {'sessionId': session.id, 'cwd': directory}
  if session.title:
    info['title'] = session.title
  if session.HasField('updated_at'):
    info['updatedAt'] = session.updated_at.ToDatetime(
      tzinfo=datetime.UTC
    ).isoformat()
  return info


def _failure(error: Exception) -> str:
  """A one-line reason for an error answered to the client."""
  if isinstance(error, grpc.aio.AioRpcError):
    return f'{error.code().name}: {error.details()}'
  if isinstance(error, TimeoutError):
    return 'timed out'
  return str(error) or type(error).__name__


@dataclasses.dataclass
class _Turn:
  """One accepted prompt, from dispatch until its response is written."""

  started: bool = False
  cancel_requested: bool = False
  cancelled: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
  # The outstanding `session/request_permission`, so a cancel can settle it.
  permission: asyncio.Future[JSON] | None = None


class _Session:
  """One ACP session: a worker, its event queue, and the active turn."""

  def __init__(
    self,
    session_id: str,
    client: rpc_client.Client,
    workspace: pathlib.Path,
    configuration: rpc_client.WorkerConfiguration,
    choices: dict[str, models.ModelOption],
    model: str,
    effort: str,
    events: asyncio.Queue[agent_pb2.SessionMessage | None],
  ) -> None:
    self.session_id = session_id
    self.client = client
    self.workspace = workspace
    # The launch settings, carrying the current model, protocol and endpoint.
    self.configuration = configuration
    # The offered models by value id, and the current one's value id.
    self.choices = choices
    self.model = model
    self.effort = effort
    self.events = events
    self.turn: _Turn | None = None
    # Set once the worker connection is lost; the session takes no more prompts.
    self.failed = False
    self.failure: str | None = None
    self.deferred = False
    self.attaching: asyncio.Task[None] | None = None
    # Set while a worker for another provider replaces the current one.
    self.switching = False


async def _attached(attaching: asyncio.Task[None], turn: _Turn) -> bool:
  """Wait for a session's first attachment, or for the prompt's cancellation.

  Returns:
    Whether the attachment finished, successfully or not, before a cancel.
  """
  cancelled = asyncio.ensure_future(turn.cancelled.wait())
  try:
    await asyncio.wait(
      (attaching, cancelled), return_when=asyncio.FIRST_COMPLETED
    )
  finally:
    cancelled.cancel()
  return attaching.done()


def _send_inference(
  session: _Session, inference: agent_pb2.InferenceConfig
) -> None:
  session.client.send(
    agent_pb2.ClientMessage(
      session_id=session.session_id,
      message_id=uuid.uuid4().hex,
      set_inference_config=inference,
    )
  )


def _discard_idle_events(session: _Session) -> None:
  """Drop what the session reported while no prompt was running.

  A kept session can run turns for another client; their events, down to the
  completion that would end the next prompt at once, belong to no prompt here.
  A lost connection is kept: it fails the next prompt.
  """
  while not session.events.empty():
    if session.events.get_nowait() is None:
      session.failed = True
  if session.failed:
    session.events.put_nowait(None)


def _config_options(session: _Session) -> list[JSON]:
  """The session's model and reasoning effort selectors."""
  labels = dict(models.WIRE_PROTOCOLS)
  groups: dict[str, list[JSON]] = {api: [] for api in labels}
  for value, option in session.choices.items():
    entry: JSON = {'value': value, 'name': option.label}
    if option.description:
      entry['description'] = option.description
    groups.setdefault(option.api, []).append(entry)
  return [
    {
      'id': _MODEL_OPTION,
      'name': 'Model',
      'category': 'model',
      'type': 'select',
      'currentValue': session.model,
      'options': [
        {
          'group': api,
          'name': labels.get(api, api) or 'Server',
          'options': entries,
        }
        for api, entries in groups.items()
        if entries
      ],
    },
    {
      'id': _EFFORT_OPTION,
      'name': 'Reasoning effort',
      'category': 'thought_level',
      'type': 'select',
      'currentValue': session.effort,
      'options': [
        {'value': effort, 'name': inference_choices.effort_label(effort)}
        for effort in inference_choices.REASONING_EFFORTS
      ],
    },
  ]


def _session_models(session: _Session) -> JSON:
  """The offered models for clients from before session config options."""
  labels = dict(models.WIRE_PROTOCOLS)
  return {
    'availableModels': [
      {
        'modelId': value,
        'name': option.label,
        'description': labels.get(option.api, option.api),
      }
      for value, option in session.choices.items()
    ],
    'currentModelId': session.model,
  }


class Server:
  """Answer one ACP client over a line-oriented reader and writer.

  Given `session_pages`, the sessions belong to a server that runs and keeps
  them: `transport` starts a session there, or rejoins one when the worker
  configuration says `create_session` is false, and `model_source` lists the
  server's models. Such sessions are listed a page at a time by offset and
  limit, each page with the total, and are loaded or resumed after the process
  that started them is gone.

  A new session on a server is attached at its first prompt, not when it is
  opened. Clients open sessions only to read what an agent offers, and a
  server can take longer to prepare a checkout than a client allows a session
  to start. The first prompt waits for the attachment instead, and is sent
  only once the session is attached, so cancelling the wait sends nothing.
  """

  def __init__(
    self,
    *,
    worker_config: WorkerConfigFactory,
    transport: TransportFactory,
    reasoning_effort: str,
    consent: Callable[[pathlib.Path], None] | None = None,
    model_source: ModelSource = offered_models,
    session_pages: SessionPages | None = None,
    hosted_attach_seconds: float = _HOSTED_ATTACH_SECONDS,
  ) -> None:
    self._worker_config = worker_config
    self._transport = transport
    self._reasoning_effort = reasoning_effort
    self._consent = consent
    self._model_source = model_source
    self._session_pages = session_pages
    self._hosted = session_pages is not None
    self._hosted_attach_seconds = hosted_attach_seconds
    self._offered: asyncio.Future[Sequence[models.ModelOption]] | None = None
    self._sessions: dict[str, _Session] = {}
    self._pending: dict[str, asyncio.Future[JSON]] = {}
    self._next_request_id = 0
    self._output: Output | None = None
    self._write_lock = asyncio.Lock()
    self._requests: set[asyncio.Task[None]] = set()

  async def serve(self, reader: asyncio.StreamReader, output: Output) -> None:
    """Dispatch client messages until the client closes its end."""
    self._output = output
    try:
      while True:
        try:
          line = await reader.readline()
        except (asyncio.LimitOverrunError, ValueError):
          await self._write_error(None, _PARSE_ERROR, 'line too long')
          continue
        if not line:
          return
        line = line.strip()
        if not line:
          continue
        try:
          message = json.loads(line)
        except ValueError as error:
          await self._write_error(None, _PARSE_ERROR, f'invalid json: {error}')
          continue
        if not isinstance(message, dict):
          await self._write_error(None, _INVALID_REQUEST, 'expected an object')
          continue
        await self._dispatch(cast(JSON, message))
    finally:
      for task in list(self._requests):
        task.cancel()
      await asyncio.gather(*self._requests, return_exceptions=True)
      if self._offered is not None:
        self._offered.cancel()
      for session in list(self._sessions.values()):
        await session.client.close()
      self._sessions.clear()

  async def _dispatch(self, message: JSON) -> None:
    method = message.get('method')
    if not isinstance(method, str):
      request_id = message.get('id')
      if isinstance(request_id, str) and request_id in self._pending:
        # The client answered one of our requests (a permission prompt). A
        # cancel may have settled it already, with the answer close behind.
        future = self._pending.pop(request_id)
        if not future.done():
          future.set_result(message)
      return
    params = message.get('params')
    params = cast(JSON, params) if isinstance(params, dict) else {}
    request_id = message.get('id')
    if request_id is None:
      await self._notification(method, params)
      return
    if method == 'session/prompt':
      try:
        session, turn, text = self._accept_prompt(params)
      except RPCError as error:
        await self._write_error(request_id, error.code, error.message)
        return
      handler = self._prompt_request(request_id, session, turn, text)
    else:
      handler = self._respond(request_id, method, params)
    # Requests run off the read loop, so a slow worker start or a long turn
    # never delays a cancel or a permission answer for another session.
    self._spawn(handler)

  def _spawn(self, work: Coroutine[object, object, None]) -> asyncio.Task[None]:
    task = asyncio.create_task(work)
    self._requests.add(task)
    task.add_done_callback(self._requests.discard)
    return task

  async def _respond(
    self, request_id: object, method: str, params: JSON
  ) -> None:
    try:
      result = await self._request(method, params)
    except RPCError as error:
      await self._write_error(request_id, error.code, error.message)
      return
    await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

  async def _request(self, method: str, params: JSON) -> JSON:
    if method == 'initialize':
      # Catalogs load while the client finishes its handshake.
      self._offered_models()
      capabilities: JSON = {
        'loadSession': self._hosted,
        'promptCapabilities': {
          'image': False,
          'audio': False,
          'embeddedContext': True,
        },
      }
      if self._hosted:
        capabilities['sessionCapabilities'] = {'list': {}, 'resume': {}}
      return {
        'protocolVersion': PROTOCOL_VERSION,
        'agentCapabilities': capabilities,
        'authMethods': [],
      }
    if method == 'authenticate':
      return {}
    if method == 'session/new':
      return await self._new_session(params)
    if self._session_pages is not None:
      if method == 'session/list':
        return await self._list_sessions(self._session_pages, params)
      if method == 'session/load':
        return await self._open_session(params, replay=True)
      if method == 'session/resume':
        return await self._open_session(params, replay=False)
    if method == 'session/set_config_option':
      return await self._set_config_option(params)
    if method == 'session/set_model':
      await self._switch_model(
        self._session(params), str(params.get('modelId', ''))
      )
      return {}
    raise RPCError(_METHOD_NOT_FOUND, f'method not supported: {method}')

  def _session(self, params: JSON) -> _Session:
    session = self._sessions.get(str(params.get('sessionId', '')))
    if session is None:
      raise RPCError(_INVALID_PARAMS, 'unknown session')
    return session

  def _offered_models(self) -> asyncio.Future[Sequence[models.ModelOption]]:
    """The models every session offers, listed until a listing has some."""
    listed = self._offered
    if listed is None or (
      listed.done()
      and not listed.cancelled()
      and listed.exception() is None
      and not listed.result()
    ):
      listed = asyncio.ensure_future(self._model_source())
      self._offered = listed
    return listed

  async def _notification(self, method: str, params: JSON) -> None:
    if method != 'session/cancel':
      return
    session = self._sessions.get(str(params.get('sessionId', '')))
    if session is None or session.turn is None:
      return
    turn = session.turn
    turn.cancel_requested = True
    turn.cancelled.set()
    if turn.permission is not None and not turn.permission.done():
      # The client may never answer a permission request it has dismissed.
      turn.permission.set_result({
        'result': {'outcome': {'outcome': 'cancelled'}}
      })
    if turn.started:
      session.client.send(
        agent_pb2.ClientMessage(
          session_id=session.session_id,
          message_id=uuid.uuid4().hex,
          interrupt=agent_pb2.Interrupt(),
        )
      )

  async def _new_session(self, params: JSON) -> JSON:
    session = await self._start_session(self._workspace(params), None)
    return {
      'sessionId': session.session_id,
      'configOptions': _config_options(session),
      'models': _session_models(session),
    }

  async def _open_session(self, params: JSON, *, replay: bool) -> JSON:
    """Rejoin a session the server kept, replaying its conversation on a load."""
    session_id = params.get('sessionId')
    if not isinstance(session_id, str) or not session_id:
      raise RPCError(_INVALID_PARAMS, 'sessionId is required')
    workspace = self._workspace(params)
    session = self._sessions.get(session_id)
    if session is not None and session.failed:
      del self._sessions[session_id]
      await session.client.close()
      session = None
    if session is None:
      session = await self._start_session(workspace, session_id)
    if replay and not session.deferred:
      try:
        records = await session.client.messages()
      except (OSError, RuntimeError, grpc.RpcError) as error:
        raise RPCError(
          _INTERNAL_ERROR,
          f'could not read the conversation ({_failure(error)})',
        ) from None
      for update in history_updates(records):
        await self._notify(
          'session/update', {'sessionId': session_id, 'update': update}
        )
    return {
      'configOptions': _config_options(session),
      'models': _session_models(session),
    }

  async def _list_sessions(self, pages: SessionPages, params: JSON) -> JSON:
    cwd = params.get('cwd')
    directory = cwd if isinstance(cwd, str) and cwd else str(pathlib.Path.cwd())
    cursor = params.get('cursor')
    if cursor is None:
      offset = 0
    elif isinstance(cursor, str) and cursor.isdecimal():
      offset = int(cursor)
    else:
      raise RPCError(_INVALID_PARAMS, 'unknown cursor')
    try:
      listed, total = await pages(offset, _SESSION_PAGE)
    except (OSError, RuntimeError, TimeoutError, grpc.RpcError) as error:
      raise RPCError(
        _INTERNAL_ERROR, f'could not list sessions ({_failure(error)})'
      ) from None
    result: JSON = {
      'sessions': [_session_info(session, directory) for session in listed]
    }
    if listed and offset + len(listed) < total:
      result['nextCursor'] = str(offset + len(listed))
    return result

  def _workspace(self, params: JSON) -> pathlib.Path:
    """The client's working directory, once its notices have been accepted."""
    cwd = params.get('cwd')
    workspace = (
      pathlib.Path(cwd) if isinstance(cwd, str) and cwd else pathlib.Path.cwd()
    )
    if not workspace.is_dir():
      raise RPCError(_INVALID_PARAMS, f'cwd is not a directory: {workspace}')
    workspace = workspace.resolve()
    if self._consent is not None:
      try:
        self._consent(workspace)
      except SystemExit as error:
        raise RPCError(_INVALID_PARAMS, str(error)) from None
    servers = _array(params.get('mcpServers'))
    if servers:
      print(
        f'mirror acp: ignoring {len(servers)} mcpServers; configure MCP in '
        'Mirror instead',
        file=sys.stderr,
      )
    return workspace

  async def _start_session(
    self, workspace: pathlib.Path, session_id: str | None
  ) -> _Session:
    """Attach a new conversation, or the kept one that `session_id` names."""
    offered = self._offered_models()
    existing = session_id is not None
    conversation_id = session_id or uuid.uuid4().hex
    config = self._worker_config(conversation_id)
    if existing:
      config = dict(config) | {'create_session': False}
    configuration = rpc_client.WorkerConfiguration(config, cwd=workspace)
    events: asyncio.Queue[agent_pb2.SessionMessage | None] = asyncio.Queue()
    deferred = self._hosted and not existing
    client, conversation_id = await self._attach(
      configuration,
      self._reasoning_effort,
      workspace,
      conversation_id,
      events,
      restart=existing,
      started=not deferred,
    )
    try:
      choices = await asyncio.shield(offered)
    except asyncio.CancelledError:
      await client.close()
      raise
    effort = self._reasoning_effort
    if self._hosted:
      configuration, effort = _kept_selection(client, configuration, effort)
    model = str(configuration.values['model'])
    start = (
      _hosted_start(choices, configuration)
      if self._hosted
      else _local_start(configuration, effort)
    )
    session = _Session(
      conversation_id,
      client,
      workspace,
      configuration,
      _hosted_choices(choices, start, new=not existing)
      if self._hosted
      else _model_choices(choices, start),
      _value_id(start),
      effort,
      events,
    )
    self._sessions[conversation_id] = session
    print(
      f'mirror acp: session {conversation_id} in {workspace} ({model})',
      file=sys.stderr,
    )
    session.deferred = deferred
    return session

  async def _finish_attaching(self, session: _Session) -> None:
    """Attach a session for its first prompt, recording why if it cannot.

    Nothing awaits this task's result, so every failure is recorded on the
    session, where the waiting prompt and later ones find it.
    """
    try:
      async with asyncio.timeout(self._hosted_attach_seconds):
        await session.client.start(replay=False)
    except Exception as error:
      session.failure = self._attach_failure(error)
      session.failed = True
      session.events.put_nowait(None)
      print(
        f'mirror acp: session {session.session_id}: {session.failure}',
        file=sys.stderr,
      )

  def _attach_failure(self, error: Exception) -> str:
    if self._hosted:
      return f'could not open the session on the server ({_failure(error)})'
    return (
      'could not start agent; check model authentication and worker '
      f'configuration ({error})'
    )

  async def _attach(
    self,
    configuration: rpc_client.WorkerConfiguration,
    effort: str,
    workspace: pathlib.Path,
    conversation_id: str,
    events: asyncio.Queue[agent_pb2.SessionMessage | None],
    *,
    restart: bool,
    started: bool = True,
  ) -> tuple[rpc_client.Client, str]:
    """Start a worker for the conversation that sends its events to `events`.

    With `started` false the client is returned before it attaches; it accepts
    requests meanwhile and sends them in order once it has.

    The launch settings go to the transport with the attachment. A hosted
    session's transport hands them to the daemon as the configuration for a
    session with no conversation yet, and opens the stream with a handshake
    that sets nothing, so a session rejoined on its server keeps the model and
    effort it runs with.
    """
    # A replacement worker's attachment first replays the conversation, which
    # the client already shows; the end of the replay means it is attached.
    replayed = asyncio.Event()
    if not restart:
      replayed.set()

    def on_message(event: agent_pb2.SessionMessage) -> None:
      if replayed.is_set():
        events.put_nowait(event)
      elif event.WhichOneof('payload') == 'replay_done':
        replayed.set()

    async def on_error(error: str) -> None:
      del error  # Raw transport errors can include provider payloads.
      events.put_nowait(None)
      failed = self._sessions.get(conversation_id)
      if failed is None:
        return
      failed.failed = True
      turn = failed.turn
      if turn is not None and turn.permission is not None:
        # Nothing can approve a call on a worker that is gone.
        if not turn.permission.done():
          turn.permission.set_result({})

    client: rpc_client.Client | None = None
    try:
      transport, conversation_id = await self._transport(
        configuration.values, workspace, conversation_id, on_message
      )
      client = rpc_client.Client(
        transport,
        conversation_id,
        on_message=on_message,
        on_error=on_error,
        web_provider=None if self._hosted else configuration.web_provider,
        inference_config=configuration.inference_config(
          str(configuration.values['model']), effort, remote=self._hosted
        ),
      )
      if started:
        async with asyncio.timeout(
          self._hosted_attach_seconds if self._hosted else _LOCAL_ATTACH_SECONDS
        ):
          await client.start(replay=False)
          await replayed.wait()
    except Exception as error:
      if client is not None:
        await client.close()
      raise RPCError(_INTERNAL_ERROR, self._attach_failure(error)) from None
    return client, conversation_id

  async def _set_config_option(self, params: JSON) -> JSON:
    session = self._session(params)
    config_id, value = params.get('configId'), params.get('value')
    if not isinstance(value, str):
      raise RPCError(_INVALID_PARAMS, 'value must be a string')
    if config_id == _MODEL_OPTION:
      await self._switch_model(session, value)
    elif config_id == _EFFORT_OPTION:
      if value not in inference_choices.REASONING_EFFORTS:
        raise RPCError(_INVALID_PARAMS, f'unknown reasoning effort: {value}')
      await self._select(session, session.model, value)
    else:
      raise RPCError(_INVALID_PARAMS, f'unknown config option: {config_id}')
    return {'configOptions': _config_options(session)}

  async def _switch_model(self, session: _Session, value: str) -> None:
    if value not in session.choices:
      raise RPCError(_INVALID_PARAMS, f'unknown model: {value}')
    await self._select(session, value, session.effort)

  async def _select(self, session: _Session, model: str, effort: str) -> None:
    """Apply a model and effort the way `/model` and `/effort` do."""
    if session.failed:
      raise RPCError(
        _INTERNAL_ERROR,
        session.failure or 'agent connection failed; start a new session',
      )
    if session.switching:
      raise RPCError(_INVALID_REQUEST, 'the session is already switching')
    option = session.choices[model]
    if self._hosted:
      configuration = session.configuration.updated({
        'model': option.model_id,
        'remote_model_api': option.api or None,
      })
      _send_inference(
        session,
        configuration.inference_config(option.model_id, effort, remote=True),
      )
    else:
      configuration = await self._select_local(session, option, effort)
    session.configuration = configuration
    session.model = model
    session.effort = effort

  async def _select_local(
    self, session: _Session, option: models.ModelOption, effort: str
  ) -> rpc_client.WorkerConfiguration:
    """Reconfigure the worker, or replace it for a model on another provider."""
    _check_selection(
      option, effort, session.configuration.values.get('max_tokens')
    )
    configuration = session.configuration.updated({
      'model': option.model_id,
      'model_api': option.api,
      'model_base_url': option.base_url or None,
      'base_url': None,
    })
    inference = configuration.inference_config(
      option.model_id, effort, remote=False
    )
    current = session.configuration.inference_config(
      option.model_id, effort, remote=False
    )
    if (inference.api, inference.base_url) == (current.api, current.base_url):
      _send_inference(session, inference)
    else:
      await self._restart(session, configuration, effort)
    return configuration

  async def _restart(
    self,
    session: _Session,
    configuration: rpc_client.WorkerConfiguration,
    effort: str,
  ) -> None:
    """Replace the worker, which takes provider credentials only at start."""
    if session.turn is not None:
      raise RPCError(
        _INVALID_REQUEST,
        'finish or cancel the running prompt before switching providers',
      )
    session.switching = True
    await session.client.close()
    attached = False
    try:
      try:
        session.client, _ = await self._attach(
          configuration,
          effort,
          session.workspace,
          session.session_id,
          session.events,
          restart=True,
        )
        attached = True
      except RPCError:
        # Restore the previous worker so the session keeps its settings.
        session.client, _ = await self._attach(
          session.configuration,
          session.effort,
          session.workspace,
          session.session_id,
          session.events,
          restart=True,
        )
        attached = True
        raise
    finally:
      session.switching = False
      if not attached:
        session.failed = True

  def _accept_prompt(self, params: JSON) -> tuple[_Session, _Turn, str]:
    """Register the turn before its task runs, so a cancel cannot miss it."""
    session = self._session(params)
    if session.failed:
      raise RPCError(
        _INTERNAL_ERROR,
        session.failure or 'agent connection failed; start a new session',
      )
    text = prompt_text(_array(params.get('prompt')))
    if not text.strip():
      raise RPCError(_INVALID_PARAMS, 'prompt has no text content')
    if session.turn is not None:
      raise RPCError(_INVALID_REQUEST, 'a prompt is already running')
    if session.switching:
      raise RPCError(_INVALID_REQUEST, 'the session is switching providers')
    if session.deferred:
      session.deferred = False
      session.attaching = self._spawn(self._finish_attaching(session))
    session.turn = _Turn()
    return session, session.turn, text

  async def _prompt_request(
    self, request_id: object, session: _Session, turn: _Turn, text: str
  ) -> None:
    try:
      result = await self._run_turn(session, turn, text)
    except RPCError as error:
      await self._write_error(request_id, error.code, error.message)
      return
    except asyncio.CancelledError:
      await self._write_error(request_id, _INTERNAL_ERROR, 'server shut down')
      raise
    finally:
      if session.turn is turn:
        session.turn = None
    await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

  async def _run_turn(self, session: _Session, turn: _Turn, text: str) -> JSON:
    if turn.cancel_requested:
      # Cancelled before the worker saw the prompt.
      return {'stopReason': 'cancelled'}
    if session.attaching is not None and not await _attached(
      session.attaching, turn
    ):
      return {'stopReason': 'cancelled'}
    if session.failed:
      raise RPCError(
        _INTERNAL_ERROR,
        session.failure or 'agent connection failed; start a new session',
      )
    _discard_idle_events(session)
    session.client.enqueue(
      agent_pb2.AgentMessage(
        id=uuid.uuid4().hex,
        chat=agent_pb2.ChatRecord(
          user=agent_pb2.UserMessage(content=text), success=True
        ),
      )
    )
    turn.started = True
    translator = Translator()
    while True:
      event = await session.events.get()
      if event is None:
        raise RPCError(
          _INTERNAL_ERROR,
          session.failure
          or 'agent connection failed; check provider configuration',
        )
      if event.WhichOneof('payload') == 'hitl_request':
        # A request already queued when the cancel or the disconnect landed:
        # nothing can act on the answer, so do not ask the client.
        if not turn.cancel_requested and not session.failed:
          await self._permission(session, turn, event.hitl_request.call)
        continue
      if _DEBUG:
        _debug_event(event)
      updates, outcome = translator.translate(event)
      for update in updates:
        await self._notify(
          'session/update', {'sessionId': session.session_id, 'update': update}
        )
      if outcome is None:
        continue
      if outcome.error is not None:
        raise RPCError(_INTERNAL_ERROR, outcome.error)
      result: JSON = {'stopReason': outcome.stop_reason}
      if outcome.usage is not None:
        result['usage'] = outcome.usage
      return result

  async def _permission(
    self, session: _Session, turn: _Turn, call: agent_pb2.ToolCall
  ) -> None:
    """Ask the client to approve a tool call and forward its answer."""
    self._next_request_id += 1
    request_id = f'mirror-{self._next_request_id}'
    future: asyncio.Future[JSON] = asyncio.get_running_loop().create_future()
    self._pending[request_id] = future
    turn.permission = future
    try:
      await self._write({
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'session/request_permission',
        'params': {
          'sessionId': session.session_id,
          'toolCall': _tool_call_update(
            call.id,
            call.function.name,
            call.function.arguments,
            'pending',
          ),
          'options': list(_PERMISSION_OPTIONS),
        },
      })
      response = await future
    finally:
      turn.permission = None
      self._pending.pop(request_id, None)
    if turn.cancel_requested or session.failed:
      # The interrupt supersedes the approval, or there is no worker to tell.
      return
    approved = False
    result = _object(response.get('result'))
    outcome = _object(result.get('outcome')) if result is not None else None
    if outcome is not None and outcome.get('outcome') == 'selected':
      approved = outcome.get('optionId') == 'allow'
    session.client.send(
      agent_pb2.ClientMessage(
        session_id=session.session_id,
        message_id=uuid.uuid4().hex,
        hitl_response=agent_pb2.HITLResponse(
          call_id=call.id,
          approved=approved,
          source=agent_pb2.APPROVAL_SOURCE_MANUAL,
        ),
      )
    )

  async def _notify(self, method: str, params: JSON) -> None:
    await self._write({'jsonrpc': '2.0', 'method': method, 'params': params})

  async def _write_error(
    self, request_id: object, code: int, message: str
  ) -> None:
    await self._write({
      'jsonrpc': '2.0',
      'id': request_id,
      'error': {'code': code, 'message': message},
    })

  async def _write(self, message: JSON) -> None:
    assert self._output is not None
    data = json.dumps(message, ensure_ascii=False).encode('utf-8') + b'\n'
    async with self._write_lock:
      self._output.write(data)
      self._output.flush()


_DEBUG = bool(os.environ.get('MIRROR_ACP_DEBUG'))


def _debug_event(event: agent_pb2.SessionMessage) -> None:
  """Trace one session message on stderr (set MIRROR_ACP_DEBUG=1)."""
  payload = event.WhichOneof('payload')
  detail = ''
  if payload == 'runtime_delta':
    detail = event.runtime_delta.WhichOneof('payload') or ''
  elif payload == 'agent_message':
    record = event.agent_message
    detail = record.WhichOneof('payload') or ''
    if detail == 'chat':
      kind = record.chat.WhichOneof('message') or ''
      detail += f'/{kind}'
      if kind == 'assistant':
        assistant = record.chat.assistant
        detail += (
          f' content={len(assistant.content)}'
          f' tool_calls={len(assistant.tool_calls)}'
          f' success={record.chat.success}'
        )
    elif detail == 'log':
      detail += f' {record.log.level}: {record.log.text[:160]!r}'
  print(f'mirror acp: event {payload} {detail}', file=sys.stderr)


async def _stdin_reader() -> asyncio.StreamReader:
  reader = asyncio.StreamReader(limit=_MAX_LINE_BYTES)
  protocol = asyncio.StreamReaderProtocol(reader)
  await asyncio.get_running_loop().connect_read_pipe(
    lambda: protocol, sys.stdin.buffer
  )
  return reader


def _consent(workspace: pathlib.Path) -> None:
  """Require the directory's notices to have been accepted interactively."""
  onboarding.ensure_consent(
    workspace, confirm=noninteractive.require_interactive_consent
  )


def run(
  worker_command: Sequence[str],
  worker_config: WorkerConfigFactory,
  *,
  reasoning_effort: str,
) -> int:
  """Serve ACP on stdin/stdout until the client disconnects."""
  return serve_stdio(
    Server(
      worker_config=worker_config,
      transport=_local_transport(worker_command),
      reasoning_effort=reasoning_effort,
      consent=_consent,
    )
  )


def serve_stdio(server: Server) -> int:
  """Run `server` on stdin/stdout until the client disconnects.

  Returns:
    The process exit status.
  """

  async def main() -> None:
    await server.serve(await _stdin_reader(), sys.stdout.buffer)

  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    return 130
  return 0
