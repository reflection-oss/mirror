"""One local agent turn without terminal UI or interactive approvals."""

from __future__ import annotations

import asyncio
import http
import pathlib
import re
import signal
import sys
import threading
import types
import urllib.parse
import uuid
from collections.abc import Mapping, Sequence

from mirror.protocol import agent_pb2

from mirror.cli import local_rpc_client, models, rpc_client

_MAX_PROMPT_BYTES = 8 * 1024 * 1024
_STDIN_HINT_SECONDS = 1.0


def read_prompt(argument: str | None) -> str:
  """Resolve an argument and stdin using the `codex exec` input convention."""
  terminal = sys.stdin.isatty()
  if argument is None and terminal:
    raise ValueError(
      'pass a prompt argument or pipe UTF-8 input to mirror exec'
    )
  prompt = argument or ''
  if argument == '-' or not terminal:
    # The read waits for EOF; with a prompt argument, say why if that takes a while.
    hint = threading.Timer(_STDIN_HINT_SECONDS, _print_stdin_hint)
    if argument not in (None, '-'):
      hint.start()
    try:
      raw = sys.stdin.buffer.read(_MAX_PROMPT_BYTES + 1)
      if len(raw) > _MAX_PROMPT_BYTES:
        raise ValueError('prompt exceeds 8 MiB')
      content = raw.decode('utf-8')
    except (OSError, UnicodeError):
      raise ValueError('could not read UTF-8 prompt from stdin') from None
    finally:
      hint.cancel()
    if argument is None or argument == '-':
      prompt = content
    elif content.strip():
      newline = '' if content.endswith('\n') else '\n'
      prompt = f'{argument}\n\n<stdin>\n{content}{newline}</stdin>'
  if len(prompt.encode('utf-8')) > _MAX_PROMPT_BYTES:
    raise ValueError('prompt exceeds 8 MiB')
  if not prompt.strip():
    raise ValueError('prompt must not be empty')
  return prompt


def _print_stdin_hint() -> None:
  print(
    'mirror: waiting for input from another program; anything it sends is '
    "added to your prompt. If you aren't sending anything, stop with Ctrl+C "
    'and add "< /dev/null" to the end of the command, e.g. '
    'mirror exec "..." < /dev/null',
    file=sys.stderr,
    flush=True,
  )


def require_interactive_consent(notice: str) -> bool:
  """Fail instead of granting directory trust or displaying an interactive notice."""
  del notice
  raise SystemExit(
    'mirror: open Mirror interactively in this directory to review and accept '
    'the usage and folder notices before using exec'
  )


def _endpoint(base_url: str) -> str:
  """scheme://host[:port] only: user info, path or query in a URL can hold credentials."""
  parts = urllib.parse.urlsplit(base_url)
  try:
    port = f':{parts.port}' if parts.port else ''
  except ValueError:
    port = ''
  return f'{parts.scheme}://{parts.hostname}{port}' if parts.hostname else ''


def _failure_message(error: str, inference: agent_pb2.InferenceConfig) -> str:
  """Explain a failed model request from its HTTP status and what was requested.

  The provider's own text after the status can echo request data, so it is never shown.
  """
  if inference.api == 'llama_cpp':
    return 'llama.cpp request failed; check that the model is loaded'
  match = re.match(r'model request failed: (\d{3})\b', error)
  if match is None:
    return 'model requests failed; see the session log for details'
  code = int(match[1])
  try:
    status = f'{code} {http.HTTPStatus(code).phrase}'
  except ValueError:
    status = str(code)
  endpoint = _endpoint(inference.base_url)
  target = f'model "{inference.model}"' + (
    f' at {endpoint}' if endpoint else ''
  )
  # exec takes credentials from the environment only; it rejects --api-key.
  key = (
    'your Codex login'
    if inference.api == 'openai-codex-responses'
    else models.key_env(inference.api)
  )
  if code == 401:
    return (
      f'the provider rejected the API key ({status}) for {target}. Check {key}.'
    )
  if code == 403:
    return f'the provider denied access ({status}) for {target}. Check that {key} can use this model.'
  if code == 404:
    return f'the provider could not find {target} ({status}). Check --model and --base-url.'
  if code == 429:
    return f'the provider is rate limiting requests ({status}) for {target}. Wait and try again.'
  if 500 <= code < 600:
    return f'the provider had a server error ({status}) for {target}. Try again later.'
  return f'model requests failed: {status}'


async def _answer(
  events: asyncio.Queue[agent_pb2.SessionMessage | None],
  inference: agent_pb2.InferenceConfig,
) -> str:
  """Wait for turn completion, retaining only the last committed answer."""
  answer = ''
  last_error = ''
  while True:
    event = await events.get()
    if event is None:
      raise RuntimeError(
        'agent connection failed; check provider configuration'
      )
    payload = event.WhichOneof('payload')
    if payload == 'hitl_request':
      raise RuntimeError(
        'tool approval required; rerun interactively to review the action. '
        'exec never grants an approval'
      )
    if payload == 'input_nack':
      raise RuntimeError('agent rejected the request; check run configuration')
    if payload == 'agent_message':
      record = event.agent_message
      if record.WhichOneof('payload') == 'log' and record.log.level == 'error':
        last_error = record.log.text
      if (
        record.WhichOneof('payload') == 'log' and record.log.level == 'warning'
      ):
        print(f'mirror: warning: {record.log.text}', file=sys.stderr)
      if record.chat.WhichOneof('message') == 'assistant':
        if record.HasField('inference'):
          last_error = ''  # A later model completion supersedes earlier errors.
        assistant = record.chat.assistant
        # Limit notices are synthetic records, not model completions.
        answer = (
          assistant.content or assistant.refusal
          if record.HasField('inference')
          and record.chat.success
          and not record.chat.truncated
          and not assistant.tool_calls
          else ''
        )
    if payload != 'runtime_delta':
      continue
    delta = event.runtime_delta.WhichOneof('payload')
    if delta == 'turn_error':
      # The error text may contain request bodies or credentials; a turn that
      # failed on a model error is explained from that error's status alone.
      if last_error:
        raise RuntimeError(_failure_message(last_error, inference))
      raise RuntimeError(
        'agent turn failed; check provider configuration and limits'
      )
    if delta == 'turn_interrupted':
      raise RuntimeError('agent turn was interrupted')
    if delta == 'turn_completed':
      if not answer.strip():
        if last_error:
          raise RuntimeError(_failure_message(last_error, inference))
        raise RuntimeError(
          'agent ended without a final answer; check run limits'
        )
      return answer


async def _run(
  worker_command: Sequence[str],
  configuration: rpc_client.WorkerConfiguration,
  prompt: str,
  reasoning_effort: str,
) -> str:
  events: asyncio.Queue[agent_pb2.SessionMessage | None] = asyncio.Queue()
  workspace = pathlib.Path.cwd()
  record = await local_rpc_client.create_workspace(
    worker_command, configuration.values, workspace
  )
  conversation_id = record.session_id
  configuration = configuration.updated({
    'conversation_id': conversation_id,
    'workspace_id': record.id,
  })
  print(f'mirror: session {conversation_id}', file=sys.stderr)

  async def on_error(error: str) -> None:
    del error  # Raw transport errors can include provider payloads.
    events.put_nowait(None)

  log_dir = pathlib.Path.home() / '.mirror' / 'logs' / conversation_id
  transport = local_rpc_client.Transport(
    worker_command,
    configuration.values,
    workspace=workspace,
    log_dir=log_dir,
    on_message=events.put_nowait,
  )
  inference = configuration.inference_config(
    str(configuration.values['model']), reasoning_effort, remote=False
  )
  client = rpc_client.Client(
    transport,
    conversation_id,
    on_message=events.put_nowait,
    on_error=on_error,
    inference_config=inference,
    web_provider=configuration.web_provider,
  )
  try:
    try:
      await asyncio.wait_for(client.start(replay=False), timeout=30)
    except (OSError, RuntimeError, ValueError, TimeoutError):
      raise RuntimeError(
        'could not attach agent; check numeric limits, model authentication and '
        'worker configuration'
      ) from None
    client.enqueue(
      agent_pb2.AgentMessage(
        id=uuid.uuid4().hex,
        chat=agent_pb2.ChatRecord(
          user=agent_pb2.UserMessage(content=prompt), success=True
        ),
      )
    )
    return await _answer(events, inference)
  finally:
    await client.close()
    await transport.close_workspace()


def run(
  worker_command: Sequence[str],
  worker_config: Mapping[str, object],
  argument: str | None,
  *,
  reasoning_effort: str,
) -> int:
  """Print one final answer; close the worker on errors and termination signals."""
  interrupted = 0

  def interrupt(signum: int, frame: types.FrameType | None) -> None:
    nonlocal interrupted
    del frame
    if not interrupted:
      interrupted = 128 + signum
      raise KeyboardInterrupt

  previous_int = signal.signal(signal.SIGINT, interrupt)
  previous_term = signal.signal(signal.SIGTERM, interrupt)
  try:
    prompt = read_prompt(argument)
    configuration = rpc_client.WorkerConfiguration(
      worker_config, cwd=pathlib.Path.cwd()
    )
    answer = asyncio.run(
      _run(worker_command, configuration, prompt, reasoning_effort)
    )
    print(answer, end='' if answer.endswith('\n') else '\n', flush=True)
    return 0
  except KeyboardInterrupt:
    print('mirror: interrupted', file=sys.stderr)
    return interrupted or 130
  except (ValueError, RuntimeError, OSError) as error:
    print(f'mirror: {error}', file=sys.stderr)
    return 1
  finally:
    signal.signal(signal.SIGINT, previous_int)
    signal.signal(signal.SIGTERM, previous_term)
