"""One local agent turn without terminal UI or interactive approvals."""

from __future__ import annotations

import asyncio
import datetime
import pathlib
import signal
import sys
import types
import uuid
from collections.abc import Mapping, Sequence

from mirror.protocol import agent_pb2

from mirror.cli import cli_store, local_rpc_client, rpc_client

_MAX_PROMPT_BYTES = 8 * 1024 * 1024


def read_prompt(argument: str | None) -> str:
  """Resolve an argument and stdin using the `codex exec` input convention."""
  terminal = sys.stdin.isatty()
  if argument is None and terminal:
    raise ValueError(
      'pass a prompt argument or pipe UTF-8 input to mirror exec'
    )
  prompt = argument or ''
  if argument == '-' or not terminal:
    try:
      raw = sys.stdin.buffer.read(_MAX_PROMPT_BYTES + 1)
      if len(raw) > _MAX_PROMPT_BYTES:
        raise ValueError('prompt exceeds 8 MiB')
      content = raw.decode('utf-8')
    except (OSError, UnicodeError):
      raise ValueError('could not read UTF-8 prompt from stdin') from None
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


def require_interactive_consent(notice: str) -> bool:
  """Fail instead of granting directory trust or displaying an interactive notice."""
  del notice
  raise SystemExit(
    'mirror: open Mirror interactively in this directory to review and accept '
    'the usage and folder notices before using exec'
  )


async def _answer(
  events: asyncio.Queue[agent_pb2.SessionMessage | None],
) -> str:
  """Wait for turn completion, retaining only the last committed answer."""
  answer = ''
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
      if record.chat.WhichOneof('message') == 'assistant':
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
      # Provider errors may contain request bodies or credentials.
      raise RuntimeError(
        'agent turn failed; check provider configuration and limits'
      )
    if delta == 'turn_interrupted':
      raise RuntimeError('agent turn was interrupted')
    if delta == 'turn_completed':
      if not answer.strip():
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
  conversation_id = str(configuration.values['conversation_id'])
  workspace = pathlib.Path.cwd()

  def on_message(event: agent_pb2.SessionMessage) -> None:
    events.put_nowait(event)
    message = event.agent_message
    if (
      message.WhichOneof('payload') != 'chat'
      or message.chat.WhichOneof('message') != 'user'
      or message.chat.HasField('display')
      or not message.chat.user.content.strip()
    ):
      return
    with cli_store.Store() as store:
      store.record_user_message(
        conversation_id,
        entry_id=message.id,
        sequence=message.sequence,
        timestamp=message.timestamp.ToDatetime(tzinfo=datetime.UTC)
        if message.HasField('timestamp')
        else datetime.datetime.now(datetime.UTC),
        content=message.chat.user.content,
        cwd=str(workspace),
        model=str(configuration.values['model']),
      )

  async def on_error(error: str) -> None:
    del error  # Raw transport errors can include provider payloads.
    events.put_nowait(None)

  transport = local_rpc_client.Transport(
    worker_command,
    configuration.values,
    workspace=workspace,
    log_dir=pathlib.Path.home() / '.mirror' / 'logs' / conversation_id,
    on_message=on_message,
  )
  client = rpc_client.Client(
    transport,
    conversation_id,
    on_message=on_message,
    on_error=on_error,
    inference_config=configuration.inference_config(
      str(configuration.values['model']), reasoning_effort, remote=False
    ),
  )
  try:
    await asyncio.wait_for(client.start(replay=False), timeout=30)
  except (OSError, RuntimeError):
    raise RuntimeError(
      'could not start agent; check model authentication and worker configuration'
    ) from None
  try:
    client.enqueue(
      agent_pb2.AgentMessage(
        id=uuid.uuid4().hex,
        chat=agent_pb2.ChatRecord(
          user=agent_pb2.UserMessage(content=prompt), success=True
        ),
      )
    )
    return await _answer(events)
  finally:
    await client.close()


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
    print(
      f'mirror: session {worker_config["conversation_id"]}', file=sys.stderr
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
