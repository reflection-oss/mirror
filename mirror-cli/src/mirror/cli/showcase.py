"""An isolated, synthetic AgentRPC worker for `mirror --showcase`.

Starts with a rendered gallery; send any message to play its events live.
Esc stops playback. /new restores the gallery. No model, host tool, external service,
or real agent store is used. CLI metadata and the workspace are temporary.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import signal
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import override

import grpc.aio
from google.protobuf import empty_pb2
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
)

from mirror.cli import cli, rpc_client


def messages() -> list[agent_pb2.AgentMessage]:
  """Build a deterministic gallery of durable rendering events."""
  entries: list[agent_pb2.AgentMessage] = []

  def chat(record: agent_pb2.ChatRecord) -> None:
    entries.append(agent_pb2.AgentMessage(chat=record))

  def user(content: str) -> None:
    chat(
      agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content=content), success=True
      )
    )

  def answer(content: str, reasoning: str = '') -> None:
    chat(
      agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(
          content=content, reasoning_content=reasoning
        ),
        usage=agent_pb2.Usage(prompt_tokens=1000, completion_tokens=120),
        success=True,
      )
    )

  def tool(
    name: str,
    arguments: Mapping[str, object],
    output: str,
    *,
    display: str | None = None,
    success: bool = True,
  ) -> None:
    call_id = f'call-{len(entries)}'
    chat(
      agent_pb2.ChatRecord(
        assistant=agent_pb2.AssistantMessage(
          tool_calls=[
            agent_pb2.ToolCall(
              id=call_id,
              type='function',
              function=agent_pb2.FunctionCall(
                name=name, arguments=json.dumps(arguments)
              ),
            )
          ]
        ),
        usage=agent_pb2.Usage(prompt_tokens=1000, completion_tokens=120),
        success=True,
      )
    )
    result = agent_pb2.ChatRecord(
      tool=agent_pb2.ToolMessage(
        name=name,
        tool_call_id=call_id,
        content=output,
      ),
      success=success,
    )
    if display is not None:
      result.display = display
    chat(result)

  user(
    'Visual showcase: explore this folder. All events below are fabricated; no tools run and no files change.'
  )
  answer(
    'I’ll inspect the workspace, then demonstrate file changes and validation.',
    '**Plan:** inspect the files, make a small change, and check it.\n\nThis is synthetic reasoning for detail mode.',
  )
  tool(
    'read',
    {'file_path': '.agents/skills/transcript-review/SKILL.md'},
    '# Transcript review\n\nCheck gutters, spacing, and activity summaries.',
  )
  for directory in ('.', 'mirror-cli', 'mirror-cli/src', 'mirror-cli/tests'):
    tool('ls', {'path': directory}, 'README.md\nsrc/\ntests/')
  for index in range(8):
    tool(
      'read',
      {'file_path': f'mirror-cli/src/example_{index}.py'},
      'from __future__ import annotations\n\ndef example():\n  return True\n'
      * 8,
    )
  tool(
    'grep',
    {'pattern': 'render|spacing', 'path': 'mirror-cli/src'},
    'Found 36 lines\nrendering.py:1: render\nstream.py:2: spacing',
  )
  tool(
    'glob',
    {'pattern': '**/*.proto', 'path': 'protocols'},
    'agent.proto\nmirror.proto',
  )
  tool(
    'read',
    {'file_path': 'missing.py'},
    'File not found: missing.py',
    success=False,
  )
  tool(
    'read',
    {'file_path': 'mirror-cli/README.md'},
    '# Mirror\n\nA coding-agent harness.',
  )
  answer(
    'This workspace contains the following components.\n\n'
    '| Folder | Purpose |\n| --- | --- |\n'
    '| `mirror-cli/` | Python terminal UI, approvals, extensions, and transcript rendering. |\n'
    '| `mirror-agent/` | Rust worker, tools, background tasks, and local persistence. |\n'
    '| `protocols/` | Shared protobuf contracts. |\n\n'
    'Useful starting points:\n\n- `mirror-cli/README.md:1`\n- `mirror-cli/src/mirror/cli/rendering.py:1`'
  )

  user(
    'Demonstrate consecutive extension calls with long arguments and results.'
  )
  tool(
    'todo',
    {
      'action': 'add',
      'text': 'Create MCP built-in extension tier with ordinary lifecycle, commands, and tool registration.',
    },
    'Added todo #1: Create MCP built-in extension tier with ordinary lifecycle, commands, and tool registration.',
  )
  tool(
    'todo',
    {
      'action': 'add',
      'text': 'Restack tool API and move generic routing/approvals/history API into that tier.',
    },
    'Added todo #2: Restack tool API and move generic routing/approvals/history API into that tier.',
  )
  tool(
    'todo',
    {
      'action': 'add',
      'text': 'Reduce bug-report tier to consumer changes; validate and push replacement stack together.',
    },
    'Added todo #3: Reduce bug-report tier to consumer changes; validate and push replacement stack together.',
  )

  tool('todo', {'action': 'toggle', 'id': 1}, 'Todo #1 marked complete.')
  tool(
    'todo',
    {'action': 'list'},
    '1. [done] Create MCP built-in extension tier.\n'
    '2. [pending] Restack tool API.\n'
    '3. [pending] Reduce bug-report tier.',
  )

  user('Demonstrate writes, edits, shell output, and failures.')
  tool(
    'write',
    {'file_path': 'showcase/example.py', 'content': 'print("hello")\n'},
    'Wrote file.',
    display='--- showcase/example.py\n+++ showcase/example.py\n@@ -0,0 +1 @@\n+print("hello")\n',
  )
  tool(
    'edit',
    {
      'file_path': 'showcase/example.py',
      'old_string': 'hello',
      'new_string': 'hello, Mirror',
    },
    'Replaced text.',
    display='--- showcase/example.py\n+++ showcase/example.py\n@@ -1 +1 @@\n-print("hello")\n+print("hello, Mirror")\n',
  )
  tool(
    'edit',
    {
      'file_path': 'showcase/example.py',
      'old_string': 'missing',
      'new_string': 'replacement',
    },
    'Expected one matching block; found zero. No file was changed.',
    success=False,
  )
  tool(
    'bash',
    {'command': 'python showcase/example.py'},
    'hello, Mirror\n[Completed in 0.1s]',
    display='python showcase/example.py',
  )
  command = (
    "jj status; jj log --no-graph -r '@::' "
    '-T \'change_id.shortest(8) ++ " " ++ bookmarks ++ "\\n"\''
  )
  tool(
    'bash',
    {'command': command},
    'Working copy changes:\n'
    'M mirror-cli/src/mirror/cli/cli.py\n'
    'M mirror-cli/src/mirror/cli/components/completion.py\n'
    'M mirror-cli/src/mirror/cli/rendering.py\n'
    '[Completed in 0.2s]',
    display=command,
  )
  command = (
    'for file in mirror-cli/src/mirror/cli/*.py; do\n'
    '  printf "Checking %s\\n" "$file"\n'
    'done'
  )
  tool(
    'bash',
    {'command': command},
    'Checking mirror-cli/src/mirror/cli/cli.py\n'
    'Checking mirror-cli/src/mirror/cli/rendering.py\n'
    'Checking mirror-cli/src/mirror/cli/showcase.py\n'
    '[Completed in 0.0s]',
    display=command,
  )
  tool(
    'bash',
    {'command': 'rm -- showcase/example.py'},
    '[Completed in 0.0s]',
    display='rm -- showcase/example.py',
  )
  tool(
    'bash',
    {'command': 'printf "same command\\n"'},
    'same command\n[Completed in 0.0s]',
    display='printf "same command\\n"',
  )
  tool(
    'bash',
    {'command': 'pytest -q'},
    'Exit code: 1\nFAILED test_example.py::test_spacing\n'
    'Expected one blank line, got two.\n[Completed in 1.4s]',
    display='pytest -q',
  )
  tool(
    'bash',
    {'command': 'sleep 60'},
    'Command timed out after 60000ms\n[Completed in 60.0s]',
    display='sleep 60',
    success=False,
  )
  tool(
    'bash',
    {'command': 'pytest -q'},
    '24 passed in 1.8s\n[Completed in 2.1s]',
    display='pytest -q',
  )
  answer(
    'The **synthetic** checks now pass.\n\n```python\ndef greet():\n    return "hello, Mirror"\n```\n\n> No commands above were executed.'
  )

  user('Show local and web exploration together, including failures.')
  tool('read', {'file_path': 'README.md'}, '# Transcript layout')
  tool(
    'web_search',
    {'query': 'transcript rendering'},
    'Transcript layout — https://example.com/layout',
  )
  tool(
    'web_browse',
    {'url': 'https://example.com/layout'},
    'Group related exploration and keep tool previews compact.',
  )
  tool(
    'web_search',
    {'query': 'transcript rendering'},
    'Web search failed: service unavailable.',
    success=False,
  )
  tool(
    'web_browse',
    {'url': 'https://example.com/layout'},
    'Web browse failed: service unavailable.',
    success=False,
  )

  user('Show host tool results and background task completion.')
  tool(
    'extension__showcase__search_docs',
    {'query': 'transcript rendering'},
    'Found 2 documents\nTranscript layout — https://example.com/layout\nTool grouping — https://example.com/tools',
  )
  tool(
    'extension__showcase__get_ticket',
    {'ticket': 'DEMO-42'},
    '{"title":"Polish transcript spacing","status":"in progress"}',
  )
  tool(
    'extension__showcase__get_ticket',
    {'ticket': 'DEMO-404'},
    'Permission denied. Ask the workspace administrator for access.',
    success=False,
  )
  tool(
    'subagent',
    {'prompt': 'Review transcript layout.'},
    'Subagent task showcase-review started. Its final answer will be delivered automatically.',
  )
  entries.append(
    agent_pb2.AgentMessage(
      task_result=agent_pb2.TaskResultRecord(
        task_id='showcase-review',
        tool_name='subagent',
        success=True,
        content='## Review\n\n- Gutters align.\n- Tool output remains inspectable.\n- No files changed.',
      )
    )
  )
  entries.append(
    agent_pb2.AgentMessage(
      task_result=agent_pb2.TaskResultRecord(
        task_id='showcase-check',
        tool_name='bash',
        success=False,
        content='Exit code: 1\nThe synthetic background check failed.\n'
        '[Completed in 0.9s]',
        display='pytest -q',
      )
    )
  )
  answer(
    'Host results and background tasks use the same transcript, without executing any integrations.'
  )

  user('Show context compaction and a longer final response.')
  entries.append(
    agent_pb2.AgentMessage(
      context_window=agent_pb2.ContextWindowRecord(
        compaction=agent_pb2.Compaction(
          summary='Synthetic showcase: explored the workspace, illustrated changes, and displayed mock tool results.',
          usage=agent_pb2.Usage(prompt_tokens=48000, completion_tokens=1600),
        ),
      )
    )
  )
  tool(
    'read',
    {'file_path': 'mirror-cli/src/mirror/cli/stream.py'},
    'class StreamPane:\n  ...',
  )
  tool(
    'grep',
    {'pattern': 'pane_height', 'path': 'mirror-cli/src'},
    'stream.py:312: def pane_height(self):',
  )
  answer(
    '## Showcase complete\n\n'
    'This conversation exercises **exploration**, *reasoning*, tables, lists, code, writes, edits, shell output, errors, host tools, background results, and compaction.\n\n'
    '1. Use `/view detail` to inspect individual calls and reasoning.\n'
    '2. Use `/view compact` to inspect grouped activity.\n'
    '3. Resize the terminal, then press Ctrl+L to redraw.\n'
    '4. Send any message to play the gallery live; Esc stops playback.\n\n'
    '**Everything in this conversation is fabricated. No files changed and no external tools ran.**'
  )
  return entries


async def events(
  entries: Sequence[agent_pb2.AgentMessage],
  *,
  delay: float = 0.2,
) -> AsyncIterator[agent_pb2.SessionMessage]:
  """Play durable records through the same transient events as a live worker."""
  for source in entries:
    record = agent_pb2.AgentMessage()
    record.CopyFrom(source)
    kind = record.WhichOneof('payload')
    assistant = (
      kind == 'chat' and record.chat.WhichOneof('message') == 'assistant'
    )
    if kind == 'context_window' and record.context_window.HasField(
      'compaction'
    ):
      yield agent_pb2.SessionMessage(
        runtime_delta=agent_pb2.RuntimeDelta(
          compaction_started=empty_pb2.Empty()
        )
      )
      await asyncio.sleep(delay * 5)
    if assistant:
      message = record.chat.assistant
      for reasoning, text in (
        (True, message.reasoning_content),
        (False, message.content),
      ):
        for start in range(0, len(text), 32):
          chunk = text[start : start + 32]
          delta = (
            agent_pb2.RuntimeDelta(
              stream_reasoning=agent_pb2.StreamReasoning(reasoning=chunk)
            )
            if reasoning
            else agent_pb2.RuntimeDelta(
              stream_content=agent_pb2.StreamContent(content=chunk)
            )
          )
          yield agent_pb2.SessionMessage(runtime_delta=delta)
          await asyncio.sleep(delay / 4)
      for index, call in enumerate(message.tool_calls):
        yield agent_pb2.SessionMessage(
          runtime_delta=agent_pb2.RuntimeDelta(
            stream_tool_call=agent_pb2.StreamToolCall(
              index=index,
              name=call.function.name,
              arguments=call.function.arguments,
            )
          )
        )
        await asyncio.sleep(delay)
    if kind == 'task_result':
      task = record.task_result
      yield _task_event(task, agent_pb2.TASK_STATUS_RUNNING)
      await asyncio.sleep(delay * 3)
      yield _task_event(
        task,
        agent_pb2.TASK_STATUS_SUCCEEDED
        if task.success
        else agent_pb2.TASK_STATUS_FAILED,
      )
    await asyncio.sleep(delay)
    yield agent_pb2.SessionMessage(agent_message=record, stream_final=assistant)
  yield agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(turn_completed=empty_pb2.Empty())
  )


def _task_event(
  task: agent_pb2.TaskResultRecord, status: agent_pb2.TaskStatus
) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      task_update=agent_pb2.TaskUpdate(
        id=task.task_id,
        tool_name=task.tool_name,
        status=status,
        detached=True,
        display='Synthetic background task',
      )
    )
  )


class ShowcaseAgent(agent_pb2_grpc.AgentRPCServicer):
  """In-memory replay and cancellable playback; never invokes real tools."""

  def __init__(self, *, delay: float = 0.2) -> None:
    self._history: list[agent_pb2.AgentMessage] = []
    self._delay = delay
    for entry in messages():
      self._record(entry)

  def _record(self, entry: agent_pb2.AgentMessage) -> None:
    entry.sequence = len(self._history) + 1
    entry.id = entry.id or f'showcase-{entry.sequence}'
    entry.timestamp.GetCurrentTime()
    self._history.append(entry)

  @override
  async def GetMessages(
    self,
    request: agent_pb2.GetMessagesRequest,
    context: grpc.aio.ServicerContext[
      agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
    ],
  ) -> agent_pb2.GetMessagesResponse:
    del context
    return agent_pb2.GetMessagesResponse(
      messages=self._history[request.offset : request.offset + request.limit],
      total=len(self._history),
    )

  @override
  async def Session(
    self,
    request_iterator: AsyncIterator[agent_pb2.ClientMessage],
    context: grpc.aio.ServicerContext[
      agent_pb2.ClientMessage, agent_pb2.SessionMessage
    ],
  ) -> AsyncIterator[agent_pb2.SessionMessage]:
    del context
    responses: asyncio.Queue[agent_pb2.SessionMessage | None] = asyncio.Queue()
    playback: asyncio.Task[None] | None = None

    async def play(request: agent_pb2.ClientMessage) -> None:
      entry = agent_pb2.AgentMessage()
      entry.CopyFrom(request.message)
      self._record(entry)
      await responses.put(agent_pb2.SessionMessage(agent_message=entry))
      async for event in events(messages(), delay=self._delay):
        if event.HasField('agent_message'):
          self._record(event.agent_message)
        await responses.put(event)

    async def receive() -> None:
      nonlocal playback
      async for request in request_iterator:
        kind = request.WhichOneof('payload')
        if kind == 'message':
          if playback is not None and not playback.done():
            await responses.put(
              agent_pb2.SessionMessage(
                input_nack=agent_pb2.InputNack(
                  message_id=request.message_id,
                  reason=agent_pb2.INPUT_NACK_REASON_INVALID_REQUEST,
                  error='Showcase is playing; press Esc to stop before restarting.',
                )
              )
            )
            continue
          await responses.put(
            agent_pb2.SessionMessage(
              input_ack=agent_pb2.InputAck(message_id=request.message_id)
            )
          )
          playback = asyncio.create_task(play(request))
        elif kind == 'interrupt' and playback is not None:
          playback.cancel()
          await asyncio.gather(playback, return_exceptions=True)
          await responses.put(
            agent_pb2.SessionMessage(
              runtime_delta=agent_pb2.RuntimeDelta(
                stream_aborted=empty_pb2.Empty()
              )
            )
          )
          await responses.put(
            agent_pb2.SessionMessage(
              runtime_delta=agent_pb2.RuntimeDelta(
                turn_interrupted=empty_pb2.Empty()
              )
            )
          )
          await responses.put(
            agent_pb2.SessionMessage(
              runtime_delta=agent_pb2.RuntimeDelta(
                turn_completed=empty_pb2.Empty()
              )
            )
          )
        elif kind in {'set_inference_config', 'set_approval_mode'}:
          await responses.put(
            agent_pb2.SessionMessage(
              input_ack=agent_pb2.InputAck(message_id=request.message_id)
            )
          )
      await responses.put(None)

    reader = asyncio.create_task(receive())
    try:
      while (response := await responses.get()) is not None:
        yield response
    finally:
      reader.cancel()
      tasks = [reader]
      if playback is not None:
        playback.cancel()
        tasks.append(playback)
      await asyncio.gather(*tasks, return_exceptions=True)


async def serve() -> None:
  """Expose the synthetic worker over the usual loopback AgentRPC transport."""
  server = grpc.aio.server(options=rpc_client.GRPC_CHANNEL_OPTIONS)  # pyright: ignore[reportUnknownMemberType] - gRPC factory has unspecialized handler types.
  agent_pb2_grpc.add_AgentRPCServicer_to_server(ShowcaseAgent(), server)  # pyright: ignore[reportUnknownMemberType] - protoc generates an untyped registration function.
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  print(f'MIRROR_RPC_ADDRESS=127.0.0.1:{port}', flush=True)
  stopped = asyncio.Event()
  asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
  try:
    await stopped.wait()
  finally:
    await server.stop(0)


def run(
  *,
  scrollback_limit: int = 1000,
  flush_scrollback: bool = False,
) -> None:
  """Run the real TUI against a synthetic worker in a disposable workspace."""
  with (
    tempfile.TemporaryDirectory(prefix='mirror-showcase-') as directory,
    contextlib.chdir(directory),
  ):
    root = pathlib.Path.cwd()
    asyncio.run(
      cli.run_tui(
        [sys.executable, '-m', 'mirror.cli.showcase'],
        {
          'conversation_id': uuid.uuid4().hex,
          'model': 'Showcase',
          'create_session': True,
        },
        database=root / 'cli.db',
        scrollback_limit=scrollback_limit,
        flush_scrollback=flush_scrollback,
      )
    )


if __name__ == '__main__':
  asyncio.run(serve())
