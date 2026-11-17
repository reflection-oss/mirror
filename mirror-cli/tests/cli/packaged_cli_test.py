"""Opt-in smoke test of the installed CLI and bundled Rust worker."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import hashlib
import http.server
import json
import os
import pathlib
import platform
import pty
import queue
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Protocol, cast, override

import grpc.aio
import pyte
import pytest
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
  daemon_pb2,
  daemon_pb2_grpc,
)

from mirror import config, distribution
from mirror.cli import (
  daemon_client,
  extension,
  local_rpc_client,
  user_extensions,
)

_PROMPT = 'Say hello from the packaged CLI'
# ELF `e_machine` values for the architectures Mirror wheels target.
_ELF_MACHINES = {'arm64': 183, 'aarch64': 183, 'x86_64': 62}
_ANSWER = 'Packaged PTY assistant response'


class _ModelServer(http.server.HTTPServer):
  def __init__(self) -> None:
    super().__init__(('127.0.0.1', 0), _ModelHandler)
    self.requests: queue.Queue[tuple[str, bytes]] = queue.Queue()
    self.release_response: queue.Queue[None] = queue.Queue()
    self.response_index = 0


def _model_response(path: str, index: int) -> bytes:
  """Complete two tool-calling responses and a final answer with reported usage."""
  tokens = (11, 7, 3)[min(index, 2)]
  answer = _ANSWER if index == 2 else f'{_ANSWER} {index}'
  call = {'name': 'ls', 'arguments': '{"path":"."}'}
  call_id = f'call-{index}'
  if path == '/responses':
    output = (
      [{'type': 'function_call', 'call_id': call_id, **call}]
      if index < 2
      else [
        {
          'type': 'message',
          'content': [{'type': 'output_text', 'text': answer}],
        }
      ]
    )
    chunk = {
      'type': 'response.completed',
      'response': {
        'status': 'completed',
        'output': output,
        'usage': {'input_tokens': 100, 'output_tokens': tokens},
      },
    }
  else:
    assert path == '/chat/completions'
    delta = (
      {
        'tool_calls': [
          {'index': 0, 'id': call_id, 'type': 'function', 'function': call}
        ]
      }
      if index < 2
      else {'content': answer}
    )
    chunk = {
      'choices': [
        {'delta': delta, 'finish_reason': 'tool_calls' if index < 2 else 'stop'}
      ],
      'usage': {'prompt_tokens': 100, 'completion_tokens': tokens},
    }
  return f'data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n'.encode()


class _ModelHandler(http.server.BaseHTTPRequestHandler):
  def do_GET(self) -> None:
    """Advertise the actual deployment limit for worker discovery."""
    assert self.path == '/models'
    body = json.dumps({
      'data': [{'id': 'packaged-smoke-model', 'max_model_len': 8192}]
    }).encode()
    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_POST(self) -> None:
    """Pause subsequent requests so the test can observe completed-response usage."""
    self.connection.settimeout(5)
    body = self.rfile.read(int(self.headers['Content-Length']))
    assert isinstance(self.server, _ModelServer)
    self.server.requests.put((self.path, body))
    index = self.server.response_index
    self.server.response_index += 1
    if index:
      self.server.release_response.get(timeout=30)
    response = _model_response(self.path, index)
    self.send_response(200)
    self.send_header('Content-Type', 'text/event-stream')
    self.send_header('Content-Length', str(len(response)))
    self.end_headers()
    self.wfile.write(response)

  @override
  def log_message(self, format: str, *args: object) -> None:
    del format, args


class _Terminal(pyte.Screen):
  synchronized = False

  @override
  def set_mode(self, *modes: int, private: bool = False) -> None:
    if private and 2026 in modes:
      self.synchronized = True
    super().set_mode(*modes, private=private)

  @override
  def reset_mode(self, *modes: int, private: bool = False) -> None:
    super().reset_mode(*modes, private=private)
    if private and 2026 in modes:
      self.synchronized = False

  def __init__(self, master: int) -> None:
    super().__init__(120, 40)
    self.master = master

  @override
  def write_process_input(self, data: str) -> None:
    # pyte answers cursor-position queries, including queries split across reads.
    os.write(self.master, data.encode())


def _wait_for_config_key(path: pathlib.Path, key: str) -> dict[str, object]:
  """Poll the user config until a background write records `key`."""
  deadline = time.monotonic() + 30
  while time.monotonic() < deadline:
    settings = json.loads(path.read_text())
    if key in settings:
      return settings
    time.sleep(0.1)
  raise AssertionError(f'{key!r} was never written to {path}')


def _wait_for_screen(
  child: subprocess.Popen[bytes],
  screen: _Terminal,
  stream: pyte.ByteStream,
  text: str | None,
  *,
  exact_row: bool = False,
) -> None:
  """Wait for a completed frame containing text, or clean exit for None."""
  deadline = time.monotonic() + 30
  while time.monotonic() < deadline:
    if text is not None and not screen.synchronized:
      visible = (
        any(row.strip() == text for row in screen.display)
        if exact_row
        else text in '\n'.join(screen.display)
      )
      if visible:
        return
    if select.select([screen.master], [], [], 0.1)[0]:
      try:
        output = os.read(screen.master, 65536)
      except OSError as error:
        if error.errno != errno.EIO:
          raise
        break
      if not output:
        break
      stream.feed(output)
    elif child.poll() is not None:
      break
  if text is None:
    assert child.wait(timeout=5) == 0, '\n'.join(screen.display)
    return
  pytest.fail(
    f'CLI did not render {text!r} (exit={child.poll()}):\n'
    + '\n'.join(screen.display)
  )


def _write_terminal(
  screen: _Terminal, stream: pyte.ByteStream, data: bytes
) -> None:
  """Drain output during large pastes so the test cannot deadlock the PTY."""
  os.set_blocking(screen.master, False)
  try:
    while data:
      readable, writable, _ = select.select(
        [screen.master], [screen.master], [], 15
      )
      if not readable and not writable:
        raise TimeoutError('Terminal did not accept input')
      if readable:
        stream.feed(os.read(screen.master, 65536))
      if writable:
        data = data[os.write(screen.master, data) :]
  finally:
    os.set_blocking(screen.master, True)


class _DaemonStub(Protocol):
  ListConversations: grpc.aio.UnaryUnaryMultiCallable[
    daemon_pb2.ListConversationsRequest, daemon_pb2.ListConversationsResponse
  ]
  GetPromptHistory: grpc.aio.UnaryUnaryMultiCallable[
    daemon_pb2.GetPromptHistoryRequest, daemon_pb2.GetPromptHistoryResponse
  ]
  GetDaemonInfo: grpc.aio.UnaryUnaryMultiCallable[
    daemon_pb2.GetDaemonInfoRequest, daemon_pb2.GetDaemonInfoResponse
  ]
  AttachWorkspace: grpc.aio.UnaryUnaryMultiCallable[
    daemon_pb2.AttachWorkspaceRequest, daemon_pb2.AttachWorkspaceResponse
  ]


class _AgentStub(Protocol):
  GetMessages: grpc.aio.UnaryUnaryMultiCallable[
    agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
  ]
  CallTool: grpc.aio.UnaryUnaryMultiCallable[
    agent_pb2.CallToolRequest, agent_pb2.CallToolResponse
  ]
  Session: grpc.aio.StreamStreamMultiCallable[
    agent_pb2.ClientMessage, agent_pb2.SessionMessage
  ]


def _worker_pids(daemon_pid: int) -> list[int]:
  processes = subprocess.run(
    ['ps', '-ax', '-o', 'pid=,ppid=,comm='],
    capture_output=True,
    text=True,
    check=True,
    timeout=5,
  )
  children: list[int] = []
  for line in processes.stdout.splitlines():
    pid, parent, command = line.split(maxsplit=2)
    if (
      int(parent) == daemon_pid and pathlib.Path(command).name == 'mirror-agent'
    ):
      children.append(int(pid))
  return children


async def _wait_for_process_exit(pid: int) -> None:
  async with asyncio.timeout(5):
    while True:
      try:
        os.kill(pid, 0)
      except ProcessLookupError:
        return
      await asyncio.sleep(0.05)


def _open_existing_cli(
  workspace: pathlib.Path,
  expected: str,
  *,
  browse: bool = False,
  create: bool = False,
  instruction: str | None = None,
) -> None:
  master, slave = pty.openpty()
  termios.tcsetwinsize(slave, (40, 120))
  child = None
  environment = dict(os.environ)
  if not create:
    environment.pop('OPENROUTER_API_KEY', None)
  try:
    child = subprocess.Popen(
      [
        sys.executable,
        '-c',
        'import os,sys; os.login_tty(0); os.execv(sys.argv[1], sys.argv[1:])',
        str(pathlib.Path(sys.executable).parent / 'mirror'),
      ],
      cwd=workspace,
      env=environment,
      stdin=slave,
      stdout=slave,
      stderr=slave,
    )
    os.close(slave)
    slave = -1
    screen = _Terminal(master)
    stream = pyte.ByteStream(screen)
    if browse:
      _wait_for_screen(child, screen, stream, 'Workspaces')
      query = (
        'packaged-smoke-model'
        if expected == 'packaged-smoke-model'
        else _PROMPT
      )
      os.write(master, query.encode())
      _wait_for_screen(child, screen, stream, '› ' + query)
      _wait_for_screen(child, screen, stream, '↻')
      os.write(master, b'\r')
    _wait_for_screen(child, screen, stream, 'manual |')
    _wait_for_screen(child, screen, stream, expected)
    if instruction is not None:
      os.write(master, f'{instruction}\r'.encode())
      _wait_for_screen(child, screen, stream, f'{_ANSWER} 3')
    if create:
      os.write(master, b'/new\r')
      _wait_for_screen(child, screen, stream, 'New working directory')
      os.write(master, b'\r')
      _wait_for_screen(child, screen, stream, 'Tab / Shift+Tab')
      os.write(master, b'\x1b[Z')
      _wait_for_screen(
        child, screen, stream, f'{_ANSWER} 3' if instruction else expected
      )
    os.write(master, b'\x04')
    _wait_for_screen(child, screen, stream, None)
  finally:
    if child is not None and child.poll() is None:
      child.kill()
      child.wait(timeout=5)
    os.close(master)
    if slave != -1:
      os.close(slave)


async def _headless_cli(workspace: pathlib.Path, mode: str) -> None:
  command = [
    str(pathlib.Path(sys.executable).parent / 'mirror'),
    mode,
    '--api',
    'openai-chat-completions',
    '--model',
    'packaged-smoke-model',
    '--base-url',
    os.environ['OPENAI_BASE_URL'],
    '--approval-mode',
    'auto',
  ]
  if mode == 'exec':
    command.append('headless exec prompt')
  child = await asyncio.create_subprocess_exec(
    *command,
    cwd=workspace,
    stdin=asyncio.subprocess.PIPE,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  try:
    if mode == 'acp':
      assert child.stdin is not None and child.stdout is not None
      child.stdin.write(
        json.dumps({
          'jsonrpc': '2.0',
          'id': 1,
          'method': 'session/new',
          'params': {'cwd': str(workspace), 'mcpServers': []},
        }).encode()
        + b'\n'
      )
      await child.stdin.drain()
      response = json.loads(await asyncio.wait_for(child.stdout.readline(), 10))
      session_id = response['result']['sessionId']
      child.stdin.write(
        json.dumps({
          'jsonrpc': '2.0',
          'id': 2,
          'method': 'session/prompt',
          'params': {
            'sessionId': session_id,
            'prompt': [{'type': 'text', 'text': 'headless acp prompt'}],
          },
        }).encode()
        + b'\n'
      )
      await child.stdin.drain()
      async with asyncio.timeout(10):
        while True:
          response = json.loads(await child.stdout.readline())
          if response.get('id') == 2:
            assert response['result']['stopReason'] == 'end_turn'
            break
    assert child.stdin is not None
    child.stdin.close()
    output, error = await asyncio.wait_for(child.communicate(), 10)
    assert child.returncode == 0, error.decode()
    if mode == 'exec':
      assert _ANSWER.encode() in output
  finally:
    if child.returncode is None:
      child.kill()
      await child.wait()


async def _check_cli_reconnect(root: pathlib.Path, version: str) -> None:
  """Run inside a subprocess with a private HOME and no inherited credentials."""
  home = pathlib.Path.home()
  workspace = root / 'workspace'
  workspace.mkdir()
  subprocess.run(
    ['git', 'init', '--initial-branch=main', str(workspace)],
    capture_output=True,
    check=True,
    timeout=10,
  )
  (home / '.mirror').mkdir()
  (home / '.mirror/config.json').write_text(
    json.dumps({
      'analytics': False,
      'usage_warning_acknowledged': True,
      'trusted_directories': [str(workspace)],
    })
  )
  worker = pathlib.Path(config.__file__).parent / 'bin/mirror-agent'
  database = home / '.mirror/state.db'
  profile = hashlib.sha256(str(database.resolve()).encode()).hexdigest()[:16]
  state = pathlib.Path(
    os.environ.get('MIRROR_DAEMON_STATE_ROOT') or home / '.mirror/run' / profile
  )
  command = (
    str(worker.with_name('mirror-daemon')),
    '--state-root',
    str(state),
    '--database',
    str(database),
    '--worker-home',
    str(home),
    '--worker-path',
    os.environ['PATH'],
    '--application-version',
    distribution.installed().version
    if version in ('same', 'default')
    else version,
  )
  daemon = await asyncio.create_subprocess_exec(
    *command,
    env={'TMPDIR': tempfile.gettempdir()},
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  metadata = None
  stream = None
  address = ''
  try:
    assert daemon.stdout is not None
    ready = await asyncio.wait_for(daemon.stdout.readline(), timeout=10)
    assert ready.startswith(b'MIRROR_DAEMON_ADDRESS=')
    address = ready.decode().strip().removeprefix('MIRROR_DAEMON_ADDRESS=')
    metadata = daemon_client.Client(address)
    async with grpc.aio.insecure_channel(
      address,
      options=(
        ('grpc.default_authority', 'localhost'),
        ('grpc.enable_http_proxy', 0),
      ),
    ) as channel:
      stub = cast(_DaemonStub, daemon_pb2_grpc.DaemonRPCStub(channel))
      agent = cast(_AgentStub, agent_pb2_grpc.AgentRPCStub(channel))
      before = await stub.GetDaemonInfo(
        daemon_pb2.GetDaemonInfoRequest(), timeout=5
      )
      record = await asyncio.to_thread(
        metadata.create,
        daemon_pb2.CreateWorkspaceRequest(
          workspace_id=str(uuid.uuid4()), directory=str(workspace)
        ),
      )
      host_tool = agent_pb2.HostToolDefinition(
        name='fixture-host',
        description='Synthetic client callback',
        parameters_json='{"type":"object"}',
        timeout_seconds=5,
      )
      configuration = daemon_pb2.AgentConfiguration(
        inference=agent_pb2.InferenceConfig(
          model='packaged-smoke-model',
          api='openai-chat-completions',
          base_url=os.environ['OPENAI_BASE_URL'],
        ),
        api_key='synthetic-reconnect-key',
        host_tools=[host_tool],
        approval_mode=agent_pb2.APPROVAL_MODE_HITL,
        include_environment=False,
        checkpoints_enabled=False,
      )
      attached = await stub.AttachWorkspace(
        daemon_pb2.AttachWorkspaceRequest(
          workspace_id=record.id,
          client_id=str(uuid.uuid4()),
          configuration=configuration,
        ),
        timeout=15,
      )
      authorization = (('mirror-attachment-id', attached.attachment_id),)
      stream = agent.Session(metadata=authorization)
      await stream.write(
        agent_pb2.ClientMessage(
          session_id=record.session_id,
          message_id='host',
        )
      )
      async with asyncio.timeout(5):
        while True:
          event = await stream.read()
          assert isinstance(event, agent_pb2.SessionMessage)
          if event.input_ack.message_id == 'host':
            break
      request = agent_pb2.GetMessagesRequest(session_id=record.session_id)
      workers = _worker_pids(daemon.pid)
      assert len(workers) == 1
      await asyncio.to_thread(
        _open_existing_cli, workspace, 'packaged-smoke-model', browse=True
      )
      await stream.write(
        agent_pb2.ClientMessage(
          session_id=record.session_id,
          message_id='seed',
          message=agent_pb2.AgentMessage(
            id='seed',
            chat=agent_pb2.ChatRecord(
              user=agent_pb2.UserMessage(content=_PROMPT)
            ),
          ),
        )
      )
      async with asyncio.timeout(10):
        while True:
          event = await stream.read()
          assert isinstance(event, agent_pb2.SessionMessage)
          if event.agent_message.chat.assistant.content == _ANSWER:
            break
      history = await agent.GetMessages(
        request, metadata=authorization, timeout=5
      )
      assert sum(message.id == 'seed' for message in history.messages) == 1
      conversations = await stub.ListConversations(
        daemon_pb2.ListConversationsRequest(), timeout=5
      )
      assert [item.id for item in conversations.conversations] == [
        record.session_id
      ]
      assert conversations.conversations[0].directory == str(workspace)
      prompts = await stub.GetPromptHistory(
        daemon_pb2.GetPromptHistoryRequest(session_id=record.session_id),
        timeout=5,
      )
      assert list(prompts.prompts) == [_PROMPT]
      await asyncio.to_thread(_open_existing_cli, workspace, _ANSWER)
      assert _worker_pids(daemon.pid) == workers
      assert (
        await stub.GetDaemonInfo(daemon_pb2.GetDaemonInfoRequest(), timeout=5)
        == before
      )
      assert (
        await agent.GetMessages(request, metadata=authorization, timeout=5)
        == history
      )
      snapshot = await asyncio.to_thread(
        metadata.snapshot, daemon_pb2.WatchWorkspacesRequest()
      )
      assert snapshot.workspaces[0].session_id == record.session_id
      assert snapshot.workspaces[0].inference == attached.workspace.inference
      stream.cancel()
      await stream.code()
      await asyncio.to_thread(
        _open_existing_cli,
        workspace,
        _ANSWER,
        create=True,
        instruction='A turn after every client detached',
      )
      created = await asyncio.to_thread(
        metadata.snapshot, daemon_pb2.WatchWorkspacesRequest()
      )
      assert len(created.workspaces) == 2
      assert {item.session_id for item in created.workspaces} > {
        record.session_id
      }
      assert workers[0] in _worker_pids(daemon.pid)
      observer = await stub.AttachWorkspace(
        daemon_pb2.AttachWorkspaceRequest(
          workspace_id=record.id,
          client_id=str(uuid.uuid4()),
          configuration=daemon_pb2.AgentConfiguration(),
        ),
        timeout=5,
      )
      assert not observer.execution_host, (
        'Different live tools cannot claim host authority'
      )
      reacquired = await stub.AttachWorkspace(
        daemon_pb2.AttachWorkspaceRequest(
          workspace_id=record.id,
          client_id=str(uuid.uuid4()),
          configuration=daemon_pb2.AgentConfiguration(host_tools=[host_tool]),
        ),
        timeout=5,
      )
      assert reacquired.execution_host
      authorization = (('mirror-attachment-id', reacquired.attachment_id),)
      stream = agent.Session(metadata=authorization)
      await stream.write(
        agent_pb2.ClientMessage(
          session_id=record.session_id, message_id='reattached'
        )
      )
      async with asyncio.timeout(5):
        while True:
          event = await stream.read()
          assert isinstance(event, agent_pb2.SessionMessage)
          if event.input_ack.message_id == 'reattached':
            break
      call = agent.CallTool(
        agent_pb2.CallToolRequest(
          session_id=record.session_id, name=host_tool.name, arguments='{}'
        ),
        metadata=authorization,
        timeout=5,
      )
      async with asyncio.timeout(5):
        while True:
          event = await stream.read()
          assert isinstance(event, agent_pb2.SessionMessage)
          if event.HasField('host_tool_call'):
            break
      assert event.host_tool_call.function.name == host_tool.name
      await stream.write(
        agent_pb2.ClientMessage(
          session_id=record.session_id,
          message_id='new-host-result',
          host_tool_result=agent_pb2.HostToolResult(
            call_id=event.host_tool_call.id,
            content='executed by the new client',
            success=True,
          ),
        )
      )
      result = await call
      assert result.success and result.content == 'executed by the new client'
      continued = await agent.GetMessages(
        request, metadata=authorization, timeout=5
      )
      assert sum(message.id == 'seed' for message in continued.messages) == 1
      assert (
        sum(
          message.chat.user.content == 'A turn after every client detached'
          for message in continued.messages
        )
        == 1
      )
      old_workers = _worker_pids(daemon.pid)
      daemon.kill()
      await asyncio.wait_for(daemon.communicate(), timeout=10)
      assert daemon.returncode == -signal.SIGKILL
      await asyncio.gather(
        *(_wait_for_process_exit(pid) for pid in old_workers)
      )
      daemon = await asyncio.create_subprocess_exec(
        *command[:-1],
        distribution.installed().version,
        env={'TMPDIR': tempfile.gettempdir()},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )
      assert daemon.stdout is not None
      assert (
        await asyncio.wait_for(daemon.stdout.readline(), timeout=10) == ready
      )
      after = await stub.GetDaemonInfo(
        daemon_pb2.GetDaemonInfoRequest(), timeout=5
      )
      assert after.instance_id != before.instance_id
      restored = await asyncio.to_thread(
        metadata.snapshot, daemon_pb2.WatchWorkspacesRequest()
      )
      assert {item.session_id for item in restored.workspaces} == {
        item.session_id for item in created.workspaces
      }
      assert not any(item.HasField('inference') for item in restored.workspaces)
      restarted = await stub.AttachWorkspace(
        daemon_pb2.AttachWorkspaceRequest(
          workspace_id=record.id,
          client_id=str(uuid.uuid4()),
          configuration=configuration,
        ),
        timeout=15,
      )
      authorization = (('mirror-attachment-id', restarted.attachment_id),)
      assert (
        await agent.GetMessages(request, metadata=authorization, timeout=5)
        == continued
      )
      await asyncio.to_thread(_open_existing_cli, workspace, f'{_ANSWER} 3')
      assert len(_worker_pids(daemon.pid)) == 1
      old_workers = _worker_pids(daemon.pid)
      upgrade = await asyncio.create_subprocess_exec(
        *command[:-1],
        '9999.1.0',
        '--ensure',
        env={'TMPDIR': tempfile.gettempdir()},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )
      output, diagnostics = await asyncio.wait_for(
        upgrade.communicate(), timeout=35
      )
      assert upgrade.returncode == 0, diagnostics.decode()
      assert output == ready
      assert b'MIRROR_DAEMON_WARNING=Daemon update deferred' in diagnostics
      assert _worker_pids(daemon.pid) == old_workers
      assert (
        await stub.GetDaemonInfo(daemon_pb2.GetDaemonInfoRequest(), timeout=5)
        == after
      )
      assert (
        await agent.GetMessages(request, metadata=authorization, timeout=5)
        == continued
      )
      transport = local_rpc_client.Transport(
        [str(worker)],
        {'database': str(database)},
        workspace=workspace,
        log_dir=home,
      )
      await transport.manage('restart')
      assert await asyncio.wait_for(daemon.wait(), timeout=10) == 0
      await asyncio.gather(
        *(_wait_for_process_exit(pid) for pid in old_workers)
      )
      replaced = await stub.GetDaemonInfo(
        daemon_pb2.GetDaemonInfoRequest(), timeout=5, wait_for_ready=True
      )
      assert replaced.instance_id != after.instance_id
      with pytest.raises(grpc.aio.AioRpcError) as denied:
        await agent.GetMessages(request, metadata=authorization, timeout=5)
      assert denied.value.code() == grpc.StatusCode.PERMISSION_DENIED
      restored = await asyncio.to_thread(
        metadata.snapshot, daemon_pb2.WatchWorkspacesRequest()
      )
      assert {item.session_id for item in restored.workspaces} == {
        item.session_id for item in created.workspaces
      }
      assert not any(item.HasField('inference') for item in restored.workspaces)
      resumed = await stub.AttachWorkspace(
        daemon_pb2.AttachWorkspaceRequest(
          workspace_id=record.id,
          client_id=str(uuid.uuid4()),
          configuration=configuration,
        ),
        timeout=15,
      )
      assert (
        await agent.GetMessages(
          request,
          metadata=(('mirror-attachment-id', resumed.attachment_id),),
          timeout=5,
        )
        == continued
      )
      await _headless_cli(workspace, 'exec')
      await _headless_cli(workspace, 'acp')
      prompts = await stub.GetPromptHistory(
        daemon_pb2.GetPromptHistoryRequest(), timeout=5
      )
      assert 'headless exec prompt' in prompts.prompts
      assert 'headless acp prompt' in prompts.prompts
      assert not (home / '.mirror/cli.db').exists()
  finally:
    if stream is not None:
      stream.cancel()
    if metadata is not None:
      metadata.close()
    remaining_workers = _worker_pids(daemon.pid)
    if daemon.returncode is None:
      daemon.terminate()
    _, diagnostics = await asyncio.wait_for(daemon.communicate(), timeout=10)
    assert daemon.returncode == 0, diagnostics.decode()
    await asyncio.gather(
      *(_wait_for_process_exit(pid) for pid in remaining_workers)
    )
    await _stop_fixture_replacement(address)
    runtime_hash = hashlib.sha256(os.fsencode(state.resolve())).hexdigest()[:16]
    runtime = pathlib.Path('/tmp') / f'm-{os.geteuid()}-{runtime_hash}'
    if address == f'unix:{runtime / "daemon.sock"}':
      shutil.rmtree(runtime)


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.asyncio
@pytest.mark.parametrize('version', ['same', '9999.0.0', 'default'])
async def test_installed_cli_reconnects_to_live_workspace(
  tmp_path: pathlib.Path,
  version: str,
) -> None:
  home = tmp_path / 'home'
  home.mkdir()
  with (
    tempfile.TemporaryDirectory(prefix='mr-') as state,
    _ModelServer() as server,
  ):
    # Seed, detached TUI, exec and ACP turns all use synthetic inference.
    server.response_index = 2
    server.release_response.put(None)
    server.release_response.put(None)
    server.release_response.put(None)
    server.release_response.put(None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = (
      'import asyncio, pathlib, runpy; '
      f'namespace = runpy.run_path({str(pathlib.Path(__file__).resolve())!r}); '
      f'asyncio.run(asyncio.wait_for(namespace[{_check_cli_reconnect.__name__!r}]('
      f'pathlib.Path({str(tmp_path)!r}), {version!r}), timeout=65))'
    )
    child = await asyncio.create_subprocess_exec(
      sys.executable,
      '-c',
      script,
      env={
        'HOME': str(home),
        'PATH': os.environ.get('PATH', os.defpath),
        'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
        'TERM': 'xterm-256color',
        'LANG': 'C.UTF-8',
        'GRPC_ENABLE_FORK_SUPPORT': '0',
        'OPENROUTER_API_KEY': 'synthetic-reconnect-key',
        # Model catalogs `mirror acp` lists come from the synthetic server too.
        'OPENROUTER_BASE_URL': f'http://127.0.0.1:{server.server_port}',
        'OPENAI_BASE_URL': f'http://127.0.0.1:{server.server_port}',
        'MIRROR_DAEMON_STATE_ROOT': '' if version == 'default' else state,
        'TMPDIR': tempfile.gettempdir(),
        'MIRROR_SKIP_UPDATE': '1',
        'MIRROR_EXTENSIONS_ENABLED': 'false',
      },
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.STDOUT,
      start_new_session=True,
    )
    try:
      output, _ = await asyncio.wait_for(child.communicate(), timeout=80)
      assert child.returncode == 0, output.decode()
      assert server.requests.qsize() == 4, (
        'Reconnecting must not repeat inference'
      )
    finally:
      if child.returncode is None:
        os.killpg(child.pid, signal.SIGKILL)
        await child.wait()
      server.shutdown()
      thread.join(timeout=5)


class _UnidentifiedDaemon(daemon_pb2_grpc.DaemonRPCServicer):
  def WatchWorkspaces(  # noqa: N802, generated RPC name
    self,
    request: daemon_pb2.WatchWorkspacesRequest,
    context: grpc.ServicerContext,
  ) -> Iterator[daemon_pb2.WatchWorkspacesResponse]:
    del request, context
    yield daemon_pb2.WatchWorkspacesResponse(
      workspaces=[daemon_pb2.Workspace(id='still-running')]
    )


async def _serve_unidentified_daemon(
  state: pathlib.Path, ignore_term: bool
) -> None:
  """Serve a legacy protocol in a disposable process, never the pytest process."""
  if ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
  endpoint = state / 'daemon.sock'
  server = grpc.aio.server()
  daemon_pb2_grpc.add_DaemonRPCServicer_to_server(  # pyright: ignore[reportUnknownMemberType] - generated registration has no stub.
    _UnidentifiedDaemon(), server
  )
  server.add_insecure_port(f'unix:{endpoint}')
  with (state / 'daemon.lock').open('w') as lock:
    os.fchmod(lock.fileno(), 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    await server.start()
    endpoint.chmod(0o600)
    print(f'MIRROR_DAEMON_ADDRESS=unix:{endpoint}', flush=True)
    await server.wait_for_termination()


def _daemon_pid(address: str) -> int:
  """Identify only the process serving this fixture's private socket."""
  with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(address.removeprefix('unix:'))
    if sys.platform == 'darwin':
      return connection.getsockopt(0, 2)  # SOL_LOCAL, LOCAL_PEERPID.
    pid, _, _ = struct.unpack(
      '3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    return pid


@contextlib.asynccontextmanager
async def _running_fixture_daemon(
  command: Sequence[str], environment: dict[str, str]
) -> AsyncIterator[tuple[asyncio.subprocess.Process, str]]:
  process = await asyncio.create_subprocess_exec(
    *command,
    env=environment,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  address = ''
  try:
    assert process.stdout is not None
    ready = await asyncio.wait_for(process.stdout.readline(), timeout=10)
    assert ready.startswith(b'MIRROR_DAEMON_ADDRESS='), ready
    address = ready.decode().strip().removeprefix('MIRROR_DAEMON_ADDRESS=')
    yield process, address
  finally:
    if process.returncode is None:
      process.kill()
    await asyncio.wait_for(process.communicate(), timeout=10)
    await _stop_fixture_replacement(address)


async def _stop_fixture_replacement(address: str) -> None:
  if not address:
    return
  with contextlib.suppress(FileNotFoundError, ConnectionRefusedError):
    pid = _daemon_pid(address)
    assert pid != os.getpid()
    workers = _worker_pids(pid)
    os.kill(pid, signal.SIGTERM)
    await _wait_for_process_exit(pid)
    await asyncio.gather(
      *(_wait_for_process_exit(worker) for worker in workers)
    )


async def _discover_fixture_daemon(
  root: pathlib.Path, *, restart: bool = False
) -> None:
  """Use the installed CLI startup path without sending an inference request."""
  transport = local_rpc_client.Transport(
    [str(pathlib.Path(config.__file__).parent / 'bin/mirror-agent')],
    {'database': str(root / 'state.db')},
    workspace=root,
    log_dir=root,
  )
  if restart:
    await transport.manage('restart')
  metadata = await transport.daemon_client()
  try:
    await asyncio.to_thread(
      metadata.snapshot, daemon_pb2.WatchWorkspacesRequest()
    )
  finally:
    metadata.close()
    await transport.close()


def _fixture_call(function: str, arguments: str) -> tuple[str, ...]:
  return (
    sys.executable,
    '-c',
    'import asyncio, pathlib, runpy; '
    f'namespace = runpy.run_path({str(pathlib.Path(__file__).resolve())!r}); '
    f'asyncio.run(namespace[{function!r}]({arguments}))',
  )


async def _launch_fixture_client(
  root: pathlib.Path,
  state: str,
  *,
  development: bool = False,
  restart: bool = False,
  expected_exit: int = 0,
) -> str:
  environment = {
    'HOME': str(root),
    'PATH': os.environ.get('PATH', os.defpath),
    'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
    'MIRROR_DAEMON_STATE_ROOT': state,
    'GRPC_ENABLE_FORK_SUPPORT': '0',
  }
  if development:
    environment['MIRROR_RUST_WORKER'] = str(
      pathlib.Path(config.__file__).parent / 'bin/mirror-agent'
    )
  process = await asyncio.create_subprocess_exec(
    *_fixture_call(
      '_discover_fixture_daemon',
      f'pathlib.Path({str(root)!r}), restart={restart!r}',
    ),
    env=environment,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )
  try:
    _, stderr = await asyncio.wait_for(process.communicate(), timeout=35)
    assert process.returncode == expected_exit, stderr.decode()
    return stderr.decode()
  finally:
    if process.returncode is None:
      process.kill()
      await process.wait()


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.asyncio
async def test_installed_cli_skips_invalid_catalog_entries(
  tmp_path: pathlib.Path,
) -> None:
  with tempfile.TemporaryDirectory(prefix='mi-', dir='/tmp') as state:
    directory = pathlib.Path(state)
    old_catalog = directory / 'workspaces.pb'
    old_bytes = b'other Mirror build: synthetic incompatible catalog'
    old_catalog.write_bytes(old_bytes)
    catalog = directory / 'workspaces-v1.pb'
    # Two record envelopes: malformed creation payload and missing identity.
    invalid = b'\x0a\x03\x12\x01\xff\x0a\x00'
    catalog.write_bytes(invalid)
    try:
      await _launch_fixture_client(tmp_path, state)
      await _stop_fixture_replacement(f'unix:{state}/daemon.sock')
      await _launch_fixture_client(tmp_path, state)
      assert old_catalog.read_bytes() == old_bytes
      assert catalog.read_bytes() == invalid
    finally:
      await _stop_fixture_replacement(f'unix:{state}/daemon.sock')


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.asyncio
async def test_installed_daemon_reports_startup_failure_without_catalog_values(
  tmp_path: pathlib.Path,
) -> None:
  with tempfile.TemporaryDirectory(prefix='mi-', dir='/tmp') as state:
    catalog = pathlib.Path(state) / 'workspaces-v1.pb'
    canary = b'synthetic-catalog-secret'
    catalog.write_bytes(b'\x12\x01\xff' + canary)
    process = await asyncio.create_subprocess_exec(
      str(pathlib.Path(config.__file__).parent / 'bin/mirror-daemon'),
      '--ensure',
      '--state-root',
      state,
      '--database',
      str(tmp_path / 'state.db'),
      '--worker-home',
      str(tmp_path),
      env={},
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    try:
      stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
      assert process.returncode != 0
      assert b'failed to decode Protobuf message' in stderr
      assert b'run without --ensure' not in stderr
      assert canary not in stdout + stderr
      assert catalog.read_bytes() == b'\x12\x01\xff' + canary
    finally:
      if process.returncode is None:
        process.kill()
        await process.wait()
      await _stop_fixture_replacement(f'unix:{state}/daemon.sock')


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.asyncio
@pytest.mark.parametrize('ignore_term', [False, True])
async def test_installed_cli_requires_explicit_restart_for_unidentified_daemon(
  tmp_path: pathlib.Path, ignore_term: bool
) -> None:
  with tempfile.TemporaryDirectory(prefix='mi-', dir='/tmp') as state:
    command = _fixture_call(
      '_serve_unidentified_daemon', f'pathlib.Path({state!r}), {ignore_term!r}'
    )
    environment = {
      'HOME': str(tmp_path),
      'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
    }
    async with _running_fixture_daemon(command, environment) as (
      previous,
      address,
    ):
      lock = pathlib.Path(state) / 'daemon.lock'
      inode = lock.stat().st_ino
      diagnostics = await _launch_fixture_client(
        tmp_path, state, expected_exit=1
      )
      assert (
        'Mirror daemon is unresponsive; restart it explicitly' in diagnostics
      )
      assert _daemon_pid(address) == previous.pid
      assert lock.stat().st_ino == inode
      await _launch_fixture_client(tmp_path, state, restart=True)
      assert await asyncio.wait_for(previous.wait(), timeout=5) == -(
        signal.SIGKILL if ignore_term else signal.SIGTERM
      )
      assert _daemon_pid(address) != previous.pid
      assert lock.stat().st_ino == inode
      async with grpc.aio.insecure_channel(
        address,
        options=(
          ('grpc.default_authority', 'localhost'),
          ('grpc.enable_http_proxy', 0),
        ),
      ) as channel:
        stub = cast(_DaemonStub, daemon_pb2_grpc.DaemonRPCStub(channel))
        before = await stub.GetDaemonInfo(
          daemon_pb2.GetDaemonInfoRequest(), timeout=5
        )
        await _launch_fixture_client(tmp_path, state)
        assert (
          await stub.GetDaemonInfo(daemon_pb2.GetDaemonInfoRequest(), timeout=5)
          == before
        )


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('version', 'worker_changed', 'development', 'replacement'),
  [
    pytest.param('same', False, False, False, id='identical-copy'),
    pytest.param('same', False, True, False, id='unchanged-development'),
    pytest.param('same', True, False, False, id='same-version-rebuild'),
    pytest.param('0.0.0', False, False, True, id='older-release'),
    pytest.param('9999.0.0', False, False, False, id='newer-release'),
    pytest.param(
      '9999.0.0', True, True, False, id='development-keeps-newer-release'
    ),
  ],
)
async def test_installed_cli_selects_current_installation(
  tmp_path: pathlib.Path,
  version: str,
  worker_changed: bool,
  development: bool,
  replacement: bool,
) -> None:
  bundle = tmp_path / 'previous'
  bundle.mkdir()
  installed = pathlib.Path(config.__file__).parent / 'bin'
  shutil.copy2(installed / 'mirror-daemon', bundle / 'mirror-daemon')
  shutil.copy2(installed / 'mirror-agent', bundle / 'mirror-agent')
  if worker_changed:
    (bundle / 'mirror-agent').write_bytes(b'#!/bin/sh\nexit 1\n')
  with tempfile.TemporaryDirectory(prefix='mi-', dir='/tmp') as state:
    command = (
      str(bundle / 'mirror-daemon'),
      '--state-root',
      state,
      '--database',
      str(tmp_path / 'state.db'),
      '--worker-home',
      str(tmp_path),
      '--application-version',
      distribution.installed().version if version == 'same' else version,
    )
    async with _running_fixture_daemon(command, {}) as (previous, address):
      await _launch_fixture_client(tmp_path, state, development=development)
      replaced = _daemon_pid(address) != previous.pid
      assert replaced is replacement
      async with grpc.aio.insecure_channel(
        address,
        options=(
          ('grpc.default_authority', 'localhost'),
          ('grpc.enable_http_proxy', 0),
        ),
      ) as channel:
        stub = cast(_DaemonStub, daemon_pb2_grpc.DaemonRPCStub(channel))
        before = await stub.GetDaemonInfo(
          daemon_pb2.GetDaemonInfoRequest(), timeout=5
        )
        await _launch_fixture_client(tmp_path, state, development=development)
        assert (
          await stub.GetDaemonInfo(daemon_pb2.GetDaemonInfoRequest(), timeout=5)
          == before
        )


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.parametrize(
  ('api', 'endpoint', 'token_key', 'input_key', 'content'),
  [
    (
      'openai-chat-completions',
      '/chat/completions',
      'max_tokens',
      'messages',
      _PROMPT,
    ),
    (
      'openai-responses',
      '/responses',
      'max_output_tokens',
      'input',
      [{'type': 'input_text', 'text': _PROMPT}],
    ),
  ],
)
def test_installed_cli_pty(
  tmp_path: pathlib.Path,
  api: str,
  endpoint: str,
  token_key: str,
  input_key: str,
  content: str | list[dict[str, str]],
) -> None:
  """Type into the installed mirror, render its real worker reply, and exit."""
  executable = pathlib.Path(sys.executable).parent / 'mirror'
  assert executable.is_file(), (
    'Install the native wheel into the test interpreter environment'
  )
  home = tmp_path / ('home-' + 'x' * 80)
  workspace = tmp_path / 'workspace'
  client_tmp = tmp_path / 'client-tmp'
  home.mkdir()
  workspace.mkdir()
  client_tmp.mkdir()
  # Never forward provider credentials, worker overrides, or user extensions.
  env = {
    'PATH': os.environ.get('PATH', os.defpath),
    'HOME': str(home),
    'TERM': 'xterm-256color',
    'LANG': 'C.UTF-8',
    'OPENROUTER_API_KEY': 'synthetic-packaged-pty-key',
    'OPENAI_API_KEY': 'synthetic-packaged-pty-key',
    'MIRROR_EXTENSIONS_ENABLED': 'false',
    'MIRROR_SKIP_UPDATE': '1',  # Updates have their own isolated startup test.
    'MIRROR_MODELS_CONFIG': str(home / 'models.json'),
    'MIRROR_STATE_DB': str(home / 'state.db'),
    'TMPDIR': str(client_tmp),
  }
  # setup-python needs its matching libpython, not the system Python library.
  if library_path := os.environ.get('LD_LIBRARY_PATH'):
    env['LD_LIBRARY_PATH'] = library_path
  # Keep repository instructions and extensions outside the fixture out of startup.
  subprocess.run(
    ['git', 'init', '--initial-branch=main', str(workspace)],
    env=env,
    check=True,
    capture_output=True,
    timeout=10,
  )
  with _ModelServer() as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    master = slave = -1
    child = None
    daemon = None
    database = home / 'state.db'
    profile = hashlib.sha256(str(database.resolve()).encode()).hexdigest()[:16]
    state_root = home / '.mirror' / 'run' / profile
    # Exercise fallback discovery across different daemon/client temporary directories.
    assert len(os.fsencode(state_root / 'daemon.sock')) > 103
    try:
      daemon = subprocess.Popen(
        [
          str(pathlib.Path(config.__file__).parent / 'bin/mirror-daemon'),
          '--application-version',
          distribution.installed().version,
          '--state-root',
          str(state_root),
          '--database',
          str(database),
          '--worker-home',
          str(home),
          '--worker-path',
          env['PATH'],
        ],
        env={'TMPDIR': tempfile.gettempdir()},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
      )
      assert daemon.stdout is not None
      assert select.select([daemon.stdout], [], [], 10)[0]
      assert daemon.stdout.readline().startswith(b'MIRROR_DAEMON_ADDRESS=')
      master, slave = pty.openpty()
      termios.tcsetwinsize(slave, (40, 120))
      child = subprocess.Popen(
        [
          sys.executable,
          '-c',
          # Give the CLI a controlling terminal, not just terminal-shaped fds.
          'import os,sys; os.login_tty(0); os.execv(sys.argv[1], sys.argv[1:])',
          str(executable),
          '--model',
          'packaged-smoke-model',
          '--api',
          api,
          '--base-url',
          f'http://127.0.0.1:{server.server_port}',
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=workspace,
        env=env,
      )
      os.close(slave)
      slave = -1
      screen = _Terminal(master)
      stream = pyte.ByteStream(screen)
      _wait_for_screen(
        child, screen, stream, 'Continue and trust this folder? [y/N]'
      )
      assert server.requests.empty()
      os.write(master, b'yes\r')
      _wait_for_screen(child, screen, stream, 'New working directory')
      os.write(master, b'\r')
      _wait_for_screen(child, screen, stream, 'packaged-smoke-model')
      _wait_for_screen(child, screen, stream, 'anonymous feature usage')
      settings = _wait_for_config_key(
        home / '.mirror' / 'config.json', 'analytics_notice_shown'
      )
      install_id = settings.pop('install_id')
      assert isinstance(install_id, str)
      assert re.fullmatch(r'[0-9a-f]{32}', install_id)
      assert settings == {
        'usage_warning_acknowledged': True,
        'trusted_directories': [str(workspace.resolve())],
        'analytics_notice_shown': True,
        'new_workspace_kind': 'normal',
      }
      os.write(master, _PROMPT.encode() + b'\r')
      _wait_for_screen(child, screen, stream, '(11 tokens)')
      server.release_response.put(None)
      _wait_for_screen(child, screen, stream, '(18 tokens)')
      server.release_response.put(None)
      _wait_for_screen(child, screen, stream, _ANSWER)
      path, body = server.requests.get(timeout=5)
      assert path == endpoint
      request = json.loads(body)
      assert request['stream'] is True
      assert 0 < request[token_key] < 8192
      assert any(
        message['role'] == 'user' and message['content'] == content
        for message in request[input_key]
      )
      long_prompt = 'Reading history\n' + '\n'.join(
        f'History row {index:03d}' for index in range(90)
      )
      server.release_response.put(None)
      _write_terminal(
        screen, stream, b'\x1b[200~' + long_prompt.encode() + b'\x1b[201~\r'
      )
      _wait_for_screen(child, screen, stream, f'{_ANSWER} 3')
      os.kill(child.pid, signal.SIGSTOP)
      _, stopped = os.waitpid(child.pid, os.WUNTRACED)
      assert os.WIFSTOPPED(stopped)
      try:
        os.write(master, b'\x1b[5~' * 10)
        _wait_for_screen(
          child, screen, stream, 'History row 000', exact_row=True
        )
      finally:
        os.kill(child.pid, signal.SIGCONT)
      os.write(master, 'draft é'.encode())
      _wait_for_screen(child, screen, stream, 'draft é')
      os.write(
        master, b'\x15\x04'
      )  # Clear the draft and request graceful shutdown.
      _wait_for_screen(child, screen, stream, None)
    finally:
      server.release_response.put(None)
      server.release_response.put(None)
      try:
        if child is not None:
          with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
          if child.poll() is None:
            child.kill()
          child.wait(timeout=5)
      finally:
        if master != -1:
          os.close(master)
        if slave != -1:
          os.close(slave)
        server.shutdown()
        thread.join(timeout=10)
        if daemon is not None:
          if daemon.poll() is None:
            daemon.terminate()
          try:
            daemon.communicate(timeout=10)
          except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.communicate(timeout=5)
          runtime_hash = hashlib.sha256(
            os.fsencode(state_root.resolve())
          ).hexdigest()[:16]
          runtime = pathlib.Path('/tmp') / f'm-{os.geteuid()}-{runtime_hash}'
          shutil.rmtree(runtime, ignore_errors=True)
        assert not thread.is_alive()


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
def test_installed_sandboxd_is_a_linux_build_for_this_machine() -> None:
  sandboxd = pathlib.Path(config.__file__).parent / 'bin/linux/mirror-sandboxd'
  image = sandboxd.read_bytes()
  (machine,) = struct.unpack_from('<H', image, 18)

  assert os.access(sandboxd, os.X_OK)
  assert image[:5] == b'\x7fELF\x02'
  assert machine == _ELF_MACHINES[platform.machine()]


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1'
  or sys.platform != 'linux',
  reason='Requires an installed native wheel on Linux',
)
def test_installed_sandboxd_loads_on_linux() -> None:
  sandboxd = pathlib.Path(config.__file__).parent / 'bin/linux/mirror-sandboxd'

  completed = subprocess.run(
    [sandboxd, 'no-such-command'],
    capture_output=True,
    text=True,
    check=False,
    timeout=30,
  )

  assert 'unsupported mirror-sandboxd command' in completed.stderr


@pytest.mark.asyncio
async def test_distribution_extensions(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.chdir(tmp_path)
  (
    tmp_path / '.git'
  ).mkdir()  # Do not discover extensions in the parent checkout.
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  packaged = pathlib.Path(user_extensions.__file__).with_name(
    'packaged_extensions'
  )
  expected = {path.parent.name for path in packaged.glob('*/extension.py')}
  registry = extension.ExtensionRegistry()
  try:
    for name in expected:
      await registry.load(packaged / name, overrides={'enabled': False})
    manager = user_extensions.UserExtensions(
      registry,
      tmp_path / 'extensions',
      cwd=tmp_path,
      settings=config.Config(tmp_path / 'config.json'),
    )
    await manager.start()
    assert {choice.label for choice in manager.selector().selector.choices} == {
      name
      for name in expected
      if registry.default_enabled(packaged / name) != 'hidden'
    }
    assert {
      name
      for name in registry.extension_ids()
      if registry.source_directory(name) is not None
    } == {
      name
      for name in expected
      if registry.default_enabled(packaged / name) is True
    }
    assert not (tmp_path / 'config.json').exists()
  finally:
    await registry.close()


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
def test_installed_daemon_management_preserves_state(
  tmp_path: pathlib.Path,
) -> None:
  binary = pathlib.Path(config.__file__).parent / 'bin/mirror-daemon'
  state = tmp_path / 'state'
  command = [
    str(binary),
    '--state-root',
    str(state),
    '--database',
    str(tmp_path / 'agent.db'),
    '--worker-home',
    str(tmp_path),
  ]

  def run(action: str) -> str:
    result = subprocess.run(
      [*command, action],
      env={},
      capture_output=True,
      text=True,
      timeout=35,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

  assert run('--status') == 'Mirror daemon stopped'
  assert run('--stop') == 'Mirror daemon stopped'
  assert not state.exists(), (
    'Observation and an absent stop must not create state'
  )
  state.mkdir(mode=0o700)
  legacy = state / 'workspaces.pb'
  legacy.write_bytes(b'other build synthetic catalog')
  address = run('--ensure').removeprefix('MIRROR_DAEMON_ADDRESS=unix:')
  try:
    before = run('--status')
    assert before.startswith('Mirror daemon running (')
    assert run('--ensure') == f'MIRROR_DAEMON_ADDRESS=unix:{address}'
    assert run('--status') == before
    assert legacy.read_bytes() == b'other build synthetic catalog'
    lock = (state / 'daemon.lock').stat().st_ino
    assert run('--stop') == 'Mirror daemon stopped'
    assert run('--status') == 'Mirror daemon stopped'
    assert run('--stop') == 'Mirror daemon stopped'
    assert (state / 'daemon.lock').stat().st_ino == lock
    assert legacy.read_bytes() == b'other build synthetic catalog'
  finally:
    run('--stop')
    runtime = pathlib.Path(address).parent
    if runtime != state.resolve():
      runtime.rmdir()


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
@pytest.mark.parametrize(
  ('version', 'worker_changed', 'restart', 'replaced'),
  [
    ('1.0.0', False, False, False),
    ('1.0.0', True, False, False),
    ('1.0.0', False, True, True),
    ('0.0.0', False, False, True),
    ('9999.0.0', False, False, False),
    ('9999.0.0', False, True, True),
  ],
)
def test_installed_daemon_replacement_without_frontend(
  tmp_path: pathlib.Path,
  version: str,
  worker_changed: bool,
  restart: bool,
  replaced: bool,
) -> None:
  installed = pathlib.Path(config.__file__).parent / 'bin'
  previous = tmp_path / 'previous'
  previous.mkdir()
  shutil.copy2(installed / 'mirror-daemon', previous / 'mirror-daemon')
  shutil.copy2(installed / 'mirror-agent', previous / 'mirror-agent')
  if worker_changed:
    (previous / 'mirror-agent').write_bytes(b'synthetic obsolete worker')
  state = tmp_path / 'state'
  arguments = [
    '--state-root',
    str(state),
    '--database',
    str(tmp_path / 'agent.db'),
    '--worker-home',
    str(tmp_path),
  ]

  def run(bundle: pathlib.Path, *action: str) -> str:
    result = subprocess.run(
      [str(bundle / 'mirror-daemon'), *arguments, *action],
      env={},
      capture_output=True,
      text=True,
      timeout=35,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

  address = run(previous, '--ensure', '--application-version', version)
  try:
    before = run(installed, '--status')
    flags = ('--replace',) if restart else ()
    assert (
      run(
        installed,
        '--ensure',
        '--application-version',
        '1.0.0',
        *flags,
      )
      == address
    )
    after = run(installed, '--status')
    assert (before != after) is replaced
    run(installed, '--ensure', '--application-version', '1.0.0', *flags)
    assert (run(installed, '--status') != after) is restart
  finally:
    run(installed, '--stop')
    runtime = pathlib.Path(
      address.removeprefix('MIRROR_DAEMON_ADDRESS=unix:')
    ).parent
    if runtime != state.resolve():
      runtime.rmdir()


@pytest.mark.skipif(
  os.environ.get('MIRROR_TEST_PACKAGED_WHEEL') != '1',
  reason='Requires an installed native wheel',
)
def test_installed_cli_daemon_commands_need_no_consent_or_credentials(
  tmp_path: pathlib.Path,
) -> None:
  with tempfile.TemporaryDirectory(prefix='md-', dir='/tmp') as temporary:
    state = pathlib.Path(temporary) / 'state'
    environment = {
      'HOME': str(tmp_path),
      'PATH': os.defpath,
      # setup-python needs its matching libpython, not the system library.
      'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
      'MIRROR_DAEMON_STATE_ROOT': str(state),
      'MIRROR_SKIP_UPDATE': '1',
    }

    def run(action: str) -> str:
      result = subprocess.run(
        [sys.executable, '-m', 'mirror.main', 'daemon', action],
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
      )
      assert result.returncode == 0, result.stderr
      return result.stdout.strip()

    assert run('status') == 'Mirror daemon stopped'
    assert run('stop') == 'Mirror daemon stopped'
    assert not state.exists()
    assert not (tmp_path / '.mirror').exists()
    try:
      started = run('start')
      assert started.startswith('Mirror daemon running (')
      assert run('start') == started
      assert run('status') == started
      assert run('stop') == 'Mirror daemon stopped'
      assert run('status') == 'Mirror daemon stopped'
      assert not (tmp_path / '.mirror').exists()
    finally:
      run('stop')
