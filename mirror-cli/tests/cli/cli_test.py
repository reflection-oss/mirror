"""Tests for the mirror TUI: rendering, footer, and worker lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import gc
import io
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from collections.abc import (
  AsyncIterator,
  Callable,
  Coroutine,
  Iterable,
  Mapping,
  Sequence,
)
from typing import Any, cast, override
from unittest import mock

import grpc.aio
import pyte
import pytest
from freezegun import api as freezegun_api
from google.protobuf import empty_pb2
from mirror.protocol import (
  agent_pb2,
  agent_pb2_grpc,
  mirror_pb2,
  mirror_pb2_grpc,
)
from prompt_toolkit import application as _application
from prompt_toolkit import buffer as _buffer
from prompt_toolkit import data_structures as _data_structures
from prompt_toolkit import document as _document
from prompt_toolkit import mouse_events as _mouse_events
from prompt_toolkit import utils as _pt_utils
from prompt_toolkit.application import current as _current
from prompt_toolkit.formatted_text import ansi as _ansi
from prompt_toolkit.input import base as _input_base
from prompt_toolkit.input import defaults as _input_defaults
from prompt_toolkit.input import vt100_parser as _vt100_parser
from prompt_toolkit.layout import containers as _containers
from prompt_toolkit.layout import controls as _controls
from prompt_toolkit.layout import mouse_handlers as _mouse_handlers
from prompt_toolkit.layout import processors as _processors
from prompt_toolkit.layout import screen as _screen
from prompt_toolkit.output import base as _output_base
from prompt_toolkit.output import vt100 as _vt100
from rich import cells as _cells
from rich import console as _console
from rich import text as _text

from mirror import config, distribution, skills, workspace
from mirror.cli import actions as _actions
from mirror.cli import (
  agents,
  cli,
  cli_store,
  extension,
  themes,
  tool_approvals,
  tool_groups,
  user_extensions,
  workspaces,
)
from mirror.cli import commands as _commands
from mirror.cli import effects as _effects
from mirror.cli import host_tools as _host_tools
from mirror.cli import keymap as _keymap
from mirror.cli import local_rpc_client as _local_rpc_client
from mirror.cli import main as cli_main
from mirror.cli import models as _models
from mirror.cli import remote_rpc_client as _remote_rpc_client
from mirror.cli import rendering as _rendering
from mirror.cli import rpc_client as _rpc_client
from mirror.cli import selection as _selection
from mirror.cli import state as _state
from mirror.cli import stream as _stream_mod
from mirror.cli import terminal as _terminal_mod
from mirror.cli import workspace_controller as _workspace_controller
from mirror.cli.components import activity as _activity
from mirror.cli.components import completion as _completion
from mirror.cli.components import failure as _failure
from mirror.cli.components import pending as _pending
from mirror.cli.components import resume as _resume
from mirror.cli.components import rewind as _rewind
from mirror.cli.components import selector as _selector
from mirror.cli.components import tasks as _tasks
from mirror.cli.components import worktree_tabs
from mirror.cli.composer import processors as _composer_processors
from mirror.cli.extensions.connections import configuration as _mcp
from mirror.cli.extensions.connections import extension as _connections
from mirror.cli.extensions.editor import extension as _editor_extension
from mirror.cli.extensions.external_editor import (
  extension as external_editor_extension,
)
from mirror.cli.extensions.themes import extension as themes_extension
from mirror.cli.extensions.tips import extension as tips_extension
from mirror.cli.extensions.user_extensions import extension as _user_extension
from mirror.cli.extensions.view_mode import extension as _view_mode
from mirror.cli.extensions.vim import extension as vim_extension


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize(
  ('settings', 'enabled'),
  [
    ({}, False),
    ({'checkpoints_enabled': True}, True),
    ({'checkpoints_enabled': False}, False),
  ],
)
async def test_client_checkpoint_setting_is_forwarded(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  remote: bool,
  settings: dict[str, bool],
  enabled: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  settings_path = tmp_path / '.mirror' / 'config.json'
  settings_path.parent.mkdir()
  settings_path.write_text(json.dumps(settings))
  bootstrap = {'conversation_id': 'test', 'model': 'test-model'}
  with (
    mock.patch.object(
      _workspace_controller, 'WorkspaceController', autospec=True
    ) as controller,
    mock.patch.object(cli, '_Repl', autospec=True) as repl,
    mock.patch.object(
      cli._workspace_application.WorkspaceApplication,
      'run_views',
      new_callable=mock.AsyncMock,
    ),
    mock.patch.object(
      user_extensions.UserExtensions, 'start', new_callable=mock.AsyncMock
    ),
  ):
    await cli.run_tui(['worker'], bootstrap)
    create_agent = controller.call_args.kwargs['create_agent']
    spec = workspaces.AgentSpec(
      config=bootstrap,
      workspace=tmp_path,
      reasoning_effort='medium',
      active=True,
      remote=remote,
    )
    # Startup arguments are a snapshot, not live-reloaded extension settings.
    settings_path.write_text('{"checkpoints_enabled":false}')
    create_agent(spec, controller.return_value)
    assert repl.call_args.kwargs['fullscreen'] is True
    assert repl.call_args.args[1] == bootstrap | {
      'checkpoints_enabled': enabled
    }
    assert 'checkpoints_enabled' not in bootstrap


def test_openrouter_base_url_selects_chat_completions(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
  monkeypatch.setenv('OPENROUTER_BASE_URL', 'http://configured')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = cli_main._parse_args()

  assert args.api == 'openai-chat-completions'
  assert args.model == 'openai/gpt-6-astra'


def test_non_streaming_chat_completions_api_is_accepted(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(
    sys,
    'argv',
    ['mirror', '--api', 'openai-chat-completions-non-streaming'],
  )

  assert cli_main._parse_args().api == 'openai-chat-completions-non-streaming'


@pytest.mark.parametrize('flags', [[], ['--trust-extensions']])
def test_trust_extensions_requires_explicit_flag(
  monkeypatch: pytest.MonkeyPatch, flags: list[str]
) -> None:
  monkeypatch.setattr(sys, 'argv', ['mirror', *flags])
  assert cli_main._parse_args().trust_extensions is bool(flags)


def test_negative_scrollback_limit_is_rejected(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(sys, 'argv', ['mirror', '--scrollback-limit', '-1'])

  # Accepting it would crop past the start of the history and show nothing.
  with pytest.raises(SystemExit):
    cli_main._parse_args()


def test_flush_scrollback_defaults_off(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
  monkeypatch.setenv('OPENROUTER_BASE_URL', 'http://configured')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = cli_main._parse_args()

  assert args.flush_scrollback is False


def test_flush_scrollback_flag_enables_dropping_saved_lines(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
  monkeypatch.setenv('OPENROUTER_BASE_URL', 'http://configured')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(sys, 'argv', ['mirror', '--flush-scrollback'])

  args = cli_main._parse_args()

  assert args.flush_scrollback is True


def test_remote_endpoint_is_disabled_by_default(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('MIRROR_REMOTE_ENDPOINT', raising=False)
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = cli_main._parse_args()

  assert args.remote_endpoint is None
  assert args.remote_insecure is False


def test_remote_endpoint_defaults_from_environment(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('MIRROR_REMOTE_ENDPOINT', 'gateway.example:443')
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = cli_main._parse_args()

  assert args.remote_endpoint == 'gateway.example:443'
  assert args.remote_insecure is False


def test_configured_endpoint_without_api_selects_chat_completions(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
) -> None:
  # An entry that only names an endpoint must not fall back to the Codex
  # protocol, which would ask for a login and talk the wrong wire format.
  config = tmp_path / 'models.json'
  config.write_text(
    json.dumps([
      {'id': 'local', 'label': 'Local', 'base_url': 'http://localhost:8000/v1'}
    ])
  )
  monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = cli_main._parse_args()

  assert args.model == 'local'
  assert args.model_base_url == 'http://localhost:8000/v1'
  assert args.api == 'openai-chat-completions'


def test_naming_a_configured_model_applies_its_endpoint(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
) -> None:
  # `--model local` has to reach the same server `/model local` reaches.
  config = tmp_path / 'models.json'
  config.write_text(
    json.dumps([
      {'id': 'local', 'label': 'Local', 'base_url': 'http://localhost:8000/v1'}
    ])
  )
  monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  monkeypatch.setattr(sys, 'argv', ['mirror', '--model', 'local'])

  args = cli_main._parse_args()

  # The endpoint stays the entry's own, so switching to a model without one
  # returns to the provider default instead of inheriting this server.
  assert args.model_base_url == 'http://localhost:8000/v1'
  assert args.api == 'openai-chat-completions'


def test_explicit_api_survives_a_configured_model(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
) -> None:
  config = tmp_path / 'models.json'
  config.write_text(
    json.dumps([
      {'id': 'local', 'label': 'Local', 'api': 'openai-chat-completions'}
    ])
  )
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  monkeypatch.setattr(sys, 'argv', ['mirror', '--api', 'openai-responses'])

  args = cli_main._parse_args()

  assert args.api == 'openai-responses'


def test_api_key_alone_never_selects_a_backend(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('OPENROUTER_API_KEY', 'configured')
  monkeypatch.delenv('OPENROUTER_BASE_URL', raising=False)
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = cli_main._parse_args()

  assert args.api == 'reflection'
  assert args.model == 'Beam-501B-A23B'


# A stand-in AgentRPC worker that echoes each instruction as an assistant turn.
_ECHO_WORKER_SOURCE = textwrap.dedent("""
  import asyncio
  import json
  import os
  import signal
  import sys

  import grpc.aio
  from google.protobuf import empty_pb2

  from mirror.protocol import agent_pb2, agent_pb2_grpc

  class Echo(agent_pb2_grpc.AgentRPCServicer):
    async def GetMessages(self, request, context):
      return agent_pb2.GetMessagesResponse()

    async def Session(self, requests, context):
      async for request in requests:
        if request.HasField('set_inference_config'):
          yield agent_pb2.SessionMessage(
            input_ack=agent_pb2.InputAck(message_id=request.message_id)
          )
        if request.WhichOneof('payload') != 'message':
          continue
        yield agent_pb2.SessionMessage(
          event_id=1,
          agent_message=agent_pb2.AgentMessage(
            id='assistant',
            chat=agent_pb2.ChatRecord(
              assistant=agent_pb2.AssistantMessage(
                content='echo: ' + request.message.chat.user.content,
              ),
              success=True,
            ),
          ),
        )
        yield agent_pb2.SessionMessage(
          event_id=2,
          runtime_delta=agent_pb2.RuntimeDelta(
            turn_completed=empty_pb2.Empty()
          ),
        )

  async def main():
    config = json.loads(sys.argv[1])
    print('cwd=' + os.getcwd(), flush=True)
    server = grpc.aio.server()
    agent_pb2_grpc.add_AgentRPCServicer_to_server(Echo(), server)
    port = server.add_insecure_port(config['rpc_address'])
    await server.start()
    print(f'MIRROR_RPC_ADDRESS=127.0.0.1:{port}', flush=True)
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    await stopped.wait()
    await server.stop(0)

  asyncio.run(main())
""")


class _RemoteEcho(agent_pb2_grpc.AgentRPCServicer):
  def __init__(self, *, echo_user: bool = False) -> None:
    self.requests: asyncio.Queue[agent_pb2.ClientMessage] = asyncio.Queue()
    self.echo_user = echo_user

  async def GetMessages(  # noqa: N802, generated RPC name
    self,
    request: agent_pb2.GetMessagesRequest,
    context: grpc.aio.ServicerContext[
      agent_pb2.GetMessagesRequest, agent_pb2.GetMessagesResponse
    ],
  ) -> agent_pb2.GetMessagesResponse:
    del request, context
    return agent_pb2.GetMessagesResponse(
      messages=[
        agent_pb2.AgentMessage(
          id='remote-history',
          chat=agent_pb2.ChatRecord(
            user=agent_pb2.UserMessage(content='saved remotely'),
            success=True,
          ),
        )
      ],
      total=1,
    )

  async def Session(  # noqa: N802, generated RPC name
    self,
    request_iterator: AsyncIterator[agent_pb2.ClientMessage],
    context: grpc.aio.ServicerContext[
      agent_pb2.ClientMessage, agent_pb2.SessionMessage
    ],
  ) -> AsyncIterator[agent_pb2.SessionMessage]:
    del context
    async for request in request_iterator:
      await self.requests.put(request)
      yield agent_pb2.SessionMessage(
        input_ack=agent_pb2.InputAck(message_id=request.message_id)
      )
      if self.echo_user and request.WhichOneof('payload') == 'message':
        yield agent_pb2.SessionMessage(
          agent_message=agent_pb2.AgentMessage(
            id=request.message_id,
            chat=agent_pb2.ChatRecord(
              user=agent_pb2.UserMessage(
                content=request.message.chat.user.content
              ),
              success=True,
            ),
          )
        )


class _RemoteMirror(mirror_pb2_grpc.MirrorRPCServicer):
  def __init__(
    self,
    session_ids: list[str],
    *,
    start_gate: asyncio.Event | None = None,
  ) -> None:
    self.session_ids = session_ids
    self.started: asyncio.Queue[str] = asyncio.Queue()
    self.deleted: asyncio.Queue[str] = asyncio.Queue()
    self.offsets: list[int] = []
    self.start_gate = start_gate
    self.forks: list[mirror_pb2.ForkSessionRequest] = []
    self.start_requests: list[mirror_pb2.StartSessionRequest] = []

  async def ForkSession(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ForkSessionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.ForkSessionRequest, mirror_pb2.ForkSessionResponse
    ],
  ) -> mirror_pb2.ForkSessionResponse:
    del context
    self.forks.append(request)
    self.session_ids.append('forked')
    return mirror_pb2.ForkSessionResponse(session_id='forked')

  async def StartSession(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.StartSessionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.StartSessionRequest, mirror_pb2.StartSessionResponse
    ],
  ) -> mirror_pb2.StartSessionResponse:
    del context
    self.start_requests.append(request)
    await self.started.put(request.session_id)
    if self.start_gate is not None:
      await self.start_gate.wait()
    return mirror_pb2.StartSessionResponse()

  async def DeleteSession(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.DeleteSessionRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.DeleteSessionRequest, mirror_pb2.DeleteSessionResponse
    ],
  ) -> mirror_pb2.DeleteSessionResponse:
    del context
    await self.deleted.put(request.session_id)
    return mirror_pb2.DeleteSessionResponse()

  async def ListSessions(  # noqa: N802, generated RPC name
    self,
    request: mirror_pb2.ListSessionsRequest,
    context: grpc.aio.ServicerContext[
      mirror_pb2.ListSessionsRequest, mirror_pb2.ListSessionsResponse
    ],
  ) -> mirror_pb2.ListSessionsResponse:
    del context
    self.offsets.append(request.offset)
    session_ids = self.session_ids[
      request.offset : request.offset + request.limit
    ]
    return mirror_pb2.ListSessionsResponse(
      sessions=[
        mirror_pb2.Session(id=session_id) for session_id in session_ids
      ],
      total=len(self.session_ids),
    )


def _repl(
  worker_source: str = _ECHO_WORKER_SOURCE,
  conversation_id: str = 'conv-1',
  model: str = 'm-test',
  output: _output_base.Output | None = None,
  workspace_path: pathlib.Path | None = None,
  database: str | pathlib.Path | None = None,
  create_session: bool = False,
  remote_endpoint: str | None = None,
  remote_secure: bool = True,
  remote_workspace: mirror_pb2.Workspace | None = None,
  extension_registry: extension.ExtensionRegistry | None = None,
  extension_manager: user_extensions.UserExtensions | None = None,
  approvals: tool_approvals.ToolApprovals | None = None,
  workspace_extensions: tuple[str, ...] = (),
  workspace_tools: tuple[str, ...] | None = None,
  active: bool = True,
  fullscreen: bool = False,
  initial_theme: str = 'dark',
) -> cli._Repl:
  """Return a REPL configured to start a small `-c` AgentRPC worker."""

  def create() -> cli._Repl:
    repl = cli._Repl(
      [sys.executable, '-c', worker_source],
      {
        'conversation_id': conversation_id,
        'model': model,
        'model_api': None,
        'model_base_url': None,
        'approval_mode': 'auto',
        'create_session': create_session,
        'remote_workspace': remote_workspace,
      },
      session_id='test-session',
      workspace=workspace_path,
      workspace_effects=asyncio.Queue(),
      workspace_controller=mock.Mock(
        spec=_workspace_controller.WorkspaceController
      ),
      extension_registry=extension_registry
      if extension_registry is not None
      else extension.ExtensionRegistry(),
      user_extensions=extension_manager,
      tool_approvals=approvals
      or tool_approvals.ToolApprovals(
        cli_store.default_database_path().with_name('tool_approvals.json')
      ),
      reasoning_effort='medium',
      workspace_extensions=workspace_extensions,
      workspace_tools=workspace_tools,
      active=active,
      fullscreen=fullscreen,
      initial_theme=initial_theme,
      database=database,
      remote_endpoint=remote_endpoint,
      remote_secure=remote_secure,
    )
    # Existing rendering assertions exercise the per-call transcript view.
    repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))
    return repl

  if output is None:
    return create()
  with _current.create_app_session(output=output):
    return create()


def test_repl_uses_120_hz_redraw_cap_and_adaptive_periodic_refresh() -> None:
  repl = _repl(output=_output_base.DummyOutput())
  app = repl._app
  assert app.min_redraw_interval == pytest.approx(1 / 120)
  assert app.refresh_interval == pytest.approx(0.1)
  assert not app.full_screen

  repl.dispatch(_actions.TurnStarted())
  assert app.refresh_interval == pytest.approx(1 / 120)

  repl.dispatch(_actions.TurnFinished())
  assert app.refresh_interval == pytest.approx(0.1)


def test_exit_hint_uses_continue() -> None:
  repl = _repl()
  with repl._terminal.console.capture() as captured:
    repl.print_resume_hint()
  assert captured.get() == 'Bye. Continue with mirror --continue\n'


@pytest.mark.asyncio
async def test_missing_workspace_extension_blocks_worker_start() -> None:
  repl = _repl(workspace_extensions=('missing/indicator',))
  repl.dispatch(_actions.SetActive(False))
  try:
    await repl.start()
    assert repl._client is None
    assert repl._state.lifecycle.failure is not None
    assert 'Required extension is unavailable' in repl._state.lifecycle.failure
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_terminal_command_restores_draft_after_child_exits(
  monkeypatch: pytest.MonkeyPatch,
  cancel: bool,
) -> None:
  entered = threading.Event()
  release = threading.Event()

  def run(
    argv: tuple[str, ...],
    *,
    stdout: object,
    stderr: object,
    close_fds: bool,
    check: bool,
  ) -> subprocess.CompletedProcess[bytes]:
    assert stdout is stderr is repl.console.file
    assert argv[-3:] == ('/workspace', 'editor', '/workspace/file with spaces')
    assert not close_fds
    assert not check
    entered.set()
    assert release.wait(5)
    return subprocess.CompletedProcess(argv, 0)

  monkeypatch.setattr(subprocess, 'run', run)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._input.text = 'unfinished draft'
    repl._input.cursor_position = 4
    ready = asyncio.Event()
    app_task = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    with _current.set_app(repl._app):
      command = asyncio.create_task(
        repl._run_terminal_command(
          ('editor', '/workspace/file with spaces'),
          '/workspace',
        )
      )
    try:
      assert await asyncio.to_thread(entered.wait, 5)
      assert repl._app._running_in_terminal
      if cancel:
        command.cancel()
        await asyncio.sleep(0)
        assert not command.done()
      release.set()
      if cancel:
        with pytest.raises(asyncio.CancelledError):
          await command
      else:
        await command
      assert not repl._app._running_in_terminal
      assert repl._input.text == 'unfinished draft'
      assert repl._input.cursor_position == 4
    finally:
      release.set()
      await asyncio.gather(command, return_exceptions=True)
      repl._app.exit()
      await app_task


@pytest.mark.asyncio
async def test_terminal_command_missing_executable_reports_error(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  dispatched: list[_actions.Action] = []
  monkeypatch.setattr(repl, 'dispatch', dispatched.append)

  await repl._run_terminal_command(
    (str(tmp_path / 'missing-editor'),), str(tmp_path)
  )

  (error,) = dispatched
  assert isinstance(error, _actions.ShowText)
  # Shells differ on whether a missing absolute executable is 126 or 127.
  assert error.text in (
    'Command exited with status 126.',
    'Command exited with status 127.',
  )
  assert error.style == 'ui.error'


@pytest.mark.asyncio
async def test_terminal_command_spawns_without_fork_and_preserves_cwd_and_arguments(
  tmp_path: pathlib.Path,
) -> None:
  cwd = tmp_path / 'space; $(echo unsafe)'
  cwd.mkdir()
  repl = _repl()
  argument = 'literal; $(echo unsafe)'
  with mock.patch.object(os, 'posix_spawn', wraps=os.posix_spawn) as spawn:
    await repl._run_terminal_command(
      (
        sys.executable,
        '-c',
        'import pathlib, sys; pathlib.Path("result").write_text(sys.argv[1])',
        argument,
      ),
      str(cwd),
    )
    spawn.assert_called_once()
  assert (cwd / 'result').read_text() == argument


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('tools', 'expected'),
  [
    ((), []),
    ((_host_tools.ToolDefinition('chosen', 'Workspace tool', {}),), ['chosen']),
  ],
)
async def test_local_transport_uses_only_workspace_host_tools(
  tmp_path: pathlib.Path,
  tools: tuple[_host_tools.ToolDefinition, ...],
  expected: list[str],
) -> None:
  transport = _local_rpc_client.Transport(
    [sys.executable, '-c', _ECHO_WORKER_SOURCE],
    {
      'conversation_id': 'transport-test',
      'model': 'm-test',
      'host_tools': [{'name': 'stale'}],
    },
    workspace=tmp_path,
    log_dir=tmp_path / 'logs',
  )
  with mock.patch.object(
    asyncio, 'create_subprocess_exec', wraps=asyncio.create_subprocess_exec
  ) as spawn:
    try:
      await transport.connect(tools)
      configuration = json.loads(spawn.call_args.args[-1])
      assert [tool['name'] for tool in configuration['host_tools']] == expected
    finally:
      await transport.close()


@pytest.mark.asyncio
async def test_extension_agent_owns_a_separate_local_worker() -> None:
  repl = _repl()
  messages: asyncio.Queue[agent_pb2.SessionMessage] = asyncio.Queue()

  async def on_error(message: str) -> None:
    pytest.fail(message)

  client = repl._extension_host._context.create_agent(
    agents.Options(), messages.put_nowait, on_error
  )
  async with client:
    transport = client.transport
    assert isinstance(transport, _local_rpc_client.Transport)
    process = transport.process
    assert process is not None
    client.enqueue(
      agent_pb2.AgentMessage(
        id='child-input',
        chat=agent_pb2.ChatRecord(
          user=agent_pb2.UserMessage(content='child request'),
          success=True,
        ),
      )
    )
    attached = await asyncio.wait_for(messages.get(), timeout=5)
    assert attached.HasField('input_ack')
    response = await asyncio.wait_for(messages.get(), timeout=5)
    assert (
      response.agent_message.chat.assistant.content == 'echo: child request'
    )
  assert process.returncode is not None
  assert repl._client is None
  assert repl._state.session.conversation_id == 'conv-1'


class _StartupExtension(extension.Extension[int, None, None]):
  def __init__(self, tool_name: str = 'startup') -> None:
    self.definition = _host_tools.ToolDefinition(tool_name, 'Startup tool', {})
    self.loaded = 0
    self.running = asyncio.Event()
    self.stopped = asyncio.Event()
    self.tools_closed = False

  @override
  def tools(self, context: extension.Context) -> list[_host_tools.HostTool]:
    return [self]

  async def execute(
    self,
    arguments: str,
    context: _host_tools.ExecutionContext = _host_tools.MODEL_CONTEXT,
  ) -> _host_tools.ToolResult:
    return _host_tools.ToolResult(arguments, True)

  @override
  async def on_unload(self, context: None) -> None:
    self.tools_closed = True

  @classmethod
  @override
  def id(cls) -> str:
    return 'test/startup'

  @staticmethod
  @override
  def initial_state() -> int:
    return 0

  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.STATUS_RIGHT

  @override
  def reduce(
    self, global_state: _state.State, action: _actions.Action
  ) -> tuple[int, tuple[extension.PrivateEffect[None], ...]]:
    if isinstance(action, _actions.ExtensionLoaded):
      self.loaded += 1
      return 1, (self.effect(None),)
    return self.require_state(global_state), ()

  @override
  def render(
    self, global_state: _state.State, extension_state: int
  ) -> _containers.AnyContainer:
    return _containers.Window()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: None
  ) -> None:
    self.running.set()
    try:
      await asyncio.Event().wait()
    finally:
      self.stopped.set()


@pytest.mark.asyncio
async def test_fresh_tab_activation_starts_extensions_without_starting_worker(
  tmp_path: pathlib.Path,
) -> None:
  registry = extension.ExtensionRegistry()
  plugin = _StartupExtension()
  await registry.register(plugin, None)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(
      create_session=True,
      database=tmp_path / 'state.db',
      extension_registry=registry,
    )
    try:
      repl._app.pre_run_callables.append(lambda: repl._app.exit())
      await repl.activate(show_context=True)
      await asyncio.wait_for(plugin.running.wait(), timeout=2)
      assert not plugin.stopped.is_set()
      assert not repl._state.lifecycle.started
      assert repl._state.extensions[plugin.id()] == 1
      assert isinstance(
        repl._extension_host.render(
          repl._state, extension.RenderRegion.STATUS_RIGHT
        ),
        _containers.VSplit,
      )
      repl._app.pre_run_callables.append(lambda: repl._app.exit())
      await repl.activate(show_context=True)
      await repl._run_turn('first prompt')
      assert repl._state.lifecycle.started
      assert plugin.loaded == 1
      assert not plugin.stopped.is_set()
    finally:
      await repl.close()
      await registry.close()
    assert plugin.stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('remote_endpoint', [None, 'remote.example:443'])
async def test_call_tool_requires_a_worker_even_for_registered_host_tools(
  remote_endpoint: str | None,
) -> None:
  repl = _repl(remote_endpoint=remote_endpoint)
  repl._host_tools_registry.register(
    'custom', [_StartupExtension('custom.echo')]
  )
  try:
    with pytest.raises(RuntimeError, match='agent is not connected'):
      await repl.call_tool('custom.echo', {'value': 1})
  finally:
    await repl.close()


class _OtherStartupExtension(_StartupExtension):
  @classmethod
  @override
  def id(cls) -> str:
    return 'test/other'


@pytest.mark.asyncio
async def test_extension_tool_changes_restart_with_the_complete_catalog(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  catalogs: list[tuple[str, ...]] = []
  connected = asyncio.Queue[None]()
  original_connect = _local_rpc_client.Transport.connect

  async def connect(
    transport: _local_rpc_client.Transport,
    host_tools: Sequence[_host_tools.ToolDefinition] = (),
  ) -> grpc.aio.Channel:
    channel = await original_connect(transport, host_tools)
    catalogs.append(tuple(tool.name for tool in host_tools))
    connected.put_nowait(None)
    return channel

  other = _OtherStartupExtension('other_tool')
  monkeypatch.setattr(_local_rpc_client.Transport, 'connect', connect)
  registry = extension.ExtensionRegistry()
  plugin = _StartupExtension()
  await registry.register(plugin, None)
  await registry.register(other, None)
  repl = _repl(
    create_session=True,
    database=tmp_path / 'state.db',
    extension_registry=registry,
  )
  try:
    await repl.start()
    await connected.get()
    await repl._host_tools_registry.wait_for_update()
    assert catalogs == [('test_startup__startup', 'test_other__other_tool')]
    assert repl._state.extensions[plugin.id()] == 1
    await registry.unload(plugin.id())
    await asyncio.wait_for(connected.get(), timeout=5)
    await repl._host_tools_registry.wait_for_update()
    assert catalogs == [
      ('test_startup__startup', 'test_other__other_tool'),
      ('test_other__other_tool',),
    ]
    assert plugin.tools_closed
    repl.dispatch(_actions.TurnStarted())
    previous_client = repl._client
    await registry.register(_StartupExtension(), None)
    await asyncio.sleep(0)
    assert repl._client is previous_client
    assert connected.empty()
    repl.dispatch(_actions.TurnFinished())
    await asyncio.wait_for(connected.get(), timeout=5)
    await repl._host_tools_registry.wait_for_update()
    assert catalogs[-1] == ('test_other__other_tool', 'test_startup__startup')
    other.definition = _host_tools.ToolDefinition(
      'other_new', 'Updated tool', {}
    )
    await registry.refresh_tools(other.id())
    await repl._host_tools_registry.wait_for_update()
    assert len(catalogs) == 4
    assert catalogs[-1] == ('test_startup__startup', 'test_other__other_new')
  finally:
    await repl.close()
    await registry.close()
  assert other.tools_closed


@pytest.mark.asyncio
async def test_user_extension_tool_updates_live_widget_and_commands(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'clock'
  directory.mkdir()
  source = directory / 'extension.py'
  source.write_text("""from mirror.cli import actions, commands, extension
from prompt_toolkit.layout import containers, controls
class Clock(extension.Extension[str, None, None]):
  @classmethod
  def id(cls): return 'clock'
  @staticmethod
  def initial_state(): return 'Clock'
  def render_region(self): return extension.RenderRegion.WIDGET_ABOVE_PROMPT
  def render(self, global_state, extension_state):
    return containers.Window(controls.FormattedTextControl(extension_state), height=1)
  def slash_commands(self):
    return (commands.CommandDef('clock', 'Show clock', lambda context, arguments: actions.ShowText('Clock')),)
""")
  bundled = tmp_path / 'bundled'
  authoring_directory = bundled / 'user_extensions'
  authoring_directory.mkdir(parents=True)
  (authoring_directory / 'extension.py').write_text(
    (
      pathlib.Path(extension.__file__).with_name('extensions')
      / 'user_extensions'
      / 'extension.py'
    )
    .read_text()
    .replace("'core/user-extensions'", "'user_extensions'")
  )
  (authoring_directory / 'SKILL.md').write_text(
    pathlib.Path(_user_extension.__file__).with_name('SKILL.md').read_text()
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path, bundled=bundled)
  await manager.start()
  repl = _repl(extension_registry=registry, extension_manager=manager)
  try:
    completion = repl._components['completion']
    assert isinstance(completion, _completion.Completion)
    initial_state = repl._state
    assert await registry.load(directory) == 'clock'
    assert 'clock' not in initial_state.extensions
    assert repl._state.extensions['clock'] == 'Clock'
    assert repl.get_available_commands()[-1].name == 'clock'
    await registry.unload('clock')
    await repl._extension_host.start()
    assert 'user_extensions' not in registry.extension_ids()
    assert 'user_extensions__extensions' not in {
      tool.name for tool in repl._host_tools_registry.tools()
    }
    await repl._run_slash('/extensions')
    selector = repl._state.selector.selector
    assert selector is not None
    authoring = next(
      choice
      for choice in selector.choices
      if choice.value == 'builtin/user_extensions'
    )
    assert authoring.prefix == '○'
    await asyncio.gather(*repl.dispatch(selector.on_select(authoring.value)))
    repl.dispatch(_actions.CloseSelector())
    height = repl._bottom_pane.preferred_height(80, 40).preferred
    result = await repl._host_tools_registry.execute(
      'user_extensions__extensions', '{"action":"load","name":"clock"}'
    )
    assert result.success
    assert repl._bottom_pane.preferred_height(80, 40).preferred == height + 2
    assert repl._state.extensions['clock'] == 'Clock'
    assert repl.get_available_commands()[-1].name == 'clock'
    source.write_text(
      source.read_text().replace("return 'Clock'", "return 'Time'")
    )
    assert (
      await _user_extension.UserExtensionsTool(manager).execute(
        '{"action":"reload","extension_id":"clock"}'
      )
    ).success
    assert repl._state.extensions['clock'] == 'Time'
    await repl._run_slash('/extensions')
    await asyncio.gather(*repl.dispatch(selector.on_select(authoring.value)))
    assert 'user_extensions' not in registry.extension_ids()
    assert 'user_extensions__extensions' not in {
      tool.name for tool in repl._host_tools_registry.tools()
    }
    assert repl._state.extensions['clock'] == 'Time'
    await repl._run_slash('/extensions')
    assert repl._state.selector.selector is not None
    assert repl._state.selector.selector.id == 'extensions'
    repl.dispatch(_actions.CloseSelector())
    assert (
      await _user_extension.UserExtensionsTool(manager).execute(
        '{"action":"unload","extension_id":"clock"}'
      )
    ).success
    assert repl._bottom_pane.preferred_height(80, 40).preferred == height
    assert (
      _commands.command_named('clock', repl.get_available_commands()) is None
    )
    assert (
      _commands.command_named('extensions', repl.get_available_commands())
      is not None
    )
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
async def test_loading_widget_initializes_dormant_tabs_before_they_receive_actions(
  tmp_path: pathlib.Path,
) -> None:
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path)
  active = _repl(extension_registry=registry)
  dormant = _repl(extension_registry=registry)
  try:
    await active._extension_host.start()
    assert (
      await _user_extension.UserExtensionsTool(manager).execute(
        '{"action":"create","name":"clock"}'
      )
    ).success
    assert (
      await _user_extension.UserExtensionsTool(manager).execute(
        '{"action":"load","name":"clock"}'
      )
    ).success
    dormant.dispatch(_actions.SetActive(True))
    await dormant._extension_host.start()
    assert dormant._state.extensions['clock'] == 'Loading widget…'
    assert active._state.extensions['clock'] == 'Loading widget…'
    status = dormant._extension_host.status('clock')
    assert status is not None
    assert status.status == 'active'
  finally:
    await active.close()
    await dormant.close()
    await registry.close()


async def _run_repl(repl: cli._Repl) -> None:
  """Exercise one agent with the lifecycle owned by the controller."""
  try:
    await repl.start()
    await repl.activate(show_context=False)
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_remote_session_listing_reads_every_page() -> None:
  session_ids = [f'session-{index}' for index in range(1001)]
  remote = _RemoteMirror(session_ids)
  server = grpc.aio.server()
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(remote, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  try:
    listed = await _remote_rpc_client.list_remote_sessions(
      f'127.0.0.1:{port}', secure=False
    )
  finally:
    await server.stop(0)

  assert listed == session_ids
  assert remote.offsets == [0, 1000]


@pytest.mark.asyncio
async def test_remote_session_deletion_calls_mirror_rpc() -> None:
  remote = _RemoteMirror([])
  server = grpc.aio.server()
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(remote, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  try:
    await _remote_rpc_client.delete_remote_session(
      f'127.0.0.1:{port}', 'session-to-delete', secure=False
    )
  finally:
    await server.stop(0)

  assert await remote.deleted.get() == 'session-to-delete'


@pytest.mark.asyncio
async def test_remote_transport_attaches_without_owning_worker() -> None:
  remote = _RemoteEcho()
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(remote, server)
  port = server.add_insecure_port('127.0.0.1:0')
  endpoint = f'127.0.0.1:{port}'
  await server.start()
  received: list[agent_pb2.SessionMessage] = []
  acknowledged = asyncio.Event()
  errors: list[str] = []

  def on_message(message: agent_pb2.SessionMessage) -> None:
    received.append(message)
    if message.WhichOneof('payload') == 'input_ack':
      acknowledged.set()

  async def on_error(message: str) -> None:
    errors.append(message)

  client = _rpc_client.Client(
    _remote_rpc_client.Transport(endpoint, secure=False),
    'remote-session',
    on_message=on_message,
    on_error=on_error,
  )
  try:
    await client.start(replay=True)
    client.send(
      agent_pb2.ClientMessage(
        session_id='remote-session',
        message_id='remote-input',
        message=agent_pb2.AgentMessage(
          id='remote-input',
          chat=agent_pb2.ChatRecord(
            user=agent_pb2.UserMessage(content='continue remotely'),
            success=True,
          ),
        ),
      )
    )
    await asyncio.wait_for(acknowledged.wait(), timeout=1)
    user_input = await remote.requests.get()

    assert user_input.WhichOneof('payload') == 'message'
    assert [message.id for message in await client.messages()] == [
      'remote-history'
    ]
    assert [message.WhichOneof('payload') for message in received] == [
      'agent_message',
      'replay_done',
      'input_ack',
    ]
    assert not errors
  finally:
    await client.close()
    probe = grpc.aio.insecure_channel(endpoint)
    try:
      await asyncio.wait_for(probe.channel_ready(), timeout=1)
    finally:
      await probe.close()
      await server.stop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'finish_loading, interrupt', [(False, False), (True, False), (True, True)]
)
async def test_run_tui_accepts_input_before_extensions_finish_loading(
  tmp_path: pathlib.Path,
  capfd: pytest.CaptureFixture[str],
  monkeypatch: pytest.MonkeyPatch,
  finish_loading: bool,
  interrupt: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.chdir(tmp_path)
  loading = asyncio.Event()
  release = asyncio.Event()
  cancelled = asyncio.Event()
  active: list[cli._Repl] = []
  loaded = _StartupExtension()
  activate = cli._Repl.activate

  async def load_extensions(
    manager: user_extensions.UserExtensions, *, trust_extensions: bool = False
  ) -> None:
    del trust_extensions
    assert active[0]._app.render_counter > 0
    loading.set()
    try:
      await release.wait()
      await manager.registry.register(loaded, None)
    except asyncio.CancelledError:
      cancelled.set()
      raise

  async def activate_repl(
    repl: cli._Repl,
    *,
    show_context: bool = True,
    label: str | None = None,
    navigation: _containers.AnyContainer | None = None,
    before_render: Callable[[_application.Application[str | None]], None]
    | None = None,
    screen: cli._workspace_application.WorkspaceApplication | None = None,
  ) -> str | None:
    active.append(repl)
    return await activate(
      repl,
      show_context=show_context,
      label=label,
      navigation=navigation,
      before_render=before_render,
      screen=screen,
    )

  monkeypatch.setattr(user_extensions.UserExtensions, 'start', load_extensions)
  monkeypatch.setattr(
    user_extensions.UserExtensions, 'apply_settings', mock.AsyncMock()
  )
  monkeypatch.setattr(cli._Repl, 'activate', activate_repl)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    running = asyncio.create_task(
      cli.run_tui(
        [sys.executable, '-c', _ECHO_WORKER_SOURCE],
        {'conversation_id': 'startup', 'model': 'test', 'create_session': True},
        database=tmp_path / 'cli.db',
      )
    )
    try:
      await asyncio.wait_for(loading.wait(), timeout=2)
      repl = active[0]
      assert repl._client is None
      pipe.send_text('hello')

      async def wait_for_input() -> None:
        while repl._input.text != 'hello':
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_input(), timeout=2)
      if finish_loading:
        pipe.send_text('\r')

        async def wait_for_queue() -> None:
          while repl._state.turn.queued != ('hello',):
            await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_queue(), timeout=2)
        assert repl._client is None
        if interrupt:
          repl.dispatch(_actions.InterruptRequested())
          assert repl._state.turn.submitted_inputs == ()
          assert repl._input.text == 'hello'
        release.set()
        await asyncio.wait_for(repl._worker_ready.wait(), timeout=5)
        assert 'test_startup__startup' in {
          tool.name for tool in repl._host_tools_registry.tools()
        }

        async def wait_for_reply() -> None:
          while not any(
            entry.chat.assistant.content == 'echo: hello'
            for entry in repl.transcript_entries()
          ):
            await asyncio.sleep(0)

        if interrupt:
          assert repl._input.text == 'hello'
          assert repl._state.turn.submitted_inputs == ()
          assert not repl.transcript_entries()
          pipe.send_text('\r')
        await asyncio.wait_for(wait_for_reply(), timeout=5)
      else:
        repl._input.text = ''
      pipe.send_text('\x04')
      await asyncio.wait_for(running, timeout=5)
      assert cancelled.is_set() is not finish_loading
      assert 'Bye. Continue with mirror --continue' in capfd.readouterr().out
    finally:
      running.cancel()
      await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('api', 'provider'),
  [('reflection', 'Reflection'), ('anthropic-messages', 'Anthropic')],
)
@pytest.mark.parametrize('cancel', [False, True])
async def test_api_key_prompt_renders_before_extensions_and_keeps_key_private(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  capfd: pytest.CaptureFixture[str],
  cancel: bool,
  api: str,
  provider: str,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  key_path = tmp_path / '.mirror' / f'{provider.lower()}_api_key'
  monkeypatch.chdir(tmp_path)
  active: list[cli._Repl] = []
  extensions_started = asyncio.Event()
  activate = cli._Repl.activate
  screen = pyte.Screen(120, 32)
  screen.set_mode(pyte.modes.LNM)
  stream = pyte.Stream(screen)
  monkeypatch.setattr(
    cli._logging.defaults,
    'create_output',
    lambda stdout: _vt100.Vt100_Output(
      stdout,
      lambda: _data_structures.Size(rows=32, columns=120),
      term='xterm-256color',
      enable_cpr=False,
    ),
  )

  async def activate_repl(
    repl: cli._Repl,
    *,
    show_context: bool = True,
    label: str | None = None,
    navigation: _containers.AnyContainer | None = None,
    before_render: Callable[[_application.Application[str | None]], None]
    | None = None,
    screen: cli._workspace_application.WorkspaceApplication | None = None,
  ) -> str | None:
    active.append(repl)
    return await activate(
      repl,
      show_context=show_context,
      label=label,
      navigation=navigation,
      before_render=before_render,
      screen=screen,
    )

  async def load_extensions(
    manager: user_extensions.UserExtensions, *, trust_extensions: bool = False
  ) -> None:
    del manager, trust_extensions
    assert key_path.read_text() == 'synthetic-api-key'
    extensions_started.set()

  monkeypatch.setattr(cli._Repl, 'activate', activate_repl)
  monkeypatch.setattr(user_extensions.UserExtensions, 'start', load_extensions)
  monkeypatch.setattr(
    user_extensions.UserExtensions, 'apply_settings', mock.AsyncMock()
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    running = asyncio.create_task(
      cli.run_tui(
        [sys.executable, '-c', _ECHO_WORKER_SOURCE],
        {
          'conversation_id': 'onboarding',
          'model': _models.default_model(api),
          'api': api,
          'create_session': True,
        },
        database=tmp_path / 'cli.db',
      )
    )
    try:

      async def wait_for_prompt() -> None:
        while True:
          stream.feed(capfd.readouterr().out)
          displayed = '\n'.join(screen.display)
          if 'Paste your key below' in displayed and 'API key ›' in displayed:
            return
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_prompt(), timeout=3)
      displayed = '\n'.join(screen.display)
      assert displayed.index('mirror') < displayed.index(
        f'Connect to {provider}'
      )
      assert _models.default_model(api) in displayed
      assert 'Enter to connect' in displayed
      assert '8;id=' not in displayed
      assert '8;;' not in displayed
      assert 'auto ·' not in displayed
      assert 'loading…' not in displayed
      repl = active[0]
      assert repl._app.render_counter > 0
      assert repl._client is None
      assert not extensions_started.is_set()
      pipe.send_text('synthetic-api-key')

      async def wait_for_key() -> None:
        while (
          repl._secret_input is not None
          and repl._secret_input.text != 'synthetic-api-key'
        ):
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_key(), timeout=3)
      assert repl._input.text == ''
      assert 'synthetic-api-key' not in repr(repl._state)
      assert not repl.transcript_entries()
      pipe.send_text('\x03' if cancel else '\r')
      if not cancel:
        await asyncio.wait_for(extensions_started.wait(), timeout=3)

        async def wait_for_connected() -> None:
          while True:
            stream.feed(capfd.readouterr().out)
            if f'Connect to {provider}' not in '\n'.join(screen.display):
              return
            await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_connected(), timeout=3)
        displayed = '\n'.join(screen.display)
        assert 'Reflection mirror' in displayed
        assert f'Connect to {provider}' not in displayed
        assert 'synthetic-api-key' not in displayed
        await asyncio.wait_for(repl._worker_ready.wait(), timeout=5)
        assert repl._secret_input is None
        pipe.send_text('\x04')
      await asyncio.wait_for(running, timeout=5)
      assert extensions_started.is_set() is not cancel
      assert key_path.exists() is not cancel
    finally:
      running.cancel()
      await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_remote_start_preserves_input_submitted_during_creation(
  tmp_path: pathlib.Path,
) -> None:
  remote = _RemoteEcho(echo_user=True)
  start_gate = asyncio.Event()
  mirror = _RemoteMirror([], start_gate=start_gate)
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(remote, server)
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(mirror, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  repl = _repl(
    database=tmp_path / 'state.db',
    create_session=True,
    model='server default',
    remote_endpoint=f'127.0.0.1:{port}',
    remote_secure=False,
  )
  repl._worker_configuration = repl._worker_configuration.updated({
    'api': 'openai-codex-responses',
    'base_url': 'https://local-codex.invalid',
  })
  startup = asyncio.create_task(repl._start_worker(replay=True))

  try:
    await asyncio.wait_for(mirror.started.get(), timeout=1)
    repl.dispatch(_actions.WorkerLoadingStarted())
    repl._input.text = 'submitted while creating'
    _submit(repl)
    assert repl._state.turn.queued == ('submitted while creating',)

    start_gate.set()
    await asyncio.wait_for(startup, timeout=1)
    requests = [
      await asyncio.wait_for(remote.requests.get(), timeout=1) for _ in range(3)
    ]

    async def wait_until_rendered() -> None:
      while repl._state.turn.queued:
        await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_rendered(), timeout=1)
    assert [request.WhichOneof('payload') for request in requests] == [
      'set_inference_config',
      'message',
      'set_approval_mode',
    ]
    assert requests[0].set_inference_config == agent_pb2.InferenceConfig(
      model='', reasoning_effort='medium'
    )
    assert requests[1].message.chat.user.content == 'submitted while creating'
  finally:
    start_gate.set()
    await asyncio.gather(startup, return_exceptions=True)
    await repl._shutdown_worker()
    await server.stop(0)


@pytest.mark.asyncio
async def test_new_session_during_startup_does_not_send_old_pending_input() -> (
  None
):
  remote = _RemoteEcho()
  mirror = _RemoteMirror([])
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(remote, server)
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(mirror, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  repl = _repl(
    create_session=True,
    remote_endpoint=f'127.0.0.1:{port}',
    remote_secure=False,
  )
  loaded = asyncio.Event()

  async def load_extensions() -> None:
    await loaded.wait()

  repl._extensions_loaded = asyncio.create_task(load_extensions())
  repl.dispatch(_actions.ReplayFinished(False))
  repl._input.text = 'only for the original conversation'
  (startup,) = repl.dispatch(_actions.ComposerSubmit())
  await asyncio.sleep(0)
  repl._input.text = 'newer draft'
  switch = repl.dispatch(_actions.NewSession())
  assert repl._workspace_effects is not None
  try:
    _, effect = await asyncio.wait_for(repl._workspace_effects.get(), timeout=2)
    assert isinstance(effect, _effects.RetargetRemote)
    replacement = repl._state.session.conversation_id
    assert replacement != 'conv-1'
    loaded.set()
    await asyncio.wait_for(asyncio.gather(startup, *switch), timeout=2)

    assert (
      await asyncio.wait_for(mirror.started.get(), timeout=2) == replacement
    )
    assert mirror.started.empty()
    configuration = await asyncio.wait_for(remote.requests.get(), timeout=2)
    permissions = await asyncio.wait_for(remote.requests.get(), timeout=2)
    assert configuration.WhichOneof('payload') == 'set_inference_config'
    assert permissions.WhichOneof('payload') == 'set_approval_mode'
    assert repl._state.turn.pending_instructions == ()
    assert repl._input.text == 'only for the original conversation\nnewer draft'

    repl._input.text = 'explicitly submitted to the new conversation'
    _submit(repl)
    request = await asyncio.wait_for(remote.requests.get(), timeout=2)
    assert request.session_id == replacement
    assert request.message.chat.user.content == (
      'explicitly submitted to the new conversation'
    )
  finally:
    loaded.set()
    await asyncio.gather(startup, *switch, return_exceptions=True)
    await repl.close()
    await server.stop(0)


@pytest.mark.asyncio
async def test_remote_repl_creates_its_session_in_the_selected_workspace(
  tmp_path: pathlib.Path,
) -> None:
  mirror = _RemoteMirror([])
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(_RemoteEcho(), server)
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(mirror, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  workspace = mirror_pb2.Workspace(
    name='project',
    repository_url='https://github.com/acme/project',
    revision='main',
    directory='src',
  )
  repl = _repl(
    database=tmp_path / 'state.db',
    create_session=True,
    remote_endpoint=f'127.0.0.1:{port}',
    remote_secure=False,
    remote_workspace=workspace,
  )
  repl._active = False

  try:
    await repl._start_worker(replay=True)
  finally:
    await repl._shutdown_worker()
    await server.stop(0)

  assert [request.workspace for request in mirror.start_requests] == [workspace]


@pytest.mark.asyncio
async def test_remote_repl_sets_approval_after_session_is_ready(
  tmp_path: pathlib.Path,
) -> None:
  remote = _RemoteEcho()
  mirror = _RemoteMirror([])
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(remote, server)
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(mirror, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  repl = _repl(
    database=tmp_path / 'state.db',
    create_session=True,
    remote_endpoint=f'127.0.0.1:{port}',
    remote_secure=False,
  )
  repl._active = False

  try:
    await repl._start_worker(replay=True)
    started = await asyncio.wait_for(mirror.started.get(), timeout=1)
    inference = await asyncio.wait_for(remote.requests.get(), timeout=1)
    approval = await asyncio.wait_for(remote.requests.get(), timeout=1)

    assert started == 'conv-1'
    assert inference.WhichOneof('payload') == 'set_inference_config'
    assert inference.set_inference_config == agent_pb2.InferenceConfig(
      model='m-test', reasoning_effort='medium'
    )
    assert approval.WhichOneof('payload') == 'set_approval_mode'
    assert approval.set_approval_mode.mode == agent_pb2.APPROVAL_MODE_AUTO
    assert remote.requests.empty()
    await repl._run_slash('/compact')
    compact = await asyncio.wait_for(remote.requests.get(), timeout=1)
    assert compact.WhichOneof('payload') == 'compact'
    assert compact.session_id == 'conv-1'
    assert compact.message_id
    assert remote.requests.empty()
    assert repl._client is not None
    assert repl._client._host_tools_registry is None  # pyright: ignore[reportPrivateUsage]
    assert isinstance(
      repl._client.transport,
      _remote_rpc_client.Transport,
    )
    with (
      cli_store.Store(tmp_path / 'state.db') as store,
      pytest.raises(cli_store.ConversationNotFoundError),
    ):
      store.get_info('conv-1')
  finally:
    await repl._shutdown_worker()
    await server.stop(0)


@pytest.mark.asyncio
async def test_fork_safe_subprocesses_do_not_leak_grpc_logs_to_tui(
  tmp_path: pathlib.Path,
  capfd: pytest.CaptureFixture[str],
) -> None:
  errors: list[str] = []

  async def on_error(message: str) -> None:
    errors.append(message)

  transports: list[_local_rpc_client.Transport] = []

  def client() -> _rpc_client.Client:
    transport = _local_rpc_client.Transport(
      [sys.executable, '-c', _ECHO_WORKER_SOURCE],
      {'conversation_id': 'conv-1'},
      workspace=tmp_path,
      log_dir=tmp_path,
    )
    transports.append(transport)
    return _rpc_client.Client(
      transport,
      'conv-1',
      on_message=lambda _message: None,
      on_error=on_error,
    )

  first = client()
  second = client()
  try:
    await first.start(replay=False)
    workspace.git_workspace_key.cache_clear()
    workspace.git_workspace_key(str(tmp_path))
    await second.start(replay=False)
    first_process = transports[0].process
    second_process = transports[1].process
    assert first_process is not None and first_process.returncode is None
    assert second_process is not None and second_process.returncode is None
    os.write(2, b'native client log\n')
  finally:
    await first.close()
    await second.close()

  assert not errors
  terminal_stderr = capfd.readouterr().err
  log = (tmp_path / 'stderr').read_text()
  assert 'native client log' in terminal_stderr
  assert 'fork_posix.cc' not in terminal_stderr
  assert 'native client log' not in log
  assert f'cwd={tmp_path}' in (tmp_path / 'stdout').read_text()


@pytest.mark.asyncio
async def test_rpc_client_queues_requests_before_worker_connects(
  tmp_path: pathlib.Path,
) -> None:
  completed = asyncio.Event()

  def on_message(message: agent_pb2.SessionMessage) -> None:
    if message.runtime_delta.HasField('turn_completed'):
      completed.set()

  async def on_error(message: str) -> None:
    pytest.fail(message)

  transport = _local_rpc_client.Transport(
    [sys.executable, '-c', _ECHO_WORKER_SOURCE],
    {'conversation_id': 'conv-1'},
    workspace=tmp_path,
    log_dir=tmp_path,
  )
  client = _rpc_client.Client(
    transport,
    'conv-1',
    on_message=on_message,
    on_error=on_error,
  )
  request = agent_pb2.ClientMessage(session_id='conv-1', message_id='message-1')
  request.message.id = request.message_id
  request.message.chat.user.content = 'queued before startup'
  request.message.chat.success = True

  client.send(request)
  try:
    await client.start(replay=False)
    await asyncio.wait_for(completed.wait(), timeout=5)
  finally:
    await client.close()


def _record_restarts(
  repl: cli._Repl, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[dict[str, object], bool, str | None]]:
  calls: list[tuple[dict[str, object], bool, str | None]] = []

  async def restart(
    updates: dict[str, object],
    *,
    replay: bool,
    opening_conversation: str | None = None,
  ) -> None:
    calls.append((updates, replay, opening_conversation))

  monkeypatch.setattr(repl, '_restart_worker', restart)
  return calls


class _SizedOutput(_output_base.DummyOutput):
  def __init__(self, *, rows: int, columns: int) -> None:
    self.size = _data_structures.Size(rows=rows, columns=columns)

  @override
  def get_size(self) -> _data_structures.Size:
    return self.size


@contextlib.asynccontextmanager
async def _visual_repl(
  columns: int = 80,
  rows: int = 24,
  registry: extension.ExtensionRegistry | None = None,
  initial_theme: str = 'dark',
) -> AsyncIterator[tuple[cli._Repl, io.BytesIO]]:
  """Capture the real layout and styles without a worker or personal paths."""
  stream = io.BytesIO()
  output = _vt100.Vt100_Output(
    io.TextIOWrapper(stream, encoding='utf-8'),
    lambda: _data_structures.Size(rows=rows, columns=columns),
    term='xterm-256color',
    default_color_depth=_output_base.ColorDepth.DEPTH_24_BIT,
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl(
      model='reflection',
      workspace_path=pathlib.Path('/workspace/mirror'),
      extension_registry=registry,
      initial_theme=initial_theme,
    )
    try:
      with _current.set_app(repl._app):
        yield repl, stream
    finally:
      await repl.close()


def _capture_frame(repl: cli._Repl, stream: io.BytesIO) -> str:
  repl._app.before_render.fire()
  repl._app.renderer.render(repl._app, repl._app.layout)
  return stream.getvalue().decode('utf-8')


@pytest.mark.asyncio
async def test_welcome_decoration_does_not_change_transcript_width() -> None:
  async with _visual_repl(200, 45) as (repl, stream):
    frame = _capture_frame(repl, stream)
    assert '⠂' in frame
    assert repl._component_state().layout.columns == 80
    assert repl._input_window.render_info is not None
    assert repl._input_window.render_info.window_width == 78
    assert repl._app.layout.current_control is repl._composer_control

    repl._handle_message(_delta({'role': 'user', 'content': 'Hello.'}))
    terminal = pyte.Screen(200, 45)
    pyte.Stream(terminal).feed(_capture_frame(repl, stream))
    assert '⠂' not in '\n'.join(terminal.display)
    assert repl._component_state().layout.columns == 200
    assert repl._input_window.render_info.window_width == 198
    assert repl._app.layout.current_control is repl._composer_control


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('variable', 'value'), [('NO_COLOR', '1'), ('TERM', 'dumb')]
)
async def test_color_disabled_completion_has_no_filled_edges(
  monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
  monkeypatch.setenv(variable, value)
  async with _visual_repl(initial_theme='light') as (repl, stream):
    repl._input.text = '/'
    terminal = pyte.Screen(80, 24)
    pyte.Stream(terminal).feed(_capture_frame(repl, stream))
    displayed = '\n'.join(terminal.display)
    assert 'Commands' in displayed
    assert '▀' * 8 not in displayed
    assert '▄' * 8 not in displayed
    assert repl._input.text == '/'


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['mono', 'terminal'])
async def test_theme_change_repaints_edges_without_losing_draft(
  monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setenv('TERM', 'xterm-256color')
  async with _visual_repl() as (repl, stream):
    repl._input.text = '/'
    terminal = pyte.Screen(80, 24)
    pyte.Stream(terminal).feed(_capture_frame(repl, stream))
    assert '▀' * 8 in '\n'.join(terminal.display)
    assert '▄' * 8 in '\n'.join(terminal.display)

    await _dispatch_and_wait(
      repl, _actions.ExtensionConfigChanged(themes.EXTENSION_ID, {'name': name})
    )
    terminal = pyte.Screen(80, 24)
    pyte.Stream(terminal).feed(_capture_frame(repl, stream))
    assert '▀' * 8 not in '\n'.join(terminal.display)
    assert '▄' * 8 not in '\n'.join(terminal.display)
    assert 'Commands' in '\n'.join(terminal.display)
    assert repl._input.text == '/'
    assert repl._app.layout.current_control is repl._composer_control

    await _dispatch_and_wait(
      repl,
      _actions.ExtensionConfigChanged(themes.EXTENSION_ID, {'name': 'dark'}),
    )
    terminal = pyte.Screen(80, 24)
    pyte.Stream(terminal).feed(_capture_frame(repl, stream))
    assert '▀' * 8 in '\n'.join(terminal.display)
    assert '▄' * 8 in '\n'.join(terminal.display)
    assert repl._input.text == '/'
    assert repl._app.layout.current_control is repl._composer_control


def _delta(
  message: dict[str, Any],
  display: str | None = None,
  entry_id: str | None = None,
) -> agent_pb2.SessionMessage:
  role = message['role']
  chat = agent_pb2.ChatRecord(success=message.get('success', True))
  if display is not None:
    chat.display = display
  if role == 'system':
    chat.system.content = message['content']
  elif role == 'user':
    chat.user.content = message['content']
  elif role == 'assistant':
    assistant = chat.assistant
    if message.get('content') is not None:
      assistant.content = message['content']
    if message.get('reasoning_content') is not None:
      assistant.reasoning_content = message['reasoning_content']
    if message.get('refusal') is not None:
      assistant.refusal = message['refusal']
    for call in message.get('tool_calls') or []:
      function = call.get('function', {})
      arguments = function.get('arguments', '{}')
      assistant.tool_calls.add(
        id=call.get('id', ''),
        type=call.get('type', ''),
        function=agent_pb2.FunctionCall(
          name=function.get('name', ''),
          arguments=arguments,
        ),
      )
  elif role == 'tool':
    chat.tool.tool_call_id = message.get('tool_call_id', '')
    chat.tool.content = message.get('content', '')
    if message.get('name') is not None:
      chat.tool.name = message['name']
  else:
    raise ValueError(f'unknown role: {role}')
  return agent_pb2.SessionMessage(
    agent_message=agent_pb2.AgentMessage(id=entry_id or '', chat=chat)
  )


def _replay_done(
  conversation_id: str = '',
  *,
  has_user_message: bool = False,
) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    replay_done=agent_pb2.ReplayDone(
      conversation_id=conversation_id,
      has_user_message=has_user_message,
    )
  )


def _runtime(delta: agent_pb2.RuntimeDelta) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(runtime_delta=delta)


def test_worker_messages_update_cli_projection(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'state.db'
  repl = _repl(database=database, create_session=True)

  repl._on_worker_message(  # pyright: ignore[reportPrivateUsage]
    _delta({'role': 'user', 'content': 'project this'}, entry_id='user-1')
  )

  with cli_store.Store(database) as store:
    assert store.get_info('conv-1').title == 'project this'
    assert store.recent_user_messages('conv-1') == ['project this']


def _stream_content(content: str) -> agent_pb2.SessionMessage:
  return _runtime(
    agent_pb2.RuntimeDelta(
      stream_content=agent_pb2.StreamContent(content=content)
    )
  )


def _stream_reasoning(reasoning: str) -> agent_pb2.SessionMessage:
  return _runtime(
    agent_pb2.RuntimeDelta(
      stream_reasoning=agent_pb2.StreamReasoning(reasoning=reasoning)
    )
  )


def _stream_refusal(refusal: str) -> agent_pb2.SessionMessage:
  return _runtime(
    agent_pb2.RuntimeDelta(
      stream_refusal=agent_pb2.StreamRefusal(refusal=refusal)
    )
  )


def _stream_tool_call(
  *, index: int, name: str = '', arguments: str = ''
) -> agent_pb2.SessionMessage:
  return _runtime(
    agent_pb2.RuntimeDelta(
      stream_tool_call=agent_pb2.StreamToolCall(
        index=index, name=name, arguments=arguments
      )
    )
  )


def _stream_aborted() -> agent_pb2.SessionMessage:
  return _runtime(agent_pb2.RuntimeDelta(stream_aborted=empty_pb2.Empty()))


def _turn_completed() -> agent_pb2.SessionMessage:
  return _runtime(agent_pb2.RuntimeDelta(turn_completed=empty_pb2.Empty()))


def _window_text(
  window: _containers.AnyContainer,
) -> list[tuple[str, str]] | str:
  assert isinstance(window, _containers.Window)
  assert isinstance(window.content, _controls.FormattedTextControl)
  assert not callable(window.content.text)
  return cast(list[tuple[str, str]] | str, window.content.text)


def _render_banner(repl: cli._Repl) -> str:
  container = repl._components['banner'].render(repl._state)
  assert isinstance(container, _containers.HSplit)
  window = container.children[0]
  assert isinstance(window, _containers.Window)
  content = window.content.create_content(repl._state.layout.columns, 100)
  return '\n'.join(
    ''.join(text for _, text in content.get_line(line))
    for line in range(content.line_count)
  )


def _render_welcome_notices(repl: cli._Repl) -> str:
  return _rendering.render_blocks(
    list(repl._transcript.welcome_notices()), 1000
  )


def _render_activity(repl: cli._Repl) -> list[tuple[str, str]]:
  window = _activity.Activity().render(repl._state)
  return cast(list[tuple[str, str]], _window_text(window))


def _render_footer(repl: cli._Repl) -> list[tuple[str, str]]:
  container = repl._components['status_bar'].render(repl._state)
  assert isinstance(container, _containers.VSplit)
  return cast(list[tuple[str, str]], _window_text(container.children[0]))


def _component_text(
  container: _containers.AnyContainer,
) -> list[tuple[str, str]]:
  return cast(list[tuple[str, str]], _window_text(container))


def _component_height(container: _containers.AnyContainer) -> int:
  return container.preferred_height(80, 100).preferred


def _render_tasks(repl: cli._Repl) -> str:
  return _fragments_text(_component_text(_tasks.Tasks().render(repl._state)))


def _render_completion(repl: cli._Repl) -> list[tuple[str, str]]:
  component = _completion.Completion()
  return _component_text(component.render(repl._state))


def _completion_height(repl: cli._Repl) -> int:
  component = _completion.Completion()
  return _component_height(component.render(repl._state))


def _completion_preview(repl: cli._Repl) -> list[tuple[str, str]] | None:
  return _completion.prompt_preview(repl._state)


def _render_resume(repl: cli._Repl) -> list[tuple[str, str]]:
  return _component_text(
    (
      _selector.Selector() if repl._state.selector.active else _resume.Resume()
    ).render(repl._state)
  )


def _resume_height(repl: cli._Repl) -> int:
  return _component_height(
    (
      _selector.Selector() if repl._state.selector.active else _resume.Resume()
    ).render(repl._state)
  )


def _render_rewind_picker(repl: cli._Repl) -> list[tuple[str, str]]:
  return _component_text(
    (
      _selector.Selector()
      if repl._state.selector.active
      else _rewind.RewindPicker()
    ).render(repl._state)
  )


def _render_rewind_banner(repl: cli._Repl) -> list[tuple[str, str]]:
  return _component_text(_rewind.RewindBanner().render(repl._state))


def _render_approval(repl: cli._Repl) -> list[str]:
  width = repl._state.layout.columns
  container = _selector.Selector().render(repl._state)
  height = container.preferred_height(width, 100).preferred
  target = _screen.Screen()
  container.write_to_screen(
    target,
    _mouse_handlers.MouseHandlers(),
    _screen.WritePosition(0, 0, width, height),
    '',
    True,
    None,
  )
  return [
    ''.join(
      target.data_buffer[row][column].char for column in range(width)
    ).rstrip()
    for row in range(height)
  ]


def _running_call_hints(
  repl: cli._Repl,
) -> dict[str, tuple[str, str, str]]:
  return {call.call_id: call.render_hint for call in repl._state.stream.running}


def _running_call_ids(repl: cli._Repl) -> tuple[str, ...]:
  return tuple(call.call_id for call in repl._state.stream.running)


def _pending_call_hints(
  repl: cli._Repl,
) -> tuple[tuple[str, str, str], ...]:
  return tuple(call.render_hint for call in repl._state.stream.pending)


def _task_delta(
  task_id: str,
  display: str,
  status: str = 'running',
  *,
  detached: bool = True,
) -> agent_pb2.SessionMessage:
  statuses = {
    'running': agent_pb2.TASK_STATUS_RUNNING,
    'succeeded': agent_pb2.TASK_STATUS_SUCCEEDED,
    'failed': agent_pb2.TASK_STATUS_FAILED,
    'cancelled': agent_pb2.TASK_STATUS_CANCELLED,
  }
  return agent_pb2.SessionMessage(
    runtime_delta=agent_pb2.RuntimeDelta(
      task_update=agent_pb2.TaskUpdate(
        id=task_id,
        tool_name='bash',
        display=display,
        status=statuses[status],
        detached=detached,
      )
    )
  )


def _task_result(
  task_id: str,
  tool_name: str,
  content: str,
  *,
  success: bool = True,
) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    agent_message=agent_pb2.AgentMessage(
      task_result=agent_pb2.TaskResultRecord(
        task_id=task_id,
        tool_name=tool_name,
        content=content,
        success=success,
      )
    )
  )


def _approval_requested(
  call_id: str, name: str, arguments: str = '{}'
) -> agent_pb2.SessionMessage:
  return agent_pb2.SessionMessage(
    hitl_request=agent_pb2.HITLRequest(
      call=agent_pb2.ToolCall(
        id=call_id,
        type='function',
        function=agent_pb2.FunctionCall(name=name, arguments=arguments),
      )
    )
  )


def _stream_final(
  message: dict[str, Any], *, completion_tokens: int | None = None
) -> agent_pb2.SessionMessage:
  payload = _delta(message)
  payload.stream_final = True
  if completion_tokens is not None:
    payload.agent_message.chat.usage.CopyFrom(
      agent_pb2.Usage(
        prompt_tokens=100_000, completion_tokens=completion_tokens
      )
    )
  return payload


def _handle(repl: cli._Repl, delta: agent_pb2.SessionMessage) -> Any:
  """An entry arriving from the running turn."""
  return repl._live_entry(delta.agent_message)


def _start_calls(
  repl: cli._Repl,
  tool_calls: list[dict[str, Any]],
) -> None:
  """The live assistant turn that starts `tool_calls`."""
  _handle(
    repl,
    _delta(
      {'role': 'assistant', 'content': None, 'tool_calls': tool_calls},
    ),
  )


def _capture_paints(
  repl: cli._Repl,
  monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, _terminal_mod.TerminalClear]]:
  """Each write the terminal receives: its ANSI, and how it cleared first."""
  paints: list[tuple[str, _terminal_mod.TerminalClear]] = []

  async def run_in_terminal(callback: Any) -> None:
    callback()

  def print_ansi(
    ansi: str,
    *,
    clear: _terminal_mod.TerminalClear = _terminal_mod.TerminalClear.KEEP,
  ) -> None:
    paints.append((ansi, clear))

  monkeypatch.setattr(cli, '_run_in_terminal', run_in_terminal)
  monkeypatch.setattr(repl._terminal, 'print_ansi', print_ansi)
  return paints


@pytest.mark.asyncio
async def test_background_start_stays_inactive_and_clears_replay_buffers(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))

  async def start_worker(*, replay: bool) -> None:
    assert replay
    assert repl._state.activity.busy
    assert repl._state.activity.status == 'loading…'
    repl._terminal.queue_emit([_text.Text('stale')])
    repl._handle_message(_replay_done('conv-1'))

  monkeypatch.setattr(repl, '_start_worker', start_worker)

  await repl.start()

  assert not repl._state.lifecycle.active
  assert not repl._state.activity.busy
  assert repl._state.activity.status == ''
  assert repl._terminal.header == []
  assert repl._terminal.emit_buffer == []


@pytest.mark.asyncio
async def test_activation_preserves_background_notices(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl.dispatch(_actions.SetReplaying(False))
  repl._terminal.queue_emit([_text.Text('transcript duplicate')])
  repl.show_error('background failure')
  paints: list[tuple[str, _terminal_mod.TerminalClear]] = []

  async def run_async() -> str | None:
    return None

  def print_ansi(
    ansi: str,
    *,
    clear: _terminal_mod.TerminalClear = _terminal_mod.TerminalClear.KEEP,
  ) -> None:
    paints.append((ansi, clear))

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(repl._terminal, 'print_ansi', print_ansi)

  await repl.activate(show_context=True)

  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'background failure' in ansi
  assert 'm-test' not in ansi
  assert 'm-test' in _render_banner(repl)
  plain = _text.Text.from_ansi(ansi).plain
  assert repl.workspace.name in plain
  assert 'transcript duplicate' not in ansi


@pytest.mark.asyncio
@pytest.mark.parametrize('show_context', [False, True])
async def test_mcp_startup_warning_survives_activation_and_replay(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
  show_context: bool,
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path))
  _mcp.add_server(
    _mcp.CatalogServer(
      'slack', 'Slack', 'https://slack.test/mcp', _mcp.CatalogOAuth()
    )
  )
  _mcp.add_server(
    _mcp.CatalogServer(
      'notion', 'Notion', 'https://notion.test/mcp', _mcp.CatalogOAuth()
    )
  )
  registry = extension.ExtensionRegistry()
  await registry.register(_connections.ConnectionsExtension(), None)
  repl = _repl(extension_registry=registry)
  repl.dispatch(_actions.SetActive(False))
  paints: list[tuple[str, _terminal_mod.TerminalClear]] = []

  async def run_async() -> str | None:
    repl._handle_message(_replay_done('conv-1'))
    await asyncio.gather(*tuple(repl._app._background_tasks))
    return None

  def print_ansi(
    ansi: str,
    *,
    clear: _terminal_mod.TerminalClear = _terminal_mod.TerminalClear.KEEP,
  ) -> None:
    paints.append((ansi, clear))

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(repl._terminal, 'print_ansi', print_ansi)
  try:
    await repl.activate(show_context=show_context)
    assert 'Sign-in required' not in paints[-1][0]
    text = _render_welcome_notices(repl)
    assert (
      'Sign-in required on this machine: /connect notion, /connect slack'
      in text
    )
    assert text.count('Sign-in required') == 1
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
async def test_supervisor_error_is_emitted_during_replay(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  emitted: list[str] = []

  async def emit(renderables: list[_console.RenderableType]) -> None:
    emitted.extend(_render_to_text(renderable) for renderable in renderables)

  monkeypatch.setattr(repl._terminal, 'emit', emit)

  repl.show_error('drop failed')
  await asyncio.gather(*tuple(repl._app._background_tasks))

  assert emitted == ['✗ drop failed\n']


@pytest.mark.asyncio
async def test_activation_of_a_lazy_worktree_clears_the_previous_conversation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # A never-started worktree has no ReplayDone event to replace its screen.
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  paints: list[tuple[str, _terminal_mod.TerminalClear]] = []

  async def run_async() -> str | None:
    return None

  def print_ansi(
    ansi: str,
    *,
    clear: _terminal_mod.TerminalClear = _terminal_mod.TerminalClear.KEEP,
  ) -> None:
    paints.append((ansi, clear))

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(repl._terminal, 'print_ansi', print_ansi)

  await repl.activate(show_context=True, label='feature alias')

  assert len(paints) == 1
  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  plain = _text.Text.from_ansi(ansi).plain
  assert 'feature alias' in plain
  assert 'm-test' not in ansi
  assert 'm-test' in _render_banner(repl)


@pytest.mark.asyncio
async def test_flush_scrollback_promotes_a_worktree_tab_paint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl._terminal.flush_scrollback = True
  paints: list[tuple[str, _terminal_mod.TerminalClear]] = []

  async def run_async() -> str | None:
    return None

  def print_ansi(
    ansi: str,
    *,
    clear: _terminal_mod.TerminalClear = _terminal_mod.TerminalClear.KEEP,
  ) -> None:
    paints.append((ansi, clear))

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(repl._terminal, 'print_ansi', print_ansi)

  await repl.activate(show_context=True)

  assert paints[0][1] is _terminal_mod.TerminalClear.SCROLLBACK


@pytest.mark.asyncio
async def test_first_activation_keeps_banner_out_of_scrollback(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  paints: list[tuple[str, _terminal_mod.TerminalClear]] = []

  async def run_async() -> str | None:
    return None

  def print_ansi(
    ansi: str,
    *,
    clear: _terminal_mod.TerminalClear = _terminal_mod.TerminalClear.KEEP,
  ) -> None:
    paints.append((ansi, clear))

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(repl._terminal, 'print_ansi', print_ansi)

  await repl.activate(show_context=False)

  assert len(paints) == 1
  assert paints[0][1] is _terminal_mod.TerminalClear.SCREEN
  assert 'Reflection mirror' not in paints[0][0]
  assert 'Reflection mirror' in _render_banner(repl)


@pytest.mark.asyncio
async def test_replay_during_first_activation_does_not_repeat_banner(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl.dispatch(_actions.WorkerStarted())
  repl.dispatch(_actions.TurnStarted())
  paints: list[str] = []

  async def run_async() -> str | None:
    repl._handle_message(_replay_done('conv-1'))
    await asyncio.gather(*tuple(repl._app._background_tasks))
    return None

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(
    repl._terminal,
    'print_ansi',
    lambda ansi, **_: paints.append(ansi),
  )

  await repl.activate(show_context=True)

  assert len(paints) == 2
  assert not paints[0]
  assert 'Reflection' not in paints[1]
  assert 'Reflection mirror' in _render_banner(repl)


@pytest.mark.asyncio
async def test_activation_restores_committed_stream_content(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl.dispatch(_actions.SetReplaying(False))
  repl._handle_message(_stream_content('partial answer'))
  repl._handle_message(
    _stream_tool_call(index=0, name='bash', arguments='{"command":"ls"}')
  )
  paints: list[str] = []

  async def run_async() -> str | None:
    return None

  monkeypatch.setattr(repl._app, 'run_async', run_async)
  monkeypatch.setattr(
    repl._terminal,
    'print_ansi',
    lambda ansi, **_: paints.append(ansi),
  )

  await repl.activate(show_context=True)

  assert 'partial answer' in paints[0]


@pytest.mark.asyncio
async def test_interruption_survives_rerender_and_worktree_switch(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl.dispatch(_actions.SetReplaying(False))
  repl._handle_message(_delta({'role': 'user', 'content': 'do work'}))
  repl._handle_message(
    _runtime(agent_pb2.RuntimeDelta(turn_interrupted=empty_pb2.Empty()))
  )
  paints = _capture_paints(repl, monkeypatch)

  await repl._rerender()

  assert 'interrupted by user' in paints[-1][0]
  assert paints[-1][1] is _terminal_mod.TerminalClear.SCREEN
  latest = repl._render_history(header=[_text.Text('workspace')])
  assert 'interrupted by user' in latest


@pytest.mark.asyncio
async def test_failed_worker_restarts_on_next_message(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl.dispatch(_actions.WorkerStarted())
  repl._worker_ready.set()

  await repl._exit_with_error('worker exited')

  assert not repl._state.lifecycle.started
  assert not repl._worker_ready.is_set()
  calls: list[str] = []

  async def shutdown() -> None:
    calls.append('shutdown')

  async def start(*, replay: bool) -> None:
    assert not replay
    calls.append('start')
    repl.dispatch(_actions.WorkerStarted())

  def send(instruction: str, *, message_id: str | None = None) -> None:
    assert instruction == 'retry'
    assert message_id is None
    calls.append('send')

  monkeypatch.setattr(repl, '_shutdown_worker', shutdown)
  monkeypatch.setattr(repl, '_start_worker', start)
  monkeypatch.setattr(repl, '_send_instruction', send)

  await repl._run_turn('retry')

  assert calls == ['shutdown', 'start', 'send']


@pytest.mark.asyncio
async def test_failed_worker_keeps_commands_available(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl.dispatch(_actions.WorkerFailed('worker exited'))
    switched: list[tuple[str, bool]] = []

    async def switch(conversation_id: str, *, create: bool) -> None:
      switched.append((conversation_id, create))
      repl._app.exit()

    async def enter_new_command() -> None:
      await asyncio.sleep(0.05)
      pipe.send_text('/new\r')

    monkeypatch.setattr(repl, '_switch_session', switch)
    repl._app.pre_run_callables.append(
      lambda: repl._app.create_background_task(enter_new_command())
    )

    assert (
      _fragments_text(_component_text(_failure.Failure().render(repl._state)))
      == 'Agent failed to load. Press Enter to try loading again'
    )
    await asyncio.wait_for(repl.activate(show_context=False), timeout=2)

    assert len(switched) == 1
    conversation_id, create = switched[0]
    assert len(conversation_id) == 32
    assert create


@pytest.mark.asyncio
async def test_worker_failure_leaves_reverse_search(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._exit_on_worker_failure = False

    async def repaint(
      *, replacing: bool, drop_scrollback: bool = False
    ) -> None:
      del replacing, drop_scrollback

    monkeypatch.setattr(repl, '_repaint', repaint)

    async def fail_during_search() -> None:
      await asyncio.sleep(0.05)
      pipe.send_text('\x12')  # Ctrl+R.
      await asyncio.sleep(0.05)
      assert repl._state.selector.active
      pipe.send_text('query')
      await asyncio.sleep(0.05)
      await repl._exit_with_error('worker exited')
      assert not repl._state.selector.active
      search_text = repl._selector_search.buffer.text
      pipe.send_text('ordinary input')
      await asyncio.sleep(0.05)
      assert repl._selector_search.buffer.text == search_text
      assert repl._input.text == 'ordinary input'
      repl._app.exit()

    repl._app.pre_run_callables.append(
      lambda: repl._app.create_background_task(fail_during_search())
    )
    await asyncio.wait_for(repl.activate(show_context=False), timeout=2)


@pytest.mark.asyncio
async def test_worker_failure_on_a_tab_replaces_the_previous_conversation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # A failed start must replace the screen, not append under the previous tab.
  repl = _repl()
  repl.dispatch(_actions.SetActive(True))
  repl._exit_on_worker_failure = False
  repl._terminal.pin_header(_text.Text('new-worktree'))
  paints = _capture_paints(repl, monkeypatch)

  await repl._exit_with_error('worker exited')

  assert len(paints) == 1
  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'new-worktree' in ansi
  assert 'worker exited' in ansi


@pytest.mark.asyncio
async def test_active_worker_failure_exits_through_an_effect(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(True))
  effects: list[_effects.Effect] = []

  async def repaint(*, replacing: bool) -> None:
    assert replacing

  monkeypatch.setattr(repl, '_run_effect', effects.append)
  monkeypatch.setattr(repl, '_repaint', repaint)

  await repl._exit_with_error('worker exited')

  assert effects == [
    _effects.ExitApp(defer_until_running=True),
  ]


def test_approval_resolution_preserves_the_next_queued_prompt() -> None:
  repl = _repl()
  repl._handle_message(_approval_requested('first', 'bash'))
  repl._handle_message(_approval_requested('second', 'bash'))
  assert repl._state.approval.request is not None
  assert repl._state.approval.request.call_id == 'first'

  repl._handle_message(
    agent_pb2.SessionMessage(
      hitl_resolved=agent_pb2.HITLResolved(call_id='first', approved=True)
    )
  )
  assert repl._state.approval.request is not None
  assert repl._state.approval.request.call_id == 'second'


def test_inactive_agent_surfaces_attention_and_completion() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetActive(False))
  repl.dispatch(_actions.WorkerStarted())

  repl._handle_message(_approval_requested('call-1', 'bash'))
  assert repl.worktree_state is _actions.WorktreeAgentState.ATTENTION

  repl._handle_message(
    agent_pb2.SessionMessage(
      hitl_resolved=agent_pb2.HITLResolved(call_id='call-1')
    )
  )
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(_turn_completed())
  assert repl.worktree_state is _actions.WorktreeAgentState.UNREAD


@contextlib.asynccontextmanager
async def _route_workspaces(
  repl: cli._Repl, controller: _workspace_controller.WorkspaceController
) -> AsyncIterator[None]:
  assert repl._workspace_effects is not None
  queue = repl._workspace_effects

  async def route() -> None:
    while True:
      agent, effect = await queue.get()
      await cli.handle_workspace_effect(controller, agent, effect)

  task = asyncio.create_task(route())
  try:
    yield
  finally:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_worktree_navigation_keybindings(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    monkeypatch.setenv('TERM_PROGRAM', 'ghostty')
    repl = _repl()
    terminal_writes: list[str] = []
    monkeypatch.setattr(repl._app.output, 'write_raw', terminal_writes.append)

    class Worktrees:
      def refresh(self) -> None:
        pass

      def can_cycle(self) -> bool:
        return True

      def can_select(self, index: int) -> bool:
        return index == 1

    controller = cast(_workspace_controller.WorkspaceController, Worktrees())
    async with _route_workspaces(repl, controller):
      forward = asyncio.create_task(repl.activate(show_context=False))
      await asyncio.sleep(0.05)
      pipe.send_text('\t')
      assert await asyncio.wait_for(forward, timeout=2) == 'next'
      assert not repl._app.erase_when_done

      backward = asyncio.create_task(repl.activate(show_context=False))
      await asyncio.sleep(0.05)
      pipe.send_text('\x1b[9;2u')  # Shift+Tab in the CSI-u protocol.
      assert await asyncio.wait_for(backward, timeout=2) == 'previous'
      assert not repl._app.erase_when_done

      numbered = asyncio.create_task(repl.activate(show_context=False))
      await asyncio.sleep(0.05)
      pipe.send_text('\x1b[49;5u')  # Ctrl+1 already names the selected tab.
      await asyncio.sleep(0.05)
      assert not numbered.done()
      pipe.send_text('\x1b[50;5u')  # Ctrl+2 in the CSI u protocol.
      assert await asyncio.wait_for(numbered, timeout=2) == 'select:1'
      assert not repl._app.erase_when_done

      quitting = asyncio.create_task(repl.activate(show_context=False))
      await asyncio.sleep(0.05)
      pipe.send_text('\x1b[100;5u')  # Ctrl+D in the CSI-u protocol.
      assert await asyncio.wait_for(quitting, timeout=2) is None
      assert not repl._app.erase_when_done
      # Level 2 rewrites ordinary shifted text as escape sequences. Mirror only
      # recognizes the concrete aliases above, so it must never enable the mode.
      terminal_output = ''.join(terminal_writes)
      assert '\x1b[>4;2m' not in terminal_output
      assert '\x1b[>4m' not in terminal_output


@pytest.mark.asyncio
async def test_worktree_navigation_keeps_a_visible_frame_between_tabs() -> None:
  """The old tab stays visible until the next tab can replace the screen."""

  class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
      return True

  class Worktrees:
    def refresh(self) -> None:
      pass

    def can_cycle(self) -> bool:
      return True

    def can_select(self, index: int) -> bool:
      return index == 1

  terminal = TerminalBuffer()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=12, columns=80),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    first = _repl(conversation_id='first')
    second = _repl(conversation_id='second')
    first._terminal.console = _console.Console(
      file=terminal, force_terminal=True, width=80, height=12
    )
    second._terminal.console = _console.Console(
      file=terminal, force_terminal=True, width=80, height=12
    )
    worktrees = cast(_workspace_controller.WorkspaceController, Worktrees())
    async with (
      _route_workspaces(first, worktrees),
      _route_workspaces(second, worktrees),
    ):
      _seed_history(first, ['first conversation'])
      _seed_history(second, ['second conversation'])
      await _settle(first)
      await _settle(second)
      first._input.text = '\n'.join(
        f'draft line {index}' for index in range(20)
      )
      first._input.cursor_position = len(first._input.text)
      second._input.text = 'second draft'
      terminal.seek(0)
      terminal.truncate()

      active = asyncio.create_task(first.activate(show_context=True))
      await asyncio.sleep(0.05)
      before_switch = _screen_lines_with_positions(terminal.getvalue(), rows=12)
      assert 'draft line 19' in '\n'.join(before_switch)

      pipe.send_text('\t')
      assert await asyncio.wait_for(active, timeout=2) == 'next'

      after_switch = _screen_lines_with_positions(terminal.getvalue(), rows=12)
      assert after_switch == before_switch
      screen = '\n'.join(after_switch)
      assert 'draft line 19' in screen

      paint_start = terminal.tell()
      active = asyncio.create_task(second.activate(show_context=True))
      await asyncio.sleep(0.05)
      paint = terminal.getvalue()[paint_start:]
      assert (
        paint.index(_terminal_mod.SYNC_BEGIN)
        < paint.index('\x1b[2J')
        < paint.index('second conversation')
        < paint.index('second draft')
        < paint.index(_terminal_mod.SYNC_END)
      )
      screen = '\n'.join(_screen_lines(terminal.getvalue(), rows=12))
      assert 'second conversation' in screen
      assert 'first conversation' not in screen

      pipe.send_text('\x04')
      assert await asyncio.wait_for(active, timeout=2) is None
      escapes = terminal.getvalue()
      assert escapes.count('\x1b[?2004h') == 2
      assert escapes.count('\x1b[?2004l') == 2


@pytest.mark.asyncio
async def test_welcome_tab_and_shift_tab_keep_selected_workspace_visible() -> (
  None
):
  tabs = (
    workspaces.PendingWorkspace('a', 'workspace-alpha'),
    workspaces.PendingWorkspace('b', 'workspace-bravo'),
    workspaces.PendingWorkspace('c', 'workspace-charlie'),
    workspaces.PendingWorkspace('d', 'workspace-delta'),
    workspaces.PendingWorkspace('e', 'workspace-echo'),
    workspaces.PendingWorkspace('f', 'workspace-foxtrot'),
  )
  selected = 'e'
  navigation = _containers.Window(
    _controls.FormattedTextControl(
      lambda: worktree_tabs.render_tabs(tabs, selected, width=200)
    ),
    dont_extend_height=True,
    wrap_lines=False,
  )
  async with _visual_repl(200, 40) as (first, stream):
    second = _repl(conversation_id='second', model='second-model')
    pipe = first._app.input
    assert isinstance(pipe, _input_base.PipeInput)
    first._input.text = 'first draft'
    second._input.text = 'second draft'
    first_painted = asyncio.Event()
    second_painted = asyncio.Event()
    first._app.after_render += lambda _: first_painted.set()
    second._app.after_render += lambda _: second_painted.set()
    active = asyncio.create_task(
      first.activate(show_context=False, navigation=navigation)
    )
    try:
      await asyncio.wait_for(first_painted.wait(), 2)
      pipe.send_text('\t')
      assert first._workspace_effects is not None
      agent, effect = await asyncio.wait_for(first._workspace_effects.get(), 2)
      assert agent is first
      assert effect == _effects.CycleWorktree(1)
      first.handoff('next')
      assert await asyncio.wait_for(active, 2) == 'next'

      selected = 'f'
      active = asyncio.create_task(
        second.activate(show_context=False, navigation=navigation)
      )
      await asyncio.wait_for(second_painted.wait(), 2)
      terminal = pyte.Screen(200, 40)
      pyte.Stream(terminal).feed(stream.getvalue().decode())
      assert 'workspace-foxtrot' in terminal.display[-1]
      assert 'Tab / Shift+Tab' in terminal.display[-1]
      assert second._app.layout.current_control is second._composer_control
      pipe.send_text('\x1b[Z')
      assert second._workspace_effects is not None
      agent, effect = await asyncio.wait_for(second._workspace_effects.get(), 2)
      assert agent is second
      assert effect == _effects.CycleWorktree(-1)
      second.handoff('previous')
      assert await asyncio.wait_for(active, 2) == 'previous'

      selected = 'e'
      first_painted.clear()
      active = asyncio.create_task(
        first.activate(show_context=False, navigation=navigation)
      )
      await asyncio.wait_for(first_painted.wait(), 2)
      terminal = pyte.Screen(200, 40)
      pyte.Stream(terminal).feed(stream.getvalue().decode())
      assert 'workspace-echo' in terminal.display[-1]
      assert first._input.text == 'first draft'
      assert second._input.text == 'second draft'
      assert first._app.layout.current_control is first._composer_control
      pipe.send_text('\x04')
      assert await asyncio.wait_for(active, 2) is None
    finally:
      active.cancel()
      await asyncio.gather(active, return_exceptions=True)
      await second.close()


@pytest.mark.asyncio
async def test_worktree_cycle_is_a_noop_without_another_worktree(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  refreshes = 0

  class Worktrees:
    def refresh(self) -> None:
      nonlocal refreshes
      refreshes += 1

    def can_cycle(self) -> bool:
      return False

  controller = cast(_workspace_controller.WorkspaceController, Worktrees())
  exits: list[str | None] = []
  monkeypatch.setattr(
    repl._app, 'exit', lambda result=None: exits.append(result)
  )
  await cli.handle_workspace_effect(controller, repl, _effects.CycleWorktree(1))
  await cli.handle_workspace_effect(
    controller, repl, _effects.CycleWorktree(-1)
  )

  assert exits == []
  assert refreshes == 2


def _seed_history(
  repl: cli._Repl,
  contents: Iterable[str],
) -> None:
  """A replayed conversation of user messages, as the worker sends it."""
  for content in contents:
    repl._handle_message(_delta({'role': 'user', 'content': content}))
  repl._handle_message(_replay_done('conv-1'))


def _render_to_text(renderable: _console.RenderableType) -> str:
  output = io.StringIO()
  _rendering.ansi_console(output, 80).print(renderable)
  return _rendering.strip_ansi(output.getvalue())


def _noop_command(request: agent_pb2.ClientMessage) -> None:
  """Drop a worker command, for tests that only exercise the terminal side."""
  del request


async def _settle(repl: cli._Repl) -> None:
  """Wait for the work a key handler scheduled on the application."""
  await asyncio.gather(*tuple(repl._app._background_tasks))


async def _dispatch_and_wait(repl: cli._Repl, action: _actions.Action) -> None:
  """Dispatch one action and wait for its immediate effects."""
  await asyncio.gather(*repl.dispatch(action))


def _submit(repl: cli._Repl) -> None:
  if repl._state.selector.active:
    repl.dispatch(*repl._selector_choice())
    return
  repl.dispatch(_actions.ComposerSubmit())


def _complete_selection(repl: cli._Repl) -> None:
  repl.dispatch(_actions.AcceptSelection())


def _move_completion(repl: cli._Repl, delta: int) -> None:
  repl.dispatch(_actions.MoveSelection(delta))


def _ansi_text(ansi: Any) -> str:
  """The plain terminal escapes behind a prompt_toolkit `ANSI`."""
  return ''.join(
    fragment[1]
    for fragment in ansi.__pt_formatted_text__()
    if fragment[0] != '[ZeroWidthEscape]'
  )


def _screen_lines(escapes: str, *, cols: int = 80, rows: int = 8) -> list[str]:
  """The non-blank rows an escape string paints on a virtual terminal."""
  return [
    line
    for line in _screen_lines_with_positions(escapes, cols=cols, rows=rows)
    if line
  ]


def _screen_lines_with_positions(
  escapes: str, *, cols: int = 80, rows: int = 8
) -> list[str]:
  """Every terminal row painted by an escape string, preserving positions."""
  screen = pyte.Screen(cols, rows)
  screen.set_mode(pyte.modes.LNM)  # LF also carriage-returns, like a real tty.
  pyte.Stream(screen).feed(escapes)
  return [line.rstrip() for line in screen.display]


def test_user_text_prefixes_prompt_and_tints_known_skills() -> None:
  rendered = _rendering.user_text('$create-tool please', {'create-tool'})

  screen = pyte.Screen(40, 5)
  screen.set_mode(pyte.modes.LNM)
  pyte.Stream(screen).feed(_rendering.renderable_to_ansi(rendered, 40))

  assert screen.display[1].rstrip() == '› $create-tool please'
  tint = themes.rich_theme().styles['ui.info'].color
  assert tint is not None
  assert screen.buffer[1][2].fg == tint.get_truecolor().hex.lstrip('#')
  assert screen.buffer[1][15].fg == 'white'
  assert screen.buffer[1][2].bg == screen.buffer[1][15].bg


@pytest.mark.parametrize('width', [12, 80])
def test_user_text_shades_full_width_with_vertical_padding(width: int) -> None:
  screen = pyte.Screen(width, 6)
  screen.set_mode(pyte.modes.LNM)
  pyte.Stream(screen).feed(
    _rendering.render_blocks(
      [_rendering.user_text('hello'), _rendering.assistant_render('reply')],
      width,
    )
  )

  assert screen.display[:5] == [
    ' ' * width,
    '› hello'.ljust(width),
    ' ' * width,
    ' ' * width,
    '● reply'.ljust(width),
  ]
  background = screen.buffer[1][2].bg
  assert background != 'default'
  assert len(screen.buffer[0]) == width
  assert len(screen.buffer[1]) == width
  assert len(screen.buffer[2]) == width
  assert {cell.bg for cell in screen.buffer[0].values()} == {background}
  assert {cell.bg for cell in screen.buffer[1].values()} == {background}
  assert {cell.bg for cell in screen.buffer[2].values()} == {background}
  assert screen.buffer[3][0].bg == 'default'
  assert screen.buffer[4][2].bg == 'default'


@pytest.mark.parametrize(
  ('content', 'expected'),
  [
    ('one two three four', ['› one two', '  three', '  four']),
    ('first\n\nsecond', ['› first', '', '  second']),
    ('a\tb', ['› a       b']),
    ('a\n\tb', ['› a', '          b']),
    ('abcdefghijk', ['› abcdefghi', '  jk']),
    ('你好你好你好', ['› 你好你好', '  你好']),
    ('**literal**', ['› **literal', '  **']),
    ('before\x1b[31mred\x1b[0m', ['› beforered']),
  ],
)
def test_user_text_wraps_in_a_two_column_gutter(
  content: str, expected: list[str]
) -> None:
  rendered = _rendering.renderable_to_ansi(_rendering.user_text(content), 12)
  lines = _rendering.strip_ansi(rendered).splitlines()

  assert [line.rstrip() for line in lines[1:-1]] == expected
  assert {_cells.cell_len(line) for line in lines} == {12}


def test_strip_ansi_covers_terminal_modes_and_string_sequences() -> None:
  contaminated = (
    'before'
    '\x1b[?1006;1000h'
    '\x1b[>4;2m'
    '\x1b]0;window title\x07'
    '\x1bPdevice control\x1b\\'
    '\x9b?2004h'
    '\x07after'
  )

  assert _rendering.strip_ansi(contaminated) == 'beforeafter'


def test_rendered_ansi_keeps_formatting_but_drops_terminal_modes() -> None:
  contaminated = (
    '\x1b[31mred\x1b[0m'
    '\x1b[?1006;1000h'
    '\x1b[?1049h'
    '\x1b[>4;2m'
    '\x1b]0;window title\x07'
  )

  sanitized = _rendering.sanitize_ansi(contaminated)

  assert sanitized == '\x1b[31mred\x1b[0m'


def test_streaming_preview_formats_markdown_and_bounds_overflow() -> None:
  preview = _state.StreamPreview(
    content='\n\n'.join(f'**paragraph {index}**' for index in range(12))
  )
  ansi, _ = _stream_mod.render_stream_preview(
    preview,
    width=80,
    max_lines=7,
  )
  live = ''.join(fragment[1] for fragment in ansi.__pt_formatted_text__())
  lines = live.splitlines()
  message_index = next(
    index
    for index, line in enumerate(lines)
    if line.strip() == 'Earlier output temporarily hidden'
  )

  assert lines[message_index].index('Earlier') > 0
  assert lines[message_index + 1].strip() == '…'
  assert 'paragraph 0' not in live
  assert 'paragraph 11' in live
  assert '**' not in live


def test_pending_tool_renders_like_its_settled_form() -> None:
  # A running tool and its completed form share one renderer, so completing a
  # tool is a body swap (running… -> result) in place, not a restyle+jump.
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {
          'name': 'edit',
          'arguments': '{"file_path":"file.py"}',
        },
      }
    ],
  )
  pending = _pending.render(repl._state)
  pending_text = _ansi_text(pending)
  live = _screen_lines(pending_text)

  settled = _handle(
    repl,
    _delta({
      'role': 'tool',
      'tool_call_id': 'c1',
      'content': 'ok',
      'name': 'edit',
    }),
  )
  settled_ansi = _rendering.renderable_to_ansi(settled, 80)
  settled_plain = _ansi_text(_ansi.ANSI(settled_ansi))
  settled_lines = _screen_lines(settled_ansi)

  assert 'edit(file.py)' in pending_text
  assert 'edit(file.py)' in settled_plain
  assert '└' in live[1] and 'running' in live[1]
  assert '└' in settled_lines[1] and 'ok' in settled_lines[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'view_mode', [_actions.ViewMode.COMPACT, _actions.ViewMode.TRANSCRIPT]
)
async def test_theme_switch_refreshes_pending_tools_without_changing_calls(
  view_mode: _actions.ViewMode,
) -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  repl.dispatch(_actions.SetViewMode(view_mode))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': 'read', 'arguments': '{"file_path":"file.py"}'},
      }
    ],
  )
  before = _pending.render(repl._state)
  stream = repl._state.stream
  rows = _pending.rows(repl._state)
  assert _pending.render(repl._state) is before
  await _dispatch_and_wait(
    repl,
    _actions.ExtensionConfigChanged(themes.EXTENSION_ID, {'name': 'light'}),
  )
  after = _pending.render(repl._state)
  assert before.value != after.value
  assert _ansi_text(before) == _ansi_text(after)
  assert _pending.rows(repl._state) == rows
  assert repl._state.stream == stream
  assert _pending.render(repl._state) is after
  await repl.close()


@pytest.mark.parametrize('repetitions', [20, 100_000])
def test_long_tool_previews_are_bounded_before_rendering(
  repetitions: int,
) -> None:
  source = 'https://example.com/' + 'very-long-path/' * repetitions
  args = _rendering.compact_args(json.dumps({'url': source}))
  preview = _rendering.result_summary(source)
  assert len(args) <= 1024
  assert len(preview.plain) <= 1024
  result = _rendering.tool_text('web_browse', 'example.com', preview)

  lines = _screen_lines(_rendering.renderable_to_ansi(result, 80))

  assert len(lines) == 2
  assert 'web_browse(example.com)' in lines[0]
  assert '└ https://example.com/' in lines[1]
  assert lines[1].endswith('…')


def test_parallel_pending_tools_have_settled_message_spacing() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"first"}'},
      },
      {
        'id': 'c2',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"second"}'},
      },
    ],
  )

  lines = _ansi_text(_pending.render(repl._state)).splitlines()

  assert 'bash(first)' in lines[0]
  assert lines[2] == ''
  assert 'bash(second)' in lines[3]


def test_stream_partial_accumulates_preview_state() -> None:
  repl = _repl()
  repl._handle_message(_stream_content('Hel'))
  repl._handle_message(_stream_content('lo'))
  repl._handle_message(_stream_refusal('No'))

  preview = repl._stream_pane.preview
  assert preview is not None
  assert preview.content == 'Hello'
  assert preview.refusal == 'No'


@pytest.mark.asyncio
async def test_stream_window_create_content_calls_pane_methods() -> None:
  """prompt_toolkit must receive callables, not a frozen Point/ANSI snapshot."""
  repl = _repl()
  control = next(
    candidate
    for candidate in repl._app.layout.find_all_controls()
    if isinstance(candidate, _controls.FormattedTextControl)
    and candidate.get_cursor_position is not None
  )
  repl._handle_message(_stream_content('Hello'))
  content = control.create_content(40, 10)
  assert content.cursor_position == repl._stream_pane.cursor_position()
  text = ''.join(
    fragment[1]
    for index in range(content.line_count)
    for fragment in content.get_line(index)
  )
  assert 'Hello' in text
  await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_pane_stays_visible_while_tool_call_assembles(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Streamed answer text must not vanish while the tool-call summary catches up.
  repl = _repl()
  emitted: list[_console.RenderableType] = []

  async def emit(
    renderables: list[_console.RenderableType],
    *,
    replacing_history: bool = False,
  ) -> None:
    del replacing_history
    emitted.extend(renderables)

  monkeypatch.setattr(repl._terminal, 'emit', emit)
  repl._handle_message(_stream_content('the answer'))
  # Streaming: the pane holds the text, plus the blank row under it.
  assert repl._stream_pane.pane_rows > 0
  assert repl._stream_pane.pane_height == repl._stream_pane.pane_rows + 1
  assert _pending.rows(repl._state) == 0

  repl._handle_message(_stream_tool_call(index=0, name='bash'))
  # The text moved to scrollback, so the pane releases its rows: with nothing
  # to sit under, its blank row goes too, leaving one gap and not two.
  assert repl._stream_pane.pane_height == 0
  assert _pending.rows(repl._state) == 0

  repl._handle_message(_stream_tool_call(index=0, arguments='{"command":"ls"}'))

  assert repl._stream_pane.pane_height == 0
  assert _pending.rows(repl._state) > 0
  # The answer text was committed to scrollback once, before the tool rows.
  await asyncio.sleep(0)
  assert len(emitted) == 1
  assert 'the answer' in _render_to_text(emitted[0])


@pytest.mark.asyncio
async def test_stream_handoff_orders_final_and_preserves_aborted_output() -> (
  None
):
  class _RecordingRepl(cli._Repl):
    def __init__(self) -> None:
      self.emitted: list[_console.RenderableType] = []
      self.written = asyncio.Event()
      super().__init__(
        [sys.executable, '-c', 'pass'],
        {'conversation_id': 'conv-1', 'model': 'm-test'},
        session_id='test-session',
        extension_registry=extension.ExtensionRegistry(),
        workspace_effects=asyncio.Queue(),
        workspace_controller=mock.Mock(
          spec=_workspace_controller.WorkspaceController
        ),
      )

      async def emit(
        renderables: list[_console.RenderableType],
      ) -> None:
        self.emitted.extend(renderables)
        self.written.set()

      self._terminal.emit = emit

  repl = _RecordingRepl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))
  repl._handle_message(_stream_reasoning('thinking'))
  repl._handle_message(_stream_content('stale answer'))
  assert repl._stream_pane.preview is not None
  await asyncio.wait_for(repl.written.wait(), timeout=1)

  repl.written.clear()
  repl._handle_message(
    _stream_final({
      'role': 'assistant',
      'reasoning_content': 'thinking',
      'content': 'final answer',
    })
  )
  await asyncio.wait_for(repl.written.wait(), timeout=1)

  assert 'thinking' in _render_to_text(repl.emitted[0])
  final = _render_to_text(repl.emitted[-1])
  assert 'thinking' not in final
  assert 'final answer' in final
  assert 'stale' not in final

  aborted = _RecordingRepl()
  aborted._handle_message(_stream_content('unfinished answer'))
  aborted._handle_message(_stream_aborted())
  await asyncio.wait_for(aborted.written.wait(), timeout=1)
  assert 'unfinished answer' in _render_to_text(aborted.emitted[-1])


@pytest.mark.asyncio
async def test_idle_composer_does_not_add_a_second_scrollback_gap() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  assert repl._bottom_pane.preferred_height(80, 24).preferred == 5
  repl.dispatch(_actions.TurnStarted())
  assert repl._bottom_pane.preferred_height(80, 24).preferred == 7
  repl.dispatch(_actions.TurnFinished())
  assert repl._bottom_pane.preferred_height(80, 24).preferred == 5
  await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_streaming_budget_counts_wrapped_input() -> None:
  repl = _repl(output=_SizedOutput(rows=12, columns=10))
  repl._input.text = 'x' * 25

  # 12 rows, less 4 input rows, top padding, input gap, footer, divider, margin.
  assert repl._streaming_max_visible_lines() == 3
  await asyncio.sleep(0)


def test_footer_uses_compact_separators() -> None:
  repl = _repl()

  left = ''.join(text for _, text in _render_footer(repl))

  assert left == ' 🛠  auto | m-test · medium '


def test_a_system_message_is_not_drawn() -> None:
  rendered = _handle(_repl(), _delta({'role': 'system', 'content': 'be good'}))
  assert rendered is None


def test_log_records_are_drawn() -> None:
  repl = _repl()
  log = agent_pb2.SessionMessage(
    agent_message=agent_pb2.AgentMessage(
      log=agent_pb2.LogRecord(level='warning', text='heads up')
    )
  )
  assert 'warning: heads up' in _render_to_text(_handle(repl, log))


def test_a_user_message_keeps_its_prompt_glyph() -> None:
  rendered = _handle(_repl(), _delta({'role': 'user', 'content': 'hi'}))
  assert rendered is not None
  assert _screen_lines(_rendering.renderable_to_ansi(rendered, 80)) == ['› hi']


def test_a_loaded_skill_is_drawn_under_the_message_that_named_it() -> None:
  repl = _repl()
  user = _handle(repl, _delta({'role': 'user', 'content': '$create-tool'}))
  skill = _handle(
    repl,
    _delta(
      {'role': 'user', 'content': '<skill>instructions</skill>'},
      display='$create-tool',
    ),
  )

  assert user is not None and skill is not None
  lines = _render_to_text(_console.Group(user, skill)).splitlines()
  assert [line.rstrip() for line in lines] == [
    '',
    '› $create-tool',
    '',
    '◆ loaded $create-tool',
  ]


def test_assistant_skill_mentions_are_tinted_after_markdown_rendering() -> None:
  rendered = _rendering.assistant_turn_renderable(
    None,
    'Use $create-tool now.',
    skill_names={'create-tool'},
  )

  assert rendered is not None
  tinted = _rendering.renderable_to_ansi(
    _text.Text('$create-tool', style='ui.info'), 80
  )
  assert tinted in _rendering.renderable_to_ansi(rendered, 80)


def test_assistant_content_renders_markdown_with_icon() -> None:
  rendered = _handle(
    _repl(), _delta({'role': 'assistant', 'content': 'the **bold** answer'})
  )
  assert rendered is not None
  text = _render_to_text(rendered)
  assert '●' in text  # The assistant gutter icon.
  assert 'bold' in text  # Markdown rendered its content.


def test_assistant_refusal_is_visible() -> None:
  rendered = _handle(
    _repl(),
    _delta({
      'role': 'assistant',
      'content': None,
      'refusal': 'I cannot help with that.',
    }),
  )
  assert rendered is not None
  assert 'I cannot help with that.' in _render_to_text(rendered)


def test_reasoning_is_drawn_before_the_answer() -> None:
  message = {
    'role': 'assistant',
    'content': 'done',
    'reasoning_content': 'thinking',
  }
  rendered = _handle(_repl(), _delta(message))
  assert rendered is not None
  text = _render_to_text(rendered)
  assert text.index('thinking') < text.index('done')


def _bash_call(
  arguments: str = '{"command": "ls"}', *, name: str = 'bash'
) -> dict[str, Any]:
  return {
    'role': 'assistant',
    'content': None,
    'tool_calls': [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': name, 'arguments': arguments},
      }
    ],
  }


def test_tool_call_is_parked_then_resolved_with_hints() -> None:
  repl = _repl()
  # The call is parked in the pending pane, not emitted to scrollback yet.
  assert _handle(repl, _delta(_bash_call())) is None
  assert _running_call_hints(repl) == {'c1': ('bash', 'ls', 'tool')}
  assert _running_call_ids(repl) == ('c1',)

  # Its result resolves the call into one block: the call line + a preview the
  # TUI derives from the protobuf content.
  result = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'raw output'}
  rendered = _handle(repl, _delta(result))
  assert _running_call_hints(repl) == {}
  assert _running_call_ids(repl) == ()
  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'bash(ls)' in text
  assert '└ raw output' in text


def test_agent_activity_has_distinct_rendering() -> None:
  repl = _repl()
  call = _bash_call('{"prompt":"Explore the repository."}', name='subagent')
  assert _handle(repl, _delta(call)) is None

  result = {
    'role': 'tool',
    'tool_call_id': 'c1',
    'content': 'Task 88766 started.',
  }
  rendered = _handle(repl, _delta(result))

  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'Subagent · Explore the repository.' in text
  assert 'subagent(' not in text
  assert 'Task 88766 started.' in text


@pytest.mark.asyncio
async def test_replayed_history_draws_without_moving_the_live_panes(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Loading a conversation is not activity: the footer status and the running
  # rows describe the turn in progress, and a replay has none.
  history = (
    _delta({'role': 'user', 'content': 'do the thing'}),
    _delta(_bash_call()),
    _delta({'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}),
    _delta({'role': 'assistant', 'content': 'done'}),
  )

  replayed = _repl()  # A fresh REPL is replaying the conversation it opened.
  paints = _capture_paints(replayed, monkeypatch)
  replayed.dispatch(_actions.SetStatus('loading conversation…'))
  for delta in history:
    replayed._handle_message(delta)
  replayed._handle_message(_replay_done('conv-1'))
  await _settle(replayed)
  assert replayed._state.activity.status == 'loading conversation…'
  assert _running_call_ids(replayed) == ()
  assert _pending.rows(replayed._state) == 0
  # The entries still reach the screen, tool result joined to its call.
  drawn = '\n'.join(_screen_lines(paints[0][0], rows=24))
  assert 'do the thing' in drawn
  assert 'bash(ls)' in drawn
  assert 'done' in drawn

  live = _repl()
  live.dispatch(_actions.SetReplaying(False))
  for delta in history[:2]:
    live._handle_message(delta)
  assert live._state.activity.status == 'running tool…'
  assert _running_call_ids(live) == ('c1',)
  assert _pending.rows(live._state) > 0
  await _settle(live)


def test_diff_result_renders_with_line_numbers() -> None:
  diff = '--- f.py\n+++ f.py\n@@ -1,2 +1,2 @@\n keep\n-old line\n+new line\n'
  assert _rendering.looks_like_diff(diff)
  plain = _rendering.render_diff(diff).plain
  assert '- old line' in plain
  assert '+ new line' in plain
  assert '2 - old line' in plain  # line number precedes the change


def test_render_diff_keeps_content_lines_starting_with_dashes() -> None:
  # A deleted line whose content starts with `--` must not be mistaken for a
  # file header, nor desync the line numbers.
  diff = '--- f.txt\n+++ f.txt\n@@ -1,2 +1,2 @@\n ctx\n---foo\n+++bar\n'
  plain = _rendering.render_diff(diff).plain
  assert '- --foo' in plain
  assert '+ ++bar' in plain


def test_render_diff_multi_file_keeps_headers_and_resets_line_numbers() -> None:
  # A `git diff` over two files (e.g. from BashTool): the second file's header
  # ends the first hunk instead of being parsed as its context, and both file
  # paths survive as plain text so hunks stay attributable.
  diff = (
    'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n'
    '@@ -1,1 +1,1 @@\n-old a\n+new a\n'
    'diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n'
    '@@ -5,1 +5,1 @@\n-old b\n+new b\n'
  )
  plain = _rendering.render_diff(diff).plain
  # The second file's header survives instead of being eaten as hunk context.
  assert 'diff --git a/b.py b/b.py' in plain
  assert '1 - old a' in plain
  assert '5 - old b' in plain  # Line numbers reset from the second hunk header.


def test_tool_display_renders_as_diff_instead_of_content() -> None:
  diff = '--- f.py\n+++ f.py\n@@ -1,1 +1,1 @@\n-old\n+new\n'
  message = {
    'role': 'tool',
    'tool_call_id': 'c1',
    'content': 'Made 1 replacement(s) in f.py.',
  }
  rendered = _handle(_repl(), _delta(message, display=diff))
  assert rendered is not None
  text = _render_to_text(rendered)
  assert '- old' in text and '+ new' in text  # The diff, not the confirmation.
  assert 'Made 1 replacement' not in text


def test_tool_content_that_looks_like_a_diff_is_not_rendered_as_a_diff() -> (
  None
):
  # A diff only renders when it arrives as a tool's `display` payload (edit /
  # write). `git diff` run through bash lands in `content` with no `display`,
  # so it stays an ordinary result preview rather than a line-numbered diff.
  git_diff = (
    'diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n'
    '@@ -1,1 +1,1 @@\n-old\n+new\n'
  )
  repl = _repl()
  _handle(repl, _delta(_bash_call('{"command": "git diff"}')))
  result = {'role': 'tool', 'tool_call_id': 'c1', 'content': git_diff}
  rendered = _handle(repl, _delta(result))
  assert rendered is not None
  text = _render_to_text(rendered)
  assert '1 - old' not in text  # No line-numbered diff rendering.
  assert '1 + new' not in text


def test_next_status_label_running_tool_after_tool_call() -> None:
  call = _delta(_bash_call()).agent_message.chat
  result = _delta({
    'role': 'tool',
    'content': 'x',
    'tool_call_id': 'c1',
  }).agent_message.chat
  assert _rendering.next_status_label(call) == 'running tool…'
  assert _rendering.next_status_label(result) == 'thinking…'


def test_next_status_label_thinking_after_assistant_message() -> None:
  message = _delta({
    'role': 'assistant',
    'content': 'Let me check that.',
  }).agent_message.chat
  assert _rendering.next_status_label(message) == 'thinking…'


@pytest.mark.asyncio
@pytest.mark.parametrize('view_mode', list(_actions.ViewMode))
async def test_activity_resumes_after_assistant_message_before_turn_finishes(
  view_mode: _actions.ViewMode,
) -> None:
  repl = _NoEmitRepl()
  repl.dispatch(
    _actions.SetReplaying(False),
    _actions.SetViewMode(view_mode),
    _actions.TurnStarted(),
  )
  try:
    repl._handle_message(_stream_content('Let me check that.'))
    repl._handle_message(
      _stream_final({'role': 'assistant', 'content': 'Let me check that.'})
    )
    assert repl._state.activity.busy
    assert 'Working' in _fragments_text(_render_activity(repl))

    repl._handle_message(_turn_completed())
    assert not repl._state.activity.busy
    assert _render_activity(repl) == []
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('view_mode', list(_actions.ViewMode))
async def test_activity_resumes_when_reasoning_follows_streamed_content(
  view_mode: _actions.ViewMode,
) -> None:
  repl = _NoEmitRepl()
  repl.dispatch(
    _actions.SetReplaying(False),
    _actions.SetViewMode(view_mode),
    _actions.TurnStarted(),
  )
  try:
    repl._handle_message(_stream_content('Let me check that.'))
    assert 'Working' in _fragments_text(_render_activity(repl))
    repl._handle_message(_stream_reasoning('Considering the next step.'))
    assert repl._state.activity.busy
    assert 'Working' in _fragments_text(_render_activity(repl))

    repl._handle_message(
      _stream_final({
        'role': 'assistant',
        'content': 'Let me check that.',
        'reasoning_content': 'Considering the next step.',
      })
    )
    repl._handle_message(_turn_completed())
    assert not repl._state.activity.busy
    assert _render_activity(repl) == []
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('view_mode', list(_actions.ViewMode))
async def test_activity_stays_visible_after_streamed_commentary(
  view_mode: _actions.ViewMode,
) -> None:
  repl = _NoEmitRepl()
  repl.dispatch(
    _actions.SetReplaying(False),
    _actions.SetViewMode(view_mode),
    _actions.TurnStarted(),
  )
  try:
    assert 'Working' in _fragments_text(_render_activity(repl))
    repl._handle_message(_stream_content('Let me check that.'))

    assert repl._state.activity.busy
    assert 'Working' in _fragments_text(_render_activity(repl))
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('view_mode', list(_actions.ViewMode))
@pytest.mark.parametrize(
  'arguments',
  ['', '{"command":', '{"command":"pwd"}'],
  ids=['tool-name-only', 'incomplete-arguments', 'displayable-arguments'],
)
async def test_activity_stays_visible_before_tool_arguments_are_displayable(
  view_mode: _actions.ViewMode,
  arguments: str,
) -> None:
  terminal = io.StringIO()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=16, columns=80),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _NoEmitRepl()
    repl.dispatch(
      _actions.SetReplaying(False),
      _actions.SetViewMode(view_mode),
      _actions.TurnStarted(),
    )
    try:
      repl._handle_message(_stream_content('Let me check that.'))
      repl._handle_message(_stream_tool_call(index=0, name='bash'))
      repl._handle_message(_stream_tool_call(index=0, arguments=arguments))
      with _current.set_app(repl._app):
        repl._app.renderer.render(repl._app, repl._app.layout)
      displayed = '\n'.join(_screen_lines(terminal.getvalue(), rows=16))

      assert repl._state.activity.busy
      assert 'Working' in displayed, displayed
    finally:
      await repl.close()


def test_stream_partial_content_preserves_thinking_status() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SetStatus('thinking…'))
  repl._handle_message(_stream_reasoning('still thinking'))
  assert repl._state.activity.status == 'thinking…'

  # Separate streams avoid scheduling a reasoning-handoff task outside an event loop.
  writing = _repl()
  writing.dispatch(_actions.TurnStarted())
  writing.dispatch(_actions.SetStatus('thinking…'))
  writing._handle_message(_stream_content('Hello'))
  assert writing._state.activity.status == 'thinking…'
  assert 'Working' in _fragments_text(_render_activity(writing))


@pytest.mark.asyncio
async def test_stream_partial_tool_call_shows_pending_and_running_tool() -> (
  None
):
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SetStatus('thinking…'))
  repl._handle_message(_stream_tool_call(index=0, name='write'))
  assert repl._state.activity.status == 'thinking…'
  assert _pending_call_hints(repl) == ()
  assert _pending.rows(repl._state) == 0

  repl._handle_message(
    _stream_tool_call(index=0, arguments='{"file_path":"a.py"}')
  )
  assert _pending_call_hints(repl) == (('write', 'a.py', 'tool'),)
  activity = ''.join(text for _, text in _render_activity(repl))
  assert 'Working' in activity
  assert 'thinking' not in activity

  # stream_final replaces the live placeholder with the durable parked call.
  repl._handle_message(
    _stream_final(
      {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
          {
            'id': 'c1',
            'type': 'function',
            'function': {
              'name': 'write',
              'arguments': '{"file_path": "a.py", "content": "x"}',
            },
          }
        ],
      },
    )
  )
  assert _running_call_hints(repl) == {'c1': ('write', 'a.py', 'tool')}
  assert _running_call_ids(repl) == ('c1',)
  assert _pending_call_hints(repl) == ()
  assert repl._state.activity.status == 'running tool…'
  assert repl._stream_pane.pane_height == 0
  await _settle(repl)


def test_skill_read_appears_after_path_and_settles_as_loaded() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  call = agent_pb2.StreamToolCall(index=0, name='read')
  repl._stream_pane.on_tool_call(call)
  assert _pending.rows(repl._state) == 0

  repl._stream_pane.on_tool_call(
    agent_pb2.StreamToolCall(
      index=0,
      arguments='{"file_path":"/skills/prs/SKILL.md","line_start":1}',
    )
  )
  pending = _pending.render(repl._state)
  assert '◆ loading $prs' in _ansi_text(pending)

  _handle(
    repl,
    _delta({
      'role': 'assistant',
      'tool_calls': [
        {
          'id': 'c1',
          'type': 'function',
          'function': {
            'name': 'read',
            'arguments': (
              '{"file_path":"/skills/prs/SKILL.md","line_start":1}'
            ),
          },
        }
      ],
    }),
  )
  result = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'instructions'}
  rendered = _handle(repl, _delta(result))
  assert rendered is not None
  assert _render_to_text(rendered).strip() == '◆ loaded $prs'


def test_stream_aborted_clears_pending_tool_call() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(
    _stream_tool_call(index=0, name='edit', arguments='{"file_path":"a.py"}')
  )
  assert repl._state.stream.pending

  repl._handle_message(_stream_aborted())

  assert _pending_call_hints(repl) == ()
  assert repl._state.activity.status == 'thinking…'


def test_streaming_edit_gutter_has_consistent_terminal_width() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  repl._handle_message(
    _stream_tool_call(
      index=0,
      name='edit',
      arguments=json.dumps({
        'file_path': 'mirror-agent/src/local/agent_local_service_test.rs',
        'old_string': 'assert_matches',
      }),
    )
  )

  first_line = _ansi_text(_pending.render(repl._state)).splitlines()[0]

  assert first_line.startswith('· ')
  assert _pt_utils.get_cwidth(first_line) == _cells.cell_len(first_line)
  assert 'edit(mirror-agent/src/local/agent_local_service_test.rs' in first_line


@pytest.mark.asyncio
@pytest.mark.parametrize('columns', [30, 60])
@pytest.mark.parametrize(
  'instruction', ['word ' * 24 + 'END', '界' * 50 + 'END']
)
async def test_queued_instruction_wraps_to_terminal_width(
  columns: int, instruction: str
) -> None:
  terminal = io.StringIO()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=30, columns=columns),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl()
    try:
      repl.dispatch(
        _actions.TurnStarted(), _actions.SubmitInstruction(instruction)
      )
      with _current.set_app(repl._app):
        repl._app.renderer.render(repl._app, repl._app.layout)
    finally:
      await repl.close()

  lines = _screen_lines(terminal.getvalue(), cols=columns, rows=30)
  displayed = ''.join(lines)
  assert instruction.replace(' ', '') in displayed.replace(' ', '')
  assert not any(instruction in line for line in lines)


@pytest.mark.asyncio
@pytest.mark.parametrize('busy', [False, True])
async def test_submit_renders_next_frame_before_transport_ack(
  busy: bool,
) -> None:
  terminal = io.StringIO()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=30, columns=80),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl()
    repl.dispatch(_actions.ReplayFinished(False), _actions.WorkerStarted())
    if busy:
      repl.dispatch(_actions.TurnStarted())
    # The real client's outbox has no connected worker: no ack can arrive.
    repl._client = _rpc_client.Client(
      repl._transport(repl._worker_configuration),
      'conv-1',
      on_message=repl._on_worker_message,
      on_error=repl._exit_with_error,
    )
    try:
      repl._input.text = 'optimistic hello'
      _submit(repl)
      assert repl._input.text == ''
      assert repl._state.turn.pending_instructions == ('optimistic hello',)
      assert repl._transcript.entries() == ()
      with _current.set_app(repl._app):
        repl._app.renderer.render(repl._app, repl._app.layout)
      lines = _screen_lines(terminal.getvalue(), cols=80, rows=30)
      assert any('› optimistic hello' in line for line in lines)
      assert not any('pending' in line for line in lines)
      message_id = repl._state.turn.submitted_inputs[0][0]
      repl._handle_message(
        agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id=message_id)
        )
      )
      assert repl._state.turn.pending_instructions == ('optimistic hello',)
      assert repl._transcript.entries() == ()
    finally:
      await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('columns', [30, 80])
@pytest.mark.usefixtures('frozen_ui_time')
async def test_committing_multiple_inputs_preserves_every_presented_frame(
  columns: int,
) -> None:
  class TerminalBuffer(io.StringIO):
    def __init__(self) -> None:
      super().__init__()
      self.screen = pyte.Screen(columns, 30)
      self.screen.set_mode(pyte.modes.LNM)
      self.stream = pyte.Stream(self.screen)
      self.frames: list[tuple[str, ...]] = []
      self.presented = asyncio.Event()

    @override
    def write(self, text: str) -> int:
      self.stream.feed(text)
      return super().write(text)

    @override
    def flush(self) -> None:
      # pyte stores private terminal modes shifted by five bits.
      if 2026 << 5 not in self.screen.mode:
        self.frames.append(tuple(self.screen.display))
        self.presented.set()

  terminal = TerminalBuffer()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=30, columns=columns),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      file=terminal, width=columns, force_terminal=True
    )
    _seed_history(repl, ['earlier message'])
    await _settle(repl)
    repl.dispatch(
      _actions.ReplayFinished(False),
      _actions.TurnStarted(),
      _actions.InputSubmitted('first', 'First message\nwith another line'),
      _actions.InputSubmitted('second', 'Second message 界' * 3),
      _actions.InputSubmitted('third', 'Third message'),
    )
    repl._input.text = 'unsent draft'
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    expected = tuple(terminal.screen.display)
    terminal.frames.clear()
    terminal.presented.clear()
    try:
      with _current.set_app(repl._app):
        repl._handle_message(
          _delta(
            {'role': 'user', 'content': 'First message\nwith another line'},
            entry_id='first',
          )
        )
        await asyncio.wait_for(terminal.presented.wait(), timeout=3)
        terminal.presented.clear()
        repl._handle_message(
          _delta(
            {'role': 'user', 'content': 'Second message 界' * 3},
            entry_id='second',
          )
        )
        await asyncio.wait_for(terminal.presented.wait(), timeout=3)
        terminal.presented.clear()
        repl._handle_message(
          _delta({'role': 'user', 'content': 'Third message'}, entry_id='third')
        )
        await asyncio.wait_for(terminal.presented.wait(), timeout=3)
      assert set(terminal.frames) == {expected}
      assert repl._state.turn.pending_instructions == ()
    finally:
      repl._app.exit()
      await running
      await repl.close()


@pytest.mark.asyncio
async def test_interrupt_before_idle_send_effect_never_starts_worker() -> None:
  repl = _repl()
  repl.dispatch(_actions.ReplayFinished(False))
  try:
    repl._input.text = 'cancel before sending'
    _submit(repl)
    assert repl._state.turn.pending_instructions == ('cancel before sending',)
    repl.dispatch(_actions.InterruptRequested())
    await asyncio.gather(*tuple(repl._app._background_tasks))
    assert repl._input.text == 'cancel before sending'
    assert repl._state.turn.pending_instructions == ()
    assert not repl._state.lifecycle.started
    assert repl._client is None
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_rapid_sends_keep_ids_order_and_reconcile_once() -> None:
  repl = _repl()
  repl.dispatch(_actions.ReplayFinished(False), _actions.WorkerStarted())
  client = _rpc_client.Client(
    repl._transport(repl._worker_configuration),
    'conv-1',
    on_message=repl._on_worker_message,
    on_error=repl._exit_with_error,
  )
  repl._client = client
  try:
    repl._input.text = 'same text'
    _submit(repl)
    repl._input.text = 'same text'
    _submit(repl)
    first = client._request_queue.get_nowait()
    second = client._request_queue.get_nowait()
    assert first is not None and second is not None
    assert first.message_id != second.message_id
    assert first.message_id == first.message.id
    assert second.message_id == second.message.id
    assert repl._state.turn.input_order == (first.message_id, second.message_id)
    # A workspace handoff does not own or discard the worker's submissions.
    repl.dispatch(_actions.SetActive(False))
    committed = agent_pb2.SessionMessage(agent_message=first.message)
    repl._handle_message(committed)
    repl._handle_message(committed)
    repl._handle_message(
      agent_pb2.SessionMessage(
        input_ack=agent_pb2.InputAck(message_id=first.message_id)
      )
    )
    repl._handle_message(
      agent_pb2.SessionMessage(
        input_nack=agent_pb2.InputNack(
          message_id=first.message_id, error='late'
        )
      )
    )
    repl.dispatch(_actions.SetActive(True))
    assert repl._state.turn.pending_instructions == ('same text',)
    assert repl._state.turn.input_order == (second.message_id,)
    assert repl._state.turn.queued == ('same text',)
    assert repl._transcript.entries() == (first.message,)
    assert repl._input.text == ''
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_closed_transport_restores_submission_without_claiming_delivery() -> (
  None
):
  repl = _repl()
  repl.dispatch(_actions.ReplayFinished(False), _actions.WorkerStarted())
  client = _rpc_client.Client(
    repl._transport(repl._worker_configuration),
    'conv-1',
    on_message=repl._on_worker_message,
    on_error=repl._exit_with_error,
  )
  await client.close()
  repl._client = client
  repl.dispatch(_actions.SetActive(False))
  try:
    repl._input.text = 'recover me'
    _submit(repl)
    assert repl._input.text == 'recover me'
    assert repl._state.turn.pending_instructions == ()
    assert repl._transcript.entries() == ()
    assert 'closed' in _rendering.strip_ansi(repl._render_history())
  finally:
    await repl.close()


@pytest.mark.parametrize('theme_name', ['dark', 'light', 'terminal', 'mono'])
def test_pending_user_text_preserves_authoritative_geometry_and_style(
  theme_name: str,
) -> None:
  repl = _repl(initial_theme=theme_name)
  repl.dispatch(
    _actions.TerminalResized(40), _actions.InputSubmitted('request', 'hello')
  )
  pending = repl._components['queued'].render(repl._state)
  assert isinstance(pending, _containers.Window)
  assert isinstance(pending.content, _controls.FormattedTextControl)
  assert isinstance(pending.content.text, _ansi.ANSI)
  settled = _ansi.ANSI(
    _rendering.render_blocks(
      [_rendering.user_text('hello')], 40, theme_name=theme_name
    )
  )
  assert (
    pending.content.text.__pt_formatted_text__()
    == settled.__pt_formatted_text__()
  )


def test_footer_shows_model_without_session_when_idle() -> None:
  repl = _repl(conversation_id='conversation-1234', model='gpt-4o-mini')
  footer = ''.join(text for _, text in _render_footer(repl))
  assert 'gpt-4o-mini' in footer
  assert 'conversa' not in footer


class _AbovePromptExtension(_StartupExtension):
  @override
  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_ABOVE_PROMPT

  @override
  def render(
    self, global_state: _state.State, extension_state: int
  ) -> _containers.AnyContainer:
    return _containers.Window(height=1)


class _InvalidStyleExtension(_AbovePromptExtension):
  @override
  def render(
    self, global_state: _state.State, extension_state: int
  ) -> _containers.AnyContainer:
    return _containers.Window(
      _controls.FormattedTextControl([('fg:ansibrightwhite', 'Broken widget')]),
      height=1,
    )


@pytest.mark.asyncio
async def test_invalid_extension_style_does_not_break_terminal_renderer() -> (
  None
):
  terminal = io.StringIO()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=12, columns=100),
    term='xterm-256color',
    enable_cpr=False,
  )
  registry = extension.ExtensionRegistry()
  await registry.register(_InvalidStyleExtension(), None)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl(extension_registry=registry)
    try:
      await repl._extension_host.start()
      with _current.set_app(repl._app):
        # Direct renderer calls bypass Application._redraw's frame counter.
        repl._app.render_counter += 1
        repl._app.renderer.render(repl._app, repl._app.layout)
        repl._app.render_counter += 1
        repl._app.renderer.render(repl._app, repl._app.layout)
      displayed = '\n'.join(
        _screen_lines(terminal.getvalue(), cols=100, rows=12)
      )
      assert 'Extension test/startup disabled:' in displayed
      assert 'Wrong color format' in displayed
      assert '›' in displayed
      await registry.unload('test/startup')
      await registry.register(_AbovePromptExtension(), None)
      with _current.set_app(repl._app):
        repl._app.render_counter += 1
        repl._app.renderer.render(repl._app, repl._app.layout)
      assert 'disabled:' not in '\n'.join(
        _screen_lines(terminal.getvalue(), cols=100, rows=12)
      )
    finally:
      await repl.close()
      await registry.close()


@pytest.mark.asyncio
async def test_activity_adds_spacing_only_while_widget_is_visible() -> None:
  registry = extension.ExtensionRegistry()
  plugin = _AbovePromptExtension()
  await registry.register(plugin, None)
  repl = _repl(extension_registry=registry)
  try:
    await repl._extension_host.start()
    idle_height = repl._bottom_pane.preferred_height(80, 40).preferred
    repl.dispatch(_actions.TurnStarted(), _actions.SetStatus('thinking…'))
    assert (
      repl._bottom_pane.preferred_height(80, 40).preferred == idle_height + 2
    )
    repl.dispatch(_actions.TurnFinished())
    assert repl._bottom_pane.preferred_height(80, 40).preferred == idle_height
    await registry.unload(plugin.id())
    empty_height = repl._bottom_pane.preferred_height(80, 40).preferred
    repl.dispatch(_actions.TurnStarted(), _actions.SetStatus('thinking…'))
    assert (
      repl._bottom_pane.preferred_height(80, 40).preferred == empty_height + 2
    )
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
async def test_activity_is_message_aligned_above_prompt_with_delayed_tip(
  tmp_path: pathlib.Path,
  frozen_ui_time: freezegun_api.FrozenDateTimeFactory,
) -> None:
  class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
      return True

  terminal = TerminalBuffer()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=8, columns=80),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    registry = extension.ExtensionRegistry()
    await registry.register(tips_extension.TipsExtension(), None)
    manager = user_extensions.UserExtensions(registry, tmp_path / 'extensions')
    repl = _repl(extension_registry=registry, extension_manager=manager)
    repl._handle_message(_delta({'role': 'user', 'content': 'Run the tool.'}))
    await repl._extension_host.start()
    repl.dispatch(
      _actions.ReplayFinished(has_user_message=False),
      _actions.TurnStarted(observed_at=time.monotonic()),
      _actions.SetStatus('running tool…'),
    )
    rendered = asyncio.Event()

    def after_render(_: _application.Application[str | None]) -> None:
      rendered.set()

    repl._app.after_render += after_render
    active = asyncio.create_task(repl.activate(show_context=False))
    try:
      with _current.set_app(repl._app):
        await asyncio.wait_for(rendered.wait(), timeout=2)
        waiting_height = repl._bottom_pane.preferred_height(80, 40).preferred
        assert 'Tip:' not in terminal.getvalue()
        rendered.clear()
        frozen_ui_time.tick(15)
        # Components retain their layout until the next render frame.
        repl._app.invalidate()
        await asyncio.wait_for(rendered.wait(), timeout=2)
        assert (
          repl._bottom_pane.preferred_height(80, 40).preferred
          == waiting_height + 1
        )
        lines = _screen_lines_with_positions(terminal.getvalue(), rows=8)
        visible_height = repl._bottom_pane.preferred_height(80, 40).preferred
        await repl._run_slash('/tips disable')
        assert (
          repl._bottom_pane.preferred_height(80, 40).preferred
          == visible_height - 1
        )
        assert json.loads((tmp_path / 'config.json').read_text()) == {
          'extensions': {'builtin/tips': {'config': {'visible': False}}},
        }
        await repl._run_slash('/tips')
        assert (
          repl._bottom_pane.preferred_height(80, 40).preferred == visible_height
        )
        await repl._run_slash('/tips enable')
        assert (
          repl._bottom_pane.preferred_height(80, 40).preferred == visible_height
        )
        assert 'Working' in _fragments_text(_render_activity(repl))
    finally:
      repl._app.exit()
      await asyncio.wait_for(active, timeout=2)
      await repl.close()
      await registry.close()

  activity_index = next(
    index for index, line in enumerate(lines) if 'Working' in line
  )
  prompt_index = next(
    index for index, line in enumerate(lines) if line.startswith(' ›')
  )
  assert lines[prompt_index - 1] == ''
  assert lines[prompt_index + 1] == ''
  assert activity_index < prompt_index - 1
  assert lines[activity_index].index('Working') == (_rendering.GUTTER_WIDTH + 2)
  assert lines[activity_index + 1].startswith('  └ Tip: ')
  assert lines[activity_index + 1].index('Tip:') == lines[activity_index].index(
    'Working'
  )
  assert activity_index + 2 < prompt_index
  assert 'running tool…' not in _fragments_text(_render_footer(repl))


def test_dispatch_batches_state_effects_and_invalidation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  invalidations: list[None] = []
  effects: list[_effects.Effect] = []
  monkeypatch.setattr(
    repl._app, 'invalidate', lambda: invalidations.append(None)
  )
  monkeypatch.setattr(repl, '_run_effect', effects.append)

  repl.dispatch(
    _actions.TurnStarted(),
    _actions.SetStatus('thinking…'),
    _actions.DetachRequested(),
    _actions.DetachRequested(),
  )

  assert repl._state.activity.busy
  assert repl._state.activity.status == 'thinking…'
  assert effects == [
    _effects.DetachTools(),
    _effects.DetachTools(),
  ]
  assert invalidations == [None]


def test_activity_is_shown_below_prompt_when_busy() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SetStatus('running tool…'))

  assert 'Working' in _fragments_text(_render_activity(repl))
  assert 'running tool…' not in _fragments_text(_render_footer(repl))


def test_footer_is_unchanged_when_busy() -> None:
  repl = _repl(conversation_id='conversation-1234', model='gpt-4o-mini')
  idle_footer = _render_footer(repl)

  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SetStatus('running tool…'))

  assert _render_footer(repl) == idle_footer


def test_activity_omits_empty_status_when_busy() -> None:
  repl = _repl(model='gpt-4o-mini')
  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SetStatus(''))

  activity = _fragments_text(_render_activity(repl))
  assert activity == ''
  assert repl._state.activity.busy


@pytest.mark.asyncio
async def test_banner_shows_identity_version_model_and_directory() -> None:
  repl = _repl(model='gpt-4o-mini')
  output = _render_banner(repl)
  assert f'Reflection mirror v{distribution.installed().version}' in output
  assert 'gpt-4o-mini' in output
  assert 'directory' in output


@pytest.mark.asyncio
async def test_new_startup_keeps_banner_live() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(create_session=True)
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    repl._app.pre_run_callables.append(repl._app.exit)
    await asyncio.wait_for(_run_repl(repl), timeout=10)

  assert 'Reflection' not in repl._terminal.console.export_text()
  assert 'Reflection mirror' in _render_banner(repl)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'contents',
  [
    '[{"keys": ["f6"], "action": {"type": "CloseSelector"}}]',
    '[{"keys": [["c-c"]], "action": null}]',
    '{',
  ],
)
async def test_startup_reports_invalid_keybindings_and_uses_defaults(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: pathlib.Path,
  contents: str,
) -> None:
  config = tmp_path / '.mirror' / 'keybindings.json'
  config.parent.mkdir()
  config.write_text(contents)
  monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(
      input=pipe, output=_SizedOutput(rows=24, columns=1000)
    ),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(record=True, width=1000)
    pipe.send_text('\x03')
    await asyncio.wait_for(_run_repl(repl), timeout=10)

  output = _render_welcome_notices(repl)
  assert 'Invalid keybindings config' in output
  assert str(config) in output
  assert 'Using default keybindings.' in output
  assert repl._client is None
  assert config.read_text() == contents


@pytest.mark.asyncio
async def test_app_quits_on_ctrl_c_and_stops_worker() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    pipe.send_text('\x03')
    await asyncio.wait_for(_run_repl(repl), timeout=10)

  assert repl._client is None


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['draft', 'first\nsecond', ' \n', '/he'])
@pytest.mark.parametrize(
  ('keys', 'sequence'),
  [(['c-c'], '\x03'), (['c-g'], '\x07'), (['c-x', 'g'], '\x18g')],
)
async def test_cancel_or_quit_is_rebindable_and_clears_the_whole_prompt(
  text: str,
  keys: list[str],
  sequence: str,
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  if keys != ['c-c']:
    path = tmp_path / '.mirror' / 'keybindings.json'
    path.parent.mkdir()
    path.write_text(
      json.dumps([
        {'action': {'type': 'CancelOrQuitRequested'}, 'keys': [keys]},
      ])
    )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    running = asyncio.create_task(repl._app.run_async())
    try:
      await asyncio.sleep(0.05)
      repl._input.text = text
      repl._input.cursor_position = 1
      if keys != ['c-c']:
        pipe.send_text('\x03')
        await asyncio.sleep(0.05)
        assert not running.done()
        assert repl._input.text == text
      pipe.send_text(sequence)
      await asyncio.sleep(0.05)
      assert not running.done()
      assert repl._input.text == ''
      assert repl._state.composer == _actions.ComposerState()
      assert not _completion.visible(repl._state)
      assert not repl._state.turn.submitted_inputs
      assert text in repl._history.get_strings()
      pipe.send_text(sequence)
      await asyncio.wait_for(running, timeout=2)
    finally:
      if not running.done():
        repl._app.exit()
      await running


@pytest.mark.asyncio
async def test_config_can_restore_immediate_quit_with_a_draft(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  path = tmp_path / '.mirror' / 'keybindings.json'
  path.parent.mkdir()
  path.write_text(
    json.dumps([
      {'action': {'type': 'QuitRequested'}, 'keys': [['c-c'], ['c-d']]},
    ])
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    pipe.send_text('draft\x03')
    await asyncio.wait_for(repl._app.run_async(), timeout=2)
  assert repl._input.text == 'draft'


@pytest.mark.asyncio
async def test_startup_reports_worker_error() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(
      worker_source="print('MIRROR_RPC_ERROR=missing Codex login', flush=True)"
    )
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    await asyncio.wait_for(_run_repl(repl), timeout=10)

  output = repl._terminal.console.export_text()
  assert 'could not start agent worker' in output
  assert 'missing Codex login' in output


@pytest.mark.asyncio
async def test_shift_enter_inserts_a_newline_without_submitting() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    # 'foo', Shift+Enter (CSI-u bytes → newline), 'bar', then Ctrl+D to quit.
    pipe.send_text('foo\x1b[13;2ubar\x04')
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  assert repl._input.text == 'foo\nbar'


@pytest.mark.asyncio
async def test_ctrl_underscore_undoes_composer_text() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    pipe.send_text('x\x1f\x03')  # type, Ctrl+_, Ctrl+C.
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  assert repl._input.text == ''


@pytest.mark.asyncio
async def test_ctrl_y_redoes_composer_text() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    pipe.send_text('x\x1f\x19\x04')  # type, Ctrl+_, Ctrl+Y, Ctrl+D.
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  assert repl._input.text == 'x'


@pytest.mark.asyncio
async def test_ctrl_shift_z_csi_u_redoes_composer_text() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    pipe.send_text('x\x1f\x1b[122;6u\x04')
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  assert repl._input.text == 'x'


@pytest.mark.asyncio
async def test_rerender_keys_do_not_clear_composer_redo() -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_view_mode.ViewModeExtension(), None)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    # type, Ctrl+_, Ctrl+L, Ctrl+T, Ctrl+Y, Ctrl+D.
    pipe.send_text('x\x1f\x0c\x14\x19\x04')
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  assert repl._input.text == 'x'

  await registry.close()


def _slash_menu_text(repl: cli._Repl) -> str:
  return ''.join(text for _, text in _render_completion(repl))


def test_slash_menu_navigation_and_completion() -> None:
  repl = _repl()
  repl._input.text = '/'
  text = _slash_menu_text(repl)
  assert text.splitlines()[1].startswith(
    f' Commands  (1/{len(repl.get_available_commands())})'
  )
  assert ' › /help' in text
  assert 'move · Enter' not in text
  assert text.endswith('▾')
  _move_completion(repl, 1)
  _complete_selection(repl)
  assert repl._input.text == '/status '


def test_completion_escape_resets_composer_through_an_effect() -> None:
  repl = _repl()
  repl._input.text = '/he'

  repl.dispatch(_actions.CloseSelector())

  assert repl._input.text == ''
  assert repl._state.composer == _actions.ComposerState()


def test_selection_actions_move_and_accept_completion() -> None:
  repl = _repl()
  repl._input.text = '/'

  repl.dispatch(_actions.MoveSelection(1))
  repl.dispatch(_actions.AcceptSelection())

  assert repl._input.text == '/status '


def test_selection_actions_move_and_accept_picker() -> None:
  repl = _repl()
  repl.dispatch(
    _actions.OpenSelector(
      _selection.Selector(
        'test',
        'Pick a prompt',
        (
          _selection.Choice('first', 'First'),
          _selection.Choice('second', 'Second'),
        ),
        _actions.RestorePrompt,
      )
    )
  )

  repl.dispatch(_actions.MoveSelection(1), _actions.AcceptSelection())

  assert not repl._state.selector.active
  assert repl._input.text == 'second'


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('numbered', 'input_height', 'active', 'query', 'status'),
  [(True, 0, False, '', 'second'), (False, 1, True, '2', '')],
)
async def test_selector_number_keys_or_search_preserve_composer(
  numbered: bool,
  input_height: int,
  active: bool,
  query: str,
  status: str,
) -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._input.text = 'unfinished draft'
    repl.dispatch(
      _actions.OpenSelector(
        _selection.Selector(
          'test',
          'Choose',
          (
            _selection.Choice('first', 'First'),
            _selection.Choice('second', 'Second 2'),
          ),
          _actions.SetStatus,
          numbered=numbered,
        )
      )
    )
    assert repl._input_window.preferred_height(80, 20).preferred == input_height
    running = asyncio.create_task(repl._app.run_async())
    try:
      await asyncio.sleep(0.05)
      pipe.send_text('2')
      await asyncio.sleep(0.05)
      assert repl._state.selector.active is active
      assert repl._state.selector.query == query
      assert repl._state.activity.status == status
      assert repl._input.text == 'unfinished draft'
      repl.dispatch(_actions.CloseSelector())
      assert repl._input_window.content is repl._composer_control
      assert repl._input_window.preferred_height(80, 20).preferred == 1
    finally:
      repl._app.exit()
      await asyncio.wait_for(running, timeout=2)
      await repl.close()


@pytest.mark.asyncio
async def test_numbered_selector_ignores_search_and_out_of_range_shortcuts() -> (
  None
):
  repl = _repl()
  try:
    repl.dispatch(
      _actions.OpenSelector(
        _selection.Selector(
          'test',
          'Choose',
          (_selection.Choice('first', 'First'),),
          _actions.SetStatus,
          numbered=True,
        )
      )
    )
    assert repl._bottom_pane.preferred_height(80, 40).max < 12
    repl.dispatch(_actions.SelectorSearchChanged('unmatched'))
    repl.dispatch(_actions.AcceptSelection(-1), _actions.AcceptSelection(1))
    assert repl._state.selector.active
    assert repl._state.selector.query == ''
    assert repl._state.activity.status == ''
    repl.dispatch(_actions.AcceptSelection())
    assert repl._state.activity.status == 'first'
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_accept_selection_returns_started_tasks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._transcript.append(
    agent_pb2.AgentMessage(
      chat=agent_pb2.ChatRecord(user=agent_pb2.UserMessage(content='hello'))
    )
  )
  emitted = asyncio.Event()

  async def emit(renderables: Iterable[_console.RenderableType]) -> None:
    del renderables
    await asyncio.sleep(0)
    emitted.set()

  monkeypatch.setattr(repl._terminal, 'emit', emit)
  tasks = repl.dispatch(
    _actions.OpenSelector(
      _selection.Selector(
        'test',
        'Pick',
        (_selection.Choice('selected', 'Selected'),),
        _actions.ShowText,
      )
    ),
    _actions.AcceptSelection(),
  )

  assert len(tasks) == 1
  await asyncio.gather(*tasks)
  assert emitted.is_set()
  assert not repl._state.selector.active


def test_close_selector_dismisses_without_accepting() -> None:
  repl = _repl()
  repl._input.text = 'draft'
  repl.dispatch(
    _actions.OpenSelector(
      _selection.Selector(
        'test',
        'Pick a prompt',
        (_selection.Choice('replacement', 'Replacement'),),
        _actions.RestorePrompt,
      )
    )
  )

  repl.dispatch(_actions.CloseSelector())

  assert not repl._state.selector.active
  assert repl._input.text == 'draft'


def test_selection_actions_do_nothing_without_a_picker() -> None:
  repl = _repl()
  repl._input.text = 'draft'

  repl.dispatch(_actions.MoveSelection(1))
  repl.dispatch(_actions.AcceptSelection())
  repl.dispatch(_actions.CloseSelector())

  assert repl._input.text == 'draft'
  assert not repl._state.selector.active


def test_slash_menu_prompt_preview() -> None:
  repl = _repl()
  repl._input.text = '/'
  assert _completion_preview(repl) == [
    ('class:slash-autocomplete', 'help'),
    ('class:slash-preview-action', '  ↵'),
  ]


@pytest.mark.asyncio
async def test_new_command_closes_slash_menu_before_switch() -> None:
  repl = _repl()
  repl._input.text = '/'
  new_index = next(
    index
    for index, command in enumerate(_commands.COMMANDS)
    if command.name == 'new'
  )
  repl.dispatch(
    _actions.MoveCompletionRequested(
      new_index,
      skill_count=0,
      command_count=len(repl.get_available_commands()),
    )
  )
  assert _completion.visible(repl._state)

  _submit(repl)

  assert repl._input.text == ''
  assert repl._state.composer == _actions.ComposerState()
  assert not _completion.visible(repl._state)
  task = next(iter(repl._app._background_tasks))
  task.cancel()
  await asyncio.gather(task, return_exceptions=True)


def test_completion_preview_renders_at_cursor_on_later_line() -> None:
  input_document = _document.Document(
    'First line\nUse $p later', cursor_position=len('First line\nUse $p')
  )
  control = _controls.BufferControl(
    buffer=_buffer.Buffer(document=input_document)
  )
  preview = [
    ('class:skill-autocomplete', 'df'),
    ('class:skill-preview-action', '  ↵'),
  ]
  processor = _composer_processors.CompletionPreviewProcessor(lambda: preview)
  first_line = processor.apply_transformation(
    _processors.TransformationInput(
      control,
      input_document,
      0,
      lambda index: index,
      [('', 'First line')],
      80,
      1,
    )
  )
  cursor_line = processor.apply_transformation(
    _processors.TransformationInput(
      control,
      input_document,
      1,
      lambda index: index,
      [('', 'Use $p later')],
      80,
      1,
    )
  )

  assert first_line.fragments == [('', 'First line')]
  assert cursor_line.fragments == [
    ('', 'Use $p'),
    *preview,
    ('', ' later'),
  ]


def test_slash_alias_preview_matches_tab_completion() -> None:
  repl = _repl()
  repl._input.text = '/exi'
  assert _completion_preview(repl) == [
    ('class:slash-autocomplete', 't'),
    ('class:slash-preview-action', '  ↵'),
  ]
  _complete_selection(repl)
  assert repl._input.text == '/exit '


def test_slash_preview_hides_after_incomplete_command_and_space() -> None:
  repl = _repl()
  repl._input.text = '/he '

  assert _completion_preview(repl) is None


def test_slash_menu_soft_scroll_when_truncated() -> None:
  repl = _repl()
  repl._input.text = '/'
  text = _slash_menu_text(repl)
  total = len(repl.get_available_commands())
  assert text.splitlines()[1].startswith(f' Commands  (1/{total})')
  assert ('class:menu-scroll-disabled', '▴') in _render_completion(repl)
  assert text.endswith('▾')
  assert '...' not in text
  scroll_column = text.splitlines()[-1].index('▾')
  assert _completion_height(repl) == repl._state.completion.slash.max_rows + 3

  selected = total // 2
  repl.dispatch(
    _actions.MoveCompletionRequested(
      selected, skill_count=0, command_count=total
    )
  )
  middle_lines = _slash_menu_text(repl).splitlines()
  assert middle_lines[1].endswith('▴')
  assert middle_lines[-1].endswith('▾')
  assert middle_lines[1].index('▴') == middle_lines[-1].index('▾')
  assert middle_lines[1].index('▴') == scroll_column

  repl.dispatch(
    _actions.MoveCompletionRequested(
      total - selected - 1,
      skill_count=0,
      command_count=total,
    )
  )
  text = _slash_menu_text(repl)
  assert text.splitlines()[1].startswith(f' Commands  ({total}/{total})')
  assert text.splitlines()[1].endswith('▴')
  assert ('class:menu-scroll-disabled', '▾') in _render_completion(repl)
  assert _completion_height(repl) == repl._state.completion.slash.max_rows + 3


def test_slash_menu_keeps_locked_commands_while_busy() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl._input.text = '/'
  names = {
    command.name
    for command in _commands.menu_commands(
      repl._state.composer.text, repl.get_available_commands()
    )
    or ()
  }
  assert names == {command.name for command in repl.get_available_commands()}
  # Navigate to clear and confirm the row lock + prompt busy hint.
  clear_index = next(
    i for i, c in enumerate(_commands.COMMANDS) if c.name == 'clear'
  )
  repl.dispatch(
    _actions.MoveCompletionRequested(
      clear_index,
      skill_count=0,
      command_count=len(repl.get_available_commands()),
    )
  )
  text = _slash_menu_text(repl)
  assert f'› {_completion.LOCK}/clear' in text
  assert _completion_preview(repl) == [
    ('class:slash-autocomplete', 'clear'),
    ('class:slash-preview-locked', '  · unavailable while busy'),
  ]


def test_submit_handler_dispatches_one_interaction_action(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._input.text = 'hello'
  actions: list[_actions.Action] = []

  def dispatch(*submitted: _actions.Action) -> tuple[asyncio.Task[None], ...]:
    actions.extend(submitted)
    return ()

  monkeypatch.setattr(repl, 'dispatch', dispatch)

  _submit(repl)

  assert actions == [_actions.ComposerSubmit()]


def test_enter_completes_argument_command(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  rename = _commands.CommandDef(
    'rename',
    'Rename the conversation',
    lambda _, args: _actions.ShowText(args),
    arguments=_commands.RequiredArguments('<title>'),
  )
  monkeypatch.setattr(_commands, 'COMMANDS', (rename,))
  repl = _repl()
  repl._input.text = '/r'
  assert _completion_preview(repl) == [
    ('class:slash-autocomplete', 'ename'),
    ('class:slash-preview-args', '  <title>'),
    ('class:slash-preview-action', '  ↵'),
  ]
  assert '<title>' not in _slash_menu_text(repl)
  _submit(repl)
  assert repl._input.text == '/rename '
  assert _completion_preview(repl) == [
    ('class:slash-argument-placeholder', '<title>'),
  ]
  repl._input.text += 'Release prep'
  assert _completion_preview(repl) is None


@pytest.mark.asyncio
async def test_commands_and_completion_follow_extension_load_and_unload() -> (
  None
):
  registry = extension.ExtensionRegistry()
  repl = _repl(extension_registry=registry)
  completion = _completion.Completion()
  await repl._extension_host.start()
  try:
    await registry.load(
      pathlib.Path(cli.__file__).with_name('extensions') / 'inference_controls'
    )
    repl._input.text = '/reason'
    assert 'reasoning' in {c.name for c in repl.get_available_commands()}
    assert 'reasoning' in ''.join(
      text for _, text in _component_text(completion.render(repl._state))
    )
    await registry.unload('core/inference-controls')
    assert 'reasoning' not in {c.name for c in repl.get_available_commands()}
    assert 'reasoning' not in ''.join(
      text for _, text in _component_text(completion.render(repl._state))
    )
  finally:
    await repl._extension_host.close()
    await registry.close()


@pytest.mark.asyncio
async def test_enter_opens_reasoning_selector_instead_of_editing_arguments() -> (
  None
):
  registry = extension.ExtensionRegistry()
  await registry.load(
    pathlib.Path(cli.__file__).with_name('extensions') / 'inference_controls'
  )
  repl = _repl(extension_registry=registry)
  await repl._extension_host.start()
  repl._input.text = '/reason'

  try:
    _submit(repl)
    assert repl._input.text == ''
    await asyncio.gather(*tuple(repl._app._background_tasks))
  finally:
    await repl._extension_host.close()
    await registry.close()

  assert repl._state.selector.active
  window = _selector.Selector().render(repl._state)
  assert isinstance(window.content, _controls.FormattedTextControl)
  assert not callable(window.content.text)
  assert 'medium  (current)' in _fragments_text(window.content.text)


def test_enter_executes_argument_free_command(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._input.text = '/he'
  scheduled: list[object] = []

  def schedule(coroutine) -> None:
    scheduled.append(coroutine)
    coroutine.close()

  monkeypatch.setattr(repl._app, 'create_background_task', schedule)
  _submit(repl)
  assert repl._input.text == ''
  assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_resume_uses_short_lived_database_access(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  current_id = '11111111-current'
  other_id = '22222222-other'
  with cli_store.Store(database) as store:
    store.record_user_message(
      current_id,
      entry_id='current-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='current work',
      cwd=str(tmp_path),
      model='m-test',
    )
    store.record_user_message(
      other_id,
      entry_id='other-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved work',
      cwd=str(tmp_path),
      model='m-test',
    )

  repl = _repl(conversation_id=current_id, database=database)
  restarts = _record_restarts(repl, monkeypatch)
  await repl._run_slash('/resume 2222')

  assert restarts == [
    (
      {
        'conversation_id': other_id,
        'create_session': False,
        'allow_workspace_mismatch': False,
      },
      True,
      other_id,
    )
  ]
  assert repl._state.session.pending_switch is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', [False, True])
async def test_host_tool_api_routes_through_rpc_without_approval(
  remote: bool,
) -> None:
  class ToolClient(_rpc_client.Client):
    @override
    async def call_tool(
      self, name: str, arguments: Mapping[str, object]
    ) -> _host_tools.ToolResult:
      assert name == 'test.save'
      assert arguments == {'value': 1}
      return _host_tools.ToolResult('from Rust', True)

  async def on_error(message: str) -> None:
    pytest.fail(message)

  repl = _repl(remote_endpoint='test.invalid:443' if remote else None)
  repl._client = ToolClient(
    _remote_rpc_client.Transport('test.invalid:443'),
    'session',
    on_message=lambda _: None,
    on_error=on_error,
  )
  repl.dispatch(_actions.SetApprovalMode('hitl'))

  repl._host_tools_registry.register('test', [_StartupExtension('test.save')])
  repl._handle_message(_approval_requested('worker-call', 'bash'))
  assert await repl.call_tool(
    'test.save', {'value': 1}
  ) == _host_tools.ToolResult('from Rust', True)
  assert repl._state.approval.request is not None
  assert repl._state.approval.request.call_id == 'worker-call'
  repl._handle_message(
    agent_pb2.SessionMessage(
      hitl_resolved=agent_pb2.HITLResolved(call_id='worker-call')
    )
  )
  assert repl._state.approval.request is None


@pytest.mark.asyncio
@pytest.mark.parametrize('remote_endpoint', [None, 'remote.example:443'])
async def test_extensions_read_the_session_transcript(
  remote_endpoint: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
  registry = extension.ExtensionRegistry()
  create_host = mock.Mock(wraps=registry.create_host)
  monkeypatch.setattr(registry, 'create_host', create_host)
  repl = _repl(remote_endpoint=remote_endpoint, extension_registry=registry)
  context = create_host.call_args.args[0]
  replayed = (_user_record('first'), _assistant_record('second'))
  live = _delta({'role': 'user', 'content': 'third'}, entry_id='live')
  try:
    assert context.transcript() == ()
    _seed_transcript(repl, *replayed)
    repl._handle_message(live)
    snapshot = context.transcript()
    assert snapshot == (*replayed, live.agent_message)
    snapshot[0].id = 'edited'
    assert repl.transcript_entries()[0].id == replayed[0].id
    repl.dispatch(
      _actions.SessionSwitchStarted(_actions.PendingSwitch('conv-2', False))
    )
    assert context.transcript() == ()
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_rename_delegates_to_workspace_controller(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl(create_session=True)
  _seed_history(repl, ['message'])
  emitted: list[_console.RenderableType] = []

  async def emit(renderables: list[_console.RenderableType]) -> None:
    emitted.extend(renderables)

  monkeypatch.setattr(repl._terminal, 'emit', emit)

  class Workspaces:
    def rename(self, alias: str) -> str:
      assert alias == 'alias'
      return alias

  controller = cast(_workspace_controller.WorkspaceController, Workspaces())
  await cli.handle_workspace_effect(
    controller, repl, _effects.RenameWorktree('alias')
  )
  await _settle(repl)
  assert "Worktree alias set to 'alias'" in _render_to_text(emitted[0])


@pytest.mark.asyncio
async def test_inference_changes_during_rust_turn_without_restarting_worker(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl(model='first-model')
  repl._worker_configuration = repl._worker_configuration.updated({
    'api': 'openai-responses',
    'base_url': 'https://first.example.test/v1',
  })
  repl.dispatch(_actions.TurnStarted())
  restarts = _record_restarts(repl, monkeypatch)
  sent: list[agent_pb2.ClientMessage] = []
  monkeypatch.setattr(repl, '_send_request', sent.append)

  repl.dispatch(_actions.SetReasoning('high'))
  await repl._switch_model(
    _models.ModelOption(
      'next-model',
      'Next',
      api='openai-chat-completions',
      base_url='https://next.example.test/v1',
    )
  )

  assert not restarts
  assert repl._state.activity.busy
  assert [request.set_inference_config for request in sent] == [
    agent_pb2.InferenceConfig(
      model='first-model',
      api='openai-responses',
      base_url='https://first.example.test/v1',
      reasoning_effort='high',
    ),
    agent_pb2.InferenceConfig(
      model='next-model',
      api='openai-chat-completions',
      base_url='https://next.example.test/v1',
      reasoning_effort='high',
    ),
  ]


def test_agent_message_executes_through_client() -> None:
  repl = _repl()
  sent: list[agent_pb2.AgentMessage] = []

  class _Client:
    def enqueue(self, message: agent_pb2.AgentMessage) -> None:
      sent.append(message)

  repl._client = _Client()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
  message = agent_pb2.AgentMessage(
    id='extension-1',
    extension=agent_pb2.ExtensionRecord(extension='todo', kind='state'),
  )

  repl._enqueue(message)  # pyright: ignore[reportPrivateUsage]

  assert sent == [message]


def test_submit_then_interrupt_is_queued_before_worker_connects() -> None:
  repl = _repl()
  sent: list[agent_pb2.ClientMessage] = []

  class _Client:
    def send(self, request: agent_pb2.ClientMessage) -> None:
      sent.append(request)

  repl._client = _Client()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
  repl.dispatch(_actions.WorkerLoadingStarted())
  repl._input.text = 'do not run this'

  _submit(repl)
  repl.dispatch(_actions.InterruptRequested())

  assert [request.WhichOneof('payload') for request in sent] == [
    'message',
    'interrupt',
  ]
  assert sent[0].message.chat.user.content == 'do not run this'
  assert repl._input.text == 'do not run this'
  assert not repl._state.turn.queued


@pytest.mark.asyncio
async def test_new_session_restarts_worker(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  restarts = _record_restarts(repl, monkeypatch)

  await _dispatch_and_wait(repl, _actions.NewSession())

  updates, replay, conversation_id = restarts[0]
  assert replay
  assert updates['create_session'] is True
  assert updates['conversation_id'] == conversation_id
  assert len(str(conversation_id)) == 32


@pytest.mark.asyncio
@pytest.mark.parametrize('create', [False, True])
async def test_session_lifecycle_across_start_new_resume_and_retry(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, create: bool
) -> None:
  repl = _repl(database=tmp_path / 'state.db', create_session=create)
  observed: list[_actions.Action] = []
  dispatch = repl.dispatch

  def record(*actions: _actions.Action) -> tuple[asyncio.Task[None], ...]:
    observed.extend(actions)
    return dispatch(*actions)

  async def start(client: _rpc_client.Client, *, replay: bool) -> None:
    del client, replay

  async def close(client: _rpc_client.Client) -> None:
    del client

  monkeypatch.setattr(repl, 'dispatch', record)
  monkeypatch.setattr(_rpc_client.Client, 'start', start)
  monkeypatch.setattr(_rpc_client.Client, 'close', close)
  await repl.start()
  await _dispatch_and_wait(repl, _actions.NewSession())
  new_id = repl._state.session.conversation_id
  await repl._switch_session('resumed', create=False)
  await repl._retry_worker()
  await repl.close()

  assert [
    action.conversation_id
    for action in observed
    if isinstance(action, _actions.SessionCreated)
  ] == (['conv-1'] if create else []) + [new_id]
  assert [
    action.conversation_id
    for action in observed
    if isinstance(action, _actions.SessionAttached)
  ] == ['conv-1', new_id, 'resumed', 'resumed']


@pytest.mark.asyncio
async def test_resume_picker_scrolls_and_selects(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  current_id = 'current'
  with cli_store.Store(database) as store:
    for index in range(10):
      conversation_id = f'conversation-{index}'
      store.record_user_message(
        conversation_id,
        entry_id=f'entry-{index}',
        sequence=0,
        timestamp=datetime.datetime.now(datetime.UTC),
        content=f'work {index}',
        cwd=str(tmp_path),
        model='m-test',
      )

  repl = _repl(conversation_id=current_id, database=database)
  restarts = _record_restarts(repl, monkeypatch)
  await _dispatch_and_wait(repl, _actions.ResumeSession())

  choices = repl._state.selector.filtered_choices()
  assert len(choices) == 10
  assert repl._input.read_only()
  assert _resume_height(repl) == 12
  for _ in range(9):
    repl.dispatch(_actions.MoveSelectorSelection(1))
  fragments = _render_resume(repl)
  rendered = ''.join(text for _, text in fragments)
  assert choices[-1].label is not None
  assert choices[-1].label in rendered
  assert '(10/10)' in rendered
  border_fragments = [
    text for style, text in fragments if style == 'class:menu-border'
  ]
  assert border_fragments.count(' ') == 8
  assert '╰─' not in border_fragments
  assert '[cwd]' in rendered.splitlines()[1]
  assert rendered.splitlines()[2].endswith('▴')
  assert 'Tab scope' in rendered.splitlines()[2]
  assert ('class:menu-scroll-disabled', '▾') in fragments

  _submit(repl)
  await _settle(repl)

  assert restarts == [
    (
      {
        'conversation_id': choices[-1].value,
        'create_session': False,
        'allow_workspace_mismatch': False,
      },
      True,
      choices[-1].value,
    )
  ]
  assert not repl._state.selector.active
  assert not repl._state.selector.filtered_choices()
  assert not repl._input.read_only()


def _user_record(
  content: str, *, display: str | None = None
) -> agent_pb2.AgentMessage:
  return _delta(
    {'role': 'user', 'content': content},
    display=display,
    entry_id=str(uuid.uuid4()),
  ).agent_message


def _assistant_record(content: str) -> agent_pb2.AgentMessage:
  return _delta(
    {'role': 'assistant', 'content': content}, entry_id=str(uuid.uuid4())
  ).agent_message


def _rewindable_repl(
  tmp_path: pathlib.Path, *entries: agent_pb2.AgentMessage
) -> cli._Repl:
  """A REPL over a saved conversation whose worker holds `entries`."""
  database = tmp_path / 'state.db'
  records: list[agent_pb2.AgentMessage] = []
  for sequence, entry in enumerate(entries):
    record = agent_pb2.AgentMessage()
    record.CopyFrom(entry)
    record.sequence = sequence
    records.append(record)

  class _Client:
    async def messages(self) -> list[agent_pb2.AgentMessage]:
      return records

    async def fork_session(self, entry_id: str) -> str:
      assert any(record.id == entry_id for record in records)
      return 'forked'

    async def close(self) -> None:
      pass

  repl = _repl(
    conversation_id='current', database=database, create_session=True
  )
  for record in records:
    repl._record_cli_message(record)  # pyright: ignore[reportPrivateUsage]
  repl._client = _Client()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
  repl.dispatch(_actions.WorkerStarted())
  return repl


def _seed_transcript(
  repl: cli._Repl,
  *entries: agent_pb2.AgentMessage,
) -> None:
  """Land stored message records in the transcript, as their replay would."""
  for entry in entries:
    repl._handle_message(agent_pb2.SessionMessage(agent_message=entry))
  repl._handle_message(_replay_done('current'))


@pytest.mark.asyncio
async def test_rewind_previews_a_turn_without_discarding_history(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  reply = _assistant_record('on it')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, reply, second)
  _seed_transcript(repl, first, reply, second)
  sent: list[agent_pb2.ClientMessage] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  paints = _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  await _dispatch_and_wait(repl, _actions.RewindSession())

  # Choices run newest first, so the top row is the turn just taken.
  choices = repl._state.selector.filtered_choices()
  assert [choice.value for choice in choices] == [second.id, first.id]
  assert repl._input.read_only()
  rendered = ''.join(text for _, text in _render_rewind_picker(repl))
  assert '↩1' in rendered and 'second' in rendered

  _submit(repl)
  await _settle(repl)

  staged = repl._state.rewind.staged
  assert staged is not None and staged.target.entry_id == second.id
  # The chosen message is handed back for editing, and only the history before
  # it is drawn. Nothing is deleted at this point.
  assert repl._input.text == 'second'
  assert not repl._state.selector.active
  assert staged.discards == 1
  # The preview is drawn from the transcript the TUI already owns, without a
  # worker round trip.
  assert sent == []
  ansi, clear = paints[-1]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'first' in ansi and 'on it' in ansi
  assert 'second' not in ansi

  # A standing banner says how to get back to the tip of the conversation.
  banner = ''.join(text for _, text in _render_rewind_banner(repl))
  assert 'rewound past 1 turn(s)' in banner
  assert 'Esc returns to the latest turn' in banner

  # A view-mode change re-renders without silently restoring the hidden tail.
  await repl._rerender()
  assert 'second' not in paints[-1][0]
  assert sent == []


@pytest.mark.asyncio
async def test_rewind_does_not_offer_injected_skill_context(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Skill text is stored as a user message, but it is part of a turn rather
  # than a turn of its own, so it is not a place to continue from.
  monkeypatch.chdir(tmp_path)
  typed = _user_record('use $helper')
  injected = _user_record('the whole skill body', display='$helper')
  repl = _rewindable_repl(tmp_path, typed, injected)

  await _dispatch_and_wait(repl, _actions.RewindSession())

  assert [
    choice.value for choice in repl._state.selector.filtered_choices()
  ] == [typed.id]


@pytest.mark.asyncio
async def test_rewind_picker_keeps_down_out_of_the_task_list(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  repl = _rewindable_repl(tmp_path, _user_record('first'))
  repl._handle_message(_task_delta('1234', 'running'))
  assert repl._can_focus_tasks()

  await _dispatch_and_wait(repl, _actions.RewindSession())

  # The picker's own Down binding is not eager, so the task list's eager one
  # must step aside while the picker is open.
  assert repl._state.selector.active
  assert not repl._can_focus_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('view_mode', 'tool_label'),
  [
    (_actions.ViewMode.COMPACT, '· bash ls'),
    (_actions.ViewMode.TRANSCRIPT, '· bash(ls)'),
  ],
)
async def test_rewind_preview_hides_unanswered_calls_from_the_tail(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  view_mode: _actions.ViewMode,
  tool_label: str,
) -> None:
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  repl.dispatch(_actions.SetViewMode(view_mode))
  # The second turn was interrupted mid-call: no result ever answered it.
  repl._handle_message(
    _delta({'role': 'user', 'content': 'first'}, entry_id=first.id)
  )
  repl._handle_message(
    _delta({'role': 'user', 'content': 'second'}, entry_id=second.id)
  )
  repl._handle_message(_delta(_bash_call()))
  repl._handle_message(_replay_done('current'))
  history = _rendering.strip_ansi(repl._render_history())
  assert tool_label in history
  assert '└' not in history
  paints = _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', _noop_command)

  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)

  ansi, clear = paints[-1]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'first' in ansi
  # The interrupted turn's dangling call is set aside with the rest of it.
  assert tool_label not in _rendering.strip_ansi(ansi)


@pytest.mark.asyncio
async def test_rewind_picker_drops_a_read_a_switch_overtook(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  repl = _rewindable_repl(tmp_path, _user_record('first'))
  read_turns = repl._user_turns

  async def read() -> tuple[_actions.UserTurn, ...]:
    turns = await read_turns()
    repl.dispatch(_actions.SessionOpened('elsewhere'))
    return turns

  monkeypatch.setattr(repl, '_user_turns', read)
  await _dispatch_and_wait(repl, _actions.RewindSession())

  # The turns belong to a conversation that is no longer on screen.
  assert not repl._state.selector.filtered_choices()


@pytest.mark.asyncio
async def test_rewinding_twice_counts_from_the_end_of_the_conversation(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  second = _user_record('second')
  third = _user_record('third')
  repl = _rewindable_repl(tmp_path, first, second, third)
  _seed_transcript(repl, first, second, third)

  def send(request: agent_pb2.ClientMessage) -> None:
    del request

  _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)

  # Rewind to the newest turn, then reopen the picker and go further back.
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  staged = repl._state.rewind.staged
  assert staged is not None and staged.discards == 1

  await _dispatch_and_wait(repl, _actions.RewindSession())
  repl.dispatch(_actions.MoveSelectorSelection(1))
  repl.dispatch(_actions.MoveSelectorSelection(1))
  _submit(repl)
  await _settle(repl)

  staged = repl._state.rewind.staged
  assert staged is not None and staged.target.entry_id == first.id
  # The count is absolute, so restaging never double counts hidden turns.
  assert staged.discards == 3


@pytest.mark.asyncio
async def test_sending_after_a_rewind_confirms_before_forking(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  _seed_transcript(repl, first, second)
  sent: list[agent_pb2.ClientMessage] = []
  emitted: list[_console.RenderableType] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  monkeypatch.setattr(repl._terminal, 'queue_emit', emitted.extend)
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  sent.clear()

  repl._input.text = 'try something else'
  _submit(repl)

  # Enter opens the confirmation instead of starting a turn.
  assert repl._state.rewind.confirming
  assert sent == []
  warning = ''.join(text for _, text in _render_rewind_picker(repl))
  assert '1 later turn(s) stay in the original conversation' in warning
  assert 'not restored' in warning  # Files are explicitly out of scope.
  assert '↵/y fork and send' in warning

  # Declining keeps both the staged rewind and the typed message.
  repl.dispatch(_actions.SetRewindConfirming(False))
  assert not repl._state.rewind.confirming
  assert repl._state.rewind.staged is not None
  assert repl._input.text == 'try something else'

  _submit(repl)
  assert repl._state.rewind.confirming

  parent = repl._client
  assert parent is not None

  async def start_child(*, replay: bool) -> None:
    assert replay
    assert repl._state.session.conversation_id == 'forked'
    assert not repl._worker_configuration.values['create_session']
    repl._transcript.clear_history()
    repl._transcript.extend([first])
    repl.dispatch(_actions.WorkerStarted(), _actions.ReplayFinished(True))

  monkeypatch.setattr(repl, '_start_worker', start_child)
  # A second Enter accepts, since Enter is what opened the confirmation.
  scheduled: list[Coroutine[Any, Any, None]] = []
  monkeypatch.setattr(repl._app, 'create_background_task', scheduled.append)
  _submit(repl)
  await scheduled[0]

  assert len(sent) == 1
  assert sent[0].session_id == 'forked'
  assert sent[0].message.chat.user.content == 'try something else'
  assert not sent[0].message.chat.HasField('rewind_to')
  assert repl._input.text == ''
  assert [record.id for record in await parent.messages()] == [
    first.id,
    second.id,
  ]
  assert repl._transcript.transcript_index(second.id) is None
  assert repl._transcript.transcript_index(first.id) is not None


@pytest.mark.asyncio
async def test_rewind_child_start_failure_keeps_child_identity_and_draft(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  first = _user_record('first')
  repl = _rewindable_repl(tmp_path, first)
  _capture_paints(repl, monkeypatch)

  async def fail_start(*, replay: bool) -> None:
    assert replay
    raise RuntimeError('child unavailable')

  observed: list[_actions.Action] = []
  dispatch = repl.dispatch

  def record(*actions: _actions.Action) -> tuple[asyncio.Task[None], ...]:
    observed.extend(actions)
    return dispatch(*actions)

  monkeypatch.setattr(repl, 'dispatch', record)
  monkeypatch.setattr(repl, '_start_worker', fail_start)
  await repl._commit_rewind(first.id, 'revised draft')
  assert _actions.SessionCreated('forked') in observed
  assert not any(isinstance(a, _actions.SessionAttached) for a in observed)

  assert repl._state.session.conversation_id == 'forked'
  assert repl._worker_configuration.values['conversation_id'] == 'forked'
  assert repl._input.text == 'revised draft'
  assert repl._state.lifecycle.failure is not None
  assert 'child unavailable' in repl._state.lifecycle.failure
  assert not repl._state.turn.submitted_inputs


@pytest.mark.asyncio
async def test_refused_rewind_keeps_parent_and_restores_draft(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  first = _user_record('first')
  repl = _rewindable_repl(tmp_path, first)
  _seed_transcript(repl, first)
  _capture_paints(repl, monkeypatch)

  async def refuse(entry_id: str) -> str:
    assert entry_id == first.id
    raise RuntimeError('session busy; retry')

  assert repl._client is not None
  monkeypatch.setattr(repl._client, 'fork_session', refuse)
  await repl._commit_rewind(first.id, 'revised draft')

  assert repl._state.session.conversation_id == 'current'
  assert repl._input.text == 'revised draft'
  assert repl._transcript.transcript_index(first.id) is not None
  assert not repl._state.turn.submitted_inputs
  assert not repl._state.activity.busy


@pytest.mark.asyncio
async def test_remote_rewind_forks_then_sends_only_to_child(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  remote = _RemoteEcho()
  mirror = _RemoteMirror(['parent'])
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(remote, server)
  mirror_pb2_grpc.add_MirrorRPCServicer_to_server(mirror, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  repl = _repl(
    conversation_id='parent',
    database=tmp_path / 'state.db',
    remote_endpoint=f'127.0.0.1:{port}',
    remote_secure=False,
  )
  _capture_paints(repl, monkeypatch)
  try:
    await repl._commit_rewind('remote-history', 'revised draft')
    inference = await asyncio.wait_for(remote.requests.get(), timeout=2)
    approval = await asyncio.wait_for(remote.requests.get(), timeout=2)
    message = await asyncio.wait_for(remote.requests.get(), timeout=2)
    assert (
      inference.session_id
      == approval.session_id
      == message.session_id
      == 'forked'
    )
    assert message.message.chat.user.content == 'revised draft'
    assert not message.message.chat.HasField('rewind_to')
    assert mirror.forks == [
      mirror_pb2.ForkSessionRequest(
        source_session_id='parent', at_entry_id='remote-history'
      )
    ]
    assert mirror.session_ids == ['parent', 'forked']
    assert repl._state.session.conversation_id == 'forked'
  finally:
    await repl.close()
    await server.stop(0)


@pytest.mark.asyncio
async def test_a_committed_rewind_cannot_be_cancelled_or_sent_again(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Committing hands the rewind to the worker, so the keys that used to reach
  # it must no longer claim to cancel it or start a second commit.
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  _seed_transcript(repl, first, second)
  sent: list[agent_pb2.ClientMessage] = []
  scheduled: list[Coroutine[Any, Any, None]] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  repl._input.text = 'try something else'
  _submit(repl)

  async def start_child(*, replay: bool) -> None:
    assert replay
    repl.dispatch(_actions.WorkerStarted(), _actions.ReplayFinished(True))

  monkeypatch.setattr(repl, '_start_worker', start_child)
  monkeypatch.setattr(repl._app, 'create_background_task', scheduled.append)
  _submit(repl)
  await scheduled[0]
  sent.clear()

  _submit(repl)
  repl.dispatch(_actions.CancelRewind())
  await _settle(repl)

  assert repl._state.rewind.staged is None
  assert len(scheduled) == 1
  assert sent == []  # No second commit, and nothing re-rendered as cancelled.


@pytest.mark.asyncio
async def test_rewind_escape_restores_the_full_history(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  _seed_transcript(repl, first, second)
  sent: list[agent_pb2.ClientMessage] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  paints = _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  assert 'second' not in paints[-1][0]

  repl.dispatch(_actions.CloseSelector())
  await _settle(repl)

  assert repl._state.rewind.staged is None
  # Returning to the latest turn is local work too: nothing asks the worker.
  assert sent == []
  assert 'second' in paints[-1][0]
  assert not repl._input.read_only()
  # Returning to the latest turn takes the rewound turn's text back out.
  assert repl._input.text == ''
  assert _render_rewind_banner(repl) == []


@pytest.mark.asyncio
async def test_rewind_survives_the_escape_that_closes_the_skill_menu(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  available = (
    skills.Skill(
      name='helper', description='', body='', path=tmp_path / 'SKILL.md'
    ),
  )
  monkeypatch.setattr(skills, 'load_skills', lambda _: available)
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  _seed_transcript(repl, first, second)

  async def send(request: agent_pb2.ClientMessage) -> None:
    del request

  _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  repl._input.text = '$hel'

  # The open menu owns Esc, so dismissing it must not unstage the rewind.
  assert repl._skill_completion.visible(
    repl._state.composer.text,
    repl._state.composer.cursor,
  )
  assert not repl._rewind_escape_active()
  repl.dispatch(_actions.CloseSelector())
  assert not repl._skill_completion.visible(
    repl._state.composer.text, repl._state.composer.cursor
  )
  assert repl._state.rewind.staged is not None


@pytest.mark.asyncio
async def test_rewind_survives_the_escape_that_closes_the_selector(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', '/nonexistent/models.json')
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  _seed_transcript(repl, first, second)

  def send(request: agent_pb2.ClientMessage) -> None:
    del request

  _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  repl.dispatch(
    _actions.OpenSelector(
      _selection.Selector(
        'choice',
        'Choose',
        (_selection.Choice('value', 'Value'),),
        lambda _: _actions.ClearSelector(),
        extension_id='test/selector',
      )
    )
  )

  # The picker owns Esc, so closing it must not unstage the rewind.
  assert repl._state.selector.active
  assert not repl._rewind_escape_active()
  repl.dispatch(_actions.CloseSelector())
  assert not repl._state.selector.active
  assert repl._state.rewind.staged is not None


@pytest.mark.asyncio
async def test_cancelling_a_rewind_keeps_an_edited_message(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  first = _user_record('first')
  second = _user_record('second')
  repl = _rewindable_repl(tmp_path, first, second)
  _seed_transcript(repl, first, second)

  def send(request: agent_pb2.ClientMessage) -> None:
    del request

  _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  await _dispatch_and_wait(repl, _actions.RewindSession())
  _submit(repl)
  await _settle(repl)
  repl._input.text = 'second, but rewritten'

  repl.dispatch(_actions.CancelRewind())
  await _settle(repl)

  # Only the untouched prefill is withdrawn; a real draft survives.
  assert repl._input.text == 'second, but rewritten'


@pytest.mark.asyncio
async def test_rewind_reports_when_there_is_nothing_to_rewind_to(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  repl = _rewindable_repl(tmp_path)
  emitted: list[_console.RenderableType] = []

  async def emit(renderables: list[_console.RenderableType]) -> None:
    emitted.extend(renderables)

  monkeypatch.setattr(repl._terminal, 'emit', emit)
  await _dispatch_and_wait(repl, _actions.RewindSession())

  assert not repl._state.selector.active
  assert not emitted
  assert 'no user turns' in _render_welcome_notices(repl)


@pytest.mark.asyncio
async def test_resume_picker_enter_dismisses_before_the_switch(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # prompt_toolkit can deliver two Enter presses in one pass; the picker must
  # close on the first so the second cannot schedule another switch.
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'saved',
      entry_id='saved-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved work',
      cwd=str(tmp_path),
      model='m-test',
    )

  repl = _repl(conversation_id='current', database=database)
  restarts = _record_restarts(repl, monkeypatch)
  scheduled: list[Coroutine[Any, Any, None]] = []

  def schedule(coroutine: Coroutine[Any, Any, None]) -> None:
    scheduled.append(coroutine)

  await _dispatch_and_wait(repl, _actions.ResumeSession())
  chosen = repl._state.selector.filtered_choices()[0]
  monkeypatch.setattr(repl._app, 'create_background_task', schedule)

  _submit(repl)
  assert not repl._state.selector.filtered_choices()
  _submit(repl)

  assert len(scheduled) == 1
  await scheduled[0]
  assert len(scheduled) == 2
  await scheduled[1]
  assert restarts == [
    (
      {
        'conversation_id': chosen.value,
        'create_session': False,
        'allow_workspace_mismatch': False,
      },
      True,
      chosen.value,
    )
  ]


@pytest.mark.asyncio
async def test_opening_resume_picker_closes_slash_menu(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'current',
      entry_id='current-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='current',
      cwd=str(tmp_path),
      model='m-test',
    )
    store.record_user_message(
      'saved',
      entry_id='saved-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved',
      cwd=str(tmp_path),
      model='m-test',
    )

  repl = _repl(conversation_id='current', database=database)
  repl._input.text = '/'
  resume_index = next(
    index
    for index, command in enumerate(_commands.COMMANDS)
    if command.name == 'resume'
  )
  repl.dispatch(
    _actions.MoveCompletionRequested(
      resume_index,
      skill_count=0,
      command_count=len(repl.get_available_commands()),
    )
  )
  assert _completion.visible(repl._state)

  _submit(repl)
  await _settle(repl)

  assert repl._state.selector.active
  assert repl._state.composer == _actions.ComposerState()
  assert not _completion.visible(repl._state)
  repl.dispatch(_actions.ClearSelector())
  assert not repl._state.resume.active
  assert not _completion.visible(repl._state)


@pytest.mark.asyncio
async def test_resume_picker_escape_cancels_without_switching(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'saved',
      entry_id='saved-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved work',
      cwd=str(tmp_path),
      model='m-test',
    )

  repl = _repl(conversation_id='current', database=database)
  await _dispatch_and_wait(repl, _actions.ResumeSession())
  repl.dispatch(_actions.ClearSelector())

  assert not repl._state.selector.active
  assert not repl._state.selector.filtered_choices()
  assert not repl._input.read_only()


@pytest.mark.asyncio
async def test_bare_resume_keeps_picker_open_when_scope_is_empty(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  repl = _repl(database=tmp_path / 'state.db')
  await _dispatch_and_wait(repl, _actions.ResumeSession())

  assert repl._state.selector.active
  assert not repl._state.selector.filtered_choices()
  assert _resume_height(repl) == 5
  rendered = ''.join(text for _, text in _render_resume(repl))
  assert 'No other conversations in this scope' in rendered
  assert '[cwd]' in rendered
  assert 'Tab scope' in rendered


@pytest.mark.asyncio
async def test_resume_tab_cycles_scope_and_confirm_allows_foreign_cwd(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  other = tmp_path / 'other'
  other.mkdir()
  with cli_store.Store(database) as store:
    store.record_user_message(
      'foreign',
      entry_id='foreign-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved elsewhere',
      cwd=str(other),
      model='m-test',
    )

  repl = _repl(conversation_id='current', database=database)
  restarts = _record_restarts(repl, monkeypatch)
  await _dispatch_and_wait(repl, _actions.ResumeSession())
  assert not repl._state.selector.filtered_choices()

  repl.dispatch(
    _actions.SetResumeScope(
      workspace.next_session_scope(repl._state.resume.scope)
    )
  )
  assert repl._state.resume.scope.value == 'git'
  repl.dispatch(
    _actions.SetResumeScope(
      workspace.next_session_scope(repl._state.resume.scope)
    )
  )
  assert repl._state.resume.scope.value == 'all'
  await _settle(repl)
  assert [
    choice.value for choice in repl._state.selector.filtered_choices()
  ] == ['foreign']
  rendered = ''.join(text for _, text in _render_resume(repl))
  assert '[all]' in rendered.splitlines()[1]
  assert 'Tab scope' in rendered.splitlines()[2]

  _submit(repl)
  await _settle(repl)
  assert repl._state.resume.confirmation is not None
  assert not restarts

  confirm_text = ''.join(text for _, text in _render_resume(repl))
  assert 'Resume it from' in confirm_text
  assert 'y resume · n/Esc cancel' in confirm_text

  repl.dispatch(_actions.AcceptResumeConfirmation())
  await _settle(repl)
  assert restarts == [
    (
      {
        'conversation_id': 'foreign',
        'create_session': False,
        'allow_workspace_mismatch': True,
      },
      True,
      'foreign',
    )
  ]


@pytest.mark.asyncio
async def test_resume_picker_reads_sessions_off_the_event_loop(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Git scope spawns a `git` probe per saved directory, so the terminal must not
  # wait on the read.
  monkeypatch.chdir(tmp_path)
  database = tmp_path / 'state.db'
  repl = _repl(conversation_id='current', database=database)
  threads: list[int] = []

  def infos(
    scope: workspace.SessionScope | None = None,
  ) -> list[cli_store.ConversationInfo]:
    del scope
    threads.append(threading.get_ident())
    return []

  monkeypatch.setattr(repl, '_session_infos', infos)
  await _dispatch_and_wait(repl, _actions.ResumeSession())

  assert threads and threading.get_ident() not in threads


@pytest.mark.asyncio
async def test_session_switch_replaces_the_visible_screen(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Empty and used conversations both keep saved scrollback; `/clear` is the
  # flush. The visible screen is replaced either way.
  empty = _repl()
  _record_restarts(empty, monkeypatch)
  paints = _capture_paints(empty, monkeypatch)
  await empty._switch_session('next', create=False)
  empty._handle_message(_delta({'role': 'user', 'content': 'restored'}))
  empty._handle_message(_replay_done('next'))
  await _settle(empty)
  assert paints[-1][1] is _terminal_mod.TerminalClear.SCREEN

  started = _repl()
  _record_restarts(started, monkeypatch)
  paints = _capture_paints(started, monkeypatch)
  started._handle_message(_delta({'role': 'user', 'content': 'saved work'}))
  started._handle_message(_replay_done(has_user_message=True))
  await _settle(started)
  paints.clear()
  await started._switch_session('next', create=False)
  started._handle_message(_delta({'role': 'user', 'content': 'next'}))
  started._handle_message(_replay_done('next', has_user_message=True))
  await _settle(started)
  assert paints[-1][1] is _terminal_mod.TerminalClear.SCREEN


@pytest.mark.asyncio
async def test_slash_help_stays_local() -> None:
  """Slash output is emitted in-process; the worker must not see stdin."""
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    del pipe
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )

    async def once() -> None:
      await repl._run_slash('/help')
      repl._app.exit()

    repl._app.pre_run_callables.append(
      lambda: repl._app.create_background_task(once())
    )
    await asyncio.wait_for(_run_repl(repl), timeout=10)
    output = _render_welcome_notices(repl)
  assert 'Available commands' in output
  assert 'echo:' not in output


@pytest.mark.asyncio
async def test_slash_quit_exits_the_app() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl()
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    pipe.send_text('/quit\r')
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  assert repl._client is None


@pytest.mark.asyncio
async def test_startup_crash_keeps_live_banner_available() -> None:
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    # The worker dies before `replay_done`, without creating conversation content.
    repl = _repl(worker_source='raise SystemExit(1)')
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    await asyncio.wait_for(_run_repl(repl), timeout=10)
  output = repl._terminal.console.export_text()
  assert 'model ' in _render_banner(repl)
  assert not repl._transcript.has_conversation
  assert 'could not start agent worker' in output


@pytest.mark.asyncio
async def test_replay_history_reaches_the_terminal_in_one_write(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  repl._handle_message(_delta({'role': 'user', 'content': 'first'}))
  repl._handle_message(_delta({'role': 'assistant', 'content': 'second'}))
  await asyncio.sleep(0)
  assert not paints

  repl._handle_message(_replay_done('conv-1'))
  await _settle(repl)

  assert len(paints) == 1
  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'first' in ansi
  assert 'second' in ansi


@pytest.mark.asyncio
async def test_replay_never_executes_controls_captured_in_tool_output(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  repl._handle_message(_delta(_bash_call()))
  repl._handle_message(
    _delta({
      'role': 'tool',
      'tool_call_id': 'c1',
      'content': (
        'before\x1b[?1006;1000h\x1b[?1002h\x1b[?1049h\x1b[>4;2m\x1b[?2004hafter'
      ),
    })
  )

  repl._handle_message(_replay_done('conv-1'))
  await _settle(repl)

  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'beforeafter' in ansi
  assert '\x1b[?1006;1000h' not in ansi
  assert '\x1b[?1002h' not in ansi
  assert '\x1b[?1049h' not in ansi
  assert '\x1b[>4;2m' not in ansi
  assert '\x1b[?2004h' not in ansi


@pytest.mark.asyncio
async def test_history_replacement_suppresses_footer_until_clear_finishes(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  chrome_during_replace: list[
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
  ] = []

  async def run_in_terminal(callback: Any) -> None:
    chrome_during_replace.append((
      _render_activity(repl),
      _render_footer(repl),
    ))
    callback()
    chrome_during_replace.append((
      _render_activity(repl),
      _render_footer(repl),
    ))

  monkeypatch.setattr(cli, '_run_in_terminal', run_in_terminal)
  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SetStatus('loading conversation…'))
  repl.dispatch(
    _actions.SetPendingSwitch(_actions.PendingSwitch('target', create=False))
  )
  repl._terminal.pin_header(_text.Text('history'))

  await repl._repaint(replacing=True)

  assert chrome_during_replace == [([], []), ([], [])]
  assert _render_footer(repl)


@pytest.mark.asyncio
async def test_printing_mid_switch_keeps_the_screen_for_the_new_history(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # `/help` and friends stay available while a switch is in flight; their output
  # must not consume the one clear that swaps in the new conversation.
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  _record_restarts(repl, monkeypatch)
  await repl._switch_session('target', create=False)

  await repl._run_slash('/help')

  assert not any(
    clear is not _terminal_mod.TerminalClear.KEEP for _, clear in paints
  )
  assert repl._state.session.pending_switch is not None

  repl._handle_message(_delta({'role': 'user', 'content': 'restored'}))
  repl._handle_message(_replay_done('target'))
  await asyncio.sleep(0)

  screen = [
    ansi
    for ansi, clear in paints
    if clear is _terminal_mod.TerminalClear.SCREEN
  ]
  assert len(screen) == 1
  assert 'restored' in screen[0]
  assert repl._state.session.pending_switch is None


@pytest.mark.asyncio
async def test_replay_done_flushes_pending_and_only_done_ends_the_turn() -> (
  None
):
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  _handle(repl, _delta(_bash_call()))
  repl._handle_message(_replay_done())
  assert repl._state.activity.busy
  assert _running_call_hints(repl) == {}
  assert _running_call_ids(repl) == ()
  repl._handle_message(_turn_completed())
  assert not repl._state.activity.busy


def test_done_stays_busy_while_a_resubmitted_instruction_is_queued(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  effects: list[_effects.Effect] = []
  monkeypatch.setattr(repl, '_run_effect', effects.append)
  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.SubmitInstruction('follow-up'))
  started = repl._state.activity.work_started

  # An interrupted turn's `done` can arrive after a follow-up was submitted; it
  # runs as its own turn, so the app must stay busy and keep the queued line.
  repl._handle_message(_turn_completed())

  assert repl._state.activity.busy
  assert repl._state.turn.queued == ('follow-up',)
  assert started is not None
  assert repl._state.activity.work_started == started


def test_waiting_task_shows_footer_hint_until_detached() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())

  repl._handle_message(_task_delta('1234', 'running', detached=False))

  assert 'Ctrl+B to detach' in _fragments_text(_render_activity(repl))
  assert not _render_tasks(repl)

  repl._handle_message(_task_delta('1234', 'running'))

  assert 'Ctrl+B to detach' not in _fragments_text(_render_activity(repl))
  assert 'bash · 1234 · running' in _render_tasks(repl)


@pytest.mark.asyncio
async def test_tasks_leave_one_row_before_footer() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  empty_height = repl._bottom_pane.preferred_height(80, 24).preferred

  repl._handle_message(_task_delta('1234', 'running'))

  task_height = repl._bottom_pane.preferred_height(80, 24).preferred
  assert task_height == empty_height + 3  # Header, task, and footer gap.
  await asyncio.sleep(0)


def test_task_deltas_render_and_update_one_row_per_running_task() -> None:
  repl = _repl()
  repl._handle_message(_task_delta('1234', 'first update'))
  repl._handle_message(_task_delta('5678', 'second update'))

  text = _render_tasks(repl)
  assert 'Tasks\n' in text
  assert '├─ bash · 1234 · first update' in text
  assert '└─ bash · 5678 · second update' in text
  assert text.count('\n') == 2

  repl._handle_message(_task_delta('1234', 'finished', 'succeeded'))

  text = _render_tasks(repl)
  assert '1234' not in text
  assert '└─ bash · 5678 · second update' in text
  assert text.count('\n') == 1


@pytest.mark.asyncio
async def test_down_focuses_task_and_cancel_sends_worker_command() -> None:
  repl = _repl()
  repl._handle_message(_task_delta('1234', 'running'))
  assert repl._can_focus_tasks()
  repl._input.text = 'first\nlast'
  repl._input.cursor_position = 0
  assert not repl._can_focus_tasks()

  repl._input.cursor_position = len(repl._input.text)
  assert repl._can_focus_tasks()
  repl.dispatch(_actions.SelectFirstTask())

  assert repl._state.tasks.selected_id == '1234'
  text = _render_tasks(repl)
  assert 'press x to cancel' in text

  sent: list[agent_pb2.ClientMessage] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  repl._send_request = send  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
  repl.dispatch(_actions.CancelSelectedTask())
  await _settle(repl)

  assert len(sent) == 1
  assert sent[0].cancel_task.task_id == '1234'
  assert repl._state.tasks.selected_id is None
  assert repl._app.layout.current_window is repl._input_window


@pytest.mark.asyncio
async def test_detach_requested_sends_detach_tools_while_busy() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  sent: list[agent_pb2.ClientMessage] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  repl._send_request = send  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
  repl.dispatch(_actions.DetachRequested())
  await asyncio.sleep(0)

  assert len(sent) == 1
  assert sent[0].WhichOneof('payload') == 'detach_tools'


def test_selected_task_completion_returns_focus_to_prompt() -> None:
  repl = _repl()
  repl._handle_message(_task_delta('1234', 'running'))
  repl.dispatch(_actions.SelectFirstTask())

  repl._handle_message(_task_delta('1234', 'finished', 'succeeded'))

  assert repl._state.tasks.selected_id is None
  assert repl._app.layout.current_window is repl._input_window
  assert not _render_tasks(repl)


def test_task_deltas_ellipsize_labels_to_one_terminal_row() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=40))
  repl._handle_message(_task_delta('1234', 'first line\n' + 'x' * 80))

  text = _render_tasks(repl)
  rows = _screen_lines(text, cols=40)

  assert len(rows) == 2
  assert rows[0] == '  Tasks'
  assert rows[1].endswith('…')
  assert 'first line' in rows[1]


def test_input_nack_releases_busy_state_without_queued_work() -> None:
  repl = _repl()
  repl.dispatch(
    _actions.TurnStarted(),
    _actions.InputSubmitted('request', 'rejected input'),
  )

  repl._handle_message(
    agent_pb2.SessionMessage(
      input_nack=agent_pb2.InputNack(message_id='request', error='refused')
    )
  )

  assert not repl._state.activity.busy


@pytest.mark.asyncio
async def test_input_nack_restores_the_submitted_text_without_losing_new_edits(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.WorkerStarted())
  submitted = 'A long first line.\n\n' + ('important details ' * 100).rstrip()
  repl._input.text = submitted
  sent: list[agent_pb2.ClientMessage] = []
  scheduled: list[Coroutine[Any, Any, None]] = []
  monkeypatch.setattr(repl, '_send_request', sent.append)
  monkeypatch.setattr(repl._app, 'create_background_task', scheduled.append)

  _submit(repl)
  assert repl._input.text == ''
  assert len(sent) == 1
  repl._input.text = 'draft typed while waiting'
  repl._handle_message(
    agent_pb2.SessionMessage(
      input_nack=agent_pb2.InputNack(
        message_id=sent[0].message_id,
        error='refused',
      )
    )
  )

  assert repl._input.text == f'{submitted}\ndraft typed while waiting'
  assert repl._input.cursor_position == len(repl._input.text)
  assert not repl._state.activity.busy


def test_handle_delta_renders_task_results() -> None:
  delta = _task_result('1234', 'bash', 'complete')

  rendered = _handle(_repl(), delta)

  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'bash(task 1234)' in text
  assert 'complete' in text


def test_agent_task_result_renders_full_markdown_answer() -> None:
  content = '# Survey\n\nFirst section.\n\nThe **full** `answer`.'
  delta = _task_result('88766', 'subagent', content)
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))

  rendered = _handle(repl, delta)

  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'Subagent · task 88766' in text
  assert 'Survey' in text
  assert 'First section.' in text
  assert 'The full answer.' in text
  assert '**' not in text


def test_compact_subagent_result_shows_only_completion_status() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  rendered = _handle(
    repl,
    _task_result('review-task', 'subagent', '## Review\n\nDetailed findings.'),
  )

  assert rendered is not None
  text = _render_to_text(rendered)
  assert [line.rstrip() for line in text.splitlines()] == [
    '· Subagent',
    '  └ Complete',
  ]


def test_compact_mode_preserves_background_command_output() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  delta = _task_result('1234', 'bash', 'line one\nline two\nline three')

  rendered = _handle(repl, delta)

  assert rendered is not None
  text = _render_to_text(rendered)
  assert '· Ran background task 1234' in text
  assert '└ line one … (+2 lines)' in text
  assert 'line two' not in text
  assert 'line three' not in text


def test_a_non_function_tool_call_is_left_out() -> None:
  repl = _repl()
  message = {
    'role': 'assistant',
    'content': None,
    'tool_calls': [{'id': 'c1', 'type': 'custom'}],
  }
  assert _handle(repl, _delta(message)) is None
  assert _running_call_hints(repl) == {}
  assert _running_call_ids(repl) == ()


def _fragments_text(fragments: list[tuple[str, str]]) -> str:
  return ''.join(text for _style, text in fragments)


class _NoEmitRepl(cli._Repl):
  """A REPL whose terminal emit is a no-op, for async tests without a running app."""

  def __init__(self) -> None:
    super().__init__(
      [sys.executable, '-c', 'pass'],
      {'conversation_id': 'conv-1', 'model': 'm-test'},
      session_id='test-session',
      extension_registry=extension.ExtensionRegistry(),
      workspace_effects=asyncio.Queue(),
      workspace_controller=mock.Mock(
        spec=_workspace_controller.WorkspaceController
      ),
    )

    async def emit(
      renderables: list[_console.RenderableType],
    ) -> None:
      del renderables  # Avoid scrollback writes in a non-running app.

    self._terminal.emit = emit


@pytest.mark.asyncio
async def test_approval_prompt_round_trip() -> None:
  repl = _repl()  # Default mode is auto.
  sent: list[agent_pb2.ClientMessage] = []

  def capture(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  repl._send_request = capture  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
  assert '🛠  auto' in _fragments_text(_render_footer(repl))
  repl.dispatch(_actions.ToggleApprovalMode())
  await _settle(repl)
  assert '🛠  manual' in _fragments_text(_render_footer(repl))
  assert sent[0].WhichOneof('payload') == 'set_approval_mode'
  sent.clear()

  repl.dispatch(_actions.TurnStarted())
  assert _render_activity(repl)
  repl._handle_message(_approval_requested('gc-1', 'bash', '{"command":"ls"}'))
  assert not _render_activity(repl)
  assert _component_height(_activity.Activity().render(repl._state)) == 0
  assert repl._state.activity.busy
  selector = repl._state.selector.selector
  assert selector is not None
  assert [choice.label for choice in selector.choices] == [
    'Approve',
    'Approve and always allow commands starting with `ls`',
    'Reject',
  ]
  assert repl._state.approval.request is not None
  prompt = '\n'.join(_render_approval(repl))
  assert 'bash(ls)' in prompt
  assert '› 1. Approve' in prompt  # Approve selected by default.

  repl.dispatch(_actions.MoveSelection(2))  # Move down to "Reject".
  repl.dispatch(_actions.AcceptSelection())
  await _settle(repl)
  assert len(sent) == 1
  assert sent[0].hitl_response.call_id == 'gc-1'
  assert not sent[0].hitl_response.approved
  assert not repl._state.approval.request is not None
  assert _render_activity(repl)
  assert repl._state.activity.work_paused is None

  # A worker-side resolution (mode switch, interrupt) also clears the prompt.
  repl._handle_message(_approval_requested('gc-2', 'bash'))
  repl._handle_message(
    agent_pb2.SessionMessage(
      hitl_resolved=agent_pb2.HITLResolved(call_id='gc-2', approved=True)
    )
  )
  assert not repl._state.approval.request is not None


def test_approval_prompt_wraps_the_exact_bash_command() -> None:
  repl = _repl()
  repl.dispatch(_actions.TerminalResized(40))
  command = (
    'cd /workspace/project && bazel test //project/toolbox/mirror:all\n'
    '\tgit push origin HEAD  '
  )
  repl._handle_message(
    _approval_requested('gc-1', 'bash', json.dumps({'command': command}))
  )
  lines = _render_approval(repl)
  assert lines[1:4] == [
    '· bash(cd /workspace/project && bazel',
    '  test //project/toolbox/mirror:all',
    '          git push origin HEAD)',
  ]
  assert '› 1. Approve' in '\n'.join(lines[3:])
  assert '3. Reject' in '\n'.join(lines[3:])


@pytest.mark.asyncio
async def test_permissions_picker_changes_worker_mode() -> None:
  repl = _repl()
  sent: list[agent_pb2.ClientMessage] = []

  def capture(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  repl._send_request = capture  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
  repl.dispatch(_actions.SelectPermissions())

  assert repl._state.selector.active
  assert 'Automatic approval  (current)' in _fragments_text(
    _component_text(_selector.Selector().render(repl._state))
  )

  repl.dispatch(
    _actions.MoveSelectorSelection(1),
  )
  repl.dispatch(*repl._selector_choice())
  await _settle(repl)

  assert repl.approval_mode == 'approve'
  assert repl._worker_configuration.values['approval_mode'] == 'approve'
  assert len(sent) == 1
  assert sent[0].set_approval_mode.mode == agent_pb2.APPROVAL_MODE_HITL


def test_pending_pane_hides_call_awaiting_approval() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  # Park a tool call (as an assistant tool-call delta would), then gate it.
  _start_calls(
    repl,
    [
      {
        'id': 'gc-1',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{}'},
      }
    ],
  )
  assert _pending.rows(repl._state) > 0

  repl._handle_message(_approval_requested('gc-1', 'bash'))

  # The gated call is shown in the approval prompt, not the running pane.
  assert _pending.rows(repl._state) == 0
  rendered = _pending.render(repl._state)
  assert _screen_lines(_ansi_text(rendered)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('query', ['/', '$'])
async def test_ctrl_j_k_navigate_completion_without_editing(
  query: str,
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  available = tuple(
    skills.Skill(
      name=name, description='', body='', path=tmp_path / name / 'SKILL.md'
    )
    for name in ('alpha', 'beta')
  )
  monkeypatch.setattr(skills, 'load_skills', lambda _: available)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(database=tmp_path / 'state.db')
    running = asyncio.create_task(repl._app.run_async())
    try:
      await asyncio.sleep(0.05)
      pipe.send_text(query)
      await asyncio.sleep(0.05)
      initial = (
        _completion.selected_skill(repl._state)
        if query == '$'
        else _completion.selected_slash_command(repl._state)
      )
      assert initial is not None
      pipe.send_text('\x0a')
      await asyncio.sleep(0.05)
      selected = (
        _completion.selected_skill(repl._state)
        if query == '$'
        else _completion.selected_slash_command(repl._state)
      )
      assert selected is not None and selected != initial
      assert repl._input.text == query
      pipe.send_text('\x0b')
      await asyncio.sleep(0.05)
      selected = (
        _completion.selected_skill(repl._state)
        if query == '$'
        else _completion.selected_slash_command(repl._state)
      )
      assert selected == initial
      assert repl._input.text == query
      assert not repl._state.turn.submitted_inputs
      repl._input.text = 'ordinary text'
      repl._input.cursor_position = 8
      pipe.send_text('\x0b')
      await asyncio.sleep(0.05)
      assert repl._input.text == 'ordinary'
      pipe.send_text('\x0a')
      await asyncio.sleep(0.05)
      assert repl._input.text == 'ordinary\n'
    finally:
      repl._app.exit()
      await asyncio.wait_for(running, timeout=2)


class _BindingExtension(extension.Extension[None, None, None]):
  @classmethod
  @override
  def id(cls) -> str:
    return 'test/bindings'

  @override
  def keybindings(self) -> Sequence[_keymap.Binding]:
    return (_keymap.Binding(('c-c',), _actions.CycleViewMode()),)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('config', 'expected'),
  [
    ([], _actions.ViewMode.COMPACT),
    (
      [{'keys': [['c-c']], 'action': {'type': 'RerenderRequested'}}],
      _actions.ViewMode.TRANSCRIPT,
    ),
  ],
)
async def test_layered_actions_can_override_the_builtin_quit_shortcut(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  config: list[dict[str, object]],
  expected: _actions.ViewMode,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  path = tmp_path / '.mirror' / 'keybindings.json'
  path.parent.mkdir()
  path.write_text(json.dumps(config))
  registry = extension.ExtensionRegistry()
  await registry.register(_BindingExtension(), None)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)
    running = asyncio.create_task(repl._app.run_async())
    try:
      await asyncio.sleep(0.05)
      pipe.send_text('\x03')
      await asyncio.sleep(0.1)
      assert not running.done()
      assert repl._state.status_bar.view_mode == expected
    finally:
      repl._app.exit()
      await running
      await repl._extension_host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_config_rebinds_editor_and_preserves_selector_escape(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  monkeypatch.setenv('HOME', str(tmp_path))
  path = tmp_path / '.mirror' / 'keybindings.json'
  path.parent.mkdir()
  path.write_text(
    json.dumps([
      {
        'keys': [['f6'], ['c-x', 'e']],
        'action': {
          'type': 'RoutedExtensionAction',
          'extension_id': 'core/editor',
          'action': {
            'type': 'mirror.cli.extensions.editor.extension.OpenEditor'
          },
        },
      },
      {'keys': [], 'action': {'type': 'InterruptRequested'}},
      {'keys': [['c-g']], 'action': {'type': 'CloseSelector'}},
    ])
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(
    registry,
    tmp_path / '.mirror' / 'extensions',
    bundled=pathlib.Path(extension.__file__).with_name('extensions'),
  )
  await manager.start()
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)
    await repl._extension_host.start()
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))

    async def send_keys(keys: str, *, selector_active: bool) -> None:
      processed = asyncio.Event()

      def after_key_press(_: object) -> None:
        processed.set()

      repl._app.key_processor.after_key_press += after_key_press
      try:
        async with asyncio.timeout(3):
          pipe.send_text(keys)
          await processed.wait()
          while repl._state.selector.active != selector_active:
            await asyncio.sleep(0)
      finally:
        repl._app.key_processor.after_key_press -= after_key_press

    try:
      await asyncio.wait_for(ready.wait(), timeout=3)
      repl._input.text = 'draft'
      await send_keys('\x10', selector_active=False)
      assert not repl._state.selector.active
      assert repl._input.text == 'draft'
      await send_keys('\x1b[17~', selector_active=True)
      assert repl._state.selector.active
      await send_keys('\x1b', selector_active=False)
      assert not repl._state.selector.active
      await send_keys('\x1b[17~', selector_active=True)
      assert repl._state.selector.active
      await send_keys('\x07', selector_active=False)
      assert not repl._state.selector.active
      assert repl._input.text == 'draft'
      await send_keys('\x18', selector_active=False)
      assert not repl._state.selector.active
      await send_keys('e', selector_active=True)
      assert repl._state.selector.active
      assert repl._input.text == 'draft'
    finally:
      repl._app.exit()
      await running
      await repl._extension_host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_editor_shortcut_hot_loads_and_preserves_composer_draft(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  registry = extension.ExtensionRegistry()
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)
    await repl._extension_host.start()
    running = asyncio.create_task(repl._app.run_async())
    try:
      repl._input.text = 'unfinished draft'
      repl._input.cursor_position = 4
      await registry.register(_editor_extension.EditorExtension(), None)
      await asyncio.sleep(0.05)
      pipe.send_text('\x10')
      await asyncio.sleep(0.1)
      request = repl._state.selector.selector
      assert request is not None
      assert request.extension_id == 'core/editor'
      assert repl._input.text == 'unfinished draft'
      assert repl._input.cursor_position == 4
      pipe.send_text('\x1b')
      await asyncio.sleep(0.1)
      assert not repl._state.selector.active
      assert repl._input.text == 'unfinished draft'
      assert repl._input.cursor_position == 4
      await registry.unload('core/editor')
      pipe.send_text('\x10')
      await asyncio.sleep(0.05)
      assert not repl._state.selector.active
      assert not any(
        binding.filter()
        for binding in repl._app.key_bindings.get_bindings_for_keys(('c-p',))
      )
    finally:
      repl._app.exit()
      await running
      await repl._extension_host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_editor_shortcut_does_not_replace_an_active_selector(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  registry = extension.ExtensionRegistry()
  await registry.register(_editor_extension.EditorExtension(), None)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)
    running = asyncio.create_task(repl._app.run_async())
    try:
      choices = (
        _selection.Choice('first', 'First'),
        _selection.Choice('second', 'Second'),
      )
      repl.dispatch(
        _actions.OpenSelector(
          _selection.Selector(
            'other',
            'Other picker',
            choices,
            _actions.SetModel,
            selected='second',
          )
        )
      )
      await asyncio.sleep(0.05)
      pipe.send_text('\x10')
      await asyncio.sleep(0.05)
      assert repl._state.selector.id == 'other'
      assert repl._state.selector.selected_choice() == choices[1]
      pipe.send_text('\x1b[A')
      await asyncio.sleep(0.05)
      assert repl._state.selector.selected_choice() == choices[0]
    finally:
      repl._app.exit()
      await running
      await repl._extension_host.close()
      await registry.close()


@pytest.mark.asyncio
async def test_editor_shortcut_respects_busy_command_availability(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  registry = extension.ExtensionRegistry()
  await registry.register(_editor_extension.EditorExtension(), None)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)
    running = asyncio.create_task(repl._app.run_async())
    try:
      repl._input.text = 'draft'
      repl.dispatch(_actions.TurnStarted())
      await asyncio.sleep(0.05)
      pipe.send_text('\x10')
      await asyncio.sleep(0.05)
      assert not repl._state.selector.active
      assert repl._input.text == 'draft'
    finally:
      repl._app.exit()
      await running
      await repl._extension_host.close()
      await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('line', ['synthetic text', '中 e\u0301 👩\u200d💻'])
async def test_typing_frames_preserve_large_multiline_paste_and_cursor_edits(
  line: str,
) -> None:
  terminal = io.StringIO()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=24, columns=80),
    term='xterm-256color',
    enable_cpr=False,
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl()
    prefix = (line + '\n') * 512
    expected = prefix + 'last'
    painted = asyncio.Event()

    def after_render(_: _application.Application[str | None]) -> None:
      if repl._input.text == expected:
        painted.set()

    repl._app.after_render += after_render
    running = asyncio.create_task(repl._app.run_async())
    try:
      # Large pastes can fill the OS pipe; let the event loop drain it.
      await asyncio.to_thread(
        pipe.send_text, '\x1b[200~' + expected + '\x1b[201~'
      )
      await asyncio.wait_for(painted.wait(), timeout=10)
      assert repl._state.composer.text == expected
      assert 'last' in '\n'.join(
        _screen_lines(terminal.getvalue(), cols=80, rows=24)
      )
      painted.clear()
      expected = prefix + 'laXst'
      pipe.send_text('\x1b[D\x1b[DX')
      await asyncio.wait_for(painted.wait(), timeout=10)
      assert repl._state.composer.text == expected
      assert repl._input.cursor_position == len(prefix) + 3
      assert 'laXst' in '\n'.join(
        _screen_lines(terminal.getvalue(), cols=80, rows=24)
      )
    finally:
      repl._app.exit()
      await running
      await repl.close()


def test_all_key_bindings_resolve_to_callable_handlers() -> None:
  # A binding referencing a renamed/removed method only fails when the key is
  # pressed; assert every handler resolves so that regresses in CI instead.
  repl = _repl()
  bindings = repl._app.key_bindings.bindings
  assert bindings and all(callable(b.handler) for b in bindings)


@pytest.mark.asyncio
async def test_up_arrow_recalls_persistent_history(
  tmp_path: pathlib.Path,
) -> None:
  """Up arrow loads a past user message from SQLite into the prompt buffer."""
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'conv-1',
      entry_id='current-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='old persistent prompt',
      cwd=str(tmp_path),
      model='m',
    )
    # A newer prompt from a different conversation.
    store.record_user_message(
      'other',
      entry_id='other-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='other prompt',
      cwd=str(tmp_path),
      model='m',
    )

  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(database=database)
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )

    async def interact() -> None:
      await asyncio.sleep(0.1)
      pipe.send_text('\x1b[A')  # Up arrow.
      await asyncio.sleep(0.1)
      # Current session's prompt comes first, not the other session's.
      assert repl._input.text == 'old persistent prompt'
      repl._app.exit()

    repl._app.pre_run_callables.append(
      lambda: repl._app.create_background_task(interact())
    )
    await asyncio.wait_for(_run_repl(repl), timeout=10)


def test_session_opened_reloads_history_without_dropping_a_draft() -> None:
  repl = _repl()
  repl._input.text = 'half-written prompt'
  repl._input.cursor_position = 4

  repl._on_session_opened('conv-2')

  assert repl._input.text == 'half-written prompt'
  assert repl._input.cursor_position == 4
  # The next recall reads the newly opened conversation.
  assert repl._history._conversation_id == 'conv-2'


def _input_styles(repl: cli._Repl) -> set[str]:
  """The style strings the input line paints, per fragment."""
  control = next(
    candidate
    for candidate in repl._app.layout.find_all_controls()
    if isinstance(candidate, _controls.BufferControl)
    and candidate.buffer is repl._input
  )
  content = control.create_content(40, 10)
  return {
    style
    for style, *_ in content.get_line(0)  # pyright: ignore[reportUnknownVariableType]
  }


@pytest.mark.asyncio
async def test_ctrl_r_search_finds_past_prompt(
  tmp_path: pathlib.Path,
) -> None:
  """Ctrl+R fuzzy recall preserves cancelled drafts and never submits."""
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'past',
      entry_id='first-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='fix the bug',
      cwd=str(tmp_path),
      model='m',
    )
    store.record_user_message(
      'past',
      entry_id='second-entry',
      sequence=1,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='add a feature',
      cwd=str(tmp_path),
      model='m',
    )

  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(database=database)
    repl._terminal.console = _console.Console(
      record=True, width=80, theme=themes.rich_theme()
    )
    searching_styles: set[str] = set()
    accepted_styles: set[str] = set()
    later_styles: set[str] = set()

    ready = asyncio.Event()
    repl._app.pre_run_callables.append(ready.set)

    async def send_keys(keys: str, predicate: Callable[[], bool]) -> None:
      rendered = asyncio.Event()

      def after_render(_app: object) -> None:
        if predicate():
          rendered.set()

      repl._app.after_render += after_render
      try:
        pipe.send_text(keys)
        await rendered.wait()
      finally:
        repl._app.after_render -= after_render

    async def interact() -> None:
      await ready.wait()
      await send_keys(
        '\x12', lambda: len(repl._state.selector.filtered_choices()) == 2
      )
      assert repl._state.selector.active
      await send_keys('\x0a', lambda: repl._state.selector.menu.selected == 1)
      assert repl._state.selector.menu.selected == 1
      assert repl._state.selector.active
      await send_keys('\x0b', lambda: repl._state.selector.menu.selected == 0)
      assert repl._state.selector.menu.selected == 0
      await send_keys('fxbg', lambda: repl._state.selector.query == 'fxbg')
      searching_styles.update(
        style
        for style, _ in _component_text(
          _selector.Selector().render(repl._state)
        )
      )
      await send_keys('\r', lambda: not repl._state.selector.active)
      assert not repl._state.selector.active
      assert repl._input.text == 'fix the bug'
      assert not repl._state.turn.submitted_inputs
      assert not repl._state.activity.busy
      accepted_styles.update(_input_styles(repl))
      repl._input.text = 'fix something else'
      later_styles.update(_input_styles(repl))
      repl._input.cursor_position = 4
      await send_keys(
        '\x12', lambda: len(repl._state.selector.filtered_choices()) == 2
      )
      await send_keys(
        'no-match', lambda: repl._state.selector.query == 'no-match'
      )
      assert repl._state.selector.filtered_choices() == ()
      await send_keys('\x1b', lambda: not repl._state.selector.active)
      assert not repl._state.selector.active
      assert repl._input.text == 'fix something else'
      assert repl._input.cursor_position == 4
      repl._app.exit()

    # Keep assertions out of prompt_toolkit's interactive exception handler.
    async with asyncio.timeout(10), asyncio.TaskGroup() as tasks:
      tasks.create_task(_run_repl(repl))
      tasks.create_task(interact())

  # The match is highlighted while the search pane is open.
  assert any('bold underline' in style for style in searching_styles)
  # ...and the highlight ends with the search, in the recalled prompt and in
  # every prompt typed after it.
  assert not any('search' in style for style in accepted_styles)
  assert not any('search' in style for style in later_styles)


@pytest.mark.asyncio
async def test_exploration_retains_detail_row_between_tools_and_activity_until_turn_finishes() -> (
  None
):
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetReplaying(False), _actions.TurnStarted())
  _start_calls(
    repl,
    [
      {
        'id': 'read-1',
        'type': 'function',
        'function': {'name': 'read', 'arguments': '{"file_path":"first.py"}'},
      }
    ],
  )
  assert _pending.rows(repl._state) == 2
  assert '└ Reading first.py' in _ansi_text(_pending.render(repl._state))
  _handle(
    repl,
    _delta({'role': 'tool', 'tool_call_id': 'read-1', 'content': 'source'}),
  )
  assert _pending.rows(repl._state) == 2
  assert '└ Read first.py' in _ansi_text(_pending.render(repl._state))

  repl._handle_message(_stream_reasoning('Checking the next file.'))
  assert _pending.rows(repl._state) == 2
  assert repl._state.tool_group is not None
  _start_calls(
    repl,
    [
      {
        'id': 'search-1',
        'type': 'function',
        'function': {
          'name': 'grep',
          'arguments': '{"pattern":"render","path":"src"}',
        },
      }
    ],
  )
  assert _pending.rows(repl._state) == 2
  assert '└ Searching render in src' in _ansi_text(_pending.render(repl._state))
  _handle(
    repl,
    _delta({'role': 'tool', 'tool_call_id': 'search-1', 'content': 'matches'}),
  )
  assert _pending.rows(repl._state) == 2
  assert '└ Searched render in src' in _ansi_text(_pending.render(repl._state))

  repl._handle_message(_stream_content('Here is the final answer.'))
  assert _pending.rows(repl._state) == 0
  assert 'Working' in _fragments_text(_render_activity(repl))
  assert repl._state.activity.busy
  repl._handle_message(_turn_completed())
  assert _render_activity(repl) == []
  assert not repl._state.activity.busy
  await _settle(repl)


@pytest.mark.parametrize(
  ('content', 'display', 'error', 'expected'),
  [
    ('[Completed in 0.0s]', 'rm -- test.txt', False, ['· Ran rm -- test.txt']),
    (
      'echo hello\n[Completed in 0.0s]',
      'echo hello',
      False,
      ['· Ran echo hello', '  └ echo hello'],
    ),
    (
      'Exit code: 1\nfailed assertion\n[Completed in 0.4s]',
      'pytest',
      False,
      ['✗ Ran pytest · exit 1', '  └ failed assertion'],
    ),
    (
      'Command timed out after 60000ms\nPartial stdout:\nwaiting\n'
      '[Completed in 60.0s]',
      'sleep 60',
      True,
      ['✗ Ran sleep 60 · timed out', '  └ waiting'],
    ),
    (
      'Command timed out after 5ms\n[Completed in 0.0s]',
      'cat log',
      False,
      ['· Ran cat log', '  └ Command timed out after 5ms'],
    ),
    (
      'Timed out.\nExit code: -1',
      'sleep 60',
      False,
      ['✗ Ran sleep 60 · timed out'],
    ),
  ],
)
def test_compact_bash_renders_command_once_and_preserves_real_output(
  content: str,
  display: str,
  error: bool,
  expected: list[str],
) -> None:
  rendered = _rendering.tool_result_renderable(
    'bash',
    display,
    content,
    display,
    compact=True,
    activity_kind='tool',
    error=error,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 80))
  assert [line.rstrip() for line in plain.splitlines()] == expected


def test_new_repl_defaults_to_compact() -> None:
  repl = _NoEmitRepl()
  assert repl._state.status_bar.view_mode is _actions.ViewMode.COMPACT


def _exploration_result(
  name: str, *, success: bool = True
) -> agent_pb2.SessionMessage:
  return _delta({
    'role': 'tool',
    'name': name,
    'tool_call_id': name,
    'content': 'raw tool output',
    'success': success,
  })


@pytest.mark.asyncio
async def test_compact_groups_results_live_and_in_history() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetReplaying(False))
  repl._handle_message(_exploration_result('read'))
  repl._handle_message(_delta({'role': 'assistant', 'content': ''}))
  repl._handle_message(_exploration_result('grep'))
  repl._handle_message(_exploration_result('read', success=False))

  live = _ansi_text(_pending.render(repl._state))
  assert live.splitlines()[0].rstrip() == '· Explored · 1 read, 1 search'
  assert 'raw tool output' not in live
  assert not repl._terminal.emit_buffer
  assert _pending.rows(repl._state) == 2

  repl._handle_message(_turn_completed())
  assert repl._state.tool_group is None
  assert _pending.rows(repl._state) == 0
  emitted = _rendering.render_blocks(repl._terminal.emit_buffer, 80)
  history = repl._render_history()
  assert (
    _rendering.strip_ansi(emitted).strip()
    == _rendering.strip_ansi(history).strip()
  )
  assert history.count('Explored') == 1
  await _settle(repl)


@pytest.mark.asyncio
@pytest.mark.parametrize('streamed', [False, True])
async def test_compact_mixed_batch_groups_until_visible_result(
  streamed: bool,
) -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetReplaying(False), _actions.TurnStarted())
  repl._handle_message(_exploration_result('read'))
  calls: list[dict[str, Any]] = [
    {
      'id': str(index),
      'type': 'function',
      'function': {'name': name, 'arguments': arguments},
    }
    for index, (name, arguments) in enumerate([
      ('read', '{"file_path":"first.py"}'),
      ('read', '{"file_path":"second.py"}'),
      ('grep', '{"pattern":"first"}'),
      ('grep', '{"pattern":"second"}'),
      ('bash', '{"command":"jj status; jj root"}'),
    ])
  ]
  if streamed:
    repl._handle_message(_stream_reasoning('Checking the next files.'))
    for index, call in enumerate(calls):
      function = call['function']
      repl._handle_message(
        _stream_tool_call(index=index, name=function['name'])
      )
      repl._handle_message(
        _stream_tool_call(index=index, arguments=function['arguments'])
      )
  assistant = _delta({
    'role': 'assistant',
    'reasoning_content': 'Checking the next files.',
    'tool_calls': calls,
  })
  assistant.stream_final = streamed
  repl._handle_message(assistant)
  for index in range(4):
    repl._handle_message(
      _delta({
        'role': 'tool',
        'tool_call_id': str(index),
        'content': 'source or matches',
      })
    )
  repl._handle_message(
    _delta({
      'role': 'tool',
      'tool_call_id': '4',
      'content': '[Completed in 0.0s]',
    })
  )

  emitted = _rendering.strip_ansi(
    _rendering.render_blocks(repl._terminal.emit_buffer, 80)
  )
  history = _rendering.strip_ansi(repl._render_history())
  for rendered in (emitted, history):
    assert rendered.count('Explored') == 1
    assert '3 reads, 2 searches' in rendered
    assert rendered.index('Explored') < rendered.index('Ran jj status; jj root')
  assert repl._state.tool_group is None
  await _settle(repl)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'boundary',
  [
    _stream_content('Here is what I found.'),
    _delta({'role': 'tool', 'name': 'edit', 'content': 'Edited f.py'}),
    _approval_requested('c1', 'read', '{"file_path":"f.py"}'),
    _turn_completed(),
  ],
)
async def test_compact_flushes_before_visible_boundary(
  boundary: agent_pb2.SessionMessage,
) -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetReplaying(False))
  repl._handle_message(_exploration_result('read'))

  repl._handle_message(boundary)

  assert repl._state.tool_group is None
  assert '1 read' in _render_to_text(repl._terminal.emit_buffer[0])
  await _settle(repl)


@pytest.mark.asyncio
async def test_compact_replay_groups_without_changing_raw_records() -> None:
  repl = _NoEmitRepl()
  repl._handle_message(_exploration_result('read'))
  repl._handle_message(_exploration_result('grep'))
  repl._handle_message(_delta({'role': 'assistant', 'content': 'Found it.'}))
  repl._handle_message(_exploration_result('ls'))
  repl._handle_message(_replay_done('conv-1'))
  await _settle(repl)

  compact = _rendering.strip_ansi(repl._render_history())
  assert compact.count('Explored') == 2
  assert compact.index('1 read, 1 search') < compact.index('Found it.')
  assert compact.index('Found it.') < compact.index('1 directory scan')
  assert 'raw tool output' not in compact

  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))
  await _settle(repl)
  transcript = _rendering.strip_ansi(repl._render_history())
  assert transcript.count('raw tool output') == 3
  assert 'Explored' not in transcript


@pytest.mark.asyncio
async def test_changing_mode_flushes_live_group_once() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetReplaying(False), _actions.TurnStarted())
  repl._handle_message(_exploration_result('read'))

  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))

  assert repl._state.tool_group is None
  assert len(repl._terminal.emit_buffer) == 1
  repl._handle_message(_exploration_result('grep'))
  assert 'raw tool output' in _render_to_text(repl._terminal.emit_buffer[-1])
  await _settle(repl)


@pytest.mark.parametrize('width', [36, 64, 100])
def test_assistant_table_has_one_blank_line_and_aligned_gutter(
  width: int,
) -> None:
  content = (
    'Overview.\n\n'
    '| Folder | Purpose |\n| --- | --- |\n'
    '| `mirror-cli/` | Terminal UI |\n'
    '| `mirror-agent/` | Agent worker |\n\n'
    'Next steps.'
  )
  rendered = _rendering.renderable_to_ansi(
    _rendering.assistant_render(content), width
  )
  lines = [
    line.rstrip() for line in _rendering.strip_ansi(rendered).splitlines()
  ]
  assert lines == [
    '● Overview.',
    '',
    '  ┌───────────────┬──────────────┐',
    '  │ Folder        │ Purpose      │',
    '  ├───────────────┼──────────────┤',
    '  │ mirror-cli/   │ Terminal UI  │',
    '  ├───────────────┼──────────────┤',
    '  │ mirror-agent/ │ Agent worker │',
    '  └───────────────┴──────────────┘',
    '',
    '  Next steps.',
  ]


def test_reasoning_renders_markdown_not_literal_delimiters() -> None:
  rendered = _render_to_text(_rendering.reasoning_text('**Check** `file.py`.'))
  assert 'Check file.py.' in rendered
  assert '**' not in rendered
  assert '`' not in rendered


def test_result_summary_compact_truncates_to_one_line() -> None:
  multi = 'first line\nsecond line\nthird'
  full = _rendering.result_summary(multi)
  compact = _rendering.result_summary(multi, compact=True)
  assert 'second line' in full.plain
  assert 'second line' not in compact.plain
  assert 'first line' in compact.plain
  assert compact.plain.count('\n') == 0

  long_line = 'x' * 100
  truncated = _rendering.result_summary(long_line, compact=True)
  block = _rendering.tool_block('Result', truncated)
  narrow = (
    _rendering
    .strip_ansi(_rendering.renderable_to_ansi(block, 40))
    .splitlines()[1]
    .strip()
  )
  wide = (
    _rendering
    .strip_ansi(_rendering.renderable_to_ansi(block, 120))
    .splitlines()[1]
    .strip()
  )
  assert narrow == '└ ' + 'x' * 35 + '…'
  assert wide == f'└ {long_line}'


def test_compact_delta_hides_reasoning_without_a_placeholder() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  rendered = _handle(
    repl,
    _delta({
      'role': 'assistant',
      'content': 'the answer',
      'reasoning_content': 'secret thoughts',
    }),
  )
  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'the answer' in text
  assert 'secret thoughts' not in text
  assert 'Thought' not in text
  assert 'chars)' not in text


def test_default_mode_shows_reasoning_in_delta() -> None:
  repl = _repl()
  rendered = _handle(
    repl,
    _delta({
      'role': 'assistant',
      'content': 'the answer',
      'reasoning_content': 'visible thoughts',
    }),
  )
  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'visible thoughts' in text


def test_compact_mode_hides_reasoning_in_stream_preview() -> None:
  preview = _state.StreamPreview(reasoning='hidden thoughts', content='answer')
  ansi, _ = _stream_mod.render_stream_preview(
    preview, width=80, max_lines=10, compact_reasoning=True
  )
  text = _ansi_text(ansi)
  assert 'answer' in text
  assert 'hidden thoughts' not in text
  assert 'Thought' not in text  # Thinking lives in the status row.


@pytest.mark.asyncio
async def test_compact_mode_retires_reasoning_when_content_arrives() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  repl._handle_message(_stream_reasoning('thinking'))
  repl._handle_message(_stream_content('answer'))
  preview = repl._stream_pane.preview
  assert preview is not None
  # Retain the prefix for final-message deduplication, without printing it.
  assert preview.committed_reasoning == 'thinking'
  assert preview.reasoning == ''
  await _settle(repl)


@pytest.mark.asyncio
async def test_compact_mode_commits_reasoning_when_tool_call_arrives() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  repl._handle_message(_stream_reasoning('thinking'))
  repl._handle_message(_stream_tool_call(index=0, name='write'))
  preview = repl._stream_pane.preview
  assert preview is not None
  assert preview.committed_reasoning == 'thinking'
  assert preview.reasoning == ''


@pytest.mark.asyncio
async def test_commit_content_strips_prefix_when_more_content_arrives() -> None:
  """Content committed before tool calls must not be duplicated in the final.

  When the model streams answer text, starts a tool call, then streams more
  answer text, the final message has the full content.  The committed prefix
  must be stripped so only the remainder is rendered.
  """

  class _RecordingRepl(cli._Repl):
    def __init__(self) -> None:
      self.emitted: list[_console.RenderableType] = []
      self.written = asyncio.Event()
      super().__init__(
        [sys.executable, '-c', 'pass'],
        {'conversation_id': 'conv-1', 'model': 'm-test'},
        session_id='test-session',
        extension_registry=extension.ExtensionRegistry(),
        workspace_effects=asyncio.Queue(),
        workspace_controller=mock.Mock(
          spec=_workspace_controller.WorkspaceController
        ),
      )

      async def emit(
        renderables: list[_console.RenderableType],
      ) -> None:
        self.emitted.extend(renderables)
        self.written.set()

      self._terminal.emit = emit

  repl = _RecordingRepl()
  repl._handle_message(_stream_content('Hello'))
  repl._handle_message(_stream_tool_call(index=0, name='write'))
  preview = repl._stream_pane.preview
  assert preview is not None
  assert preview.committed_content == 'Hello'
  await asyncio.wait_for(repl.written.wait(), timeout=1)

  # More content arrives while tool calls are streaming.
  repl._handle_message(_stream_content(' world'))

  repl.written.clear()
  repl._handle_message(
    _stream_final(
      {
        'role': 'assistant',
        'content': 'Hello world',
        'tool_calls': [
          {'id': 'c0', 'type': 'function', 'function': {'name': 'write'}}
        ],
      },
    )
  )
  await asyncio.wait_for(repl.written.wait(), timeout=1)

  committed = _render_to_text(repl.emitted[0])
  assert 'Hello' in committed
  final = _render_to_text(repl.emitted[-1])
  # The full "Hello world" must not appear in the final renderable —
  # only the uncommitted remainder " world" should be there.
  assert 'Hello world' not in final
  assert 'world' in final
  assert 'Hello world' in repl._render_history()


def test_compact_mode_preserves_diff_without_duplicate_headers() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': 'edit', 'arguments': '{"file_path":"f.py"}'},
      }
    ],
  )
  diff = '--- f.py\n+++ f.py\n@@ -1,2 +1,2 @@\n-old\n+new\n'
  rendered = _handle(
    repl,
    _delta(
      {
        'role': 'tool',
        'tool_call_id': 'c1',
        'content': 'done',
        'name': 'edit',
      },
      display=diff,
    ),
  )
  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'Edited f.py' in text
  assert '- old' in text
  assert '+ new' in text
  assert '--- f.py' not in text
  assert '+++ f.py' not in text
  assert 'edit(' not in text


def test_compact_bash_keeps_a_bounded_output_preview() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.COMPACT))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"ls"}'},
      }
    ],
  )
  result = {
    'role': 'tool',
    'tool_call_id': 'c1',
    'content': 'line one\nline two\nline three',
  }
  rendered = _handle(repl, _delta(result))
  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'Ran ls' in text
  assert 'line one' in text
  assert '… (+2 lines)' in text
  assert 'line two' not in text
  assert 'line three' not in text


def test_compact_bash_wraps_command_and_ellipsizes_output() -> None:
  rendered = _rendering.tool_result_renderable(
    'bash',
    '',
    '界' * 100 + '\n[Completed in 0.0s]',
    'cd /workspace/project &&\n  bazel test //project/toolbox/mirror:all',
    compact=True,
    activity_kind='tool',
    error=False,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 30))
  assert [line.rstrip() for line in plain.splitlines()] == [
    '· Ran cd /workspace/project &&',
    '  bazel test',
    '  //project/toolbox/mirror:all',
    '  └ 界界界界界界界界界界界界 …',
  ]


def test_compact_bash_bounds_a_long_command() -> None:
  rendered = _rendering.tool_result_renderable(
    'bash',
    '',
    'Exit code: 1\n[Completed in 0.0s]',
    'echo ' + 'x' * 1000,
    compact=True,
    activity_kind='tool',
    error=False,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 80))
  assert [line.rstrip() for line in plain.splitlines()] == [
    '✗ Ran echo',
    '  ' + 'x' * 78,
    '  ' + 'x' * 78,
    '  ' + 'x' * 78,
    '  … · exit 1',
  ]


def test_compact_bash_failure_has_one_line_preview() -> None:
  rendered = _rendering.tool_result_renderable(
    'bash',
    '',
    'Exit code: 1\nfirst\nsecond\nthird\nfourth\n[Completed in 1.2s]',
    'pytest',
    compact=True,
    activity_kind='tool',
    error=False,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 80))
  assert [line.rstrip() for line in plain.splitlines()] == [
    '✗ Ran pytest · exit 1',
    '  └ first … (+3 lines)',
  ]


def test_compact_running_bash_wraps_its_command() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.TerminalResized(40), _actions.TurnStarted())
  command = 'cd /workspace/project && bazel test //project/toolbox/mirror:all'
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {
          'name': 'bash',
          'arguments': json.dumps({'command': command}),
        },
      }
    ],
  )
  rendered = _rendering.strip_ansi(_ansi_text(_pending.render(repl._state)))
  assert [line.rstrip() for line in rendered.splitlines()] == [
    '· Running cd /workspace/project && bazel',
    '  test //project/toolbox/mirror:all',
  ]
  assert _pending.rows(repl._state) == 2


def test_compact_unanswered_bash_wraps_its_command() -> None:
  rendered = _rendering.unanswered_call(
    (
      'bash',
      'cd /workspace/project && bazel test //project/toolbox/mirror:all',
      'tool',
    ),
    compact=True,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 40))
  assert [line.rstrip() for line in plain.splitlines()] == [
    '· bash cd /workspace/project && bazel',
    '  test //project/toolbox/mirror:all',
  ]


def test_expanded_bash_wraps_a_long_command() -> None:
  rendered = _rendering.tool_result_renderable(
    'bash',
    '',
    'ok',
    'cd /workspace/project && bazel test //project/toolbox/mirror:all',
    compact=False,
    activity_kind='tool',
    error=False,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 40))
  assert [line.rstrip() for line in plain.splitlines()] == [
    '· bash(cd /workspace/project && bazel',
    '  test //project/toolbox/mirror:all)',
    '  └ ok',
  ]


def test_expanded_bash_preserves_full_command_and_output() -> None:
  rendered = _rendering.tool_result_renderable(
    'bash',
    'shortened…',
    'first\nsecond\nthird\nfourth',
    'echo first\necho second',
    compact=False,
    activity_kind='tool',
    error=False,
  )
  plain = _rendering.strip_ansi(_rendering.renderable_to_ansi(rendered, 80))
  assert 'echo first' in plain
  assert 'echo second' in plain
  assert 'fourth' in plain
  assert 'shortened' not in plain


@pytest.mark.parametrize('mode', list(_actions.ViewMode))
def test_core_footer_does_not_render_view_mode(mode: _actions.ViewMode) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(mode))
  assert _fragments_text(_render_footer(repl)) == ' 🛠  auto | m-test · medium '


def test_extra_compact_hides_reasoning_and_keeps_the_answer() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.EXTRA_COMPACT))
  rendered = _handle(
    repl,
    _delta({
      'role': 'assistant',
      'content': 'the answer',
      'reasoning_content': 'secret thoughts',
    }),
  )
  assert rendered is not None
  text = _render_to_text(rendered)
  assert 'the answer' in text
  assert 'secret thoughts' not in text
  assert 'Thought' not in text


def test_extra_compact_hides_tool_results() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.EXTRA_COMPACT))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"ls"}'},
      }
    ],
  )
  rendered = _handle(
    repl,
    _delta({
      'role': 'tool',
      'tool_call_id': 'c1',
      'content': 'line one\nline two',
      'name': 'bash',
    }),
  )
  assert rendered is None
  assert 'c1' not in _running_call_hints(repl)


def test_extra_compact_hides_task_results() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.EXTRA_COMPACT))
  rendered = _handle(repl, _task_result('1234', 'bash', 'complete'))
  assert rendered is None


def test_extra_compact_hides_pending_tool_rows() -> None:
  repl = _repl(output=_SizedOutput(rows=24, columns=80))
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.EXTRA_COMPACT))
  _start_calls(
    repl,
    [
      {
        'id': 'c1',
        'type': 'function',
        'function': {'name': 'bash', 'arguments': '{"command":"ls"}'},
      }
    ],
  )
  assert _running_call_ids(repl) == ('c1',)
  assert _pending.rows(repl._state) == 0
  assert _pending.count(repl._state) == 0


def test_extra_compact_hides_reasoning_in_stream_preview() -> None:
  preview = _state.StreamPreview(reasoning='hidden thoughts', content='answer')
  ansi, _ = _stream_mod.render_stream_preview(
    preview, width=80, max_lines=10, hide_reasoning=True
  )
  text = _ansi_text(ansi)
  assert 'answer' in text
  assert 'hidden thoughts' not in text
  assert 'Thought' not in text
  assert 'Thinking' not in text


@pytest.mark.asyncio
async def test_extra_compact_does_not_commit_reasoning_to_scrollback() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.EXTRA_COMPACT))
  repl._handle_message(_stream_reasoning('thinking'))
  repl._handle_message(_stream_content('answer'))
  preview = repl._stream_pane.preview
  assert preview is not None
  assert preview.committed_reasoning == 'thinking'
  assert preview.reasoning == ''
  assert repl._stream_pane.committed_blocks() == []
  ansi_text = _ansi_text(repl._stream_pane.ansi())
  assert 'answer' in ansi_text
  assert 'thinking' not in ansi_text
  assert 'Thought' not in ansi_text


def test_extra_compact_history_shows_only_user_and_assistant_text() -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.EXTRA_COMPACT))
  repl._transcript.extend([
    _delta({'role': 'user', 'content': 'please fix it'}).agent_message,
    _delta({
      'role': 'assistant',
      'content': 'all done',
      'reasoning_content': 'secret plan',
      'tool_calls': [
        {
          'id': 'c1',
          'type': 'function',
          'function': {'name': 'bash', 'arguments': '{"command":"ls"}'},
        }
      ],
    }).agent_message,
    _delta({
      'role': 'tool',
      'tool_call_id': 'c1',
      'content': 'file.py',
      'name': 'bash',
    }).agent_message,
  ])
  text = repl._render_history()
  assert 'please fix it' in text
  assert 'all done' in text
  assert 'secret plan' not in text
  assert 'file.py' not in text
  assert 'bash(ls)' not in text


@pytest.mark.asyncio
async def test_the_view_command_redraws_the_history_like_the_key_does(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Density is a property of the drawn history, so asking for it redraws,
  # whichever way it was asked for. Nothing is written to say it changed.
  registry = extension.ExtensionRegistry()
  await registry.register(_view_mode.ViewModeExtension(), None)
  repl = _repl(extension_registry=registry)
  await repl._extension_host.start()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  await repl._run_slash('/view compact')
  await _settle(repl)

  assert repl._state.status_bar.view_mode is _actions.ViewMode.COMPACT
  assert len(paints) == 1
  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'first ask' in ansi
  assert 'View mode' not in ansi

  await repl._extension_host.close()
  await registry.close()


@pytest.mark.asyncio
async def test_changing_density_mid_turn_redraws_history(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_view_mode.ViewModeExtension(), None)
  repl = _repl(extension_registry=registry)
  await repl._extension_host.start()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()
  repl.dispatch(_actions.TurnStarted())

  await repl._run_slash('/view')
  await _settle(repl)

  assert repl._state.status_bar.view_mode is _actions.ViewMode.COMPACT
  assert len(paints) == 1
  assert paints[0][1] is _terminal_mod.TerminalClear.SCREEN
  assert 'first ask' in paints[0][0]
  assert repl._state.activity.busy

  await repl._extension_host.close()
  await registry.close()


@pytest.mark.asyncio
async def test_drawing_history_leaves_the_next_turn_no_calls_to_answer(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Drawing the history pairs every call with its result in order to render
  # it. That pairing belongs to the drawing: were it left behind, the end of
  # the next turn would print a resultless row for every tool the
  # conversation has ever run.
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  repl._handle_message(_delta({'role': 'user', 'content': 'ask'}))
  repl._handle_message(
    _delta(
      {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
          {
            'id': 'c1',
            'type': 'function',
            'function': {'name': 'bash', 'arguments': '{"command":"ls"}'},
          }
        ],
      },
    )
  )
  repl._handle_message(
    _delta({
      'role': 'tool',
      'tool_call_id': 'c1',
      'content': 'out',
      'name': 'bash',
    })
  )
  repl._handle_message(_replay_done('conv-1'))
  await _settle(repl)
  assert 'bash' in paints[0][0]  # The history did draw the answered call.
  paints.clear()

  repl._handle_message(_turn_completed())
  await _settle(repl)

  assert paints == []


@pytest.mark.asyncio
async def test_rerendered_history_replaces_the_screen_from_the_transcript(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Re-rendering draws the conversation the TUI already has: no worker round
  # trip, and no replay to wait for.
  repl = _repl()
  sent: list[agent_pb2.ClientMessage] = []

  def send(request: agent_pb2.ClientMessage) -> None:
    sent.append(request)

  paints = _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl, '_send_request', send)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  await repl._rerender()
  await _settle(repl)

  assert sent == []
  assert len(paints) == 1
  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCREEN
  assert 'first ask' in ansi


@pytest.mark.asyncio
async def test_application_uses_color_capability_or_output_default() -> None:
  output = _SizedOutput(rows=24, columns=80)
  repl = _repl(output=output)
  try:
    assert repl._app.color_depth == (
      themes.prompt_color_depth() or output.get_default_color_depth()
    )
  finally:
    await repl.close()


def test_initial_theme_applies_before_deferred_extension_loading(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setenv('TERM', 'xterm-256color')
  repl = _repl(initial_theme='light')
  assert themes.current(repl._state.extensions) == 'light'
  assert themes.EXTENSION_ID not in repl._state.extensions
  repl.dispatch(_actions.ExtensionUnloaded(themes.EXTENSION_ID))
  assert themes.current(repl._state.extensions) == 'dark'


@pytest.mark.asyncio
async def test_themes_persist_reload_and_keep_extension_consent(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setenv('TERM', 'xterm-256color')
  bundled = tmp_path / 'bundled'
  source = bundled / 'themes' / 'extension.py'
  source.parent.mkdir(parents=True)
  source.write_text(pathlib.Path(themes_extension.__file__).read_text())
  settings = config.Config(tmp_path / 'config.json')
  settings.path.write_text(
    '{"untouched":42,"extensions":{"user/unapproved":{"enabled":false}}}'
  )
  registry = extension.ExtensionRegistry()
  await registry.register(
    themes_extension.ThemesExtension(), None, directory=source.parent
  )
  manager = user_extensions.UserExtensions(
    registry, tmp_path / 'extensions', bundled=bundled, settings=settings
  )
  await manager.start()
  first = _repl(extension_registry=registry, extension_manager=manager)
  second = _repl(
    extension_registry=registry, extension_manager=manager, active=False
  )
  try:
    await first._extension_host.start()
    await second._extension_host.start()
    first._input.text = 'keep this draft'
    await first._run_slash('/themes')
    first.dispatch(_actions.SelectorSearchChanged('light'))
    await asyncio.gather(*first.dispatch(*first._selector_choice()))
    assert first._input.text == 'keep this draft'
    assert themes.current(first._state.extensions) == 'light'
    assert settings.read() == {
      'untouched': 42,
      'extensions': {
        'user/unapproved': {'enabled': False},
        'builtin/themes': {'config': {'name': 'light'}},
      },
    }
    await first._run_slash('/themes')
    assert first._state.selector.selector is not None
    assert first._state.selector.selector.current == 'light'
    await manager.apply_settings(settings.read().get('extensions'))
    assert themes.current(second._state.extensions) == 'light'
    assert themes.saved(config.Config(settings.path).read()) == 'light'
    settings.set_extension('builtin/themes', False)
    await manager.apply_settings(settings.read().get('extensions'))
    assert themes.current(first._state.extensions) == 'dark'
    assert themes.current(second._state.extensions) == 'dark'
    assert themes.saved(settings.read()) == 'dark'
    assert 'themes' not in [
      command.name for command in first.get_available_commands()
    ]
    assert settings.read()['extensions'] == {
      'user/unapproved': {'enabled': False},
      'builtin/themes': {'enabled': False, 'config': {'name': 'light'}},
    }
  finally:
    await first.close()
    await second.close()
    await registry.close()


@pytest.mark.asyncio
async def test_themes_report_invalid_names_and_save_errors(
  tmp_path: pathlib.Path,
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(themes_extension.ThemesExtension(), None)
  manager = user_extensions.UserExtensions(registry, tmp_path / 'extensions')
  repl = _repl(extension_registry=registry, extension_manager=manager)
  try:
    await repl._extension_host.start()
    await repl._run_slash('/themes missing')
    assert 'Usage: /themes' in _render_welcome_notices(repl)
    assert not (tmp_path / 'config.json').exists()
    (tmp_path / 'config.json').write_text('invalid json')
    await repl._run_slash('/themes light')
    assert 'Could not save theme:' in _render_welcome_notices(repl)
    assert (tmp_path / 'config.json').read_text() == 'invalid json'
    assert repl._state.extensions.configuration(themes.EXTENSION_ID) == {
      'name': 'dark'
    }
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('defer', 'resume'),
  [
    (_actions.TurnStarted(), _actions.TurnFinished()),
    (_actions.SetActive(False), _actions.SetActive(True)),
  ],
  ids=['turn', 'background'],
)
async def test_theme_repaint_waits_for_idle_foreground_without_decorations(
  monkeypatch: pytest.MonkeyPatch,
  defer: _actions.Action,
  resume: _actions.Action,
) -> None:
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setenv('TERM', 'xterm-256color')
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['literal #ff0000 user text'])
  await _settle(repl)
  before = repl._render_history()
  paints.clear()
  repl._input.text = 'unfinished draft'
  repl.dispatch(defer)
  await _dispatch_and_wait(
    repl,
    _actions.ExtensionConfigChanged(themes.EXTENSION_ID, {'name': 'light'}),
  )
  assert paints == []
  assert repl._app.style is not None
  assert repl._app.style.get_attrs_for_style_str('class:composer') == (
    themes.prompt_style('light').get_attrs_for_style_str('class:composer')
  )
  await _dispatch_and_wait(repl, resume)
  await _settle(repl)
  assert len(paints) == 1
  assert paints[0][0] != before
  assert _rendering.strip_ansi(paints[0][0]) == _rendering.strip_ansi(before)
  assert repl._input.text == 'unfinished draft'
  assert not repl._state.session.rerendering
  await repl.close()


@pytest.mark.asyncio
async def test_theme_repaint_rechecks_idle_after_a_turn_starts(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setenv('TERM', 'xterm-256color')
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()
  repaint_tasks = repl.dispatch(
    _actions.ExtensionConfigChanged(themes.EXTENSION_ID, {'name': 'light'})
  )
  repl.dispatch(_actions.TurnStarted())
  await asyncio.gather(*repaint_tasks)
  assert paints == []
  await _dispatch_and_wait(repl, _actions.TurnFinished())
  await _settle(repl)
  assert len(paints) == 1
  assert 'first ask' in paints[0][0]
  assert 'History re-rendered' not in paints[0][0]
  await repl.close()


@pytest.mark.asyncio
async def test_theme_switch_refreshes_stream_cache_without_changing_deltas(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.setenv('TERM', 'xterm-256color')
  repl = _repl()
  repl._handle_message(_stream_content('A partial `literal #ff0000` answer'))
  before = repl._stream_pane.ansi().value
  cursor = repl._stream_pane.cursor_position()
  preview = repl._state.stream.preview
  await _dispatch_and_wait(
    repl,
    _actions.ExtensionConfigChanged(themes.EXTENSION_ID, {'name': 'light'}),
  )
  after = repl._stream_pane.ansi().value
  assert before != after
  assert _rendering.strip_ansi(before) == _rendering.strip_ansi(after)
  assert repl._state.stream.preview == preview
  assert repl._stream_pane.cursor_position() == cursor
  repl._handle_message(_stream_content(' continues'))
  assert 'continues' in repl._stream_pane.ansi().value
  await repl.close()


def _resize_repl(
  monkeypatch: pytest.MonkeyPatch,
  *,
  columns: int = 80,
  rows: int = 24,
) -> tuple[cli._Repl, _SizedOutput]:
  monkeypatch.setattr(cli, '_RESIZE_SETTLE_SECONDS', 0)
  output = _SizedOutput(rows=rows, columns=columns)
  repl = _repl(output=output)
  repl._terminal.console.size = (columns, rows)
  return repl, output


def _fire_resize(repl: cli._Repl) -> None:
  repl._app._on_resize()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'initial_columns, columns, rows',
  [(32, 160, 40), (320, 80, 24), (320, 32, 12)],
)
async def test_activation_uses_live_dimensions_after_inactive_resize(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  initial_columns: int,
  columns: int,
  rows: int,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  output = _SizedOutput(rows=50, columns=initial_columns)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl(active=False)
    repl._input.text = 'unfinished draft'
    output.size = _data_structures.Size(rows=rows, columns=columns)
    observed: list[tuple[int, str]] = []

    def first_frame(app: _application.Application[str | None]) -> None:
      app.after_render -= first_frame
      observed.append((repl._state.layout.columns, _render_banner(repl)))
      app.exit()

    repl._app.after_render += first_frame
    try:
      await repl.activate(show_context=True)
      assert observed[0][0] == columns
      assert (
        max(_cells.cell_len(line) for line in observed[0][1].splitlines())
        <= columns
      )
      assert 'm-test' in observed[0][1]
      assert repl._input.text == 'unfinished draft'
      assert not repl._transcript.has_conversation
    finally:
      await repl.close()


@pytest.mark.asyncio
async def test_width_change_hints_to_redraw_instead_of_rerendering(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, output = _resize_repl(monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()
  relayouts: list[object] = []
  repl._relayout_prompt = lambda: relayouts.append(True)

  output.size = _data_structures.Size(rows=24, columns=120)
  _fire_resize(repl)
  await _settle(repl)

  assert paints == []
  assert relayouts == [True]
  assert repl._state.status_bar.resize_dirty
  assert 'Ctrl+L to redraw' in ''.join(text for _, text in _render_footer(repl))
  assert repl._terminal.console.width == 120
  assert repl._state.session.history_width == 80


@pytest.mark.asyncio
async def test_height_only_resize_does_not_hint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, output = _resize_repl(monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  output.size = _data_structures.Size(rows=40, columns=80)
  _fire_resize(repl)
  await _settle(repl)

  assert paints == []
  assert not repl._state.status_bar.resize_dirty
  assert 'Ctrl+L' not in ''.join(text for _, text in _render_footer(repl))


@pytest.mark.asyncio
async def test_resize_does_not_relayout_the_prompt_until_settle(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, output = _resize_repl(monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  relayouts: list[object] = []
  repl._relayout_prompt = lambda: relayouts.append(output.size)

  output.size = _data_structures.Size(rows=40, columns=80)
  _fire_resize(repl)
  output.size = _data_structures.Size(rows=48, columns=80)
  _fire_resize(repl)

  assert relayouts == []

  await _settle(repl)

  assert relayouts == [_data_structures.Size(rows=48, columns=80)]


@pytest.mark.asyncio
async def test_resize_back_to_original_width_still_hints(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Intermediate widths rewrap on-screen ANSI even if the drag ends where it
  # started.
  repl, output = _resize_repl(monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  output.size = _data_structures.Size(rows=24, columns=100)
  _fire_resize(repl)
  output.size = _data_structures.Size(rows=24, columns=80)
  _fire_resize(repl)
  await _settle(repl)

  assert paints == []
  assert repl._state.status_bar.resize_dirty
  assert repl._state.session.history_width == 80


@pytest.mark.asyncio
async def test_ctrl_l_rerenders_without_cycling_view_mode(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, _output = _resize_repl(monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()
  mode = repl._state.status_bar.view_mode

  repl.dispatch(_actions.RerenderRequested())
  await _settle(repl)

  assert repl._state.status_bar.view_mode is mode
  assert len(paints) == 1
  assert paints[0][1] is _terminal_mod.TerminalClear.SCREEN
  assert 'first ask' in paints[0][0]


@pytest.mark.asyncio
async def test_ctrl_l_clears_the_redraw_hint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, output = _resize_repl(monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  output.size = _data_structures.Size(rows=24, columns=120)
  _fire_resize(repl)
  await _settle(repl)
  paints.clear()

  repl.dispatch(_actions.RerenderRequested())
  await _settle(repl)

  assert len(paints) == 1
  assert not repl._state.status_bar.resize_dirty
  assert 'Ctrl+L' not in ''.join(text for _, text in _render_footer(repl))
  assert repl._state.session.history_width == 120


@pytest.mark.asyncio
@pytest.mark.parametrize('busy', [False, True])
@pytest.mark.parametrize(
  ('initial', 'expected'),
  [
    (_actions.ViewMode.TRANSCRIPT, _actions.ViewMode.COMPACT),
    (_actions.ViewMode.COMPACT, _actions.ViewMode.EXTRA_COMPACT),
    (_actions.ViewMode.EXTRA_COMPACT, _actions.ViewMode.TRANSCRIPT),
  ],
)
async def test_ctrl_t_cycles_view_mode_and_rerenders(
  monkeypatch: pytest.MonkeyPatch,
  busy: bool,
  initial: _actions.ViewMode,
  expected: _actions.ViewMode,
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_view_mode.ViewModeExtension(), None)
  repl = _repl(extension_registry=registry)
  await repl._extension_host.start()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  await _dispatch_and_wait(repl, _actions.SetViewMode(initial))
  paints.clear()
  if busy:
    repl.dispatch(_actions.AgentTurnStarted())
  repl._input.document = _document.Document('keep draft', cursor_position=4)
  turn = repl._state.turn
  parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)

  with _application.set_app(repl._app):
    parser.feed('\x14')
    repl._app.key_processor.process_keys()
    await _settle(repl)

  assert repl._state.status_bar.view_mode is expected
  assert len(paints) == 1
  assert paints[0][1] is _terminal_mod.TerminalClear.SCREEN
  assert 'first ask' in paints[0][0]
  assert repl._state.activity.busy is busy
  assert repl._state.turn == turn
  assert repl._input.document == _document.Document(
    'keep draft', cursor_position=4
  )

  await repl._extension_host.close()
  await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('remote_endpoint', [None, 'localhost:12345'])
async def test_view_change_restores_committed_stream_reasoning(
  monkeypatch: pytest.MonkeyPatch,
  remote_endpoint: str | None,
) -> None:
  repl = _repl(remote_endpoint=remote_endpoint)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _dispatch_and_wait(
    repl, _actions.SetViewMode(_actions.ViewMode.COMPACT)
  )
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(_stream_reasoning('available reasoning'))
  repl._handle_message(_stream_content('live answer'))
  preview = repl._state.stream.preview
  await _settle(repl)
  paints.clear()

  await _dispatch_and_wait(
    repl, _actions.SetViewMode(_actions.ViewMode.TRANSCRIPT)
  )

  assert 'Reasoning' in paints[-1][0]
  assert paints[-1][0].count('available reasoning') == 1
  assert 'live answer' in _ansi_text(repl._stream_pane.ansi())
  assert repl._state.stream.preview == preview
  assert repl._state.activity.busy
  assert repl._state.activity.status == 'thinking…'

  await _dispatch_and_wait(repl, _actions.CycleViewMode())

  assert 'available reasoning' not in paints[-1][0]
  assert 'Reasoning' not in paints[-1][0]
  assert 'live answer' in _ansi_text(repl._stream_pane.ansi())
  assert repl._state.stream.preview == preview


@pytest.mark.asyncio
async def test_view_redraw_includes_events_received_while_waiting_for_terminal(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  repl.dispatch(_actions.AgentTurnStarted())
  entered = asyncio.Event()
  resume = asyncio.Event()

  async def suspend(callback: Callable[[], None]) -> None:
    entered.set()
    await resume.wait()
    callback()

  monkeypatch.setattr(cli, '_run_in_terminal', suspend)
  redraw = repl.dispatch(_actions.CycleViewMode())
  await entered.wait()
  repl._handle_message(
    _delta({'role': 'assistant', 'content': 'arrived during redraw'})
  )
  repl._handle_message(_stream_reasoning('next request reasoning'))
  repl._handle_message(_stream_content('next request answer'))
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))
  resume.set()
  await asyncio.gather(*redraw)
  await _settle(repl)

  assert paints[-1][0].count('arrived during redraw') == 1
  assert paints[-1][0].count('next request reasoning') == 1
  assert not repl._terminal.emit_buffer
  assert 'next request answer' in _ansi_text(repl._stream_pane.ansi())
  assert repl._state.activity.busy


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', list(_actions.ViewMode))
async def test_reasoning_visibility_matches_live_and_saved_records(
  mode: _actions.ViewMode,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.SetViewMode(mode))
  repl._handle_message(_stream_reasoning('available reasoning'))
  live = _ansi_text(repl._stream_pane.ansi())
  saved = _handle(
    repl,
    _delta({
      'role': 'assistant',
      'content': 'answer',
      'reasoning_content': 'available reasoning',
    }),
  )
  assert saved is not None
  text = _render_to_text(saved)
  assert ('available reasoning' in live) is (
    mode is _actions.ViewMode.TRANSCRIPT
  )
  assert ('Reasoning' in live) is (mode is _actions.ViewMode.TRANSCRIPT)
  assert ('available reasoning' in text) is (
    mode is _actions.ViewMode.TRANSCRIPT
  )
  assert ('Reasoning' in text) is (mode is _actions.ViewMode.TRANSCRIPT)
  assert 'answer' in text
  await _settle(repl)


def test_answer_without_reasoning_has_no_reasoning_label() -> None:
  rendered = _rendering.assistant_turn_renderable(None, 'answer')
  assert rendered is not None
  assert 'Reasoning' not in _render_to_text(rendered)


@pytest.mark.asyncio
async def test_ctrl_l_is_ignored_while_a_turn_is_in_progress(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, _output = _resize_repl(monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  repl.dispatch(_actions.TurnStarted())
  repl.dispatch(_actions.RerenderRequested())
  await _settle(repl)

  assert paints == []


@pytest.mark.asyncio
async def test_slash_clear_flushes_scrollback_and_repaints(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  await repl._run_slash('/clear')
  await _settle(repl)

  assert len(paints) == 1
  ansi, clear = paints[0]
  assert clear is _terminal_mod.TerminalClear.SCROLLBACK
  assert 'first ask' in ansi
  assert 'Screen cleared' in ansi


@pytest.mark.asyncio
async def test_flush_scrollback_promotes_ctrl_l_to_drop_saved_lines(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl, _output = _resize_repl(monkeypatch)
  repl._terminal.flush_scrollback = True
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  repl.dispatch(_actions.RerenderRequested())
  await _settle(repl)

  assert len(paints) == 1
  assert paints[0][1] is _terminal_mod.TerminalClear.SCROLLBACK
  assert 'first ask' in paints[0][0]


@pytest.mark.asyncio
async def test_slash_clear_is_blocked_while_a_turn_is_in_progress(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  _seed_history(repl, ['first ask'])
  await _settle(repl)
  paints.clear()

  repl.dispatch(_actions.TurnStarted())
  await repl._run_slash('/clear')
  await _settle(repl)

  assert paints
  assert all(clear is _terminal_mod.TerminalClear.KEEP for _, clear in paints)
  assert 'unavailable' in paints[0][0]


@pytest.mark.asyncio
async def test_rerender_keeps_thinking_from_reasoning_content(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)
  repl._handle_message(_replay_done('conv-1'))
  await _settle(repl)
  repl._handle_message(
    _delta({
      'role': 'assistant',
      'content': 'the answer',
      'reasoning_content': 'secret thinking',
    })
  )
  paints.clear()

  await repl._rerender()
  await _settle(repl)

  assert 'secret thinking' in paints[-1][0]
  assert 'the answer' in paints[-1][0]


def test_working_timer_runs_continuously_across_model_waits_and_tools(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted(observed_at=10))
  monkeypatch.setattr(_rendering.time, 'monotonic', lambda: 22)

  fragments = _render_activity(repl)
  assert 'Working · 12s' in _fragments_text(fragments)

  _start_calls(
    repl,
    [
      {
        'id': 'read-1',
        'type': 'function',
        'function': {'name': 'read', 'arguments': '{"file_path":"first.py"}'},
      }
    ],
  )
  assert 'Exploring · 12s' in _fragments_text(_render_activity(repl))
  repl.dispatch(_actions.SetStatus('thinking…'))
  monkeypatch.setattr(_rendering.time, 'monotonic', lambda: 30)
  assert 'Exploring · 20s' in _fragments_text(_render_activity(repl))
  repl.dispatch(
    _actions.ClearRunningToolCalls(), _actions.SetStatus('running tool…')
  )
  assert 'Working · 20s' in _fragments_text(_render_activity(repl))
  repl.dispatch(_actions.TurnFinished())
  assert repl._state.activity.work_started is None
  repl.dispatch(_actions.TurnStarted(observed_at=30))
  assert 'Working · 0s' in _fragments_text(_render_activity(repl))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', list(_actions.ViewMode))
async def test_activity_accumulates_reported_output_across_tool_calls(
  mode: _actions.ViewMode,
) -> None:
  repl = _repl()
  repl.dispatch(
    _actions.SetReplaying(False),
    _actions.SetViewMode(mode),
    _actions.TurnStarted(),
  )
  repl._handle_message(_stream_reasoning('unreported reasoning'))
  repl._handle_message(
    _runtime(
      agent_pb2.RuntimeDelta(
        stream_usage=agent_pb2.StreamUsage(completion_tokens=534)
      )
    )
  )
  assert 'tokens' not in _fragments_text(_render_activity(repl))

  repl._handle_message(
    _stream_final(
      {
        'role': 'assistant',
        'tool_calls': [
          {
            'id': 'read-1',
            'type': 'function',
            'function': {
              'name': 'read',
              'arguments': '{"file_path":"first.py"}',
            },
          }
        ],
      },
      completion_tokens=534,
    )
  )
  assert 'Exploring (534 tokens)' in _fragments_text(_render_activity(repl))

  repl._handle_message(
    _delta({
      'role': 'tool',
      'tool_call_id': 'read-1',
      'content': 'contents',
    })
  )
  repl._handle_message(_stream_content('next response'))
  assert '(534 tokens)' in _fragments_text(_render_activity(repl))
  repl._handle_message(
    _stream_final(
      {'role': 'assistant', 'content': 'next response'}, completion_tokens=1000
    )
  )
  assert 'Working (1,534 tokens)' in _fragments_text(_render_activity(repl))
  await _settle(repl)


@pytest.mark.asyncio
@pytest.mark.parametrize('tokens', [None, 0])
async def test_activity_does_not_estimate_missing_or_zero_usage(
  tokens: int | None,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(_stream_content('x' * 4000))
  assert 'tokens' not in _fragments_text(_render_activity(repl))
  repl._handle_message(
    _stream_final(
      {'role': 'assistant', 'content': 'x' * 4000}, completion_tokens=tokens
    )
  )
  assert 'tokens' not in _fragments_text(_render_activity(repl))
  await _settle(repl)


@pytest.mark.asyncio
async def test_activity_preserves_completed_usage_after_stream_abort() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(
    _stream_final(
      {'role': 'assistant', 'content': 'first'}, completion_tokens=534
    )
  )
  repl._handle_message(_stream_content('unfinished'))
  repl._handle_message(
    _runtime(
      agent_pb2.RuntimeDelta(
        stream_usage=agent_pb2.StreamUsage(completion_tokens=1000)
      )
    )
  )
  repl._handle_message(_stream_aborted())
  assert '(534 tokens)' in _fragments_text(_render_activity(repl))
  repl._handle_message(_turn_completed())
  assert _render_activity(repl) == []
  assert repl._state.turn.output_tokens == 0
  await _settle(repl)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'boundary',
  [
    _actions.TurnStarted(),
    _actions.UserMessageObserved('next queued prompt'),
    _actions.SessionOpened('next-session'),
  ],
)
async def test_activity_resets_output_usage_for_new_turns(
  boundary: _actions.Action,
) -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(
    _stream_final(
      {'role': 'assistant', 'content': 'first'}, completion_tokens=534
    )
  )
  repl.dispatch(boundary)
  assert 'tokens' not in _fragments_text(_render_activity(repl))
  repl._handle_message(
    _stream_final(
      {'role': 'assistant', 'content': 'second'}, completion_tokens=7
    )
  )
  assert '(7 tokens)' in _fragments_text(_render_activity(repl))
  await _settle(repl)


def test_activity_does_not_count_replayed_response_usage() -> None:
  repl = _repl()
  repl.dispatch(_actions.TurnStarted(), _actions.SetReplaying(True))
  response = _stream_final(
    {'role': 'assistant', 'content': 'historical answer'},
    completion_tokens=1000,
  )
  response.stream_final = False
  repl._handle_message(response)
  assert repl._state.turn.output_tokens == 0
  assert 'tokens' not in _fragments_text(_render_activity(repl))


def test_compact_reasoning_uses_status_instead_of_a_transcript_row() -> None:
  preview = _state.StreamPreview(reasoning='x' * 1500)
  ansi, _ = _stream_mod.render_stream_preview(
    preview, width=80, max_lines=10, compact_reasoning=True
  )
  text = _ansi_text(ansi)
  assert text == ''


def test_compact_stream_preview_only_shows_the_answer() -> None:
  preview = _state.StreamPreview(reasoning='x' * 500, content='answer')
  ansi, _ = _stream_mod.render_stream_preview(
    preview, width=80, max_lines=10, compact_reasoning=True
  )
  text = _ansi_text(ansi)
  assert 'Thought' not in text
  assert 'answer' in text


@pytest.mark.asyncio
async def test_cropped_startup_replay_keeps_a_warning(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._terminal.console = _console.Console(width=80)
  repl._scrollback_limit = 5
  paints = _capture_paints(repl, monkeypatch)
  repl._terminal.pin_header(
    _rendering.log_text('warning', 'context window unknown')
  )
  _seed_history(repl, (f'line {i}' for i in range(30)))
  await _settle(repl)

  text = paints[0][0]

  # A startup warning is not history, so cropping the history keeps it.
  assert 'context window unknown' in text
  assert 'line 29' in text
  assert 'line 0' not in text


@pytest.mark.asyncio
async def test_cropped_switch_replay_keeps_divider_without_banner(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl(model='gpt-4o-mini')
  repl._terminal.console = _console.Console(width=80)
  repl._scrollback_limit = 5

  _record_restarts(repl, monkeypatch)
  paints = _capture_paints(repl, monkeypatch)
  await repl._switch_session('conv-2', create=False)
  for i in range(30):
    repl._handle_message(_delta({'role': 'user', 'content': f'line {i}'}))
  repl._handle_message(_replay_done('conv-2'))
  await _settle(repl)

  text = paints[0][0]

  assert 'Resumed conversation' in text
  assert 'Reflection mirror' not in text
  assert repl._transcript.has_conversation
  assert 'line 29' in text
  assert 'line 0' not in text


@pytest.mark.asyncio
async def test_an_empty_startup_replay_clears_the_visible_screen(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  # An empty replay has a screen to clear (leftover shell output).
  repl = _repl()
  paints = _capture_paints(repl, monkeypatch)

  repl._handle_message(_replay_done('conv-1'))
  await _settle(repl)

  assert len(paints) == 1
  assert paints[0][1] is _terminal_mod.TerminalClear.SCREEN
  assert not repl._state.session.replaying


def test_a_repaint_writes_history_beyond_the_viewport() -> None:
  # A SCREEN paint writes the history budget, not only the viewport.
  repl = _repl()
  written = io.StringIO()
  repl._terminal.console = _console.Console(
    file=written, force_terminal=True, width=80, height=5
  )
  history = '\n'.join(f'line {i}' for i in range(20)) + '\n'

  repl._terminal.print_ansi(history, clear=_terminal_mod.TerminalClear.SCREEN)

  frame = written.getvalue()
  assert frame.index('\x1b[2J') < frame.index('line 0')
  assert 'line 19' in frame
  assert '\x1b[3J' not in frame
  assert _terminal_mod.SYNC_BEGIN not in frame


def test_terminal_output_rejects_terminal_commands_in_rendered_text() -> None:
  repl = _repl()
  written = io.StringIO()
  repl._terminal.console = _console.Console(
    file=written, force_terminal=True, width=80
  )

  repl._terminal.print_ansi(
    'before\x1b[?1006;1000h\x1b[?1049h\x1b[>4;2mafter',
    clear=_terminal_mod.TerminalClear.SCREEN,
  )

  frame = written.getvalue()
  assert frame == '\x1b[2J\x1b[Hbeforeafter'


def test_a_repaint_clears_and_draws_in_one_terminal_write() -> None:
  class RecordingOutput(io.StringIO):
    def __init__(self) -> None:
      super().__init__()
      self.writes: list[str] = []

    def write(self, text: str) -> int:
      self.writes.append(text)
      return super().write(text)

  repl = _repl()
  written = RecordingOutput()
  repl._terminal.console = _console.Console(
    file=written, force_terminal=True, width=80
  )
  written.writes.clear()

  repl._terminal.print_ansi(
    'complete history\n', clear=_terminal_mod.TerminalClear.SCREEN
  )

  assert written.writes == ['\x1b[2J\x1b[Hcomplete history\n']


def test_a_scrollback_flush_keeps_the_new_history() -> None:
  # CSI 3J runs before the new history so overflow can stay in scrollback.
  repl = _repl()
  written = io.StringIO()
  repl._terminal.console = _console.Console(
    file=written, force_terminal=True, width=80, height=5
  )
  history = '\n'.join(f'line {i}' for i in range(20)) + '\n'

  repl._terminal.print_ansi(
    history, clear=_terminal_mod.TerminalClear.SCROLLBACK
  )

  frame = written.getvalue()
  assert frame.index('\x1b[2J') < frame.index('\x1b[3J') < frame.index('line 0')
  assert 'line 19' in frame
  assert _terminal_mod.SYNC_BEGIN not in frame


def test_flush_scrollback_promotes_screen_clears_only() -> None:
  repl = _repl()
  repl._terminal.flush_scrollback = True
  screen = _terminal_mod.TerminalClear.SCREEN
  keep = _terminal_mod.TerminalClear.KEEP
  scrollback = _terminal_mod.TerminalClear.SCROLLBACK

  assert repl._terminal.effective_clear(screen) is scrollback
  assert repl._terminal.effective_clear(keep) is keep
  assert repl._terminal.effective_clear(scrollback) is scrollback


def test_flush_scrollback_does_not_flush_on_a_live_emit() -> None:
  repl = _repl()
  repl._terminal.flush_scrollback = True
  written = io.StringIO()
  repl._terminal.console = _console.Console(
    file=written, force_terminal=True, width=80
  )

  repl._terminal.print_ansi('live\n', clear=_terminal_mod.TerminalClear.KEEP)

  assert '\x1b[3J' not in written.getvalue()
  assert '\x1b[2J' not in written.getvalue()


def test_cropped_replay_renders_entries_like_the_uncropped_path() -> None:
  repl = _repl()
  repl._terminal.console = _console.Console(width=80)
  repl._scrollback_limit = 50
  entry = _text.Text('entry', style='red')
  repl._terminal.pin_header(_text.Text('divider'))
  repl._terminal.trailer.append(entry)

  text = repl._render_history()

  # Cropping only chooses what to show: styling matches a normal emit, so a
  # replayed screen does not lose the colours the live transcript has.
  assert _rendering.renderable_to_ansi(entry, 80).rstrip() in text


@pytest.mark.asyncio
async def test_rendered_history_keeps_the_header_and_crops_old_entries(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._terminal.console = _console.Console(width=80)
  repl._scrollback_limit = 5
  paints = _capture_paints(repl, monkeypatch)
  repl._terminal.pin_header(_text.Text('=== divider ==='))
  _seed_history(repl, (f'entry {i}' for i in range(50)))
  await _settle(repl)

  text = paints[0][0]

  assert '=== divider ===' in text
  assert 'entry 49' in text
  assert 'entry 0' not in text
  assert 'earlier entries not shown' in text


@pytest.mark.asyncio
async def test_rendered_history_keeps_everything_under_the_limit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl._terminal.console = _console.Console(width=80)
  repl._scrollback_limit = 2000
  paints = _capture_paints(repl, monkeypatch)
  repl._terminal.pin_header(_text.Text('header'))
  _seed_history(repl, ['only entry'])
  await _settle(repl)

  text = paints[0][0]

  assert 'header' in text
  assert 'only entry' in text
  assert 'earlier entries not shown' not in text


def test_numbered_selector_renders_shortcuts_and_hint() -> None:
  text = _fragments_text(
    _selector.render_choices(
      _state.MenuState(active=True),
      (_selection.Choice('no', 'Disable'), _selection.Choice('yes', 'Enable')),
      title='Approval',
      numbered=True,
    )
  )
  assert '1. Disable' in text
  assert '2. Enable' in text
  assert '1–9 to select' in text


@pytest.mark.parametrize(
  ('status', 'expected'),
  [
    (None, ' › Memory  Remember context'),
    ('< On >', ' › Memory  < On >  Remember context'),
    ('     >', ' › Memory       >  Remember context'),
    ('', ' › Memory    Remember context'),
  ],
)
def test_render_choices_status_column_follows_name(
  status: str | None, expected: str
) -> None:
  rendered = _fragments_text(
    _selector.render_choices(
      _state.MenuState(active=True),
      (
        _selection.Choice(
          'memory', 'Memory', 'Remember context', status=status
        ),
      ),
      title='Settings',
    )
  )

  assert rendered.splitlines()[1] == expected


def test_selector_status_icons_align_and_details_have_distinct_styles() -> None:
  fragments = _selector.render_choices(
    _state.MenuState(active=True),
    (
      _selection.Choice(
        'a',
        'Clock',
        'Local time',
        prefix='●',
        prefix_style='class:extension-on',
      ),
      _selection.Choice(
        'b',
        'Timer',
        'Elapsed time',
        prefix='○',
        prefix_style='class:extension-off',
      ),
    ),
    title='Extensions',
  )
  lines = _fragments_text(fragments).splitlines()
  assert lines[1].index('Clock') == lines[2].index('Timer')
  assert ('class:menu-selected class:extension-on ', '●') in fragments
  assert ('class:menu-item class:extension-off ', '○') in fragments
  assert ('class:menu-detail', '  Elapsed time') in fragments


@pytest.mark.asyncio
async def test_extensions_selector_enter_toggles_live_widget(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, tmp_path / 'extensions')
  assert (
    await _user_extension.UserExtensionsTool(manager).execute(
      '{"action":"create","name":"clock"}'
    )
  ).success
  assert (
    await _user_extension.UserExtensionsTool(manager).execute(
      '{"action":"create","name":"alarm"}'
    )
  ).success
  repl = _repl(extension_registry=registry, extension_manager=manager)
  try:
    await repl._extension_host.start()
    await repl._run_slash('/extensions')
    repl.dispatch(_actions.SelectorSearchChanged('clo'))
    await asyncio.gather(*repl.dispatch(*repl._selector_choice()))
    assert 'clock' in repl._state.extensions
    selector = repl._state.selector.selector
    assert selector is not None
    assert [choice.label for choice in selector.choices] == ['alarm', 'clock']
    assert selector.choices[1].prefix == '●'
    assert repl._state.selector.query == 'clo'
    assert repl._state.selector.selected_choice() == selector.choices[1]
    await asyncio.gather(*repl.dispatch(*repl._selector_choice()))
    assert 'clock' not in repl._state.extensions
    selector = repl._state.selector.selector
    assert selector is not None
    assert [choice.label for choice in selector.choices] == ['alarm', 'clock']
    assert selector.choices[1].prefix == '○'
    assert repl._state.selector.query == 'clo'
    assert repl._state.selector.selected_choice() == selector.choices[1]
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('first_index', 'first_enabled'), [(0, False), (1, True)]
)
@pytest.mark.parametrize(
  ('second_index', 'second_enabled'), [(0, False), (1, True)]
)
async def test_extension_approval_advances_through_two_extensions(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  first_index: int,
  first_enabled: bool,
  second_index: int,
  second_enabled: bool,
) -> None:
  monkeypatch.setenv('MIRROR_EXTENSIONS_ENABLED', 'true')
  root = tmp_path / 'extensions'
  first = root / 'bug_report'
  second = root / 'usage_telemetry'
  first.mkdir(parents=True)
  second.mkdir()
  source = (
    'from mirror.cli import extension\n'
    'class Plugin(extension.Extension[None, None, None]):\n'
    '  @classmethod\n'
    '  def id(cls): return "NAME"\n'
  )
  (first / 'extension.py').write_text(source.replace('NAME', 'bug_report'))
  (second / 'extension.py').write_text(
    source.replace('NAME', 'usage_telemetry')
  )
  registry = extension.ExtensionRegistry()
  manager = user_extensions.UserExtensions(registry, root)
  await manager.start()
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry, extension_manager=manager)
    gc.collect()
    running = asyncio.create_task(repl._app.run_async())
    try:
      await repl._extension_host.start()
      await manager.on_foreground(repl._extension_context)
      await asyncio.sleep(0.05)
      request = repl._state.selector.selector
      assert request is not None
      assert str(first / 'extension.py') in request.title
      assert registry.extension_ids() == ()
      pipe.send_text(str(first_index + 1))
      await asyncio.sleep(0.05)
      request = repl._state.selector.selector
      assert request is not None
      assert str(second / 'extension.py') in request.title
      assert ('bug_report' in registry.extension_ids()) is first_enabled
      assert 'usage_telemetry' not in registry.extension_ids()
      pipe.send_text(str(second_index + 1))
      await asyncio.sleep(0.05)
      assert not repl._state.selector.active
      assert ('usage_telemetry' in registry.extension_ids()) is second_enabled
      assert json.loads((tmp_path / 'config.json').read_text())[
        'extensions'
      ] == {
        str(first / 'extension.py'): {'enabled': first_enabled},
        str(second / 'extension.py'): {'enabled': second_enabled},
      }
    finally:
      repl._app.exit()
      await asyncio.wait_for(running, timeout=2)
      await repl.close()
      await registry.close()


@pytest.mark.asyncio
async def test_async_selector_keeps_rows_and_fixed_height_while_searching() -> (
  None
):
  repl = _repl()
  choices = (
    _selection.Choice('alpha', 'alpha.py'),
    _selection.Choice('beta', 'beta.py'),
    _selection.Choice('gamma', 'gamma.py'),
  )
  repl.dispatch(
    _actions.OpenSelector(
      _selection.Selector(
        'files',
        'Files',
        choices,
        _actions.SetModel,
        prefiltered=True,
      )
    )
  )
  before = _selector.Selector().render(repl._state)

  repl.dispatch(_actions.SelectorSearchChanged('alpha'))
  pending = _selector.Selector().render(repl._state)
  assert repl._state.selector.filtered_choices() == choices
  assert repl._selector_choice() == ()
  assert repl._state.selector.active

  repl.dispatch(
    _actions.SelectorResultsLoaded('files', 'alpha', choices[:1], False)
  )
  matched = _selector.Selector().render(repl._state)
  assert repl._selector_choice() == (
    _actions.SelectorChoiceSelected('files', 'alpha'),
    _actions.SetModel('alpha'),
  )
  repl.dispatch(_actions.SelectorSearchChanged('absent'))
  repl.dispatch(_actions.SelectorResultsLoaded('files', 'absent', (), False))
  empty = _selector.Selector().render(repl._state)

  assert (
    _component_height(before),
    _component_height(pending),
    _component_height(matched),
    _component_height(empty),
  ) == (11, 11, 11, 11)
  assert len(_fragments_text(_component_text(before)).splitlines()) == 11
  assert len(_fragments_text(_component_text(pending)).splitlines()) == 11
  assert len(_fragments_text(_component_text(matched)).splitlines()) == 11
  assert len(_fragments_text(_component_text(empty)).splitlines()) == 11


@pytest.mark.asyncio
async def test_close_selector_dispatches_its_cancellation_action() -> None:
  repl = _repl()
  try:
    repl.dispatch(
      _actions.OpenSelector(
        _selection.Selector(
          'authentication',
          'Authenticating',
          (),
          lambda _: _actions.ClearSelector(),
          cancel_action=_actions.SetStatus('cancelled'),
        )
      )
    )
    await asyncio.gather(*repl.dispatch(_actions.CloseSelector()))
    assert not repl._state.selector.active
    assert repl._state.activity.status == 'cancelled'
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_selector_dispatches_the_selected_choice() -> None:
  registry = extension.ExtensionRegistry()
  await registry.load(
    pathlib.Path(cli.__file__).with_name('extensions') / 'inference_controls'
  )
  repl = _repl(extension_registry=registry)
  await repl._extension_host.start()

  try:
    await repl._run_slash('/reasoning')
    selector = repl._state.selector.selector
    assert selector is not None
    selected = repl._selector_choice()
    await asyncio.gather(*repl.dispatch(*selected))
  finally:
    await repl._extension_host.close()
    await registry.close()

  assert selected == (
    _actions.SelectorChoiceSelected('reasoning', 'medium'),
    _actions.SetReasoning('medium'),
  )
  assert repl._selector_choice() == (_actions.ClearSelector(),)
  assert not repl._state.selector.active


@pytest.mark.asyncio
async def test_an_unreadable_model_config_reports_instead_of_an_empty_picker(
  monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
  # A directory where the config belongs is the readable form of a bad file.
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(tmp_path))
  registry = extension.ExtensionRegistry()
  await registry.load(
    pathlib.Path(cli.__file__).with_name('extensions') / 'inference_controls'
  )
  repl = _repl(extension_registry=registry)
  emitted: list[str] = []

  async def emit(
    renderables: list[Any], *, replacing_history: bool = False
  ) -> None:
    del replacing_history
    emitted.extend(str(renderable) for renderable in renderables)

  monkeypatch.setattr(repl._terminal, 'emit', emit)
  await repl._extension_host.start()

  try:
    await repl._run_slash('/model')
  finally:
    await repl._extension_host.close()
    await registry.close()

  assert not emitted
  assert _render_welcome_notices(repl)
  assert not repl._state.selector.active


@pytest.mark.asyncio
async def test_mcp_tools_share_one_group_live_and_in_history() -> None:
  repl = _NoEmitRepl()
  repl.dispatch(_actions.SetReplaying(False), _actions.TurnStarted())
  repl._handle_message(_exploration_result('mcp_search'))
  repl._handle_message(_exploration_result('mcp_search'))
  repl._handle_message(_exploration_result('mcp_describe'))
  repl._handle_message(
    _delta({
      'role': 'assistant',
      'tool_calls': [
        {
          'id': 'mcp_call',
          'type': 'function',
          'function': {
            'name': 'mcp_call',
            'arguments': json.dumps({
              'calls': [
                {
                  'name': 'mcp__slack__send_message',
                  'arguments': {'message': 'hello'},
                }
              ],
            }),
          },
        }
      ],
    })
  )

  live = _ansi_text(_pending.render(repl._state))
  assert live.splitlines()[0].rstrip() == '· mcp · 2 search, 1 describe'
  assert "└ Call slack.send_message({'message': 'hello'})" in live
  assert 'Using mcp' in _fragments_text(_render_activity(repl))

  repl._handle_message(_exploration_result('mcp_call'))
  live = _ansi_text(_pending.render(repl._state))
  assert live.splitlines()[0].rstrip() == (
    '· mcp · 2 search, 1 describe, 1 call'
  )
  assert not repl._terminal.emit_buffer

  repl._handle_message(_turn_completed())
  emitted = _rendering.render_blocks(repl._terminal.emit_buffer, 80)
  history = repl._render_history()
  assert (
    _rendering.strip_ansi(emitted).strip()
    == _rendering.strip_ansi(history).strip()
    == '· mcp · 2 search, 1 describe, 1 call'
  )
  repl.dispatch(_actions.SetViewMode(_actions.ViewMode.TRANSCRIPT))
  await _settle(repl)
  transcript = _rendering.strip_ansi(repl._render_history())
  assert 'mcp_call(' in transcript
  assert "slack.send_message({'message': 'hello'})" in transcript


@pytest.mark.parametrize(
  ('arguments', 'expected'),
  [
    (
      '{"calls":[{"name":"mcp__slack__search","arguments":{}},'
      '{"name":"mcp__linear__search","arguments":{}}]}',
      'slack.search({}), linear.search({})',
    ),
    (
      '{"calls":[{"name":"mcp__slack__search","arguments":',
      'slack.search({})',
    ),
    ('{"calls":[null,{}, {"name":1}]}', ''),
  ],
)
def test_mcp_call_preview_handles_batches_and_partial_arguments(
  arguments: str, expected: str
) -> None:
  assert _rendering.compact_args(arguments, tool_name='mcp_call') == expected


def test_namespaced_tool_presentation_is_provider_neutral() -> None:
  name = 'extension__calendar__search'
  assert tool_groups.group_key(name) == 'extension:calendar'
  assert tool_groups.display_name(name) == 'search'
  assert tool_groups.status_label('extension:calendar') == 'Using calendar'
  group = tool_groups.result(name, success=True)
  assert group is not None
  assert group.summary().plain == '· calendar · 1 search'


class _PickerExtension(_StartupExtension):
  @override
  def slash_commands(self) -> tuple[_commands.CommandDef, ...]:
    return (
      _commands.CommandDef(
        'choose',
        'Choose an item',
        lambda context, args: _actions.OpenSelector(
          _selection.Selector(
            'test',
            'Choose',
            (),
            lambda _: _actions.ClearSelector(),
            extension_id=self.id(),
          )
        ),
      ),
    )


@pytest.mark.asyncio
async def test_extension_command_opens_searchable_picker() -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_PickerExtension(), None)
  repl = _repl(extension_registry=registry)
  try:
    await repl._run_slash('/choose')
    assert repl._state.selector.active
    assert repl._app.layout.current_window is repl._input_window
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
async def test_extension_command_available_with_remote_agent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_PickerExtension(), None)
  repl = _repl(
    remote_endpoint='remote.example:443', extension_registry=registry
  )
  emitted: list[str] = []

  async def emit(
    renderables: list[Any], *, replacing_history: bool = False
  ) -> None:
    del replacing_history
    emitted.extend(str(renderable) for renderable in renderables)

  monkeypatch.setattr(repl._terminal, 'emit', emit)

  await repl._run_slash('/choose')

  assert repl._state.selector.active
  assert not emitted
  assert 'Unknown command: /choose' not in _render_welcome_notices(repl)
  await repl.close()
  await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('scrollback_limit', [0, 1000])
async def test_worktree_activation_defers_older_history_until_redraw(
  monkeypatch: pytest.MonkeyPatch, scrollback_limit: int
) -> None:
  repl = _repl(active=False)
  repl._scrollback_limit = scrollback_limit
  _seed_history(repl, (f'entry {i}' for i in range(200)))
  paints = _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl._app, 'run_async', mock.AsyncMock(return_value=None))

  await repl.activate(show_context=True)

  assert 'entry 199' in paints[0][0]
  assert 'entry 0' not in paints[0][0]
  assert 'Ctrl+L to restore history' in paints[0][0]
  assert len(repl.transcript_entries()) == 200

  await repl._rerender()

  assert 'entry 0' in paints[-1][0]
  assert 'entry 199' in paints[-1][0]
  assert 'Ctrl+L to restore history' not in paints[-1][0]


@pytest.mark.asyncio
async def test_first_worktree_replay_uses_short_activation_paint(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl(active=False)
  repl.dispatch(_actions.WorkerStarted())
  paints = _capture_paints(repl, monkeypatch)

  async def run_async() -> str | None:
    _seed_history(repl, (f'entry {i}' for i in range(200)))
    await _settle(repl)
    return None

  monkeypatch.setattr(repl._app, 'run_async', run_async)

  await repl.activate(show_context=True)

  assert not paints[0][0]
  assert 'entry 199' in paints[-1][0]
  assert 'entry 0' not in paints[-1][0]
  assert 'Ctrl+L to restore history' in paints[-1][0]


@pytest.mark.asyncio
async def test_activation_respects_smaller_scrollback_limit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl(active=False)
  repl._scrollback_limit = 1
  _seed_history(repl, ['old entry', 'new entry'])
  paints = _capture_paints(repl, monkeypatch)
  monkeypatch.setattr(repl._app, 'run_async', mock.AsyncMock(return_value=None))

  await repl.activate(show_context=True)

  assert 'new entry' in paints[0][0]
  assert 'old entry' not in paints[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('api', 'provider'),
  [('reflection', 'Reflection'), ('anthropic-messages', 'Anthropic')],
)
@pytest.mark.parametrize('cancel', [False, True])
@pytest.mark.parametrize('via_picker', [False, True])
async def test_switch_provider_prompts_privately_and_preserves_draft(
  capfd: pytest.CaptureFixture[str],
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  cancel: bool,
  via_picker: bool,
  api: str,
  provider: str,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.delenv('ANTHROPIC_BASE_URL', raising=False)
  key_path = tmp_path / '.mirror' / f'{provider.lower()}_api_key'
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(model='previous-model')
    repl._worker_configuration = repl._worker_configuration.updated({
      'api': 'openai-responses',
    })
    repl._input.text = 'unfinished draft'
    previous = repl.model_selection
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    with _current.set_app(repl._app):
      switching = asyncio.create_task(
        _dispatch_and_wait(
          repl,
          _actions.ModelResolved(
            _models.ModelOption(_models.default_model(api), api, api=api)
          ),
        )
      )
    try:

      async def wait_for_prompt() -> None:
        while repl._secret_input is None:
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_prompt(), timeout=3)
      assert repl.model_selection == previous
      root = repl._app.layout.container
      assert isinstance(root, _containers.DynamicContainer)
      layout = root.get_container()
      assert isinstance(layout, _containers.HSplit)
      banner = layout.children[0]
      assert isinstance(banner, _containers.ConditionalContainer)
      assert not banner.filter()
      pipe.send_text('synthetic-switch-key')

      async def wait_for_key() -> None:
        while (
          repl._secret_input is not None
          and repl._secret_input.text != 'synthetic-switch-key'
        ):
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_key(), timeout=3)
      assert 'synthetic-switch-key' not in repr(repl._state)
      assert not repl.transcript_entries()
      if via_picker:
        pipe.send_text('\x1b')

        async def wait_for_picker() -> None:
          while repl._state.selector.id != 'model':
            await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_picker(), timeout=3)
        assert repl._state.selector.query == ''
        pipe.send_text('\x1b' if cancel else f'{provider}\r')
        if not cancel:
          await asyncio.wait_for(wait_for_prompt(), timeout=3)
          assert repl._secret_input is not None
          assert repl._secret_input.text == ''
          pipe.send_text('synthetic-switch-key\r')
      else:
        pipe.send_text('\x03' if cancel else '\r')
      await asyncio.wait_for(switching, timeout=3)
      assert repl._secret_input is None
      assert repl._input.text == 'unfinished draft'
      assert 'synthetic-switch-key' not in capfd.readouterr().out
      assert 'synthetic-switch-key' not in _render_welcome_notices(repl)
      assert banner.filter()
      assert not running.done()
      if cancel:
        assert repl.model_selection == previous
        assert not key_path.exists()
      else:
        assert repl.model_selection.api == api
        assert key_path.read_text() == 'synthetic-switch-key'
        saved_selection = _models.load_last_selection()
        assert saved_selection is not None
        assert saved_selection.api == api
    finally:
      repl._app.exit()
      switching.cancel()
      await asyncio.gather(switching, running, return_exceptions=True)
      await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('environment', [False, True])
@pytest.mark.parametrize('saved', [False, True])
async def test_logout_reports_saved_key_and_environment_separately(
  capfd: pytest.CaptureFixture[str],
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  environment: bool,
  saved: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  if environment:
    monkeypatch.setenv('REFLECTION_API_KEY', 'synthetic-env-key')
  if saved:
    cli._reflection.save_key('synthetic-saved-key')
  repl = _repl()
  repl._worker_configuration = repl._worker_configuration.updated({
    'api': 'reflection' if environment else 'openai-responses',
  })
  await _dispatch_and_wait(repl, _actions.ForgetReflectionKey())
  await _settle(repl)
  text = capfd.readouterr().out + _render_welcome_notices(repl)
  assert ('Removed the saved' if saved else 'No saved') in text
  assert ('REFLECTION_API_KEY is still set' in text) is environment
  assert 'synthetic-env-key' not in text
  assert 'synthetic-saved-key' not in text
  assert not cli._reflection.key_path().exists()
  assert repl._secret_input is None


@pytest.mark.asyncio
@pytest.mark.parametrize('environment', [False, True])
async def test_switch_to_reflection_reuses_available_credentials(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  environment: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  if environment:
    monkeypatch.setenv('REFLECTION_API_KEY', 'synthetic-env-key')
  else:
    cli._reflection.save_key('synthetic-saved-key')
  repl = _repl(model='previous-model')

  await _dispatch_and_wait(
    repl,
    _actions.ModelResolved(
      _models.ModelOption('Beam-501B-A23B', 'Reflection', api='reflection')
    ),
  )

  assert repl.model_selection.api == 'reflection'
  assert repl._secret_input is None


@pytest.mark.asyncio
async def test_remote_model_switch_needs_no_local_sign_in(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  repl = _repl(model='previous-model', remote_endpoint='remote.example:443')

  await _dispatch_and_wait(
    repl, _actions.SetRemoteModel('Beam-501B-A23B', 'reflection')
  )
  await _dispatch_and_wait(repl, _actions.SetRemoteModel('Beam-502B', ''))

  assert repl._secret_input is None
  assert repl.model_selection == _models.ModelOption(
    'Beam-502B', 'Beam-502B', api='reflection'
  )


@pytest.mark.asyncio
async def test_initial_sign_in_can_choose_another_provider_before_worker_starts(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  capfd: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  worker = _ECHO_WORKER_SOURCE.replace(
    'config = json.loads(sys.argv[1])',
    "config = json.loads(sys.argv[1])\n  assert config['model_api'] == 'openai-codex-responses'",
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(worker_source=worker, model='Beam-501B-A23B')
    repl._worker_configuration = repl._worker_configuration.updated({
      'api': 'reflection'
    })
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()

    async def prepare() -> None:
      assert await repl.ensure_api_key()

    with _current.set_app(repl._app):
      repl._extensions_loaded = asyncio.create_task(prepare())
      starting = asyncio.create_task(repl.start())
    try:

      async def wait_for_prompt() -> None:
        while repl._secret_input is None:
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_prompt(), timeout=3)
      secret_buffer = repl._secret_input
      assert secret_buffer is not None
      pipe.send_text('synthetic-discarded-key')

      async def wait_for_key() -> None:
        while secret_buffer.text != 'synthetic-discarded-key':
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_key(), timeout=3)
      pipe.send_text('\x1b')

      async def wait_for_picker() -> None:
        while repl._state.selector.id != 'model':
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_picker(), timeout=3)
      assert secret_buffer.text == ''
      assert repl._state.selector.query == ''
      assert not cli._reflection.key_path().exists()
      assert repl._client is None
      pipe.send_text('ChatGPT')

      async def wait_for_search() -> None:
        while (
          repl._state.selector.query != 'ChatGPT'
          or repl._state.selector.search_pending
        ):
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_search(), timeout=3)
      pipe.send_text('\r')

      async def wait_for_models() -> None:
        while repl._state.selector.id != 'codex-models':
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_models(), timeout=3)
      pipe.send_text('\r')
      await asyncio.wait_for(starting, timeout=5)
      assert repl.model_selection.api == 'openai-codex-responses'
      assert repl._client is not None
      assert not cli._reflection.key_path().exists()
      assert not repl._state.selector.active
      assert repl._secret_input is None
      assert 'synthetic-discarded-key' not in repr(repl._state)
      assert 'synthetic-discarded-key' not in capfd.readouterr().out
    finally:
      repl._app.exit()
      starting.cancel()
      repl._extensions_loaded.cancel()
      await asyncio.gather(
        starting, repl._extensions_loaded, running, return_exceptions=True
      )
      await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('pick_reflection', [False, True])
async def test_initial_picker_returns_to_sign_in_without_a_usable_model(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  pick_reflection: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(model='Beam-501B-A23B')
    repl._worker_configuration = repl._worker_configuration.updated({
      'api': 'reflection'
    })
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    with _current.set_app(repl._app):
      onboarding = asyncio.create_task(repl.ensure_api_key())
    try:

      async def wait_for_prompt() -> None:
        while repl._secret_input is None:
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_prompt(), timeout=3)
      pipe.send_text('\x1b')

      async def wait_for_picker() -> None:
        while repl._state.selector.id != 'model':
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_picker(), timeout=3)
      pipe.send_text('\r' if pick_reflection else '\x1b')
      await asyncio.wait_for(wait_for_prompt(), timeout=3)
      assert not onboarding.done()
      assert repl._secret_input is not None
      assert repl._secret_input.text == ''
      pipe.send_text('synthetic-key\r')
      assert await asyncio.wait_for(onboarding, timeout=3)
      assert repl.model_selection.api == 'reflection'
    finally:
      repl._app.exit()
      onboarding.cancel()
      await asyncio.gather(onboarding, running, return_exceptions=True)
      await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('saved', [False, True])
async def test_logout_reopens_sign_in_for_the_active_reflection_model(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  saved: bool,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.delenv('REFLECTION_API_KEY', raising=False)
  if saved:
    cli._reflection.save_key('synthetic-old-key')
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(model='Beam-501B-A23B')
    repl._worker_configuration = repl._worker_configuration.updated({
      'api': 'reflection'
    })
    repl._input.text = 'unfinished draft'
    previous = repl.model_selection
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    with _current.set_app(repl._app):
      logout = asyncio.create_task(
        _dispatch_and_wait(repl, _actions.ForgetReflectionKey())
      )
    try:

      async def wait_for_prompt() -> None:
        while repl._secret_input is None:
          await asyncio.sleep(0)

      await asyncio.wait_for(wait_for_prompt(), timeout=3)
      assert not cli._reflection.key_path().exists()
      assert not logout.done()
      assert repl.model_selection == previous
      pipe.send_text('synthetic-new-key\r')
      await asyncio.wait_for(logout, timeout=3)
      assert cli._reflection.key_path().read_text() == 'synthetic-new-key'
      assert repl._secret_input is None
      assert repl._input.text == 'unfinished draft'
      assert repl.model_selection == previous
      assert not running.done()
    finally:
      repl._app.exit()
      logout.cancel()
      await asyncio.gather(logout, running, return_exceptions=True)
      await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('columns', [45, 100])
@pytest.mark.parametrize('role', ['user', 'assistant'])
async def test_live_banner_updates_through_notices_and_disappears_for_conversation(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  columns: int,
  role: str,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  terminal = io.StringIO()
  output = _vt100.Vt100_Output(
    terminal,
    lambda: _data_structures.Size(rows=30, columns=columns),
    term='xterm-256color',
    enable_cpr=False,
  )
  screen = pyte.Screen(columns, 30)
  screen.set_mode(pyte.modes.LNM)
  stream = pyte.Stream(screen)
  offset = 0

  async def wait_for_screen(predicate: Callable[[str], bool]) -> str:
    nonlocal offset
    while True:
      content = terminal.getvalue()
      stream.feed(content[offset:])
      offset = len(content)
      displayed = '\n'.join(screen.display)
      if predicate(displayed):
        return displayed
      await asyncio.sleep(0)

  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=output),
  ):
    repl = _repl(model='old-model')
    repl._terminal.console = _console.Console(
      file=terminal, width=columns, force_terminal=True
    )
    ready = asyncio.Event()
    running = asyncio.create_task(repl._app.run_async(pre_run=ready.set))
    await ready.wait()
    try:
      await asyncio.wait_for(
        wait_for_screen(lambda text: 'model      old-model' in text), timeout=3
      )
      repl._handle_message(_replay_done('conv-1'))
      with _current.set_app(repl._app):
        await _dispatch_and_wait(
          repl, _actions.ShowText('Beta mode is enabled.')
        )
        await repl._switch_model(
          _models.ModelOption('new-model', 'New', api='openai-responses')
        )
      displayed = await asyncio.wait_for(
        wait_for_screen(lambda text: 'model      new-model' in text), timeout=3
      )
      assert 'old-model' not in displayed
      assert displayed.count('Reflection mirror') == 1
      assert displayed.count('Beta mode is enabled.') == 1
      assert displayed.index('Reflection mirror') < displayed.index(
        'Beta mode is enabled.'
      )
      assert not repl._transcript.has_conversation
      await repl._rerender()
      displayed = await asyncio.wait_for(
        wait_for_screen(lambda text: 'model      new-model' in text), timeout=3
      )
      assert displayed.count('Beta mode is enabled.') == 1
      assert displayed.index('Reflection mirror') < displayed.index(
        'Beta mode is enabled.'
      )

      repl._handle_message(_delta({'role': role, 'content': 'hello world'}))
      displayed = await asyncio.wait_for(
        wait_for_screen(
          lambda text: 'hello world' in text and 'Reflection mirror' not in text
        ),
        timeout=3,
      )
      assert 'model      ' not in displayed
      assert displayed.count('Beta mode is enabled.') == 1
      assert displayed.index('Beta mode is enabled.') < displayed.index(
        'hello world'
      )
      assert repl._transcript.has_conversation
      await repl._rerender()
      displayed = await asyncio.wait_for(
        wait_for_screen(lambda text: 'hello world' in text), timeout=3
      )
      assert displayed.count('Beta mode is enabled.') == 1
      assert 'Reflection mirror' not in displayed
    finally:
      repl._app.exit()
      await running
      await repl.close()


class _SecretExtension(extension.Extension[None, str, None]):
  def __init__(self) -> None:
    self.received = asyncio.Queue[str]()

  @classmethod
  @override
  def id(cls) -> str:
    return 'test/secret'

  @override
  async def handle_effect(
    self, context: extension.Context, effect: str
  ) -> None:
    self.received.put_nowait(effect)


@pytest.mark.asyncio
async def test_secret_prompt_masks_input_and_hands_it_to_its_extension() -> (
  None
):
  registry = extension.ExtensionRegistry()
  plugin = _SecretExtension()
  await registry.register(plugin, None)
  observed: dict[str, object] = {}
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(extension_registry=registry)

    async def enter_secret() -> None:
      try:
        await asyncio.sleep(0.05)
        repl.dispatch(
          _actions.RequestSecret(
            _actions.SecretRequest(
              'token', 'Paste a token', plugin.id(), 'received {}'.format
            )
          )
        )
        pipe.send_text('\x1b[200~ghp_secret\x1b[201~')
        await asyncio.sleep(0.05)
        line = repl._extension_secret_input.control.create_content(
          80, 1
        ).get_line(0)
        observed['rendered'] = ''.join(fragment[1] for fragment in line)
        observed['length'] = repl._state.secret.length
        observed['composer'] = repl._input.text
        pipe.send_text('\r')
        observed['received'] = await asyncio.wait_for(plugin.received.get(), 2)
      finally:
        repl._app.exit()

    repl._app.pre_run_callables.append(
      lambda: repl._app.create_background_task(enter_secret())
    )
    try:
      await asyncio.wait_for(repl.activate(show_context=False), timeout=5)
    finally:
      await repl.close()
      await registry.close()
  assert observed == {
    'rendered': f'› {"•" * 10} ',
    'length': 10,
    'composer': '',
    'received': 'received ghp_secret',
  }
  assert not repl._state.secret.active
  assert not repl._extension_secret_input.buffer.text
  assert 'ghp_secret' not in repr(repl._state)


@pytest.mark.asyncio
async def test_skill_only_extension_refreshes_worker_and_completion(
  tmp_path: pathlib.Path,
) -> None:
  directory = tmp_path / 'helper'
  directory.mkdir()
  (directory / 'extension.py').write_text(
    'import pathlib\n'
    'from mirror.cli import extension\n'
    'class Helper(extension.Extension[None, None, None]):\n'
    '  @classmethod\n'
    '  def id(cls): return "helper"\n'
    '  def skills(self):\n'
    '    return (pathlib.Path(__file__).with_name("SKILL.md"),)\n'
  )
  path = directory / 'SKILL.md'
  path.write_text(
    '---\nname: skill-only\ndescription: Local helper.\n---\nOriginal'
  )
  capture = tmp_path / 'worker-config.json'
  worker = _ECHO_WORKER_SOURCE.replace(
    'config = json.loads(sys.argv[1])',
    'config = json.loads(sys.argv[1])\n'
    f'  with open({str(capture)!r}, "w") as f: json.dump(config, f)',
  )
  registry = extension.ExtensionRegistry()
  repl = _repl(worker_source=worker, extension_registry=registry)
  try:
    await repl.start()
    assert json.loads(capture.read_text())['additional_skills'] == []
    await registry.load(directory)
    await repl._host_tools_registry.wait_for_update()
    assert (
      repl._skill_completion.matches('$skill-only', len('$skill-only'))[0].body
      == 'Original'
    )
    assert json.loads(capture.read_text())['additional_skills'] == [
      {
        'name': 'skill-only',
        'description': 'Local helper.',
        'body': 'Original',
        'path': str(path.resolve()),
        'disable_model_invocation': False,
      }
    ]

    repl.dispatch(_actions.TurnStarted())
    previous_client = repl._client
    path.write_text(
      '---\nname: skill-only\ndescription: Local helper.\n---\nUpdated'
    )
    await registry.reload('helper')
    await asyncio.sleep(0)
    assert repl._client is previous_client
    repl.dispatch(_actions.TurnFinished())
    await repl._host_tools_registry.wait_for_update()
    assert (
      json.loads(capture.read_text())['additional_skills'][0]['body']
      == 'Updated'
    )

    await registry.unload('helper')
    await repl._host_tools_registry.wait_for_update()
    assert (
      repl._skill_completion.matches('$skill-only', len('$skill-only')) == ()
    )
    assert json.loads(capture.read_text())['additional_skills'] == []
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
async def test_project_skill_overrides_extension_in_completion(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.chdir(tmp_path)
  directory = tmp_path / '.agents' / 'skills' / 'mirror-extensions'
  directory.mkdir(parents=True)
  path = directory / 'SKILL.md'
  path.write_text(
    '---\ndescription: Project authoring guide.\n---\nProject instructions.'
  )
  registry = extension.ExtensionRegistry()
  await registry.register(
    _user_extension.UserExtensions(), None, overrides={'enabled': True}
  )
  repl = _repl(extension_registry=registry)
  try:
    matches = repl._skill_completion.matches(
      '$mirror-extensions', len('$mirror-extensions')
    )
    assert len(matches) == 1
    assert matches[0].path == path
    assert matches[0].body == 'Project instructions.'
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'name,arguments',
  [
    ('bash', '{"command":"ls; pwd"}'),
    ('write', '{"file_path":"a.txt","content":"text"}'),
    ('mcp__calendar__create_event', '{"date":"2030-01-01","title":"test"}'),
  ],
)
async def test_remembered_approval_skips_prompt_after_restart(
  tmp_path: pathlib.Path,
  name: str,
  arguments: str,
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(_approval_requested('first', name, arguments))
  assert '2. Approve and remember' in '\n'.join(_render_approval(repl))
  assert _actions.AcceptSelection(1) in {
    binding.action
    for binding in _keymap.defaults(
      repl._state,
      completion_visible=False,
      modal_active=False,
      can_focus_tasks=False,
    )
    if binding.keys == ('2',)
  }
  repl.dispatch(_actions.AcceptSelection(1))
  await _settle(repl)
  assert sent[-1].hitl_response.approved
  assert sent[-1].hitl_response.source == agent_pb2.APPROVAL_SOURCE_MANUAL

  restarted = _repl(approvals=tool_approvals.ToolApprovals(path))
  restarted._send_request = sent.append
  restarted._handle_message(_approval_requested('repeat', name, arguments))
  assert restarted._state.approval.request is None
  assert sent[-1].hitl_response.call_id == 'repeat'
  assert (
    sent[-1].hitl_response.source == agent_pb2.APPROVAL_SOURCE_REMEMBERED_EXACT
  )
  assert sent[-1].hitl_response.approved
  restarted._handle_message(
    _approval_requested('different', name, arguments + ' ')
  )
  assert restarted._state.approval.request is not None
  assert restarted._state.approval.request.call_id == 'different'


@pytest.mark.asyncio
@pytest.mark.parametrize('approved', [None, 0, 2])
async def test_single_approval_and_rejection_are_not_remembered(
  tmp_path: pathlib.Path, approved: int | None
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(_approval_requested('first', 'bash', '{"command":"ls"}'))
  repl.dispatch(_actions.AcceptSelection(approved))
  await _settle(repl)
  assert sent[-1].hitl_response.approved is (approved != 2)
  assert sent[-1].hitl_response.source == agent_pb2.APPROVAL_SOURCE_MANUAL
  assert not path.exists()
  repl._handle_message(
    _approval_requested('repeat', 'bash', '{"command":"ls"}')
  )
  assert repl._state.approval.request is not None


@pytest.mark.asyncio
async def test_failed_remember_approves_once_with_notice(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(_approval_requested('first', 'bash', '{"command":"ls"}'))
  path.mkdir()
  repl.dispatch(_actions.AcceptSelection(1))
  repl.dispatch(_actions.AcceptSelection(0))
  await _settle(repl)
  assert sent[-1].hitl_response.approved
  assert 'Could not save tool approval' in _render_welcome_notices(repl)
  repl._handle_message(
    _approval_requested('repeat', 'bash', '{"command":"ls"}')
  )
  assert repl._state.approval.request is not None


def test_corrupt_approvals_prompt_with_notice(tmp_path: pathlib.Path) -> None:
  path = tmp_path / 'approvals.json'
  path.write_text('{')
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  repl._handle_message(_approval_requested('call', 'bash', '{"command":"ls"}'))
  assert repl._state.approval.request is not None
  assert repl._state.approval.request.remember is None
  assert 'Remembered tool approvals are unavailable' in _render_welcome_notices(
    repl
  )


def test_worker_resolution_does_not_remember_command(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  repl._handle_message(_approval_requested('call', 'bash', '{"command":"ls"}'))
  repl._handle_message(
    agent_pb2.SessionMessage(
      hitl_resolved=agent_pb2.HITLResolved(call_id='call', approved=True)
    )
  )
  assert repl._state.approval.request is None
  assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'endpoint,session,secure,remembered',
  [
    ('first:1234', 'remote-session', True, True),
    ('second:1234', 'remote-session', True, False),
    ('first:1234', 'other-session', True, False),
    ('first:1234', 'remote-session', False, False),
    (None, 'remote-session', True, False),
  ],
)
async def test_remote_approval_is_scoped_to_endpoint_session_and_transport(
  tmp_path: pathlib.Path,
  endpoint: str | None,
  session: str,
  secure: bool,
  remembered: bool,
) -> None:
  approvals = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  original = _repl(
    approvals=approvals,
    remote_endpoint='first:1234',
    conversation_id='remote-session',
  )
  sent: list[agent_pb2.ClientMessage] = []
  original._send_request = sent.append
  original._handle_message(
    _approval_requested('first', 'write', '{"path":"test"}')
  )
  original.dispatch(_actions.AcceptSelection(1))
  await _settle(original)
  assert sent[-1].hitl_response.approved
  sent.clear()

  other = _repl(
    approvals=approvals,
    remote_endpoint=endpoint,
    conversation_id=session,
    remote_secure=secure,
  )
  other._send_request = sent.append
  other._handle_message(_approval_requested('next', 'write', '{"path":"test"}'))
  assert (other._state.approval.request is None) is remembered
  assert bool(sent) is remembered


@pytest.mark.asyncio
async def test_local_approval_does_not_cross_workspace_override(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  approvals = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  monkeypatch.setenv('MIRROR_WORKSPACE', '/first')
  original = _repl(approvals=approvals)
  sent: list[agent_pb2.ClientMessage] = []
  original._send_request = sent.append
  original._handle_message(
    _approval_requested('first', 'bash', '{"command":"ls"}')
  )
  original.dispatch(_actions.AcceptSelection(1))
  await _settle(original)
  assert sent[-1].hitl_response.approved

  monkeypatch.setenv('MIRROR_WORKSPACE', '/second')
  other = _repl(approvals=approvals)
  other._handle_message(_approval_requested('next', 'bash', '{"command":"ls"}'))
  assert other._state.approval.request is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('key', ['\x1b', '\x03'])
@pytest.mark.parametrize('command', ['ls', 'git status'])
async def test_approval_cancel_interrupts_remote_turn_without_reject_and_retry(
  tmp_path: pathlib.Path, key: str, command: str
) -> None:
  path = tmp_path / 'approvals.json'
  remote = _RemoteEcho()
  server = grpc.aio.server()
  agent_pb2_grpc.add_AgentRPCServicer_to_server(remote, server)
  port = server.add_insecure_port('127.0.0.1:0')
  await server.start()
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(
      approvals=tool_approvals.ToolApprovals(path),
      remote_endpoint=f'127.0.0.1:{port}',
      remote_secure=False,
      conversation_id='focused-session',
    )
    repl._client = _rpc_client.Client(
      _remote_rpc_client.Transport(f'127.0.0.1:{port}', secure=False),
      'focused-session',
      on_message=repl._handle_message,
      on_error=repl._exit_with_error,
    )
    await repl._client.start(replay=False)
    repl._input.text = 'unfinished draft'
    repl.dispatch(_actions.TurnStarted())
    repl._handle_message(
      _approval_requested('first', 'bash', json.dumps({'command': command}))
    )
    repl._handle_message(_approval_requested('second', 'write'))
    repl.dispatch(_actions.MoveSelection(1))
    running = asyncio.create_task(
      repl._app.run_async(pre_run=lambda: pipe.send_text(key))
    )
    try:
      request = await asyncio.wait_for(remote.requests.get(), timeout=2)
      assert request.WhichOneof('payload') == 'interrupt'
      assert request.session_id == 'focused-session'
      assert request.message_id
      assert remote.requests.empty()
      assert not running.done()
      assert repl._input.text == 'unfinished draft'
      assert not path.exists()
      # Only the worker resolves approvals; no reject can resume the model first.
      assert len(repl._state.approval.requests) == 2
      repl._handle_message(
        agent_pb2.SessionMessage(
          hitl_resolved=agent_pb2.HITLResolved(call_id='first', approved=False)
        )
      )
      repl._handle_message(
        agent_pb2.SessionMessage(
          hitl_resolved=agent_pb2.HITLResolved(call_id='second', approved=False)
        )
      )
      repl.dispatch(_actions.TurnFinished())
      assert repl._state.approval.request is None
      assert not repl._state.selector.active
      assert not running.done()
      assert repl._input.text == 'unfinished draft'
    finally:
      repl._app.exit()
      await asyncio.wait_for(running, timeout=2)
      await repl.close()
      await server.stop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'action', [_actions.CloseSelector(), _actions.CancelOrQuitRequested()]
)
async def test_cancel_dismisses_other_selector_before_pending_approval(
  action: _actions.Action,
) -> None:
  repl = _repl()
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._input.text = 'unfinished draft'
  repl.dispatch(
    _actions.OpenSelector(
      _selection.Selector(
        'other', 'Other menu', (), lambda _: _actions.ClearSelector()
      )
    )
  )
  repl._handle_message(_approval_requested('call', 'write'))
  repl.dispatch(action)
  await _settle(repl)
  assert not sent
  assert repl._input.text == 'unfinished draft'
  assert repl._state.selector.id == 'approval:call'
  await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'key,approved,remembered',
  [
    ('1', True, False),
    ('2', True, True),
    ('3', False, False),
  ],
)
async def test_approval_uses_shared_selector_shortcuts(
  tmp_path: pathlib.Path,
  key: str,
  approved: bool,
  remembered: bool,
) -> None:
  path = tmp_path / 'approvals.json'
  decision: asyncio.Future[agent_pb2.ClientMessage] = (
    asyncio.get_running_loop().create_future()
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(approvals=tool_approvals.ToolApprovals(path))
    repl._send_request = decision.set_result
    repl._input.text = 'unfinished draft'
    repl._handle_message(
      _approval_requested('call', 'write', '{"path":"test"}')
    )
    assert [
      choice.value for choice in repl._state.selector.filtered_choices()
    ] == [
      'approve',
      'remember',
      'reject',
    ]
    running = asyncio.create_task(
      repl._app.run_async(pre_run=lambda: pipe.send_text(key))
    )
    try:
      response = await asyncio.wait_for(decision, timeout=2)
      assert response.hitl_response.approved is approved
      assert path.exists() is remembered
      assert not repl._state.selector.active
      assert repl._input.text == 'unfinished draft'
    finally:
      repl._app.exit()
      await asyncio.wait_for(running, timeout=2)
      await repl.close()


@pytest.mark.asyncio
async def test_selected_prefix_approves_varying_literal_arguments_after_restart(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl.dispatch(_actions.TurnStarted())
  repl._handle_message(
    _approval_requested('first', 'bash', '{"command":"git status"}')
  )
  assert not _render_activity(repl)
  assert repl._state.activity.work_paused is not None
  assert [
    choice.label for choice in repl._state.selector.filtered_choices()
  ] == [
    'Approve',
    'Approve and always allow commands starting with `git status`',
    'Reject',
  ]
  assert not path.exists()
  repl.dispatch(_actions.AcceptSelection(1))
  await _settle(repl)
  assert len(sent) == 1
  assert sent[0].hitl_response.approved
  assert sent[0].hitl_response.source == agent_pb2.APPROVAL_SOURCE_MANUAL
  assert not repl._state.selector.active
  assert _render_activity(repl)
  assert repl._state.activity.work_paused is None
  assert json.loads(path.read_text())['entries'][0]['argv'] == ['git', 'status']

  restarted = _repl(approvals=tool_approvals.ToolApprovals(path))
  restarted._send_request = sent.append
  restarted._handle_message(
    _approval_requested(
      'next', 'bash', '{"command":"git \'status\' --porcelain"}'
    )
  )
  assert restarted._state.approval.request is None
  assert sent[-1].hitl_response.call_id == 'next'
  assert (
    sent[-1].hitl_response.source == agent_pb2.APPROVAL_SOURCE_REMEMBERED_PREFIX
  )
  assert sent[-1].hitl_response.approved
  restarted._handle_message(
    _approval_requested('boundary', 'bash', '{"command":"git status-other"}')
  )
  assert restarted._state.approval.request is not None
  assert restarted._state.approval.request.call_id == 'boundary'


@pytest.mark.asyncio
async def test_existing_exact_approval_is_not_widened(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  repl = _repl(approvals=store)
  exact = tool_approvals.from_call(
    'bash', '{"command":"git status"}', repl._local_approval_scope
  )
  assert exact is not None
  store.remember(exact)
  content = path.read_text()
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(
    _approval_requested('first', 'bash', '{"command":"git status"}')
  )
  assert sent[-1].hitl_response.approved
  assert repl._state.approval.request is None
  repl._handle_message(
    _approval_requested('next', 'bash', '{"command":"git status --short"}')
  )
  assert repl._state.approval.request is not None
  assert len(sent) == 1
  assert path.read_text() == content


@pytest.mark.parametrize(
  'arguments',
  [
    '{"command":"git status; echo other"}',
    '{"command":"git status > output"}',
    '{"command":"git status $(id)"}',
    '{"command":"git status","cwd":"/other"}',
  ],
)
def test_prefix_grant_does_not_allow_unsupported_shell_calls(
  tmp_path: pathlib.Path, arguments: str
) -> None:
  store = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  repl = _repl(approvals=store)
  store.remember(
    tool_approvals.ArgvPrefixApproval(
      repl._local_approval_scope, 'bash', ('git', 'status')
    )
  )
  repl._handle_message(_approval_requested('call', 'bash', arguments))
  assert repl._state.approval.request is not None
  assert isinstance(
    repl._state.approval.request.remember, tool_approvals.ToolApproval
  )
  assert (
    repl._state.selector.filtered_choices()[1].label
    == 'Approve and remember this exact call'
  )
  assert [
    choice.value for choice in repl._state.selector.filtered_choices()
  ] == ['approve', 'remember', 'reject']


def test_repeated_approval_request_is_asked_once(
  tmp_path: pathlib.Path,
) -> None:
  repl = _repl(
    approvals=tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  )
  request = _approval_requested('call', 'bash', '{"command":"ls"}')
  repl._handle_message(request)
  repl._handle_message(request)
  assert [queued.call_id for queued in repl._state.approval.requests] == [
    'call'
  ]


def test_remote_shell_keeps_exact_matching(tmp_path: pathlib.Path) -> None:
  store = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  repl = _repl(
    approvals=store, remote_endpoint='remote:1234', conversation_id='session'
  )
  scope = json.dumps(('remote-v1', 'remote:1234', True, 'session'))
  store.remember(tool_approvals.ArgvPrefixApproval(scope, 'bash', ('cat',)))
  repl._handle_message(
    _approval_requested('call', 'bash', '{"command":"cat first.txt"}')
  )
  assert repl._state.approval.request is not None
  assert isinstance(
    repl._state.approval.request.remember, tool_approvals.ToolApproval
  )
  assert (
    repl._state.selector.filtered_choices()[1].label
    == 'Approve and remember this exact call'
  )

  assert [
    choice.value for choice in repl._state.selector.filtered_choices()
  ] == [
    'approve',
    'remember',
    'reject',
  ]


@pytest.mark.asyncio
async def test_stale_prefix_choice_does_not_approve_next_call(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(
    _approval_requested('first', 'bash', '{"command":"git status"}')
  )
  repl._handle_message(
    _approval_requested('next', 'bash', '{"command":"git status --short"}')
  )
  selector = repl._state.selector.selector
  assert selector is not None
  repl._handle_message(
    agent_pb2.SessionMessage(
      hitl_resolved=agent_pb2.HITLResolved(call_id='first', approved=True)
    )
  )
  repl.dispatch(selector.on_select('remember'))
  await _settle(repl)
  assert repl._state.approval.request is not None
  assert repl._state.approval.request.call_id == 'next'
  assert not sent
  assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('keys', ['2', '\x1b[B\r'])
async def test_prefix_approval_requires_only_one_selection(
  tmp_path: pathlib.Path, keys: str
) -> None:
  path = tmp_path / 'approvals.json'
  decision: asyncio.Future[agent_pb2.ClientMessage] = (
    asyncio.get_running_loop().create_future()
  )
  with (
    _input_defaults.create_pipe_input() as pipe,
    _current.create_app_session(input=pipe, output=_output_base.DummyOutput()),
  ):
    repl = _repl(approvals=tool_approvals.ToolApprovals(path))
    repl._send_request = decision.set_result
    repl._input.text = 'unfinished draft'
    repl._handle_message(
      _approval_requested('call', 'bash', '{"command":"git status --short"}')
    )
    running = asyncio.create_task(
      repl._app.run_async(pre_run=lambda: pipe.send_text(keys))
    )
    try:
      response = await asyncio.wait_for(decision, timeout=2)
      assert response.hitl_response.approved
      entry = json.loads(path.read_text())['entries'][0]
      assert entry['kind'] == 'argv_prefix'
      assert entry['argv'] == ['git', 'status', '--short']
      assert not repl._state.selector.active
      assert repl._input.text == 'unfinished draft'
    finally:
      repl._app.exit()
      await asyncio.wait_for(running, timeout=2)
      await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'index,remember,argv',
  [
    (0, True, ['git', 'status', '--short']),
    (1, False, ['git', 'status', '--short']),
  ],
)
async def test_prefix_approval_supports_configured_approval_actions(
  tmp_path: pathlib.Path,
  index: int,
  remember: bool,
  argv: list[str],
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(
    _approval_requested('call', 'bash', '{"command":"git status --short"}')
  )
  repl.dispatch(
    _actions.MoveApproval(index), _actions.DecideApproval(remember=remember)
  )
  await _settle(repl)
  assert sent[-1].hitl_response.approved
  assert not repl._state.selector.active
  assert json.loads(path.read_text())['entries'][0]['argv'] == argv


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'command,argv,label',
  [
    ('ls -l', ['ls', '-l'], 'ls -l'),
    ('git status --short', ['git', 'status', '--short'], 'git status --short'),
    ("echo 'two words'", ['echo', 'two words'], "echo 'two words'"),
  ],
)
async def test_remember_saves_only_the_displayed_full_prefix(
  tmp_path: pathlib.Path, command: str, argv: list[str], label: str
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(
    _approval_requested('call', 'bash', json.dumps({'command': command}))
  )
  assert repl._state.selector.filtered_choices()[1].label == (
    f'Approve and always allow commands starting with `{label}`'
  )
  repl.dispatch(_actions.AcceptSelection(1))
  await _settle(repl)
  assert sent[-1].hitl_response.approved
  assert json.loads(path.read_text())['entries'] == [
    {
      'kind': 'argv_prefix',
      'scope': repl._local_approval_scope,
      'tool': 'bash',
      'argv': argv,
    }
  ]
  repl._handle_message(
    _approval_requested('other', 'bash', json.dumps({'command': argv[0]}))
  )
  assert repl._state.approval.request is not None
  assert len(sent) == 1


class _ComposerWidget(extension.Extension[int, None, None]):
  def __init__(self, height: int) -> None:
    self.height = height

  @classmethod
  def id(cls) -> str:
    return cls.__name__

  @staticmethod
  def initial_state() -> int:
    return 1

  def render_region(self) -> extension.RenderRegion:
    return extension.RenderRegion.WIDGET_BELOW_PROMPT

  def render(
    self, global_state: _state.State, extension_state: int
  ) -> _containers.AnyContainer:
    return _containers.Window(
      _controls.FormattedTextControl(self.id()), height=self.height
    )


class _SecondComposerWidget(_ComposerWidget):
  pass


class _SyntheticMemoryFault(extension.Extension[None, None, None]):
  @classmethod
  def id(cls) -> str:
    return 'core/memory'

  async def on_foreground(self, context: extension.Context) -> None:
    raise AttributeError("'Context' object has no attribute 'extensions'")


class _SyntheticWorktreeFault(_SyntheticMemoryFault):
  @classmethod
  def id(cls) -> str:
    return 'user/worktree-agents'

  async def on_foreground(self, context: extension.Context) -> None:
    raise AttributeError("'WorktreeTab' object has no attribute 'identity'")


@pytest.mark.asyncio
async def test_extension_faults_appear_above_selector_without_taking_focus() -> (
  None
):
  registry = extension.ExtensionRegistry()
  await registry.register(_SyntheticMemoryFault(), None)
  await registry.register(_SyntheticWorktreeFault(), None)
  repl = _repl(extension_registry=registry)
  try:
    await repl._extension_host.start()
    await repl._extension_host.foreground()
    repl._input.text = 'unsent draft'
    repl.dispatch(
      _actions.OpenSelector(
        _selection.Selector(
          'reasoning',
          'Reasoning effort',
          (_selection.Choice('high', 'High'),),
          _actions.SetReasoning,
        )
      )
    )
    height = repl._bottom_pane.preferred_height(80, 40).preferred
    target = _screen.Screen()
    repl._bottom_pane.write_to_screen(
      target,
      _mouse_handlers.MouseHandlers(),
      _screen.WritePosition(0, 0, 80, height),
      '',
      True,
      None,
    )
    rendered = '\n'.join(
      ''.join(target.data_buffer[row][column].char for column in range(80))
      for row in range(height)
    )
    assert rendered.index('Extension core/memory disabled') < rendered.index(
      'Reasoning effort'
    )
    assert rendered.index(
      'Extension user/worktree-agents disabled'
    ) < rendered.index('Reasoning effort')
    assert "'extensions'" in rendered
    assert "'identity'" in rendered
    assert repl._app.layout.current_control is repl._selector_search.control
    assert not repl._transcript.has_conversation
    repl.dispatch(_actions.CloseSelector())
    assert repl._input.text == 'unsent draft'
    assert repl._app.layout.current_control is repl._composer_control
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('widget_height', [0, 1])
@pytest.mark.parametrize('tabs_height', [0, 1])
async def test_composer_spaces_visible_widgets_and_tabs(
  widget_height: int, tabs_height: int
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_ComposerWidget(widget_height), None)
  await registry.register(_SecondComposerWidget(widget_height), None)
  repl = _repl(extension_registry=registry)
  try:
    await repl._extension_host.start()
    repl._navigation = _containers.Window(
      _controls.FormattedTextControl('tabs'), height=tabs_height
    )
    height = 5 + 3 * widget_height + tabs_height
    assert repl._bottom_pane.preferred_height(80, 40).preferred == height
    target = _screen.Screen()
    repl._bottom_pane.write_to_screen(
      target,
      _mouse_handlers.MouseHandlers(),
      _screen.WritePosition(0, 0, 80, height),
      '',
      True,
      None,
    )
    rows = [
      ''.join(
        target.data_buffer[row][column].char for column in range(80)
      ).rstrip()
      for row in range(5, height)
    ]
    assert rows == (
      [_ComposerWidget.id(), '', _SecondComposerWidget.id()] * widget_height
      + ['tabs'] * tabs_height
    )
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
async def test_vim_prompt_and_history(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  registry = extension.ExtensionRegistry()
  assert not await registry.register(vim_extension.VimExtension(), None)
  assert await registry.register(
    vim_extension.VimExtension(), None, overrides={'enabled': True}
  )
  repl = _repl(extension_registry=registry)
  try:
    with _application.set_app(repl._app):
      parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)
      parser.feed('one two')
      repl._app.key_processor.process_keys()
      parser.feed('\x1b')
      parser.flush()
      repl._app.key_processor.process_keys()
      assert vim_extension.indicator(repl._app) == 'NORMAL'
      parser.feed('0dw')
      repl._app.key_processor.process_keys()
      assert repl._input.text == 'two'
      parser.feed('\r')
      repl._app.key_processor.process_keys()
      await asyncio.sleep(0.05)
      assert repl._input.text == ''
      assert vim_extension.indicator(repl._app) == 'INSERT'
      parser.feed('\x12')
      repl._app.key_processor.process_keys()
      await asyncio.sleep(0.05)
      assert repl.selector.id == 'history'
      assert vim_extension.indicator(repl._app) == ''
      parser.feed('\x1b')
      parser.flush()
      repl._app.key_processor.process_keys()
      await asyncio.sleep(0.05)
      assert not repl.selector.active
      assert vim_extension.indicator(repl._app) == 'INSERT'
  finally:
    await repl._app.cancel_and_wait_for_background_tasks()
    await registry.close()


@pytest.mark.asyncio
async def test_vim_approval_and_unload(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  registry = extension.ExtensionRegistry()
  await registry.register(
    vim_extension.VimExtension(), None, overrides={'enabled': True}
  )
  repl = _repl(extension_registry=registry)
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  try:
    with _application.set_app(repl._app):
      parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)
      parser.feed('draft')
      repl._app.key_processor.process_keys()
      parser.feed('\x1b')
      parser.flush()
      repl._app.key_processor.process_keys()
      assert vim_extension.indicator(repl._app) == 'NORMAL'
      repl._handle_message(
        _approval_requested('call', 'bash', '{"command":"ls"}')
      )
      assert vim_extension.indicator(repl._app) == ''
      parser.feed('3')
      repl._app.key_processor.process_keys()
      await asyncio.sleep(0.05)
      assert not sent[-1].hitl_response.approved
      assert repl._input.text == 'draft'
      assert vim_extension.indicator(repl._app) == 'INSERT'
      await registry.unload(vim_extension.VimExtension.id())
      assert vim_extension.indicator(repl._app) == ''
      parser.feed('h')
      repl._app.key_processor.process_keys()
      assert repl._input.text == 'drafht'
  finally:
    await repl._app.cancel_and_wait_for_background_tasks()
    await registry.close()


@pytest.mark.asyncio
async def test_vim_secret_prompt_is_not_modal(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  registry = extension.ExtensionRegistry()
  await registry.register(
    vim_extension.VimExtension(), None, overrides={'enabled': True}
  )
  repl = _repl(extension_registry=registry)
  try:
    with _application.set_app(repl._app):
      parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)
      parser.feed('\x1b')
      parser.flush()
      repl._app.key_processor.process_keys()
      assert vim_extension.indicator(repl._app) == 'NORMAL'
      repl.dispatch(
        _actions.RequestSecret(
          _actions.SecretRequest('test', 'Synthetic input', 'test', str)
        )
      )
      assert vim_extension.indicator(repl._app) == ''
      parser.feed('dw')
      repl._app.key_processor.process_keys()
      assert repl._state.secret.length == 2
      assert repl._input.text == ''
      parser.feed('\x1b')
      parser.flush()
      repl._app.key_processor.process_keys()
      assert not repl._state.secret.active
      assert vim_extension.indicator(repl._app) == 'INSERT'
  finally:
    await repl._app.cancel_and_wait_for_background_tasks()
    await registry.close()


@pytest.mark.asyncio
async def test_vim_widget_is_in_footer_not_prompt(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  registry = extension.ExtensionRegistry()
  await registry.register(
    vim_extension.VimExtension(), None, overrides={'enabled': True}
  )
  repl = _repl(extension_registry=registry)

  def footer() -> str:
    rendered = repl._components['status_bar'].render(repl._state)
    painted = _screen.Screen()
    rendered.write_to_screen(
      painted,
      _mouse_handlers.MouseHandlers(),
      _screen.WritePosition(0, 0, 100, 1),
      '',
      True,
      None,
    )
    return ''.join(painted.data_buffer[0][x].char for x in range(100))

  try:
    await repl._extension_host.start()
    with _application.set_app(repl._app):
      repl._input.text = 'hello'
      repl._input.cursor_position = 5
      prompt = repl._composer_control.create_content(100, 1).get_line(0)
      assert ''.join(fragment[1] for fragment in prompt).strip() == '› hello'
      assert ' INSERT ' in footer()
      parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)
      parser.feed_and_flush('\x1b')
      repl._app.key_processor.process_keys()
      assert ' NORMAL ' in footer()
      parser.feed('v')
      repl._app.key_processor.process_keys()
      assert ' VISUAL ' in footer()
      assert repl._app.style is not None
      assert (
        repl._app.style.get_attrs_for_style_str('class:selected').bgcolor
        == themes
        .prompt_style(themes.current(repl._state.extensions))
        .get_attrs_for_style_str('class:selected')
        .bgcolor
      )
      with mock.patch.object(repl._app.output, 'reset_cursor_shape') as reset:
        await registry.unload(vim_extension.VimExtension.id())
        reset.assert_called_once()
      assert 'VISUAL' not in footer()
      assert repl._app.style.get_attrs_for_style_str(
        'class:selected'
      ) == themes.prompt_style(
        themes.current(repl._state.extensions)
      ).get_attrs_for_style_str('class:selected')
  finally:
    await repl._app.cancel_and_wait_for_background_tasks()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('keys', [None, '', '\x1b', '\x1bf\x1b', '\x1bd\x1b'])
async def test_external_editor_round_trip(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  keys: str | None,
) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('VISUAL', 'test-editor --wait')
  registry = extension.ExtensionRegistry()
  assert await registry.register(
    external_editor_extension.ExternalEditorExtension(), None
  )
  if keys is not None:
    await registry.register(
      vim_extension.VimExtension(), None, overrides={'enabled': True}
    )
  repl = _repl(extension_registry=registry)
  repl._input.text = 'original draft'
  paths: list[pathlib.Path] = []

  def run(
    argv: tuple[str, ...],
    *,
    stdout: object,
    stderr: object,
    close_fds: bool,
    check: bool,
  ) -> subprocess.CompletedProcess[bytes]:
    assert argv[-3:-1] == ('test-editor', '--wait')
    path = pathlib.Path(argv[-1])
    paths.append(path)
    assert path.read_text() == 'original draft\n'
    path.write_text('edited\n世界\n', encoding='utf-8')
    return subprocess.CompletedProcess(argv, 0)

  monkeypatch.setattr(subprocess, 'run', run)
  try:
    with _application.set_app(repl._app):
      parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)
      parser.feed_and_flush(keys or '')
      repl._app.key_processor.process_keys()
      parser.feed('\x07')
      repl._app.key_processor.process_keys()
      async with asyncio.timeout(5):
        while repl._input.text == 'original draft':
          await asyncio.sleep(0.01)
      assert repl._input.text == 'edited\n世界'
      assert repl._input.document.cursor_position_row == 1
      assert repl._input.document.cursor_position_col == 2
      assert len(paths) == 1
      assert not paths[0].exists()
      repl._input.undo()
      assert repl._input.text == 'original draft'
  finally:
    await repl._app.cancel_and_wait_for_background_tasks()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  'failure', ['exit', 'missing', 'encoding', 'changed', 'secret']
)
async def test_external_editor_preserves_draft_on_failure(
  monkeypatch: pytest.MonkeyPatch,
  failure: str,
) -> None:
  monkeypatch.setenv('VISUAL', 'test-editor')
  repl = _repl()
  repl._input.text = 'original draft'
  paths: list[pathlib.Path] = []

  def run(
    argv: tuple[str, ...],
    *,
    stdout: object,
    stderr: object,
    close_fds: bool,
    check: bool,
  ) -> subprocess.CompletedProcess[bytes]:
    path = pathlib.Path(argv[-1])
    paths.append(path)
    path.write_bytes(b'\xff' if failure == 'encoding' else b'edited')
    if failure == 'missing':
      raise FileNotFoundError('test-editor')
    return subprocess.CompletedProcess(argv, 1 if failure == 'exit' else 0)

  async def handoff(
    callback: Callable[[], object], *, in_executor: bool
  ) -> object:
    result = callback()
    if failure == 'changed':
      repl._input.text = 'new draft'
    if failure == 'secret':
      repl.dispatch(
        _actions.RequestSecret(
          _actions.SecretRequest('test', 'Synthetic', 'test', str)
        )
      )
    return result

  monkeypatch.setattr(subprocess, 'run', run)
  monkeypatch.setattr(cli, '_run_in_terminal', handoff)
  await repl._edit_composer()
  assert repl._input.text == (
    'new draft' if failure == 'changed' else 'original draft'
  )
  assert not paths[0].exists()
  assert not repl._editing_prompt


@pytest.mark.asyncio
async def test_external_editor_does_not_launch_for_secret(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl()
  repl.dispatch(
    _actions.RequestSecret(
      _actions.SecretRequest('test', 'Synthetic', 'test', str)
    )
  )
  with mock.patch.object(subprocess, 'run') as run:
    await repl._edit_composer()
    run.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('original', 'saved', 'expected'),
  [
    ('draft', 'draft\n', 'draft'),
    ('draft\n', 'draft\n\n', 'draft\n'),
    ('draft', 'edited\n\n', 'edited\n'),
    ('draft', 'edited', 'edited'),
    ('draft', '', ''),
    ('', '\n', ''),
  ],
)
async def test_external_editor_file_newline(
  monkeypatch: pytest.MonkeyPatch,
  original: str,
  saved: str,
  expected: str,
) -> None:
  monkeypatch.setenv('VISUAL', 'test-editor')
  repl = _repl()
  repl._input.text = original

  def run(
    argv: tuple[str, ...],
    *,
    stdout: object,
    stderr: object,
    close_fds: bool,
    check: bool,
  ) -> subprocess.CompletedProcess[bytes]:
    path = pathlib.Path(argv[-1])
    assert path.read_text() == original + '\n'
    path.write_text(saved, encoding='utf-8')
    return subprocess.CompletedProcess(argv, 0)

  monkeypatch.setattr(subprocess, 'run', run)
  await repl._edit_composer()
  assert repl._input.text == expected


@pytest.mark.asyncio
async def test_cat_command_choice_persists_and_matches_other_literal_arguments(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  repl = _repl(approvals=tool_approvals.ToolApprovals(path))
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(
    _approval_requested('first', 'bash', '{"command":"cat first.txt"}')
  )
  choices = repl._state.selector.filtered_choices()
  assert choices[2].value == 'reject'
  assert len(choices) == 3
  assert choices[1].value == 'remember'
  assert (
    choices[1].label == 'Approve and always allow commands starting with `cat`'
  )
  assert not path.exists()
  repl.dispatch(_actions.AcceptSelection(1))
  await _settle(repl)
  assert sent[-1].hitl_response.approved
  assert json.loads(path.read_text())['entries'][0]['argv'] == ['cat']

  restarted = _repl(approvals=tool_approvals.ToolApprovals(path))
  restarted._send_request = sent.append
  restarted._handle_message(
    _approval_requested(
      'next', 'bash', '{"command":"cat -n \'other file.txt\'"}'
    )
  )
  assert restarted._state.approval.request is None
  assert sent[-1].hitl_response.call_id == 'next'
  assert (
    sent[-1].hitl_response.source == agent_pb2.APPROVAL_SOURCE_REMEMBERED_PREFIX
  )


@pytest.mark.parametrize(
  'name,arguments',
  [
    ('bash', '{"command":"catfish first.txt"}'),
    ('bash', '{"command":"/bin/cat first.txt"}'),
    ('bash', '{"command":"./cat first.txt"}'),
    ('bash', '{"command":"env cat first.txt"}'),
    ('bash', '{"command":"sudo cat first.txt"}'),
    ('bash', '{"command":"command cat first.txt"}'),
    ('bash', '{"command":"cat first.txt; echo other"}'),
    ('bash', '{"command":"cat first.txt && echo other"}'),
    ('bash', '{"command":"cat first.txt | cat"}'),
    ('bash', '{"command":"cat $(echo first.txt)"}'),
    ('bash', '{"command":"cat first.txt > output"}'),
    ('bash', '{"command":"cat first.txt","cwd":"/other"}'),
    ('bash', '{"command":"cat first.txt","command":"echo other"}'),
    ('other__bash', '{"command":"cat first.txt"}'),
  ],
)
def test_cat_prefix_does_not_approve_other_execution(
  tmp_path: pathlib.Path,
  name: str,
  arguments: str,
) -> None:
  store = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  repl = _repl(approvals=store)
  store.remember(
    tool_approvals.ArgvPrefixApproval(
      repl._local_approval_scope,
      'bash',
      ('cat',),
    )
  )
  sent: list[agent_pb2.ClientMessage] = []
  repl._send_request = sent.append
  repl._handle_message(_approval_requested('boundary', name, arguments))
  assert repl._state.approval.request is not None
  assert not sent


@pytest.mark.asyncio
async def test_fullscreen_command_output_stays_in_the_selected_transcript() -> (
  None
):
  repl = _repl(fullscreen=True)
  try:
    repl._transcript.append(
      _delta({'role': 'user', 'content': 'hello'}).agent_message
    )
    task = repl._run_effect(_effects.EmitText('COMMAND OUTPUT', ''))
    if task is not None:
      await task
    content = repl._viewport.create_content(80, 15)
    assert 'COMMAND OUTPUT' in '\n'.join(
      ''.join(fragment[1] for fragment in content.get_line(index))
      for index in range(content.line_count)
    )
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_fullscreen_large_composer_keeps_a_bounded_dock() -> None:
  repl = _repl(fullscreen=True, output=_SizedOutput(rows=30, columns=80))
  repl._input.text = '\n'.join('long input' for _ in range(300))
  assert repl._input_window.preferred_height(80, 30).max <= 10
  assert repl._dock_height() <= 20


@pytest.mark.asyncio
async def test_fullscreen_repaint_preserves_the_current_reading_position() -> (
  None
):
  repl = _repl(fullscreen=True)
  try:
    for index in range(20):
      repl._handle_message(
        _delta({'role': 'assistant', 'content': f'answer {index}'})
      )
    repl._handle_message(_replay_done('conv-1'))
    repl._viewport.create_content(80, 8)
    repl._viewport.first()
    await repl._rerender()
    assert repl._viewport.reading
    content = repl._viewport.create_content(80, 8)
    visible = '\n'.join(
      ''.join(fragment[1] for fragment in content.get_line(index))
      for index in range(content.line_count)
    )
    assert 'answer 0' in visible
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_fullscreen_catalog_restart_refreshes_durable_history(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  repl = _repl(fullscreen=True)
  monkeypatch.setattr(repl, '_shutdown_worker', mock.AsyncMock())
  repl._handle_message(_stream_content('OBSOLETE PREVIEW'))

  async def start(
    *,
    replay: bool,
    configuration: _rpc_client.WorkerConfiguration | None = None,
  ) -> None:
    assert replay
    assert configuration is not None
    assert repl._state.session.replaying
    repl._handle_message(
      _delta({'role': 'assistant', 'content': 'DURABLE GAP MESSAGE'})
    )
    repl._handle_message(_replay_done('conv-1'))

  monkeypatch.setattr(repl, '_start_worker', start)
  try:
    assert await repl._restart_worker({}, replay=False)
    assert repl._stream_pane.preview is None
    content = repl._viewport.create_content(80, 8)
    visible = '\n'.join(
      ''.join(fragment[1] for fragment in content.get_line(index))
      for index in range(content.line_count)
    )
    assert 'DURABLE GAP MESSAGE' in visible
    assert 'OBSOLETE PREVIEW' not in visible
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_fullscreen_keeps_pinned_sign_in_instructions_visible() -> None:
  repl = _repl(fullscreen=True, output=_SizedOutput(rows=30, columns=80))
  repl._terminal.pin_header(
    _text.Text('SIGN IN GUIDE', style='link=https://example.test')
  )
  repl._secret_input = _buffer.Buffer()
  frame = _screen.Screen(initial_width=80, initial_height=30)
  try:
    with _current.set_app(repl._app):
      repl._app.layout.container.write_to_screen(
        frame,
        _mouse_handlers.MouseHandlers(),
        _screen.WritePosition(xpos=0, ypos=0, width=80, height=30),
        '',
        False,
        None,
      )
    visible = '\n'.join(
      ''.join(row[index].char for index in sorted(row))
      for _, row in sorted(frame.data_buffer.items())
    )
    assert 'SIGN IN GUIDE' in visible
    assert 'https://example.test' not in visible
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('rows, columns', [(15, 40), (30, 80), (50, 120)])
async def test_fullscreen_pending_overflow_keeps_the_status_bar_visible(
  rows: int,
  columns: int,
) -> None:
  repl = _repl(fullscreen=True, output=_SizedOutput(rows=rows, columns=columns))
  repl.dispatch(_actions.TerminalResized(columns=columns))
  repl.dispatch(_actions.SetReplaying(False))
  repl._handle_message(_delta({'role': 'user', 'content': 'begin'}))
  for index in range(40):
    repl._handle_message(
      _stream_tool_call(
        index=index,
        name='read',
        arguments='{"file_path":"example.py"}',
      )
    )
  repl._input.text = '\n'.join('long input' for _ in range(300))
  frame = _screen.Screen(initial_width=columns, initial_height=rows)
  try:
    with _current.set_app(repl._app):
      repl._app.layout.container.write_to_screen(
        frame,
        _mouse_handlers.MouseHandlers(),
        _screen.WritePosition(xpos=0, ypos=0, width=columns, height=rows),
        '',
        False,
        None,
      )
    visible = [
      ''.join(frame.data_buffer[y][x].char for x in range(columns))
      for y in range(rows)
    ]
    status_row = next(i for i, line in enumerate(visible) if 'm-test' in line)
    input_row = next(
      i for i, line in enumerate(visible) if 'long input' in line
    )
    assert input_row < status_row
    assert 'long input' in visible[status_row - 2]
    assert not visible[status_row - 1].strip()
    assert visible[-1] == '▀' * columns
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('active', [True, False])
async def test_fullscreen_aborted_stream_is_retained_after_its_preview_closes(
  active: bool,
) -> None:
  repl = _repl(fullscreen=True, active=active)
  try:
    repl.dispatch(_actions.SetReplaying(False), _actions.TurnStarted())
    repl._handle_message(_stream_content('PARTIAL ABORTED ANSWER'))
    repl._handle_message(_stream_aborted())
    assert repl._stream_pane.preview is None
    visible = _rendering.strip_ansi(
      _rendering.render_blocks(
        [block.renderable for block in repl._transcript.blocks()],
        80,
      )
    )
    assert 'PARTIAL ABORTED ANSWER' in visible
    assert not repl.transcript_entries()
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_history_escape_keeps_modal_precedence_and_preserves_the_draft() -> (
  None
):
  repl = _repl(fullscreen=True)
  try:
    repl.dispatch(_actions.SetReplaying(False))
    repl._handle_message(
      _delta({
        'role': 'assistant',
        'content': '\n\n'.join(f'Paragraph {i}' for i in range(40)),
      })
    )
    repl._viewport.create_content(80, 8)
    repl.dispatch(_actions.NavigateHistory('oldest'))
    assert repl._viewport.reading
    repl._input.text = 'unfinished draft'
    with _current.set_app(repl._app):
      assert repl._dismissal_action() == _actions.NavigateHistory('latest')
    repl.dispatch(
      _actions.OpenSelector(
        _selection.Selector('test', 'Pick', (), _actions.RestorePrompt)
      )
    )
    with _current.set_app(repl._app):
      assert repl._dismissal_action() == _actions.CloseSelector()
    assert repl._viewport.reading
    repl.dispatch(_actions.ClearSelector())
    repl.dispatch(_actions.NavigateHistory('latest'))
    assert not repl._viewport.reading
    assert repl._input.text == 'unfinished draft'
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fullscreen', [False, True])
async def test_conflicting_durable_user_id_cannot_change_the_local_history_index(
  tmp_path: pathlib.Path, fullscreen: bool
) -> None:
  repl = _repl(database=tmp_path / 'history.db', fullscreen=fullscreen)
  try:
    repl.dispatch(_actions.SetReplaying(False))
    entry = agent_pb2.AgentMessage(
      id='durable-user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='ORIGINAL USER')
      ),
    )
    repl._on_worker_message(agent_pb2.SessionMessage(agent_message=entry))
    conflicting = agent_pb2.AgentMessage()
    conflicting.CopyFrom(entry)
    conflicting.chat.user.content = 'CONFLICTING USER'
    with pytest.raises(ValueError, match='Conflicting durable message'):
      repl._on_worker_message(
        agent_pb2.SessionMessage(agent_message=conflicting)
      )
    assert repl.transcript_entries() == (entry,)
    with repl._store() as store:
      assert store.recent_user_messages('conv-1') == ['ORIGINAL USER']
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fullscreen', [False, True])
async def test_redacted_startup_metadata_does_not_replace_canonical_history(
  fullscreen: bool,
) -> None:
  repl = _repl(fullscreen=fullscreen)
  try:
    repl.dispatch(_actions.SetReplaying(False))
    canonical = agent_pb2.AgentMessage(
      id='context',
      sequence=1,
      context_window=agent_pb2.ContextWindowRecord(
        system_prompt='synthetic prompt'
      ),
    )
    redacted = agent_pb2.AgentMessage()
    redacted.CopyFrom(canonical)
    redacted.context_window.system_prompt = ''
    repl._on_worker_startup_message(
      agent_pb2.SessionMessage(agent_message=redacted)
    )
    assert not repl.transcript_entries()
    repl._on_worker_message(agent_pb2.SessionMessage(agent_message=canonical))
    assert repl.transcript_entries() == (canonical,)
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_reading_history_does_not_steal_escape_from_a_native_approval() -> (
  None
):
  repl = _repl(fullscreen=True)
  try:
    repl.dispatch(_actions.SetReplaying(False))
    repl._handle_message(
      _delta({
        'role': 'assistant',
        'content': '\n\n'.join(f'Paragraph {i}' for i in range(40)),
      })
    )
    repl._viewport.create_content(80, 8)
    repl.dispatch(_actions.NavigateHistory('oldest'))
    repl._handle_message(_approval_requested('call', 'bash'))
    assert repl._state.approval.request is not None
    parser = _vt100_parser.Vt100Parser(repl._app.key_processor.feed)
    with _current.set_app(repl._app):
      assert repl._dismissal_action() == _actions.CloseSelector()
      parser.feed('\x1b')
      parser.flush()
      repl._app.key_processor.process_keys()
    assert repl._viewport.reading
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('rows, columns', [(15, 40), (30, 80), (80, 180)])
async def test_fullscreen_approval_is_visible_in_the_assembled_dock(
  rows: int, columns: int
) -> None:
  repl = _repl(fullscreen=True, output=_SizedOutput(rows=rows, columns=columns))
  try:
    repl.dispatch(_actions.SetReplaying(False))
    repl._handle_message(_approval_requested('call', 'bash'))
    frame = _screen.Screen()
    with _current.set_app(repl._app):
      repl._app.layout.container.write_to_screen(
        frame,
        _mouse_handlers.MouseHandlers(),
        _screen.WritePosition(0, 0, columns, rows),
        '',
        False,
        None,
      )
    visible = '\n'.join(
      ''.join(frame.data_buffer[y][x].char for x in range(columns))
      for y in range(rows)
    )
    assert 'Tool approval' in visible
    assert '1. Approve' in visible
    assert '3. Reject' in visible
    assert 'Esc' in visible
    assert 'm-test' in visible
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ('columns', 'composer_start', 'composer_end'),
  [(40, 3, 37), (80, 3, 77), (180, 50, 130)],
)
@pytest.mark.parametrize('rows', [30, 32])
async def test_fullscreen_retains_widgets_and_mirror_composer_styling(
  columns: int,
  composer_start: int,
  composer_end: int,
  rows: int,
) -> None:
  registry = extension.ExtensionRegistry()
  await registry.register(_ComposerWidget(1), None)
  await registry.register(_SecondComposerWidget(1), None)
  repl = _repl(
    fullscreen=True,
    output=_SizedOutput(rows=rows, columns=columns),
    extension_registry=registry,
    workspace_path=pathlib.Path('/workspace/mirror'),
  )
  try:
    await repl._extension_host.start()
    repl._navigation = _containers.Window(
      _controls.FormattedTextControl('git main'), height=1
    )
    repl._input.text = 'VISIBLE DRAFT'
    frame = _screen.Screen()
    with _current.set_app(repl._app):
      repl._app.layout.container.write_to_screen(
        frame,
        _mouse_handlers.MouseHandlers(),
        _screen.WritePosition(0, 0, columns, rows),
        '',
        False,
        None,
      )
    visible = [
      ''.join(frame.data_buffer[y][x].char for x in range(columns))
      for y in range(rows)
    ]
    assert _ComposerWidget.id() in '\n'.join(visible)
    assert _SecondComposerWidget.id() in '\n'.join(visible)
    input_row = next(
      y for y, line in enumerate(visible) if '› VISIBLE DRAFT' in line
    )
    widget_row = next(
      y for y, line in enumerate(visible) if _ComposerWidget.id() in line
    )
    assert visible[-1].strip() == 'git main'
    assert input_row < widget_row < rows - 1
    assert visible[input_row].index('› VISIBLE DRAFT') == composer_start + 1
    assert (
      visible[input_row][composer_start]
      == visible[input_row][composer_end - 1]
      == ' '
    )
    assert (
      frame.data_buffer[input_row][composer_start].style
      == frame.data_buffer[input_row][composer_end - 1].style
    )
    assert (
      'class:composer' in frame.data_buffer[input_row][composer_start].style
    )
    status_row = next(
      y for y, line in enumerate(visible) if 'auto | m-test' in line
    )
    status_column = visible[status_row].index('auto | m-test')
    assert (
      'class:composer' in frame.data_buffer[status_row][status_column].style
    )
    assert input_row < status_row < widget_row
  finally:
    await repl.close()
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('selected', [0, 9])
@pytest.mark.parametrize('section', ['', 'Options'])
async def test_small_fullscreen_selector_keeps_its_selected_choice_visible(
  selected: int,
  section: str,
) -> None:
  repl = _repl(fullscreen=True, output=_SizedOutput(rows=15, columns=40))
  try:
    repl.dispatch(
      _actions.OpenSelector(
        _selection.Selector(
          'test',
          'CHOOSE',
          tuple(
            _selection.Choice(str(i), f'CHOICE_{i}', section=section)
            for i in range(10)
          ),
          _actions.RestorePrompt,
        )
      ),
      _actions.MoveSelection(selected),
    )
    frame = _screen.Screen()
    with _current.set_app(repl._app):
      repl._app.before_render.fire()
      repl._app.layout.container.write_to_screen(
        frame,
        _mouse_handlers.MouseHandlers(),
        _screen.WritePosition(0, 0, 40, 15),
        '',
        False,
        None,
      )
    visible = '\n'.join(
      ''.join(frame.data_buffer[y][x].char for x in range(40))
      for y in range(15)
    )
    assert 'CHOOSE' in visible
    assert f'› CHOICE_{selected}' in visible
    assert section in visible
    assert 'Enter · Esc' in visible
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_fullscreen_approval_command_scrolls_without_losing_choices() -> (
  None
):
  output = _SizedOutput(rows=30, columns=80)
  repl = _repl(fullscreen=True, output=output)

  def paint() -> tuple[list[str], _mouse_handlers.MouseHandlers]:
    size = output.get_size()
    frame = _screen.Screen()
    handlers = _mouse_handlers.MouseHandlers()
    repl._app.before_render.fire()
    repl._app.layout.update_parents_relations()
    repl._app.layout.container.write_to_screen(
      frame,
      handlers,
      _screen.WritePosition(0, 0, size.columns, size.rows),
      '',
      False,
      None,
    )
    return [
      ''.join(frame.data_buffer[y][x].char for x in range(size.columns))
      for y in range(size.rows)
    ], handlers

  try:
    repl._handle_message(
      _approval_requested(
        'call',
        'bash',
        json.dumps({
          'command': '\n'.join(f'echo LINE_{i:03d}' for i in range(30))
        }),
      )
    )
    with _current.set_app(repl._app):
      initial, handlers = paint()
      assert 'LINE_000' in '\n'.join(initial)
      row = next(y for y, line in enumerate(initial) if 'LINE_000' in line)
      column = initial[row].index('LINE_000')
      handlers.mouse_handlers[row][column](
        _mouse_events.MouseEvent(
          position=_data_structures.Point(column, row),
          event_type=_mouse_events.MouseEventType.SCROLL_DOWN,
          button=_mouse_events.MouseButton.NONE,
          modifiers=frozenset(),
        )
      )
      scrolled, _ = paint()
      assert 'LINE_000' not in '\n'.join(scrolled)
      assert 'LINE_001' in '\n'.join(scrolled)
      assert '1. Approve' in '\n'.join(scrolled)
      assert '3. Reject' in '\n'.join(scrolled)
      output.size = _data_structures.Size(rows=20, columns=60)
      repl.dispatch(_actions.TerminalResized(columns=60))
      resized, _ = paint()
      assert 'LINE_001' in '\n'.join(resized)
      assert '1. Approve' in '\n'.join(resized)
      assert '3. Reject' in '\n'.join(resized)
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_conflicting_replay_id_cannot_change_the_local_history_index(
  tmp_path: pathlib.Path,
) -> None:
  repl = _repl(database=tmp_path / 'history.db', fullscreen=True)
  try:
    entry = agent_pb2.AgentMessage(
      id='replayed-user',
      sequence=1,
      chat=agent_pb2.ChatRecord(
        user=agent_pb2.UserMessage(content='ORIGINAL USER')
      ),
    )
    repl._on_worker_message(agent_pb2.SessionMessage(agent_message=entry))
    conflicting = agent_pb2.AgentMessage()
    conflicting.CopyFrom(entry)
    conflicting.chat.user.content = 'CONFLICTING USER'
    with pytest.raises(ValueError, match='Conflicting durable message'):
      repl._on_worker_message(
        agent_pb2.SessionMessage(agent_message=conflicting)
      )
    assert repl._transcript.replayed() == (entry,)
    with repl._store() as store:
      assert store.recent_user_messages('conv-1') == ['ORIGINAL USER']
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_fullscreen_abort_preserves_the_live_reading_position() -> None:
  repl = _repl(fullscreen=True)
  try:
    repl.dispatch(_actions.SetReplaying(False), _actions.TurnStarted())
    repl._handle_message(
      _stream_content('\n\n'.join(f'LIVE PARAGRAPH {i:03d}' for i in range(80)))
    )
    repl._viewport.create_content(80, 12)
    repl._viewport.scroll(-30)
    before = repl._viewport.create_content(80, 12)
    first = ''.join(fragment[1] for fragment in before.get_line(0))
    repl._handle_message(_stream_aborted())
    after = repl._viewport.create_content(80, 12)
    assert ''.join(fragment[1] for fragment in after.get_line(0)) == first
  finally:
    await repl.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('acknowledged', [False, True])
async def test_session_switch_keeps_pending_input_and_drafts_in_the_origin(
  acknowledged: bool,
) -> None:
  repl = _repl(fullscreen=True)
  try:
    repl.dispatch(_actions.InputSubmitted('pending-alpha', 'ALPHA ONLY'))
    if acknowledged:
      repl._handle_message(
        agent_pb2.SessionMessage(
          input_ack=agent_pb2.InputAck(message_id='pending-alpha'),
        )
      )
    repl._input.text = 'ALPHA DRAFT'
    repl._on_session_opened('beta')
    assert repl._state.turn.pending_instructions == ()
    assert repl._input.text == ''
    repl._input.text = 'BETA DRAFT'
    repl._on_session_opened('conv-1')
    assert repl._input.text == 'ALPHA ONLY\nALPHA DRAFT'
    assert repl._state.turn.pending_instructions == ()
    repl._on_session_opened('beta')
    assert repl._input.text == 'BETA DRAFT'
  finally:
    await repl.close()


@pytest.mark.asyncio
async def test_fullscreen_jump_hint_is_centered_and_click_returns_to_latest() -> (
  None
):
  repl = _repl(fullscreen=True, output=_SizedOutput(rows=15, columns=80))
  try:
    repl.dispatch(_actions.SetReplaying(False))
    for index in range(20):
      repl._handle_message(
        _delta({'role': 'assistant', 'content': f'answer {index}'})
      )
    repl._viewport.create_content(80, 5)
    repl._viewport.first()
    frame = _screen.Screen()
    handlers = _mouse_handlers.MouseHandlers()
    with _current.set_app(repl._app):
      repl._app.layout.update_parents_relations()
      repl._app.layout.container.write_to_screen(
        frame, handlers, _screen.WritePosition(0, 0, 80, 15), '', False, None
      )
      visible = [
        ''.join(frame.data_buffer[y][x].char for x in range(80))
        for y in range(15)
      ]
      row = next(
        y for y, line in enumerate(visible) if 'Back to bottom' in line
      )
      assert (
        abs(
          len(visible[row])
          - len(visible[row].lstrip())
          - (len(visible[row]) - len(visible[row].rstrip()))
        )
        <= 1
      )
      column = visible[row].index('Back to bottom')
      handlers.mouse_handlers[row][column](
        _mouse_events.MouseEvent(
          position=_data_structures.Point(column, row),
          event_type=_mouse_events.MouseEventType.MOUSE_UP,
          button=_mouse_events.MouseButton.LEFT,
          modifiers=frozenset(),
        )
      )
    assert not repl._viewport.reading
  finally:
    await repl.close()
