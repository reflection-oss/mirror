"""Opt-in smoke test of the installed CLI and bundled Rust worker."""

from __future__ import annotations

import contextlib
import errno
import http.server
import json
import os
import pathlib
import platform
import pty
import queue
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from typing import override

import pyte
import pytest

from mirror import config
from mirror.cli import extension, user_extensions

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
  tokens = (11, 7, 3)[index]
  call = {'name': 'ls', 'arguments': '{"path":"."}'}
  call_id = f'call-{index}'
  if path == '/responses':
    output = (
      [{'type': 'function_call', 'call_id': call_id, **call}]
      if index < 2
      else [
        {
          'type': 'message',
          'content': [{'type': 'output_text', 'text': _ANSWER}],
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
      else {'content': _ANSWER}
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
) -> None:
  """Drain the PTY until text renders, or until clean exit when text is None."""
  deadline = time.monotonic() + 30
  while time.monotonic() < deadline:
    if text is not None and text in '\n'.join(screen.display):
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
  home = tmp_path / 'home'
  workspace = tmp_path / 'workspace'
  home.mkdir()
  workspace.mkdir()
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
    'MIRROR_CLI_DB': str(home / 'cli.db'),
  }
  # setup-python needs its matching libpython, not the system Python library.
  if library_path := os.environ.get('LD_LIBRARY_PATH'):
    env['LD_LIBRARY_PATH'] = library_path
  with _ModelServer() as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    master = slave = -1
    child = None
    try:
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
      _wait_for_screen(child, screen, stream, 'packaged-smoke-model')
      _wait_for_screen(child, screen, stream, 'anonymous feature usage')
      settings = _wait_for_config_key(
        home / '.mirror' / 'config.json', 'analytics_notice_shown'
      )
      assert re.fullmatch(r'[0-9a-f]{32}', settings.pop('install_id'))
      assert settings == {
        'usage_warning_acknowledged': True,
        'trusted_directories': [str(workspace.resolve())],
        'analytics_notice_shown': True,
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
      os.write(master, b'\x04')  # Ctrl+D requests a graceful CLI shutdown.
      _wait_for_screen(child, screen, stream, None)
    finally:
      server.release_response.put(None)
      server.release_response.put(None)
      try:
        if child is not None:
          with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
          child.wait(timeout=5)
      finally:
        if master != -1:
          os.close(master)
        if slave != -1:
          os.close(slave)
        server.shutdown()
        thread.join(timeout=10)
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
