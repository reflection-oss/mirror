"""Tests for Mirror CLI conversation selection."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import logging
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Mapping
from unittest import mock

import pytest
from prompt_toolkit.application import current
from rich import console

from mirror import distribution
from mirror.cli import (
  cli_store,
  local_rpc_client,
  main,
  models,
  onboarding,
  rpc_client,
  showcase,
)
from mirror.cli import logging as mirror_logging


def _args(
  *,
  conversation_id: str | None = None,
  continue_recent: bool = False,
  allow_foreign_workspace: bool = False,
) -> argparse.Namespace:
  return argparse.Namespace(
    conversation_id=conversation_id,
    continue_recent=continue_recent,
    allow_foreign_workspace=allow_foreign_workspace,
  )


@pytest.mark.parametrize('command', ['exec', 'acp'])
@pytest.mark.parametrize('firecrawl', [False, True])
def test_noninteractive_worker_receives_default_search_connection(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  command: str,
  firecrawl: bool,
) -> None:
  monkeypatch.setenv('MIRROR_MCP_DIR', str(tmp_path / 'mcp'))
  monkeypatch.setenv('FIRECRAWL_API_KEY', 'configured-key' if firecrawl else '')
  monkeypatch.setattr(sys, 'argv', ['mirror', command])
  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]
  config = main._worker_config(  # pyright: ignore[reportPrivateUsage]
    args, 'test', create_session=True, allow_workspace_mismatch=False
  )
  expected = (
    []
    if firecrawl
    else [
      {
        'name': 'parallel',
        'config': {
          'url': 'https://search.parallel.ai/mcp',
          'enabledTools': ['web_search', 'web_fetch'],
        },
        'credential_directory': str(tmp_path / 'mcp' / 'credentials'),
      }
    ]
  )
  assert config['mcp_connections'] == expected


@pytest.mark.parametrize('fail', [False, True])
def test_showcase_restores_cwd(
  monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
  original = pathlib.Path.cwd()
  workspaces: list[pathlib.Path] = []

  async def run_tui(
    worker_command: list[str],
    worker_config: Mapping[str, object],
    **kwargs: object,
  ) -> None:
    del worker_command, worker_config
    cwd = pathlib.Path.cwd()
    workspaces.append(cwd)
    assert cwd != original
    assert kwargs['database'] == cwd / 'cli.db'
    assert 'workspace' not in kwargs
    if fail:
      raise RuntimeError('showcase failed')

  monkeypatch.setattr(showcase.cli, 'run_tui', run_tui)
  if fail:
    with pytest.raises(RuntimeError, match='showcase failed'):
      showcase.run()
  else:
    showcase.run()
  assert pathlib.Path.cwd() == original
  assert len(workspaces) == 1
  assert not workspaces[0].exists()


def test_database_paths_must_be_distinct(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  database = tmp_path / 'state.db'
  monkeypatch.setenv('MIRROR_CLI_DB', str(database))

  with pytest.raises(SystemExit, match='must use different paths'):
    main._database_paths(str(database))  # pyright: ignore[reportPrivateUsage]


def test_run_configures_logging_before_starting_tui(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  configured = object()
  calls: list[object] = []
  monkeypatch.setattr(
    'mirror.cli.main._logging.configure_logging',
    lambda: calls.append('configure_logging') or configured,
  )
  monkeypatch.setattr(main, 'main', lambda logs: calls.append(logs))

  main.run()

  assert calls == ['configure_logging', configured]


def test_logging_flushes_startup_records_to_conversation(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  root = logging.getLogger()
  existing_handlers = tuple(root.handlers)
  previous_level = root.level
  previous_disable = root.manager.disable
  monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
  try:
    logs = mirror_logging.configure_logging()
    logging.getLogger('mirror.test').info('before conversation')
    logs.set_conversation('conversation-1')
    logging.getLogger('mirror.test').info('after conversation')

    no_id = (tmp_path / '.mirror/logs/no-id.log').read_text()
    conversation = (
      tmp_path / '.mirror/logs/conversation-1/cli-stderr'
    ).read_text()
    assert 'before conversation' in no_id
    assert 'after conversation' not in no_id
    assert 'before conversation' in conversation
    assert 'after conversation' in conversation
    assert (tmp_path / '.mirror/logs/no-id.log').stat().st_mode & 0o777 == 0o600
  finally:
    for handler in tuple(root.handlers):
      root.removeHandler(handler)
      handler.close()
    for handler in existing_handlers:
      root.addHandler(handler)
    root.setLevel(previous_level)
    logging.disable(previous_disable)


def test_logging_does_not_require_a_writable_home(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  root = logging.getLogger()
  existing_handlers = tuple(root.handlers)
  previous_level = root.level
  previous_disable = root.manager.disable
  unavailable = tmp_path / 'home'
  unavailable.write_text('not a directory')
  monkeypatch.setattr(pathlib.Path, 'home', lambda: unavailable)
  try:
    logs = mirror_logging.configure_logging()
    logging.getLogger('mirror.test').info('before conversation')
    logs.set_conversation('conversation-1')
    logging.getLogger('mirror.test').info('after conversation')
  finally:
    for handler in tuple(root.handlers):
      root.removeHandler(handler)
      handler.close()
    for handler in existing_handlers:
      root.addHandler(handler)
    root.setLevel(previous_level)
    logging.disable(previous_disable)


def test_capture_output_keeps_only_ui_output_on_terminal(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  capfd: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
  original_stdout, original_stderr = sys.stdout, sys.stderr
  # capfd's sys.stderr owns a separate file; use the real fd-2 stream.
  handler = logging.StreamHandler(sys.__stderr__)
  with mirror_logging.capture_output('conversation-1') as terminal:
    print('python stdout')
    print('python stderr', file=sys.stderr)
    os.write(1, b'native stdout\n')
    os.write(2, b'native stderr\n')
    thread = threading.Thread(target=os.write, args=(2, b'background stderr\n'))
    thread.start()
    thread.join()
    handler.handle(logging.makeLogRecord({'msg': 'existing handler'}))
    subprocess.run(
      [
        sys.executable,
        '-c',
        'print("child stdout"); import os; os.write(2, b"child stderr\\n")',
      ],
      check=True,
    )
    console.Console(file=terminal, color_system=None).print('rich UI')
    output = current.get_app_session().output
    output.write('prompt UI\n')
    output.flush()
  assert sys.stdout is original_stdout
  assert sys.stderr is original_stderr
  captured = capfd.readouterr()
  assert captured.out == 'rich UI\nprompt UI\n'
  assert captured.err == ''
  stdout = (tmp_path / '.mirror/logs/conversation-1/cli-stdout').read_text()
  stderr = (tmp_path / '.mirror/logs/conversation-1/cli-stderr').read_text()
  assert 'python stdout' in stdout
  assert 'native stdout' in stdout
  assert 'child stdout' in stdout
  assert 'stderr' not in stdout
  assert 'existing handler' not in stdout
  assert 'python stderr' in stderr
  assert 'native stderr' in stderr
  assert 'background stderr' in stderr
  assert 'existing handler' in stderr
  assert 'child stderr' in stderr
  assert 'stdout' not in stderr
  assert 'UI' not in stdout + stderr
  assert (
    (tmp_path / '.mirror/logs/conversation-1/cli-stdout').stat().st_mode & 0o777
  ) == 0o600
  os.write(1, b'restored stdout\n')
  os.write(2, b'restored stderr\n')
  restored = capfd.readouterr()
  assert restored.out == 'restored stdout\n'
  assert restored.err == 'restored stderr\n'


def test_capture_output_restores_terminal_after_failure(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  capfd: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setattr(pathlib.Path, 'home', lambda: tmp_path)
  original_stdout, original_stderr = sys.stdout, sys.stderr
  session = current.get_app_session()
  with pytest.raises(RuntimeError, match='extension failed'):
    with mirror_logging.capture_output('conversation-1'):
      print('captured')
      raise RuntimeError('extension failed')
  assert sys.stdout is original_stdout
  assert sys.stderr is original_stderr
  assert current.get_app_session() is session
  os.write(1, b'restored stdout\n')
  os.write(2, b'restored stderr\n')
  captured = capfd.readouterr()
  assert captured.out == 'restored stdout\n'
  assert captured.err == 'restored stderr\n'


def test_parse_args_version_prints_installed_version(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(sys, 'argv', ['mirror', '--version'])

  with pytest.raises(SystemExit) as exit_info:
    main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert exit_info.value.code == 0
  assert (
    capsys.readouterr().out == f'mirror {distribution.installed().version}\n'
  )


def test_remote_endpoint_requires_configuration(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv('MIRROR_REMOTE_ENDPOINT', raising=False)
  monkeypatch.setattr(sys, 'argv', ['mirror'])

  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert args.remote_endpoint is None


@pytest.mark.parametrize(
  'effort', ['none', 'low', 'medium', 'high', 'xhigh', 'max']
)
def test_parse_args_accepts_named_reasoning_effort(
  monkeypatch: pytest.MonkeyPatch, effort: str
) -> None:
  monkeypatch.setattr(sys, 'argv', ['mirror', '--reasoning-effort', effort])

  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert args.reasoning_effort == effort
  assert args.reasoning_effort_override


@pytest.mark.parametrize(
  ('arguments', 'command', 'prompt'),
  [
    ([], None, None),
    (['Explain this project'], 'exec', 'Explain this project'),
    (
      ['--model', 'test', 'Explain this project'],
      'exec',
      'Explain this project',
    ),
    (
      ['Explain this project', '--model', 'test'],
      'exec',
      'Explain this project',
    ),
    (['--', '--literal'], 'exec', '--literal'),
    (['-'], 'exec', '-'),
    ([''], 'exec', ''),
    (['exec'], 'exec', None),
    (['acp'], 'acp', None),
    (['exec', 'Explain this project'], 'exec', 'Explain this project'),
    (
      ['exec', '--model', 'test', 'Explain this project'],
      'exec',
      'Explain this project',
    ),
    (['exec', '--', '--literal'], 'exec', '--literal'),
  ],
)
def test_parse_args_routes_prompt_to_exec(
  monkeypatch: pytest.MonkeyPatch,
  arguments: list[str],
  command: str | None,
  prompt: str | None,
) -> None:
  monkeypatch.setattr(sys, 'argv', ['mirror', *arguments])

  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert args.command == command
  assert args.prompt == prompt


@pytest.mark.parametrize(
  'arguments',
  [
    ['first prompt', 'second prompt'],
    ['acp', 'unexpected prompt'],
    ['Explain this project', '--continue'],
    ['Explain this project', '--trust-extensions'],
    ['Explain this project', '--api-key', 'synthetic-cli-key'],
    ['Explain this project', '--log-level', 'debug'],
  ],
)
def test_parse_args_rejects_invalid_prompt_invocations(
  monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
  monkeypatch.setattr(sys, 'argv', ['mirror', *arguments])

  with pytest.raises(SystemExit) as exit_info:
    main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert exit_info.value.code == 2


@pytest.mark.parametrize(
  'argv',
  [
    ['mirror', '--reasoning-effort', 'lots'],
    ['mirror', '--reasoning-effort', 'min'],
    ['mirror', '--reasoning-effort', '64'],
    ['mirror', '--extra-body', '{}'],
  ],
)
def test_parse_args_rejects_unsupported_inference_options(
  monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
  monkeypatch.setattr(sys, 'argv', argv)

  with pytest.raises(SystemExit):
    main._parse_args()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
  'flag',
  [
    '--max-iterations',
    '--max-tokens',
    '--context-window',
    '--compaction-threshold',
  ],
)
def test_main_rejects_negative_worker_limits(
  tmp_path: pathlib.Path, flag: str
) -> None:
  result = subprocess.run(
    [sys.executable, '-m', 'mirror.cli.main', flag, '-1', 'exec', 'hello'],
    env={'HOME': str(tmp_path), 'TERM': 'xterm-256color'},
    cwd=tmp_path,
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 2
  assert f'{flag} cannot be negative' in result.stderr
  assert 'could not start agent' not in result.stderr


def test_resolve_conversation_creates_new_by_default(
  tmp_path: pathlib.Path,
):
  with cli_store.Store(tmp_path / 'state.db') as store:
    conversation_id, create, allow = main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
      _args(), store
    )

  assert create
  assert not allow
  assert len(conversation_id) == 32


def test_continue_without_history_creates_a_new_conversation(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.chdir(tmp_path)
  with cli_store.Store(tmp_path / 'state.db') as store:
    conversation_id, create, allow = main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
      _args(continue_recent=True), store
    )

  assert len(conversation_id) == 32
  assert create
  assert not allow


def test_resolve_exact_and_continue_require_existing_conversation(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
  monkeypatch.chdir(tmp_path)
  other = tmp_path / 'other'
  other.mkdir()
  with cli_store.Store(tmp_path / 'state.db') as store:
    store.record_user_message(
      'c1',
      entry_id='entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='saved work',
      cwd=str(tmp_path),
      model='m',
    )
    store.record_user_message(
      'elsewhere',
      entry_id='elsewhere-entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='other work',
      cwd=str(other),
      model='m',
    )

    assert main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
      _args(conversation_id='c1'), store
    ) == ('c1', False, False)
    assert main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
      _args(continue_recent=True), store
    ) == ('c1', False, False)
    with pytest.raises(SystemExit, match='does not exist'):
      main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
        _args(conversation_id='missing'), store
      )
    with pytest.raises(SystemExit, match='was started in'):
      main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
        _args(conversation_id='elsewhere'), store
      )
    assert main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
      _args(conversation_id='elsewhere', allow_foreign_workspace=True),
      store,
    ) == ('elsewhere', False, True)


def test_resolve_rejects_malformed_conversation_id_cleanly(
  tmp_path: pathlib.Path,
) -> None:
  with (
    cli_store.Store(tmp_path / 'state.db') as store,
    pytest.raises(SystemExit, match='mirror: conversation id .* must match'),
  ):
    main._resolve_conversation(  # pyright: ignore[reportPrivateUsage]
      _args(conversation_id='invalid/id'), store
    )


def test_compaction_model_uses_its_configured_route(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config = tmp_path / 'models.json'
  config.write_text(
    '[{"id":"main","label":"Main"},'
    '{"id":"compact","label":"Compact","api":"openai-responses",'
    '"base_url":"https://compact.example/v1"}]'
  )
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(config))
  monkeypatch.setattr(
    main.sys,
    'argv',
    ['mirror', '--model', 'main', '--compaction-model', 'compact'],
  )

  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert args.compaction_model_api == 'openai-responses'
  assert args.compaction_model_base_url == 'https://compact.example/v1'


@pytest.mark.parametrize(
  ('api', 'api_key'),
  [
    ('openai-chat-completions', None),
    ('anthropic-messages', 'synthetic-cli-key'),
  ],
)
def test_main_starts_one_long_lived_worker(
  tmp_path: pathlib.Path,
  monkeypatch: pytest.MonkeyPatch,
  api: str,
  api_key: str | None,
) -> None:
  args = argparse.Namespace(
    api=api,
    api_key=api_key,
    showcase=False,
    command=None,
    prompt=None,
    base_url=None,
    database=str(tmp_path / 'state.db'),
    conversation_id=None,
    continue_recent=False,
    model='m-test',
    model_api=None,
    model_base_url=None,
    approval_mode='auto',
    reasoning_effort='medium',
    scrollback_limit=1000,
    flush_scrollback=False,
    remote_endpoint='gateway.example:443',
    remote_insecure=False,
    trust_extensions=False,
  )
  monkeypatch.setattr(main, '_parse_args', lambda: args)
  monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
  monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
  monkeypatch.setenv('MIRROR_RUST_WORKER', '/checkout/mirror-agent')
  opened: list[tuple[str, bool]] = []

  async def run_tui(
    worker_command: list[str],
    worker_config: Mapping[str, object],
    *,
    reasoning_effort: str = 'medium',
    database: str | pathlib.Path | None,
    scrollback_limit: int = 1000,
    flush_scrollback: bool = False,
    continue_recent: bool = False,
    remote_endpoint: str | None = None,
    remote_secure: bool = True,
    trust_extensions: bool = False,
  ) -> None:
    assert worker_command == ['/checkout/mirror-agent']
    if api_key is not None:
      assert main.os.environ['ANTHROPIC_API_KEY'] == api_key
    assert worker_config.get('api_key') is None
    assert scrollback_limit == args.scrollback_limit
    assert flush_scrollback is args.flush_scrollback
    assert database == cli_store.default_database_path()
    assert continue_recent is args.continue_recent
    assert remote_endpoint == args.remote_endpoint
    assert remote_secure
    assert trust_extensions is args.trust_extensions
    assert worker_config['approval_mode'] == args.approval_mode
    assert worker_config['database'] == args.database
    assert 'reasoning_effort' not in worker_config
    assert 'flush_scrollback' not in worker_config
    assert 'remote_endpoint' not in worker_config
    assert 'remote_insecure' not in worker_config
    assert reasoning_effort == args.reasoning_effort
    opened.append((
      str(worker_config['conversation_id']),
      bool(worker_config['create_session']),
    ))

  monkeypatch.setattr(main.cli, 'run_tui', run_tui)

  monkeypatch.setattr(main.onboarding, 'ensure_consent', lambda _: None)
  monkeypatch.setattr(main.onboarding, 'ensure_install_id', lambda: None)
  main.main()

  assert len(opened[0][0]) == 32
  assert opened == [(opened[0][0], True)]


def test_worker_command_selects_rust_worker(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv('MIRROR_RUST_WORKER', '/custom/bin/mirror-agent')

  command = main._worker_command()  # pyright: ignore[reportPrivateUsage]

  assert command == ['/custom/bin/mirror-agent']


def test_worker_command_selects_bundled_worker(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  worker = tmp_path / 'mirror/bin/mirror-agent'
  worker.parent.mkdir(parents=True)
  worker.touch(mode=0o755)
  monkeypatch.setattr(main, '__file__', str(tmp_path / 'mirror/cli/main.py'))
  monkeypatch.delenv('MIRROR_RUST_WORKER', raising=False)

  command = main._worker_command()  # pyright: ignore[reportPrivateUsage]

  assert command == [str(worker)]


@pytest.mark.parametrize('present', [False, True])
def test_worker_command_requires_executable_worker(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
  worker = tmp_path / 'mirror/bin/mirror-agent'
  worker.parent.mkdir(parents=True)
  if present:
    worker.touch(mode=0o644)
  monkeypatch.setattr(main, '__file__', str(tmp_path / 'mirror/cli/main.py'))
  monkeypatch.delenv('MIRROR_RUST_WORKER', raising=False)

  with pytest.raises(SystemExit, match='no executable agent bundled'):
    main._worker_command()  # pyright: ignore[reportPrivateUsage]


def test_worker_resumes_from_the_configured_database(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  agent_database = tmp_path / 'state.db'
  cli_database = tmp_path / 'cli.db'
  monkeypatch.setenv('MIRROR_CLI_DB', str(cli_database))
  with cli_store.Store(agent_database) as store:
    store.record_user_message(
      'saved',
      entry_id='entry',
      sequence=0,
      timestamp=datetime.datetime.now(datetime.UTC),
      content='continue this',
      cwd=str(tmp_path),
      model='old-model',
    )
  args = argparse.Namespace(
    api='openai-codex-responses',
    api_key=None,
    showcase=False,
    command=None,
    prompt=None,
    base_url=None,
    database=str(agent_database),
    conversation_id='saved',
    continue_recent=False,
    allow_foreign_workspace=False,
    model='gpt-5.6-sol',
    model_api=None,
    model_base_url=None,
    approval_mode='auto',
    reasoning_effort='medium',
    scrollback_limit=1000,
    flush_scrollback=False,
    remote_endpoint=None,
    remote_insecure=False,
    trust_extensions=False,
  )
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(main, '_parse_args', lambda: args)
  monkeypatch.setenv('MIRROR_RUST_WORKER', '/checkout/mirror-agent')

  async def run_tui(
    worker_command: list[str],
    worker_config: Mapping[str, object],
    **kwargs: object,
  ) -> None:
    assert worker_command == ['/checkout/mirror-agent']
    assert worker_config['database'] == str(agent_database)
    assert worker_config['conversation_id'] == 'saved'
    assert worker_config['create_session'] is False
    assert kwargs['database'] == cli_database

  monkeypatch.setattr(main.cli, 'run_tui', run_tui)

  monkeypatch.setattr(main.onboarding, 'ensure_consent', lambda _: None)
  monkeypatch.setattr(main.onboarding, 'ensure_install_id', lambda: None)
  main.main()

  with cli_store.Store(cli_database) as store:
    assert store.get_info('saved').title == 'continue this'


def test_anthropic_messages_defaults_to_claude(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv('MIRROR_MODELS_CONFIG', str(tmp_path / 'models.json'))
  monkeypatch.setattr(sys, 'argv', ['mirror', '--api', 'anthropic-messages'])

  args = main._parse_args()  # pyright: ignore[reportPrivateUsage]

  assert args.api == 'anthropic-messages'
  assert args.model == 'claude-opus-4-8'


def test_anthropic_messages_reads_its_own_credentials() -> None:
  assert models.key_env('anthropic-messages') == 'ANTHROPIC_API_KEY'
  assert models.base_env('anthropic-messages') == 'ANTHROPIC_BASE_URL'


def test_a_configured_endpoint_counts_as_self_hosted() -> None:
  args = argparse.Namespace(
    base_url=None, model_base_url='http://localhost:8000/v1'
  )

  # Such a server usually needs no key, so startup must not demand one.
  assert main._custom_endpoint(args, 'OPENROUTER_BASE_URL')  # pyright: ignore[reportPrivateUsage]


def test_a_configured_protocol_decides_startup_key_requirements(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # An entry naming only a protocol runs on it, so its OpenRouter key satisfies
  # startup even when the session default is plan-billed Codex.
  args = argparse.Namespace(
    api='openai-codex-responses',
    api_key=None,
    showcase=False,
    command=None,
    prompt=None,
    base_url=None,
    database=str(tmp_path / 'state.db'),
    conversation_id=None,
    continue_recent=False,
    model='openai/gpt-5.6-sol',
    model_api='openai-chat-completions',
    model_base_url=None,
    approval_mode='auto',
    reasoning_effort='medium',
    scrollback_limit=1000,
    flush_scrollback=False,
    remote_endpoint=None,
    remote_insecure=False,
    trust_extensions=False,
  )
  monkeypatch.setattr(main, '_parse_args', lambda: args)
  monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
  monkeypatch.setenv('MIRROR_RUST_WORKER', '/checkout/mirror-agent')
  started = False

  async def run_tui(
    worker_command: list[str],
    worker_config: Mapping[str, object],
    **kwargs: object,
  ) -> None:
    del worker_config, kwargs
    nonlocal started
    assert worker_command == ['/checkout/mirror-agent']
    started = True

  monkeypatch.setattr(main.cli, 'run_tui', run_tui)

  monkeypatch.setattr(main.onboarding, 'ensure_consent', lambda _: None)
  monkeypatch.setattr(main.onboarding, 'ensure_install_id', lambda: None)
  main.main()

  assert started


@pytest.mark.asyncio
async def test_worker_startup_failure_preserves_stderr(
  tmp_path: pathlib.Path,
) -> None:
  worker = tmp_path / 'missing-worker'
  transport = local_rpc_client.Transport(
    [str(worker)],
    {},
    workspace=tmp_path,
    log_dir=tmp_path / 'logs',
  )
  with pytest.raises(RuntimeError, match='before reporting its address'):
    await transport.connect()

  assert 'FileNotFoundError' in (tmp_path / 'logs/stderr').read_text()
  assert transport.process is None


@pytest.mark.skipif(
  not os.environ.get('MIRROR_TEST_PACKAGED_WHEEL'),
  reason='Requires an installed native wheel',
)
@pytest.mark.asyncio
async def test_installed_worker_tools(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  worker = pathlib.Path(main.__file__).parents[1] / 'bin/mirror-agent'
  assert os.access(worker, os.X_OK)
  subprocess.run([worker, '--version'], check=True)
  monkeypatch.setenv('PATH', str(tmp_path))
  monkeypatch.delenv('MIRROR_RG_PATH', raising=False)
  assert shutil.which('rg') is None
  (tmp_path / 'input.txt').write_text('bundled-wheel-marker\n')
  transport = local_rpc_client.Transport(
    [str(worker)],
    {
      'conversation_id': 'wheel-smoke',
      'model': 'test-model',
      'model_api': 'openai-chat-completions',
      'base_url': 'http://127.0.0.1:1',
      'database': str(tmp_path / 'state.db'),
      'create_session': True,
      'tools': ['bash', 'grep'],
    },
    workspace=tmp_path,
    log_dir=tmp_path / 'logs',
  )
  async with (
    rpc_client.Client(
      transport,
      'wheel-smoke',
      on_message=lambda _: None,
      on_error=mock.AsyncMock(side_effect=RuntimeError('Worker RPC failed')),
    ) as client,
    asyncio.timeout(30),
  ):
    bash = await client.call_tool(
      'bash', {'command': 'printf packaged-bash-ok'}
    )
    assert bash.success and 'packaged-bash-ok' in bash.content, bash
    grep = await client.call_tool(
      'grep',
      {
        'pattern': 'bundled-wheel-marker',
        'path': 'input.txt',
        'output_mode': 'content',
      },
    )
    assert grep.success and 'bundled-wheel-marker' in grep.content, grep


def test_incompatible_database_exits_cleanly_without_changing_it(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with sqlite3.connect(database) as connection:
    connection.execute('PRAGMA user_version = 3')
  onboarding.ensure_consent(
    tmp_path,
    path=tmp_path / '.mirror' / 'config.json',
    confirm=lambda _: True,
  )
  before = database.read_bytes()
  result = subprocess.run(
    [
      sys.executable,
      '-m',
      'mirror.cli.main',
      '--model',
      'synthetic',
      '--api',
      'openai-chat-completions',
      '--base-url',
      'http://127.0.0.1:1',
    ],
    cwd=tmp_path,
    capture_output=True,
    text=True,
    timeout=20,
    check=False,
    env={
      'HOME': str(tmp_path),
      'PATH': os.environ['PATH'],
      'MIRROR_CLI_DB': str(database),
      'MIRROR_STATE_DB': str(tmp_path / 'state.db'),
      'MIRROR_MODELS_CONFIG': str(tmp_path / 'models.json'),
      'MIRROR_SKIP_UPDATE': '1',
    },
  )
  assert result.returncode != 0
  assert 'newer than this binary' in result.stderr
  assert 'Traceback' not in result.stderr
  assert database.read_bytes() == before
