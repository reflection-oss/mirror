"""Entrypoint for the mirror TUI."""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import uuid
from collections.abc import Mapping, Sequence

from mirror import config as _config
from mirror import distribution
from mirror.cli import approval_modes
from mirror.cli import (
  cli,
  local_rpc_client,
  noninteractive,
  onboarding,
  showcase,
  themes,
)
from mirror.cli import logging as _logging
from mirror.cli import models as _models
from mirror.cli.extensions.memory import extension as _memory


def _startup_model(
  configured: Sequence[_models.ModelOption], requested: str | None
) -> _models.ModelOption | None:
  """The configured entry to start from, so its endpoint and protocol apply."""
  if (
    requested is None
    and (selection := _models.load_last_selection()) is not None
  ):
    return selection
  model_id = requested if requested is not None else _models.load_last_model()
  option = next((o for o in configured if o.model_id == model_id), None)
  if option is not None:
    return option
  if model_id == _models.GEMINI_MODEL:
    return _models.ModelOption(model_id, 'Gemini 4 Argon', api='gemini')
  if model_id is not None and _models.is_reflection_default(model_id):
    return _models.ModelOption(model_id, 'Reflection', api='reflection')
  return configured[0] if configured and requested is None else None


def _custom_endpoint(
  args: argparse.Namespace, base_env: str | None
) -> str | None:
  """The self-hosted endpoint in play: a flag, the model's entry, or the env."""
  return (
    args.base_url
    or args.model_base_url
    or (os.environ.get(base_env) if base_env else None)
  )


def _agent_database_path(configured: str | None) -> pathlib.Path:
  """Return the configured agent database path."""
  if configured is not None:
    return pathlib.Path(configured)
  if path := os.environ.get('MIRROR_STATE_DB'):
    return pathlib.Path(path)
  return pathlib.Path.home() / '.mirror' / 'state.db'


def _parse_args() -> argparse.Namespace:
  """Parse the mirror command-line flags."""
  parser = argparse.ArgumentParser(
    prog='mirror',
    description='Mirror coding agent.',
    epilog='Use mirror update to check for updates without starting a session.',
  )
  parser.add_argument(
    '--no-update',
    action='store_true',
    help='Skip the startup update check; use the selected release.',
  )
  parser.add_argument(
    '--version',
    action='version',
    version=f'%(prog)s {distribution.installed().version}',
  )
  parser.add_argument(
    '--remote-endpoint',
    default=os.environ.get('MIRROR_REMOTE_ENDPOINT') or None,
    help=(
      'AgentRPC gateway (or $MIRROR_REMOTE_ENDPOINT) for /remote, which the '
      'remotes extension provides; it is off by default, enable it in '
      '/extensions.'
    ),
  )
  parser.add_argument(
    '--remote-insecure',
    action='store_true',
    help='Use plaintext gRPC for a loopback --remote-endpoint.',
  )
  parser.add_argument(
    '--model',
    default=None,
    help='Model id.',
  )
  parser.add_argument(
    '--compaction-model',
    default=None,
    help=(
      'Model id used to summarize history when compacting. Defaults to --model.'
    ),
  )
  parser.add_argument(
    '--api-key',
    default=None,
    help='API key for the selected provider (defaults to its environment variable).',
  )
  parser.add_argument(
    '--base-url',
    default=None,
    help=(
      'Base URL of an OpenAI-compatible endpoint (e.g. an internal model '
      'server). Defaults to OpenRouter (or $OPENROUTER_BASE_URL).'
    ),
  )
  parser.add_argument(
    '--api',
    choices=[
      'reflection',
      'openai-codex-responses',
      'openai-responses',
      'openai-chat-completions',
      'openai-chat-completions-non-streaming',
      'openrouter',
      'gemini',
      'vercel',
      'anthropic-messages',
    ],
    default=None,
    help=('Wire protocol to speak.'),
  )
  parser.add_argument(
    '--reasoning-effort',
    choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'],
    default=None,
    help='Reasoning effort for model requests (default: max).',
  )
  parser.add_argument(
    '--max-iterations',
    type=int,
    default=None,
    help='Maximum model/tool loop iterations (default: unlimited).',
  )
  parser.add_argument(
    '--compaction-threshold',
    type=int,
    default=None,
    help=(
      'Prompt-token count that triggers history compaction. When omitted, '
      'compacts at 90%% of --context-window: 5%% is reserved for safety '
      'and 5%% for a completion (or --max-tokens, if that is smaller).'
    ),
  )
  parser.add_argument(
    '--max-tokens',
    type=int,
    default=64000,
    help=(
      'Soft ceiling on completion tokens per request. Shrinks so prompt plus '
      'completion stay within 90%% of --context-window.'
    ),
  )
  parser.add_argument(
    '--stream',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Stream model output as it is generated (default: enabled).',
  )
  parser.add_argument(
    '--stream-usage',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Request token usage in streaming responses (default: enabled).',
  )
  parser.add_argument(
    '--strict-finish-reason',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Reject responses without a standard finish reason (default: enabled).',
  )
  parser.add_argument(
    '--context-window',
    type=int,
    default=None,
    help=(
      'Override the model context length. By default the Rust worker reads '
      'GET /models for the selected model (max_model_len, context_length, '
      'max_input_tokens, context_window, meta.n_ctx, or meta.n_ctx_train). '
      'If unavailable, log a diagnostic and use the provider fallback: '
      '256000 generally, 272000 for Codex, 200000 for Anthropic. Anthropic discovers '
      'max_input_tokens through GET /v1/models/{model}. Completion budgets '
      'and compaction use the resolved limit.'
    ),
  )
  parser.add_argument(
    '--database',
    default=None,
    help=(
      'SQLite agent database (default: $MIRROR_STATE_DB, or '
      '~/.mirror/state.db).'
    ),
  )
  conversations = parser.add_mutually_exclusive_group()
  conversations.add_argument(
    '--showcase',
    action='store_true',
    help='Inspect synthetic history and live playback without a model or real tools.',
  )
  conversations.add_argument(
    '--continue',
    dest='continue_session',
    action='store_true',
    help='Restore the last tab layout for this directory (the default).',
  )
  conversations.add_argument(
    '--conversation-id',
    default=None,
    help='Existing conversation id to resume.',
  )
  parser.add_argument(
    '--approval-mode',
    choices=['approve', 'auto', 'bypass'],
    default=None,
    help='approve for manual review, auto for automatic approvals, bypass to run tools without approval (remembered; default: approve)',
  )
  parser.add_argument(
    '--log-level',
    choices=['debug', 'info', 'warning', 'error', 'critical'],
    default='info',
    help=(
      'Worker log level. Logs are written to '
      '~/.mirror/logs/<conversation>/stderr, not the transcript. Avoid '
      '`debug`, which logs request URLs/headers (leaking the API key).'
    ),
  )
  parser.add_argument(
    '--scrollback-limit',
    type=int,
    default=1000,
    help=(
      'Maximum terminal lines to paint when drawing history (on resume '
      'or session switch). Older history stays loaded and '
      'is omitted from the screen with a crop indicator. Set to 0 for no '
      'limit (default: 1000).'
    ),
  )
  parser.add_argument(
    '--flush-scrollback',
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      'On every screen replace (startup, session switch, worktree '
      'tab), also drop saved terminal scrollback. '
      'Default: keep scrollback.'
    ),
  )
  parser.add_argument(
    '--trust-extensions',
    action='store_true',
    help='Remember new external extensions as enabled without prompting; keep disabled ones off.',
  )
  parser.add_argument(
    'command',
    nargs='?',
    metavar='PROMPT|exec|acp|daemon',
    help=(
      'A prompt or exec runs one new local session without a TUI; approvals '
      'fail rather than prompt. acp serves the Agent Client Protocol on '
      'stdin/stdout for clients such as Omnigent or Zed; approvals are '
      'forwarded to the client. daemon start|stop|status manages the shared backend; '
      'daemon address prints the Unix socket address of an already running daemon.'
    ),
  )
  parser.add_argument(
    'prompt',
    nargs='?',
    metavar='PROMPT',
    help=(
      'Instructions for exec. Omit or use - to read stdin; '
      'piped stdin alongside an argument is appended as a <stdin> block.'
    ),
  )
  args, remaining = parser.parse_known_args()
  if remaining:
    args = parser.parse_intermixed_args()
  if args.command == 'daemon':
    if args.prompt not in ('start', 'stop', 'status', 'address'):
      parser.error('expected daemon start, stop, status, or address')
    if args.prompt != 'address':
      return args
  if args.command is not None and args.command not in ('exec', 'acp', 'daemon'):
    if args.prompt is not None:
      parser.error('expected one prompt; quote prompts containing spaces')
    args.prompt = args.command
    args.command = 'exec'
  args.reasoning_effort_override = args.reasoning_effort is not None
  if args.reasoning_effort is None:
    args.reasoning_effort = 'max'
  for flag, value in (
    ('--max-iterations', args.max_iterations),
    ('--max-tokens', args.max_tokens),
    ('--context-window', args.context_window),
    ('--compaction-threshold', args.compaction_threshold),
  ):
    if value is not None and value < 0:
      parser.error(f'{flag} cannot be negative')
  if args.scrollback_limit < 0:
    parser.error('--scrollback-limit cannot be negative (0 means no limit)')
  if args.command == 'acp' and args.prompt is not None:
    parser.error('acp takes no prompt; the client sends prompts over stdio')
  if args.command in ('exec', 'acp'):
    if args.api_key is not None:
      parser.error(
        f'{args.command} requires provider credentials from the environment or saved login, not --api-key'
      )
    if any((
      args.trust_extensions,
      args.showcase,
      args.continue_session,
      args.conversation_id,
    )):
      parser.error(
        f'{args.command} does not load extensions or resume existing sessions'
      )
    if args.log_level == 'debug':
      parser.error(
        f'{args.command} does not support credential-sensitive debug logs'
      )
    worker_workspace = os.environ.get('MIRROR_WORKSPACE')
    if (
      args.command == 'exec'
      and worker_workspace is not None
      and pathlib.Path(worker_workspace).resolve() != pathlib.Path.cwd()
    ):
      parser.error('exec uses the current directory; unset MIRROR_WORKSPACE')
  if args.showcase:
    return args
  args.approval_mode_override = args.approval_mode is not None
  if args.approval_mode is None:
    args.approval_mode = (
      saved
      if (saved := _settings().get('approval_mode')) in approval_modes.NAMES
      else 'approve'
    )
  elif args.command not in ('exec', 'acp'):
    _config.save_approval_mode(args.approval_mode)
  explicit_api = args.api is not None
  explicit_base_url = args.base_url is not None
  try:
    configured = _models.load_models()
  except (ValueError, OSError) as error:
    raise SystemExit(f'mirror: {error}') from error
  args.model_selection_override = any((
    args.model is not None,
    explicit_api,
    explicit_base_url,
  ))
  if args.model is not None:
    option = _startup_model(configured, args.model)
  elif explicit_api:
    option = _models.ModelOption(
      _models.default_model(args.api), args.api, api=args.api
    )
  else:
    option = _startup_model(configured, None)
  args.model_api = None
  args.model_base_url = None
  args.compaction_model_api = None
  args.compaction_model_base_url = None
  if option is not None:
    args.model = option.model_id
    if option.api and not explicit_api:
      args.model_api = option.api
    if option.base_url and not explicit_base_url:
      args.model_base_url = option.base_url
  if args.compaction_model is not None:
    compaction = _startup_model(configured, args.compaction_model)
    if compaction is not None:
      args.compaction_model_api = compaction.api or None
      args.compaction_model_base_url = compaction.base_url or None
  if args.api == 'openrouter':
    args.api = 'openai-chat-completions'
  if args.api is None:
    has_custom_endpoint = _custom_endpoint(args, 'OPENROUTER_BASE_URL')
    args.api = (
      'openai-chat-completions' if has_custom_endpoint else 'reflection'
    )
  if args.model is None:
    args.model = _models.default_model(args.api)
  if not args.reasoning_effort_override:
    args.reasoning_effort = _models.inherited_reasoning_effort(
      args.model_api or args.api, args.reasoning_effort
    )
  return args


def _worker_config(
  args: argparse.Namespace,
  conversation_id: str,
  *,
  create_session: bool,
  allow_workspace_mismatch: bool,
) -> Mapping[str, object]:
  """Return subprocess configuration without client settings or secrets."""
  config = dict(vars(args))
  config.pop('showcase')
  config.pop('continue_session', None)
  config.pop('command', None)
  config.pop('prompt', None)
  config.pop('update', None)
  config.pop('no_update', None)
  config.pop('reasoning_effort')
  config.pop('flush_scrollback')
  config.pop('remote_endpoint')
  config.pop('remote_insecure')
  config['database'] = str(_agent_database_path(args.database))
  settings = _config.Config(_config.default_path()).read()
  return (
    config
    | {
      'conversation_id': conversation_id,
      'create_session': create_session,
      'allow_workspace_mismatch': allow_workspace_mismatch,
      'api_key': None,
      'web_provider': _config.web_provider(settings),
      'root_search_guard': True,
    }
    | _memory.worker_config(settings)
  )


def _settings() -> dict[str, object]:
  """The user's settings, or nothing when the file is missing or invalid."""
  try:
    return _config.Config(_config.default_path()).read()
  except (OSError, ValueError):
    return {}


def _worker_command() -> list[str]:
  """Return the Rust AgentRPC worker command."""
  worker = os.environ.get('MIRROR_RUST_WORKER')
  if worker is not None:
    return [worker]
  bundled = pathlib.Path(__file__).parents[1] / 'bin' / 'mirror-agent'
  if bundled.is_file() and os.access(bundled, os.X_OK):
    return [str(bundled)]
  raise SystemExit(
    'mirror: no executable agent bundled; install a platform wheel or '
    'set MIRROR_RUST_WORKER to a built mirror-agent executable'
  )


def main(logs: _logging.Logs | None = None) -> None:
  """Parse flags and run the TUI over the agent worker subprocess."""
  args = _parse_args()
  if args.command == 'daemon':
    transport = local_rpc_client.Transport(
      _worker_command(),
      dict(
        _worker_config(
          args, '', create_session=False, allow_workspace_mismatch=False
        )
      )
      | {'reasoning_effort': args.reasoning_effort}
      if args.prompt == 'address'
      else {'database': str(_agent_database_path(args.database))},
      workspace=pathlib.Path.cwd(),
      log_dir=pathlib.Path.cwd(),
    )
    try:
      print(asyncio.run(transport.manage(args.prompt)))
    except local_rpc_client.DaemonConnectionError as error:
      raise SystemExit(f'mirror: {error}') from None
    return
  logs = logs or _logging.configure_logging()
  if args.showcase:
    if themes.preference(_settings()) == themes.AUTO:
      themes.detect_background()
    showcase.run(
      scrollback_limit=args.scrollback_limit,
      flush_scrollback=args.flush_scrollback,
    )
    return
  api = args.model_api or args.api
  if args.command in ('exec', 'acp'):
    try:
      _models.validate_selection(
        _models.ModelOption(
          args.model,
          args.model,
          api=api,
          base_url=args.base_url or args.model_base_url or '',
        ),
        args.reasoning_effort,
        max_tokens=args.max_tokens,
      )
      _models.validate_connection(
        api, args.base_url or args.model_base_url or ''
      )
    except ValueError as error:
      raise SystemExit(f'mirror: {error}') from None
  if args.command == 'acp':
    from mirror.cli import acp

    raise SystemExit(
      acp.run(
        _worker_command(),
        lambda conversation_id: _worker_config(
          args,
          conversation_id,
          create_session=True,
          allow_workspace_mismatch=False,
        ),
        reasoning_effort=args.reasoning_effort,
      )
    )
  if args.command == 'exec':
    try:
      onboarding.ensure_consent(
        pathlib.Path.cwd(), confirm=noninteractive.require_interactive_consent
      )
    except ValueError as error:
      raise SystemExit(f'mirror: {error}') from None
    raise SystemExit(
      noninteractive.run(
        _worker_command(),
        _worker_config(
          args,
          uuid.uuid4().hex,
          create_session=True,
          allow_workspace_mismatch=False,
        ),
        args.prompt,
        reasoning_effort=args.reasoning_effort,
      )
    )
  try:
    onboarding.ensure_consent(pathlib.Path.cwd())
  except ValueError as error:
    raise SystemExit(f'mirror: {error}') from None
  onboarding.ensure_install_id()
  if args.api_key:
    os.environ[_models.key_env(api)] = args.api_key
  if args.conversation_id:
    logs.set_conversation(args.conversation_id)
  if themes.preference(_settings()) == themes.AUTO:
    themes.detect_background()
  try:
    asyncio.run(
      cli.run_tui(
        _worker_command(),
        _worker_config(
          args,
          args.conversation_id or '',
          create_session=False,
          allow_workspace_mismatch=False,
        ),
        reasoning_effort=args.reasoning_effort,
        selected_session=args.conversation_id,
        scrollback_limit=args.scrollback_limit,
        flush_scrollback=args.flush_scrollback,
        remote_endpoint=args.remote_endpoint,
        remote_secure=not args.remote_insecure,
        trust_extensions=args.trust_extensions,
      )
    )
  except KeyboardInterrupt:
    sys.exit(130)


def run() -> None:
  """Console-script entry point (`mirror`)."""
  try:
    main()
  except local_rpc_client.DaemonConnectionError as error:
    raise SystemExit(f'mirror: {error}') from None


if __name__ == '__main__':
  run()
