"""Entrypoint for the mirror TUI."""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import sys
import uuid
from collections.abc import Mapping, Sequence

from rich import console as _console

from mirror import config as _config
from mirror import distribution, workspace
from mirror.cli import (
  cli,
  cli_store,
  noninteractive,
  onboarding,
  reflection,
  showcase,
  worktrees,
)
from mirror.cli import logging as _logging
from mirror.cli import models as _models
from mirror.cli import themes
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
  if model_id is not None and model_id == _models.default_model('reflection'):
    return _models.ModelOption(model_id, 'Reflection', api='reflection')
  return configured[0] if configured and requested is None else None


def _workspace_model() -> _models.ModelOption | None:
  """Resolve the launch worktree's selection before provider validation."""
  current = worktrees.current_worktree(pathlib.Path.cwd())
  repository_key = workspace.git_workspace_key(str(current.path)) or str(
    current.path
  )
  with cli_store.Store() as store:
    overlays = store.worktree_overlays(repository_key)
  overlay = overlays.get(current.id) or overlays.get(
    workspace.resolve_cwd(current.path)
  )
  if overlay is None or not overlay.preferences.HasField('model'):
    return None
  preferences = overlay.preferences
  configured = _models.find_model(preferences.model)
  return _models.ModelOption(
    preferences.model,
    configured.label,
    api=preferences.model_api
    if preferences.HasField('model_api')
    else configured.api,
    base_url=preferences.model_base_url
    if preferences.HasField('model_base_url')
    else configured.base_url,
  )


def _custom_endpoint(
  args: argparse.Namespace, base_env: str | None
) -> str | None:
  """The self-hosted endpoint in play: a flag, the model's entry, or the env."""
  return (
    args.base_url
    or args.model_base_url
    or (os.environ.get(base_env) if base_env else None)
  )


def _key_env(api: str) -> str:
  """The environment variable holding the API key for a wire protocol."""
  if api == 'reflection':
    return 'REFLECTION_API_KEY'
  if api == 'openai-responses':
    return 'OPENAI_API_KEY'
  if api == 'anthropic-messages':
    return 'ANTHROPIC_API_KEY'
  return 'OPENROUTER_API_KEY'


def _base_env(api: str) -> str | None:
  """The environment variable holding the base URL for a wire protocol."""
  if api == 'reflection':
    return None
  if api == 'openai-responses':
    return 'OPENAI_BASE_URL'
  if api == 'anthropic-messages':
    return 'ANTHROPIC_BASE_URL'
  return 'OPENROUTER_BASE_URL'


def _agent_database_path(configured: str | None) -> pathlib.Path:
  """Return the configured agent database path."""
  if configured is not None:
    return pathlib.Path(configured)
  if path := os.environ.get('MIRROR_STATE_DB'):
    return pathlib.Path(path)
  return pathlib.Path.home() / '.mirror' / 'state.db'


def _database_paths(
  configured_agent: str | None,
) -> tuple[pathlib.Path, pathlib.Path]:
  """Return distinct CLI and agent database paths."""
  cli_database = cli_store.default_database_path()
  agent_database = _agent_database_path(configured_agent)
  if cli_database.resolve() == agent_database.resolve():
    raise SystemExit('mirror: CLI and agent databases must use different paths')
  return cli_database, agent_database


def _parse_args() -> argparse.Namespace:
  """Parse the mirror command-line flags."""
  parser = argparse.ArgumentParser(
    prog='mirror', description='Mirror coding agent.'
  )
  parser.add_argument(
    '--update',
    action='store_true',
    help='Allow a longer startup update check (handled by the launcher).',
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
    help='AgentRPC gateway used by /remote (or $MIRROR_REMOTE_ENDPOINT).',
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
      'anthropic-messages',
    ],
    default=None,
    help=('Wire protocol to speak.'),
  )
  parser.add_argument(
    '--reasoning-effort',
    choices=['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'],
    default='max',
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
      'max_input_tokens, context_window, or meta.n_ctx). If unavailable, '
      'log a diagnostic and use the provider fallback: 256000 generally, '
      '360000 for Codex, 200000 for Anthropic. Anthropic discovers '
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
    '--conversation-id',
    default=None,
    help='Existing conversation id to resume.',
  )
  conversations.add_argument(
    '--continue',
    action='store_true',
    dest='continue_recent',
    help='Resume the most recent conversation in each worktree.',
  )
  parser.add_argument(
    '--approval-mode',
    choices=['auto', 'approve'],
    default=None,
    help='auto for automatic approvals, approve for manual review (remembered; default: approve)',
  )
  parser.add_argument(
    '--allow-foreign-workspace',
    action='store_true',
    help=(
      'Allow --conversation-id to resume a conversation started in another '
      'directory without prompting.'
    ),
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
      'Maximum terminal lines to paint when drawing history (on resume, '
      'session switch, or Ctrl+L re-render). Older history stays loaded and '
      'is omitted from the screen with a crop indicator. Set to 0 for no '
      'limit (default: 1000).'
    ),
  )
  parser.add_argument(
    '--flush-scrollback',
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
      'On every screen replace (startup, session switch, Ctrl+L, worktree '
      'tab), also drop saved terminal scrollback. `/clear` already does this. '
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
    choices=['exec'],
    help='exec runs one new local session without a TUI; approvals fail rather than prompt.',
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
  args = parser.parse_intermixed_args()
  if args.scrollback_limit < 0:
    # A negative cap would crop past the start of the history and show nothing.
    parser.error('--scrollback-limit cannot be negative (0 means no limit)')
  if args.command == 'exec':
    if args.api_key is not None:
      parser.error(
        'exec requires provider credentials from the environment or saved login, not --api-key'
      )
    if any((
      args.trust_extensions,
      args.allow_foreign_workspace,
      args.showcase,
      args.conversation_id,
      args.continue_recent,
    )):
      parser.error('exec does not load extensions or resume existing sessions')
    if args.log_level == 'debug':
      parser.error('exec does not support credential-sensitive debug logs')
    worker_workspace = os.environ.get('MIRROR_WORKSPACE')
    if (
      worker_workspace is not None
      and pathlib.Path(worker_workspace).resolve() != pathlib.Path.cwd()
    ):
      parser.error('exec uses the current directory; unset MIRROR_WORKSPACE')
  if args.showcase:
    return args
  if args.approval_mode is None:
    args.approval_mode = (
      'auto' if _settings().get('approval_mode') == 'auto' else 'approve'
    )
  elif args.command != 'exec':
    _config.save_approval_mode(args.approval_mode)
  # Explicit flags win over the config file's model entry.
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
    option = _workspace_model() or _startup_model(configured, None)
  # An entry's protocol and endpoint belong to the model it names, so they stay
  # out of the flags: switching to an entry without one must not inherit it.
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
    # Custom endpoints use Chat Completions; otherwise default to Reflection.
    has_custom_endpoint = _custom_endpoint(args, 'OPENROUTER_BASE_URL')
    args.api = (
      'openai-chat-completions' if has_custom_endpoint else 'reflection'
    )
  if args.model is None:
    args.model = _models.default_model(args.api)
  return args


def _confirm_foreign_workspace(
  info: cli_store.ConversationInfo, cwd: str
) -> bool:
  """Ask whether to resume a conversation from a different directory."""
  if not sys.stdin.isatty():
    return False
  prompt = (
    f'Conversation {info.id[:8]} was started in {info.cwd}.\n'
    f'Resume it from {cwd}? [y/N] '
  )
  try:
    answer = input(prompt)
  except EOFError:
    return False
  return answer.strip().lower() in {'y', 'yes'}


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
  config.pop('command', None)
  config.pop('prompt', None)
  config.pop('update', None)
  config.pop('no_update', None)
  config.pop('reasoning_effort')
  config.pop('flush_scrollback')
  config.pop('remote_endpoint')
  config.pop('remote_insecure')
  config['database'] = str(_agent_database_path(args.database))
  return (
    config
    | {
      'conversation_id': conversation_id,
      'create_session': create_session,
      'allow_workspace_mismatch': allow_workspace_mismatch,
      'approval_mode_override': True,
      'api_key': None,
    }
    | _memory.worker_config(_settings())
  )


def _settings() -> dict[str, object]:
  """The user's settings, or nothing when the file is missing or invalid."""
  try:
    return _config.Config(_config.default_path()).read()
  except (OSError, ValueError):
    return {}


def _resolve_conversation(
  args: argparse.Namespace,
  store: cli_store.Store,
) -> tuple[str, bool, bool]:
  """Choose conversation id, create flag, and workspace-mismatch allowance."""
  cwd = os.getcwd()
  try:
    if args.conversation_id:
      conversation_id = args.conversation_id
    elif args.continue_recent:
      info = store.latest(cwd=cwd)
      if info is None:
        return uuid.uuid4().hex, True, False
      conversation_id = info.id
    else:
      return uuid.uuid4().hex, True, False
    try:
      store.validate_workspace(conversation_id, cwd)
    except cli_store.WorkspaceMismatchError as error:
      info = store.get_info(conversation_id)
      if args.allow_foreign_workspace or _confirm_foreign_workspace(info, cwd):
        return conversation_id, False, True
      raise SystemExit(
        f'mirror: {error}. Re-run with --allow-foreign-workspace, or '
        'confirm when prompted in a terminal.'
      ) from error
  except (cli_store.SessionOpenError, ValueError) as error:
    raise SystemExit(f'mirror: {error}') from error
  return conversation_id, False, False


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
  if args.showcase:
    showcase.run(
      scrollback_limit=args.scrollback_limit,
      flush_scrollback=args.flush_scrollback,
    )
    return
  # What to authenticate is what the worker will build: the startup model's own
  # protocol when its entry names one, and the session flag for the rest.
  api = args.model_api or args.api
  # Needs an API key for the selected endpoint or a base url for self-hosted
  key_env = _key_env(api)
  base_env = _base_env(api)
  has_key = bool((args.api_key or os.environ.get(key_env, '')).strip())
  has_custom_endpoint = _custom_endpoint(args, base_env)
  if api == 'reflection':
    if (
      has_custom_endpoint
      and has_custom_endpoint.rstrip('/') != reflection.BASE_URL
    ):
      raise SystemExit(
        'mirror: --api reflection uses the Reflection endpoint; use --api openai-chat-completions for custom endpoints'
      )
  elif (
    api not in ('openai-codex-responses', 'anthropic-messages')
    and not has_key
    and not has_custom_endpoint
  ):
    try:
      theme_name = themes.saved(_config.Config(_config.default_path()).read())
    except (OSError, ValueError):
      theme_name = themes.effective('dark')
    err = _console.Console(
      stderr=True,
      highlight=False,
      theme=themes.rich_theme(theme_name),
    )
    err.print('[ui.error.bold]mirror:[/ui.error.bold] no API key found.')
    err.print(
      f'Export [ui.info]{key_env}[/ui.info].'
      if args.command == 'exec'
      else f'Export [ui.info]{key_env}[/ui.info] or pass [ui.info]--api-key[/ui.info].'
    )
    err.print(
      'To use a keyless self-hosted endpoint instead, pass '
      '[ui.info]--base-url[/ui.info].'
    )
    raise SystemExit(1)
  if args.command == 'exec':
    onboarding.ensure_consent(
      pathlib.Path.cwd(), confirm=noninteractive.require_interactive_consent
    )
    _database_paths(args.database)
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
  onboarding.ensure_consent(pathlib.Path.cwd())
  onboarding.ensure_install_id()
  if args.api_key:
    os.environ[key_env] = args.api_key
  tui_database, agent_database = _database_paths(args.database)
  with cli_store.Store(tui_database, import_from=agent_database) as store:
    conversation_id, create_session, allow_mismatch = _resolve_conversation(
      args, store
    )
  if logs is not None:
    logs.set_conversation(conversation_id)
  try:
    asyncio.run(
      cli.run_tui(
        _worker_command(),
        _worker_config(
          args,
          conversation_id,
          create_session=create_session,
          allow_workspace_mismatch=allow_mismatch,
        ),
        reasoning_effort=args.reasoning_effort,
        database=tui_database,
        scrollback_limit=args.scrollback_limit,
        flush_scrollback=args.flush_scrollback,
        continue_recent=args.continue_recent,
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
    main(_logging.configure_logging())
  except cli_store.DatabaseCompatibilityError as error:
    raise SystemExit(f'mirror: {error}') from None


if __name__ == '__main__':
  run()
