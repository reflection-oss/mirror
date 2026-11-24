"""Built-in slash commands for the Mirror TUI."""

from __future__ import annotations

import dataclasses
import enum
import shlex
from collections.abc import Callable, Sequence

from mirror import config
from mirror.cli import actions, models, selection
from mirror.cli.components import card
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event


@dataclasses.dataclass(frozen=True)
class CommandContext:
  """Read-only REPL state a command may inspect."""

  model: str
  reasoning_effort: str
  conversation_id: str
  busy: bool
  cwd: str
  available_commands: tuple[CommandDef, ...] = ()
  config: config.JsonObject = dataclasses.field(
    default_factory=dict[str, config.JsonValue]
  )
  remote: bool = False
  session_id: str = ''
  speed: str = 'normal'
  daemon_models: tuple[models.ModelOption, ...] | None = None
  model_selection: models.ModelOption = models.ModelOption('', '')
  has_session: bool = True


type CommandHandler = Callable[[CommandContext, str], actions.Action | None]


@dataclasses.dataclass(frozen=True)
class RequiredArguments:
  """Arguments the command cannot run without, described by `hint`."""

  hint: str


@dataclasses.dataclass(frozen=True)
class OptionalArguments:
  """Optional arguments shown as `hint` before Enter runs without them."""

  hint: str


type ArgumentSpec = RequiredArguments | OptionalArguments


class CommandId(enum.Enum):
  """Bounded analytics IDs, independent of resolved command spellings."""

  CUSTOM = 'custom'
  HELP = 'help'
  STATUS = 'status'
  DAEMON_RESTART = 'daemon-restart'
  RESUME = 'resume'
  RENAME = 'rename'
  TAG = 'tag'
  REWIND = 'rewind'
  FORK = 'fork'
  WORKSPACES = 'workspaces'
  OPEN = 'open'
  WORKTREE = 'worktree'
  SANDBOX = 'sandbox'
  FOLDER = 'folder'
  CLOSE = 'close'
  DROP = 'drop'
  VIEW = 'view'
  PERMISSIONS = 'permissions'
  EXTENSIONS = 'extensions'
  COMPACT = 'compact'
  EXTENSIONS_ADD = 'extensions-add'
  EXTENSIONS_UPDATE = 'extensions-update'
  EXTENSIONS_REMOVE = 'extensions-remove'
  NEW = 'new'
  CLEAR = 'clear'
  QUIT = 'quit'


@dataclasses.dataclass(frozen=True)
class CommandDef:
  """One slash command (name without the leading `/`)."""

  name: str
  id: CommandId = dataclasses.field(default=CommandId.CUSTOM, kw_only=True)
  description: str
  handler: CommandHandler
  aliases: tuple[str, ...] = ()
  # None takes no arguments at all; see `RequiredArguments`/`OptionalArguments`.
  arguments: ArgumentSpec | None = None
  edit_arguments_on_select: bool = False
  # When False, the REPL rejects the command while a turn is in flight.
  available_when_busy: bool = False
  # Unattached UI hosts expose no tools or agent services.
  requires_session: bool = True
  hidden: bool = False
  extension_id: str | None = None
  # Reference the action, never a default key that configuration can replace.
  shortcut_action: actions.Action | None = None

  def names(self) -> tuple[str, ...]:
    return (self.name, *self.aliases)

  def matching_name(self, prefix: str) -> str | None:
    """Return the canonical name or alias matching `prefix`."""
    matches = selection.fuzzy_matches(
      self.names(), prefix, lambda name: (name,)
    )
    return matches[0] if matches else None

  def is_available(self, *, busy: bool) -> bool:
    return not busy or self.available_when_busy

  @property
  def accepts_arguments(self) -> bool:
    return self.arguments is not None

  @property
  def prompt_hint(self) -> str | None:
    """The argument hint shown while Enter waits for editable input."""
    return self.arguments.hint if self.arguments is not None else None

  @property
  def arguments_are_optional(self) -> bool:
    """Whether Enter may run the command without the hinted arguments."""
    return isinstance(self.arguments, OptionalArguments)


def resolve_commands(available: Sequence[CommandDef]) -> tuple[CommandDef, ...]:
  """Resolve namespaced definitions and expose only their winning spellings."""
  by_identity = {
    (command.extension_id, command.name): command for command in available
  }
  active = tuple(
    command
    for command in available
    if by_identity[command.extension_id, command.name] is command
  )
  by_name: dict[str, CommandDef] = {}
  for command in active:
    for name in command.names():
      by_name[name] = command
  resolved: list[CommandDef] = []
  for command in active:
    names = tuple(name for name in command.names() if by_name[name] is command)
    if names:
      resolved.append(
        dataclasses.replace(command, name=names[0], aliases=names[1:])
      )
  return tuple(resolved)


def looks_like_slash_command(text: str) -> bool:
  """Whether `text` should be treated as a slash command, not a file path."""
  if not text.startswith('/'):
    return False
  first = text.split(None, 1)[0]
  return '/' not in first[1:]


def _parse(text: str) -> tuple[str, str] | None:
  """Return `(name, args)` for a slash command, or None if not one."""
  if not looks_like_slash_command(text):
    return None
  body = text[1:].strip()
  if not body:
    return '', ''
  parts = body.split(maxsplit=1)
  return parts[0].lower(), parts[1] if len(parts) == 2 else ''


def _handle_help(context: CommandContext, args: str) -> actions.Action:
  del args
  available = tuple(
    command
    for command in (context.available_commands or COMMANDS)
    if not command.hidden
  )
  width = max((len(command.name) for command in available), default=0)
  lines = ['Available commands:']
  for command in sorted(available, key=lambda command: command.name):
    label = f'/{command.name}'
    alias = (
      f' (aliases: {", ".join(f"/{a}" for a in command.aliases)})'
      if command.aliases
      else ''
    )
    requirement = (
      ' (requires an open session)'
      if command.requires_session and not context.has_session
      else ''
    )
    lines.append(
      f'  {label:<{width + 1}}  {command.description}{alias}{requirement}'
    )
  return actions.ShowText('\n'.join(lines))


def _handle_status(context: CommandContext, args: str) -> actions.Action:
  del args
  return actions.ShowText(
    card.sections(
      'Session status',
      (
        (
          ('Model', context.model),
          ('Reasoning', context.reasoning_effort),
          ('Speed', context.speed),
        ),
        (
          ('Session', context.session_id),
          ('Conversation', context.conversation_id),
          ('Directory', context.cwd),
        ),
        (('Turn in flight', 'yes' if context.busy else 'no'),),
      ),
    )
  )


def _handle_daemon_restart(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.RestartDaemon()


def _handle_resume(_: CommandContext, args: str) -> actions.Action:
  if not args:
    return actions.ResumeSession()
  if len(args.split()) != 1:
    return actions.ShowText(
      'Conversation id must not contain spaces.', style='ui.warning'
    )
  return actions.ResumeSession(args)


def _handle_clear(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.NewSession()


def _handle_new(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.OpenNewWorkspace()


def _handle_tag(_: CommandContext, args: str) -> actions.Action:
  return actions.WorkspaceTagCommand(args or 'menu')


def _handle_rename(_: CommandContext, args: str) -> actions.Action:
  if not args:
    return actions.ShowText('Usage: /rename <alias>', style='ui.warning')
  return actions.RenameWorktree(args)


def _handle_compact(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.CompactConversation()


def _handle_rewind(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.RewindSession()


def _handle_workspaces(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.OpenWorkspaces()


def _handle_open(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.OpenWorkspaceBrowser()


def _handle_folder(_: CommandContext, args: str) -> actions.Action:
  return actions.OpenFolder(args)


def _handle_close(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.CloseWorkspace()


def _handle_fork(_: CommandContext, args: str) -> actions.Action:
  return actions.ForkWorkspace(args or None)


def _handle_worktree(_: CommandContext, args: str) -> actions.Action:
  return actions.CreateWorktree(args or None)


def _handle_sandbox(_: CommandContext, args: str) -> actions.Action:
  return actions.CreateSandbox(args or None)


def _handle_drop(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.Drop()


def _handle_extensions(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.ShowExtensions()


def _handle_extensions_add(_: CommandContext, args: str) -> actions.Action:
  try:
    arguments = shlex.split(args)
  except ValueError as error:
    return actions.ShowText(str(error), style='ui.warning')
  match arguments:
    case [source] if source:
      return actions.AddExtension(source)
    case _:
      return actions.ShowText(
        'Usage: /extensions-add <GitHub URL or host directory>',
        style='ui.warning',
      )


def _handle_extensions_update(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.UpdateExtensions()


def _handle_extensions_remove(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.RemoveExtension()


def _handle_quit(_: CommandContext, args: str) -> actions.Action:
  del args
  return actions.QuitApp()


def _handle_view(context: CommandContext, args: str) -> actions.Action:
  del context
  token = args.strip().lower()
  if not token:
    return actions.CycleViewMode()
  try:
    mode = actions.ViewMode(token)
  except ValueError:
    modes = ', '.join(mode.value for mode in actions.ViewMode)
    return actions.ShowText(
      f'Unknown view mode: {args}. Modes: {modes}', style='ui.warning'
    )
  return actions.SetViewMode(mode)


COMMANDS: tuple[CommandDef, ...] = (
  CommandDef(
    'help',
    'Show available commands',
    _handle_help,
    id=CommandId.HELP,
    available_when_busy=True,
    requires_session=False,
  ),
  CommandDef(
    'status',
    'Show session details',
    _handle_status,
    id=CommandId.STATUS,
    available_when_busy=True,
  ),
  CommandDef(
    'daemon-restart',
    'Restart the shared daemon with this installation (interrupts local workers)',
    _handle_daemon_restart,
    id=CommandId.DAEMON_RESTART,
    available_when_busy=True,
  ),
  CommandDef(
    'resume',
    'Choose a conversation',
    _handle_resume,
    id=CommandId.RESUME,
    arguments=OptionalArguments('[reference]'),
  ),
  CommandDef(
    'tag',
    'Choose or customize a workspace icon',
    _handle_tag,
    id=CommandId.TAG,
    arguments=OptionalArguments('[options]'),
    available_when_busy=True,
  ),
  CommandDef(
    'rename',
    'Set an alias for the current worktree',
    _handle_rename,
    id=CommandId.RENAME,
    arguments=RequiredArguments('<alias>'),
    available_when_busy=True,
  ),
  CommandDef(
    'rewind',
    'Go back to an earlier user turn',
    _handle_rewind,
    id=CommandId.REWIND,
  ),
  CommandDef(
    'workspaces',
    'Search and switch between workspaces',
    _handle_workspaces,
    id=CommandId.WORKSPACES,
    shortcut_action=actions.OpenWorkspaces(),
    available_when_busy=True,
    requires_session=False,
  ),
  CommandDef(
    'open',
    'Open or reopen a workspace',
    _handle_open,
    id=CommandId.OPEN,
    available_when_busy=True,
    requires_session=False,
  ),
  CommandDef(
    'folder',
    'Open a directory (absolute or relative to this workspace)',
    _handle_folder,
    id=CommandId.FOLDER,
    arguments=RequiredArguments('<path>'),
    available_when_busy=True,
    requires_session=False,
  ),
  CommandDef(
    'close',
    'Close workspace, keeping files',
    _handle_close,
    id=CommandId.CLOSE,
  ),
  CommandDef(
    'fork',
    'Fork this conversation and current files into a new worktree tab',
    _handle_fork,
    id=CommandId.FORK,
    arguments=OptionalArguments('[name]'),
    edit_arguments_on_select=True,
  ),
  CommandDef(
    'worktree',
    'Create a worktree workspace from the remote default branch',
    _handle_worktree,
    id=CommandId.WORKTREE,
    arguments=OptionalArguments('[name]'),
    edit_arguments_on_select=True,
    available_when_busy=True,
  ),
  CommandDef(
    'sandbox',
    'Create a sandboxed workspace',
    _handle_sandbox,
    id=CommandId.SANDBOX,
    arguments=OptionalArguments('[name]'),
    edit_arguments_on_select=True,
    available_when_busy=True,
  ),
  CommandDef(
    'drop',
    'Delete this managed workspace',
    _handle_drop,
    id=CommandId.DROP,
  ),
  CommandDef(
    'extensions',
    'Enable or disable extensions',
    _handle_extensions,
    id=CommandId.EXTENSIONS,
    available_when_busy=True,
  ),
  CommandDef(
    'view',
    'Cycle view: detail, default, focus',
    _handle_view,
    id=CommandId.VIEW,
    shortcut_action=actions.CycleViewMode(),
    arguments=OptionalArguments('[mode]'),
    available_when_busy=True,
  ),
  CommandDef(
    'compact',
    'Compact conversation context',
    _handle_compact,
    id=CommandId.COMPACT,
  ),
  CommandDef(
    'extensions-add',
    'Add a GitHub or host-local extension source',
    _handle_extensions_add,
    id=CommandId.EXTENSIONS_ADD,
    arguments=RequiredArguments('<GitHub URL or host directory>'),
    available_when_busy=True,
  ),
  CommandDef(
    'extensions-update',
    'Choose GitHub extensions to update',
    _handle_extensions_update,
    id=CommandId.EXTENSIONS_UPDATE,
    available_when_busy=True,
  ),
  CommandDef(
    'extensions-remove',
    'Choose an extension source to remove',
    _handle_extensions_remove,
    id=CommandId.EXTENSIONS_REMOVE,
    available_when_busy=True,
  ),
  CommandDef(
    'clear',
    'Start a fresh conversation in this workspace; retain its history',
    _handle_clear,
    id=CommandId.CLEAR,
  ),
  CommandDef(
    'new',
    'Create a new workspace',
    _handle_new,
    id=CommandId.NEW,
    available_when_busy=True,
    requires_session=False,
  ),
  CommandDef(
    'quit',
    'Exit mirror',
    _handle_quit,
    id=CommandId.QUIT,
    shortcut_action=actions.QuitRequested(),
    aliases=('exit',),
    available_when_busy=True,
    requires_session=False,
  ),
)


def menu_commands(
  text: str, available: Sequence[CommandDef] | None = None
) -> tuple[CommandDef, ...] | None:
  """Commands matching the name currently being typed.

  Shown for `/` and `/partial` on a single line with no arguments yet. Hidden
  once a space introduces args, or when the token looks like a path.
  """
  if '\n' in text or not text.startswith('/'):
    return None
  query = text[1:]
  if any(character.isspace() for character in query) or '/' in query:
    return None
  commands_to_search = COMMANDS if available is None else available
  return selection.fuzzy_matches(
    sorted(
      (command for command in commands_to_search if not command.hidden),
      key=lambda command: command.name.casefold(),
    ),
    query,
    CommandDef.names,
  )


def command_named(
  name: str, available: Sequence[CommandDef] | None = None
) -> CommandDef | None:
  """Look up a command by canonical name or alias."""
  needle = name.lower()
  commands_to_search = COMMANDS if available is None else available
  return next(
    (
      command
      for command in reversed(commands_to_search)
      if needle in command.names()
    ),
    None,
  )


def selected_submission(
  command: actions.CommandSelection, *, busy: bool = False
) -> str | actions.RestorePrompt:
  """Complete required arguments or submit the selected command."""
  if command.edit_arguments and (not busy or command.available_when_busy):
    return actions.RestorePrompt(f'/{command.completion} ')
  return f'/{command.name}'


def dispatch(
  text: str,
  context: CommandContext,
  available: Sequence[CommandDef] | None = None,
) -> actions.Action | None:
  """Run a slash command, returning its action when it has one."""
  parsed = _parse(text)
  if parsed is None:
    return None
  name, args = parsed
  if not name:
    return actions.ShowText(
      'Type a command name, or /help for the list.', style='ui.warning'
    )
  available_commands = tuple(COMMANDS if available is None else available)
  command = command_named(name, available_commands)
  if command is None:
    return actions.ShowText(
      f'Unknown command: /{name}. Type /help.', style='ui.error'
    )
  if command.requires_session and not context.has_session:
    return actions.ShowText(
      f'/{command.name} requires an open session.', style='ui.warning'
    )
  if not command.is_available(busy=context.busy):
    return actions.ShowText(
      f'/{command.name} is unavailable while a turn is in progress.',
      style='ui.warning',
    )
  if args and not command.accepts_arguments:
    return actions.ShowText(
      f'/{command.name} does not accept arguments.', style='ui.warning'
    )
  action = command.handler(
    dataclasses.replace(context, available_commands=available_commands), args
  )

  # ExtensionHost counts contributed handlers inside its fault boundary.
  if command.extension_id is None:
    statsig_client.get().log_event(
      statsig_event.Event(
        'mirror_slash_command_dispatched',
        value=1,
        metadata={'command': command.id.value},
      )
    )
  return action
