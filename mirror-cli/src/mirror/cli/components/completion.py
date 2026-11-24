"""Composer-token adapters for the shared selector renderer."""

from collections.abc import Callable, Sequence
from typing import override

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, dimension

from mirror import skills
from mirror.cli import actions, commands, selection, state, themes
from mirror.cli.components import component, menu, selector
from mirror.cli.composer import skill_completion

LOCK = '⊘'


def choices_changed(
  text: str,
  cursor: int,
  skills_completion: skill_completion.SkillCompletion,
  available_commands: Sequence[commands.CommandDef],
  *,
  has_session: bool = True,
) -> actions.CompletionChanged:
  """Snapshot completion inputs when the composer or command catalog changes."""
  if text.lstrip().startswith('!'):
    return actions.CompletionChanged((), ())
  hint = (
    commands.command_named(text[1:-1], available_commands)
    if text.startswith('/') and text.endswith(' ')
    else None
  )
  query = skill_completion.query(text, cursor)
  return actions.CompletionChanged(
    skills_completion.matches(text, cursor),
    tuple(
      _command_selection(command, text, has_session=has_session)
      for command in commands.menu_commands(text, available_commands) or ()
    ),
    hint.prompt_hint if hint is not None else None,
    hint.arguments_are_optional if hint is not None else False,
    empty_skills=not skills_completion.available
    and query is not None
    and not query.value,
  )


def _command_selection(
  command: commands.CommandDef, text: str, *, has_session: bool = True
) -> actions.CommandSelection:
  return actions.CommandSelection(
    name=command.name,
    completion=command.matching_name(text[1:]) or command.name,
    edit_arguments=(has_session or not command.requires_session)
    and command.prompt_hint is not None
    and (
      not command.arguments_are_optional or command.edit_arguments_on_select
    ),
    available_when_busy=command.available_when_busy,
    description=(
      f'{command.description} · requires an open session'
      if command.requires_session and not has_session
      else command.description
    ),
    prompt_hint=command.prompt_hint,
    arguments_are_optional=command.arguments_are_optional,
    shortcut_action=command.shortcut_action,
  )


class Completion(component.Component):
  @override
  def cache_key(self, state: state.State) -> object | None:
    # Visible choices include shortcut callbacks owned by the current view.
    return () if not visible(state) else None

  def __init__(
    self,
    shortcut_hint: Callable[[actions.Action | None], str] = lambda _: '',
  ) -> None:
    self._shortcut_hint = shortcut_hint

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    if state.selector.active:
      return containers.Window(height=dimension.Dimension.exact(0))
    skills = state.completion.skills
    if skills or state.completion.empty_skills:
      choices = tuple(
        selection.Choice(
          skill.name, f'${skill.name}', ' '.join(skill.description.split())
        )
        for skill in skills
      )
      query = skill_completion.query(state.composer.text, state.composer.cursor)
      menu_state = state.completion.skill
      title, needle = 'Skills', query.value if query else ''
    else:
      matches = state.completion.commands
      choices = tuple(
        selection.Choice(
          command.name,
          f'{"" if not state.activity.busy or command.available_when_busy else LOCK}/{command.completion}',
          command.description,
          shortcut=self._shortcut_hint(command.shortcut_action)
          if not state.activity.busy or command.available_when_busy
          else '',
        )
        for command in matches
      )
      menu_state = state.completion.slash
      title, needle = 'Commands', state.composer.text[1:]
    fragments = (
      selector.render_choices(
        menu_state,
        choices,
        title=title,
        query=needle,
        width=state.layout.columns,
        wrap_details=not skills,
        empty='No skills found. Add .agents/skills/<name>/SKILL.md.',
        footer='Esc to cancel' if state.completion.empty_skills else None,
      )
      if choices or state.completion.empty_skills
      else []
    )
    return menu.window(
      fragments,
      state.layout.columns,
      theme_name=themes.current(state.extensions, state.extension_themes),
    )


def visible(state: state.State) -> bool:
  return not state.selector.active and state.completion.active


def selected_slash_command(
  state: state.State,
) -> actions.CommandSelection | None:
  return state.completion.slash.selected_item(state.completion.commands)


def selected_skill(state: state.State) -> skills.Skill | None:
  return state.completion.skill.selected_item(state.completion.skills)


def prompt_preview(state: state.State) -> StyleAndTextTuples | None:
  if state.completion.skills:
    skill = selected_skill(state)
    query = skill_completion.query(state.composer.text, state.composer.cursor)
    if skill is None or query is None:
      return None
    suffix = (
      skill.name[len(query.value) :]
      if skill.name.lower().startswith(query.value.lower())
      else f' → ${skill.name}'
    )
    fragments: StyleAndTextTuples = (
      [('class:slash-autocomplete', suffix)] if suffix else []
    )
    return [*fragments, ('class:slash-preview-action', '  ↵')]
  text = state.composer.text
  if state.completion.argument_hint is not None:
    fragments = [
      ('class:slash-argument-placeholder', state.completion.argument_hint)
    ]
    if state.completion.argument_optional:
      fragments.append(('class:slash-preview-action', '  · optional'))
    return fragments
  command = selected_slash_command(state)
  if command is None:
    return None
  query = text[1:]
  name = command.completion
  suffix = (
    name[len(query) :]
    if name.lower().startswith(query.lower())
    else f' → /{name}'
  )
  fragments = [('class:slash-autocomplete', suffix)] if suffix else []
  if state.activity.busy and not command.available_when_busy:
    fragments.append((
      'class:slash-preview-locked',
      '  · unavailable while busy',
    ))
  elif command.prompt_hint is not None:
    fragments.extend((
      ('class:slash-preview-args', f'  {command.prompt_hint}'),
      ('class:slash-preview-action', '  ↵'),
    ))
  else:
    fragments.append(('class:slash-preview-action', '  ↵'))
  return fragments
