"""Named workspace icons in the existing user configuration."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from typing import cast

from prompt_toolkit import utils

from mirror import config
from mirror.cli import actions, selection, themes


@dataclasses.dataclass(frozen=True)
class WorkspaceTag:
  """A workspace icon with a stable ID, picker name and optional RGB color."""

  id: str
  name: str
  color: str | None = None
  icon: str = '■'

  def __post_init__(self) -> None:
    if (
      not self.icon.isprintable()
      or any(char.isspace() for char in self.icon)
      or utils.get_cwidth(self.icon) not in (1, 2)
    ):
      raise ValueError('tag icon must be a printable one- or two-cell symbol')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', self.id):
      raise ValueError(
        'tag ID must be 1-64 letters, numbers, underscores or dashes'
      )
    if (
      not self.name.strip()
      or len(self.name) > 64
      or not self.name.isprintable()
    ):
      raise ValueError('tag name must be 1-64 printable characters')
    if self.color is not None and not re.fullmatch(
      r'#[0-9a-fA-F]{6}', self.color
    ):
      raise ValueError('tag color must be #RRGGBB or omitted')


DEFAULT_TAGS = (
  WorkspaceTag('blocked', 'Red', themes.TAG_COLORS[0][0]),
  WorkspaceTag('in-progress', 'Blue', themes.TAG_COLORS[1][0]),
)


def definitions(settings: Mapping[str, object]) -> tuple[WorkspaceTag, ...]:
  """Validate definitions in their configured cycling order."""
  if 'workspace_tags' not in settings:
    return DEFAULT_TAGS
  raw = settings['workspace_tags']
  if not isinstance(raw, list):
    raise ValueError('workspace_tags must be a list')
  tags: list[WorkspaceTag] = []
  for entry in cast(list[object], raw):
    if not isinstance(entry, dict):
      raise ValueError('each workspace tag must be an object')
    fields = cast(dict[str, object], entry)
    tag_id, name, color = (
      fields.get('id'),
      fields.get('name'),
      fields.get('color'),
    )
    if (
      not isinstance(tag_id, str)
      or not isinstance(name, str)
      or (color is not None and not isinstance(color, str))
    ):
      raise ValueError(
        'workspace tags require string id/name and optional string color'
      )
    icon = fields.get('icon', '■')
    if not isinstance(icon, str):
      raise ValueError('workspace tag icon must be a string')
    tag = WorkspaceTag(tag_id, name, color, icon)
    if any(existing.id == tag.id for existing in tags):
      raise ValueError(f'duplicate workspace tag ID {tag.id!r}')
    tags.append(tag)
  return tuple(tags)


def save(settings: config.Config, tag: WorkspaceTag) -> None:
  """Create or edit a definition without changing its position or other settings."""

  def change(current: dict[str, object]) -> None:
    tags = list(definitions(current))
    index = next(
      (index for index, existing in enumerate(tags) if existing.id == tag.id),
      len(tags),
    )
    tags[index : index + 1] = [tag]
    current['workspace_tags'] = [dataclasses.asdict(item) for item in tags]

  settings.update(change)


def delete(settings: config.Config, tag_id: str) -> None:
  """Delete a definition while preserving the order of remaining tags."""

  def change(current: dict[str, object]) -> None:
    current['workspace_tags'] = [
      dataclasses.asdict(tag)
      for tag in definitions(current)
      if tag.id != tag_id
    ]

  settings.update(change)


_ICONS = (
  ('■', 'Square'),
  ('●', 'Circle'),
  ('★', 'Star'),
  ('⚑', 'Flag'),
  ('✓', 'Check'),
  ('!', 'Alert'),
  ('🚀', 'Rocket'),
)
_COLORS = (('-', 'No color'), *themes.TAG_COLORS)


def _choice(tag: WorkspaceTag, command: str) -> selection.Choice:
  return selection.Choice(
    command,
    tag.name,
    prefix=tag.icon,
    prefix_style=f'fg:{tag.color}' if tag.color else '',
  )


def menu(
  tags: tuple[WorkspaceTag, ...],
  current: WorkspaceTag | None,
) -> actions.OpenSelector:
  """Choose a tag without printing metadata into the transcript."""
  return actions.OpenSelector(
    selection.Selector(
      id='workspace-tags',
      title='Workspace tag · Ctrl+S cycles',
      choices=(
        selection.Choice('clear', 'Untagged', prefix='—'),
        *(_choice(tag, f'assign {tag.id}') for tag in tags),
        selection.Choice('new', 'New tag…', prefix='+'),
        selection.Choice('manage', 'Edit tags…'),
      ),
      current=f'assign {current.id}' if current else 'clear',
      selected=f'assign {current.id}' if current else 'clear',
      on_select=actions.WorkspaceTagCommand,
    )
  )


def manage(
  tags: tuple[WorkspaceTag, ...],
) -> actions.OpenSelector:
  """Choose the definition to edit, rather than assigning it."""
  return actions.OpenSelector(
    selection.Selector(
      id='workspace-tags-manage',
      title='Edit tags',
      choices=(
        *(_choice(tag, f'edit {tag.id}') for tag in tags),
        selection.Choice('new', 'New tag…', prefix='+'),
      ),
      on_select=actions.WorkspaceTagCommand,
      cancel_action=actions.WorkspaceTagCommand('menu'),
    )
  )


def edit(tag: WorkspaceTag) -> actions.OpenSelector:
  """Offer icon, color, name and deletion without exposing tag IDs."""
  return actions.OpenSelector(
    selection.Selector(
      id='workspace-tag-edit',
      title=f'{tag.icon} {tag.name}',
      choices=(
        selection.Choice(f'icons {tag.id}', 'Change icon…'),
        selection.Choice(f'colors {tag.id}', 'Change color…'),
        selection.Choice(f'rename {tag.id}', 'Rename…'),
        selection.Choice(f'delete {tag.id}', 'Delete tag'),
      ),
      on_select=lambda value: (
        actions.RestorePrompt(f'/tag rename {tag.id} {tag.name}')
        if value.startswith('rename ')
        else actions.WorkspaceTagCommand(value)
      ),
      cancel_action=actions.WorkspaceTagCommand('manage'),
    )
  )


def icons(tag_id: str | None = None) -> actions.OpenSelector:
  """Choose a preset or enter a custom symbol in the composer."""
  prefix = f'symbol {tag_id}' if tag_id else 'new-color'
  return actions.OpenSelector(
    selection.Selector(
      id='workspace-tag-icon',
      title='Choose an icon',
      choices=(
        *(
          selection.Choice(f'{prefix} {icon}', name, prefix=icon)
          for icon, name in _ICONS
        ),
        selection.Choice('custom', 'Custom symbol or emoji…'),
      ),
      on_select=lambda value: (
        actions.RestorePrompt(
          f'/tag symbol {tag_id} ' if tag_id else '/tag new '
        )
        if value == 'custom'
        else actions.WorkspaceTagCommand(value)
      ),
      cancel_action=actions.WorkspaceTagCommand(
        f'edit {tag_id}' if tag_id else 'menu'
      ),
    )
  )


def colors(icon: str, tag: WorkspaceTag | None = None) -> actions.OpenSelector:
  """Preview each color on the chosen icon before applying it."""
  choices = tuple(
    selection.Choice(
      f'color {tag.id} {color}' if tag else f'create {color} {icon}',
      name,
      prefix=icon,
      prefix_style=f'fg:{color}' if color != '-' else '',
    )
    for color, name in _COLORS
  )
  return actions.OpenSelector(
    selection.Selector(
      id='workspace-tag-color',
      title='Choose a color',
      choices=(*choices, selection.Choice('custom', 'Custom hex color…')),
      current=f'color {tag.id} {tag.color or "-"}' if tag else None,
      selected=f'color {tag.id} {tag.color or "-"}' if tag else None,
      on_select=lambda value: (
        actions.RestorePrompt(
          f'/tag color {tag.id} ' if tag else f'/tag create-color {icon} '
        )
        if value == 'custom'
        else actions.WorkspaceTagCommand(value)
      ),
      cancel_action=actions.WorkspaceTagCommand(
        f'edit {tag.id}' if tag else 'new'
      ),
    )
  )


def icon_name(icon: str) -> str:
  """Give a new icon a neutral, editable picker label."""
  return next((name for value, name in _ICONS if value == icon), 'Custom')
