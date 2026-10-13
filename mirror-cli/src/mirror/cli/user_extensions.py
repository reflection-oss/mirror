"""Core extension discovery, consent, settings, and lifecycle management."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import logging
import os
import pathlib

from mirror import config, project
from mirror.cli import (
  actions,
  extension,
  extension_sources,
  selection,
)
from mirror.cli.statsig.events import client as statsig_client
from mirror.cli.statsig.events import event as statsig_event


def _description(directory: pathlib.Path) -> str:
  """Read extension metadata without executing disabled or unapproved code."""
  try:
    module = ast.parse((directory / 'extension.py').read_text())
    description = ast.get_docstring(module)
    return (
      description.splitlines()[0] if description else 'No description provided.'
    )
  except (OSError, SyntaxError, UnicodeError):
    return 'Description unavailable.'


class UserExtensions:
  """Own extension discovery, settings, consent, and lifecycle management."""

  def __init__(
    self,
    registry: extension.ExtensionRegistry,
    root: pathlib.Path,
    *,
    cwd: pathlib.Path | None = None,
    bundled: pathlib.Path | None = None,
    settings: config.Config | None = None,
    packaged: pathlib.Path | None = None,
  ) -> None:
    self.registry = registry
    self.root = root.resolve()
    self._packaged = (
      packaged
      if packaged is not None
      else pathlib.Path(__file__).with_name('packaged_extensions')
    ).resolve()
    self._sources = extension_sources.Sources(self.root)
    self._cwd = cwd
    self._bundled = bundled.resolve() if bundled is not None else None
    self._config = (
      settings
      if settings is not None
      else config.Config(self.root.parent / 'config.json')
    )
    self._applied: dict[str, config.ExtensionSettings] | None = None
    self._apply_lock = asyncio.Lock()
    self._foreground: extension.Context | None = None
    self._pending_approval: dict[str, pathlib.Path] = {}

  def _key(self, directory: pathlib.Path) -> str:
    if key := self._sources.key(directory):
      return key
    if directory.parent == self._bundled:
      return f'builtin/{directory.name}'
    if directory.parent == self._packaged:
      return f'packaged/{directory.name}'
    return str((directory / 'extension.py').resolve())

  def _directories(
    self, *, ignore_environment: bool = False
  ) -> dict[str, pathlib.Path]:
    roots = (
      project.resource_directories(self._cwd, 'extensions', self.root)
      if self._cwd is not None
      else [self.root]
    )
    if (
      not ignore_environment
      and os.getenv('MIRROR_EXTENSIONS_ENABLED', 'true').lower() != 'true'
    ):
      roots = []
    else:
      roots.insert(0, self._packaged)
    found: dict[str, pathlib.Path] = {}
    if roots:
      found.update({
        directory.name: directory
        for directory in self._sources.directories(
          self._config.read().get('extensions')
        ).values()
      })
    for root in roots:
      if root.is_dir():
        for directory in sorted(root.iterdir()):
          if (directory / 'extension.py').is_file():
            directory = directory.resolve()
            found[directory.name] = directory
    directories: dict[str, pathlib.Path] = {}
    if self._bundled is not None and self._bundled.is_dir():
      for directory in sorted(self._bundled.iterdir()):
        if (directory / 'extension.py').is_file():
          directories[self._key(directory)] = directory
    directories.update({
      self._key(directory): directory for directory in found.values()
    })
    return directories

  def _trusted(self, directory: pathlib.Path) -> bool:
    return directory.parent in (self._bundled, self._packaged)

  def _emit_metrics(self) -> None:
    """Report applied inventory after settings changes, not workspace focus."""
    if self._applied is None:
      return
    counts = {
      'enabled': 0,
      'disabled': 0,
      'hidden': 0,
      'external_enabled': 0,
      'external_disabled': 0,
    }
    extension_state_events: list[statsig_event.Event] = []

    for key, settings in self._applied.items():
      if settings.enabled == 'hidden':
        state = 'hidden'
      elif settings.enabled:
        state = 'enabled'
      else:
        state = 'disabled'
      counts[state] += 1

      if key.startswith(('builtin/', 'packaged/')):
        extension_state_events.append(
          statsig_event.Event(
            'mirror_extension_state', value=key, metadata={'state': state}
          )
        )
      elif state == 'enabled':
        counts['external_enabled'] += 1
      elif state == 'disabled':
        counts['external_disabled'] += 1

    client = statsig_client.get()
    for name, count in counts.items():
      client.log_event(
        statsig_event.Event(f'mirror_extension_{name}_count', value=count)
      )
    for event in extension_state_events:
      client.log_event(event)

  async def start(self, *, trust_extensions: bool = False) -> None:
    """Load enabled extensions; ask once about newly discovered external code."""
    entries = self._config.read().get('extensions')
    if os.getenv('MIRROR_EXTENSIONS_ENABLED', 'true').lower() == 'true':
      await self._sources.prepare(entries)
    for key, directory in self._directories().items():
      try:
        overrides = config.extension_overrides(entries, key)
        settings = config.resolve_extension(overrides)
      except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning(
          'Ignoring config for extension %s: %s', key, error
        )
        continue
      if 'enabled' not in overrides:
        if self._trusted(directory) or settings.enabled is not True:
          continue
        if trust_extensions:
          self._config.set_extension(key, True)
        else:
          self._pending_approval[key] = directory
    await self.apply_settings(self._config.read().get('extensions'))
    if self._foreground is not None:
      self._show_approval(self._foreground)

  async def apply_settings(self, settings: object) -> None:
    """Enable and disable extensions to match their configured settings."""
    async with self._apply_lock:
      if os.getenv('MIRROR_EXTENSIONS_ENABLED', 'true').lower() == 'true':
        await self._sources.prepare(settings)
      directories = self._directories()
      previous = self._applied or {}
      loaded_directories = {
        name: self.registry.source_directory(name)
        for name in self.registry.extension_ids()
      }
      for name, directory in loaded_directories.items():
        if directory is not None:
          key = self._key(directory)
          if key in previous and directories.get(key) != directory:
            await self.registry.unload(name)
      self._pending_approval = {
        key: directory
        for key, directory in self._pending_approval.items()
        if directories.get(key) == directory
      }
      applied: dict[str, config.ExtensionSettings] = {}
      for key, directory in directories.items():
        await asyncio.sleep(0)
        try:
          overrides = config.extension_overrides(settings, key)
          resolved = config.resolve_extension(overrides)
          if 'enabled' in overrides or resolved.enabled is not True:
            self._pending_approval.pop(key, None)
          if (
            not self._trusted(directory)
            and resolved.enabled is True
            and overrides.get('enabled') is not True
          ):
            resolved = dataclasses.replace(resolved, enabled=False)
          try:
            loaded = await self._apply_extension(
              directory, resolved.enabled is True, overrides
            )
          except extension.ExtensionLoadError:
            loaded = (
              None  # The registry logged the diagnostic; keep loading others.
            )
          if resolved.enabled is True and loaded is None:
            resolved = dataclasses.replace(
              resolved,
              enabled='hidden'
              if 'enabled' not in overrides
              and self.registry.default_enabled(directory) == 'hidden'
              else False,
            )
          applied[key] = resolved
        except (OSError, ValueError) as error:
          logging.getLogger(__name__).warning(
            'Ignoring config for extension %s: %s', key, error
          )
          if key in previous:
            applied[key] = previous[key]
      self._applied = applied
      self._refresh_selector()
      self._emit_metrics()

  async def on_foreground(self, context: extension.Context) -> None:
    self._foreground = context
    self._refresh_selector()
    self._show_approval(context)

  async def on_background(self, context: extension.Context) -> None:
    if (
      self._foreground is not None
      and self._foreground.dispatch is context.dispatch
    ):
      self._foreground = None

  def _refresh_selector(self) -> None:
    context = self._foreground
    if context is None:
      return
    current = context.get_state().selector
    if current.selector is not None and current.selector.id == 'extensions':
      choice = current.selected_choice()
      context.dispatch(
        self.selector(selected=choice.value if choice is not None else None)
      )

  def _show_approval(self, context: extension.Context) -> None:
    if not self._pending_approval:
      return
    key, directory = next(iter(self._pending_approval.items()))
    context.dispatch(
      actions.OpenSelector(
        selection.Selector(
          'extension-approval',
          f'Enable {directory.name}? Runs unsandboxed code with your permissions. {key}',
          (
            selection.Choice('disable', 'Disable', key),
            selection.Choice('enable', 'Enable', key),
          ),
          lambda choice: actions.ExtensionConsent(key, choice == 'enable'),
          selected='disable',
          numbered=True,
          cancel_action=actions.ExtensionConsent(key, None),
        )
      )
    )

  def _choices(self) -> dict[str, bool]:
    return (
      {key: settings.enabled is True for key, settings in self._applied.items()}
      if self._applied is not None
      else config.extension_choices(self._config.read().get('extensions'))
    )

  def _hidden(self, key: str, directory: pathlib.Path) -> bool:
    if self._applied is not None and key in self._applied:
      return self._applied[key].enabled == 'hidden'
    try:
      overrides = config.extension_overrides(
        self._config.read().get('extensions'), key
      )
      return config.resolve_extension(overrides).enabled == 'hidden'
    except (OSError, ValueError):
      return False

  def _detail(self, key: str, directory: pathlib.Path) -> str:
    if error := self._sources.errors.get(key):
      return f'{key} · {error}'
    description = self.registry.description(directory) or _description(
      directory
    )
    return (
      f'{key} · {description}' if self._sources.key(directory) else description
    )

  def selector(self, selected: str | None = None) -> actions.OpenSelector:
    """Build the core extension selector, independent of loaded extensions."""
    choices = self._choices()
    return actions.OpenSelector(
      selection.Selector(
        'extensions',
        'Extensions · Enter toggles',
        tuple(
          selection.Choice(
            key,
            directory.name,
            self._detail(key, directory),
            prefix='●' if choices.get(key, False) else '○',
            prefix_style='class:extension-on'
            if choices.get(key, False)
            else 'class:extension-off',
          )
          for key, directory in sorted(
            self._directories().items(),
            key=lambda item: (item[1].name.casefold(), item[0]),
          )
          if not self._hidden(key, directory)
        ),
        actions.ToggleExtension,
        selected=selected,
        empty='No extensions found.',
        keep_open=True,
      )
    )

  def _source_selector(self, *, update: bool) -> actions.OpenSelector:
    configured = extension_sources.repositories(
      self._config.read().get('extensions'), self.root
    )
    choices = tuple(
      selection.Choice(key, repository.name, key)
      for key, repository in sorted(
        configured.items(), key=lambda item: item[1].name
      )
    )
    if update and choices:
      choices = (selection.Choice('', 'All extensions'), *choices)
    return actions.OpenSelector(
      selection.Selector(
        'extensions-update' if update else 'extensions-remove',
        'Update GitHub extensions' if update else 'Remove a GitHub extension',
        choices,
        actions.UpdateExtensions if update else actions.RemoveExtension,
        empty='No GitHub extension sources configured.',
      )
    )

  def _remove_confirmation(self, key: str) -> actions.OpenSelector:
    configured = extension_sources.repositories(
      self._config.read().get('extensions'), self.root
    )
    if key not in configured:
      raise ValueError('Extension source is no longer configured')
    return actions.OpenSelector(
      selection.Selector(
        'extensions-remove-confirm',
        f'Remove {configured[key].name}? Cached files will be kept.',
        (
          selection.Choice('cancel', 'Cancel'),
          selection.Choice('remove', 'Remove', key),
        ),
        lambda choice: (
          actions.RemoveExtension(key, confirmed=True)
          if choice == 'remove'
          else actions.CloseSelector()
        ),
        selected='cancel',
        numbered=True,
      )
    )

  async def _add_source(self, context: extension.Context, url: str) -> None:
    async with self._apply_lock:
      self._sources.add(self._config, url)
    context.dispatch(actions.ShowText('Downloading extension…'))
    await self.apply_settings(self._config.read().get('extensions'))
    if error := self._sources.errors.get(url):
      raise ValueError(
        f'Source added, but download failed: {error}. '
        'Retry with /extensions-update.'
      )
    directory = self._directories().get(url)
    if directory is None:
      raise ValueError('Source is shadowed or no longer configured')
    overrides = config.extension_overrides(
      self._config.read().get('extensions'), url
    )
    settings = config.resolve_extension(overrides)
    if 'enabled' not in overrides and settings.enabled is True:
      self._pending_approval[url] = directory
    context.dispatch(
      actions.ShowText(
        f'{directory.name} downloaded. Approval required to enable.'
      )
    )
    self._show_approval(context)

  async def _activate_source(
    self, repository: extension_sources.Repository, directory: pathlib.Path
  ) -> str:
    key = repository.key
    settings = self._config.read().get('extensions')
    if (
      extension_sources.repositories(settings, self.root).get(key) != repository
    ):
      raise ValueError(
        'Source changed during download; retry /extensions-update'
      )
    directories = self._directories()
    previous = directories.get(key)
    if previous is None and any(
      path.name == directory.name for path in directories.values()
    ):
      self._sources.use(key, directory)
      return 'downloaded; shadowed by a local extension'
    overrides = config.extension_overrides(settings, key)
    resolved = config.resolve_extension(overrides)
    loaded = next(
      (
        name
        for name in self.registry.extension_ids()
        if previous is not None
        and self.registry.source_directory(name) == previous
      ),
      None,
    )
    if overrides.get('enabled') is True:
      if loaded is not None:
        await self.registry.reload(
          loaded, directory=directory, overrides=overrides
        )
      else:
        await self.registry.load(directory, overrides=overrides)
        for status in self.registry.statuses(directory.name):
          if status.error is not None:
            await self.registry.unload(directory.name)
            raise extension.ExtensionLoadError(status.error)
      self._pending_approval.pop(key, None)
      message = 'updated and reloaded'
    else:
      if loaded is not None:
        await self.registry.unload(loaded)
      self._pending_approval.pop(key, None)
      message = 'updated; disabled'
      if 'enabled' not in overrides and resolved.enabled is True:
        self._pending_approval[key] = directory
        message = 'downloaded; approval required to enable'
        resolved = dataclasses.replace(resolved, enabled=False)
    self._sources.use(key, directory)
    self._applied = {**(self._applied or {}), key: resolved}
    return message

  async def _update_sources(
    self, context: extension.Context, name: str
  ) -> None:
    context.dispatch(actions.ShowText('Updating GitHub extensions…'))
    async with self._apply_lock:
      results = await self._sources.update(
        self._config.read().get('extensions'), name
      )
      for repository, directory in results.items():
        key = repository.key
        try:
          if directory is None:
            raise ValueError(self._sources.errors[key])
          message = await self._activate_source(repository, directory)
          await asyncio.to_thread(repository.publish, directory)
          context.dispatch(actions.ShowText(f'{directory.name}: {message}.'))
        except (OSError, ValueError) as error:
          self._sources.errors[key] = str(error)
          context.dispatch(
            actions.ShowText(f'{key}: update failed: {error}', style='ui.error')
          )
    self._refresh_selector()
    self._show_approval(context)

  async def _change_source(
    self,
    context: extension.Context,
    request: actions.AddExtension
    | actions.UpdateExtensions
    | actions.RemoveExtension,
  ) -> None:
    if os.getenv('MIRROR_EXTENSIONS_ENABLED', 'true').lower() != 'true':
      raise ValueError('External extensions are disabled')
    if isinstance(request, actions.AddExtension):
      await self._add_source(context, request.url)
    elif isinstance(request, actions.UpdateExtensions):
      if request.name is None:
        context.dispatch(self._source_selector(update=True))
        return
      await self._update_sources(context, request.name)
    elif request.key is None:
      context.dispatch(self._source_selector(update=False))
    elif not request.confirmed:
      context.dispatch(self._remove_confirmation(request.key))
    else:
      async with self._apply_lock:
        self._sources.remove(self._config, request.key)
      await self.apply_settings(self._config.read().get('extensions'))
      context.dispatch(
        actions.ShowText('Extension removed. Cached files were kept.')
      )

  async def change(
    self,
    context: extension.Context,
    effect: actions.ToggleExtension
    | actions.ExtensionConsent
    | actions.AddExtension
    | actions.UpdateExtensions
    | actions.RemoveExtension,
  ) -> None:
    """Apply a toggle, trust decision, or source operation."""
    if isinstance(
      effect,
      (actions.AddExtension, actions.UpdateExtensions, actions.RemoveExtension),
    ):
      try:
        await self._change_source(context, effect)
      except (OSError, ValueError) as error:
        context.dispatch(actions.ShowText(str(error), style='ui.error'))
      return
    if isinstance(effect, actions.ExtensionConsent):
      directory = self._pending_approval.pop(effect.key, None)
      try:
        if directory is not None and effect.enabled is not None:
          entries = self._config.read().get('extensions')
          overrides = config.extension_overrides(entries, effect.key)
          settings = config.resolve_extension(overrides)
          if (
            'enabled' not in overrides
            and settings.enabled is True
            and self._directories().get(effect.key) == directory
          ):
            await self.set_enabled(directory, effect.enabled)
            context.dispatch(
              actions.ShowText(
                f'{directory.name} {"enabled" if effect.enabled else "disabled"}.'
              )
            )
        elif directory is not None:
          context.dispatch(
            actions.ShowText(
              f'{directory.name} not enabled; approval cancelled.'
            )
          )
      except (ValueError, OSError) as error:
        context.dispatch(actions.ShowText(str(error), style='ui.error'))
      self._show_approval(context)
      return
    directory = self._directories().get(effect.key)
    if directory is None:
      context.dispatch(
        actions.ShowText(
          'Extension no longer exists. Reopen /extensions.', style='ui.warning'
        )
      )
      return
    enabled = not self._choices().get(effect.key, False)
    try:
      await self.set_enabled(directory, enabled)
    except (ValueError, OSError) as error:
      context.dispatch(actions.ShowText(str(error), style='ui.error'))
      return
    context.dispatch(self.selector(selected=effect.key))

  async def set_enabled(
    self, directory: pathlib.Path, enabled: bool
  ) -> str | None:
    """Persist enablement and apply it to every workspace."""
    async with self._apply_lock:
      key = self._key(directory)
      if (
        enabled
        and self._directories(ignore_environment=True).get(key) != directory
      ):
        raise ValueError(
          f'Extension is shadowed or no longer discoverable: {directory}'
        )
      self._config.set_extension(key, enabled)
      self._pending_approval.pop(key, None)
      overrides = config.extension_overrides(
        self._config.read().get('extensions'), key
      )
      settings = config.resolve_extension(overrides)
      loaded = await self._apply_extension(directory, enabled, overrides)
      if enabled and loaded is None:
        raise ValueError(f'Could not load {directory}; see the CLI log.')
      self._applied = {**(self._applied or {}), key: settings}
      self._refresh_selector()
      self._emit_metrics()
      return loaded

  async def _apply_extension(
    self, directory: pathlib.Path, enabled: bool, overrides: config.JsonObject
  ) -> str | None:
    loaded = next(
      (
        name
        for name in self.registry.extension_ids()
        if self.registry.source_directory(name) == directory
      ),
      None,
    )
    if enabled and loaded is None:
      for name in self.registry.extension_ids():
        previous = self.registry.source_directory(name)
        if previous is not None and previous.name == directory.name:
          if previous.parent != self._bundled:
            await self.registry.unload(name)
            break
      loaded = await self.registry.load(directory, overrides=overrides)
    elif enabled and loaded is not None:
      if not await self.registry.configure(loaded, overrides):
        return None
    elif not enabled and loaded is not None:
      await self.registry.unload(loaded)
    return loaded

  async def reload(self, extension_id: str) -> str | None:
    """Reload an extension using its saved settings, retaining menu state."""
    async with self._apply_lock:
      directory = self.registry.source_directory(extension_id)
      if directory is None:
        raise ValueError(f'extension has no source directory: {extension_id!r}')
      key = self._key(directory)
      if self._directories(ignore_environment=True).get(key) != directory:
        raise ValueError(
          f'Extension is shadowed or no longer discoverable: {directory}'
        )
      overrides = config.extension_overrides(
        self._config.read().get('extensions'), key
      )
      settings = config.resolve_extension(overrides)
      if settings.enabled is not True:
        await self.registry.unload(extension_id)
        loaded = None
      else:
        loaded = await self.registry.reload(extension_id, overrides=overrides)
      self._applied = {**(self._applied or {}), key: settings}
      self._refresh_selector()
      return loaded
