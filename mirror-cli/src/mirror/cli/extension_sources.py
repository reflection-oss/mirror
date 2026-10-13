"""GitHub extension checkouts, separate from consent and extension execution."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import fcntl
import logging
import os
import pathlib
import re
import subprocess
import tempfile
from typing import cast

from mirror import config


@dataclasses.dataclass(frozen=True)
class Repository:
  """One configured repository and its privately managed checkouts."""

  key: str
  url: str
  name: str
  directory: pathlib.Path
  ref: str | None

  def installed(self) -> pathlib.Path | None:
    """Resolve the installed revision; ref changes require an explicit update."""
    saved = config.Config(self.directory / 'installed.json').read()
    revision = saved.get('revision')
    if revision is None:
      return None
    if not isinstance(revision, str) or not re.fullmatch(
      r'[0-9a-f]{40,64}', revision
    ):
      raise ValueError('Invalid installed extension revision')
    directory = self.directory / revision / self.name
    return directory if (directory / 'extension.py').is_file() else None

  def install(self, *, update: bool = False) -> pathlib.Path:
    """Install a missing source, or stage an update without selecting it."""
    self.directory.mkdir(parents=True, exist_ok=True)
    with (self.directory / 'install.lock').open('a') as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      installed = self.installed()
      if installed is not None and not update:
        return installed
      if installed is not None:
        _check_clean(installed)
      with tempfile.TemporaryDirectory(dir=self.directory) as temporary:
        checkout = pathlib.Path(temporary) / self.name
        _git(
          self.directory,
          'clone',
          '--no-checkout',
          '--no-recurse-submodules',
          '--depth=1',
          '--no-tags',
          '--template=',
          '--',
          self.url,
          str(checkout),
        )
        target = 'HEAD'
        if self.ref is not None:
          _git(
            checkout,
            'fetch',
            '--no-recurse-submodules',
            '--depth=1',
            '--no-tags',
            'origin',
            self.ref,
          )
          target = 'FETCH_HEAD'
        revision = _git(
          checkout, 'rev-parse', '--verify', f'{target}^{{commit}}'
        )
        if not re.fullmatch(r'[0-9a-f]{40,64}', revision):
          raise ValueError('Git returned an invalid extension revision')
        _git(checkout, 'checkout', '--detach', revision)
        destination = self.directory / revision / self.name
        if destination.exists():
          _check_clean(destination)
          checkout = destination
        entrypoint = checkout / 'extension.py'
        if entrypoint.is_symlink() or not entrypoint.is_file():
          raise ValueError(
            'Repository must contain an extension.py file at its root'
          )
        try:
          ast.parse(entrypoint.read_text())
        except (SyntaxError, UnicodeError) as error:
          raise ValueError(
            'Repository extension.py is not valid Python'
          ) from error
        if checkout != destination:
          destination.parent.mkdir(exist_ok=True)
          checkout.rename(destination)
        if not update:
          self.publish(destination)
        return destination

  def publish(self, directory: pathlib.Path) -> None:
    """Select a downloaded checkout for subsequent CLI launches."""
    config.Config(self.directory / 'installed.json').update(
      lambda saved: saved.update(ref=self.ref, revision=directory.parent.name)
    )


def _check_clean(directory: pathlib.Path) -> None:
  if _git(directory, 'status', '--porcelain', '--untracked-files=no'):
    raise ValueError(
      'Managed checkout has local edits; move them to a local extension before updating'
    )


def _git(directory: pathlib.Path, *arguments: str) -> str:
  """Use Git's credential helpers, without inheriting model keys or logging output."""
  environment = {
    key: value
    for key, value in os.environ.items()
    if key in {'HOME', 'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'XDG_CONFIG_HOME'}
  }
  environment.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
  if arguments[0] not in {'clone', 'fetch'}:
    # Keep credential helpers for downloads, but never run configured worktree filters.
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
  try:
    result = subprocess.run(
      (
        'git',
        '-c',
        f'core.hooksPath={os.devnull}',
        '-c',
        'core.fsmonitor=false',
        '-c',
        'submodule.recurse=false',
        *arguments,
      ),
      cwd=directory,
      env=environment,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
      timeout=60,
      check=False,
    )
  except FileNotFoundError as error:
    raise ValueError('Git is required to install GitHub extensions') from error
  except subprocess.TimeoutExpired as error:
    raise ValueError('Extension download timed out') from error
  if result.returncode:
    raise ValueError(
      'Git operation failed; check repository access and the configured ref'
    )
  return result.stdout.strip()


def repositories(
  settings: object, root: pathlib.Path, *, strict: bool = False
) -> dict[str, Repository]:
  """Recognize GitHub URLs in user settings, never executable install recipes."""
  repositories: dict[str, Repository] = {}
  if not isinstance(settings, dict):
    return repositories
  entries = cast(dict[str, object], settings)
  seen: dict[str, str] = {}
  for key in entries:
    if '://' not in key:
      continue
    try:
      match = re.fullmatch(
        r'https://github\.com/([A-Za-z0-9][A-Za-z0-9-]*)/([A-Za-z0-9_.-]+?)(?:\.git)?/?',
        key,
      )
      if match is None or match[2] in {'.', '..'}:
        raise ValueError(
          'Extension sources must be https://github.com/owner/repository URLs without credentials or query parameters'
        )
      owner, name = match[1].lower(), match[2].lower()
      overrides = config.extension_overrides(entries, key)
      config.resolve_extension(overrides)
      ref = overrides.get('ref')
      if ref is not None and (
        not isinstance(ref, str)
        or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]*', ref)
      ):
        raise ValueError('Extension ref must be a branch, tag, or commit SHA')
      directory = root / '.managed' / 'github.com' / owner / name
      if name in seen:
        repositories.pop(seen[name], None)
        raise ValueError('GitHub extension repository names must be unique')
      seen[name] = key
      repositories[key] = Repository(
        key, f'https://github.com/{owner}/{name}.git', name, directory, ref
      )
    except ValueError as error:
      if strict:
        raise
      logging.getLogger(__name__).warning(
        'Ignoring extension source: %s', error
      )
  return repositories


class Sources:
  """Acquire sources while retaining each running CLI's chosen checkout."""

  def __init__(self, root: pathlib.Path) -> None:
    self._root = root
    self._directories: dict[str, pathlib.Path] = {}
    self._keys: dict[pathlib.Path, str] = {}
    self._configured: set[str] = set()
    self.errors: dict[str, str] = {}

  def key(self, directory: pathlib.Path) -> str | None:
    return self._keys.get(directory)

  def directories(self, settings: object) -> dict[str, pathlib.Path]:
    if not isinstance(settings, dict):
      return {}
    return {
      key: path
      for key, path in self._directories.items()
      if key in settings and key in self._configured
    }

  def use(self, key: str, directory: pathlib.Path) -> None:
    """Choose a checkout for this CLI without changing other running CLIs."""
    self._directories[key] = directory
    self._keys[directory] = key
    self._configured.add(key)

  def add(self, configuration: config.Config, url: str) -> None:
    """Save a validated source declaration, leaving execution consent unset."""

    def change(settings: dict[str, object]) -> None:
      entries = settings.get('extensions', {})
      if not isinstance(entries, dict):
        raise ValueError('extensions must be a JSON object')
      entries = cast(dict[str, object], entries)
      if url in entries:
        raise ValueError('This extension source is already configured')
      updated: dict[str, object] = {**entries, url: {}}
      if url not in repositories(updated, self._root, strict=True):
        raise ValueError('Use an HTTPS GitHub repository URL')
      settings['extensions'] = updated

    configuration.update(change)
    self._directories.pop(url, None)

  def remove(self, configuration: config.Config, key: str) -> None:
    """Remove only the source declaration, retaining every cached revision."""

    def change(settings: dict[str, object]) -> None:
      entries = settings.get('extensions')
      configured = (
        cast(dict[str, object], entries) if isinstance(entries, dict) else {}
      )
      if key not in repositories(configured, self._root):
        raise ValueError('Extension source is no longer configured')
      del configured[key]

    configuration.update(change)

  async def prepare(self, settings: object) -> None:
    """Install missing sources; never replace a checkout in a running process."""
    configured = repositories(settings, self._root)
    self._configured = set(configured)
    for key, repository in configured.items():
      if key in self._directories:
        continue
      directory = repository.directory / 'uninstalled' / repository.name
      try:
        directory = await asyncio.to_thread(repository.install)
        self.errors.pop(key, None)
      except (OSError, ValueError) as error:
        self.errors[key] = str(error)
        logging.getLogger(__name__).warning('Extension %s: %s', key, error)
      self.use(key, directory)

  async def update(
    self, settings: object, name: str
  ) -> dict[Repository, pathlib.Path | None]:
    """Stage checkouts for the caller to activate and publish; retain failures."""
    configured = repositories(settings, self._root)
    selected = [
      repository
      for key, repository in configured.items()
      if not name
      or name
      in (
        key,
        repository.name,
        f'{repository.directory.parent.name}/{repository.name}',
      )
    ]
    if not selected:
      raise ValueError('No matching GitHub extension sources configured')
    results: dict[Repository, pathlib.Path | None] = {}
    for repository in selected:
      try:
        results[repository] = await asyncio.to_thread(
          repository.install, update=True
        )
        self.errors.pop(repository.key, None)
      except (OSError, ValueError) as error:
        self.errors[repository.key] = str(error)
        results[repository] = None
    return results
