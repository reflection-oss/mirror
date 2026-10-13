"""Offer PyPI upgrades before CLI startup without mutating a running install."""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import importlib.metadata
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
from collections.abc import Sequence
from typing import cast

from packaging import version as packaging_version

_RELEASE_ID = re.compile(r'([0-9a-z.+]+)-([0-9a-f]{64})')
_INDEX = 'https://pypi.org/simple/'
_ARTIFACT_BASE = 'https://files.pythonhosted.org/packages/'


@dataclasses.dataclass(frozen=True)
class Launch:
  """The interpreter and entrypoint to replace the startup process with."""

  python: pathlib.Path
  module: str = 'mirror.cli.main'


@dataclasses.dataclass(frozen=True)
class Release:
  """An exact compatible wheel from the configured artifact source."""

  version: str
  url: str
  sha256: str

  @property
  def directory(self) -> str:
    return f'{self.version}-{self.sha256}'


def _version(value: str) -> packaging_version.Version:
  parsed = packaging_version.Version(value)
  if (
    str(parsed) != value
    or parsed.epoch
    or len(parsed.release) != 3
    or parsed.is_prerelease
    or parsed.is_postrelease
  ):
    raise ValueError('Unsupported release version')
  return parsed


def _record(value: object) -> dict[str, object]:
  if not isinstance(value, dict):
    raise TypeError('Expected a record')
  return cast(dict[str, object], value)


def _one(value: object) -> dict[str, object]:
  if not isinstance(value, list):
    raise TypeError('Expected a list')
  records = cast(list[object], value)
  if len(records) != 1:
    raise ValueError('Expected one compatible package or wheel')
  return _record(records[0])


def _https(url: str) -> urllib.parse.SplitResult:
  parsed = urllib.parse.urlsplit(url)
  path = urllib.parse.unquote(parsed.path)
  if (
    any(character.isspace() or ord(character) < 32 for character in url)
    or parsed.scheme != 'https'
    or not parsed.hostname
    or parsed.port not in (None, 443)
    or parsed.username
    or parsed.password
    or parsed.query
    or parsed.fragment
    or '\\' in path
    or any(part in ('.', '..') for part in path.split('/'))
  ):
    raise ValueError('Invalid release source')
  return parsed


def _installed_version() -> str | None:
  try:
    installed = importlib.metadata.distribution('mirror-agent')
  except importlib.metadata.PackageNotFoundError:
    return None
  # Installing the package elsewhere must not opt a source checkout in.
  if (
    pathlib.Path(str(installed.locate_file('mirror/updater.py'))).resolve()
    != pathlib.Path(__file__).resolve()
  ):
    return None
  return installed.version


def _environment() -> dict[str, str]:
  names = [
    'PATH',
    'HOME',
    'TMPDIR',
    'XDG_DATA_HOME',
    'XDG_CACHE_HOME',
    'XDG_CONFIG_HOME',
  ]
  return {name: os.environ[name] for name in names if name in os.environ}


def _run(
  argv: Sequence[str],
  *,
  timeout: float = 180,
  stdin: bytes | None = None,
) -> bytes:
  """Bound trusted installer execution without collecting diagnostic stderr."""
  with tempfile.TemporaryDirectory(prefix='mirror-update-') as directory:
    with subprocess.Popen(
      argv,
      cwd=directory,
      env=_environment(),
      start_new_session=True,
      stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
    ) as process:
      try:
        output, _ = process.communicate(stdin, timeout=timeout)
      except BaseException:
        with contextlib.suppress(ProcessLookupError):
          os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
      if process.returncode:
        raise RuntimeError('Release tooling failed')
      return output


def latest(current: str, *, timeout: float) -> Release | None:
  """Resolve the latest compatible mirror-agent wheel from PyPI."""
  minimum = _version(current).public
  output = _run(
    (
      'uv',
      '--no-config',
      'pip',
      'compile',
      '--no-deps',
      '--prerelease',
      'disallow',
      '--only-binary',
      ':all:',
      '--format',
      'pylock.toml',
      '--python',
      sys.executable,
      '--default-index',
      _INDEX,
      '--refresh',
      '-',
    ),
    timeout=timeout,
    stdin=f'mirror-agent>={minimum}\n'.encode(),
  )
  package = _one(tomllib.loads(output.decode()).get('packages'))
  version = package.get('version')
  if package.get('name') != 'mirror-agent' or not isinstance(version, str):
    raise ValueError('Invalid resolved package')
  if _version(version) <= _version(current):
    return None
  wheel = _one(package.get('wheels'))
  url, digest = wheel.get('url'), _record(wheel.get('hashes')).get('sha256')
  if not isinstance(url, str) or not isinstance(digest, str):
    raise TypeError('Missing wheel URL or SHA-256')
  _https(url)
  if (
    not url.startswith(_ARTIFACT_BASE)
    or not url.endswith('.whl')
    or re.fullmatch('[0-9a-f]{64}', digest) is None
  ):
    raise ValueError('Wheel is outside the configured artifact source')
  return Release(version, url, digest)


def _root() -> pathlib.Path:
  data = pathlib.Path(
    os.environ.get('XDG_DATA_HOME', pathlib.Path.home() / '.local' / 'share')
  )
  return data / 'mirror' / 'releases' / 'public'


def _selected(root: pathlib.Path, current: str) -> tuple[pathlib.Path, str]:
  python = pathlib.Path(sys.executable)
  try:
    name = (root / 'selected').read_text()
  except FileNotFoundError:
    return python, current
  match = _RELEASE_ID.fullmatch(name)
  if match is None:
    raise ValueError('Invalid selected release')
  version = match[1]
  directory = root / name
  candidate = directory / 'bin' / 'python'
  if _version(version) <= _version(current):
    return python, current
  if not (directory / 'ready').is_file() or not candidate.is_file():
    raise ValueError('Selected release is incomplete')
  return candidate, version


def install(release: Release, root: pathlib.Path, current: str) -> pathlib.Path:
  """Publish a complete side-by-side environment; never mutate an old release."""
  _version(release.version)
  _https(release.url)
  if (
    re.fullmatch('[0-9a-f]{64}', release.sha256) is None
    or not release.url.startswith(_ARTIFACT_BASE)
    or not release.url.endswith('.whl')
  ):
    raise ValueError('Invalid release')
  root.mkdir(parents=True, exist_ok=True, mode=0o700)
  with (root / 'install.lock').open('a') as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      raise RuntimeError('Another update is being prepared') from None
    selected, version = _selected(root, current)
    if _version(version) >= _version(release.version):
      return selected
    directory = root / release.directory
    python = directory / 'bin' / 'python'
    if not (directory / 'ready').is_file():
      # Incomplete directories were never selected and cannot host a live CLI.
      shutil.rmtree(directory, ignore_errors=True)
      try:
        _run(
          (
            'uv',
            '--no-config',
            'venv',
            '--python',
            sys.executable,
            str(directory),
          ),
        )
        requirement = (
          f'mirror-agent @ {release.url} --hash=sha256:{release.sha256}\n'
        )
        _run(
          (
            'uv',
            '--no-config',
            'pip',
            'install',
            '--python',
            str(python),
            '--only-binary',
            ':all:',
            '--default-index',
            'https://pypi.org/simple/',
            '-r',
            '-',
          ),
          stdin=requirement.encode(),
        )
        # Read distribution metadata without importing application code.
        output = _run(
          (
            str(python),
            '-I',
            '-c',
            'import importlib.metadata; print(importlib.metadata.version("mirror-agent"))',
          ),
        )
        if output.decode().strip() != release.version:
          raise ValueError('Installed release version mismatch')
        (directory / 'ready').touch()
      except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    temporary = root / 'selected.tmp'
    temporary.write_text(release.directory)
    temporary.replace(root / 'selected')
    return python


def prepare(*, check: bool, force: bool, interactive: bool) -> Launch:
  """Select a cached release and optionally offer an update before CLI startup."""
  python = pathlib.Path(sys.executable)
  if os.environ.get('MIRROR_SKIP_UPDATE') == '1' or os.environ.get(
    'MIRROR_RUST_WORKER'
  ):
    return Launch(python)
  try:
    current = _installed_version()
    if current is None:
      return Launch(python)
    root = _root()
    python, current = _selected(root, current)
    if python != pathlib.Path(sys.executable):
      # Let the selected release own checks, so updater fixes take effect too.
      return Launch(python, 'mirror.main')
    if not check or not interactive:
      return Launch(python)
    release = latest(current, timeout=120 if force else 3)
    if release is None:
      return Launch(python)
    print(
      f'Mirror {release.version} is available. Update and start? [y/N] ',
      end='',
      file=sys.stderr,
      flush=True,
    )
    if input().strip().lower() not in ('y', 'yes'):
      return Launch(python)
    print('Preparing Mirror update…', file=sys.stderr)
    return Launch(install(release, root, current))
  except (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    subprocess.TimeoutExpired,
    EOFError,
    importlib.metadata.PackageNotFoundError,
  ):
    print(
      'Mirror update unavailable; starting the existing release. Check uv and network access; retry with mirror --update.',
      file=sys.stderr,
    )
    return Launch(python)
