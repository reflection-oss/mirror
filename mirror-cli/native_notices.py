"""Collect original crate license texts from worker and ripgrep Cargo JSON logs."""

# cargo-about can synthesize SPDX text; cargo-bundle-licenses loses paths/NOTICE.
# Inventory build/proc-macro inputs conservatively, not exact linkage or headers.

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile
import tomllib
from typing import cast
from urllib import request

_CONVENTIONAL = re.compile(
  r'^(licen[cs]e|copying|unlicense|notice|copyright)(?:$|[._-])', re.IGNORECASE
)
_SOURCE_SUFFIXES = (
  '.rs',
  '.c',
  '.h',
  '.cc',
  '.cpp',
  '.hpp',
  '.py',
  '.js',
  '.ts',
  '.go',
  '.sh',
)
# Cargo >= 1.77: https://doc.rust-lang.org/cargo/reference/pkgid-spec.html
_PACKAGE_ID = re.compile(
  r'(registry|git|path)\+[a-z][a-z0-9+.-]*://[^\s#]+#[^\s#]+'
)
_FIRST_PARTY = {
  'mirror-agent': 'mirror-agent/Cargo.toml',
  'mirror-protocol': 'protocols/mirror_agent/Cargo.toml',
  'mirror-agent-python': 'mirror-agent-py/Cargo.toml',
  'mirror-sandbox': 'mirror-sandbox/Cargo.toml',
  'mirror-sandbox-microsandbox': 'mirror-sandbox/providers/microsandbox/Cargo.toml',
  'mirror-sandboxd': 'mirror-sandboxd/Cargo.toml',
}
# A binary Zig cross-links embeds Zig's compiler runtime and glibc's start files.
_ZIG_RUNTIME_FILES = (
  ('Zig and its compiler runtime', pathlib.Path('LICENSE')),
  ('glibc start files', pathlib.Path('lib/libc/glibc/LICENSES')),
)


def _object(value: object) -> dict[str, object]:
  if not isinstance(value, dict):
    raise ValueError('Expected an object/table')
  return cast(dict[str, object], value)


def _string(record: dict[str, object], key: str) -> str:
  value = record.get(key)
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f'Missing or invalid {key!r}: {value!r}')
  return value


def _artifacts(log: pathlib.Path) -> set[tuple[str, pathlib.Path]]:
  """Read every artifact, irrespective of target kind or Cargo freshness."""
  packages: set[tuple[str, pathlib.Path]] = set()
  with log.open(encoding='utf-8') as stream:
    for line_number, line in enumerate(stream, 1):
      try:
        record = _object(json.loads(line))
        if _string(record, 'reason') != 'compiler-artifact':
          continue
        package_id = _string(record, 'package_id')
        if not _PACKAGE_ID.fullmatch(package_id):
          raise ValueError(f'Invalid Cargo >= 1.77 package_id: {package_id!r}')
        manifest = pathlib.Path(_string(record, 'manifest_path'))
        if not manifest.is_absolute() or manifest.name != 'Cargo.toml':
          raise ValueError(f'Invalid manifest_path: {str(manifest)!r}')
        packages.add((package_id, manifest))
      except ValueError as error:
        raise ValueError(f'{log}:{line_number}: {error}') from error
  if not packages:
    raise ValueError(f'{log}: no compiler-artifact records')
  return packages


def _walk_error(error: OSError) -> None:
  raise error


def _paths(
  root: pathlib.Path, explicit: str | None
) -> dict[pathlib.Path, bool]:
  """Find attribution files; NOTICE/COPYRIGHT alone are not license texts."""
  paths: dict[pathlib.Path, bool] = {}
  for directory, directories, filenames in root.walk(on_error=_walk_error):
    directories[:] = [name for name in directories if name != '.git']
    for name in filenames:
      relative = (directory / name).relative_to(root)
      if (directory / name).is_symlink() and (directory / name).is_dir():
        raise ValueError(
          f'Cannot inventory symlink directory: {directory / name}'
        )
      in_licenses = any(
        part.lower() == 'licenses' for part in relative.parts[:-1]
      )
      if not in_licenses and relative.suffix.lower() in _SOURCE_SUFFIXES:
        continue
      match = _CONVENTIONAL.match(name)
      if match or in_licenses:
        paths[relative] = not match or match[1].lower() not in (
          'notice',
          'copyright',
        )
  if explicit is not None:
    relative = pathlib.Path(explicit)
    if relative.is_absolute() or '..' in relative.parts:
      raise ValueError(f'license-file must be within the crate: {explicit!r}')
    paths[relative] = True
  return paths


def _text(root: pathlib.Path, relative: pathlib.Path) -> bytes:
  """Read original bytes, refusing missing, empty, non-file or escaped paths."""
  path = root / relative
  if not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
    raise ValueError(f'Not a regular file within the crate: {path}')
  content = path.read_bytes()
  if not content.strip() or b'\0' in content:
    raise ValueError(f'Empty or non-text attribution file: {path}')
  return content


def _section(label: str, content: bytes) -> bytes:
  digest = hashlib.sha256(content).hexdigest()
  return f'{label}\nSHA256: {digest}\n\n'.encode() + content + b'\n\n'


def _supplement(
  source: dict[str, object],
  root: pathlib.Path,
  version: str,
  expression: str,
) -> bytes:
  """Fetch only a version/license/VCS-matched, checksum-pinned source text."""
  if (
    _string(source, 'version') != version
    or _string(source, 'license') != expression
  ):
    raise ValueError(
      f'{root}: supplemental version/license mismatch; review sources'
    )
  vcs = _text(root, pathlib.Path('.cargo_vcs_info.json'))
  git = _object(_object(json.loads(vcs)).get('git'))
  commit = _string(source, 'git_sha1')
  if _string(git, 'sha1') != commit or git.get('dirty', False) is not False:
    raise ValueError(f'{root}: supplemental VCS mismatch; review sources')
  url, checksum = _string(source, 'url'), _string(source, 'sha256')
  if not url.startswith(('https://', 'file://')) or not re.fullmatch(
    '[0-9a-f]{64}', checksum
  ):
    raise ValueError(f'{root}: invalid supplemental URL/checksum')
  label = f'Supplemental source: {url}\nSource VCS git SHA1: {commit}\nReason: {_string(source, "reason")}'
  if 'selected_license' in source:
    label += f'\nSelected license for this distribution: {_string(source, "selected_license")}'
  with request.urlopen(url, timeout=30) as response:
    content: bytes = response.read()
  if hashlib.sha256(content).hexdigest() != checksum:
    raise ValueError(f'{url}: supplemental SHA256 mismatch')
  if not content.strip() or b'\0' in content:
    raise ValueError(f'{url}: empty or non-text supplement')
  return _section('File: .cargo_vcs_info.json', vcs) + _section(label, content)


def _notices(
  package_id: str,
  manifest: pathlib.Path,
  sources: dict[str, object],
  workspace: pathlib.Path,
) -> bytes:
  """Render actual source files or an explicitly guarded supplemental text."""
  with manifest.open('rb') as stream:
    package = _object(tomllib.load(stream).get('package'))
  name, version = _string(package, 'name'), _string(package, 'version')
  if package_id.rsplit('#', 1)[1] not in (version, f'{name}@{version}'):
    raise ValueError(f'{manifest}: metadata does not match {package_id}')
  if (
    package_id.startswith('path+')
    and name in _FIRST_PARTY
    and manifest.resolve() == workspace / _FIRST_PARTY[name]
  ):
    return b''
  expression = (
    _string(package, 'license') if 'license' in package else '(not declared)'
  )
  explicit = (
    _string(package, 'license-file') if 'license-file' in package else None
  )
  root = manifest.parent.resolve(strict=True)
  paths = _paths(root, explicit)
  header = f'Package: {package_id}\nName: {name}\nVersion: {version}\nLicense expression: {expression}\n'
  if 'authors' in package:
    authors = package['authors']
    if not isinstance(authors, list) or not all(
      isinstance(author, str) for author in cast(list[object], authors)
    ):
      raise ValueError(f'{manifest}: invalid authors')
    header += f'Authors: {json.dumps(authors, ensure_ascii=False)}\n'
  sections = [(header + '\n').encode()]
  for relative in sorted(paths):
    sections.append(
      _section(f'File: {relative.as_posix()}', _text(root, relative))
    )
  if name in sources:
    if any(paths.values()):
      raise ValueError(
        f'{package_id}: source now ships license text; review supplemental entry'
      )
    sections.append(
      _supplement(_object(sources[name]), root, version, expression)
    )
  elif not any(paths.values()):
    raise ValueError(
      f'{package_id}: no actual license text (NOTICE alone is insufficient)'
    )
  return b''.join(sections)


def _rust_notices(root: pathlib.Path) -> bytes:
  """Retain the build toolchain's standard-library attribution and license texts."""
  root = root.resolve(strict=True)
  paths = {
    pathlib.Path('COPYRIGHT-library.html'),
    pathlib.Path('licenses/Apache-2.0.txt'),
    pathlib.Path('licenses/MIT.txt'),
  }
  paths.update(
    path.relative_to(root) for path in (root / 'licenses').glob('*.txt')
  )
  return b''.join(
    _section(
      f'Rust standard library file: {path.as_posix()}', _text(root, path)
    )
    for path in sorted(paths)
  )


def _zig_runtime_notices(root: pathlib.Path) -> bytes:
  """Retain the attribution of the runtime a Zig toolchain links into a binary."""
  root = root.resolve(strict=True)
  sections: list[bytes] = []
  for name, path in _ZIG_RUNTIME_FILES:
    if not (root / path).is_file():
      raise ValueError(f'{root}: missing Zig runtime attribution {path}')
    sections.append(
      _section(
        f'Zig runtime ({name}) file: {path.as_posix()}', _text(root, path)
      )
    )
  return b''.join(sections)


def main() -> None:
  """Atomically write notices only after both logs and every crate validate."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('output', type=pathlib.Path)
  parser.add_argument('worker_json', type=pathlib.Path)
  parser.add_argument('ripgrep_json', type=pathlib.Path)
  parser.add_argument(
    '--cargo-json',
    type=pathlib.Path,
    action='append',
    default=[],
    help='Cargo JSON log of another native build packaged alongside the worker',
  )
  parser.add_argument('--rust-docs', type=pathlib.Path, required=True)
  parser.add_argument(
    '--zig-runtime',
    type=pathlib.Path,
    help='Zig package whose runtime a cross-linked packaged binary embeds',
  )
  parser.add_argument(
    '--sources',
    type=pathlib.Path,
    default=pathlib.Path(__file__).with_name('native_notices_sources.toml'),
  )
  parser.add_argument(
    '--workspace-root',
    type=pathlib.Path,
    default=pathlib.Path(__file__).resolve().parent.parent,
    help='Mirror workspace root; only known first-party names at exact paths are exempt',
  )
  args = parser.parse_args()
  try:
    with args.sources.open('rb') as stream:
      sources = _object(tomllib.load(stream))
    artifacts = _artifacts(args.worker_json) | _artifacts(args.ripgrep_json)
    for log in args.cargo_json:
      artifacts |= _artifacts(log)
    packages: dict[str, bytes] = {}
    for package_id, manifest in sorted(artifacts):
      rendered = _notices(
        package_id, manifest, sources, args.workspace_root.resolve()
      )
      if package_id in packages and packages[package_id] != rendered:
        raise ValueError(
          f'Conflicting source texts or metadata for {package_id}'
        )
      packages[package_id] = rendered
    content = (
      b'Third-party notices: native build inputs (not an exact linked inventory).\n\n'
      + b''.join(packages.values())
    )
    content += _rust_notices(args.rust_docs)
    if args.zig_runtime is not None:
      content += _zig_runtime_notices(args.zig_runtime)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
      staged = pathlib.Path(temporary) / 'notices'
      staged.write_bytes(content)
      staged.replace(args.output)
  except (OSError, ValueError) as error:
    parser.error(str(error))


if __name__ == '__main__':
  main()
