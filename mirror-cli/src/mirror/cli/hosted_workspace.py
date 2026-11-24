"""The hosted workspace that matches a local directory."""

from __future__ import annotations

import asyncio
import pathlib
import urllib.parse

from mirror.protocol import mirror_pb2

_GIT_TIMEOUT_SECONDS = 5


def repository_url(value: str) -> str:
  """Normalize a local HTTPS or Git SSH remote to a credential-free HTTPS URL."""
  if value.startswith('git@') and ':' in value:
    host, path = value[4:].split(':', 1)
    value = f'https://{host}/{path}'
  parsed = urllib.parse.urlsplit(value)
  if (
    parsed.scheme == 'ssh' and parsed.username == 'git' and parsed.port is None
  ):
    parsed = parsed._replace(scheme='https', netloc=parsed.hostname or '')
  if (
    parsed.scheme != 'https'
    or not parsed.hostname
    or parsed.username is not None
    or parsed.password is not None
    or parsed.query
    or parsed.fragment
    or any(character.isspace() for character in value)
  ):
    raise ValueError(
      'Workspace repositories require a credential-free HTTPS URL.'
    )
  path = parsed.path.rstrip('/').removesuffix('.git')
  if not path or any(part in ('.', '..') for part in path.split('/')):
    raise ValueError('Repository URL must identify a repository.')
  return urllib.parse.urlunsplit(('https', parsed.netloc.lower(), path, '', ''))


def _origin_url(remote: str) -> str | None:
  """Derive a credential-free identity from a local origin, if it has one."""
  parsed = urllib.parse.urlsplit(remote)
  if parsed.scheme == 'https':
    remote = urllib.parse.urlunsplit(
      parsed._replace(netloc=parsed.netloc.rpartition('@')[2])
    )
  try:
    return repository_url(remote)
  except ValueError:
    return None


async def _git(directory: pathlib.Path, *arguments: str) -> str | None:
  """One Git answer about `directory`, or None when Git has none."""
  try:
    process = await asyncio.create_subprocess_exec(
      'git',
      '-C',
      str(directory),
      *arguments,
      stdin=asyncio.subprocess.DEVNULL,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.DEVNULL,
    )
  except OSError:
    return None
  try:
    stdout, _ = await asyncio.wait_for(
      process.communicate(), _GIT_TIMEOUT_SECONDS
    )
  except TimeoutError:
    process.kill()
    await process.wait()
    return None
  if process.returncode != 0:
    return None
  return stdout.decode(errors='replace').strip()


async def _origin_branch(directory: pathlib.Path) -> tuple[str, str] | None:
  """The branch on `origin` that the current local branch tracks, if any.

  Returns:
    The branch's name on `origin` and the local ref that holds its copy, which
    a custom fetch mapping can put outside `refs/remotes/origin`.
  """
  head = await _git(directory, 'symbolic-ref', '--quiet', 'HEAD')
  if head is None:
    return None
  upstream = await _git(
    directory,
    'for-each-ref',
    '--format=%(upstream:remotename)%00%(upstream:remoteref)%00%(upstream)',
    head,
  )
  remote, ref, copy = ((upstream or '').split('\0') + ['', '', ''])[:3]
  if remote != 'origin' or not ref.startswith('refs/heads/'):
    return None
  return ref.removeprefix('refs/heads/'), copy


async def _checked_out_directory(
  root: pathlib.Path, directory: str, revision: str
) -> str:
  """Find the deepest ancestor of `directory` that a hosted checkout will have.

  The checkout is judged by the local copy of `revision`. A directory that is
  only in local commits is not there, a path that is a file there is not a
  directory, and nothing is there when the revision has no local copy; the
  repository root always is.
  """
  path = pathlib.PurePosixPath(directory)
  for candidate in (path, *path.parents):
    name = candidate.as_posix()
    if name == '.':
      return ''
    if await _git(root, 'cat-file', '-t', f'{revision}:{name}') == 'tree':
      return name
  return ''


async def for_directory(directory: pathlib.Path) -> mirror_pb2.Workspace | None:
  """The repository checkout a hosted session starts in for `directory`.

  Local changes are not uploaded: the revision is the branch on `origin` that
  the local branch tracks, or the repository's default branch, and the
  directory is the deepest one that revision has.

  Args:
    directory: A local directory, usually inside a clone.

  Returns:
    The workspace to check out, or None when the directory is not in a
    repository whose `origin` has a credential-free HTTPS identity.
  """
  root = await _git(directory, 'rev-parse', '--show-toplevel')
  if root is None:
    return None
  remote = await _git(directory, 'remote', 'get-url', 'origin')
  url = _origin_url(remote) if remote else None
  if url is None:
    return None
  prefix = await _git(directory, 'rev-parse', '--show-prefix')
  branch, copy = await _origin_branch(directory) or (
    'HEAD',
    'refs/remotes/origin/HEAD',
  )
  return mirror_pb2.Workspace(
    name=pathlib.PurePosixPath(urllib.parse.urlsplit(url).path).name,
    repository_url=url,
    revision=branch,
    directory=await _checked_out_directory(
      pathlib.Path(root), prefix or '', copy
    ),
  )
