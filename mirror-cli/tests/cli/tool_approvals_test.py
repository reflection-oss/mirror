from __future__ import annotations

import json
import pathlib
import stat

import pytest
from mirror.cli import tool_approvals


def _key(
  command: str = 'git status', scope: str = '/workspace', tool: str = 'bash'
) -> tool_approvals.ToolApproval:
  approval = tool_approvals.from_call(
    tool, json.dumps({'command': command}), scope
  )
  assert approval is not None
  return approval


def test_remember_persists_without_storing_command_text(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  key = _key('echo synthetic-secret-canary')
  assert not store.contains(key)
  assert not path.exists()

  store.remember(key)

  assert store.contains(key)
  assert tool_approvals.ToolApprovals(path).contains(key)
  assert 'synthetic-secret-canary' not in path.read_text()
  assert json.loads(path.read_text()) == {
    'version': 1,
    'entries': [
      {
        'kind': 'exact_arguments',
        'scope': key.scope,
        'tool': key.tool,
        'arguments_sha256': key.arguments_sha256,
      }
    ],
  }
  assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
  'command,scope',
  [
    ('git status --short', '/workspace'),
    ('git status; echo extra', '/workspace'),
    ('git status\necho extra', '/workspace'),
    ('git  status', '/workspace'),
    ('git status', '/other-worktree'),
  ],
)
def test_remember_matches_only_exact_command_and_scope(
  tmp_path: pathlib.Path, command: str, scope: str
) -> None:
  store = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  store.remember(_key())
  assert not store.contains(_key(command, scope))


@pytest.mark.parametrize(
  'name,arguments,scope',
  [
    ('', '{}', '/workspace'),
    ('bash', '{}', ''),
    ('bash', '\ud800', '/workspace'),
  ],
)
def test_from_call_rejects_unusable_identity(
  name: str, arguments: str, scope: str
) -> None:
  assert tool_approvals.from_call(name, arguments, scope) is None


def test_from_call_matches_arguments_verbatim() -> None:
  assert (
    tool_approvals.from_call('bash', ' {"command":"git status"} ', '/workspace')
    != _key()
  )
  assert _key(tool='extension__bash') != _key()


@pytest.mark.parametrize(
  'tool,arguments',
  [
    ('bash', '{"command":"ls"}'),
    ('write', '{"file_path":"a.txt","content":"text"}'),
    ('mcp__calendar__create_event', '{"date":"2030-01-01","title":"test"}'),
  ],
)
def test_remember_supports_any_tool(
  tmp_path: pathlib.Path, tool: str, arguments: str
) -> None:
  store = tool_approvals.ToolApprovals(tmp_path / 'approvals.json')
  approval = tool_approvals.from_call(tool, arguments, '/workspace')
  assert approval is not None
  store.remember(approval)
  assert store.contains(approval)
  other = tool_approvals.from_call('other', arguments, '/workspace')
  assert other is not None
  assert not store.contains(other)


def test_cache_and_duplicate_remember_do_not_touch_disk(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  store.remember(_key())
  path.unlink()
  path.mkdir()  # Any subsequent read/write would fail rather than hit the cache.

  assert store.contains(_key())
  assert not store.contains(_key('pwd'))
  store.remember(_key())
  assert path.is_dir()


def test_writers_merge_without_losing_other_process_approvals(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  first = tool_approvals.ToolApprovals(path)
  second = tool_approvals.ToolApprovals(path)
  assert not first.contains(_key())
  assert not second.contains(_key('pwd'))

  first.remember(_key())
  second.remember(_key('pwd'))

  restarted = tool_approvals.ToolApprovals(path)
  assert restarted.contains(_key())
  assert restarted.contains(_key('pwd'))


def test_unknown_kind_is_inert_and_preserved_by_a_stale_writer(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  store.remember(_key('pwd'))
  key = _key()
  unknown = {
    'kind': 'future_matcher',
    'scope': key.scope,
    'tool': key.tool,
    'arguments_sha256': key.arguments_sha256,
    'options': {'pattern': ['git', 'status']},
  }
  settings = json.loads(path.read_text())
  settings['entries'].append(unknown)
  path.write_text(json.dumps(settings))
  restarted = tool_approvals.ToolApprovals(path)

  assert restarted.contains(_key('pwd'))
  assert not restarted.contains(key)
  assert restarted.available
  store.remember(_key('ls'))

  saved = json.loads(path.read_text())
  assert saved['entries'][:2] == settings['entries']
  assert len(saved['entries']) == 3
  latest = tool_approvals.ToolApprovals(path)
  assert latest.contains(_key('pwd'))
  assert latest.contains(_key('ls'))
  assert not latest.contains(key)


def test_writers_do_not_duplicate_the_same_approval(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  first = tool_approvals.ToolApprovals(path)
  second = tool_approvals.ToolApprovals(path)
  assert not first.contains(_key())
  assert not second.contains(_key())

  first.remember(_key())
  second.remember(_key())

  assert second.contains(_key())
  assert len(json.loads(path.read_text())['entries']) == 1


@pytest.mark.parametrize(
  'content',
  [
    b'{',
    b'\xff',
    b'[]',
    b'{"version":2,"entries":[]}',
    b'{"version":true,"entries":[]}',
    b'{"version":1,"entries":[{"kind":"exact_arguments","scope":"/workspace",'
    b'"tool":"bash","arguments_sha256":"*"}]}',
  ],
)
def test_invalid_file_fails_closed_and_is_not_overwritten(
  tmp_path: pathlib.Path, content: bytes
) -> None:
  path = tmp_path / 'approvals.json'
  path.write_bytes(content)
  store = tool_approvals.ToolApprovals(path)

  with pytest.raises(ValueError):
    store.contains(_key())
  assert not store.contains(_key())
  with pytest.raises(ValueError):
    store.remember(_key())
  assert path.read_bytes() == content


@pytest.mark.parametrize(
  'updates',
  [
    {'kind': ''},
    {'kind': False},
    {'scope': None},
    {'tool': ''},
    {'arguments_sha256': 'g' * 64},
    {'constraint': 'not understood'},
    {'kind': 'future_matcher', 'scope': []},
  ],
)
def test_invalid_entry_is_not_matched_or_overwritten(
  tmp_path: pathlib.Path, updates: dict[str, object]
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  store.remember(_key())
  settings = json.loads(path.read_text())
  settings['entries'][0].update(updates)
  content = json.dumps(settings)
  path.write_text(content)
  restarted = tool_approvals.ToolApprovals(path)

  with pytest.raises(ValueError):
    restarted.contains(_key())
  assert not restarted.contains(_key())
  with pytest.raises(ValueError):
    restarted.remember(_key('pwd'))
  assert path.read_text() == content


def test_untagged_v1_approval_remains_exact_and_writable(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  key = _key()
  entry = {
    'scope': key.scope,
    'tool': key.tool,
    'arguments_sha256': key.arguments_sha256,
  }
  content = json.dumps({'version': 1, 'entries': [entry]})
  path.write_text(content)
  store = tool_approvals.ToolApprovals(path)

  assert store.contains(key)
  assert store.available
  assert not store.contains(_key('git status --short'))
  assert path.read_text() == content
  store.remember(_key('pwd'))
  assert json.loads(path.read_text())['entries'] == [
    entry,
    {
      'kind': 'exact_arguments',
      'scope': key.scope,
      'tool': key.tool,
      'arguments_sha256': _key('pwd').arguments_sha256,
    },
  ]
  restarted = tool_approvals.ToolApprovals(path)
  assert restarted.contains(key)
  assert restarted.contains(_key('pwd'))


@pytest.mark.parametrize(
  'updates',
  [
    {'arguments_sha256': 'g' * 64},
    {'argv': ['git']},
    {'kind': None},
  ],
)
def test_malformed_untagged_approval_fails_closed(
  tmp_path: pathlib.Path, updates: dict[str, object]
) -> None:
  path = tmp_path / 'approvals.json'
  key = _key()
  content = json.dumps({
    'version': 1,
    'entries': [
      {
        'scope': key.scope,
        'tool': key.tool,
        'arguments_sha256': key.arguments_sha256,
        **updates,
      }
    ],
  })
  path.write_text(content)
  store = tool_approvals.ToolApprovals(path)
  with pytest.raises(ValueError):
    store.contains(key)
  with pytest.raises(ValueError):
    store.remember(_key('pwd'))
  assert path.read_text() == content


def test_unreadable_file_fails_closed(tmp_path: pathlib.Path) -> None:
  store = tool_approvals.ToolApprovals(tmp_path)
  with pytest.raises(OSError):
    store.contains(_key())
  assert not store.contains(_key())


def test_failed_save_does_not_cache_new_approval(
  tmp_path: pathlib.Path,
) -> None:
  parent = tmp_path / 'not-a-directory'
  path = parent / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  assert not store.contains(_key())
  parent.write_text('block writes')

  with pytest.raises(OSError):
    store.remember(_key())
  assert not store.contains(_key())
  assert parent.read_text() == 'block writes'


def test_corruption_after_load_does_not_get_overwritten(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  store.remember(_key())
  path.write_text('{')

  with pytest.raises(ValueError):
    store.remember(_key('pwd'))
  assert not store.contains(_key('pwd'))
  assert not store.contains(_key())
  assert path.read_text() == '{'


def test_invalid_entry_disables_entire_file(tmp_path: pathlib.Path) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  store.remember(_key())
  path.write_text(path.read_text().replace('"entries": [', '"entries": [{},'))
  restarted = tool_approvals.ToolApprovals(path)
  with pytest.raises(ValueError):
    restarted.contains(_key())
  assert not restarted.contains(_key())


def test_prefix_and_exact_grants_survive_restart(
  tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'approvals.json'
  store = tool_approvals.ToolApprovals(path)
  prefix = tool_approvals.ArgvPrefixApproval(
    '/workspace', 'bash', ('git', 'status')
  )
  store.remember(_key('pwd'))
  store.remember(prefix)

  restarted = tool_approvals.ToolApprovals(path)
  assert restarted.contains(_key('pwd'))
  assert restarted.contains(prefix)
  assert not restarted.contains(_key())
  assert not restarted.contains(
    tool_approvals.ArgvPrefixApproval('/other', 'bash', prefix.argv)
  )
  assert not restarted.contains(
    tool_approvals.ArgvPrefixApproval('/workspace', 'other', prefix.argv)
  )
  assert json.loads(path.read_text())['entries'][1] == {
    'kind': 'argv_prefix',
    'scope': '/workspace',
    'tool': 'bash',
    'argv': ['git', 'status'],
  }
  path.unlink()
  path.mkdir()
  assert restarted.contains(prefix)
  restarted.remember(prefix)
  assert path.is_dir()


@pytest.mark.parametrize(
  'argv',
  [None, [], [''], 'git', [123], ['git', '\n'], ['git', '\x1bhidden']],
)
def test_invalid_prefix_fails_closed(
  tmp_path: pathlib.Path, argv: object
) -> None:
  path = tmp_path / 'approvals.json'
  content = json.dumps({
    'version': 1,
    'entries': [
      {
        'kind': 'argv_prefix',
        'scope': '/workspace',
        'tool': 'bash',
        'argv': argv,
      }
    ],
  })
  path.write_text(content)
  store = tool_approvals.ToolApprovals(path)
  with pytest.raises(ValueError):
    store.contains(_key())
  with pytest.raises(ValueError):
    store.remember(_key())
  assert path.read_text() == content
