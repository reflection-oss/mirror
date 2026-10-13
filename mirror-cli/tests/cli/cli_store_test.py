"""Tests for the CLI store."""

from __future__ import annotations

import datetime
import pathlib
import sqlite3

import pytest

from mirror import workspace
from mirror.cli import cli_store
from mirror.protocol import (
  agent_pb2,
  client_pb2,
)


def _timestamp(value: str) -> datetime.datetime:
  return datetime.datetime.fromisoformat(value)


def test_store_lists_and_finds_conversations(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'state.db'
  other = tmp_path / 'other'
  other.mkdir()
  with cli_store.Store(database) as store:
    store.record_user_message(
      'current',
      entry_id='typed',
      sequence=0,
      timestamp=_timestamp('2025-01-03T00:00:00+00:00'),
      content='typed prompt',
      cwd=str(tmp_path),
      model='model-a',
    )
    store.record_user_message(
      'other',
      entry_id='other-prompt',
      sequence=0,
      timestamp=_timestamp('2025-01-02T00:00:00+00:00'),
      content='other prompt',
      cwd=str(other),
      model='model-b',
    )

    assert [info.id for info in store.list_infos(cwd=str(tmp_path))] == [
      'current'
    ]
    latest = store.latest(cwd=str(tmp_path))
    assert latest is not None
    assert latest.id == 'current'
    assert latest.title == 'typed prompt'
    assert store.find_by_reference('curr')[0].id == 'current'
    assert store.get_info('current').title == 'typed prompt'


def test_store_supplies_prompt_history_and_rewinds(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'c1',
      entry_id='first',
      sequence=0,
      timestamp=_timestamp('2025-01-01T00:00:00+00:00'),
      content='first prompt',
      cwd=str(tmp_path),
      model='old-model',
    )
    store.record_user_message(
      'c1',
      entry_id='second',
      sequence=2,
      timestamp=_timestamp('2025-01-01T00:00:02+00:00'),
      content='second prompt',
      cwd=str(tmp_path),
      model='new-model',
    )

    assert store.recent_user_messages('c1') == [
      'second prompt',
      'first prompt',
    ]
    assert store.get_info('c1').model == 'new-model'


def test_store_handles_a_fresh_database(
  tmp_path: pathlib.Path,
) -> None:
  with cli_store.Store(tmp_path / 'state.db') as store:
    assert store.list_infos(cwd=str(tmp_path)) == []
    assert store.recent_user_messages() == []


def test_worktree_overlays_store_aliases_and_preferences(
  tmp_path: pathlib.Path,
) -> None:
  with cli_store.Store(tmp_path / 'state.db') as store:
    assert (
      store.set_worktree_alias('repo-a', 'first', '  Shared   name ')
      == 'Shared name'
    )
    store.set_worktree_preferences(
      'repo-a',
      'first',
      client_pb2.WorkspacePreferences(
        reasoning_effort='high', approval_mode=agent_pb2.APPROVAL_MODE_HITL
      ),
    )
    with pytest.raises(ValueError, match='already in use'):
      store.set_worktree_alias('repo-a', 'second', 'shared name')

    store.set_worktree_alias('repo-b', 'second', 'shared name')
    assert store.worktree_overlays('repo-a') == {
      'first': cli_store.WorktreeOverlay(
        alias='Shared name',
        preferences=client_pb2.WorkspacePreferences(
          reasoning_effort='high', approval_mode=agent_pb2.APPROVAL_MODE_HITL
        ),
      )
    }


@pytest.mark.parametrize(
  'tools',
  [
    None,
    client_pb2.WorkspacePreferences.ToolAllowlist(),
    client_pb2.WorkspacePreferences.ToolAllowlist(names=['read']),
    client_pb2.WorkspacePreferences.ToolAllowlist(names=['review_checklist']),
    client_pb2.WorkspacePreferences.ToolAllowlist(
      names=['read', 'review_checklist']
    ),
  ],
)
def test_workspace_preferences_restore_without_losing_presence(
  tmp_path: pathlib.Path,
  tools: client_pb2.WorkspacePreferences.ToolAllowlist | None,
) -> None:
  preferences = client_pb2.WorkspacePreferences(
    reasoning_effort='high',
    model='review-model',
    system_prompt='',
    tools_allowlist=tools,
    extensions=['review/indicator'],
  )
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_preferences('repo', 'review', preferences)
    store.set_worktree_alias('repo', 'review', 'Reviewer')
  with cli_store.Store(database) as store:
    overlay = store.worktree_overlays('repo')['review']
    assert overlay.alias == 'Reviewer'
    assert overlay.preferences == preferences
    assert overlay.preferences.HasField('system_prompt')
    assert overlay.preferences.HasField('tools_allowlist') == (
      tools is not None
    )
    overlay.preferences.reasoning_effort = 'low'
    store.set_worktree_preferences('repo', 'review', overlay.preferences)
  with cli_store.Store(database) as store:
    preferences.reasoning_effort = 'low'
    assert store.worktree_overlays('repo')['review'].preferences == preferences


def test_corrupt_worktree_preferences_reject_unsafe_default_restore(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'state.db'
  with cli_store.Store(database) as store:
    store.set_worktree_alias('repo-a', 'first', 'Existing alias')
  with sqlite3.connect(database) as connection:
    connection.execute(
      'UPDATE cli_worktree_overlays SET preferences = ?', (b'corrupt',)
    )

  with cli_store.Store(database) as store:
    with pytest.raises(ValueError, match='corrupt workspace configuration'):
      store.worktree_overlays('repo-a')


def test_store_records_schema_version(tmp_path: pathlib.Path) -> None:
  database = tmp_path / 'cli.db'
  with cli_store.Store(database):
    pass

  with sqlite3.connect(database) as connection:
    version = connection.execute('PRAGMA user_version').fetchone()[0]
    tables = connection.execute(
      """
      SELECT name FROM sqlite_master
      WHERE type = 'table' AND name LIKE 'cli_%' ORDER BY name
      """
    ).fetchall()
    indexes = connection.execute(
      """
      SELECT name FROM sqlite_master
      WHERE type = 'index' AND name LIKE 'cli_%' ORDER BY name
      """
    ).fetchall()
  assert version == 1
  assert tables == [
    ('cli_user_messages',),
    ('cli_worktree_overlays',),
  ]
  assert indexes == [('cli_user_messages_recent',)]


def test_store_preserves_older_schema_data(tmp_path: pathlib.Path) -> None:
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.record_user_message(
      'preserved',
      entry_id='entry',
      sequence=0,
      timestamp=_timestamp('2025-01-01T00:00:00+00:00'),
      content='preserved prompt',
      cwd=str(tmp_path),
      model='model',
    )
  with sqlite3.connect(database) as connection:
    connection.execute('PRAGMA user_version = 0')

  with cli_store.Store(database) as store:
    assert store.get_info('preserved').title == 'preserved prompt'


def test_store_imports_cli_state_from_agent_database(
  tmp_path: pathlib.Path,
) -> None:
  agent_database = tmp_path / 'state.db'
  cli_database = tmp_path / 'cli.db'
  with cli_store.Store(agent_database) as store:
    store.record_user_message(
      'remote-session',
      entry_id='entry',
      sequence=0,
      timestamp=_timestamp('2025-01-01T00:00:00+00:00'),
      content='preserved prompt',
      cwd=str(tmp_path),
      model='model',
    )
    store.set_worktree_alias('repo-a', 'worktree-a', 'Preserved alias')
    store.set_worktree_preferences(
      'repo-a',
      'worktree-a',
      client_pb2.WorkspacePreferences(
        reasoning_effort='high', approval_mode=agent_pb2.APPROVAL_MODE_HITL
      ),
    )
  with sqlite3.connect(agent_database) as connection:
    connection.execute(
      """
      CREATE TABLE cli_conversation_names (
        conversation_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
      )
      """
    )
    connection.execute(
      'INSERT INTO cli_conversation_names VALUES (?, ?)',
      ('remote-session', 'Preserved remote'),
    )
    connection.execute(
      """
      CREATE TABLE cli_worktree_aliases (
        repository_key TEXT NOT NULL,
        provider TEXT NOT NULL,
        worktree_id TEXT NOT NULL,
        alias TEXT NOT NULL COLLATE NOCASE,
        PRIMARY KEY (repository_key, provider, worktree_id),
        UNIQUE (repository_key, alias)
      )
      """
    )
    connection.execute(
      'INSERT INTO cli_worktree_aliases VALUES (?, ?, ?, ?)',
      ('repo-a', 'local', 'legacy-worktree', 'Legacy alias'),
    )
    connection.execute('CREATE TABLE react_sessions(id TEXT PRIMARY KEY)')

  with cli_store.Store(cli_database, import_from=agent_database) as store:
    assert store.get_info('remote-session').title == 'preserved prompt'
    assert store.worktree_overlays('repo-a') == {
      'legacy-worktree': cli_store.WorktreeOverlay(alias='Legacy alias'),
      'worktree-a': cli_store.WorktreeOverlay(
        alias='Preserved alias',
        preferences=client_pb2.WorkspacePreferences(
          reasoning_effort='high', approval_mode=agent_pb2.APPROVAL_MODE_HITL
        ),
      ),
    }
    repository_key = workspace.resolve_cwd(tmp_path)
    assert store.worktree_overlays(repository_key)['remote-session'].alias == (
      'Preserved remote'
    )

  with sqlite3.connect(cli_database) as connection:
    tables = connection.execute(
      "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
  assert tables == [
    ('cli_user_messages',),
    ('cli_worktree_overlays',),
  ]


def test_store_rejects_newer_schema(tmp_path: pathlib.Path) -> None:
  database = tmp_path / 'cli.db'
  with sqlite3.connect(database) as connection:
    connection.execute('PRAGMA user_version = 3')

  with pytest.raises(RuntimeError, match='newer than this binary'):
    cli_store.Store(database)


def test_workspace_tags_share_the_existing_preferences_blob(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_alias('repo', 'default', 'Main')
    store.set_worktree_preferences(
      'repo', 'default', client_pb2.WorkspacePreferences(model='test-model')
    )
    store.set_workspace_tag('repo', 'default', 'blocked')
    store.set_workspace_tag('other-repo', 'feature-id', 'blocked')
  with cli_store.Store(database) as store:
    overlay = store.worktree_overlays('repo')['default']
    assert overlay.tag_id == 'blocked'
    assert overlay.preferences.tag_id == 'blocked'
    assert overlay.preferences.model == 'test-model'
    assert overlay.alias == 'Main'
    store.set_workspace_tag('repo', 'default', None)
    assert not store.worktree_overlays('repo')['default'].preferences.HasField(
      'tag_id'
    )
    store.delete_workspace_tag('blocked')
    assert store.worktree_overlays('other-repo')['feature-id'].tag_id is None
    assert (
      store.worktree_overlays('repo')['default'].preferences.model
      == 'test-model'
    )
  with sqlite3.connect(database) as connection:
    assert connection.execute('PRAGMA user_version').fetchone()[0] == 1
    assert connection.execute(
      'SELECT preferences FROM cli_worktree_overlays WHERE repository_key = ?',
      ('repo',),
    ).fetchone()[0] == client_pb2.WorkspacePreferences(
      model='test-model'
    ).SerializeToString(deterministic=True)


def test_store_upgrades_v1_without_changing_worktree_overlays(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with sqlite3.connect(database) as connection:
    connection.executescript("""
      CREATE TABLE cli_worktree_overlays (
        repository_key TEXT NOT NULL, id TEXT NOT NULL,
        alias TEXT, preferences BLOB NOT NULL,
        PRIMARY KEY (repository_key, id)
      );
      INSERT INTO cli_worktree_overlays VALUES ('repo', 'existing', 'saved name', X'');
      PRAGMA user_version = 1;
    """)
  with cli_store.Store(database) as store:
    store.save_workspace('repo', 'new-id', str(tmp_path), 'folder')
  with cli_store.Store(database) as store:
    assert store.saved_workspaces('repo') == [
      ('new-id', str(tmp_path), 'folder')
    ]
    assert store.worktree_overlays('repo')['existing'].alias == 'saved name'


def test_additive_schema_keeps_older_readers_and_writers_compatible(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.save_workspace('repo', 'folder-id', str(tmp_path), 'Notes')
    store.set_worktree_alias('repo', 'checkout', 'Before')
  with sqlite3.connect(database) as old_client:
    assert old_client.execute('PRAGMA user_version').fetchone()[0] == 1
    old_client.executescript("""
      UPDATE cli_worktree_overlays SET alias = 'After' WHERE id = 'checkout';
      INSERT INTO cli_worktree_overlays(repository_key, id, alias, preferences)
        VALUES ('repo', 'old-id', 'Old-created', X'');
      UPDATE cli_worktree_overlays SET alias = 'Folder alias' WHERE id = 'folder-id';
    """)
  with cli_store.Store(database) as store:
    assert store.worktree_overlays('repo')['checkout'].alias == 'After'
    assert store.worktree_overlays('repo')['old-id'].alias == 'Old-created'
    assert store.worktree_overlays('repo')['folder-id'].alias == 'Folder alias'
    assert store.saved_workspaces('repo') == [
      ('folder-id', str(tmp_path), 'Notes')
    ]


def test_store_preserves_unknown_additive_data(tmp_path: pathlib.Path) -> None:
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.set_worktree_alias('repo', 'checkout', 'Before')
  with sqlite3.connect(database) as future_client:
    future_client.executescript("""
      ALTER TABLE cli_worktree_overlays ADD COLUMN future_option TEXT DEFAULT 'keep';
      CREATE TABLE future_feature(value TEXT);
      INSERT INTO future_feature VALUES ('keep');
    """)
  with cli_store.Store(database) as store:
    store.set_worktree_alias('repo', 'checkout', 'After')
    store.set_worktree_preferences(
      'repo',
      'checkout',
      client_pb2.WorkspacePreferences(reasoning_effort='high'),
    )
  with sqlite3.connect(database) as future_client:
    assert future_client.execute(
      'SELECT future_option FROM cli_worktree_overlays'
    ).fetchone() == ('keep',)
    assert future_client.execute(
      'SELECT value FROM future_feature'
    ).fetchone() == ('keep',)


@pytest.mark.parametrize(
  'legacy_sql',
  [
    '',
    """
    CREATE TABLE cli_folder_workspaces AS
      SELECT repository_key, id, path, name, created_at FROM cli_workspaces WHERE closed_at IS NULL;
    CREATE TABLE cli_closed_workspaces AS
      SELECT repository_key, id, closed_at FROM cli_workspaces WHERE closed_at IS NOT NULL;
    DROP TABLE cli_workspaces;
  """,
  ],
)
@pytest.mark.parametrize('version', [1, 2])
def test_store_repairs_both_workspace_preview_schemas(
  tmp_path: pathlib.Path, legacy_sql: str, version: int
) -> None:
  database = tmp_path / 'cli.db'
  with sqlite3.connect(database) as preview:
    preview.executescript("""
      CREATE TABLE cli_user_messages (
        conversation_id TEXT, entry_id TEXT, sequence INTEGER, timestamp TEXT,
        content TEXT, cwd TEXT, git_key TEXT, model TEXT
      );
      CREATE TABLE cli_worktree_overlays (
        repository_key TEXT, id TEXT, alias TEXT, preferences BLOB NOT NULL,
        PRIMARY KEY (repository_key, id)
      );
      INSERT INTO cli_worktree_overlays VALUES ('repo', 'open-id', 'Keep alias', X'');
      CREATE TABLE cli_workspaces (
        repository_key TEXT, id TEXT, path TEXT, name TEXT, created_at TEXT, closed_at TEXT
      );
      INSERT INTO cli_workspaces VALUES ('repo', 'open-id', '/notes', 'Notes', 'created', NULL);
      INSERT INTO cli_workspaces VALUES ('repo', 'closed-id', '/closed', 'Closed', 'created', 'closed');
    """)
    preview.executescript(legacy_sql)
    preview.execute(f'PRAGMA user_version = {version}')
  with cli_store.Store(database) as store:
    assert store.saved_workspaces('repo') == [('open-id', '/notes', 'Notes')]
    assert store.closed_workspaces('repo') == {'closed-id'}
    assert store.worktree_overlays('repo')['open-id'].alias == 'Keep alias'
  with sqlite3.connect(database) as old_client:
    assert old_client.execute('PRAGMA user_version').fetchone()[0] == 1
  with cli_store.Store(database) as store:
    assert store.saved_workspaces('repo') == [('open-id', '/notes', 'Notes')]


def test_store_does_not_downgrade_unrecognized_version_two(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with sqlite3.connect(database) as unknown:
    unknown.executescript("""
      CREATE TABLE unrelated(value TEXT);
      INSERT INTO unrelated VALUES ('keep');
      PRAGMA user_version = 2;
    """)
  before = database.read_bytes()
  with pytest.raises(RuntimeError, match='database left unchanged'):
    cli_store.Store(database)
  assert database.read_bytes() == before


def test_failed_preview_import_rolls_back_column_additions(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with sqlite3.connect(database) as preview:
    preview.executescript("""
      CREATE TABLE cli_worktree_overlays (
        repository_key TEXT, id TEXT, alias TEXT, preferences BLOB NOT NULL,
        PRIMARY KEY (repository_key, id)
      );
      CREATE TABLE cli_workspaces (unexpected TEXT);
      INSERT INTO cli_workspaces VALUES ('keep');
      PRAGMA user_version = 1;
    """)
  before = database.read_bytes()
  with pytest.raises(
    cli_store.DatabaseCompatibilityError, match='database left unchanged'
  ):
    cli_store.Store(database)
  assert database.read_bytes() == before


def test_preview_reopened_after_upgrade_preserves_extended_records(
  tmp_path: pathlib.Path,
) -> None:
  database = tmp_path / 'cli.db'
  with cli_store.Store(database) as store:
    store.save_workspace('repo', 'folder-id', str(tmp_path), 'Notes')
  with sqlite3.connect(database) as preview:
    preview.executescript("""
      CREATE TABLE cli_workspaces (
        repository_key TEXT, id TEXT, path TEXT, name TEXT, created_at TEXT, closed_at TEXT
      );
      PRAGMA user_version = 2;
    """)
  with cli_store.Store(database) as store:
    assert store.saved_workspaces('repo') == [
      ('folder-id', str(tmp_path), 'Notes')
    ]
  with sqlite3.connect(database) as connection:
    assert connection.execute('PRAGMA user_version').fetchone()[0] == 1
