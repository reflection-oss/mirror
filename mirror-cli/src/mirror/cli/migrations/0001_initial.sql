CREATE TABLE IF NOT EXISTS cli_user_messages (
  conversation_id TEXT NOT NULL,
  entry_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  timestamp TEXT NOT NULL,
  content TEXT NOT NULL,
  cwd TEXT NOT NULL,
  git_key TEXT,
  model TEXT NOT NULL,
  PRIMARY KEY (conversation_id, entry_id)
);
CREATE INDEX IF NOT EXISTS cli_user_messages_recent
  ON cli_user_messages(timestamp DESC);
CREATE TABLE IF NOT EXISTS cli_worktree_overlays (
  repository_key TEXT NOT NULL,
  id TEXT NOT NULL,
  alias TEXT COLLATE NOCASE,
  preferences BLOB NOT NULL,
  PRIMARY KEY (repository_key, id),
  UNIQUE (repository_key, alias)
);
