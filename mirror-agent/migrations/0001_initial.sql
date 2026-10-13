CREATE TABLE IF NOT EXISTS react_sessions (
  id TEXT PRIMARY KEY,
  state_id TEXT NOT NULL,
  workspace TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS react_entries (
  session_id TEXT NOT NULL REFERENCES react_sessions(id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL,
  entry_id TEXT NOT NULL,
  payload BLOB NOT NULL,
  PRIMARY KEY (session_id, sequence),
  UNIQUE (session_id, entry_id)
);
