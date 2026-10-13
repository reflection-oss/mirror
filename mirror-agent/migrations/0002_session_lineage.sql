ALTER TABLE react_sessions RENAME TO react_session;
ALTER TABLE react_session ADD COLUMN lineage TEXT NOT NULL DEFAULT '[]';
ALTER TABLE react_session ADD COLUMN fork_points TEXT NOT NULL DEFAULT '[]';
UPDATE react_session SET lineage = json_array(id), fork_points = json_array(9223372036854775807);
ALTER TABLE react_session DROP COLUMN state_id;

CREATE TABLE react_entry (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES react_session(id) ON DELETE RESTRICT,
  sequence INTEGER NOT NULL,
  entry_id TEXT NOT NULL,
  payload BLOB NOT NULL,
  UNIQUE (session_id, sequence),
  UNIQUE (session_id, entry_id)
);
INSERT INTO react_entry(session_id, sequence, entry_id, payload)
  SELECT session_id, sequence, entry_id, payload FROM react_entries;
DROP TABLE react_entries;
