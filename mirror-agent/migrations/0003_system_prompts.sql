CREATE TABLE react_system_prompt (
  hash BLOB PRIMARY KEY NOT NULL,
  content TEXT NOT NULL
);
ALTER TABLE react_entry ADD COLUMN system_prompt_hash BLOB REFERENCES react_system_prompt(hash);
CREATE INDEX react_entry_window ON react_entry(session_id, sequence DESC) WHERE system_prompt_hash IS NOT NULL;
