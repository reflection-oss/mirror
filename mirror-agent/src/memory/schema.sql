-- The memory store's schema, created on open. Nothing distributed has written
-- it yet, so there are no migrations; the first schema change after a release
-- adds them.
CREATE TABLE IF NOT EXISTS memory_record (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('preference', 'fact', 'procedure', 'pitfall', 'reference')),
    status TEXT NOT NULL CHECK (status IN ('active', 'retired')),
    version INTEGER NOT NULL,
    actor TEXT NOT NULL CHECK (actor IN ('user', 'extractor', 'consolidator', 'import')),
    repository TEXT NOT NULL DEFAULT '',
    path_globs TEXT NOT NULL DEFAULT '[]',
    situation TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    content TEXT NOT NULL,
    -- Extracted memories carry verbatim quotes from the work and the command
    -- that proved them; imported memories name the harness, directory and file
    -- they came from. Empty for what was written here.
    evidence TEXT NOT NULL DEFAULT '[]',
    verification TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    observed_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS memory_record_scope ON memory_record(status, repository, type);

CREATE TABLE IF NOT EXISTS memory_event (
    seq INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL,
    op TEXT NOT NULL CHECK (op IN ('accept', 'correct', 'retire', 'forget')),
    actor TEXT NOT NULL CHECK (actor IN ('user', 'extractor', 'consolidator', 'import')),
    session_id TEXT,
    version INTEGER NOT NULL,
    at INTEGER NOT NULL
) STRICT;
CREATE INDEX IF NOT EXISTS memory_event_record ON memory_event(memory_id, seq);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    id UNINDEXED, key, situation, keywords, content,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS memory_maintenance (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_consolidated_at INTEGER NOT NULL,
    last_event_seq INTEGER NOT NULL
) STRICT;
