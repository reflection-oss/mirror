ALTER TABLE react_session ADD COLUMN title TEXT NOT NULL DEFAULT '';
ALTER TABLE react_session ADD COLUMN model TEXT NOT NULL DEFAULT '';
ALTER TABLE react_session ADD COLUMN created_at_seconds INTEGER;
ALTER TABLE react_session ADD COLUMN created_at_nanos INTEGER;
ALTER TABLE react_session ADD COLUMN updated_at_seconds INTEGER;
ALTER TABLE react_session ADD COLUMN updated_at_nanos INTEGER;
CREATE INDEX react_session_activity ON react_session(COALESCE(updated_at_seconds, 0) DESC, COALESCE(updated_at_nanos, 0) DESC, id DESC);
