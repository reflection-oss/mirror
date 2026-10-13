"""Bounded caches for native batch matching and visible-row highlights."""

import functools

import frizbee


@functools.lru_cache(maxsize=16)
def _haystacks(texts: tuple[str, ...]) -> frizbee.Haystacks:
  return frizbee.Haystacks(texts)


@functools.lru_cache(maxsize=64)
def ranked_indices(texts: tuple[str, ...], query: str) -> tuple[int, ...]:
  """Copy each candidate list once and score it in one native call per query."""
  matcher = frizbee.Matcher(
    query, max_typos=0, casing='smart', sort='score_then_index_asc'
  )
  return tuple(match.index for match in matcher.match_list(_haystacks(texts)))


@functools.lru_cache(maxsize=128)
def positions(text: str, query: str) -> tuple[int, ...] | None:
  """Translate native UTF-8 byte offsets to Python character positions."""
  if not query:
    return ()
  match = frizbee.Matcher(query, max_typos=0, casing='smart').match_one_indices(
    text, 0
  )
  if match is None:
    return None
  offsets = set(match.indices)
  if text.isascii():
    return tuple(sorted(offsets))
  matched: list[int] = []
  offset = 0
  for index, character in enumerate(text):
    if offset in offsets:
      matched.append(index)
    offset += len(character.encode('utf-8'))
  return tuple(matched)
