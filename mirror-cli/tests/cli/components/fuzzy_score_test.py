"""Native matching, cache reuse, and Unicode highlight regression tests."""

from mirror.cli import selection
from mirror.cli.components import fuzzy_score, selector


def test_batch_matching_keeps_ties_stable_and_deduplicates_terms() -> None:
  choices = (('fix', 'fix'), ('fix', 'other'), ('no', 'match'))
  assert (
    selection.fuzzy_matches(choices, 'fx', lambda terms: terms) == choices[:2]
  )
  assert selection.fuzzy_matches(choices, '', lambda terms: terms) == choices
  assert selection.fuzzy_matches(choices, 'zzzz', lambda terms: terms) == ()


def test_matching_and_highlights_search_the_full_long_prompt() -> None:
  text = 'a' * 20000 + ' zebra'
  assert fuzzy_score.ranked_indices((text, 'unrelated'), 'zbr') == (0,)
  assert fuzzy_score.positions(text, 'zbr') == (20001, 20003, 20004)


def test_unicode_highlights_use_characters_not_utf8_bytes() -> None:
  text = '😀 café'
  assert fuzzy_score.positions(text, 'é') == (5,)
  fragments = selector.highlight(text, 'é', 'class:menu-item')
  assert ''.join(fragment[1] for fragment in fragments) == text
  assert [part for style, part in fragments if 'underline' in style] == ['é']
  assert fuzzy_score.positions('😀 fix', 'fx') == (2, 4)


def test_prepared_candidates_and_query_results_are_reused() -> None:
  fuzzy_score._haystacks.cache_clear()
  fuzzy_score.ranked_indices.cache_clear()
  texts = ('fix the bug', 'add a feature')
  first = fuzzy_score.ranked_indices(texts, 'fx')
  assert fuzzy_score.ranked_indices((*texts,), 'fx') is first
  fuzzy_score.ranked_indices(texts, 'ad')
  assert fuzzy_score._haystacks.cache_info().misses == 1
  assert fuzzy_score.ranked_indices.cache_info().hits == 1
  fuzzy_score.ranked_indices((*texts, 'fix another bug'), 'fx')
  assert fuzzy_score._haystacks.cache_info().misses == 2
