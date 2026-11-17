import pytest
from rich import cells

from mirror.cli.components import banner


@pytest.mark.parametrize(
  'columns, rows', [(100, 40), (40, 24), (16, 24), (80, 10)]
)
def test_artwork_fits_terminal(columns: int, rows: int) -> None:
  image = banner.artwork(columns, rows, 'utf-8')
  assert (
    max((cells.cell_len(line) for line in image.plain.splitlines()), default=0)
    <= columns
  )
  assert len(image.plain.splitlines()) <= max(0, rows - 12)


def test_artwork_uses_reflection_when_room_is_available() -> None:
  image = banner.artwork(100, 40, 'utf-8')
  assert '██' in image.plain
  assert '┈' in image.plain


@pytest.mark.parametrize('encoding', ['ascii', None])
def test_artwork_ascii_output_has_a_readable_fallback(
  encoding: str | None,
) -> None:
  assert banner.artwork(100, 40, encoding).plain == 'Mirror'


def test_artwork_short_terminal_keeps_room_for_input() -> None:
  assert not banner.artwork(100, 10, 'utf-8').plain


@pytest.mark.parametrize(
  'columns, rows, height',
  [
    (120, 40, 5),
    (83, 40, 5),
    (84, 30, 5),
    (120, 24, 5),
    (80, 23, 1),
    (120, 12, 1),
    (19, 40, 1),
    (5, 40, 0),
  ],
)
def test_artwork_matches_metacarbon_breakpoints(
  columns: int, rows: int, height: int
) -> None:
  assert (
    len(banner.artwork(columns, rows, 'utf-8').plain.splitlines()) == height
  )


def test_artwork_compact_uses_exact_quadrant_glyphs() -> None:
  assert banner.artwork(80, 40, 'utf-8').plain.splitlines() == [
    '▗▞▀▀▚▖  ▗▟██▙▖',
    '█    █  ██████   Mirror',
    '▝▚▄▄▞▘  ▝▜██▛▘',
    '┈┈┈┈┈┈┈┈┈┈┈┈┈┈',
    '▗▝▀▀▚   ▗▐█▜▙',
  ]


@pytest.mark.parametrize(
  'available, height', [(14, 5), (6, 5), (5, 3), (4, 3), (3, 1), (1, 0)]
)
def test_artwork_growing_composer_drops_whole_parts(
  available: int, height: int
) -> None:
  assert (
    len(
      banner.artwork(
        120, 40, 'utf-8', available_rows=available
      ).plain.splitlines()
    )
    == height
  )
