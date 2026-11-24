from mirror.cli import themes


def test_rich_theme_fades_a_recoloured_scroll_thumb_in_its_own_colour() -> None:
  custom = themes.Theme('dark', 'Custom dark', {'faint': '#123456'})
  faded = themes.rich_theme(custom).styles['ui.scrollbar-fade-3']
  assert faded.color is not None
  assert faded.color.name == '#123456'


def test_rich_theme_fades_an_unchanged_scroll_thumb_like_its_base() -> None:
  custom = themes.Theme('dark', 'Custom dark', {'accent': '#ff0000'})
  role = 'ui.scrollbar-fade-3'
  assert (
    themes.rich_theme(custom).styles[role]
    == themes.rich_theme('dark').styles[role]
  )
