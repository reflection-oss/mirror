"""Palette contracts and a guard against reintroducing component color literals."""

import ast
import pathlib
import re
import subprocess
import sys

import pyte
import pytest
from rich import color, text

from mirror.cli import extension, rendering, themes
from mirror.cli.components import banner, spinner
from mirror.cli.extensions.themes import extension as themes_extension


@pytest.mark.parametrize('name', themes.NAMES)
def test_theme_covers_all_roles_and_keeps_selection_distinct(name: str) -> None:
  palette = themes.rich_theme(name)
  assert palette.styles['ui.selected'] != palette.styles['ui.accent']
  assert palette.styles['ui.selected'].bold
  sheet = themes.prompt_style(name)
  assert sheet.get_attrs_for_style_str('class:menu-selected').bold
  assert sheet.get_attrs_for_style_str('class:worktree-failed').bold
  assert sheet.get_attrs_for_style_str('class:resume-scope-active').bold


@pytest.mark.parametrize('name', ['dark', 'light'])
def test_brand_accents_are_green_not_turquoise(name: str) -> None:
  palette = themes.rich_theme(name)
  foreground = palette.styles['ui.accent'].color
  assert foreground is not None
  rgb = foreground.get_truecolor()
  assert rgb.green > rgb.red
  assert rgb.green - rgb.blue >= 25
  assert palette.styles['ui.selected'].bgcolor == color.Color.parse('#1e3207')


@pytest.mark.parametrize(
  ('role', 'expected'), [('accent', '#72b58e'), ('success', '#3cb371')]
)
def test_dark_accents_match_metacarbon(role: str, expected: str) -> None:
  assert themes.rich_theme('dark').styles[
    f'ui.{role}'
  ].color == color.Color.parse(expected)


@pytest.mark.parametrize(
  ('accent', 'role'),
  [
    (spinner.Color.PALE_PINK, 'ui.accent'),
    (spinner.Color.NEON_YELLOW, 'ui.warning'),
    (spinner.Color.LIGHT_GREEN, 'ui.success'),
    (spinner.Color.BLUE, 'ui.info'),
    (spinner.Color.PURPLE, 'ui.heading'),
  ],
)
def test_spinner_uses_shared_semantic_roles(
  accent: spinner.Color, role: str
) -> None:
  assert accent.value == role
  assert (
    themes.rich_theme('dark').styles[role]
    != (themes.rich_theme('light').styles[role])
  )
  assert themes.rich_theme('mono').styles[role].color in {
    None,
    color.Color.default(),
  }


@pytest.mark.parametrize('name', ['dark', 'light'])
def test_activity_shimmer_uses_shared_muted_and_highlight_roles(
  name: str,
) -> None:
  sheet = themes.prompt_style(name)
  assert sheet.get_attrs_for_style_str('class:ui.highlight-0').color == (
    sheet.get_attrs_for_style_str('class:ui.muted').color
  )
  assert sheet.get_attrs_for_style_str('class:ui.highlight-127').color == (
    sheet.get_attrs_for_style_str('class:ui.highlight').color
  )


@pytest.mark.parametrize('name', ['terminal', 'mono'])
def test_activity_shimmer_uses_emphasis_without_overriding_terminal_colors(
  name: str,
) -> None:
  sheet = themes.prompt_style(name)
  base = sheet.get_attrs_for_style_str('class:ui.highlight-0')
  peak = sheet.get_attrs_for_style_str('class:ui.highlight-127')
  assert base.color == peak.color == 'default'
  assert not base.bold and peak.bold


def _luminance(value: str) -> float:
  channels = color.Color.parse(value).get_truecolor().normalized
  linear = [
    channel / 12.92
    if channel <= 0.04045
    else ((channel + 0.055) / 1.055) ** 2.4
    for channel in channels
  ]
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


@pytest.mark.parametrize('name', ['dark', 'light'])
@pytest.mark.parametrize(
  ('role', 'background'),
  [
    ('view-compact', '5f87af'),
    ('view-focus', '87af87'),
    ('view-detail', 'af87af'),
  ],
)
def test_view_badges_keep_original_colors_and_text_contrast(
  name: str, role: str, background: str
) -> None:
  badge = themes.prompt_style(name).get_attrs_for_style_str(
    f'class:composer class:footer class:ui.{role}'
  )
  assert (badge.color, badge.bgcolor) == ('000000', background)
  assert badge.bold and not badge.reverse
  assert (_luminance(f'#{badge.bgcolor}') + 0.05) / 0.05 >= 4.5


@pytest.mark.parametrize('role', ['accent', 'info', 'success'])
def test_dark_accents_have_contrast_on_the_composer(role: str) -> None:
  palette = themes.rich_theme('dark').styles
  foreground = palette[f'ui.{role}'].color
  background = palette['ui.input'].bgcolor
  assert foreground is not None and background is not None
  assert (_luminance(foreground.name) + 0.05) / (
    _luminance(background.name) + 0.05
  ) >= 5.5


@pytest.mark.parametrize('name', ['dark', 'light'])
def test_composer_model_uses_primary_text_not_muted_footer(name: str) -> None:
  sheet = themes.prompt_style(name)
  model = sheet.get_attrs_for_style_str(
    'class:composer class:footer class:footer-model'
  )
  assert model.color == sheet.get_attrs_for_style_str('class:composer').color
  assert (
    model.color
    != sheet.get_attrs_for_style_str('class:composer class:footer').color
  )


@pytest.mark.parametrize('name', ['dark', 'light'])
def test_information_is_blue_not_another_brand_green(name: str) -> None:
  foreground = themes.rich_theme(name).styles['ui.info'].color
  assert foreground is not None
  rgb = foreground.get_truecolor()
  assert rgb.blue > rgb.green > rgb.red


@pytest.mark.parametrize(
  ('name', 'background'), [('dark', '282828'), ('light', 'eeeeee')]
)
def test_composer_and_user_message_have_the_same_background(
  name: str, background: str
) -> None:
  sheet = themes.prompt_style(name)
  assert sheet.get_attrs_for_style_str('class:composer').bgcolor == background
  assert (
    sheet.get_attrs_for_style_str('class:composer-bottom-divider').color
    == background
  )
  assert themes.rich_theme(name).styles[
    'ui.message'
  ].bgcolor == color.Color.parse(f'#{background}')


def test_dark_menu_preserves_sage_mint_and_green_surface() -> None:
  sheet = themes.prompt_style('dark')
  surface = sheet.get_attrs_for_style_str('class:menu-surface')
  selected = sheet.get_attrs_for_style_str('class:menu-selected')
  detail = sheet.get_attrs_for_style_str(
    'class:menu-selected class:menu-detail-selected'
  )
  assert (surface.color, surface.bgcolor) == ('87a995', '171d1a')
  assert (selected.color, selected.bgcolor) == ('ffffff', '3f8f68')
  assert (detail.color, detail.bgcolor, detail.bold) == (
    'b7d7c3',
    '3f8f68',
    False,
  )


def test_startup_artwork_uses_distinct_metacarbon_roles() -> None:
  image = banner.artwork(160, 40, 'utf-8')
  assert image.style == 'ui.welcome-logo'
  assert banner.artwork(40, 24, 'utf-8').style == 'ui.welcome-logo'
  assert [span.style for span in image.spans] == [
    'ui.welcome-axis',
    'ui.welcome-reflection',
    'ui.welcome-reflection-deep',
  ]


@pytest.mark.parametrize(
  'role, expected',
  [
    ('welcome-logo', '#72b58e'),
    ('welcome-axis', '#383838'),
    ('welcome-reflection', '#585858'),
    ('welcome-reflection-deep', '#383838'),
    ('welcome-dots', '#3a3a3a'),
  ],
)
def test_dark_welcome_colors_match_metacarbon(role: str, expected: str) -> None:
  assert themes.rich_theme('dark').styles[
    f'ui.{role}'
  ].color == color.Color.parse(expected)


@pytest.mark.parametrize('name', themes.NAMES)
@pytest.mark.parametrize('indicator', ['extension-on', 'extension-off'])
def test_selected_extension_indicator_keeps_selection_contrast(
  name: str, indicator: str
) -> None:
  sheet = themes.prompt_style(name)
  selected = sheet.get_attrs_for_style_str('class:menu-selected')
  indicator_style = sheet.get_attrs_for_style_str(
    f'class:menu-selected class:{indicator}'
  )
  assert indicator_style.color == selected.color
  assert indicator_style.bgcolor == selected.bgcolor
  assert indicator_style.reverse == selected.reverse
  assert (
    sheet.get_attrs_for_style_str('class:menu-selected fg:#123456').color
    == '123456'
  )


@pytest.mark.parametrize('name', themes.NAMES)
@pytest.mark.parametrize('surface', ['worktree-tab-selected', 'menu-selected'])
@pytest.mark.parametrize(
  'indicator',
  [
    'worktree',
    'worktree-running',
    'worktree-attention',
    'worktree-failed',
    'worktree-running class:activity-spinner-highlight',
  ],
)
def test_selected_workspace_status_keeps_its_surface_colors(
  name: str, surface: str, indicator: str
) -> None:
  sheet = themes.prompt_style(name)
  selected = sheet.get_attrs_for_style_str(f'class:{surface}')
  status = sheet.get_attrs_for_style_str(f'class:{surface} class:{indicator}')
  assert (status.color, status.bgcolor, status.reverse) == (
    selected.color,
    selected.bgcolor,
    selected.reverse,
  )


@pytest.mark.parametrize('name', ['terminal', 'mono'])
def test_terminal_palettes_do_not_force_a_background(name: str) -> None:
  sheet = themes.prompt_style(name)
  assert sheet.get_attrs_for_style_str('class:composer').bgcolor == ''
  assert sheet.get_attrs_for_style_str('class:menu-selected').reverse
  assert themes.rich_theme(name).styles['ui.message'].bgcolor is None


@pytest.mark.parametrize('environment', [{'TERM': 'dumb'}, {'NO_COLOR': ''}])
def test_terminal_capabilities_override_color_choice(
  environment: dict[str, str],
) -> None:
  assert themes.effective('light', environment) == 'mono'


@pytest.mark.parametrize(
  ('environment', 'expected'),
  [
    ({}, 'None'),
    ({'COLORTERM': 'truecolor'}, 'DEPTH_24_BIT'),
    ({'COLORTERM': '24BIT'}, 'DEPTH_24_BIT'),
    ({'COLORTERM': 'unknown'}, 'None'),
    (
      {'COLORTERM': 'truecolor', 'PROMPT_TOOLKIT_COLOR_DEPTH': 'DEPTH_8_BIT'},
      'DEPTH_8_BIT',
    ),
    (
      {'COLORTERM': 'truecolor', 'PROMPT_TOOLKIT_COLOR_DEPTH': 'DEPTH_4_BIT'},
      'DEPTH_4_BIT',
    ),
    (
      {'COLORTERM': 'truecolor', 'PROMPT_TOOLKIT_COLOR_DEPTH': 'DEPTH_1_BIT'},
      'DEPTH_1_BIT',
    ),
    (
      {'COLORTERM': 'truecolor', 'PROMPT_TOOLKIT_COLOR_DEPTH': 'invalid'},
      'DEPTH_24_BIT',
    ),
    ({'COLORTERM': 'truecolor', 'NO_COLOR': ''}, 'DEPTH_1_BIT'),
    (
      {'NO_COLOR': '1', 'PROMPT_TOOLKIT_COLOR_DEPTH': 'DEPTH_24_BIT'},
      'DEPTH_1_BIT',
    ),
    (
      {'TERM': 'dumb', 'PROMPT_TOOLKIT_COLOR_DEPTH': 'DEPTH_24_BIT'},
      'DEPTH_1_BIT',
    ),
  ],
)
def test_prompt_color_depth_respects_terminal_capabilities_and_overrides(
  environment: dict[str, str], expected: str
) -> None:
  result = subprocess.run(
    [
      sys.executable,
      '-c',
      'from mirror.cli import themes; '
      'depth = themes.prompt_color_depth(); '
      'print(depth.value if depth is not None else None)',
    ],
    env={'TERM': 'xterm-256color', 'PYTHONDONTWRITEBYTECODE': '1'}
    | environment,
    text=True,
    capture_output=True,
    check=True,
  )
  assert result.stdout.strip() == expected


def test_unknown_theme_falls_back_without_changing_settings() -> None:
  settings = {'name': 'not-a-theme', 'extra': True}
  assert themes.configured(settings) == 'dark'
  assert settings == {'name': 'not-a-theme', 'extra': True}


def test_rich_themes_resolve_at_render_time_without_recoloring_text() -> None:
  value = text.Text('literal #ff0000 yellow bg:red', style='ui.error')
  dark = rendering.renderable_to_ansi(value, 60, theme_name='dark')
  light = rendering.renderable_to_ansi(value, 60, theme_name='light')
  assert dark != light
  assert (
    rendering.strip_ansi(dark) == rendering.strip_ansi(light) == value.plain
  )
  assert rendering.renderable_to_ansi(value, 60, theme_name='dark') == dark


def test_mono_rendering_keeps_text_and_omits_color_sequences() -> None:
  rendered = rendering.renderable_to_ansi(
    text.Text('Error', style='ui.error'), 20, theme_name='mono'
  )
  assert rendering.strip_ansi(rendered) == 'Error'
  assert '\x1b[38;' not in rendered
  assert '\x1b[48;' not in rendered


def test_terminal_ansi_is_not_theme_rewritten() -> None:
  value = '\x1b[38;2;123;45;67mtool output\x1b[0m'
  assert rendering.sanitize_ansi(value) == value


@pytest.mark.asyncio
async def test_builtin_theme_loads_with_saved_configuration() -> None:
  registry = extension.ExtensionRegistry()
  try:
    loaded = await registry.load(
      pathlib.Path(themes_extension.__file__).parent,
      overrides={'config': {'name': 'light'}},
    )
    assert loaded == themes.EXTENSION_ID
  finally:
    await registry.close()


@pytest.mark.parametrize('name', themes.NAMES)
def test_diff_roles_do_not_replace_syntax_foregrounds(name: str) -> None:
  assert themes.rich_theme(name).styles['ui.addition'].color is None
  assert themes.rich_theme(name).styles['ui.deletion'].color is None


@pytest.mark.parametrize(
  ('name', 'filled'),
  [('dark', True), ('light', True), ('terminal', False), ('mono', False)],
)
def test_edit_panel_uses_palette_fill_without_changing_text(
  name: str, filled: bool
) -> None:
  block = rendering.tool_result_renderable(
    'edit',
    'f.py',
    'Replaced text.',
    '--- f.py\n+++ f.py\n@@ -1 +1 @@\n-old = 1\n+new = 2\n',
    compact=True,
    activity_kind='tool',
    error=False,
  )
  assert block is not None
  terminal = pyte.Screen(40, 8)
  pyte.Stream(terminal).feed(
    rendering.renderable_to_ansi(block, 40, theme_name=name).replace(
      '\n', '\r\n'
    )
  )
  assert terminal.display[2].startswith('    ┃    1 - old = 1')
  assert terminal.display[3].startswith('    ┃    1 + new = 2')
  assert (terminal.buffer[2][39].bg != 'default') == filled
  assert (terminal.buffer[3][39].bg != 'default') == filled
  assert terminal.buffer[2][4].bg == 'default'


_COLOR = re.compile(r'#[0-9a-fA-F]{6}\b|(?:fg:|bg:)(?:#[0-9a-fA-F]|ansi[a-z])')
_NAMED = re.compile(
  r'(?:(?:bold|italic|dim) )?(?:red|green|yellow|blue|cyan|magenta|white|black|grey\d+)\Z'
)


def _color_literals(source: pathlib.Path) -> list[str]:
  return [
    node.value
    for node in ast.walk(ast.parse(source.read_text()))
    if isinstance(node, ast.Constant)
    and isinstance(node.value, str)
    and (_COLOR.search(node.value) or _NAMED.fullmatch(node.value))
  ]


@pytest.mark.parametrize(
  'source',
  sorted(
    path
    for path in pathlib.Path(themes.__file__).parent.rglob('*.py')
    if path.name != 'themes.py' and not path.name.endswith('_test.py')
  ),
)
def test_first_party_components_have_no_color_literals(
  source: pathlib.Path,
) -> None:
  assert _color_literals(source) == [], (
    f'UI colors belong in themes.py: {source}'
  )
