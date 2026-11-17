"""Palette contracts and a guard against reintroducing component color literals."""

import ast
import dataclasses
import io
import pathlib
import re
import subprocess
import sys

import pyte
import pytest
from mirror.cli.extensions.tokyonight import extension as tokyonight
from prompt_toolkit import application, data_structures
from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.input import defaults
from prompt_toolkit.layout import containers, controls, layout
from prompt_toolkit.output import color_depth, vt100
from rich import color, text

from mirror.cli import extension, extension_state, rendering, themes
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


@pytest.mark.parametrize('name', themes.NAMES)
@pytest.mark.parametrize(
  ('component', 'role'),
  [
    ('code-action', 'muted'),
    ('code-action-hover', 'info'),
    ('code-hint', 'muted'),
    ('code-copied', 'success'),
    ('code-copy-error', 'warning'),
  ],
)
def test_code_copy_controls_use_semantic_foregrounds(
  name: str, component: str, role: str
) -> None:
  sheet = themes.prompt_style(name)
  control = sheet.get_attrs_for_style_str(f'class:ui.code class:{component}')
  semantic = sheet.get_attrs_for_style_str(f'class:ui.{role}')
  assert control.color == semantic.color
  assert (
    control.bgcolor == sheet.get_attrs_for_style_str('class:ui.code').bgcolor
  )


@pytest.mark.parametrize('name', ['dark', 'light'])
def test_brand_accents_are_green_not_turquoise(name: str) -> None:
  palette = themes.rich_theme(name)
  foreground = palette.styles['ui.accent'].color
  assert foreground is not None
  rgb = foreground.get_truecolor()
  assert rgb.green > rgb.red
  assert rgb.green - rgb.blue >= 25
  selected = palette.styles['ui.selected'].bgcolor
  assert selected is not None
  surface = selected.get_truecolor()
  assert surface.green > max(surface.red, surface.blue)


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
    ('view-default', '5f87af'),
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
  ('name', 'background'), [('dark', '282828'), ('light', 'e8e8e8')]
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


@pytest.mark.parametrize('name', themes.NAMES)
@pytest.mark.parametrize('surface', ['', 'class:menu-surface', 'bg:#123456'])
def test_menu_text_inherits_its_container_background(
  name: str, surface: str
) -> None:
  output = io.StringIO()
  sheet = themes.prompt_style(name)
  with defaults.create_pipe_input() as pipe:
    app = application.Application[None](
      layout=layout.Layout(
        containers.Window(
          controls.FormattedTextControl([
            ('class:menu-item', 'Workspace'),
            ('class:menu-detail', ' Running'),
          ]),
          style=surface,
        )
      ),
      style=sheet,
      full_screen=True,
      color_depth=color_depth.ColorDepth.DEPTH_24_BIT,
      input=pipe,
      output=vt100.Vt100_Output(
        output,
        lambda: data_structures.Size(rows=2, columns=40),
        enable_cpr=False,
      ),
    )
    with application.set_app(app):
      app.renderer.render(app, app.layout)
  terminal = pyte.Screen(40, 2)
  pyte.Stream(terminal).feed(output.getvalue())
  background = sheet.get_attrs_for_style_str(surface).bgcolor or 'default'
  assert terminal.display[0] == 'Workspace Running'.ljust(40)
  assert terminal.buffer[0][0].bg == background
  assert terminal.buffer[0][10].bg == background
  assert terminal.buffer[0][39].bg == background
  assert terminal.buffer[1][0].bg == background
  assert sheet.get_attrs_for_style_str('class:menu-item').color == (
    sheet.get_attrs_for_style_str('class:menu-surface').color
  )


def test_startup_artwork_uses_distinct_metacarbon_roles() -> None:
  image = banner.artwork(160, 40, 'utf-8')
  assert image.style == 'ui.welcome-logo'
  assert banner.artwork(40, 24, 'utf-8').style == 'ui.welcome-logo'
  assert [span.style for span in image.spans] == [
    'bold',
    'ui.welcome-axis',
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


@pytest.mark.parametrize(
  'name', [*themes.NAMES, *tokyonight.TokyoNight().themes()]
)
def test_extension_enablement_uses_shared_accent_and_muted_roles(
  name: str | themes.Theme,
) -> None:
  sheet = themes.prompt_style(name)
  enabled = sheet.get_attrs_for_style_str(
    'class:menu-surface class:extension-on'
  )
  disabled = sheet.get_attrs_for_style_str(
    'class:menu-surface class:extension-off'
  )
  assert enabled.color == sheet.get_attrs_for_style_str('class:ui.accent').color
  assert (
    enabled.color
    == sheet.get_attrs_for_style_str('class:worktree-running').color
  )
  assert enabled.bold
  assert (
    sheet.get_attrs_for_style_str('class:ui.extension-active.bold').color
    == enabled.color
  )
  assert (
    themes.rich_theme(name).styles['ui.extension-active']
    == themes.rich_theme(name).styles['ui.accent']
  )
  assert disabled.color == sheet.get_attrs_for_style_str('class:ui.muted').color
  assert (
    enabled.bgcolor
    == disabled.bgcolor
    == sheet.get_attrs_for_style_str('class:menu-surface').bgcolor
  )


@pytest.mark.parametrize(
  'name', [*themes.NAMES, *tokyonight.TokyoNight().themes()]
)
@pytest.mark.parametrize('indicator', ['extension-on', 'extension-off'])
def test_selected_extension_indicator_keeps_selection_contrast(
  name: str | themes.Theme, indicator: str
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
@pytest.mark.parametrize(
  'surface',
  [
    'worktree-tab-selected',
    'menu-selected',
    'workspace-browser class:menu-selected',
  ],
)
@pytest.mark.parametrize(
  'indicator',
  [
    'worktree',
    'worktree-running',
    'worktree-attention',
    'worktree-failed',
    'worktree-running class:activity-spinner-highlight',
    'worktree-running class:ui.success',
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


def test_theme_contributions_snapshot_styles_and_namespace_ids() -> None:
  styles = {'accent': '#123456'}
  custom = themes.Theme('dark', 'Custom dark', styles)
  styles['accent'] = 'invalid'
  first = themes.contributions('first', [custom])[0]
  second = themes.contributions('second', (custom,))[0]
  assert (first.id, second.id) == ('first:dark', 'second:dark')
  assert themes.rich_theme(first).styles[
    'ui.accent'
  ].color == color.Color.parse('#123456')
  assert (
    themes.prompt_style(first).get_attrs_for_style_str('class:ui.accent').color
    == '123456'
  )
  assert (
    themes.rich_theme('dark').styles['ui.accent']
    != themes.rich_theme(first).styles['ui.accent']
  )
  assert themes.configured({'name': 'dark'}, (first, second)) == 'dark'
  changed = dataclasses.replace(first, styles={'accent': '#654321'})
  assert themes.rich_theme(changed) != themes.rich_theme(first)
  assert (
    themes
    .prompt_style(changed)
    .get_attrs_for_style_str('class:ui.accent')
    .color
    == '654321'
  )
  with pytest.raises(ValueError, match='provider ID'):
    themes.contributions('first:second', [custom])


@pytest.mark.parametrize(
  ('values', 'message'),
  [
    ("'dark'", 'sequence'),
    ('None', 'sequence'),
    ("['dark']", 'Theme values'),
    ("[themes.Theme('night', 'Night')] * 2", 'Duplicate'),
    ("[themes.Theme('other:night', 'Night')]", 'local theme ID'),
    ("[themes.Theme('', 'Night')]", 'local theme ID'),
    ("[themes.Theme('night', 'bad\\nlabel')]", 'labels'),
    ("[themes.Theme('night', 'Night', base='missing')]", 'base'),
    ("[themes.Theme('night', 'Night', {'missing': 'red'})]", 'role'),
    ("[themes.Theme('night', 'Night', {'accent': 'not-a-style'})]", 'style'),
    ("[themes.Theme('night', 'Night', {'accent': 42})]", 'AttributeError'),
    (
      "[themes.Theme('night', 'Night', {'text': 'white on black'})]",
      'foreground-only',
    ),
    (
      "[themes.Theme('night', 'Night', {'muted': 'reverse'})]",
      'foreground-only',
    ),
    (
      "[themes.Theme('night', 'Night', {'canvas': 'white on black'})]",
      'only a background',
    ),
    (
      "[themes.Theme('night', 'Night', {'addition': 'green'})]",
      'only a background',
    ),
    ("[themes.Theme('night', None)]", 'AttributeError'),
    ("[themes.Theme(None, 'Night')]", 'TypeError'),
    ("[themes.Theme('night', 'Night'), object()]", 'Theme values'),
  ],
)
@pytest.mark.asyncio
async def test_theme_contributions_reject_invalid_provider_atomically(
  tmp_path: pathlib.Path, values: str, message: str
) -> None:
  source = tmp_path / 'invalid' / 'extension.py'
  source.parent.mkdir()
  source.write_text(
    'from mirror.cli import extension, themes\n'
    'class Invalid(extension.Extension):\n'
    '  @classmethod\n'
    '  def id(cls): return "invalid"\n'
    f'  def themes(self): return {values}\n'
  )
  registry = extension.ExtensionRegistry()
  try:
    with pytest.raises(extension.ExtensionLoadError, match=message):
      await registry.load(source.parent)
    assert registry.extension_ids() == ()
  finally:
    await registry.close()


@pytest.mark.parametrize('environment', [{'NO_COLOR': ''}, {'TERM': 'dumb'}])
def test_custom_theme_respects_terminal_capabilities(
  environment: dict[str, str],
) -> None:
  custom = themes.Theme('night', 'Night', {'accent': '#123456'})
  assert themes.effective(custom, environment) == 'mono'


def test_custom_theme_shared_roles_reach_components_without_repetition() -> (
  None
):
  palette = themes.rich_theme(
    themes.Theme(
      'shared',
      'Shared roles',
      {
        'canvas': 'on #111122',
        'text': '#ddddff',
        'muted': '#9999cc',
        'accent': '#aabbcc',
        'success': '#44bb88',
        'input': '#ddddff on #222233',
        'selected': 'bold #ffffff on #334466 not reverse',
      },
    )
  ).styles
  assert palette['ui.menu'] == palette['ui.message'] == palette['ui.input']
  assert palette['ui.menu-selected'] == palette['ui.selected']
  assert (
    palette['ui.menu-selected-detail'].bgcolor == palette['ui.selected'].bgcolor
  )
  assert not palette['ui.menu-selected-detail'].bold
  assert (
    palette['ui.menu-accent']
    == palette['ui.welcome-logo']
    == palette['ui.accent']
  )
  assert (
    palette['ui.menu-muted']
    == palette['ui.tool-heading']
    == palette['ui.muted']
  )
  assert palette['ui.text'].bgcolor is None
  assert (
    palette['ui.menu-edge'].bgcolor
    == palette['ui.input-edge'].bgcolor
    == palette['ui.canvas'].bgcolor
  )


def test_unavailable_theme_keeps_saved_preference() -> None:
  settings = {'name': 'missing:night'}
  extensions = extension_state.ExtensionStates(
    configs={themes.EXTENSION_ID: settings}
  )
  assert themes.current(extensions) == themes.effective('dark')
  assert extensions.configuration(themes.EXTENSION_ID) == settings
  assert themes.saved({
    'extensions': {'builtin/themes': {'config': settings}}
  }) == themes.effective('dark')


def test_startup_preference_keeps_custom_theme_before_extensions_load() -> None:
  settings = {
    'extensions': {'builtin/themes': {'config': {'name': 'custom:light'}}}
  }
  assert themes.preference(settings) == 'custom:light'


def test_unknown_theme_falls_back_without_changing_settings() -> None:
  settings = {'name': 'not-a-theme', 'extra': True}
  assert themes.configured(settings) == themes.AUTO
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
  assert terminal.display[2].startswith('     ┃    1 - old = 1')
  assert terminal.display[3].startswith('     ┃    1 + new = 2')
  assert (terminal.buffer[2][39].bg != 'default') == filled
  assert (terminal.buffer[3][39].bg != 'default') == filled
  assert terminal.buffer[2][5].bg == 'default'


def _render_diff_terminal(
  name: str | themes.Theme, compact: bool, depth: color_depth.ColorDepth
) -> pyte.Screen:
  """Capture edit output through both Rich and the terminal's color reduction."""
  block = rendering.tool_result_renderable(
    'edit',
    'f.py',
    'Replaced text.',
    '--- f.py\n+++ f.py\n@@ -1,2 +1,2 @@\n keep = 1\n-old = "before"\n+new = "after"\n',
    compact=compact,
    activity_kind='tool',
    error=False,
  )
  assert block is not None
  output = io.StringIO()
  with defaults.create_pipe_input() as pipe:
    app = application.Application[None](
      layout=layout.Layout(
        containers.Window(
          controls.FormattedTextControl(
            ansi.ANSI(rendering.renderable_to_ansi(block, 48, theme_name=name))
          ),
          style='class:theme-root',
        )
      ),
      style=themes.prompt_style(name),
      full_screen=True,
      color_depth=depth,
      input=pipe,
      output=vt100.Vt100_Output(
        output,
        lambda: data_structures.Size(rows=12, columns=48),
        enable_cpr=False,
      ),
    )
    with application.set_app(app):
      app.renderer.render(app, app.layout)
  terminal = pyte.Screen(48, 12)
  pyte.Stream(terminal).feed(output.getvalue())
  return terminal


@pytest.mark.parametrize(
  'name', ['dark', 'light', *tokyonight.TokyoNight().themes()]
)
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize(
  'depth',
  [color_depth.ColorDepth.DEPTH_24_BIT, color_depth.ColorDepth.DEPTH_8_BIT],
)
def test_diff_fills_remain_red_and_green_after_terminal_quantization(
  name: str | themes.Theme, compact: bool, depth: color_depth.ColorDepth
) -> None:
  terminal = _render_diff_terminal(name, compact, depth)
  removed = next(
    row for row, line in enumerate(terminal.display) if '- old' in line
  )
  added = next(
    row for row, line in enumerate(terminal.display) if '+ new' in line
  )
  unchanged = next(
    row for row, line in enumerate(terminal.display) if 'keep = 1' in line
  )
  start = terminal.display[removed].index('┃') + 1
  red = terminal.buffer[removed][47].bg
  green = terminal.buffer[added][47].bg
  assert {
    terminal.buffer[removed][column].bg for column in range(start, 48)
  } == {red}
  assert {terminal.buffer[added][column].bg for column in range(start, 48)} == {
    green
  }
  red_rgb = color.Color.parse(f'#{red}').get_truecolor()
  green_rgb = color.Color.parse(f'#{green}').get_truecolor()
  assert red_rgb.red > red_rgb.green + 20
  assert green_rgb.green > green_rgb.red + 20
  assert terminal.buffer[unchanged][47].bg not in {red, green}
  assert (
    terminal.buffer[removed][start].bg != terminal.buffer[removed][start - 1].bg
  )


@pytest.mark.parametrize('compact', [False, True])
def test_custom_diff_roles_change_fills_without_recoloring_syntax(
  compact: bool,
) -> None:
  custom = themes.contributions(
    'custom',
    [
      themes.Theme(
        'diff',
        'Custom diff',
        {
          'addition': 'on #123456',
          'deletion': 'on #654321',
        },
      )
    ],
  )[0]
  original = _render_diff_terminal(
    'dark', compact, color_depth.ColorDepth.DEPTH_24_BIT
  )
  terminal = _render_diff_terminal(
    custom, compact, color_depth.ColorDepth.DEPTH_24_BIT
  )
  removed = next(
    row for row, line in enumerate(terminal.display) if '- old' in line
  )
  added = next(
    row for row, line in enumerate(terminal.display) if '+ new' in line
  )
  assert terminal.display == original.display
  assert terminal.buffer[removed][47].bg == '654321'
  assert terminal.buffer[added][47].bg == '123456'
  assert [terminal.buffer[removed][column].fg for column in range(48)] == [
    original.buffer[removed][column].fg for column in range(48)
  ]
  assert [terminal.buffer[added][column].fg for column in range(48)] == [
    original.buffer[added][column].fg for column in range(48)
  ]


@pytest.mark.parametrize('name', ['terminal', 'mono'])
def test_ansi_and_monochrome_diffs_keep_change_markers(name: str) -> None:
  terminal = pyte.Screen(40, 4)
  pyte.Stream(terminal).feed(
    rendering.renderable_to_ansi(
      rendering.render_diff(
        '--- f.py\n+++ f.py\n@@ -1 +1 @@\n-old\n+new\n', show_file_headers=False
      ),
      40,
      theme_name=name,
    ).replace('\n', '\r\n')
  )
  assert '- old' in terminal.display[0]
  assert '+ new' in terminal.display[1]


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
    if path.name != 'themes.py'
    and path != pathlib.Path(tokyonight.__file__)
    and not path.name.endswith('_test.py')
  ),
)
def test_first_party_components_have_no_color_literals(
  source: pathlib.Path,
) -> None:
  assert _color_literals(source) == [], (
    f'UI colors belong in palette definitions, not components: {source}'
  )
