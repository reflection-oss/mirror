# Terminal themes

`/themes` opens the usual searchable selector. Choose with Enter; Escape cancels
and restores the draft. `/themes dark`, `/themes light`, `/themes terminal`, and
`/themes mono` select directly, including during an active turn.

| Theme | Use |
| --- | --- |
| `dark` (default) | Deep-green selection, bright green accents and blue information; neutral charcoal composer/user-message fills. |
| `light` | Deep-green accents and selection, blue information, and light input/message fills. |
| `terminal` | Standard ANSI colors, terminal-default backgrounds, reverse selection. |
| `mono` | No first-party colors; bold/reverse selection and textual status cues. |

`NO_COLOR` (including an empty value) or `TERM=dumb` forces monochrome without
changing the saved preference. Use `terminal` for an ANSI-only terminal. Mirror
does not probe or change the terminal's background. Narrow/short terminals keep
the same controls; startup artwork shrinks or disappears to leave input space.

`COLORTERM=truecolor` or `COLORTERM=24bit` preserves the exact RGB palette instead
of quantizing it to 256 colors. A valid `PROMPT_TOOLKIT_COLOR_DEPTH` takes
precedence over that capability; `NO_COLOR` and `TERM=dumb` still disable colors.
Without an advertised capability or override, prompt-toolkit chooses the depth.
The composer and user messages share the same neutral fill: `#282828` in dark,
`#eeeeee` in light, and the terminal background in terminal/mono.

## Persistence and refresh

The built-in `core/themes` extension uses the existing configuration lifecycle:

```json
{"extensions": {"builtin/themes": {"config": {"name": "light"}}}}
```

This lives in `~/.mirror/config.json`. Writes merge atomically with other
settings and do not grant extension consent or change enablement. Invalid names
fall back to `dark` on load; invalid commands or failed saves report a warning or
error without applying a new preference. Disabling `builtin/themes` removes the
command and restores the default palette.

The initiating workspace updates immediately. Other workspaces and CLI processes
pick up changes through the existing configuration watcher (normally five
seconds). Chrome and live previews resolve the palette on their next render.
Historical conversation output repaints only when foreground and idle, after
replay/rerender/session switching finishes. Repainting uses the existing
transcript; it adds no messages or history divider and does not restart workers.
Already-scrolled-off terminal history may retain its previous colors.

## Semantic contract

`src/mirror/cli/themes.py` is the sole first-party palette definition. Rich uses
`ui.<role>` styles; prompt-toolkit uses `class:ui.<role>` or the component classes
mapped there. `.bold` and `.italic` variants compose emphasis without new colors.
There is no process-global active palette: each render/application resolves its
own immutable preference snapshot.

The base roles are:

- `text`, `muted`, `faint`: normal text, secondary information, quiet chrome.
- `accent`, `info`: navigation/emphasis and informational content.
- `success`, `warning`, `error`: status with labels/icons as non-color cues.
- `input`, `message`, `selected`: paired foreground/background surfaces.
- `addition`, `deletion`: diff backgrounds; `+`/`-` remain visible without color.
- `view-compact`, `view-focus`, `view-detail`: independent blue, green and purple
  badge surfaces, preserving the original mode colors in dark/light. Terminal uses
  ANSI equivalents; mono uses bold reversed text and the unchanged mode labels.

Selectors additionally resolve `menu`, `menu-accent`, `menu-muted`,
`menu-selected`, `menu-selected-detail`, `menu-scroll`, and `menu-disabled`.
Their dark palette preserves the green inset surface and sage/mint text hierarchy;
light uses its own contrasting surface, and terminal/mono inherit the terminal
background. Surface-edge styles derive from their surface's background.
Selected workspace status icons inherit the selected surface's colors, including
animated dots; their glyphs retain status distinctions without reverse-video
turning status foreground colors into mismatched background patches.

This covers startup/cards, menus/resume, workspace/task tabs, composer/completions,
approvals, footer/activity, notices/failures, built-in extensions, packaged voice
UI, and transcript/stream chrome. Extension errors remain above modal selectors.
Cards use single-line borders; code and edit panels share a shaded surface and
left rail. The model name uses primary composer text rather than muted metadata.

The palette does **not** rewrite user/tool content, ANSI output, syntax token
styles, configured workspace-tag colors, or third-party extension literals.
Extensions may opt in to semantic styles; their existing literal styles still
work. ANSI and syntax colors can be reduced by terminal capabilities/monochrome,
not remapped through the semantic roles. Tag defaults are listed centrally but
are user data, not palette roles.
