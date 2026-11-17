# mirror-transcript

Rust transcript renderer with a Python client.

- Python owns source order, composer, menus, extensions, and workspace lifecycle.
- Rust owns transcript layout, caches, reading anchors, scrolling, and painting.
- Python chrome and the native transcript share one native terminal writer.
- Scrolling works without a Python round trip. Foreground terminal use waits for
  the renderer to acknowledge a pause.

## Process boundary

Python starts a renderer subprocess. Separate pipes carry length-prefixed protobuf
commands and events:

- Python → Rust: source updates, order, themes, navigation, rendered chrome rows,
  geometry, cursor, input routes, and terminal handoff.
- Rust → Python: forwarded input, presentation acknowledgements, reading state,
  resize notifications, copy/link actions, rejections, and startup readiness.

`transcript.proto` defines commands, events, and content types. Bindings are
generated at build time. `ChromeFrame` can publish source updates, navigation,
and chrome together. Composer and menus cross as rendered chrome.

## Storage limits

- Per view: 32 MiB serialized source and 100,000 elements; at most 128 views.
- Aggregate charged source: 128 MiB; active layout/row cache budget: 16 MiB.
- Wire frame: 8 MiB; native event backlog: 1 MiB; Python command backlog: 16 MiB.

Inactive views retain sources and reading anchors, but discard layout caches.
These budgets bound retained storage, not peak RSS or temporary allocations.

## Development

The client launches the bundled `bin/mirror-transcript` by default.
Set `MIRROR_TRANSCRIPT_BIN` to use a locally built executable.
