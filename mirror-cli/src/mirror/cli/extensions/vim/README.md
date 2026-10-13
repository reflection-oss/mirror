# Vim

Enable `vim` in `/extensions` for prompt-toolkit's Vi-style editing.

- `Esc`: dismiss UI or return to Normal mode; it does not interrupt in Vi mode.
- `i`/`a`: Insert; `v`/`V`: Visual. Standard motions, operators and text objects work.
- Enter submits, Ctrl+J adds a newline, Ctrl+R opens history. Cancel pending Vi commands with Esc before using shortcuts.

Not full Vim: no command line, `.vimrc`, or plugins. Native `cw` consumes the word separator; `ciw` preserves it.
