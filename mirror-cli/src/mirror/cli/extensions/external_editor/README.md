# External editor

Ctrl+G edits the draft using `$VISUAL`, then `$EDITOR`, then `vi`.
For VS Code use `EDITOR="code --wait"`.
Saving and closing updates the draft **without submitting**; failure keeps the original.
Enabled by default, independently of Vim motions. Disable `external_editor` in `/extensions`.
