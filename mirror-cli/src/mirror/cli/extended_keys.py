"""Augment prompt-toolkit's input parsing with extended-key aliases."""

from __future__ import annotations

from prompt_toolkit import keys
from prompt_toolkit.input import ansi_escape_sequences


def install() -> int:
  """Register the extended sequences used by Mirror's key bindings."""
  aliases = {
    '\x1b[9;2u': keys.Keys.BackTab,
    '\x1b[27;2;9~': keys.Keys.BackTab,
    '\x1b[27;2;9u': keys.Keys.BackTab,
    '\x1b[13;2u': keys.Keys.ControlJ,
    '\x1b[27;2;13~': keys.Keys.ControlJ,
    '\x1b[27;2;13u': keys.Keys.ControlJ,
    '\x1b[13;5u': keys.Keys.ControlJ,
    '\x1b[27;5;13~': keys.Keys.ControlJ,
    '\x1b[27;5;13u': keys.Keys.ControlJ,
    '\x1b[122;6u': keys.Keys.ControlY,
    '\x1b[27;6;122~': keys.Keys.ControlY,
    '\x1b[27;6;122u': keys.Keys.ControlY,
    '\x1b[100;5u': keys.Keys.ControlD,
    '\x1b[27;5;100~': keys.Keys.ControlD,
    '\x1b[27;5;100u': keys.Keys.ControlD,
  }
  for index in range(1, 10):
    key = keys.Keys(f'c-{index}')
    codepoint = ord(str(index))
    aliases[f'\x1b[{codepoint};5u'] = key
    aliases[f'\x1b[27;5;{codepoint}~'] = key
    aliases[f'\x1b[27;5;{codepoint}u'] = key

  changed = 0
  for sequence, key in aliases.items():
    if ansi_escape_sequences.ANSI_SEQUENCES.get(sequence) != key:
      ansi_escape_sequences.ANSI_SEQUENCES[sequence] = key
      changed += 1
  return changed
