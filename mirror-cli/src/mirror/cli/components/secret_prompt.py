"""Masked input for secrets that only the requesting extension receives."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from prompt_toolkit import buffer, history
from prompt_toolkit.layout import containers, controls, dimension, processors

from mirror.cli import state
from mirror.cli.components import component, menu


class SecretPrompt(component.Component):
  """The masked prompt's title, character count, and keys."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    request = state.secret.request
    if request is None:
      return containers.Window(height=0)
    length = state.secret.length
    panel = menu.ConfirmPanel(
      title=request.title,
      lines=(f'{length} character{"" if length == 1 else "s"}',),
      hint=request.hint,
    )
    return containers.Window(
      controls.FormattedTextControl(panel.render()),
      height=dimension.Dimension.exact(panel.height),
      wrap_lines=False,
    )


class SecretInput:
  """Retained masked input that publishes only its length."""

  def __init__(self, changed: Callable[[int], object]) -> None:
    def on_change(value: buffer.Buffer) -> None:
      changed(len(value.text))

    self.buffer = buffer.Buffer(
      multiline=False,
      history=history.DummyHistory(),
      on_text_changed=on_change,
    )
    self.control = controls.BufferControl(
      buffer=self.buffer,
      include_default_input_processors=False,
      input_processors=[
        processors.PasswordProcessor('•'),
        processors.BeforeInput('› '),
      ],
    )

  def take(self) -> str:
    """Return the input and clear it along with its undo history."""
    secret = self.buffer.text
    self.clear()
    return secret

  def clear(self) -> None:
    """Clear the input and its undo history."""
    self.buffer.reset()
