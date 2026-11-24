import pytest
from mirror_transcript import prompt_input
from prompt_toolkit import keys
from prompt_toolkit.input import base
from prompt_toolkit.key_binding import key_processor


@pytest.mark.parametrize('partial', [b'\x1b[200~paste', b'\x1b[', b'\xc3'])
def test_reconnect_discards_partial_input_and_keeps_completed_keys(
  partial: bytes,
) -> None:
  input = prompt_input.Input(base.DummyInput())
  input.feed(7, b'a' + partial, 'old selection')

  input.reconnect()

  assert input.sequence == 0
  assert input.take_selection() == ''
  input.feed(1, b'x\r\x03')
  assert input.read_keys() == [
    key_processor.KeyPress('a'),
    key_processor.KeyPress('x'),
    key_processor.KeyPress(keys.Keys.ControlM, '\r'),
    key_processor.KeyPress(keys.Keys.ControlC, '\x03'),
  ]
