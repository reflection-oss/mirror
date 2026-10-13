from mirror.cli.components import secret_prompt


def test_secret_input_publishes_only_its_length_and_forgets_on_take() -> None:
  lengths: list[int] = []
  secret = secret_prompt.SecretInput(lengths.append)

  secret.buffer.insert_text('ghp_secret')

  assert lengths == [10]
  assert secret.take() == 'ghp_secret'
  assert not secret.buffer.text
  secret.buffer.undo()
  assert not secret.buffer.text
