"""Tests for `$skill-name` typeahead completion."""

from __future__ import annotations

import pathlib

from prompt_toolkit import buffer, document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout import containers, controls, processors

from mirror.cli import actions, reducer, state
from mirror.cli.components import completion as completion_component
from mirror import skills
from mirror.cli.composer import skill_completion


def _skill(name: str, description: str) -> skills.Skill:
  return skills.Skill(
    name,
    description,
    'body',
    pathlib.Path(f'/skills/{name}/SKILL.md'),
  )


def _text(fragments: StyleAndTextTuples) -> str:
  return ''.join(fragment[1] for fragment in fragments)


def _root(text: str, cursor: int) -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState(
      model='test',
      reasoning_effort='medium',
      approval_mode='auto',
    ),
    session=state.SessionState(conversation_id='test'),
    lifecycle=state.LifecycleState(active=True),
    composer=actions.ComposerState(text=text, cursor=cursor),
  )


def test_typeahead_filters_renders_and_completes_selected_skill() -> None:
  completion = skill_completion.SkillCompletion([
    _skill('pdf', 'Work with PDF files.'),
    _skill('prs', 'Review and ship pull requests.'),
    _skill('training', 'Debug training jobs.'),
  ])

  input_text = 'Use $p'
  cursor = len(input_text)
  assert completion.visible(input_text, cursor)
  current, _ = reducer.reduce(
    _root(input_text, cursor),
    completion_component.choices_changed(input_text, cursor, completion, ()),
  )
  container = completion_component.Completion().render(current)
  assert isinstance(container, containers.Window)
  assert isinstance(container.content, controls.FormattedTextControl)
  assert not callable(container.content.text)
  rendered = _text(container.content.text)
  assert 'Skills  (1/2)' in rendered
  assert '$pdf' in rendered
  assert '$prs' in rendered
  assert '$training' not in rendered

  matches = completion.matches(input_text, cursor)
  root, _ = reducer.reduce(
    _root(input_text, cursor),
    actions.MoveCompletionRequested(
      1, skill_count=len(matches), command_count=0
    ),
  )
  selected = root.completion.skill.selected_item(matches)
  assert selected is not None and selected.name == 'prs'
  edit = completion.complete('Use $p', len('Use $p'), selected)
  assert edit == skill_completion.TextEdit('Use $prs ', len('Use $prs '))


def test_typeahead_replaces_token_at_cursor_without_losing_suffix() -> None:
  pdf = _skill('pdf', 'Work with PDF files.')
  completion = skill_completion.SkillCompletion([pdf])
  text = '$p summarize this'

  edit = completion.complete(text, len('$p'), pdf)

  assert edit == skill_completion.TextEdit('$pdf summarize this', len('$pdf '))
  assert not completion.visible(edit.text, edit.cursor)


def test_typeahead_completion_before_newline_stays_on_current_line() -> None:
  pdf = _skill('pdf', 'Work with PDF files.')
  completion = skill_completion.SkillCompletion([pdf])
  text = 'Use $p\nFinal line'
  cursor = len('Use $p')

  edit = completion.complete(text, cursor, pdf)

  assert edit == skill_completion.TextEdit(
    'Use $pdf \nFinal line', len('Use $pdf ')
  )
  assert edit.text[edit.cursor] == '\n'


def test_typeahead_completion_and_cancellation_in_multiline_input() -> None:
  pdf = _skill('pdf', 'Work with PDF files.')
  completion = skill_completion.SkillCompletion([pdf])
  text = 'First line\nUse $p later\nFinal line'
  cursor = text.index('$p') + len('$p')

  assert completion.complete(text, cursor, pdf) == skill_completion.TextEdit(
    'First line\nUse $pdf later\nFinal line',
    len('First line\nUse $pdf '),
  )
  assert completion.cancel(text, cursor) == skill_completion.TextEdit(
    'First line\nUse  later\nFinal line',
    len('First line\nUse '),
  )


def test_input_processor_tints_only_exact_loaded_skill_tokens() -> None:
  text = '$prs $unknown $prs,'
  input_document = document.Document(text, cursor_position=len(text))
  control = controls.BufferControl(
    buffer=buffer.Buffer(document=input_document)
  )
  transformation_input = processors.TransformationInput(
    control,
    input_document,
    0,
    lambda index: index,
    [('', text)],
    80,
    1,
  )
  processor = skill_completion.SkillMentionProcessor([
    _skill('prs', 'Review pull requests.')
  ])

  transformed = processor.apply_transformation(transformation_input)

  assert transformed.fragments == [
    ('class:skill-mention-input', '$prs'),
    ('', ' $unknown $prs,'),
  ]

  processor.update(())
  assert processor.apply_transformation(transformation_input).fragments == [
    ('', text)
  ]


def test_typeahead_ignores_non_skill_tokens() -> None:
  completion = skill_completion.SkillCompletion([
    _skill('prs', 'Review pull requests.')
  ])

  assert not completion.visible('echo $HOME', len('echo $HOME'))
  assert not completion.visible('price$prs', len('price$prs'))
