from __future__ import annotations

import json

import pytest
from mirror.cli import bash_approvals, tool_approvals


@pytest.mark.parametrize(
  'command,argv',
  [
    ('git  status --short', ('git', 'status', '--short')),
    ('"git" sta"tus"', ('git', 'status')),
    (
      "printf '%s' 'literal ; | & > $ ` *'",
      ('printf', '%s', 'literal ; | & > $ ` *'),
    ),
    ('echo "two words"', ('echo', 'two words')),
    ("echo ''", ('echo', '')),
    ("echo 'we'\"'\"'re'", ('echo', "we're")),
    ('cat ./café.txt', ('cat', './café.txt')),
  ],
)
def test_prefixes_decode_literal_bash_words(
  command: str, argv: tuple[str, ...]
) -> None:
  prefixes = bash_approvals.prefixes(
    json.dumps({'command': command}), '/workspace'
  )
  assert prefixes[0] == tool_approvals.ArgvPrefixApproval(
    '/workspace', 'bash', argv
  )
  assert prefixes[-1] == tool_approvals.ArgvPrefixApproval(
    '/workspace', 'bash', argv[:1]
  )


def test_prefixes_are_narrowest_first() -> None:
  assert bash_approvals.prefixes(
    '{"command":"git status --short"}', '/workspace'
  ) == (
    tool_approvals.ArgvPrefixApproval(
      '/workspace', 'bash', ('git', 'status', '--short')
    ),
    tool_approvals.ArgvPrefixApproval('/workspace', 'bash', ('git', 'status')),
    tool_approvals.ArgvPrefixApproval('/workspace', 'bash', ('git',)),
  )


@pytest.mark.parametrize(
  'command',
  [
    '',
    'git status; echo other',
    'git status;',
    'git status && echo other',
    'git status || echo other',
    'git status | cat',
    'git status &',
    'git status\necho other',
    'git\tstatus',
    'git status > output',
    'git status 2>&1',
    'cat <<EOF\ntext\nEOF',
    'echo $(id)',
    'echo `id`',
    'echo "$(id)"',
    'echo "$HOME"',
    'echo ${HOME}',
    'echo $((1+2))',
    'cat <(id)',
    'cat *',
    'cat file?.txt',
    'cat [ab].txt',
    'echo ~',
    'echo {a,b}',
    'echo "$"',
    'echo escaped\\ word',
    'echo "escaped\\$value"',
    "echo $'escaped'",
    'echo $"translated"',
    'A=value git status',
    '(git status)',
    '{ git status; }',
    'if true; then git status; fi',
    'for value in a; do echo "$value"; done',
    'function command() { echo hi; }',
    '! git status',
    'git status # comment',
    'git "unterminated',
    'echo \x1b[31mhidden',
    'echo \x00',
    'echo \ud800',
    'echo \u202ehidden',
    "'' argument",
  ],
)
def test_prefixes_reject_nonliteral_or_compound_shell_syntax(
  command: str,
) -> None:
  assert not bash_approvals.prefixes(
    json.dumps({'command': command}), '/workspace'
  )


@pytest.mark.parametrize(
  'arguments',
  [
    '{',
    'null',
    '[]',
    '[["command", "ls"]]',
    '{}',
    '{"command":123}',
    '{"command":"ls","command":"pwd"}',
    '{"command":"ls","cwd":"/other"}',
    '{"command":"ls","timeout_seconds":5}',
    '{"command":"ls","detached":true}',
  ],
)
def test_prefixes_require_unambiguous_command_only_arguments(
  arguments: str,
) -> None:
  assert not bash_approvals.prefixes(arguments, '/workspace')


@pytest.mark.parametrize(
  'argv,expected',
  [
    (('cat', 'first.txt'), ('cat',)),
    (('head', '-n', '5', 'first.txt'), ('head',)),
    (('tail', 'first.txt'), ('tail',)),
    (('wc', '-l', 'first.txt'), ('wc',)),
    (('git', 'status'), ('git', 'status')),
    (('bash', '-c', 'cat first.txt'), ('bash', '-c', 'cat first.txt')),
    (('python', '-c', 'print(1)'), ('python', '-c', 'print(1)')),
    (('env', 'cat', 'first.txt'), ('env', 'cat', 'first.txt')),
    (('./cat', 'first.txt'), ('./cat', 'first.txt')),
  ],
)
def test_suggested_rule_only_broadens_file_viewer_prefixes(
  argv: tuple[str, ...],
  expected: tuple[str, ...],
) -> None:
  rule = tool_approvals.ArgvPrefixApproval('/workspace', 'bash', argv)
  assert bash_approvals.suggested_rule(
    rule
  ) == tool_approvals.ArgvPrefixApproval(
    '/workspace',
    'bash',
    expected,
  )
